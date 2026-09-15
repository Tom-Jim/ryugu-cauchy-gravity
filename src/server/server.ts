import { join } from "path";
import { existsSync, readFileSync, unlinkSync } from "fs";
import {
  bootMasconStatus,
  masconStatusResponse,
  startMasconBake,
} from "./mascon_server";
import { bootRtfpStatus, rtfpStatusResponse, startRtfpBake } from "./rtfp_server";

const ROOT = join(import.meta.dir, "../..");
const PORT = Number(Bun.env.PORT ?? 3000);
const BAKE_BIN = join(ROOT, "bakes/build/ryugu_gradient_bake_cpp");
const BAKE_OBJ =
  Bun.env.BAKE_OBJ ??
  join(ROOT, "../Ryugu_wasm/assets/models/SHAPE_SFM_200k_v20180804.obj");
const BAKE_OUT = join(ROOT, "assets/records/gradient_faces.bin");
const BAKE_ORDER = join(ROOT, "assets/records/.bake_order.bin");
const BAKE_LOG = join(ROOT, "assets/records/.bake_progress.log");
const BAKE_MAGIC = 0x52484746;
/** Shared observation-surface bounds: 1 mm – 32 m, all algorithms start at 16 m. */
const WERNER_STANDOFF_DEFAULT_MM = 16000;
const WERNER_STANDOFF_MAX_MM = 32000;

const MIME: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript",
  ".mjs": "application/javascript",
  ".wasm": "application/wasm",
  ".glb": "model/gltf-binary",
  ".bin": "application/octet-stream",
  ".wgsl": "text/plain; charset=utf-8",
  ".json": "application/json; charset=utf-8",
};

type BakeStatus = {
  state: "idle" | "running" | "done" | "error";
  current: number;
  total: number;
  percent: number;
  message: string;
  error: string;
  canResume: boolean;
};

let bakeChild: ReturnType<typeof Bun.spawn> | null = null;
/** Observation-surface height (mm) of the current record / run. */
let bakeStandoffMm = WERNER_STANDOFF_DEFAULT_MM;
let status: BakeStatus = {
  state: "idle",
  current: 0,
  total: 0,
  percent: 0,
  message: "未开始",
  error: "",
  canResume: false,
};

/** Every algorithm shares the slider (1 mm – 32 m) and the 16 m start height. */
function clampStandoffMm(mm: unknown): number {
  const v = Number(mm);
  if (!Number.isFinite(v)) return WERNER_STANDOFF_DEFAULT_MM;
  return Math.min(WERNER_STANDOFF_MAX_MM, Math.max(1, v));
}

/** Status payload plus the observation height the record was baked at. */
function statusPayload(): BakeStatus & { standoffMm: number; done: boolean } {
  return { ...status, standoffMm: bakeStandoffMm, done: status.state === "done" };
}

function externalBakePid(): number | null {
  try {
    const out = Bun.spawnSync(["pgrep", "-f", "[r]yugu_gradient_bake_cpp"], {
      stdout: "pipe",
      stderr: "pipe",
    });
    if (out.exitCode !== 0) return null;
    const text = new TextDecoder().decode(out.stdout).trim();
    const pid = Number(text.split(/\s+/)[0]);
    return Number.isFinite(pid) && pid > 0 ? pid : null;
  } catch {
    return null;
  }
}

function isBakeRunning() {
  if (bakeChild) {
    if (bakeChild.exitCode !== null) {
      bakeChild = null;
    } else {
      // Child handle can linger after SIGKILL; confirm the OS pid still exists.
      try {
        process.kill(bakeChild.pid, 0);
        return true;
      } catch {
        bakeChild = null;
      }
    }
  }
  return externalBakePid() !== null;
}

/** Kill any in-flight bake so a new observation height can start cleanly. */
function stopBake() {
  const pids = new Set<number>();
  if (bakeChild) {
    try {
      pids.add(bakeChild.pid);
    } catch {
      /* ignore */
    }
    bakeChild = null;
  }
  const ext = externalBakePid();
  if (ext) pids.add(ext);
  for (const pid of pids) {
    try {
      process.kill(pid, "SIGTERM");
    } catch {
      /* already gone */
    }
  }
  // Brief wait so unlink can replace the open output file.
  Bun.sleepSync(150);
  for (const pid of pids) {
    try {
      process.kill(pid, "SIGKILL");
    } catch {
      /* ignore */
    }
  }
  Bun.sleepSync(50);
}

function parseProgressText(text: string) {
  const verts = [...text.matchAll(/VERT_PROGRESS\s+(\d+)\s+(\d+)/g)].pop();
  // Negative lookbehind so VERT_PROGRESS is not treated as face PROGRESS.
  const faces = [...text.matchAll(/(?<![A-Z_])faces\s+(\d+)\s+\/\s+(\d+)/g)].pop();
  const prog = [...text.matchAll(/(?<![A-Z_])PROGRESS\s+(\d+)\s+(\d+)/g)].pop();
  const evaled = [...text.matchAll(/evaluated\s+(\d+)\s+\/\s+(\d+)/g)].pop();
  // Prefer face progress when present; else vertex Hessian phase.
  if ((faces || prog) && !verts) {
    const m = faces ?? prog!;
    const current = Number(m[1]);
    const total = Number(m[2]);
    status.current = current;
    status.total = total;
    status.percent = total > 0 ? Math.min(100, (100 * current) / total) : 0;
    status.message = `面片 ${current} / ${total}`;
  } else if (faces || prog) {
    // Both exist: faces take priority for display numbers
    const m = faces ?? prog!;
    const current = Number(m[1]);
    const total = Number(m[2]);
    status.current = current;
    status.total = total;
    status.percent = total > 0 ? Math.min(100, (100 * current) / total) : 0;
    status.message = `面片 ${current} / ${total}`;
  } else if (verts) {
    const current = Number(verts[1]);
    const total = Number(verts[2]);
    status.current = current;
    status.total = total;
    status.percent = total > 0 ? Math.min(100, (100 * current) / total) : 0;
    status.message = `顶点 Hessian ${current} / ${total}`;
  } else if (evaled) {
    const current = Number(evaled[1]);
    const total = Number(evaled[2]);
    status.current = current;
    status.total = total;
    status.percent = total > 0 ? Math.min(100, (100 * current) / total) : 0;
    status.message = `顶点 ${current} / ${total}`;
  }
  if (/wrote\s+\d+\s+dense face|wrote\s+\d+\s+face/.test(text)) {
    status.state = "done";
    status.percent = 100;
    status.message = "烘焙完成";
    status.current = status.total || status.current;
    status.canResume = false;
  }
  if (/face assemble|vertex Hessian batches/.test(text) && status.state === "running") {
    // keep latest message from VERT/PROGRESS lines
  }
}

/** Read progressive bake checkpoint from gradient_faces.bin. */
function readCheckpoint(): {
  ok: boolean;
  current: number;
  total: number;
  done: boolean;
  canResume: boolean;
  standoffMm: number;
} {
  if (!existsSync(BAKE_OUT)) {
    return { ok: false, current: 0, total: 0, done: false, canResume: false, standoffMm: 0 };
  }
  try {
    const buf = Buffer.from(readFileSync(BAKE_OUT));
    if (buf.byteLength < 28) {
      return { ok: false, current: 0, total: 0, done: false, canResume: false, standoffMm: 0 };
    }
    const magic = buf.readUInt32LE(0);
    const version = buf.readUInt32LE(4);
    const total = buf.readUInt32LE(8);
    const headerDone = buf.readUInt32LE(12);
    if (magic !== BAKE_MAGIC || version !== 5 || total === 0) {
      return { ok: false, current: 0, total: 0, done: false, canResume: false, standoffMm: 0 };
    }
    const need = 28 + total * 4;
    if (buf.byteLength < need) {
      return { ok: false, current: 0, total, done: false, canResume: false, standoffMm: 0 };
    }
    // Reserved v5 header slot: observation height in mm (0 = written before the
    // height was recorded, i.e. treat as the 1 mm default).
    const rawMm = buf.readFloatLE(16);
    const standoffMm =
      Number.isFinite(rawMm) && rawMm >= 1 && rawMm <= WERNER_STANDOFF_MAX_MM ? rawMm : 0;
    let current = 0;
    for (let i = 0; i < total; i++) {
      const s = buf.readFloatLE(28 + i * 4);
      if (Number.isFinite(s)) current++;
    }
    // Prefer denser count; header lags until flush.
    current = Math.max(current, headerDone);
    const done = current >= total;
    return {
      ok: true,
      current,
      total,
      done,
      canResume: current > 0 && !done,
      standoffMm,
    };
  } catch {
    return { ok: false, current: 0, total: 0, done: false, canResume: false, standoffMm: 0 };
  }
}

function applyCheckpointToStatus() {
  const cp = readCheckpoint();
  status.canResume = cp.canResume;
  if (cp.standoffMm > 0) bakeStandoffMm = cp.standoffMm;
  if (!cp.ok) return cp;
  // Don't clobber live VERT_PROGRESS / face PROGRESS from the bake log.
  const logSaysVerts = /^顶点 Hessian/.test(status.message || "");
  const logSaysFaces = /^面片 /.test(status.message || "");
  if (!logSaysVerts) {
    status.current = cp.current;
    status.total = cp.total;
    status.percent = cp.total > 0 ? Math.min(100, (100 * cp.current) / cp.total) : 0;
  }
  if (cp.done) {
    if (!isBakeRunning()) {
      status.state = "done";
      status.message = "烘焙完成（已有完整记录）";
      status.percent = 100;
      status.canResume = false;
    }
  } else if (isBakeRunning()) {
    status.state = "running";
    if (!logSaysVerts && !logSaysFaces) {
      status.message = `面片 ${cp.current} / ${cp.total}`;
    }
    status.canResume = false;
  } else if (cp.canResume) {
    status.state = "idle";
    status.message = `可继续 · 面片 ${cp.current} / ${cp.total}`;
  }
  return cp;
}

function corsJson(data: unknown, statusCode = 200) {
  return new Response(JSON.stringify(data), {
    status: statusCode,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Resource-Policy": "same-origin",
      "Cache-Control": "no-store",
    },
  });
}

function syncFromLog() {
  if (!existsSync(BAKE_LOG)) return;
  try {
    parseProgressText(readFileSync(BAKE_LOG, "utf8"));
  } catch {
    /* ignore */
  }
}

function clearBakeRecords() {
  for (const p of [BAKE_OUT, BAKE_ORDER, BAKE_LOG]) {
    try {
      if (existsSync(p)) unlinkSync(p);
    } catch {
      /* ignore */
    }
  }
}

/** Minimal v5 header so /assets/records/gradient_faces.bin never 404s on restart. */
async function writeBakeStub(totalFaces = 196608, standoffMm = bakeStandoffMm) {
  const buf = Buffer.alloc(28 + totalFaces * 4);
  buf.writeUInt32LE(BAKE_MAGIC, 0);
  buf.writeUInt32LE(5, 4);
  buf.writeUInt32LE(totalFaces, 8);
  buf.writeUInt32LE(0, 12);
  buf.writeFloatLE(standoffMm, 16);
  buf.writeFloatLE(Number.POSITIVE_INFINITY, 20);
  buf.writeFloatLE(Number.NEGATIVE_INFINITY, 24);
  for (let i = 0; i < totalFaces; i++) {
    buf.writeFloatLE(Number.NaN, 28 + i * 4);
  }
  await Bun.write(BAKE_OUT, buf);
}

async function startBake(
  mode: "restart" | "resume",
  requestedStandoffMm?: number,
): Promise<Response> {
  if (isBakeRunning()) {
    applyCheckpointToStatus();
    // A restart at a different observation height has to kill the running bake;
    // otherwise the requested height would be silently ignored.
    const wantsNewHeight = mode === "restart"
      && requestedStandoffMm !== undefined
      && Math.abs(clampStandoffMm(requestedStandoffMm) - bakeStandoffMm) > 1e-6;
    if (!wantsNewHeight) {
      return corsJson({ ok: true, ...statusPayload(), alreadyRunning: true });
    }
    stopBake();
  }
  if (!existsSync(BAKE_BIN)) {
    status = {
      state: "error",
      current: 0,
      total: 0,
      percent: 0,
      message: "烘焙程序缺失",
      error: `缺少 ${BAKE_BIN}，请先 bun run esa:bridge`,
      canResume: false,
    };
    return corsJson(statusPayload(), 500);
  }
  if (!existsSync(BAKE_OBJ)) {
    status = {
      state: "error",
      current: 0,
      total: 0,
      percent: 0,
      message: "网格缺失",
      error: `缺少 OBJ: ${BAKE_OBJ}`,
      canResume: false,
    };
    return corsJson(statusPayload(), 500);
  }

  // Observation height: a resume has to continue the record's own surface, a
  // restart starts at the height the UI slider asked for (1 mm – 32 m).
  const recordMm = readCheckpoint();
  const usedStandoffMm =
    mode === "resume" && recordMm.standoffMm > 0
      ? recordMm.standoffMm
      : clampStandoffMm(requestedStandoffMm ?? bakeStandoffMm);
  bakeStandoffMm = usedStandoffMm;

  if (mode === "resume") {
    const cp = readCheckpoint();
    if (!cp.canResume && !cp.done) {
      status = {
        state: "error",
        current: cp.current,
        total: cp.total,
        percent: 0,
        message: "没有可恢复的记录",
        error: "请先点「重新计算」，或确认 assets/records/gradient_faces.bin 未损坏",
        canResume: false,
      };
      return corsJson(statusPayload(), 400);
    }
    if (cp.done) {
      status = {
        state: "done",
        current: cp.current,
        total: cp.total,
        percent: 100,
        message: "记录已完整，无需继续",
        error: "",
        canResume: false,
      };
      return corsJson({ ok: true, ...statusPayload() });
    }
  } else {
    // 重新计算：清空记录后写占位 bin，避免前端轮询 404
    clearBakeRecords();
    await writeBakeStub(196608, usedStandoffMm);
  }

  await Bun.write(BAKE_LOG, "");
  const cp0 = mode === "resume" ? readCheckpoint() : { current: 0, total: 196608 };
  status = {
    state: "running",
    current: cp0.current || 0,
    total: cp0.total || 196608,
    percent:
      cp0.total && cp0.current
        ? Math.min(100, (100 * cp0.current) / cp0.total)
        : 0,
    message:
      mode === "resume"
        ? `继续全网格逐面烘焙… · 观测面 ${usedStandoffMm} mm`
        : `启动全网格逐面烘焙… · 观测面 ${usedStandoffMm} mm`,
    error: "",
    canResume: false,
  };

  const args = [
    BAKE_BIN,
    "--obj",
    BAKE_OBJ,
    "--out",
    BAKE_OUT,
    "--order",
    BAKE_ORDER,
    "--standoff-mm",
    String(usedStandoffMm),
  ];
  if (mode === "resume") args.push("--resume");

  const proc = Bun.spawn(args, {
    cwd: ROOT,
    stdout: "pipe",
    stderr: "pipe",
  });
  bakeChild = proc;

  (async () => {
    const out = proc.stdout;
    const err = proc.stderr;
    let buf = "";
    const append = async (stream: ReadableStream<Uint8Array> | null) => {
      if (!stream) return;
      const reader = stream.getReader();
      const dec = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = dec.decode(value, { stream: true });
        buf += chunk;
        await Bun.write(BAKE_LOG, buf);
        parseProgressText(buf);
        applyCheckpointToStatus();
        if (isBakeRunning()) status.state = "running";
      }
    };
    await Promise.all([append(out), append(err)]);
    const code = await proc.exited;
    bakeChild = null;
    syncFromLog();
    applyCheckpointToStatus();
    if (code === 0) {
      status.state = "done";
      status.percent = 100;
      status.message = "烘焙完成，记录已保存";
      status.error = "";
      status.canResume = false;
    } else if (status.state === "running") {
      applyCheckpointToStatus();
      status.state = "error";
      status.message = status.canResume ? "烘焙中断（可继续）" : "烘焙失败";
      status.error = `exit ${code}`;
    }
  })().catch((e) => {
    bakeChild = null;
    applyCheckpointToStatus();
    status.state = "error";
    status.message = status.canResume ? "烘焙异常（可继续）" : "烘焙异常";
    status.error = String(e);
  });

  return corsJson({ ok: true, ...statusPayload(), mode });
}

Bun.serve({
  port: PORT,
  hostname: "0.0.0.0",
  async fetch(req) {
    const url = new URL(req.url);
    const pathname = url.pathname;

    if (pathname === "/api/bake/status") {
      syncFromLog();
      applyCheckpointToStatus();
      if (isBakeRunning()) status.state = "running";
      else if (status.state === "running") {
        // 进程已不在，但日志曾标 running：按记录纠正
        const cp = readCheckpoint();
        status.state = cp.done ? "done" : cp.canResume ? "idle" : "idle";
        if (cp.canResume) status.message = `可继续 · 面片 ${cp.current} / ${cp.total}`;
      }
      return corsJson(statusPayload());
    }
    if (pathname === "/api/bake/start" && req.method === "POST") {
      let mode: "restart" | "resume" = "restart";
      let standoffMm: number | undefined;
      try {
        const body = (await req.json()) as { mode?: string; standoffMm?: number };
        if (body?.mode === "resume") mode = "resume";
        else if (body?.mode === "restart") mode = "restart";
        if (body?.standoffMm !== undefined) standoffMm = body.standoffMm;
      } catch {
        mode = "restart";
      }
      return startBake(mode, standoffMm);
    }

    if (pathname === "/api/rtfp/status") {
      return rtfpStatusResponse();
    }
    if (pathname === "/api/rtfp/start" && req.method === "POST") {
      let mode: "restart" | "resume" = "restart";
      let standoffMm: number | undefined;
      let density: string | undefined;
      try {
        const body = (await req.json()) as {
          mode?: string;
          standoffMm?: number;
          density?: string;
        };
        if (body?.mode === "resume") mode = "resume";
        else if (body?.mode === "restart") mode = "restart";
        if (body?.standoffMm !== undefined) standoffMm = body.standoffMm;
        if (body?.density !== undefined) density = body.density;
      } catch {
        mode = "restart";
      }
      return startRtfpBake(mode, standoffMm, density);
    }

    if (pathname === "/api/mascon/status") {
      return masconStatusResponse();
    }
    if (pathname === "/api/mascon/start" && req.method === "POST") {
      let mode: "restart" | "resume" = "restart";
      let standoffMm: number | undefined;
      try {
        const body = (await req.json()) as { mode?: string; standoffMm?: number };
        if (body?.mode === "resume") mode = "resume";
        else if (body?.mode === "restart") mode = "restart";
        if (body?.standoffMm !== undefined) standoffMm = body.standoffMm;
      } catch {
        mode = "restart";
      }
      return startMasconBake(mode, standoffMm);
    }

    const relative = pathname === "/" ? "src/web/index.html" : pathname.slice(1);
    const file = Bun.file(join(ROOT, relative));
    if (!(await file.exists())) {
      return new Response("Not Found", { status: 404 });
    }
    const ext = pathname === "/" ? ".html" : pathname.slice(pathname.lastIndexOf("."));
    return new Response(file, {
      headers: {
        "Content-Type": MIME[ext] ?? "application/octet-stream",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
        "Cross-Origin-Resource-Policy": "same-origin",
        "Cache-Control": "no-store",
      },
    });
  },
});

console.log(`http://127.0.0.1:${PORT}`);

// Boot: prefer on-disk bake record over stale log "running".
applyCheckpointToStatus();
if (existsSync(BAKE_LOG)) {
  try {
    const text = readFileSync(BAKE_LOG, "utf8");
    parseProgressText(text);
  } catch {
    /* ignore */
  }
}
if (!isBakeRunning()) {
  const cp = readCheckpoint();
  if (cp.done) {
    status.state = "done";
    status.message = "烘焙完成（已有完整记录）";
  } else if (cp.canResume) {
    status.state = "idle";
    status.message = `可继续 · 面片 ${cp.current} / ${cp.total}`;
  } else if (status.state === "running") {
    status.state = "idle";
    status.message = "未开始";
  }
}

bootMasconStatus();
bootRtfpStatus();
