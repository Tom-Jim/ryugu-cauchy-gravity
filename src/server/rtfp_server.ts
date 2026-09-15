/**
 * RT-FP bake — same RHGF v5 checkpoint protocol as Werner and Mascon.
 *
 * The solver is the Rust crate in `bakes/rtfp/`, and *all* of its parallelism
 * lives in WGSL compute shaders (polyhedral near-field tensor, BVH ray
 * traversal, deviation quadrature). The host process is single-threaded by
 * construction — `run` only marshals buffers and reads results back.
 */
import { existsSync, readFileSync, unlinkSync } from "fs";
import { join } from "path";

const ROOT = join(import.meta.dir, "../..");
const RTFP_BIN = join(ROOT, "target/release/rtfp-bake");
const RTFP_OBJ =
  Bun.env.BAKE_OBJ ??
  join(ROOT, "../Ryugu_wasm/assets/models/SHAPE_SFM_200k_v20180804.obj");
const RTFP_OUT = join(ROOT, "assets/records/rtfp_faces.bin");
const RTFP_ORDER = join(ROOT, "assets/records/.rtfp_order.bin");
const RTFP_LOG = join(ROOT, "assets/records/.rtfp_progress.log");
const RTFP_DENSITY = join(ROOT, "assets/density/cauchy.toml");
const RTFP_MAGIC = 0x52484746;
const RTFP_DIRECTIONS = Number(Bun.env.RTFP_DIRECTIONS ?? 288);

export type RtfpDensityMode = "cauchy" | "constant";

export type RtfpStatus = {
  state: "idle" | "running" | "done" | "error";
  current: number;
  total: number;
  percent: number;
  message: string;
  error: string;
  canResume: boolean;
};

let rtfpChild: ReturnType<typeof Bun.spawn> | null = null;
/** Observation-surface height (mm) of the current record / run. */
let rtfpStandoffMm = RTFP_STANDOFF_DEFAULT_MM;
/** Cauchy (toml kernels) vs the constant-density control case. */
let rtfpDensityMode: RtfpDensityMode = "cauchy";
let rtfpStatus: RtfpStatus = {
  state: "idle",
  current: 0,
  total: 0,
  percent: 0,
  message: "未开始",
  error: "",
  canResume: false,
};

/** Slider bounds: 1 mm – 32 m, same two-segment track as the other algorithms. */
const RTFP_STANDOFF_MAX_MM = 32000;
const RTFP_STANDOFF_DEFAULT_MM = 16000;

function clampStandoffMm(mm: unknown): number {
  const v = Number(mm);
  if (!Number.isFinite(v)) return RTFP_STANDOFF_DEFAULT_MM;
  return Math.min(RTFP_STANDOFF_MAX_MM, Math.max(1, v));
}

function rtfpPayload(): RtfpStatus & {
  standoffMm: number;
  densityMode: RtfpDensityMode;
  done: boolean;
} {
  // `done` is part of the payload, not derived by the viewer: the record on disk
  // is the only thing that decides it, and the same flag gates the compare panel.
  return {
    ...rtfpStatus,
    standoffMm: rtfpStandoffMm,
    densityMode: rtfpDensityMode,
    done: rtfpStatus.state === "done",
  };
}

function corsJson(data: unknown, status = 200) {
  return Response.json(data, {
    status,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Cache-Control": "no-store",
    },
  });
}

function externalRtfpPid(): number | null {
  try {
    // Bracket trick so this pgrep's own argv does not match the pattern.
    const out = Bun.spawnSync(["pgrep", "-f", "[r]tfp-bake"], {
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

function isRtfpRunning() {
  if (rtfpChild) {
    if (rtfpChild.exitCode !== null) {
      rtfpChild = null;
    } else {
      try {
        process.kill(rtfpChild.pid, 0);
        return true;
      } catch {
        rtfpChild = null;
      }
    }
  }
  return externalRtfpPid() !== null;
}

/** Kill any in-flight RT-FP bake so「重新计算」actually restarts. */
function stopRtfpBake() {
  const pids = new Set<number>();
  if (rtfpChild) {
    try {
      pids.add(rtfpChild.pid);
    } catch {
      /* ignore */
    }
    rtfpChild = null;
  }
  const ext = externalRtfpPid();
  if (ext) pids.add(ext);
  for (const pid of pids) {
    try {
      process.kill(pid, "SIGTERM");
    } catch {
      /* already gone */
    }
  }
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

function parseRtfpProgress(text: string) {
  const faces = [...text.matchAll(/(?<![A-Z_])faces\s+(\d+)\s+\/\s+(\d+)/g)].pop();
  // `PROGRESS_R <vertices done> <vertices total>` is the quad phase; faces are
  // written progressively after it, so both lines can move the bar.
  const prog = [...text.matchAll(/PROGRESS_R\s+(\d+)\s+(\d+)/g)].pop();
  if (faces || prog) {
    const m = faces ?? prog!;
    const current = Number(m[1]);
    const total = Number(m[2]);
    rtfpStatus.current = current;
    rtfpStatus.total = total;
    rtfpStatus.percent =
      total > 0 ? Math.min(current >= total ? 99.9 : (100 * current) / total, 99.9) : 0;
    rtfpStatus.message =
      current >= total && total > 0 ? `面片 ${current} / ${total}（写回中）` : `面片 ${current} / ${total}`;
  }
  if (/analytic tensor/.test(text)) {
    rtfpStatus.message = "解析近场张量（GPU 求积）…";
    rtfpStatus.percent = Math.max(rtfpStatus.percent, 0.5);
  }
  if (/wrote\s+\d+\s+dense face/.test(text)) {
    // Completion is decided by the on-disk record, never by this log line.
    rtfpStatus.message = "正在核验完整记录…";
  }
}

function readRtfpCheckpoint(): {
  ok: boolean;
  current: number;
  total: number;
  done: boolean;
  canResume: boolean;
  standoffMm: number;
} {
  if (!existsSync(RTFP_OUT)) {
    return { ok: false, current: 0, total: 0, done: false, canResume: false, standoffMm: 0 };
  }
  try {
    const buf = Buffer.from(readFileSync(RTFP_OUT));
    if (buf.byteLength < 28) {
      return { ok: false, current: 0, total: 0, done: false, canResume: false, standoffMm: 0 };
    }
    const magic = buf.readUInt32LE(0);
    const version = buf.readUInt32LE(4);
    const total = buf.readUInt32LE(8);
    const headerDone = buf.readUInt32LE(12);
    if (magic !== RTFP_MAGIC || version !== 5 || total === 0) {
      return { ok: false, current: 0, total: 0, done: false, canResume: false, standoffMm: 0 };
    }
    const need = 28 + total * 4;
    if (buf.byteLength < need) {
      return { ok: false, current: 0, total, done: false, canResume: false, standoffMm: 0 };
    }
    // Reserved v5 header slot: observation height in mm (0 = not recorded).
    const rawMm = buf.readFloatLE(16);
    const standoffMm = Number.isFinite(rawMm) && rawMm >= 1 && rawMm <= RTFP_STANDOFF_MAX_MM ? rawMm : 0;
    let finite = 0;
    for (let i = 0; i < total; i++) {
      const s = buf.readFloatLE(28 + i * 4);
      if (Number.isFinite(s)) finite++;
    }
    const current = Math.min(total, Math.max(finite, headerDone));
    const done = finite >= total && total > 0;
    return { ok: true, current, total, done, canResume: finite > 0 && !done, standoffMm };
  } catch {
    return { ok: false, current: 0, total: 0, done: false, canResume: false, standoffMm: 0 };
  }
}

function applyRtfpCheckpoint() {
  const cp = readRtfpCheckpoint();
  const running = isRtfpRunning();
  if (cp.standoffMm > 0) rtfpStandoffMm = cp.standoffMm;
  // Never advertise resume while a bake process is alive.
  rtfpStatus.canResume = running ? false : cp.canResume;
  if (!cp.ok) {
    if (running && rtfpStatus.percent >= 100) rtfpStatus.percent = 0;
    return;
  }
  const logSaysFaces = /^面片 /.test(rtfpStatus.message || "");
  if (!logSaysFaces) {
    rtfpStatus.current = cp.current;
    rtfpStatus.total = cp.total;
    rtfpStatus.percent = cp.total > 0 ? Math.min(100, (100 * cp.current) / cp.total) : 0;
  } else {
    const cur = Math.max(rtfpStatus.current, cp.current);
    const tot = Math.max(rtfpStatus.total, cp.total);
    rtfpStatus.current = cur;
    rtfpStatus.total = tot;
    if (!cp.done) {
      rtfpStatus.percent = tot > 0 ? Math.min(99.9, (100 * cur) / tot) : 0;
    }
  }
  if (cp.done && !running) {
    rtfpStatus.state = "done";
    rtfpStatus.message = "烘焙完成（已有完整记录）";
    rtfpStatus.percent = 100;
    rtfpStatus.current = cp.total;
    rtfpStatus.total = cp.total;
    rtfpStatus.canResume = false;
  } else if (running) {
    rtfpStatus.state = "running";
    if (!cp.done && rtfpStatus.percent >= 100) {
      rtfpStatus.percent = cp.total > 0 ? Math.min(99.9, (100 * cp.current) / cp.total) : 0;
    }
  } else if (cp.canResume) {
    rtfpStatus.state = "idle";
    rtfpStatus.message = `可继续 · 面片 ${cp.current} / ${cp.total}`;
    rtfpStatus.current = cp.current;
    rtfpStatus.total = cp.total;
    rtfpStatus.percent = cp.total > 0 ? Math.min(99.9, (100 * cp.current) / cp.total) : 0;
  }
}

function clearRtfpRecords() {
  for (const p of [RTFP_OUT, RTFP_ORDER, RTFP_LOG]) {
    try {
      if (existsSync(p)) unlinkSync(p);
    } catch {
      /* ignore */
    }
  }
}

async function writeRtfpStub(totalFaces = 196608, standoffMm = rtfpStandoffMm) {
  const buf = Buffer.alloc(28 + totalFaces * 4);
  buf.writeUInt32LE(RTFP_MAGIC, 0);
  buf.writeUInt32LE(5, 4);
  buf.writeUInt32LE(totalFaces, 8);
  buf.writeUInt32LE(0, 12);
  buf.writeFloatLE(standoffMm, 16);
  buf.writeFloatLE(Number.POSITIVE_INFINITY, 20);
  buf.writeFloatLE(Number.NEGATIVE_INFINITY, 24);
  // Must be NaN — 0.0 is finite and looks like a finished bake.
  for (let i = 0; i < totalFaces; i++) {
    buf.writeFloatLE(Number.NaN, 28 + i * 4);
  }
  await Bun.write(RTFP_OUT, buf);
}

export function bootRtfpStatus() {
  applyRtfpCheckpoint();
  if (existsSync(RTFP_LOG)) {
    try {
      parseRtfpProgress(readFileSync(RTFP_LOG, "utf8"));
    } catch {
      /* ignore */
    }
  }
  if (!isRtfpRunning()) {
    const cp = readRtfpCheckpoint();
    if (cp.done) {
      rtfpStatus.state = "done";
      rtfpStatus.message = "烘焙完成（已有完整记录）";
    } else if (cp.canResume) {
      rtfpStatus.state = "idle";
      rtfpStatus.message = `可继续 · 面片 ${cp.current} / ${cp.total}`;
    } else if (rtfpStatus.state === "running") {
      rtfpStatus.state = "idle";
      rtfpStatus.message = "未开始";
    }
  }
}

export function rtfpStatusResponse(): Response {
  if (existsSync(RTFP_LOG)) {
    try {
      parseRtfpProgress(readFileSync(RTFP_LOG, "utf8"));
    } catch {
      /* ignore */
    }
  }
  applyRtfpCheckpoint();
  if (isRtfpRunning()) rtfpStatus.state = "running";
  else if (rtfpStatus.state === "running") {
    const cp = readRtfpCheckpoint();
    rtfpStatus.state = cp.done ? "done" : "idle";
    if (cp.canResume) rtfpStatus.message = `可继续 · 面片 ${cp.current} / ${cp.total}`;
  }
  return corsJson(rtfpPayload());
}

export async function startRtfpBake(
  mode: "restart" | "resume",
  requestedStandoffMm?: number,
  requestedDensity?: string,
): Promise<Response> {
  if (!isRtfpRunning() && rtfpStatus.state === "running") {
    rtfpChild = null;
    applyRtfpCheckpoint();
    const cp = readRtfpCheckpoint();
    rtfpStatus.state = cp.done ? "done" : "idle";
    if (cp.canResume) rtfpStatus.message = `可继续 · 面片 ${cp.current} / ${cp.total}`;
  }
  if (isRtfpRunning()) {
    if (mode === "resume") {
      applyRtfpCheckpoint();
      return corsJson({ ok: true, ...rtfpPayload(), alreadyRunning: true });
    }
    stopRtfpBake();
  }
  if (!existsSync(RTFP_BIN)) {
    rtfpStatus = {
      state: "error",
      current: 0,
      total: 0,
      percent: 0,
      message: "RT-FP 烘焙程序缺失",
      error: `缺少 ${RTFP_BIN}，请先 bun run rtfp:build`,
      canResume: false,
    };
    return corsJson(rtfpPayload(), 500);
  }
  if (!existsSync(RTFP_OBJ)) {
    rtfpStatus = {
      state: "error",
      current: 0,
      total: 0,
      percent: 0,
      message: "网格缺失",
      error: `缺少 OBJ: ${RTFP_OBJ}`,
      canResume: false,
    };
    return corsJson(rtfpPayload(), 500);
  }
  if (!existsSync(RTFP_DENSITY)) {
    rtfpStatus = {
      state: "error",
      current: 0,
      total: 0,
      percent: 0,
      message: "密度 TOML 缺失",
      error: `缺少 ${RTFP_DENSITY}`,
      canResume: false,
    };
    return corsJson(rtfpPayload(), 500);
  }

  // A resume continues the record's own density mode and height; a restart
  // takes both from the UI.
  const nextDensity: RtfpDensityMode =
    mode === "resume"
      ? rtfpDensityMode
      : requestedDensity === "constant"
        ? "constant"
        : "cauchy";
  const recordMm = readRtfpCheckpoint();
  const usedStandoffMm =
    mode === "resume" && recordMm.standoffMm > 0
      ? recordMm.standoffMm
      : clampStandoffMm(requestedStandoffMm ?? rtfpStandoffMm);
  rtfpStandoffMm = usedStandoffMm;
  rtfpDensityMode = nextDensity;

  if (mode === "resume") {
    const cp = readRtfpCheckpoint();
    if (!cp.canResume && !cp.done) {
      rtfpStatus = {
        state: "error",
        current: cp.current,
        total: cp.total,
        percent: 0,
        message: "没有可恢复的记录",
        error: "请先点「重新计算」，或确认 assets/records/rtfp_faces.bin 未损坏",
        canResume: false,
      };
      return corsJson(rtfpPayload(), 400);
    }
    if (cp.done) {
      rtfpStatus = {
        state: "done",
        current: cp.current,
        total: cp.total,
        percent: 100,
        message: "记录已完整，无需继续",
        error: "",
        canResume: false,
      };
      return corsJson({ ok: true, ...rtfpPayload() });
    }
  } else {
    clearRtfpRecords();
    await writeRtfpStub(196608, usedStandoffMm);
  }

  await Bun.write(RTFP_LOG, "");
  const cp0 = mode === "resume" ? readRtfpCheckpoint() : { current: 0, total: 196608 };
  rtfpStatus = {
    state: "running",
    current: cp0.current || 0,
    total: cp0.total || 196608,
    percent:
      mode === "restart"
        ? 0
        : cp0.total && cp0.current
          ? Math.min(99.9, (100 * cp0.current) / cp0.total)
          : 0,
    message:
      (mode === "resume" ? "继续" : "启动") +
      ` RT-FP 全 GPU 烘焙 · ${nextDensity === "constant" ? "常密度" : "Cauchy"} · 观测面 ${usedStandoffMm} mm`,
    error: "",
    canResume: false,
  };

  const args = [
    RTFP_BIN,
    "--obj",
    RTFP_OBJ,
    "--out",
    RTFP_OUT,
    "--order",
    RTFP_ORDER,
    "--density",
    RTFP_DENSITY,
    "--mode",
    nextDensity,
    "--directions",
    String(RTFP_DIRECTIONS),
    "--standoff-mm",
    String(usedStandoffMm),
  ];
  if (mode === "resume") args.push("--resume");

  const proc = Bun.spawn(args, { cwd: ROOT, stdout: "pipe", stderr: "pipe" });
  rtfpChild = proc;

  (async () => {
    let buf = "";
    // Serialize merges — parallel stdout/stderr `buf +=` races freeze progress.
    let chain: Promise<void> = Promise.resolve();
    const append = (stream: ReadableStream<Uint8Array> | null) => {
      if (!stream) return Promise.resolve();
      const reader = stream.getReader();
      const dec = new TextDecoder();
      const pump = async () => {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = dec.decode(value, { stream: true });
          chain = chain.then(async () => {
            buf += chunk;
            await Bun.write(RTFP_LOG, buf);
            parseRtfpProgress(buf);
            applyRtfpCheckpoint();
            if (isRtfpRunning()) rtfpStatus.state = "running";
          });
          await chain;
        }
      };
      return pump();
    };
    await Promise.all([append(proc.stdout), append(proc.stderr)]);
    const code = await proc.exited;
    rtfpChild = null;
    if (existsSync(RTFP_LOG)) {
      try {
        parseRtfpProgress(readFileSync(RTFP_LOG, "utf8"));
      } catch {
        /* ignore */
      }
    }
    applyRtfpCheckpoint();
    if (code === 0) {
      const cpDone = readRtfpCheckpoint();
      if (cpDone.done) {
        rtfpStatus.state = "done";
        rtfpStatus.percent = 100;
        rtfpStatus.current = cpDone.total;
        rtfpStatus.total = cpDone.total;
        rtfpStatus.message = "烘焙完成，记录已保存";
        rtfpStatus.error = "";
        rtfpStatus.canResume = false;
      } else {
        applyRtfpCheckpoint();
        rtfpStatus.state = cpDone.canResume ? "idle" : "error";
        rtfpStatus.message = cpDone.canResume
          ? `可继续 · 面片 ${cpDone.current} / ${cpDone.total}`
          : "烘焙结束但记录不完整";
        rtfpStatus.error = cpDone.canResume ? "" : "exit 0 but incomplete checkpoint";
      }
    } else if (rtfpStatus.state === "running") {
      applyRtfpCheckpoint();
      rtfpStatus.state = "error";
      rtfpStatus.message = rtfpStatus.canResume ? "烘焙中断（可继续）" : "烘焙失败";
      rtfpStatus.error = `exit ${code}`;
    }
  })().catch((e) => {
    rtfpChild = null;
    applyRtfpCheckpoint();
    rtfpStatus.state = "error";
    rtfpStatus.message = rtfpStatus.canResume ? "烘焙异常（可继续）" : "烘焙异常";
    rtfpStatus.error = String(e);
  });

  return corsJson({
    ok: true,
    ...rtfpPayload(),
    mode,
    directions: RTFP_DIRECTIONS,
  });
}
