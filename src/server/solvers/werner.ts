/**
 * Werner solver endpoint — the constant-density reference.
 *
 * This is the ESA `polyhedral-gravity-model` library (vendored, wrapped by
 * `bakes/common/esa_bridge.cpp`) evaluating the closed-form polyhedral
 * gravity-gradient tensor of a *uniform* density body. It is the ground truth
 * for the constant-density comparisons: any solver that claims to reproduce the
 * uniform-density limit has to agree with this record face by face.
 *
 * Like every other solver here it writes an RHGF v5 record, so the viewer and
 * the comparison panel treat it identically — only the binary and the progress
 * dialect differ.
 */
import { existsSync, readFileSync, unlinkSync } from "node:fs";
import { join } from "node:path";
import { FACE_COUNT, ROOT, corsJson } from "./gpu_solver";
import {
  STANDOFF_DEFAULT_MM,
  blankRecord,
  clampStandoffMm,
  recordSummary,
  type RecordSummary,
} from "./record";
import { liveRecordPath, liveWorkPath } from "./live_paths";

const BIN = join(ROOT, "bakes/build/ryugu_gradient_bake_cpp");
const OBJ =
  process.env.BAKE_OBJ ?? join(ROOT, "../Ryugu_wasm/assets/models/SHAPE_SFM_200k_v20180804.obj");
const DENSITY = "uniform";
const OUT = liveRecordPath("werner", DENSITY);
const ORDER = liveWorkPath("werner", DENSITY, "order");
const LOG = liveWorkPath("werner", DENSITY, "log");

export function wernerRecordPath(): string {
  return OUT;
}

export type WernerStatus = {
  state: "idle" | "running" | "done" | "error";
  current: number;
  total: number;
  percent: number;
  message: string;
  error: string;
  canResume: boolean;
};

let child: ReturnType<typeof Bun.spawn> | null = null;
let standoffMm = STANDOFF_DEFAULT_MM;
let status: WernerStatus = {
  state: "idle",
  current: 0,
  total: 0,
  percent: 0,
  message: "Not started",
  error: "",
  canResume: false,
};

const readFileOrNull = (path: string): Buffer | null => {
  try {
    return existsSync(path) ? readFileSync(path) : null;
  } catch {
    return null;
  }
};

const summary = (): RecordSummary =>
  recordSummary(readFileOrNull(OUT) ?? Buffer.alloc(0));

function payload() {
  return { ...status, standoffMm, done: status.state === "done" };
}

/** Find a bake process this server did not spawn (bracket avoids self-match). */
function externalPid(): number | null {
  try {
    const out = Bun.spawnSync(["pgrep", "-f", "[r]yugu_gradient_bake_cpp"], {
      stdout: "pipe",
      stderr: "pipe",
    });
    if (out.exitCode !== 0) return null;
    const pid = Number(new TextDecoder().decode(out.stdout).trim().split(/\s+/)[0]);
    return Number.isFinite(pid) && pid > 0 ? pid : null;
  } catch {
    return null;
  }
}

function isRunning() {
  if (child) {
    if (child.exitCode !== null) {
      child = null;
    } else {
      // The handle can linger after SIGKILL; confirm the OS pid still exists.
      try {
        process.kill(child.pid, 0);
        return true;
      } catch {
        child = null;
      }
    }
  }
  return externalPid() !== null;
}

/** Kill any in-flight bake so a new observation height can start cleanly. */
function stop() {
  const pids = new Set<number>();
  if (child) {
    try {
      pids.add(child.pid);
    } catch {
      /* ignore */
    }
    child = null;
  }
  const ext = externalPid();
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

/**
 * The Werner binary reports progress in four dialects (vertex Hessian batches,
 * face writes, and two generic counters). Faces win over vertices because the
 * face count is what the record — and therefore the viewer — actually tracks.
 */
function parseProgress(text: string) {
  const verts = [...text.matchAll(/VERT_PROGRESS\s+(\d+)\s+(\d+)/g)].pop();
  // Negative lookbehind so VERT_PROGRESS is not read as face PROGRESS.
  const faces = [...text.matchAll(/(?<![A-Z_])faces\s+(\d+)\s+\/\s+(\d+)/g)].pop();
  const prog = [...text.matchAll(/(?<![A-Z_])PROGRESS\s+(\d+)\s+(\d+)/g)].pop();
  const evaluated = [...text.matchAll(/evaluated\s+(\d+)\s+\/\s+(\d+)/g)].pop();
  const set = (current: number, total: number, message: string) => {
    status.current = current;
    status.total = total;
    status.percent = total > 0 ? Math.min(100, (100 * current) / total) : 0;
    status.message = message;
  };
  const faceMatch = faces ?? prog;
  if (faceMatch) {
    set(Number(faceMatch[1]), Number(faceMatch[2]), `Faces ${faceMatch[1]} / ${faceMatch[2]}`);
  } else if (verts) {
    set(Number(verts[1]), Number(verts[2]), `Vertex Hessians ${verts[1]} / ${verts[2]}`);
  } else if (evaluated) {
    set(Number(evaluated[1]), Number(evaluated[2]), `Vertices ${evaluated[1]} / ${evaluated[2]}`);
  }
  if (/wrote\s+\d+\s+dense face|wrote\s+\d+\s+face/.test(text)) {
    status.state = "done";
    status.percent = 100;
    status.message = "Bake complete";
    status.current = status.total || status.current;
    status.canResume = false;
  }
}

function applyCheckpoint() {
  const cp = summary();
  const resumable = cp.exists && cp.current > 0 && !cp.done;
  status.canResume = resumable;
  if (cp.standoffMm > 0) standoffMm = cp.standoffMm;
  if (!cp.exists) return cp;
  // Don't clobber live vertex/face progress coming from the log.
  const logOwnsBar = /^(Vertex Hessians|Vertices) /.test(status.message || "");
  if (!logOwnsBar) {
    status.current = cp.current;
    status.total = cp.total;
    status.percent = cp.total > 0 ? Math.min(100, (100 * cp.current) / cp.total) : 0;
  }
  if (cp.done) {
    if (!isRunning()) {
      status.state = "done";
      status.message = "Bake complete (record already on disk)";
      status.percent = 100;
      status.canResume = false;
    }
  } else if (isRunning()) {
    status.state = "running";
    if (!logOwnsBar) status.message = `Faces ${cp.current} / ${cp.total}`;
    status.canResume = false;
  } else if (resumable) {
    status.state = "idle";
    status.message = `Resumable · faces ${cp.current} / ${cp.total}`;
  }
  return cp;
}

function syncFromLog() {
  const log = readFileOrNull(LOG);
  if (!log) return;
  try {
    parseProgress(log.toString("utf8"));
  } catch {
    /* ignore */
  }
}

function clearRecords() {
  for (const p of [OUT, ORDER, LOG]) {
    try {
      if (existsSync(p)) unlinkSync(p);
    } catch {
      /* ignore */
    }
  }
}

export function bootWernerStatus() {
  applyCheckpoint();
  syncFromLog();
  if (isRunning()) return;
  const cp = summary();
  if (cp.done) {
    status.state = "done";
    status.message = "Bake complete (record already on disk)";
  } else if (cp.exists && cp.current > 0) {
    status.state = "idle";
    status.message = `Resumable · faces ${cp.current} / ${cp.total}`;
  } else if (status.state === "running") {
    status.state = "idle";
    status.message = "Not started";
  }
}

export function wernerStatusResponse(): Response {
  syncFromLog();
  applyCheckpoint();
  if (isRunning()) {
    status.state = "running";
  } else if (status.state === "running") {
    const cp = summary();
    status.state = cp.done ? "done" : "idle";
  }
  return corsJson(payload());
}

export async function startWernerBake(
  mode: "restart" | "resume",
  requestedStandoffMm?: number,
): Promise<Response> {
  if (isRunning()) {
    applyCheckpoint();
    // A restart at a different observation height has to kill the running bake;
    // otherwise the requested height would be silently ignored.
    const wantsNewHeight =
      mode === "restart" &&
      requestedStandoffMm !== undefined &&
      Math.abs(clampStandoffMm(requestedStandoffMm) - standoffMm) > 1e-6;
    if (!wantsNewHeight) {
      return corsJson({ ok: true, ...payload(), alreadyRunning: true });
    }
    stop();
  }
  if (!existsSync(BIN)) {
    status = {
      ...status,
      state: "error",
      message: "Werner bake program missing",
      error: `missing ${BIN}; run \`bun run bakes:build\` first`,
    };
    return corsJson(payload(), 500);
  }
  if (!existsSync(OBJ)) {
    status = { ...status, state: "error", message: "Mesh missing", error: `missing OBJ: ${OBJ}` };
    return corsJson(payload(), 500);
  }

  // A resume only reuses a record produced on the requested surface.
  const recordMm = summary();
  const usedStandoffMm = clampStandoffMm(requestedStandoffMm ?? standoffMm);
  if (
    mode === "resume"
    && recordMm.standoffMm > 0
    && Math.abs(recordMm.standoffMm - usedStandoffMm) > 1e-3
  ) {
    status = {
      ...status,
      state: "error",
      message: "Observation height changed",
      error: `saved result is at ${recordMm.standoffMm} mm; recompute at ${usedStandoffMm} mm`,
      canResume: false,
    };
    return corsJson(payload(), 409);
  }
  standoffMm = usedStandoffMm;

  if (mode === "resume") {
    const cp = summary();
    const resumable = cp.exists && cp.current > 0 && !cp.done;
    if (!resumable && !cp.done) {
      status = {
        ...status,
        state: "error",
        current: cp.current,
        total: cp.total,
        percent: 0,
        message: "No resumable record",
        error: `press "Recompute" first, or check ${OUT}`,
        canResume: false,
      };
      return corsJson(payload(), 400);
    }
    if (cp.done) {
      status = {
        state: "done",
        current: cp.total,
        total: cp.total,
        percent: 100,
        message: "Record already complete",
        error: "",
        canResume: false,
      };
      return corsJson({ ok: true, ...payload() });
    }
  } else {
    // Recompute: clear first, then write a stub so the viewer never 404s.
    clearRecords();
    await Bun.write(OUT, blankRecord(FACE_COUNT, usedStandoffMm));
  }

  await Bun.write(LOG, "");
  const cp0 = mode === "resume" ? summary() : { current: 0, total: FACE_COUNT };
  status = {
    state: "running",
    current: cp0.current || 0,
    total: cp0.total || FACE_COUNT,
    percent: cp0.total && cp0.current ? Math.min(100, (100 * cp0.current) / cp0.total) : 0,
    message: `${mode === "resume" ? "Resuming" : "Starting"} Werner uniform-density bake · ${usedStandoffMm} mm`,
    error: "",
    canResume: false,
  };

  const args = [
    BIN,
    "--obj",
    OBJ,
    "--out",
    OUT,
    "--order",
    ORDER,
    "--standoff-mm",
    String(usedStandoffMm),
  ];
  if (mode === "resume") args.push("--resume");

  const proc = Bun.spawn(args, { cwd: ROOT, stdout: "pipe", stderr: "pipe" });
  child = proc;

  (async () => {
    let buf = "";
    let chain: Promise<void> = Promise.resolve();
    const append = (stream: ReadableStream<Uint8Array> | null) => {
      if (!stream) return Promise.resolve();
      const reader = stream.getReader();
      const dec = new TextDecoder();
      const pump = async () => {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = dec.decode(value, { stream: true });
          chain = chain.then(async () => {
            buf += chunk;
            await Bun.write(LOG, buf);
            parseProgress(buf);
            applyCheckpoint();
            if (isRunning()) status.state = "running";
          });
          await chain;
        }
      };
      return pump();
    };
    await Promise.all([append(proc.stdout), append(proc.stderr)]);
    const code = await proc.exited;
    child = null;
    syncFromLog();
    applyCheckpoint();
    if (code === 0) {
      status.state = "done";
      status.percent = 100;
      status.message = "Bake complete, record saved";
      status.error = "";
      status.canResume = false;
    } else if (status.state === "running") {
      applyCheckpoint();
      status.state = "error";
      status.message = status.canResume ? "Bake interrupted (resumable)" : "Bake failed";
      status.error = `exit ${code}`;
    }
  })().catch((e) => {
    child = null;
    applyCheckpoint();
    status.state = "error";
    status.message = status.canResume ? "Bake aborted (resumable)" : "Bake aborted";
    status.error = String(e);
  });

  return corsJson({ ok: true, ...payload(), mode });
}
