/**
 * Generic controller for the native `rtfp-bake` binary, shared by the RT-FP and
 * Carlson solvers.
 *
 * Both solvers are the same Rust crate writing the same RHGF v5 record with the
 * same progress protocol; they differ only in which `--solver` value they pass
 * and which files they own on disk. Everything else — process lifetime, resume
 * semantics, checkpoint reading, progress parsing — is identical, so it lives
 * here once.
 *
 * Progress is reported by two independent sources and they are deliberately not
 * merged into a single truth:
 *
 *  * the child's stdout, parsed for `PROGRESS_R` / `faces n / m` lines, which is
 *    what moves the bar smoothly, and
 *  * the record file on disk, which is the only thing that can declare a bake
 *    *finished* (`done`), because it is also the thing the viewer renders.
 */
import {
  closeSync,
  existsSync,
  openSync,
  readFileSync,
  readSync,
  statSync,
  unlinkSync,
} from "node:fs";
import { join } from "node:path";
import {
  STANDOFF_DEFAULT_MM,
  blankRecord,
  clampStandoffMm,
  recordSummary,
  type RecordSummary,
} from "./record";

export const ROOT = join(import.meta.dir, "../../..");
export const LAUNCHER_BIN = join(ROOT, "target/release/rtfp-bake");
export const DEFAULT_OBJ =
  process.env.BAKE_OBJ ?? join(ROOT, "../Ryugu_wasm/assets/models/SHAPE_SFM_200k_v20180804.obj");

/** Face count of the Ryugu observation mesh, and the record's fixed size. */
export const FACE_COUNT = 196608;
const CHECKPOINT_HEADER_BYTES = 36;
const MAX_CAPTURED_LOG_CHARS = 512 * 1024;
const LOG_FLUSH_INTERVAL_MS = 100;
const VERTEX_PROGRESS_START = 1;
const VERTEX_PROGRESS_END = 90;
const FACE_PROGRESS_START = 90;
const FACE_PROGRESS_END = 99.9;

export type DensityMode = "cauchy" | "constant" | "elliptic";
export type DensityModeMap = Partial<Record<DensityMode, string>>;

export type SolverState = "idle" | "running" | "done" | "error";

export type SolverStatus = {
  state: SolverState;
  current: number;
  total: number;
  percent: number;
  message: string;
  error: string;
  canResume: boolean;
};

export type SolverStatusPayload = SolverStatus & {
  standoffMm: number;
  densityMode: DensityMode;
  done: boolean;
  records: Record<string, RecordSummary>;
};

export type SolverSpec = {
  /** `--solver` value handed to the binary. */
  id: "ray" | "carlson" | "carlson-alpha";
  /** Human-readable name used in status messages. */
  label: string;
  /** Density selected before the UI explicitly requests another one. */
  defaultDensity: DensityMode;
  /** Record written for each density mode. */
  outByMode: DensityModeMap;
  /** Density TOML consumed by each mode. */
  densityByMode: DensityModeMap;
  /** stdout capture per density mode. */
  logByMode: DensityModeMap;
  /** Per-vertex GPU tensor prefixes, one per density mode. */
  checkpointByMode: DensityModeMap;
  /** Progressive face-write order per density mode. */
  orderByMode: DensityModeMap;
  /** Extra CLI arguments beyond the shared ones. */
  extraArgs: string[];
  /** Extra flags for a specific density mode (e.g. `--directions`). */
  modeArgs: Partial<Record<DensityMode, string[]>>;
};

export function corsJson(data: unknown, status = 200) {
  return Response.json(data, {
    status,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Resource-Policy": "same-origin",
      "Cache-Control": "no-store",
    },
  });
}

function readFileOrNull(path: string): Buffer | null {
  try {
    return existsSync(path) ? readFileSync(path) : null;
  } catch {
    return null;
  }
}

function vertexPercent(current: number, total: number) {
  if (total <= 0) return VERTEX_PROGRESS_START;
  const ratio = Math.max(0, Math.min(1, current / total));
  return VERTEX_PROGRESS_START + (VERTEX_PROGRESS_END - VERTEX_PROGRESS_START) * ratio;
}

function facePercent(current: number, total: number) {
  if (total <= 0) return FACE_PROGRESS_START;
  const ratio = Math.max(0, Math.min(1, current / total));
  return FACE_PROGRESS_START + (FACE_PROGRESS_END - FACE_PROGRESS_START) * ratio;
}

function sameStandoff(a: number, b: number) {
  return Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= 1e-3;
}

/** Build a controller bound to one `SolverSpec`. */
export function createGpuSolver(spec: SolverSpec) {
  let child: ReturnType<typeof Bun.spawn> | null = null;
  let standoffMm = STANDOFF_DEFAULT_MM;
  let densityMode: DensityMode = spec.defaultDensity;
  let status: SolverStatus = {
    state: "idle",
    current: 0,
    total: 0,
    percent: 0,
    message: "Not started",
    error: "",
    canResume: false,
  };

  function modePath(paths: DensityModeMap, density: DensityMode, label: string): string {
    const path = paths[density];
    if (!path) throw new Error(`${spec.label}: missing ${label} path for density ${density}`);
    return path;
  }

  function recordPath(density: DensityMode): string {
    return modePath(spec.outByMode, density, "record");
  }

  function orderPath(density: DensityMode): string {
    return modePath(spec.orderByMode, density, "order");
  }

  function logPath(density: DensityMode): string {
    return modePath(spec.logByMode, density, "log");
  }

  function checkpointPath(density: DensityMode): string {
    return modePath(spec.checkpointByMode, density, "checkpoint");
  }

  const outPath = () => recordPath(densityMode);

  /**
   * A server restart has no client-side density selection to read. Prefer a
   * partially written record so Resume remains reachable, then the most recently
   * touched complete record.
   */
  function bootDensityMode(): DensityMode {
    const modes = Object.keys(spec.outByMode) as DensityMode[];
    let best: { mode: DensityMode; partial: boolean; mtime: number } | null = null;
    let defaultComplete = false;
    for (const mode of modes) {
      const path = modePath(spec.outByMode, mode, "record");
      const cp = recordSummary(readFileOrNull(path) ?? Buffer.alloc(0));
      const vertex = vertexCheckpointFor(mode);
      if (!cp.exists && !vertex.exists) continue;
      if (mode === spec.defaultDensity && cp.done) defaultComplete = true;
      let mtime = 0;
      try {
        if (cp.exists) mtime = statSync(path).mtimeMs;
      } catch {
        /* the record disappeared between existsSync and statSync */
      }
      try {
        const checkpoint = checkpointPath(mode);
        if (vertex.exists) mtime = Math.max(mtime, statSync(checkpoint).mtimeMs);
      } catch {
        /* the checkpoint disappeared between the header read and statSync */
      }
      const candidate = {
        mode,
        partial: (cp.current > 0 && !cp.done)
          || (vertex.exists && vertex.current > 0 && vertex.current < vertex.total),
        mtime,
      };
      if (
        !best
        || (candidate.partial && !best.partial)
        || (candidate.partial === best.partial && candidate.mtime > best.mtime)
      ) {
        best = candidate;
      }
    }
    if (best?.partial) return best.mode;
    if (defaultComplete) return spec.defaultDensity;
    return best?.mode ?? spec.defaultDensity;
  }

  function densityLabel(density: DensityMode) {
    if (density === "constant") return "constant";
    if (density === "elliptic") return "fractional Cauchy";
    return "Cauchy";
  }

  function startMessage(prefix: string, density: DensityMode, mm: number) {
    return `${prefix} ${spec.label} · ${densityLabel(density)} density · ${mm} mm`;
  }

  function payload(): SolverStatusPayload {
    return {
      ...status,
      standoffMm,
      densityMode,
      done: status.state === "done",
      records: Object.fromEntries(
        Object.entries(spec.outByMode).map(([mode, path]) => [
          mode,
          recordSummary(readFileOrNull(path) ?? Buffer.alloc(0)),
        ]),
      ),
    };
  }

  /**
   * Find a bake process that this server did not spawn. The bracket in the
   * pattern keeps `pgrep`'s own argv from matching itself.
   */
  function externalPid(): number | null {
    try {
      const pattern = `[r]tfp-bake.*--solver ${spec.id}([[:space:]]|$)`;
      const out = Bun.spawnSync(["pgrep", "-f", pattern], {
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

  /** Stop any in-flight bake so "Recompute" really restarts it. */
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

  /** The active record's checkpoint, read straight from the file. */
  function checkpoint(summary = recordSummary(readFileOrNull(outPath()) ?? Buffer.alloc(0))) {
    return {
      ...summary,
      canResume: summary.exists && summary.current > 0 && !summary.done,
    };
  }

  function vertexCheckpointFor(
    density: DensityMode,
  ): { exists: boolean; current: number; total: number; standoffMm: number } {
    let fd: number | null = null;
    let buf: Buffer;
    try {
      fd = openSync(checkpointPath(density), "r");
      buf = Buffer.allocUnsafe(CHECKPOINT_HEADER_BYTES);
      if (readSync(fd, buf, 0, CHECKPOINT_HEADER_BYTES, 0) < CHECKPOINT_HEADER_BYTES) {
        return { exists: false, current: 0, total: 0, standoffMm: 0 };
      }
    } catch {
      return { exists: false, current: 0, total: 0, standoffMm: 0 };
    } finally {
      if (fd !== null) closeSync(fd);
    }
    if (buf.subarray(0, 8).toString("ascii") !== "RYVHCP01") {
      return { exists: false, current: 0, total: 0, standoffMm: 0 };
    }
    if (buf.readUInt32LE(8) !== 2) {
      return { exists: false, current: 0, total: 0, standoffMm: 0 };
    }
    const total = Number(buf.readBigUInt64LE(12));
    const current = Number(buf.readBigUInt64LE(20));
    const standoffMm = buf.readDoubleLE(28);
    if (
      !Number.isSafeInteger(total)
      || !Number.isSafeInteger(current)
      || current > total
      || !Number.isFinite(standoffMm)
      || standoffMm <= 0
    ) {
      return { exists: false, current: 0, total: 0, standoffMm: 0 };
    }
    return { exists: current > 0, current, total, standoffMm };
  }

  function vertexCheckpoint(): {
    exists: boolean;
    current: number;
    total: number;
    standoffMm: number;
  } {
    return vertexCheckpointFor(densityMode);
  }

  function parseProgress(text: string) {
    // `PROGRESS_R <vertices done> <vertices total>` is the remainder phase; face
    // lines follow it as the record is written, so both can move the bar.
    const faces = [...text.matchAll(/(?<![A-Z_])faces\s+(\d+)\s+\/\s+(\d+)/g)].pop();
    const prog = [...text.matchAll(/PROGRESS_R\s+(\d+)\s+(\d+)/g)].pop();
    const m = faces ?? prog;
    if (m) {
      const current = Number(m[1]);
      const total = Number(m[2]);
      if (!faces && prog) {
        status.current = current;
        status.total = total;
        status.percent = vertexPercent(current, total);
        status.message = `GPU vertices ${current} / ${total}`;
      } else {
        status.current = current;
        status.total = total;
        status.percent = facePercent(current, total);
        status.message =
          current >= total && total > 0
            ? `Faces ${current} / ${total} (writing)`
            : `Faces ${current} / ${total}`;
      }
    } else if (/analytic tensor/.test(text)) {
      status.message = "Analytic near-field tensor (GPU)";
      status.percent = Math.max(status.percent, VERTEX_PROGRESS_START);
    } else {
      const evaluating = [...text.matchAll(/evaluating\s+(\d+)\s+vertices/g)].pop();
      if (evaluating && status.current === 0) {
        status.total = Number(evaluating[1]);
        status.current = 0;
        status.percent = 0;
        status.message = `GPU points 0 / ${evaluating[1]}`;
      } else if (/^solver=|^jump surfaces:/m.test(text)) {
        status.message = spec.id === "carlson"
          ? "Building the jump-surface list"
          : spec.id === "carlson-alpha"
            ? "Preparing radial finite-part buffers"
            : "Preparing GPU buffers";
        status.percent = Math.max(status.percent, VERTEX_PROGRESS_START);
      }
    }
    if (/wrote\s+\d+\s+dense face/.test(text)) {
      // Completion is decided by the on-disk record, never by this log line.
      status.message = "Verifying the record";
    }
  }

  function applyCheckpoint() {
    const cp = checkpoint();
    const vertex = vertexCheckpoint();
    const running = isRunning();
    if (cp.standoffMm > 0) standoffMm = cp.standoffMm;
    else if (vertex.standoffMm > 0) standoffMm = vertex.standoffMm;
    // Never advertise resume while a bake process is alive.
    const vertexResumable = vertex.exists && vertex.current < vertex.total;
    status.canResume = running ? false : cp.canResume || vertexResumable;
    if (!cp.exists) {
      if (running && status.percent >= 100) status.percent = 0;
      if (!running && vertexResumable) {
        status.state = "idle";
        status.message = `Resumable · vertices ${vertex.current} / ${vertex.total}`;
        status.current = vertex.current;
        status.total = vertex.total;
        status.percent = vertexPercent(vertex.current, vertex.total);
      } else if (!running) {
        status = {
          state: "idle",
          current: 0,
          total: 0,
          percent: 0,
          message: "Not started",
          error: "",
          canResume: false,
        };
      }
      return;
    }
    if (cp.done && !running) {
      status.state = "done";
      status.message = "Bake complete (record already on disk)";
      status.percent = 100;
      status.current = cp.total;
      status.total = cp.total;
      status.canResume = false;
    } else if (running) {
      status.state = "running";
      if (status.current === 0 && status.total === 0) {
        if (vertex.current > 0) {
          status.current = vertex.current;
          status.total = vertex.total;
          status.percent = vertexPercent(vertex.current, vertex.total);
          status.message = `GPU vertices ${vertex.current} / ${vertex.total}`;
        } else if (cp.current > 0) {
          status.current = cp.current;
          status.total = cp.total;
          status.percent = facePercent(cp.current, cp.total);
          status.message = `Faces ${cp.current} / ${cp.total}`;
        }
      }
    } else if (cp.canResume) {
      status.state = "idle";
      status.message = `Resumable · faces ${cp.current} / ${cp.total}`;
      status.current = cp.current;
      status.total = cp.total;
      status.percent = facePercent(cp.current, cp.total);
    } else if (vertexResumable) {
      status.state = "idle";
      status.message = `Resumable · vertices ${vertex.current} / ${vertex.total}`;
      status.current = vertex.current;
      status.total = vertex.total;
      status.percent = vertexPercent(vertex.current, vertex.total);
    } else if (!running) {
      status = {
        state: "idle",
        current: 0,
        total: cp.total || 0,
        percent: 0,
        message: "Not started",
        error: "",
        canResume: false,
      };
    }
  }

  function clearRecords() {
    const checkpoint = checkpointPath(densityMode);
    const paths = [
      outPath(),
      orderPath(densityMode),
      logPath(densityMode),
      checkpoint,
      `${checkpoint}.tmp`,
    ];
    for (const p of paths) {
      try {
        if (existsSync(p)) unlinkSync(p);
      } catch {
        /* ignore */
      }
    }
  }

  function boot() {
    densityMode = bootDensityMode();
    const log = readFileOrNull(logPath(densityMode));
    if (log) {
      try {
        parseProgress(log.toString("utf8"));
      } catch {
        /* ignore */
      }
    }
    applyCheckpoint();
    if (!isRunning()) {
      const cp = checkpoint();
      if (cp.done) {
        status.state = "done";
        status.message = "Bake complete (record already on disk)";
      } else if (cp.canResume) {
        status.state = "idle";
        status.message = `Resumable · faces ${cp.current} / ${cp.total}`;
      } else if (status.state === "running") {
        status.state = "idle";
        status.message = "Not started";
      }
    }
  }

  function statusResponse(): Response {
    const log = readFileOrNull(logPath(densityMode));
    if (log) {
      try {
        parseProgress(log.toString("utf8"));
      } catch {
        /* ignore */
      }
    }
    applyCheckpoint();
    if (isRunning()) {
      status.state = "running";
    } else if (status.state === "running") {
      const cp = checkpoint();
      status.state = cp.done ? "done" : "idle";
      if (cp.canResume) status.message = `Resumable · faces ${cp.current} / ${cp.total}`;
    }
    return corsJson(payload());
  }

  function fail(message: string, error: string, httpStatus = 500): Response {
    status = { ...status, state: "error", message, error };
    return corsJson(payload(), httpStatus);
  }

  async function start(
    mode: "restart" | "resume",
    requestedStandoffMm?: number,
    requestedDensity?: string,
  ): Promise<Response> {
    if (!isRunning() && status.state === "running") {
      child = null;
      applyCheckpoint();
      const cp = checkpoint();
      status.state = cp.done ? "done" : "idle";
      if (cp.canResume) status.message = `Resumable · faces ${cp.current} / ${cp.total}`;
    }
    if (isRunning()) {
      if (mode === "resume") {
        applyCheckpoint();
        return corsJson({ ok: true, ...payload(), alreadyRunning: true });
      }
      stop();
    }
    if (!existsSync(LAUNCHER_BIN)) {
      return fail(
        `${spec.label} bake program missing`,
        `missing ${LAUNCHER_BIN}; run \`bun run rtfp:build\` first`,
      );
    }
    const obj = DEFAULT_OBJ;
    if (!existsSync(obj)) return fail("Mesh missing", `missing OBJ: ${obj}`);
    // Resume and restart both use the requested density and height. A resume
    // only reuses a checkpoint produced on the exact same observation surface.
    // An unsupported density is an error rather than a silent fallback.
    const requested = requestedDensity ?? densityMode;
    if (requestedDensity !== undefined && !(requestedDensity in spec.outByMode)) {
      return fail(
        "Density mode unsupported",
        `${spec.label} cannot evaluate density ${JSON.stringify(requestedDensity)}`,
        400,
      );
    }
    const requestedMode = requested as DensityMode;
    const nextDensity: DensityMode = spec.outByMode[requestedMode]
      ? requestedMode
      : spec.defaultDensity;
    const densityToml = modePath(spec.densityByMode, nextDensity, "density TOML");
    if (!existsSync(densityToml)) {
      return fail("Density TOML missing", `missing ${densityToml}`);
    }
    const nextOutPath = recordPath(nextDensity);
    const record = recordSummary(readFileOrNull(nextOutPath) ?? Buffer.alloc(0));
    const vertex = vertexCheckpointFor(nextDensity);
    const savedStandoffMm = record.standoffMm > 0 ? record.standoffMm : vertex.standoffMm;
    const usedStandoffMm = clampStandoffMm(requestedStandoffMm ?? standoffMm);
    standoffMm = usedStandoffMm;
    densityMode = nextDensity;

    if (mode === "resume") {
      if (savedStandoffMm > 0 && !sameStandoff(savedStandoffMm, usedStandoffMm)) {
        return fail(
          "Observation height changed",
          `saved result is at ${savedStandoffMm} mm; recompute at ${usedStandoffMm} mm`,
          409,
        );
      }
      const cp = record;
      const resumable = (cp.exists && cp.current > 0 && !cp.done)
        || (vertex.exists && vertex.current > 0 && vertex.current <= vertex.total);
      if (!resumable && !cp.done) {
        return fail(
          "No resumable record",
          `press "Recompute" first, or check ${nextOutPath.replace(`${ROOT}/`, "")}`,
          400,
        );
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
      clearRecords();
      await Bun.write(outPath(), blankRecord(FACE_COUNT, usedStandoffMm));
    }

    await Bun.write(logPath(nextDensity), "");
    const vertex0 = vertexCheckpoint();
    const record0 = checkpoint();
    const resumeFromVertex = mode === "resume" && vertex0.current > 0;
    const resumeFromFaces = mode === "resume" && record0.current > 0;
    status = {
      state: "running",
      current: resumeFromVertex
        ? vertex0.current
        : resumeFromFaces
          ? record0.current
          : 0,
      total: resumeFromVertex
        ? vertex0.total
        : resumeFromFaces
          ? record0.total
          : FACE_COUNT,
      percent: resumeFromVertex
        ? vertexPercent(vertex0.current, vertex0.total)
        : resumeFromFaces
          ? facePercent(record0.current, record0.total)
          : 0,
      message: startMessage(mode === "resume" ? "Resuming" : "Starting", nextDensity, usedStandoffMm),
      error: "",
      canResume: false,
    };

    const args = [
      LAUNCHER_BIN,
      "--obj",
      obj,
      "--out",
      outPath(),
      "--order",
      orderPath(nextDensity),
      "--checkpoint",
      checkpointPath(nextDensity),
      "--density",
      densityToml,
      "--mode",
      nextDensity,
      "--solver",
      spec.id,
      "--standoff-mm",
      String(usedStandoffMm),
      ...spec.extraArgs,
      ...(spec.modeArgs[nextDensity] ?? []),
    ];
    if (mode === "resume") args.push("--resume");

    const proc = Bun.spawn(args, { cwd: ROOT, stdout: "pipe", stderr: "pipe" });
    child = proc;

    (async () => {
      let buf = "";
      let lastFlush = 0;
      // Serialize merges — parallel stdout/stderr `buf +=` races freeze progress.
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
              if (buf.length > MAX_CAPTURED_LOG_CHARS) {
                buf = buf.slice(-MAX_CAPTURED_LOG_CHARS);
              }
              parseProgress(buf);
              status.state = "running";
              const now = Date.now();
              if (now - lastFlush >= LOG_FLUSH_INTERVAL_MS) {
                lastFlush = now;
                await Bun.write(logPath(nextDensity), buf);
              }
            });
            await chain;
          }
        };
        return pump();
      };
      await Promise.all([append(proc.stdout), append(proc.stderr)]);
      await Bun.write(logPath(nextDensity), buf);
      const code = await proc.exited;
      child = null;
      const log = readFileOrNull(logPath(nextDensity));
      if (log) {
        try {
          parseProgress(log.toString("utf8"));
        } catch {
          /* ignore */
        }
      }
      applyCheckpoint();
      if (code === 0) {
        const cpDone = checkpoint();
        if (cpDone.done) {
          status.state = "done";
          status.percent = 100;
          status.current = cpDone.total;
          status.total = cpDone.total;
          status.message = "Bake complete, record saved";
          status.error = "";
          status.canResume = false;
        } else if (cpDone.canResume) {
          status.state = "idle";
          status.message = `Resumable · faces ${cpDone.current} / ${cpDone.total}`;
        } else {
          status.state = "error";
          status.message = "Bake ended with an incomplete record";
          status.error = "exit 0 but incomplete checkpoint";
        }
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

    return corsJson({ ok: true, ...payload(), mode, args });
  }

  return { spec, boot, statusResponse, start };
}
