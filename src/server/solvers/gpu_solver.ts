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
import { existsSync, readFileSync, unlinkSync } from "node:fs";
import { join } from "node:path";
import {
  RHGF_HEADER,
  STANDOFF_DEFAULT_MM,
  STANDOFF_MAX_MM,
  blankRecord,
  clampStandoffMm,
  recordSummary,
  type RecordSummary,
} from "./record";

export const ROOT = join(import.meta.dir, "../../..");
export const LAUNCHER_BIN = join(ROOT, "target/release/rtfp-bake");
export const DEFAULT_OBJ =
  process.env.BAKE_OBJ ?? join(ROOT, "../Ryugu_wasm/assets/models/SHAPE_SFM_200k_v20180804.obj");
export const DENSITY_TOML = join(ROOT, "assets/density/cauchy.toml");

/** Face count of the Ryugu observation mesh, and the record's fixed size. */
export const FACE_COUNT = 196608;

export type DensityMode = "cauchy" | "constant";

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
  records: Record<DensityMode, RecordSummary>;
};

export type SolverSpec = {
  /** `--solver` value handed to the binary. */
  id: "ray" | "carlson";
  /** Human-readable name used in status messages. */
  label: string;
  /** Record written for each density mode. */
  outByMode: Record<DensityMode, string>;
  /** Progressive write order, so a resumed run keeps the same face sequence. */
  order: string;
  /** stdout capture of the current run. */
  log: string;
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
    return existsSync(path) ? Buffer.from(readFileSync(path)) : null;
  } catch {
    return null;
  }
}

/** Build a controller bound to one `SolverSpec`. */
export function createGpuSolver(spec: SolverSpec) {
  let child: ReturnType<typeof Bun.spawn> | null = null;
  let standoffMm = STANDOFF_DEFAULT_MM;
  let densityMode: DensityMode = "cauchy";
  let status: SolverStatus = {
    state: "idle",
    current: 0,
    total: 0,
    percent: 0,
    message: "Not started",
    error: "",
    canResume: false,
  };

  const outPath = () => spec.outByMode[densityMode];

  function startMessage(prefix: string, density: DensityMode, mm: number) {
    return `${prefix} ${spec.label} · ${density === "constant" ? "constant" : "Cauchy"} density · ${mm} mm`;
  }

  function payload(): SolverStatusPayload {
    return {
      ...status,
      standoffMm,
      densityMode,
      done: status.state === "done",
      records: {
        cauchy: recordSummary(readFileOrNull(spec.outByMode.cauchy) ?? Buffer.alloc(0)),
        constant: recordSummary(readFileOrNull(spec.outByMode.constant) ?? Buffer.alloc(0)),
      },
    };
  }

  /**
   * Find a bake process that this server did not spawn. The bracket in the
   * pattern keeps `pgrep`'s own argv from matching itself.
   */
  function externalPid(): number | null {
    try {
      const out = Bun.spawnSync(["pgrep", "-f", "[r]tfp-bake"], {
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

  function parseProgress(text: string) {
    // `PROGRESS_R <vertices done> <vertices total>` is the remainder phase; face
    // lines follow it as the record is written, so both can move the bar.
    const faces = [...text.matchAll(/(?<![A-Z_])faces\s+(\d+)\s+\/\s+(\d+)/g)].pop();
    const prog = [...text.matchAll(/PROGRESS_R\s+(\d+)\s+(\d+)/g)].pop();
    const m = faces ?? prog;
    if (m) {
      const current = Number(m[1]);
      const total = Number(m[2]);
      status.current = current;
      status.total = total;
      status.percent =
        total > 0 ? Math.min(current >= total ? 99.9 : (100 * current) / total, 99.9) : 0;
      status.message =
        current >= total && total > 0
          ? `Faces ${current} / ${total} (writing)`
          : `Faces ${current} / ${total}`;
    }
    if (/analytic tensor/.test(text)) {
      status.message = "Analytic near-field tensor (GPU)";
      status.percent = Math.max(status.percent, 0.5);
    }
    if (/^solver=|^jump surfaces:/.test(text)) {
      status.message = "Building the jump-surface list";
      status.percent = Math.max(status.percent, 0.5);
    }
    if (/wrote\s+\d+\s+dense face/.test(text)) {
      // Completion is decided by the on-disk record, never by this log line.
      status.message = "Verifying the record";
    }
  }

  function applyCheckpoint() {
    const cp = checkpoint();
    const running = isRunning();
    if (cp.standoffMm > 0) standoffMm = cp.standoffMm;
    // Never advertise resume while a bake process is alive.
    status.canResume = running ? false : cp.canResume;
    if (!cp.exists) {
      if (running && status.percent >= 100) status.percent = 0;
      return;
    }
    const logOwnsBar = /^Faces /.test(status.message || "");
    if (!logOwnsBar) {
      status.current = cp.current;
      status.total = cp.total;
      status.percent = cp.total > 0 ? Math.min(100, (100 * cp.current) / cp.total) : 0;
    } else {
      const cur = Math.max(status.current, cp.current);
      const tot = Math.max(status.total, cp.total);
      status.current = cur;
      status.total = tot;
      if (!cp.done) status.percent = tot > 0 ? Math.min(99.9, (100 * cur) / tot) : 0;
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
      if (!cp.done && status.percent >= 100) {
        status.percent = cp.total > 0 ? Math.min(99.9, (100 * cp.current) / cp.total) : 0;
      }
    } else if (cp.canResume) {
      status.state = "idle";
      status.message = `Resumable · faces ${cp.current} / ${cp.total}`;
      status.current = cp.current;
      status.total = cp.total;
      status.percent = cp.total > 0 ? Math.min(99.9, (100 * cp.current) / cp.total) : 0;
    }
  }

  function clearRecords() {
    for (const p of [outPath(), spec.order, spec.log]) {
      try {
        if (existsSync(p)) unlinkSync(p);
      } catch {
        /* ignore */
      }
    }
  }

  function boot() {
    applyCheckpoint();
    const log = readFileOrNull(spec.log);
    if (log) {
      try {
        parseProgress(log.toString("utf8"));
      } catch {
        /* ignore */
      }
    }
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
    const log = readFileOrNull(spec.log);
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
    if (!existsSync(DENSITY_TOML)) {
      return fail("Density TOML missing", `missing ${DENSITY_TOML}`);
    }

    // A resume continues the record's own density mode and height; a restart
    // takes both from the UI.
    const nextDensity: DensityMode =
      mode === "resume"
        ? densityMode
        : requestedDensity === "constant"
          ? "constant"
          : "cauchy";
    const record = checkpoint();
    const usedStandoffMm =
      mode === "resume" && record.standoffMm > 0
        ? record.standoffMm
        : clampStandoffMm(requestedStandoffMm ?? standoffMm);
    standoffMm = usedStandoffMm;
    densityMode = nextDensity;

    if (mode === "resume") {
      const cp = recordSummary(
        readFileOrNull(spec.outByMode[nextDensity]) ?? Buffer.alloc(0),
      );
      const resumable = cp.exists && cp.current > 0 && !cp.done;
      if (!resumable && !cp.done) {
        return fail(
          "No resumable record",
          `press "Recompute" first, or check ${
            spec.outByMode[nextDensity].replace(`${ROOT}/`, "")
          }`,
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

    await Bun.write(spec.log, "");
    const cp0 = checkpoint();
    status = {
      state: "running",
      current: cp0.current || 0,
      total: cp0.total || FACE_COUNT,
      percent:
        mode === "restart"
          ? 0
          : cp0.total && cp0.current
            ? Math.min(99.9, (100 * cp0.current) / cp0.total)
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
      spec.order,
      "--density",
      DENSITY_TOML,
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
              await Bun.write(spec.log, buf);
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
      const log = readFileOrNull(spec.log);
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

export type GpuSolver = ReturnType<typeof createGpuSolver>;
