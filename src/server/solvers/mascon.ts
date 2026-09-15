/**
 * Mascon solver endpoint — the mainstream voxel direct-summation baseline.
 *
 * A mascon ("mass concentration") model replaces the body with a regular grid
 * of point masses, so the gravity-gradient tensor is a plain O(N) sum over
 * 192^3 = 7.1e6 voxels per evaluation point. It is the algorithm most
 * commonly used in flight dynamics, which is exactly why it is here: it is the
 * reference the reader is presumed to already trust.
 *
 * It shares no machinery with the other solvers — different binary, different
 * progress dialect, different failure modes — but it writes the same RHGF v5
 * record, so the viewer and the comparison panel treat it identically.
 */
import { existsSync, readFileSync, unlinkSync } from "fs";
import { join } from "path";
import { corsJson } from "./gpu_solver";

const ROOT = join(import.meta.dir, "../../..");
const MASCON_BIN = join(ROOT, "bakes/build/ryugu_mascon_bake_cpp");
const MASCON_OBJ =
  Bun.env.BAKE_OBJ ??
  join(ROOT, "../Ryugu_wasm/assets/models/SHAPE_SFM_200k_v20180804.obj");
const MASCON_OUT = join(ROOT, "assets/records/mascon_faces.bin");
const MASCON_ORDER = join(ROOT, "assets/records/.mascon_order.bin");
const MASCON_LOG = join(ROOT, "assets/records/.mascon_progress.log");
const MASCON_DENSITY = join(ROOT, "assets/density/cauchy.toml");
const MASCON_MAGIC = 0x52484746;
/**
 * 192³ halved the discretisation gap against RT-FP (0.74 % → 0.31 % median at a
 * 16 m standoff); 256³ is better still but 8× slower than 128³.
 */
const MASCON_GRID = Number(Bun.env.MASCON_GRID ?? 192);
/**
 * Slider bounds: 1 mm – 32 m. Mascon is a voxel direct sum with 7.9 m cells, so
 * 1 mm is meaningless for it: at 1 mm the field is ~250 % off, at 16 m (two cells
 * out) the residual against RT-FP is 0.76 % median.
 */
const MASCON_STANDOFF_DEFAULT_MM = 16000;
const MASCON_STANDOFF_MAX_MM = 32000;

export type MasconStatus = {
  state: "idle" | "running" | "done" | "error";
  current: number;
  total: number;
  percent: number;
  message: string;
  error: string;
  canResume: boolean;
};

let masconChild: ReturnType<typeof Bun.spawn> | null = null;
/** Observation-surface height (mm) of the current record / run. */
let masconStandoffMm = MASCON_STANDOFF_DEFAULT_MM;
let masconStatus: MasconStatus = {
  state: "idle",
  current: 0,
  total: 0,
  percent: 0,
  message: "Not started",
  error: "",
  canResume: false,
};

function clampStandoffMm(mm: unknown): number {
  const v = Number(mm);
  if (!Number.isFinite(v)) return MASCON_STANDOFF_DEFAULT_MM;
  return Math.min(MASCON_STANDOFF_MAX_MM, Math.max(1, v));
}

/** Status payload plus the observation height the record was baked at. */
function masconPayload(): MasconStatus & { standoffMm: number; done: boolean } {
  return { ...masconStatus, standoffMm: masconStandoffMm, done: masconStatus.state === "done" };
}

function externalMasconPid(): number | null {
  try {
    // Bracket trick so this pgrep's own argv does not match the pattern.
    const out = Bun.spawnSync(["pgrep", "-f", "[r]yugu_mascon_bake_cpp"], {
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

function isMasconRunning() {
  // Mirror Werner: trust our spawned child first. pgrep-only was wrongly
  // reporting "not running", which flipped status to "record already complete"
  // mid-bake
  // whenever a complete (or stale-complete) checkpoint was on disk — and also
  // allowed a second bake to start and truncate the live file.
  if (masconChild) {
    if (masconChild.exitCode !== null) {
      masconChild = null;
    } else {
      try {
        process.kill(masconChild.pid, 0);
        return true;
      } catch {
        masconChild = null;
      }
    }
  }
  return externalMasconPid() !== null;
}


/** Kill any in-flight mascon bake so "Recompute" really restarts it. */
function stopMasconBake() {
  const pids = new Set<number>();
  if (masconChild) {
    try {
      pids.add(masconChild.pid);
    } catch {
      /* ignore */
    }
    masconChild = null;
  }
  const ext = externalMasconPid();
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

function parseMasconProgress(text: string) {
  const faces = [...text.matchAll(/(?<![A-Z_])faces\s+(\d+)\s+\/\s+(\d+)/g)].pop();
  const prog = [...text.matchAll(/(?<![A-Z_])PROGRESS\s+(\d+)\s+(\d+)/g)].pop();
  const voxel = [...text.matchAll(/voxelize z\s+(\d+)\s+\/\s+(\d+)/g)].pop();
  if (faces || prog) {
    const m = faces ?? prog!;
    const current = Number(m[1]);
    const total = Number(m[2]);
    masconStatus.current = current;
    masconStatus.total = total;
    // Face phase owns the bar; never report 100% until disk checkpoint is complete.
    masconStatus.percent =
      total > 0 ? Math.min(current >= total ? 99.9 : (100 * current) / total, 99.9) : 0;
    if (current >= total && total > 0) {
      // Leave final 100% / done to applyMasconCheckpoint (finite == total on disk).
      masconStatus.message = `Faces ${current} / ${total} (writing back)`;
    } else {
      masconStatus.message = `Faces ${current} / ${total}`;
    }
  } else if (voxel) {
    const vz = Number(voxel[1]);
    const vt = Number(voxel[2]);
    // Prelude only: grid z-slices must NOT drive the main bar to 100% (that was the
    // "jumps to 100% halfway" bug — voxelize 128/128 looked like a finished bake).
    const faceTotal = masconStatus.total > vt ? masconStatus.total : 196608;
    masconStatus.total = faceTotal;
    masconStatus.current = 0;
    masconStatus.percent = vt > 0 ? Math.min(2.5, (2.5 * vz) / vt) : 0;
    masconStatus.message = `Voxelising ${vz} / ${vt}`;
  }
  if (/wrote\s+\d+\s+dense face/.test(text)) {
    // Do NOT set state=done / percent=100 here — the log line can appear while the
    // process is still exiting, and a stale "wrote" must not override an incomplete
    // checkpoint. applyMasconCheckpoint is the sole authority for completion.
    masconStatus.message = "Verifying the record...";
  }
}

function readMasconCheckpoint(): {
  ok: boolean;
  current: number;
  total: number;
  done: boolean;
  canResume: boolean;
  standoffMm: number;
} {
  if (!existsSync(MASCON_OUT)) {
    return { ok: false, current: 0, total: 0, done: false, canResume: false, standoffMm: 0 };
  }
  try {
    const buf = Buffer.from(readFileSync(MASCON_OUT));
    if (buf.byteLength < 28) {
      return { ok: false, current: 0, total: 0, done: false, canResume: false, standoffMm: 0 };
    }
    const magic = buf.readUInt32LE(0);
    const version = buf.readUInt32LE(4);
    const total = buf.readUInt32LE(8);
    const headerDone = buf.readUInt32LE(12);
    if (magic !== MASCON_MAGIC || version !== 5 || total === 0) {
      return { ok: false, current: 0, total: 0, done: false, canResume: false, standoffMm: 0 };
    }
    const need = 28 + total * 4;
    if (buf.byteLength < need) {
      return { ok: false, current: 0, total, done: false, canResume: false, standoffMm: 0 };
    }
    // Reserved v5 header slot: observation height in mm (0 = not recorded).
    const rawMm = buf.readFloatLE(16);
    const standoffMm = Number.isFinite(rawMm) && rawMm >= 1 && rawMm <= MASCON_STANDOFF_MAX_MM ? rawMm : 0;
    let finite = 0;
    for (let i = 0; i < total; i++) {
      const s = buf.readFloatLE(28 + i * 4);
      if (Number.isFinite(s)) finite++;
    }
    // Progress may use headerCompleted as a hint, but "done" must mean every
    // face scalar is finite — header can briefly disagree during a flush.
    const current = Math.min(total, Math.max(finite, headerDone));
    const done = finite >= total && total > 0;
    return { ok: true, current, total, done, canResume: finite > 0 && !done, standoffMm };
  } catch {
    return { ok: false, current: 0, total: 0, done: false, canResume: false, standoffMm: 0 };
  }
}

function applyMasconCheckpoint() {
  const cp = readMasconCheckpoint();
  const running = isMasconRunning();
  if (cp.standoffMm > 0) masconStandoffMm = cp.standoffMm;
  // Mirror Werner: never advertise resume while a bake process is alive.
  masconStatus.canResume = running ? false : cp.canResume;
  if (!cp.ok) {
    // Incomplete / missing file while running: never leave a stale 100% bar.
    if (running && masconStatus.percent >= 100) masconStatus.percent = 0;
    return;
  }
  const logSaysFaces = /^Faces /.test(masconStatus.message || "");
  const logSaysVoxel = /^Voxelising /.test(masconStatus.message || "");
  if (logSaysVoxel && running) {
    // Keep prelude percent from parseMasconProgress; once faces exist on disk,
    // prefer checkpoint face counts so the bar never sticks at voxel 100%.
    if (cp.current > 0) {
      masconStatus.current = cp.current;
      masconStatus.total = cp.total;
      masconStatus.percent =
        cp.total > 0 ? Math.min(99.9, (100 * cp.current) / cp.total) : 0;
      masconStatus.message = `Faces ${cp.current} / ${cp.total}`;
    }
  } else if (!logSaysFaces && !logSaysVoxel) {
    masconStatus.current = cp.current;
    masconStatus.total = cp.total;
    masconStatus.percent =
      cp.total > 0 ? Math.min(100, (100 * cp.current) / cp.total) : 0;
  } else if (logSaysFaces && cp.ok) {
    // Prefer denser of log vs disk; clamp below 100 until truly done.
    const cur = Math.max(masconStatus.current, cp.current);
    const tot = Math.max(masconStatus.total, cp.total);
    masconStatus.current = cur;
    masconStatus.total = tot;
    if (!cp.done) {
      masconStatus.percent = tot > 0 ? Math.min(99.9, (100 * cur) / tot) : 0;
    }
  }
  if (cp.done && !running) {
    masconStatus.state = "done";
    masconStatus.message = "Bake complete (record already on disk)";
    masconStatus.percent = 100;
    masconStatus.current = cp.total;
    masconStatus.total = cp.total;
    masconStatus.canResume = false;
  } else if (running) {
    masconStatus.state = "running";
    // Hard rule: incomplete checkpoint ⇒ bar must not read as finished.
    if (!cp.done && masconStatus.percent >= 100) {
      masconStatus.percent =
        cp.total > 0 ? Math.min(99.9, (100 * cp.current) / cp.total) : 0;
    }
  } else if (cp.canResume) {
    masconStatus.state = "idle";
    masconStatus.message = `Resumable · faces ${cp.current} / ${cp.total}`;
    masconStatus.current = cp.current;
    masconStatus.total = cp.total;
    masconStatus.percent =
      cp.total > 0 ? Math.min(99.9, (100 * cp.current) / cp.total) : 0;
  }
}

function clearMasconRecords() {
  for (const p of [MASCON_OUT, MASCON_ORDER, MASCON_LOG]) {
    try {
      if (existsSync(p)) unlinkSync(p);
    } catch {
      /* ignore */
    }
  }
}

async function writeMasconStub(totalFaces = 196608, standoffMm = masconStandoffMm) {
  const buf = Buffer.alloc(28 + totalFaces * 4);
  buf.writeUInt32LE(MASCON_MAGIC, 0);
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
  await Bun.write(MASCON_OUT, buf);
}

export function bootMasconStatus() {
  applyMasconCheckpoint();
  if (existsSync(MASCON_LOG)) {
    try {
      parseMasconProgress(readFileSync(MASCON_LOG, "utf8"));
    } catch {
      /* ignore */
    }
  }
  if (!isMasconRunning()) {
    const cp = readMasconCheckpoint();
    if (cp.done) {
      masconStatus.state = "done";
      masconStatus.message = "Bake complete (record already on disk)";
    } else if (cp.canResume) {
      masconStatus.state = "idle";
      masconStatus.message = `Resumable · faces ${cp.current} / ${cp.total}`;
    } else if (masconStatus.state === "running") {
      masconStatus.state = "idle";
      masconStatus.message = "Not started";
    }
  }
}

export function masconStatusResponse(): Response {
  if (existsSync(MASCON_LOG)) {
    try {
      parseMasconProgress(readFileSync(MASCON_LOG, "utf8"));
    } catch {
      /* ignore */
    }
  }
  applyMasconCheckpoint();
  if (isMasconRunning()) masconStatus.state = "running";
  else if (masconStatus.state === "running") {
    const cp = readMasconCheckpoint();
    masconStatus.state = cp.done ? "done" : "idle";
    if (cp.canResume) {
      masconStatus.message = `Resumable · faces ${cp.current} / ${cp.total}`;
    }
  }
  return corsJson(masconPayload());
}

export async function startMasconBake(
  mode: "restart" | "resume",
  requestedStandoffMm?: number,
): Promise<Response> {
  // Drop stale in-memory "running" when the OS process is already gone.
  if (!isMasconRunning() && masconStatus.state === "running") {
    masconChild = null;
    applyMasconCheckpoint();
    const cp = readMasconCheckpoint();
    masconStatus.state = cp.done ? "done" : "idle";
    if (cp.canResume) {
      masconStatus.message = `Resumable · faces ${cp.current} / ${cp.total}`;
    }
  }
  if (isMasconRunning()) {
    if (mode === "resume") {
      applyMasconCheckpoint();
      return corsJson({ ok: true, ...masconPayload(), alreadyRunning: true });
    }
    // Recompute: kill the old process first, then clear and restart. Otherwise
    // the UI sticks on stale progress and then jumps to 100%.
    stopMasconBake();
  }
  if (!existsSync(MASCON_BIN)) {
    masconStatus = {
      state: "error",
      current: 0,
      total: 0,
      percent: 0,
      message: "Mascon bake program missing",
      error: `missing ${MASCON_BIN}; run \`bun run bakes:build\` first`,
      canResume: false,
    };
    return corsJson(masconPayload(), 500);
  }
  if (!existsSync(MASCON_OBJ)) {
    masconStatus = {
      state: "error",
      current: 0,
      total: 0,
      percent: 0,
      message: "Mesh missing",
      error: `missing OBJ: ${MASCON_OBJ}`,
      canResume: false,
    };
    return corsJson(masconPayload(), 500);
  }
  if (!existsSync(MASCON_DENSITY)) {
    masconStatus = {
      state: "error",
      current: 0,
      total: 0,
      percent: 0,
      message: "Density TOML missing",
      error: `missing ${MASCON_DENSITY}`,
      canResume: false,
    };
    return corsJson(masconPayload(), 500);
  }

  // Observation height: a resume has to continue the record's own surface, a
  // restart starts at the height the UI slider asked for (1 mm – 32 m).
  const recordMm = readMasconCheckpoint();
  const usedStandoffMm =
    mode === "resume" && recordMm.standoffMm > 0
      ? recordMm.standoffMm
      : clampStandoffMm(requestedStandoffMm ?? masconStandoffMm);
  masconStandoffMm = usedStandoffMm;

  if (mode === "resume") {
    const cp = readMasconCheckpoint();
    if (!cp.canResume && !cp.done) {
      masconStatus = {
        state: "error",
        current: cp.current,
        total: cp.total,
        percent: 0,
        message: "No resumable record",
        error: 'press "Recompute" first, or check assets/records/mascon_faces.bin',
        canResume: false,
      };
      return corsJson(masconPayload(), 400);
    }
    if (cp.done) {
      masconStatus = {
        state: "done",
        current: cp.current,
        total: cp.total,
        percent: 100,
        message: "Record already complete",
        error: "",
        canResume: false,
      };
      return corsJson({ ok: true, ...masconPayload() });
    }
  } else {
    clearMasconRecords();
    await writeMasconStub(196608, usedStandoffMm);
  }

  await Bun.write(MASCON_LOG, "");
  const cp0 = mode === "resume" ? readMasconCheckpoint() : { current: 0, total: 196608 };
  masconStatus = {
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
      mode === "resume"
        ? `Resuming mascon voxel direct sum · standoff ${usedStandoffMm} mm`
        : `Starting mascon voxel direct sum · standoff ${usedStandoffMm} mm`,
    error: "",
    canResume: false,
  };

  const args = [
    MASCON_BIN,
    "--obj",
    MASCON_OBJ,
    "--out",
    MASCON_OUT,
    "--order",
    MASCON_ORDER,
    "--density",
    MASCON_DENSITY,
    "--grid",
    String(MASCON_GRID),
    "--standoff-mm",
    String(usedStandoffMm),
  ];
  if (mode === "resume") args.push("--resume");

  const proc = Bun.spawn(args, {
    cwd: ROOT,
    stdout: "pipe",
    stderr: "pipe",
  });
  masconChild = proc;

  (async () => {
    let buf = "";
    // Serialize merges — parallel stdout/stderr buf+= races froze progress then jumped to 100%.
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
            await Bun.write(MASCON_LOG, buf);
            parseMasconProgress(buf);
            applyMasconCheckpoint();
            if (isMasconRunning()) masconStatus.state = "running";
          });
          await chain;
        }
      };
      return pump();
    };
    await Promise.all([append(proc.stdout), append(proc.stderr)]);
    const code = await proc.exited;
    masconChild = null;
    if (existsSync(MASCON_LOG)) {
      try {
        parseMasconProgress(readFileSync(MASCON_LOG, "utf8"));
      } catch {
        /* ignore */
      }
    }
    applyMasconCheckpoint();
    if (code === 0) {
      const cpDone = readMasconCheckpoint();
      if (cpDone.done) {
        masconStatus.state = "done";
        masconStatus.percent = 100;
        masconStatus.current = cpDone.total;
        masconStatus.total = cpDone.total;
        masconStatus.message = "Bake complete, record saved";
        masconStatus.error = "";
        masconStatus.canResume = false;
      } else {
        applyMasconCheckpoint();
        masconStatus.state = cpDone.canResume ? "idle" : "error";
        masconStatus.message = cpDone.canResume
          ? `Resumable · faces ${cpDone.current} / ${cpDone.total}`
          : "Bake ended with an incomplete record";
        masconStatus.error = cpDone.canResume ? "" : "exit 0 but incomplete checkpoint";
      }
    } else if (masconStatus.state === "running") {
      applyMasconCheckpoint();
      masconStatus.state = "error";
      masconStatus.message = masconStatus.canResume ? "Bake interrupted (resumable)" : "Bake failed";
      masconStatus.error = `exit ${code}`;
    }
  })().catch((e) => {
    masconChild = null;
    applyMasconCheckpoint();
    masconStatus.state = "error";
    masconStatus.message = masconStatus.canResume ? "Bake aborted (resumable)" : "Bake aborted";
    masconStatus.error = String(e);
  });

  return corsJson({ ok: true, ...masconPayload(), mode, grid: MASCON_GRID });
}
