/**
 * RHGF v5 record I/O — the on-disk face-scalar format every solver writes and
 * the viewer reads.
 *
 * Layout (little endian):
 *
 *   offset  bytes  field
 *   0       4      magic 0x52484746 ("RHGF")
 *   4       4      version (5)
 *   8       4      total face count
 *   12      4      faces finished (header mirror of the scan below)
 *   16      4      observation standoff in mm (f32)
 *   20      4      s_min over the finished faces (f32)
 *   24      4      s_max over the finished faces (f32)
 *   28      4*N    one f32 per face, NaN until computed
 *
 * A record is *complete* when every face slot is finite, and nothing else is
 * authoritative: progress and "done" flags are derived from this file, so a
 * stale log line can never make an unfinished bake look finished.
 */

export const RHGF_MAGIC = 0x52484746;
export const RHGF_VERSION = 5;
export const RHGF_HEADER = 28;

/** Observation heights the UI slider can request, in millimetres. */
export const STANDOFF_MIN_MM = 1;
export const STANDOFF_MAX_MM = 32000;
export const STANDOFF_DEFAULT_MM = 16000;

export type RecordSummary = {
  exists: boolean;
  current: number;
  total: number;
  done: boolean;
  standoffMm: number;
};

export const EMPTY_RECORD: RecordSummary = {
  exists: false,
  current: 0,
  total: 0,
  done: false,
  standoffMm: 0,
};

export function clampStandoffMm(mm: unknown): number {
  const value = Number(mm);
  if (!Number.isFinite(value)) return STANDOFF_DEFAULT_MM;
  return Math.min(STANDOFF_MAX_MM, Math.max(STANDOFF_MIN_MM, value));
}

/**
 * Compact state of one on-disk record. The panel needs several records at once
 * (both density modes of each solver, plus the Werner and mascon references),
 * so status payloads carry one summary per record rather than only the active
 * one.
 */
export function recordSummary(buf: Buffer): RecordSummary {
  if (buf.byteLength < RHGF_HEADER) return EMPTY_RECORD;
  if (buf.readUInt32LE(0) !== RHGF_MAGIC || buf.readUInt32LE(4) !== RHGF_VERSION) {
    return EMPTY_RECORD;
  }
  const total = buf.readUInt32LE(8);
  const headerDone = buf.readUInt32LE(12);
  if (total === 0) return EMPTY_RECORD;
  if (buf.byteLength < RHGF_HEADER + total * 4) {
    return { exists: true, current: 0, total, done: false, standoffMm: 0 };
  }
  const rawMm = buf.readFloatLE(16);
  const standoffMm =
    Number.isFinite(rawMm) && rawMm >= STANDOFF_MIN_MM && rawMm <= STANDOFF_MAX_MM ? rawMm : 0;
  let finite = 0;
  for (let i = 0; i < total; i++) {
    if (Number.isFinite(buf.readFloatLE(RHGF_HEADER + i * 4))) finite++;
  }
  return {
    exists: true,
    current: Math.min(total, Math.max(finite, headerDone)),
    total,
    done: finite >= total,
    standoffMm,
  };
}

/** Allocate an all-NaN record so the viewer can start drawing face by face. */
export function blankRecord(totalFaces: number, standoffMm: number): Buffer {
  const buf = Buffer.alloc(RHGF_HEADER + totalFaces * 4);
  buf.writeUInt32LE(RHGF_MAGIC, 0);
  buf.writeUInt32LE(RHGF_VERSION, 4);
  buf.writeUInt32LE(totalFaces, 8);
  buf.writeUInt32LE(0, 12);
  buf.writeFloatLE(standoffMm, 16);
  buf.writeFloatLE(Number.POSITIVE_INFINITY, 20);
  buf.writeFloatLE(Number.NEGATIVE_INFINITY, 24);
  // Must be NaN — 0.0 is finite and would look like a finished bake.
  for (let i = 0; i < totalFaces; i++) buf.writeFloatLE(Number.NaN, RHGF_HEADER + i * 4);
  return buf;
}
