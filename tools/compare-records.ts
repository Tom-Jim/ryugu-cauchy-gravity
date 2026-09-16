#!/usr/bin/env bun
/**
 * Face-by-face comparison of two RHGF v5 records.
 *
 * Every solver in this project writes one finite `‖H‖_F` value per mesh face,
 * so a residual is always a difference between two arrays with the same index
 * meaning. The norm is ~1e-6, which makes an absolute tolerance meaningless:
 * this tool reports relative differences, plus the fraction of faces above a
 * threshold, and the observation height of each record so a comparison taken at
 * two different heights is immediately visible as such.
 *
 * Usage:
 *   bun tools/compare-records.ts <a.bin> <b.bin> [--rel 0.05] [--json]
 */

const RHGF_MAGIC = 0x52484746;
const RHGF_VERSION = 5;
const HEADER_BYTES = 28;

type Record = {
  path: string;
  standoffMm: number;
  total: number;
  finite: number;
  scalars: Float32Array;
};

function parse(path: string, bytes: Uint8Array): Record {
  const fail = (why: string): never => {
    throw new Error(`${path}: ${why}`);
  };
  if (bytes.byteLength < HEADER_BYTES) fail("shorter than the RHGF header");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(0, true) !== RHGF_MAGIC) fail("not an RHGF record");
  if (view.getUint32(4, true) !== RHGF_VERSION) fail("not an RHGF v5 record");
  const total = view.getUint32(8, true);
  if (bytes.byteLength < HEADER_BYTES + total * 4) fail("truncated scalar table");
  const scalars = new Float32Array(total);
  let finite = 0;
  for (let i = 0; i < total; i++) {
    const v = view.getFloat32(HEADER_BYTES + i * 4, true);
    scalars[i] = v;
    if (Number.isFinite(v)) finite++;
  }
  // Offset 16 is the reserved slot holding the observation height in mm.
  const rawMm = view.getFloat32(16, true);
  return {
    path,
    standoffMm: Number.isFinite(rawMm) ? rawMm : 0,
    total,
    finite,
    scalars,
  };
}

async function read(path: string): Promise<Record> {
  const file = Bun.file(path);
  if (!(await file.exists())) throw new Error(`${path}: no such file`);
  return parse(path, new Uint8Array(await file.arrayBuffer()));
}

function quantile(sorted: Float64Array, q: number): number {
  return sorted[Math.min(sorted.length - 1, Math.round((sorted.length - 1) * q))] ?? Number.NaN;
}

function fmt(x: number): string {
  return Number.isFinite(x) ? `${(100 * x).toFixed(3)}%` : "n/a";
}

const args = Bun.argv.slice(2);
const flag = (name: string, fallback: number): number => {
  const at = args.indexOf(name);
  const raw = at >= 0 ? Number(args[at + 1]) : Number.NaN;
  return Number.isFinite(raw) ? raw : fallback;
};
const positional = args.filter((a, i) => !a.startsWith("--") && !args[i - 1]?.startsWith("--"));
if (positional.length !== 2) {
  console.error("usage: bun tools/compare-records.ts <a.bin> <b.bin> [--rel 0.05] [--json]");
  process.exit(2);
}
const relEps = flag("--rel", 0.05);
const asJson = args.includes("--json");

const a = await read(positional[0]!);
const b = await read(positional[1]!);
const n = Math.min(a.total, b.total);
if (n === 0) throw new Error("nothing to compare");

// Packed storage avoids one boxed JavaScript Number per finite face.
const rel = new Float64Array(n);
let count = 0;
let sumA = 0;
let sumB = 0;
let maxA = 0;
let maxB = 0;
for (let i = 0; i < n; i++) {
  const sa = a.scalars[i]!;
  const sb = b.scalars[i]!;
  if (!Number.isFinite(sa) || !Number.isFinite(sb)) continue;
  rel[count++] = Math.abs(sa - sb) / Math.max(Math.abs(sa), Math.abs(sb), 1e-300);
  sumA += sa;
  sumB += sb;
  maxA = Math.max(maxA, sa);
  maxB = Math.max(maxB, sb);
}
if (count === 0) throw new Error("no face is finite in both records");
const sorted = rel.subarray(0, count);
sorted.sort();

const summary = {
  a: { path: a.path, standoffMm: a.standoffMm, faces: a.total, finite: a.finite },
  b: { path: b.path, standoffMm: b.standoffMm, faces: b.total, finite: b.finite },
  compared: count,
  relEps,
  overThreshold: sorted.reduce((total, r) => total + (r > relEps ? 1 : 0), 0),
  mean: sorted.reduce((s, r) => s + r, 0) / count,
  median: quantile(sorted, 0.5),
  p90: quantile(sorted, 0.9),
  p99: quantile(sorted, 0.99),
  max: sorted[count - 1]!,
  aggregateRatio: sumB !== 0 ? sumA / sumB : Number.NaN,
  maxScalarRatio: maxB !== 0 ? maxA / maxB : Number.NaN,
};

if (asJson) {
  console.log(JSON.stringify(summary, null, 2));
} else {
  const label = (r: Record) =>
    `${r.path} · standoff ${(r.standoffMm / 1000).toFixed(2)} m · ${r.finite}/${r.total} finite`;
  console.log(`a: ${label(a)}`);
  console.log(`b: ${label(b)}`);
  if (Math.abs(a.standoffMm - b.standoffMm) > 1e-3) {
    console.log("   !! the two records were evaluated at different heights");
  }
  console.log(`compared ${summary.compared} faces (|a-b| / max(|a|,|b|))`);
  console.log(
    `  mean ${fmt(summary.mean)} · median ${fmt(summary.median)} · p90 ${fmt(summary.p90)}` +
      ` · p99 ${fmt(summary.p99)} · max ${fmt(summary.max)}`,
  );
  console.log(
    `  over ${fmt(relEps)}: ${summary.overThreshold} faces` +
      ` (${((100 * summary.overThreshold) / summary.compared).toFixed(2)} %)`,
  );
  console.log(
    `  aggregate a/b ${summary.aggregateRatio.toFixed(6)}` +
      ` · largest scalar ratio ${summary.maxScalarRatio.toFixed(6)}`,
  );
}
