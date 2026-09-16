#!/usr/bin/env bun
/**
 * Build the compact observer geometry and deterministic source samples used by
 * the browser-side continuous-height preview.
 */

import { mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

const root = join(import.meta.dir, "..");
const objPath =
  process.env.BAKE_OBJ ??
  join(root, "../Ryugu_wasm/assets/models/SHAPE_SFM_200k_v20180804.obj");
const outDir = join(root, "assets/live");

type Vec3 = [number, number, number];
type Kernel = { c: Vec3; sigma: number; w: number; alpha: number };

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const length = (a: Vec3) => Math.hypot(a[0], a[1], a[2]);

function parseObj(text: string) {
  const verts: Vec3[] = [];
  const faces: { center: Vec3; normal: Vec3 }[] = [];
  for (const line of text.split("\n")) {
    if (line.startsWith("v ")) {
      const p = line.trim().split(/\s+/);
      verts.push([Number(p[1]), Number(p[2]), Number(p[3])]);
      continue;
    }
    if (!line.startsWith("f ")) continue;
    const p = line.trim().split(/\s+/).slice(1, 4);
    if (p.length !== 3) continue;
    const idx = p.map((token) => {
      const raw = Number(token.split("/")[0]);
      return raw > 0 ? raw - 1 : verts.length + raw;
    });
    const a = verts[idx[0]!];
    const b = verts[idx[1]!];
    const c = verts[idx[2]!];
    if (!a || !b || !c) continue;
    const center: Vec3 = [
      (a[0] + b[0] + c[0]) / 3,
      (a[1] + b[1] + c[1]) / 3,
      (a[2] + b[2] + c[2]) / 3,
    ];
    const n = cross(sub(b, a), sub(c, a));
    const l = Math.max(length(n), 1e-20);
    faces.push({ center, normal: [n[0] / l, n[1] / l, n[2] / l] });
  }
  return { verts, faces };
}

function parseKernels(path: string): Kernel[] {
  const text = readFileSync(path, "utf8");
  const alphaDefault = Number(text.match(/alpha_default\s*=\s*([0-9.]+)/)?.[1] ?? 1);
  const kernels: Kernel[] = [];
  for (const match of text.matchAll(/\{[^{}]*\}/gs)) {
    const block = match[0];
    const c = block.match(/c\s*=\s*\[\s*([^,\]]+),\s*([^,\]]+),\s*([^,\]]+)\s*\]/);
    const sigma = block.match(/sigma\s*=\s*([0-9.eE+-]+)/);
    const w = block.match(/\bw\s*=\s*([0-9.eE+-]+)/);
    if (!c || !sigma || !w) continue;
    const alpha = Number(block.match(/alpha\s*=\s*([0-9.eE+-]+)/)?.[1] ?? alphaDefault);
    kernels.push({
      c: [Number(c[1]) * 1000, Number(c[2]) * 1000, Number(c[3]) * 1000],
      sigma: Number(sigma[1]) / 1000,
      w: Number(w[1]),
      alpha,
    });
  }
  return kernels;
}

function rng(seed: number) {
  let state = seed >>> 0;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 4294967296;
  };
}

function randomUnit(r: () => number): Vec3 {
  const z = 2 * r() - 1;
  const phi = 2 * Math.PI * r();
  const s = Math.sqrt(Math.max(0, 1 - z * z));
  return [s * Math.cos(phi), s * Math.sin(phi), z];
}

function sampleKernels(kernels: Kernel[], count: number, bodyRadius: number) {
  const scales = kernels.map((k) => Math.max(1 / Math.max(k.sigma, 1e-12), 1));
  const masses = kernels.map((k, i) => {
    const truncated = Math.min(scales[i]!, bodyRadius);
    return k.w * Math.pow(truncated, 3);
  });
  const total = masses.reduce((s, m) => s + Math.abs(m), 0) || 1;
  const counts = masses.map((m) => Math.max(1, Math.floor((count * Math.abs(m)) / total)));
  let totalCount = counts.reduce((s, n) => s + n, 0);
  let cursor = 0;
  while (totalCount < count) {
    const at = cursor++ % counts.length;
    counts[at] = counts[at]! + 1;
    totalCount++;
  }
  while (totalCount > count) {
    let at = -1;
    for (let i = counts.length - 1; i >= 0; i--) {
      if (counts[i]! > 1) {
        at = i;
        break;
      }
    }
    if (at < 0) break;
    counts[at] = counts[at]! - 1;
    totalCount--;
  }

  const random = rng(0x51a7e);
  const out: number[] = [];
  for (let i = 0; i < kernels.length; i++) {
    const k = kernels[i]!;
    const n = counts[i]!;
    const scale = scales[i]!;
    const mass = masses[i]! / n;
    for (let j = 0; j < n; j++) {
      const u = random();
      const radius = Math.min(
        bodyRadius * 2.5,
        scale * Math.tan(Math.PI * (u - 0.5)),
      );
      const dir = randomUnit(random);
      const r = Number.isFinite(radius) ? Math.abs(radius) : bodyRadius;
      out.push(
        k.c[0] + dir[0] * r,
        k.c[1] + dir[1] * r,
        k.c[2] + dir[2] * r,
        mass,
      );
    }
  }
  return new Float32Array(out);
}

function sampleUniform(count: number, extents: Vec3) {
  const random = rng(0x51a7f);
  const out = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    const dir = randomUnit(random);
    const radius = Math.cbrt(random());
    out[i * 4] = dir[0] * extents[0] * radius;
    out[i * 4 + 1] = dir[1] * extents[1] * radius;
    out[i * 4 + 2] = dir[2] * extents[2] * radius;
    out[i * 4 + 3] = 1 / count;
  }
  return out;
}

console.log("reading mesh...");
const mesh = parseObj(await Bun.file(objPath).text());
console.log(`parsed ${mesh.faces.length} faces`);
if (mesh.faces.length !== 196608) {
  throw new Error(`expected 196608 faces, got ${mesh.faces.length}`);
}

const min: Vec3 = [Infinity, Infinity, Infinity];
const max: Vec3 = [-Infinity, -Infinity, -Infinity];
for (const p of mesh.verts) {
  for (let axis = 0; axis < 3; axis++) {
    min[axis] = Math.min(min[axis]!, p[axis]!);
    max[axis] = Math.max(max[axis]!, p[axis]!);
  }
}
console.log("computed bounds");
const extent: Vec3 = [
  (max[0] - min[0]) / 2,
  (max[1] - min[1]) / 2,
  (max[2] - min[2]) / 2,
];
const bodyRadius = Math.max(...extent);

const observerStride = 16 + 6 * 4 + mesh.faces.length * 12;
const observers = new Uint8Array(observerStride);
const observerView = new DataView(observers.buffer);
observerView.setUint32(0, 0x424f5952, true); // RYOB
observerView.setUint32(4, 1, true);
observerView.setUint32(8, mesh.faces.length, true);
observerView.setUint32(12, 0, true);
for (let axis = 0; axis < 3; axis++) {
  observerView.setFloat32(16 + axis * 4, min[axis]!, true);
  observerView.setFloat32(28 + axis * 4, max[axis]!, true);
}
const quantizePosition = (value: number, axis: number) => {
  const range = Math.max(max[axis]! - min[axis]!, 1e-20);
  return Math.max(-32768, Math.min(32767, Math.round(((value - min[axis]!) / range) * 65535 - 32768)));
};
let offset = 40;
for (const face of mesh.faces) {
  for (let axis = 0; axis < 3; axis++) {
    observerView.setInt16(offset, quantizePosition(face.center[axis]!, axis), true);
    observerView.setInt16(
      offset + 6,
      Math.max(-32767, Math.min(32767, Math.round(face.normal[axis]! * 32767))),
      true,
    );
    offset += 2;
  }
  offset += 6;
}
console.log("quantized observers");

const cauchy = sampleKernels(parseKernels(join(root, "assets/density/cauchy.toml")), 768, bodyRadius);
console.log("sampled cauchy sources");
const elliptic = sampleKernels(
  parseKernels(join(root, "assets/density/cauchy_elliptic.toml")),
  512,
  bodyRadius,
);
console.log("sampled elliptic sources");
const uniform = sampleUniform(384, extent);
const sets = [
  { id: 0, data: cauchy },
  { id: 1, data: elliptic },
  { id: 2, data: uniform },
];

const sourceBytes = 16 + sets.reduce((n, set) => n + 16 + set.data.byteLength, 0);
const sources = new Uint8Array(sourceBytes);
const sourceView = new DataView(sources.buffer);
sourceView.setUint32(0, 0x31535952, true); // RYS1
sourceView.setUint32(4, 1, true);
sourceView.setUint32(8, sets.length, true);
sourceView.setUint32(12, 0, true);
let sourceOffset = 16;
for (const set of sets) {
  sourceView.setUint32(sourceOffset, set.id, true);
  sourceView.setUint32(sourceOffset + 4, set.data.length / 4, true);
  sourceView.setBigUint64(sourceOffset + 8, 0n, true);
  sourceOffset += 16;
  sources.set(new Uint8Array(set.data.buffer), sourceOffset);
  sourceOffset += set.data.byteLength;
}

mkdirSync(outDir, { recursive: true });
await Bun.write(join(outDir, "face_observers.bin"), observers);
await Bun.write(join(outDir, "preview_sources.bin"), sources);
console.log(`wrote ${mesh.faces.length} observers and ${sets.length} source sets`);
