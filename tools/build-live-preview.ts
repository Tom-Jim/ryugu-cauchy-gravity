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
type Triangle = { a: Vec3; b: Vec3; c: Vec3; center: Vec3; normal: Vec3; area: number };

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const length = (a: Vec3) => Math.hypot(a[0], a[1], a[2]);

function parseObj(text: string) {
  const verts: Vec3[] = [];
  const faces: Triangle[] = [];
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
    const doubleArea = length(n);
    const l = Math.max(doubleArea, 1e-20);
    faces.push({
      a,
      b,
      c,
      center,
      normal: [n[0] / l, n[1] / l, n[2] / l],
      area: 0.5 * doubleArea,
    });
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

function densityAt(kernels: Kernel[], point: Vec3) {
  let density = 0;
  for (const kernel of kernels) {
    const dx = point[0] - kernel.c[0];
    const dy = point[1] - kernel.c[1];
    const dz = point[2] - kernel.c[2];
    const r2 = dx * dx + dy * dy + dz * dz;
    density += kernel.w / Math.pow(1 + kernel.sigma * kernel.sigma * r2, kernel.alpha);
  }
  return density;
}

function faceDensities(faces: Triangle[], kernels: Kernel[], normalizeMass: number) {
  const out = new Float64Array(faces.length);
  for (let i = 0; i < faces.length; i++) out[i] = densityAt(kernels, faces[i]!.center);
  if (normalizeMass <= 0) return out;

  let total = 0;
  for (let i = 0; i < faces.length; i++) total += out[i]! * faces[i]!.area;
  const scale = total !== 0 ? normalizeMass / total : 1;
  for (let i = 0; i < out.length; i++) out[i] = out[i]! * scale;
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

const cauchy = faceDensities(
  mesh.faces,
  parseKernels(join(root, "assets/density/cauchy.toml")),
  4.50e11,
);
console.log("sampled Cauchy density on every triangle");
const elliptic = faceDensities(
  mesh.faces,
  parseKernels(join(root, "assets/density/cauchy_elliptic.toml")),
  0,
);
console.log("sampled elliptic density on every triangle");

const sourceBytes = 16 + mesh.faces.length * 12 * 4;
const sources = new Uint8Array(sourceBytes);
const sourceView = new DataView(sources.buffer);
sourceView.setUint32(0, 0x31535952, true); // RYS1
sourceView.setUint32(4, 2, true);
sourceView.setUint32(8, mesh.faces.length, true);
sourceView.setUint32(12, 0, true);
let sourceOffset = 16;
for (let i = 0; i < mesh.faces.length; i++) {
  const face = mesh.faces[i]!;
  for (const vertex of [face.a, face.b, face.c]) {
    sourceView.setFloat32(sourceOffset, vertex[0], true);
    sourceView.setFloat32(sourceOffset + 4, vertex[1], true);
    sourceView.setFloat32(sourceOffset + 8, vertex[2], true);
    sourceOffset += 12;
  }
  sourceView.setFloat32(sourceOffset, cauchy[i]!, true);
  sourceView.setFloat32(sourceOffset + 4, elliptic[i]!, true);
  sourceView.setFloat32(sourceOffset + 8, 1, true);
  sourceOffset += 12;
}

mkdirSync(outDir, { recursive: true });
await Bun.write(join(outDir, "face_observers.bin"), observers);
await Bun.write(join(outDir, "preview_sources.bin"), sources);
console.log(`wrote ${mesh.faces.length} observers and full-face source geometry`);
