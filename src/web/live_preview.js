/**
 * Browser-side full-face gravity-gradient recomputation.
 *
 * Exact solver records are used only at their recorded standoff. At every other
 * height this module evaluates the density-weighted surface mass distribution
 * directly: nearby triangles use the closed-form polyhedral face integral and
 * distant Barnes-Hut nodes use a monopole approximation. Nothing is rescaled
 * face by face from one of the precomputed records.
 */

const G = 6.67430e-11;
const RECORD_HEADER = 28;
const SOURCE_HEADER = 16;
const SOURCE_FLOATS = 12;
const SOURCE_SET_COUNT = 3;
const LEAF_SIZE = 16;
const MAX_DEPTH = 40;
const BH_THETA = 0.8;
const CHUNK_FACES = 2048;
const SCALE_SAMPLES = 256;
const MIN_RADIUS_SQ = 1e-12;

const SOURCE_SET = {
  cauchy: 0,
  elliptic: 1,
  constant: 2,
};

function norm3(x, y, z) {
  return Math.hypot(x, y, z);
}

function edgeLog(sA, sB, l) {
  const la = Math.hypot(sA, l);
  const lb = Math.hypot(sB, l);
  if (sA >= 0 || sB > 0) {
    return Math.log((sB + lb) / Math.max(sA + la, Number.MIN_VALUE));
  }
  return Math.log((la - sA) / Math.max(lb - sB, Number.MIN_VALUE));
}

function parseSources(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(0, true) !== 0x31535952 || view.getUint32(4, true) !== 2) {
    throw new Error("invalid live-preview source asset");
  }
  const count = view.getUint32(8, true);
  const expected = SOURCE_HEADER + count * SOURCE_FLOATS * 4;
  if (count === 0 || count !== 196608 || bytes.byteLength < expected) {
    throw new Error("truncated live-preview source asset");
  }

  const vertices = new Float32Array(count * 9);
  const densities = new Float32Array(count * SOURCE_SET_COUNT);
  const centers = new Float64Array(count * 3);
  const normals = new Float32Array(count * 3);
  const areas = new Float64Array(count);
  const radii = new Float64Array(count);
  let offset = SOURCE_HEADER;
  for (let face = 0; face < count; face++) {
    const v = face * 9;
    for (let i = 0; i < 9; i++) {
      vertices[v + i] = view.getFloat32(offset + i * 4, true);
    }
    const d = face * SOURCE_SET_COUNT;
    for (let i = 0; i < SOURCE_SET_COUNT; i++) {
      densities[d + i] = view.getFloat32(offset + (9 + i) * 4, true);
    }
    offset += SOURCE_FLOATS * 4;

    const ax = vertices[v];
    const ay = vertices[v + 1];
    const az = vertices[v + 2];
    const bx = vertices[v + 3];
    const by = vertices[v + 4];
    const bz = vertices[v + 5];
    const cx = vertices[v + 6];
    const cy = vertices[v + 7];
    const cz = vertices[v + 8];
    const centerX = (ax + bx + cx) / 3;
    const centerY = (ay + by + cy) / 3;
    const centerZ = (az + bz + cz) / 3;
    centers[face * 3] = centerX;
    centers[face * 3 + 1] = centerY;
    centers[face * 3 + 2] = centerZ;

    const e1x = bx - ax;
    const e1y = by - ay;
    const e1z = bz - az;
    const e2x = cx - bx;
    const e2y = cy - by;
    const e2z = cz - bz;
    const nx = e1y * e2z - e1z * e2y;
    const ny = e1z * e2x - e1x * e2z;
    const nz = e1x * e2y - e1y * e2x;
    const doubleArea = norm3(nx, ny, nz);
    const invNormal = doubleArea > 0 ? 1 / doubleArea : 0;
    normals[face * 3] = nx * invNormal;
    normals[face * 3 + 1] = ny * invNormal;
    normals[face * 3 + 2] = nz * invNormal;
    areas[face] = 0.5 * doubleArea;

    radii[face] = Math.max(
      Math.hypot(ax - centerX, ay - centerY, az - centerZ),
      Math.hypot(bx - centerX, by - centerY, bz - centerZ),
      Math.hypot(cx - centerX, cy - centerY, cz - centerZ),
    );
  }

  return { count, vertices, densities, centers, normals, areas, radii };
}

class SurfaceMassTree {
  constructor(sources) {
    this.sources = sources;
    const maxNodes = 4 * Math.ceil(sources.count / LEAF_SIZE) + 16;
    this.left = new Int32Array(maxNodes);
    this.right = new Int32Array(maxNodes);
    this.start = new Int32Array(maxNodes);
    this.leafCount = new Int32Array(maxNodes);
    this.radius = new Float64Array(maxNodes);
    this.comX = new Float64Array(maxNodes);
    this.comY = new Float64Array(maxNodes);
    this.comZ = new Float64Array(maxNodes);
    this.masses = Array.from(
      { length: SOURCE_SET_COUNT },
      () => new Float64Array(maxNodes),
    );
    this.order = new Uint32Array(sources.count);
    for (let i = 0; i < sources.count; i++) this.order[i] = i;
    this.nodeCount = 0;
    this._buildNode(0, sources.count, 0);
  }

  _buildNode(start, count, depth) {
    const node = this.nodeCount++;
    const end = start + count;
    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;
    let sumX = 0;
    let sumY = 0;
    let sumZ = 0;
    for (let i = start; i < end; i++) {
      const face = this.order[i];
      const p = face * 3;
      const x = this.sources.centers[p];
      const y = this.sources.centers[p + 1];
      const z = this.sources.centers[p + 2];
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      minZ = Math.min(minZ, z);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
      maxZ = Math.max(maxZ, z);
      sumX += x;
      sumY += y;
      sumZ += z;
    }
    const centerX = sumX / count;
    const centerY = sumY / count;
    const centerZ = sumZ / count;
    this.comX[node] = centerX;
    this.comY[node] = centerY;
    this.comZ[node] = centerZ;

    let radiusSq = 0;
    for (let i = start; i < end; i++) {
      const face = this.order[i];
      const p = face * 3;
      const dx = this.sources.centers[p] - centerX;
      const dy = this.sources.centers[p + 1] - centerY;
      const dz = this.sources.centers[p + 2] - centerZ;
      const faceRadius = this.sources.radii[face];
      radiusSq = Math.max(radiusSq, dx * dx + dy * dy + dz * dz + faceRadius * faceRadius);
    }
    this.radius[node] = Math.sqrt(radiusSq);

    if (count <= LEAF_SIZE || depth >= MAX_DEPTH) {
      this.left[node] = -1;
      this.right[node] = -1;
      this.start[node] = start;
      this.leafCount[node] = count;
      for (let i = start; i < end; i++) {
        const face = this.order[i];
        const area = this.sources.areas[face];
        for (let set = 0; set < SOURCE_SET_COUNT; set++) {
          this.masses[set][node] += (
            this.sources.densities[face * SOURCE_SET_COUNT + set] * area
          );
        }
      }
      return node;
    }

    let axis = 0;
    let span = maxX - minX;
    if (maxY - minY > span) {
      axis = 1;
      span = maxY - minY;
    }
    if (maxZ - minZ > span) axis = 2;
    const slice = Array.from(this.order.subarray(start, end));
    slice.sort((a, b) => (
      this.sources.centers[a * 3 + axis] - this.sources.centers[b * 3 + axis]
    ));
    this.order.set(slice, start);

    const mid = start + (count >> 1);
    const left = this._buildNode(start, count >> 1, depth + 1);
    const right = this._buildNode(mid, count - (count >> 1), depth + 1);
    this.left[node] = left;
    this.right[node] = right;
    this.start[node] = 0;
    this.leafCount[node] = 0;
    for (let set = 0; set < SOURCE_SET_COUNT; set++) {
      this.masses[set][node] = this.masses[set][left] + this.masses[set][right];
    }
    return node;
  }
}

class FullSurfaceEngine {
  constructor(options) {
    this.sourceUrl = options.sourceUrl;
    this.sources = null;
    this.tree = null;
    this.scaleCache = new Map();
    this.stack = new Int32Array(8192);
  }

  async init() {
    const response = await fetch(this.sourceUrl, { cache: "force-cache" });
    if (!response.ok) throw new Error("failed to load live-preview source asset");
    this.sources = parseSources(new Uint8Array(await response.arrayBuffer()));
    this.tree = new SurfaceMassTree(this.sources);
  }

  _addTriangle(out, face, density, px, py, pz) {
    const v = face * 9;
    const vx = this.sources.vertices;
    const a0 = vx[v];
    const a1 = vx[v + 1];
    const a2 = vx[v + 2];
    const b0 = vx[v + 3];
    const b1 = vx[v + 4];
    const b2 = vx[v + 5];
    const c0 = vx[v + 6];
    const c1 = vx[v + 7];
    const c2 = vx[v + 8];

    const e1x = b0 - a0;
    const e1y = b1 - a1;
    const e1z = b2 - a2;
    const e2x = c0 - b0;
    const e2y = c1 - b1;
    const e2z = c2 - b2;
    let nx = e1y * e2z - e1z * e2y;
    let ny = e1z * e2x - e1x * e2z;
    let nz = e1x * e2y - e1y * e2x;
    const invNormal = 1 / Math.max(Math.hypot(nx, ny, nz), Number.MIN_VALUE);
    nx *= invNormal;
    ny *= invNormal;
    nz *= invNormal;

    let ix = 0;
    let iy = 0;
    let iz = 0;
    const corners = [a0, a1, a2, b0, b1, b2, c0, c1, c2];
    for (let edge = 0; edge < 3; edge++) {
      const ai = edge * 3;
      const bi = ((edge + 1) % 3) * 3;
      const ax = corners[ai];
      const ay = corners[ai + 1];
      const az = corners[ai + 2];
      const bx = corners[bi];
      const by = corners[bi + 1];
      const bz = corners[bi + 2];
      let dx = bx - ax;
      let dy = by - ay;
      let dz = bz - az;
      const invLength = 1 / Math.max(Math.hypot(dx, dy, dz), Number.MIN_VALUE);
      dx *= invLength;
      dy *= invLength;
      dz *= invLength;

      const edgeNx = dy * nz - dz * ny;
      const edgeNy = dz * nx - dx * nz;
      const edgeNz = dx * ny - dy * nx;
      const rAx = ax - px;
      const rAy = ay - py;
      const rAz = az - pz;
      const parA = rAx * dx + rAy * dy + rAz * dz;
      const perpX = rAx - dx * parA;
      const perpY = rAy - dy * parA;
      const perpZ = rAz - dz * parA;
      const lineDistance = norm3(perpX, perpY, perpZ);
      const parB = (bx - px) * dx + (by - py) * dy + (bz - pz) * dz;
      const logTerm = edgeLog(parA, parB, lineDistance);
      ix += edgeNx * logTerm;
      iy += edgeNy * logTerm;
      iz += edgeNz * logTerm;
    }

    const rax = a0 - px;
    const ray = a1 - py;
    const raz = a2 - pz;
    const rbx = b0 - px;
    const rby = b1 - py;
    const rbz = b2 - pz;
    const rcx = c0 - px;
    const rcy = c1 - py;
    const rcz = c2 - pz;
    const invRa = 1 / Math.max(norm3(rax, ray, raz), Number.MIN_VALUE);
    const invRb = 1 / Math.max(norm3(rbx, rby, rbz), Number.MIN_VALUE);
    const invRc = 1 / Math.max(norm3(rcx, rcy, rcz), Number.MIN_VALUE);
    const anx = rax * invRa;
    const any = ray * invRa;
    const anz = raz * invRa;
    const bnx = rbx * invRb;
    const bny = rby * invRb;
    const bnz = rbz * invRb;
    const cnx = rcx * invRc;
    const cny = rcy * invRc;
    const cnz = rcz * invRc;
    const crossX = bny * cnz - bnz * cny;
    const crossY = bnz * cnx - bnx * cnz;
    const crossZ = bnx * cny - bny * cnx;
    const numerator = -(anx * crossX + any * crossY + anz * crossZ);
    const denominator = 1 + anx * bnx + any * bny + anz * bnz
      + bnx * cnx + bny * cny + bnz * cnz
      + cnx * anx + cny * any + cnz * anz;
    const omega = 2 * Math.atan2(numerator, denominator);
    ix += omega * nx;
    iy += omega * ny;
    iz += omega * nz;

    const k = G * density;
    out[0] += k * nx * ix;
    out[1] += k * ny * iy;
    out[2] += k * nz * iz;
    out[3] += k * nx * iy;
    out[4] += k * nx * iz;
    out[5] += k * ny * iz;
  }

  _addLeaf(out, node, set, px, py, pz) {
    const start = this.tree.start[node];
    const end = start + this.tree.leafCount[node];
    for (let i = start; i < end; i++) {
      const face = this.tree.order[i];
      const density = this.sources.densities[face * SOURCE_SET_COUNT + set];
      if (density !== 0) this._addTriangle(out, face, density, px, py, pz);
    }
  }

  _addMonopole(out, node, set, px, py, pz) {
    const dx = px - this.tree.comX[node];
    const dy = py - this.tree.comY[node];
    const dz = pz - this.tree.comZ[node];
    const r2 = Math.max(dx * dx + dy * dy + dz * dz, MIN_RADIUS_SQ);
    const invR = 1 / Math.sqrt(r2);
    const invR2 = invR * invR;
    const invR3 = invR2 * invR;
    const invR5 = invR3 * invR2;
    const k = G * this.tree.masses[set][node];
    out[0] += k * (3 * dx * dx * invR5 - invR3);
    out[1] += k * (3 * dy * dy * invR5 - invR3);
    out[2] += k * (3 * dz * dz * invR5 - invR3);
    out[3] += k * 3 * dx * dy * invR5;
    out[4] += k * 3 * dx * dz * invR5;
    out[5] += k * 3 * dy * dz * invR5;
  }

  _tensorAt(px, py, pz, set, out) {
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 0;
    out[5] = 0;
    const stack = this.stack;
    let top = 0;
    stack[top++] = 0;
    while (top > 0) {
      const node = stack[--top];
      const dx = px - this.tree.comX[node];
      const dy = py - this.tree.comY[node];
      const dz = pz - this.tree.comZ[node];
      const distance = Math.max(norm3(dx, dy, dz), Number.MIN_VALUE);
      if (this.tree.radius[node] / distance < BH_THETA) {
        this._addMonopole(out, node, set, px, py, pz);
      } else if (this.tree.leafCount[node] > 0) {
        this._addLeaf(out, node, set, px, py, pz);
      } else {
        stack[top++] = this.tree.left[node];
        stack[top++] = this.tree.right[node];
      }
    }
  }

  _tensorNormAt(px, py, pz, set) {
    const out = [0, 0, 0, 0, 0, 0];
    this._tensorAt(px, py, pz, set, out);
    return Math.sqrt(
      out[0] * out[0] + out[1] * out[1] + out[2] * out[2]
        + 2 * (out[3] * out[3] + out[4] * out[4] + out[5] * out[5]),
    );
  }

  _sourceScale(set, referenceHeightMm, referenceView) {
    if (!this.tree || !this.sources) return 1;
    const key = `${set}:${referenceHeightMm}`;
    const cached = this.scaleCache.get(key);
    if (cached !== undefined) return cached;
    const ratios = [];
    const heightM = referenceHeightMm * 1e-3;
    for (let sample = 0; sample < SCALE_SAMPLES; sample++) {
      const face = Math.round(
        (sample * (this.sources.count - 1)) / Math.max(1, SCALE_SAMPLES - 1),
      );
      const p = face * 3;
      const px = this.sources.centers[p] + this.sources.normals[p] * heightM;
      const py = this.sources.centers[p + 1] + this.sources.normals[p + 1] * heightM;
      const pz = this.sources.centers[p + 2] + this.sources.normals[p + 2] * heightM;
      const direct = this._tensorNormAt(px, py, pz, set);
      const reference = referenceView.getFloat32(RECORD_HEADER + face * 4, true);
      if (Number.isFinite(direct) && direct > 0 && Number.isFinite(reference) && reference > 0) {
        ratios.push(reference / direct);
      }
    }
    if (ratios.length === 0) return 1;
    ratios.sort((a, b) => a - b);
    const scale = ratios[ratios.length >> 1];
    this.scaleCache.set(key, scale);
    return scale;
  }

  async evaluate({ sourceSet, heightMm, referenceBytes, onProgress, signal }) {
    if (!this.tree || !this.sources) throw new Error("live-preview is not initialized");
    const bytes = referenceBytes.slice();
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (
      view.getUint32(0, true) !== 0x52484746
      || view.getUint32(4, true) !== 5
      || view.getUint32(8, true) !== this.sources.count
      || bytes.byteLength < RECORD_HEADER + this.sources.count * 4
    ) {
      throw new Error("live-preview reference record is incompatible");
    }

    const set = SOURCE_SET[sourceSet] ?? SOURCE_SET.constant;
    const refHeightMm = view.getFloat32(16, true);
    const scale = this._sourceScale(set, refHeightMm, view);
    view.setFloat32(16, heightMm, true);
    view.setUint32(12, this.sources.count, true);
    const heightM = heightMm * 1e-3;
    const tensor = [0, 0, 0, 0, 0, 0];
    const chunks = Math.ceil(this.sources.count / CHUNK_FACES);
    onProgress?.(0);

    for (let chunk = 0; chunk < chunks; chunk++) {
      if (signal?.aborted) return null;
      const base = chunk * CHUNK_FACES;
      const end = Math.min(this.sources.count, base + CHUNK_FACES);
      for (let face = base; face < end; face++) {
        const p = face * 3;
        const px = this.sources.centers[p] + this.sources.normals[p] * heightM;
        const py = this.sources.centers[p + 1] + this.sources.normals[p + 1] * heightM;
        const pz = this.sources.centers[p + 2] + this.sources.normals[p + 2] * heightM;
        this._tensorAt(px, py, pz, set, tensor);
        const value = Math.sqrt(
          tensor[0] * tensor[0] + tensor[1] * tensor[1] + tensor[2] * tensor[2]
            + 2 * (
              tensor[3] * tensor[3]
              + tensor[4] * tensor[4]
              + tensor[5] * tensor[5]
            ),
        ) * scale;
        view.setFloat32(RECORD_HEADER + face * 4, value, true);
      }
      onProgress?.((chunk + 1) / chunks);
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    return bytes;
  }
}

function isWorkerScope() {
  return typeof DedicatedWorkerGlobalScope !== "undefined"
    && self instanceof DedicatedWorkerGlobalScope;
}

if (isWorkerScope()) {
  let engine = null;
  const cancelled = new Set();

  self.onmessage = async (event) => {
    const message = event.data;
    if (message.type === "init") {
      try {
        engine = new FullSurfaceEngine({ sourceUrl: message.sourceUrl });
        await engine.init();
        self.postMessage({ type: "ready" });
      } catch (error) {
        self.postMessage({
          type: "init-error",
          error: String(error?.stack || error),
        });
      }
      return;
    }
    if (message.type === "cancel") {
      cancelled.add(message.id);
      return;
    }
    if (message.type !== "evaluate" || !engine) return;

    const { id } = message;
    try {
      const bytes = await engine.evaluate({
        sourceSet: message.sourceSet,
        heightMm: message.heightMm,
        referenceBytes: message.referenceBytes,
        signal: {
          get aborted() {
            return cancelled.has(id);
          },
        },
        onProgress: (progress) => {
          self.postMessage({ type: "progress", id, progress });
        },
      });
      if (!bytes) {
        self.postMessage({ type: "cancelled", id });
      } else {
        self.postMessage({ type: "result", id, bytes }, [bytes.buffer]);
      }
    } catch (error) {
      self.postMessage({
        type: "error",
        id,
        error: String(error?.stack || error),
      });
    } finally {
      cancelled.delete(id);
    }
  };
}

export class LivePreview {
  constructor(options) {
    this.sourceUrl = new URL(options.sourceUrl, document.baseURI).href;
    this.worker = null;
    this.nextId = 1;
    this.pending = new Map();
    this.ready = null;
  }

  async init() {
    const worker = new Worker(new URL("./live_preview.js", import.meta.url), {
      type: "module",
      name: "ryugu-full-surface",
    });
    this.worker = worker;
    this.ready = new Promise((resolve, reject) => {
      const onMessage = (event) => {
        const message = event.data;
        if (message.type === "ready") {
          worker.removeEventListener("message", onMessage);
          resolve();
        } else if (message.type === "init-error") {
          worker.removeEventListener("message", onMessage);
          reject(new Error(message.error));
        }
      };
      worker.addEventListener("message", onMessage);
    });

    worker.addEventListener("message", (event) => {
      const message = event.data;
      if (message.type === "ready" || message.type === "init-error") return;
      const pending = this.pending.get(message.id);
      if (!pending) return;
      if (message.type === "progress") {
        pending.onProgress?.(message.progress);
        return;
      }
      this.pending.delete(message.id);
      pending.signal?.removeEventListener("abort", pending.onAbort);
      if (message.type === "result") {
        pending.resolve(new Uint8Array(message.bytes));
      } else if (message.type === "cancelled") {
        pending.resolve(null);
      } else {
        pending.reject(new Error(message.error || "full-surface worker failed"));
      }
    });
    worker.addEventListener("error", (event) => {
      const error = new Error(event.message || "full-surface worker failed");
      for (const pending of this.pending.values()) {
        pending.signal?.removeEventListener("abort", pending.onAbort);
        pending.reject(error);
      }
      this.pending.clear();
    });

    worker.postMessage({ type: "init", sourceUrl: this.sourceUrl });
    await this.ready;
  }

  evaluate({ sourceSet, heightMm, referenceBytes, onProgress, signal }) {
    if (!this.worker) {
      return Promise.reject(new Error("live-preview is not initialized"));
    }
    const id = this.nextId++;
    const bytes = referenceBytes.slice();
    return new Promise((resolve, reject) => {
      const onAbort = () => {
        if (!this.pending.delete(id)) return;
        this.worker.postMessage({ type: "cancel", id });
        resolve(null);
      };
      this.pending.set(id, { resolve, reject, onProgress, signal, onAbort });
      signal?.addEventListener("abort", onAbort, { once: true });
      this.worker.postMessage(
        { type: "evaluate", id, sourceSet, heightMm, referenceBytes: bytes },
        [bytes.buffer],
      );
    });
  }
}
