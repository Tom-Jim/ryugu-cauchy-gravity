/**
 * Continuous-height GPU preview for the static viewer.
 *
 * Exact solver records are anchor points. Between anchors this module evaluates
 * the height response of a compact source model on the GPU, then scales the
 * anchor record face by face. The result is continuous, progressive and keeps
 * each solver's face-to-face structure from its exact reference record.
 */

const CHUNK_FACES = 8192;
const WORKGROUP_SIZE = 64;
const OBSERVER_HEADER = 40;
const RECORD_HEADER = 28;

const PREVIEW_WGSL = `
struct Globals {
  n_faces: u32,
  base_face: u32,
  n_sources: u32,
  height_m: f32,
  ref_height_m: f32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> observers: array<f32>;
@group(0) @binding(2) var<storage, read> sources: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> refs: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

fn point_at(face: u32, height_m: f32) -> vec3<f32> {
  let p = 6u * face;
  let center = vec3<f32>(observers[p], observers[p + 1u], observers[p + 2u]);
  let normal = vec3<f32>(observers[p + 3u], observers[p + 4u], observers[p + 5u]);
  return center + normal * height_m;
}

fn tensor_norm_at(point: vec3<f32>) -> f32 {
  var xx = 0.0;
  var yy = 0.0;
  var zz = 0.0;
  var xy = 0.0;
  var xz = 0.0;
  var yz = 0.0;
  for (var k = 0u; k < globals.n_sources; k = k + 1u) {
    let src = sources[k];
    let d = point - src.xyz;
    let r2 = max(dot(d, d), 0.01);
    let inv_r = inverseSqrt(r2);
    let inv_r2 = inv_r * inv_r;
    let inv_r3 = inv_r2 * inv_r;
    let inv_r5 = inv_r3 * inv_r2;
    let m = src.w;
    xx += m * (3.0 * d.x * d.x * inv_r5 - inv_r3);
    yy += m * (3.0 * d.y * d.y * inv_r5 - inv_r3);
    zz += m * (3.0 * d.z * d.z * inv_r5 - inv_r3);
    xy += m * 3.0 * d.x * d.y * inv_r5;
    xz += m * 3.0 * d.x * d.z * inv_r5;
    yz += m * 3.0 * d.y * d.z * inv_r5;
  }
  return sqrt(xx * xx + yy * yy + zz * zz + 2.0 * (xy * xy + xz * xz + yz * yz));
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn preview(@builtin(global_invocation_id) gid: vec3<u32>) {
  let face = globals.base_face + gid.x;
  if (face >= globals.n_faces) {
    return;
  }
  let reference = refs[face];
  if (!(reference == reference) || reference == 0.0) {
    output[face] = reference;
    return;
  }
  let at_height = tensor_norm_at(point_at(face, globals.height_m));
  let at_reference = tensor_norm_at(point_at(face, globals.ref_height_m));
  output[face] = reference * (at_height / max(at_reference, 1e-30));
}
`;

function parseObservers(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(0, true) !== 0x424f5952 || view.getUint32(4, true) !== 1) {
    throw new Error("invalid live-preview observer asset");
  }
  const count = view.getUint32(8, true);
  if (bytes.byteLength < OBSERVER_HEADER + count * 12) {
    throw new Error("truncated live-preview observer asset");
  }
  const min = [view.getFloat32(16, true), view.getFloat32(20, true), view.getFloat32(24, true)];
  const max = [view.getFloat32(28, true), view.getFloat32(32, true), view.getFloat32(36, true)];
  const out = new Float32Array(count * 6);
  for (let face = 0; face < count; face++) {
    const src = OBSERVER_HEADER + face * 12;
    const dst = face * 6;
    for (let axis = 0; axis < 3; axis++) {
      const q = view.getInt16(src + axis * 2, true);
      out[dst + axis] = min[axis] + ((q + 32768) / 65535) * (max[axis] - min[axis]);
      const n = view.getInt16(src + 6 + axis * 2, true);
      out[dst + 3 + axis] = n / 32767;
    }
  }
  return { count, data: out };
}

function parseSources(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(0, true) !== 0x31535952 || view.getUint32(4, true) !== 1) {
    throw new Error("invalid live-preview source asset");
  }
  const count = view.getUint32(8, true);
  let offset = 16;
  const sets = new Map();
  for (let i = 0; i < count; i++) {
    const id = view.getUint32(offset, true);
    const n = view.getUint32(offset + 4, true);
    offset += 16;
    if (bytes.byteLength < offset + n * 16) {
      throw new Error("truncated live-preview source asset");
    }
    const data = new Float32Array(n * 4);
    for (let j = 0; j < data.length; j++) data[j] = view.getFloat32(offset + j * 4, true);
    sets.set(id, data);
    offset += data.byteLength;
  }
  return sets;
}

const SOURCE_SET = {
  cauchy: 0,
  elliptic: 1,
  constant: 2,
};

export class LivePreview {
  constructor(device, options) {
    this.device = device;
    // Keep the adapter alive for the page lifetime. Some WebGPU
    // implementations tie the device's external instance to the adapter that
    // created it, and allowing that wrapper to be collected invalidates later
    // mapAsync calls.
    this.adapter = options.adapter;
    this.observerUrl = options.observerUrl;
    this.sourceUrl = options.sourceUrl;
    this.faceCount = 0;
    this.observerBuffer = null;
    this.sourceBuffers = new Map();
    this.sourceCounts = new Map();
    this.refBuffer = null;
    this.outputBuffer = null;
    this.staging = null;
    this.uniform = null;
    this.pipeline = null;
    this.layout = null;
    this.bindGroups = new Map();
  }

  async init() {
    const [observerRes, sourceRes] = await Promise.all([
      fetch(this.observerUrl, { cache: "force-cache" }),
      fetch(this.sourceUrl, { cache: "force-cache" }),
    ]);
    if (!observerRes.ok || !sourceRes.ok) {
      throw new Error("failed to load live-preview GPU assets");
    }
    const observers = parseObservers(new Uint8Array(await observerRes.arrayBuffer()));
    const sources = parseSources(new Uint8Array(await sourceRes.arrayBuffer()));
    this.faceCount = observers.count;

    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.observerBuffer = this.device.createBuffer({
      label: "live-preview observers",
      size: Math.max(observers.data.byteLength, 4),
      usage: storage,
    });
    this.device.queue.writeBuffer(this.observerBuffer, 0, observers.data);

    for (const [id, data] of sources) {
      const buffer = this.device.createBuffer({
        label: `live-preview sources ${id}`,
        size: Math.max(data.byteLength, 4),
        usage: storage,
      });
      this.device.queue.writeBuffer(buffer, 0, data);
      this.sourceBuffers.set(id, buffer);
      this.sourceCounts.set(id, data.length / 4);
    }

    const recordBytes = this.faceCount * 4;
    this.refBuffer = this.device.createBuffer({
      label: "live-preview reference scalars",
      size: recordBytes,
      usage: storage,
    });
    this.outputBuffer = this.device.createBuffer({
      label: "live-preview output scalars",
      size: recordBytes,
      usage: storage | GPUBufferUsage.COPY_SRC,
    });
    this.staging = this.device.createBuffer({
      label: "live-preview readback",
      size: CHUNK_FACES * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.uniform = this.device.createBuffer({
      label: "live-preview globals",
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const module = this.device.createShaderModule({
      label: "live-preview",
      code: PREVIEW_WGSL,
    });
    this.layout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
      ],
    });
    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [this.layout],
    });
    this.pipeline = this.device.createComputePipeline({
      label: "live-preview",
      layout: pipelineLayout,
      compute: { module, entryPoint: "preview" },
    });
  }

  _bindGroup(sourceSet) {
    const id = SOURCE_SET[sourceSet] ?? SOURCE_SET.constant;
    if (this.bindGroups.has(id)) return { bindGroup: this.bindGroups.get(id), id };
    const bindGroup = this.device.createBindGroup({
      label: `live-preview ${sourceSet}`,
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.uniform } },
        { binding: 1, resource: { buffer: this.observerBuffer } },
        { binding: 2, resource: { buffer: this.sourceBuffers.get(id) } },
        { binding: 3, resource: { buffer: this.refBuffer } },
        { binding: 4, resource: { buffer: this.outputBuffer } },
      ],
    });
    this.bindGroups.set(id, bindGroup);
    return { bindGroup, id };
  }

  async evaluate({ sourceSet, heightMm, referenceBytes, onProgress, signal }) {
    if (!this.pipeline) throw new Error("live-preview is not initialized");
    const bytes = referenceBytes.slice();
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (
      view.getUint32(0, true) !== 0x52484746
      || view.getUint32(4, true) !== 5
      || view.getUint32(8, true) !== this.faceCount
      || bytes.byteLength < RECORD_HEADER + this.faceCount * 4
    ) {
      throw new Error("live-preview reference record is incompatible");
    }
    const refHeightMm = view.getFloat32(16, true);
    view.setFloat32(16, heightMm, true);
    if (Math.abs(heightMm - refHeightMm) < 1e-6) {
      onProgress?.(1);
      return bytes;
    }

    const refScalars = new Float32Array(this.faceCount);
    for (let i = 0; i < this.faceCount; i++) {
      refScalars[i] = view.getFloat32(RECORD_HEADER + i * 4, true);
    }
    const outputScalars = new Float32Array(bytes.buffer, bytes.byteOffset + RECORD_HEADER, this.faceCount);
    this.device.queue.writeBuffer(this.refBuffer, 0, refScalars);

    const { bindGroup, id } = this._bindGroup(sourceSet);
    const nSources = this.sourceCounts.get(id) ?? 0;
    const chunks = Math.ceil(this.faceCount / CHUNK_FACES);
    onProgress?.(0);
    for (let chunk = 0; chunk < chunks; chunk++) {
      if (signal?.aborted) return null;
      const base = chunk * CHUNK_FACES;
      const count = Math.min(CHUNK_FACES, this.faceCount - base);
      const globals = new ArrayBuffer(32);
      const g = new DataView(globals);
      g.setUint32(0, this.faceCount, true);
      g.setUint32(4, base, true);
      g.setUint32(8, nSources, true);
      g.setFloat32(12, heightMm * 1e-3, true);
      g.setFloat32(16, refHeightMm * 1e-3, true);
      this.device.queue.writeBuffer(this.uniform, 0, globals);

      const encoder = this.device.createCommandEncoder({ label: `live-preview ${chunk}` });
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(count / WORKGROUP_SIZE));
      pass.end();
      encoder.copyBufferToBuffer(this.outputBuffer, base * 4, this.staging, 0, count * 4);
      this.device.queue.submit([encoder.finish()]);

      await this.staging.mapAsync(GPUMapMode.READ, 0, count * 4);
      const mapped = new Float32Array(this.staging.getMappedRange(0, count * 4));
      outputScalars.set(mapped, base);
      this.staging.unmap();
      if (signal?.aborted) return null;
      onProgress?.((chunk + 1) / chunks);
    }
    return bytes;
  }
}
