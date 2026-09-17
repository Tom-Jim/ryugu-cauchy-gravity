// Final tensor composition and Frobenius reduction for browser computation.
// Keeping these operations in WGSL prevents the JavaScript coordinator from
// becoming a second numerical implementation of the solvers.

struct Globals {
  n_points: u32,
  n_kernels: u32,
  _pad0: u32,
  _pad1: u32,
};

struct Kernel {
  c: vec3<f32>,
  sigma: f32,
  w: f32,
  alpha: f32,
  _pad1: f32,
  _pad2: f32,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> analytic_tensors: array<f32>;
@group(0) @binding(2) var<storage, read> remainder_tensors: array<f32>;
@group(0) @binding(3) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> kernels: array<Kernel>;
@group(0) @binding(5) var<storage, read_write> face_scalars: array<f32>;

fn frobenius(h: array<f32, 6>) -> f32 {
  return sqrt(
    h[0] * h[0] + h[1] * h[1] + h[2] * h[2]
      + 2.0 * (h[3] * h[3] + h[4] * h[4] + h[5] * h[5]),
  );
}

@compute @workgroup_size(64)
fn analytic_scalar(@builtin(global_invocation_id) gid: vec3<u32>) {
  let point = gid.x;
  if (point >= globals.n_points) {
    return;
  }
  let base = point * 6u;
  var h = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  for (var component = 0u; component < 6u; component = component + 1u) {
    h[component] = analytic_tensors[base + component];
  }
  face_scalars[point] = frobenius(h);
}

@compute @workgroup_size(64)
fn ray_scalar(@builtin(global_invocation_id) gid: vec3<u32>) {
  let point = gid.x;
  if (point >= globals.n_points) {
    return;
  }
  let x = points[point].xyz;
  var density = 0.0;
  for (var k = 0u; k < globals.n_kernels; k = k + 1u) {
    let kernel = kernels[k];
    let d = x - kernel.c;
    density = density + kernel.w
      * pow(max(1.0 + kernel.sigma * kernel.sigma * dot(d, d), 1e-30), -kernel.alpha);
  }
  let base = point * 6u;
  var h = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  for (var component = 0u; component < 6u; component = component + 1u) {
    h[component] = analytic_tensors[base + component] * density
      + remainder_tensors[base + component];
  }
  face_scalars[point] = frobenius(h);
}
