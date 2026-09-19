// Carlson uniform-density boundary pipeline.
//
// Every triangle carries a unit surface weight.  The physical constant density
// is applied once by the dispatch scale, so the tensor is one direct boundary
// integral:
//
//   H_ij(x) = G rho sum_T n_j(T) I_T[i](x)
//
// This dispatch shares no compute pipeline with RT-FP. In particular it does
// not read the RT-FP radial remainder or execute the ray traversal. The mesh
// and star-cone triangles are evaluated completely from their boundary data.

struct CarlsonGlobals {
  n_points: u32,
  n_faces: u32,
  g: f32,
  grid_x: u32,
  point_offset: u32,
}

@group(0) @binding(0) var<uniform> globals: CarlsonGlobals;
// Four vec4 rows per triangle: A, B, C, then (unit normal, density jump).
@group(0) @binding(1) var<storage, read> triangles: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> observers: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> carlson_out: array<f32>;

const WG: u32 = 64u;
const COMPONENTS: u32 = 6u;

var<workgroup> sums: array<f32, WG * COMPONENTS>;

fn signed_solid_angle(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
  let ua = normalize(a);
  let ub = normalize(b);
  let uc = normalize(c);
  let denom = 1.0 + dot(ua, ub) + dot(ub, uc) + dot(uc, ua);
  return 2.0 * atan2(-dot(ua, cross(ub, uc)), denom);
}

fn stable_asinh(x: f32) -> f32 {
  let a = abs(x);
  var value = 0.0;
  if (a < 0.01) {
    let a2 = a * a;
    value = a * (1.0 + a2 * (-0.1666666667 + 0.075 * a2));
  } else if (a > 4096.0) {
    value = log(a) + 0.6931471805599453;
  } else {
    value = log(a + sqrt(1.0 + a * a));
  }
  return select(-value, value, x >= 0.0);
}

fn edge_log_integral(
  p: vec3<f32>,
  q: vec3<f32>,
  observer: vec3<f32>,
  normal: vec3<f32>,
) -> vec3<f32> {
  let edge = q - p;
  let edge_len = length(edge);
  if (edge_len <= 0.0) {
    return vec3<f32>(0.0);
  }
  let tangent = edge / edge_len;
  let rp = p - observer;
  let rq = q - observer;
  let along_p = dot(rp, tangent);
  let along_q = dot(rq, tangent);
  let perpendicular = rp - along_p * tangent;
  let line_distance = length(perpendicular);
  var value = 0.0;
  if (line_distance > 0.0) {
    value = stable_asinh(along_q / line_distance)
      - stable_asinh(along_p / line_distance);
  } else {
    value = log(max(abs(along_q), 1e-30)) - log(max(abs(along_p), 1e-30));
  }
  return cross(tangent, normal) * value;
}

fn triangle_flux(triangle: u32, observer: vec3<f32>) -> vec3<f32> {
  let row = 4u * triangle;
  let a = triangles[row].xyz;
  let b = triangles[row + 1u].xyz;
  let c = triangles[row + 2u].xyz;
  let normal = triangles[row + 3u].xyz;
  var flux = normal * signed_solid_angle(a - observer, b - observer, c - observer);
  flux += edge_log_integral(a, b, observer, normal);
  flux += edge_log_integral(b, c, observer, normal);
  flux += edge_log_integral(c, a, observer, normal);
  return flux;
}

@compute @workgroup_size(WG)
fn carlson_surface(
  @builtin(workgroup_id) workgroup: vec3<u32>,
  @builtin(local_invocation_id) local: vec3<u32>,
) {
  let local_point = workgroup.x + workgroup.y * globals.grid_x;
  let point = local_point + globals.point_offset;
  let point_is_active = local_point < globals.n_points;
  var observer = vec3<f32>(0.0);
  if (point_is_active) {
    observer = observers[point].xyz;
  }
  var accumulated = array<f32, COMPONENTS>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  for (var face = local.x; face < globals.n_faces; face += WG) {
    let row = 4u * face;
    let normal = triangles[row + 3u].xyz;
    let weight = triangles[row + 3u].w;
    let flux = triangle_flux(face, observer);
    accumulated[0] += weight * normal.x * flux.x;
    accumulated[1] += weight * normal.y * flux.y;
    accumulated[2] += weight * normal.z * flux.z;
    // The continuum sum is symmetric, but each individual face contribution
    // need not be. Accumulate the symmetric projection before reduction so a
    // different summation order cannot leave a hidden H_ij/H_ji discrepancy.
    accumulated[3] += 0.5 * weight * (normal.x * flux.y + normal.y * flux.x);
    accumulated[4] += 0.5 * weight * (normal.x * flux.z + normal.z * flux.x);
    accumulated[5] += 0.5 * weight * (normal.y * flux.z + normal.z * flux.y);
  }

  for (var component = 0u; component < COMPONENTS; component += 1u) {
    sums[local.x * COMPONENTS + component] = accumulated[component];
  }
  workgroupBarrier();

  // WG is fixed at 64. Keep the six reduction stages explicit so the shader
  // has no dynamic reduction exit or branch divergence.
  if (local.x < 32u) {
    for (var component = 0u; component < COMPONENTS; component += 1u) {
      let slot = local.x * COMPONENTS + component;
      sums[slot] += sums[slot + 32u * COMPONENTS];
    }
  }
  workgroupBarrier();
  if (local.x < 16u) {
    for (var component = 0u; component < COMPONENTS; component += 1u) {
      let slot = local.x * COMPONENTS + component;
      sums[slot] += sums[slot + 16u * COMPONENTS];
    }
  }
  workgroupBarrier();
  if (local.x < 8u) {
    for (var component = 0u; component < COMPONENTS; component += 1u) {
      let slot = local.x * COMPONENTS + component;
      sums[slot] += sums[slot + 8u * COMPONENTS];
    }
  }
  workgroupBarrier();
  if (local.x < 4u) {
    for (var component = 0u; component < COMPONENTS; component += 1u) {
      let slot = local.x * COMPONENTS + component;
      sums[slot] += sums[slot + 4u * COMPONENTS];
    }
  }
  workgroupBarrier();
  if (local.x < 2u) {
    for (var component = 0u; component < COMPONENTS; component += 1u) {
      let slot = local.x * COMPONENTS + component;
      sums[slot] += sums[slot + 2u * COMPONENTS];
    }
  }
  workgroupBarrier();
  if (local.x == 0u && point_is_active) {
    for (var component = 0u; component < COMPONENTS; component += 1u) {
      sums[component] += sums[COMPONENTS];
    }
  }
  workgroupBarrier();

  if (local.x == 0u && point_is_active) {
    let base = point * COMPONENTS;
    for (var component = 0u; component < COMPONENTS; component += 1u) {
      carlson_out[base + component] = sums[component] * globals.g;
    }
  }
}
