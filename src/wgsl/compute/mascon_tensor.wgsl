// Diagnostic-only Mascon tensor output.
// This mirrors mascon_tree.wgsl exactly, but preserves all six Hessian
// components instead of reducing them to a face scalar.

struct Globals {
  counts: vec4<f32>,
  bounds_min: vec4<f32>,
  cell: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> globals: Globals;
@group(0) @binding(1) var<storage, read> nodes: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> point_records: array<u32>;
@group(0) @binding(3) var<storage, read> point_order: array<u32>;
@group(0) @binding(4) var<storage, read> observers: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> tensors: array<f32>;

const STACK: u32 = 128u;

fn point_position(index: u32) -> vec3<f32> {
  let word = point_records[index * 4u];
  let x = f32(word & 0xffffu);
  let y = f32((word >> 16u) & 0xffffu);
  let word_z = point_records[index * 4u + 1u];
  let z = f32(word_z & 0xffffu);
  return globals.bounds_min.xyz + (vec3<f32>(x, y, z) + vec3<f32>(0.5)) * globals.cell.xyz;
}

fn point_mass(index: u32) -> f32 {
  let mode = u32(globals.bounds_min.w + 0.5);
  if (mode == 0u) {
    return bitcast<f32>(point_records[index * 4u + 2u]);
  }
  if (mode == 1u) {
    return bitcast<f32>(point_records[index * 4u + 3u]);
  }
  return globals.cell.w * globals.cell.x * globals.cell.y * globals.cell.z;
}

fn add_point_mass(out: ptr<function, array<f32, 6>>, observer: vec3<f32>, source: vec3<f32>, mass: f32) {
  let d = observer - source;
  let r2 = max(dot(d, d), 1e-12);
  let inv_r = inverseSqrt(r2);
  let inv_r2 = inv_r * inv_r;
  let inv_r3 = inv_r2 * inv_r;
  let inv_r5 = inv_r3 * inv_r2;
  let k = globals.counts.z * mass;
  (*out)[0] += k * (3.0 * d.x * d.x * inv_r5 - inv_r3);
  (*out)[1] += k * (3.0 * d.y * d.y * inv_r5 - inv_r3);
  (*out)[2] += k * (3.0 * d.z * d.z * inv_r5 - inv_r3);
  (*out)[3] += k * 3.0 * d.x * d.y * inv_r5;
  (*out)[4] += k * 3.0 * d.x * d.z * inv_r5;
  (*out)[5] += k * 3.0 * d.y * d.z * inv_r5;
}

fn add_leaf(out: ptr<function, array<f32, 6>>, node: u32, observer: vec3<f32>) {
  let base = node * 5u;
  let first = u32(nodes[base + 2u].w + 0.5);
  let count = u32(nodes[base + 3u].w + 0.5);
  for (var i = 0u; i < count; i = i + 1u) {
    let point = point_order[first + i];
    add_point_mass(out, observer, point_position(point), point_mass(point));
  }
}

fn add_node_monopole(out: ptr<function, array<f32, 6>>, node: u32, observer: vec3<f32>) {
  let base = node * 5u;
  let mode = u32(globals.bounds_min.w + 0.5);
  var mass = select(nodes[base + 1u].y, nodes[base + 1u].x, mode == 0u);
  if (mode == 2u) {
    mass = globals.cell.w * globals.cell.x * globals.cell.y * globals.cell.z
      * nodes[base + 4u].w;
  }
  let center = select(
    select(nodes[base + 4u].xyz, nodes[base + 3u].xyz, mode == 1u),
    nodes[base + 2u].xyz,
    mode == 0u,
  );
  add_point_mass(out, observer, center, mass);
}

@compute @workgroup_size(64)
fn mascon_tensor(@builtin(global_invocation_id) gid: vec3<u32>) {
  let face = gid.x;
  if (face >= u32(globals.counts.x + 0.5)) {
    return;
  }
  let observer = observers[face].xyz;
  var tensor = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  var stack: array<u32, STACK>;
  var top = 0u;
  stack[top] = 0u;
  top = 1u;
  let theta = globals.counts.w;
  loop {
    if (top == 0u) {
      break;
    }
    top = top - 1u;
    let node = stack[top];
    let base = node * 5u;
    let center = nodes[base].xyz;
    let half = nodes[base].w;
    let distance = max(length(observer - center), 1e-12);
    let left_f = nodes[base + 1u].z;
    let right_f = nodes[base + 1u].w;
    let leaf_count = u32(nodes[base + 3u].w + 0.5);
    if (leaf_count > 0u) {
      add_leaf(&tensor, node, observer);
    } else if (half / distance < theta) {
      add_node_monopole(&tensor, node, observer);
    } else if (top + 2u <= STACK) {
      if (right_f >= 0.0) {
        stack[top] = u32(right_f + 0.5);
        top = top + 1u;
      }
      if (left_f >= 0.0) {
        stack[top] = u32(left_f + 0.5);
        top = top + 1u;
      }
    } else {
      add_node_monopole(&tensor, node, observer);
    }
  }
  let base = face * 6u;
  for (var component = 0u; component < 6u; component = component + 1u) {
    tensors[base + component] = tensor[component];
  }
}
