// Mascon gravity-gradient evaluation over a Barnes-Hut tree.
//
// This is the browser GPU mirror of the point-mass direct sum. Leaves are
// evaluated one voxel at a time; well-separated octree nodes use their exact
// monopole mass and centre of mass. The tree is built from the same 192^3
// occupied-cell source asset as the native Mascon solver, and no value from
// another solver path is read.

struct Globals {
  counts: vec4<f32>,       // n_points, n_nodes, G, theta
  bounds_min: vec4<f32>,   // xyz, density mode (0 Cauchy, 1 elliptic, 2 constant)
  cell: vec4<f32>,         // xyz, constant density
};

@group(0) @binding(0) var<storage, read> globals: Globals;
// Four vec4 rows per node:
//   centre.xyz + half extent
//   mass Cauchy + mass elliptic + left + right
//   centre of mass.xyz + leaf start
//   leaf count + total occupied-point count
@group(0) @binding(1) var<storage, read> nodes: array<vec4<f32>>;
// Original point records: x | y<<16, z | pad<<16, mass Cauchy bits, mass elliptic bits.
@group(0) @binding(2) var<storage, read> point_records: array<u32>;
// Point indices ordered by tree leaf.
@group(0) @binding(3) var<storage, read> point_order: array<u32>;
@group(0) @binding(4) var<storage, read> observers: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> face_scalars: array<f32>;

const STACK: u32 = 128u;

fn node_f32(index: u32, slot: u32) -> f32 {
  return nodes[index * 4u + slot].x;
}

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
  let base = node * 4u;
  let first = u32(nodes[base + 2u].w + 0.5);
  let count = u32(nodes[base + 3u].x + 0.5);
  for (var i = 0u; i < count; i = i + 1u) {
    let point = point_order[first + i];
    add_point_mass(out, observer, point_position(point), point_mass(point));
  }
}

fn add_node_monopole(out: ptr<function, array<f32, 6>>, node: u32, observer: vec3<f32>) {
  let base = node * 4u;
  let mode = u32(globals.bounds_min.w + 0.5);
  var mass = select(
    nodes[base + 1u].y,
    nodes[base + 1u].x,
    mode == 0u,
  );
  if (mode == 2u) {
    mass = globals.cell.w * globals.cell.x * globals.cell.y * globals.cell.z
      * nodes[base + 3u].y;
  }
  add_point_mass(out, observer, nodes[base + 2u].xyz, mass);
}

@compute @workgroup_size(64)
fn mascon_tree(@builtin(global_invocation_id) gid: vec3<u32>) {
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
    let base = node * 4u;
    // Node records are SI: `nodes[base].xyz` is the box centre in metres and
    // `nodes[base].w` its largest half edge in metres. The same convention is
    // used by the native mirror (`src/rust/compute/native/mascon.rs`).
    let center = nodes[base].xyz;
    let half = nodes[base].w;
    let distance = max(length(observer - center), 1e-12);
    // Children are indices, or exactly -1.0 for "absent". Test the sign first:
    // casting -1.0 straight to an integer would fold it onto child 0.
    let left_f = nodes[base + 1u].z;
    let right_f = nodes[base + 1u].w;
    let leaf_count = u32(nodes[base + 3u].x + 0.5);
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
      // The stack cannot hold both children: keep the mass by merging instead
      // of dropping the subtree.
      add_node_monopole(&tensor, node, observer);
    }
  }
  face_scalars[face] = sqrt(
    tensor[0] * tensor[0] + tensor[1] * tensor[1] + tensor[2] * tensor[2]
      + 2.0 * (tensor[3] * tensor[3] + tensor[4] * tensor[4] + tensor[5] * tensor[5]),
  );
}
