// Build the observation surface on the GPU.  JavaScript only selects the
// height and face range; all geometric arithmetic remains in WGSL.

struct Globals {
  count: u32,
  face_offset: u32,
  height_m: f32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> globals: Globals;
// Four vec4 rows per observation face: A, B, C, outward unit normal.
@group(0) @binding(1) var<storage, read> faces: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> observer_points: array<vec4<f32>>;

@compute @workgroup_size(64)
fn build_observers(@builtin(global_invocation_id) gid: vec3<u32>) {
  let local_face = gid.x;
  if (local_face >= globals.count) {
    return;
  }
  let face = globals.face_offset + local_face;
  let base = face * 4u;
  let center = (faces[base].xyz + faces[base + 1u].xyz + faces[base + 2u].xyz) / 3.0;
  let point = center + faces[base + 3u].xyz * globals.height_m;
  observer_points[local_face] = vec4<f32>(point, 1.0);
}
