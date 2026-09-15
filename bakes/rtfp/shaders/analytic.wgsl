// Uniform-density gravity-gradient tensor, one workgroup per observation point.
//
//   W_ij(x) = G Σ_F n_j(F) · I_F[i](x)
//   I_F(x)  = Ω(x) n  +  Σ_edges (t × n) · ln((s_B + l_B)/(s_A + l_A))
//
// `l` in the edge log is the distance from `x` to the edge *line* — not to the
// face plane, which is the classic slip that still passes on the symmetry axes.
// Face data is precomputed, so the kernel is branch-free per edge and one face
// costs one `atan2` and six `ln`. Matches `src/analytic.rs` term for term.

struct Globals {
    n_points: u32,
    n_faces: u32,
    g: f32,
    // Workgroups dispatched along `x`; `y` is the stride for points past the
    // 65535-per-dimension dispatch cap.
    grid_x: u32,
}

@group(0) @binding(0) var<uniform> globals: Globals;
// Four rows per face: corner A, corner B, corner C, (unit normal, plane offset).
// Fourth component of the last row is the face's density jump (kg/m³): 1 for the
// unit-density polyhedral tensor, Δρ for the Carlson density-jump representation.
@group(0) @binding(1) var<storage, read> faces: array<vec4<f32>>;
// Observation points, xyz + pad.
@group(0) @binding(2) var<storage, read> points: array<vec4<f32>>;
// Six components per point: xx, yy, zz, xy, xz, yz.
@group(0) @binding(3) var<storage, read_write> out_w: array<f32>;

const WG: u32 = 64u;
const SLOTS: u32 = 6u;

var<workgroup> partials: array<f32, WG * SLOTS>;

fn solid_angle(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
    let an = normalize(a);
    let bn = normalize(b);
    let cn = normalize(c);
    let den = 1.0 + dot(an, bn) + dot(bn, cn) + dot(cn, an);
    return 2.0 * atan2(-dot(an, cross(bn, cn)), den);
}

/// `asinh` without the `x + √(x²+1)` cancellation for large `|x|`.
fn asinh_f(x: f32) -> f32 {
    let a = abs(x);
    return sign(x) * log(max(a + sqrt(a * a + 1.0), 1e-30));
}

/// `∫ ds/√(s² + L²)` over one edge, as the outward-normal vector.
fn edge_term(p: vec3<f32>, q: vec3<f32>, x: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let d = q - p;
    let len = length(d);
    if (len <= 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let t = d / len;
    let r = p - x;
    let s_a = dot(r, t);
    // Distance to the edge line, as a vector length so a point directly above
    // the edge does not lose it to cancellation.
    let perp = r - s_a * t;
    let l = length(perp);
    let s_b = dot(q - x, t);
    var term = 0.0;
    if (l > 0.0) {
        // ∫ ds/√(s² + l²) = asinh(s_b/l) − asinh(s_a/l). Both terms stay well
        // conditioned when `l ≪ |s|`, where `s + √(s² + l²)` cancels to zero in
        // f32 and a guard turns the term into a spurious −69.
        term = asinh_f(s_b / l) - asinh_f(s_a / l);
    } else {
        // Degenerate: the point sits exactly on the edge line, so the integral is
        // logarithmic and finite as long as the foot is outside the segment.
        term = log(max(abs(s_b), 1e-30)) - log(max(abs(s_a), 1e-30));
    }
    return cross(t, n) * term;
}

/// `I_F(x)` for face `f`.
fn face_integral(f: u32, x: vec3<f32>) -> vec3<f32> {
    let base = 4u * f;
    let a = faces[base].xyz;
    let b = faces[base + 1u].xyz;
    let c = faces[base + 2u].xyz;
    let nrm = faces[base + 3u].xyz;
    var i_vec = nrm * solid_angle(a - x, b - x, c - x);
    i_vec = i_vec + edge_term(a, b, x, nrm);
    i_vec = i_vec + edge_term(b, c, x, nrm);
    i_vec = i_vec + edge_term(c, a, x, nrm);
    return i_vec;
}

@compute @workgroup_size(WG)
fn analytic(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let pid = wid.x + wid.y * globals.grid_x;
    if (pid >= globals.n_points) {
        return;
    }
    let x = points[pid].xyz;
    var w = array<f32, SLOTS>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let n_faces = globals.n_faces;
    for (var f = lid.x; f < n_faces; f = f + WG) {
        let base = 4u * f;
        let i_vec = face_integral(f, x);
        let nrm = faces[base + 3u].xyz;
        // `weight` is the per-face density jump; it is 1.0 for the polyhedral
        // tensor the RT-FP ray solver scales by ρ(x) itself.
        let weight = faces[base + 3u].w;
        w[0] = w[0] + weight * nrm.x * i_vec.x;
        w[1] = w[1] + weight * nrm.y * i_vec.y;
        w[2] = w[2] + weight * nrm.z * i_vec.z;
        w[3] = w[3] + weight * nrm.x * i_vec.y;
        w[4] = w[4] + weight * nrm.x * i_vec.z;
        w[5] = w[5] + weight * nrm.y * i_vec.z;
    }
    for (var k = 0u; k < SLOTS; k = k + 1u) {
        partials[lid.x * SLOTS + k] = w[k];
    }
    workgroupBarrier();
    var step = WG / 2u;
    loop {
        if (step == 0u) {
            break;
        }
        if (lid.x < step) {
            for (var k = 0u; k < SLOTS; k = k + 1u) {
                let at = lid.x * SLOTS + k;
                partials[at] = partials[at] + partials[at + step * SLOTS];
            }
        }
        workgroupBarrier();
        step = step / 2u;
    }
    if (lid.x == 0u) {
        for (var k = 0u; k < SLOTS; k = k + 1u) {
            out_w[pid * SLOTS + k] = partials[k] * globals.g;
        }
    }
}
