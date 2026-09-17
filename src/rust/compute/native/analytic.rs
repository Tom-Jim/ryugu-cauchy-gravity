//! Uniform-density gravity-gradient tensor in closed form, per observation point,
//! as a **surface integral** over the mesh.
//!
//! For a homogeneous body the divergence theorem gives
//!
//! ```text
//!   W_ij(x) = ∂_i∂_j V(x) = G ρ ∮_{∂V} (x_i − y_i)/|x − y|³ · n_j(y) dS(y)
//! ```
//!
//! (valid for `x` outside the body; inside, an extra `−4πGρ/3 · δ_ij` contact
//! term appears — the RT-FP observation surface is always outside, so the bake
//! never needs it). For one flat triangle `F` with unit normal `n` the vector
//! integral is
//!
//! ```text
//!   I_F(x) = ∫_F (x − y)/|x − y|³ dS
//!          = Ω(x) · n  +  Σ_{edges e} n_e · ln((s_B + l_B)/(s_A + l_A))
//! ```
//!
//! with `Ω` the signed solid angle of `F` seen from `x`, `n_e` the in-plane
//! *outward* edge normal `dir × n`, `s_A = (A − x)·dir`, `s_B = s_A + |AB|` the
//! signed distances of the two edge ends measured along the edge, and
//! `l_A = |x − A|`, `l_B = |x − B|` the **3-D** distances to those ends.
//!
//! The `l` in the edge log is the distance from `x` to the *edge line*, not to
//! the face plane: writing `L² = |x − A|² − s_A²` and `l = √(s² + L²)` makes the
//! edge integrand `∫ ds/√(s² + L²)` exact. Using the plane distance `h` instead
//! is the classic slip — it still reproduces the symmetry axes (face centre,
//! far field), where the off-diagonal components vanish identically, but it is
//! wrong everywhere else. `--selftest` pins this down against the ESA library at
//! 1 mm above a face centre, an edge, a corner and a vertex.
//!
//! Assembling `W_ij = G ρ Σ_F n_j(F) · I_F[i]` costs one `atan2` and six `ln`
//! per face — the same order as the ESA library's Tsoulis formulation, but with
//! no projection case analysis and no per-point plane solve, so one face maps
//! onto one GPU thread with no branches. `src/wgsl/compute/werner.wgsl` is the same
//! arithmetic in f32; `--selftest` compares both against the ESA library.

use crate::mesh::Mesh;
use crate::tensor::Sym6;
use crate::G;

/// Point-independent per-face data — exactly the buffer the analytic shader
/// consumes (13 `f32` in, four `vec4` slots per face).
#[derive(Clone, Copy, Debug, Default)]
pub struct Face {
    /// Corner positions, in the mesh winding order.
    pub corners: [[f64; 3]; 3],
    /// Unit outward normal, `normal(v1−v0, v2−v1)` (the ESA library's
    /// `buildUnitNormalOfPlane`), so the orientation matches the Werner record.
    pub n: [f64; 3],
    /// Density jump across the face, in kg/m³. `precompute` leaves this at 1 so
    /// the polyhedral tensor stays "per unit density"; `carlson.rs` fills in the
    /// per-face Δρ of its density-jump surface representation instead.
    pub weight: f64,
}

/// Precompute the point-independent face data (once per mesh).
pub fn precompute(mesh: &Mesh) -> Vec<Face> {
    let mut out = Vec::with_capacity(mesh.face_count());
    for f in 0..mesh.face_count() {
        let [i0, i1, i2] = mesh.face(f);
        let corners = [
            mesh.vertex(i0 as usize),
            mesh.vertex(i1 as usize),
            mesh.vertex(i2 as usize),
        ];
        let n = normalize(cross(
            sub(corners[1], corners[0]),
            sub(corners[2], corners[1]),
        ));
        out.push(Face {
            corners,
            n,
            weight: 1.0,
        });
    }
    out
}

/// `∫ ds/√(s² + L²)` over the edge, i.e. `ln((s_B + l_B)/(s_A + l_A))`.
///
/// `l` is the distance from `x` to the edge **line** (so `l → 0` when `x` sits
/// exactly above the edge). For `s < 0` the direct `s + l` loses every
/// significant digit as `l → 0`; the identity `s + l = L²/(l − s)` is
/// algebraically identical and never subtracts two nearly equal numbers, and it
/// stays finite in the `l = 0` limit as long as both ends are on the same side.
#[inline]
pub fn edge_log(s_a: f64, s_b: f64, l: f64) -> f64 {
    let l_a = (s_a * s_a + l * l).sqrt();
    let l_b = (s_b * s_b + l * l).sqrt();
    if s_a >= 0.0 || s_b > 0.0 {
        ((s_b + l_b) / (s_a + l_a).max(f64::MIN_POSITIVE)).ln()
    } else {
        ((l_a - s_a) / (l_b - s_b).max(f64::MIN_POSITIVE)).ln()
    }
}

/// Signed solid angle of the triangle with corner vectors `a, b, c` (all taken
/// from the observation point), positive when `x` is on the `n`-side of the face.
///
/// Van Oosterom–Strackee with the triple product taken in the sign convention
/// that pairs with `n = normal(v1−v0, v2−v1)`; `atan2` keeps it correct past the
/// `π` wrap, which matters at 1 mm standoff where a face can subtend nearly `2π`.
#[inline]
pub fn solid_angle(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let (an, bn, cn) = (normalize(a), normalize(b), normalize(c));
    let denom = 1.0 + dot(an, bn) + dot(bn, cn) + dot(cn, an);
    2.0 * (-dot(an, cross(bn, cn))).atan2(denom)
}

/// `I_F(x)` — the vector surface integral of one face.
pub fn face_integral(f: &Face, x: [f64; 3]) -> [f64; 3] {
    let mut i_vec = [0.0f64; 3];
    for e in 0..3 {
        let a = f.corners[e];
        let b = f.corners[(e + 1) % 3];
        let dir = normalize(sub(b, a));
        // ∫_F ∇^{2D}(1/r) dS = ∮_{∂F} (1/r) n_e ds (2-D gradient theorem), with
        // the edge's outward in-plane normal `dir × n`.
        let outward = cross(dir, f.n);
        let r_a = sub(a, x);
        let par = dot(r_a, dir);
        // Distance to the edge *line*, kept as a vector length so that a point
        // directly above the edge does not lose it to cancellation.
        let perp = sub(r_a, scale(dir, par));
        let l = norm(perp);
        let log_term = edge_log(par, dot(sub(b, x), dir), l);
        for k in 0..3 {
            i_vec[k] += outward[k] * log_term;
        }
    }
    let omega = solid_angle(
        sub(f.corners[0], x),
        sub(f.corners[1], x),
        sub(f.corners[2], x),
    );
    for (slot, n) in i_vec.iter_mut().zip(f.n) {
        *slot += omega * n;
    }
    i_vec
}

/// Accumulate one face into `W_ij = G Σ_F n_j · I_F[i]` (unit density).
#[inline]
pub fn add_face(w: &mut Sym6, f: &Face, x: [f64; 3]) {
    let i_vec = face_integral(f, x);
    let k = G * f.weight;
    w[0] += k * f.n[0] * i_vec[0];
    w[1] += k * f.n[1] * i_vec[1];
    w[2] += k * f.n[2] * i_vec[2];
    w[3] += k * f.n[0] * i_vec[1];
    w[4] += k * f.n[0] * i_vec[2];
    w[5] += k * f.n[1] * i_vec[2];
}

/// `W(x)` for unit density (the caller scales by the density at `x`).
pub fn hessian6(faces: &[Face], x: [f64; 3]) -> Sym6 {
    let mut w = [0.0f64; 6];
    for f in faces {
        add_face(&mut w, f, x);
    }
    w
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: [f64; 3], k: f64) -> [f64; 3] {
    [a[0] * k, a[1] * k, a[2] * k]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn normalize(a: [f64; 3]) -> [f64; 3] {
    let n = norm(a);
    if n > f64::MIN_POSITIVE {
        [a[0] / n, a[1] / n, a[2] / n]
    } else {
        [0.0, 0.0, 0.0]
    }
}
