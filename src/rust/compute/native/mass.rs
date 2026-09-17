//! Mass integrals for the Cauchy kernel field, reduced to surface integrals.
//!
//! The bake needs one scalar per kernel that the closed-form radial split has
//! no use for: the *total mass* it puts inside the mesh,
//!
//! ```text
//!   M = ∫_V ρ(y) dV = Σ_k w_k ∫_V (1 + σ_k²‖y − c_k‖²)^(−α_k) dV .
//! ```
//!
//! For `α = 1` the divergence theorem turns each kernel volume integral into a
//! surface integral with no special functions at all. Look for a radial field
//! `F(y) = g(r)·(y − c)`, `r = ‖y − c‖`, with `∇·F = (1 + σ²r²)^(−1)`:
//!
//! ```text
//!   ∇·F = 3g(r) + r g′(r) = (1 + σ²r²)^(−1)
//! ```
//!
//! and `g(r) = (t − atan t)/t³` with `t = σr` solves it exactly (verified by
//! substituting: the σ drops out, leaving `3g + t g′ = 1/(1+t²)`). At the
//! centre `g → 1/3` — the Taylor series below keeps that limit accurate without
//! ever forming `0/0`, which matters because the TOML kernels sit *inside* the
//! body and `r` is only guaranteed large on the boundary.
//!
//! So
//!
//! ```text
//!   ∫_V (1 + σ²‖y − c‖²)^(−1) dV = ∮_{∂V} g(σ‖y − c‖) (y − c)·n(y) dS .
//! ```
//!
//! The boundary integrand is smooth (the kernels are strictly interior), so a
//! 7-point degree-5 Dunavant rule per triangle converges to f64 round-off. The
//! `σ = 0` case degenerates to `g = 1/3` and reproduces the exact enclosed
//! volume, which is what [`body_volume`] uses and what `--selftest` pins down.

use crate::density::KernelSi;
use crate::mesh::Mesh;

/// 7-point degree-5 Dunavant rule on the reference triangle (barycentric
/// coordinates, weights summing to 1 so the triangle area factors out).
const DUNANT7: [([f64; 3], f64); 7] = [
    ([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], 0.225),
    (
        [
            0.059_715_871_789_769_81,
            0.470_142_064_105_115_1,
            0.470_142_064_105_115_1,
        ],
        0.132_394_152_788_506_2,
    ),
    (
        [
            0.470_142_064_105_115_1,
            0.059_715_871_789_769_81,
            0.470_142_064_105_115_1,
        ],
        0.132_394_152_788_506_2,
    ),
    (
        [
            0.470_142_064_105_115_1,
            0.470_142_064_105_115_1,
            0.059_715_871_789_769_81,
        ],
        0.132_394_152_788_506_2,
    ),
    (
        [
            0.797_426_985_353_087_3,
            0.101_286_507_323_456_34,
            0.101_286_507_323_456_34,
        ],
        0.125_939_180_544_827_13,
    ),
    (
        [
            0.101_286_507_323_456_34,
            0.797_426_985_353_087_3,
            0.101_286_507_323_456_34,
        ],
        0.125_939_180_544_827_13,
    ),
    (
        [
            0.101_286_507_323_456_34,
            0.101_286_507_323_456_34,
            0.797_426_985_353_087_3,
        ],
        0.125_939_180_544_827_13,
    ),
];

/// `g(t) = (t − atan t)/t³`, the radial factor of the divergence-free-adjoint
/// field.
///
/// The direct quotient loses digits badly as `t → 0`: `t − atan t` is `O(t³)`,
/// so forming it from two `O(t)` operands costs `3ε/t²` of relative accuracy
/// (already `3e-10` at `t = 1e-3`). The series
/// `1/3 − t²/5 + t⁴/7 − t⁶/9 + …` has no cancellation at all, and truncated
/// after `t¹⁴` its first dropped term at `t = 0.1` is `t¹⁶/17 ≈ 6e-18`, i.e.
/// below f64 round-off. Past the cutoff the direct form is accurate to ~1e-13.
#[inline]
pub fn radial_g(t: f64) -> f64 {
    if t.abs() < 0.1 {
        radial_g_series(t)
    } else {
        (t - t.atan()) / (t * t * t)
    }
}

/// The cancellation-free branch, kept separate so `--selftest` can compare it
/// against the direct quotient at the point where they meet.
#[inline]
fn radial_g_series(t: f64) -> f64 {
    let t2 = t * t;
    // term_k = (−1)^k t^(2k) / (2k + 3): 1/3 − t²/5 + t⁴/7 − t⁶/9 + …
    // Successive terms are the previous one times −t²(2k+1)/(2k+3).
    let mut acc = 1.0 / 3.0;
    let mut term = 1.0 / 3.0;
    for k in 1..10u32 {
        term *= -t2 * (2.0 * k as f64 + 1.0) / (2.0 * k as f64 + 3.0);
        acc += term;
    }
    acc
}

/// `∫_V (1 + σ²‖y − c‖²)^(−1) dV` by the surface reduction above.
pub fn kernel_volume(mesh: &Mesh, c: [f64; 3], sigma: f64) -> f64 {
    let mut sum = 0.0f64;
    for f in 0..mesh.face_count() {
        let [i0, i1, i2] = mesh.face(f);
        let p0 = mesh.vertex(i0 as usize);
        let p1 = mesh.vertex(i1 as usize);
        let p2 = mesh.vertex(i2 as usize);
        // Raw (un-normalised) face normal: ‖n‖ = 2·Area, which is exactly the
        // Jacobian that converts the barycentric rule into a surface integral.
        let n = cross(sub(p1, p0), sub(p2, p1));
        for (bary, w) in DUNANT7 {
            let y = [
                bary[0] * p0[0] + bary[1] * p1[0] + bary[2] * p2[0],
                bary[0] * p0[1] + bary[1] * p1[1] + bary[2] * p2[1],
                bary[0] * p0[2] + bary[1] * p1[2] + bary[2] * p2[2],
            ];
            let d = sub(y, c);
            let r = norm(d);
            // (y−c)·n̂ dS = (d·n)/‖n‖ · (‖n‖/2)·Σw·(…) = ½ Σw (d·n) g(σr).
            sum += w * radial_g(sigma * r) * dot(d, n) * 0.5;
        }
    }
    sum
}

/// Exact enclosed volume (`σ = 0` limit of [`kernel_volume`], independent of the
/// expansion centre because `∮ (y − c)·n dS = 3V`).
pub fn body_volume(mesh: &Mesh) -> f64 {
    kernel_volume(mesh, [0.0, 0.0, 0.0], 0.0)
}

/// Volume centroid `c = (1/2V) ∮ ‖y‖² n(y) dS`, which follows from
/// `∇·(½‖y‖² e_i) = y_i`. `‖y‖²` is quadratic, so the 7-point degree-5 rule is
/// exact here. The Carlson decomposition needs a point that is actually inside
/// the body, and the volume centroid is the natural choice.
pub fn volume_centroid(mesh: &Mesh) -> [f64; 3] {
    let mut acc = [0.0f64; 3];
    for f in 0..mesh.face_count() {
        let [i0, i1, i2] = mesh.face(f);
        let p0 = mesh.vertex(i0 as usize);
        let p1 = mesh.vertex(i1 as usize);
        let p2 = mesh.vertex(i2 as usize);
        let n = cross(sub(p1, p0), sub(p2, p1));
        for (bary, w) in DUNANT7 {
            let y = [
                bary[0] * p0[0] + bary[1] * p1[0] + bary[2] * p2[0],
                bary[0] * p0[1] + bary[1] * p1[1] + bary[2] * p2[1],
                bary[0] * p0[2] + bary[1] * p1[2] + bary[2] * p2[2],
            ];
            let r2 = dot(y, y);
            for k in 0..3 {
                acc[k] += w * 0.5 * r2 * n[k];
            }
        }
    }
    let v = body_volume(mesh);
    if v.abs() > 1e-30 {
        [acc[0] / v, acc[1] / v, acc[2] / v]
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// `∫_V ρ(y) dV` for the Cauchy kernel field, in kg.
pub fn kernel_field_mass(mesh: &Mesh, kernels: &[KernelSi]) -> f64 {
    let mut total = 0.0;
    for k in kernels {
        total += k.w * kernel_volume(mesh, k.c, k.sigma);
    }
    total
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// Closed form `4π (R − atan(σR)/σ)/σ²` for a ball of radius `R` centred at the
/// kernel centre — what `--selftest` compares the surface reduction against.
pub fn ball_kernel_volume(radius: f64, sigma: f64) -> f64 {
    if sigma.abs() < 1e-12 {
        4.0 / 3.0 * std::f64::consts::PI * radius.powi(3)
    } else {
        4.0 * std::f64::consts::PI * (radius - (sigma * radius).atan() / sigma) / (sigma * sigma)
    }
}

/// Triangulated UV sphere, used by `--selftest` to compare [`kernel_volume`]
/// against [`ball_kernel_volume`]. `n_lat × 2·n_lat` quads, closed at both poles.
pub fn uv_sphere(radius: f64, n_lat: usize) -> Mesh {
    let n_lon = 2 * n_lat;
    let mut mesh = Mesh::default();
    for i in 0..=n_lat {
        let theta = std::f64::consts::PI * (i as f64) / (n_lat as f64);
        let (st, ct) = theta.sin_cos();
        for j in 0..n_lon {
            let phi = 2.0 * std::f64::consts::PI * (j as f64) / (n_lon as f64);
            let (sp, cp) = phi.sin_cos();
            mesh.xyz
                .extend_from_slice(&[radius * st * cp, radius * st * sp, radius * ct]);
        }
    }
    let idx = |i: usize, j: usize| (i * n_lon + (j % n_lon)) as u32;
    for i in 0..n_lat {
        for j in 0..n_lon {
            let (a, b) = (idx(i, j), idx(i, j + 1));
            let (c, d) = (idx(i + 1, j + 1), idx(i + 1, j));
            // Skip the degenerate triangles that sit exactly on a pole.
            if i != 0 {
                mesh.faces.extend_from_slice(&[a, c, b]);
            }
            if i + 1 != n_lat {
                mesh.faces.extend_from_slice(&[a, d, c]);
            }
        }
    }
    mesh
}
