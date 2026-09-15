//! Carlson-style density-jump surface representation — the independent check on
//! the RT-FP ray/remainder split.
//!
//! ## Why it is a different algorithm
//!
//! RT-FP writes `H(x) = ρ(x)·W(x) + G Σ_q ω_q T(u_q) Σ_k w_k R_k(u_q)`: an
//! analytic polyhedral tensor for the *uniform* part, plus a directional
//! quadrature that carries the deviation from uniform density. Both halves lean
//! on the ray/BVH machinery and on the closed-form radial remainder `R_k`.
//!
//! This module never touches a ray. It uses the other classical reduction —
//! Carlson's boundary-integral formulation — namely that a *piecewise constant*
//! density field is completely described by its jump surfaces, so the Hessian is
//! a pure surface integral over those surfaces with the jump as the weight:
//!
//! ```text
//!   H_ij(x) = G Σ_F Δρ_F · n_j(F) · I_F[i](x)
//! ```
//!
//! with the same `I_F` (solid angle + edge logs) `analytic.rs` uses. That is one
//! GPU pass over a face list and no quadrature at all.
//!
//! ## Where the jump surfaces come from
//!
//! Star-cone the body from its volume centroid `o`: every mesh face `f` becomes a
//! tetrahedron `(o, v0, v1, v2)` carrying the constant density `ρ_f`. Each
//! tetrahedron contributes its own closed boundary, oriented outward from that
//! tetrahedron; interior faces are shared by two tetrahedra and cancel by
//! superposition. Summing the individual boundaries instead of pre-cancelling
//! them through an edge-adjacency table removes a whole class of orientation
//! bugs.
//!
//! ## Radial refinement — why one constant per cone is not enough
//!
//! A cone spans the whole way from the centroid to the surface, so with a single
//! density per cone the representation error is the *radial* variation of `ρ`
//! across ~0.3 km. That is fine for a nearly uniform field and useless for a
//! stratified one: a mantle–core jump lands inside a cone and is smeared over its
//! whole length.
//!
//! So each cone is cut into `K` slabs by the scale `s` along the cone, and each
//! slab carries the volume-mean density of the shell it covers. Slab boundaries
//! are placed by *equidistributing the variation* of `ρ` along the cone axis, so
//! a steep transition collects boundaries instead of a smooth gradient. `K` is
//! chosen per cone from the cone's own total variation against a tolerance, then
//! a global face budget relaxes that tolerance until the emitted triangle count
//! fits — the refinement is therefore self-limiting rather than a fixed cost, and
//! the tolerance it actually achieved is reported by the bake.
//!
//! The slab boundary between scales `s_a < s_b` is a frustum. Its closed boundary
//! is three planar side quads (six triangles) plus the far cap, and the near cap
//! is the previous slab's far cap with the opposite orientation, which is why the
//! caps can be summed as `g_i − g_{i+1}` instead of being emitted twice.
//!
//! ## Why the constant part is *not* carried by the cone
//!
//! A cone from one point only tiles a body that is star-shaped from that point.
//! Ryugu is a top with a concave equatorial waist, and it is not: measured on the
//! real mesh, `|Σ signed tet volume| / Σ|tet volume| = 0.99898`, i.e. ≈0.05 % of
//! the volume sits in inverted tets. Left in the constant term that defect is a
//! ≈1e-3 relative error — worse than the mascon discretisation this solver is
//! supposed to adjudicate.
//!
//! So split the field at a reference density `ρ_ref = ρ(centroid)` and let each
//! half go where it is exact:
//!
//! ```text
//!   H = ρ_ref · W_mesh (exact, mesh faces, weight ρ_ref)
//!     + G Σ_cone (ρ_f − ρ_ref) · n_j I_F[i]   (star-cone faces, tiny weights)
//! ```
//!
//! The mesh faces give `W` exactly for *any* shape — that is the whole point of
//! the surface-integral form. The cone then only has to carry the ≈1e-5 (relative)
//! deviation from uniform density, so the same 0.05 % star-shape defect lands on
//! a 1e-5 signal and contributes ≈1e-8. The 4 cone faces per mesh face plus the
//! 196 608 mesh faces give 983 040 triangles in one analytic pass.
//!
//! ## What it is good for
//!
//! * **Constant density is an exact identity.** With one density everywhere every
//!   cone weight `ρ_f − ρ_ref` is zero, so the face list collapses to `ρ_ref·W`,
//!   the quantity Werner and the RT-FP analytic term compute from completely
//!   different code. `--selftest` asserts this on the real mesh.
//! * **Cauchy density is accurate to O(∇ρ·h/ρ).** With `σ ≈ 5.5e-5 m⁻¹` the TOML
//!   density varies by only ≈0.05 % across the whole body, and a face is a few
//!   metres, so the piecewise-constant error is ≈1e-5 — two orders of magnitude
//!   tighter than the mascon voxel field's ≈3e-3 at 192³. Whatever gap is left
//!   between this and mascon is therefore mascon's discretisation, not the Cauchy
//!   field.
//!
//! The one thing this representation does *not* do is the Carlson `R_F/R_D/R_J`
//! elliptic reduction of `∫ds/[(s+b)(s²+d²)^α]` that the derivation PDF carries
//! for general `α`: that reduction is needed when the radial integral is
//! genuinely elliptic (`α = m/4`), and for the `α = 1` this TOML uses — and which
//! `density.rs` already refuses to go beyond — the radial integral is elementary
//! and no `R_F/R_D/R_J` is involved at any point.

use crate::analytic::Face;
use crate::density::{Density, DensityMode};
use crate::mass;
use crate::mesh::Mesh;

#[derive(Clone, Copy, Debug)]
pub struct Refine {
    /// Target within-slab density variation, relative to `|ρ_ref|`.
    pub tol: f64,
    /// Hard cap on slabs per cone, which also caps the per-cone triangle cost.
    pub max_slabs: usize,
    /// Axis samples per cone: where the variation is measured and how finely the
    /// slab boundaries can be placed.
    pub samples: usize,
    /// Hard cap on emitted cone triangles. The tolerance is relaxed until the
    /// estimate fits, because triangle count is exactly the GPU cost of the bake.
    pub face_budget: usize,
}

impl Default for Refine {
    fn default() -> Self {
        Self {
            tol: 1e-3,
            max_slabs: 8,
            samples: 24,
            face_budget: 8_000_000,
        }
    }
}

/// Diagnostics from building the jump-surface list.
#[derive(Clone, Copy, Debug, Default)]
pub struct CarlsonStats {
    /// Star-cone centre actually used.
    pub origin: [f64; 3],
    /// Reference density split out into the exact polyhedral term, kg/m³.
    pub rho_ref: f64,
    /// `Σ |tetrahedron volume|` — the volume the decomposition covers.
    pub covered_volume: f64,
    /// Signed sum, i.e. `Σ ±tetrahedron volume`. Equal to [`Self::covered_volume`]
    /// exactly when the body is star-shaped from `origin`.
    pub signed_volume: f64,
    /// `∮ (1/3)(y − o)·n dS`, the mesh's own enclosed volume.
    pub mesh_volume: f64,
    /// Surface triangles emitted by the star cone (4 per mesh face).
    pub n_faces: usize,
    /// Mesh faces carried at `ρ_ref` on top of the cone.
    pub n_mesh_faces: usize,
    /// Largest and smallest `|Δρ|` over those triangles, in kg/m³.
    pub max_jump: f64,
    pub min_jump: f64,
    /// Total slabs emitted over all cones (1 per cone = no radial refinement).
    pub slabs: usize,
    /// Largest per-cone slab count.
    pub max_slabs_used: usize,
    /// Relative tolerance actually achieved after the budget was applied.
    pub tol: f64,
    /// Worst within-slab density variation reached, relative to `|ρ_ref|`. This
    /// is the quantity that bounds the representation error of this face list.
    pub worst_slab_variation: f64,
    /// Cone triangles the budget refused (0 when the requested tolerance fitted).
    pub over_budget: bool,
}

impl CarlsonStats {
    /// Fraction of the mesh volume the star-cone actually covers. Below 1 the
    /// body is not star-shaped from the centroid and the decomposition is only a
    /// signed approximation of it.
    pub fn coverage(&self) -> f64 {
        if self.mesh_volume.abs() > 1e-30 {
            self.covered_volume / self.mesh_volume
        } else {
            1.0
        }
    }

    /// `|signed|/covered` — 1 for a clean star-shaped decomposition.
    pub fn signed_ratio(&self) -> f64 {
        if self.covered_volume.abs() > 1e-30 {
            self.signed_volume.abs() / self.covered_volume
        } else {
            1.0
        }
    }
}

/// The complete Carlson face list for `mesh` under `density`:
/// the mesh faces at `weight = ρ_ref` followed by the star-cone faces at
/// `weight = ρ_f − ρ_ref`. Feeding it to the analytic pipeline produces
/// `H_ij = G Σ_F w_F n_j I_F[i]` directly — no separate `ρ(x)` factor and no
/// quadrature, unlike the RT-FP ray solver.
pub fn face_list(mesh: &Mesh, density: &Density, mode: DensityMode) -> (Vec<Face>, CarlsonStats) {
    let origin = mass::volume_centroid(mesh);
    let rho_ref = density.rho(&origin, mode);
    let (cone, mut stats) = star_faces(mesh, density, mode, origin, rho_ref, &Refine::default());
    let mut faces = crate::analytic::precompute(mesh);
    for f in faces.iter_mut() {
        f.weight = rho_ref;
    }
    stats.n_mesh_faces = faces.len();
    faces.extend(cone);
    (faces, stats)
}

/// Star-cone jump surfaces only, each carrying `weight = ρ(tet) − rho_ref`.
///
/// Exposed separately because `--selftest` uses `rho_ref = 0` on the unit cube,
/// where the cone tiles the body exactly, to check the orientation logic against
/// the polyhedral tensor.
pub fn star_faces(
    mesh: &Mesh,
    density: &Density,
    mode: DensityMode,
    origin: [f64; 3],
    rho_ref: f64,
    refine: &Refine,
) -> (Vec<Face>, CarlsonStats) {
    let n_mesh_faces = mesh.face_count();
    let samples = refine.samples.max(4);

    // Pass 1 — the only per-cone quantity that matters is how much ρ varies along
    // its axis, so measure that first. Everything else (slab counts, boundary
    // placement, the tolerance actually achieved) follows from it.
    let mut corners: Vec<[[f64; 3]; 3]> = Vec::with_capacity(n_mesh_faces);
    let mut variation: Vec<f64> = Vec::with_capacity(n_mesh_faces);
    let mut covered = 0.0f64;
    let mut signed = 0.0f64;
    for f in 0..n_mesh_faces {
        let [i0, i1, i2] = mesh.face(f);
        let v0 = mesh.vertex(i0 as usize);
        let v1 = mesh.vertex(i1 as usize);
        let v2 = mesh.vertex(i2 as usize);
        corners.push([v0, v1, v2]);
        let vol = tet_signed_volume(origin, v0, v1, v2);
        covered += vol.abs();
        signed += vol;
        let profile = axis_profile(origin, face_centre([v0, v1, v2]), density, mode, samples);
        variation.push(total_variation(&profile));
    }

    // Triangle count *is* the GPU cost of the bake, so relax the tolerance until
    // the estimate fits the budget instead of discovering the cost afterwards.
    let mut tol = refine.tol.max(f64::MIN_POSITIVE);
    let estimate = |tol: f64| -> f64 {
        variation
            .iter()
            .map(|tv| (7.0 * slabs_for(*tv, tol, refine.max_slabs, rho_ref) as f64 - 3.0).max(0.0))
            .sum()
    };
    let mut over_budget = false;
    while estimate(tol) > refine.face_budget as f64 {
        if tol >= 1.0 {
            over_budget = true;
            break;
        }
        tol *= 1.5;
    }

    // Pass 2 — emit. Each slab contributes its own closed frustum boundary, which
    // keeps the orientation logic trivial: every triangle is oriented away from
    // the slab centre and the shared faces cancel by superposition.
    let mut faces: Vec<Face> = Vec::new();
    let mut max_jump: f64 = 0.0;
    let mut min_jump = f64::INFINITY;
    let mut slabs_total = 0usize;
    let mut max_slabs_used = 0usize;
    let mut worst_variation = 0.0f64;
    let scale = rho_ref.abs().max(f64::MIN_POSITIVE);

    for (f, tri) in corners.iter().enumerate() {
        let [v0, v1, v2] = *tri;
        let profile = axis_profile(origin, face_centre(*tri), density, mode, samples);
        let k = slabs_for(variation[f], tol, refine.max_slabs, rho_ref);
        let bounds = slab_boundaries(&profile, k);
        slabs_total += k;
        max_slabs_used = max_slabs_used.max(k);

        let mut lo = 0.0f64;
        for (i, hi) in bounds.iter().enumerate() {
            let hi = *hi;
            let rho_slab = slab_mean(&profile, lo, hi);
            let (mn, mx) = slab_range(&profile, lo, hi);
            worst_variation = worst_variation.max((mx - mn) / scale);
            let g = rho_slab - rho_ref;
            max_jump = max_jump.max(g.abs());
            min_jump = min_jump.min(g.abs());

            let centre = frustum_centre(origin, *tri, lo, hi);
            for (a, b) in [(v0, v1), (v1, v2), (v2, v0)] {
                let p = lerp(origin, a, lo);
                let q = lerp(origin, b, lo);
                let r = lerp(origin, b, hi);
                let s = lerp(origin, a, hi);
                if lo > 0.0 {
                    push_face(&mut faces, p, q, r, centre, g);
                    push_face(&mut faces, p, r, s, centre, g);
                } else {
                    // The near ring collapses onto the apex: one triangle, not two.
                    push_face(&mut faces, origin, r, s, centre, g);
                }
            }
            // Cross-section at `hi`. Its twin is the previous slab's cap with the
            // opposite orientation, so the two are summed here as a single jump.
            let next = if i + 1 < k {
                let rho_next = slab_mean(&profile, hi, bounds[i + 1]);
                rho_next - rho_ref
            } else {
                0.0
            };
            let cap = g - next;
            push_face(
                &mut faces,
                lerp(origin, v0, hi),
                lerp(origin, v1, hi),
                lerp(origin, v2, hi),
                centre,
                cap,
            );
            max_jump = max_jump.max(cap.abs());
            min_jump = min_jump.min(cap.abs());
            lo = hi;
        }
    }

    let stats = CarlsonStats {
        origin,
        rho_ref,
        covered_volume: covered,
        signed_volume: signed,
        mesh_volume: mass::body_volume(mesh),
        n_faces: faces.len(),
        n_mesh_faces: 0,
        max_jump,
        min_jump: if min_jump.is_finite() { min_jump } else { 0.0 },
        slabs: slabs_total,
        max_slabs_used,
        tol,
        worst_slab_variation: worst_variation,
        over_budget,
    };
    (faces, stats)
}

/// `ρ` sampled along the cone axis at `s_j = (j + ½)/m`.
fn axis_profile(
    origin: [f64; 3],
    apex: [f64; 3],
    density: &Density,
    mode: DensityMode,
    m: usize,
) -> Vec<f64> {
    (0..m)
        .map(|j| density.rho(&lerp(origin, apex, (j as f64 + 0.5) / m as f64), mode))
        .collect()
}

fn total_variation(v: &[f64]) -> f64 {
    v.windows(2).map(|w| (w[1] - w[0]).abs()).sum()
}

/// Slabs for one cone: enough that the variation inside a slab falls below
/// `tol·|ρ_ref|`, never more than `max_slabs`.
fn slabs_for(tv: f64, tol: f64, max_slabs: usize, rho_ref: f64) -> usize {
    let scale = rho_ref.abs();
    if tv <= 0.0 || tv.is_nan() || scale <= f64::MIN_POSITIVE || max_slabs <= 1 {
        return 1;
    }
    let want = (tv / (tol * scale)).ceil();
    if !want.is_finite() {
        return max_slabs;
    }
    (want.max(1.0) as usize).min(max_slabs)
}

/// Slab boundaries in the cone scale `s ∈ (0, 1]`, placed so every slab carries
/// the same amount of variation of `ρ` (equidistribution). A steep transition
/// therefore collects boundaries; a flat stretch gets almost none.
fn slab_boundaries(profile: &[f64], k: usize) -> Vec<f64> {
    let m = profile.len();
    let mut cumulative = vec![0.0f64; m];
    for j in 1..m {
        cumulative[j] = cumulative[j - 1] + (profile[j] - profile[j - 1]).abs();
    }
    let tv = cumulative[m - 1];
    if k <= 1 || tv <= 0.0 || tv.is_nan() {
        return vec![1.0];
    }
    let sample_s = |j: usize| (j as f64 + 0.5) / m as f64;
    let gap = 0.5 / m as f64;
    let mut out = Vec::with_capacity(k);
    for i in 1..=k {
        let target = tv * i as f64 / k as f64;
        let mut s = 1.0;
        for j in 1..m {
            if cumulative[j] >= target {
                let span = cumulative[j] - cumulative[j - 1];
                let frac = if span > 0.0 {
                    (target - cumulative[j - 1]) / span
                } else {
                    0.0
                };
                s = sample_s(j - 1) + frac * (sample_s(j) - sample_s(j - 1));
                break;
            }
        }
        out.push(s);
    }
    // Keep the sequence strictly increasing and inside the cone.
    let last = out.len() - 1;
    out[last] = 1.0;
    for i in 0..last {
        let lower = if i == 0 { gap } else { out[i - 1] + gap };
        let upper = (1.0 - (last - i) as f64 * gap).max(lower);
        out[i] = out[i].clamp(lower, upper);
    }
    out
}

/// Volume mean of `ρ` over the frustum between scales `lo` and `hi`.
///
/// The volume element of a cone scales as `s² ds`, so each axis sample stands for
/// the third-power mass of the slice around it.
fn slab_mean(profile: &[f64], lo: f64, hi: f64) -> f64 {
    let m = profile.len();
    let gap = 0.5 / m as f64;
    let (mut num, mut den) = (0.0, 0.0);
    for (j, v) in profile.iter().enumerate() {
        let s = (j as f64 + 0.5) / m as f64;
        let a = (s - gap).max(lo);
        let b = (s + gap).min(hi);
        if b <= a {
            continue;
        }
        let w = (b * b * b - a * a * a) / 3.0;
        num += w * v;
        den += w;
    }
    if den > 0.0 {
        num / den
    } else {
        let mid = 0.5 * (lo + hi);
        profile[(((mid * m as f64 - 0.5) as isize).clamp(0, m as isize - 1)) as usize]
    }
}

/// `(min, max)` of `ρ` over the axis samples inside one slab.
fn slab_range(profile: &[f64], lo: f64, hi: f64) -> (f64, f64) {
    let m = profile.len();
    let (mut mn, mut mx) = (f64::INFINITY, f64::NEG_INFINITY);
    for (j, v) in profile.iter().enumerate() {
        let s = (j as f64 + 0.5) / m as f64;
        if s >= lo && s <= hi {
            mn = mn.min(*v);
            mx = mx.max(*v);
        }
    }
    if mn.is_finite() {
        (mn, mx)
    } else {
        (0.0, 0.0)
    }
}

fn face_centre(tri: [[f64; 3]; 3]) -> [f64; 3] {
    [
        (tri[0][0] + tri[1][0] + tri[2][0]) / 3.0,
        (tri[0][1] + tri[1][1] + tri[2][1]) / 3.0,
        (tri[0][2] + tri[1][2] + tri[2][2]) / 3.0,
    ]
}

fn lerp(origin: [f64; 3], target: [f64; 3], s: f64) -> [f64; 3] {
    [
        origin[0] + s * (target[0] - origin[0]),
        origin[1] + s * (target[1] - origin[1]),
        origin[2] + s * (target[2] - origin[2]),
    ]
}

/// Mean of the ring vertices of a frustum — always interior, so it orients every
/// emitted triangle outward without a special case for the apex slab.
fn frustum_centre(origin: [f64; 3], tri: [[f64; 3]; 3], lo: f64, hi: f64) -> [f64; 3] {
    let mut c = [0.0f64; 3];
    let mut n = 0.0f64;
    for v in tri {
        for s in [lo.max(1e-6), hi] {
            let p = lerp(origin, v, s);
            for i in 0..3 {
                c[i] += p[i];
            }
            n += 1.0;
        }
    }
    [c[0] / n, c[1] / n, c[2] / n]
}

/// RT-FP's tensor at `x`, evaluated in f64 — the reference the jump-surface
/// representation is measured against in the tuning tests.
///
/// It is the same two halves the bake uses: `ρ(x)·W(x)` from the closed-form
/// polyhedral tensor, plus the directional quadrature of the radial remainder.
/// Nothing here is shared with the face list, which is the point.
// Mirrors the argument list of the bake's own split so the two can be read side
// by side; every entry is an immutable input buffer of one evaluation.
#[allow(clippy::too_many_arguments)]
pub fn rtfp_reference(
    mesh: &Mesh,
    tree: &crate::bvh::Bvh,
    uniform: &[Face],
    density: &Density,
    kernels: &[crate::density::KernelSi],
    x: &[f64; 3],
    t_max: f64,
    dirs: &[([f64; 3], f64)],
) -> [f64; 6] {
    use crate::{geom, split, tensor};
    let mut h = crate::analytic::hessian6(uniform, *x);
    let rho = density.rho(x, DensityMode::Cauchy);
    for t in h.iter_mut() {
        *t *= rho;
    }
    let mut hits = Vec::new();
    geom::bvh_crossings(
        tree,
        mesh,
        *x,
        [-1.0, 0.0, 0.0],
        crate::GEOM_EPS_M,
        t_max,
        &mut hits,
    );
    let inside = hits.len() % 2 == 1;
    for (u, omega) in dirs {
        geom::bvh_crossings(
            tree,
            mesh,
            *x,
            [-u[0], -u[1], -u[2]],
            crate::GEOM_EPS_M,
            t_max,
            &mut hits,
        );
        let slots = split::intervals(&hits, inside);
        let s = split::remainder_scalar(kernels, *x, *u, &slots);
        tensor::add_tensor_term(&mut h, u, crate::G * *omega * s);
    }
    h
}

/// Append one triangle, flipping two corners when needed so the unit normal
/// points away from `inside` (the tetrahedron centre).
fn push_face(
    out: &mut Vec<Face>,
    p: [f64; 3],
    q: [f64; 3],
    r: [f64; 3],
    inside: [f64; 3],
    weight: f64,
) {
    let (mut q, mut r) = (q, r);
    let mut n = cross(sub(q, p), sub(r, p));
    let face_centre = [
        (p[0] + q[0] + r[0]) / 3.0,
        (p[1] + q[1] + r[1]) / 3.0,
        (p[2] + q[2] + r[2]) / 3.0,
    ];
    if dot(n, sub(face_centre, inside)) < 0.0 {
        std::mem::swap(&mut q, &mut r);
        n = cross(sub(q, p), sub(r, p));
    }
    let len = norm(n);
    let unit = if len > 1e-30 {
        [n[0] / len, n[1] / len, n[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    };
    out.push(Face {
        corners: [p, q, r],
        n: unit,
        weight,
    });
}

/// `(b−a) · ((c−a) × (d−a)) / 6`.
fn tet_signed_volume(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    dot(sub(b, a), cross(sub(c, a), sub(d, a))) / 6.0
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytic;
    use crate::density::Normalization;
    use crate::tensor::{frobenius, Sym6};

    fn cube() -> Mesh {
        let mut mesh = Mesh::default();
        for &(x, y, z) in &[
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
        ] {
            mesh.xyz.extend_from_slice(&[x, y, z]);
        }
        for f in [
            [0u32, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [1, 2, 6],
            [1, 6, 5],
            [3, 0, 4],
            [3, 4, 7],
        ] {
            mesh.faces.extend_from_slice(&f);
        }
        mesh
    }

    #[test]
    fn centroid_and_coverage_of_cube() {
        let mesh = cube();
        let c = mass::volume_centroid(&mesh);
        assert!(c.iter().all(|v| v.abs() < 1e-12), "centroid {c:?}");
        let density = Density::homogeneous(1190.0, &mesh);
        // `rho_ref = 0` keeps the full density in the cone weights, which is the
        // configuration that exercises the orientation logic.
        let (faces, stats) = star_faces(
            &mesh,
            &density,
            DensityMode::Constant,
            c,
            0.0,
            &Refine::default(),
        );
        assert_eq!(faces.len(), 4 * mesh.face_count());
        assert!(
            (stats.coverage() - 1.0).abs() < 1e-12,
            "coverage {}",
            stats.coverage()
        );
        assert!((stats.signed_ratio() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn constant_density_equals_polyhedral_tensor() {
        // The identity the whole solver is justified by: with a single density
        // everywhere the jump faces reproduce `ρ · W(x)` exactly.
        let mesh = cube();
        let rho = 1190.0;
        let density = Density::homogeneous(rho, &mesh);
        let origin = mass::volume_centroid(&mesh);
        let (faces, _) = star_faces(
            &mesh,
            &density,
            DensityMode::Constant,
            origin,
            0.0,
            &Refine::default(),
        );
        let reference = analytic::precompute(&mesh);
        for x in [[0.0, 0.0, 2.5], [3.0, -1.0, 0.5], [0.4, 0.4, 1.9]] {
            let got = analytic::hessian6(&faces, x);
            let mut want = analytic::hessian6(&reference, x);
            for t in want.iter_mut() {
                *t *= rho;
            }
            let d = sub6(&got, &want);
            let rel = frobenius(&d) / frobenius(&want).max(1e-300);
            assert!(rel < 1e-12, "x={x:?} rel={rel} got={got:?} want={want:?}");
        }
    }

    #[test]
    fn cauchy_field_is_refined_within_tolerance() {
        // The claim the refinement has to back up: cutting each cone until the
        // density variation inside a slab is small bounds the representation
        // error of the whole face list, whatever the TOML looks like.
        let root = crate::root();
        let Some(mesh) = ryugu_mesh_or_skip("cauchy_field_is_refined_within_tolerance") else {
            return;
        };
        let density = Density::from_toml(
            &root.join("assets/density/cauchy.toml"),
            &mesh,
            Normalization::TotalMass,
        )
        .expect("density");
        let (faces, stats) = face_list(&mesh, &density, DensityMode::Cauchy);
        eprintln!(
            "cone triangles {} ({} slabs, max {}), tol {:.3e}, worst slab variation {:.4}, \
             budget hit {}",
            stats.n_faces,
            stats.slabs,
            stats.max_slabs_used,
            stats.tol,
            stats.worst_slab_variation,
            stats.over_budget
        );
        assert_eq!(faces.len(), stats.n_faces + stats.n_mesh_faces);
        assert!(stats.coverage() > 0.99, "coverage {}", stats.coverage());
        assert!(
            stats.max_slabs_used <= Refine::default().max_slabs,
            "slabs per cone {}",
            stats.max_slabs_used
        );
        assert!(
            stats.worst_slab_variation < 0.05,
            "worst within-slab variation {:.4} of ρ_ref",
            stats.worst_slab_variation
        );
    }

    /// The 200k-face Ryugu OBJ lives outside this repository, so it is absent on
    /// a bare CI checkout. Tests that need it report the skip and pass.
    fn ryugu_mesh_or_skip(test: &str) -> Option<Mesh> {
        let path = std::env::var_os("RYUGU_OBJ")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                crate::root().join("../Ryugu_wasm/assets/models/SHAPE_SFM_200k_v20180804.obj")
            });
        if !path.is_file() {
            eprintln!(
                "skipping {test}: observation mesh not found at {}",
                path.display()
            );
            return None;
        }
        Some(Mesh::load_obj(&path, crate::KM_TO_M).expect("mesh"))
    }

    fn sub6(a: &Sym6, b: &Sym6) -> Sym6 {
        [
            a[0] - b[0],
            a[1] - b[1],
            a[2] - b[2],
            a[3] - b[3],
            a[4] - b[4],
            a[5] - b[5],
        ]
    }
}
