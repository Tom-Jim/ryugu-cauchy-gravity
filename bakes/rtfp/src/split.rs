//! The analytic near-field split.
//!
//! The ray form of the Hessian is
//! `H = G Σ_q ω_q T(u_q) Σ_k w_k J_k(u_q)` minus the contact term
//! `4πG/3 · χ ρ(x) I`, with `T(u) = 3uuᵀ − I` and the finite-part radial
//! integral `J_k = ∫_visible ρ_k(x − R u) dR/R` (`ρ_k = w_k / A_k`).
//!
//! Writing `L(u) = Σ_intervals ln(r₁/r₀)` for the finite part of the log and
//! splitting `Σ_k w_k J_k = ρ(x)·L(u) + Σ_k w_k R_k(u)` turns that into
//!
//! ```text
//!   H = ρ(x) · [ G Σ_q ω_q T(u_q) L(u_q) − (4πG/3) χ I ]   exact polyhedral tensor
//!     + G Σ_q ω_q T(u_q) Σ_k w_k R_k(u_q)                   smooth remainder
//! ```
//!
//! The bracket is exactly the uniform-density gravity-gradient tensor, which the
//! closed form in `analytic.rs` (WGSL: `shaders/analytic.wgsl`) evaluates. So the
//! logarithmic near-field divergence
//! and the terminator kink of `ln r₁(u)` — the two features a fixed sphere rule
//! cannot resolve, and the reason 288 directions alone still left ~2 % — never
//! enter the quadrature. `R_k` keeps no `ln r` term at all, which also removes
//! the finite-part reference scale `ℓ₀` from the pipeline.
//!
//! Everything in this module is the **CPU reference** behind `--selftest`; the
//! bake evaluates the same formulas in `shaders/rays.wgsl` and
//! `shaders/remainder.wgsl`.

use crate::density::KernelSi;

/// Interval slots per (point, direction); mirrors `MAX_INTERVALS` in `rays.wgsl`.
pub const MAX_INTERVALS: usize = 16;

/// `D(R) = A − 2σ²bR + σ²R²`.
#[inline]
fn d_of(r: f64, a: f64, b: f64, s2: f64) -> f64 {
    (a - 2.0 * s2 * b * r + s2 * r * r).max(1e-300)
}

/// Visible intervals of the ray `x − R u`, as `(r₀, r₁)` pairs.
///
/// The returned flag is true when the ray had more intervals than the fixed GPU
/// capacity. The bake reports that condition from `rays.wgsl`; this CPU mirror
/// exposes it so reference checks cannot silently truncate either.
#[inline]
pub fn intervals(hits: &[f64], inside: bool) -> ([Option<(f64, f64)>; MAX_INTERVALS], bool) {
    let mut out = [None; MAX_INTERVALS];
    let mut n = 0usize;
    if inside {
        if hits.is_empty() {
            return (out, false);
        }
        out[n] = Some((0.0, hits[0]));
        n += 1;
        let mut i = 1;
        while i + 1 < hits.len() && n < MAX_INTERVALS {
            out[n] = Some((hits[i], hits[i + 1]));
            n += 1;
            i += 2;
        }
    } else {
        let mut i = 0;
        while i + 1 < hits.len() && n < MAX_INTERVALS {
            out[n] = Some((hits[i], hits[i + 1]));
            n += 1;
            i += 2;
        }
    }
    let needed = if inside && !hits.is_empty() {
        1 + (hits.len() - 1) / 2
    } else {
        hits.len() / 2
    };
    (out, needed > MAX_INTERVALS)
}

/// `Σ_k w_k R_k(u)` for one direction: the radial integrals with their log part
/// removed, summed over the visible intervals of `x − R u`.
pub fn remainder_scalar(
    kernels: &[KernelSi],
    x: [f64; 3],
    u: [f64; 3],
    ivals: &[Option<(f64, f64)>; MAX_INTERVALS],
) -> f64 {
    if kernels.is_empty() || ivals.iter().all(|s| s.is_none()) {
        return 0.0;
    }
    let mut total = 0.0;
    for k in kernels {
        let p = [x[0] - k.c[0], x[1] - k.c[1], x[2] - k.c[2]];
        let p2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
        let s2 = k.sigma * k.sigma;
        let a = 1.0 + s2 * p2;
        let b = p[0] * u[0] + p[1] * u[1] + p[2] * u[2];
        let d = (1.0 / s2 + p2 - b * b).max(1e-30).sqrt();
        let mut jk = 0.0;
        for slot in ivals.iter().flatten() {
            let (r0, r1) = *slot;
            if r1 <= r0 {
                continue;
            }
            let dr1 = d_of(r1, a, b, s2);
            let phi1 = (b / d) * (r1 - b).atan2(d);
            jk += if r0 > 0.0 {
                // J_k − ln(r₁/r₀)/A with J_k = P(r₁) − P(r₀)
                let dr0 = d_of(r0, a, b, s2);
                (-0.5 * (dr1 / dr0).ln() + phi1 - (b / d) * (r0 - b).atan2(d)) / a
            } else {
                // Finite part on [0, r₁], with its ln(r₁/ℓ₀)/A term removed:
                // the ℓ₀ dependence lives in the analytic tensor and cancels
                // against Σω_q T(u_q) = 0.
                (-0.5 * (dr1 / a).ln() + phi1 + (b / d) * b.atan2(d)) / a
            };
        }
        total += k.w * jk;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::{intervals, MAX_INTERVALS};

    #[test]
    fn interval_capacity_is_explicit() {
        let hits: Vec<f64> = (1..=2 * MAX_INTERVALS as i32 + 2).map(f64::from).collect();
        let (slots, overflow) = intervals(&hits, false);
        assert_eq!(slots.iter().flatten().count(), MAX_INTERVALS);
        assert!(overflow);

        let (_, inside_overflow) = intervals(&hits, true);
        assert!(inside_overflow);
    }
}
