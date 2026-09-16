//! General-alpha radial finite part and the Carlson special-function backend.
//!
//! The legacy Carlson jump-surface solver is intentionally an `alpha = 1`
//! representation. `CarlsonAlpha` supports the general rational Cauchy kernel
//!
//! ```text
//!   phi_k(x - R u) = (A R^2 - 2 B R + C)^(-alpha)
//! ```
//!
//! through the regularized radial remainder
//!
//! ```text
//!   R_k = integral_a^b [phi_k(x - R u) - phi_k(x)] dR / R.
//! ```
//!
//! The subtraction removes the finite-part logarithmic singularity, including
//! when `a = 0`. The production bake evaluates the same formula with a
//! fixed-order GPU rule; this module contains the f64 reference, the Carlson
//! `R_F/R_D/R_J/R_C` library bridge used by the verification checks, and
//! reduction identities for quartic boundary arcs.

use crate::density::KernelSi;

/// High-order reference rule for one radial remainder interval.
pub fn radial_remainder_reference(
    kernel: &KernelSi,
    x: [f64; 3],
    u: [f64; 3],
    a: f64,
    b: f64,
) -> f64 {
    if !a.is_finite() || !b.is_finite() || b <= a {
        return 0.0;
    }
    let s2 = kernel.sigma * kernel.sigma;
    let p = [x[0] - kernel.c[0], x[1] - kernel.c[1], x[2] - kernel.c[2]];
    let p2 = dot(p, p);
    let aa = 1.0 + s2 * p2;
    let bb = dot(p, u);
    let phi0 = aa.powf(-kernel.alpha);
    let mid = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    let (nodes, weights) = gauss_legendre(64);
    let mut sum = 0.0;
    for (node, weight) in nodes.iter().zip(&weights) {
        let r = mid + half * node;
        let d = (aa - 2.0 * s2 * bb * r + s2 * r * r).max(f64::MIN_POSITIVE);
        sum += weight * (d.powf(-kernel.alpha) - phi0) / r;
    }
    kernel.w * half * sum
}

/// One `G Σω T(u) Σk wk Rk` contribution with the f64 reference rule.
pub fn remainder_scalar_reference(
    kernels: &[KernelSi],
    x: [f64; 3],
    u: [f64; 3],
    intervals: &[(f64, f64)],
) -> f64 {
    let mut sum = 0.0;
    for kernel in kernels {
        for &(a, b) in intervals {
            sum += radial_remainder_reference(kernel, x, u, a, b);
        }
    }
    sum
}

/// Carlson symmetric integrals used by the genus-one boundary reduction.
///
/// These are thin wrappers so the numerical backend stays in one place and so
/// callers cannot accidentally pull in a different special-function convention.
pub mod special {
    pub fn rf(x: f64, y: f64, z: f64) -> Result<f64, String> {
        ellip::elliprf(x, y, z).map_err(|e| format!("RF({x}, {y}, {z}): {e}"))
    }

    pub fn rd(x: f64, y: f64, z: f64) -> Result<f64, String> {
        ellip::elliprd(x, y, z).map_err(|e| format!("RD({x}, {y}, {z}): {e}"))
    }

    pub fn rj(x: f64, y: f64, z: f64, p: f64) -> Result<f64, String> {
        ellip::elliprj(x, y, z, p).map_err(|e| format!("RJ({x}, {y}, {z}, {p}): {e}"))
    }

    pub fn rc(x: f64, y: f64) -> Result<f64, String> {
        ellip::elliprc(x, y).map_err(|e| format!("RC({x}, {y}): {e}"))
    }
}

/// Canonical infinite third-kind integral
///
/// ```text
///   integral_0^inf dt / ((t + p) sqrt((t + x)(t + y)(t + z)))
///     = (2 / 3) RJ(x, y, z, p).
/// ```
///
/// The finite boundary arcs are first split at roots and mapped to this tail
/// form before they reach the special-function backend.
pub fn third_kind_tail(x: f64, y: f64, z: f64, p: f64) -> Result<f64, String> {
    if x < 0.0 || y < 0.0 || z < 0.0 || p <= 0.0 {
        return Err("third_kind_tail requires x,y,z >= 0 and p > 0".into());
    }
    Ok(2.0 * special::rj(x, y, z, p)? / 3.0)
}

fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    for i in 0..n {
        let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut derivative = 0.0;
        for _ in 0..100 {
            let (mut p1, mut p2) = (1.0, 0.0);
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                p1 = ((2 * j + 1) as f64 * x * p2 - j as f64 * p3) / (j + 1) as f64;
            }
            derivative = n as f64 * (x * p1 - p2) / (x * x - 1.0);
            let step = p1 / derivative;
            x -= step;
            if step.abs() < 1e-15 {
                break;
            }
        }
        nodes[i] = x;
        weights[i] = 2.0 / ((1.0 - x * x) * derivative * derivative);
    }
    (nodes, weights)
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kernel(alpha: f64) -> KernelSi {
        KernelSi {
            c: [0.1, -0.2, 0.3],
            sigma: 0.8,
            w: 1.0,
            alpha,
        }
    }

    #[test]
    fn carlson_library_identities() {
        let pi = std::f64::consts::PI;
        assert!((special::rf(0.0, 1.0, 1.0).unwrap() - pi / 2.0).abs() < 1e-14);
        assert!((special::rd(0.0, 1.0, 1.0).unwrap() - 3.0 * pi / 4.0).abs() < 1e-14);
        assert!((special::rc(0.0, 1.0).unwrap() - pi / 2.0).abs() < 1e-14);
        assert!((special::rj(1.0, 1.0, 1.0, 1.0).unwrap() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn alpha_one_matches_closed_form() {
        let k = kernel(1.0);
        let x = [0.35, 0.15, 0.55];
        let u = [0.2, -0.4, 0.8];
        let s2 = k.sigma * k.sigma;
        let p = [x[0] - k.c[0], x[1] - k.c[1], x[2] - k.c[2]];
        let p2 = dot(p, p);
        let a = 1.0 + s2 * p2;
        let b = dot(p, u);
        let d = (1.0 / s2 + p2 - b * b).sqrt();
        let (r0, r1) = (0.7, 2.1);
        let d1 = (a - 2.0 * s2 * b * r1 + s2 * r1 * r1).max(f64::MIN_POSITIVE);
        let d0 = (a - 2.0 * s2 * b * r0 + s2 * r0 * r0).max(f64::MIN_POSITIVE);
        let phi1 = (b / d) * (r1 - b).atan2(d);
        let phi0 = (b / d) * (r0 - b).atan2(d);
        let want = (-0.5 * (d1 / d0).ln() + phi1 - phi0) / a;
        let got = radial_remainder_reference(&k, x, u, r0, r1);
        assert!((got - want).abs() < 2e-10, "got {got}, want {want}");
    }

    #[test]
    fn fractional_alpha_is_finite_at_zero() {
        for alpha in [0.75, 1.25, 1.5, 2.25] {
            let value = radial_remainder_reference(
                &kernel(alpha),
                [0.4, 0.2, 0.6],
                [0.1, -0.7, 0.7],
                0.0,
                1.3,
            );
            assert!(value.is_finite(), "alpha={alpha}: {value}");
        }
    }
}
