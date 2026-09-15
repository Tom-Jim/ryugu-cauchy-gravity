//! Spherical quadrature for the direction integral `∫ T(u) S(u) dΩ`.
//!
//! The derivation (工程实现 3. 球面求积方向) only requires `Σω = 4π` and a
//! centre-symmetric node set (`(u, ω), (−u, ω)`), because `Σω T(u) = 0` is what
//! cancels the `log(standoff)` divergence. A product rule — Gauss–Legendre in
//! `cos θ` times a uniform azimuth — satisfies both exactly and, unlike the
//! icosahedral 12-point rule, resolves the logarithmic near-surface structure of
//! the integrand.
//!
//! Measured against the ESA polyhedral record (24 faces, 1 mm standoff):
//! 12 dirs → 18 % median, 72 → 7.6 %, 288 → 1.7 %, 800 → 1.4 %.

/// A node set: `(direction, solid-angle weight)` pairs. The buffer layout the
/// compute shader consumes is exactly this, so the same slice feeds the CPU
/// reference and the GPU dispatch.
pub type Dirs = [([f64; 3], f64)];

/// Legendre nodes/weights on [−1, 1] (Newton iteration on the recurrence).
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    for i in 0..n {
        let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut dp = 0.0;
        for _ in 0..100 {
            let (mut p1, mut p2) = (1.0, 0.0);
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                p1 = ((2 * j + 1) as f64 * x * p2 - j as f64 * p3) / (j + 1) as f64;
            }
            dp = n as f64 * (x * p1 - p2) / (x * x - 1.0);
            let dx = p1 / dp;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }
        nodes[i] = x;
        weights[i] = 2.0 / ((1.0 - x * x) * dp * dp);
    }
    (nodes, weights)
}

/// Product rule with `n_theta` polar nodes and `2·n_theta` azimuthal nodes.
pub fn product_rule(n_theta: usize) -> Vec<([f64; 3], f64)> {
    let n_phi = 2 * n_theta;
    let (nodes, weights) = gauss_legendre(n_theta);
    let mut dirs = Vec::with_capacity(n_theta * n_phi);
    for i in 0..n_theta {
        let r = (1.0 - nodes[i] * nodes[i]).max(0.0).sqrt();
        for j in 0..n_phi {
            let phi = 2.0 * std::f64::consts::PI * (j as f64 + 0.5) / n_phi as f64;
            dirs.push((
                [r * phi.cos(), r * phi.sin(), nodes[i]],
                weights[i] * 2.0 * std::f64::consts::PI / n_phi as f64,
            ));
        }
    }
    dirs
}

/// Pick the product rule closest to `total` nodes (default 288 = 12 × 24).
pub fn directions(total: usize) -> Vec<([f64; 3], f64)> {
    let want = total.max(8) as f64;
    let n_theta = ((want / 2.0).sqrt().round() as usize).max(2);
    product_rule(n_theta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_sum_to_four_pi() {
        for n in [2usize, 6, 12, 20] {
            let s: f64 = product_rule(n).iter().map(|(_, w)| w).sum();
            assert!((s - 4.0 * std::f64::consts::PI).abs() < 1e-12, "n={n} sum={s}");
        }
    }

    #[test]
    fn tensor_moment_vanishes() {
        // Σω T(u) = 0 exactly (this is what kills the log-standoff divergence).
        for n in [2usize, 6, 12, 20] {
            let mut m = [0.0f64; 3];
            let mut off = [0.0f64; 3];
            for (u, w) in product_rule(n) {
                for i in 0..3 {
                    m[i] += w * (3.0 * u[i] * u[i] - 1.0);
                }
                off[0] += w * 3.0 * u[0] * u[1];
                off[1] += w * 3.0 * u[0] * u[2];
                off[2] += w * 3.0 * u[1] * u[2];
            }
            for v in m.iter().chain(off.iter()) {
                assert!(v.abs() < 1e-12, "n={n} moment {v}");
            }
        }
    }

    #[test]
    fn default_direction_count() {
        assert_eq!(directions(288).len(), 288);
    }
}
