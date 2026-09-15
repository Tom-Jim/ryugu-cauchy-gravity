//! Symmetric 3x3 tensor packed as the six independent components
//! `[xx, yy, zz, xy, xz, yz]` — the same layout the ESA polyhedral library
//! returns and the same one the records store implicitly through ‖H‖_F.

pub type Sym6 = [f64; 6];

/// Frobenius norm of the symmetric tensor (off-diagonal terms counted twice).
pub fn frobenius(a: &Sym6) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + 2.0 * (a[3] * a[3] + a[4] * a[4] + a[5] * a[5]))
        .sqrt()
}

/// `t += s · T(u)` with `T(u) = 3uuᵀ − I` (the kernel of the ray representation).
pub fn add_tensor_term(t: &mut Sym6, u: &[f64; 3], s: f64) {
    t[0] += s * (3.0 * u[0] * u[0] - 1.0);
    t[1] += s * (3.0 * u[1] * u[1] - 1.0);
    t[2] += s * (3.0 * u[2] * u[2] - 1.0);
    t[3] += s * 3.0 * u[0] * u[1];
    t[4] += s * 3.0 * u[0] * u[2];
    t[5] += s * 3.0 * u[1] * u[2];
}

#[cfg(test)]
mod tests {
    use super::*;

    const ZERO: Sym6 = [0.0; 6];

    #[test]
    fn frobenius_matches_matrix_form() {
        let t: Sym6 = [1.0, -2.0, 0.5, 0.25, -0.75, 1.5];
        let expected = (1.0f64 + 4.0 + 0.25 + 2.0 * (0.0625 + 0.5625 + 2.25)).sqrt();
        assert!((frobenius(&t) - expected).abs() < 1e-12);
    }

    #[test]
    fn traceless_axisymmetric_tensor() {
        // T(z) = diag(-1, -1, 2)
        let mut t = ZERO;
        add_tensor_term(&mut t, &[0.0, 0.0, 1.0], 1.0);
        assert!((t[0] + 1.0).abs() < 1e-12);
        assert!((t[1] + 1.0).abs() < 1e-12);
        assert!((t[2] - 2.0).abs() < 1e-12);
        assert!(t[3].abs() < 1e-12 && t[4].abs() < 1e-12 && t[5].abs() < 1e-12);
    }
}
