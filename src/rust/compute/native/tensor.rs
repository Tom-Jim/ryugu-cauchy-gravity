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

