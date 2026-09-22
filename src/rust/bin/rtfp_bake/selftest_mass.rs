fn body_radius(mesh: &Mesh) -> f64 {
    (0..mesh.vertex_count())
        .map(|i| {
            let p = mesh.vertex(i);
            (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
        })
        .fold(0.0f64, f64::max)
}

/// Print both normalisation conventions side by side, so the constant factor
/// between this record and one baked under the other convention is on the log
/// rather than buried inside a face-by-face comparison.
fn print_mass_report(density: &Density, mode: DensityMode) {
    let m = &density.mass;
    let d = density.decomposition;
    println!(
        "mass integral (divergence theorem): body volume {:.6e} m³",
        m.volume
    );
    match mode {
        DensityMode::Cauchy => {
            let target = if m.target_mass > 0.0 {
                format!("{:.6e} kg", m.target_mass)
            } else {
                "absent from TOML".to_string()
            };
            println!(
                "  raw TOML weights: ∫ρ dV = {:.6e} kg  →  ρ(0)=1190 convention ×{:.9e} = {:.6e} kg",
                m.raw_mass, m.rho0_scale, m.rho0_mass
            );
            println!(
                "  normalization={:?}: ×{:.9e} = {:.6e} kg (target {target})",
                m.normalization, m.applied_scale, m.applied_mass
            );
            if m.rho0_mass.abs() > 1e-30 {
                println!(
                    "  scale vs the old ρ(0) convention: ×{:.9} ({:+.4} %) — a constant on every face",
                    m.scale_vs_rho0(),
                    100.0 * (m.scale_vs_rho0() - 1.0)
                );
            }
            println!(
                "  alpha→1 decomposition: {} → {} kernels, max peak-relative error {:.3e}",
                d.original_kernels, d.expanded_kernels, d.max_peak_error
            );
        }
        DensityMode::Elliptic => println!(
            "  general-alpha field: original exponents preserved, normalization={:?}, weight scale ×{:.9e}",
            m.normalization, m.applied_scale
        ),
        DensityMode::Constant => println!(
            "  constant {:.1} kg/m³ × V = {:.6e} kg (the Werner bake's density, unchanged)",
            density.bulk_density, m.applied_mass
        ),
    }
    std::io::Write::flush(&mut std::io::stdout()).ok();
}

/// Checks the volume→surface reduction in `mass.rs` against closed forms: the
/// exact enclosed volume of a cube, and the analytic ball integral
/// `4π(R − atan(σR)/σ)/σ²` for the sphere tessellations the mesh format allows.
fn selftest_mass() -> Result<(), String> {
    let cube = unit_cube();
    let v = mass::body_volume(&cube);
    println!(
        "  cube volume: {v:.12} (exact 8) → rel {:.3e}",
        (v - 8.0).abs() / 8.0
    );
    if (v - 8.0).abs() > 1e-10 {
        return Err(format!("cube volume {v} != 8"));
    }

    let mut worst = (0.0f64, 0.0f64);
    for sigma in [0.0f64, 0.05, 0.28, 1.0, 8.0] {
        let sphere = mass::uv_sphere(0.5, 128);
        let got = mass::kernel_volume(&sphere, [0.0, 0.0, 0.0], sigma);
        let want = mass::ball_kernel_volume(0.5, sigma);
        let rel = (got - want).abs() / want;
        if rel > worst.0 {
            worst = (rel, sigma);
        }
    }
    println!(
        "  sphere ∫(1+σ²r²)⁻¹dV vs closed form: worst rel {:.3e} (σ={})",
        worst.0, worst.1
    );
    if worst.0 > 3e-4 {
        return Err(format!(
            "surface mass integral disagrees with the closed form by {:.3e}",
            worst.0
        ));
    }
    Ok(())
}

fn rel6(a: &Sym6, b: &Sym6) -> f64 {
    let d = frobenius(&[
        a[0] - b[0],
        a[1] - b[1],
        a[2] - b[2],
        a[3] - b[3],
        a[4] - b[4],
        a[5] - b[5],
    ]);
    d / frobenius(b).max(1e-300)
}

