fn unit_cube() -> Mesh {
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

/// Checks that pin the two halves of the split down against trusted references:
/// the ESA polyhedral library (closed form), a direct point-mass sum (the same
/// body finely subdivided), and the CPU reference tracer for the WGSL pipelines.
fn selftest(args: &Args) -> Result<(), String> {
    let side = 2.0f64;
    let mesh = unit_cube();
    let p = [0.0, 0.0, 1.0 + 0.35];
    let ev = esa::Evaluable::new(&mesh, 1.0)?;
    let cube_faces = analytic::precompute(&mesh);
    for (label, q) in [
        ("far field", p),
        ("1 mm above face centre", [0.0, 0.0, 1.0 + 1e-3]),
        ("1 mm above edge", [1.0, 0.0, 1.0 + 1e-3]),
        // Exactly above the corner the point sits *in* the planes of two faces,
        // where the surface integral itself diverges; a hair to the side is the
        // same near-field test without the degeneracy.
        ("1 mm above corner vertex", [1.0001, 1.0001, 1.0 + 1e-3]),
        ("1 mm above face corner", [0.5, 0.5, 1.0 + 1e-3]),
    ] {
        let mine = analytic::hessian6(&cube_faces, q);
        let lib = ev.hessian6(q)?;
        let rel = rel6(&mine, &lib);
        println!("  cube {label}: rel diff {rel:.3e}");
    }

    // Direct sum over a finely subdivided cube (n³ point masses).
    let n = 24usize;
    let cell = side / n as f64;
    let m = cell * cell * cell;
    let mut direct = [0.0f64; 6];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let y = [
                    -side / 2.0 + (i as f64 + 0.5) * cell,
                    -side / 2.0 + (j as f64 + 0.5) * cell,
                    -side / 2.0 + (k as f64 + 0.5) * cell,
                ];
                let d = [y[0] - p[0], y[1] - p[1], y[2] - p[2]];
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                let r3 = r2 * r2.sqrt();
                let gm = G * m;
                let c = 3.0 * gm / (r3 * r2);
                let dd = gm / r3;
                direct[0] += c * d[0] * d[0] - dd;
                direct[1] += c * d[1] * d[1] - dd;
                direct[2] += c * d[2] * d[2] - dd;
                direct[3] += c * d[0] * d[1];
                direct[4] += c * d[0] * d[2];
                direct[5] += c * d[1] * d[2];
            }
        }
    }
    let lib = ev.hessian6(p)?;
    println!(
        "  cube direct sum vs ESA: rel {:.3e} (same sign: {})",
        rel6(&direct, &lib),
        lib[2] * direct[2] > 0.0
    );
    selftest_mass()?;
    selftest_surface_form(args)?;
    selftest_carlson(args)?;
    selftest_carlson_alpha(args)?;
    selftest_gpu(args)
}
