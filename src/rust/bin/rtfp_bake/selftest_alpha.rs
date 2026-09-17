fn selftest_carlson_alpha(args: &Args) -> Result<(), String> {
    use crate::carlson_alpha::special;
    let pi = std::f64::consts::PI;
    let rf = special::rf(0.0, 1.0, 1.0)?;
    let rd = special::rd(0.0, 1.0, 1.0)?;
    let rc = special::rc(0.0, 1.0)?;
    let rj = special::rj(1.0, 1.0, 1.0, 1.0)?;
    let tail = carlson_alpha::third_kind_tail(1.0, 2.0, 3.0, 4.0)?;
    println!("  Carlson library: RF={rf:.12} RD={rd:.12} RC={rc:.12} RJ={rj:.12} tail={tail:.12}");
    if (rf - pi / 2.0).abs() > 1e-12
        || (rd - 3.0 * pi / 4.0).abs() > 1e-12
        || (rc - pi / 2.0).abs() > 1e-12
        || (rj - 1.0).abs() > 1e-12
        || !tail.is_finite()
    {
        return Err("Carlson library identity check failed".into());
    }

    let mesh = unit_cube();
    let faces = analytic::precompute(&mesh);
    let bvh = bvh::Bvh::build(&mesh);
    let points = mesh.observation_points(1e-2);
    let sample: Vec<[f64; 3]> = (0..24).map(|i| points[i * points.len() / 24]).collect();
    let dirs = quadrature::directions(args.directions.min(256));
    let kernel = crate::density::KernelSi {
        c: [0.1, -0.1, 0.2],
        sigma: 0.8,
        w: 1.0,
        alpha: 1.25,
    };
    let device = gpu::Device::new()?;
    let scene = gpu::Scene::new(device, &mesh, &bvh, &faces, &sample, &dirs, sample.len(), 1);
    let radius = body_radius(&mesh);
    let t_max = radius * 4.0;
    let gpu = scene.block_carlson_alpha(
        &sample,
        &dirs,
        std::slice::from_ref(&kernel),
        t_max as f32,
        GEOM_EPS_M as f32,
    )?;

    let mut hits = Vec::new();
    let mut errs = Vec::new();
    for (i, x) in sample.iter().enumerate() {
        // `block_carlson_alpha` returns the regularized radial remainder only.
        // The analytic uniform-density tensor belongs to the separate near-field
        // term, so comparing it against a full `rho·W + remainder` reference
        // would manufacture a large false disagreement.
        let mut want = [0.0; 6];
        for (u, omega) in &dirs {
            geom::bvh_crossings(
                &bvh,
                &mesh,
                *x,
                [-u[0], -u[1], -u[2]],
                GEOM_EPS_M,
                t_max,
                &mut hits,
            );
            let (slots, overflow) = split::intervals(&hits, false);
            if overflow {
                return Err("alpha selftest ray exceeded interval capacity".into());
            }
            let intervals: Vec<(f64, f64)> = slots.iter().flatten().copied().collect();
            let scalar = carlson_alpha::remainder_scalar_reference(
                std::slice::from_ref(&kernel),
                *x,
                *u,
                &intervals,
            );
            tensor::add_tensor_term(&mut want, u, G * *omega * scalar);
        }
        errs.push(rel6(&gpu[i], &want));
    }
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let worst = errs[errs.len() - 1];
    println!(
        "  CarlsonAlpha alpha=1.25 GPU vs f64/Carlson reference: median {:.3e}, worst {:.3e}",
        errs[errs.len() / 2],
        worst
    );
    if worst.is_nan() || worst > 2e-2 {
        return Err(format!(
            "CarlsonAlpha general-alpha GPU path disagrees by {worst:.3e}"
        ));
    }
    Ok(())
}

/// The identity the Carlson solver rests on, checked end-to-end on the real mesh:
/// with one density everywhere every cone weight `ρ_f − ρ_ref` vanishes, so the
/// 983 040-triangle face list has to collapse back onto `ρ_ref·W(x)` — the same
/// quantity `analytic.rs`, the Werner bake and the ESA library compute from
/// entirely different code. This is what proves the per-face weights survive the
/// trip through the shader and that the mesh faces carry `ρ_ref`.
///
/// The star-cone *orientations* are checked separately, and on the unit cube:
/// a cone only tiles a body that is star-shaped from its apex, and Ryugu is not
/// (see the coverage/signed-ratio line below), so running the raw cone on the
/// real mesh would be measuring the body's shape rather than the code.

