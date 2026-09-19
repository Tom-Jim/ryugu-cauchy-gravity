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
        // Independent reference for the hybrid residual. Perturbing each
        // direction re-traces the mesh, so the f64 finite difference includes
        // moving interval endpoints without reusing the GPU face-normal formula.
        let mut want = [0.0; 6];
        for (u, omega) in &dirs {
            let mut scalar_at = |direction: [f64; 3]| -> Result<f64, String> {
                geom::bvh_crossings(
                    &bvh,
                    &mesh,
                    *x,
                    [-direction[0], -direction[1], -direction[2]],
                    GEOM_EPS_M,
                    t_max,
                    &mut hits,
                );
                let (slots, overflow) = split::intervals(&hits, false);
                if overflow {
                    return Err("alpha selftest ray exceeded interval capacity".into());
                }
                let intervals: Vec<(f64, f64)> = slots.iter().flatten().copied().collect();
                Ok(carlson_alpha::remainder_scalar_reference(
                    std::slice::from_ref(&kernel),
                    *x,
                    direction,
                    &intervals,
                ))
            };
            let epsilon = 2.0e-4;
            let mut gradient = [0.0; 3];
            for axis in 0..3 {
                let mut tangent = [0.0; 3];
                tangent[axis] = 1.0;
                for component in 0..3 {
                    tangent[component] -= u[axis] * u[component];
                }
                let mut plus = [0.0; 3];
                let mut minus = [0.0; 3];
                for component in 0..3 {
                    plus[component] = u[component] + epsilon * tangent[component];
                    minus[component] = u[component] - epsilon * tangent[component];
                }
                let plus_norm = (plus[0] * plus[0] + plus[1] * plus[1] + plus[2] * plus[2]).sqrt();
                let minus_norm =
                    (minus[0] * minus[0] + minus[1] * minus[1] + minus[2] * minus[2]).sqrt();
                for component in 0..3 {
                    plus[component] /= plus_norm;
                    minus[component] /= minus_norm;
                }
                gradient[axis] = (scalar_at(plus)? - scalar_at(minus)?) / (2.0 * epsilon);
            }
            let amplitude = G * *omega;
            want[0] += amplitude * u[0] * gradient[0];
            want[1] += amplitude * u[1] * gradient[1];
            want[2] += amplitude * u[2] * gradient[2];
            want[3] += 0.5 * amplitude * (u[1] * gradient[0] + u[0] * gradient[1]);
            want[4] += 0.5 * amplitude * (u[2] * gradient[0] + u[0] * gradient[2]);
            want[5] += 0.5 * amplitude * (u[2] * gradient[1] + u[1] * gradient[2]);
        }
        errs.push(rel6(&gpu[i], &want));
    }
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let worst = errs[errs.len() - 1];
    println!(
        "  CarlsonAlpha alpha=1.25 hybrid GPU vs f64 directional derivative: median {:.3e}, worst {:.3e}",
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
