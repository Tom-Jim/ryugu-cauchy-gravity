fn selftest_surface_form(args: &Args) -> Result<(), String> {
    let mesh =
        Mesh::load_glb(&args.mesh, KM_TO_M).map_err(|e| format!("{}: {e}", args.mesh.display()))?;
    let faces = analytic::precompute(&mesh);
    let ev = esa::Evaluable::new(&mesh, 1.0)?;
    let nv = mesh.vertex_count();
    let mut rng = 0x2545_F491_4F6C_DD1Du64;
    let mut next_u32 = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as u32
    };
    let radius = (0..nv)
        .map(|i| {
            let p = mesh.vertex(i);
            (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
        })
        .fold(0.0f64, f64::max);

    let mut samples: Vec<([f64; 3], String)> = Vec::new();
    for mm in [1.0f64, 10.0, 100.0, 500.0] {
        let pts = mesh.observation_points(mm * 1e-3);
        for _ in 0..8 {
            let i = next_u32() as usize % nv;
            samples.push((pts[i], format!("surface@{mm}mm v{i}")));
        }
    }
    for k in 0..8 {
        let theta = 2.0 * std::f64::consts::PI * (k as f64) / 8.0;
        let r = radius * (1.15 + 0.05 * k as f64);
        samples.push((
            [
                r * theta.cos(),
                0.3 * r * theta.sin(),
                0.2 * r * theta.cos(),
            ],
            format!("free r={r:.1}m"),
        ));
    }

    let mut worst = (0.0f64, String::new());
    let mut errs = Vec::new();
    for (x, label) in samples.iter() {
        let lib = ev.hessian6(*x)?;
        let mine = analytic::hessian6(&faces, *x);
        let rel = rel6(&mine, &lib);
        errs.push(rel);
        if rel > worst.0 {
            worst = (rel, label.clone());
        }
    }
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "  closed form vs ESA on the real mesh: {} samples, median {:.3e}, worst {:.3e} ({})",
        samples.len(),
        errs[errs.len() / 2],
        worst.0,
        worst.1
    );
    if worst.0 > 1e-6 {
        return Err(format!(
            "closed form disagrees with the ESA library by {:.3e} ({})",
            worst.0, worst.1
        ));
    }
    Ok(())
}

/// WGSL pipelines against the references that do not share their code:
/// `analytic.rs` (closed form, itself checked against ESA) for the tensor, and a
/// brute-force f64 tracer + `split.rs` for the ray/remainder chain. The BVH
/// traversal the shader uses is checked separately, by walking the same flat
/// arrays from Rust.
fn selftest_gpu(args: &Args) -> Result<(), String> {
    let mesh =
        Mesh::load_glb(&args.mesh, KM_TO_M).map_err(|e| format!("{}: {e}", args.mesh.display()))?;
    let faces = analytic::precompute(&mesh);
    let bvh = bvh::Bvh::build(&mesh);
    let radius = body_radius(&mesh);
    let t_max = radius * 4.0;
    let dirs = quadrature::directions(args.directions);
    let density = match args.mode {
        DensityMode::Cauchy => Density::from_toml(&args.density, &mesh, args.normalize)?,
        DensityMode::Elliptic => {
            return Err("mode=elliptic requires --solver carlson-alpha".into());
        }
        DensityMode::Constant => Density::homogeneous(1190.0, &mesh),
    };
    let kernels = density.kernels();
    let all = mesh.observation_points(1e-3);
    let nv = mesh.vertex_count();
    let sample: Vec<[f64; 3]> = (0..192).map(|i| all[i * nv / 192]).collect();

    let device = gpu::Device::new()?;
    let scene = gpu::Scene::new(
        device,
        &mesh,
        &bvh,
        &faces,
        &sample,
        &dirs,
        sample.len(),
        kernels.len(),
    );

    // 1. Analytic tensor: WGSL (f32) vs the f64 closed form.
    let w_gpu = scene.analytic_tensors()?;
    let mut errs: Vec<f64> = sample
        .iter()
        .enumerate()
        .map(|(i, p)| rel6(&analytic::hessian6(&faces, *p), &w_gpu[i]))
        .collect();
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let worst_w = errs[errs.len() - 1];
    println!(
        "  WGSL analytic vs f64 closed form: median {:.3e}, worst {:.3e}",
        errs[errs.len() / 2],
        worst_w
    );

    // 2. Ray traversal: the shader's BVH layout, walked from Rust, vs brute force.
    let brute = BruteTracer::new(&mesh);
    let mut hits = Vec::new();
    let mut bvh_hits = Vec::new();
    let mut worst_ray = 0.0f64;
    for (i, p) in sample.iter().enumerate().take(32) {
        for (q, (u, _)) in dirs.iter().enumerate() {
            let d = [-u[0], -u[1], -u[2]];
            brute.crossings(*p, d, GEOM_EPS_M, t_max, &mut hits);
            geom::bvh_crossings(&bvh, &mesh, *p, d, GEOM_EPS_M, t_max, &mut bvh_hits);
            if hits.len() != bvh_hits.len() {
                worst_ray = f64::INFINITY;
                println!(
                    "    point {i} direction {q}: {} crossings, BVH found {}",
                    hits.len(),
                    bvh_hits.len()
                );
                continue;
            }
            for (a, b) in hits.iter().zip(bvh_hits.iter()) {
                worst_ray = worst_ray.max((a - b).abs() / a.abs().max(1e-30));
            }
        }
    }
    println!("  BVH traversal vs brute force: worst crossing rel {worst_ray:.3e}");

    // 3. Remainder: full WGSL chain (probe + rays + remainder) vs f64 brute force.
    let rem_gpu =
        scene.block_remainder(&sample, &dirs, kernels, t_max as f32, GEOM_EPS_M as f32)?;
    let mut errs: Vec<f64> = Vec::new();
    let mut worst = (0.0f64, 0usize);
    for (i, p) in sample.iter().enumerate() {
        brute.crossings(*p, [1.0, 0.0, 0.0], GEOM_EPS_M, t_max, &mut hits);
        let inside = hits.len() % 2 == 1;
        let mut acc = [0.0f64; 6];
        for (u, w) in &dirs {
            brute.crossings(*p, [-u[0], -u[1], -u[2]], GEOM_EPS_M, t_max, &mut hits);
            let (slots, overflow) = split::intervals(&hits, inside);
            if overflow {
                return Err(format!(
                    "reference ray at {p:?} exceeded {} visible intervals",
                    split::MAX_INTERVALS
                ));
            }
            let s = split::remainder_scalar(kernels, *p, *u, &slots);
            tensor::add_tensor_term(&mut acc, u, G * *w * s);
        }
        let rel = rel6(&acc, &rem_gpu[i]);
        errs.push(rel);
        if rel > worst.0 {
            worst = (rel, i);
        }
    }
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let over = errs.iter().filter(|e| **e > 5e-2).count();
    println!(
        "  WGSL rays+remainder vs f64 brute force: median {:.3e}, p90 {:.3e}, worst {:.3e} ({over} of {} >5%)",
        errs[errs.len() / 2],
        errs[errs.len() * 9 / 10],
        worst.0,
        errs.len()
    );
    // A handful of rays graze a mesh vertex, where a crossing sits exactly on the
    // f32/f64 hit tolerance and can flip; the rest of the chain has to be tight.
    let median_rem = errs[errs.len() / 2];
    if worst_w > 1e-2 || worst_ray > 1e-6 || median_rem > 1e-3 || worst.0 > 2e-1 {
        return Err("GPU pipelines disagree with the reference".into());
    }
    Ok(())
}
