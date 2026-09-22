#[derive(Clone, Copy)]
enum RayAlgorithm {
    Rtfp,
    Carlson,
    CarlsonAlpha,
}

fn run_rtfp(args: &Args) -> Result<(), String> {
    run_ray_algorithm(args, RayAlgorithm::Rtfp)
}

fn run_carlson(args: &Args) -> Result<(), String> {
    run_ray_algorithm(args, RayAlgorithm::Carlson)
}

fn run_carlson_alpha(args: &Args) -> Result<(), String> {
    run_ray_algorithm(args, RayAlgorithm::CarlsonAlpha)
}

fn run_ray_algorithm(args: &Args, algorithm: RayAlgorithm) -> Result<(), String> {
    let standoff_m = args.standoff_mm * 1e-3;
    let mesh =
        Mesh::load_glb(&args.mesh, KM_TO_M).map_err(|e| format!("{}: {e}", args.mesh.display()))?;
    let nv = mesh.vertex_count();
    let nf = mesh.face_count();
    println!("observation standoff: {:.3} mm", args.standoff_mm);
    println!("mesh: {nv} vertices, {nf} faces (meters)");
    println!(
        "solver={} mode={:?} directions={}",
        match algorithm {
            RayAlgorithm::Rtfp => "rtfp",
            RayAlgorithm::Carlson => "carlson",
            RayAlgorithm::CarlsonAlpha => "carlson-alpha",
        },
        args.mode,
        args.directions
    );
    std::io::Write::flush(&mut std::io::stdout()).ok();

    let points = mesh.observation_points(standoff_m);
    let faces = analytic::precompute(&mesh);
    let bvh = bvh::Bvh::build(&mesh);
    println!("bvh: {} nodes, depth {}", bvh.node_count(), bvh.depth);
    let radius = body_radius(&mesh);
    let t_min = GEOM_EPS_M as f32;
    let t_max = (radius * 4.0) as f32;

    let density = match args.mode {
        DensityMode::Cauchy | DensityMode::Elliptic
            if matches!(algorithm, RayAlgorithm::CarlsonAlpha) =>
        {
            Density::from_toml_general_alpha(&args.density, &mesh, args.normalize)?
        }
        DensityMode::Cauchy | DensityMode::Elliptic => {
            Density::from_toml(&args.density, &mesh, args.normalize)?
        }
        DensityMode::Constant => Density::homogeneous(1190.0, &mesh),
    };
    print_mass_report(&density, args.mode);
    let dirs = quadrature::directions(args.directions);
    let kernels = density.kernels();
    println!(
        "quadrature: {} directions (exact: Σω=4π, ΣωT=0)",
        dirs.len()
    );
    std::io::Write::flush(&mut std::io::stdout()).ok();

    let device = gpu::Device::new()?;
    let scene = gpu::Scene::new(
        device,
        &mesh,
        &bvh,
        &faces,
        &points,
        &dirs,
        POINTS_PER_BLOCK,
        kernels.len(),
    );
    // Which vertices the record still needs (resume).
    let record = if args.resume {
        let existing = record::Record::open_resume(&args.out, nf)
            .map_err(|e| format!("{}: {e}", args.out.display()))?;
        match existing {
            Some(r) if (r.standoff_mm - args.standoff_mm as f32).abs() < 1e-3 => {
                println!("RESUME from {} / {} faces", r.n_done, nf);
                r
            }
            _ => record::Record::create(&args.out, nf, args.standoff_mm as f32)
                .map_err(|e| format!("{}: {e}", args.out.display()))?,
        }
    } else {
        record::Record::create(&args.out, nf, args.standoff_mm as f32)
            .map_err(|e| format!("{}: {e}", args.out.display()))?
    };
    let mut needed = vec![false; nv];
    let mut todo = Vec::new();
    for f in 0..nf {
        if !record.scalars[f].is_finite() {
            todo.push(f as u32);
            for v in mesh.face(f) {
                needed[v as usize] = true;
            }
        }
    }
    println!(
        "evaluating {} vertices ({} faces remaining)",
        needed.iter().filter(|b| **b).count(),
        todo.len()
    );
    std::io::Write::flush(&mut std::io::stdout()).ok();

    // Per-vertex tensors, not scalars: the face value has to be the norm of the
    // three-vertex-averaged tensor used by every RHGF producer. Averaging the
    // three per-vertex norms
    // instead is a *different* (always larger, by the triangle inequality)
    // quantity — it was worth 1.9 % median against the Werner record.
    let mut vertex_h = vec![[f64::NAN; 6]; nv];
    let resumed_vertices = if args.resume {
        match checkpoint::load(&args.checkpoint, nv, args.standoff_mm)
            .map_err(|e| format!("{}: {e}", args.checkpoint.display()))?
        {
            Some(saved) => {
                let count = saved.len();
                vertex_h[..count].copy_from_slice(&saved);
                println!("RESUME from {count} / {nv} vertex tensors");
                count
            }
            None => 0,
        }
    } else {
        0
    };
    if !todo.is_empty() {
        for (block, lo) in (0..nv).step_by(POINTS_PER_BLOCK).enumerate() {
            let hi = (lo + POINTS_PER_BLOCK).min(nv);
            if hi <= resumed_vertices {
                println!("PROGRESS_R {hi} {nv}");
                continue;
            }
            let (w_block, remainder) = match algorithm {
                RayAlgorithm::Rtfp => (
                    scene.rtfp_analytic_tensors_block(lo, hi)?,
                    scene.block_remainder(&points[lo..hi], &dirs, kernels, t_max, t_min)?,
                ),
                RayAlgorithm::Carlson | RayAlgorithm::CarlsonAlpha => (
                    scene.carlson_alpha_analytic_tensors_block(lo, hi)?,
                    scene.block_carlson_alpha(&points[lo..hi], &dirs, kernels, t_max, t_min)?,
                ),
            };
            for (i, rem) in remainder.into_iter().enumerate() {
                let vi = lo + i;
                if !needed[vi] {
                    continue;
                }
                let rho = density.rho(&points[vi], args.mode);
                let mut h = w_block[i];
                for t in h.iter_mut() {
                    *t *= rho;
                }
                for (t, r) in h.iter_mut().zip(rem.iter()) {
                    *t += r;
                }
                vertex_h[vi] = h;
            }
            if (block + 1) % CHECKPOINT_BLOCKS == 0 || hi == nv {
                checkpoint::save(&args.checkpoint, &vertex_h, hi, args.standoff_mm)
                    .map_err(|e| format!("{}: {e}", args.checkpoint.display()))?;
            }
            println!("PROGRESS_R {hi} {nv}");
            std::io::Write::flush(&mut std::io::stdout()).ok();
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    write_face_record(args, &mesh, &vertex_h, record)
}

/// Per-face scalars from per-vertex tensors, written progressively in the saved
/// (shuffled) order so the viewer can start drawing before the bake finishes.
///
/// The face value is the norm of the *averaged* tensor, exactly like
/// Shared RHGF face convention:
/// averaging the three per-vertex norms instead is a different (always larger,
/// by the triangle inequality) quantity.
fn write_face_record(
    args: &Args,
    mesh: &Mesh,
    vertex_h: &[[f64; 6]],
    mut record: record::Record,
) -> Result<(), String> {
    let nf = mesh.face_count();
    let saved_order = record::load_order(&args.order, nf)
        .map_err(|e| format!("{}: {e}", args.order.display()))?;
    let order = match saved_order {
        Some(o) => o,
        None => {
            let mut o: Vec<u32> = (0..nf as u32).collect();
            let mut rng = 0x9E37_79B9_7F4A_7C15u64
                ^ (args.standoff_mm as u64)
                ^ (std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.subsec_nanos() as u64)
                    .unwrap_or(0));
            for i in (1..o.len()).rev() {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                o.swap(i, (rng as usize) % (i + 1));
            }
            record::write_order(&args.order, &o)
                .map_err(|e| format!("{}: {e}", args.order.display()))?;
            o
        }
    };
    let mut step = 0usize;
    for f in order {
        let fi = f as usize;
        if record.scalars[fi].is_finite() {
            continue;
        }
        let face = mesh.face(fi);
        let mut hbar = [0.0f64; 6];
        for k in 0..6 {
            hbar[k] = (vertex_h[face[0] as usize][k]
                + vertex_h[face[1] as usize][k]
                + vertex_h[face[2] as usize][k])
                / 3.0;
        }
        let s = frobenius(&hbar) as f32;
        record
            .set_face(fi, s)
            .map_err(|e| format!("{}: {e}", args.out.display()))?;
        step += 1;
        if step.is_multiple_of(256) || record.n_done == nf {
            record
                .flush_header()
                .map_err(|e| format!("{}: {e}", args.out.display()))?;
            println!("faces {} / {}", record.n_done, nf);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
    }
    record
        .flush_header()
        .map_err(|e| format!("{}: {e}", args.out.display()))?;
    println!(
        "wrote {} dense face scalars to {} (s_min={:e} s_max={:e})",
        nf,
        args.out.display(),
        record.s_min,
        record.s_max
    );
    Ok(())
}
