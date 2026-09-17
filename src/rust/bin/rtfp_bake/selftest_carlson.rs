fn selftest_carlson(args: &Args) -> Result<(), String> {
    let rho = 1190.0f64;
    let mesh =
        Mesh::load_glb(&args.mesh, KM_TO_M).map_err(|e| format!("{}: {e}", args.mesh.display()))?;
    let density = Density::homogeneous(rho, &mesh);
    let (faces, stats) = carlson::face_list(&mesh, &density, DensityMode::Constant);
    println!(
        "  carlson face list: {} triangles ({} mesh + {} cone), cone coverage {:.5} %, \
         |signed|/covered {:.6}",
        faces.len(),
        stats.n_mesh_faces,
        stats.n_faces,
        100.0 * stats.coverage(),
        stats.signed_ratio()
    );

    let all = mesh.observation_points(1e-3);
    let nv = mesh.vertex_count();
    let sample: Vec<[f64; 3]> = (0..64).map(|i| all[i * nv / 64]).collect();
    let reference = analytic::precompute(&mesh);
    let dirs = quadrature::directions(8);
    let bvh = bvh::Bvh::build(&mesh);
    let device = gpu::Device::new()?;
    let scene = gpu::Scene::new(
        device,
        &mesh,
        &bvh,
        &faces,
        &sample,
        &dirs,
        POINTS_PER_BLOCK,
        0,
    );
    let got = scene.carlson_surface_tensors()?;
    let mut errs: Vec<f64> = sample
        .iter()
        .enumerate()
        .map(|(i, x)| {
            let mut want = analytic::hessian6(&reference, *x);
            for t in want.iter_mut() {
                *t *= rho;
            }
            rel6(&got[i], &want)
        })
        .collect();
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let worst = errs[errs.len() - 1];
    println!(
        "  carlson constant-density identity vs polyhedral ρ·W: median {:.3e}, worst {:.3e} \
         (f32 analytic alone reaches ~3e-3 at 1 mm)",
        errs[errs.len() / 2],
        worst
    );
    if worst > 1e-2 {
        return Err(format!(
            "Carlson jump surfaces disagree with ρ·W by {worst:.3e}; the face weights or \
             orientations are wrong",
        ));
    }
    selftest_cross_solver(&mesh)?;
    Ok(())
}

/// The Carlson face list against the RT-FP tensor on the *real* density field.
///
/// This is the cross-validation the two solvers exist for, and it runs on the CPU
/// in seconds, so it can gate CI without a GPU or a bake. What it measures is the
/// jump-surface representation error — how much of `ρ` a piecewise-constant cell
/// fails to carry — not the arithmetic of either solver; the two share no code
/// beyond the mesh and the face kernel.
fn selftest_cross_solver(mesh: &Mesh) -> Result<(), String> {
    let root = crate::root();
    let path = root.join("assets/density/cauchy.toml");
    let density = Density::from_toml(&path, mesh, Normalization::TotalMass)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let kernels = density.kernels();
    let (faces, stats) = carlson::face_list(mesh, &density, DensityMode::Cauchy);
    println!(
        "  carlson Cauchy face list: {} triangles ({} slabs, worst within-slab variation \
         {:.3e} of ρ_ref)",
        faces.len(),
        stats.slabs,
        stats.worst_slab_variation
    );

    let uniform = analytic::precompute(mesh);
    let tree = bvh::Bvh::build(mesh);
    let radius = body_radius(mesh);
    let t_max = radius * 4.0;
    // 288 directions is what the RT-FP bake itself uses, and on this field its own
    // quadrature error against a 1568-direction run is ~8e-5, two orders below the
    // representation error being measured here.
    let dirs = quadrature::directions(288);
    // Sweep the slider's useful range. The near end is the hard one: a millimetre
    // above a face, the tensor is dominated by that face, so a cell-level density
    // error is amplified by the near-field weighting that also makes the voxel
    // direct sum unusable there.
    for standoff_mm in [1.0f64, 1_000.0, 4_000.0, 8_000.0, 16_000.0, 32_000.0] {
        let all = mesh.observation_points(standoff_mm * 1e-3);
        let nv = mesh.vertex_count();
        let sample: Vec<[f64; 3]> = (0..32).map(|i| all[i * nv / 32]).collect();
        let mut errs: Vec<f64> = sample
            .iter()
            .map(|x| -> Result<f64, String> {
                let want = carlson::rtfp_reference(
                    mesh, &tree, &uniform, &density, kernels, x, t_max, &dirs,
                )?;
                Ok(rel6(&analytic::hessian6(&faces, *x), &want))
            })
            .collect::<Result<_, _>>()?;
        errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let worst = errs[errs.len() - 1];
        println!(
            "  carlson vs RT-FP at standoff {standoff_mm} mm: median {:.3e}, worst {:.3e}",
            errs[errs.len() / 2],
            worst
        );
        // A representation error, not an arithmetic one, and it is bounded by the
        // within-slab density variation the face list reports above. The bound is
        // loose on purpose: what it has to catch is a wrong weight or a wrong
        // orientation, which shows up at O(1).
        if worst.is_nan() || worst >= 6e-2 {
            return Err(format!(
                "Carlson and RT-FP disagree by {worst:.3e} at {standoff_mm} mm; either the \
                 jump-surface representation has lost the density field or one of the two \
                 solvers is wrong"
            ));
        }
    }
    Ok(())
}

/// Closed form vs the ESA library on the real mesh, at every standoff the UI
/// slider can ask for. The symmetry-axis cube cases pass even with the classic
/// `h²`-for-`L²` slip; this is the test that catches it.
