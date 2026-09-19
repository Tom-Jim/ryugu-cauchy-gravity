/// Star-cone jump surfaces only, each carrying `weight = rho(tet) - rho_ref`.
pub fn star_faces(
    mesh: &Mesh,
    density: &Density,
    mode: DensityMode,
    origin: [f64; 3],
    rho_ref: f64,
    refine: &Refine,
) -> (Vec<Face>, CarlsonStats) {
    let n_mesh_faces = mesh.face_count();
    let samples = refine.samples.max(4);

    let mut corners: Vec<[[f64; 3]; 3]> = Vec::with_capacity(n_mesh_faces);
    let mut variation: Vec<f64> = Vec::with_capacity(n_mesh_faces);
    let mut covered = 0.0f64;
    let mut signed = 0.0f64;
    for index in 0..n_mesh_faces {
        let [i0, i1, i2] = mesh.face(index);
        let v0 = mesh.vertex(i0 as usize);
        let v1 = mesh.vertex(i1 as usize);
        let v2 = mesh.vertex(i2 as usize);
        corners.push([v0, v1, v2]);
        let volume = tet_signed_volume(origin, v0, v1, v2);
        covered += volume.abs();
        signed += volume;
        let profile = axis_profile(origin, face_centre([v0, v1, v2]), density, mode, samples);
        variation.push(total_variation(&profile));
    }

    // Triangle count is the GPU cost. Relax the tolerance before emitting data
    // when the requested refinement would exceed the configured budget.
    let mut tol = refine.tol.max(f64::MIN_POSITIVE);
    let estimate = |tol: f64| -> f64 {
        variation
            .iter()
            .map(|value| {
                (7.0 * slabs_for(*value, tol, refine.max_slabs, rho_ref) as f64 - 3.0).max(0.0)
            })
            .sum()
    };
    while estimate(tol) > refine.face_budget as f64 {
        if tol >= 1.0 {
            break;
        }
        tol *= 1.5;
    }

    let mut faces: Vec<Face> = Vec::new();
    let mut slabs_total = 0usize;
    let mut worst_variation = 0.0f64;
    let scale = rho_ref.abs().max(f64::MIN_POSITIVE);

    for (index, tri) in corners.iter().enumerate() {
        let [v0, v1, v2] = *tri;
        let profile = axis_profile(origin, face_centre(*tri), density, mode, samples);
        let slab_count = slabs_for(variation[index], tol, refine.max_slabs, rho_ref);
        let bounds = slab_boundaries(&profile, slab_count);
        slabs_total += slab_count;

        let mut lo = 0.0f64;
        for (slab, hi) in bounds.iter().copied().enumerate() {
            let rho_slab = slab_mean(&profile, lo, hi);
            let (min_density, max_density) = slab_range(&profile, lo, hi);
            worst_variation = worst_variation.max((max_density - min_density) / scale);
            let jump = rho_slab - rho_ref;

            let centre = frustum_centre(origin, *tri, lo, hi);
            for (a, b) in [(v0, v1), (v1, v2), (v2, v0)] {
                let p = lerp(origin, a, lo);
                let q = lerp(origin, b, lo);
                let r = lerp(origin, b, hi);
                let s = lerp(origin, a, hi);
                if lo > 0.0 {
                    push_face(&mut faces, p, q, r, centre, jump);
                    push_face(&mut faces, p, r, s, centre, jump);
                } else {
                    push_face(&mut faces, origin, r, s, centre, jump);
                }
            }

            let next_jump = if slab + 1 < slab_count {
                slab_mean(&profile, hi, bounds[slab + 1]) - rho_ref
            } else {
                0.0
            };
            let cap = jump - next_jump;
            push_face(
                &mut faces,
                lerp(origin, v0, hi),
                lerp(origin, v1, hi),
                lerp(origin, v2, hi),
                centre,
                cap,
            );
            lo = hi;
        }
    }

    let stats = CarlsonStats {
        covered_volume: covered,
        signed_volume: signed,
        mesh_volume: mass::body_volume(mesh),
        n_faces: faces.len(),
        n_mesh_faces: 0,
        slabs: slabs_total,
        worst_slab_variation: worst_variation,
    };
    (faces, stats)
}
