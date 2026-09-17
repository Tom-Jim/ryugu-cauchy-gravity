/// Independent f64 RT-FP reference used by Carlson tuning self-tests.
#[allow(clippy::too_many_arguments)]
pub fn rtfp_reference(
    mesh: &Mesh,
    tree: &crate::bvh::Bvh,
    uniform: &[Face],
    density: &Density,
    kernels: &[crate::density::KernelSi],
    x: &[f64; 3],
    t_max: f64,
    dirs: &[([f64; 3], f64)],
) -> Result<[f64; 6], String> {
    use crate::{geom, split, tensor};

    let mut hessian = crate::analytic::hessian6(uniform, *x);
    let rho = density.rho(x, DensityMode::Cauchy);
    for value in &mut hessian {
        *value *= rho;
    }

    let mut hits = Vec::new();
    geom::bvh_crossings(
        tree,
        mesh,
        *x,
        [-1.0, 0.0, 0.0],
        crate::GEOM_EPS_M,
        t_max,
        &mut hits,
    );
    let inside = hits.len() % 2 == 1;
    for (direction, weight) in dirs {
        geom::bvh_crossings(
            tree,
            mesh,
            *x,
            [-direction[0], -direction[1], -direction[2]],
            crate::GEOM_EPS_M,
            t_max,
            &mut hits,
        );
        let (intervals, overflow) = split::intervals(&hits, inside);
        if overflow {
            return Err(format!(
                "RT-FP reference ray at {x:?} exceeded {} visible intervals",
                split::MAX_INTERVALS
            ));
        }
        let scalar = split::remainder_scalar(kernels, *x, *direction, &intervals);
        tensor::add_tensor_term(&mut hessian, direction, crate::G * *weight * scalar);
    }
    Ok(hessian)
}
