fn axis_profile(
    origin: [f64; 3],
    apex: [f64; 3],
    density: &Density,
    mode: DensityMode,
    samples: usize,
) -> Vec<f64> {
    (0..samples)
        .map(|index| {
            density.rho(
                &lerp(origin, apex, (index as f64 + 0.5) / samples as f64),
                mode,
            )
        })
        .collect()
}

fn total_variation(values: &[f64]) -> f64 {
    values.windows(2).map(|pair| (pair[1] - pair[0]).abs()).sum()
}

fn slabs_for(variation: f64, tol: f64, max_slabs: usize, rho_ref: f64) -> usize {
    let scale = rho_ref.abs();
    if variation <= 0.0
        || variation.is_nan()
        || scale <= f64::MIN_POSITIVE
        || max_slabs <= 1
    {
        return 1;
    }
    let wanted = (variation / (tol * scale)).ceil();
    if wanted.is_finite() {
        (wanted.max(1.0) as usize).min(max_slabs)
    } else {
        max_slabs
    }
}

/// Equidistribute density variation over the cone scale `(0, 1]`.
fn slab_boundaries(profile: &[f64], count: usize) -> Vec<f64> {
    let samples = profile.len();
    let mut cumulative = vec![0.0f64; samples];
    for index in 1..samples {
        cumulative[index] =
            cumulative[index - 1] + (profile[index] - profile[index - 1]).abs();
    }
    let variation = cumulative[samples - 1];
    if count <= 1 || variation <= 0.0 || variation.is_nan() {
        return vec![1.0];
    }
    let sample_scale = |index: usize| (index as f64 + 0.5) / samples as f64;
    let gap = 0.5 / samples as f64;
    let mut bounds = Vec::with_capacity(count);
    for slab in 1..=count {
        let target = variation * slab as f64 / count as f64;
        let mut scale = 1.0;
        for index in 1..samples {
            if cumulative[index] >= target {
                let span = cumulative[index] - cumulative[index - 1];
                let fraction = if span > 0.0 {
                    (target - cumulative[index - 1]) / span
                } else {
                    0.0
                };
                scale = sample_scale(index - 1)
                    + fraction * (sample_scale(index) - sample_scale(index - 1));
                break;
            }
        }
        bounds.push(scale);
    }
    let last = bounds.len() - 1;
    bounds[last] = 1.0;
    for index in 0..last {
        let lower = if index == 0 {
            gap
        } else {
            bounds[index - 1] + gap
        };
        let upper = (1.0 - (last - index) as f64 * gap).max(lower);
        bounds[index] = bounds[index].clamp(lower, upper);
    }
    bounds
}

/// Volume mean over a frustum. The cone volume element scales as `s^2 ds`.
fn slab_mean(profile: &[f64], lo: f64, hi: f64) -> f64 {
    let samples = profile.len();
    let gap = 0.5 / samples as f64;
    let (mut numerator, mut denominator) = (0.0, 0.0);
    for (index, density) in profile.iter().enumerate() {
        let scale = (index as f64 + 0.5) / samples as f64;
        let a = (scale - gap).max(lo);
        let b = (scale + gap).min(hi);
        if b <= a {
            continue;
        }
        let weight = (b * b * b - a * a * a) / 3.0;
        numerator += weight * density;
        denominator += weight;
    }
    if denominator > 0.0 {
        numerator / denominator
    } else {
        let middle = 0.5 * (lo + hi);
        profile[(((middle * samples as f64 - 0.5) as isize)
            .clamp(0, samples as isize - 1)) as usize]
    }
}

fn slab_range(profile: &[f64], lo: f64, hi: f64) -> (f64, f64) {
    let samples = profile.len();
    let (mut minimum, mut maximum) = (f64::INFINITY, f64::NEG_INFINITY);
    for (index, density) in profile.iter().enumerate() {
        let scale = (index as f64 + 0.5) / samples as f64;
        if scale >= lo && scale <= hi {
            minimum = minimum.min(*density);
            maximum = maximum.max(*density);
        }
    }
    if minimum.is_finite() {
        (minimum, maximum)
    } else {
        (0.0, 0.0)
    }
}
