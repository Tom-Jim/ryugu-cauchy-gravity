#[derive(Clone, Copy, Debug, Deserialize)]
struct Kernel {
    c: Vec3,
    sigma: f64,
    w: f64,
    alpha: f64,
}

#[derive(Deserialize)]
struct DensityFile {
    #[serde(default = "one")]
    alpha_default: f64,
    #[serde(default)]
    total_mass_target: f64,
    kernels: Vec<KernelEntry>,
}

struct ParsedDensity {
    kernels: Vec<Kernel>,
    total_mass_target: f64,
}

#[derive(Deserialize)]
struct KernelEntry {
    c: [f64; 3],
    sigma: f64,
    w: f64,
    #[serde(default)]
    alpha: Option<f64>,
}

fn one() -> f64 {
    1.0
}

fn parse_density_text(text: &str, label: &str) -> Result<ParsedDensity, String> {
    let file: DensityFile = toml::from_str(text).map_err(|error| format!("{label}: {error}"))?;
    let kernels = file
        .kernels
        .iter()
        .map(|entry| Kernel {
            c: [
                entry.c[0] * KM_TO_M,
                entry.c[1] * KM_TO_M,
                entry.c[2] * KM_TO_M,
            ],
            sigma: entry.sigma / KM_TO_M,
            w: entry.w,
            alpha: entry.alpha.unwrap_or(file.alpha_default),
        })
        .collect();
    Ok(ParsedDensity {
        kernels,
        total_mass_target: file.total_mass_target.max(0.0),
    })
}

fn density_at(kernels: &[Kernel], point: Vec3) -> f64 {
    let mut density = 0.0;
    for kernel in kernels {
        let d = sub(point, kernel.c);
        density += kernel.w / (1.0 + kernel.sigma * kernel.sigma * dot(d, d)).powf(kernel.alpha);
    }
    density
}

/// `(t - atan t) / t³`, with the series that keeps it finite at the origin.
fn radial_g(t: f64) -> f64 {
    if t.abs() >= 0.1 {
        return (t - t.atan()) / (t * t * t);
    }
    let t2 = t * t;
    let mut term = 1.0 / 3.0;
    let mut total = term;
    for k in 1..10 {
        let k = k as f64;
        term *= -t2 * (2.0 * k + 1.0) / (2.0 * k + 3.0);
        total += term;
    }
    total
}

/// Dunant 7-point rule in barycentric coordinates.
const DUNANT7: [([f64; 3], f64); 7] = [
    ([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], 0.225),
    (
        [0.05971587178976981, 0.4701420641051151, 0.4701420641051151],
        0.1323941527885062,
    ),
    (
        [0.4701420641051151, 0.05971587178976981, 0.4701420641051151],
        0.1323941527885062,
    ),
    (
        [0.4701420641051151, 0.4701420641051151, 0.05971587178976981],
        0.1323941527885062,
    ),
    (
        [0.7974269853530873, 0.10128650732345634, 0.10128650732345634],
        0.12593918054482713,
    ),
    (
        [0.10128650732345634, 0.7974269853530873, 0.10128650732345634],
        0.12593918054482713,
    ),
    (
        [0.10128650732345634, 0.10128650732345634, 0.7974269853530873],
        0.12593918054482713,
    ),
];

fn triangle_point(triangle: &Triangle, bary: [f64; 3]) -> Vec3 {
    [
        bary[0] * triangle.a[0] + bary[1] * triangle.b[0] + bary[2] * triangle.c[0],
        bary[0] * triangle.a[1] + bary[1] * triangle.b[1] + bary[2] * triangle.c[1],
        bary[0] * triangle.a[2] + bary[1] * triangle.b[2] + bary[2] * triangle.c[2],
    ]
}

/// Enclosed volume of the Cauchy kernel field, by the divergence theorem.
fn kernel_volume(triangles: &[Triangle], center: Vec3, sigma: f64) -> f64 {
    let mut total = 0.0;
    for triangle in triangles {
        let raw_normal = cross(sub(triangle.b, triangle.a), sub(triangle.c, triangle.b));
        for (bary, weight) in DUNANT7 {
            let y = triangle_point(triangle, bary);
            total += 0.5
                * weight
                * radial_g(sigma * norm(sub(y, center)))
                * dot(sub(y, center), raw_normal);
        }
    }
    total
}

fn normalize_kernels(
    triangles: &[Triangle],
    kernels: &[Kernel],
    target_mass: f64,
) -> Result<Vec<Kernel>, String> {
    if kernels
        .iter()
        .any(|kernel| (kernel.alpha - 1.0).abs() > 1e-9)
    {
        return Err(
            "Cauchy total-mass normalization requires alpha=1 kernels; use CarlsonAlpha or Mascon for general alpha"
                .into(),
        );
    }
    let raw_mass: f64 = kernels
        .iter()
        .map(|kernel| kernel.w * kernel_volume(triangles, kernel.c, kernel.sigma))
        .sum();
    let scale_factor = if target_mass > 0.0 && raw_mass.abs() > 1e-30 {
        target_mass / raw_mass
    } else {
        1.0
    };
    Ok(kernels
        .iter()
        .map(|kernel| Kernel {
            w: kernel.w * scale_factor,
            ..*kernel
        })
        .collect())
}
