#[derive(Clone, Copy, Debug, Deserialize)]
pub(crate) struct Kernel {
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
    #[serde(default)]
    mean_density: Option<f64>,
    #[serde(default)]
    bulk_density_ref: Option<f64>,
    kernels: Vec<KernelEntry>,
}

pub(crate) struct ParsedDensity {
    pub(crate) kernels: Vec<Kernel>,
    pub(crate) total_mass_target: f64,
    pub(crate) mean_density_target: Option<f64>,
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

pub(crate) fn parse_density_text(text: &str, label: &str) -> Result<ParsedDensity, String> {
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
    let mean_density_target = file
        .mean_density
        .or(file.bulk_density_ref)
        .filter(|v| *v > 0.0);
    Ok(ParsedDensity {
        kernels,
        total_mass_target: file.total_mass_target.max(0.0),
        mean_density_target,
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

fn mesh_enclosed_volume(triangles: &[Triangle]) -> f64 {
    let mut total = 0.0;
    for t in triangles {
        let n_raw = cross(sub(t.b, t.a), sub(t.c, t.b));
        let centroid = [
            (t.a[0] + t.b[0] + t.c[0]) / 3.0,
            (t.a[1] + t.b[1] + t.c[1]) / 3.0,
            (t.a[2] + t.b[2] + t.c[2]) / 3.0,
        ];
        total += dot(centroid, n_raw) / 6.0;
    }
    total.abs()
}

fn calibrate_kernels(
    triangles: &[Triangle],
    density: &ParsedDensity,
) -> Result<Vec<Kernel>, String> {
    if density.kernels.is_empty() {
        return Ok(Vec::new());
    }

    let max_r = triangles
        .iter()
        .flat_map(|t| [t.a, t.b, t.c])
        .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
        .fold(0.0f64, f64::max);

    // Reference bounding radius for Ryugu model in ryugu.glb is ~528.5 m
    const RYUGU_REF_RADIUS_M: f64 = 528.5;
    let s_geom = if max_r > 10.0 {
        max_r / RYUGU_REF_RADIUS_M
    } else {
        1.0
    };

    let scaled_kernels: Vec<Kernel> = density
        .kernels
        .iter()
        .map(|kernel| Kernel {
            c: [
                kernel.c[0] * s_geom,
                kernel.c[1] * s_geom,
                kernel.c[2] * s_geom,
            ],
            sigma: kernel.sigma / s_geom,
            w: kernel.w,
            alpha: kernel.alpha,
        })
        .collect();

    let target_mass = if let Some(mean_rho) = density.mean_density_target {
        let vol = mesh_enclosed_volume(triangles);
        mean_rho * vol
    } else if density.total_mass_target > 0.0 {
        density.total_mass_target
    } else {
        0.0
    };

    if target_mass <= 0.0 {
        return Ok(scaled_kernels);
    }

    normalize_kernels(triangles, &scaled_kernels, target_mass)
}
