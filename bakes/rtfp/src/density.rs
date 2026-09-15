//! Cauchy density model (`assets/density/cauchy.toml`) plus the homogeneous mode.
//!
//! `ρ(y) = Σ_k w_k [1 + σ_k²‖y − c_k‖²]^(−α_k)`, kernels stored in km / km⁻¹ in
//! the TOML and converted to SI here. Weights are rescaled so that ρ(0) matches
//! `bulk_density_ref`, which is what makes the CAUCHY field and the MASCON voxel
//! field describe the same mass distribution.

use serde::Deserialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DensityMode {
    Cauchy,
    Constant,
}

impl DensityMode {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "cauchy" => Ok(Self::Cauchy),
            "constant" => Ok(Self::Constant),
            other => Err(format!("unknown density mode {other:?} (cauchy|constant)")),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct CauchyKernel {
    pub c: [f64; 3],
    pub sigma: f64,
    pub w: f64,
    #[serde(default = "default_alpha")]
    pub alpha: f64,
}

fn default_alpha() -> f64 {
    1.0
}

fn default_bulk() -> f64 {
    1190.0
}

#[derive(Clone, Debug, Deserialize)]
struct CauchyDensityFile {
    #[serde(default = "default_bulk")]
    bulk_density_ref: f64,
    #[serde(default = "default_alpha")]
    alpha_default: f64,
    #[serde(default)]
    total_mass_target: f64,
    kernels: Vec<CauchyKernel>,
}

/// One kernel in SI units (centres in meters, σ in m⁻¹, w in kg/m³).
#[derive(Clone, Copy, Debug)]
pub struct KernelSi {
    pub c: [f64; 3],
    pub sigma: f64,
    pub w: f64,
    pub alpha: f64,
}

#[derive(Clone, Debug)]
pub struct Density {
    kernels: Vec<KernelSi>,
    pub bulk_density: f64,
}

impl Density {
    pub fn from_toml(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let raw: CauchyDensityFile =
            toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let alpha_default = raw.alpha_default;
        let mut kernels: Vec<KernelSi> = raw
            .kernels
            .iter()
            .map(|k| KernelSi {
                c: [k.c[0] * 1000.0, k.c[1] * 1000.0, k.c[2] * 1000.0],
                sigma: k.sigma / 1000.0,
                w: k.w,
                alpha: if k.alpha > 0.0 { k.alpha } else { alpha_default },
            })
            .collect();
        let bulk_density = if raw.bulk_density_ref > 0.0 { raw.bulk_density_ref } else { 1190.0 };
        let total_mass_target = raw.total_mass_target.max(0.0);
        if total_mass_target > 0.0 {
            let rho0 = densitize(&kernels, &[0.0, 0.0, 0.0]);
            if rho0.abs() > 1e-30 {
                let scale = bulk_density / rho0;
                for k in &mut kernels {
                    k.w *= scale;
                }
            }
        }
        if let Some(bad) = kernels.iter().find(|k| (k.alpha - 1.0).abs() > 1e-9) {
            return Err(format!(
                "kernel alpha={} unsupported: the closed-form radial integral is α=1 only \
                 (see 推导 七, 一般 α 需要径向求积)",
                bad.alpha
            ));
        }
        Ok(Self { kernels, bulk_density })
    }

    /// One embedded-constant kernel — used for the homogeneous mode so that the
    /// analytic split below covers both modes with a single code path.
    pub fn homogeneous(bulk_density: f64) -> Self {
        Self { kernels: Vec::new(), bulk_density }
    }

    pub fn kernels(&self) -> &[KernelSi] {
        &self.kernels
    }

    /// Density of the *model* at `x` (meters). For the homogeneous mode this is
    /// the bulk density everywhere, which is exactly the value the analytic
    /// uniform-density term has to carry (the mesh boundary, not ρ, cuts the body).
    pub fn rho(&self, x: &[f64; 3], mode: DensityMode) -> f64 {
        match mode {
            DensityMode::Cauchy => densitize(&self.kernels, x),
            DensityMode::Constant => self.bulk_density,
        }
    }
}

pub fn densitize(kernels: &[KernelSi], x: &[f64; 3]) -> f64 {
    let mut rho = 0.0;
    for k in kernels {
        let dx = x[0] - k.c[0];
        let dy = x[1] - k.c[1];
        let dz = x[2] - k.c[2];
        let base = 1.0 + k.sigma * k.sigma * (dx * dx + dy * dy + dz * dz);
        rho += if (k.alpha - 1.0).abs() < 1e-9 {
            k.w / base
        } else {
            k.w * base.powf(-k.alpha)
        };
    }
    rho
}
