//! Cauchy density model (`assets/density/cauchy.toml`) plus the homogeneous mode.
//!
//! `ρ(y) = Σ_k w_k [1 + σ_k²‖y − c_k‖²]^(−α_k)`, kernels stored in km / km⁻¹ in
//! the TOML and converted to SI here.
//!
//! Two normalisation conventions are possible and they are *not* the same field:
//!
//! * [`Normalization::Rho0`] — scale the kernel weights so that `ρ(0)` equals
//!   `bulk_density_ref` (1190 kg/m³). This is what the bake used to do.
//! * [`Normalization::TotalMass`] — scale them so that `∫_V ρ dV` equals
//!   `total_mass_target` (4.5e11 kg, the Ryugu mass scale the TOML intends).
//
// Only the second one puts the Cauchy field on the same mass budget as the
// mascon voxel field, which also normalises to `total_mass_target`. The two
// differ by ≈ +0.30 % (mascon higher), and that difference is a constant
// multiplier on every face, so it used to look like a solver disagreement when
// it was really just a bookkeeping choice. `mass.rs` supplies the volume
// integral that makes the total-mass convention available.

use serde::Deserialize;

use crate::mass;
use crate::mesh::Mesh;

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

    /// Short suffix used in record file names (`carlson_cauchy_faces.bin`).
    pub fn tag(self) -> &'static str {
        match self {
            Self::Cauchy => "cauchy",
            Self::Constant => "constant",
        }
    }
}

/// Which scalar the kernel weights are rescaled to match.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Normalization {
    #[default]
    Rho0,
    TotalMass,
}

impl Normalization {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "rho0" => Ok(Self::Rho0),
            "total_mass" | "total-mass" => Ok(Self::TotalMass),
            other => Err(format!("unknown normalization {other:?} (rho0|total_mass)")),
        }
    }
}

/// What the two conventions would have produced, so a bake can print the
/// bookkeeping difference instead of leaving it to be discovered as a solver gap.
#[derive(Clone, Copy, Debug, Default)]
pub struct MassReport {
    /// `∫_V ρ dV` with the TOML weights exactly as written.
    pub raw_mass: f64,
    /// Weight multiplier that realises the `ρ(0) = bulk_density_ref` convention.
    pub rho0_scale: f64,
    /// `∫_V ρ dV` under that convention.
    pub rho0_mass: f64,
    /// Weight multiplier actually applied (1.0 for the homogeneous mode).
    pub applied_scale: f64,
    /// `∫_V ρ dV` after the applied scale.
    pub applied_mass: f64,
    /// TOML `total_mass_target`, 0 when absent.
    pub target_mass: f64,
    /// Enclosed volume of the mesh.
    pub volume: f64,
    /// Convention the bake ran under.
    pub normalization: Normalization,
}

impl MassReport {
    /// `applied_mass / rho0_mass` — the constant factor the normalisation puts
    /// between this record and one baked with the old `ρ(0)` convention.
    pub fn scale_vs_rho0(&self) -> f64 {
        if self.rho0_mass.abs() > 1e-30 {
            self.applied_mass / self.rho0_mass
        } else {
            1.0
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
    /// Bookkeeping for the chosen normalisation (see [`MassReport`]).
    pub mass: MassReport,
}

impl Density {
    pub fn from_toml(
        path: &std::path::Path,
        mesh: &Mesh,
        normalization: Normalization,
    ) -> Result<Self, String> {
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
                alpha: if k.alpha > 0.0 {
                    k.alpha
                } else {
                    alpha_default
                },
            })
            .collect();
        let bulk_density = if raw.bulk_density_ref > 0.0 {
            raw.bulk_density_ref
        } else {
            1190.0
        };
        let total_mass_target = raw.total_mass_target.max(0.0);
        if let Some(bad) = kernels.iter().find(|k| (k.alpha - 1.0).abs() > 1e-9) {
            return Err(format!(
                "kernel alpha={} unsupported: the closed-form radial integral is α=1 only \
                 (general α would require a radial quadrature rule)",
                bad.alpha
            ));
        }

        // The volume integral is linear in the weights, so one pass with the raw
        // TOML weights gives both conventions at once.
        let raw_mass = mass::kernel_field_mass(mesh, &kernels);
        let rho0 = densitize(&kernels, &[0.0, 0.0, 0.0]);
        let rho0_scale = if rho0.abs() > 1e-30 {
            bulk_density / rho0
        } else {
            1.0
        };
        let applied_scale = match normalization {
            Normalization::Rho0 => rho0_scale,
            Normalization::TotalMass if total_mass_target > 0.0 && raw_mass.abs() > 1e-30 => {
                total_mass_target / raw_mass
            }
            // No target in the TOML: fall back to the ρ(0) convention.
            Normalization::TotalMass => rho0_scale,
        };
        for k in &mut kernels {
            k.w *= applied_scale;
        }
        let report = MassReport {
            raw_mass,
            rho0_scale,
            rho0_mass: raw_mass * rho0_scale,
            applied_scale,
            applied_mass: raw_mass * applied_scale,
            target_mass: total_mass_target,
            volume: mass::body_volume(mesh),
            normalization,
        };
        Ok(Self {
            kernels,
            bulk_density,
            mass: report,
        })
    }

    /// One embedded-constant kernel — used for the homogeneous mode so that the
    /// analytic split below covers both modes with a single code path.
    pub fn homogeneous(bulk_density: f64, mesh: &Mesh) -> Self {
        let volume = mass::body_volume(mesh);
        let report = MassReport {
            raw_mass: bulk_density * volume,
            rho0_scale: 1.0,
            rho0_mass: bulk_density * volume,
            applied_scale: 1.0,
            applied_mass: bulk_density * volume,
            target_mass: 0.0,
            volume,
            normalization: Normalization::Rho0,
        };
        Self {
            kernels: Vec::new(),
            bulk_density,
            mass: report,
        }
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
