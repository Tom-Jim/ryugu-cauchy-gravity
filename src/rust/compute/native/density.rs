//! Arbitrary density model (`assets/density/cauchy.toml`) plus the homogeneous mode.
//!
//! `ρ(y) = Σ_k w_k [1 + σ_k²‖y − c_k‖²]^(−α_k)`, kernels stored in km / km⁻¹ in
//! the TOML and converted to SI here. Non-unit exponents are adaptively
//! decomposed into a finite Cauchy mixture (`α = 1`) before any RT-FP/Carlson
//! pipeline consumes the kernels; this gives all three solvers one compatible
//! representation while retaining the original arbitrary-density TOML input.
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
    /// Non-unit Cauchy exponents. The legacy jump-surface Carlson solver and
    /// RT-FP's closed-form radial remainder deliberately do not consume this
    /// model; CarlsonAlpha evaluates the general radial finite part instead.
    Elliptic,
    Constant,
}

impl DensityMode {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "cauchy" => Ok(Self::Cauchy),
            "elliptic" => Ok(Self::Elliptic),
            "constant" => Ok(Self::Constant),
            other => Err(format!(
                "unknown density mode {other:?} (cauchy|elliptic|constant)"
            )),
        }
    }

    /// Short suffix used in record file names (`carlson_cauchy_faces.bin`).
    pub fn tag(self) -> &'static str {
        match self {
            Self::Cauchy => "cauchy",
            Self::Elliptic => "elliptic",
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
    /// Keep the TOML kernel weights unchanged.
    Raw,
}

impl Normalization {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "rho0" => Ok(Self::Rho0),
            "total_mass" | "total-mass" => Ok(Self::TotalMass),
            "raw" => Ok(Self::Raw),
            other => Err(format!(
                "unknown normalization {other:?} (rho0|total_mass|raw)"
            )),
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
    #[serde(rename = "type", default)]
    kind: Option<String>,
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

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct Density {
    kernels: Vec<KernelSi>,
    pub bulk_density: f64,
    /// Bookkeeping for the chosen normalisation (see [`MassReport`]).
    pub mass: MassReport,
    /// Error and size of the adaptive α→1 Cauchy decomposition.
    pub decomposition: DecompositionReport,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DecompositionReport {
    pub original_kernels: usize,
    pub expanded_kernels: usize,
    /// Worst sampled deviation, as a fraction of the kernel's own peak density.
    ///
    /// A local relative error is meaningless over this range: the samples reach
    /// `u = sigma * body_radius`, where the source density has already decayed
    /// to ~1e-8 of its peak, so an absolute error of 1e-8 out there reports as
    /// 100 %. Referencing the peak keeps the number tied to the part of the
    /// field that actually carries mass.
    pub max_peak_error: f64,
}

impl Density {
    /// Load an arbitrary-density file, expand it into the `alpha = 1` Cauchy
    /// mixture the solvers consume, and apply the requested weight convention.
    ///
    /// The mode a caller later observes the field through
    /// ([`Density::rho`]) does not change this load path: the decomposition has
    /// already left every kernel at `alpha = 1`, so one load serves Cauchy,
    /// Elliptic and the RT-FP/Carlson mixture assets alike.
    pub fn from_toml(
        path: &std::path::Path,
        mesh: &Mesh,
        normalization: Normalization,
    ) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let raw: CauchyDensityFile =
            toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        if let Some(kind) = raw.kind.as_deref() {
            let known = matches!(
                kind,
                "cauchy_kernels" | "arbitrary_density" | "arbitrary_density_cauchy_mixture"
            );
            if !known {
                return Err(format!(
                    "{}: unsupported density type {kind:?}; expected arbitrary_density",
                    path.display()
                ));
            }
        }
        let alpha_default = raw.alpha_default;
        let original_kernels: Vec<KernelSi> = raw
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
        let radius = crate::body_radius(mesh);
        let (mut kernels, decomposition) = adaptive_cauchy_decompose(&original_kernels, radius);
        let bulk_density = if raw.bulk_density_ref > 0.0 {
            raw.bulk_density_ref
        } else {
            1190.0
        };
        let total_mass_target = raw.total_mass_target.max(0.0);
        // The volume integral is linear in the weights, so one pass gives both
        // conventions at once. It runs on the *decomposed* kernels: they are all
        // α = 1, which is exactly the case `mass::kernel_field_mass`'s closed
        // form covers. Skipping it for a fractional file (as an earlier revision
        // did) left `raw_mass` at zero, which silently turned
        // `--normalize total_mass` into the ρ(0) convention.
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
            Normalization::Raw => 1.0,
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
            decomposition,
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
            decomposition: DecompositionReport::default(),
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
            DensityMode::Cauchy | DensityMode::Elliptic => densitize(&self.kernels, x),
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

/// Replace every non-unit exponent by a deterministic, adaptive finite sum of
/// unit-exponent Cauchy kernels. The fit is radial around each kernel centre and
/// uses logarithmic samples over the body radius, so narrow and broad kernels
/// receive the same relative accuracy. Signed source weights remain signed;
/// this is required for the TOML's deficit/compensation terms.
fn adaptive_cauchy_decompose(
    source: &[KernelSi],
    body_radius: f64,
) -> (Vec<KernelSi>, DecompositionReport) {
    let mut out = Vec::new();
    let mut max_error = 0.0_f64;
    let radius = body_radius.max(1.0);
    for kernel in source {
        let delta = (kernel.alpha - 1.0).abs();
        if delta <= 1e-9 {
            out.push(*kernel);
            continue;
        }
        let terms = (8.0 + 5.0 * delta.ceil()).clamp(8.0, 16.0) as usize;
        let umax = (kernel.sigma * radius).max(1.0);
        let mut scales = Vec::with_capacity(terms);
        for i in 0..terms {
            let t = if terms == 1 {
                0.5
            } else {
                i as f64 / (terms - 1) as f64
            };
            // Cover two decades around the source scale; the fit is in u=σr.
            scales.push(10.0_f64.powf(-1.0 + 2.0 * t));
        }
        let samples = (terms * 3).max(24);
        let mut normal = vec![vec![0.0_f64; terms + 1]; terms];
        for j in 0..samples {
            let t = j as f64 / (samples - 1) as f64;
            let u = if j == 0 {
                0.0
            } else {
                10.0_f64.powf(-4.0 + (umax.log10() + 4.0) * t)
            };
            let target = (1.0 + u * u).powf(-kernel.alpha);
            let basis: Vec<f64> = scales
                .iter()
                .map(|scale| 1.0 / (1.0 + (scale * u) * (scale * u)))
                .collect();
            for (a, row) in normal.iter_mut().enumerate().take(terms) {
                let wa = basis[a];
                for (b, cell) in row.iter_mut().enumerate().take(terms) {
                    *cell += wa * basis[b];
                }
                row[terms] += wa * target;
            }
        }
        for (i, row) in normal.iter_mut().enumerate().take(terms) {
            row[i] += 1e-12;
        }
        let coefficients = solve_linear_system(&mut normal).unwrap_or_else(|| {
            let mut fallback = vec![0.0; terms];
            fallback[terms / 2] = 1.0;
            fallback
        });
        // `u = 0` is the kernel peak, and the target is normalised to it.
        let peak = 1.0_f64;
        let mut max_abs_error = 0.0_f64;
        for j in 0..samples {
            let t = j as f64 / (samples - 1) as f64;
            let u = if j == 0 {
                0.0
            } else {
                10.0_f64.powf(-4.0 + (umax.log10() + 4.0) * t)
            };
            let target = (1.0 + u * u).powf(-kernel.alpha);
            let approx = scales
                .iter()
                .zip(&coefficients)
                .map(|(scale, coefficient)| coefficient / (1.0 + (scale * u).powi(2)))
                .sum::<f64>();
            max_abs_error = max_abs_error.max((approx - target).abs());
        }
        max_error = max_error.max(max_abs_error / peak);
        for (scale, coefficient) in scales.into_iter().zip(coefficients) {
            if coefficient.abs() > 1e-12 {
                out.push(KernelSi {
                    c: kernel.c,
                    sigma: kernel.sigma * scale,
                    w: kernel.w * coefficient,
                    alpha: 1.0,
                });
            }
        }
    }
    let report = DecompositionReport {
        original_kernels: source.len(),
        expanded_kernels: out.len(),
        max_peak_error: max_error,
    };
    (out, report)
}

fn solve_linear_system(matrix: &mut [Vec<f64>]) -> Option<Vec<f64>> {
    let n = matrix.len();
    for col in 0..n {
        let (pivot, value) = (col..n)
            .map(|row| (row, matrix[row][col].abs()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))?;
        if value <= 1e-18 {
            return None;
        }
        matrix.swap(col, pivot);
        let divisor = matrix[col][col];
        for value in matrix[col].iter_mut().skip(col) {
            *value /= divisor;
        }
        // One clone per column keeps the row elimination borrow-checker clean
        // without splitting the matrix; `n` is bounded by the kernel count.
        let pivot_row = matrix[col].clone();
        for (row_index, row) in matrix.iter_mut().enumerate() {
            if row_index == col {
                continue;
            }
            let factor = row[col];
            if factor.abs() <= 1e-30 {
                continue;
            }
            for (j, value) in row.iter_mut().enumerate().skip(col) {
                *value -= factor * pivot_row[j];
            }
        }
    }
    Some((0..n).map(|row| matrix[row][n]).collect())
}
