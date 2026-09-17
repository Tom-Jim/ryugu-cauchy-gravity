//! RT-FP bake: ray-traced finite-part Hessian with an analytic near-field split.
//!
//! ```text
//!   H(x) = ρ(x) · W(x)  +  G Σ_q ω_q T(u_q) Σ_k w_k R_k(u_q)
//!          ^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//!          exact uniform-   smooth remainder, 288 directions
//!          density tensor
//! ```
//!
//! `W` is the polyhedral gravity-gradient tensor of the mesh at `x` in closed
//! form (surface integral over the 196 k faces); `R_k` is the radial finite-part
//! integral with its logarithmic near-field part removed. See `split.rs` for the
//! derivation, `gpu.rs` for the three WGSL pipelines that do all of the work, and
//! `--selftest` for the checks that pin the split down against the ESA library.

#[path = "../compute/native/mod.rs"]
mod native;

use native::{
    analytic, bvh, carlson, carlson_alpha, checkpoint, density, esa, geom, gpu, mass, mesh,
    quadrature, record, split, tensor,
};

use density::{Density, DensityMode, Normalization};
use geom::BruteTracer;
use mesh::Mesh;
use std::path::PathBuf;
use std::process::ExitCode;
use tensor::{Sym6, frobenius};

/// Gravitational constant (CODATA 2018), shared by every native solver.
pub const G: f64 = 6.674_30e-11;
/// Finite-part reference length; cancels against Σω_q T(u_q) = 0 (see split.rs).
pub const ELL0_M: f64 = 1.0;
const KM_TO_M: f64 = 1000.0;
/// Ray start offset above the observation point.
const GEOM_EPS_M: f64 = 1e-9;
const DIRS_DEFAULT: usize = 288;
/// Observation-surface bounds, shared with the C++ bakes: mascon's record lives
/// up to 16 m above the surface, and the viewer's slider has to reach it.
const STANDOFF_MAX_MM: f64 = 32000.0;
/// Observation points per GPU block. Smaller submissions keep the desktop
/// compositor responsive while the WGSL pipelines are saturated.
// Keep one GPU submission short enough that the browser/compositor and
// progress poller continue to run. A 512-point Carlson block can monopolise
// the Metal queue for over a minute on large jump-surface scenes.
const POINTS_PER_BLOCK: usize = 32;
const CHECKPOINT_BLOCKS: usize = 32;

/// Which solver produces the face scalars.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Solver {
    /// `H = ρ(x)·W(x) + G Σω T Σw R_k` — analytic polyhedral near field plus a
    /// ray-traced directional quadrature.
    Ray,
    /// `H_ij = G Σ_F Δρ_F n_j I_F[i]` over the star-cone jump surfaces.
    Carlson,
    /// General-alpha radial finite part with the Carlson symmetric-function
    /// verification backend. Unlike [`Self::Ray`], this path accepts non-unit
    /// Cauchy exponents.
    CarlsonAlpha,
}

impl Solver {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "ray" | "rtfp" => Ok(Self::Ray),
            "carlson" => Ok(Self::Carlson),
            "carlson-alpha" | "carlsonalpha" => Ok(Self::CarlsonAlpha),
            other => Err(format!(
                "unknown solver {other:?} (ray|carlson|carlson-alpha)"
            )),
        }
    }
}

struct Args {
    mesh: PathBuf,
    out: PathBuf,
    order: PathBuf,
    checkpoint: PathBuf,
    density: PathBuf,
    mode: DensityMode,
    normalize: Normalization,
    solver: Solver,
    standoff_mm: f64,
    directions: usize,
    resume: bool,
    selftest: bool,
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn runtime_path(name: &str) -> PathBuf {
    std::env::temp_dir()
        .join("ryugu-cauchy-gravity-runtime")
        .join(name)
}

fn parse_args() -> Result<Args, String> {
    let r = root();
    let mut args = Args {
        mesh: r.join("assets/models/ryugu.glb"),
        out: runtime_path("rtfp-cauchy.bin"),
        order: runtime_path(".rtfp-cauchy-order"),
        checkpoint: runtime_path(".rtfp-cauchy-vertex-checkpoint"),
        density: r.join("assets/density/cauchy.toml"),
        mode: DensityMode::Cauchy,
        // The TOML asks for the Ryugu mass scale, so that is the default; the
        // old ρ(0) convention stays reachable for reproducing older records.
        normalize: Normalization::TotalMass,
        solver: Solver::Ray,
        // Same start height as the other two bakes: comparable records out of the box.
        standoff_mm: 16000.0,
        directions: DIRS_DEFAULT,
        resume: false,
        selftest: false,
    };
    let mut it = std::env::args().skip(1);
    // Track explicit values so solver-specific defaults never overwrite them.
    let (mut out_set, mut order_set, mut checkpoint_set, mut normalize_set) =
        (false, false, false, false);
    while let Some(a) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{a} needs a value"));
        match a.as_str() {
            "--mesh" => args.mesh = PathBuf::from(value()?),
            "--out" => {
                args.out = PathBuf::from(value()?);
                out_set = true;
            }
            "--order" => {
                args.order = PathBuf::from(value()?);
                order_set = true;
            }
            "--checkpoint" => {
                args.checkpoint = PathBuf::from(value()?);
                checkpoint_set = true;
            }
            "--density" => args.density = PathBuf::from(value()?),
            "--mode" => args.mode = DensityMode::parse(&value()?)?,
            "--normalize" => {
                args.normalize = Normalization::parse(&value()?)?;
                normalize_set = true;
            }
            "--solver" => args.solver = Solver::parse(&value()?)?,
            "--directions" => {
                args.directions = value()?.parse().map_err(|e| format!("--directions: {e}"))?
            }
            "--standoff-mm" => {
                let mm: f64 = value()?
                    .parse()
                    .map_err(|e| format!("--standoff-mm: {e}"))?;
                if !(1.0..=STANDOFF_MAX_MM).contains(&mm) {
                    return Err(format!("--standoff-mm {mm} outside 1..{STANDOFF_MAX_MM}"));
                }
                args.standoff_mm = mm;
            }
            "--resume" => args.resume = true,
            "--selftest" => args.selftest = true,
            "--probe-gpu" => {
                return match gpu::Device::new() {
                    Ok(dev) => {
                        println!("GPU adapter: {}", dev.name());
                        std::process::exit(0);
                    }
                    Err(e) => Err(e),
                };
            }
            "--help" | "-h" => {
                println!(
                    "rtfp-bake --mesh assets/models/ryugu.glb [--out path] [--order path] [--checkpoint path]\n\
                     \x20         [--density toml] [--mode cauchy|elliptic|constant] [--normalize rho0|total_mass|raw]\n\
                     \x20         [--solver ray|carlson|carlson-alpha] [--standoff-mm 1..32000] [--directions 288]\n\
                     \x20         [--resume] [--selftest]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if args.solver == Solver::CarlsonAlpha && args.mode == DensityMode::Elliptic && !normalize_set {
        args.normalize = Normalization::Raw;
    }

    let prefix = match args.solver {
        Solver::Ray => format!("rtfp-{}", args.mode.tag()),
        Solver::Carlson => format!("carlson-{}", args.mode.tag()),
        Solver::CarlsonAlpha => format!("carlsonalpha-{}", args.mode.tag()),
    };
    if !out_set {
        args.out = runtime_path(&format!("{prefix}.bin"));
    }
    if !order_set {
        args.order = runtime_path(&format!(".{prefix}-order"));
    }
    if !checkpoint_set {
        args.checkpoint = runtime_path(&format!(".{prefix}-vertex-checkpoint"));
    }
    Ok(args)
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let result = if args.selftest {
        selftest(&args)
    } else {
        match args.solver {
            Solver::Ray => run_rtfp(&args),
            Solver::Carlson => run_carlson(&args),
            Solver::CarlsonAlpha => run_carlson_alpha(&args),
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("rtfp-bake: {e}");
            ExitCode::FAILURE
        }
    }
}

// Unit cube (12 triangles) used as the closed-form test body.

include!("rtfp_bake/selftest_core.rs");
include!("rtfp_bake/selftest_alpha.rs");
include!("rtfp_bake/selftest_carlson.rs");
include!("rtfp_bake/selftest_surface.rs");
include!("rtfp_bake/selftest_mass.rs");
include!("rtfp_bake/execute.rs");
