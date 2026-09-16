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

mod analytic;
mod bvh;
mod carlson;
mod carlson_alpha;
mod checkpoint;
mod density;
mod esa;
mod geom;
mod gpu;
mod mass;
mod mesh;
mod quadrature;
mod record;
mod split;
mod tensor;

use density::{Density, DensityMode, Normalization};
use geom::BruteTracer;
use mesh::Mesh;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use tensor::{frobenius, Sym6};

/// Gravitational constant (CODATA 2018), matching the C++ bakes in `bakes/`.
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
    obj: PathBuf,
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
    // <project>/bakes/rtfp -> <project>
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crate lives in <project>/bakes/rtfp")
        .to_path_buf()
}

fn runtime_path(name: &str) -> PathBuf {
    std::env::temp_dir()
        .join("ryugu-cauchy-gravity-live")
        .join(name)
}

fn parse_args() -> Result<Args, String> {
    let r = root();
    let mut args = Args {
        obj: r.join("../Ryugu_wasm/assets/models/SHAPE_SFM_200k_v20180804.obj"),
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
    // Tracked so the Carlson defaults below never override an explicit path.
    let (mut out_set, mut order_set, mut checkpoint_set, mut normalize_set) =
        (false, false, false, false);
    while let Some(a) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{a} needs a value"));
        match a.as_str() {
            "--obj" => args.obj = PathBuf::from(value()?),
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
                }
            }
            "--help" | "-h" => {
                println!(
                    "rtfp-bake --obj mesh.obj [--out path] [--order path] [--checkpoint path]\n\
                     \x20         [--density toml] [--mode cauchy|elliptic|constant] [--normalize rho0|total_mass|raw]\n\
                     \x20         [--solver ray|carlson|carlson-alpha] [--standoff-mm 1..32000] [--directions 288]\n\
                     \x20         [--resume] [--selftest]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    // A Carlson run must not clobber the RT-FP record, so give it its own paths
    // unless the caller asked for something specific.
    if args.solver == Solver::Carlson {
        if !out_set {
            args.out = runtime_path(&format!("carlson-{}.bin", args.mode.tag()));
        }
        if !order_set {
            args.order = runtime_path(&format!(".carlson-{}-order", args.mode.tag()));
        }
        if !checkpoint_set {
            args.checkpoint =
                runtime_path(&format!(".carlson-{}-vertex-checkpoint", args.mode.tag()));
        }
        // The Carlson face list depends on the density field, so there is no
        // standoff-replay cache on this path.
    }
    if args.solver == Solver::CarlsonAlpha {
        if !out_set {
            args.out = runtime_path(&format!("carlsonalpha-{}.bin", args.mode.tag()));
        }
        if !order_set {
            args.order = runtime_path(&format!(".carlsonalpha-{}-order", args.mode.tag()));
        }
        if !checkpoint_set {
            args.checkpoint = runtime_path(&format!(
                ".carlsonalpha-{}-vertex-checkpoint",
                args.mode.tag()
            ));
        }
        // The fractional-Cauchy TOML carries raw weights and Mascon keeps that
        // scale when total_mass_target is zero. Match it unless the caller
        // explicitly requested another convention.
        if args.mode == DensityMode::Elliptic && !normalize_set {
            args.normalize = Normalization::Raw;
        }
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
            Solver::Ray => run(&args),
            Solver::Carlson => run_carlson(&args),
            Solver::CarlsonAlpha => run_raylike(&args, true),
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

/// Unit cube (12 triangles) used as the closed-form test body.
fn unit_cube() -> Mesh {
    let mut mesh = Mesh::default();
    for &(x, y, z) in &[
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ] {
        mesh.xyz.extend_from_slice(&[x, y, z]);
    }
    for f in [
        [0u32, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [2, 3, 7],
        [2, 7, 6],
        [1, 2, 6],
        [1, 6, 5],
        [3, 0, 4],
        [3, 4, 7],
    ] {
        mesh.faces.extend_from_slice(&f);
    }
    mesh
}

/// Checks that pin the two halves of the split down against trusted references:
/// the ESA polyhedral library (closed form), a direct point-mass sum (the same
/// body finely subdivided), and the CPU reference tracer for the WGSL pipelines.
fn selftest(args: &Args) -> Result<(), String> {
    let side = 2.0f64;
    let mesh = unit_cube();
    let p = [0.0, 0.0, 1.0 + 0.35];
    let ev = esa::Evaluable::new(&mesh, 1.0)?;
    let cube_faces = analytic::precompute(&mesh);
    for (label, q) in [
        ("far field", p),
        ("1 mm above face centre", [0.0, 0.0, 1.0 + 1e-3]),
        ("1 mm above edge", [1.0, 0.0, 1.0 + 1e-3]),
        // Exactly above the corner the point sits *in* the planes of two faces,
        // where the surface integral itself diverges; a hair to the side is the
        // same near-field test without the degeneracy.
        ("1 mm above corner vertex", [1.0001, 1.0001, 1.0 + 1e-3]),
        ("1 mm above face corner", [0.5, 0.5, 1.0 + 1e-3]),
    ] {
        let mine = analytic::hessian6(&cube_faces, q);
        let lib = ev.hessian6(q)?;
        let rel = rel6(&mine, &lib);
        println!("  cube {label}: rel diff {rel:.3e}");
    }

    // Direct sum over a finely subdivided cube (n³ point masses).
    let n = 24usize;
    let cell = side / n as f64;
    let m = cell * cell * cell;
    let mut direct = [0.0f64; 6];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let y = [
                    -side / 2.0 + (i as f64 + 0.5) * cell,
                    -side / 2.0 + (j as f64 + 0.5) * cell,
                    -side / 2.0 + (k as f64 + 0.5) * cell,
                ];
                let d = [y[0] - p[0], y[1] - p[1], y[2] - p[2]];
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                let r3 = r2 * r2.sqrt();
                let gm = G * m;
                let c = 3.0 * gm / (r3 * r2);
                let dd = gm / r3;
                direct[0] += c * d[0] * d[0] - dd;
                direct[1] += c * d[1] * d[1] - dd;
                direct[2] += c * d[2] * d[2] - dd;
                direct[3] += c * d[0] * d[1];
                direct[4] += c * d[0] * d[2];
                direct[5] += c * d[1] * d[2];
            }
        }
    }
    let lib = ev.hessian6(p)?;
    println!(
        "  cube direct sum vs ESA: rel {:.3e} (same sign: {})",
        rel6(&direct, &lib),
        lib[2] * direct[2] > 0.0
    );
    selftest_mass()?;
    selftest_surface_form(args)?;
    selftest_carlson(args)?;
    selftest_carlson_alpha(args)?;
    selftest_gpu(args)
}

/// General-alpha radial finite part against the f64 reference and the Carlson
/// symmetric-function library. This is the check that keeps `CarlsonAlpha`
/// distinct from the alpha=1 jump-surface implementation.
fn selftest_carlson_alpha(args: &Args) -> Result<(), String> {
    use crate::carlson_alpha::special;
    let pi = std::f64::consts::PI;
    let rf = special::rf(0.0, 1.0, 1.0)?;
    let rd = special::rd(0.0, 1.0, 1.0)?;
    let rc = special::rc(0.0, 1.0)?;
    let rj = special::rj(1.0, 1.0, 1.0, 1.0)?;
    let tail = carlson_alpha::third_kind_tail(1.0, 2.0, 3.0, 4.0)?;
    println!("  Carlson library: RF={rf:.12} RD={rd:.12} RC={rc:.12} RJ={rj:.12} tail={tail:.12}");
    if (rf - pi / 2.0).abs() > 1e-12
        || (rd - 3.0 * pi / 4.0).abs() > 1e-12
        || (rc - pi / 2.0).abs() > 1e-12
        || (rj - 1.0).abs() > 1e-12
        || !tail.is_finite()
    {
        return Err("Carlson library identity check failed".into());
    }

    let mesh = unit_cube();
    let faces = analytic::precompute(&mesh);
    let bvh = bvh::Bvh::build(&mesh);
    let points = mesh.observation_points(1e-2);
    let sample: Vec<[f64; 3]> = (0..24).map(|i| points[i * points.len() / 24]).collect();
    let dirs = quadrature::directions(args.directions.min(256));
    let kernel = crate::density::KernelSi {
        c: [0.1, -0.1, 0.2],
        sigma: 0.8,
        w: 1.0,
        alpha: 1.25,
    };
    let device = gpu::Device::new()?;
    let scene = gpu::Scene::new(device, &mesh, &bvh, &faces, &sample, &dirs, sample.len(), 1);
    let radius = body_radius(&mesh);
    let t_max = radius * 4.0;
    let gpu = scene.block_carlson_alpha(
        &sample,
        &dirs,
        std::slice::from_ref(&kernel),
        t_max as f32,
        GEOM_EPS_M as f32,
    )?;

    let mut hits = Vec::new();
    let mut errs = Vec::new();
    for (i, x) in sample.iter().enumerate() {
        // `block_carlson_alpha` returns the regularized radial remainder only.
        // The analytic uniform-density tensor belongs to the separate near-field
        // term, so comparing it against a full `rho·W + remainder` reference
        // would manufacture a large false disagreement.
        let mut want = [0.0; 6];
        for (u, omega) in &dirs {
            geom::bvh_crossings(
                &bvh,
                &mesh,
                *x,
                [-u[0], -u[1], -u[2]],
                GEOM_EPS_M,
                t_max,
                &mut hits,
            );
            let (slots, overflow) = split::intervals(&hits, false);
            if overflow {
                return Err("alpha selftest ray exceeded interval capacity".into());
            }
            let intervals: Vec<(f64, f64)> = slots.iter().flatten().copied().collect();
            let scalar = carlson_alpha::remainder_scalar_reference(
                std::slice::from_ref(&kernel),
                *x,
                *u,
                &intervals,
            );
            tensor::add_tensor_term(&mut want, u, G * *omega * scalar);
        }
        errs.push(rel6(&gpu[i], &want));
    }
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let worst = errs[errs.len() - 1];
    println!(
        "  CarlsonAlpha alpha=1.25 GPU vs f64/Carlson reference: median {:.3e}, worst {:.3e}",
        errs[errs.len() / 2],
        worst
    );
    if worst.is_nan() || worst > 2e-2 {
        return Err(format!(
            "CarlsonAlpha general-alpha GPU path disagrees by {worst:.3e}"
        ));
    }
    Ok(())
}

/// The identity the Carlson solver rests on, checked end-to-end on the real mesh:
/// with one density everywhere every cone weight `ρ_f − ρ_ref` vanishes, so the
/// 983 040-triangle face list has to collapse back onto `ρ_ref·W(x)` — the same
/// quantity `analytic.rs`, the Werner bake and the ESA library compute from
/// entirely different code. This is what proves the per-face weights survive the
/// trip through the shader and that the mesh faces carry `ρ_ref`.
///
/// The star-cone *orientations* are checked separately, and on the unit cube:
/// a cone only tiles a body that is star-shaped from its apex, and Ryugu is not
/// (see the coverage/signed-ratio line below), so running the raw cone on the
/// real mesh would be measuring the body's shape rather than the code.
fn selftest_carlson(args: &Args) -> Result<(), String> {
    let rho = 1190.0f64;
    let mesh =
        Mesh::load_obj(&args.obj, KM_TO_M).map_err(|e| format!("{}: {e}", args.obj.display()))?;
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
fn selftest_surface_form(args: &Args) -> Result<(), String> {
    let mesh =
        Mesh::load_obj(&args.obj, KM_TO_M).map_err(|e| format!("{}: {e}", args.obj.display()))?;
    let faces = analytic::precompute(&mesh);
    let ev = esa::Evaluable::new(&mesh, 1.0)?;
    let nv = mesh.vertex_count();
    let mut rng = 0x2545_F491_4F6C_DD1Du64;
    let mut next_u32 = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as u32
    };
    let radius = (0..nv)
        .map(|i| {
            let p = mesh.vertex(i);
            (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
        })
        .fold(0.0f64, f64::max);

    let mut samples: Vec<([f64; 3], String)> = Vec::new();
    for mm in [1.0f64, 10.0, 100.0, 500.0] {
        let pts = mesh.observation_points(mm * 1e-3);
        for _ in 0..8 {
            let i = next_u32() as usize % nv;
            samples.push((pts[i], format!("surface@{mm}mm v{i}")));
        }
    }
    for k in 0..8 {
        let theta = 2.0 * std::f64::consts::PI * (k as f64) / 8.0;
        let r = radius * (1.15 + 0.05 * k as f64);
        samples.push((
            [
                r * theta.cos(),
                0.3 * r * theta.sin(),
                0.2 * r * theta.cos(),
            ],
            format!("free r={r:.1}m"),
        ));
    }

    let mut worst = (0.0f64, String::new());
    let mut errs = Vec::new();
    for (x, label) in samples.iter() {
        let lib = ev.hessian6(*x)?;
        let mine = analytic::hessian6(&faces, *x);
        let rel = rel6(&mine, &lib);
        errs.push(rel);
        if rel > worst.0 {
            worst = (rel, label.clone());
        }
    }
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "  closed form vs ESA on the real mesh: {} samples, median {:.3e}, worst {:.3e} ({})",
        samples.len(),
        errs[errs.len() / 2],
        worst.0,
        worst.1
    );
    if worst.0 > 1e-6 {
        return Err(format!(
            "closed form disagrees with the ESA library by {:.3e} ({})",
            worst.0, worst.1
        ));
    }
    Ok(())
}

/// WGSL pipelines against the references that do not share their code:
/// `analytic.rs` (closed form, itself checked against ESA) for the tensor, and a
/// brute-force f64 tracer + `split.rs` for the ray/remainder chain. The BVH
/// traversal the shader uses is checked separately, by walking the same flat
/// arrays from Rust.
fn selftest_gpu(args: &Args) -> Result<(), String> {
    let mesh =
        Mesh::load_obj(&args.obj, KM_TO_M).map_err(|e| format!("{}: {e}", args.obj.display()))?;
    let faces = analytic::precompute(&mesh);
    let bvh = bvh::Bvh::build(&mesh);
    let radius = body_radius(&mesh);
    let t_max = radius * 4.0;
    let dirs = quadrature::directions(args.directions);
    let density = match args.mode {
        DensityMode::Cauchy => Density::from_toml(&args.density, &mesh, args.normalize)?,
        DensityMode::Elliptic => {
            return Err("mode=elliptic requires --solver carlson-alpha".into());
        }
        DensityMode::Constant => Density::homogeneous(1190.0, &mesh),
    };
    let kernels = density.kernels();
    let all = mesh.observation_points(1e-3);
    let nv = mesh.vertex_count();
    let sample: Vec<[f64; 3]> = (0..192).map(|i| all[i * nv / 192]).collect();

    let device = gpu::Device::new()?;
    let scene = gpu::Scene::new(
        device,
        &mesh,
        &bvh,
        &faces,
        &sample,
        &dirs,
        sample.len(),
        kernels.len(),
    );

    // 1. Analytic tensor: WGSL (f32) vs the f64 closed form.
    let w_gpu = scene.analytic_tensors()?;
    let mut errs: Vec<f64> = sample
        .iter()
        .enumerate()
        .map(|(i, p)| rel6(&analytic::hessian6(&faces, *p), &w_gpu[i]))
        .collect();
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let worst_w = errs[errs.len() - 1];
    println!(
        "  WGSL analytic vs f64 closed form: median {:.3e}, worst {:.3e}",
        errs[errs.len() / 2],
        worst_w
    );

    // 2. Ray traversal: the shader's BVH layout, walked from Rust, vs brute force.
    let brute = BruteTracer::new(&mesh);
    let mut hits = Vec::new();
    let mut bvh_hits = Vec::new();
    let mut worst_ray = 0.0f64;
    for (i, p) in sample.iter().enumerate().take(32) {
        for (q, (u, _)) in dirs.iter().enumerate() {
            let d = [-u[0], -u[1], -u[2]];
            brute.crossings(*p, d, GEOM_EPS_M, t_max, &mut hits);
            geom::bvh_crossings(&bvh, &mesh, *p, d, GEOM_EPS_M, t_max, &mut bvh_hits);
            if hits.len() != bvh_hits.len() {
                worst_ray = f64::INFINITY;
                println!(
                    "    point {i} direction {q}: {} crossings, BVH found {}",
                    hits.len(),
                    bvh_hits.len()
                );
                continue;
            }
            for (a, b) in hits.iter().zip(bvh_hits.iter()) {
                worst_ray = worst_ray.max((a - b).abs() / a.abs().max(1e-30));
            }
        }
    }
    println!("  BVH traversal vs brute force: worst crossing rel {worst_ray:.3e}");

    // 3. Remainder: full WGSL chain (probe + rays + remainder) vs f64 brute force.
    let rem_gpu =
        scene.block_remainder(&sample, &dirs, kernels, t_max as f32, GEOM_EPS_M as f32)?;
    let mut errs: Vec<f64> = Vec::new();
    let mut worst = (0.0f64, 0usize);
    for (i, p) in sample.iter().enumerate() {
        brute.crossings(*p, [1.0, 0.0, 0.0], GEOM_EPS_M, t_max, &mut hits);
        let inside = hits.len() % 2 == 1;
        let mut acc = [0.0f64; 6];
        for (u, w) in &dirs {
            brute.crossings(*p, [-u[0], -u[1], -u[2]], GEOM_EPS_M, t_max, &mut hits);
            let (slots, overflow) = split::intervals(&hits, inside);
            if overflow {
                return Err(format!(
                    "reference ray at {p:?} exceeded {} visible intervals",
                    split::MAX_INTERVALS
                ));
            }
            let s = split::remainder_scalar(kernels, *p, *u, &slots);
            tensor::add_tensor_term(&mut acc, u, G * *w * s);
        }
        let rel = rel6(&acc, &rem_gpu[i]);
        errs.push(rel);
        if rel > worst.0 {
            worst = (rel, i);
        }
    }
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let over = errs.iter().filter(|e| **e > 5e-2).count();
    println!(
        "  WGSL rays+remainder vs f64 brute force: median {:.3e}, p90 {:.3e}, worst {:.3e} ({over} of {} >5%)",
        errs[errs.len() / 2],
        errs[errs.len() * 9 / 10],
        worst.0,
        errs.len()
    );
    // A handful of rays graze a mesh vertex, where a crossing sits exactly on the
    // f32/f64 hit tolerance and can flip; the rest of the chain has to be tight.
    let median_rem = errs[errs.len() / 2];
    if worst_w > 1e-2 || worst_ray > 1e-6 || median_rem > 1e-3 || worst.0 > 2e-1 {
        return Err("GPU pipelines disagree with the reference".into());
    }
    Ok(())
}

fn body_radius(mesh: &Mesh) -> f64 {
    (0..mesh.vertex_count())
        .map(|i| {
            let p = mesh.vertex(i);
            (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
        })
        .fold(0.0f64, f64::max)
}

/// Print both normalisation conventions side by side, so the constant factor
/// between this record and one baked under the other convention is on the log
/// rather than buried inside a face-by-face comparison.
fn print_mass_report(density: &Density, mode: DensityMode) {
    let m = &density.mass;
    println!(
        "mass integral (divergence theorem): body volume {:.6e} m³",
        m.volume
    );
    match mode {
        DensityMode::Cauchy | DensityMode::Elliptic => {
            let target = if m.target_mass > 0.0 {
                format!("{:.6e} kg", m.target_mass)
            } else {
                "absent from TOML".to_string()
            };
            if mode == DensityMode::Cauchy {
                println!(
                    "  raw TOML weights: ∫ρ dV = {:.6e} kg  →  ρ(0)=1190 convention ×{:.9e} = {:.6e} kg",
                    m.raw_mass, m.rho0_scale, m.rho0_mass
                );
            } else {
                println!(
                    "  elliptic fractional-alpha field: weights kept raw, scale is shared with Mascon"
                );
            }
            println!(
                "  normalization={:?}: ×{:.9e} = {:.6e} kg (target {target})",
                m.normalization, m.applied_scale, m.applied_mass
            );
            if m.rho0_mass.abs() > 1e-30 {
                println!(
                    "  scale vs the old ρ(0) convention: ×{:.9} ({:+.4} %) — a constant on every face",
                    m.scale_vs_rho0(),
                    100.0 * (m.scale_vs_rho0() - 1.0)
                );
            }
        }
        DensityMode::Constant => println!(
            "  constant {:.1} kg/m³ × V = {:.6e} kg (the Werner bake's density, unchanged)",
            density.bulk_density, m.applied_mass
        ),
    }
    std::io::Write::flush(&mut std::io::stdout()).ok();
}

/// Checks the volume→surface reduction in `mass.rs` against closed forms: the
/// exact enclosed volume of a cube, and the analytic ball integral
/// `4π(R − atan(σR)/σ)/σ²` for the sphere tessellations the mesh format allows.
fn selftest_mass() -> Result<(), String> {
    let cube = unit_cube();
    let v = mass::body_volume(&cube);
    println!(
        "  cube volume: {v:.12} (exact 8) → rel {:.3e}",
        (v - 8.0).abs() / 8.0
    );
    if (v - 8.0).abs() > 1e-10 {
        return Err(format!("cube volume {v} != 8"));
    }

    let mut worst = (0.0f64, 0.0f64);
    for sigma in [0.0f64, 0.05, 0.28, 1.0, 8.0] {
        let sphere = mass::uv_sphere(0.5, 128);
        let got = mass::kernel_volume(&sphere, [0.0, 0.0, 0.0], sigma);
        let want = mass::ball_kernel_volume(0.5, sigma);
        let rel = (got - want).abs() / want;
        if rel > worst.0 {
            worst = (rel, sigma);
        }
    }
    println!(
        "  sphere ∫(1+σ²r²)⁻¹dV vs closed form: worst rel {:.3e} (σ={})",
        worst.0, worst.1
    );
    if worst.0 > 3e-4 {
        return Err(format!(
            "surface mass integral disagrees with the closed form by {:.3e}",
            worst.0
        ));
    }
    Ok(())
}

fn rel6(a: &Sym6, b: &Sym6) -> f64 {
    let d = frobenius(&[
        a[0] - b[0],
        a[1] - b[1],
        a[2] - b[2],
        a[3] - b[3],
        a[4] - b[4],
        a[5] - b[5],
    ]);
    d / frobenius(b).max(1e-300)
}

fn run(args: &Args) -> Result<(), String> {
    run_raylike(args, false)
}

fn run_raylike(args: &Args, general_alpha: bool) -> Result<(), String> {
    let standoff_m = args.standoff_mm * 1e-3;
    let mesh =
        Mesh::load_obj(&args.obj, KM_TO_M).map_err(|e| format!("{}: {e}", args.obj.display()))?;
    let nv = mesh.vertex_count();
    let nf = mesh.face_count();
    println!("observation standoff: {:.3} mm", args.standoff_mm);
    println!("mesh: {nv} vertices, {nf} faces (meters)");
    println!(
        "solver={} mode={:?} directions={}",
        if general_alpha {
            "carlson-alpha"
        } else {
            "rtfp"
        },
        args.mode,
        args.directions
    );
    std::io::Write::flush(&mut std::io::stdout()).ok();

    let points = mesh.observation_points(standoff_m);
    let faces = analytic::precompute(&mesh);
    let bvh = bvh::Bvh::build(&mesh);
    println!("bvh: {} nodes, depth {}", bvh.node_count(), bvh.depth);
    let radius = body_radius(&mesh);
    let t_min = GEOM_EPS_M as f32;
    let t_max = (radius * 4.0) as f32;

    let density = match args.mode {
        DensityMode::Cauchy => Density::from_toml(&args.density, &mesh, args.normalize)?,
        DensityMode::Elliptic => {
            Density::from_toml_for_mode(&args.density, &mesh, args.normalize, args.mode)?
        }
        DensityMode::Constant => Density::homogeneous(1190.0, &mesh),
    };
    print_mass_report(&density, args.mode);
    let dirs = quadrature::directions(args.directions);
    let kernels = density.kernels();
    println!(
        "quadrature: {} directions (exact: Σω=4π, ΣωT=0)",
        dirs.len()
    );
    std::io::Write::flush(&mut std::io::stdout()).ok();

    let device = gpu::Device::new()?;
    let scene = gpu::Scene::new(
        device,
        &mesh,
        &bvh,
        &faces,
        &points,
        &dirs,
        POINTS_PER_BLOCK,
        kernels.len(),
    );
    // Which vertices the record still needs (resume).
    let record = if args.resume {
        let existing = record::Record::open_resume(&args.out, nf)
            .map_err(|e| format!("{}: {e}", args.out.display()))?;
        match existing {
            Some(r) if (r.standoff_mm - args.standoff_mm as f32).abs() < 1e-3 => {
                println!("RESUME from {} / {} faces", r.n_done, nf);
                r
            }
            _ => record::Record::create(&args.out, nf, args.standoff_mm as f32)
                .map_err(|e| format!("{}: {e}", args.out.display()))?,
        }
    } else {
        record::Record::create(&args.out, nf, args.standoff_mm as f32)
            .map_err(|e| format!("{}: {e}", args.out.display()))?
    };
    let mut needed = vec![false; nv];
    let mut todo = Vec::new();
    for f in 0..nf {
        if !record.scalars[f].is_finite() {
            todo.push(f as u32);
            for v in mesh.face(f) {
                needed[v as usize] = true;
            }
        }
    }
    println!(
        "evaluating {} vertices ({} faces remaining)",
        needed.iter().filter(|b| **b).count(),
        todo.len()
    );
    std::io::Write::flush(&mut std::io::stdout()).ok();

    // Per-vertex tensors, not scalars: the face value has to be the norm of the
    // three-vertex-averaged tensor, exactly like `bakes/werner/bake_main.cpp` and
    // `bakes/mascon/mascon_bake_main.cpp` do. Averaging the three per-vertex norms
    // instead is a *different* (always larger, by the triangle inequality)
    // quantity — it was worth 1.9 % median against the Werner record.
    let mut vertex_h = vec![[f64::NAN; 6]; nv];
    let resumed_vertices = if args.resume {
        match checkpoint::load(&args.checkpoint, nv)
            .map_err(|e| format!("{}: {e}", args.checkpoint.display()))?
        {
            Some(saved) => {
                let count = saved.len();
                vertex_h[..count].copy_from_slice(&saved);
                println!("RESUME from {count} / {nv} vertex tensors");
                count
            }
            None => 0,
        }
    } else {
        0
    };
    if !todo.is_empty() {
        for (block, lo) in (0..nv).step_by(POINTS_PER_BLOCK).enumerate() {
            let hi = (lo + POINTS_PER_BLOCK).min(nv);
            if hi <= resumed_vertices {
                println!("PROGRESS_R {hi} {nv}");
                continue;
            }
            let w_block = scene.analytic_tensors_block(lo, hi)?;
            let remainder = if general_alpha {
                scene.block_carlson_alpha(&points[lo..hi], &dirs, kernels, t_max, t_min)?
            } else {
                scene.block_remainder(&points[lo..hi], &dirs, kernels, t_max, t_min)?
            };
            for (i, rem) in remainder.into_iter().enumerate() {
                let vi = lo + i;
                if !needed[vi] {
                    continue;
                }
                let rho = density.rho(&points[vi], args.mode);
                let mut h = w_block[i];
                for t in h.iter_mut() {
                    *t *= rho;
                }
                for (t, r) in h.iter_mut().zip(rem.iter()) {
                    *t += r;
                }
                vertex_h[vi] = h;
            }
            if (block + 1) % CHECKPOINT_BLOCKS == 0 || hi == nv {
                checkpoint::save(&args.checkpoint, &vertex_h, hi)
                    .map_err(|e| format!("{}: {e}", args.checkpoint.display()))?;
            }
            println!("PROGRESS_R {hi} {nv}");
            std::io::Write::flush(&mut std::io::stdout()).ok();
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    write_face_record(args, &mesh, &vertex_h, record)
}

/// Per-face scalars from per-vertex tensors, written progressively in the saved
/// (shuffled) order so the viewer can start drawing before the bake finishes.
///
/// The face value is the norm of the *averaged* tensor, exactly like
/// `bakes/werner/bake_main.cpp` and `bakes/mascon/mascon_bake_main.cpp`:
/// averaging the three per-vertex norms instead is a different (always larger,
/// by the triangle inequality) quantity.
fn write_face_record(
    args: &Args,
    mesh: &Mesh,
    vertex_h: &[[f64; 6]],
    mut record: record::Record,
) -> Result<(), String> {
    let nf = mesh.face_count();
    let saved_order = record::load_order(&args.order, nf)
        .map_err(|e| format!("{}: {e}", args.order.display()))?;
    let order = match saved_order {
        Some(o) => o,
        None => {
            let mut o: Vec<u32> = (0..nf as u32).collect();
            let mut rng = 0x9E37_79B9_7F4A_7C15u64
                ^ (args.standoff_mm as u64)
                ^ (std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.subsec_nanos() as u64)
                    .unwrap_or(0));
            for i in (1..o.len()).rev() {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                o.swap(i, (rng as usize) % (i + 1));
            }
            record::write_order(&args.order, &o)
                .map_err(|e| format!("{}: {e}", args.order.display()))?;
            o
        }
    };
    let mut step = 0usize;
    for f in order {
        let fi = f as usize;
        if record.scalars[fi].is_finite() {
            continue;
        }
        let face = mesh.face(fi);
        let mut hbar = [0.0f64; 6];
        for k in 0..6 {
            hbar[k] = (vertex_h[face[0] as usize][k]
                + vertex_h[face[1] as usize][k]
                + vertex_h[face[2] as usize][k])
                / 3.0;
        }
        let s = frobenius(&hbar) as f32;
        record
            .set_face(fi, s)
            .map_err(|e| format!("{}: {e}", args.out.display()))?;
        step += 1;
        if step.is_multiple_of(256) || record.n_done == nf {
            record
                .flush_header()
                .map_err(|e| format!("{}: {e}", args.out.display()))?;
            println!("faces {} / {}", record.n_done, nf);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
    }
    record
        .flush_header()
        .map_err(|e| format!("{}: {e}", args.out.display()))?;
    println!(
        "wrote {} dense face scalars to {} (s_min={:e} s_max={:e})",
        nf,
        args.out.display(),
        record.s_min,
        record.s_max
    );
    Ok(())
}

/// Carlson density-jump solver: one analytic pass over the star-cone jump
/// surfaces, no rays and no directional quadrature.
fn run_carlson(args: &Args) -> Result<(), String> {
    let standoff_m = args.standoff_mm * 1e-3;
    let mesh =
        Mesh::load_obj(&args.obj, KM_TO_M).map_err(|e| format!("{}: {e}", args.obj.display()))?;
    let nv = mesh.vertex_count();
    let nf = mesh.face_count();
    println!("solver=carlson (density-jump surface integral, no ray tracing)");
    println!("observation standoff: {:.3} mm", args.standoff_mm);
    println!("mesh: {nv} vertices, {nf} faces (meters)");
    println!("mode={:?}", args.mode);
    std::io::Write::flush(&mut std::io::stdout()).ok();

    let points = mesh.observation_points(standoff_m);
    let density = match args.mode {
        DensityMode::Cauchy => Density::from_toml(&args.density, &mesh, args.normalize)?,
        DensityMode::Constant => Density::homogeneous(1190.0, &mesh),
        DensityMode::Elliptic => {
            return Err("mode=elliptic requires --solver carlson-alpha".into());
        }
    };
    print_mass_report(&density, args.mode);

    let (faces, stats) = carlson::face_list(&mesh, &density, args.mode);
    println!(
        "jump surfaces: {} triangles = {nf} mesh (weight ρ_ref) + {} star-cone (weight ρ_f − ρ_ref)",
        faces.len(),
        stats.n_faces
    );
    println!(
        "  star cone from [{:.3}, {:.3}, {:.3}] m, ρ_ref = {:.6e} kg/m³",
        stats.origin[0], stats.origin[1], stats.origin[2], stats.rho_ref
    );
    println!(
        "  cone coverage {:.5} % of the mesh volume {:.6e} m³, |signed|/covered {:.6} \
         (below 1 ⇒ not star-shaped; the split keeps that defect off the constant term)",
        100.0 * stats.coverage(),
        stats.mesh_volume,
        stats.signed_ratio()
    );
    println!(
        "  |ρ_f − ρ_ref| range {:.6e} .. {:.6e} kg/m³ ({:.3e} of ρ_ref)",
        stats.min_jump,
        stats.max_jump,
        (stats.max_jump - stats.min_jump) / stats.rho_ref.abs().max(1e-300)
    );
    println!(
        "  radial refinement: {} slabs (max {} per cone), requested tolerance {:.1e}, \
         worst within-slab variation {:.3e} of ρ_ref{}",
        stats.slabs,
        stats.max_slabs_used,
        stats.tol,
        stats.worst_slab_variation,
        if stats.over_budget {
            " [face budget reached]"
        } else {
            ""
        }
    );
    std::io::Write::flush(&mut std::io::stdout()).ok();

    // The analytic pass needs the mesh and BVH buffers on the device, but none of
    // the ray/remainder state is ever dispatched from here.
    let bvh = bvh::Bvh::build(&mesh);
    let dirs = quadrature::directions(8);
    let device = gpu::Device::new()?;
    let scene = gpu::Scene::new(
        device,
        &mesh,
        &bvh,
        &faces,
        &points,
        &dirs,
        POINTS_PER_BLOCK,
        0,
    );
    println!(
        "evaluating {nv} vertices × {} faces on the GPU",
        faces.len()
    );
    std::io::Write::flush(&mut std::io::stdout()).ok();
    let mut vertex_h = vec![[f64::NAN; 6]; nv];
    let resumed_vertices = if args.resume {
        match checkpoint::load(&args.checkpoint, nv)
            .map_err(|e| format!("{}: {e}", args.checkpoint.display()))?
        {
            Some(saved) => {
                let count = saved.len();
                vertex_h[..count].copy_from_slice(&saved);
                println!("RESUME from {count} / {nv} vertex tensors");
                count
            }
            None => 0,
        }
    } else {
        0
    };
    for (block, lo) in (0..nv).step_by(POINTS_PER_BLOCK).enumerate() {
        let hi = (lo + POINTS_PER_BLOCK).min(nv);
        if hi <= resumed_vertices {
            println!("PROGRESS_R {hi} {nv}");
            continue;
        }
        scene.carlson_surface_block(lo, hi, &mut vertex_h)?;
        if (block + 1) % CHECKPOINT_BLOCKS == 0 || hi == nv {
            checkpoint::save(&args.checkpoint, &vertex_h, hi)
                .map_err(|e| format!("{}: {e}", args.checkpoint.display()))?;
        }
        println!("PROGRESS_R {hi} {nv}");
        std::io::Write::flush(&mut std::io::stdout()).ok();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    if let Some(bad) = vertex_h.iter().flatten().find(|v| !v.is_finite()) {
        return Err(format!(
            "carlson tensor produced a non-finite component ({bad})"
        ));
    }

    let record = if args.resume {
        let existing = record::Record::open_resume(&args.out, nf)
            .map_err(|e| format!("{}: {e}", args.out.display()))?;
        match existing {
            Some(r) if (r.standoff_mm - args.standoff_mm as f32).abs() < 1e-3 => {
                println!("RESUME from {} / {} faces", r.n_done, nf);
                r
            }
            _ => record::Record::create(&args.out, nf, args.standoff_mm as f32)
                .map_err(|e| format!("{}: {e}", args.out.display()))?,
        }
    } else {
        record::Record::create(&args.out, nf, args.standoff_mm as f32)
            .map_err(|e| format!("{}: {e}", args.out.display()))?
    };
    write_face_record(args, &mesh, &vertex_h, record)
}
