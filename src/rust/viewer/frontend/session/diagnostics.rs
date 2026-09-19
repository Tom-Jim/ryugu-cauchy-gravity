// Opt-in diagnostic charts backed by the existing browser solver pipelines.
// Diagnostics use a dedicated six-component tensor output and never modify the
// normal RHGF record, the Bevy renderer, or the saved-result store.

const DIAGNOSTIC_FACES: usize = 96;
// Heights used by the exterior Laplace chart, ordered from far to near so the
// log-scale plot reads left-to-right as the surface is approached.
const CONSISTENCY_HEIGHTS_MM: [f64; 8] = [
    1_000_000.0,
    100_000.0,
    10_000.0,
    1_000.0,
    300.0,
    30.0,
    3.0,
    1.0,
];
// Heights used by the near-surface residual chart. This is intentionally a
// complete geometric-ish sample of 10 m down to 1 mm, not a prefix of the
// slider positions; the chart compares solver consistency at common heights.
const STABILITY_HEIGHTS_MM: [f64; 6] = [10_000.0, 1_000.0, 300.0, 30.0, 3.0, 1.0];
const SWEEP_THETA: [f64; 4] = [0.5, 0.25, 0.1, 0.05];
const SWEEP_DIRECTIONS: [f64; 4] = [8.0, 16.0, 32.0, 64.0];
// Each value selects a complete Gauss-Legendre rule.  A 12-point prefix of the
// GL16 table is not a quadrature rule and produced a misleading sweep.
const SWEEP_QUADRATURE: [f64; 3] = [4.0, 8.0, 16.0];

#[derive(Clone, Copy)]
struct DiagnosticSolver {
    label: &'static str,
    algorithm: &'static str,
    source_set: &'static str,
    density: &'static str,
    color: &'static str,
}

#[derive(Clone, Debug)]
struct DiagnosticTensor {
    values: Vec<[f64; 6]>,
}

impl DiagnosticTensor {
    fn len(&self) -> usize {
        self.values.len()
    }
}

fn diagnostic_solvers() -> [DiagnosticSolver; 9] {
    [
        DiagnosticSolver {
            label: "Werner - uniform",
            algorithm: "werner",
            source_set: "constant",
            density: "uniform",
            color: "#8ce9ff",
        },
        DiagnosticSolver {
            label: "Mascon - fractional Cauchy",
            algorithm: "mascon",
            source_set: "elliptic",
            density: "fractional-cauchy",
            color: "#ffd27e",
        },
        DiagnosticSolver {
            label: "Mascon - Cauchy",
            algorithm: "mascon",
            source_set: "cauchy",
            density: "cauchy",
            color: "#ffad66",
        },
        DiagnosticSolver {
            label: "RT-FP - Cauchy",
            algorithm: "rtfp",
            source_set: "cauchy",
            density: "cauchy",
            color: "#7dffbf",
        },
        DiagnosticSolver {
            label: "RT-FP - uniform",
            algorithm: "rtfp",
            source_set: "constant",
            density: "uniform",
            color: "#4ee0a5",
        },
        DiagnosticSolver {
            label: "Carlson - Cauchy",
            algorithm: "carlson",
            source_set: "cauchy",
            density: "cauchy",
            color: "#d39cff",
        },
        DiagnosticSolver {
            label: "Carlson - uniform",
            algorithm: "carlson",
            source_set: "constant",
            density: "uniform",
            color: "#a983ff",
        },
        DiagnosticSolver {
            label: "CarlsonAlpha - fractional Cauchy",
            algorithm: "carlsonalpha",
            source_set: "elliptic",
            density: "fractional-cauchy",
            color: "#f08cff",
        },
        DiagnosticSolver {
            label: "CarlsonAlpha - uniform",
            algorithm: "carlsonalpha",
            source_set: "constant",
            density: "uniform",
            color: "#b9a7ff",
        },
    ]
}

fn diagnostic_title(kind: &str) -> Option<(&'static str, &'static str, &'static str)> {
    match kind {
        "pareto" => Some((
            "Pareto frontier - parameter sweeps",
            "Batch wall time / point (us, log scale)",
            "Relative error (log scale)",
        )),
        "stability" => Some((
            "Near-surface residual - density facets",
            "Height above surface (m, log scale)",
            "Relative difference to reference (log scale)",
        )),
        "symmetry" => Some((
            "Tensor consistency - exterior Laplace check",
            "Height above surface (m, log scale)",
            "|trace(H)| / ||H||F (log scale)",
        )),
        _ => None,
    }
}

fn start_diagnostic(core: &Rc<RefCell<Core>>, kind: &str) {
    let Some((title, x_label, y_label)) = diagnostic_title(kind) else {
        return;
    };
    {
        let mut state = core.borrow_mut();
        if state.compute_running || state.diagnostic_running {
            set_bool(&state.ui.diagnostics, "busy", false);
            set_bool(&state.ui.diagnostics, "ready", false);
            set_str(&state.ui.diagnostics, "status", "Diagnostic unavailable");
            set_str(
                &state.ui.diagnostics,
                "error",
                "A full computation is already running. Wait for it to finish first.",
            );
            return;
        }
        state.diagnostic_running = true;
        set_bool(&state.ui.diagnostics, "visible", true);
        set_bool(&state.ui.diagnostics, "busy", true);
        set_bool(&state.ui.diagnostics, "ready", false);
        set_f64(&state.ui.diagnostics, "progress", 0.0);
        set_str(&state.ui.diagnostics, "kind", kind);
        set_str(&state.ui.diagnostics, "title", title);
        set_str(&state.ui.diagnostics, "xLabel", x_label);
        set_str(&state.ui.diagnostics, "yLabel", y_label);
        set_str(
            &state.ui.diagnostics,
            "status",
            "Preparing diagnostic tensor pipeline...",
        );
        set_str(&state.ui.diagnostics, "error", "");
        set_str(&state.ui.diagnostics, "summary", "");
        set(&state.ui.diagnostics, "series", Array::new().into());
    }
    let kind = kind.to_string();
    let task_core = core.clone();
    spawn_local(async move {
        let result = run_diagnostic(&task_core, &kind).await;
        let ui = task_core.borrow().ui.diagnostics.clone();
        let mut state = task_core.borrow_mut();
        state.diagnostic_running = false;
        set_bool(&ui, "busy", false);
        match result {
            Ok(()) => {
                set_bool(&ui, "ready", true);
                set_f64(&ui, "progress", 100.0);
                set_str(&ui, "status", "Diagnostic complete");
            }
            Err(error) => {
                set_bool(&ui, "ready", false);
                set_str(&ui, "error", &error);
                set_str(&ui, "status", "Diagnostic failed");
            }
        }
    });
}

async fn run_diagnostic(core: &Rc<RefCell<Core>>, kind: &str) -> Result<(), String> {
    match kind {
        "pareto" => run_pareto(core).await,
        "stability" => run_stability(core).await,
        "symmetry" => run_tensor_symmetry(core).await,
        _ => Err(format!("unknown diagnostic {kind}")),
    }
}

fn parse_tensor(bytes: &[u8]) -> Option<DiagnosticTensor> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(24) {
        return None;
    }
    let mut values = Vec::with_capacity(bytes.len() / 24);
    for chunk in bytes.as_chunks::<24>().0 {
        let mut tensor = [0.0; 6];
        for (index, value) in tensor.iter_mut().enumerate() {
            let start = index * 4;
            *value = f32::from_le_bytes(chunk[start..start + 4].try_into().ok()?) as f64;
        }
        if tensor.iter().any(|value| !value.is_finite()) {
            return None;
        }
        values.push(tensor);
    }
    Some(DiagnosticTensor { values })
}

fn set_optional_f64(options: &Object, name: &str, value: Option<f64>) {
    if let Some(value) = value {
        set_f64(options, name, value);
    }
}

#[allow(clippy::too_many_arguments)]
async fn evaluate_tensor_sample(
    core: &Rc<RefCell<Core>>,
    solver: DiagnosticSolver,
    height_mm: f64,
    face_count: usize,
    direction_limit: Option<f64>,
    quadrature_limit: Option<f64>,
    mascon_theta: Option<f64>,
    progress_base: f64,
    progress_span: f64,
) -> Result<(DiagnosticTensor, f64), String> {
    let ui = core.borrow().ui.diagnostics.clone();
    set_str(
        &ui,
        "status",
        &format!("Running {} - {:.0} m", solver.label, height_mm / 1000.0),
    );
    let progress_ui = ui.clone();
    let progress = Closure::<dyn FnMut(f64)>::new(move |fraction: f64| {
        let percent = progress_base + progress_span * fraction.clamp(0.0, 1.0);
        set_f64(&progress_ui, "progress", percent);
        set_str(
            &progress_ui,
            "status",
            &format!("Running diagnostic pipeline - {:.0}%", percent),
        );
    });
    let options = Object::new();
    set_str(&options, "algorithm", solver.algorithm);
    set_str(&options, "sourceSet", solver.source_set);
    set_f64(&options, "heightMm", height_mm);
    set_f64(&options, "faceCount", face_count as f64);
    set_optional_f64(&options, "directionLimit", direction_limit);
    set_optional_f64(&options, "quadratureLimit", quadrature_limit);
    set_optional_f64(&options, "masconTheta", mascon_theta);
    set(&options, "onProgress", progress.as_ref().clone());
    let mut engine = core
        .borrow_mut()
        .engine
        .take()
        .ok_or_else(|| "WebGPU computation backend is unavailable".to_string())?;
    let started = js_sys::Date::now();
    let value = engine.evaluate_diagnostic(options.into()).await;
    core.borrow_mut().engine = Some(engine);
    drop(progress);
    let value = value.map_err(|error| format!("{} failed: {error:?}", solver.label))?;
    if value.is_null() || value.is_undefined() {
        return Err(format!("{} was aborted", solver.label));
    }
    let bytes = value
        .dyn_into::<Uint8Array>()
        .map_err(|_| format!("{} returned invalid tensor bytes", solver.label))?
        .to_vec();
    let tensor = parse_tensor(&bytes)
        .ok_or_else(|| format!("{} returned a non-finite tensor sample", solver.label))?;
    let elapsed_us = (js_sys::Date::now() - started).max(0.001) * 1000.0;
    let per_point_us = elapsed_us / tensor.len().max(1) as f64;
    Ok((tensor, per_point_us))
}

fn tensor_norm(tensor: &[f64; 6]) -> f64 {
    (tensor[0] * tensor[0]
        + tensor[1] * tensor[1]
        + tensor[2] * tensor[2]
        + 2.0 * (tensor[3] * tensor[3] + tensor[4] * tensor[4] + tensor[5] * tensor[5]))
        .sqrt()
}

fn mean_relative_difference(a: &DiagnosticTensor, b: &DiagnosticTensor) -> f64 {
    let reference_scale = b
        .values
        .iter()
        .map(tensor_norm)
        .fold(0.0_f64, f64::max)
        .max(f64::MIN_POSITIVE);
    let denominator_floor = (reference_scale * 1e-8).max(f64::MIN_POSITIVE);
    let mut difference_squared = 0.0;
    let mut reference_squared = 0.0;
    for (left, right) in a.values.iter().zip(&b.values) {
        let difference = [
            left[0] - right[0],
            left[1] - right[1],
            left[2] - right[2],
            left[3] - right[3],
            left[4] - right[4],
            left[5] - right[5],
        ];
        let difference_norm = tensor_norm(&difference);
        let reference_norm = tensor_norm(right);
        difference_squared += difference_norm * difference_norm;
        reference_squared += reference_norm * reference_norm;
    }
    if difference_squared == 0.0 && reference_squared == 0.0 {
        0.0
    } else {
        difference_squared.sqrt() / reference_squared.sqrt().max(denominator_floor)
    }
}

// Keep only non-dominated points. A parameter sweep is not a Pareto frontier
// merely because its samples are joined in timing order; the rising arm of a
// V-shaped curve is dominated and must not be presented as optimal.
fn pareto_frontier(mut points: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    points.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.total_cmp(&right.1))
    });
    let mut best_error = f64::INFINITY;
    points.retain(|(_, error)| {
        if *error < best_error {
            best_error = *error;
            true
        } else {
            false
        }
    });
    points
}

fn solver_reference(solver: DiagnosticSolver) -> DiagnosticSolver {
    match solver.density {
        "uniform" => diagnostic_solvers()[0],
        // The continuous-density full-direction RT-FP path is a materially
        // better reference than the discretised Mascon field.
        "cauchy" => diagnostic_solvers()[3],
        // General alpha has no universal elliptic closed form.  Use the full
        // full-direction, GL16 CarlsonAlpha result as the browser reference;
        // swept candidates top out at 128 directions and therefore cannot
        // compare equal merely by selecting the same radial order.
        _ => diagnostic_solvers()[7],
    }
}

fn sweep_for(solver: DiagnosticSolver) -> Option<&'static [f64]> {
    match solver.algorithm {
        "mascon" => Some(&SWEEP_THETA),
        "rtfp" => Some(&SWEEP_DIRECTIONS),
        "carlson" if solver.density == "cauchy" => Some(&SWEEP_DIRECTIONS),
        "carlsonalpha" if solver.density == "fractional-cauchy" => Some(&SWEEP_QUADRATURE),
        _ => None,
    }
}

fn sweep_options(solver: DiagnosticSolver, value: f64) -> (Option<f64>, Option<f64>, Option<f64>) {
    match solver.algorithm {
        "mascon" => (None, None, Some(value)),
        "rtfp" | "carlson" => (Some(value), None, None),
        // Couple angular and radial refinement. Holding the direction count at
        // 64 made the angular error floor hide every GL refinement and produced
        // a misleading horizontal CarlsonAlpha plateau.
        "carlsonalpha" => (Some((value * 8.0).clamp(32.0, 128.0)), Some(value), None),
        _ => (None, None, None),
    }
}

fn stability_options(
    solver: DiagnosticSolver,
) -> (Option<f64>, Option<f64>, Option<f64>) {
    match (solver.algorithm, solver.density) {
        ("rtfp" | "carlson", "cauchy") => (Some(128.0), None, None),
        ("carlsonalpha", "fractional-cauchy") => (Some(128.0), Some(16.0), None),
        ("mascon", _) => (None, None, Some(0.1)),
        _ => (None, None, None),
    }
}

async fn run_pareto(core: &Rc<RefCell<Core>>) -> Result<(), String> {
    let solvers = diagnostic_solvers();
    let plotted: Vec<DiagnosticSolver> = solvers
        .into_iter()
        .filter(|solver| sweep_for(*solver).is_some())
        .collect();
    let total = plotted
        .iter()
        .filter_map(|solver| sweep_for(*solver))
        .map(|values| values.len())
        .sum::<usize>();
    let span = 100.0 / total.max(1) as f64;
    let height = core.borrow().standoff_mm.max(1.0);
    let mut series = Vec::new();
    let mut references: Vec<(&'static str, DiagnosticTensor)> = Vec::new();
    let mut progress = 0usize;
    for solver in plotted {
        let values = sweep_for(solver).unwrap();
        let reference_solver = solver_reference(solver);
        let reference = if let Some((_, tensor)) = references
            .iter()
            .find(|(density, _)| *density == solver.density)
        {
            tensor.clone()
        } else {
            let (tensor, _) = evaluate_tensor_sample(
                core,
                reference_solver,
                height,
                DIAGNOSTIC_FACES,
                None,
                None,
                None,
                progress as f64 * span,
                span,
            )
            .await?;
            references.push((solver.density, tensor.clone()));
            tensor
        };
        // Compile the pipeline, populate immutable caches and pay the first
        // allocation outside the timed sweep. The warm-up has one point, so it
        // is cheap and removes the cold-start kink that previously created a
        // spurious V-shaped timing curve.
        let (warm_direction, warm_quadrature, warm_theta) =
            sweep_options(solver, values[0]);
        let _ = evaluate_tensor_sample(
            core,
            solver,
            height,
            1,
            warm_direction,
            warm_quadrature,
            warm_theta,
            progress as f64 * span,
            0.0,
        )
        .await?;
        let mut points = Vec::new();
        for value in values {
            let (direction, quadrature, theta) = sweep_options(solver, *value);
            let (tensor, micros) = evaluate_tensor_sample(
                core,
                solver,
                height,
                DIAGNOSTIC_FACES,
                direction,
                quadrature,
                theta,
                progress as f64 * span,
                span,
            )
            .await?;
            progress += 1;
            points.push((
                micros.max(0.001),
                mean_relative_difference(&tensor, &reference).max(1e-15),
            ));
        }
        let points = pareto_frontier(points);
        series.push((
            solver.label.to_string(),
            solver.color.to_string(),
            "all".to_string(),
            diagnostic_dash(solver).to_string(),
            points,
        ));
    }
    publish_diagnostic(core, "pareto", "log", "log", series, "Only non-dominated sweep samples are drawn. Each solver receives a one-point untimed warm-up, removing asset/pipeline cold-start distortion. CarlsonAlpha couples angular and radial refinement so its GL sweep is not hidden behind a fixed angular-error floor. Timings remain batch wall time per point, including submission and readback; they are not isolated shader timestamps.".to_string(), "pareto-frontier.svg");
    Ok(())
}

async fn run_stability(core: &Rc<RefCell<Core>>) -> Result<(), String> {
    let solvers = diagnostic_solvers();
    let plotted: Vec<DiagnosticSolver> = solvers
        .into_iter()
        .filter(|solver| solver.algorithm != "werner")
        .collect();
    let total = plotted.len() * STABILITY_HEIGHTS_MM.len();
    let span = 100.0 / total.max(1) as f64;
    let mut series = Vec::new();
    let mut references: Vec<(&'static str, f64, DiagnosticTensor)> = Vec::new();
    let mut progress = 0usize;
    for solver in plotted {
        let reference_solver = solver_reference(solver);
        let mut points = Vec::new();
        for height in STABILITY_HEIGHTS_MM {
            let reference = if let Some((_, _, tensor)) =
                references.iter().find(|(density, cached_height, _)| {
                    *density == solver.density && *cached_height == height
                }) {
                tensor.clone()
            } else {
                let (tensor, _) = evaluate_tensor_sample(
                    core,
                    reference_solver,
                    height,
                    DIAGNOSTIC_FACES,
                    None,
                    None,
                    None,
                    progress as f64 * span,
                    span,
                )
                .await?;
                references.push((solver.density, height, tensor.clone()));
                tensor
            };
            let (direction, quadrature, theta) = stability_options(solver);
            let (tensor, _) = evaluate_tensor_sample(
                core,
                solver,
                height,
                DIAGNOSTIC_FACES,
                direction,
                quadrature,
                theta,
                progress as f64 * span,
                span,
            )
            .await?;
            progress += 1;
            points.push((
                height / 1000.0,
                mean_relative_difference(&tensor, &reference).max(1e-15),
            ));
        }
        series.push((
            solver.label.to_string(),
            solver.color.to_string(),
            diagnostic_density_label(solver.density).to_string(),
            diagnostic_dash(solver).to_string(),
            points,
        ));
    }
    publish_diagnostic(core, "stability", "log", "log", series, "Uniform density uses Werner. Cauchy candidates use the same 128-direction rule and are compared with full-direction RT-FP; fractional candidates use a fixed GL16 radial rule and are compared with full-direction GL16 CarlsonAlpha. References are cached per density and height. These f32 browser baselines test solver consistency, not native-f64 truth.".to_string(), "near-surface-stability.svg");
    Ok(())
}

async fn run_tensor_symmetry(core: &Rc<RefCell<Core>>) -> Result<(), String> {
    let solvers = diagnostic_solvers();
    let total = solvers.len() * CONSISTENCY_HEIGHTS_MM.len();
    let span = 100.0 / total.max(1) as f64;
    let mut series = Vec::new();
    let mut progress = 0usize;
    for solver in solvers {
        let mut points = Vec::new();
        for height in CONSISTENCY_HEIGHTS_MM {
            let (tensor, _) = evaluate_tensor_sample(
                core,
                solver,
                height,
                DIAGNOSTIC_FACES,
                None,
                None,
                None,
                progress as f64 * span,
                span,
            )
            .await?;
            progress += 1;
            let trace_squared = tensor
                .values
                .iter()
                .map(|value| (value[0] + value[1] + value[2]).powi(2))
                .sum::<f64>();
            let norm_squared = tensor
                .values
                .iter()
                .map(|value| tensor_norm(value).powi(2))
                .sum::<f64>();
            // All diagnostic tensors originate in f32. Values below f32 epsilon
            // are censored at the representable precision floor instead of
            // being advertised as an f64/machine-precision measurement.
            let trace_ratio = (trace_squared.sqrt()
                / norm_squared.sqrt().max(f64::MIN_POSITIVE))
            .max(f32::EPSILON as f64);
            points.push((height / 1000.0, trace_ratio));
        }
        series.push((
            solver.label.to_string(),
            solver.color.to_string(),
            diagnostic_density_label(solver.density).to_string(),
            diagnostic_dash(solver).to_string(),
            points,
        ));
    }
    publish_diagnostic(core, "symmetry", "log", "log", series, "This is the aggregate exterior Laplace residual sqrt(sum trace(H)^2)/sqrt(sum ||H||F^2), not an H_ij-H_ji test: six-component storage is symmetric by construction. Values are censored at f32 epsilon; the former ~2.5e-16 line was therefore not a defensible machine-precision claim. A nine-component or finite-difference curl diagnostic is still required for an independent conservative-field test.".to_string(), "tensor-consistency.svg");
    Ok(())
}

fn diagnostic_density_label(mode: &str) -> &'static str {
    match mode {
        "uniform" => "Uniform",
        "cauchy" => "Cauchy",
        _ => "Fractional Cauchy",
    }
}

fn diagnostic_dash(solver: DiagnosticSolver) -> &'static str {
    match solver.algorithm {
        "mascon" => "dashed",
        "rtfp" => "dotted",
        "carlson" => "dashdot",
        "carlsonalpha" => "longdash",
        _ => "solid",
    }
}

#[allow(clippy::type_complexity)]
fn publish_diagnostic(
    core: &Rc<RefCell<Core>>,
    kind: &str,
    x_scale: &str,
    y_scale: &str,
    series: Vec<(String, String, String, String, Vec<(f64, f64)>)>,
    summary: String,
    filename: &str,
) {
    let ui = core.borrow().ui.diagnostics.clone();
    set_str(&ui, "kind", kind);
    set_str(&ui, "xScale", x_scale);
    set_str(&ui, "yScale", y_scale);
    set_str(&ui, "summary", &summary);
    set_str(&ui, "downloadName", filename);
    let series_array = Array::new();
    for (name, color, facet, dash, points) in series {
        let item = Object::new();
        set_str(&item, "name", &name);
        set_str(&item, "color", &color);
        set_str(&item, "facet", &facet);
        set_str(&item, "dash", &dash);
        let point_array = Array::new();
        for (x, y) in points {
            let point = Object::new();
            set_f64(&point, "x", x);
            set_f64(&point, "y", y);
            point_array.push(&point);
        }
        set(&item, "points", point_array.into());
        series_array.push(&item);
    }
    set(&ui, "series", series_array.into());
}
