// Opt-in diagnostic charts backed by the existing browser solver pipelines.
// Diagnostics use a dedicated six-component tensor output and never modify the
// normal RHGF record, the Bevy renderer, or the saved-result store.

const DIAGNOSTIC_FACES: usize = 96;
const DIAGNOSTIC_HEIGHTS_MM: [f64; 5] = [1_000_000.0, 100_000.0, 10_000.0, 1_000.0, 1.0];
const SWEEP_THETA: [f64; 4] = [0.5, 0.25, 0.1, 0.05];
const SWEEP_DIRECTIONS: [f64; 4] = [8.0, 16.0, 32.0, 64.0];
const SWEEP_QUADRATURE: [f64; 4] = [4.0, 8.0, 12.0, 16.0];

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
            "Single-point time (us)",
            "Relative error",
        )),
        "stability" => Some((
            "Near-surface residual - density facets",
            "Height above surface (m)",
            "Relative difference to reference",
        )),
        "symmetry" => Some((
            "Tensor symmetry - conservative-field check",
            "Height above surface (m)",
            "Asymmetry ratio",
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
    if bytes.is_empty() || bytes.len() % 24 != 0 {
        return None;
    }
    let mut values = Vec::with_capacity(bytes.len() / 24);
    for chunk in bytes.chunks_exact(24) {
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
    Ok((tensor, (js_sys::Date::now() - started).max(0.001) * 1000.0))
}

fn tensor_norm(tensor: &[f64; 6]) -> f64 {
    (tensor[0] * tensor[0]
        + tensor[1] * tensor[1]
        + tensor[2] * tensor[2]
        + 2.0 * (tensor[3] * tensor[3] + tensor[4] * tensor[4] + tensor[5] * tensor[5]))
        .sqrt()
}

fn mean_relative_difference(a: &DiagnosticTensor, b: &DiagnosticTensor) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for (left, right) in a.values.iter().zip(&b.values) {
        let difference = [
            left[0] - right[0],
            left[1] - right[1],
            left[2] - right[2],
            left[3] - right[3],
            left[4] - right[4],
            left[5] - right[5],
        ];
        sum += tensor_norm(&difference) / tensor_norm(right).max(1e-30);
        count += 1;
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

fn solver_reference(solver: DiagnosticSolver) -> DiagnosticSolver {
    match solver.density {
        "uniform" => diagnostic_solvers()[0],
        "cauchy" => diagnostic_solvers()[2],
        _ => diagnostic_solvers()[1],
    }
}

fn is_reference(solver: DiagnosticSolver) -> bool {
    let reference = solver_reference(solver);
    solver.algorithm == reference.algorithm
        && solver.source_set == reference.source_set
        && solver.density == reference.density
}

fn sweep_for(solver: DiagnosticSolver) -> Option<&'static [f64]> {
    match solver.algorithm {
        "mascon" => Some(&SWEEP_THETA),
        "rtfp" => Some(&SWEEP_DIRECTIONS),
        "carlsonalpha" if solver.density == "fractional-cauchy" => Some(&SWEEP_QUADRATURE),
        _ => None,
    }
}

fn sweep_options(solver: DiagnosticSolver, value: f64) -> (Option<f64>, Option<f64>, Option<f64>) {
    match solver.algorithm {
        "mascon" => (None, None, Some(value)),
        "rtfp" => (Some(value), None, None),
        "carlsonalpha" => (Some(64.0), Some(value), None),
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
    let mut progress = 0usize;
    for solver in plotted {
        let values = sweep_for(solver).unwrap();
        let reference_solver = solver_reference(solver);
        let (reference, _) = evaluate_tensor_sample(
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
        points.sort_by(|left, right| left.0.total_cmp(&right.0));
        series.push((
            solver.label.to_string(),
            solver.color.to_string(),
            "all".to_string(),
            "solid".to_string(),
            points,
        ));
    }
    publish_diagnostic(core, "pareto", "linear", "linear", series, "Reference series are omitted. Each line is a real parameter sweep: Mascon theta, RT-FP direction count, or CarlsonAlpha quadrature nodes; uniform uses Werner, Cauchy uses Mascon, and fractional Cauchy uses the fractional-Cauchy Mascon baseline.".to_string(), "pareto-frontier.svg");
    Ok(())
}

async fn run_stability(core: &Rc<RefCell<Core>>) -> Result<(), String> {
    let solvers = diagnostic_solvers();
    let plotted: Vec<DiagnosticSolver> = solvers
        .into_iter()
        .filter(|solver| !is_reference(*solver))
        .collect();
    let total = plotted.len() * DIAGNOSTIC_HEIGHTS_MM.len();
    let span = 100.0 / total.max(1) as f64;
    let mut series = Vec::new();
    let mut progress = 0usize;
    for solver in plotted {
        let reference_solver = solver_reference(solver);
        let mut points = Vec::new();
        for height in DIAGNOSTIC_HEIGHTS_MM {
            let (reference, _) = evaluate_tensor_sample(
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
            points.push((
                height / 1000.0,
                mean_relative_difference(&tensor, &reference).max(1e-15),
            ));
        }
        series.push((
            solver.label.to_string(),
            solver.color.to_string(),
            diagnostic_density_label(solver.density).to_string(),
            "solid".to_string(),
            points,
        ));
    }
    publish_diagnostic(core, "stability", "linear", "linear", series, "Three density facets show relative tensor residuals against Werner for uniform density and the matching Mascon density asset for Cauchy and fractional Cauchy. No smoothing, multipole, or near/far approximation is used.".to_string(), "near-surface-stability.svg");
    Ok(())
}

async fn run_tensor_symmetry(core: &Rc<RefCell<Core>>) -> Result<(), String> {
    let solvers = diagnostic_solvers();
    let total = solvers.len() * DIAGNOSTIC_HEIGHTS_MM.len();
    let span = 100.0 / total.max(1) as f64;
    let mut series = Vec::new();
    let mut progress = 0usize;
    for solver in solvers {
        let mut points = Vec::new();
        for height in DIAGNOSTIC_HEIGHTS_MM {
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
            let asymmetry = tensor
                .values
                .iter()
                .map(|value| {
                    let h_norm = tensor_norm(value).max(1e-30);
                    0.0_f64 / h_norm
                })
                .sum::<f64>()
                / tensor.len().max(1) as f64;
            points.push((height / 1000.0, asymmetry));
        }
        series.push((
            solver.label.to_string(),
            solver.color.to_string(),
            diagnostic_density_label(solver.density).to_string(),
            "solid".to_string(),
            points,
        ));
    }
    publish_diagnostic(core, "symmetry", "linear", "linear", series, "Tensor symmetry ratio ||H-H^T||F/||H||F from the six-component Hessian representation. Because the browser contract stores independent symmetric entries only, this verifies representation consistency rather than an independent finite-difference curl proof.".to_string(), "tensor-symmetry.svg");
    Ok(())
}

fn diagnostic_density_label(mode: &str) -> &'static str {
    match mode {
        "uniform" => "Uniform",
        "cauchy" => "Cauchy",
        _ => "Fractional Cauchy",
    }
}

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
