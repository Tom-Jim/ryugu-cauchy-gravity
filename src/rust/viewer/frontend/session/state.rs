// ---------------------------------------------------------------------------
// Session state
// ---------------------------------------------------------------------------

/// The Vue store the controller writes into, plus the sub-objects it touches.
#[derive(Clone)]
struct Ui {
    root: JsValue,
    message: JsValue,
    compute: JsValue,
    mobile: JsValue,
    algo: JsValue,
    standoff: JsValue,
    bake: JsValue,
    density: JsValue,
    reload: JsValue,
    compare: JsValue,
    saved: JsValue,
    diagnostics: JsValue,
}

impl Ui {
    fn new(viewer_ui: &JsValue, saved: &JsValue, diagnostics: &JsValue) -> Ui {
        let buttons = field(viewer_ui, "buttons");
        Ui {
            root: viewer_ui.clone(),
            message: field(viewer_ui, "message"),
            compute: field(viewer_ui, "compute"),
            mobile: field(viewer_ui, "mobile"),
            algo: field(viewer_ui, "algo"),
            standoff: field(viewer_ui, "standoff"),
            bake: field(&buttons, "bake"),
            density: field(&buttons, "density"),
            reload: field(&buttons, "reload"),
            compare: field(viewer_ui, "compare"),
            saved: saved.clone(),
            diagnostics: diagnostics.clone(),
        }
    }

    fn status(&self, text: &str) {
        set_str(&self.root, "status", text);
    }

    fn bar_percent(&self, percent: f64) {
        set_f64(&self.root, "barPercent", percent.clamp(0.0, 100.0));
    }

    fn show_message(&self, text: &str) {
        set_bool(&self.message, "visible", true);
        set_str(&self.message, "text", text);
    }

    fn compute_indicator(&self, visible: bool, text: &str) {
        set_bool(&self.compute, "visible", visible);
        set_str(&self.compute, "text", text);
    }

    fn standoff_position(&self, pos: f64) {
        set_f64(&self.standoff, "pos", pos);
    }
}

/// One runtime result, kept only in this tab.
#[derive(Clone, Debug)]
struct ComputeResult {
    bytes: Vec<u8>,
    parsed: ParsedRecord,
    algo: String,
    algorithm: String,
    density_mode: String,
    standoff_mm: f64,
    revision: u64,
}

#[derive(Clone, Copy, Debug)]
struct Job {
    mm: f64,
    preserve_window: bool,
    force_compute: bool,
    generation: u64,
}

#[derive(Clone, Debug, Default)]
struct Status {
    state: String,
    done: bool,
    message: String,
    percent: f64,
    total: usize,
    standoff_mm: f64,
    density_mode: String,
    error: String,
}

struct Core {
    base_url: String,
    ui: Ui,
    store: Rc<Store>,
    engine: Option<ComputeEngine>,
    pending_model: Option<Vec<u8>>,
    pending_cauchy: Option<String>,
    pending_elliptic: Option<String>,
    algo: String,
    density_mode: String,
    density_selections: HashMap<String, String>,
    standoff_mm: f64,
    standoff_dragging: bool,
    record_generation: u64,
    stats_generation: Option<u64>,
    compute_running: bool,
    diagnostic_running: bool,
    compute_target: Option<Job>,
    compute_generation: u64,
    compute_abort: Option<web_sys::AbortController>,
    latest_results: HashMap<String, ComputeResult>,
    latest_revision: u64,
    compare_key: String,
    last_status: Option<Status>,
    scalar_stats: Option<ScalarStats>,
    saved_items: Vec<SavedRow>,
    saved_current_id: Option<String>,
    memory_guard_running: bool,
    tab_id: String,
    tab_identity: String,
    channel: Option<web_sys::BroadcastChannel>,
}

impl Core {
    fn spec(&self) -> &'static Algo {
        algo_spec(&self.algo)
    }

    fn density_selectable(&self) -> bool {
        self.spec().density_modes.len() > 1
    }

    /// The density model a result is keyed by.
    fn result_mode(&self) -> String {
        if self.density_selectable() {
            self.density_mode.clone()
        } else {
            "uniform".to_string()
        }
    }

    fn result_key_for(&self, algo: &str, mode: &str) -> String {
        let spec = algo_spec(algo);
        let resolved = if spec.density_modes.len() > 1 {
            mode
        } else {
            "uniform"
        };
        format!("{algo}:{resolved}")
    }

    fn active_key(&self) -> String {
        let mode = self.result_mode();
        self.result_key_for(&self.algo, &mode)
    }

    fn latest_result_for(&self, algo: &str, mode: &str) -> Option<&ComputeResult> {
        let key = self.result_key_for(algo, mode);
        self.latest_results.get(&key)
    }

    /// Source buffer set to load for one `(algorithm, density)` pair.
    fn compute_source_set(&self, algo: &str, mode: &str) -> &'static str {
        if algo == "werner" {
            return "constant";
        }
        let spec = algo_spec(algo);
        let selected = if spec.density_modes.len() > 1 {
            mode
        } else {
            spec.default_density
        };
        match selected {
            "elliptic" => "elliptic",
            "constant" => "constant",
            _ => "cauchy",
        }
    }

    fn next_density_mode(&self) -> String {
        let modes = self.spec().density_modes;
        if modes.is_empty() {
            return self.density_mode.clone();
        }
        let index = modes
            .iter()
            .position(|mode| *mode == self.density_mode)
            .map(|index| index + 1)
            .unwrap_or(0);
        modes[index % modes.len()].to_string()
    }
}
