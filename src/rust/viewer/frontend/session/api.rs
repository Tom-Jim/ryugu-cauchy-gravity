// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct Session {
    core: Rc<RefCell<Core>>,
}

#[wasm_bindgen]
impl Session {
    /// `new Session(viewerUi, saved, baseUrl)`.
    #[wasm_bindgen(constructor)]
    pub fn new(viewer_ui: JsValue, saved: JsValue, base_url: Option<String>) -> Session {
        let core = Core {
            base_url: base_url.unwrap_or_else(|| "./".to_string()),
            ui: Ui::new(&viewer_ui, &saved),
            store: Rc::new(Store::new()),
            engine: None,
            algo: "werner".to_string(),
            density_mode: "cauchy".to_string(),
            density_selections: HashMap::new(),
            standoff_mm: STANDOFF_DEFAULT_MM,
            standoff_dragging: false,
            record_generation: 0,
            stats_generation: None,
            compute_running: false,
            compute_target: None,
            compute_generation: 0,
            compute_abort: None,
            latest_results: HashMap::new(),
            latest_revision: 0,
            compare_key: String::new(),
            last_status: None,
            scalar_stats: None,
            saved_items: Vec::new(),
            saved_current_id: None,
            memory_guard_running: false,
            tab_id: String::new(),
            tab_identity: String::new(),
            channel: None,
        };
        Session {
            core: Rc::new(RefCell::new(core)),
        }
    }

    /// Load the viewer and start the recomputation engine.
    pub async fn boot(&self) -> Result<(), JsValue> {
        let core = self.core.clone();
        if !enforce_single_tab(&core) {
            return Ok(());
        }
        init_viewer_state(&core);
        if !webgpu_available() {
            let ui = core.borrow().ui.clone();
            ui.show_message("WebGPU is required (navigator.gpu is unavailable in this browser).");
            return Ok(());
        }
        watch_unhandled_errors();
        if let Err(error) = start_engine(&core).await {
            let ui = core.borrow().ui.clone();
            ui.compute_indicator(false, "");
            ui.bar_percent(0.0);
            set_bool(&ui.bake, "disabled", true);
            set_str(&ui.bake, "text", "Unavailable");
            ui.show_message(&error);
        }
        Ok(())
    }

    pub fn on_algo(&self) {
        let core = self.core.clone();
        let current = core.borrow().algo.clone();
        let index = ALGO_ORDER
            .iter()
            .position(|key| *key == current)
            .unwrap_or(0);
        let next = ALGO_ORDER[(index + 1) % ALGO_ORDER.len()].to_string();
        select_algorithm(&core, &next);
    }

    pub fn on_bake(&self) {
        start_bake(&self.core.clone());
    }

    pub fn on_density(&self) {
        let core = self.core.clone();
        let label = {
            let mut state = core.borrow_mut();
            let mode = state.next_density_mode();
            let algo = state.algo.clone();
            state.density_mode = mode.clone();
            state.density_selections.insert(algo, mode.clone());
            state.record_generation += 1;
            density_label(&mode)
        };
        apply_algo_labels(&core);
        publish_saved_selection(&core, false);
        core.borrow_mut().compare_key.clear();
        core.borrow()
            .ui
            .status(&format!("Switched to {label} density…"));
        let mm = core.borrow().standoff_mm;
        request_static_compute(&core, mm, true, false);
    }

    /// Re-paint the latest in-memory result with a shared colour window.
    pub fn on_reload(&self) {
        let core = self.core.clone();
        let ui = core.borrow().ui.clone();
        set_bool(&ui.reload, "disabled", true);
        core.borrow_mut().compare_key.clear();
        core.borrow_mut().scalar_stats = None;
        ui.status("Reloading the latest temporary result…");
        spawn_local(async move {
            let key = core.borrow().active_key();
            if core.borrow().latest_results.contains_key(&key) {
                let generation = core.borrow().record_generation;
                let standoff = core.borrow().latest_results[&key].standoff_mm;
                let message = format!("Temporary result · {}", fmt_mm(standoff));
                render_completed_result(&core, &key, true, generation, &message).await;
            } else {
                let ui = core.borrow().ui.clone();
                ui.status("The current algorithm has no temporary result to render.");
            }
            let ui = core.borrow().ui.clone();
            set_bool(&ui.reload, "disabled", false);
        });
    }

    /// Dragging only moves the handle and the readout: no computation starts,
    /// and the URL is written once on release so a refresh restores the height
    /// the user actually stopped on.
    pub fn on_standoff_input(&self, pos: f64) {
        let core = self.core.clone();
        core.borrow_mut().standoff_dragging = true;
        set_standoff_ui(&core, pos_to_mm(pos), false);
    }

    pub fn on_standoff_commit(&self) {
        let core = self.core.clone();
        core.borrow_mut().standoff_dragging = false;
        let mm = core.borrow().standoff_mm;
        apply_standoff(&core, mm);
    }

    pub fn on_mobile_dismiss(&self) {
        let core = self.core.clone();
        let ui = core.borrow().ui.clone();
        set_bool(&ui.mobile, "visible", false);
    }

    /// Save the current in-memory result as the durable 100 % record.
    pub fn on_save_current(&self) {
        let core = self.core.clone();
        spawn_local(async move {
            save_current(&core).await;
        });
    }

    pub fn on_download_current(&self) {
        let core = self.core.clone();
        spawn_local(async move { download_current(&core).await; });
    }

    pub fn on_delete_current(&self) {
        let core = self.core.clone();
        let id = core.borrow().saved_current_id.clone();
        spawn_local(async move {
            delete_item(&core, id.as_deref()).await;
        });
    }

    pub fn on_delete_item(&self, id: String) {
        let core = self.core.clone();
        spawn_local(async move {
            delete_item(&core, Some(&id)).await;
        });
    }

    /// Fill the saved-results panel from the browser's own store.
    pub fn refresh_saved(&self) {
        let core = self.core.clone();
        spawn_local(async move {
            refresh_saved(&core).await;
        });
    }
}
