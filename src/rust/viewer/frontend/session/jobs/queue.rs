fn request_static_compute(
    core: &Rc<RefCell<Core>>,
    mm: f64,
    preserve_window: bool,
    force_compute: bool,
) {
    let generation = {
        let mut state = core.borrow_mut();
        state.record_generation += 1;
        state.record_generation
    };
    cancel_compute(core);
    restore_original_model(core);
    hide_compute_indicator(core);
    set_bar_percent(core, 0.0);
    set_standoff_ui(core, mm, true);
    set_standoff_record_label(core, f64::NAN);
    {
        let mut state = core.borrow_mut();
        state.compare_key.clear();
        state.scalar_stats = None;
    }
    if core.borrow().engine.is_none() {
        core.borrow()
            .ui
            .status("Browser recomputation is not initialized");
        return;
    }
    queue_compute(
        core,
        Job {
            mm,
            preserve_window,
            force_compute,
            generation,
        },
    );
}

fn cancel_compute(core: &Rc<RefCell<Core>>) {
    let abort = {
        let mut state = core.borrow_mut();
        state.compute_target = None;
        state.compute_generation += 1;
        state.compute_abort.take()
    };
    if let Some(controller) = abort {
        controller.abort();
    }
}

fn queue_compute(core: &Rc<RefCell<Core>>, job: Job) {
    let start = {
        let mut state = core.borrow_mut();
        if state.engine.is_none() {
            return;
        }
        state.compute_target = Some(job);
        if state.compute_running {
            false
        } else {
            state.compute_running = true;
            true
        }
    };
    if start {
        let core = core.clone();
        spawn_local(async move {
            run_compute_queue(core).await;
        });
    }
}

async fn run_compute_queue(core: Rc<RefCell<Core>>) {
    {
        let ui = core.borrow().ui.clone();
        set_bool(&ui.bake, "disabled", true);
        set_str(&ui.bake, "text", "Computing…");
    }
    loop {
        let job = core.borrow_mut().compute_target.take();
        let Some(job) = job else {
            break;
        };
        let token = core.borrow().compute_generation;
        let controller = web_sys::AbortController::new().ok();
        let signal = controller
            .as_ref()
            .map(|controller| JsValue::from(controller.signal()));
        core.borrow_mut().compute_abort = controller.clone();
        if let Err(error) = run_job(&core, &job, token, signal).await
            && token == core.borrow().compute_generation
        {
            hide_compute_indicator(&core);
            core.borrow()
                .ui
                .status(&format!("Full surface recompute failed: {error}"));
        }
        let mut state = core.borrow_mut();
        let owns_abort = state
            .compute_abort
            .as_ref()
            .zip(controller.as_ref())
            .is_some_and(|(active, ours)| active == ours);
        if owns_abort {
            state.compute_abort = None;
        }
    }
    core.borrow_mut().compute_running = false;
    hide_compute_indicator(&core);
    let ui = core.borrow().ui.clone();
    set_bool(&ui.bake, "disabled", false);
    set_str(&ui.bake, "text", "Recompute");
}

/// The density key a persisted row is filed under.
fn mode_key(algo: &str, mode: &str) -> String {
    if algo_spec(algo).density_modes.len() > 1 {
        mode.to_string()
    } else {
        "uniform".to_string()
    }
}

fn remember_latest_result(
    core: &Rc<RefCell<Core>>,
    bytes: Vec<u8>,
    parsed: ParsedRecord,
    algo: &str,
    algorithm: &str,
    density_mode: &str,
) -> String {
    let mut state = core.borrow_mut();
    state.latest_revision += 1;
    let revision = state.latest_revision;
    let key = state.result_key_for(algo, density_mode);
    let resolved_mode = mode_key(algo, density_mode);
    let standoff_mm = parsed.standoff_mm;
    state.latest_results.insert(
        key.clone(),
        ComputeResult {
            bytes,
            parsed,
            algo: algo.to_string(),
            algorithm: algorithm.to_string(),
            density_mode: resolved_mode,
            standoff_mm,
            revision,
        },
    );
    key
}
