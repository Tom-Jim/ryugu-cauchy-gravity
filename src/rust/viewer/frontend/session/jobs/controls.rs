fn start_bake(core: &Rc<RefCell<Core>>) {
    cancel_compute(core);
    core.borrow_mut().compare_key.clear();
    let mm = core.borrow().standoff_mm;
    request_static_compute(core, mm, false, true);
}

/// Apply a new observation height to whichever algorithm is on screen.
fn apply_standoff(core: &Rc<RefCell<Core>>, mm: f64) {
    set_standoff_ui(core, mm, true);
    publish_saved_selection(core, false);
    // The visible result now belongs to the previous height.
    set_standoff_record_label(core, f64::NAN);
    let mm = core.borrow().standoff_mm;
    request_static_compute(core, mm, false, false);
}

fn select_algorithm(core: &Rc<RefCell<Core>>, next: &str) {
    let current = core.borrow().algo.clone();
    if next == current || !ALGO_ORDER.contains(&next) {
        return;
    }
    cancel_compute(core);
    {
        let mut state = core.borrow_mut();
        state.record_generation += 1;
        state.compute_generation += 1;
        state.compute_target = None;
        state.algo = next.to_string();
        state.density_mode = state
            .density_selections
            .get(next)
            .cloned()
            .unwrap_or_else(|| algo_spec(next).default_density.to_string());
        state.compare_key.clear();
        state.scalar_stats = None;
    }
    let (ui, compare, name) = {
        let state = core.borrow();
        (
            state.ui.clone(),
            state.spec().compare,
            state.spec().name.to_string(),
        )
    };
    set_bool(&ui.compare, "visible", compare);
    set_str(&ui.compare, "text", "");
    set_storage(ALGO_KEY, next);
    if let Some(url) = current_url() {
        url.search_params().set("algo", next);
        url.search_params().delete("fresh");
        // Keep one browser-history entry: Back leaves the viewer instead of
        // restoring the previous algorithm and creating another WebGPU context.
        replace_history(&url.href());
    }
    apply_algo_labels(core);
    publish_saved_selection(core, false);
    ui.status(&format!("Loading {name}…"));
    let mm = core.borrow().standoff_mm;
    request_static_compute(core, mm, false, false);
}
