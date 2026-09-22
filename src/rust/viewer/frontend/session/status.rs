// ---------------------------------------------------------------------------
// Status rendering
// ---------------------------------------------------------------------------

fn render_status(core: &Rc<RefCell<Core>>, status: Status) {
    let pct = if status.percent.is_finite() {
        status.percent.clamp(0.0, 100.0)
    } else {
        0.0
    };
    core.borrow().ui.bar_percent(pct);
    let mut line = if status.message.is_empty() {
        status.state.clone()
    } else {
        status.message.clone()
    };
    if status.total > 0 {
        line.push_str(&format!(" · {pct:.1}%"));
    }
    if !status.error.is_empty() {
        line.push_str(&format!(" · {}", status.error));
    }
    if status.standoff_mm.is_finite() {
        sync_standoff_record_label(core, status.standoff_mm);
    }
    if let Some(stats) = core.borrow().scalar_stats.clone() {
        line.push_str(&stats.describe());
    }

    let (ui, running, compute_running, name) = {
        let mut state = core.borrow_mut();
        let selectable = state.density_selectable();
        if selectable
            && state
                .spec()
                .density_modes
                .contains(&status.density_mode.as_str())
        {
            state.density_mode = status.density_mode.clone();
        }
        (
            state.ui.clone(),
            status.state == "running",
            state.compute_running,
            state.spec().name.to_string(),
        )
    };

    ui.status(&line);
    if running {
        show_compute_indicator(core, pct.min(99.0), &name);
    } else {
        hide_compute_indicator(core);
    }
    set_bool(&ui.bake, "hidden", false);
    set_bool(&ui.bake, "disabled", compute_running);
    set_str(
        &ui.bake,
        "text",
        if compute_running {
            "Computing…"
        } else {
            "Recompute"
        },
    );
    set_bool(&ui.reload, "hidden", status.state != "done");
    set_bool(&ui.density, "disabled", running);
    apply_algo_labels(core);
    publish_saved_selection(core, status.done && pct >= 100.0);
    core.borrow_mut().last_status = Some(status.clone());
    if status.state != "done" {
        core.borrow_mut().scalar_stats = None;
    }
    let core_for_compare = core.clone();
    spawn_local(async move {
        update_algorithm_compare(&core_for_compare, &status).await;
    });
}

fn set_bar_percent(core: &Rc<RefCell<Core>>, percent: f64) {
    let bounded = if percent.is_finite() {
        percent.clamp(0.0, 100.0)
    } else {
        0.0
    };
    core.borrow().ui.bar_percent(bounded);
}

fn show_compute_indicator(core: &Rc<RefCell<Core>>, percent: f64, label: &str) {
    let bounded = if percent.is_finite() {
        percent.clamp(0.0, 99.0)
    } else {
        0.0
    };
    let text = if label.is_empty() {
        format!("Computing · {bounded:.1}%")
    } else {
        format!("Computing · {label} · {bounded:.1}%")
    };
    core.borrow().ui.compute_indicator(true, &text);
}

fn hide_compute_indicator(core: &Rc<RefCell<Core>>) {
    core.borrow().ui.compute_indicator(false, "Computing · 0.0%");
}

/// Reset the renderer before a new algorithm/density result arrives. This is a
/// separate event from the bake payload, so the first new block cannot erase it.
fn reset_renderer_state(core: &Rc<RefCell<Core>>) {
    crate::reset_bake_paint();
    core.borrow_mut().scalar_stats = None;
}
