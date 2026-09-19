// ---------------------------------------------------------------------------
// Standoff and labels
// ---------------------------------------------------------------------------

fn set_standoff_ui(core: &Rc<RefCell<Core>>, mm: f64, persist: bool) {
    let clamped = clamp_standoff(mm);
    let (ui, position, value) = {
        let mut state = core.borrow_mut();
        state.standoff_mm = clamped;
        (state.ui.clone(), mm_to_pos(clamped), fmt_mm(clamped))
    };
    ui.standoff_position(position);
    set_str(&ui.standoff, "value", &value);
    if persist && let Some(url) = current_url() {
        url.search_params().set("standoff", &format!("{clamped}"));
        replace_history(&url.href());
    }
}

fn set_standoff_record_label(core: &Rc<RefCell<Core>>, mm: f64) {
    let text = if mm.is_finite() && (STANDOFF_MIN_MM..=STANDOFF_MAX_MM).contains(&mm) {
        format!("latest {}", fmt_mm(mm))
    } else {
        String::new()
    };
    let ui = core.borrow().ui.clone();
    set_str(&ui.standoff, "record", &text);
}

fn sync_standoff_record_label(core: &Rc<RefCell<Core>>, mm: f64) {
    if core.borrow().standoff_dragging {
        return;
    }
    set_standoff_record_label(core, mm);
}

fn algo_short(spec: &Algo, mode: &str, selectable: bool) -> String {
    if !selectable {
        return spec.short.to_string();
    }
    match spec.name {
        "Mascon" => format!("Mascon · {} density", density_label(mode)),
        "RT-FP" => {
            if mode == "constant" {
                "RT-FP · uniform density".to_string()
            } else {
                format!("RT-FP · {} density", density_label(mode))
            }
        }
        "Carlson" => {
            if mode == "constant" {
                "Carlson · uniform density".to_string()
            } else {
                format!("Carlson · {} density", density_label(mode))
            }
        }
        _ => format!("CarlsonAlpha · {} density", density_label(mode)),
    }
}

fn algo_title(spec: &Algo, mode: &str, selectable: bool) -> String {
    if !selectable {
        return spec.title.to_string();
    }
    match spec.name {
        "Mascon" => {
            if mode == "elliptic" {
                "Mascon · 5.2 m voxel Barnes–Hut tree · fractional Cauchy density".to_string()
            } else {
                "Mascon · 5.2 m voxel Barnes–Hut tree · Cauchy density".to_string()
            }
        }
        "RT-FP" => {
            if mode == "constant" {
                "RT-FP · uniform density ρ₀ · analytic near field only".to_string()
            } else {
                format!(
                    "RT-FP · {} density · analytic near field + GPU quadrature",
                    density_label(mode)
                )
            }
        }
        "Carlson" => {
            if mode == "constant" {
                "Carlson · uniform density · exact polyhedral boundary".to_string()
            } else {
                format!(
                    "Carlson · {} density · Carlson radial residual",
                    density_label(mode)
                )
            }
        }
        _ => {
            if mode == "constant" {
                "CarlsonAlpha · uniform density · exact polyhedral boundary".to_string()
            } else {
                "CarlsonAlpha · general-α Cauchy · Carlson radial residual".to_string()
            }
        }
    }
}

fn apply_algo_labels(core: &Rc<RefCell<Core>>) {
    let (ui, short, title, button, show_density, density_text) = {
        let state = core.borrow();
        let spec = state.spec();
        let selectable = state.density_selectable();
        let next = state.next_density_mode();
        (
            state.ui.clone(),
            algo_short(spec, &state.density_mode, selectable),
            algo_title(spec, &state.density_mode, selectable),
            spec.next.to_string(),
            selectable,
            format!("Switch to {} density", density_label(&next)),
        )
    };
    set_str(&ui.algo, "label", &short);
    set_str(&ui.algo, "button", &button);
    set_str(&ui.algo, "panelTitle", &title);
    set_bool(&ui.density, "hidden", !show_density);
    if show_density {
        set_str(&ui.density, "text", &density_text);
    }
}

fn publish_saved_selection(core: &Rc<RefCell<Core>>, done: bool) {
    let (ui, algo, algorithm, density, mm) = {
        let state = core.borrow();
        let spec = state.spec();
        (
            state.ui.clone(),
            state.algo.clone(),
            spec.name.to_string(),
            state.result_mode(),
            state.standoff_mm,
        )
    };
    let current = field(&ui.saved, "current");
    set_str(&current, "algo", &algo);
    set_str(&current, "algorithm", &algorithm);
    set_str(&current, "density", &density);
    set_f64(&current, "standoffMm", mm);
    set_bool(&current, "done", done);
    let state = core.borrow();
    update_saved_flags(&state);
}
