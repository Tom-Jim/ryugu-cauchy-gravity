async fn render_completed_result(
    core: &Rc<RefCell<Core>>,
    key: &str,
    preserve_window: bool,
    generation: u64,
    message: &str,
) -> bool {
    if generation != core.borrow().record_generation {
        return false;
    }
    let Some((standoff, total, density_mode)) = ({
        let state = core.borrow();
        state.latest_results.get(key).map(|result| {
            (
                result.standoff_mm,
                result.parsed.total,
                result.density_mode.clone(),
            )
        })
    }) else {
        return false;
    };
    let label = if message.is_empty() {
        format!("Latest temporary result · {}", fmt_mm(standoff))
    } else {
        message.to_string()
    };
    show_compute_indicator(core, 99.0, &format!("Rendering {}", fmt_mm(standoff)));
    core.borrow()
        .ui
        .status(&format!("{label} · rendering · 99.0%"));
    {
        let state = core.borrow();
        if let Some(result) = state.latest_results.get(key) {
            if preserve_window {
                crate::push_bake_update_preserving_window(&result.bytes);
            } else {
                crate::push_bake_update(&result.bytes);
            }
        }
    }
    wait_for_rendered_frames(2).await;
    if generation != core.borrow().record_generation {
        return false;
    }
    core.borrow_mut().scalar_stats = None;
    set_standoff_record_label(core, standoff);
    refresh_bake_scalar_stats(core, key);
    if generation != core.borrow().record_generation {
        return false;
    }
    hide_compute_indicator(core);
    render_status(
        core,
        Status {
            state: "done".to_string(),
            done: true,
            message: format!(
                "{label}{}",
                if preserve_window {
                    " · shared density colour scale"
                } else {
                    ""
                }
            ),
            percent: 100.0,
            total,
            standoff_mm: standoff,
            density_mode,
            error: String::new(),
        },
    );
    true
}

fn refresh_bake_scalar_stats(core: &Rc<RefCell<Core>>, key: &str) {
    let generation = core.borrow().record_generation;
    if core.borrow().stats_generation == Some(generation) {
        return;
    }
    core.borrow_mut().stats_generation = Some(generation);
    let stats = core
        .borrow()
        .latest_results
        .get(key)
        .and_then(|result| bake_scalar_stats(&result.parsed.scalars));
    if generation == core.borrow().record_generation {
        core.borrow_mut().scalar_stats = stats;
    }
    if core.borrow().stats_generation == Some(generation) {
        core.borrow_mut().stats_generation = None;
    }
}
