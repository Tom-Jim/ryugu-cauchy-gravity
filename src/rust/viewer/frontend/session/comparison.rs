// ---------------------------------------------------------------------------
// Comparison panel
// ---------------------------------------------------------------------------

/// Resolve a comparison reference at the target height.
///
/// A saved 100 % record is the durable source for later comparisons, so it is
/// consulted whenever this session has no runtime result for that track. Both
/// sources carry the height they were baked at, which is what makes a
/// face-by-face diff meaningful.
async fn reference_result_for(
    core: &Rc<RefCell<Core>>,
    reference: &CompareRef,
    target_mm: f64,
) -> Option<ParsedRecord> {
    if let Some(result) = core
        .borrow()
        .latest_result_for(reference.algo, reference.density_mode)
    {
        return Some(result.parsed.clone());
    }
    let store = core.borrow().store.clone();
    let id = saved_id(reference.algo, reference.density_mode, target_mm).await;
    // A missing or unreadable save must not break the panel.
    let row = store.get_saved(&id).await.ok().flatten()?;
    parse_face_scalars(&row.bytes?)
}

/// One comparison line, or the reason it cannot be produced yet.
async fn compare_against(
    core: &Rc<RefCell<Core>>,
    reference: &CompareRef,
    mine: &ParsedRecord,
    mine_standoff_mm: f64,
) -> String {
    let label = format!("对比 {} ({})", reference.name, reference.kind);
    let Some(parsed) = reference_result_for(core, reference, mine_standoff_mm).await else {
        return format!(
            "{label}: 尚未计算。请先切换到 {} · {}，在 {} 计算一次或保存该高度的结果。",
            reference.name,
            reference.kind,
            fmt_mm(mine_standoff_mm),
        );
    };
    if !same_standoff(parsed.standoff_mm, mine_standoff_mm) {
        // Face-by-face differences only mean something on one observation
        // surface, so the hint names the height the reference must be at.
        return format!(
            "{label}: 高度不一致。目标高度 {}（{} · {} 当前为 {}），请把 {} 也拉到 {} 重新计算后再对比。",
            fmt_mm(mine_standoff_mm),
            reference.name,
            reference.kind,
            fmt_mm(parsed.standoff_mm),
            reference.name,
            fmt_mm(mine_standoff_mm),
        );
    }
    let Some(diff) = scalar_diff_stats(mine, &parsed, 0.05) else {
        return format!("{label}: 可比较的有限面不足。");
    };
    let pct = |value: f64| format!("{:.2}%", 100.0 * value);
    let partial = if diff.compared < diff.total {
        format!(" · 仅 {}/{} 面", diff.compared, diff.total)
    } else {
        String::new()
    };
    format!(
        "{label}: 中位差 {} · 最大 {} · 超过 {} 的面 {}/{}{partial}",
        pct(diff.med),
        pct(diff.max),
        pct(diff.rel_eps),
        diff.over,
        diff.compared,
    )
}

/// The panel text used whenever no comparison is possible yet.
fn compare_height_notice(standoff_mm: f64) -> String {
    format!(
        "对比只在同一高度发生：目标高度 {}。 计算完成后会用各算法在该高度的最新临时结果比较；参考算法必须在同一高度完成计算。",
        fmt_mm(standoff_mm)
    )
}

async fn update_algorithm_compare(core: &Rc<RefCell<Core>>, status: &Status) {
    let (mode, algo, name, active_key, standoff_mm) = {
        let state = core.borrow();
        (
            state.result_mode(),
            state.algo.clone(),
            state.spec().name.to_string(),
            state.active_key(),
            state.standoff_mm,
        )
    };
    let order = compare_order(&algo, &mode);
    let ui = core.borrow().ui.clone();
    set_bool(&ui.compare, "visible", true);
    if order.is_empty() {
        core.borrow_mut().compare_key.clear();
        set_str(
            &ui.compare,
            "text",
            &format!(
                "{name} 是基准算法。切换到 RT-FP / Carlson / CarlsonAlpha，可查看它们与本基准在同一高度上的面间对比。"
            ),
        );
        return;
    }
    let mine = core.borrow().latest_results.get(&active_key).cloned();
    let Some(mine) = mine else {
        core.borrow_mut().compare_key.clear();
        set_str(&ui.compare, "text", &compare_height_notice(standoff_mm));
        return;
    };
    if !status.done || !same_standoff(mine.standoff_mm, status.standoff_mm) {
        core.borrow_mut().compare_key.clear();
        set_str(&ui.compare, "text", &compare_height_notice(standoff_mm));
        return;
    }
    let key = {
        let state = core.borrow();
        let refs: Vec<String> = order
            .iter()
            .map(|reference_key| {
                let revision = compare_ref(reference_key)
                    .and_then(|reference| {
                        state.latest_result_for(reference.algo, reference.density_mode)
                    })
                    .map(|result| result.revision.to_string())
                    .unwrap_or_else(|| "none".to_string());
                format!("{reference_key}:{revision}")
            })
            .collect();
        format!("{}|{}", mine.revision, refs.join("|"))
    };
    if key == core.borrow().compare_key {
        return;
    }
    core.borrow_mut().compare_key = key;
    let generation = core.borrow().record_generation;
    let mut lines = Vec::with_capacity(order.len());
    for reference_key in order {
        let Some(reference) = compare_ref(reference_key) else {
            continue;
        };
        lines.push(compare_against(core, &reference, &mine.parsed, mine.standoff_mm).await);
    }
    // A switch that landed during the saved-record read owns the panel now.
    if generation != core.borrow().record_generation {
        return;
    }
    set_str(
        &ui.compare,
        "text",
        &format!("{}\n{SAMPLING_NOTE}", lines.join("\n")),
    );
}
