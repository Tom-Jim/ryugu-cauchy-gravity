// ---------------------------------------------------------------------------
// Saved results
// ---------------------------------------------------------------------------

fn format_saved_at(value: f64) -> String {
    if !value.is_finite() {
        return String::new();
    }
    js_sys::Date::new(&JsValue::from_f64(value))
        .to_locale_string("", &Object::new())
        .into()
}

fn format_faces(faces: f64) -> String {
    js_sys::Number::from(faces).to_locale_string("").into()
}

fn saved_item_object(row: &SavedRow) -> JsValue {
    let object = Object::new();
    set_str(&object, "id", &row.id);
    set_str(&object, "algo", &row.algo);
    set_str(&object, "algorithm", &row.algorithm);
    set_str(&object, "density", &row.density);
    set_f64(&object, "standoffMm", row.standoff_mm);
    set_str(&object, "standoffText", &fmt_mm(row.standoff_mm));
    set_str(&object, "densityText", density_label(&row.density));
    set_str(
        &object,
        "savedAtText",
        &format_saved_at(js_sys::Date::parse(&row.saved_at)),
    );
    set_str(&object, "facesText", &format_faces(row.faces));
    object.into()
}

async fn refresh_saved(core: &Rc<RefCell<Core>>) {
    {
        let ui = core.borrow().ui.clone();
        set_bool(&ui.saved, "loading", true);
    }
    let store = core.borrow().store.clone();
    match store.list_saved().await {
        Ok(items) => {
            let array = Array::new();
            for item in &items {
                array.push(&saved_item_object(item));
            }
            let mut state = core.borrow_mut();
            state.saved_current_id = current_saved_id(&items, &state);
            state.saved_items = items;
            let ui = state.ui.clone();
            set(&ui.saved, "items", array.into());
            update_saved_flags(&state);
        }
        Err(error) => {
            let ui = core.borrow().ui.clone();
            set_str(&ui.saved, "message", &format!("读取保存列表失败：{error}"));
        }
    }
    let ui = core.borrow().ui.clone();
    set_bool(&ui.saved, "loading", false);
}

fn current_saved_id(items: &[SavedRow], state: &Core) -> Option<String> {
    let algo = state.algo.clone();
    let density = state.result_mode();
    let mm = state.standoff_mm;
    items
        .iter()
        .find(|item| {
            item.algo == algo && item.density == density && same_standoff(item.standoff_mm, mm)
        })
        .map(|item| item.id.clone())
}

fn update_saved_flags(state: &Core) {
    let ui = state.ui.clone();
    let current = field(&ui.saved, "current");
    let done = get_bool(&current, "done");
    let mm = get_f64(&current, "standoffMm");
    let busy = get_bool(&ui.saved, "busy");
    set_bool(&ui.saved, "canSave", done && mm > 0.0 && !busy);
    set_bool(
        &ui.saved,
        "hasCurrentSaved",
        state.saved_current_id.is_some(),
    );
}

async fn save_current(core: &Rc<RefCell<Core>>) {
    let (can_save, record) = {
        let mut state = core.borrow_mut();
        let current_id = current_saved_id(&state.saved_items, &state);
        state.saved_current_id = current_id;
        let key = state.active_key();
        let record = state.latest_results.get(&key).cloned();
        let done = get_bool(&field(&state.ui.saved, "current"), "done");
        let busy = get_bool(&state.ui.saved, "busy");
        (done && state.standoff_mm > 0.0 && !busy, record)
    };
    if !can_save {
        return;
    }
    let ui = core.borrow().ui.clone();
    set_bool(&ui.saved, "busy", true);
    set_str(&ui.saved, "message", "正在保存当前结果…");
    let Some(record) = record else {
        set_str(
            &ui.saved,
            "message",
            "保存失败：Error: 当前结果不在内存中，无法保存",
        );
        set_bool(&ui.saved, "busy", false);
        return;
    };
    let row = SavedRow {
        id: saved_id(&record.algo, &record.density_mode, record.standoff_mm).await,
        algo: record.algo.clone(),
        algorithm: record.algorithm.clone(),
        density: record.density_mode.clone(),
        standoff_mm: record.standoff_mm,
        saved_at: String::from(js_sys::Date::new_0().to_iso_string()),
        faces: record.parsed.total as f64,
        bytes: Some(record.bytes.clone()),
    };
    let described = format!("{} · {}", row.algorithm, fmt_mm(row.standoff_mm));
    let store = core.borrow().store.clone();
    match store.put_saved(&row).await {
        Ok(()) => {
            set_str(&ui.saved, "message", &format!("已保存 {described}"));
            refresh_saved(core).await;
        }
        Err(error) => set_str(&ui.saved, "message", &format!("保存失败：{error}")),
    }
    set_bool(&ui.saved, "busy", false);
    // The panel keeps saved records as candidate comparison sources, so re-run
    // the last status through it.
    let last = core.borrow().last_status.clone();
    if let Some(status) = last {
        update_algorithm_compare(core, &status).await;
    }
}

async fn delete_item(core: &Rc<RefCell<Core>>, id: Option<&str>) {
    let Some(id) = id.map(str::to_string) else {
        return;
    };
    let ui = core.borrow().ui.clone();
    if get_bool(&ui.saved, "busy") {
        return;
    }
    set_bool(&ui.saved, "busy", true);
    set_str(&ui.saved, "message", "正在删除…");
    let item = core
        .borrow()
        .saved_items
        .iter()
        .find(|item| item.id == id)
        .cloned();
    let store = core.borrow().store.clone();
    match store.delete_saved(&id).await {
        Ok(()) => {
            let described = item
                .map(|item| format!("{} · {}", item.algorithm, fmt_mm(item.standoff_mm)))
                .unwrap_or_else(|| id.clone());
            set_str(&ui.saved, "message", &format!("已删除 {described}"));
            refresh_saved(core).await;
            let last = core.borrow().last_status.clone();
            if let Some(status) = last {
                update_algorithm_compare(core, &status).await;
            }
        }
        Err(error) => set_str(&ui.saved, "message", &format!("删除失败：{error}")),
    }
    set_bool(&ui.saved, "busy", false);
}

