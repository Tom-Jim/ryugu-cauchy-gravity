// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

fn webgpu_available() -> bool {
    let Some(window) = web_sys::window() else {
        return false;
    };
    let gpu = field(&window.navigator(), "gpu");
    !gpu.is_undefined() && !gpu.is_null()
}

/// True when this tab owns the viewer; false when an active tab already does.
fn enforce_single_tab(core: &Rc<RefCell<Core>>) -> bool {
    let here = tab_identity(&current_href());
    let id = session_storage(TAB_ID_KEY)
        .filter(|id| !id.is_empty())
        .unwrap_or_else(|| {
            let generated = random_tab_id();
            set_session_storage(TAB_ID_KEY, &generated);
            generated
        });
    let channel = web_sys::BroadcastChannel::new(TAB_CHANNEL_KEY).ok();
    let duplicate = read_lease()
        .map(|lease| {
            lease.id != id && lease.identity == here && js_sys::Date::now() - lease.ts < LEASE_MS
        })
        .unwrap_or(false);

    {
        let mut state = core.borrow_mut();
        state.tab_id = id.clone();
        state.tab_identity = here.clone();
        state.channel = channel.clone();
    }

    if duplicate {
        if let Some(channel) = &channel {
            let message = Object::new();
            set_str(&message, "type", "activate");
            set_str(&message, "identity", &here);
            set_str(&message, "url", &current_href());
            set_str(&message, "from", &id);
            let _ = channel.post_message(&message.into());
        }
        if let Some(window) = web_sys::window() {
            let _ = window.open_with_url_and_target("", "_self");
            let _ = window.close();
        }
        show_duplicate_notice();
        return false;
    }

    write_lease(&id, &here);
    watch_tab_lease(core.clone());
    true
}

fn show_duplicate_notice() {
    let Some(document) = document() else {
        return;
    };
    if document.get_element_by_id("duplicate-tab-notice").is_some() {
        return;
    }
    let Ok(notice) = document.create_element("div") else {
        return;
    };
    let _ = notice.set_attribute("id", "duplicate-tab-notice");
    let _ = notice.set_attribute(
        "style",
        "position:fixed;inset:0;z-index:9999;display:grid;place-items:center;\
         padding:2rem;background:#050508;color:#e8e6e3;text-align:center;\
         font:14px/1.5 ui-sans-serif,system-ui,sans-serif",
    );
    notice.set_text_content(Some(
        "This viewer is already open in another tab. The existing tab was activated.",
    ));
    if let Some(body) = document.body() {
        let _ = body.append_child(&notice);
    }
}

/// Keep the lease fresh, hand it back on unload, and adopt an activation ping.
fn watch_tab_lease(core: Rc<RefCell<Core>>) {
    if let Some(channel) = core.borrow().channel.clone() {
        let identity = core.borrow().tab_identity.clone();
        let on_message = Closure::<dyn FnMut(JsValue)>::new(move |event: JsValue| {
            let message = field(&event, "data");
            if field(&message, "type").as_string().as_deref() != Some("activate") {
                return;
            }
            if field(&message, "identity").as_string().as_deref() != Some(identity.as_str()) {
                return;
            }
            if let Some(url) = field(&message, "url").as_string()
                && let Some(window) = web_sys::window()
            {
                let location = window.location();
                if location.replace(&url).is_ok() {
                    return;
                }
                let _ = window.focus();
            }
        })
        .into_js_value();
        channel.set_onmessage(Some(on_message.unchecked_ref()));
    }

    let heartbeat_core = core.clone();
    spawn_local(async move {
        loop {
            sleep_ms(1000).await;
            let (id, identity) = {
                let state = heartbeat_core.borrow();
                if state.tab_id.is_empty() {
                    return;
                }
                (state.tab_id.clone(), state.tab_identity.clone())
            };
            match read_lease() {
                Some(current) if current.id != id => continue,
                Some(_) => write_lease(&id, &identity),
                None => continue,
            }
        }
    });

    let pagehide_core = core.clone();
    let on_pagehide = Closure::<dyn FnMut(JsValue)>::new(move |_| {
        let channel = {
            let mut state = pagehide_core.borrow_mut();
            state.tab_id.clear();
            state.channel.clone()
        };
        if let Some(storage) = local_storage() {
            let _ = storage.remove_item(TAB_LEASE_KEY);
        }
        if let Some(channel) = channel {
            channel.close();
        }
    })
    .into_js_value();
    if let Some(window) = web_sys::window() {
        let _ = window.add_event_listener_with_callback("pagehide", on_pagehide.unchecked_ref());
    }
}

fn watch_unhandled_errors() {
    let rejection = Closure::<dyn FnMut(JsValue)>::new(move |event: JsValue| {
        let reason = field(&event, "reason").as_string().unwrap_or_default();
        web_sys::console::warn_1(
            &format!(
                "[unhandledrejection] {}",
                reason.chars().take(500).collect::<String>()
            )
            .into(),
        );
    })
    .into_js_value();
    let error = Closure::<dyn FnMut(JsValue)>::new(move |event: JsValue| {
        let message = field(&event, "message").as_string().unwrap_or_default();
        web_sys::console::warn_1(
            &format!(
                "[window.error] {}",
                message.chars().take(500).collect::<String>()
            )
            .into(),
        );
    })
    .into_js_value();
    if let Some(window) = web_sys::window() {
        let _ = window
            .add_event_listener_with_callback("unhandledrejection", rejection.unchecked_ref());
        let _ = window.add_event_listener_with_callback("error", error.unchecked_ref());
    }
}

fn mobile_environment() -> bool {
    let Some(window) = web_sys::window() else {
        return false;
    };
    let navigator = window.navigator();
    let user_agent = field(&navigator, "userAgent")
        .as_string()
        .unwrap_or_default();
    let pattern = js_sys::RegExp::new("Android|iPhone|iPad|iPod|Mobile", "i");
    if pattern.test(&user_agent) {
        return true;
    }
    if field(&field(&navigator, "userAgentData"), "mobile").as_bool() == Some(true) {
        return true;
    }
    let screen = field(&window, "screen");
    let small_touch = get_f64(&navigator, "maxTouchPoints") > 1.0
        && get_f64(&screen, "width").min(get_f64(&screen, "height")) <= 820.0;
    let narrow = window
        .match_media("(max-width: 720px)")
        .ok()
        .flatten()
        .map(|query| query.matches())
        .unwrap_or(false);
    small_touch || narrow
}

/// Reads the URL, the persisted algorithm and the initial labels.
fn init_viewer_state(core: &Rc<RefCell<Core>>) {
    let mobile = mobile_environment();
    {
        let ui = core.borrow().ui.clone();
        set_bool(&ui.mobile, "visible", mobile);
    }

    let params = current_url().map(|url| url.search_params());
    let requested = params
        .as_ref()
        .and_then(|params| params.get("algo"))
        .filter(|algo| ALGO_ORDER.contains(&algo.as_str()));
    let algo = requested
        .or_else(|| storage(ALGO_KEY).filter(|algo| ALGO_ORDER.contains(&algo.as_str())))
        .unwrap_or_else(|| "werner".to_string());
    set_storage(ALGO_KEY, &algo);

    let startup_mm = params
        .as_ref()
        .and_then(|params| params.get("standoff"))
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|mm| mm.is_finite() && *mm >= STANDOFF_MIN_MM && *mm <= STANDOFF_MAX_MM)
        .unwrap_or(STANDOFF_DEFAULT_MM);

    {
        let mut state = core.borrow_mut();
        state.density_mode = algo_spec(&algo).default_density.to_string();
        state.algo = algo;
    }
    set_standoff_ui(core, startup_mm, false);

    let ui = core.borrow().ui.clone();
    set_bool(&ui.bake, "hidden", false);
    set_bool(&ui.bake, "disabled", false);
    set_str(&ui.bake, "text", "Recompute");
    set_bool(&ui.standoff, "staticHidden", false);
    set_str(&ui.standoff, "staticText", "Rust WASM / WebGPU");
    set_str(&ui.standoff, "title", "Observation height");
    set_str(
        &ui.standoff,
        "staticTitle",
        "Computation runs in Rust WASM and WebGPU WGSL; the server only serves files.",
    );

    apply_algo_labels(core);
    publish_saved_selection(core, false);
    ui.bar_percent(0.0);
    ui.status("Initializing the full-surface recomputation engine…");
}

async fn start_engine(core: &Rc<RefCell<Core>>) -> Result<(), String> {
    // Start the renderer on an empty record so the original model, with its own
    // materials and vertex colours, is on screen while the solvers run.
    let stub = [0u8; RECORD_HEADER];
    crate::run_with_bake(&stub);
    wait_for_rendered_frames(2).await;
    core.borrow().ui.status("Loading the browser solver…");
    refresh_saved(core).await;

    // The browser compute coordinator is Rust: `ComputeEngine` drives the WGSL
    // kernels through the same WebGPU device the viewer renders with.
    let base_url = core.borrow().base_url.clone();
    let mut engine = ComputeEngine::new(Some(base_url));
    engine.init().await.map_err(|error| format!("{error:?}"))?;
    core.borrow_mut().engine = Some(engine);

    let ui = core.borrow().ui.clone();
    ui.bar_percent(0.0);
    set_bool(&ui.bake, "disabled", false);
    set_str(&ui.bake, "text", "Recompute");
    install_memory_guard(core.clone());
    let standoff_mm = core.borrow().standoff_mm;
    request_static_compute(core, standoff_mm, false, true);
    Ok(())
}
