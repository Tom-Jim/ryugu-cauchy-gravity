// ---------------------------------------------------------------------------
// Memory guard
// ---------------------------------------------------------------------------

fn wasm_memory_bytes() -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        (core::arch::wasm32::memory_size(0) as f64) * 65536.0
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        0.0
    }
}

async fn sample_page_memory() -> f64 {
    let wasm_bytes = wasm_memory_bytes();
    let Some(window) = web_sys::window() else {
        return wasm_bytes;
    };
    let performance = field(&window, "performance");
    let js_bytes = get_f64(&field(&performance, "memory"), "usedJSHeapSize");
    let mut measured_bytes = 0.0;
    if field(&window, "crossOriginIsolated").as_bool() == Some(true) {
        let measure = field(&performance, "measureUserAgentSpecificMemory");
        if let Some(function) = measure.dyn_ref::<Function>()
            && let Ok(value) = function.call0(&performance)
            && let Ok(value) = JsFuture::from(Promise::from(value)).await
        {
            // The API is intentionally unavailable in most browser contexts.
            measured_bytes = get_f64(&value, "bytes");
        }
    }
    measured_bytes.max(wasm_bytes + js_bytes)
}

fn install_memory_guard(core: Rc<RefCell<Core>>) {
    if core.borrow().memory_guard_running {
        return;
    }
    core.borrow_mut().memory_guard_running = true;
    spawn_local(async move {
        loop {
            sleep_ms(5000).await;
            if !core.borrow().memory_guard_running {
                return;
            }
            if document()
                .map(|document| document.hidden())
                .unwrap_or(false)
            {
                continue;
            }
            let bytes = sample_page_memory().await;
            if bytes < RELOAD_MEMORY_LIMIT_BYTES {
                continue;
            }
            {
                let mut state = core.borrow_mut();
                state.memory_guard_running = false;
                state.latest_results.clear();
                state.scalar_stats = None;
            }
            set_bar_percent(&core, 0.0);
            core.borrow()
                .ui
                .status("Browser memory ceiling reached; reloading a clean viewer context…");
            let text = format!(
                "Viewer memory reached {:.2} GiB (hard ceiling {:.1} GiB). Reloading before the tab can exceed 6 GiB.",
                bytes / 1024.0_f64.powi(3),
                MEMORY_LIMIT_BYTES / 1024.0_f64.powi(3),
            );
            core.borrow().ui.show_message(&text);
            sleep_ms(1200).await;
            reload_page();
            return;
        }
    });
}

