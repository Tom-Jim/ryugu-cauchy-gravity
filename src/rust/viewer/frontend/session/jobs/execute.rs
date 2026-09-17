async fn run_job(
    core: &Rc<RefCell<Core>>,
    job: &Job,
    token: u64,
    signal: Option<JsValue>,
) -> Result<(), String> {
    // The progress and checkpoint callbacks outlive this frame, so the job is
    // copied into them instead of borrowed.
    let job = *job;
    if job.generation != core.borrow().record_generation {
        return Ok(());
    }
    let (algo, density_mode, name, source_set) = {
        let state = core.borrow();
        (
            state.algo.clone(),
            state.density_mode.clone(),
            state.spec().name.to_string(),
            state.compute_source_set(&state.algo, &state.density_mode),
        )
    };
    let label = format!("Full surface recompute · {}", fmt_mm(job.mm));

    let cached_key = if job.force_compute {
        None
    } else {
        Some(core.borrow().result_key_for(&algo, &density_mode))
    };
    if let Some(key) = cached_key.as_deref() {
        let cached = core
            .borrow()
            .latest_results
            .get(key)
            .map(|result| result.standoff_mm);
        if let Some(standoff) = cached
            && same_standoff(standoff, job.mm)
        {
            render_completed_result(
                core,
                key,
                job.preserve_window,
                job.generation,
                &format!("Temporary result · {}", fmt_mm(job.mm)),
            )
            .await;
            return Ok(());
        }
    }

    let mut checkpoint: Option<Vec<u8>> = None;
    let mut start_face = 0usize;
    if !job.force_compute {
        let id = temporary_id(&algo, &mode_key(&algo, &density_mode), job.mm);
        let store = core.borrow().store.clone();
        // IndexedDB may be unavailable; compute from scratch instead.
        if let Ok(Some(temporary)) = store.get_temp(&id).await
            && !temporary.bytes.is_empty()
        {
            if temporary.complete {
                if let Some(parsed) = parse_face_scalars(&temporary.bytes)
                    && record_has_signal(&parsed)
                {
                    let key = remember_latest_result(
                        core,
                        temporary.bytes.clone(),
                        parsed,
                        &algo,
                        &name,
                        &density_mode,
                    );
                    render_completed_result(
                        core,
                        &key,
                        job.preserve_window,
                        job.generation,
                        &format!("Temporary result · {}", fmt_mm(job.mm)),
                    )
                    .await;
                    return Ok(());
                }
            } else {
                let completed = record_face_count(&temporary.bytes);
                start_face = completed.min(temporary.completed.max(0.0) as usize);
                checkpoint = Some(temporary.bytes);
            }
        }
    }

    let start_percent = if start_face > 0 {
        (100.0 * start_face as f64 / FACE_TOTAL).min(99.0)
    } else {
        0.0
    };
    set_bar_percent(core, start_percent);
    show_compute_indicator(core, start_percent, &format!("全表面 {}", fmt_mm(job.mm)));
    core.borrow().ui.status(&if start_face > 0 {
        format!("Resuming {label} · {start_percent:.1}%")
    } else {
        format!("{label} · 0.0%")
    });
    set_standoff_record_label(core, job.mm);

    let options = Object::new();
    set_str(&options, "algorithm", &algo);
    set_str(&options, "sourceSet", source_set);
    set_f64(&options, "heightMm", job.mm);
    set_f64(&options, "startFace", start_face as f64);
    if let Some(signal) = signal {
        set(&options, "signal", signal);
    }
    if let Some(bytes) = &checkpoint {
        set(
            &options,
            "checkpointBytes",
            Uint8Array::from(bytes.as_slice()).into(),
        );
    }

    let progress_core = core.clone();
    let progress_label = label.clone();
    let progress_mm = job.mm;
    let progress = Closure::<dyn FnMut(f64)>::new(move |percent: f64| {
        if token != progress_core.borrow().compute_generation {
            return;
        }
        let compute_percent = 99.0 * percent.clamp(0.0, 1.0);
        set_bar_percent(&progress_core, compute_percent);
        show_compute_indicator(
            &progress_core,
            compute_percent,
            &format!("全表面 {}", fmt_mm(progress_mm)),
        );
        progress_core
            .borrow()
            .ui
            .status(&format!("{progress_label} · {compute_percent:.1}%"));
    });

    let checkpoint_core = core.clone();
    let checkpoint_mm = job.mm;
    let on_checkpoint = Closure::<dyn FnMut(JsValue)>::new(move |payload: JsValue| {
        if token != checkpoint_core.borrow().compute_generation {
            return;
        }
        let bytes = field(&payload, "bytes")
            .dyn_into::<Uint8Array>()
            .map(|array| array.to_vec())
            .unwrap_or_default();
        if bytes.is_empty() {
            return;
        }
        let row = {
            let state = checkpoint_core.borrow();
            TempRow {
                id: temporary_id(&state.algo, &state.result_mode(), checkpoint_mm),
                algo: state.algo.clone(),
                algorithm: state.spec().name.to_string(),
                density: state.result_mode(),
                standoff_mm: checkpoint_mm,
                completed: get_f64(&payload, "completed"),
                total: get_f64(&payload, "total"),
                complete: get_bool(&payload, "complete"),
                updated_at: String::from(js_sys::Date::new_0().to_iso_string()),
                bytes,
            }
        };
        let store = checkpoint_core.borrow().store.clone();
        spawn_local(async move {
            // A failed checkpoint must not stop the calculation.
            let _ = store.put_temp(&row).await;
        });
    });

    set(&options, "onProgress", progress.as_ref().clone());
    set(&options, "onCheckpoint", on_checkpoint.as_ref().clone());

    let mut engine = core.borrow_mut().engine.take();
    let outcome = match engine.as_mut() {
        Some(engine) => engine.evaluate(options.into()).await,
        None => return Err("WebGPU computation backend is unavailable".to_string()),
    };
    core.borrow_mut().engine = engine;
    let value = outcome.map_err(|error| format!("{error:?}"))?;
    drop(progress);
    drop(on_checkpoint);

    if token != core.borrow().compute_generation || value.is_null() || value.is_undefined() {
        return Ok(());
    }
    let bytes = value
        .dyn_into::<Uint8Array>()
        .map(|array| array.to_vec())
        .map_err(|_| "invalid runtime result".to_string())?;
    let Some(parsed) = parse_face_scalars(&bytes) else {
        return Err("invalid runtime result".to_string());
    };
    if !record_has_signal(&parsed) {
        return Err(
            "计算没有产生任何非零结果：GPU 管线未执行，请检查 WebGPU 设备限制与管线校验错误。"
                .to_string(),
        );
    }
    let density = mode_key(&algo, &density_mode);
    let key = remember_latest_result(core, bytes.clone(), parsed.clone(), &algo, &name, &density);
    let row = TempRow {
        id: temporary_id(&algo, &density, job.mm),
        algo: algo.clone(),
        algorithm: name.clone(),
        density,
        standoff_mm: job.mm,
        bytes,
        completed: parsed.total as f64,
        total: parsed.total as f64,
        complete: true,
        updated_at: String::from(js_sys::Date::new_0().to_iso_string()),
    };
    let store = core.borrow().store.clone();
    let _ = store.put_temp(&row).await;
    if token != core.borrow().compute_generation {
        return Ok(());
    }
    render_completed_result(core, &key, job.preserve_window, job.generation, &label).await;
    Ok(())
}
