// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct ComputeEngine {
    base_url: String,
    gpu: Option<GpuSolver>,
}

#[wasm_bindgen]
impl ComputeEngine {
    /// `new ComputeEngine({ baseUrl })` resolved against the document base URI.
    #[wasm_bindgen(constructor)]
    pub fn new(base_url: Option<String>) -> ComputeEngine {
        // The session controller normally installs this, but `ComputeEngine` is
        // also driven directly (probes, tests); without it a Rust panic inside a
        // future surfaces as a bare `RuntimeError: unreachable` with no message.
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        let base = base_url.unwrap_or_else(|| "./".to_string());
        let base_uri = web_sys::window()
            .and_then(|window| window.document())
            .and_then(|document| document.base_uri().ok().flatten())
            .unwrap_or_default();
        let resolved = web_sys::Url::new_with_base(&base, &base_uri)
            .map(|url| url.href())
            .ok()
            .unwrap_or_else(|| base.clone());
        ComputeEngine {
            base_url: resolved,
            gpu: None,
        }
    }

    /// Acquire the WebGPU adapter and device.
    pub async fn init(&mut self) -> Result<(), JsValue> {
        let solver = GpuSolver::new(self.base_url.clone())
            .await
            .map_err(|error| JsValue::from_str(&error))?;
        self.gpu = Some(solver);
        Ok(())
    }

    /// Run one algorithm to completion, block by block, emitting progress and
    /// checkpoint events. Returns the RHGF v5 record, or `null` when aborted.
    pub async fn evaluate(&mut self, options: JsValue) -> Result<JsValue, JsValue> {
        let algorithm = field_string(&options, "algorithm").unwrap_or_default();
        let start_face = field_f64(&options, "startFace")
            .unwrap_or(0.0)
            .clamp(0.0, FACE_COUNT as f64) as usize;
        let height_mm = field_f64(&options, "heightMm").unwrap_or(0.0);
        let source_set = field_string(&options, "sourceSet").unwrap_or_default();
        let signal = field(&options, "signal");
        let on_progress = field(&options, "onProgress");
        let on_checkpoint = field(&options, "onCheckpoint");
        let checkpoint = field(&options, "checkpointBytes");
        let checkpoint_bytes: Option<Vec<u8>> = checkpoint
            .dyn_ref::<js_sys::Uint8Array>()
            .map(|array| array.to_vec());

        let mut bytes = make_record(checkpoint_bytes.as_deref(), height_mm, FACE_COUNT);
        set_completed(&mut bytes, start_face);
        let mut completed = start_face;
        let mut block_counter = 0usize;

        let emit_progress = |completed: usize, on_progress: &JsValue| {
            if let Some(callback) = on_progress.dyn_ref::<js_sys::Function>() {
                let fraction = if FACE_COUNT > 0 {
                    completed as f64 / FACE_COUNT as f64
                } else {
                    1.0
                };
                let _ = callback.call1(&JsValue::NULL, &JsValue::from_f64(fraction));
            }
        };
        let emit_checkpoint = |bytes: &[u8],
                               completed: usize,
                               force: bool,
                               block_counter: usize,
                               on_checkpoint: &JsValue| {
            if !force && !block_counter.is_multiple_of(CHECKPOINT_BLOCKS) {
                return;
            }
            if let Some(callback) = on_checkpoint.dyn_ref::<js_sys::Function>() {
                let payload = js_sys::Object::new();
                let _ = js_sys::Reflect::set(
                    &payload,
                    &JsValue::from_str("bytes"),
                    &js_sys::Uint8Array::from(bytes).into(),
                );
                let _ = js_sys::Reflect::set(
                    &payload,
                    &JsValue::from_str("completed"),
                    &JsValue::from_f64(completed as f64),
                );
                let _ = js_sys::Reflect::set(
                    &payload,
                    &JsValue::from_str("total"),
                    &JsValue::from_f64(FACE_COUNT as f64),
                );
                let _ = js_sys::Reflect::set(
                    &payload,
                    &JsValue::from_str("complete"),
                    &JsValue::from_bool(completed >= FACE_COUNT),
                );
                let _ = callback.call1(&JsValue::NULL, &payload);
            }
        };

        emit_progress(completed, &on_progress);
        if completed >= FACE_COUNT {
            emit_checkpoint(&bytes, completed, true, block_counter, &on_checkpoint);
            return Ok(js_sys::Uint8Array::from(bytes.as_slice()).into());
        }

        let Some(gpu) = self.gpu.as_mut() else {
            return Err(JsValue::from_str(
                "WebGPU computation backend is unavailable",
            ));
        };

        let size = block_size(&algorithm);
        let mut block_start = start_face;
        while block_start < FACE_COUNT {
            if signal_aborted(&signal) {
                return Ok(JsValue::NULL);
            }
            let block_end = FACE_COUNT.min(block_start + size);
            let constant = source_set == "constant";
            let scalars = match algorithm.as_str() {
                "werner" => gpu
                    .run_werner(block_start, block_end, height_mm, &signal)
                    .await,
                "mascon" => gpu
                    .run_mascon(
                        block_start,
                        block_end,
                        height_mm,
                        if source_set == "elliptic" { "elliptic" } else { "cauchy" },
                        &signal,
                    )
                    .await,
                "rtfp" => gpu
                    .run_rtfp(constant, block_start, block_end, height_mm, &signal)
                    .await,
                "carlson" => gpu
                    .run_carlson(constant, block_start, block_end, height_mm, &signal)
                    .await,
                "carlsonalpha" => gpu
                    .run_carlson_alpha(constant, block_start, block_end, height_mm, &signal)
                    .await,
                other => Err(format!("unknown algorithm {other}")),
            }
            .map_err(|error| JsValue::from_str(&error))?;
            let Some(scalars) = scalars else {
                return Ok(JsValue::NULL);
            };
            write_scalars(&mut bytes, block_start, &scalars);
            completed = block_end;
            set_completed(&mut bytes, completed);
            block_counter += 1;
            emit_checkpoint(&bytes, completed, false, block_counter, &on_checkpoint);
            emit_progress(completed, &on_progress);
            yield_to_browser().await;
            block_start += size;
        }
        emit_checkpoint(&bytes, completed, true, block_counter, &on_checkpoint);
        Ok(js_sys::Uint8Array::from(bytes.as_slice()).into())
    }
}
