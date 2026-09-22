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
    /// Replace the browser compute source selected by the asset library.
    /// This invalidates only derived source buffers; WGSL kernels and Bevy
    /// rendering remain untouched.
    pub fn set_model_bytes(&mut self, bytes: js_sys::Uint8Array) {
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_model_bytes(bytes.to_vec());
        }
    }

    pub fn set_density_text(&mut self, name: String, text: String) {
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_density_text(&name, text);
        }
    }
    /// Diagnostic-only tensor path. It returns six Hessian components per
    /// observation point and never writes an RHGF record or touches rendering.
    pub async fn evaluate_diagnostic(&mut self, options: JsValue) -> Result<JsValue, JsValue> {
        let algorithm = field_string(&options, "algorithm").unwrap_or_default();
        let height_mm = field_f64(&options, "heightMm").unwrap_or(0.0);
        let source_set = field_string(&options, "sourceSet").unwrap_or_default();
        let direction_limit =
            field_f64(&options, "directionLimit").map(|value| value.max(1.0) as usize);
        let quadrature_limit =
            field_f64(&options, "quadratureLimit").map(|value| value.clamp(1.0, 16.0) as usize);
        let signal = field(&options, "signal");
        let on_progress = field(&options, "onProgress");
        let Some(gpu) = self.gpu.as_mut() else {
            return Err(JsValue::from_str(
                "WebGPU computation backend is unavailable",
            ));
        };
        let model_faces = gpu
            .model_face_count()
            .await
            .map_err(|error| JsValue::from_str(&error))?;
        let face_count = field_f64(&options, "faceCount")
            .unwrap_or(128.0)
            .clamp(1.0, model_faces as f64) as usize;
        let mut output = Vec::with_capacity(face_count * 6);
        let mut start = 0usize;
        let size = block_size(&algorithm);
        while start < face_count {
            if signal_aborted(&signal) {
                return Ok(JsValue::NULL);
            }
            let end = face_count.min(start + size);
            let values = match algorithm.as_str() {
                "werner" => gpu.run_werner_tensor(start, end, height_mm, &signal).await,
                "mascon" => {
                    gpu.run_mascon(
                        start,
                        end,
                        height_mm,
                        match source_set.as_str() {
                            "elliptic" => "elliptic",
                            "constant" => "constant",
                            _ => "cauchy",
                        },
                        &signal,
                        OutputMode::Tensor,
                        field_f64(&options, "masconTheta"),
                    )
                    .await
                }
                "rtfp" => {
                    gpu.run_rtfp_tensor(
                        source_set == "constant",
                        start,
                        end,
                        height_mm,
                        direction_limit,
                        &signal,
                    )
                    .await
                }
                "carlson" => {
                    gpu.run_carlson_tensor(
                        source_set == "constant",
                        start,
                        end,
                        height_mm,
                        direction_limit,
                        &signal,
                    )
                    .await
                }
                "carlsonalpha" => {
                    gpu.run_carlson_alpha_tensor(
                        source_set == "constant",
                        start,
                        end,
                        height_mm,
                        direction_limit,
                        quadrature_limit,
                        &signal,
                    )
                    .await
                }
                other => Err(format!("unknown algorithm {other}")),
            }
            .map_err(|error| JsValue::from_str(&error))?;
            let Some(values) = values else {
                return Ok(JsValue::NULL);
            };
            if values.len() != (end - start) * 6 {
                return Err(JsValue::from_str(
                    "diagnostic tensor block has an invalid size",
                ));
            }
            output.extend(values);
            start = end;
            if let Some(callback) = on_progress.dyn_ref::<js_sys::Function>() {
                let _ = callback.call1(
                    &JsValue::NULL,
                    &JsValue::from_f64(start as f64 / face_count as f64),
                );
            }
            yield_to_browser().await;
        }
        let mut bytes = Vec::with_capacity(output.len() * 4);
        for value in output {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Ok(js_sys::Uint8Array::from(bytes.as_slice()).into())
    }

    pub async fn export_vtk(
        &mut self,
        record: js_sys::Uint8Array,
    ) -> Result<js_sys::Uint8Array, JsValue> {
        let source = self
            .gpu
            .as_mut()
            .ok_or_else(|| JsValue::from_str("WebGPU computation backend is unavailable"))?
            .runtime_source()
            .await
            .map_err(|e| JsValue::from_str(&e))?;
        let bytes = record.to_vec();
        if bytes.len() < RECORD_HEADER
            || u32_at(&bytes, 0) != RECORD_MAGIC
            || u32_at(&bytes, 4) != RECORD_VERSION
        {
            return Err(JsValue::from_str("invalid RHGF record"));
        }
        let count = u32_at(&bytes, 8) as usize;
        if bytes.len() < RECORD_HEADER + count * 4 {
            return Err(JsValue::from_str("truncated RHGF record"));
        }
        if count != source.face_count() {
            return Err(JsValue::from_str(
                "RHGF face count does not match the loaded model",
            ));
        }
        let scalars = (0..count)
            .map(|i| f32_at(&bytes, RECORD_HEADER + i * 4))
            .collect::<Vec<_>>();
        Ok(js_sys::Uint8Array::from(
            source.vtk_polydata(&scalars).as_slice(),
        ))
    }
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
        let Some(gpu) = self.gpu.as_mut() else {
            return Err(JsValue::from_str(
                "WebGPU computation backend is unavailable",
            ));
        };
        let face_count = gpu
            .model_face_count()
            .await
            .map_err(|error| JsValue::from_str(&error))?;
        let start_face = field_f64(&options, "startFace")
            .unwrap_or(0.0)
            .clamp(0.0, face_count as f64) as usize;
        let height_mm = field_f64(&options, "heightMm").unwrap_or(0.0);
        let source_set = field_string(&options, "sourceSet").unwrap_or_default();
        let signal = field(&options, "signal");
        let on_progress = field(&options, "onProgress");
        let on_checkpoint = field(&options, "onCheckpoint");
        let checkpoint = field(&options, "checkpointBytes");
        let checkpoint_bytes: Option<Vec<u8>> = checkpoint
            .dyn_ref::<js_sys::Uint8Array>()
            .map(|array| array.to_vec());

        let mut bytes = make_record(checkpoint_bytes.as_deref(), height_mm, face_count);
        let start_face = prepare_record(&mut bytes, start_face);
        let mut completed = start_face;
        let mut block_counter = 0usize;

        let emit_progress = |completed: usize, on_progress: &JsValue| {
            if let Some(callback) = on_progress.dyn_ref::<js_sys::Function>() {
                let fraction = if face_count > 0 {
                    completed as f64 / face_count as f64
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
                    &JsValue::from_f64(face_count as f64),
                );
                let _ = js_sys::Reflect::set(
                    &payload,
                    &JsValue::from_str("complete"),
                    &JsValue::from_bool(completed >= face_count),
                );
                let _ = callback.call1(&JsValue::NULL, &payload);
            }
        };

        emit_progress(completed, &on_progress);
        if completed >= face_count {
            emit_checkpoint(&bytes, completed, true, block_counter, &on_checkpoint);
            return Ok(js_sys::Uint8Array::from(bytes.as_slice()).into());
        }

        let size = block_size(&algorithm);
        let mut block_start = start_face;
        while block_start < face_count {
            if signal_aborted(&signal) {
                return Ok(JsValue::NULL);
            }
            let block_end = face_count.min(block_start + size);
            let constant = source_set == "constant";
            let scalars = match algorithm.as_str() {
                "werner" => {
                    gpu.run_werner(block_start, block_end, height_mm, &signal)
                        .await
                }
                "mascon" => {
                    gpu.run_mascon(
                        block_start,
                        block_end,
                        height_mm,
                        match source_set.as_str() {
                            "elliptic" => "elliptic",
                            "constant" => "constant",
                            _ => "cauchy",
                        },
                        &signal,
                        OutputMode::Scalar,
                        None,
                    )
                    .await
                }
                "rtfp" => {
                    gpu.run_rtfp(constant, block_start, block_end, height_mm, &signal)
                        .await
                }
                "carlson" => {
                    gpu.run_carlson(constant, block_start, block_end, height_mm, &signal)
                        .await
                }
                "carlsonalpha" => {
                    gpu.run_carlson_alpha(constant, block_start, block_end, height_mm, &signal)
                        .await
                }
                other => Err(format!("unknown algorithm {other}")),
            }
            .map_err(|error| JsValue::from_str(&error))?;
            let Some(scalars) = scalars else {
                return Ok(JsValue::NULL);
            };
            let expected = block_end - block_start;
            if scalars.len() != expected {
                return Err(JsValue::from_str(&format!(
                    "{algorithm} returned {} scalars for a {expected}-face block",
                    scalars.len()
                )));
            }
            if let Some((offset, value)) = scalars
                .iter()
                .copied()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
            {
                return Err(JsValue::from_str(&format!(
                    "{algorithm} produced a non-finite scalar at face {} ({value})",
                    block_start + offset
                )));
            }
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
