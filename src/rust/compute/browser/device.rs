impl GpuSolver {
async fn new(base_url: String) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|error| format!("no WebGPU adapter: {error}"))?;
        let adapter_limits = adapter.limits();
        // rays.wgsl binds nine storage buffers in one stage, one past the
        // WebGPU baseline of eight, so the device has to be requested with the
        // adapter's own limits. Without this the shader fails validation, every
        // dispatch drops, and the run still reports a finished record of zeros.
        let required_limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: adapter_limits
                .max_storage_buffers_per_shader_stage,
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            max_buffer_size: adapter_limits.max_buffer_size,
            ..wgpu::Limits::default()
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("browser-compute"),
                required_features: wgpu::Features::empty(),
                required_limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                ..Default::default()
            })
            .await
            .map_err(|error| format!("no WebGPU device: {error}"))?;
        Ok(Self {
            base_url,
            device,
            queue,
            pipelines: HashMap::new(),
            source: None,
            density_files: HashMap::new(),
            assets: HashMap::new(),
            face_buffers: HashMap::new(),
            mesh_buffers: HashMap::new(),
            mascon_buffers: HashMap::new(),
        })
    }

    fn shader_source(name: &str) -> Option<&'static str> {
        match name {
            "werner" => Some(SHADER_WERNER),
            "rtfp_near" => Some(SHADER_RTFP_NEAR),
            "carlson_surface" => Some(SHADER_CARLSON_SURFACE),
            "carlson_alpha_near" => Some(SHADER_CARLSON_ALPHA_NEAR),
            "rays" => Some(SHADER_RAYS),
            "remainder" => Some(SHADER_REMAINDER),
            "carlson_alpha" => Some(SHADER_CARLSON_ALPHA),
            "observers" => Some(SHADER_OBSERVERS),
            "tensor_scalar" => Some(SHADER_TENSOR_SCALAR),
            "mascon_tree" => Some(SHADER_MASCON_TREE),
            _ => None,
        }
    }

    async fn pipeline(
        &mut self,
        name: &str,
        entry_point: &str,
    ) -> Result<wgpu::ComputePipeline, String> {
        let key = format!("{name}:{entry_point}");
        if let Some(cached) = self.pipelines.get(&key) {
            return Ok(cached.clone());
        }
        let code = Self::shader_source(name).ok_or_else(|| format!("unknown shader {name}"))?;
        // A pipeline that fails validation comes back as an invalid object and
        // only logs a warning, so the later dispatches drop silently and the
        // block reads back zeros. Capture the validation error and fail loudly.
        let error_scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(code.into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&key),
                layout: None,
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            });
        if let Some(error) = error_scope.pop().await {
            return Err(format!("WebGPU pipeline {key} failed validation: {error}"));
        }
        self.pipelines.insert(key, pipeline.clone());
        Ok(pipeline)
    }

    async fn runtime_source(&mut self) -> Result<Rc<RuntimeSource>, String> {
        if self.source.is_none() {
            let url = pipeline_url(MODEL_PATH, &self.base_url);
            let glb = fetch_bytes(&url).await.map_err(js_error)?;
            self.source = Some(Rc::new(RuntimeSource::from_glb(&glb)?));
        }
        Ok(self.source.as_ref().unwrap().clone())
    }

    async fn density_text(&mut self, path: &str) -> Result<Rc<String>, String> {
        if !self.density_files.contains_key(path) {
            let url = pipeline_url(path, &self.base_url);
            let bytes = fetch_bytes(&url).await.map_err(js_error)?;
            let text = String::from_utf8(bytes)
                .map_err(|error| format!("{path} is not UTF-8: {error}"))?;
            self.density_files.insert(path.to_string(), Rc::new(text));
        }
        Ok(self.density_files[path].clone())
    }

    async fn asset(&mut self, url: &str) -> Result<Rc<Vec<u8>>, String> {
        if !self.assets.contains_key(url) {
            let source = self.runtime_source().await?;
            if matches!(url, ASSET_MASCON | ASSET_MASCON_TREE) {
                let cauchy = self.density_text(CAUCHY_PATH).await?;
                let elliptic = self.density_text(ELLIPTIC_PATH).await?;
                let (points, tree) = source.mascon(cauchy.as_str(), elliptic.as_str())?;
                self.assets
                    .insert(ASSET_MASCON.to_string(), Rc::new(points));
                self.assets
                    .insert(ASSET_MASCON_TREE.to_string(), Rc::new(tree));
            } else {
                let bytes = match url {
                    ASSET_GEOMETRY => source.geometry(),
                    ASSET_RTFP => {
                        let density = self.density_text(CAUCHY_PATH).await?;
                        source.rtfp(density.as_str())?
                    }
                    ASSET_CARLSON_CAUCHY => {
                        let density = self.density_text(CAUCHY_PATH).await?;
                        source.carlson(Some(density.as_str()))?
                    }
                    ASSET_CARLSON_CONSTANT => source.carlson(None)?,
                    ASSET_CARLSON_ALPHA => {
                        let density = self.density_text(ELLIPTIC_PATH).await?;
                        source.carlson_alpha(density.as_str())?
                    }
                    _ => return Err(format!("unknown runtime source {url}")),
                };
                self.assets.insert(url.to_string(), Rc::new(bytes));
            }
        }
        Ok(self.assets[url].clone())
    }

    fn storage_buffer(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
        // `queue.writeBuffer` stages through a small ring buffer, so the mascon
        // tree and its voxel table (62 MB + 41 MB) queue far more staging data
        // than the ring can hold and the submit that would drain it never
        // returns. Mapping at creation copies straight into the buffer instead.
        self.create_buffer(label, data.len().max(16) as u64, true, data)
    }

    /// `pipeline_buffer(label, bytes, initial)` with the upload path chosen for
    /// the size: small buffers use the queue, large ones map at creation.
    fn create_buffer(&self, label: &str, size: u64, storage: bool, initial: &[u8]) -> wgpu::Buffer {
        let usage = if storage {
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST
        } else {
            wgpu::BufferUsages::COPY_DST
        };
        if initial.is_empty() {
            return self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            });
        }
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: true,
        });
        let mut view = buffer.slice(0..initial.len() as u64).get_mapped_range_mut();
        view.copy_from_slice(initial);
        drop(view);
        buffer.unmap();
        buffer
    }

    async fn observer_resources(
        &mut self,
        geometry: &wgpu::Buffer,
        start: usize,
        count: usize,
        height_mm: f64,
    ) -> Result<ObserverResources, String> {
        let pipeline = self.pipeline("observers", "build_observers").await?;
        let points = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("observation points"),
            size: (count * 16).max(16) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let mut globals = [0u8; 16];
        set_u32(&mut globals, 0, count as u32);
        set_u32(&mut globals, 4, start as u32);
        set_f32(&mut globals, 8, (height_mm * 1e-3) as f32);
        let global_buffer = self.storage_buffer("observer globals", &globals);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("observers"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: global_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: geometry.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: points.as_entire_binding(),
                },
            ],
        });
        Ok(ObserverResources {
            pipeline,
            bind_group,
            points,
            global_buffer,
        })
    }

    fn encode_observers(
        encoder: &mut wgpu::CommandEncoder,
        resources: &ObserverResources,
        count: usize,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("observers"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&resources.pipeline);
        pass.set_bind_group(0, &resources.bind_group, &[]);
        pass.dispatch_workgroups(count.div_ceil(64) as u32, 1, 1);
    }

    async fn read_buffer(
        &self,
        source: &wgpu::Buffer,
        byte_length: u64,
        extra: Option<(&wgpu::Buffer, u64, u64)>,
    ) -> Result<Vec<u8>, String> {
        let extra_length = extra.map(|entry| entry.2).unwrap_or(0);
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("browser-compute readback"),
            size: byte_length + extra_length,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(source, 0, &readback, 0, byte_length);
        if let Some((extra_source, offset, length)) = extra.filter(|entry| entry.2 > 0) {
            encoder.copy_buffer_to_buffer(extra_source, offset, &readback, byte_length, length);
        }
        self.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        // A rejected `mapAsync` (lost device, destroyed buffer) used to resolve
        // this promise anyway, and the following `get_mapped_range` then trapped
        // the whole module. Record the outcome and surface it as an error.
        let mapped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = mapped.clone();
        let mut executor = |resolve: js_sys::Function, _reject: js_sys::Function| {
            let flag = flag.clone();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                flag.store(result.is_ok(), std::sync::atomic::Ordering::Relaxed);
                let _ = resolve.call0(&JsValue::NULL);
            });
        };
        let _ = JsFuture::from(js_sys::Promise::new(&mut executor)).await;
        if !mapped.load(std::sync::atomic::Ordering::Relaxed) {
            readback.destroy();
            return Err(format!(
                "WebGPU readback of {byte_length} bytes failed: the mapping was rejected \
                 (the device was most likely lost while the dispatch was running)"
            ));
        }
        let data = readback.slice(..).get_mapped_range().to_vec();
        readback.unmap();
        readback.destroy();
        Ok(data)
    }

}
