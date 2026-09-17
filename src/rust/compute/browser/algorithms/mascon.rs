impl GpuSolver {
async fn run_mascon(
        &mut self,
        start: usize,
        end: usize,
        height_mm: f64,
        density_mode: &str,
        signal: &JsValue,
    ) -> Result<Option<Vec<f32>>, String> {
        let tree_url = pipeline_url(ASSET_MASCON_TREE, &self.base_url);
        let point_url = pipeline_url(ASSET_MASCON, &self.base_url);
        let point_bytes = self.asset(&point_url).await?;
        if point_bytes.len() < 64 {
            return Err("invalid Mascon point asset".into());
        }
        if u32_at(&point_bytes, 0) != 0x314d_5952 {
            return Err("invalid Mascon point magic".into());
        }
        let point_count = u32_at(&point_bytes, 12) as usize;
        let tree_bytes = self.asset(&tree_url).await?;
        let tree = parse_mascon_tree(&tree_bytes)?;

        // The three voxel/tree buffers total 113 MB and stay valid for every
        // block of the run, so they are built once. Uploading them per block
        // would copy 113 MB forty-eight times over a full sweep.
        let key = format!("{tree_url}:mascon");
        if !self.mascon_buffers.contains_key(&key) {
            let buffers = MasconBuffers {
                nodes: self.storage_buffer("Mascon nodes", &tree_bytes[tree.nodes.clone()]),
                points: self
                    .storage_buffer("Mascon points", &point_bytes[64..64 + point_count * 16]),
                order: self.storage_buffer("Mascon point order", &tree_bytes[tree.order.clone()]),
            };
            self.mascon_buffers.insert(key.clone(), Rc::new(buffers));
        }
        let buffers = self.mascon_buffers[&key].clone();
        let pipeline = self.pipeline("mascon_tree", "mascon_tree").await?;
        let count = end - start;

        let geometry_url = pipeline_url(ASSET_GEOMETRY, &self.base_url);
        let geometry_bytes = self.asset(&geometry_url).await?;
        let geometry = parse_face_asset(&geometry_bytes, FACE_MAGIC_WERNER, Some(FACE_COUNT))?;
        let geometry_records = geometry_bytes[geometry.records.clone()].to_vec();
        let geometry_key = format!("{geometry_url}:faces");
        if !self.face_buffers.contains_key(&geometry_key) {
            let buffer = self.storage_buffer("observation faces", &geometry_records);
            self.face_buffers.insert(geometry_key.clone(), buffer);
        }
        let geometry_buffer = self.face_buffers[&geometry_key].clone();

        let observer = self
            .observer_resources(&geometry_buffer, start, count, height_mm)
            .await?;
        let observer_buffer = observer.points.clone();
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mascon scalars"),
            size: (count * 4).max(16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut globals = [0f32; 12];
        globals[0] = count as f32;
        globals[1] = tree.node_count as f32;
        globals[2] = G as f32;
        globals[3] = MASCON_THETA;
        globals[4] = tree.min[0] as f32;
        globals[5] = tree.min[1] as f32;
        globals[6] = tree.min[2] as f32;
        globals[7] = match density_mode {
            "cauchy" => 0.0,
            "elliptic" => 1.0,
            _ => 2.0,
        };
        globals[8] = ((tree.max[0] - tree.min[0]) / tree.grid as f64) as f32;
        globals[9] = ((tree.max[1] - tree.min[1]) / tree.grid as f64) as f32;
        globals[10] = ((tree.max[2] - tree.min[2]) / tree.grid as f64) as f32;
        globals[11] = tree.constant_density;
        let mut global_bytes = [0u8; 48];
        for (index, value) in globals.iter().enumerate() {
            set_f32(&mut global_bytes, index * 4, *value);
        }
        let global_buffer = self.storage_buffer("Mascon globals", &global_bytes);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mascon"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: global_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.nodes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.points.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.order.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: observer_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        if signal_aborted(signal) {
            observer_buffer.destroy();
            observer.global_buffer.destroy();
            output.destroy();
            global_buffer.destroy();
            return Ok(None);
        }
        let error_scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        Self::encode_observers(&mut encoder, &observer, count);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mascon"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(count.div_ceil(64) as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        if let Some(error) = error_scope.pop().await {
            return Err(format!("WebGPU Mascon pass failed validation: {error}"));
        }
        let raw = self.read_buffer(&output, (count * 4) as u64, None).await;
        observer_buffer.destroy();
        observer.global_buffer.destroy();
        output.destroy();
        global_buffer.destroy();
        Ok(Some(bytes_to_f32(&raw?, count)))
    }

}
