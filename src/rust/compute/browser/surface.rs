impl GpuSolver {
#[allow(clippy::too_many_arguments)]
    async fn run_surface(
        &mut self,
        shader_name: &str,
        entry_point: &str,
        face_asset: &str,
        face_magic: u32,
        geometry_asset: &str,
        geometry_magic: u32,
        start: usize,
        end: usize,
        height_mm: f64,
        gravitational_scale: f64,
        signal: &JsValue,
    ) -> Result<Option<Vec<f32>>, String> {
        let pipeline = self.pipeline(shader_name, entry_point).await?;
        let face_bytes = self.asset(face_asset).await?;
        let faces = parse_face_asset(&face_bytes, face_magic, None)?;
        let face_records = face_bytes[faces.records.clone()].to_vec();
        let face_buffer_key = format!("{face_asset}:faces");
        if !self.face_buffers.contains_key(&face_buffer_key) {
            let buffer = self.storage_buffer("surface faces", &face_records);
            self.face_buffers.insert(face_buffer_key.clone(), buffer);
        }
        let face_buffer = self.face_buffers[&face_buffer_key].clone();

        let geometry_bytes = self.asset(geometry_asset).await?;
        let geometry = parse_face_asset(&geometry_bytes, geometry_magic, Some(FACE_COUNT))?;
        let geometry_records = geometry_bytes[geometry.records.clone()].to_vec();
        let geometry_buffer_key = format!("{geometry_asset}:faces");
        if !self.face_buffers.contains_key(&geometry_buffer_key) {
            let buffer = self.storage_buffer("observation faces", &geometry_records);
            self.face_buffers.insert(geometry_buffer_key.clone(), buffer);
        }
        let geometry_buffer = self.face_buffers[&geometry_buffer_key].clone();

        let count = end - start;
        let (analytic_groups_x, analytic_groups_y) = point_grid(count);
        let observer = self
            .observer_resources(&geometry_buffer, start, count, height_mm)
            .await?;
        let points = observer.points.clone();
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface tensor output"),
            size: (count * 6 * 4).max(16) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let mut globals = [0u8; 32];
        set_u32(&mut globals, 0, count as u32);
        set_u32(&mut globals, 4, faces.count as u32);
        set_f32(&mut globals, 8, gravitational_scale as f32);
        set_u32(&mut globals, 12, analytic_groups_x);
        // The observer buffer contains only this block. Keep the point offset
        // local to that compact buffer; the absolute face index reads past it.
        set_u32(&mut globals, 16, 0);
        let global_buffer = self.storage_buffer("surface globals", &globals);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(shader_name),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: global_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: face_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: points.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output.as_entire_binding(),
                },
            ],
        });

        let scalar_pipeline = self.pipeline("tensor_scalar", "analytic_scalar").await?;
        let scalar_output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("analytic face scalars"),
            size: (count * 4).max(16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut scalar_globals = [0u8; 16];
        set_u32(&mut scalar_globals, 0, count as u32);
        let scalar_global_buffer = self.storage_buffer("scalar globals", &scalar_globals);
        let scalar_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("analytic scalars"),
            layout: &scalar_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scalar_global_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: scalar_output.as_entire_binding(),
                },
            ],
        });

        if signal_aborted(signal) {
            for buffer in [
                &points,
                &observer.global_buffer,
                &output,
                &global_buffer,
                &scalar_output,
                &scalar_global_buffer,
            ] {
                buffer.destroy();
            }
            return Ok(None);
        }

        let error_scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        Self::encode_observers(&mut encoder, &observer, count);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(shader_name),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(analytic_groups_x, analytic_groups_y, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("analytic scalars"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&scalar_pipeline);
            pass.set_bind_group(0, &scalar_bind, &[]);
            pass.dispatch_workgroups(count.div_ceil(64) as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        if let Some(error) = error_scope.pop().await {
            return Err(format!("WebGPU {shader_name} pass failed validation: {error}"));
        }
        let raw = self
            .read_buffer(&scalar_output, (count * 4) as u64, None)
            .await;
        for buffer in [
            &points,
            &observer.global_buffer,
            &output,
            &global_buffer,
            &scalar_output,
            &scalar_global_buffer,
        ] {
            buffer.destroy();
        }
        Ok(Some(bytes_to_f32(&raw?, count)))
    }

}
