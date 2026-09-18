impl GpuSolver {
async fn mesh_buffers(&mut self, asset_url: &str) -> Result<(), String> {
        let key = format!("{asset_url}:mesh");
        if self.mesh_buffers.get(&key).is_some() {
            return Ok(());
        }
        let bytes = self.asset(asset_url).await?;
        let layout = parse_mesh_pipeline(&bytes)?;
        let buffers = MeshBuffers {
            positions: self.storage_buffer("ray positions", &bytes[layout.positions.clone()]),
            indices: self.storage_buffer("ray indices", &bytes[layout.indices.clone()]),
            bounds: self.storage_buffer("ray bounds", &bytes[layout.bounds.clone()]),
            links: self.storage_buffer("ray links", &bytes[layout.links.clone()]),
            directions: self.storage_buffer("ray directions", &bytes[layout.directions.clone()]),
            kernels: self.storage_buffer("ray kernels", &bytes[layout.kernels.clone()]),
            faces: self.storage_buffer("ray faces", &bytes[layout.faces.clone()]),
            layout,
        };
        self.mesh_buffers.insert(key, Rc::new(buffers));
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    /// Shared WebGPU buffer/dispatch plumbing for ray-based solvers. Algorithm
    /// selection happens in each solver module; this function has no mode flag
    /// and never chooses one numerical kernel on behalf of another.
    async fn dispatch_ray_pipeline(
        &mut self,
        asset_url: &str,
        start: usize,
        end: usize,
        height_mm: f64,
        signal: &JsValue,
        near_shader: &str,
        remainder_shader: &str,
        solver_label: &str,
    ) -> Result<Option<Vec<f32>>, String> {
        self.mesh_buffers(asset_url).await?;
        let key = format!("{asset_url}:mesh");
        let data = self.mesh_buffers[&key].clone();
        let dir_count = data.layout.dir_count;
        let face_count = data.layout.face_count;
        let kernel_count = data.layout.kernel_count;
        let count = end - start;

        let analytic_pipeline = self.pipeline(near_shader, near_shader).await?;
        let inside_pipeline = self.pipeline("rays", "inside_probe").await?;
        let rays_pipeline = self.pipeline("rays", "rays").await?;
        let remainder_pipeline = self
            .pipeline(remainder_shader, remainder_shader)
            .await?;
        let scalar_pipeline = self.pipeline("tensor_scalar", "ray_scalar").await?;
        let (analytic_groups_x, analytic_groups_y) = point_grid(count);
        let observer = self
            .observer_resources(&data.faces, start, count, height_mm)
            .await?;
        let points = observer.points.clone();

        let w_output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("analytic tensors"),
            size: (count * 6 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let mut analytic_globals = [0u8; 32];
        set_u32(&mut analytic_globals, 0, count as u32);
        set_u32(&mut analytic_globals, 4, face_count as u32);
        set_f32(&mut analytic_globals, 8, G as f32);
        set_u32(&mut analytic_globals, 12, analytic_groups_x);
        set_u32(&mut analytic_globals, 16, 0);
        let analytic_global_buffer = self.storage_buffer("analytic globals", &analytic_globals);
        let analytic_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ray analytic"),
            layout: &analytic_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: analytic_global_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: data.faces.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: points.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: w_output.as_entire_binding(),
                },
            ],
        });

        let counts = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ray counts"),
            size: (count * dir_count * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let intervals = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ray intervals"),
            size: (count * dir_count * MAX_INTERVALS * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let inside = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("inside and overflow"),
            size: ((count + 1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // `rays` writes every `out_counts[idx]` and all MAX_INTERVALS interval
        // slots for each (point, direction), so neither buffer needs clearing.
        // Only the per-point inside flag plus its atomic overflow counter must
        // start at zero, because `inside_probe` stores the flag and the trace
        // path accumulates the overflow counter. Clearing the interval buffer
        // cost a ~19 MB allocation and upload on every block.
        self.queue
            .write_buffer(&inside, 0, &vec![0u8; (count + 1) * 4]);
        let mut ray_globals = [0u8; 32];
        set_u32(&mut ray_globals, 0, count as u32);
        set_u32(&mut ray_globals, 4, dir_count as u32);
        set_u32(&mut ray_globals, 8, face_count as u32);
        let ray_groups = (count * dir_count).div_ceil(64).max(1);
        let ray_grid = grid2d(ray_groups, 1);
        set_u32(&mut ray_globals, 12, ray_grid.0);
        set_f32(&mut ray_globals, 16, T_MIN as f32);
        set_f32(&mut ray_globals, 20, T_MAX as f32);
        let ray_global_buffer = self.storage_buffer("ray globals", &ray_globals);
        let rays_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rays"),
            layout: &rays_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ray_global_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: data.positions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: data.indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: data.bounds.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: data.links.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: points.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: data.directions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: intervals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: inside.as_entire_binding(),
                },
            ],
        });
        // `inside_probe` is a second entry point of the same module. With a
        // default (auto) layout each entry point gets its *own* layout, holding
        // only the bindings it actually uses — the probe traces rays and never
        // touches the direction quadrature, the interval counts or the interval
        // slots. Binding the full `rays` group to the probe pass therefore
        // invalidates the whole command buffer, and a discarded submission reads
        // back as an all-zero block rather than as an error.
        let inside_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rays inside probe"),
            layout: &inside_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ray_global_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: data.positions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: data.indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: data.bounds.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: data.links.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: points.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: inside.as_entire_binding(),
                },
            ],
        });

        let remainder_output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("remainder tensors"),
            size: (count * 6 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let mut remainder_globals = [0u8; 16];
        set_u32(&mut remainder_globals, 0, count as u32);
        set_u32(&mut remainder_globals, 4, dir_count as u32);
        set_u32(&mut remainder_globals, 8, kernel_count as u32);
        set_f32(&mut remainder_globals, 12, G as f32);
        let remainder_global_buffer = self.storage_buffer("remainder globals", &remainder_globals);
        let remainder_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("remainder"),
            layout: &remainder_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: remainder_global_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: intervals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: points.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: data.kernels.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: data.directions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: remainder_output.as_entire_binding(),
                },
            ],
        });
        let scalar_output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ray face scalars"),
            size: (count * 4).max(16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut scalar_globals = [0u8; 16];
        set_u32(&mut scalar_globals, 0, count as u32);
        set_u32(&mut scalar_globals, 4, kernel_count as u32);
        let scalar_global_buffer = self.storage_buffer("ray scalar globals", &scalar_globals);
        let scalar_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ray scalars"),
            layout: &scalar_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scalar_global_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: w_output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: remainder_output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: points.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: data.kernels.as_entire_binding(),
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
                &w_output,
                &analytic_global_buffer,
                &counts,
                &intervals,
                &inside,
                &ray_global_buffer,
                &remainder_output,
                &remainder_global_buffer,
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
        for pass_kind in 0..5 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            match pass_kind {
                0 => {
                    pass.set_pipeline(&analytic_pipeline);
                    pass.set_bind_group(0, &analytic_bind, &[]);
                    pass.dispatch_workgroups(analytic_groups_x, analytic_groups_y, 1);
                }
                1 => {
                    pass.set_pipeline(&inside_pipeline);
                    pass.set_bind_group(0, &inside_bind, &[]);
                    pass.dispatch_workgroups(count.div_ceil(64) as u32, 1, 1);
                }
                2 => {
                    pass.set_pipeline(&rays_pipeline);
                    pass.set_bind_group(0, &rays_bind, &[]);
                    pass.dispatch_workgroups(ray_grid.0, ray_grid.1, 1);
                }
                3 => {
                    pass.set_pipeline(&remainder_pipeline);
                    pass.set_bind_group(0, &remainder_bind, &[]);
                    pass.dispatch_workgroups(count.div_ceil(64) as u32, 1, 1);
                }
                _ => {
                    pass.set_pipeline(&scalar_pipeline);
                    pass.set_bind_group(0, &scalar_bind, &[]);
                    pass.dispatch_workgroups(count.div_ceil(64) as u32, 1, 1);
                }
            }
        }
        self.queue.submit(Some(encoder.finish()));
        if let Some(error) = error_scope.pop().await {
            return Err(format!(
                "WebGPU {solver_label} pipeline failed validation: {error}"
            ));
        }

        // `out_inside_overflow[count]` is the atomic counter `rays` bumps
        // whenever crossings or visible intervals exceed the fixed shader
        // capacity. Reading it back turns a silently truncated integral into a
        // visible failure.
        let raw = self
            .read_buffer(
                &scalar_output,
                (count * 4) as u64,
                Some((&inside, (count * 4) as u64, 4)),
            )
            .await;
        for buffer in [
            &points,
            &observer.global_buffer,
            &w_output,
            &analytic_global_buffer,
            &counts,
            &intervals,
            &inside,
            &ray_global_buffer,
            &remainder_output,
            &remainder_global_buffer,
            &scalar_output,
            &scalar_global_buffer,
        ] {
            buffer.destroy();
        }
        let raw = raw?;
        let overflow = u32::from_le_bytes(raw[count * 4..count * 4 + 4].try_into().unwrap());
        if overflow != 0 {
            return Err(format!(
                "ray traversal exceeded a hit, interval, or BVH-stack capacity {overflow} times \
                 at height {height_mm} mm"
            ));
        }
        Ok(Some(bytes_to_f32(&raw, count)))
    }

}
