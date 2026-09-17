/// Upload the mesh, the BVH and the observation points once.
    // Every argument is a distinct immutable input buffer of the upload, and
    // bundling them into a builder would only move the list one level up.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: Device,
        mesh: &Mesh,
        bvh: &Bvh,
        faces: &[Face],
        points: &[[f64; 3]],
        dirs: &Dirs,
        block_capacity: usize,
        max_kernels: usize,
    ) -> Self {
        let Device {
            device,
            queue,
            adapter_name: _,
            inside_pipeline,
            rays_pipeline,
            analytic_pipeline,
            carlson_surface_pipeline,
            remainder_pipeline,
            carlson_alpha_pipeline,
            rays_layout,
            analytic_layout,
            remainder_layout,
        } = device;
        let n_points = points.len();
        let n_dirs = dirs.len();
        let n_tris = mesh.face_count();

        let (positions, indices, nodes, links) = mesh_buffers(&device, &queue, mesh, bvh);
        let face_buf = upload(
            &device,
            &queue,
            "faces",
            &faces_to_f32(faces),
            wgpu::BufferUsages::STORAGE,
        );
        let dirs_buf = upload(
            &device,
            &queue,
            "dirs",
            &dirs_to_f32(dirs),
            wgpu::BufferUsages::STORAGE,
        );
        let points_all = upload(
            &device,
            &queue,
            "points_all",
            &points_to_f32(points),
            wgpu::BufferUsages::STORAGE,
        );
        let points_block = storage_buffer(&device, "points_block", block_capacity * 16);
        let kernels = storage_buffer(&device, "kernels", max_kernels.max(1) * 8 * 4);
        let counts = storage_buffer(&device, "counts", block_capacity * n_dirs * 4);
        let ivals = storage_buffer(
            &device,
            "ivals",
            block_capacity * n_dirs * MAX_INTERVALS * 8,
        );
        // The final word is an atomic overflow counter shared by both ray entry
        // points. Keeping it in this buffer avoids adding another storage binding.
        let inside = storage_buffer(&device, "inside_and_overflow", (block_capacity + 1) * 4);
        let w_out = storage_buffer(&device, "w_out", n_points * 6 * 4);
        let rem_out = storage_buffer(&device, "rem_out", block_capacity * 6 * 4);
        let rays_globals = uniform_buffer(&device, "rays_globals", 32);
        let analytic_globals = uniform_buffer(&device, "analytic_globals", 32);
        let remainder_globals = uniform_buffer(&device, "remainder_globals", 16);

        fn entry<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
            wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }
        }
        let rays_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rays"),
            layout: &rays_layout,
            entries: &[
                entry(0, &rays_globals),
                entry(1, &positions),
                entry(2, &indices),
                entry(3, &nodes),
                entry(4, &links),
                entry(5, &points_block),
                entry(6, &dirs_buf),
                entry(7, &counts),
                entry(8, &ivals),
                entry(9, &inside),
            ],
        });
        let analytic_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("analytic"),
            layout: &analytic_layout,
            entries: &[
                entry(0, &analytic_globals),
                entry(1, &face_buf),
                entry(2, &points_all),
                entry(3, &w_out),
            ],
        });
        let remainder_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("remainder"),
            layout: &remainder_layout,
            entries: &[
                entry(0, &remainder_globals),
                entry(1, &ivals),
                entry(2, &counts),
                entry(3, &points_block),
                entry(4, &kernels),
                entry(5, &dirs_buf),
                entry(6, &rem_out),
            ],
        });

        // One readback buffer, sized for the largest result (six tensor values
        // plus one inside flag per point, then the overflow counter).
        let readback_len = (n_points.max(block_capacity) * 7 * 4 + 4).max(16);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: readback_len as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            n_points,
            n_dirs,
            n_tris,
            n_faces: faces.len(),
            block_capacity,
            inside_pipeline,
            rays_pipeline,
            analytic_pipeline,
            carlson_surface_pipeline,
            remainder_pipeline,
            carlson_alpha_pipeline,
            rays_globals,
            analytic_globals,
            remainder_globals,
            points_block,
            kernels,
            inside,
            w_out,
            rem_out,
            rays_bind,
            analytic_bind,
            remainder_bind,
            readback,
            readback_len,
        }
    }

