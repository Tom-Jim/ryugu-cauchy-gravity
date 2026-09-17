/// `G Σ_q ω_q T(u_q) Σ_k w_k R_k(u_q)` for one block of points.
    ///
    /// Runs the whole GPU chain: inside probe → ray intervals → remainder.
    pub fn block_remainder(
        &self,
        points: &[[f64; 3]],
        dirs: &Dirs,
        kernels: &[KernelSi],
        t_max: f32,
        t_min: f32,
    ) -> Result<Vec<Sym6>, String> {
        self.block_remainder_impl(
            points,
            dirs,
            kernels,
            t_max,
            t_min,
            &self.remainder_pipeline,
            "remainder",
            false,
        )
    }

    /// General-alpha radial finite part for the CarlsonAlpha solver.
    pub fn block_carlson_alpha(
        &self,
        points: &[[f64; 3]],
        dirs: &Dirs,
        kernels: &[KernelSi],
        t_max: f32,
        t_min: f32,
    ) -> Result<Vec<Sym6>, String> {
        self.block_remainder_impl(
            points,
            dirs,
            kernels,
            t_max,
            t_min,
            &self.carlson_alpha_pipeline,
            "carlson_alpha",
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn block_remainder_impl(
        &self,
        points: &[[f64; 3]],
        dirs: &Dirs,
        kernels: &[KernelSi],
        t_max: f32,
        t_min: f32,
        pipeline: &wgpu::ComputePipeline,
        label: &str,
        allow_inside: bool,
    ) -> Result<Vec<Sym6>, String> {
        let n = points.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n > self.block_capacity {
            return Err(format!(
                "block of {n} exceeds capacity {}",
                self.block_capacity
            ));
        }
        if kernels.is_empty() {
            return Ok(vec![[0.0; 6]; n]);
        }
        let pts = points_to_f32(points);
        self.queue
            .write_buffer(&self.points_block, 0, as_bytes(&pts));
        let kbuf = kernels_to_f32(kernels);
        self.queue.write_buffer(&self.kernels, 0, as_bytes(&kbuf));
        let zero = 0u32.to_le_bytes();
        self.queue
            .write_buffer(&self.inside, (self.block_capacity * 4) as u64, &zero);

        // struct Globals { n_points, n_dirs, n_tris, grid_x, t_min, t_max, _pad, _pad }
        let ray_groups = ((n * self.n_dirs) as u32).div_ceil(64);
        let (rays_grid_x, rays_grid_y) = grid_2d(ray_groups);
        let mut rays_globals = [0u8; 32];
        rays_globals[0..4].copy_from_slice(&(n as u32).to_le_bytes());
        rays_globals[4..8].copy_from_slice(&(dirs.len() as u32).to_le_bytes());
        rays_globals[8..12].copy_from_slice(&(self.n_tris as u32).to_le_bytes());
        rays_globals[12..16].copy_from_slice(&rays_grid_x.to_le_bytes());
        rays_globals[16..20].copy_from_slice(&t_min.to_le_bytes());
        rays_globals[20..24].copy_from_slice(&t_max.to_le_bytes());
        self.queue
            .write_buffer(&self.rays_globals, 0, &rays_globals);

        let mut rem_globals = [0u8; 16];
        rem_globals[0..4].copy_from_slice(&(n as u32).to_le_bytes());
        rem_globals[4..8].copy_from_slice(&(dirs.len() as u32).to_le_bytes());
        rem_globals[8..12].copy_from_slice(&(kernels.len() as u32).to_le_bytes());
        rem_globals[12..16].copy_from_slice(&(G as f32).to_le_bytes());
        self.queue
            .write_buffer(&self.remainder_globals, 0, &rem_globals);

        let groups = (n as u32).div_ceil(64);
        let mut enc = self.device.create_command_encoder(&empty_encoder("rays"));
        {
            let mut pass = enc.begin_compute_pass(&compute_pass("inside_probe"));
            pass.set_pipeline(&self.inside_pipeline);
            pass.set_bind_group(0, &self.rays_bind, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        {
            let mut pass = enc.begin_compute_pass(&compute_pass("rays"));
            pass.set_pipeline(&self.rays_pipeline);
            pass.set_bind_group(0, &self.rays_bind, &[]);
            pass.dispatch_workgroups(rays_grid_x, rays_grid_y, 1);
        }
        {
            let mut pass = enc.begin_compute_pass(&compute_pass(label));
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.remainder_bind, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.rem_out, 0, &self.readback, 0, (n * 24) as u64);
        enc.copy_buffer_to_buffer(
            &self.inside,
            0,
            &self.readback,
            (n * 24) as u64,
            (n * 4) as u64,
        );
        enc.copy_buffer_to_buffer(
            &self.inside,
            (self.block_capacity * 4) as u64,
            &self.readback,
            (n * 28) as u64,
            4,
        );
        self.queue.submit(Some(enc.finish()));
        let data = self.read_f32(n * 7 + 1)?;
        if let Some(i) = data[n * 6..n * 7].iter().position(|inside| *inside != 0.0) {
            if !allow_inside {
                return Err(format!(
                    "observation point {i} of {n} lies inside the mesh; the contact term is not \
                     enabled for this solver"
                ));
            }
        }
        if data[n * 7] != 0.0 {
            let dropped = data[n * 7].to_bits();
            return Err(format!(
                "ray traversal exceeded the {MAX_INTERVALS}-interval / {}-hit capacity \
                 ({dropped} dropped intersections/intervals; raw bits {:#010x})",
                crate::geom::MAX_HITS,
                dropped
            ));
        }
        Ok(data[..n * 6]
            .as_chunks::<6>()
            .0
            .iter()
            .map(|c| {
                [
                    c[0] as f64,
                    c[1] as f64,
                    c[2] as f64,
                    c[3] as f64,
                    c[4] as f64,
                    c[5] as f64,
                ]
            })
            .collect())
    }

