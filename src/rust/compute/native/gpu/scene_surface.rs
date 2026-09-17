impl Scene {
/// `W(x)` (unit density, `G` included) at every observation point.
    pub fn analytic_tensors(&self) -> Result<Vec<Sym6>, String> {
        self.analytic_tensors_impl(self.n_faces, &self.analytic_pipeline, "analytic")
    }

    /// Uniform-density polyhedral tensor for one observation block.
    ///
    /// The full analytic pass is independent of the radial remainder, so a
    /// resumed RT-FP run only evaluates this for points after the saved prefix.
    pub fn analytic_tensors_block(&self, lo: usize, hi: usize) -> Result<Vec<Sym6>, String> {
        let n = hi - lo;
        let (grid_x, grid_y) = grid_2d(n as u32);
        let mut globals = [0u8; 32];
        globals[0..4].copy_from_slice(&(n as u32).to_le_bytes());
        globals[4..8].copy_from_slice(&(self.n_faces as u32).to_le_bytes());
        globals[8..12].copy_from_slice(&(G as f32).to_le_bytes());
        globals[12..16].copy_from_slice(&grid_x.to_le_bytes());
        globals[16..20].copy_from_slice(&(lo as u32).to_le_bytes());
        self.queue.write_buffer(&self.analytic_globals, 0, &globals);

        let mut enc = self
            .device
            .create_command_encoder(&empty_encoder("analytic_block"));
        {
            let mut pass = enc.begin_compute_pass(&compute_pass("analytic_block"));
            pass.set_pipeline(&self.analytic_pipeline);
            pass.set_bind_group(0, &self.analytic_bind, &[]);
            pass.dispatch_workgroups(grid_x, grid_y, 1);
        }
        enc.copy_buffer_to_buffer(
            &self.w_out,
            (lo * 24) as u64,
            &self.readback,
            0,
            (n * 24) as u64,
        );
        self.queue.submit(Some(enc.finish()));
        let data = self.read_f32(n * 6)?;
        Ok(data
            .as_chunks::<6>()
            .0
            .iter()
            .map(|chunk| {
                [
                    chunk[0] as f64,
                    chunk[1] as f64,
                    chunk[2] as f64,
                    chunk[3] as f64,
                    chunk[4] as f64,
                    chunk[5] as f64,
                ]
            })
            .collect())
    }

    /// `H(x)` for the Carlson density-jump surface list.
    ///
    /// This dispatch uses its own WGSL entry point and never executes the
    /// RT-FP radial-remainder or ray pipelines.
    pub fn carlson_surface_tensors(&self) -> Result<Vec<Sym6>, String> {
        let mut out = vec![[0.0; 6]; self.n_points];
        for lo in (0..self.n_points).step_by(self.block_capacity) {
            let hi = (lo + self.block_capacity).min(self.n_points);
            self.carlson_surface_block(lo, hi, &mut out)?;
            println!("PROGRESS_R {hi} {}", self.n_points);
            std::io::Write::flush(&mut std::io::stdout()).ok();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        Ok(out)
    }

    pub fn carlson_surface_block(
        &self,
        lo: usize,
        hi: usize,
        out: &mut [[f64; 6]],
    ) -> Result<(), String> {
        let n = hi - lo;
        let (grid_x, grid_y) = grid_2d(n as u32);
        let mut globals = [0u8; 32];
        globals[0..4].copy_from_slice(&(n as u32).to_le_bytes());
        globals[4..8].copy_from_slice(&(self.n_faces as u32).to_le_bytes());
        globals[8..12].copy_from_slice(&(G as f32).to_le_bytes());
        globals[12..16].copy_from_slice(&grid_x.to_le_bytes());
        globals[16..20].copy_from_slice(&(lo as u32).to_le_bytes());
        self.queue.write_buffer(&self.analytic_globals, 0, &globals);

        let mut enc = self
            .device
            .create_command_encoder(&empty_encoder("carlson_surface"));
        {
            let mut pass = enc.begin_compute_pass(&compute_pass("carlson_surface"));
            pass.set_pipeline(&self.carlson_surface_pipeline);
            pass.set_bind_group(0, &self.analytic_bind, &[]);
            pass.dispatch_workgroups(grid_x, grid_y, 1);
        }
        enc.copy_buffer_to_buffer(
            &self.w_out,
            (lo * 24) as u64,
            &self.readback,
            0,
            (n * 24) as u64,
        );
        self.queue.submit(Some(enc.finish()));
        let data = self.read_f32(n * 6)?;
        for (i, chunk) in data.as_chunks::<6>().0.iter().enumerate() {
            out[lo + i] = [
                chunk[0] as f64,
                chunk[1] as f64,
                chunk[2] as f64,
                chunk[3] as f64,
                chunk[4] as f64,
                chunk[5] as f64,
            ];
        }
        Ok(())
    }

    fn analytic_tensors_impl(
        &self,
        n_faces: usize,
        pipeline: &wgpu::ComputePipeline,
        label: &str,
    ) -> Result<Vec<Sym6>, String> {
        let (grid_x, grid_y) = grid_2d(self.n_points as u32);
        let mut globals = [0u8; 32];
        globals[0..4].copy_from_slice(&(self.n_points as u32).to_le_bytes());
        globals[4..8].copy_from_slice(&(n_faces as u32).to_le_bytes());
        globals[8..12].copy_from_slice(&(G as f32).to_le_bytes());
        globals[12..16].copy_from_slice(&grid_x.to_le_bytes());
        self.queue.write_buffer(&self.analytic_globals, 0, &globals);

        let mut enc = self.device.create_command_encoder(&empty_encoder(label));
        {
            let mut pass = enc.begin_compute_pass(&compute_pass(label));
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.analytic_bind, &[]);
            pass.dispatch_workgroups(grid_x, grid_y, 1);
        }
        enc.copy_buffer_to_buffer(
            &self.w_out,
            0,
            &self.readback,
            0,
            (self.n_points * 24) as u64,
        );
        self.queue.submit(Some(enc.finish()));
        let data = self.read_f32(self.n_points * 6)?;
        Ok(data
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

}
