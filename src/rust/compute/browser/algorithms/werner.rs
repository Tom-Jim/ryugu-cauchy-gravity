impl GpuSolver {
    async fn run_werner(
        &mut self,
        start: usize,
        end: usize,
        height_mm: f64,
        signal: &JsValue,
    ) -> Result<Option<Vec<f32>>, String> {
        let base_url = self.base_url.clone();
        self.run_surface(
            "werner",
            "werner",
            &pipeline_url(ASSET_GEOMETRY, &base_url),
            FACE_MAGIC_WERNER,
            &pipeline_url(ASSET_GEOMETRY, &base_url),
            FACE_MAGIC_WERNER,
            start,
            end,
            height_mm,
            G * CONSTANT_DENSITY,
            signal,
            OutputMode::Scalar,
        )
        .await
    }

    async fn run_werner_tensor(
        &mut self,
        start: usize,
        end: usize,
        height_mm: f64,
        signal: &JsValue,
    ) -> Result<Option<Vec<f32>>, String> {
        let base_url = self.base_url.clone();
        self.run_surface(
            "werner",
            "werner",
            &pipeline_url(ASSET_GEOMETRY, &base_url),
            FACE_MAGIC_WERNER,
            &pipeline_url(ASSET_GEOMETRY, &base_url),
            FACE_MAGIC_WERNER,
            start,
            end,
            height_mm,
            G * CONSTANT_DENSITY,
            signal,
            OutputMode::Tensor,
        )
        .await
    }
}
