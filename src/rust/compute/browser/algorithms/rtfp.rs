impl GpuSolver {
    async fn run_rtfp(
        &mut self,
        constant: bool,
        start: usize,
        end: usize,
        height_mm: f64,
        signal: &JsValue,
    ) -> Result<Option<Vec<f32>>, String> {
        let base_url = self.base_url.clone();
        if constant {
            return self
                .run_surface(
                    "rtfp_near",
                    "rtfp_near",
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
                .await;
        }
        self.run_rtfp_ray_pipeline(
            &pipeline_url(ASSET_RTFP, &base_url),
            start,
            end,
            height_mm,
            signal,
        )
        .await
    }

    async fn run_rtfp_ray_pipeline(
        &mut self,
        asset_url: &str,
        start: usize,
        end: usize,
        height_mm: f64,
        signal: &JsValue,
    ) -> Result<Option<Vec<f32>>, String> {
        self.dispatch_ray_pipeline(
            asset_url,
            start,
            end,
            height_mm,
            signal,
            "rtfp_near",
            "remainder",
            "RT-FP",
            OutputMode::Scalar,
            None,
            None,
        )
        .await
    }

    async fn run_rtfp_tensor(
        &mut self,
        constant: bool,
        start: usize,
        end: usize,
        height_mm: f64,
        ray_nodes: Option<usize>,
        signal: &JsValue,
    ) -> Result<Option<Vec<f32>>, String> {
        let base_url = self.base_url.clone();
        if constant {
            return self
                .run_surface(
                    "rtfp_near",
                    "rtfp_near",
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
                .await;
        }
        self.dispatch_ray_pipeline(
            &pipeline_url(ASSET_RTFP, &base_url),
            start,
            end,
            height_mm,
            signal,
            "rtfp_near",
            "remainder",
            "RT-FP",
            OutputMode::Tensor,
            ray_nodes,
            None,
        )
        .await
    }
}
