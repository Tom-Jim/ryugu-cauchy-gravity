impl GpuSolver {
    async fn run_carlson_alpha(
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
                    "carlson_surface",
                    "carlson_surface",
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
        self.run_carlson_alpha_ray_pipeline(
            &pipeline_url(ASSET_CARLSON_ALPHA, &base_url),
            start,
            end,
            height_mm,
            signal,
        )
        .await
    }

    async fn run_carlson_alpha_ray_pipeline(
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
            "carlson_alpha",
            "CarlsonAlpha",
            OutputMode::Scalar,
            None,
            None,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_carlson_alpha_tensor(
        &mut self,
        constant: bool,
        start: usize,
        end: usize,
        height_mm: f64,
        ray_nodes: Option<usize>,
        quadrature_limit: Option<usize>,
        signal: &JsValue,
    ) -> Result<Option<Vec<f32>>, String> {
        let base_url = self.base_url.clone();
        if constant {
            return self
                .run_surface(
                    "carlson_surface",
                    "carlson_surface",
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
            &pipeline_url(ASSET_CARLSON_ALPHA, &base_url),
            start,
            end,
            height_mm,
            signal,
            "rtfp_near",
            "carlson_alpha",
            "CarlsonAlpha",
            OutputMode::Tensor,
            ray_nodes,
            quadrature_limit,
        )
        .await
    }
}
