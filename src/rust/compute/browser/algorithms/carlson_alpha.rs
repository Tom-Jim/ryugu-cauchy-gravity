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
                "carlson_alpha_near",
                "carlson_alpha_near",
                &pipeline_url(ASSET_GEOMETRY, &base_url),
                FACE_MAGIC_WERNER,
                &pipeline_url(ASSET_GEOMETRY, &base_url),
                FACE_MAGIC_WERNER,
                start,
                end,
                height_mm,
                G * CONSTANT_DENSITY,
                signal,
            )
            .await;
    }
    self.run_ray_pipeline(
        &pipeline_url(ASSET_CARLSON_ALPHA, &base_url),
        start,
        end,
        height_mm,
        signal,
        "carlson_alpha_near",
        true,
    )
    .await
}
