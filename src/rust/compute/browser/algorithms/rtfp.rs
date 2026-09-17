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
            )
            .await;
    }
    self.run_ray_pipeline(
        &pipeline_url(ASSET_RTFP, &base_url),
        start,
        end,
        height_mm,
        signal,
        "rtfp_near",
        false,
    )
    .await
}
