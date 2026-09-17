async fn run_carlson(
    &mut self,
    constant: bool,
    start: usize,
    end: usize,
    height_mm: f64,
    signal: &JsValue,
) -> Result<Option<Vec<f32>>, String> {
    let asset = if constant {
        ASSET_CARLSON_CONSTANT
    } else {
        ASSET_CARLSON_CAUCHY
    };
    let base_url = self.base_url.clone();
    self.run_surface(
        "carlson_surface",
        "carlson_surface",
        &pipeline_url(asset, &base_url),
        FACE_MAGIC_CARLSON,
        &pipeline_url(ASSET_GEOMETRY, &base_url),
        FACE_MAGIC_WERNER,
        start,
        end,
        height_mm,
        G,
        signal,
    )
    .await
}

