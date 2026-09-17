// ---------------------------------------------------------------------------
// JavaScript interop helpers
// ---------------------------------------------------------------------------

/// `object[name]`, i.e. `options.onProgress` and friends.
fn field(object: &JsValue, name: &str) -> JsValue {
    js_sys::Reflect::get(object, &JsValue::from_str(name)).unwrap_or(JsValue::UNDEFINED)
}

fn field_f64(object: &JsValue, name: &str) -> Option<f64> {
    field(object, name).as_f64()
}

fn field_string(object: &JsValue, name: &str) -> Option<String> {
    field(object, name).as_string()
}

/// `signal?.aborted`.
fn signal_aborted(signal: &JsValue) -> bool {
    signal
        .dyn_ref::<js_sys::Object>()
        .map(|_| field(signal, "aborted").as_bool().unwrap_or(false))
        .unwrap_or(false)
}

/// `await new Promise((resolve) => setTimeout(resolve, 0))`: hands the event
/// loop back between blocks so progress events and the compositor can run.
async fn yield_to_browser() {
    let mut executor = |resolve: js_sys::Function, _reject: js_sys::Function| {
        if let Some(window) = web_sys::window() {
            let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 0);
        }
    };
    let _ = JsFuture::from(js_sys::Promise::new(&mut executor)).await;
}

/// `fetch(url, { cache: "force-cache" })` followed by `arrayBuffer()`.
async fn fetch_bytes(url: &str) -> Result<Vec<u8>, JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;
    let init = RequestInit::new();
    init.set_cache(RequestCache::ForceCache);
    let request = web_sys::Request::new_with_str_and_init(url, &init)?;
    let response: web_sys::Response = JsFuture::from(window.fetch_with_request(&request))
        .await?
        .dyn_into()?;
    if !response.ok() {
        return Err(JsValue::from_str(&format!(
            "failed to load {url} ({})",
            response.status()
        )));
    }
    let buffer = JsFuture::from(response.array_buffer()?).await?;
    Ok(js_sys::Uint8Array::new(&buffer).to_vec())
}

fn js_error(error: JsValue) -> String {
    error
        .as_string()
        .unwrap_or_else(|| "browser resource request failed".to_string())
}

