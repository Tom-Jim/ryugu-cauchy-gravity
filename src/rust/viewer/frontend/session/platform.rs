// ---------------------------------------------------------------------------
// JavaScript glue
// ---------------------------------------------------------------------------

fn field(object: &JsValue, name: &str) -> JsValue {
    Reflect::get(object, &JsValue::from_str(name)).unwrap_or(JsValue::UNDEFINED)
}

fn set(object: &JsValue, name: &str, value: JsValue) {
    let _ = Reflect::set(object, &JsValue::from_str(name), &value);
}

fn set_str(object: &JsValue, name: &str, value: &str) {
    set(object, name, JsValue::from_str(value));
}

fn set_f64(object: &JsValue, name: &str, value: f64) {
    set(object, name, JsValue::from_f64(value));
}

fn set_bool(object: &JsValue, name: &str, value: bool) {
    set(object, name, JsValue::from_bool(value));
}

fn get_bool(object: &JsValue, name: &str) -> bool {
    field(object, name).as_bool().unwrap_or(false)
}

fn get_f64(object: &JsValue, name: &str) -> f64 {
    field(object, name).as_f64().unwrap_or(0.0)
}

fn document() -> Option<web_sys::Document> {
    web_sys::window().and_then(|window| window.document())
}

async fn await_promise(promise: Promise) {
    let _ = JsFuture::from(promise).await;
}

async fn sleep_ms(ms: i32) {
    let promise = Promise::new(&mut |resolve, _reject| {
        let callback = Closure::once_into_js(move || {
            let _ = resolve.call0(&JsValue::NULL);
        });
        if let Some(window) = web_sys::window() {
            let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(
                callback.unchecked_ref(),
                ms,
            );
        }
    });
    await_promise(promise).await;
}

async fn wait_for_frame() {
    let promise = Promise::new(&mut |resolve, _reject| {
        let callback = Closure::once_into_js(move || {
            let _ = resolve.call0(&JsValue::NULL);
        });
        if let Some(window) = web_sys::window() {
            let _ = window.request_animation_frame(callback.unchecked_ref());
        }
    });
    await_promise(promise).await;
}

async fn wait_for_rendered_frames(count: usize) {
    for _ in 0..count {
        wait_for_frame().await;
    }
}

fn current_href() -> String {
    web_sys::window()
        .and_then(|window| window.location().href().ok())
        .unwrap_or_default()
}

fn current_url() -> Option<web_sys::Url> {
    web_sys::Url::new(&current_href()).ok()
}

fn replace_history(url: &str) {
    if let Some(window) = web_sys::window()
        && let Ok(history) = window.history()
    {
        let _ = history.replace_state_with_url(&JsValue::NULL, "", Some(url));
    }
}

fn local_storage() -> Option<web_sys::Storage> {
    web_sys::window()?.local_storage().ok().flatten()
}

fn storage(key: &str) -> Option<String> {
    local_storage()?.get_item(key).ok().flatten()
}

fn set_storage(key: &str, value: &str) {
    if let Some(storage) = local_storage() {
        let _ = storage.set_item(key, value);
    }
}

fn session_storage(key: &str) -> Option<String> {
    web_sys::window()?
        .session_storage()
        .ok()
        .flatten()?
        .get_item(key)
        .ok()
        .flatten()
}

fn set_session_storage(key: &str, value: &str) {
    if let Some(window) = web_sys::window()
        && let Ok(Some(storage)) = window.session_storage()
    {
        let _ = storage.set_item(key, value);
    }
}

fn reload_page() {
    if let Some(window) = web_sys::window() {
        let location = window.location();
        let _ = location.reload();
    }
}

/// `new URL(raw, location.href)` reduced to `origin + pathname`.
fn tab_identity(raw: &str) -> String {
    let Ok(url) = web_sys::Url::new_with_base(raw, &current_href()) else {
        return raw.to_string();
    };
    format!("{}{}", url.origin(), url.pathname())
}

fn random_tab_id() -> String {
    let Some(window) = web_sys::window() else {
        return format!("{}-{}", js_sys::Date::now(), js_sys::Math::random());
    };
    let crypto = field(&window, "crypto");
    let random_uuid = field(&crypto, "randomUUID");
    if let Some(function) = random_uuid.dyn_ref::<Function>()
        && let Ok(value) = function.call0(&crypto)
        && let Some(id) = value.as_string()
        && !id.is_empty()
    {
        return id;
    }
    format!("{}-{}", js_sys::Date::now(), js_sys::Math::random())
}

/// The lease written to `localStorage` by the tab that owns the viewer.
#[derive(Clone, Debug, Default)]
struct Lease {
    id: String,
    identity: String,
    ts: f64,
}

fn read_lease() -> Option<Lease> {
    let raw = storage(TAB_LEASE_KEY)?;
    let value: JsValue = js_sys::JSON::parse(&raw).ok()?;
    Some(Lease {
        id: field(&value, "id").as_string().unwrap_or_default(),
        identity: field(&value, "identity").as_string().unwrap_or_default(),
        ts: get_f64(&value, "ts"),
    })
}

fn write_lease(id: &str, identity: &str) {
    let value = Object::new();
    set_str(&value, "id", id);
    set_str(&value, "identity", identity);
    set_f64(&value, "ts", js_sys::Date::now());
    let raw: String = value.to_string().into();
    set_storage(TAB_LEASE_KEY, &raw);
}

