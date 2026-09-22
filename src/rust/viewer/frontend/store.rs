//! Browser-managed temporary storage for completed and checkpointed RHGF records.
//!
//! This is the Rust replacement for the storage half of `src/web/app.js`. The
//! database layout is unchanged, so a viewer that already has saved results can
//! keep reading them:
//!
//! * database `ryugu-cauchy-gravity`, version 3,
//! * object store `saved-results`, key path `id`, holding completed temporary records,
//! * object store `temporary-results`, key path `id`, holding resumable
//!   checkpoints.
//! * object store `resources`, key path `id`, holding selected GLB/TOML assets.
//!
//! Nothing here computes anything; it moves bytes in and out of the browser's
//! own temporary storage so a reload can resume instead of restarting. The
//! browser owns eviction of this storage; no project file is written.

use std::cell::RefCell;
use std::rc::Rc;

use js_sys::{Array, Promise, Reflect, Uint8Array};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

const DB_NAME: &str = "ryugu-cauchy-gravity";
const DB_VERSION: u32 = 3;
const SAVED_STORE: &str = "saved-results";
const TEMP_STORE: &str = "temporary-results";
const RESOURCE_STORE: &str = "resources";

/// Bump whenever a solver source buffer, kernel set or WGSL pipeline changes.
///
/// An old checkpoint would otherwise be mistaken for a resumable result from
/// the new geometry.
pub const LIVE_GEOMETRY_VERSION: &str = "2026-09-17-device-limits-v3";

/// Metadata of one saved 100 % record, without its payload.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SavedRow {
    pub id: String,
    pub algo: String,
    pub algorithm: String,
    pub density: String,
    pub standoff_mm: f64,
    pub saved_at: String,
    pub faces: f64,
    pub bytes: Option<Vec<u8>>,
}

/// One resumable checkpoint.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TempRow {
    pub id: String,
    pub algo: String,
    pub algorithm: String,
    pub density: String,
    pub standoff_mm: f64,
    pub completed: f64,
    pub total: f64,
    pub complete: bool,
    pub updated_at: String,
    pub bytes: Vec<u8>,
}

fn to_error(value: JsValue) -> String {
    value.as_string().unwrap_or_else(|| format!("{value:?}"))
}

fn field(object: &JsValue, name: &str) -> JsValue {
    Reflect::get(object, &JsValue::from_str(name)).unwrap_or(JsValue::UNDEFINED)
}

fn string_field(object: &JsValue, name: &str) -> String {
    field(object, name).as_string().unwrap_or_default()
}

fn number_field(object: &JsValue, name: &str) -> f64 {
    field(object, name).as_f64().unwrap_or(0.0)
}

fn set_field(object: &JsValue, name: &str, value: JsValue) {
    let _ = Reflect::set(object, &JsValue::from_str(name), &value);
}

/// Wrap an IndexedDB request in a promise.
///
/// The `onsuccess` / `onerror` handlers are leaked on purpose: an event-handler
/// property keeps the JavaScript function alive, but the Rust `Closure` that
/// owns its body must outlive it, and there is no scope to tie it to here.
fn request_promise(request: &web_sys::IdbRequest) -> Promise {
    Promise::new(&mut |resolve, reject| {
        let resolve_on_success = resolve.clone();
        let request_on_success = request.clone();
        let on_success = Closure::<dyn FnMut(JsValue)>::new(move |_| {
            let value = request_on_success.result().unwrap_or(JsValue::NULL);
            let _ = resolve_on_success.call1(&JsValue::NULL, &value);
        })
        .into_js_value();
        request.set_onsuccess(Some(on_success.unchecked_ref()));

        let reject_on_error = reject.clone();
        let request_on_error = request.clone();
        let on_error = Closure::<dyn FnMut(JsValue)>::new(move |_| {
            let value = request_on_error
                .error()
                .ok()
                .flatten()
                .map(JsValue::from)
                .unwrap_or_else(|| JsValue::from_str("IndexedDB request failed"));
            let _ = reject_on_error.call1(&JsValue::NULL, &value);
        })
        .into_js_value();
        request.set_onerror(Some(on_error.unchecked_ref()));
    })
}

async fn await_request(request: &web_sys::IdbRequest) -> Result<JsValue, String> {
    JsFuture::from(request_promise(request))
        .await
        .map_err(to_error)
}

/// The viewer's persistence handle. One per session.
///
/// The handle is shared through an `Rc` and every method takes `&self`, so a
/// checkpoint write can run while the compute loop is borrowing the session.
#[derive(Clone)]
pub struct Store {
    db: Rc<RefCell<Option<web_sys::IdbDatabase>>>,
}

impl Store {
    pub fn new() -> Store {
        Store {
            db: Rc::new(RefCell::new(None)),
        }
    }

    /// Open (and on first use create) the viewer database.
    async fn open(&self) -> Result<web_sys::IdbDatabase, String> {
        if let Some(db) = self.db.borrow().as_ref() {
            return Ok(db.clone());
        }
        let db = open_db().await?;
        *self.db.borrow_mut() = Some(db.clone());
        Ok(db)
    }

    async fn run(
        &self,
        store_name: &str,
        mode: web_sys::IdbTransactionMode,
        action: impl FnOnce(&web_sys::IdbObjectStore) -> Result<web_sys::IdbRequest, JsValue>,
    ) -> Result<JsValue, String> {
        let db = self.open().await?;
        let transaction = db
            .transaction_with_str_and_mode(store_name, mode)
            .map_err(to_error)?;
        let store = transaction.object_store(store_name).map_err(to_error)?;
        let request = action(&store).map_err(to_error)?;
        await_request(&request).await
    }

    /// Metadata of every saved record, ordered the way the panel lists them.
    ///
    /// Payloads are deliberately left behind: the list is rendered on every
    /// save, and a 196 608-face record is 786 KiB.
    pub async fn list_saved(&self) -> Result<Vec<SavedRow>, String> {
        let value = self
            .run(
                SAVED_STORE,
                web_sys::IdbTransactionMode::Readonly,
                |store| store.get_all(),
            )
            .await?;
        let rows = Array::from(&value);
        let mut items: Vec<SavedRow> = rows
            .iter()
            .map(|row| {
                let mut item = saved_from_object(&row);
                item.bytes = None;
                item
            })
            .collect();
        items.sort_by(|a, b| {
            a.algorithm
                .cmp(&b.algorithm)
                .then(
                    a.standoff_mm
                        .partial_cmp(&b.standoff_mm)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
                .then(a.density.cmp(&b.density))
        });
        Ok(items)
    }

    pub async fn get_saved(&self, id: &str) -> Result<Option<SavedRow>, String> {
        let value = self
            .run(
                SAVED_STORE,
                web_sys::IdbTransactionMode::Readonly,
                |store| store.get(&JsValue::from_str(id)),
            )
            .await?;
        Ok(saved_bytes(&value).map(|bytes| {
            let mut row = saved_from_object(&value);
            row.bytes = Some(bytes);
            row
        }))
    }

    pub async fn put_saved(&self, row: &SavedRow) -> Result<(), String> {
        let object = saved_to_object(row);
        self.run(
            SAVED_STORE,
            web_sys::IdbTransactionMode::Readwrite,
            |store| store.put(&object),
        )
        .await
        .map(|_| ())
    }

    pub async fn delete_saved(&self, id: &str) -> Result<(), String> {
        self.run(
            SAVED_STORE,
            web_sys::IdbTransactionMode::Readwrite,
            |store| store.delete(&JsValue::from_str(id)),
        )
        .await
        .map(|_| ())
    }

    pub async fn get_temp(&self, id: &str) -> Result<Option<TempRow>, String> {
        let value = self
            .run(TEMP_STORE, web_sys::IdbTransactionMode::Readonly, |store| {
                store.get(&JsValue::from_str(id))
            })
            .await?;
        if value.is_undefined() || value.is_null() {
            return Ok(None);
        }
        Ok(Some(TempRow {
            id: string_field(&value, "id"),
            algo: string_field(&value, "algo"),
            algorithm: string_field(&value, "algorithm"),
            density: string_field(&value, "density"),
            standoff_mm: number_field(&value, "standoffMm"),
            completed: number_field(&value, "completed"),
            total: number_field(&value, "total"),
            complete: field(&value, "complete").as_bool().unwrap_or(false),
            updated_at: string_field(&value, "updatedAt"),
            bytes: temp_bytes(&value).unwrap_or_default(),
        }))
    }

    pub async fn put_temp(&self, row: &TempRow) -> Result<(), String> {
        let object = js_sys::Object::new();
        set_field(&object, "id", JsValue::from_str(&row.id));
        set_field(&object, "algo", JsValue::from_str(&row.algo));
        set_field(&object, "algorithm", JsValue::from_str(&row.algorithm));
        set_field(&object, "density", JsValue::from_str(&row.density));
        set_field(&object, "standoffMm", JsValue::from_f64(row.standoff_mm));
        set_field(&object, "completed", JsValue::from_f64(row.completed));
        set_field(&object, "total", JsValue::from_f64(row.total));
        set_field(&object, "complete", JsValue::from_bool(row.complete));
        set_field(&object, "updatedAt", JsValue::from_str(&row.updated_at));
        set_field(
            &object,
            "bytes",
            Uint8Array::from(row.bytes.as_slice()).into(),
        );
        let object: JsValue = object.into();
        self.run(
            TEMP_STORE,
            web_sys::IdbTransactionMode::Readwrite,
            |store| store.put(&object),
        )
        .await
        .map(|_| ())
    }
}

fn saved_from_object(value: &JsValue) -> SavedRow {
    if value.is_undefined() || value.is_null() {
        return SavedRow::default();
    }
    SavedRow {
        id: string_field(value, "id"),
        algo: string_field(value, "algo"),
        algorithm: string_field(value, "algorithm"),
        density: string_field(value, "density"),
        standoff_mm: number_field(value, "standoffMm"),
        saved_at: string_field(value, "savedAt"),
        faces: number_field(value, "faces"),
        bytes: saved_bytes(value),
    }
}

fn saved_bytes(value: &JsValue) -> Option<Vec<u8>> {
    field(value, "bytes")
        .dyn_into::<Uint8Array>()
        .ok()
        .map(|bytes| bytes.to_vec())
}

fn temp_bytes(value: &JsValue) -> Option<Vec<u8>> {
    saved_bytes(value)
}

fn saved_to_object(row: &SavedRow) -> JsValue {
    let object = js_sys::Object::new();
    set_field(&object, "id", JsValue::from_str(&row.id));
    set_field(&object, "algo", JsValue::from_str(&row.algo));
    set_field(&object, "algorithm", JsValue::from_str(&row.algorithm));
    set_field(&object, "density", JsValue::from_str(&row.density));
    set_field(&object, "standoffMm", JsValue::from_f64(row.standoff_mm));
    set_field(&object, "savedAt", JsValue::from_str(&row.saved_at));
    set_field(&object, "faces", JsValue::from_f64(row.faces));
    if let Some(bytes) = &row.bytes {
        set_field(&object, "bytes", Uint8Array::from(bytes.as_slice()).into());
    }
    object.into()
}

/// Key of one `(algorithm, density, height)` triple.
pub fn temporary_id(algo: &str, density: &str, standoff_mm: f64) -> String {
    format!(
        "{LIVE_GEOMETRY_VERSION}|{algo}|{density}|{:.3}",
        standoff_mm
    )
}

/// Key of one saved 100 % record: SHA-256 of the same triple, first 20 hex digits.
///
/// Async because the digest comes from the platform's own cryptographic hash.
pub async fn saved_id(algo: &str, density: &str, standoff_mm: f64) -> String {
    let input = format!("{algo}|{density}|{:.3}", standoff_mm);
    let digest = sha256(input.as_bytes()).await;
    digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>()
        .chars()
        .take(20)
        .collect()
}

async fn sha256(bytes: &[u8]) -> Vec<u8> {
    let window = match web_sys::window() {
        Some(window) => window,
        None => return Vec::new(),
    };
    let Ok(crypto) = window.crypto() else {
        // A record without a stable digest is still usable in this session; it
        // simply cannot be found again after a reload.
        return bytes.iter().copied().take(20).collect();
    };
    let subtle = crypto.subtle();
    let source = Uint8Array::from(bytes);
    let promise = match subtle.digest_with_str_and_buffer_source("SHA-256", &source) {
        Ok(promise) => promise,
        Err(_) => return bytes.iter().copied().take(20).collect(),
    };
    let value = match JsFuture::from(promise).await {
        Ok(value) => value,
        Err(_) => return bytes.iter().copied().take(20).collect(),
    };
    Uint8Array::new(&value).to_vec()
}

async fn open_db() -> Result<web_sys::IdbDatabase, String> {
    let window = web_sys::window().ok_or("no browser window")?;
    let factory = window
        .indexed_db()
        .map_err(to_error)?
        .ok_or("IndexedDB is unavailable in this browser context")?;
    let request = factory
        .open_with_u32(DB_NAME, DB_VERSION)
        .map_err(to_error)?;

    let upgrade_request = request.clone();
    let on_upgrade = Closure::<dyn FnMut(JsValue)>::new(move |_| {
        let Ok(db) = upgrade_request
            .result()
            .and_then(|value| value.dyn_into::<web_sys::IdbDatabase>())
        else {
            return;
        };
        let names = db.object_store_names();
        for name in [SAVED_STORE, TEMP_STORE, RESOURCE_STORE] {
            if names.contains(name) {
                continue;
            }
            let parameters = web_sys::IdbObjectStoreParameters::new();
            parameters.set_key_path(&JsValue::from_str("id"));
            let _ = db.create_object_store_with_optional_parameters(name, &parameters);
        }
    })
    .into_js_value();
    request.set_onupgradeneeded(Some(on_upgrade.unchecked_ref()));

    let value = await_request(&request).await?;
    value
        .dyn_into::<web_sys::IdbDatabase>()
        .map_err(|_| "IndexedDB open did not produce a database".to_string())
}
