//! Browser session controller.
//!
//! This is the Rust replacement for the inline `type="module"` script that used
//! to carry the whole viewer session in `src/web/index.html`. Everything that
//! used to be computed in JavaScript now happens here: the observation-height
//! slider algebra, the RHGF v5 record reader, the colour-window statistics, the
//! face-by-face algorithm comparison, the single-tab lease and the browser compute
//! queue. The page keeps only a view layer: a Vue store the controller writes
//! into, and the IndexedDB adapter in `store.rs`.
//!
//! The controller never computes gravity itself. It drives `ComputeEngine`, which
//! dispatches the WGSL kernels, and it owns the presentation state around it.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use js_sys::{Array, Function, Object, Promise, Reflect, Uint8Array};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::{JsFuture, spawn_local};

use super::store::{SavedRow, Store, TempRow, saved_id, temporary_id};
use crate::compute::ComputeEngine;

include!("height.rs");
include!("algorithms.rs");
include!("record.rs");
include!("statistics.rs");
include!("comparison_spec.rs");
include!("platform.rs");
include!("state.rs");
include!("diagnostics.rs");
include!("api.rs");
include!("bootstrap.rs");
include!("labels.rs");
include!("status.rs");
include!("jobs/controls.rs");
include!("jobs/queue.rs");
include!("jobs/execute.rs");
include!("jobs/render.rs");
include!("comparison.rs");
include!("saved.rs");
include!("memory.rs");
