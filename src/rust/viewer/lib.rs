//! WebAssembly entry point, split into frontend state, WebGPU computation and
//! Bevy rendering modules.

#![cfg(target_arch = "wasm32")]

mod compute;
mod frontend;
mod render;

pub use frontend::Session;
pub use render::{
    push_bake_update, push_bake_update_preserving_window, run_with_bake,
};
