//! WebAssembly entry point, split into frontend state, WebGPU computation and
//! Bevy rendering modules.

#![cfg(target_arch = "wasm32")]

mod compute;
mod frontend;
mod render;

pub use frontend::Session;
pub use render::{
    push_bake_update, push_bake_update_preserving_window, reset_bake_paint, run_with_bake,
    set_display_model, set_display_model_path, set_display_quaternion, set_display_rotation_period,
    set_display_scale, start_renderer,
};
