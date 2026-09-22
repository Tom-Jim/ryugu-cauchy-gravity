//! Browser-side solver coordinator.
//!
//! Each algorithm owns a separate input construction and a separate numerical
//! entry point. The numerical work stays in the checked-in WGSL kernels; this
//! module owns the WebGPU device, resources, checkpoints and progress events,
//! exactly as the JavaScript coordinator it replaces did. Nothing here is
//! JavaScript: the shaders are embedded at compile time and every dispatch,
//! buffer copy and readback is issued through `wgpu`.

use super::source::{RuntimeSource, quadrature_directions};
use std::borrow::Cow;
use std::collections::HashMap;
use std::ops::Range;
use std::rc::Rc;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{RequestCache, RequestInit};

const G: f64 = 6.674_30e-11;
const RECORD_HEADER: usize = 28;
const RECORD_MAGIC: u32 = 0x5248_4746;
const RECORD_VERSION: u32 = 5;
const CONSTANT_DENSITY: f64 = 1190.0;
const ANALYTIC_BLOCK_FACES: usize = 8192;
/// Faces per dispatch block. Each block is one upload/dispatch/readback round
/// trip, so a larger block directly cuts the sequential round-trip count. The
/// ray path is the constraint: its interval buffer is
/// `faces * directions * MAX_INTERVALS * 16` bytes, including two endpoint
/// face ids per interval. At 288 directions a 512-face block uses 36 MiB for
/// intervals; 1024 faces would use 72 MiB and exceed common 64 MiB WebGPU
/// storage-binding limits while worsening allocation spikes and UI latency.
const RAY_BLOCK_FACES: usize = 512;
const MASCON_BLOCK_FACES: usize = 4096;
/// Opening angle of the Mascon Barnes-Hut walk. A smaller value retains more
/// tree nodes near the observer and reduces voxel approximation error, at the
/// cost of more point-mass work. The native mirror in
/// `src/rust/compute/native/mascon.rs` deliberately uses the same value so
/// native and browser records remain comparable. The deployed browser
/// snapshot and its comparison caveats are recorded in
/// `docs/online-test-data.md`; this tuning constant is not itself an accuracy
/// guarantee.
const MASCON_THETA: f32 = 0.1;
const CHECKPOINT_BLOCKS: usize = 8;
const MAX_INTERVALS: usize = 16;
const T_MIN: f64 = 1e-4;
const T_MAX: f64 = 3000.0;
const SHADER_OBSERVERS: &str = include_str!("../../../wgsl/compute/observers.wgsl");
const SHADER_TENSOR_SCALAR: &str = include_str!("../../../wgsl/compute/tensor_scalar.wgsl");
const SHADER_TENSOR_COMPOSE: &str = include_str!("../../../wgsl/compute/tensor_compose.wgsl");
const SHADER_MASCON_TREE: &str = include_str!("../../../wgsl/compute/mascon_tree.wgsl");
const SHADER_MASCON_TENSOR: &str = include_str!("../../../wgsl/compute/mascon_tensor.wgsl");
const SHADER_WERNER: &str = include_str!("../../../wgsl/compute/werner.wgsl");
const SHADER_RTFP_NEAR: &str = include_str!("../../../wgsl/compute/rtfp_near.wgsl");
const SHADER_CARLSON_SURFACE: &str = include_str!("../../../wgsl/compute/carlson_surface.wgsl");
const SHADER_CARLSON_SYMMETRIC: &str = include_str!("../../../wgsl/compute/carlson_symmetric.wgsl");
const SHADER_RAYS: &str = include_str!("../../../wgsl/compute/rays.wgsl");
const SHADER_REMAINDER: &str = include_str!("../../../wgsl/compute/remainder.wgsl");
const SHADER_CARLSON_ALPHA: &str = include_str!("../../../wgsl/compute/carlson_alpha.wgsl");

const MODEL_PATH: &str = "./assets/models/ryugu.glb";
const ELLIPTIC_PATH: &str = "./assets/density/cauchy_elliptic.toml";

const ASSET_GEOMETRY: &str = "runtime:geometry";
const ASSET_MASCON_TREE: &str = "runtime:mascon-tree";
const ASSET_MASCON: &str = "runtime:mascon-points";
const ASSET_RTFP: &str = "runtime:rtfp";
const ASSET_CARLSON_ALPHA: &str = "runtime:carlson-alpha";

const FACE_MAGIC_WERNER: u32 = 0x3152_5752;

include!("interop.rs");
include!("record.rs");
include!("dispatch.rs");
include!("assets.rs");

struct ObserverResources {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    points: wgpu::Buffer,
    global_buffer: wgpu::Buffer,
}

struct MeshBuffers {
    layout: MeshLayout,
    positions: wgpu::Buffer,
    indices: wgpu::Buffer,
    bounds: wgpu::Buffer,
    links: wgpu::Buffer,
    directions: wgpu::Buffer,
    kernels: wgpu::Buffer,
    faces: wgpu::Buffer,
}

struct MasconBuffers {
    nodes: wgpu::Buffer,
    points: wgpu::Buffer,
    order: wgpu::Buffer,
}

struct GpuSolver {
    base_url: String,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: HashMap<String, wgpu::ComputePipeline>,
    source: Option<Rc<RuntimeSource>>,
    model_bytes: Option<Vec<u8>>,
    density_files: HashMap<String, Rc<String>>,
    assets: HashMap<String, Rc<Vec<u8>>>,
    face_buffers: HashMap<String, wgpu::Buffer>,
    mesh_buffers: HashMap<String, Rc<MeshBuffers>>,
    mascon_buffers: HashMap<String, Rc<MasconBuffers>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OutputMode {
    Scalar,
    Tensor,
}

include!("device.rs");
include!("surface.rs");
include!("ray.rs");
include!("algorithms/werner.rs");
include!("algorithms/mascon.rs");
include!("algorithms/rtfp.rs");
include!("algorithms/carlson_alpha.rs");

fn bytes_to_f32(bytes: &[u8], count: usize) -> Vec<f32> {
    (0..count)
        .map(|index| f32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap()))
        .collect()
}

include!("entry.rs");
