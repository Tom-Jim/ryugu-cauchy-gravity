//! Every parallel loop in this crate is a WGSL compute dispatch, and this module
//! is the whole of it:
//!
//! | shader                     | parallel over      | produces                       |
//! |----------------------------|--------------------|--------------------------------|
//! | `src/wgsl/compute/werner.wgsl` | (point, face) | uniform-density tensor `W(x)` |
//! | `src/wgsl/compute/rays.wgsl` | (point, direction) | visible intervals of `x − R u` |
//! | `src/wgsl/compute/remainder.wgsl` | point | deviation quadrature `G Σω T Σw R_k` |
//! | `src/wgsl/compute/carlson_alpha.wgsl` | point | general-α radial finite part |
//!
//! The Rust side only loads the mesh, builds the BVH (`bvh.rs`) and does scalar
//! bookkeeping. WebGPU has no ray-tracing stage, so the traversal itself is
//! WGSL — there is no CPU parallel path and no CPU fallback.

use crate::G;
use crate::analytic::Face;
use crate::bvh::Bvh;
use crate::density::KernelSi;
use crate::mesh::Mesh;
use crate::quadrature::Dirs;
use crate::tensor::Sym6;
use std::future::Future;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

/// Interval slots per (point, direction); mirrors `MAX_INTERVALS` in `rays.wgsl`.
pub const MAX_INTERVALS: usize = 16;

struct ThreadWaker(std::thread::Thread);

impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

/// Minimal executor: `wgpu`'s adapter/device futures resolve on internal threads.
fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut fut = std::pin::pin!(future);
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => std::thread::park_timeout(std::time::Duration::from_millis(5)),
        }
    }
}

pub struct Device {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
    inside_pipeline: wgpu::ComputePipeline,
    rays_pipeline: wgpu::ComputePipeline,
    analytic_pipeline: wgpu::ComputePipeline,
    carlson_alpha_near_pipeline: wgpu::ComputePipeline,
    carlson_surface_pipeline: wgpu::ComputePipeline,
    remainder_pipeline: wgpu::ComputePipeline,
    carlson_alpha_pipeline: wgpu::ComputePipeline,
    rays_layout: wgpu::BindGroupLayout,
    analytic_layout: wgpu::BindGroupLayout,
    remainder_layout: wgpu::BindGroupLayout,
}

impl Device {
    pub fn new() -> Result<Self, String> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            #[cfg(not(target_arch = "wasm32"))]
            apply_limit_buckets: false,
        }))
        .map_err(|e| format!("no GPU adapter: {e}"))?;
        let info = adapter.get_info();

        // The ray traversal binds ten storage buffers in one stage, which is past
        // the WebGPU baseline of eight, so take the adapter's own limits.
        let adapter_limits = adapter.limits();
        let mut limits = wgpu::Limits::defaults().using_resolution(adapter_limits.clone());
        limits.max_storage_buffers_per_shader_stage =
            adapter_limits.max_storage_buffers_per_shader_stage;
        limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size;
        limits.max_buffer_size = adapter_limits.max_buffer_size;
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("rtfp-bake"),
            required_features: wgpu::Features::empty(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| format!("request_device: {e}"))?;

        let rays_module = shader(
            &device,
            "rays",
            include_str!("../../../wgsl/compute/rays.wgsl"),
        );
        let rays_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rays"),
            entries: &[
                uniform(0),
                storage(1, false),
                storage(2, false),
                storage(3, false),
                storage(4, false),
                storage(5, false),
                storage(6, false),
                storage(7, true),
                storage(8, true),
                storage(9, true),
            ],
        });
        let inside_pipeline = pipeline(&device, &rays_layout, &rays_module, "inside_probe");
        let rays_pipeline = pipeline(&device, &rays_layout, &rays_module, "rays");

        let analytic_module = shader(
            &device,
            "rtfp_near",
            include_str!("../../../wgsl/compute/rtfp_near.wgsl"),
        );
        let analytic_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("analytic"),
            entries: &[
                uniform(0),
                storage(1, false),
                storage(2, false),
                storage(3, true),
            ],
        });
        let analytic_pipeline = pipeline(&device, &analytic_layout, &analytic_module, "rtfp_near");
        let carlson_alpha_near_module = shader(
            &device,
            "carlson_alpha_near",
            include_str!("../../../wgsl/compute/carlson_alpha_near.wgsl"),
        );
        let carlson_alpha_near_pipeline = pipeline(
            &device,
            &analytic_layout,
            &carlson_alpha_near_module,
            "carlson_alpha_near",
        );
        let carlson_surface_module = shader(
            &device,
            "carlson_surface",
            include_str!("../../../wgsl/compute/carlson_surface.wgsl"),
        );
        let carlson_surface_pipeline = pipeline(
            &device,
            &analytic_layout,
            &carlson_surface_module,
            "carlson_surface",
        );

        let remainder_module = shader(
            &device,
            "remainder",
            include_str!("../../../wgsl/compute/remainder.wgsl"),
        );
        let remainder_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("remainder"),
            entries: &[
                uniform(0),
                storage(1, false),
                storage(2, false),
                storage(3, false),
                storage(4, false),
                storage(5, false),
                storage(6, true),
            ],
        });
        let remainder_pipeline =
            pipeline(&device, &remainder_layout, &remainder_module, "remainder");
        let carlson_alpha_module = shader(
            &device,
            "carlson_alpha",
            include_str!("../../../wgsl/compute/carlson_alpha.wgsl"),
        );
        let carlson_alpha_pipeline = pipeline(
            &device,
            &remainder_layout,
            &carlson_alpha_module,
            "carlson_alpha",
        );

        println!("GPU: {} ({:?})", info.name, info.backend);
        Ok(Self {
            device,
            queue,
            adapter_name: info.name,
            inside_pipeline,
            rays_pipeline,
            analytic_pipeline,
            carlson_alpha_near_pipeline,
            carlson_surface_pipeline,
            remainder_pipeline,
            carlson_alpha_pipeline,
            rays_layout,
            analytic_layout,
            remainder_layout,
        })
    }

    pub fn name(&self) -> &str {
        &self.adapter_name
    }
}

fn shader(device: &wgpu::Device, label: &str, src: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    })
}

fn pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    module: &wgpu::ShaderModule,
    entry: &str,
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(entry),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(&pipeline_layout),
        module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage(binding: u32, read_write: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: if read_write {
                wgpu::BufferBindingType::Storage { read_only: false }
            } else {
                wgpu::BufferBindingType::Storage { read_only: true }
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A GPU-resident bake: mesh, BVH, quadrature nodes and the per-bake scratch.
pub struct Scene {
    device: wgpu::Device,
    queue: wgpu::Queue,
    n_points: usize,
    n_dirs: usize,
    n_tris: usize,
    n_faces: usize,
    block_capacity: usize,
    // pipelines / layouts moved out of `Device` (the device itself is shared)
    inside_pipeline: wgpu::ComputePipeline,
    rays_pipeline: wgpu::ComputePipeline,
    analytic_pipeline: wgpu::ComputePipeline,
    carlson_alpha_near_pipeline: wgpu::ComputePipeline,
    carlson_surface_pipeline: wgpu::ComputePipeline,
    remainder_pipeline: wgpu::ComputePipeline,
    carlson_alpha_pipeline: wgpu::ComputePipeline,
    // scene buffers
    rays_globals: wgpu::Buffer,
    analytic_globals: wgpu::Buffer,
    remainder_globals: wgpu::Buffer,
    points_block: wgpu::Buffer,
    kernels: wgpu::Buffer,
    inside: wgpu::Buffer,
    w_out: wgpu::Buffer,
    rem_out: wgpu::Buffer,
    rays_bind: wgpu::BindGroup,
    analytic_bind: wgpu::BindGroup,
    remainder_bind: wgpu::BindGroup,
    readback: wgpu::Buffer,
    readback_len: usize,
}

include!("gpu/scene_new.rs");
include!("gpu/scene_surface.rs");
include!("gpu/scene_remainder.rs");
include!("gpu/readback.rs");

fn empty_encoder(label: &str) -> wgpu::CommandEncoderDescriptor<'_> {
    wgpu::CommandEncoderDescriptor { label: Some(label) }
}

/// Splits `groups` workgroups across a two-dimensional dispatch grid.
///
/// A dispatch dimension is capped at 65 535, which the full mesh blows through
/// (99 846 points for the analytic pass, 449 308 workgroups for the ray pass).
/// `grid_x` comes back so the shader can rebuild a flat index as
/// `gid.x + gid.y * grid_x * workgroup_size`.
fn grid_2d(groups: u32) -> (u32, u32) {
    const MAX_X: u32 = 16_384;
    let grid_x = groups.clamp(1, MAX_X);
    (grid_x, groups.max(1).div_ceil(grid_x))
}

fn compute_pass(label: &str) -> wgpu::ComputePassDescriptor<'_> {
    wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    }
}

fn storage_buffer(device: &wgpu::Device, label: &str, size: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn uniform_buffer(device: &wgpu::Device, label: &str, size: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn upload(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    data: &[f32],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let bytes = as_bytes(data);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len().max(4) as u64,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytes);
    buffer
}

/// Mesh + BVH as flat GPU buffers. Triangles are written **in BVH leaf order**,
/// so a leaf's range indexes the triangle buffer directly.
fn mesh_buffers(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mesh: &Mesh,
    bvh: &Bvh,
) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
    let nv = mesh.vertex_count();
    let mut positions = Vec::with_capacity(nv * 4);
    for i in 0..nv {
        let p = mesh.vertex(i);
        positions.extend_from_slice(&[p[0] as f32, p[1] as f32, p[2] as f32, 0.0]);
    }
    let mut indices = Vec::with_capacity(bvh.order.len() * 3);
    for &t in &bvh.order {
        let f = mesh.face(t as usize);
        indices.extend_from_slice(&f);
    }
    let mut bounds = Vec::with_capacity(bvh.bounds.len() * 4);
    for row in &bvh.bounds {
        bounds.extend_from_slice(row);
    }
    let mut links = Vec::with_capacity(bvh.meta.len() * 4);
    for row in &bvh.meta {
        links.extend_from_slice(row);
    }

    let positions = upload(
        device,
        queue,
        "positions",
        &positions,
        wgpu::BufferUsages::STORAGE,
    );
    let nodes = upload(device, queue, "nodes", &bounds, wgpu::BufferUsages::STORAGE);
    let indices_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("indices"),
        size: as_bytes(&indices).len().max(4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&indices_buf, 0, as_bytes(&indices));
    let links_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("links"),
        size: as_bytes(&links).len().max(4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&links_buf, 0, as_bytes(&links));
    (positions, indices_buf, nodes, links_buf)
}

fn points_to_f32(points: &[[f64; 3]]) -> Vec<f32> {
    let mut out = Vec::with_capacity(points.len() * 4);
    for p in points {
        out.extend_from_slice(&[p[0] as f32, p[1] as f32, p[2] as f32, 0.0]);
    }
    out
}

fn dirs_to_f32(dirs: &Dirs) -> Vec<f32> {
    let mut out = Vec::with_capacity(dirs.len() * 4);
    for (u, w) in dirs {
        out.extend_from_slice(&[u[0] as f32, u[1] as f32, u[2] as f32, *w as f32]);
    }
    out
}

/// Four `vec4<f32>` per face: three corners, then `(unit normal, density jump)`.
fn faces_to_f32(faces: &[Face]) -> Vec<f32> {
    let mut out = Vec::with_capacity(faces.len() * 16);
    for f in faces {
        for c in f.corners {
            out.extend_from_slice(&[c[0] as f32, c[1] as f32, c[2] as f32, 0.0]);
        }
        out.extend_from_slice(&[f.n[0] as f32, f.n[1] as f32, f.n[2] as f32, f.weight as f32]);
    }
    out
}

/// `struct Kernel { c: vec3<f32>, sigma: f32, w: f32, alpha: f32, _pad, _pad }`.
fn kernels_to_f32(kernels: &[KernelSi]) -> Vec<f32> {
    let mut out = Vec::with_capacity(kernels.len() * 8);
    for k in kernels {
        out.extend_from_slice(&[
            k.c[0] as f32,
            k.c[1] as f32,
            k.c[2] as f32,
            k.sigma as f32,
            k.w as f32,
            k.alpha as f32,
            0.0,
            0.0,
        ]);
    }
    out
}

fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
