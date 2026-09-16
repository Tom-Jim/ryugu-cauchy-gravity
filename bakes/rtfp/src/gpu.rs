//! Every parallel loop in this crate is a WGSL compute dispatch, and this module
//! is the whole of it:
//!
//! | shader                     | parallel over      | produces                       |
//! |----------------------------|--------------------|--------------------------------|
//! | `shaders/analytic.wgsl`    | (point, face)      | uniform-density tensor `W(x)`  |
//! | `shaders/rays.wgsl`        | (point, direction) | visible intervals of `x − R u`  |
//! | `shaders/remainder.wgsl`   | point              | deviation quadrature `G Σω T Σw R_k` |
//!
//! The Rust side only loads the mesh, builds the BVH (`bvh.rs`) and does scalar
//! bookkeeping. WebGPU has no ray-tracing stage, so the traversal itself is
//! WGSL — there is no CPU parallel path and no CPU fallback.

use crate::analytic::Face;
use crate::bvh::Bvh;
use crate::density::KernelSi;
use crate::mesh::Mesh;
use crate::quadrature::Dirs;
use crate::tensor::Sym6;
use crate::G;
use std::future::Future;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

/// Interval slots per (point, direction); mirrors `MAX_INTERVALS` in `rays.wgsl`.
pub const MAX_INTERVALS: usize = 8;

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
    remainder_pipeline: wgpu::ComputePipeline,
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

        let rays_module = shader(&device, "rays", include_str!("../shaders/rays.wgsl"));
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
            "analytic",
            include_str!("../shaders/analytic.wgsl"),
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
        let analytic_pipeline = pipeline(&device, &analytic_layout, &analytic_module, "analytic");

        let remainder_module = shader(
            &device,
            "remainder",
            include_str!("../shaders/remainder.wgsl"),
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

        println!("GPU: {} ({:?})", info.name, info.backend);
        Ok(Self {
            device,
            queue,
            adapter_name: info.name,
            inside_pipeline,
            rays_pipeline,
            analytic_pipeline,
            remainder_pipeline,
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
    remainder_pipeline: wgpu::ComputePipeline,
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

impl Scene {
    /// Upload the mesh, the BVH and the observation points once.
    // Every argument is a distinct immutable input buffer of the upload, and
    // bundling them into a builder would only move the list one level up.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: Device,
        mesh: &Mesh,
        bvh: &Bvh,
        faces: &[Face],
        points: &[[f64; 3]],
        dirs: &Dirs,
        block_capacity: usize,
        max_kernels: usize,
    ) -> Self {
        let Device {
            device,
            queue,
            adapter_name: _,
            inside_pipeline,
            rays_pipeline,
            analytic_pipeline,
            remainder_pipeline,
            rays_layout,
            analytic_layout,
            remainder_layout,
        } = device;
        let n_points = points.len();
        let n_dirs = dirs.len();
        let n_tris = mesh.face_count();

        let (positions, indices, nodes, links) = mesh_buffers(&device, &queue, mesh, bvh);
        let face_buf = upload(
            &device,
            &queue,
            "faces",
            &faces_to_f32(faces),
            wgpu::BufferUsages::STORAGE,
        );
        let dirs_buf = upload(
            &device,
            &queue,
            "dirs",
            &dirs_to_f32(dirs),
            wgpu::BufferUsages::STORAGE,
        );
        let points_all = upload(
            &device,
            &queue,
            "points_all",
            &points_to_f32(points),
            wgpu::BufferUsages::STORAGE,
        );
        let points_block = storage_buffer(&device, "points_block", block_capacity * 16);
        let kernels = storage_buffer(&device, "kernels", max_kernels.max(1) * 8 * 4);
        let counts = storage_buffer(&device, "counts", block_capacity * n_dirs * 4);
        let ivals = storage_buffer(
            &device,
            "ivals",
            block_capacity * n_dirs * MAX_INTERVALS * 8,
        );
        // The final word is an atomic overflow counter shared by both ray entry
        // points. Keeping it in this buffer avoids adding another storage binding.
        let inside = storage_buffer(&device, "inside_and_overflow", (block_capacity + 1) * 4);
        let w_out = storage_buffer(&device, "w_out", n_points * 6 * 4);
        let rem_out = storage_buffer(&device, "rem_out", block_capacity * 6 * 4);
        let rays_globals = uniform_buffer(&device, "rays_globals", 32);
        let analytic_globals = uniform_buffer(&device, "analytic_globals", 16);
        let remainder_globals = uniform_buffer(&device, "remainder_globals", 16);

        fn entry<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
            wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }
        }
        let rays_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rays"),
            layout: &rays_layout,
            entries: &[
                entry(0, &rays_globals),
                entry(1, &positions),
                entry(2, &indices),
                entry(3, &nodes),
                entry(4, &links),
                entry(5, &points_block),
                entry(6, &dirs_buf),
                entry(7, &counts),
                entry(8, &ivals),
                entry(9, &inside),
            ],
        });
        let analytic_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("analytic"),
            layout: &analytic_layout,
            entries: &[
                entry(0, &analytic_globals),
                entry(1, &face_buf),
                entry(2, &points_all),
                entry(3, &w_out),
            ],
        });
        let remainder_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("remainder"),
            layout: &remainder_layout,
            entries: &[
                entry(0, &remainder_globals),
                entry(1, &ivals),
                entry(2, &counts),
                entry(3, &points_block),
                entry(4, &kernels),
                entry(5, &dirs_buf),
                entry(6, &rem_out),
            ],
        });

        // One readback buffer, sized for the largest result (six tensor values
        // plus one inside flag per point, then the overflow counter).
        let readback_len = (n_points.max(block_capacity) * 7 * 4 + 4).max(16);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: readback_len as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            n_points,
            n_dirs,
            n_tris,
            n_faces: faces.len(),
            block_capacity,
            inside_pipeline,
            rays_pipeline,
            analytic_pipeline,
            remainder_pipeline,
            rays_globals,
            analytic_globals,
            remainder_globals,
            points_block,
            kernels,
            inside,
            w_out,
            rem_out,
            rays_bind,
            analytic_bind,
            remainder_bind,
            readback,
            readback_len,
        }
    }

    /// `W(x)` (unit density, `G` included) at every observation point.
    pub fn analytic_tensors(&self) -> Result<Vec<Sym6>, String> {
        self.analytic_tensors_impl(self.n_faces)
    }

    fn analytic_tensors_impl(&self, n_faces: usize) -> Result<Vec<Sym6>, String> {
        let (grid_x, grid_y) = grid_2d(self.n_points as u32);
        let mut globals = [0u8; 16];
        globals[0..4].copy_from_slice(&(self.n_points as u32).to_le_bytes());
        globals[4..8].copy_from_slice(&(n_faces as u32).to_le_bytes());
        globals[8..12].copy_from_slice(&(G as f32).to_le_bytes());
        globals[12..16].copy_from_slice(&grid_x.to_le_bytes());
        self.queue.write_buffer(&self.analytic_globals, 0, &globals);

        let mut enc = self
            .device
            .create_command_encoder(&empty_encoder("analytic"));
        {
            let mut pass = enc.begin_compute_pass(&compute_pass("analytic"));
            pass.set_pipeline(&self.analytic_pipeline);
            pass.set_bind_group(0, &self.analytic_bind, &[]);
            pass.dispatch_workgroups(grid_x, grid_y, 1);
        }
        enc.copy_buffer_to_buffer(
            &self.w_out,
            0,
            &self.readback,
            0,
            (self.n_points * 24) as u64,
        );
        self.queue.submit(Some(enc.finish()));
        let data = self.read_f32(self.n_points * 6)?;
        Ok(data
            .as_chunks::<6>()
            .0
            .iter()
            .map(|c| {
                [
                    c[0] as f64,
                    c[1] as f64,
                    c[2] as f64,
                    c[3] as f64,
                    c[4] as f64,
                    c[5] as f64,
                ]
            })
            .collect())
    }

    /// `G Σ_q ω_q T(u_q) Σ_k w_k R_k(u_q)` for one block of points.
    ///
    /// Runs the whole GPU chain: inside probe → ray intervals → remainder.
    pub fn block_remainder(
        &self,
        points: &[[f64; 3]],
        dirs: &Dirs,
        kernels: &[KernelSi],
        t_max: f32,
        t_min: f32,
    ) -> Result<Vec<Sym6>, String> {
        let n = points.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n > self.block_capacity {
            return Err(format!(
                "block of {n} exceeds capacity {}",
                self.block_capacity
            ));
        }
        if kernels.is_empty() {
            return Ok(vec![[0.0; 6]; n]);
        }
        let pts = points_to_f32(points);
        self.queue
            .write_buffer(&self.points_block, 0, as_bytes(&pts));
        let kbuf = kernels_to_f32(kernels);
        self.queue.write_buffer(&self.kernels, 0, as_bytes(&kbuf));
        let zero = 0u32.to_le_bytes();
        self.queue
            .write_buffer(&self.inside, (self.block_capacity * 4) as u64, &zero);

        // struct Globals { n_points, n_dirs, n_tris, grid_x, t_min, t_max, _pad, _pad }
        let ray_groups = ((n * self.n_dirs) as u32).div_ceil(64);
        let (rays_grid_x, rays_grid_y) = grid_2d(ray_groups);
        let mut rays_globals = [0u8; 32];
        rays_globals[0..4].copy_from_slice(&(n as u32).to_le_bytes());
        rays_globals[4..8].copy_from_slice(&(dirs.len() as u32).to_le_bytes());
        rays_globals[8..12].copy_from_slice(&(self.n_tris as u32).to_le_bytes());
        rays_globals[12..16].copy_from_slice(&rays_grid_x.to_le_bytes());
        rays_globals[16..20].copy_from_slice(&t_min.to_le_bytes());
        rays_globals[20..24].copy_from_slice(&t_max.to_le_bytes());
        self.queue
            .write_buffer(&self.rays_globals, 0, &rays_globals);

        let mut rem_globals = [0u8; 16];
        rem_globals[0..4].copy_from_slice(&(n as u32).to_le_bytes());
        rem_globals[4..8].copy_from_slice(&(dirs.len() as u32).to_le_bytes());
        rem_globals[8..12].copy_from_slice(&(kernels.len() as u32).to_le_bytes());
        rem_globals[12..16].copy_from_slice(&(G as f32).to_le_bytes());
        self.queue
            .write_buffer(&self.remainder_globals, 0, &rem_globals);

        let groups = (n as u32).div_ceil(64);
        let mut enc = self.device.create_command_encoder(&empty_encoder("rays"));
        {
            let mut pass = enc.begin_compute_pass(&compute_pass("inside_probe"));
            pass.set_pipeline(&self.inside_pipeline);
            pass.set_bind_group(0, &self.rays_bind, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        {
            let mut pass = enc.begin_compute_pass(&compute_pass("rays"));
            pass.set_pipeline(&self.rays_pipeline);
            pass.set_bind_group(0, &self.rays_bind, &[]);
            pass.dispatch_workgroups(rays_grid_x, rays_grid_y, 1);
        }
        {
            let mut pass = enc.begin_compute_pass(&compute_pass("remainder"));
            pass.set_pipeline(&self.remainder_pipeline);
            pass.set_bind_group(0, &self.remainder_bind, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.rem_out, 0, &self.readback, 0, (n * 24) as u64);
        enc.copy_buffer_to_buffer(
            &self.inside,
            0,
            &self.readback,
            (n * 24) as u64,
            (n * 4) as u64,
        );
        enc.copy_buffer_to_buffer(
            &self.inside,
            (self.block_capacity * 4) as u64,
            &self.readback,
            (n * 28) as u64,
            4,
        );
        self.queue.submit(Some(enc.finish()));
        let data = self.read_f32(n * 7 + 1)?;
        if data[n * 6..n * 7].iter().any(|inside| *inside != 0.0) {
            return Err(
                "observation point lies inside the mesh; the contact term is not enabled".into(),
            );
        }
        if data[n * 7] != 0.0 {
            return Err(format!(
                "ray traversal exceeded the {MAX_INTERVALS}-interval / {}-hit capacity \
                 ({} dropped intersections/intervals)",
                crate::geom::MAX_HITS,
                data[n * 7] as u32
            ));
        }
        Ok(data[..n * 6]
            .as_chunks::<6>()
            .0
            .iter()
            .map(|c| {
                [
                    c[0] as f64,
                    c[1] as f64,
                    c[2] as f64,
                    c[3] as f64,
                    c[4] as f64,
                    c[5] as f64,
                ]
            })
            .collect())
    }

    fn read_f32(&self, count: usize) -> Result<Vec<f32>, String> {
        let bytes = count * 4;
        if bytes > self.readback_len {
            return Err("readback buffer too small".into());
        }
        let (tx, rx) = std::sync::mpsc::channel();
        self.readback
            .slice(..bytes as u64)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        loop {
            self.device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .map_err(|e| format!("device poll: {e}"))?;
            match rx.try_recv() {
                Ok(Ok(())) => break,
                Ok(Err(e)) => return Err(format!("readback failed: {e}")),
                Err(std::sync::mpsc::TryRecvError::Empty) => continue,
                Err(e) => return Err(format!("readback channel: {e}")),
            }
        }
        let mapped = self
            .readback
            .slice(..bytes as u64)
            .get_mapped_range()
            .map_err(|e| format!("readback map range: {e}"))?;
        let mut out = Vec::with_capacity(count);
        for chunk in mapped.as_chunks::<4>().0 {
            out.push(f32::from_le_bytes(*chunk));
        }
        drop(mapped);
        self.readback.unmap();
        Ok(out)
    }
}

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

/// `struct Kernel { c: vec3<f32>, sigma: f32, w: f32, _pad, _pad, _pad }`.
fn kernels_to_f32(kernels: &[KernelSi]) -> Vec<f32> {
    let mut out = Vec::with_capacity(kernels.len() * 8);
    for k in kernels {
        out.extend_from_slice(&[
            k.c[0] as f32,
            k.c[1] as f32,
            k.c[2] as f32,
            k.sigma as f32,
            k.w as f32,
            0.0,
            0.0,
            0.0,
        ]);
    }
    out
}

fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
