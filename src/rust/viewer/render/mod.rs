//! Bevy scene and face-colour renderer.
//!
//! Every algorithm (Werner, Mascon, RT-FP, Carlson and CarlsonAlpha) runs as
//! its own browser-side compute track: the kernels under `src/wgsl/` are
//! dispatched by the Rust compute coordinator and driven by the frontend
//! session controller. The static server only distributes files. This module
//! owns the display surface: it receives an RHGF v5 face
//! record per completed block and paints it onto the original model, restoring
//! the model's own material and vertex colours while a computation is still
//! running.

mod gradient;

use bevy::asset::{AssetLoadFailedEvent, AssetMetaCheck};
use bevy::camera::primitives::Aabb;
use bevy::log::{Level, LogPlugin};
use bevy::mesh::VertexAttributeValues;
use bevy::prelude::*;
use bevy::render::settings::{
    Backends, MemoryHints, RenderCreation, WgpuSettings, WgpuSettingsPriority,
};
use bevy::render::{RenderPlugin, render_resource::WgpuLimits};
use bevy::window::PresentMode;
use bevy::winit::{UpdateMode, WinitSettings};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use gradient::{
    BakePaint, DisplayWindow, colormap_scalar, display_window_for, ensure_base_colors,
    explode_mesh_for_flat_faces, paint_face_on_colors, push_bake_bytes,
    push_bake_bytes_preserving_window, request_paint_reset, take_pending_bake, triangle_count,
};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};
use wasm_bindgen::prelude::wasm_bindgen;

const PERIOD_S: f64 = 7.63 * 3600.0;
const SPIN_AXIS: Vec3 = Vec3::new(-0.043, -0.914, 0.405);
const TIME_SCALE: f64 = 1000.0;
const TARGET_SIZE: f32 = 900.0;
/// Faces colored per frame while catching up to bake progress.
const PAINT_PER_FRAME: usize = 4000;

static PENDING_MODEL_URL: OnceLock<Mutex<Option<String>>> = OnceLock::new();
static DISPLAY_SCALE_BITS: AtomicU32 = AtomicU32::new(1.0f32.to_bits());

#[wasm_bindgen]
pub fn set_display_scale(scale_meters_per_unit: f32) {
    if scale_meters_per_unit.is_finite() && scale_meters_per_unit > 0.0 {
        DISPLAY_SCALE_BITS.store(scale_meters_per_unit.to_bits(), Ordering::Relaxed);
    }
}

fn pending_model_url() -> &'static Mutex<Option<String>> {
    PENDING_MODEL_URL.get_or_init(|| Mutex::new(None))
}

/// Queue a GLB for the display scene. The existing scene is kept visible until
/// Bevy has accepted the new asset, so a bad upload cannot blank the viewer.
#[wasm_bindgen]
pub fn set_display_model(bytes: js_sys::Uint8Array) -> Result<(), wasm_bindgen::JsValue> {
    request_paint_reset();
    let parts = js_sys::Array::new();
    parts.push(&bytes);
    let blob = web_sys::Blob::new_with_u8_array_sequence(&parts)?;
    let url = web_sys::Url::create_object_url_with_blob(&blob)?;
    *pending_model_url()
        .lock()
        .map_err(|_| wasm_bindgen::JsValue::from_str("model queue unavailable"))? = Some(url);
    Ok(())
}

#[wasm_bindgen]
pub fn set_display_model_path(path: String) -> Result<(), wasm_bindgen::JsValue> {
    request_paint_reset();
    *pending_model_url()
        .lock()
        .map_err(|_| wasm_bindgen::JsValue::from_str("model queue unavailable"))? = Some(path);
    Ok(())
}

#[derive(Component)]
struct Ryugu;

#[derive(Component)]
struct Sized;

#[derive(Component)]
struct PaintTarget;

/// Every spawnable body mesh with the material the GLB gave it.
type BodyMeshQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static Mesh3d,
        Option<&'static MeshMaterial3d<StandardMaterial>>,
    ),
    Without<PaintTarget>,
>;

/// The largest body mesh, plus the triangle count used to choose it.
type BestBody = (
    Entity,
    Handle<Mesh>,
    usize,
    Option<Handle<StandardMaterial>>,
);

#[derive(Resource, Default)]
struct Clock(f64);

#[wasm_bindgen]
pub fn run_with_bake(bytes: &[u8]) {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();
    push_bake_bytes(bytes);
    run_app();
}

/// Start the display scene without manufacturing a fake bake record. The
/// original GLB material remains visible until a real result is submitted.
#[wasm_bindgen]
pub fn start_renderer() {
    run_app();
}

#[wasm_bindgen]
pub fn push_bake_update(bytes: &[u8]) {
    push_bake_bytes(bytes);
}

#[wasm_bindgen]
pub fn push_bake_update_preserving_window(bytes: &[u8]) {
    push_bake_bytes_preserving_window(bytes);
}

/// Reset only the display state. The next bake remains in the normal pending
/// slot and cannot overwrite this reset request.
#[wasm_bindgen]
pub fn reset_bake_paint() {
    request_paint_reset();
}

fn run_app() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.06, 0.07, 0.10)))
        .init_resource::<Clock>()
        .insert_resource(BakePaint {
            rng: 0xc0ffee_u64 ^ 0x9e3779b97f4a7c15,
            ..default()
        })
        .insert_resource(WinitSettings {
            focused_mode: UpdateMode::Continuous,
            // A hidden tab does not need a 60 Hz spin loop. Keep the body alive
            // at low frequency, then return to continuous updates on focus.
            unfocused_mode: UpdateMode::reactive_low_power(std::time::Duration::from_secs(1)),
        })
        .add_plugins(
            DefaultPlugins
                .build()
                .set(LogPlugin {
                    level: Level::ERROR,
                    filter: "error".into(),
                    ..default()
                })
                .set(AssetPlugin {
                    meta_check: AssetMetaCheck::Never,
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        #[cfg(target_arch = "wasm32")]
                        canvas: Some("#bevy".into()),
                        fit_canvas_to_parent: true,
                        prevent_default_event_handling: true,
                        present_mode: PresentMode::AutoVsync,
                        title: "Ryugu".into(),
                        ..default()
                    }),
                    ..default()
                })
                .set(RenderPlugin {
                    render_creation: RenderCreation::Automatic(Box::new(WgpuSettings {
                        backends: Some(Backends::BROWSER_WEBGPU),
                        priority: WgpuSettingsPriority::WebGPU,
                        limits: WgpuLimits::default(),
                        memory_hints: MemoryHints::MemoryUsage,
                        ..default()
                    })),
                    ..default()
                }),
        )
        .add_plugins(PanOrbitCameraPlugin)
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                normalize,
                ingest_model_selection,
                report_model_load_failures,
                prepare_paint_target,
                ingest_bake_updates,
                paint_faces_from_queue,
                tick,
                spin,
            )
                .chain(),
        )
        .run();
}

fn report_model_load_failures(mut failures: MessageReader<AssetLoadFailedEvent<Gltf>>) {
    for failure in failures.read() {
        error!(
            "GLB display load failed: path={} error={:?}",
            failure.path, failure.error
        );
    }
}

fn ingest_model_selection(
    mut commands: Commands,
    assets: Res<AssetServer>,
    roots: Query<Entity, With<Ryugu>>,
    mut paint: ResMut<BakePaint>,
) {
    let Ok(mut pending) = pending_model_url().try_lock() else {
        return;
    };
    let Some(url) = pending.take() else {
        return;
    };
    request_paint_reset();
    for root in &roots {
        // Visibility on the GLTF root is not sufficient: Bevy's scene
        // spawner has already created child mesh entities, and those can keep
        // rendering independently. Remove the complete old scene before the
        // replacement is spawned so gradients cannot paint two models at once.
        commands.entity(root).despawn_related::<Children>();
        commands.entity(root).despawn();
    }
    paint.face_count = 0;
    paint.painted.clear();
    paint.scalars.clear();
    paint.queue.clear();
    paint.reset_to_base = true;
    commands.spawn((
        // The explicit Scene label selects Bevy's GLTF loader. Do not append a
        // query suffix: blob URLs with a query are rejected by some browsers
        // before the GLTF asset reader sees them.
        WorldAssetRoot(assets.load(GltfAssetLabel::Scene(0).from_asset(url))),
        Transform::default(),
        Ryugu,
    ));
}

fn setup(mut commands: Commands, assets: Res<AssetServer>) {
    commands.insert_resource(GlobalAmbientLight {
        color: Color::WHITE,
        brightness: 80.0,
        ..default()
    });
    commands.spawn((
        DirectionalLight {
            illuminance: 25_000.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_xyz(1000.0, 2000.0, 1500.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            far: 100_000.0,
            near: 0.1,
            ..default()
        }),
        Transform::from_xyz(0.0, 800.0, 2500.0).looking_at(Vec3::ZERO, Vec3::Y),
        PanOrbitCamera::default(),
    ));
    commands.spawn((
        // Bevy's asset root *is* `assets/`, so the GLB sitting at
        // `assets/models/ryugu.glb` on disk is the asset path `models/ryugu.glb`.
        WorldAssetRoot(assets.load(GltfAssetLabel::Scene(0).from_asset("models/ryugu.glb"))),
        Transform::default(),
        Ryugu,
    ));
}

fn prepare_paint_target(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut paint: ResMut<BakePaint>,
    mesh_q: BodyMeshQuery,
) {
    // Take the largest body mesh, but retain its original material and texture.
    // Only the per-face vertex colour is replaced as bake records arrive.
    let mut best: Option<BestBody> = None;
    for (entity, mesh3d, material) in &mesh_q {
        let Some(src) = meshes.get(&mesh3d.0) else {
            continue;
        };
        let Some(n_faces) = triangle_count(src) else {
            continue;
        };
        if best
            .as_ref()
            .is_none_or(|(_, _, count, _)| n_faces > *count)
        {
            best = Some((
                entity,
                mesh3d.0.clone(),
                n_faces,
                material.map(|material| material.0.clone()),
            ));
        }
    }
    let Some((entity, src_handle, _, source_material)) = best else {
        return;
    };
    for (other, _, _) in &mesh_q {
        if other != entity {
            commands.entity(other).insert(Visibility::Hidden);
        }
    }
    let Some(src) = meshes.get(&src_handle) else {
        return;
    };
    // Explode so each face keeps a flat color (indexed GLB otherwise bleeds
    // neighbors). The pre-explode index buffer is no longer needed.
    let Some((mut mesh, face_count)) = explode_mesh_for_flat_faces(src) else {
        return;
    };
    if !ensure_base_colors(&mut mesh) {
        return;
    }
    let base_colors = match mesh.attribute(Mesh::ATTRIBUTE_COLOR) {
        Some(VertexAttributeValues::Float32x4(colors)) => colors.clone(),
        _ => return,
    };
    paint.face_count = face_count;
    paint.base_colors = base_colors;
    let handle = meshes.add(mesh);
    let mat = source_material
        .filter(|handle| materials.get(handle).is_some())
        .unwrap_or_else(|| {
            // Fallback only applies to malformed assets with no source material;
            // normal GLB paths keep their original colour, texture and lighting.
            materials.add(StandardMaterial::default())
        });
    commands.entity(entity).insert((
        Mesh3d(handle),
        MeshMaterial3d(mat),
        Visibility::Visible,
        PaintTarget,
    ));
}

fn ingest_bake_updates(mut paint: ResMut<BakePaint>) {
    let reset_epoch = gradient::paint_reset_epoch();
    if paint.reset_epoch != reset_epoch {
        paint.reset_epoch = reset_epoch;
        paint.scalars.clear();
        paint.painted.clear();
        paint.queue.clear();
        paint.window = DisplayWindow::default();
        paint.last_finite = 0;
        // Keep face_count/base_colors: an algorithm switch keeps the same
        // mesh, so the next bake can repaint it immediately. Model replacement
        // clears those two fields in ingest_model_selection below.
        paint.reset_to_base = true;
    }
    let Some(pending) = take_pending_bake() else {
        return;
    };
    // A callback from an aborted computation may have raced the reset. Its
    // epoch is stale and must never repaint the newly selected algorithm.
    if pending.reset_epoch != reset_epoch {
        return;
    }
    let Ok(face_scalar) = pending.baked else {
        return;
    };
    if paint.painted.len() != face_scalar.len() {
        paint.painted = vec![false; face_scalar.len()];
        paint.scalars = vec![f32::NAN; face_scalar.len()];
        paint.queue.clear();
        paint.reset_to_base = true;
        paint.last_finite = 0;
    }

    let finite_file = face_scalar.iter().filter(|s| s.is_finite()).count();
    let finite_local = paint.scalars.iter().filter(|s| s.is_finite()).count();
    let same_snapshot = paint.scalars.len() == face_scalar.len()
        && paint
            .scalars
            .iter()
            .zip(&face_scalar)
            .all(|(old, new)| old.to_bits() == new.to_bits());

    // Unchanged complete (or unchanged progressive) snapshot — do not re-sort / re-queue.
    if same_snapshot
        && finite_file == paint.last_finite
        && finite_file == finite_local
        && paint.queue.is_empty()
        && !paint.reset_to_base
    {
        return;
    }

    // Bake restart / stub: file went backwards — wipe previous full coloring.
    if finite_file < finite_local {
        paint.painted.fill(false);
        paint.scalars = vec![f32::NAN; face_scalar.len()];
        paint.queue.clear();
        paint.window = DisplayWindow::default();
        paint.reset_to_base = true;
        paint.last_finite = 0;
    }

    // A full recolor replaces the queue with every face below, so collecting a
    // second list of "new" indices first is wasted allocation on every algorithm
    // switch. Compute the decision before walking the snapshot.
    let full_recolor = !same_snapshot || finite_file != paint.last_finite;
    let mut newly = Vec::new();
    for (f, &s) in face_scalar.iter().enumerate() {
        if s.is_finite() {
            let was = paint.scalars.get(f).copied().unwrap_or(f32::NAN);
            if !full_recolor && !paint.painted[f] && !was.is_finite() {
                newly.push(f as u32);
            }
            paint.scalars[f] = s;
        } else if paint.scalars.get(f).is_some_and(|v| v.is_finite()) {
            paint.scalars[f] = f32::NAN;
            paint.painted[f] = false;
            paint.reset_to_base = true;
        }
    }

    // Recompute the percentile stretch whenever the record content changes.
    // Different completed records can have the same finite count while every
    // scalar differs, so finite count alone is not a record identity.
    if full_recolor {
        // Density switches preserve the previous scale so a roughly uniform
        // amplitude change remains visible instead of being normalized away.
        // Height and algorithm changes still derive a fresh robust window.
        if !pending.preserve_window || !paint.window.is_valid() {
            paint.window = display_window_for(&paint.scalars);
        }
        // Full recolor when the display mapping changes.
        let n_faces = paint.scalars.len() as u32;
        paint.painted.fill(false);
        paint.queue.clear();
        paint.queue.extend(0..n_faces);
        paint.reset_to_base = true;
    }
    paint.last_finite = finite_file;

    if !full_recolor && !newly.is_empty() {
        // Complete file / huge catch-up: skip shuffle so we can dump colors in one frame.
        if newly.len() > 8_000 {
            paint.queue.extend(newly);
        } else {
            paint.shuffle_push(newly);
        }
    }
}

fn paint_faces_from_queue(
    mut paint: ResMut<BakePaint>,
    mut meshes: ResMut<Assets<Mesh>>,
    targets: Query<&Mesh3d, With<PaintTarget>>,
) {
    // A record is indexed in the source GLB primitive's triangle order. Refuse
    // a partial prefix paint if the displayed primitive is not that exact mesh.
    if paint.face_count == 0 || paint.face_count != paint.scalars.len() {
        return;
    }
    let range_ok = paint.window.is_valid();
    // Bake complete (or a large backlog): paint the whole queue this frame.
    let budget = if paint.queue.len() > 8_000 {
        paint.queue.len()
    } else {
        PAINT_PER_FRAME
    };

    for mesh3d in &targets {
        let Some(mut mesh) = meshes.get_mut(&mesh3d.0) else {
            continue;
        };
        let Some(VertexAttributeValues::Float32x4(colors)) =
            mesh.attribute_mut(Mesh::ATTRIBUTE_COLOR)
        else {
            continue;
        };

        if paint.reset_to_base {
            for (i, c) in colors.iter_mut().enumerate() {
                *c = paint.base_colors.get(i).copied().unwrap_or([1.0; 4]);
            }
        }

        if paint.queue.is_empty() || paint.face_count == 0 || !range_ok {
            continue;
        }

        let mut n = 0usize;
        while n < budget {
            let Some(face) = paint.queue.pop_front() else {
                break;
            };
            let fi = face as usize;
            if fi >= paint.painted.len() || paint.painted[fi] {
                continue;
            }
            let s = paint.scalars.get(fi).copied().unwrap_or(f32::NAN);
            if !s.is_finite() {
                continue;
            }
            let Some(rgba) = colormap_scalar(&paint.window, s) else {
                continue;
            };
            if paint_face_on_colors(colors, fi, rgba) {
                paint.painted[fi] = true;
                n += 1;
            }
        }
    }
    paint.reset_to_base = false;
}

fn normalize(
    mut commands: Commands,
    targets: Query<Entity, (With<Ryugu>, Without<Sized>)>,
    children: Query<&Children>,
    aabbs: Query<&Aabb>,
) {
    for entity in &targets {
        let Some(extent) = max_extent(entity, &children, &aabbs) else {
            continue;
        };
        if extent <= 1e-5 {
            continue;
        }
        // Preserve the established visible framing, then apply the single
        // user-controlled GLB-to-Bevy scale coefficient on top of it.
        let coefficient = f32::from_bits(DISPLAY_SCALE_BITS.load(Ordering::Relaxed));
        let s = (TARGET_SIZE / extent) * coefficient;
        commands.entity(entity).insert(Sized);
        commands
            .entity(entity)
            .entry::<Transform>()
            .and_modify(move |mut t| t.scale = Vec3::splat(s));
    }
}

fn max_extent(entity: Entity, children: &Query<&Children>, aabbs: &Query<&Aabb>) -> Option<f32> {
    let mut max = 0.0_f32;
    let mut found = false;
    let mut stack = vec![entity];
    while let Some(curr) = stack.pop() {
        if let Ok(aabb) = aabbs.get(curr) {
            let size = aabb.half_extents * 2.0;
            max = max.max(size.x).max(size.y).max(size.z);
            found = true;
        }
        if let Ok(kids) = children.get(curr) {
            stack.extend(kids.iter());
        }
    }
    found.then_some(max)
}

fn tick(time: Res<Time>, mut clock: ResMut<Clock>) {
    clock.0 += time.delta_secs_f64() * TIME_SCALE;
}

fn spin(mut q: Query<&mut Transform, With<Ryugu>>, clock: Res<Clock>) {
    let r = Quat::from_axis_angle(
        SPIN_AXIS.normalize(),
        (std::f64::consts::TAU / PERIOD_S * clock.0) as f32,
    );
    for mut t in &mut q {
        t.rotation = r;
    }
}
