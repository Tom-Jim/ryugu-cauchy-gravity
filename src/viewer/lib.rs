//! Ryugu WebGPU viewer.
//!
//! Every algorithm (Werner, Mascon, RT-FP) bakes to disk through its own
//! server-spawned solver and hands the viewer an RHGF v5 face record; this crate
//! only paints it. There is no in-browser solver left — the RT-FP WGSL compute
//! shaders live in `bakes/rtfp/shaders/` and run in the bake process.

mod gradient;

use bevy::asset::AssetMetaCheck;
use bevy::camera::primitives::Aabb;
use bevy::log::{Level, LogPlugin};
use bevy::mesh::VertexAttributeValues;
use bevy::prelude::*;
use bevy::render::settings::{Backends, RenderCreation, WgpuSettings, WgpuSettingsPriority};
use bevy::render::{RenderPlugin, render_resource::WgpuLimits};
use bevy::window::PresentMode;
use bevy::winit::{UpdateMode, WinitSettings};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use gradient::{
    BakePaint, DisplayWindow, colormap_scalar, display_window_for, explode_mesh_for_flat_faces,
    init_gray_colors, paint_face_on_colors, parse_bake, push_bake_bytes, take_pending_bake,
};
use wasm_bindgen::prelude::wasm_bindgen;

const PERIOD_S: f64 = 7.63 * 3600.0;
const SPIN_AXIS: Vec3 = Vec3::new(-0.043, -0.914, 0.405);
const TIME_SCALE: f64 = 1000.0;
const TARGET_SIZE: f32 = 900.0;
/// Faces colored per frame while catching up to bake progress.
const PAINT_PER_FRAME: usize = 4000;

#[derive(Component)]
struct Ryugu;

#[derive(Component)]
struct Sized;

#[derive(Component)]
struct PaintTarget;

#[derive(Resource, Default)]
struct Clock(f64);

#[wasm_bindgen]
pub fn run_with_bake(bytes: &[u8]) {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();
    push_bake_bytes(bytes.to_vec());
    run_app();
}

#[wasm_bindgen]
pub fn push_bake_update(bytes: &[u8]) {
    push_bake_bytes(bytes.to_vec());
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
            unfocused_mode: UpdateMode::Continuous,
            ..default()
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
                        priority: WgpuSettingsPriority::Functionality,
                        limits: WgpuLimits::default(),
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
    mesh_q: Query<(Entity, &Mesh3d), Without<PaintTarget>>,
) {
    // Same path as Werner: take the largest body mesh, flat gray + vertex colors.
    let mut best: Option<(Entity, Handle<Mesh>, usize)> = None;
    for (entity, mesh3d) in &mesh_q {
        let Some(src) = meshes.get(&mesh3d.0) else {
            continue;
        };
        let n = src.count_vertices();
        if n == 0 {
            continue;
        }
        if best.as_ref().is_none_or(|(_, _, c)| n > *c) {
            best = Some((entity, mesh3d.0.clone(), n));
        }
    }
    let Some((entity, src_handle, _)) = best else {
        return;
    };
    for (other, _) in &mesh_q {
        if other != entity {
            commands.entity(other).insert(Visibility::Hidden);
        }
    }
    let Some(src) = meshes.get(&src_handle) else {
        return;
    };
    // Explode so each face keeps a flat color (indexed GLB otherwise bleeds
    // neighbors). The pre-explode index buffer is no longer needed.
    let Some((mut mesh, indices)) = explode_mesh_for_flat_faces(src) else {
        return;
    };
    mesh.remove_attribute(Mesh::ATTRIBUTE_UV_0);
    if !init_gray_colors(&mut mesh) {
        return;
    }
    paint.face_indices = indices;
    let handle = meshes.add(mesh);
    let mat = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        base_color_texture: None,
        normal_map_texture: None,
        metallic_roughness_texture: None,
        occlusion_texture: None,
        emissive_texture: None,
        // Unlit: lighting was exaggerating flat-face "色块" on noisy mascon fields.
        unlit: true,
        perceptual_roughness: 1.0,
        metallic: 0.0,
        ..default()
    });
    commands
        .entity(entity)
        .insert((Mesh3d(handle), MeshMaterial3d(mat), Visibility::Visible, PaintTarget));
}

fn ingest_bake_updates(mut paint: ResMut<BakePaint>) {
    let Some(bytes) = take_pending_bake() else {
        return;
    };
    let Ok(baked) = parse_bake(&bytes) else {
        return;
    };
    if paint.painted.len() != baked.face_scalar.len() {
        paint.painted = vec![false; baked.face_scalar.len()];
        paint.scalars = vec![f32::NAN; baked.face_scalar.len()];
        paint.queue.clear();
        paint.reset_to_gray = true;
        paint.last_finite = 0;
    }

    let finite_file = baked
        .face_scalar
        .iter()
        .filter(|s| s.is_finite())
        .count();
    let finite_local = paint.scalars.iter().filter(|s| s.is_finite()).count();

    // Unchanged complete (or unchanged progressive) snapshot — do not re-sort / re-queue.
    if finite_file == paint.last_finite
        && finite_file == finite_local
        && paint.queue.is_empty()
        && !paint.reset_to_gray
    {
        return;
    }

    // Bake restart / stub: file went backwards — wipe previous full coloring.
    if finite_file < finite_local {
        paint.painted.fill(false);
        paint.scalars = vec![f32::NAN; baked.face_scalar.len()];
        paint.queue.clear();
        paint.window = DisplayWindow::default();
        paint.reset_to_gray = true;
        paint.last_finite = 0;
    }

    let mut newly = Vec::new();
    for (f, &s) in baked.face_scalar.iter().enumerate() {
        if s.is_finite() {
            let was = paint.scalars.get(f).copied().unwrap_or(f32::NAN);
            if !paint.painted[f] && !was.is_finite() {
                newly.push(f as u32);
            }
            paint.scalars[f] = s;
        } else if paint.scalars.get(f).is_some_and(|v| v.is_finite()) {
            paint.scalars[f] = f32::NAN;
            paint.painted[f] = false;
            paint.reset_to_gray = true;
        }
    }

    // Only recompute percentile stretch when the finite set grew (expensive sort).
    if finite_file != paint.last_finite {
        // Same window rule as the RT-FP path, derived from the raw face scalars:
        // equal values therefore map to equal colours in every viewer.
        paint.window = display_window_for(&paint.scalars);
        // Full recolor when the display mapping changes.
        let n_faces = paint.scalars.len() as u32;
        paint.painted.fill(false);
        paint.queue.clear();
        paint.queue.extend(0..n_faces);
        paint.reset_to_gray = true;
    }
    paint.last_finite = finite_file;

    if !newly.is_empty() {
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
    let range_ok = paint.window.is_valid();
    // Werner complete (or large backlog): paint the whole queue this frame — "秒渲染".
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

        if paint.reset_to_gray {
            let gray = [0.58, 0.58, 0.62, 1.0];
            for c in colors.iter_mut() {
                *c = gray;
            }
        }

        if paint.queue.is_empty() || paint.face_indices.is_empty() || !range_ok {
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
            if paint_face_on_colors(colors, &paint.face_indices, fi, rgba) {
                paint.painted[fi] = true;
                n += 1;
            }
        }
    }
    paint.reset_to_gray = false;
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
        let s = TARGET_SIZE / extent;
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
