//! Ryugu WebGPU viewer: progressive random-order face coloring from bake.

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
    BakePaint, colormap_rgba, face_vertex_indices, init_gray_colors, paint_face_on_colors,
    parse_bake, push_bake_bytes, take_pending_bake,
};
use wasm_bindgen::prelude::wasm_bindgen;

const PERIOD_S: f64 = 7.63 * 3600.0;
const SPIN_AXIS: Vec3 = Vec3::new(-0.043, -0.914, 0.405);
const TIME_SCALE: f64 = 1000.0;
const TARGET_SIZE: f32 = 900.0;
/// Faces colored per frame while catching up to bake progress.
const PAINT_PER_FRAME: usize = 800;

#[derive(Component)]
struct Ryugu;

#[derive(Component)]
struct Sized;

#[derive(Component)]
struct PaintTarget;

#[derive(Resource, Default)]
struct Clock(f64);

/// Viewer mode: clean GLB display vs Werner constant-density gradient bake.
#[derive(Resource, Clone, Copy, PartialEq, Eq)]
enum ViewMode {
    Clean,
    Werner,
}

fn is_werner(mode: Res<ViewMode>) -> bool {
    *mode == ViewMode::Werner
}

#[wasm_bindgen]
pub fn run_clean() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();
    run_app(ViewMode::Clean);
}

#[wasm_bindgen]
pub fn run_with_bake(bytes: &[u8]) {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();
    push_bake_bytes(bytes.to_vec());
    run_app(ViewMode::Werner);
}

#[wasm_bindgen]
pub fn push_bake_update(bytes: &[u8]) {
    push_bake_bytes(bytes.to_vec());
}

fn run_app(mode: ViewMode) {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.02, 0.02, 0.05)))
        .insert_resource(mode)
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
                prepare_paint_target.run_if(is_werner),
                ingest_bake_updates.run_if(is_werner),
                paint_faces_from_queue.run_if(is_werner),
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
    for (entity, mesh3d) in &mesh_q {
        let Some(src) = meshes.get(&mesh3d.0) else {
            continue;
        };
        if src.count_vertices() == 0 {
            continue;
        }
        let Some(indices) = face_vertex_indices(src) else {
            continue;
        };
        let mut mesh = src.clone();
        if !init_gray_colors(&mut mesh) {
            continue;
        }
        paint.face_indices = indices;
        let handle = meshes.add(mesh);
        let mat = materials.add(StandardMaterial {
            base_color: Color::WHITE,
            unlit: true,
            perceptual_roughness: 1.0,
            metallic: 0.0,
            ..default()
        });
        commands
            .entity(entity)
            .insert((Mesh3d(handle), MeshMaterial3d(mat), PaintTarget));
    }
}

fn ingest_bake_updates(mut paint: ResMut<BakePaint>) {
    let Some(bytes) = take_pending_bake() else {
        return;
    };
    let Ok(baked) = parse_bake(&bytes) else {
        return;
    };
    if !baked.s_min.is_finite() || !baked.s_max.is_finite() || baked.s_max < baked.s_min {
        // still warming up
    } else {
        paint.s_min = baked.s_min;
        paint.s_max = baked.s_max.max(baked.s_min + 1e-20);
    }
    if paint.painted.len() != baked.face_scalar.len() {
        paint.painted = vec![false; baked.face_scalar.len()];
        paint.scalars = vec![f32::NAN; baked.face_scalar.len()];
    }
    let mut newly = Vec::new();
    for (f, &s) in baked.face_scalar.iter().enumerate() {
        if s.is_finite() && !paint.painted[f] && !paint.scalars[f].is_finite() {
            newly.push(f as u32);
        }
        if s.is_finite() {
            paint.scalars[f] = s;
        }
    }
    if !newly.is_empty() {
        paint.shuffle_push(newly);
    }
}

fn paint_faces_from_queue(
    mut paint: ResMut<BakePaint>,
    mut meshes: ResMut<Assets<Mesh>>,
    targets: Query<&Mesh3d, With<PaintTarget>>,
) {
    if paint.queue.is_empty() || paint.face_indices.is_empty() {
        return;
    }
    let span = (paint.s_max - paint.s_min).max(1e-20);
    if !(paint.s_min.is_finite() && paint.s_max.is_finite()) {
        return;
    }

    for mesh3d in &targets {
        let Some(mut mesh) = meshes.get_mut(&mesh3d.0) else {
            continue;
        };
        let Some(VertexAttributeValues::Float32x4(colors)) =
            mesh.attribute_mut(Mesh::ATTRIBUTE_COLOR)
        else {
            continue;
        };

        let mut n = 0usize;
        while n < PAINT_PER_FRAME {
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
            let rgba = colormap_rgba((s - paint.s_min) / span);
            if paint_face_on_colors(colors, &paint.face_indices, fi, rgba) {
                paint.painted[fi] = true;
                n += 1;
            }
        }
    }
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
