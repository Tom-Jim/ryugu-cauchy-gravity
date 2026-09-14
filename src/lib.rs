//! Minimal Ryugu viewer: GLB body + ESA gravity-gradient WGSL overlay.

mod gradient;

use bevy::asset::AssetMetaCheck;
use bevy::camera::primitives::Aabb;
use bevy::log::{Level, LogPlugin};
use bevy::prelude::*;
use bevy::render::settings::{Backends, RenderCreation, WgpuSettings, WgpuSettingsPriority};
use bevy::render::{RenderPlugin, render_resource::WgpuLimits};
use bevy::window::PresentMode;
use bevy::winit::{UpdateMode, WinitSettings};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use gradient::{GradientMaterial, embedded_bake, spawn_gradient_overlay};
use std::time::Duration;
use wasm_bindgen::prelude::wasm_bindgen;

const RYUGU_ROTATION_PERIOD_SECS: f64 = 7.63 * 3600.0;
const RYUGU_SPIN_AXIS: Vec3 = Vec3::new(-0.043, -0.914, 0.405);
const DISPLAY_TIME_SCALE: f64 = 1000.0;
const RYUGU_TARGET_SIZE: f32 = 900.0;

#[derive(Component)]
struct RyuguMarker;

#[derive(Component)]
struct TargetSize(f32);

#[derive(Component)]
struct ScaleNormalized;

#[derive(Resource, Default)]
struct BodyClock {
    elapsed_seconds: f64,
}

#[derive(Resource, Default)]
struct GradientSpawned(bool);

#[wasm_bindgen(start)]
pub fn run() {
    let mut app = App::new();
    app.insert_resource(ClearColor(Color::srgb(0.02, 0.02, 0.05)))
        .init_resource::<BodyClock>()
        .init_resource::<GradientSpawned>()
        .insert_resource(WinitSettings {
            focused_mode: UpdateMode::Continuous,
            unfocused_mode: if cfg!(target_arch = "wasm32") {
                UpdateMode::Continuous
            } else {
                UpdateMode::reactive_low_power(Duration::from_secs_f64(1.0 / 30.0))
            },
        })
        .add_plugins(
            DefaultPlugins
                .build()
                .set(LogPlugin {
                    level: Level::WARN,
                    filter: "warn,wgpu=error,naga=error,bevy_render=error".into(),
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
                        title: "Ryugu gravity gradient".into(),
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
        .add_plugins(MaterialPlugin::<GradientMaterial>::default())
        .add_systems(Startup, setup_scene)
        .add_systems(
            Update,
            (
                normalize_model_scale_system,
                spawn_overlay_when_ready,
                advance_clock,
                rotate_ryugu,
            )
                .chain(),
        );

    app.run();
}

fn setup_scene(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.85, 0.85, 1.0),
        brightness: 180.0,
        ..default()
    });

    commands.spawn((
        DirectionalLight {
            illuminance: 80_000.0,
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
        WorldAssetRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset("models/ryugu.glb"))),
        TargetSize(RYUGU_TARGET_SIZE),
        Transform::default(),
        RyuguMarker,
    ));
}

fn spawn_overlay_when_ready(
    mut commands: Commands,
    mut spawned: ResMut<GradientSpawned>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<GradientMaterial>>,
    ryugu: Query<Entity, (With<RyuguMarker>, With<ScaleNormalized>)>,
) {
    if spawned.0 {
        return;
    }
    let Ok(parent) = ryugu.single() else {
        return;
    };
    let Ok(baked) = embedded_bake() else {
        spawned.0 = true;
        return;
    };
    spawn_gradient_overlay(&mut commands, &mut meshes, &mut materials, parent, &baked);
    spawned.0 = true;
}

fn normalize_model_scale_system(
    mut commands: Commands,
    targets: Query<(Entity, &TargetSize), Without<ScaleNormalized>>,
    children: Query<&Children>,
    aabbs: Query<&Aabb>,
) {
    for (entity, target) in &targets {
        let Some(max_dim) = max_aabb_extent(entity, &children, &aabbs) else {
            continue;
        };
        if max_dim <= 1e-5 {
            continue;
        }
        let scale = target.0 / max_dim;
        commands.entity(entity).insert(ScaleNormalized);
        commands
            .entity(entity)
            .entry::<Transform>()
            .and_modify(move |mut t| t.scale = Vec3::splat(scale));
    }
}

fn max_aabb_extent(
    entity: Entity,
    children: &Query<&Children>,
    aabbs: &Query<&Aabb>,
) -> Option<f32> {
    let mut max_extent = 0.0_f32;
    let mut found = false;
    let mut stack = vec![entity];
    while let Some(curr) = stack.pop() {
        if let Ok(aabb) = aabbs.get(curr) {
            let size = aabb.half_extents * 2.0;
            max_extent = max_extent.max(size.x).max(size.y).max(size.z);
            found = true;
        }
        if let Ok(kids) = children.get(curr) {
            for child in kids.iter() {
                stack.push(child);
            }
        }
    }
    found.then_some(max_extent)
}

fn advance_clock(time: Res<Time>, mut clock: ResMut<BodyClock>) {
    clock.elapsed_seconds += time.delta_secs_f64() * DISPLAY_TIME_SCALE;
}

fn rotate_ryugu(mut query: Query<&mut Transform, With<RyuguMarker>>, clock: Res<BodyClock>) {
    let omega = std::f64::consts::TAU / RYUGU_ROTATION_PERIOD_SECS;
    let rotation = Quat::from_axis_angle(
        RYUGU_SPIN_AXIS.normalize(),
        (omega * clock.elapsed_seconds) as f32,
    );
    for mut transform in &mut query {
        transform.rotation = rotation;
    }
}
