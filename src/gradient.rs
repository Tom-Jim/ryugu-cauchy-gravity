//! Baked ESA near-surface gravity-gradient overlay (WGSL colormap material).

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;

const MAGIC: u32 = 0x52484746;

#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub struct GradientUniforms {
    pub s_min: f32,
    pub s_max: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct GradientMaterial {
    #[uniform(0)]
    pub gradient: GradientUniforms,
}

impl Material for GradientMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/gradient_material.wgsl".into()
    }
}

#[derive(Component)]
pub struct GradientOverlay;

#[derive(Clone, Debug)]
pub struct BakedFaces {
    pub scalars: Vec<f32>,
    pub positions_km: Vec<[f32; 3]>,
    pub s_min: f32,
    pub s_max: f32,
}

pub fn parse_bake(bytes: &[u8]) -> Result<BakedFaces, String> {
    if bytes.len() < 28 {
        return Err("bake file too short".into());
    }
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    if magic != MAGIC {
        return Err(format!("bad magic {magic:#x}"));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let n_out = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let s_min = f32::from_le_bytes(bytes[20..24].try_into().unwrap());
    let s_max = f32::from_le_bytes(bytes[24..28].try_into().unwrap());
    if version != 3 {
        return Err(format!("unsupported bake version {version}"));
    }
    let idx_bytes = n_out * 4;
    let scalar_bytes = n_out * 4;
    let pos_bytes = n_out * 9 * 4;
    let need = 28 + idx_bytes + scalar_bytes + pos_bytes;
    if bytes.len() < need {
        return Err("truncated bake payload".into());
    }
    let mut scalars = Vec::with_capacity(n_out);
    let sbase = 28 + idx_bytes;
    for i in 0..n_out {
        let o = sbase + i * 4;
        scalars.push(f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()));
    }
    let mut positions_km = Vec::with_capacity(n_out * 3);
    let pbase = sbase + scalar_bytes;
    for i in 0..(n_out * 3) {
        let o = pbase + i * 12;
        positions_km.push([
            f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()),
            f32::from_le_bytes(bytes[o + 4..o + 8].try_into().unwrap()),
            f32::from_le_bytes(bytes[o + 8..o + 12].try_into().unwrap()),
        ]);
    }
    Ok(BakedFaces {
        scalars,
        positions_km,
        s_min,
        s_max,
    })
}

pub fn spawn_gradient_overlay(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<GradientMaterial>,
    parent: Entity,
    baked: &BakedFaces,
) {
    let mut positions = Vec::with_capacity(baked.positions_km.len());
    let mut colors = Vec::with_capacity(baked.positions_km.len());
    let mut indices = Vec::with_capacity(baked.scalars.len() * 3);
    for (face, &scalar) in baked.scalars.iter().enumerate() {
        let base = positions.len() as u32;
        for k in 0..3 {
            positions.push(baked.positions_km[face * 3 + k]);
            // R channel carries \|H̄\|_F for the WGSL colormap.
            colors.push([scalar, 0.0, 0.0, 1.0]);
        }
        indices.extend_from_slice(&[base, base + 1, base + 2]);
    }
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
    mesh.compute_normals();

    let gradient_mat = materials.add(GradientMaterial {
        gradient: GradientUniforms {
            s_min: baked.s_min,
            s_max: baked.s_max.max(baked.s_min + 1e-20),
            _pad0: 0.0,
            _pad1: 0.0,
        },
    });

    commands.entity(parent).with_children(|child| {
        child.spawn((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(gradient_mat),
            Transform::default(),
            GradientOverlay,
            Name::new("GravityGradientOverlay"),
        ));
    });
}

pub fn embedded_bake() -> Result<BakedFaces, String> {
    parse_bake(include_bytes!("../assets/gradient_faces.bin"))
}
