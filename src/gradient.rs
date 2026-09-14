//! Progressive per-face bake parse + flat face coloring.

use bevy::mesh::Indices;
use bevy::prelude::*;
use std::collections::VecDeque;
use std::sync::Mutex;

const MAGIC: u32 = 0x52484746;
const GRAY: [f32; 4] = [0.14, 0.14, 0.16, 1.0];

pub struct BakedFaces {
    pub face_scalar: Vec<f32>,
    pub completed: u32,
    pub s_min: f32,
    pub s_max: f32,
}

pub fn parse_bake(bytes: &[u8]) -> Result<BakedFaces, String> {
    if bytes.len() < 28 {
        return Err("bake too short".into());
    }
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    if magic != MAGIC {
        return Err(format!("bad magic {magic:#x}"));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let n_mesh = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let n_out = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let s_min = f32::from_le_bytes(bytes[20..24].try_into().unwrap());
    let s_max = f32::from_le_bytes(bytes[24..28].try_into().unwrap());

    match version {
        5 => {
            let need = 28 + n_mesh * 4;
            if bytes.len() < need {
                return Err("truncated v5 bake".into());
            }
            let mut face_scalar = Vec::with_capacity(n_mesh);
            for i in 0..n_mesh {
                let o = 28 + i * 4;
                face_scalar.push(f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()));
            }
            Ok(BakedFaces {
                face_scalar,
                completed: n_out as u32,
                s_min,
                s_max,
            })
        }
        4 => {
            let need = 28 + n_mesh * 4;
            if bytes.len() < need || n_out != n_mesh {
                return Err("truncated v4 bake".into());
            }
            let mut face_scalar = Vec::with_capacity(n_mesh);
            for i in 0..n_mesh {
                let o = 28 + i * 4;
                face_scalar.push(f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()));
            }
            Ok(BakedFaces {
                face_scalar,
                completed: n_mesh as u32,
                s_min,
                s_max,
            })
        }
        3 => {
            let need = 28 + n_out * 4 + n_out * 4 + n_out * 36;
            if bytes.len() < need {
                return Err("truncated v3 bake".into());
            }
            let mut face_scalar = vec![f32::NAN; n_mesh.max(n_out)];
            let ibase = 28;
            let sbase = 28 + n_out * 4;
            for i in 0..n_out {
                let face =
                    u32::from_le_bytes(bytes[ibase + i * 4..ibase + i * 4 + 4].try_into().unwrap())
                        as usize;
                let s = f32::from_le_bytes(bytes[sbase + i * 4..sbase + i * 4 + 4].try_into().unwrap());
                if face >= face_scalar.len() {
                    face_scalar.resize(face + 1, f32::NAN);
                }
                face_scalar[face] = s;
            }
            Ok(BakedFaces {
                face_scalar,
                completed: n_out as u32,
                s_min,
                s_max,
            })
        }
        v => Err(format!("unsupported bake v{v}")),
    }
}

static PENDING_BAKE: Mutex<Option<Vec<u8>>> = Mutex::new(None);

pub fn push_bake_bytes(bytes: Vec<u8>) {
    if let Ok(mut g) = PENDING_BAKE.lock() {
        *g = Some(bytes);
    }
}

pub fn take_pending_bake() -> Option<Vec<u8>> {
    PENDING_BAKE.lock().ok().and_then(|mut g| g.take())
}

pub fn colormap_rgba(t: f32) -> [f32; 4] {
    let x = t.clamp(0.0, 1.0);
    let c0 = [0.20, 0.05, 0.45];
    let c1 = [0.90, 0.15, 0.50];
    let c2 = [1.00, 0.65, 0.10];
    let mix = |a: [f32; 3], b: [f32; 3], u: f32| {
        [
            a[0] + (b[0] - a[0]) * u,
            a[1] + (b[1] - a[1]) * u,
            a[2] + (b[2] - a[2]) * u,
        ]
    };
    let rgb = if x < 0.5 {
        mix(c0, c1, x * 2.0)
    } else {
        mix(c1, c2, (x - 0.5) * 2.0)
    };
    [rgb[0], rgb[1], rgb[2], 1.0]
}

pub fn face_vertex_indices(mesh: &Mesh) -> Option<Vec<u32>> {
    if let Some(raw) = mesh.indices() {
        let indices: Vec<u32> = match raw {
            Indices::U16(v) => v.iter().map(|&i| i as u32).collect(),
            Indices::U32(v) => v.to_vec(),
        };
        if indices.len() >= 3 && indices.len().is_multiple_of(3) {
            return Some(indices);
        }
        return None;
    }
    let n = mesh.count_vertices();
    if n >= 3 && n.is_multiple_of(3) {
        return Some((0..n as u32).collect());
    }
    None
}

pub fn init_gray_colors(mesh: &mut Mesh) -> bool {
    let n = mesh.count_vertices();
    if n == 0 {
        return false;
    }
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, vec![GRAY; n]);
    true
}

pub fn paint_face_on_colors(
    colors: &mut [[f32; 4]],
    indices: &[u32],
    face: usize,
    rgba: [f32; 4],
) -> bool {
    if face * 3 + 2 >= indices.len() {
        return false;
    }
    for k in 0..3 {
        let vi = indices[face * 3 + k] as usize;
        if vi >= colors.len() {
            return false;
        }
        colors[vi] = rgba;
    }
    true
}

#[derive(Resource, Default)]
pub struct BakePaint {
    pub scalars: Vec<f32>,
    pub s_min: f32,
    pub s_max: f32,
    pub painted: Vec<bool>,
    pub queue: VecDeque<u32>,
    pub face_indices: Vec<u32>,
    pub rng: u64,
}

impl BakePaint {
    pub fn shuffle_push(&mut self, mut faces: Vec<u32>) {
        // Fisher–Yates with xorshift
        let mut i = faces.len();
        while i > 1 {
            i -= 1;
            self.rng ^= self.rng << 13;
            self.rng ^= self.rng >> 7;
            self.rng ^= self.rng << 17;
            if self.rng == 0 {
                self.rng = 0x9e3779b97f4a7c15;
            }
            let j = (self.rng as usize) % (i + 1);
            faces.swap(i, j);
        }
        self.queue.extend(faces);
    }
}
