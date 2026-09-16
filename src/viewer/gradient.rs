//! Progressive per-face bake parse + flat face coloring.

use bevy::mesh::{Indices, VertexAttributeValues};
use bevy::prelude::*;
use std::collections::VecDeque;
use std::sync::Mutex;

const MAGIC: u32 = 0x52484746;
const GRAY: [f32; 4] = [0.58, 0.58, 0.62, 1.0];

pub struct BakedFaces {
    pub face_scalar: Vec<f32>,
}

/// Parse an RHGF v5 face record: a 28-byte header
/// (`magic, version, n_faces, n_done, standoff_mm, s_min, s_max`) followed by one
/// `f32` per face, `NaN` until the bake reaches that face.
///
/// `standoff_mm`, `s_min` and `s_max` are not used here: the display window is
/// derived from robust quantiles of the scalars themselves, so equal values map
/// to equal colours no matter which algorithm produced the record. The older
/// v3/v4 layouts had a different face indexing and are rejected outright.
pub fn parse_bake(bytes: &[u8]) -> Result<BakedFaces, String> {
    if bytes.len() < 28 {
        return Err("bake too short".into());
    }
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    if magic != MAGIC {
        return Err(format!("bad magic {magic:#x}"));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != 5 {
        return Err(format!("unsupported bake v{version}"));
    }
    let n_faces = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    if bytes.len() < 28 + n_faces * 4 {
        return Err("truncated v5 bake".into());
    }
    let mut face_scalar = Vec::with_capacity(n_faces);
    for i in 0..n_faces {
        let o = 28 + i * 4;
        face_scalar.push(f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()));
    }
    Ok(BakedFaces { face_scalar })
}

static PENDING_BAKE: Mutex<Option<Result<Vec<f32>, String>>> = Mutex::new(None);

pub fn push_bake_bytes(bytes: &[u8]) {
    // Parse at the WASM boundary so only one face-scalar allocation survives
    // into the next frame; retaining the raw bytes as well doubled the peak.
    let parsed = parse_bake(bytes).map(|baked| baked.face_scalar);
    if let Ok(mut g) = PENDING_BAKE.lock() {
        *g = Some(parsed);
    }
}

pub fn take_pending_bake() -> Option<Result<Vec<f32>, String>> {
    PENDING_BAKE.lock().ok().and_then(|mut g| g.take())
}

pub fn colormap_rgba(t: f32) -> [f32; 4] {
    // Smooth 4-stop ramp: deep blue → violet → magenta → soft amber (avoids
    // speckly neon yellow on heavy-tailed mascon outliers).
    let x = t.clamp(0.0, 1.0);
    let stops: [[f32; 3]; 4] = [
        [0.12, 0.10, 0.42],
        [0.45, 0.18, 0.72],
        [0.88, 0.28, 0.55],
        [0.98, 0.78, 0.35],
    ];
    let mix = |a: [f32; 3], b: [f32; 3], u: f32| {
        [
            a[0] + (b[0] - a[0]) * u,
            a[1] + (b[1] - a[1]) * u,
            a[2] + (b[2] - a[2]) * u,
        ]
    };
    let seg = (x * 3.0).floor().min(2.0);
    let u = (x * 3.0 - seg).clamp(0.0, 1.0);
    let i = seg as usize;
    let rgb = mix(stops[i], stops[i + 1], u);
    [rgb[0], rgb[1], rgb[2], 1.0]
}

/// How far up a linear 2 %–98 % ramp the median face may sit before the body is
/// considered squashed onto one colour. 1/8 is a wide margin: ‖H‖_F lands at
/// ≈0.34 for Werner / RT-FP (mild, stays linear) and ≈0.05 for mascon (heavy
/// right tail, switches to asinh), so neither can flip on a small change.
const HEAVY_TAIL_MED_T: f32 = 0.125;

/// (2 %, 50 %, 98 %) of the finite, strictly positive scalars.
///
/// One sort, and no min/max anywhere: a handful of extreme faces must not move
/// the mapping, so two runs whose outliers differ slightly still colour alike.
fn robust_quantiles(scalars: &[f32]) -> Option<(f32, f32, f32)> {
    let mut vals: Vec<f32> = scalars
        .iter()
        .copied()
        .filter(|s| s.is_finite() && *s > 0.0)
        .collect();
    if vals.is_empty() {
        return None;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = vals.len();
    let at = |q: f32| vals[((n - 1) as f32 * q).round() as usize];
    Some((at(0.02), at(0.5), at(0.98)))
}

/// The scalar → colour window shared by **every** viewer path.
///
/// Werner bake, mascon bake and RT-FP all derive their window with
/// [`display_window_for`] and map scalars with [`DisplayWindow::map`], so two
/// faces whose values agree also land on (nearly) the same colour no matter
/// which algorithm produced them.
#[derive(Clone, Copy, Debug)]
pub struct DisplayWindow {
    /// Bounds in mapped space (`t = 0` at `lo`, `t = 1` at `hi`).
    pub lo: f32,
    pub hi: f32,
    /// 0 = linear, 1 = log10, 2 = asinh(s / median).
    pub scale_mode: u8,
    /// Median used by the asinh mapping.
    pub asinh_med: f32,
}

impl Default for DisplayWindow {
    fn default() -> Self {
        Self {
            lo: f32::INFINITY,
            hi: f32::NEG_INFINITY,
            scale_mode: 0,
            asinh_med: 1.0,
        }
    }
}

impl DisplayWindow {
    /// False until a finite window has been derived from real scalars.
    pub fn is_valid(&self) -> bool {
        self.lo.is_finite() && self.hi.is_finite()
    }

    /// Map one scalar into colormap space; `colormap_rgba` clamps to `[0, 1]`.
    pub fn map(&self, s: f32) -> Option<f32> {
        if !s.is_finite() || !self.is_valid() {
            return None;
        }
        let span = (self.hi - self.lo).max(1e-30);
        let t = match self.scale_mode {
            1 => {
                if s <= 0.0 {
                    return None;
                }
                (s.log10() - self.lo) / span
            }
            2 => {
                if s <= 0.0 {
                    return None;
                }
                let med = self.asinh_med.max(1e-30);
                ((s / med).asinh() - self.lo) / span
            }
            _ => (s - self.lo) / span,
        };
        Some(t)
    }
}

/// Derive the display window for a set of face scalars.
///
/// Both candidates are honest 2 %–98 % percentile stretches, so the choice only
/// decides *how* the body is spread, never how much of the tail is kept:
///
/// * linear — the default; puts the median at t≈0.34 for a well-behaved field.
/// * `asinh(s / median)` — for a heavy right tail, where the median face would
///   land in the bottom eighth of a linear ramp and the body would collapse onto
///   one colour (mascon ‖H‖_F: 75 % of faces in the bottom 10 % vs 25 %).
///
/// The decision comes from robust quantiles instead of min/max, so two runs
/// whose extreme faces differ slightly always pick the same mapping. This single
/// rule is what keeps Werner / mascon / RT-FP colouring comparable.
pub fn display_window_for(scalars: &[f32]) -> DisplayWindow {
    let Some((lo, med, hi)) = robust_quantiles(scalars) else {
        return DisplayWindow::default();
    };
    // `!(hi > lo)` also catches a non-finite pair; state it the readable way.
    if !hi.is_finite() || !lo.is_finite() || hi <= lo {
        // Degenerate (constant) field: pin it to the bottom of the ramp.
        let hi = lo + (lo.abs() * 1e-6).max(1e-30);
        return DisplayWindow {
            lo,
            hi,
            scale_mode: 0,
            asinh_med: 1.0,
        };
    }
    if (med - lo) / (hi - lo) < HEAVY_TAIL_MED_T {
        let (alo, ahi) = ((lo / med).asinh(), (hi / med).asinh());
        if ahi > alo {
            return DisplayWindow {
                lo: alo,
                hi: ahi,
                scale_mode: 2,
                asinh_med: med,
            };
        }
    }
    DisplayWindow {
        lo,
        hi,
        scale_mode: 0,
        asinh_med: 1.0,
    }
}

/// Scalar → colour through a shared [`DisplayWindow`].
pub fn colormap_scalar(window: &DisplayWindow, s: f32) -> Option<[f32; 4]> {
    window.map(s).map(colormap_rgba)
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

/// Duplicate vertices per face so flat face colors never bleed across shared edges.
/// Returns the exploded mesh and its triangle count; vertices remain in face order,
/// so no second index array has to survive in the viewer.
pub fn explode_mesh_for_flat_faces(src: &Mesh) -> Option<(Mesh, usize)> {
    let indices = face_vertex_indices(src)?;
    let Some(VertexAttributeValues::Float32x3(pos)) = src.attribute(Mesh::ATTRIBUTE_POSITION)
    else {
        return None;
    };
    let normals = match src.attribute(Mesh::ATTRIBUTE_NORMAL) {
        Some(VertexAttributeValues::Float32x3(n)) => Some(n.as_slice()),
        _ => None,
    };

    let n_corners = indices.len();
    let mut new_pos = Vec::with_capacity(n_corners);
    let mut new_nor = Vec::with_capacity(n_corners);

    for face in 0..(n_corners / 3) {
        let i0 = indices[face * 3] as usize;
        let i1 = indices[face * 3 + 1] as usize;
        let i2 = indices[face * 3 + 2] as usize;
        if i0 >= pos.len() || i1 >= pos.len() || i2 >= pos.len() {
            return None;
        }
        let p0 = pos[i0];
        let p1 = pos[i1];
        let p2 = pos[i2];
        let face_n = {
            let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            let cx = e1[1] * e2[2] - e1[2] * e2[1];
            let cy = e1[2] * e2[0] - e1[0] * e2[2];
            let cz = e1[0] * e2[1] - e1[1] * e2[0];
            let len = (cx * cx + cy * cy + cz * cz).sqrt().max(1e-20);
            [cx / len, cy / len, cz / len]
        };
        for vi in [i0, i1, i2] {
            new_pos.push(pos[vi]);
            if let Some(nattr) = normals {
                new_nor.push(if vi < nattr.len() { nattr[vi] } else { face_n });
            } else {
                new_nor.push(face_n);
            }
        }
    }

    let mut mesh = Mesh::new(
        bevy::mesh::PrimitiveTopology::TriangleList,
        bevy::asset::RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, new_pos);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, new_nor);
    Some((mesh, n_corners / 3))
}

pub fn init_gray_colors(mesh: &mut Mesh) -> bool {
    let n = mesh.count_vertices();
    if n == 0 {
        return false;
    }
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, vec![GRAY; n]);
    true
}

pub fn paint_face_on_colors(colors: &mut [[f32; 4]], face: usize, rgba: [f32; 4]) -> bool {
    if face * 3 + 2 >= colors.len() {
        return false;
    }
    for k in 0..3 {
        colors[face * 3 + k] = rgba;
    }
    true
}

#[derive(Resource, Default)]
pub struct BakePaint {
    pub scalars: Vec<f32>,
    /// Window shared with the RT-FP path (identical values → identical colours).
    pub window: DisplayWindow,
    pub painted: Vec<bool>,
    pub queue: VecDeque<u32>,
    /// Triangle count of the exploded mesh.
    pub face_count: usize,
    pub rng: u64,
    /// When true, next paint pass restores the whole mesh to gray (bake restart).
    pub reset_to_gray: bool,
    /// Skip redundant ingest when the on-disk finite count has not changed.
    pub last_finite: usize,
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
