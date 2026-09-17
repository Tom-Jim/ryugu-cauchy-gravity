//! GLB loading and the shared observation-surface recipe.

use std::fs;
use std::path::Path;

#[derive(Clone, Debug, Default)]
pub struct Mesh {
    /// xyz per vertex, meters.
    pub xyz: Vec<f64>,
    /// Triangle indices, 0-based.
    pub faces: Vec<u32>,
}

impl Mesh {
    pub fn load_glb(path: &Path, scale: f64) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
        let model = gltf::Gltf::from_slice(&bytes)
            .map_err(|error| format!("{}: invalid GLB: {error}", path.display()))?;
        let blob = model
            .blob
            .as_deref()
            .ok_or_else(|| format!("{}: GLB has no binary buffer", path.display()))?;
        let primitive = model
            .meshes()
            .flat_map(|mesh| mesh.primitives())
            .filter(|primitive| primitive.mode() == gltf::mesh::Mode::Triangles)
            .max_by_key(|primitive| primitive.indices().map_or(0, |indices| indices.count()))
            .ok_or_else(|| format!("{}: no indexed triangle primitive", path.display()))?;
        let reader = primitive.reader(|buffer| match buffer.source() {
            gltf::buffer::Source::Bin => Some(blob),
            gltf::buffer::Source::Uri(_) => None,
        });
        let positions = reader
            .read_positions()
            .ok_or_else(|| format!("{}: mesh has no POSITION attribute", path.display()))?;
        let mut mesh = Mesh::default();
        for point in positions {
            mesh.xyz.extend_from_slice(&[
                point[0] as f64 * scale,
                point[1] as f64 * scale,
                point[2] as f64 * scale,
            ]);
        }
        mesh.faces = reader
            .read_indices()
            .ok_or_else(|| format!("{}: mesh has no index accessor", path.display()))?
            .into_u32()
            .collect();
        if !mesh.faces.len().is_multiple_of(3) {
            return Err(format!(
                "{}: triangle index count is invalid",
                path.display()
            ));
        }
        let mut signed_six_volume = 0.0f64;
        for face in 0..mesh.face_count() {
            let [i0, i1, i2] = mesh.face(face);
            signed_six_volume += dot(
                mesh.vertex(i0 as usize),
                cross(mesh.vertex(i1 as usize), mesh.vertex(i2 as usize)),
            );
        }
        if signed_six_volume < 0.0 {
            for face in mesh.faces.chunks_exact_mut(3) {
                face.swap(1, 2);
            }
        }
        Ok(mesh)
    }

    pub fn vertex_count(&self) -> usize {
        self.xyz.len() / 3
    }

    pub fn face_count(&self) -> usize {
        self.faces.len() / 3
    }

    pub fn vertex(&self, i: usize) -> [f64; 3] {
        [self.xyz[3 * i], self.xyz[3 * i + 1], self.xyz[3 * i + 2]]
    }

    pub fn face(&self, f: usize) -> [u32; 3] {
        [
            self.faces[3 * f],
            self.faces[3 * f + 1],
            self.faces[3 * f + 2],
        ]
    }

    /// Area-weighted vertex normals from accumulated face cross products.
    pub fn vertex_normals(&self) -> Vec<[f64; 3]> {
        let mut normals = vec![[0.0f64; 3]; self.vertex_count()];
        for face in 0..self.face_count() {
            let [i0, i1, i2] = self.face(face);
            let (i0, i1, i2) = (i0 as usize, i1 as usize, i2 as usize);
            let (p0, p1, p2) = (self.vertex(i0), self.vertex(i1), self.vertex(i2));
            let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            let normal = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            for index in [i0, i1, i2] {
                normals[index][0] += normal[0];
                normals[index][1] += normal[1];
                normals[index][2] += normal[2];
            }
        }
        for (index, normal) in normals.iter_mut().enumerate() {
            let length = norm(*normal);
            if length > 1e-30 {
                *normal = [normal[0] / length, normal[1] / length, normal[2] / length];
            } else {
                let point = self.vertex(index);
                let radius = norm(point);
                *normal = if radius > 1e-30 {
                    [point[0] / radius, point[1] / radius, point[2] / radius]
                } else {
                    [0.0, 1.0, 0.0]
                };
            }
            let point = self.vertex(index);
            if dot(*normal, point) < 0.0 {
                *normal = [-normal[0], -normal[1], -normal[2]];
            }
        }
        normals
    }

    pub fn observation_points(&self, standoff_m: f64) -> Vec<[f64; 3]> {
        let normals = self.vertex_normals();
        (0..self.vertex_count())
            .map(|index| {
                let point = self.vertex(index);
                let normal = normals[index];
                [
                    point[0] + standoff_m * normal[0],
                    point[1] + standoff_m * normal[1],
                    point[2] + standoff_m * normal[2],
                ]
            })
            .collect()
    }
}

fn norm(value: [f64; 3]) -> f64 {
    dot(value, value).sqrt()
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
