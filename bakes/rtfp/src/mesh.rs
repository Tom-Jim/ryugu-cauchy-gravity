//! Mesh loading and the shared observation-surface recipe.
//!
//! The token rules of the OBJ reader (`v ` / `f `, three indices, 1-based) are
//! deliberately identical to `bakes/werner/bake_main.cpp`, so the RT-FP observation
//! surface is the one the Werner / mascon records were baked at.

use std::fs;
use std::io;
use std::path::Path;

#[derive(Clone, Debug, Default)]
pub struct Mesh {
    /// xyz per vertex, meters.
    pub xyz: Vec<f64>,
    /// Triangle indices, 0-based.
    pub faces: Vec<u32>,
}

impl Mesh {
    pub fn load_obj(path: &Path, km_to_m: f64) -> io::Result<Self> {
        let text = fs::read_to_string(path)?;
        let mut mesh = Mesh::default();
        for line in text.lines() {
            let bytes = line.as_bytes();
            if bytes.len() < 2 {
                continue;
            }
            if bytes[0] == b'v' && bytes[1] == b' ' {
                let mut it = line[2..].split_whitespace();
                let (Some(x), Some(y), Some(z)) = (it.next(), it.next(), it.next()) else {
                    continue;
                };
                let (Ok(x), Ok(y), Ok(z)) = (x.parse::<f64>(), y.parse::<f64>(), z.parse::<f64>())
                else {
                    continue;
                };
                mesh.xyz
                    .extend_from_slice(&[x * km_to_m, y * km_to_m, z * km_to_m]);
            } else if bytes[0] == b'f' && bytes[1] == b' ' {
                let mut idx = [0u32; 3];
                let mut n = 0;
                for token in line[2..].split_whitespace() {
                    if n == 3 {
                        break;
                    }
                    let head: String = token.chars().take_while(|c| c.is_ascii_digit()).collect();
                    let Ok(v) = head.parse::<u32>() else { continue };
                    idx[n] = v.saturating_sub(1);
                    n += 1;
                }
                if n == 3 {
                    mesh.faces.extend_from_slice(&idx);
                }
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

    /// Area-weighted vertex normals from the accumulated, un-normalised face
    /// cross products; zero-length sums fall back to the radial direction and
    /// every normal is flipped outward (`n·p >= 0`).
    pub fn vertex_normals(&self) -> Vec<[f64; 3]> {
        let nv = self.vertex_count();
        let mut normals = vec![[0.0f64; 3]; nv];
        for f in 0..self.face_count() {
            let [i0, i1, i2] = self.face(f);
            let (i0, i1, i2) = (i0 as usize, i1 as usize, i2 as usize);
            let (p0, p1, p2) = (self.vertex(i0), self.vertex(i1), self.vertex(i2));
            let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            let n = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            for i in [i0, i1, i2] {
                normals[i][0] += n[0];
                normals[i][1] += n[1];
                normals[i][2] += n[2];
            }
        }
        for (i, n) in normals.iter_mut().enumerate() {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len > 1e-30 {
                n[0] /= len;
                n[1] /= len;
                n[2] /= len;
            } else {
                let p = self.vertex(i);
                let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                *n = if r > 1e-30 {
                    [p[0] / r, p[1] / r, p[2] / r]
                } else {
                    [0.0, 1.0, 0.0]
                };
            }
            let p = self.vertex(i);
            if n[0] * p[0] + n[1] * p[1] + n[2] * p[2] < 0.0 {
                *n = [-n[0], -n[1], -n[2]];
            }
        }
        normals
    }

    pub fn observation_points(&self, standoff_m: f64) -> Vec<[f64; 3]> {
        let normals = self.vertex_normals();
        (0..self.vertex_count())
            .map(|i| {
                let p = self.vertex(i);
                let n = normals[i];
                [
                    p[0] + standoff_m * n[0],
                    p[1] + standoff_m * n[1],
                    p[2] + standoff_m * n[2],
                ]
            })
            .collect()
    }
}
