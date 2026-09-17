//! Reference ray tracers, used **only** by `--selftest`. Nothing in the bake
//! path calls them: `shaders/rays.wgsl` does the tracing.
//!
//! Two independent checks live here. [`BruteTracer`] tests every triangle in
//! f64 and is the ground truth; [`bvh_crossings`] walks the *same flat BVH the
//! shader consumes* (built by `bvh.rs`) with `Rust` recursion, so agreement
//! between the two isolates the WGSL traversal and its f32 arithmetic.

use crate::bvh::Bvh;
use crate::mesh::Mesh;

/// Matches `MAX_HITS` in `rays.wgsl`.
pub const MAX_HITS: usize = 32;

/// Exact crossings of every triangle of the mesh along `o + t·d`.
pub struct BruteTracer<'a> {
    mesh: &'a Mesh,
}

impl<'a> BruteTracer<'a> {
    pub fn new(mesh: &'a Mesh) -> Self {
        Self { mesh }
    }

    /// Crossing distances in `(t_min, t_max)`, ascending, capped at [`MAX_HITS`].
    /// Shared-edge/vertex hits are de-duplicated because they are one boundary
    /// crossing even though two or more triangles report it.
    pub fn crossings(&self, o: [f64; 3], d: [f64; 3], t_min: f64, t_max: f64, out: &mut Vec<f64>) {
        out.clear();
        for f in 0..self.mesh.face_count() {
            let [i0, i1, i2] = self.mesh.face(f);
            let v0 = self.mesh.vertex(i0 as usize);
            let v1 = self.mesh.vertex(i1 as usize);
            let v2 = self.mesh.vertex(i2 as usize);
            if let Some(t) = intersect(v0, v1, v2, o, d)
                && t > t_min
                && t < t_max
            {
                out.push(t);
            }
        }
        out.sort_by(|a, b| a.partial_cmp(b).unwrap());
        dedup_crossings(out);
        out.truncate(MAX_HITS);
    }
}

/// Walk the flat BVH exactly as `shaders/rays.wgsl` does, but in f64.
pub fn bvh_crossings(
    bvh: &Bvh,
    mesh: &Mesh,
    o: [f64; 3],
    d: [f64; 3],
    t_min: f64,
    t_max: f64,
    out: &mut Vec<f64>,
) {
    out.clear();
    let mut stack = vec![0u32];
    while let Some(node) = stack.pop() {
        let (lo, hi) = (
            &bvh.bounds[2 * node as usize],
            &bvh.bounds[2 * node as usize + 1],
        );
        let mut near = t_min;
        let mut far = t_max;
        for k in 0..3 {
            let inv = 1.0 / d[k];
            let a = (lo[k] as f64 - o[k]) * inv;
            let b = (hi[k] as f64 - o[k]) * inv;
            near = near.max(a.min(b));
            far = far.min(a.max(b));
        }
        if far < near {
            continue;
        }
        let meta = bvh.meta[node as usize];
        if meta[3] > 0 {
            for i in 0..meta[3] {
                let tri = bvh.order[(meta[2] + i) as usize];
                let [i0, i1, i2] = mesh.face(tri as usize);
                let v0 = mesh.vertex(i0 as usize);
                let v1 = mesh.vertex(i1 as usize);
                let v2 = mesh.vertex(i2 as usize);
                if let Some(t) = intersect(v0, v1, v2, o, d)
                    && t > t_min
                    && t < t_max
                {
                    out.push(t);
                }
            }
        } else {
            stack.push(meta[0]);
            stack.push(meta[1]);
        }
    }
    out.sort_by(|a, b| a.partial_cmp(b).unwrap());
    dedup_crossings(out);
    out.truncate(MAX_HITS);
}

fn dedup_crossings(hits: &mut Vec<f64>) {
    hits.dedup_by(|a, b| {
        let scale = a.abs().max(b.abs()).max(1.0);
        (*a - *b).abs() <= (8e-7 * scale).max(1e-5)
    });
}

/// Möller–Trumbore in f64, no backface culling.
fn intersect(v0: [f64; 3], v1: [f64; 3], v2: [f64; 3], o: [f64; 3], d: [f64; 3]) -> Option<f64> {
    let e1 = sub(v1, v0);
    let e2 = sub(v2, v0);
    let pvec = cross(d, e2);
    let det = dot(e1, pvec);
    if det.abs() < 1e-18 {
        return None;
    }
    let inv = 1.0 / det;
    let tvec = sub(o, v0);
    let u = dot(tvec, pvec) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let qvec = cross(tvec, e1);
    let v = dot(d, qvec) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = dot(e2, qvec) * inv;
    if t.is_finite() { Some(t) } else { None }
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
