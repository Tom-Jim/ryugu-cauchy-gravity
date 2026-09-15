//! Binary BVH over the mesh triangles, in the flat layout the WGSL traversal
//! consumes.
//!
//! WebGPU has no ray-tracing stage, so the tracer lives in
//! `shaders/rays.wgsl`; this module only *builds* the acceleration structure —
//! a serial build of an index permutation, not a parallel numeric loop (every
//! parallel loop in this crate runs as a compute dispatch).
//!
//! Layout: two `vec4<f32>` rows of bounds per node (`min`, `max`) and one
//! `vec4<u32>` of links `(left, right, start, count)`. `count > 0` marks a leaf
//! whose triangles are `order[start .. start + count]`.

use crate::mesh::Mesh;

/// Triangles per leaf. Small enough that a leaf test is cheap, large enough that
/// 196 k triangles do not turn into 196 k nodes.
const LEAF_SIZE: usize = 8;
const MAX_DEPTH: u32 = 40;

#[derive(Clone, Copy)]
struct Node {
    min: [f32; 3],
    max: [f32; 3],
    left: u32,
    right: u32,
    start: u32,
    count: u32,
}

impl Default for Node {
    fn default() -> Self {
        Self {
            min: [0.0; 3],
            max: [0.0; 3],
            left: 0,
            right: 0,
            start: 0,
            count: 0,
        }
    }
}

pub struct Bvh {
    /// Two rows per node: `(min.xyz, 0)`, `(max.xyz, 0)`.
    pub bounds: Vec<[f32; 4]>,
    /// One row per node: `(left, right, start, count)`.
    pub meta: Vec<[u32; 4]>,
    /// Triangle permutation; leaf ranges index into this.
    pub order: Vec<u32>,
    pub depth: u32,
}

impl Bvh {
    pub fn node_count(&self) -> usize {
        self.meta.len()
    }

    pub fn build(mesh: &Mesh) -> Self {
        let nf = mesh.face_count();
        let mut order: Vec<u32> = (0..nf as u32).collect();
        let mut tris: Vec<([f32; 3], [f32; 3], [f32; 3])> = Vec::with_capacity(nf);
        for f in 0..nf {
            let [i0, i1, i2] = mesh.face(f);
            let c = [
                mesh.vertex(i0 as usize),
                mesh.vertex(i1 as usize),
                mesh.vertex(i2 as usize),
            ];
            let lo = [
                c[0][0].min(c[1][0]).min(c[2][0]) as f32,
                c[0][1].min(c[1][1]).min(c[2][1]) as f32,
                c[0][2].min(c[1][2]).min(c[2][2]) as f32,
            ];
            let hi = [
                c[0][0].max(c[1][0]).max(c[2][0]) as f32,
                c[0][1].max(c[1][1]).max(c[2][1]) as f32,
                c[0][2].max(c[1][2]).max(c[2][2]) as f32,
            ];
            let centroid = [
                ((lo[0] + hi[0]) * 0.5),
                ((lo[1] + hi[1]) * 0.5),
                ((lo[2] + hi[2]) * 0.5),
            ];
            // Pad by a hair so a hit exactly on a face is never missed by the
            // f32 slab test.
            let pad = 1e-4f32;
            tris.push((
                [lo[0] - pad, lo[1] - pad, lo[2] - pad],
                [hi[0] + pad, hi[1] + pad, hi[2] + pad],
                centroid,
            ));
        }
        let mut b = Builder {
            tris: &tris,
            nodes: Vec::with_capacity(2 * nf / LEAF_SIZE + 8),
            depth: 0,
        };
        b.build(&mut order, 0, 0);
        let mut bounds = Vec::with_capacity(b.nodes.len() * 2);
        let mut meta = Vec::with_capacity(b.nodes.len());
        for n in &b.nodes {
            bounds.push([n.min[0], n.min[1], n.min[2], 0.0]);
            bounds.push([n.max[0], n.max[1], n.max[2], 0.0]);
            meta.push([n.left, n.right, n.start, n.count]);
        }
        Self {
            bounds,
            meta,
            order,
            depth: b.depth,
        }
    }
}

struct Builder<'a> {
    tris: &'a [([f32; 3], [f32; 3], [f32; 3])],
    nodes: Vec<Node>,
    depth: u32,
}

impl Builder<'_> {
    fn build(&mut self, idx: &mut [u32], offset: usize, depth: u32) -> u32 {
        let me = self.nodes.len() as u32;
        self.nodes.push(Node::default());
        self.depth = self.depth.max(depth);
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        let mut clo = [f32::INFINITY; 3];
        let mut chi = [f32::NEG_INFINITY; 3];
        for &t in idx.iter() {
            let (tlo, thi, tc) = self.tris[t as usize];
            for k in 0..3 {
                lo[k] = lo[k].min(tlo[k]);
                hi[k] = hi[k].max(thi[k]);
                clo[k] = clo[k].min(tc[k]);
                chi[k] = chi[k].max(tc[k]);
            }
        }
        self.nodes[me as usize].min = lo;
        self.nodes[me as usize].max = hi;
        if idx.len() <= LEAF_SIZE || depth >= MAX_DEPTH {
            let slot = &mut self.nodes[me as usize];
            slot.start = offset as u32;
            slot.count = idx.len() as u32;
            return me;
        }
        // Split on the widest centroid axis; ties fall back to the widest extent.
        let mut axis = 0usize;
        let mut best = chi[0] - clo[0];
        for k in 1..3 {
            let e = chi[k] - clo[k];
            if e > best {
                best = e;
                axis = k;
            }
        }
        let centroids = self.tris;
        let mid = idx.len() / 2;
        idx.select_nth_unstable_by(mid, |a, b| {
            let ca = centroids[*a as usize].2[axis];
            let cb = centroids[*b as usize].2[axis];
            ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
        });
        let (left_slice, right_slice) = idx.split_at_mut(mid);
        let left = self.build(left_slice, offset, depth + 1);
        let right = self.build(right_slice, offset + mid, depth + 1);
        let slot = &mut self.nodes[me as usize];
        slot.left = left;
        slot.right = right;
        me
    }
}
