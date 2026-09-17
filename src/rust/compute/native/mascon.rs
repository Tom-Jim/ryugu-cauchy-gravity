//! Rust CPU Barnes-Hut mascon evaluator used by the native five-solver host.
//!
//! The browser and native paths consume the same compact point/tree assets.
//! This keeps mascon independent from the analytic, ray and Carlson kernels
//! while avoiding a JavaScript or C++ numerical implementation.

use crate::tensor::Sym6;
use std::fs;
use std::path::Path;

/// Opening angle of the Barnes-Hut walk, kept equal to `MASCON_THETA` in
/// `src/rust/compute/browser/`; the test below asserts the accuracy this value
/// has to deliver, so the two cannot drift apart silently.
#[cfg(test)]
pub const SHIPPED_THETA: f64 = 0.1;

#[derive(Clone)]
pub struct Tree {
    pub grid: u32,
    pub point_count: usize,
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub node_count: usize,
    pub nodes: Vec<f32>,
    pub order: Vec<u32>,
    pub points: Vec<u32>,
    pub cauchy_scale: f32,
    pub constant_density: f32,
}

/// Node-visit accounting for the Barnes-Hut walk. The bench and the asset tests
/// use it to show how much of the tree a single face actually opens, which is
/// the only knob (besides `theta`) that controls the cost of a dispatch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WalkStats {
    /// Tree nodes popped off the stack.
    pub nodes: u64,
    /// Nodes expanded into their two children.
    pub opened: u64,
    /// Nodes accepted as a single monopole.
    pub merged: u64,
    /// Leaf nodes whose voxel lists were summed individually.
    pub leaves: u64,
    /// Voxels summed individually through the leaf lists.
    pub points: u64,
}

fn u32_at(bytes: &[u8], offset: usize) -> Result<u32, String> {
    let end = offset.checked_add(4).ok_or("offset overflow")?;
    let raw = bytes.get(offset..end).ok_or("truncated mascon asset")?;
    Ok(u32::from_le_bytes(raw.try_into().unwrap()))
}

fn f32_at(bytes: &[u8], offset: usize) -> Result<f32, String> {
    Ok(f32::from_bits(u32_at(bytes, offset)?))
}

impl Tree {
    pub fn load(tree_path: &Path, point_path: &Path) -> Result<Self, String> {
        let tree = fs::read(tree_path).map_err(|e| format!("{}: {e}", tree_path.display()))?;
        let points = fs::read(point_path).map_err(|e| format!("{}: {e}", point_path.display()))?;
        if u32_at(&tree, 0)? != 0x314d_5452 || u32_at(&tree, 4)? != 2 {
            return Err("invalid mascon tree header".into());
        }
        if u32_at(&points, 0)? != 0x314d_5952 || u32_at(&points, 4)? != 1 {
            return Err("invalid mascon point header".into());
        }
        let grid = u32_at(&tree, 8)?;
        let point_count = u32_at(&tree, 12)? as usize;
        let node_count = u32_at(&tree, 56)? as usize;
        if u32_at(&points, 12)? as usize != point_count {
            return Err("mascon tree/point count mismatch".into());
        }
        let node_bytes = node_count.checked_mul(80).ok_or("node count overflow")?;
        let order_offset = 64usize
            .checked_add(node_bytes)
            .ok_or("order offset overflow")?;
        let order_bytes = point_count.checked_mul(4).ok_or("point count overflow")?;
        if tree.len() < order_offset + order_bytes || points.len() < 64 + point_count * 16 {
            return Err("truncated mascon tree or point asset".into());
        }
        let mut nodes = Vec::with_capacity(node_count * 20);
        for offset in (64..64 + node_bytes).step_by(4) {
            nodes.push(f32_at(&tree, offset)?);
        }
        let mut order = Vec::with_capacity(point_count);
        for offset in (order_offset..order_offset + order_bytes).step_by(4) {
            order.push(u32_at(&tree, offset)?);
        }
        let mut point_words = Vec::with_capacity(point_count * 4);
        for offset in (64..64 + point_count * 16).step_by(4) {
            point_words.push(u32_at(&points, offset)?);
        }
        Ok(Self {
            grid,
            point_count,
            node_count,
            nodes,
            order,
            points: point_words,
            cauchy_scale: f32_at(&tree, 16)?,
            constant_density: f32_at(&tree, 20)?,
            min: [f32_at(&tree, 32)?, f32_at(&tree, 36)?, f32_at(&tree, 40)?],
            max: [f32_at(&tree, 44)?, f32_at(&tree, 48)?, f32_at(&tree, 52)?],
        })
    }

    fn cell(&self) -> [f64; 3] {
        [
            (self.max[0] - self.min[0]) as f64 / self.grid as f64,
            (self.max[1] - self.min[1]) as f64 / self.grid as f64,
            (self.max[2] - self.min[2]) as f64 / self.grid as f64,
        ]
    }

    fn node(&self, index: usize, slot: usize) -> f64 {
        self.nodes[index * 20 + slot] as f64
    }

    fn point_position(&self, index: usize, cell: [f64; 3]) -> [f64; 3] {
        let word = self.points[index * 4];
        let zword = self.points[index * 4 + 1];
        [
            self.min[0] as f64 + (((word & 0xffff) as f64) + 0.5) * cell[0],
            self.min[1] as f64 + (((word >> 16) & 0xffff) as f64 + 0.5) * cell[1],
            self.min[2] as f64 + (((zword & 0xffff) as f64) + 0.5) * cell[2],
        ]
    }

    fn point_mass(&self, index: usize, mode: u8, cell: [f64; 3]) -> f64 {
        match mode {
            0 => f32::from_bits(self.points[index * 4 + 2]) as f64,
            1 => f32::from_bits(self.points[index * 4 + 3]) as f64,
            _ => self.constant_density as f64 * cell[0] * cell[1] * cell[2],
        }
    }

    #[inline]
    fn add_point(out: &mut Sym6, observer: [f64; 3], source: [f64; 3], mass: f64) {
        let d = [
            observer[0] - source[0],
            observer[1] - source[1],
            observer[2] - source[2],
        ];
        let r2 = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).max(1e-24);
        let inv_r = 1.0 / r2.sqrt();
        let inv_r3 = inv_r * inv_r * inv_r;
        let inv_r5 = inv_r3 * inv_r * inv_r;
        let gm = crate::G * mass;
        let c = 3.0 * gm * inv_r5;
        let q = gm * inv_r3;
        out[0] += c * d[0] * d[0] - q;
        out[1] += c * d[1] * d[1] - q;
        out[2] += c * d[2] * d[2] - q;
        out[3] += c * d[0] * d[1];
        out[4] += c * d[0] * d[2];
        out[5] += c * d[1] * d[2];
    }

    /// Exact O(N) direct sum over every occupied voxel, i.e. the brute-force
    /// definition the Barnes-Hut walk approximates. It is the independent
    /// reference the asset tests below compare against.
    #[cfg(test)]
    pub fn direct_sum(&self, observer: [f64; 3], mode: u8) -> Sym6 {
        let cell = self.cell();
        let mut out = [0.0; 6];
        for point in 0..self.point_count {
            Self::add_point(
                &mut out,
                observer,
                self.point_position(point, cell),
                self.point_mass(point, mode, cell),
            );
        }
        out
    }

    pub fn tensor(&self, observer: [f64; 3], mode: u8, theta: f64) -> Sym6 {
        self.walk(observer, mode, theta, None)
    }

    /// The Barnes-Hut walk, optionally reporting how much of the tree it opens.
    pub fn walk(
        &self,
        observer: [f64; 3],
        mode: u8,
        theta: f64,
        mut stats: Option<&mut WalkStats>,
    ) -> Sym6 {
        let cell = self.cell();
        let mut out = [0.0; 6];
        let mut stack = vec![0usize];
        while let Some(node) = stack.pop() {
            if let Some(stats) = stats.as_deref_mut() {
                stats.nodes += 1;
            }
            let center = [self.node(node, 0), self.node(node, 1), self.node(node, 2)];
            let half = self.node(node, 3);
            let distance = ((observer[0] - center[0]).powi(2)
                + (observer[1] - center[1]).powi(2)
                + (observer[2] - center[2]).powi(2))
            .sqrt()
            .max(1e-12);
            let left = self.node(node, 6) as i32;
            let right = self.node(node, 7) as i32;
            let leaf_count = self.node(node, 15) as usize;
            if leaf_count > 0 {
                if let Some(stats) = stats.as_deref_mut() {
                    stats.leaves += 1;
                    stats.points += leaf_count as u64;
                }
                let first = self.node(node, 11) as usize;
                for slot in first..first + leaf_count {
                    let point = self.order[slot] as usize;
                    Self::add_point(
                        &mut out,
                        observer,
                        self.point_position(point, cell),
                        self.point_mass(point, mode, cell),
                    );
                }
            } else if half / distance < theta {
                if let Some(stats) = stats.as_deref_mut() {
                    stats.merged += 1;
                }
                let (mass, com_slot) = match mode {
                    0 => (self.node(node, 4), 8),
                    1 => (self.node(node, 5), 12),
                    _ => {
                        (
                            self.constant_density as f64
                                * cell[0]
                                * cell[1]
                                * cell[2]
                                * self.node(node, 19),
                            16,
                        )
                    }
                };
                Self::add_point(
                    &mut out,
                    observer,
                    [
                        self.node(node, com_slot),
                        self.node(node, com_slot + 1),
                        self.node(node, com_slot + 2),
                    ],
                    mass,
                );
            } else {
                if let Some(stats) = stats.as_deref_mut() {
                    stats.opened += 1;
                }
                if right >= 0 {
                    stack.push(right as usize);
                }
                if left >= 0 {
                    stack.push(left as usize);
                }
            }
        }
        out
    }
}
