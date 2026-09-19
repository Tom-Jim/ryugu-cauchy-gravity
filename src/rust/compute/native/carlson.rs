//! Validation-only density-jump model retained for regression comparisons.
//!
//! Production `Solver::Carlson` no longer uses this approximation. This module
//! builds a piecewise-constant density field as weighted jump surfaces only for
//! tests that quantify the legacy representation error against the
//! continuous-density path; it must not be presented as the browser solver.
//! The exact uniform term stays on the original mesh; a refined star-cone carries
//! only the density deviation, which limits the error from Ryugu's concavity.

use crate::analytic::Face;
use crate::density::{Density, DensityMode};
use crate::mass;
use crate::mesh::Mesh;

#[derive(Clone, Copy, Debug)]
pub struct Refine {
    /// Target within-slab density variation, relative to `|rho_ref|`.
    pub tol: f64,
    /// Hard cap on slabs per cone.
    pub max_slabs: usize,
    /// Axis samples used to place slab boundaries.
    pub samples: usize,
    /// Hard cap on emitted cone triangles.
    pub face_budget: usize,
}

impl Default for Refine {
    fn default() -> Self {
        Self {
            tol: 1e-3,
            max_slabs: 8,
            samples: 24,
            face_budget: 8_000_000,
        }
    }
}

/// Diagnostics from building the jump-surface list.
#[derive(Clone, Copy, Debug, Default)]
pub struct CarlsonStats {
    pub covered_volume: f64,
    pub signed_volume: f64,
    pub mesh_volume: f64,
    pub n_faces: usize,
    pub n_mesh_faces: usize,
    pub slabs: usize,
    pub worst_slab_variation: f64,
}

impl CarlsonStats {
    pub fn coverage(&self) -> f64 {
        if self.mesh_volume.abs() > 1e-30 {
            self.covered_volume / self.mesh_volume
        } else {
            1.0
        }
    }

    pub fn signed_ratio(&self) -> f64 {
        if self.covered_volume.abs() > 1e-30 {
            self.signed_volume.abs() / self.covered_volume
        } else {
            1.0
        }
    }
}

/// Original mesh at `rho_ref`, followed by the refined density-jump surfaces.
pub fn face_list(mesh: &Mesh, density: &Density, mode: DensityMode) -> (Vec<Face>, CarlsonStats) {
    let origin = mass::volume_centroid(mesh);
    let rho_ref = density.rho(&origin, mode);
    let (cone, mut stats) = star_faces(mesh, density, mode, origin, rho_ref, &Refine::default());
    let mut faces = crate::analytic::precompute(mesh);
    for face in &mut faces {
        face.weight = rho_ref;
    }
    stats.n_mesh_faces = faces.len();
    faces.extend(cone);
    (faces, stats)
}

include!("carlson/surface.rs");
include!("carlson/refinement.rs");
include!("carlson/geometry.rs");
include!("carlson/reference.rs");
