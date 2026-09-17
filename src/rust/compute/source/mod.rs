//! Runtime construction of the five independent browser solver inputs.
//!
//! The browser loads the GLB model first. Each solver requests only the density
//! description it needs and builds its own GPU input in memory.

use serde::Deserialize;
use std::collections::HashMap;

const FACE_COUNT: u32 = 196_608;
const GRID_SIZE: usize = 192;
const CONSTANT_DENSITY: f64 = 1190.0;
const CAUCHY_TARGET_MASS: f64 = 4.50e11;
const KM_TO_M: f64 = 1000.0;
const EPS: f64 = 1e-12;

const LEAF_SIZE: usize = 8;
const MAX_BVH_DEPTH: usize = 40;
const BVH_PAD: f64 = 1e-4;
const MASCON_LEAF_POINTS: f64 = 8.0;
const DIRECTION_TOTAL: usize = 288;
const PI: f64 = std::f64::consts::PI;

include!("model.rs");
include!("density.rs");
include!("inside.rs");
include!("carlson.rs");
include!("mascon_grid.rs");
include!("mesh_pipeline.rs");
include!("mascon_tree.rs");

pub struct RuntimeSource {
    triangles: Vec<Triangle>,
}

impl RuntimeSource {
    pub fn from_glb(glb: &[u8]) -> Result<Self, String> {
        Ok(Self {
            triangles: parse_glb_model(glb)?,
        })
    }

    pub fn geometry(&self) -> Vec<u8> {
        write_werner(&self.triangles)
    }

    pub fn rtfp(&self, cauchy_toml: &str) -> Result<Vec<u8>, String> {
        let kernels = normalize_kernels(
            &self.triangles,
            &parse_kernels_text(cauchy_toml, "cauchy.toml")?,
            CAUCHY_TARGET_MASS,
        );
        Ok(build_mesh_pipeline(&self.triangles, &kernels))
    }

    pub fn carlson(&self, cauchy_toml: Option<&str>) -> Result<Vec<u8>, String> {
        let (kernels, mode) = match cauchy_toml {
            Some(text) => (
                normalize_kernels(
                    &self.triangles,
                    &parse_kernels_text(text, "cauchy.toml")?,
                    CAUCHY_TARGET_MASS,
                ),
                CarlsonMode::Cauchy,
            ),
            None => (Vec::new(), CarlsonMode::Constant),
        };
        build_carlson_faces(&self.triangles, &kernels, mode, CONSTANT_DENSITY)
    }

    pub fn carlson_alpha(&self, elliptic_toml: &str) -> Result<Vec<u8>, String> {
        let kernels = parse_kernels_text(elliptic_toml, "cauchy_elliptic.toml")?;
        Ok(build_mesh_pipeline(&self.triangles, &kernels))
    }

    pub fn mascon(
        &self,
        cauchy_toml: &str,
        elliptic_toml: &str,
    ) -> Result<(Vec<u8>, Vec<u8>), String> {
        let cauchy = parse_kernels_text(cauchy_toml, "cauchy.toml")?;
        let elliptic = parse_kernels_text(elliptic_toml, "cauchy_elliptic.toml")?;
        let points = build_mascon(&self.triangles, &cauchy, &elliptic)?;
        let source = parse_mascon_bytes(&points, "runtime Mascon source")?;
        let (tree, _, _) = build_mascon_tree(&source);
        Ok((points, tree))
    }
}
