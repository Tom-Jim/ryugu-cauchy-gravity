//! Runtime construction of the five independent browser solver inputs.
//!
//! The browser loads the GLB model first. Each solver requests only the density
//! description it needs and builds its own GPU input in memory.

use serde::Deserialize;
use std::collections::HashMap;

const GRID_SIZE: usize = 192;
const CONSTANT_DENSITY: f64 = 1190.0;
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
include!("binary.rs");
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

    pub fn face_count(&self) -> usize {
        self.triangles.len()
    }

    pub fn rtfp(&self, elliptic_toml: &str) -> Result<Vec<u8>, String> {
        let density = parse_density_text(elliptic_toml, "cauchy_elliptic.toml")?;
        let kernels = calibrate_kernels(&self.triangles, &density)?;
        build_mesh_pipeline(&self.triangles, &kernels)
    }

    pub fn carlson_alpha(&self, elliptic_toml: &str) -> Result<Vec<u8>, String> {
        let density = parse_density_text(elliptic_toml, "cauchy_elliptic.toml")?;
        let kernels = calibrate_kernels(&self.triangles, &density)?;
        build_mesh_pipeline(&self.triangles, &kernels)
    }

    pub fn mascon(
        &self,
        cauchy_toml: &str,
        elliptic_toml: &str,
    ) -> Result<(Vec<u8>, Vec<u8>), String> {
        let cauchy = parse_density_text(cauchy_toml, "cauchy.toml")?;
        let elliptic = parse_density_text(elliptic_toml, "cauchy_elliptic.toml")?;
        let cauchy_kernels = calibrate_kernels(&self.triangles, &cauchy)?;
        let elliptic_kernels = calibrate_kernels(&self.triangles, &elliptic)?;
        let target_mass = cauchy
            .mean_density_target
            .map(|rho| rho * mesh_enclosed_volume(&self.triangles))
            .unwrap_or(cauchy.total_mass_target);
        let constant_rho = cauchy.mean_density_target.unwrap_or(CONSTANT_DENSITY);
        let points = build_mascon(
            &self.triangles,
            &cauchy_kernels,
            &elliptic_kernels,
            target_mass,
            constant_rho,
        )?;
        let source = parse_mascon_bytes(&points, "runtime Mascon source")?;
        let (tree, _, _) = build_mascon_tree(&source);
        Ok((points, tree))
    }

    pub fn vtk_polydata(&self, scalars: &[f32]) -> Vec<u8> {
        let points = self
            .triangles
            .iter()
            .flat_map(|t| [t.a, t.b, t.c])
            .collect::<Vec<_>>();
        let triangle_count = self.triangles.len();
        let mut out = String::from(
            "<?xml version=\"1.0\"?>\n\
<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n\
  <PolyData>\n",
        );
        out.push_str(&format!(
            "    <Piece NumberOfPoints=\"{}\" NumberOfVerts=\"0\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"{}\">\n",
            points.len(), triangle_count
        ));
        out.push_str("      <Points>\n        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n");
        for point in points {
            out.push_str(&format!(
                "          {:.9} {:.9} {:.9}\n",
                point[0], point[1], point[2]
            ));
        }
        out.push_str("        </DataArray>\n      </Points>\n      <Polys>\n        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n");
        for index in 0..triangle_count {
            let base = index * 3;
            out.push_str(&format!("          {} {} {}\n", base, base + 1, base + 2));
        }
        out.push_str("        </DataArray>\n        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n");
        for index in 1..=triangle_count {
            out.push_str(&format!("          {}\n", index * 3));
        }
        out.push_str("        </DataArray>\n      </Polys>\n      <CellData Scalars=\"gravity_gradient\">\n        <DataArray type=\"Float32\" Name=\"gravity_gradient\" format=\"ascii\">\n");
        for index in 0..triangle_count {
            out.push_str(&format!(
                "          {:.9}\n",
                scalars.get(index).copied().unwrap_or(0.0)
            ));
        }
        out.push_str(
            "        </DataArray>\n      </CellData>\n    </Piece>\n  </PolyData>\n</VTKFile>\n",
        );
        out.into_bytes()
    }
}
