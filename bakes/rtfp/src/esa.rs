//! Bindings to the vendored ESA polyhedral-gravity library, which supplies the
//! **analytic** half of the split: the exact uniform-density gravity-gradient
//! tensor of the mesh, in closed form, for every observation point.
//!
//! Used by `--selftest` only: the bake's own tensor is the WGSL surface integral
//! in `shaders/analytic.wgsl`, and this library is the independent reference it
//! is checked against.

use crate::mesh::Mesh;

#[repr(C)]
pub struct EsaPgHandle {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn esa_pg_create(
        vertices_xyz: *const f64,
        vertex_count: usize,
        faces: *const u32,
        face_count: usize,
        density_kg_m3: f64,
        check_mesh: i32,
    ) -> *mut EsaPgHandle;
    fn esa_pg_destroy(handle: *mut EsaPgHandle);
    fn esa_pg_eval(
        handle: *mut EsaPgHandle,
        position_m: *const f64,
        out_potential: *mut f64,
        out_acceleration: *mut f64,
        out_h6: *mut f64,
    ) -> i32;
}

/// Owned handle; destroyed on drop.
pub struct Evaluable {
    handle: *mut EsaPgHandle,
}

unsafe impl Send for Evaluable {}

impl Evaluable {
    pub fn new(mesh: &Mesh, density_kg_m3: f64) -> Result<Self, String> {
        let handle = unsafe {
            esa_pg_create(
                mesh.xyz.as_ptr(),
                mesh.vertex_count(),
                mesh.faces.as_ptr(),
                mesh.face_count(),
                density_kg_m3,
                0,
            )
        };
        if handle.is_null() {
            return Err("esa_pg_create failed".into());
        }
        Ok(Self { handle })
    }

    /// `[Vxx, Vyy, Vzz, Vxy, Vxz, Vyz]` at `p` (meters).
    pub fn hessian6(&self, p: [f64; 3]) -> Result<[f64; 6], String> {
        let mut h6 = [0.0f64; 6];
        let rc = unsafe {
            esa_pg_eval(
                self.handle,
                p.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                h6.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(format!("esa_pg_eval failed at {p:?}: rc={rc}"));
        }
        Ok(h6)
    }
}

impl Drop for Evaluable {
    fn drop(&mut self) {
        unsafe { esa_pg_destroy(self.handle) }
    }
}
