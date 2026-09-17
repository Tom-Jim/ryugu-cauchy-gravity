/// Little-endian writer over a fixed-size buffer.
struct Out {
    bytes: Vec<u8>,
}

impl Out {
    fn new(len: usize) -> Self {
        Self {
            bytes: vec![0u8; len],
        }
    }

    fn u16(&mut self, at: usize, value: u16) {
        self.bytes[at..at + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn u32(&mut self, at: usize, value: u32) {
        self.bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn f32(&mut self, at: usize, value: f64) {
        self.bytes[at..at + 4].copy_from_slice(&(value as f32).to_le_bytes());
    }
}

/// One Carlson triangle record: three corners, then `(unit normal, density jump)`.
fn write_carlson_face(
    out: &mut Out,
    offset: usize,
    a: Vec3,
    b: Vec3,
    c: Vec3,
    weight: f64,
) -> usize {
    let normal = cross(sub(b, a), sub(c, a));
    let length = norm(normal);
    if length <= 1e-30 {
        return offset;
    }
    let unit_normal = scale(normal, 1.0 / length);
    let mut at = offset;
    for point in [a, b, c] {
        out.f32(at, point[0]);
        out.f32(at + 4, point[1]);
        out.f32(at + 8, point[2]);
        out.f32(at + 12, 0.0);
        at += 16;
    }
    out.f32(at, unit_normal[0]);
    out.f32(at + 4, unit_normal[1]);
    out.f32(at + 8, unit_normal[2]);
    out.f32(at + 12, weight);
    at + 16
}

#[allow(clippy::too_many_arguments)]
fn emit_tetra_face(
    out: &mut Out,
    offset: usize,
    tetra_center: Vec3,
    a: Vec3,
    b: Vec3,
    c: Vec3,
    weight: f64,
) -> usize {
    let face_center = scale(add(add(a, b), c), 1.0 / 3.0);
    let normal = cross(sub(b, a), sub(c, a));
    // The surface integral expects outward normals. The old branch flipped an
    // already-outward face, reversing every density-jump contribution.
    let (b, c) = if dot(normal, sub(face_center, tetra_center)) < 0.0 {
        (c, b)
    } else {
        (b, c)
    };
    write_carlson_face(out, offset, a, b, c, weight)
}

#[derive(Clone, Copy, PartialEq)]
enum CarlsonMode {
    Cauchy,
    Constant,
}

fn build_carlson_faces(
    triangles: &[Triangle],
    kernels: &[Kernel],
    mode: CarlsonMode,
    constant_density: f64,
) -> Result<Vec<u8>, String> {
    let origin = volume_centroid(triangles);
    let reference_density = match mode {
        CarlsonMode::Constant => constant_density,
        CarlsonMode::Cauchy => density_at(kernels, origin),
    };
    let cone_faces = if mode == CarlsonMode::Constant { 0 } else { 4 };
    let count = triangles.len() * (1 + cone_faces);
    let mut out = Out::new(16 + count * 64);
    out.u32(0, 0x3143_5952); // RYC1
    out.u32(4, 1);
    out.u32(8, count as u32);
    out.u32(12, if mode == CarlsonMode::Constant { 1 } else { 0 });
    let mut offset = 16;
    for triangle in triangles {
        offset = write_carlson_face(
            &mut out,
            offset,
            triangle.a,
            triangle.b,
            triangle.c,
            reference_density,
        );
    }
    if mode != CarlsonMode::Constant {
        for triangle in triangles {
            let tetra_center = scale(
                add(add(add(origin, triangle.a), triangle.b), triangle.c),
                0.25,
            );
            let weight = density_at(kernels, tetra_center) - reference_density;
            offset = emit_tetra_face(
                &mut out,
                offset,
                tetra_center,
                origin,
                triangle.b,
                triangle.c,
                weight,
            );
            offset = emit_tetra_face(
                &mut out,
                offset,
                tetra_center,
                origin,
                triangle.c,
                triangle.a,
                weight,
            );
            offset = emit_tetra_face(
                &mut out,
                offset,
                tetra_center,
                origin,
                triangle.a,
                triangle.b,
                weight,
            );
            offset = emit_tetra_face(
                &mut out,
                offset,
                tetra_center,
                triangle.a,
                triangle.b,
                triangle.c,
                weight,
            );
        }
    }
    if offset != out.bytes.len() {
        return Err(format!(
            "Carlson face writer ended at {offset}, expected {}",
            out.bytes.len()
        ));
    }
    Ok(out.bytes)
}
