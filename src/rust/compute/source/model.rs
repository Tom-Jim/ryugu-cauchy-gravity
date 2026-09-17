type Vec3 = [f64; 3];

#[inline]
fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn scale(a: Vec3, k: f64) -> Vec3 {
    [a[0] * k, a[1] * k, a[2] * k]
}

#[inline]
fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: Vec3) -> f64 {
    dot(a, a).sqrt()
}

#[derive(Clone, Copy)]
struct Triangle {
    a: Vec3,
    b: Vec3,
    c: Vec3,
    center: Vec3,
    normal: Vec3,
}

fn make_triangle(a: Vec3, b: Vec3, c: Vec3) -> Triangle {
    let center = scale(add(add(a, b), c), 1.0 / 3.0);
    let raw_normal = cross(sub(b, a), sub(c, a));
    let length = norm(raw_normal).max(1e-20);
    Triangle {
        a,
        b,
        c,
        center,
        normal: scale(raw_normal, 1.0 / length),
    }
}

fn parse_glb_model(glb: &[u8]) -> Result<Vec<Triangle>, String> {
    let model = gltf::Gltf::from_slice(glb).map_err(|error| format!("invalid GLB: {error}"))?;
    let blob = model.blob.as_deref().ok_or("GLB has no binary buffer")?;
    let primitive = model
        .meshes()
        .flat_map(|mesh| mesh.primitives())
        .filter(|primitive| primitive.mode() == gltf::mesh::Mode::Triangles)
        .max_by_key(|primitive| primitive.indices().map_or(0, |indices| indices.count()))
        .ok_or("GLB contains no indexed triangle primitive")?;
    let reader = primitive.reader(|buffer| match buffer.source() {
        gltf::buffer::Source::Bin => Some(blob),
        gltf::buffer::Source::Uri(_) => None,
    });
    let vertices: Vec<Vec3> = reader
        .read_positions()
        .ok_or("GLB triangle primitive has no POSITION attribute")?
        .map(|point| {
            [
                point[0] as f64 * KM_TO_M,
                point[1] as f64 * KM_TO_M,
                point[2] as f64 * KM_TO_M,
            ]
        })
        .collect();
    let indices: Vec<u32> = reader
        .read_indices()
        .ok_or("GLB triangle primitive has no index accessor")?
        .into_u32()
        .collect();
    if indices.len() % 3 != 0 {
        return Err("GLB triangle index count is not divisible by three".into());
    }
    let mut triangles = Vec::with_capacity(indices.len() / 3);
    for face in indices.chunks_exact(3) {
        let point = |index: u32| {
            vertices
                .get(index as usize)
                .copied()
                .ok_or_else(|| format!("GLB index {index} exceeds {} vertices", vertices.len()))
        };
        triangles.push(make_triangle(point(face[0])?, point(face[1])?, point(face[2])?));
    }
    if triangles.len() != FACE_COUNT as usize {
        return Err(format!(
            "expected {FACE_COUNT} model faces, found {}",
            triangles.len()
        ));
    }
    Ok(triangles)
}
