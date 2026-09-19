type Vec3 = [f64; 3];

#[inline]
fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
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
    normal: Vec3,
}

fn make_triangle(a: Vec3, b: Vec3, c: Vec3) -> Triangle {
    let raw_normal = cross(sub(b, a), sub(c, a));
    let length = norm(raw_normal).max(1e-20);
    Triangle {
        a,
        b,
        c,
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
    if !indices.len().is_multiple_of(3) {
        return Err("GLB triangle index count is not divisible by three".into());
    }
    let (faces, remainder) = indices.as_chunks::<3>();
    debug_assert!(remainder.is_empty());
    let mut signed_six_volume = 0.0f64;
    for &[i0, i1, i2] in faces {
        let a = vertices
            .get(i0 as usize)
            .copied()
            .ok_or_else(|| format!("GLB index {i0} exceeds {} vertices", vertices.len()))?;
        let b = vertices
            .get(i1 as usize)
            .copied()
            .ok_or_else(|| format!("GLB index {i1} exceeds {} vertices", vertices.len()))?;
        let c = vertices
            .get(i2 as usize)
            .copied()
            .ok_or_else(|| format!("GLB index {i2} exceeds {} vertices", vertices.len()))?;
        signed_six_volume += dot(a, cross(b, c));
    }
    let reverse_winding = signed_six_volume < 0.0;
    let mut triangles = Vec::with_capacity(indices.len() / 3);
    for &[i0, i1, i2] in faces {
        let point = |index: u32| {
            vertices
                .get(index as usize)
                .copied()
                .ok_or_else(|| format!("GLB index {index} exceeds {} vertices", vertices.len()))
        };
        let a = point(i0)?;
        let b = point(i1)?;
        let c = point(i2)?;
        triangles.push(if reverse_winding {
            make_triangle(a, c, b)
        } else {
            make_triangle(a, b, c)
        });
    }
    if triangles.len() != FACE_COUNT as usize {
        return Err(format!(
            "expected {FACE_COUNT} model faces, found {}",
            triangles.len()
        ));
    }
    Ok(triangles)
}
