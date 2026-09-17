struct FaceAsset {
    count: usize,
    records: Range<usize>,
}

fn parse_face_asset(
    bytes: &[u8],
    magic: u32,
    expected: Option<usize>,
) -> Result<FaceAsset, String> {
    if bytes.len() < 16 {
        return Err("truncated face asset".into());
    }
    if u32_at(bytes, 0) != magic {
        return Err("invalid face asset magic".into());
    }
    let count = u32_at(bytes, 8) as usize;
    if expected.is_some_and(|want| want != count) || bytes.len() < 16 + count * 64 {
        return Err(format!("invalid face asset count {count}"));
    }
    Ok(FaceAsset {
        count,
        records: 16..16 + count * 64,
    })
}

/// Byte ranges of the RTP1 v2 mesh pipeline sections.
struct MeshLayout {
    face_count: usize,
    dir_count: usize,
    kernel_count: usize,
    positions: Range<usize>,
    indices: Range<usize>,
    bounds: Range<usize>,
    links: Range<usize>,
    directions: Range<usize>,
    kernels: Range<usize>,
    faces: Range<usize>,
}

fn parse_mesh_pipeline(bytes: &[u8]) -> Result<MeshLayout, String> {
    if bytes.len() < 32 || u32_at(bytes, 0) != 0x3150_5452 || u32_at(bytes, 4) != 2 {
        return Err("invalid ray-pipeline asset".into());
    }
    let vertex_count = u32_at(bytes, 8) as usize;
    let face_count = u32_at(bytes, 12) as usize;
    let dir_count = u32_at(bytes, 16) as usize;
    let kernel_count = u32_at(bytes, 20) as usize;
    let node_count = u32_at(bytes, 24) as usize;
    if face_count != FACE_COUNT {
        return Err("invalid ray-pipeline face count".into());
    }
    let mut offset = 32usize;
    let mut take = |bytes: usize| {
        let range = offset..offset + bytes;
        offset += bytes;
        range
    };
    let positions = take(vertex_count * 16);
    let indices = take(face_count * 3 * 4);
    let bounds = take(node_count * 2 * 16);
    let links = take(node_count * 4 * 4);
    let directions = take(dir_count * 16);
    let kernels = take(kernel_count * 32);
    // `build_face_records` writes four `vec4<f32>` rows per triangle (A, B, C,
    // outward unit normal) — 64 bytes — and `observers.wgsl` indexes the binding
    // as `array<vec4<f32>>` with `base = face * 4`. A 16-byte stride truncated
    // the section to a quarter of its length, so every face past the first
    // 49 152 read its observation point from out of bounds and the ray path
    // returned a plausible-looking but wrong tensor.
    let faces = take(face_count * 64);
    if bytes.len() < offset {
        return Err("truncated ray-pipeline asset".into());
    }
    Ok(MeshLayout {
        face_count,
        dir_count,
        kernel_count,
        positions,
        indices,
        bounds,
        links,
        directions,
        kernels,
        faces,
    })
}

struct MasconTree {
    grid: u32,
    constant_density: f32,
    min: [f64; 3],
    max: [f64; 3],
    node_count: usize,
    nodes: Range<usize>,
    order: Range<usize>,
}

fn parse_mascon_tree(bytes: &[u8]) -> Result<MasconTree, String> {
    if bytes.len() < 64 || u32_at(bytes, 0) != 0x314d_5452 || u32_at(bytes, 4) != 2 {
        return Err("invalid Mascon tree asset".into());
    }
    let grid = u32_at(bytes, 8);
    let point_count = u32_at(bytes, 12) as usize;
    let node_count = u32_at(bytes, 56) as usize;
    let node_offset = 64usize;
    let order_offset = node_count
        .checked_mul(80)
        .and_then(|node_bytes| node_offset.checked_add(node_bytes))
        .ok_or("Mascon tree size overflow")?;
    let order_end = point_count
        .checked_mul(4)
        .and_then(|order_bytes| order_offset.checked_add(order_bytes))
        .ok_or("Mascon point-order size overflow")?;
    if bytes.len() < order_end {
        return Err("truncated Mascon tree asset".into());
    }
    Ok(MasconTree {
        grid,
        constant_density: f32_at(bytes, 20),
        min: [
            f32_at(bytes, 32) as f64,
            f32_at(bytes, 36) as f64,
            f32_at(bytes, 40) as f64,
        ],
        max: [
            f32_at(bytes, 44) as f64,
            f32_at(bytes, 48) as f64,
            f32_at(bytes, 52) as f64,
        ],
        node_count,
        nodes: node_offset..order_offset,
        order: order_offset..order_end,
    })
}

// ---------------------------------------------------------------------------
// GPU resources
// ---------------------------------------------------------------------------
