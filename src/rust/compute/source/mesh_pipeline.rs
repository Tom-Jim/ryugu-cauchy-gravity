fn build_face_records(triangles: &[Triangle], weight: f64) -> Vec<f32> {
    let mut out = vec![0.0f32; triangles.len() * 16];
    for (face, triangle) in triangles.iter().enumerate() {
        let base = face * 16;
        let values = [
            triangle.a[0],
            triangle.a[1],
            triangle.a[2],
            0.0,
            triangle.b[0],
            triangle.b[1],
            triangle.b[2],
            0.0,
            triangle.c[0],
            triangle.c[1],
            triangle.c[2],
            0.0,
            triangle.normal[0],
            triangle.normal[1],
            triangle.normal[2],
            weight,
        ];
        for (slot, value) in values.iter().enumerate() {
            out[base + slot] = *value as f32;
        }
    }
    out
}

fn scalar_min(a: f64, b: f64) -> f64 {
    a.min(b)
}

fn scalar_max(a: f64, b: f64) -> f64 {
    a.max(b)
}

fn scalar_round(value: f64) -> f64 {
    (value + 0.5).floor()
}

fn vertex_key(point: Vec3) -> (u32, u32, u32) {
    (
        (point[0] as f32).to_bits(),
        (point[1] as f32).to_bits(),
        (point[2] as f32).to_bits(),
    )
}

/// `writeWerner`: the closed-form face list, identical for Werner, RT-FP's
/// constant reference and CarlsonAlpha's constant reference.
fn write_werner(triangles: &[Triangle]) -> Vec<u8> {
    let records = build_face_records(triangles, 1.0);
    let mut out = Out::new(16 + triangles.len() * 64);
    out.u32(0, 0x3152_5752); // RWR1
    out.u32(4, 1);
    out.u32(8, triangles.len() as u32);
    for (index, value) in records.iter().enumerate() {
        out.f32(16 + index * 4, *value as f64);
    }
    out.bytes
}
#[derive(Clone, Copy)]
struct BvhBounds {
    low: Vec3,
    high: Vec3,
    centroid: Vec3,
}
struct BvhNode {
    min: Vec3,
    max: Vec3,
    left: u32,
    right: u32,
    start: u32,
    count: u32,
}

struct Bvh {
    nodes: Vec<BvhNode>,
    order: Vec<u32>,
}

impl Bvh {
    fn build(&mut self, bounds: &[BvhBounds], start: usize, count: usize, depth: usize) -> usize {
        let index = self.nodes.len();
        self.nodes.push(BvhNode {
            min: [f64::INFINITY; 3],
            max: [f64::NEG_INFINITY; 3],
            left: 0,
            right: 0,
            start: 0,
            count: 0,
        });
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        let mut centroid_min = [f64::INFINITY; 3];
        let mut centroid_max = [f64::NEG_INFINITY; 3];
        for slot in start..start + count {
            let triangle = bounds[self.order[slot] as usize];
            for axis in 0..3 {
                min[axis] = scalar_min(min[axis], triangle.low[axis]);
                max[axis] = scalar_max(max[axis], triangle.high[axis]);
                centroid_min[axis] = scalar_min(centroid_min[axis], triangle.centroid[axis]);
                centroid_max[axis] = scalar_max(centroid_max[axis], triangle.centroid[axis]);
            }
        }
        self.nodes[index].min = min;
        self.nodes[index].max = max;
        if count <= LEAF_SIZE || depth >= MAX_BVH_DEPTH {
            self.nodes[index].start = start as u32;
            self.nodes[index].count = count as u32;
            return index;
        }
        let mut axis = 0usize;
        for candidate in 1..3 {
            if centroid_max[candidate] - centroid_min[candidate]
                > centroid_max[axis] - centroid_min[axis]
            {
                axis = candidate;
            }
        }
        let mut slice: Vec<u32> = self.order[start..start + count].to_vec();
        slice.sort_by(|a, b| {
            let delta = bounds[*a as usize].centroid[axis] - bounds[*b as usize].centroid[axis];
            delta.partial_cmp(&0.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        self.order[start..start + count].copy_from_slice(&slice);
        let left_count = count >> 1;
        let left = self.build(bounds, start, left_count, depth + 1);
        let right = self.build(bounds, start + left_count, count - left_count, depth + 1);
        self.nodes[index].left = left as u32;
        self.nodes[index].right = right as u32;
        index
    }
}

fn build_bvh(triangles: &[Triangle]) -> Bvh {
    let bounds: Vec<BvhBounds> = triangles
        .iter()
        .map(|triangle| {
            let low = [
                scalar_min(scalar_min(triangle.a[0], triangle.b[0]), triangle.c[0]) - BVH_PAD,
                scalar_min(scalar_min(triangle.a[1], triangle.b[1]), triangle.c[1]) - BVH_PAD,
                scalar_min(scalar_min(triangle.a[2], triangle.b[2]), triangle.c[2]) - BVH_PAD,
            ];
            let high = [
                scalar_max(scalar_max(triangle.a[0], triangle.b[0]), triangle.c[0]) + BVH_PAD,
                scalar_max(scalar_max(triangle.a[1], triangle.b[1]), triangle.c[1]) + BVH_PAD,
                scalar_max(scalar_max(triangle.a[2], triangle.b[2]), triangle.c[2]) + BVH_PAD,
            ];
            BvhBounds {
                low,
                high,
                centroid: [
                    (low[0] + high[0]) * 0.5,
                    (low[1] + high[1]) * 0.5,
                    (low[2] + high[2]) * 0.5,
                ],
            }
        })
        .collect();
    let mut bvh = Bvh {
        nodes: Vec::new(),
        order: (0..triangles.len() as u32).collect(),
    };
    if !bounds.is_empty() {
        bvh.build(&bounds, 0, bounds.len(), 0);
    }
    bvh
}

/// Gauss-Legendre nodes and weights on `[-1, 1]` by Newton iteration.
fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0f64; n];
    let mut weights = vec![0.0f64; n];
    for i in 0..n {
        let mut x = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut derivative = 0.0f64;
        for _ in 0..100 {
            let mut p1 = 1.0f64;
            let mut p2 = 0.0f64;
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j as f64 + 1.0) * x * p2 - j as f64 * p3) / (j as f64 + 1.0);
            }
            derivative = n as f64 * (x * p1 - p2) / (x * x - 1.0);
            let step = p1 / derivative;
            x -= step;
            if step.abs() < 1e-15 {
                break;
            }
        }
        nodes[i] = x;
        weights[i] = 2.0 / ((1.0 - x * x) * derivative * derivative);
    }
    (nodes, weights)
}

/// `(x, y, z, weight)` direction table over the sphere.
fn quadrature_directions(total: usize) -> Vec<f32> {
    let n_theta = 2.max(scalar_round((scalar_max(total as f64, 8.0) / 2.0).sqrt()) as usize);
    let n_phi = 2 * n_theta;
    let (nodes, weights) = gauss_legendre(n_theta);
    let mut out = vec![0.0f32; n_theta * n_phi * 4];
    let mut cursor = 0usize;
    for i in 0..n_theta {
        let radius = scalar_max(0.0, 1.0 - nodes[i] * nodes[i]).sqrt();
        for j in 0..n_phi {
            let phi = 2.0 * PI * (j as f64 + 0.5) / n_phi as f64;
            out[cursor] = (radius * phi.cos()) as f32;
            out[cursor + 1] = (radius * phi.sin()) as f32;
            out[cursor + 2] = nodes[i] as f32;
            out[cursor + 3] = (weights[i] * 2.0 * PI / n_phi as f64) as f32;
            cursor += 4;
        }
    }
    out
}

/// `buildMeshPipeline`: positions, indices, BVH bounds, BVH links, directions,
/// kernels and face records behind an RTP1 v2 header.
fn build_mesh_pipeline(triangles: &[Triangle], kernels: &[Kernel]) -> Vec<u8> {
    // The source asset is indexed by exact coordinates; rebuild that compact map
    // so the ray shader reads a real indexed mesh rather than a triangle soup.
    let mut map: HashMap<(u32, u32, u32), u32> = HashMap::new();
    let mut positions: Vec<Vec3> = Vec::new();
    for triangle in triangles {
        for point in [triangle.a, triangle.b, triangle.c] {
            let key = vertex_key(point);
            if !map.contains_key(&key) {
                map.insert(key, positions.len() as u32);
                positions.push(point);
            }
        }
    }

    let bvh = build_bvh(triangles);
    // The ray shader indexes the triangle buffer with a leaf's `start`/`count`,
    // which are positions in `bvh.order`, so the index buffer has to be written
    // in BVH leaf order — exactly what `gpu.rs::mesh_buffers` does for the native
    // bake. Writing it in mesh order made every leaf trace a different triangle
    // than the one it bounds: the traced crossings were wrong, so the remainder
    // integral came out a few percent low (and the general-alpha one lower
    // still).
    let mut indices = vec![0u32; triangles.len() * 3];
    for (slot, face) in bvh.order.iter().enumerate() {
        let triangle = &triangles[*face as usize];
        for (offset, point) in [triangle.a, triangle.b, triangle.c].into_iter().enumerate() {
            indices[slot * 3 + offset] = map[&vertex_key(point)];
        }
    }
    let node_count = bvh.nodes.len();
    let dirs = quadrature_directions(DIRECTION_TOTAL);
    let dir_count = dirs.len() / 4;
    let header_bytes = 32usize;
    let mut out = Out::new(
        header_bytes
            + positions.len() * 16
            + indices.len() * 4
            + node_count * 32
            + node_count * 16
            + dirs.len() * 4
            + kernels.len() * 32
            + triangles.len() * 64,
    );
    out.u32(0, 0x3150_5452); // RTP1
    out.u32(4, 2);
    out.u32(8, positions.len() as u32);
    out.u32(12, triangles.len() as u32);
    out.u32(16, dir_count as u32);
    out.u32(20, kernels.len() as u32);
    out.u32(24, node_count as u32);
    out.u32(28, 0);

    let mut offset = header_bytes;
    for point in &positions {
        out.f32(offset, point[0]);
        out.f32(offset + 4, point[1]);
        out.f32(offset + 8, point[2]);
        offset += 16;
    }
    for value in &indices {
        out.bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        offset += 4;
    }
    for node in &bvh.nodes {
        out.f32(offset, node.min[0]);
        out.f32(offset + 4, node.min[1]);
        out.f32(offset + 8, node.min[2]);
        offset += 16;
        out.f32(offset, node.max[0]);
        out.f32(offset + 4, node.max[1]);
        out.f32(offset + 8, node.max[2]);
        offset += 16;
    }
    for node in &bvh.nodes {
        out.u32(offset, node.left);
        out.u32(offset + 4, node.right);
        out.u32(offset + 8, node.start);
        out.u32(offset + 12, node.count);
        offset += 16;
    }
    for value in &dirs {
        out.bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        offset += 4;
    }
    for kernel in kernels {
        out.f32(offset, kernel.c[0]);
        out.f32(offset + 4, kernel.c[1]);
        out.f32(offset + 8, kernel.c[2]);
        out.f32(offset + 12, kernel.sigma);
        out.f32(offset + 16, kernel.w);
        out.f32(offset + 20, kernel.alpha);
        offset += 32;
    }
    for (index, value) in build_face_records(triangles, 1.0).iter().enumerate() {
        out.bytes[offset + index * 4..offset + index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    out.bytes
}
