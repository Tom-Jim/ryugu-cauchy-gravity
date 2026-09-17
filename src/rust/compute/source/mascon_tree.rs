struct MasconPoint {
    x: u32,
    y: u32,
    z: u32,
    cauchy_mass: f64,
    elliptic_mass: f64,
}

struct MasconSource {
    grid: usize,
    count: usize,
    constant_density: f64,
    cell_volume: f64,
    cauchy_scale: f64,
    min: Vec3,
    max: Vec3,
    points: Vec<MasconPoint>,
}

fn parse_mascon_bytes(bytes: &[u8], label: &str) -> Result<MasconSource, String> {
    let fail = |why: &str| format!("{label}: {why}");
    if bytes.len() < 64 {
        return Err(fail("truncated mascon asset"));
    }
    let u32_at = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    let u16_at = |at: usize| u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap()) as u32;
    let f32_at = |at: usize| f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as f64;
    if u32_at(0) != 0x314d_5952 || u32_at(4) != 1 {
        return Err(fail("invalid mascon source asset"));
    }
    let grid = u32_at(8) as usize;
    let count = u32_at(12) as usize;
    if bytes.len() < 64 + count * 16 {
        return Err(fail("truncated mascon point table"));
    }
    let mut points = Vec::with_capacity(count);
    let mut offset = 64usize;
    for _ in 0..count {
        points.push(MasconPoint {
            x: u16_at(offset),
            y: u16_at(offset + 2),
            z: u16_at(offset + 4),
            cauchy_mass: f32_at(offset + 8),
            elliptic_mass: f32_at(offset + 12),
        });
        offset += 16;
    }
    Ok(MasconSource {
        grid,
        count,
        constant_density: f32_at(16),
        cell_volume: f32_at(20),
        cauchy_scale: f32_at(24),
        min: [f32_at(32), f32_at(36), f32_at(40)],
        max: [f32_at(44), f32_at(48), f32_at(52)],
        points,
    })
}

struct MasconNode {
    min: Vec3,
    max: Vec3,
    point_count: f64,
    mass_cauchy: f64,
    mass_elliptic: f64,
    com: Vec3,
    left: f64,
    right: f64,
    leaf_start: f64,
    leaf_count: f64,
}

#[allow(clippy::too_many_arguments)]
fn build_mascon_node(
    nodes: &mut Vec<MasconNode>,
    point_order: &mut Vec<u32>,
    lookup: &[i64],
    points: &[MasconPoint],
    grid: usize,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
    z0: usize,
    z1: usize,
) -> usize {
    let node_index = nodes.len();
    nodes.push(MasconNode {
        min: [x0 as f64, y0 as f64, z0 as f64],
        max: [x1 as f64, y1 as f64, z1 as f64],
        point_count: 0.0,
        mass_cauchy: 0.0,
        mass_elliptic: 0.0,
        com: [0.0; 3],
        left: -1.0,
        right: -1.0,
        leaf_start: 0.0,
        leaf_count: 0.0,
    });
    let mut count = 0.0f64;
    let mut sum_cauchy = 0.0f64;
    let mut sum_elliptic = 0.0f64;
    let (mut sum_x, mut sum_y, mut sum_z) = (0.0f64, 0.0f64, 0.0f64);
    let mut points_in_box: Vec<u32> = Vec::new();
    for z in z0..z1 {
        for y in y0..y1 {
            for x in x0..x1 {
                let point_index = lookup[(z * grid + y) * grid + x];
                if point_index < 0 {
                    continue;
                }
                let point = points[point_index as usize];
                count += 1.0;
                sum_cauchy += point.cauchy_mass;
                sum_elliptic += point.elliptic_mass;
                sum_x += point.x as f64 * point.cauchy_mass;
                sum_y += point.y as f64 * point.cauchy_mass;
                sum_z += point.z as f64 * point.cauchy_mass;
                points_in_box.push(point_index as u32);
            }
        }
    }
    nodes[node_index].point_count = count;
    nodes[node_index].mass_cauchy = sum_cauchy;
    nodes[node_index].mass_elliptic = sum_elliptic;
    if count > 0.0 && sum_cauchy.abs() > 1e-30 {
        nodes[node_index].com = [sum_x / sum_cauchy, sum_y / sum_cauchy, sum_z / sum_cauchy];
    } else if count > 0.0 {
        nodes[node_index].com = [
            (x0 + x1 - 1) as f64 * 0.5,
            (y0 + y1 - 1) as f64 * 0.5,
            (z0 + z1 - 1) as f64 * 0.5,
        ];
    }
    if count == 0.0 {
        return node_index;
    }
    let width_x = x1 - x0;
    let width_y = y1 - y0;
    let width_z = z1 - z0;
    if count <= MASCON_LEAF_POINTS || (width_x <= 1 && width_y <= 1 && width_z <= 1) {
        nodes[node_index].leaf_start = point_order.len() as f64;
        nodes[node_index].leaf_count = points_in_box.len() as f64;
        point_order.extend_from_slice(&points_in_box);
        return node_index;
    }
    let mut axis = 0usize;
    let mut width = width_x;
    if width_y > width {
        axis = 1;
        width = width_y;
    }
    if width_z > width {
        axis = 2;
    }
    let (left, right) = if axis == 0 {
        let mid = x0 + (width_x >> 1);
        (
            build_mascon_node(
                nodes,
                point_order,
                lookup,
                points,
                grid,
                x0,
                mid,
                y0,
                y1,
                z0,
                z1,
            ),
            build_mascon_node(
                nodes,
                point_order,
                lookup,
                points,
                grid,
                mid,
                x1,
                y0,
                y1,
                z0,
                z1,
            ),
        )
    } else if axis == 1 {
        let mid = y0 + (width_y >> 1);
        (
            build_mascon_node(
                nodes,
                point_order,
                lookup,
                points,
                grid,
                x0,
                x1,
                y0,
                mid,
                z0,
                z1,
            ),
            build_mascon_node(
                nodes,
                point_order,
                lookup,
                points,
                grid,
                x0,
                x1,
                mid,
                y1,
                z0,
                z1,
            ),
        )
    } else {
        let mid = z0 + (width_z >> 1);
        (
            build_mascon_node(
                nodes,
                point_order,
                lookup,
                points,
                grid,
                x0,
                x1,
                y0,
                y1,
                z0,
                mid,
            ),
            build_mascon_node(
                nodes,
                point_order,
                lookup,
                points,
                grid,
                x0,
                x1,
                y0,
                y1,
                mid,
                z1,
            ),
        )
    };
    nodes[node_index].left = left as f64;
    nodes[node_index].right = right as f64;
    if nodes[node_index].left >= 0.0 && nodes[left].mass_cauchy == 0.0 {
        nodes[node_index].left = -1.0;
    }
    if nodes[node_index].right >= 0.0 && nodes[right].mass_cauchy == 0.0 {
        nodes[node_index].right = -1.0;
    }
    node_index
}

fn build_mascon_tree(source: &MasconSource) -> (Vec<u8>, usize, usize) {
    let grid = source.grid;
    // The tree asset is SI: every node centre, half extent and centre of mass is
    // written in metres, so the WGSL kernel and the native mirror can read the
    // records verbatim. Only the leaf *indices* stay quantized.
    let cell = [
        (source.max[0] - source.min[0]) / grid as f64,
        (source.max[1] - source.min[1]) / grid as f64,
        (source.max[2] - source.min[2]) / grid as f64,
    ];
    let mut lookup = vec![-1i64; grid * grid * grid];
    for (index, point) in source.points.iter().enumerate() {
        lookup[(point.z as usize * grid + point.y as usize) * grid + point.x as usize] =
            index as i64;
    }
    let mut point_order: Vec<u32> = Vec::new();
    let mut nodes: Vec<MasconNode> = Vec::new();
    build_mascon_node(
        &mut nodes,
        &mut point_order,
        &lookup,
        &source.points,
        grid,
        0,
        grid,
        0,
        grid,
        0,
        grid,
    );

    let header_bytes = 64usize;
    let mut out = Out::new(header_bytes + nodes.len() * 64 + point_order.len() * 4);
    out.u32(0, 0x314d_5452); // RTM1
    out.u32(4, 1);
    out.u32(8, grid as u32);
    out.u32(12, source.count as u32);
    out.f32(16, source.cauchy_scale);
    out.f32(20, source.constant_density);
    out.f32(24, source.cell_volume);
    out.f32(28, 0.0);
    for axis in 0..3 {
        out.f32(32 + axis * 4, source.min[axis]);
        out.f32(44 + axis * 4, source.max[axis]);
    }
    out.u32(56, nodes.len() as u32);
    out.u32(60, 0);

    let mut offset = header_bytes;
    for node in &nodes {
        let center_indices = [
            (node.min[0] + node.max[0]) * 0.5,
            (node.min[1] + node.max[1]) * 0.5,
            (node.min[2] + node.max[2]) * 0.5,
        ];
        let center = [
            source.min[0] + (center_indices[0] + 0.5) * cell[0],
            source.min[1] + (center_indices[1] + 0.5) * cell[1],
            source.min[2] + (center_indices[2] + 0.5) * cell[2],
        ];
        // Largest metric half edge: an upper bound on the box's half extent, so
        // the Barnes-Hut test stays conservative while the node is opened only
        // as far as the accuracy target actually needs.
        let half = scalar_max(
            scalar_max(
                (node.max[0] - node.min[0]) * 0.5 * cell[0],
                (node.max[1] - node.min[1]) * 0.5 * cell[1],
            ),
            (node.max[2] - node.min[2]) * 0.5 * cell[2],
        );
        let com = [
            source.min[0] + (node.com[0] + 0.5) * cell[0],
            source.min[1] + (node.com[1] + 0.5) * cell[1],
            source.min[2] + (node.com[2] + 0.5) * cell[2],
        ];
        out.f32(offset, center[0]);
        out.f32(offset + 4, center[1]);
        out.f32(offset + 8, center[2]);
        out.f32(offset + 12, half);
        out.f32(offset + 16, node.mass_cauchy);
        out.f32(offset + 20, node.mass_elliptic);
        out.f32(offset + 24, node.left);
        out.f32(offset + 28, node.right);
        out.f32(offset + 32, com[0]);
        out.f32(offset + 36, com[1]);
        out.f32(offset + 40, com[2]);
        out.f32(offset + 44, node.leaf_start);
        out.f32(offset + 48, node.leaf_count);
        out.f32(offset + 52, node.point_count);
        offset += 64;
    }
    for point_index in &point_order {
        out.u32(offset, *point_index);
        offset += 4;
    }
    let nodes_len = nodes.len();
    let ordered_points = point_order.len();
    (out.bytes, nodes_len, ordered_points)
}
