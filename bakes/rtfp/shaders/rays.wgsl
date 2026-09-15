// Software BVH ray tracing for the RT-FP deviation integral.
//
// One invocation per (observation point, direction): walk the flat BVH built by
// `src/bvh.rs`, collect the first `MAX_HITS` triangle crossings along `x − R u`,
// and emit the visible `(r₀, r₁)` intervals the remainder shader integrates.
// A second entry point, `inside_probe`, runs one `+X` ray per point to get the
// crossing parity that decides whether the first interval starts at the point
// itself. Both entry points share one bind group; the probe pass is dispatched
// first, so the parity is already in memory when `rays` reads it.

struct Globals {
    n_points: u32,
    n_dirs: u32,
    n_tris: u32,
    // Workgroups dispatched along `x` in the `rays` pass; `y` is the stride
    // that keeps each dispatch dimension under the 65535 cap.
    grid_x: u32,
    t_min: f32,
    t_max: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> g: Globals;
// Vertex positions, xyz + pad.
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
// Triangle indices, already permuted into BVH leaf order (3 per triangle).
@group(0) @binding(2) var<storage, read> indices: array<u32>;
// Two rows per node: bounds min, then bounds max.
@group(0) @binding(3) var<storage, read> nodes: array<vec4<f32>>;
// One row per node: (left, right, first_triangle, triangle_count); count > 0 = leaf.
@group(0) @binding(4) var<storage, read> links: array<vec4<u32>>;
// Observation points, xyz + pad.
@group(0) @binding(5) var<storage, read> points: array<vec4<f32>>;
// Directions, xyz + quadrature weight.
@group(0) @binding(6) var<storage, read> dirs: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> out_counts: array<u32>;
@group(0) @binding(8) var<storage, read_write> out_ivals: array<vec2<f32>>;
@group(0) @binding(9) var<storage, read_write> out_inside: array<u32>;

const MAX_HITS: u32 = 8u;
const MAX_INTERVALS: u32 = 4u;
const STACK: u32 = 64u;

/// Möller–Trumbore. Returns the crossing distance, or −1 when the ray misses.
fn tri_hit(o: vec3<f32>, d: vec3<f32>, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>, t_lo: f32) -> f32 {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let pv = cross(d, e2);
    let det = dot(e1, pv);
    if (abs(det) < 1e-16) {
        return -1.0;
    }
    let inv = 1.0 / det;
    let tv = o - v0;
    let u = dot(tv, pv) * inv;
    if (u < 0.0 || u > 1.0) {
        return -1.0;
    }
    let qv = cross(tv, e1);
    let v = dot(d, qv) * inv;
    if (v < 0.0 || u + v > 1.0) {
        return -1.0;
    }
    let t = dot(e2, qv) * inv;
    if (t > t_lo) {
        return t;
    }
    return -1.0;
}

fn slab_hit(node: u32, o: vec3<f32>, inv_d: vec3<f32>, t_lo: f32, t_hi: f32) -> bool {
    let lo = nodes[2u * node].xyz;
    let hi = nodes[2u * node + 1u].xyz;
    let a = (lo - o) * inv_d;
    let b = (hi - o) * inv_d;
    let t0 = min(a, b);
    let t1 = max(a, b);
    let near = max(max(t0.x, t0.y), max(t0.z, t_lo));
    let far = min(min(t1.x, t1.y), min(t1.z, t_hi));
    return far >= near;
}

/// Insert `t` into the ascending hit list, keeping at most `MAX_HITS`.
fn insert_hit(hits: ptr<function, array<f32, MAX_HITS>>, n: ptr<function, u32>, t: f32) {
    var i = *n;
    if (i >= MAX_HITS) {
        i = MAX_HITS - 1u;
    } else {
        *n = *n + 1u;
    }
    loop {
        if (i == 0u) {
            break;
        }
        if ((*hits)[i - 1u] <= t) {
            break;
        }
        (*hits)[i] = (*hits)[i - 1u];
        i = i - 1u;
    }
    (*hits)[i] = t;
}

/// All triangle crossings of `o + t·d` with `t_min < t < t_max`, ascending.
fn trace(
    o: vec3<f32>,
    d: vec3<f32>,
    hits: ptr<function, array<f32, MAX_HITS>>,
    n_hits: ptr<function, u32>,
) {
    *n_hits = 0u;
    let inv_d = 1.0 / d;
    var t_hi = g.t_max;
    var stack: array<u32, STACK>;
    var sp = 0u;
    stack[0] = 0u;
    sp = 1u;
    loop {
        if (sp == 0u) {
            break;
        }
        sp = sp - 1u;
        let node = stack[sp];
        if (!slab_hit(node, o, inv_d, g.t_min, t_hi)) {
            continue;
        }
        let link = links[node];
        if (link.w > 0u) {
            for (var i = 0u; i < link.w; i = i + 1u) {
                let f = 3u * (link.z + i);
                let v0 = positions[indices[f]].xyz;
                let v1 = positions[indices[f + 1u]].xyz;
                let v2 = positions[indices[f + 2u]].xyz;
                let t = tri_hit(o, d, v0, v1, v2, g.t_min);
                if (t > 0.0 && t < t_hi) {
                    insert_hit(hits, n_hits, t);
                    if (*n_hits == MAX_HITS) {
                        t_hi = (*hits)[MAX_HITS - 1u];
                    }
                }
            }
        } else if (sp + 2u <= STACK) {
            stack[sp] = link.x;
            sp = sp + 1u;
            stack[sp] = link.y;
            sp = sp + 1u;
        }
    }
}

/// `+X` crossing parity: 1 when the observation point sits inside the mesh.
@compute @workgroup_size(64)
fn inside_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if (pid >= g.n_points) {
        return;
    }
    var hits: array<f32, MAX_HITS>;
    var n_hits: u32;
    trace(points[pid].xyz, vec3<f32>(1.0, 0.0, 0.0), &hits, &n_hits);
    out_inside[pid] = n_hits % 2u;
}

/// Visible `(r₀, r₁)` intervals per (point, direction).
@compute @workgroup_size(64)
fn rays(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * (g.grid_x * 64u);
    if (idx >= g.n_points * g.n_dirs) {
        return;
    }
    let pid = idx / g.n_dirs;
    let u = dirs[idx % g.n_dirs].xyz;
    let x = points[pid].xyz;
    var hits: array<f32, MAX_HITS>;
    var n_hits: u32;
    trace(x, -u, &hits, &n_hits);

    let inside = out_inside[pid] == 1u;
    var slots: array<vec2<f32>, MAX_INTERVALS>;
    var count = 0u;
    if (inside) {
        if (n_hits > 0u) {
            slots[count] = vec2<f32>(0.0, hits[0]);
            count = count + 1u;
        }
        var i = 1u;
        loop {
            if (i + 1u >= n_hits || count >= MAX_INTERVALS) {
                break;
            }
            slots[count] = vec2<f32>(hits[i], hits[i + 1u]);
            count = count + 1u;
            i = i + 2u;
        }
    } else {
        var i = 0u;
        loop {
            if (i + 1u >= n_hits || count >= MAX_INTERVALS) {
                break;
            }
            slots[count] = vec2<f32>(hits[i], hits[i + 1u]);
            count = count + 1u;
            i = i + 2u;
        }
    }
    out_counts[idx] = count;
    for (var k = 0u; k < MAX_INTERVALS; k = k + 1u) {
        out_ivals[idx * MAX_INTERVALS + k] =
            select(vec2<f32>(0.0, 0.0), slots[k], k < count);
    }
}
