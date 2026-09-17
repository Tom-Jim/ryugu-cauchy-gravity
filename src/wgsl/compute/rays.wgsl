// Software BVH ray tracing for the RT-FP deviation integral.
//
// One invocation per (observation point, direction): walk the flat BVH built by
// `src/bvh.rs`, collect up to `MAX_HITS` triangle crossings along `x − R u`,
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
// One inside flag per point, followed by one atomic overflow counter at
// index `n_points`. The counter catches both dropped crossings and intervals
// that do not fit in `MAX_INTERVALS`, so the bake fails instead of silently
// integrating an incomplete ray.
@group(0) @binding(9) var<storage, read_write> out_inside_overflow: array<atomic<u32>>;

const MAX_HITS: u32 = 32u;
const MAX_INTERVALS: u32 = 16u;
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

fn axis_interval(lo: f32, hi: f32, origin: f32, direction: f32) -> vec2<f32> {
    if (abs(direction) < 1e-20) {
        if (origin < lo || origin > hi) {
            return vec2<f32>(1.0, -1.0);
        }
        return vec2<f32>(-1e30, 1e30);
    }
    let a = (lo - origin) / direction;
    let b = (hi - origin) / direction;
    return vec2<f32>(min(a, b), max(a, b));
}

fn slab_hit(node: u32, o: vec3<f32>, d: vec3<f32>, t_lo: f32, t_hi: f32) -> bool {
    let lo = nodes[2u * node].xyz;
    let hi = nodes[2u * node + 1u].xyz;
    let tx = axis_interval(lo.x, hi.x, o.x, d.x);
    let ty = axis_interval(lo.y, hi.y, o.y, d.y);
    let tz = axis_interval(lo.z, hi.z, o.z, d.z);
    let near = max(max(tx.x, ty.x), max(tz.x, t_lo));
    let far = min(min(tx.y, ty.y), min(tz.y, t_hi));
    return far >= near;
}

fn same_crossing(a: f32, b: f32) -> bool {
    let scale = max(max(abs(a), abs(b)), 1.0);
    return abs(a - b) <= max(1e-5, 8e-7 * scale);
}

/// Insert `t` into the ascending hit list, keeping at most `MAX_HITS`.
fn insert_hit(hits: ptr<function, array<f32, MAX_HITS>>, n: ptr<function, u32>, t: f32) {
    for (var j = 0u; j < *n; j = j + 1u) {
        // Adjacent triangles share an edge crossing. It is one boundary event,
        // not two intervals; retaining both corrupts parity and can erase the
        // following interior segment.
        if (same_crossing((*hits)[j], t)) {
            return;
        }
    }
    var i = *n;
    if (i >= MAX_HITS) {
        atomicAdd(&out_inside_overflow[g.n_points], 1u);
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
        if (!slab_hit(node, o, d, g.t_min, t_hi)) {
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
        } else {
            if (sp + 2u <= STACK) {
                stack[sp] = link.x;
                sp = sp + 1u;
                stack[sp] = link.y;
                sp = sp + 1u;
            } else {
                // Never turn a capacity failure into a plausible partial
                // integral. The Rust coordinator reads this counter and fails
                // the block with a diagnostic.
                atomicAdd(&out_inside_overflow[g.n_points], 1u);
            }
        }
    }
}

/// Crossing parity is unstable when a single ray grazes a shared edge or
/// vertex. Cast three non-collinear rays and take the majority; this keeps the
/// probe cheap while making the inside/outside decision robust on the real mesh.
@compute @workgroup_size(64)
fn inside_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if (pid >= g.n_points) {
        return;
    }
    var hits: array<f32, MAX_HITS>;
    var n_hits: u32;
    trace(points[pid].xyz, vec3<f32>(1.0, 0.0, 0.0), &hits, &n_hits);
    var votes = n_hits % 2u;
    trace(points[pid].xyz, vec3<f32>(0.371, 0.542, 0.753), &hits, &n_hits);
    votes = votes + n_hits % 2u;
    trace(points[pid].xyz, vec3<f32>(-0.613, 0.211, 0.761), &hits, &n_hits);
    votes = votes + n_hits % 2u;
    atomicStore(&out_inside_overflow[pid], select(0u, 1u, votes >= 2u));
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

    let inside = atomicLoad(&out_inside_overflow[pid]) == 1u;
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
    var needed = n_hits / 2u;
    if (inside && n_hits > 0u) {
        needed = 1u + (n_hits - 1u) / 2u;
    }
    if (needed > MAX_INTERVALS) {
        atomicAdd(&out_inside_overflow[g.n_points], needed - MAX_INTERVALS);
    }
    out_counts[idx] = count;
    for (var k = 0u; k < MAX_INTERVALS; k = k + 1u) {
        out_ivals[idx * MAX_INTERVALS + k] =
            select(vec2<f32>(0.0, 0.0), slots[k], k < count);
    }
}
