// RT-FP deviation quadrature: one invocation per observation point.
//
//   H_rem(x) = G Σ_q ω_q T(u_q) Σ_k w_k R_k(u_q)
//
// with R_k the radial finite-part integral of kernel k with its logarithmic
// near-field part removed (that part lives in the analytic polyhedral tensor):
//
//   r₀ > 0 :  R = [ −½ ln(D(r₁)/D(r₀)) + (b/d)(atan2(r₁−b,d) − atan2(r₀−b,d)) ] / A
//   r₀ = 0 :  R = [ −½ ln(D(r₁)/A)     + (b/d)(atan2(r₁−b,d) + atan2(b,d))     ] / A
//
// A = 1 + σ²‖p‖², b = p·u, d² = 1/σ² + ‖p‖² − b², D(r) = A − 2σ²br + σ²r²,
// p = x − c_k, T(u) = 3uuᵀ − I.  Matches `src/split.rs` term for term.

struct Globals {
    n_points: u32,
    n_dirs: u32,
    n_kernels: u32,
    g: f32,
}

struct Kernel {
    c: vec3<f32>,
    sigma: f32,
    w: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct Dir {
    u: vec3<f32>,
    omega: f32,
}

@group(0) @binding(0) var<uniform> globals: Globals;
// Visible (r₀, r₁) slots per (point, direction), written by `rays.wgsl`.
const MAX_INTERVALS: u32 = 16u;
@group(0) @binding(1) var<storage, read> ivals: array<vec2<f32>>;
// Visible-interval count per (point, direction).
@group(0) @binding(2) var<storage, read> counts: array<u32>;
// xyz, .w = 1 when the point is inside the mesh.
@group(0) @binding(3) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> kernels: array<Kernel>;
@group(0) @binding(5) var<storage, read> dirs: array<Dir>;
// Six Hessian components per point: xx, yy, zz, xy, xz, yz.
@group(0) @binding(6) var<storage, read_write> out_t: array<f32>;

fn d_of(r: f32, a: f32, b: f32, s2: f32) -> f32 {
    return max(a - 2.0 * s2 * b * r + s2 * r * r, 1e-30);
}

@compute @workgroup_size(64)
fn remainder(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if (pid >= globals.n_points) {
        return;
    }
    let pw = points[pid];
    let x = pw.xyz;
    let nd = globals.n_dirs;
    let nk = globals.n_kernels;
    var h = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    for (var q = 0u; q < nd; q = q + 1u) {
        let idx = pid * nd + q;
        let cnt = counts[idx];
        if (cnt == 0u || nk == 0u) {
            continue;
        }
        let u = dirs[q].u;
        var s = 0.0;
        for (var k = 0u; k < nk; k = k + 1u) {
            let kk = kernels[k];
            let p = x - kk.c;
            let p2 = dot(p, p);
            let s2 = kk.sigma * kk.sigma;
            let a = 1.0 + s2 * p2;
            let b = dot(p, u);
            let d = sqrt(max(1.0 / s2 + p2 - b * b, 1e-30));
            var jk = 0.0;
            for (var t = 0u; t < MAX_INTERVALS; t = t + 1u) {
                if (t >= cnt) {
                    break;
                }
                let slot = ivals[idx * MAX_INTERVALS + t];
                let r0 = slot.x;
                let r1 = slot.y;
                if (r1 <= r0) {
                    continue;
                }
                let d1 = d_of(r1, a, b, s2);
                let phi1 = (b / d) * atan2(r1 - b, d);
                if (r0 > 0.0) {
                    let d0 = d_of(r0, a, b, s2);
                    jk += (-0.5 * log(d1 / d0) + phi1 - (b / d) * atan2(r0 - b, d)) / a;
                } else {
                    jk += (-0.5 * log(d1 / a) + phi1 + (b / d) * atan2(b, d)) / a;
                }
            }
            s += kk.w * jk;
        }
        let amp = globals.g * dirs[q].omega * s;
        h[0] += amp * (3.0 * u.x * u.x - 1.0);
        h[1] += amp * (3.0 * u.y * u.y - 1.0);
        h[2] += amp * (3.0 * u.z * u.z - 1.0);
        h[3] += amp * (3.0 * u.x * u.y);
        h[4] += amp * (3.0 * u.x * u.z);
        h[5] += amp * (3.0 * u.y * u.z);
    }

    let base = pid * 6u;
    for (var i = 0u; i < 6u; i = i + 1u) {
        out_t[base + i] = h[i];
    }
}
