// RT-FP deviation quadrature: one workgroup per observation point.
//
//   H_rem(x) = G sum_q omega_q T(u_q) sum_k w_k R_k(u_q)
//
// with R_k the radial finite-part integral of kernel k with its logarithmic
// near-field part removed (that part lives in the analytic polyhedral tensor):
//
//   r0 > 0 : R = [-0.5 ln(D(r1)/D(r0))
//                  +(b/d)(atan2(r1-b,d)-atan2(r0-b,d))] / A
//   r0 = 0 : R = [-0.5 ln(D(r1)/A)
//                  +(b/d)(atan2(r1-b,d)+atan2(b,d))] / A
//
// A=1+sigma^2|p|^2, b=p.u, d^2=1/sigma^2+|p|^2-b^2 and
// D(r)=A-2 sigma^2 b r+sigma^2 r^2. This remains the independent direct
// T*S path; CarlsonAlpha uses the integration-by-parts hybrid residual.

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
const MAX_INTERVALS: u32 = 16u;
// The last two words are endpoint face ids used only by CarlsonAlpha.
@group(0) @binding(1) var<storage, read> ivals: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> counts: array<u32>;
@group(0) @binding(3) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> kernels: array<Kernel>;
@group(0) @binding(5) var<storage, read> dirs: array<Dir>;
@group(0) @binding(6) var<storage, read_write> out_t: array<f32>;

const WG: u32 = 64u;
const COMPONENTS: u32 = 6u;
var<workgroup> partials: array<f32, WG * COMPONENTS>;

fn d_of(r: f32, a: f32, b: f32, s2: f32) -> f32 {
    return max(a - 2.0 * s2 * b * r + s2 * r * r, 1e-30);
}

@compute @workgroup_size(WG)
fn remainder(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let pid = workgroup.x;
    let x = points[pid].xyz;
    let nd = globals.n_dirs;
    let nk = globals.n_kernels;
    var h = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    for (var q = local.x; q < nd; q += WG) {
        let idx = pid * nd + q;
        let cnt = counts[idx];
        if (cnt == 0u || nk == 0u) {
            continue;
        }
        let u = dirs[q].u;
        var s = 0.0;
        for (var k = 0u; k < nk; k += 1u) {
            let kk = kernels[k];
            let p = x - kk.c;
            let p2 = dot(p, p);
            let s2 = kk.sigma * kk.sigma;
            let a = 1.0 + s2 * p2;
            let b = dot(p, u);
            let d = sqrt(max(1.0 / s2 + p2 - b * b, 1e-30));
            var jk = 0.0;
            for (var t = 0u; t < MAX_INTERVALS; t += 1u) {
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
                    jk += (-0.5 * log(d1 / d0) + phi1
                        - (b / d) * atan2(r0 - b, d)) / a;
                } else {
                    jk += (-0.5 * log(d1 / a) + phi1
                        + (b / d) * atan2(b, d)) / a;
                }
            }
            s += kk.w * jk;
        }
        let amp = globals.g * dirs[q].omega * s;
        h[0] += amp * (3.0 * u.x * u.x - 1.0);
        h[1] += amp * (3.0 * u.y * u.y - 1.0);
        h[2] += amp * (3.0 * u.z * u.z - 1.0);
        h[3] += amp * 3.0 * u.x * u.y;
        h[4] += amp * 3.0 * u.x * u.z;
        h[5] += amp * 3.0 * u.y * u.z;
    }

    for (var component = 0u; component < COMPONENTS; component += 1u) {
        partials[local.x * COMPONENTS + component] = h[component];
    }
    workgroupBarrier();

    var stride = WG / 2u;
    loop {
        if (local.x < stride) {
            for (var component = 0u; component < COMPONENTS; component += 1u) {
                let slot = local.x * COMPONENTS + component;
                partials[slot] += partials[slot + stride * COMPONENTS];
            }
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride /= 2u;
    }
    if (local.x == 0u) {
        let base = pid * COMPONENTS;
        for (var component = 0u; component < COMPONENTS; component += 1u) {
            out_t[base + component] = partials[component];
        }
    }
}
