// CarlsonAlpha general-alpha radial finite part.
//
// The angular geometry and tensor reduction match RT-FP, but the radial
// remainder is integrated directly for an arbitrary positive alpha:
//
//   Rk = integral_a^b [(A - 2 s^2 b R + s^2 R^2)^(-alpha) - C^(-alpha)] dR/R
//
// Subtracting C^(-alpha) removes the logarithmic finite-part singularity, so
// the integrand is regular even when a = 0. A 16-point Gauss-Legendre rule is
// used per visible interval. The CPU mirror in `carlson_alpha.rs` uses a
// 64-point f64 rule, and the Carlson RF/RD/RJ/RC library bridge verifies the
// boundary-reduction identities used by the special-function path.

struct Globals {
    n_points: u32,
    n_dirs: u32,
    n_kernels: u32,
    g: f32,
    quadrature_nodes: u32,
}

struct Kernel {
    c: vec3<f32>,
    sigma: f32,
    w: f32,
    alpha: f32,
    _pad1: f32,
    _pad2: f32,
}

struct Dir {
    u: vec3<f32>,
    omega: f32,
}

@group(0) @binding(0) var<uniform> globals: Globals;
const MAX_INTERVALS: u32 = 16u;
@group(0) @binding(1) var<storage, read> ivals: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> counts: array<u32>;
@group(0) @binding(3) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> kernels: array<Kernel>;
@group(0) @binding(5) var<storage, read> dirs: array<Dir>;
@group(0) @binding(6) var<storage, read_write> out_t: array<f32>;

const GL_X: array<f32, 16> = array<f32, 16>(
    9.894009349916499385e-01,
    9.445750230732326003e-01,
    8.656312023878317552e-01,
    7.554044083550029987e-01,
    6.178762444026437706e-01,
    4.580167776572273697e-01,
    2.816035507792589154e-01,
    9.501250983763744051e-02,
    -9.501250983763744051e-02,
    -2.816035507792589154e-01,
    -4.580167776572273697e-01,
    -6.178762444026437706e-01,
    -7.554044083550029987e-01,
    -8.656312023878317552e-01,
    -9.445750230732326003e-01,
    -9.894009349916499385e-01,
);

const GL_W: array<f32, 16> = array<f32, 16>(
    2.715245941175185168e-02,
    6.225352393864777567e-02,
    9.515851168249289671e-02,
    1.246289712555339463e-01,
    1.495959888165768192e-01,
    1.691565193950024248e-01,
    1.826034150449236115e-01,
    1.894506104550684744e-01,
    1.894506104550684744e-01,
    1.826034150449236115e-01,
    1.691565193950024248e-01,
    1.495959888165768192e-01,
    1.246289712555339463e-01,
    9.515851168249289671e-02,
    6.225352393864777567e-02,
    2.715245941175185168e-02,
);

fn radial_remainder(k: Kernel, p: vec3<f32>, u: vec3<f32>, a: f32, b: f32) -> f32 {
    if (b <= a) {
        return 0.0;
    }
    let s2 = k.sigma * k.sigma;
    let aa = 1.0 + s2 * dot(p, p);
    let bb = dot(p, u);
    let phi0 = pow(max(aa, 1e-30), -k.alpha);
    let mid = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    var sum = 0.0;
    for (var i = 0u; i < min(globals.quadrature_nodes, 16u); i = i + 1u) {
        let r = mid + half * GL_X[i];
        let d = max(aa - 2.0 * s2 * bb * r + s2 * r * r, 1e-30);
        sum = sum + GL_W[i] * (pow(d, -k.alpha) - phi0) / max(r, 1e-30);
    }
    return k.w * half * sum;
}

@compute @workgroup_size(64)
fn carlson_alpha(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if (pid >= globals.n_points) {
        return;
    }
    let x = points[pid].xyz;
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
            for (var t = 0u; t < MAX_INTERVALS; t = t + 1u) {
                if (t >= cnt) {
                    break;
                }
                let slot = ivals[idx * MAX_INTERVALS + t];
                s = s + radial_remainder(kk, p, u, slot.x, slot.y);
            }
        }
        let amp = globals.g * dirs[q].omega * s;
        h[0] += amp * (3.0 * u.x * u.x - 1.0);
        h[1] += amp * (3.0 * u.y * u.y - 1.0);
        h[2] += amp * (3.0 * u.z * u.z - 1.0);
        h[3] += amp * 3.0 * u.x * u.y;
        h[4] += amp * 3.0 * u.x * u.z;
        h[5] += amp * 3.0 * u.y * u.z;
    }

    let base = pid * 6u;
    for (var i = 0u; i < 6u; i = i + 1u) {
        out_t[base + i] = h[i];
    }
}
