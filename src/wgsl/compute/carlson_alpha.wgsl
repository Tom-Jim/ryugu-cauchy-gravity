// CarlsonAlpha hybrid residual for an arbitrary positive alpha.
//
// The radial finite-part scalar is differentiated exactly with respect to
// direction (including moving mesh endpoints).  On the closed direction
// sphere, globally compatible edge traces cancel and
//
//   integral T_ij S dOmega = (1/6) integral grad(T_ij).grad(S) dOmega.
//
// The fixed-endpoint beta derivative is integrated for arbitrary alpha:
//
//   Rk = integral_a^b [(A - 2 s^2 b R + s^2 R^2)^(-alpha) - C^(-alpha)] dR/R
//
//   partial_beta Rk = integral 2 alpha sigma^2 D^(-alpha-1) dR.
//
// Alpha=1 uses the exact RC-backed endpoint-angle formula. Complete
// GL4/GL8/GL16 rules are used for general alpha.

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
// (r0, r1, bitcast(entry_face), bitcast(exit_face)).
@group(0) @binding(1) var<storage, read> ivals: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> counts: array<u32>;
@group(0) @binding(3) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> kernels: array<Kernel>;
@group(0) @binding(5) var<storage, read> dirs: array<Dir>;
@group(0) @binding(6) var<storage, read_write> out_t: array<f32>;
@group(0) @binding(7) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> indices: array<u32>;

const WG: u32 = 64u;
const COMPONENTS: u32 = 6u;
const INVALID_FACE: u32 = 0xffffffffu;
var<workgroup> partials: array<f32, WG * COMPONENTS>;

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

const GL4_X: array<f32, 4> = array<f32, 4>(
    -8.611363115940525753e-01, -3.399810435848562648e-01,
     3.399810435848562648e-01,  8.611363115940525753e-01,
);
const GL4_W: array<f32, 4> = array<f32, 4>(
    3.478548451374538573e-01, 6.521451548625461427e-01,
    6.521451548625461427e-01, 3.478548451374538573e-01,
);

const GL8_X: array<f32, 8> = array<f32, 8>(
    -9.602898564975362317e-01, -7.966664774136267396e-01,
    -5.255324099163289858e-01, -1.834346424956498050e-01,
     1.834346424956498050e-01,  5.255324099163289858e-01,
     7.966664774136267396e-01,  9.602898564975362317e-01,
);
const GL8_W: array<f32, 8> = array<f32, 8>(
    1.012285362903762593e-01, 2.223810344533744706e-01,
    3.137066458778872874e-01, 3.626837833783619830e-01,
    3.626837833783619830e-01, 3.137066458778872874e-01,
    2.223810344533744706e-01, 1.012285362903762593e-01,
);

fn stable_expm1(value: f32) -> f32 {
    let magnitude = abs(value);
    if (magnitude < 1e-3) {
        let square = value * value;
        return value + 0.5 * square + value * square / 6.0
            + square * square / 24.0;
    }
    return exp(value) - 1.0;
}

fn stable_log1p(value: f32) -> f32 {
    if (abs(value) < 1e-3) {
        let square = value * value;
        return value - 0.5 * square + value * square / 3.0
            - 0.25 * square * square;
    }
    return log(1.0 + value);
}

fn carlson_atan_positive(value: f32) -> f32 {
    if (value > 1.0) {
        let reciprocal = 1.0 / value;
        let rc = carlson_rc(1.0, 1.0 + reciprocal * reciprocal);
        return 0.5 * 3.14159265358979323846 - reciprocal * rc.value;
    }
    let rc = carlson_rc(1.0, 1.0 + value * value);
    return value * rc.value;
}

// x1 >= x0 here.  The atan2 form avoids subtracting two nearly equal endpoint
// angles; RC supplies atan through atan(t)=t*RC(1+t^2,1).
fn carlson_atan_difference(x1: f32, x0: f32) -> f32 {
    let cross_term = max(x1 - x0, 0.0);
    let dot_term = 1.0 + x1 * x0;
    if (abs(dot_term) <= 1e-20) {
        return 0.5 * 3.14159265358979323846;
    }
    let acute = carlson_atan_positive(cross_term / abs(dot_term));
    return select(3.14159265358979323846 - acute, acute, dot_term > 0.0);
}

fn endpoint_gradient(face: u32, r: f32, u: vec3<f32>) -> vec3<f32> {
    if (face == INVALID_FACE || !(r > 0.0)) {
        return vec3<f32>(0.0);
    }
    let base = 3u * face;
    let v0 = positions[indices[base]].xyz;
    let v1 = positions[indices[base + 1u]].xyz;
    let v2 = positions[indices[base + 2u]].xyz;
    let normal = cross(v1 - v0, v2 - v0);
    let denominator = dot(normal, u);
    // The ray kernel has already rejected scale-relative grazing determinants.
    return -r * (normal - denominator * u) / denominator;
}

fn alpha_one_beta_derivative(
    r0: f32,
    r1: f32,
    beta: f32,
    d: f32,
    s2: f32,
) -> f32 {
    let y0 = r0 - beta;
    let y1 = r1 - beta;
    let d2 = d * d;
    let endpoint0 = y0 / (d2 * (y0 * y0 + d2));
    let endpoint1 = y1 / (d2 * (y1 * y1 + d2));
    let angle = carlson_atan_difference(y1 / d, y0 / d);
    return (endpoint1 - endpoint0 + angle / (d2 * d)) / s2;
}

fn endpoint_value(k: Kernel, aa: f32, beta: f32, s2: f32, r: f32) -> f32 {
    if (!(r > 0.0)) {
        return 2.0 * k.alpha * s2 * beta * pow(aa, -k.alpha - 1.0);
    }
    let d = max(aa - 2.0 * s2 * beta * r + s2 * r * r, 1e-30);
    let relative = (d - aa) / aa;
    return pow(aa, -k.alpha) * stable_expm1(-k.alpha * stable_log1p(relative)) / r;
}

fn beta_derivative_value(k: Kernel, aa: f32, beta: f32, s2: f32, r: f32) -> f32 {
    let d = max(aa - 2.0 * s2 * beta * r + s2 * r * r, 1e-30);
    return 2.0 * k.alpha * s2 * pow(d, -k.alpha - 1.0);
}

fn integrate_beta4(k: Kernel, aa: f32, beta: f32, s2: f32, mid: f32, half: f32) -> f32 {
    var sum = 0.0;
    for (var i = 0u; i < 4u; i += 1u) {
        sum += GL4_W[i] * beta_derivative_value(k, aa, beta, s2, mid + half * GL4_X[i]);
    }
    return half * sum;
}

fn integrate_beta8(k: Kernel, aa: f32, beta: f32, s2: f32, mid: f32, half: f32) -> f32 {
    var sum = 0.0;
    for (var i = 0u; i < 8u; i += 1u) {
        sum += GL8_W[i] * beta_derivative_value(k, aa, beta, s2, mid + half * GL8_X[i]);
    }
    return half * sum;
}

fn integrate_beta16(k: Kernel, aa: f32, beta: f32, s2: f32, mid: f32, half: f32) -> f32 {
    var sum = 0.0;
    for (var i = 0u; i < 16u; i += 1u) {
        sum += GL_W[i] * beta_derivative_value(k, aa, beta, s2, mid + half * GL_X[i]);
    }
    return half * sum;
}

fn fixed_endpoint_beta_derivative(
    k: Kernel,
    p: vec3<f32>,
    beta: f32,
    r0: f32,
    r1: f32,
) -> f32 {
    let s2 = k.sigma * k.sigma;
    let aa = 1.0 + s2 * dot(p, p);
    if (abs(k.alpha - 1.0) <= 8.0 * 1.1920929e-7) {
        let d = sqrt(max(1.0 / s2 + dot(p, p) - beta * beta, 1e-30));
        return alpha_one_beta_derivative(r0, r1, beta, d, s2);
    }
    let mid = 0.5 * (r0 + r1);
    let half = 0.5 * (r1 - r0);
    if (globals.quadrature_nodes <= 4u) {
        return integrate_beta4(k, aa, beta, s2, mid, half);
    } else if (globals.quadrature_nodes <= 8u) {
        return integrate_beta8(k, aa, beta, s2, mid, half);
    }
    return integrate_beta16(k, aa, beta, s2, mid, half);
}

@compute @workgroup_size(WG)
fn carlson_alpha(
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
        var grad_s = vec3<f32>(0.0);
        for (var k = 0u; k < nk; k = k + 1u) {
            let kk = kernels[k];
            let p = x - kk.c;
            let s2 = kk.sigma * kk.sigma;
            let aa = 1.0 + s2 * dot(p, p);
            let beta = dot(p, u);
            let grad_beta = p - beta * u;
            var grad_jk = vec3<f32>(0.0);
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
                let grad_r0 = endpoint_gradient(bitcast<u32>(slot.z), r0, u);
                let grad_r1 = endpoint_gradient(bitcast<u32>(slot.w), r1, u);
                let beta_term = fixed_endpoint_beta_derivative(kk, p, beta, r0, r1);
                grad_jk += beta_term * grad_beta
                    + endpoint_value(kk, aa, beta, s2, r1) * grad_r1
                    - endpoint_value(kk, aa, beta, s2, r0) * grad_r0;
            }
            grad_s += kk.w * grad_jk;
        }
        grad_s -= u * dot(u, grad_s);
        let amp = globals.g * dirs[q].omega;
        h[0] += amp * u.x * grad_s.x;
        h[1] += amp * u.y * grad_s.y;
        h[2] += amp * u.z * grad_s.z;
        h[3] += 0.5 * amp * (u.y * grad_s.x + u.x * grad_s.y);
        h[4] += 0.5 * amp * (u.z * grad_s.x + u.x * grad_s.z);
        h[5] += 0.5 * amp * (u.z * grad_s.y + u.y * grad_s.z);
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
