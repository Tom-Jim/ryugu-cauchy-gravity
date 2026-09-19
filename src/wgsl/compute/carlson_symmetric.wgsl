// Carlson symmetric elliptic integrals for real WebGPU workloads.
//
// The implementation follows the duplication/Taylor construction of
// B. C. Carlson (1995).  Arguments are scaled before iteration, each
// duplication contracts their differences by four, and the final value is
// recovered with the appropriate homogeneity.  The production gravity paths
// use real, non-negative arguments.  A pole on the integration path is not
// silently continued: RJ requires p > 0 and RC requires y > 0.

const CARLSON_OK: u32 = 0u;
const CARLSON_DOMAIN: u32 = 1u;
const CARLSON_SINGULAR: u32 = 2u;
const CARLSON_NO_CONVERGENCE: u32 = 3u;
const CARLSON_MAX_ITERS: u32 = 16u;
const CARLSON_TAYLOR_TOL: f32 = 2.5e-3;

struct CarlsonResult {
  value: f32,
  status: u32,
  iterations: u32,
  _pad: u32,
}

// Preprocessed real-arc packet. `arguments.xyz` are RF/RD/RJ arguments and
// `.w` is the RJ pole argument. `correction.xy` are RC arguments. The four
// coefficients are respectively elliptic, RC, algebraic endpoint and oriented
// radical scale. No quartic roots or complex arithmetic reach the GPU.
struct CarlsonArcPacket {
  arguments: vec4<f32>,
  correction: vec4<f32>,
  coefficients: vec4<f32>,
  metadata: vec4<u32>,
}

const CARLSON_ARC_RF: u32 = 0u;
const CARLSON_ARC_RD: u32 = 1u;
const CARLSON_ARC_RJ_RC: u32 = 2u;

fn carlson_result(value: f32, status: u32, iterations: u32) -> CarlsonResult {
  return CarlsonResult(value, status, iterations, 0u);
}

// WGSL constant expressions must be representable finite f32 values.  Domain
// failures are carried exclusively by `status`; callers must not rely on a NaN
// payload (several WebGPU frontends reject a bitcast-created NaN at parse time).
fn carlson_failure_value() -> f32 {
  return 0.0;
}

fn carlson_is_finite(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}

fn carlson_bad3(x: f32, y: f32, z: f32) -> bool {
  return !carlson_is_finite(x) || !carlson_is_finite(y) || !carlson_is_finite(z)
    || x < 0.0 || y < 0.0 || z < 0.0;
}

fn carlson_two_zeros3(x: f32, y: f32, z: f32) -> bool {
  return select(0u, 1u, x == 0.0) + select(0u, 1u, y == 0.0)
    + select(0u, 1u, z == 0.0) > 1u;
}

fn carlson_taylor(e2: f32, e3: f32, e4: f32, e5: f32) -> f32 {
  return 1.0 - 3.0 * e2 / 14.0 + e3 / 6.0
    + 9.0 * e2 * e2 / 88.0 - 3.0 * e4 / 22.0
    - 9.0 * e2 * e3 / 52.0 + 3.0 * e5 / 26.0
    - e2 * e2 * e2 / 16.0 + 3.0 * e3 * e3 / 40.0
    + 3.0 * e2 * e4 / 20.0 + 45.0 * e2 * e2 * e3 / 272.0
    - 9.0 * (e3 * e4 + e2 * e5) / 68.0;
}

fn carlson_rf(x: f32, y: f32, z: f32) -> CarlsonResult {
  if (carlson_bad3(x, y, z)) {
    return carlson_result(carlson_failure_value(), CARLSON_DOMAIN, 0u);
  }
  if (carlson_two_zeros3(x, y, z)) {
    return carlson_result(carlson_failure_value(), CARLSON_SINGULAR, 0u);
  }
  let scale = max(max(x, y), z);
  if (!(scale > 0.0)) {
    return carlson_result(carlson_failure_value(), CARLSON_SINGULAR, 0u);
  }
  var xn = x / scale;
  var yn = y / scale;
  var zn = z / scale;
  if (carlson_two_zeros3(xn, yn, zn)) {
    // A formally positive input can collapse to two zeros after f32 scaling.
    // Do not turn that out-of-dynamic-range case into a plausible finite value.
    return carlson_result(carlson_failure_value(), CARLSON_SINGULAR, 0u);
  }
  var iteration = 0u;
  loop {
    let mean = (xn + yn + zn) / 3.0;
    let dx = (mean - xn) / mean;
    let dy = (mean - yn) / mean;
    let dz = (mean - zn) / mean;
    let spread = max(max(abs(dx), abs(dy)), abs(dz));
    if (spread <= CARLSON_TAYLOR_TOL) {
      let e2 = dx * dy - dz * dz;
      let e3 = dx * dy * dz;
      let series = 1.0 + e3 * (1.0 / 14.0 + 3.0 * e3 / 104.0)
        + e2 * (-0.1 + e2 / 24.0 - 3.0 * e3 / 44.0
          - 5.0 * e2 * e2 / 208.0 + e2 * e3 / 16.0);
      return carlson_result(series * inverseSqrt(mean * scale), CARLSON_OK, iteration);
    }
    if (iteration >= CARLSON_MAX_ITERS) {
      return carlson_result(carlson_failure_value(), CARLSON_NO_CONVERGENCE, iteration);
    }
    let rx = sqrt(xn);
    let ry = sqrt(yn);
    let rz = sqrt(zn);
    let lambda = rx * ry + rx * rz + ry * rz;
    xn = 0.25 * (xn + lambda);
    yn = 0.25 * (yn + lambda);
    zn = 0.25 * (zn + lambda);
    iteration += 1u;
  }
  return carlson_result(carlson_failure_value(), CARLSON_NO_CONVERGENCE, 0u);
}

// RC is the degenerate RF(x,y,y).  Keeping it on the same duplication path
// gives one set of domain and convergence rules on CPU and GPU.
fn carlson_rc(x: f32, y: f32) -> CarlsonResult {
  if (!carlson_is_finite(x) || !carlson_is_finite(y) || x < 0.0 || y < 0.0) {
    return carlson_result(carlson_failure_value(), CARLSON_DOMAIN, 0u);
  }
  if (y == 0.0) {
    return carlson_result(carlson_failure_value(), CARLSON_SINGULAR, 0u);
  }
  return carlson_rf(x, y, y);
}

fn carlson_rd(x: f32, y: f32, z: f32) -> CarlsonResult {
  if (carlson_bad3(x, y, z) || z == 0.0) {
    return carlson_result(carlson_failure_value(), CARLSON_DOMAIN, 0u);
  }
  if (x == 0.0 && y == 0.0) {
    return carlson_result(carlson_failure_value(), CARLSON_SINGULAR, 0u);
  }
  let scale = max(max(x, y), z);
  var xn = x / scale;
  var yn = y / scale;
  var zn = z / scale;
  if (xn == 0.0 && yn == 0.0) {
    return carlson_result(carlson_failure_value(), CARLSON_SINGULAR, 0u);
  }
  var sum = 0.0;
  var factor = 1.0;
  var iteration = 0u;
  loop {
    let mean = (xn + yn + 3.0 * zn) / 5.0;
    let dx = (mean - xn) / mean;
    let dy = (mean - yn) / mean;
    let dz = (mean - zn) / mean;
    let spread = max(max(abs(dx), abs(dy)), abs(dz));
    if (spread <= CARLSON_TAYLOR_TOL) {
      let xyz = dx * dy * dz;
      let dz2 = dz * dz;
      let dz3 = dz2 * dz;
      let e2 = dx * dy - 6.0 * dz2;
      let e3 = 3.0 * xyz - 8.0 * dz3;
      let e4 = 3.0 * (xyz - dz3) * dz;
      let e5 = xyz * dz2;
      let normalized = factor * carlson_taylor(e2, e3, e4, e5)
        / (mean * sqrt(mean)) + 3.0 * sum;
      return carlson_result(
        normalized / (scale * sqrt(scale)), CARLSON_OK, iteration,
      );
    }
    if (iteration >= CARLSON_MAX_ITERS) {
      return carlson_result(carlson_failure_value(), CARLSON_NO_CONVERGENCE, iteration);
    }
    let rx = sqrt(xn);
    let ry = sqrt(yn);
    let rz = sqrt(zn);
    let lambda = rx * ry + rx * rz + ry * rz;
    sum += factor / (rz * (zn + lambda));
    factor *= 0.25;
    xn = 0.25 * (xn + lambda);
    yn = 0.25 * (yn + lambda);
    zn = 0.25 * (zn + lambda);
    iteration += 1u;
  }
  return carlson_result(carlson_failure_value(), CARLSON_NO_CONVERGENCE, 0u);
}

fn carlson_rj(x: f32, y: f32, z: f32, p: f32) -> CarlsonResult {
  if (carlson_bad3(x, y, z) || !carlson_is_finite(p) || p < 0.0) {
    return carlson_result(carlson_failure_value(), CARLSON_DOMAIN, 0u);
  }
  if (carlson_two_zeros3(x, y, z) || p == 0.0) {
    return carlson_result(carlson_failure_value(), CARLSON_SINGULAR, 0u);
  }
  let scale = max(max(max(x, y), z), p);
  var xn = x / scale;
  var yn = y / scale;
  var zn = z / scale;
  var pn = p / scale;
  if (carlson_two_zeros3(xn, yn, zn) || pn == 0.0) {
    return carlson_result(carlson_failure_value(), CARLSON_SINGULAR, 0u);
  }
  var sum = 0.0;
  var factor = 1.0;
  var delta = (pn - xn) * (pn - yn) * (pn - zn);
  var iteration = 0u;
  loop {
    let mean = (xn + yn + zn + 2.0 * pn) / 5.0;
    let dx = (mean - xn) / mean;
    let dy = (mean - yn) / mean;
    let dz = (mean - zn) / mean;
    let dp = (mean - pn) / mean;
    let spread = max(max(abs(dx), abs(dy)), max(abs(dz), abs(dp)));
    if (spread <= CARLSON_TAYLOR_TOL) {
      let xyz = dx * dy * dz;
      let dp2 = dp * dp;
      let dp3 = dp2 * dp;
      let e2 = dx * dy + dx * dz + dy * dz - 3.0 * dp2;
      let e3 = xyz + 2.0 * e2 * dp + 4.0 * dp3;
      let e4 = (2.0 * xyz + e2 * dp + 3.0 * dp3) * dp;
      let e5 = xyz * dp2;
      let normalized = factor * carlson_taylor(e2, e3, e4, e5)
        / (mean * sqrt(mean)) + 6.0 * sum;
      return carlson_result(
        normalized / (scale * sqrt(scale)), CARLSON_OK, iteration,
      );
    }
    if (iteration >= CARLSON_MAX_ITERS) {
      return carlson_result(carlson_failure_value(), CARLSON_NO_CONVERGENCE, iteration);
    }
    let rx = sqrt(xn);
    let ry = sqrt(yn);
    let rz = sqrt(zn);
    let rp = sqrt(pn);
    let lambda = rx * ry + rx * rz + ry * rz;
    let dn = (rp + rx) * (rp + ry) * (rp + rz);
    let en = delta / (dn * dn);
    let rc = carlson_rc(1.0, 1.0 + en);
    if (rc.status != CARLSON_OK) {
      return carlson_result(carlson_failure_value(), rc.status, iteration);
    }
    sum += factor * rc.value / dn;
    factor *= 0.25;
    xn = 0.25 * (xn + lambda);
    yn = 0.25 * (yn + lambda);
    zn = 0.25 * (zn + lambda);
    pn = 0.25 * (pn + lambda);
    delta *= 1.0 / 64.0;
    iteration += 1u;
  }
  return carlson_result(carlson_failure_value(), CARLSON_NO_CONVERGENCE, 0u);
}

fn carlson_arc(packet: CarlsonArcPacket) -> CarlsonResult {
  var elliptic = carlson_failure_value();
  var status = CARLSON_DOMAIN;
  var iterations = 0u;
  if (packet.metadata.x == CARLSON_ARC_RF) {
    let result = carlson_rf(packet.arguments.x, packet.arguments.y, packet.arguments.z);
    elliptic = result.value;
    status = result.status;
    iterations = result.iterations;
  } else if (packet.metadata.x == CARLSON_ARC_RD) {
    let result = carlson_rd(packet.arguments.x, packet.arguments.y, packet.arguments.z);
    elliptic = result.value;
    status = result.status;
    iterations = result.iterations;
  } else if (packet.metadata.x == CARLSON_ARC_RJ_RC) {
    let rj = carlson_rj(
      packet.arguments.x, packet.arguments.y, packet.arguments.z, packet.arguments.w,
    );
    if (rj.status != CARLSON_OK) {
      return rj;
    }
    let rc = carlson_rc(packet.correction.x, packet.correction.y);
    if (rc.status != CARLSON_OK) {
      return rc;
    }
    let value = packet.coefficients.w * (
      packet.coefficients.x * rj.value
        + packet.coefficients.y * rc.value
        + packet.coefficients.z
    );
    return carlson_result(value, CARLSON_OK, max(rj.iterations, rc.iterations));
  }
  if (status != CARLSON_OK) {
    return carlson_result(carlson_failure_value(), status, iterations);
  }
  let value = packet.coefficients.w * (
    packet.coefficients.x * elliptic + packet.coefficients.z
  );
  return carlson_result(value, CARLSON_OK, iterations);
}
