/** Copied/adapted from Ryugu_wasm Zig basilisk_bridge direct_sum (+ analytic Hessian). */
#include "mascon_direct.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {
constexpr double kGravityConstant = 6.67430e-11;

int direct_sum(const double *source_xyz, const double *source_mass, uint64_t source_count,
               const double position[3], double acceleration[3], double *potential) {
    if (source_xyz == nullptr || source_mass == nullptr || position == nullptr
        || acceleration == nullptr || potential == nullptr || source_count == 0) {
        return -2;
    }
    acceleration[0] = acceleration[1] = acceleration[2] = 0.0;
    *potential = 0.0;
    for (uint64_t index = 0; index < source_count; ++index) {
        const double dx = source_xyz[3 * index + 0] - position[0];
        const double dy = source_xyz[3 * index + 1] - position[1];
        const double dz = source_xyz[3 * index + 2] - position[2];
        const double radius_squared = std::max(dx * dx + dy * dy + dz * dz, 1.0e-24);
        const double inverse_radius = 1.0 / std::sqrt(radius_squared);
        const double scale = kGravityConstant * source_mass[index] * inverse_radius / radius_squared;
        acceleration[0] += scale * dx;
        acceleration[1] += scale * dy;
        acceleration[2] += scale * dz;
        *potential += kGravityConstant * source_mass[index] * inverse_radius;
    }
    return 0;
}

int direct_sum_hessian(const double *source_xyz, const double *source_mass, uint64_t source_count,
                       const double position[3], double H[6]) {
    if (source_xyz == nullptr || source_mass == nullptr || position == nullptr || H == nullptr
        || source_count == 0) {
        return -2;
    }
    H[0] = H[1] = H[2] = H[3] = H[4] = H[5] = 0.0;
    for (uint64_t index = 0; index < source_count; ++index) {
        const double dx = source_xyz[3 * index + 0] - position[0];
        const double dy = source_xyz[3 * index + 1] - position[1];
        const double dz = source_xyz[3 * index + 2] - position[2];
        const double r2 = std::max(dx * dx + dy * dy + dz * dz, 1.0e-24);
        const double r = std::sqrt(r2);
        const double r3 = r2 * r;
        const double r5 = r3 * r2;
        const double gm = kGravityConstant * source_mass[index];
        const double c = 3.0 * gm / r5;
        const double d = gm / r3;
        // Γ_ij = G m (3 dx_i dx_j / r^5 − δ_ij / r^3), dx = source − field point
        H[0] += c * dx * dx - d; // xx
        H[1] += c * dy * dy - d; // yy
        H[2] += c * dz * dz - d; // zz
        H[3] += c * dx * dy;     // xy
        H[4] += c * dx * dz;     // xz
        H[5] += c * dy * dz;     // yz
    }
    return 0;
}
} // namespace

int32_t ryugu_direct_sum_eval(const double *source_xyz, const double *source_mass,
                              uint64_t source_count, const double position_m[3],
                              double acceleration_mps2[3], double *potential_m2ps2) {
    return direct_sum(source_xyz, source_mass, source_count, position_m, acceleration_mps2,
                      potential_m2ps2);
}

int32_t ryugu_direct_sum_hessian_eval(const double *source_xyz, const double *source_mass,
                                      uint64_t source_count, const double position_m[3],
                                      double hessian6[6]) {
    return direct_sum_hessian(source_xyz, source_mass, source_count, position_m, hessian6);
}
