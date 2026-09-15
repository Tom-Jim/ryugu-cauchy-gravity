/** Brute-force N-body gravity — same ABI as Ryugu_wasm Zig `ryugu_direct_sum_eval`. */
#ifndef RYUGU_MASCON_DIRECT_H
#define RYUGU_MASCON_DIRECT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Acceleration + potential at `position_m` from point masses (SI). */
int32_t ryugu_direct_sum_eval(const double *source_xyz,
                              const double *source_mass,
                              uint64_t source_count,
                              const double position_m[3],
                              double acceleration_mps2[3],
                              double *potential_m2ps2);

/**
 * Symmetric Hessian Γ_ij = ∂g_i/∂x_j of the same direct-sum field.
 * Packed as [xx, yy, zz, xy, xz, yz] (same layout as ESA polyhedral bake).
 */
int32_t ryugu_direct_sum_hessian_eval(const double *source_xyz,
                                      const double *source_mass,
                                      uint64_t source_count,
                                      const double position_m[3],
                                      double hessian6[6]);

#ifdef __cplusplus
}
#endif

#endif
