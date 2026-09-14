#ifndef ESA_PG_BRIDGE_H
#define ESA_PG_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct EsaPgHandle EsaPgHandle;

/** Create a cached GravityEvaluable. vertices: xyz xyz..., faces: i j k (0-based). */
EsaPgHandle *esa_pg_create(
    const double *vertices_xyz,
    size_t vertex_count,
    const uint32_t *faces,
    size_t face_count,
    double density_kg_m3,
    int check_mesh /* 0 = disable integrity check */);

void esa_pg_destroy(EsaPgHandle *handle);

/**
 * Evaluate potential, acceleration, and full Hessian (6 unique components)
 * at one point. out_H6 = [Vxx, Vyy, Vzz, Vxy, Vxz, Vyz].
 * Returns 0 on success.
 */
int32_t esa_pg_eval(
    EsaPgHandle *handle,
    const double position_m[3],
    double *out_potential,
    double out_acceleration[3],
    double out_H6[6]);

/**
 * Evaluate many points (serial loop over GravityEvaluable).
 * potentials[n], accelerations[3n], H6[6n]. Returns 0 on success.
 */
int32_t esa_pg_eval_many(
    EsaPgHandle *handle,
    const double *positions_xyz,
    size_t point_count,
    double *out_potentials,
    double *out_accelerations,
    double *out_H6);

#ifdef __cplusplus
}
#endif

#endif
