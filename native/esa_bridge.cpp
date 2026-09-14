#include "esa_bridge.h"

#include "polyhedralGravity/model/GravityEvaluable.h"
#include "polyhedralGravity/model/Polyhedron.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <new>
#include <vector>

using polyhedralGravity::Array3;
using polyhedralGravity::GravityEvaluable;
using polyhedralGravity::GravityModelResult;
using polyhedralGravity::IndexArray3;
using polyhedralGravity::MetricUnit;
using polyhedralGravity::NormalOrientation;
using polyhedralGravity::Polyhedron;
using polyhedralGravity::PolyhedronIntegrity;

struct EsaPgHandle {
    GravityEvaluable evaluable;
};

extern "C" EsaPgHandle *esa_pg_create(
    const double *vertices_xyz,
    size_t vertex_count,
    const uint32_t *faces,
    size_t face_count,
    double density_kg_m3,
    int check_mesh) {
    if (vertices_xyz == nullptr || faces == nullptr || vertex_count == 0 || face_count == 0) {
        return nullptr;
    }
    try {
        std::vector<Array3> vertices(vertex_count);
        for (size_t i = 0; i < vertex_count; ++i) {
            vertices[i] = {
                vertices_xyz[3 * i + 0],
                vertices_xyz[3 * i + 1],
                vertices_xyz[3 * i + 2],
            };
        }
        std::vector<IndexArray3> triangles(face_count);
        for (size_t i = 0; i < face_count; ++i) {
            triangles[i] = {
                static_cast<size_t>(faces[3 * i + 0]),
                static_cast<size_t>(faces[3 * i + 1]),
                static_cast<size_t>(faces[3 * i + 2]),
            };
        }
        const auto integrity =
            check_mesh ? PolyhedronIntegrity::VERIFY : PolyhedronIntegrity::DISABLE;
        Polyhedron polyhedron{
            vertices,
            triangles,
            density_kg_m3,
            NormalOrientation::OUTWARDS,
            integrity,
            MetricUnit::METER,
        };
        return new (std::nothrow) EsaPgHandle{GravityEvaluable{polyhedron}};
    } catch (...) {
        return nullptr;
    }
}

extern "C" void esa_pg_destroy(EsaPgHandle *handle) {
    delete handle;
}

static void unpack_result(
    const GravityModelResult &result,
    double *out_potential,
    double out_acceleration[3],
    double out_H6[6]) {
    const auto &potential = std::get<0>(result);
    const auto &acceleration = std::get<1>(result);
    const auto &tensor = std::get<2>(result);
    if (out_potential != nullptr) {
        *out_potential = potential;
    }
    if (out_acceleration != nullptr) {
        out_acceleration[0] = acceleration[0];
        out_acceleration[1] = acceleration[1];
        out_acceleration[2] = acceleration[2];
    }
    if (out_H6 != nullptr) {
        for (size_t i = 0; i < 6; ++i) {
            out_H6[i] = tensor[i];
        }
    }
}

extern "C" int32_t esa_pg_eval(
    EsaPgHandle *handle,
    const double position_m[3],
    double *out_potential,
    double out_acceleration[3],
    double out_H6[6]) {
    if (handle == nullptr || position_m == nullptr) {
        return -1;
    }
    try {
        const Array3 point{position_m[0], position_m[1], position_m[2]};
        const auto result = std::get<GravityModelResult>(handle->evaluable(point, false));
        unpack_result(result, out_potential, out_acceleration, out_H6);
        return 0;
    } catch (...) {
        return -2;
    }
}

extern "C" int32_t esa_pg_eval_many(
    EsaPgHandle *handle,
    const double *positions_xyz,
    size_t point_count,
    double *out_potentials,
    double *out_accelerations,
    double *out_H6) {
    if (handle == nullptr || positions_xyz == nullptr || point_count == 0) {
        return -1;
    }
    try {
        std::vector<Array3> points(point_count);
        for (size_t i = 0; i < point_count; ++i) {
            points[i] = {
                positions_xyz[3 * i + 0],
                positions_xyz[3 * i + 1],
                positions_xyz[3 * i + 2],
            };
        }
        const auto results =
            std::get<std::vector<GravityModelResult>>(handle->evaluable(points, true));
        for (size_t i = 0; i < point_count; ++i) {
            double *pot = out_potentials != nullptr ? &out_potentials[i] : nullptr;
            double *acc = out_accelerations != nullptr ? &out_accelerations[3 * i] : nullptr;
            double *h6 = out_H6 != nullptr ? &out_H6[6 * i] : nullptr;
            unpack_result(results[i], pot, acc, h6);
        }
        return 0;
    } catch (...) {
        return -2;
    }
}
