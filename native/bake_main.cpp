/**
 * Near-surface gravity-gradient bake (ESA polyhedral via C ABI).
 * Used as the computation core; Zig CLI links the same C ABI.
 */
#include "esa_bridge.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

static constexpr double RYUGU_BULK_DENSITY_KG_M3 = 1190.0;
static constexpr double NEAR_SURFACE_EPS_M = 1.0e-3;
static constexpr uint32_t MAGIC = 0x52484746; // 'RHGF'

struct Mesh {
    std::vector<double> xyz;
    std::vector<uint32_t> faces;
};

static Mesh load_obj_km(const char *path, double km_to_m) {
    Mesh mesh;
    std::ifstream in(path);
    if (!in) throw std::runtime_error(std::string("cannot open ") + path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.size() < 2) continue;
        if (line[0] == 'v' && line[1] == ' ') {
            double x, y, z;
            if (std::sscanf(line.c_str() + 2, "%lf %lf %lf", &x, &y, &z) == 3) {
                mesh.xyz.push_back(x * km_to_m);
                mesh.xyz.push_back(y * km_to_m);
                mesh.xyz.push_back(z * km_to_m);
            }
        } else if (line[0] == 'f' && line[1] == ' ') {
            uint32_t idx[3] = {};
            int found = 0;
            const char *p = line.c_str() + 2;
            while (*p && found < 3) {
                while (*p == ' ') ++p;
                if (!*p) break;
                char *end = nullptr;
                unsigned long v = std::strtoul(p, &end, 10);
                if (end == p) break;
                idx[found++] = static_cast<uint32_t>(v - 1);
                p = end;
                while (*p && *p != ' ') ++p;
            }
            if (found == 3) {
                mesh.faces.push_back(idx[0]);
                mesh.faces.push_back(idx[1]);
                mesh.faces.push_back(idx[2]);
            }
        }
    }
    return mesh;
}

static void accumulate_vertex_normals(
    const Mesh &mesh, std::vector<double> &nx, std::vector<double> &ny, std::vector<double> &nz) {
    const size_t nv = mesh.xyz.size() / 3;
    nx.assign(nv, 0.0);
    ny.assign(nv, 0.0);
    nz.assign(nv, 0.0);
    for (size_t f = 0; f + 2 < mesh.faces.size(); f += 3) {
        const uint32_t i0 = mesh.faces[f], i1 = mesh.faces[f + 1], i2 = mesh.faces[f + 2];
        const double *p0 = &mesh.xyz[3 * i0];
        const double *p1 = &mesh.xyz[3 * i1];
        const double *p2 = &mesh.xyz[3 * i2];
        const double e1[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
        const double e2[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
        const double n[3] = {
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        };
        for (uint32_t idx : {i0, i1, i2}) {
            nx[idx] += n[0];
            ny[idx] += n[1];
            nz[idx] += n[2];
        }
    }
    for (size_t i = 0; i < nv; ++i) {
        const double len = std::sqrt(nx[i] * nx[i] + ny[i] * ny[i] + nz[i] * nz[i]);
        if (len > 1e-30) {
            nx[i] /= len;
            ny[i] /= len;
            nz[i] /= len;
        } else {
            const double *p = &mesh.xyz[3 * i];
            const double r = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
            if (r > 1e-30) {
                nx[i] = p[0] / r;
                ny[i] = p[1] / r;
                nz[i] = p[2] / r;
            } else {
                nx[i] = 0;
                ny[i] = 1;
                nz[i] = 0;
            }
        }
        const double *p = &mesh.xyz[3 * i];
        if (nx[i] * p[0] + ny[i] * p[1] + nz[i] * p[2] < 0) {
            nx[i] = -nx[i];
            ny[i] = -ny[i];
            nz[i] = -nz[i];
        }
    }
}

static double frobenius6(const double H[6]) {
    return std::sqrt(
        H[0] * H[0] + H[1] * H[1] + H[2] * H[2] + 2.0 * (H[3] * H[3] + H[4] * H[4] + H[5] * H[5]));
}

int main(int argc, char **argv) {
    const char *obj_path = nullptr;
    const char *out_path = "assets/gradient_faces.bin";
    size_t stride = 1;
    size_t max_faces = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--obj" && i + 1 < argc) obj_path = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (a == "--stride" && i + 1 < argc) stride = std::max<size_t>(1, std::stoul(argv[++i]));
        else if (a == "--max-faces" && i + 1 < argc) max_faces = std::stoul(argv[++i]);
        else if (a == "--help") {
            std::puts("ryugu_gradient_bake --obj mesh.obj [--out path] [--stride N] [--max-faces N]");
            return 0;
        }
    }
    if (!obj_path) {
        std::fprintf(stderr, "missing --obj\n");
        return 1;
    }

    Mesh mesh = load_obj_km(obj_path, 1000.0);
    const size_t nv = mesh.xyz.size() / 3;
    const size_t nf = mesh.faces.size() / 3;
    std::printf("mesh: %zu vertices, %zu faces (meters), density=%.0f kg/m^3\n",
                nv, nf, RYUGU_BULK_DENSITY_KG_M3);

    std::vector<uint32_t> selected_faces;
    selected_faces.reserve(nf / stride + 1);
    for (size_t f = 0; f < nf; f += stride) {
        selected_faces.push_back(static_cast<uint32_t>(f));
        if (max_faces != 0 && selected_faces.size() >= max_faces) break;
    }

    std::unordered_map<uint32_t, uint32_t> local_of;
    std::vector<uint32_t> unique_verts;
    unique_verts.reserve(selected_faces.size() * 2);
    for (uint32_t f : selected_faces) {
        for (int k = 0; k < 3; ++k) {
            const uint32_t v = mesh.faces[3 * f + k];
            if (local_of.emplace(v, static_cast<uint32_t>(unique_verts.size())).second) {
                unique_verts.push_back(v);
            }
        }
    }
    std::printf("baking %zu faces (%zu unique near-surface vertices)\n",
                selected_faces.size(), unique_verts.size());

    EsaPgHandle *handle = esa_pg_create(
        mesh.xyz.data(), nv, mesh.faces.data(), nf, RYUGU_BULK_DENSITY_KG_M3, 0);
    if (!handle) {
        std::fprintf(stderr, "esa_pg_create failed\n");
        return 1;
    }

    std::vector<double> nx, ny, nz;
    accumulate_vertex_normals(mesh, nx, ny, nz);

    std::vector<double> positions(unique_verts.size() * 3);
    for (size_t i = 0; i < unique_verts.size(); ++i) {
        const uint32_t v = unique_verts[i];
        positions[3 * i + 0] = mesh.xyz[3 * v + 0] + NEAR_SURFACE_EPS_M * nx[v];
        positions[3 * i + 1] = mesh.xyz[3 * v + 1] + NEAR_SURFACE_EPS_M * ny[v];
        positions[3 * i + 2] = mesh.xyz[3 * v + 2] + NEAR_SURFACE_EPS_M * nz[v];
    }

    std::vector<double> H6(unique_verts.size() * 6, 0.0);
    const size_t chunk = 32;
    for (size_t start = 0; start < unique_verts.size(); start += chunk) {
        const size_t count = std::min(chunk, unique_verts.size() - start);
        if (esa_pg_eval_many(
                handle,
                positions.data() + 3 * start,
                count,
                nullptr,
                nullptr,
                H6.data() + 6 * start) != 0) {
            std::fprintf(stderr, "esa_pg_eval_many failed at %zu\n", start);
            esa_pg_destroy(handle);
            return 1;
        }
        std::printf("evaluated %zu / %zu vertices\n", start + count, unique_verts.size());
        std::fflush(stdout);
    }
    esa_pg_destroy(handle);

    std::vector<uint32_t> face_indices;
    std::vector<float> face_scalar;
    std::vector<float> face_positions_km; // 9 floats per face, original OBJ km units for display
    face_indices.reserve(selected_faces.size());
    face_scalar.reserve(selected_faces.size());
    face_positions_km.reserve(selected_faces.size() * 9);
    float s_min = 1e30f, s_max = -1e30f;
    for (uint32_t f : selected_faces) {
        const uint32_t i0 = local_of[mesh.faces[3 * f]];
        const uint32_t i1 = local_of[mesh.faces[3 * f + 1]];
        const uint32_t i2 = local_of[mesh.faces[3 * f + 2]];
        double Hbar[6];
        for (int k = 0; k < 6; ++k) {
            Hbar[k] = (H6[6 * i0 + k] + H6[6 * i1 + k] + H6[6 * i2 + k]) / 3.0;
        }
        const float s = static_cast<float>(frobenius6(Hbar));
        face_indices.push_back(f);
        face_scalar.push_back(s);
        for (int corner = 0; corner < 3; ++corner) {
            const uint32_t v = mesh.faces[3 * f + corner];
            // Convert meters back to km for display alignment with ryugu.glb / OBJ.
            face_positions_km.push_back(static_cast<float>(mesh.xyz[3 * v + 0] / 1000.0));
            face_positions_km.push_back(static_cast<float>(mesh.xyz[3 * v + 1] / 1000.0));
            face_positions_km.push_back(static_cast<float>(mesh.xyz[3 * v + 2] / 1000.0));
        }
        s_min = std::min(s_min, s);
        s_max = std::max(s_max, s);
    }

    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "cannot write %s\n", out_path);
        return 1;
    }
    const uint32_t version = 3;
    const uint32_t n_mesh_faces = static_cast<uint32_t>(nf);
    const uint32_t n_out = static_cast<uint32_t>(face_scalar.size());
    const uint32_t face_stride = static_cast<uint32_t>(stride);
    out.write(reinterpret_cast<const char *>(&MAGIC), 4);
    out.write(reinterpret_cast<const char *>(&version), 4);
    out.write(reinterpret_cast<const char *>(&n_mesh_faces), 4);
    out.write(reinterpret_cast<const char *>(&n_out), 4);
    out.write(reinterpret_cast<const char *>(&face_stride), 4);
    out.write(reinterpret_cast<const char *>(&s_min), 4);
    out.write(reinterpret_cast<const char *>(&s_max), 4);
    out.write(reinterpret_cast<const char *>(face_indices.data()), n_out * sizeof(uint32_t));
    out.write(reinterpret_cast<const char *>(face_scalar.data()), n_out * sizeof(float));
    out.write(reinterpret_cast<const char *>(face_positions_km.data()), face_positions_km.size() * sizeof(float));
    std::printf("wrote %u face entries to %s (s_min=%g s_max=%g)\n", n_out, out_path, s_min, s_max);
    return 0;
}
