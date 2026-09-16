/** Progressive mascon-voxel gravity-gradient bake (brute-force direct sum).
 *
 * Uses the same C ABI as Ryugu_wasm Zig `ryugu_direct_sum_eval`, plus analytic
 * Hessian. Density: Cauchy kernels from assets/density/cauchy.toml.
 * Output: same RHGF v5 face-scalar format as Werner bake.
 */
#include "mascon_direct.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

/** Observation-surface height bounds (mm), matching the viewer slider. */
static constexpr double STANDOFF_MIN_MM = 1.0;
static constexpr double STANDOFF_MAX_MM = 32000.0;
static constexpr double STANDOFF_DEFAULT_MM = 16000.0;
/** Height the current run evaluates at; overridden by `--standoff-mm`. */
static double g_standoff_mm = STANDOFF_DEFAULT_MM;
static double g_standoff_m = STANDOFF_DEFAULT_MM * 1.0e-3;
static constexpr uint32_t MAGIC = 0x52484746; // 'RHGF'
static constexpr uint32_t VERSION = 5;
static constexpr size_t FACE_WINDOW = 128;
static constexpr size_t FLUSH_EVERY = 16;

static unsigned configured_thread_count() {
    const unsigned hardware = std::max(1u, std::thread::hardware_concurrency());
    const char *requested = std::getenv("RYUGU_THREADS");
    if (requested && std::strcmp(requested, "all") == 0) return hardware;
    unsigned threads = std::min(hardware, 4u);
    if (requested) {
        char *end = nullptr;
        const unsigned long value = std::strtoul(requested, &end, 10);
        if (end != requested && value > 0) {
            threads = static_cast<unsigned>(std::min<unsigned long>(value, hardware));
        }
    }
    return std::max(1u, threads);
}
/**
 * Voxel grid. This is the one accuracy dial the direct sum has: cells are
 * `extent/grid` wide, and the monopole-per-cell error falls roughly like the
 * cell size to the 1.7. Measured against the RT-FP Cauchy record at a 16 m
 * observation surface (same norm, same faces), the discretisation part of the
 * gap is 0.711 % median at 128³ (7.9 m cells) and 0.297 % at 192³ (5.253 m
 * cells). The same comparison at a 1 mm standoff is 40.6 % median, because the
 * surface is then inside a cell. Override with `--grid`.
 */
static constexpr size_t DEFAULT_GRID = 192;

struct Mesh {
    std::vector<double> xyz;
    std::vector<uint32_t> faces;
};

struct Kernel {
    double c[3];
    double sigma;
    double w;
    double alpha;
};

struct DensityModel {
    std::vector<Kernel> kernels;
    double total_mass_target = 4.50e11;
    double bulk_density_ref = 1190.0;
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
        const double cx = e1[1] * e2[2] - e1[2] * e2[1];
        const double cy = e1[2] * e2[0] - e1[0] * e2[2];
        const double cz = e1[0] * e2[1] - e1[1] * e2[0];
        nx[i0] += cx;
        ny[i0] += cy;
        nz[i0] += cz;
        nx[i1] += cx;
        ny[i1] += cy;
        nz[i1] += cz;
        nx[i2] += cx;
        ny[i2] += cy;
        nz[i2] += cz;
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

static void write_header(std::fstream &out, uint32_t n_faces, uint32_t completed, float s_min,
                         float s_max) {
    out.seekp(0);
    // Offset 16 is reserved in v5: store the observation height (mm as f32) so the
    // viewer can tell which surface the record was baked at.
    const float standoff_mm = static_cast<float>(g_standoff_mm);
    out.write(reinterpret_cast<const char *>(&MAGIC), 4);
    out.write(reinterpret_cast<const char *>(&VERSION), 4);
    out.write(reinterpret_cast<const char *>(&n_faces), 4);
    out.write(reinterpret_cast<const char *>(&completed), 4);
    out.write(reinterpret_cast<const char *>(&standoff_mm), 4);
    out.write(reinterpret_cast<const char *>(&s_min), 4);
    out.write(reinterpret_cast<const char *>(&s_max), 4);
}

static bool load_checkpoint(const char *path, size_t nf, std::vector<float> &face_scalar,
                            float &s_min, float &s_max, size_t &n_done) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    uint32_t magic = 0, version = 0, n_mesh = 0, completed = 0, pad = 0;
    float lo = 0, hi = 0;
    in.read(reinterpret_cast<char *>(&magic), 4);
    in.read(reinterpret_cast<char *>(&version), 4);
    in.read(reinterpret_cast<char *>(&n_mesh), 4);
    in.read(reinterpret_cast<char *>(&completed), 4);
    in.read(reinterpret_cast<char *>(&pad), 4);
    in.read(reinterpret_cast<char *>(&lo), 4);
    in.read(reinterpret_cast<char *>(&hi), 4);
    if (!in || magic != MAGIC || version != VERSION || n_mesh != nf) return false;
    face_scalar.assign(nf, std::nanf(""));
    in.read(reinterpret_cast<char *>(face_scalar.data()),
            static_cast<std::streamsize>(nf * sizeof(float)));
    if (!in) return false;
    n_done = 0;
    s_min = 1e30f;
    s_max = -1e30f;
    for (size_t i = 0; i < nf; ++i) {
        const float s = face_scalar[i];
        if (std::isfinite(s)) {
            ++n_done;
            s_min = std::min(s_min, s);
            s_max = std::max(s_max, s);
        }
    }
    if (n_done == 0) {
        s_min = 1e30f;
        s_max = -1e30f;
    }
    (void)completed;
    return true;
}

static bool write_order(const char *path, const std::vector<uint32_t> &order) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    const uint32_t n = static_cast<uint32_t>(order.size());
    out.write(reinterpret_cast<const char *>(&n), 4);
    out.write(reinterpret_cast<const char *>(order.data()), static_cast<std::streamsize>(n * 4));
    return static_cast<bool>(out);
}

static bool load_order(const char *path, size_t nf, std::vector<uint32_t> &order) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    uint32_t n = 0;
    in.read(reinterpret_cast<char *>(&n), 4);
    if (!in || n != nf) return false;
    order.resize(nf);
    in.read(reinterpret_cast<char *>(order.data()), static_cast<std::streamsize>(nf * 4));
    if (!in) return false;
    std::vector<uint8_t> seen(nf, 0);
    for (uint32_t f : order) {
        if (f >= nf || seen[f]) return false;
        seen[f] = 1;
    }
    return true;
}

/**
 * Read one number at `at`. Returns false when there is none.
 */
static bool toml_number(const std::string &s, size_t at, double &out, size_t *next = nullptr) {
    size_t q = at;
    while (q < s.size() && std::isspace(static_cast<unsigned char>(s[q]))) ++q;
    char *end = nullptr;
    out = std::strtod(s.c_str() + q, &end);
    if (end == s.c_str() + q) return false;
    if (next) *next = static_cast<size_t>(end - s.c_str());
    return true;
}

/**
 * Position just past `key` in `key = ...`, or `std::string::npos`.
 *
 * The key has to be a whole token: `c` must not match the `c` of `core`, which
 * is what makes the search safe on a text that still contains role names.
 */
static size_t toml_key(const std::string &s, const char *key) {
    const std::string pat(key);
    for (size_t pos = s.find(pat); pos != std::string::npos; pos = s.find(pat, pos + 1)) {
        const bool left_ok = pos == 0
            || !(std::isalnum(static_cast<unsigned char>(s[pos - 1])) || s[pos - 1] == '_');
        size_t q = pos + pat.size();
        if (!left_ok || q >= s.size()) continue;
        if (!std::isspace(static_cast<unsigned char>(s[q])) && s[q] != '=') continue;
        while (q < s.size() && std::isspace(static_cast<unsigned char>(s[q]))) ++q;
        if (q < s.size() && s[q] == '=') return q + 1;
    }
    return std::string::npos;
}

/** `key = <number>` out of one kernel record. */
static bool record_scalar(const std::string &s, const char *key, double &out) {
    const size_t after = toml_key(s, key);
    return after != std::string::npos && toml_number(s, after, out);
}

/** `key = [x, y, z]` out of one kernel record. */
static bool record_vector3(const std::string &s, const char *key, double out[3]) {
    size_t q = toml_key(s, key);
    if (q == std::string::npos) return false;
    while (q < s.size() && std::isspace(static_cast<unsigned char>(s[q]))) ++q;
    if (q >= s.size() || s[q] != '[') return false;
    ++q;
    for (int n = 0; n < 3; ++n) {
        while (q < s.size()
               && (std::isspace(static_cast<unsigned char>(s[q])) || s[q] == ','))
            ++q;
        if (!toml_number(s, q, out[n], &q)) return false;
    }
    return true;
}

/**
 * Reader for the Cauchy-kernel density file.
 *
 * Deliberately minimal: the Rust host parses the TOML properly, but the two
 * layouts this project writes for the same `kernels` table are simple enough
 * that a full TOML dependency is not needed here. Both are accepted:
 *
 *   [[kernels]]                     kernels = [
 *   c = [..]                          { c = [..], sigma = .., w = .. },
 *   sigma = ..                      ]
 *
 * Comments are removed first; each kernel is then reduced to the `key = value`
 * pairs inside its braces, or between its table header and the next one.
 */
static DensityModel load_cauchy_toml(const char *path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error(std::string("cannot open density toml: ") + path);
    const std::string raw((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());

    std::string text;
    text.reserve(raw.size());
    bool quoted = false;
    for (size_t i = 0; i < raw.size(); ++i) {
        const char c = raw[i];
        if (quoted) {
            text.push_back(c);
            if (c == '\\' && i + 1 < raw.size()) {
                text.push_back(raw[++i]);
            } else if (c == '"') {
                quoted = false;
            }
            continue;
        }
        if (c == '"') {
            quoted = true;
            text.push_back(c);
        } else if (c == '#') {
            while (i < raw.size() && raw[i] != '\n') ++i;
            text.push_back('\n');
        } else {
            text.push_back(c);
        }
    }

    DensityModel model;
    double value = 0.0;
    if (record_scalar(text, "total_mass_target", value)) model.total_mass_target = value;
    if (record_scalar(text, "bulk_density_ref", value)) model.bulk_density_ref = value;

    auto push = [&model](const std::string &rec) {
        Kernel k{};
        if (!record_vector3(rec, "c", k.c)) return;
        if (!record_scalar(rec, "sigma", k.sigma)) return;
        if (!record_scalar(rec, "w", k.w)) return;
        if (!record_scalar(rec, "alpha", k.alpha) || k.alpha <= 0.0) k.alpha = 1.0;
        // A vanishing or negative sigma would make the kernel a constant or make
        // the density grow with distance; neither is a density this file means.
        if (!(k.sigma > 0.0)) return;
        model.kernels.push_back(k);
    };

    // Inline array form: every `{ ... }` group.
    for (size_t i = text.find('{'); i != std::string::npos; i = text.find('{', i + 1)) {
        const size_t close = text.find('}', i);
        if (close == std::string::npos) break;
        push(text.substr(i + 1, close - i - 1));
        i = close;
    }

    // Table form: `[[kernels]]` up to the next table header.
    for (size_t i = text.find("[[kernels]]"); i != std::string::npos;
         i = text.find("[[kernels]]", i + 1)) {
        const size_t body = i + std::strlen("[[kernels]]");
        size_t end = text.size();
        for (size_t j = body; j + 1 < text.size(); ++j) {
            if (text[j] == '\n' && text[j + 1] == '[') {
                end = j;
                break;
            }
        }
        push(text.substr(body, end - body));
    }

    if (model.kernels.empty()) throw std::runtime_error("no Cauchy kernels in toml");
    return model;
}

static double density_at_m(const DensityModel &model, double x, double y, double z) {
    constexpr double M_TO_KM = 0.001;
    const double px = x * M_TO_KM, py = y * M_TO_KM, pz = z * M_TO_KM;
    double rho = 0.0;
    for (const auto &k : model.kernels) {
        const double dx = px - k.c[0];
        const double dy = py - k.c[1];
        const double dz = pz - k.c[2];
        const double r2 = dx * dx + dy * dy + dz * dz;
        const double base = 1.0 + (k.sigma * k.sigma) * r2;
        rho += k.w * std::pow(base, -k.alpha);
    }
    return rho;
}

/** Möller–Trumbore; returns t if hit with t>eps, else -1. */
static double ray_tri_t(const double o[3], const double d[3], const double *v0, const double *v1,
                        const double *v2) {
    constexpr double EPS = 1e-12;
    const double e1[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
    const double e2[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
    const double h[3] = {d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2],
                         d[0] * e2[1] - d[1] * e2[0]};
    const double a = e1[0] * h[0] + e1[1] * h[1] + e1[2] * h[2];
    if (a > -EPS && a < EPS) return -1.0;
    const double f = 1.0 / a;
    const double s[3] = {o[0] - v0[0], o[1] - v0[1], o[2] - v0[2]};
    const double u = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);
    if (u < 0.0 || u > 1.0) return -1.0;
    const double q[3] = {s[1] * e1[2] - s[2] * e1[1], s[2] * e1[0] - s[0] * e1[2],
                         s[0] * e1[1] - s[1] * e1[0]};
    const double v = f * (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]);
    if (v < 0.0 || u + v > 1.0) return -1.0;
    const double t = f * (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]);
    return t > EPS ? t : -1.0;
}

static bool point_inside_mesh(const Mesh &mesh, const double p[3]) {
    // Odd-parity ray cast along +X.
    const double dir[3] = {1.0, 0.0, 0.0};
    int hits = 0;
    for (size_t f = 0; f + 2 < mesh.faces.size(); f += 3) {
        const double *v0 = &mesh.xyz[3 * mesh.faces[f]];
        const double *v1 = &mesh.xyz[3 * mesh.faces[f + 1]];
        const double *v2 = &mesh.xyz[3 * mesh.faces[f + 2]];
        if (ray_tri_t(p, dir, v0, v1, v2) > 0.0) ++hits;
    }
    return (hits & 1) != 0;
}

/** Uniform grid hash for fast +/-X ray-triangle tests (odd parity). */
struct TriAccel {
    double bmin[3], bmax[3];
    int nx = 64, ny = 64, nz = 64;
    std::vector<std::vector<uint32_t>> cells;
    const Mesh *mesh = nullptr;

    void build(const Mesh &m) {
        mesh = &m;
        bmin[0] = bmin[1] = bmin[2] = 1e300;
        bmax[0] = bmax[1] = bmax[2] = -1e300;
        const size_t nv = m.xyz.size() / 3;
        for (size_t i = 0; i < nv; ++i) {
            for (int k = 0; k < 3; ++k) {
                bmin[k] = std::min(bmin[k], m.xyz[3 * i + k]);
                bmax[k] = std::max(bmax[k], m.xyz[3 * i + k]);
            }
        }
        for (int k = 0; k < 3; ++k) {
            const double pad = 1e-3 * (bmax[k] - bmin[k] + 1.0);
            bmin[k] -= pad;
            bmax[k] += pad;
        }
        cells.assign(static_cast<size_t>(nx) * ny * nz, {});
        const size_t nf = m.faces.size() / 3;
        for (size_t fi = 0; fi < nf; ++fi) {
            const double *v0 = &m.xyz[3 * m.faces[3 * fi]];
            const double *v1 = &m.xyz[3 * m.faces[3 * fi + 1]];
            const double *v2 = &m.xyz[3 * m.faces[3 * fi + 2]];
            double tmin[3], tmax[3];
            for (int k = 0; k < 3; ++k) {
                tmin[k] = std::min({v0[k], v1[k], v2[k]});
                tmax[k] = std::max({v0[k], v1[k], v2[k]});
            }
            int i0 = std::max(0, std::min(nx - 1, (int)((tmin[0] - bmin[0]) / (bmax[0] - bmin[0]) * nx)));
            int i1 = std::max(0, std::min(nx - 1, (int)((tmax[0] - bmin[0]) / (bmax[0] - bmin[0]) * nx)));
            int j0 = std::max(0, std::min(ny - 1, (int)((tmin[1] - bmin[1]) / (bmax[1] - bmin[1]) * ny)));
            int j1 = std::max(0, std::min(ny - 1, (int)((tmax[1] - bmin[1]) / (bmax[1] - bmin[1]) * ny)));
            int k0 = std::max(0, std::min(nz - 1, (int)((tmin[2] - bmin[2]) / (bmax[2] - bmin[2]) * nz)));
            int k1 = std::max(0, std::min(nz - 1, (int)((tmax[2] - bmin[2]) / (bmax[2] - bmin[2]) * nz)));
            for (int kk = k0; kk <= k1; ++kk)
                for (int jj = j0; jj <= j1; ++jj)
                    for (int ii = i0; ii <= i1; ++ii)
                        cells[static_cast<size_t>((kk * ny + jj) * nx + ii)].push_back(static_cast<uint32_t>(fi));
        }
    }

    bool inside(const double p[3]) const {
        const double dir[3] = {1.0, 0.0, 0.0};
        // Walk grid cells along +X from p.
        int hits = 0;
        std::vector<uint8_t> seen(mesh->faces.size() / 3, 0);
        const double inv_dx = nx / (bmax[0] - bmin[0]);
        const double inv_dy = ny / (bmax[1] - bmin[1]);
        const double inv_dz = nz / (bmax[2] - bmin[2]);
        int j = std::max(0, std::min(ny - 1, (int)((p[1] - bmin[1]) * inv_dy)));
        int k = std::max(0, std::min(nz - 1, (int)((p[2] - bmin[2]) * inv_dz)));
        int i_start = std::max(0, std::min(nx - 1, (int)((p[0] - bmin[0]) * inv_dx)));
        for (int i = i_start; i < nx; ++i) {
            const auto &bucket = cells[static_cast<size_t>((k * ny + j) * nx + i)];
            for (uint32_t fi : bucket) {
                if (seen[fi]) continue;
                seen[fi] = 1;
                const double *v0 = &mesh->xyz[3 * mesh->faces[3 * fi]];
                const double *v1 = &mesh->xyz[3 * mesh->faces[3 * fi + 1]];
                const double *v2 = &mesh->xyz[3 * mesh->faces[3 * fi + 2]];
                if (ray_tri_t(p, dir, v0, v1, v2) > 0.0) ++hits;
            }
        }
        return (hits & 1) != 0;
    }
};

static void build_mascons(const Mesh &mesh, const DensityModel &density, size_t grid,
                          std::vector<double> &xyz, std::vector<double> &mass) {
    double bmin[3] = {1e300, 1e300, 1e300};
    double bmax[3] = {-1e300, -1e300, -1e300};
    const size_t nv = mesh.xyz.size() / 3;
    for (size_t i = 0; i < nv; ++i) {
        for (int k = 0; k < 3; ++k) {
            bmin[k] = std::min(bmin[k], mesh.xyz[3 * i + k]);
            bmax[k] = std::max(bmax[k], mesh.xyz[3 * i + k]);
        }
    }
    // Slight inflate so boundary cells still sample interior.
    const double pad = 0.5 * std::min({bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]})
        / static_cast<double>(grid);
    for (int k = 0; k < 3; ++k) {
        bmin[k] -= pad;
        bmax[k] += pad;
    }
    const double dx = (bmax[0] - bmin[0]) / static_cast<double>(grid);
    const double dy = (bmax[1] - bmin[1]) / static_cast<double>(grid);
    const double dz = (bmax[2] - bmin[2]) / static_cast<double>(grid);
    const double vol = dx * dy * dz;

    TriAccel accel;
    accel.build(mesh);
    std::printf("voxel accel grid %dx%dx%d ready\n", accel.nx, accel.ny, accel.nz);
    std::fflush(stdout);

    xyz.clear();
    mass.clear();
    xyz.reserve(grid * grid * grid / 4);
    mass.reserve(grid * grid * grid / 4);

    size_t tested = 0, kept = 0;
    for (size_t iz = 0; iz < grid; ++iz) {
        for (size_t iy = 0; iy < grid; ++iy) {
            for (size_t ix = 0; ix < grid; ++ix) {
                const double p[3] = {
                    bmin[0] + (ix + 0.5) * dx,
                    bmin[1] + (iy + 0.5) * dy,
                    bmin[2] + (iz + 0.5) * dz,
                };
                ++tested;
                if (!accel.inside(p)) continue;
                const double rho = density_at_m(density, p[0], p[1], p[2]);
                xyz.push_back(p[0]);
                xyz.push_back(p[1]);
                xyz.push_back(p[2]);
                mass.push_back(rho * vol);
                ++kept;
            }
        }
        std::printf("voxelize z %zu / %zu · inside %zu\n", iz + 1, grid, kept);
        std::fflush(stdout);
    }

    double msum = 0.0;
    for (double m : mass) msum += m;
    std::printf("mascons: %zu / %zu cells, raw_mass=%.6e kg, target=%.6e kg, cell=%.3f m\n", kept,
                tested, msum, density.total_mass_target, std::cbrt(vol));
    if (std::fabs(msum) > 1e-30 && density.total_mass_target > 0.0) {
        const double scale = density.total_mass_target / msum;
        for (double &m : mass) m *= scale;
        std::printf("rescaled mascon masses by %.6f → total=%.6e kg\n", scale,
                    density.total_mass_target);
    }
    std::fflush(stdout);
}

int main(int argc, char **argv) {
    const char *obj_path = nullptr;
    const char *out_path = "assets/records/mascon_faces.bin";
    const char *order_path = "assets/records/.mascon_order.bin";
    const char *density_path = "assets/density/cauchy.toml";
    bool resume = false;
    size_t grid = DEFAULT_GRID;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--obj" && i + 1 < argc) obj_path = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (a == "--order" && i + 1 < argc) order_path = argv[++i];
        else if (a == "--density" && i + 1 < argc) density_path = argv[++i];
        else if (a == "--grid" && i + 1 < argc) grid = static_cast<size_t>(std::strtoul(argv[++i], nullptr, 10));
        else if (a == "--standoff-mm" && i + 1 < argc) {
            const double mm = std::strtod(argv[++i], nullptr);
            if (mm >= STANDOFF_MIN_MM && mm <= STANDOFF_MAX_MM) {
                g_standoff_mm = mm;
                g_standoff_m = mm * 1.0e-3;
            }
        }
        else if (a == "--resume") resume = true;
        else if (a == "--help") {
            std::puts(
                "ryugu_mascon_bake --obj mesh.obj [--density toml] [--grid N] [--out path] "
                "[--order path] [--standoff-mm 1..32000] [--resume]");
            return 0;
        }
    }
    if (!obj_path) {
        std::fprintf(stderr, "missing --obj\n");
        return 1;
    }
    if (grid < 4 || grid > 256) {
        std::fprintf(stderr, "grid must be in [4,256], got %zu\n", grid);
        return 1;
    }

    Mesh mesh = load_obj_km(obj_path, 1000.0);
    const size_t nv = mesh.xyz.size() / 3;
    const size_t nf = mesh.faces.size() / 3;
    DensityModel density = load_cauchy_toml(density_path);
    std::printf("mesh: %zu verts, %zu faces · Cauchy kernels=%zu · grid=%zu^3\n", nv, nf,
                density.kernels.size(), grid);
    std::printf("observation standoff: %.3f mm\n", g_standoff_mm);
    std::fflush(stdout);

    std::vector<double> src_xyz, src_mass;
    build_mascons(mesh, density, grid, src_xyz, src_mass);
    const uint64_t nsrc = src_mass.size();
    if (nsrc == 0) {
        std::fprintf(stderr, "no interior mascons\n");
        return 1;
    }

    // Sanity: one direct_sum call (Zig ABI).
    {
        double acc[3] = {}, pot = 0.0;
        const double origin[3] = {0, 0, 0};
        if (ryugu_direct_sum_eval(src_xyz.data(), src_mass.data(), nsrc, origin, acc, &pot) != 0) {
            std::fprintf(stderr, "ryugu_direct_sum_eval failed\n");
            return 1;
        }
        std::printf("direct_sum@origin |a|=%.6e pot=%.6e (zig ABI ok)\n",
                    std::sqrt(acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2]), pot);
        std::fflush(stdout);
    }

    std::vector<double> nx, ny, nz;
    accumulate_vertex_normals(mesh, nx, ny, nz);

    std::vector<uint8_t> vert_ready(nv, 0);
    std::vector<double> H_vert(nv * 6, 0.0);

    std::vector<float> face_scalar(nf, std::nanf(""));
    float s_min = 1e30f, s_max = -1e30f;
    size_t n_done = 0;
    std::fstream out;

    if (resume) {
        if (!load_checkpoint(out_path, nf, face_scalar, s_min, s_max, n_done)) {
            std::fprintf(stderr, "resume failed: no valid checkpoint at %s\n", out_path);
            return 1;
        }
        out.open(out_path, std::ios::binary | std::ios::in | std::ios::out);
        if (!out) {
            std::fprintf(stderr, "cannot reopen %s for resume\n", out_path);
            return 1;
        }
        std::printf("RESUME from %zu / %zu faces\n", n_done, nf);
        std::fflush(stdout);
        if (n_done >= nf) {
            write_header(out, static_cast<uint32_t>(nf), static_cast<uint32_t>(nf), s_min, s_max);
            out.flush();
            std::printf("wrote %zu dense face scalars to %s (s_min=%g s_max=%g)\n", nf, out_path,
                        s_min, s_max);
            return 0;
        }
    } else {
        out.open(out_path, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
        if (!out) {
            {
                std::ofstream create(out_path, std::ios::binary | std::ios::trunc);
                if (!create) {
                    std::fprintf(stderr, "cannot write %s\n", out_path);
                    return 1;
                }
            }
            out.open(out_path, std::ios::binary | std::ios::in | std::ios::out);
        }
        if (!out) {
            std::fprintf(stderr, "cannot open %s for update\n", out_path);
            return 1;
        }
        write_header(out, static_cast<uint32_t>(nf), 0, s_min, s_max);
        out.write(reinterpret_cast<const char *>(face_scalar.data()), nf * sizeof(float));
        out.flush();
        n_done = 0;
    }

    std::vector<uint32_t> order;
    if (resume && load_order(order_path, nf, order)) {
        std::printf("using saved face order from %s\n", order_path);
    } else {
        order.resize(nf);
        for (size_t i = 0; i < nf; ++i) order[i] = static_cast<uint32_t>(i);
        std::random_device rd;
        std::mt19937 rng(rd());
        std::shuffle(order.begin(), order.end(), rng);
        if (!write_order(order_path, order)) {
            std::fprintf(stderr, "warning: cannot write order file %s\n", order_path);
        }
    }

    std::vector<uint32_t> todo;
    todo.reserve(nf - n_done);
    for (uint32_t f : order) {
        if (!std::isfinite(face_scalar[f])) todo.push_back(f);
    }
    std::printf("baking %zu remaining faces · mascon brute force (%llu sources)\n", todo.size(),
                static_cast<unsigned long long>(nsrc));
    std::fflush(stdout);

    unsigned nthreads = configured_thread_count();
    std::printf("parallel vert hessian: %u threads, window=%zu faces\n", nthreads, FACE_WINDOW);
    std::fflush(stdout);

    auto eval_vert = [&](uint32_t v) -> bool {
        if (vert_ready[v]) return true;
        const double p[3] = {
            mesh.xyz[3 * v + 0] + g_standoff_m * nx[v],
            mesh.xyz[3 * v + 1] + g_standoff_m * ny[v],
            mesh.xyz[3 * v + 2] + g_standoff_m * nz[v],
        };
        if (ryugu_direct_sum_hessian_eval(src_xyz.data(), src_mass.data(), nsrc, p,
                                          H_vert.data() + 6 * v)
            != 0) {
            return false;
        }
        vert_ready[v] = 1;
        return true;
    };

    size_t step = 0;
    for (size_t base = 0; base < todo.size(); base += FACE_WINDOW) {
        const size_t end = std::min(base + FACE_WINDOW, todo.size());
        std::vector<uint32_t> miss;
        {
            std::unordered_set<uint32_t> uniq;
            for (size_t i = base; i < end; ++i) {
                const uint32_t f = todo[i];
                for (int k = 0; k < 3; ++k) {
                    const uint32_t v = mesh.faces[3 * f + k];
                    if (!vert_ready[v]) uniq.insert(v);
                }
            }
            miss.assign(uniq.begin(), uniq.end());
        }

        if (!miss.empty()) {
            std::atomic<size_t> next_i{0};
            std::atomic<int> fail{0};
            std::vector<std::thread> pool;
            pool.reserve(nthreads);
            for (unsigned t = 0; t < nthreads; ++t) {
                pool.emplace_back([&]() {
                    while (fail.load() == 0) {
                        const size_t i = next_i.fetch_add(1);
                        if (i >= miss.size()) break;
                        if (!eval_vert(miss[i])) {
                            fail.store(1);
                            break;
                        }
                    }
                });
            }
            for (auto &th : pool) th.join();
            if (fail.load() != 0) {
                std::fprintf(stderr, "mascon hessian failed near face window %zu\n", base);
                return 1;
            }
        }

        for (size_t i = base; i < end; ++i) {
            const uint32_t f = todo[i];
            const uint32_t v0 = mesh.faces[3 * f + 0];
            const uint32_t v1 = mesh.faces[3 * f + 1];
            const uint32_t v2 = mesh.faces[3 * f + 2];
            double Hbar[6];
            for (int k = 0; k < 6; ++k) {
                Hbar[k] = (H_vert[6 * v0 + k] + H_vert[6 * v1 + k] + H_vert[6 * v2 + k]) / 3.0;
            }
            const float s = static_cast<float>(frobenius6(Hbar));
            face_scalar[f] = s;
            s_min = std::min(s_min, s);
            s_max = std::max(s_max, s);
            ++n_done;
            ++step;

            const uint64_t scalar_off = 28ull + static_cast<uint64_t>(f) * 4ull;
            out.seekp(static_cast<std::streamoff>(scalar_off));
            out.write(reinterpret_cast<const char *>(&s), 4);

            if (step % FLUSH_EVERY == 0 || n_done == nf) {
                write_header(out, static_cast<uint32_t>(nf), static_cast<uint32_t>(n_done), s_min,
                             s_max);
                out.flush();
                std::printf("PROGRESS %zu %zu\n", n_done, nf);
                std::printf("faces %zu / %zu\n", n_done, nf);
                std::fflush(stdout);
            }
        }
        std::this_thread::yield();
    }

    write_header(out, static_cast<uint32_t>(nf), static_cast<uint32_t>(nf), s_min, s_max);
    out.flush();
    std::printf("wrote %zu dense face scalars to %s (s_min=%g s_max=%g)\n", nf, out_path, s_min,
                s_max);
    return 0;
}
