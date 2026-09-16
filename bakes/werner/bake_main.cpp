/** Progressive random-order per-face gravity-gradient bake (ESA Tsoulis).
 *  Supports --resume: keep existing gradient_faces.bin and finish unfinished faces.
 */
#include "esa_bridge.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

static constexpr double RYUGU_BULK_DENSITY_KG_M3 = 1190.0;
/** Observation-surface height bounds (mm), matching the viewer slider. */
static constexpr double STANDOFF_MIN_MM = 1.0;
static constexpr double STANDOFF_MAX_MM = 32000.0;
static constexpr double STANDOFF_DEFAULT_MM = 16000.0;
/** Height the current run evaluates at; overridden by `--standoff-mm`. */
static double g_standoff_mm = STANDOFF_DEFAULT_MM;
static double g_standoff_m = STANDOFF_DEFAULT_MM * 1.0e-3;
static constexpr uint32_t MAGIC = 0x52484746; // 'RHGF'
static constexpr uint32_t VERSION = 5;
/** Faces per parallel window: eval missing verts with multi-handle threads, then flush. */
static constexpr size_t FACE_WINDOW = 128;
static constexpr size_t FLUSH_EVERY = 64;

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

static void write_header(std::fstream &out, uint32_t n_faces, uint32_t completed, float s_min, float s_max) {
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

/** Load v5 checkpoint; returns true if usable. */
static bool load_checkpoint(
    const char *path,
    size_t nf,
    std::vector<float> &face_scalar,
    float &s_min,
    float &s_max,
    size_t &n_done) {
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
    in.read(reinterpret_cast<char *>(face_scalar.data()), static_cast<std::streamsize>(nf * sizeof(float)));
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

int main(int argc, char **argv) {
    const char *obj_path = nullptr;
    const char *out_path = "assets/records/gradient_faces.bin";
    const char *order_path = "assets/records/.bake_order.bin";
    bool resume = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--obj" && i + 1 < argc) obj_path = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (a == "--order" && i + 1 < argc) order_path = argv[++i];
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
                "ryugu_gradient_bake --obj mesh.obj [--out path] [--order path] "
                "[--standoff-mm 1..32000] [--resume]");
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
    std::printf("observation standoff: %.3f mm\n", g_standoff_mm);
    std::printf("mesh: %zu vertices, %zu faces (meters), density=%.0f\n", nv, nf, RYUGU_BULK_DENSITY_KG_M3);

    EsaPgHandle *handle = esa_pg_create(
        mesh.xyz.data(), nv, mesh.faces.data(), nf, RYUGU_BULK_DENSITY_KG_M3, 0);
    if (!handle) {
        std::fprintf(stderr, "esa_pg_create failed\n");
        return 1;
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
            esa_pg_destroy(handle);
            return 1;
        }
        out.open(out_path, std::ios::binary | std::ios::in | std::ios::out);
        if (!out) {
            std::fprintf(stderr, "cannot reopen %s for resume\n", out_path);
            esa_pg_destroy(handle);
            return 1;
        }
        std::printf("RESUME from %zu / %zu faces\n", n_done, nf);
        std::fflush(stdout);
        if (n_done >= nf) {
            write_header(out, static_cast<uint32_t>(nf), static_cast<uint32_t>(nf), s_min, s_max);
            out.flush();
            esa_pg_destroy(handle);
            std::printf("wrote %zu dense face scalars to %s (s_min=%g s_max=%g)\n", nf, out_path, s_min, s_max);
            return 0;
        }
    } else {
        out.open(out_path, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
        if (!out) {
            {
                std::ofstream create(out_path, std::ios::binary | std::ios::trunc);
                if (!create) {
                    std::fprintf(stderr, "cannot write %s\n", out_path);
                    esa_pg_destroy(handle);
                    return 1;
                }
            }
            out.open(out_path, std::ios::binary | std::ios::in | std::ios::out);
        }
        if (!out) {
            std::fprintf(stderr, "cannot open %s for update\n", out_path);
            esa_pg_destroy(handle);
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
        } else {
            std::printf("saved face order to %s\n", order_path);
        }
    }
    std::fflush(stdout);

    std::vector<uint32_t> todo;
    todo.reserve(nf - n_done);
    for (uint32_t f : order) {
        if (!std::isfinite(face_scalar[f])) todo.push_back(f);
    }
    std::printf("baking %zu remaining faces (progressive%s)\n", todo.size(), resume ? ", resume" : "");
    std::fflush(stdout);

    // Multi-handle workers: each ESA GravityEvaluable is NOT shared across threads.
    unsigned nthreads = configured_thread_count();
    std::vector<EsaPgHandle *> handles(nthreads, nullptr);
    handles[0] = handle;
    for (unsigned t = 1; t < nthreads; ++t) {
        handles[t] = esa_pg_create(
            mesh.xyz.data(), nv, mesh.faces.data(), nf, RYUGU_BULK_DENSITY_KG_M3, 0);
        if (!handles[t]) {
            std::fprintf(stderr, "esa_pg_create worker %u failed; falling back to 1 thread\n", t);
            nthreads = 1;
            break;
        }
    }
    std::printf("parallel vert eval: %u threads, window=%zu faces\n", nthreads, FACE_WINDOW);
    std::fflush(stdout);

    auto eval_vert = [&](EsaPgHandle *h, uint32_t v) -> bool {
        if (vert_ready[v]) return true;
        const double p[3] = {
            mesh.xyz[3 * v + 0] + g_standoff_m * nx[v],
            mesh.xyz[3 * v + 1] + g_standoff_m * ny[v],
            mesh.xyz[3 * v + 2] + g_standoff_m * nz[v],
        };
        if (esa_pg_eval(h, p, nullptr, nullptr, H_vert.data() + 6 * v) != 0) return false;
        vert_ready[v] = 1;
        return true;
    };

    size_t step = 0;
    for (size_t base = 0; base < todo.size(); base += FACE_WINDOW) {
        const size_t end = std::min(base + FACE_WINDOW, todo.size());

        // Unique missing verts for this window.
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
                pool.emplace_back([&, t]() {
                    EsaPgHandle *h = handles[t];
                    while (fail.load() == 0) {
                        const size_t i = next_i.fetch_add(1);
                        if (i >= miss.size()) break;
                        if (!eval_vert(h, miss[i])) {
                            fail.store(1);
                            break;
                        }
                    }
                });
            }
            for (auto &th : pool) th.join();
            if (fail.load() != 0) {
                std::fprintf(stderr, "parallel vert eval failed near face window %zu\n", base);
                for (unsigned t = 1; t < nthreads; ++t) esa_pg_destroy(handles[t]);
                esa_pg_destroy(handle);
                return 1;
            }
        }

        // Write face scalars for this window (progressive paint).
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
                write_header(out, static_cast<uint32_t>(nf), static_cast<uint32_t>(n_done), s_min, s_max);
                out.flush();
                std::printf("PROGRESS %zu %zu\n", n_done, nf);
                std::printf("faces %zu / %zu\n", n_done, nf);
                std::fflush(stdout);
            }
        }
        std::this_thread::yield();
    }

    for (unsigned t = 1; t < nthreads; ++t) esa_pg_destroy(handles[t]);
    esa_pg_destroy(handle);
    write_header(out, static_cast<uint32_t>(nf), static_cast<uint32_t>(nf), s_min, s_max);
    out.flush();
    std::printf("wrote %zu dense face scalars to %s (s_min=%g s_max=%g)\n", nf, out_path, s_min, s_max);
    return 0;
}
