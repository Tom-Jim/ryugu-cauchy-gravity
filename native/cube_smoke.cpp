#include "esa_bridge.h"

#include <array>
#include <cstdio>
#include <vector>

int main() {
    // Unit cube centered at origin, density 1, outwards faces (ESA README).
    const std::vector<double> vertices = {
        -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1,
    };
    const std::vector<uint32_t> faces = {
        1, 3, 2, 0, 3, 1, 0, 1, 5, 0, 5, 4, 0, 7, 3, 0, 4, 7,
        1, 2, 6, 1, 6, 5, 2, 3, 6, 3, 7, 6, 4, 5, 6, 4, 6, 7,
    };
    EsaPgHandle *handle = esa_pg_create(
        vertices.data(), vertices.size() / 3, faces.data(), faces.size() / 3, 1.0, 1);
    if (!handle) {
        std::fprintf(stderr, "esa_pg_create failed\n");
        return 1;
    }
    const double p[3] = {0, 0, 0};
    double V = 0;
    double g[3] = {};
    double H[6] = {};
    if (esa_pg_eval(handle, p, &V, g, H) != 0) {
        std::fprintf(stderr, "esa_pg_eval failed\n");
        esa_pg_destroy(handle);
        return 1;
    }
    std::printf("V=%.6g g=(%.6g,%.6g,%.6g) H6=(%.6g,%.6g,%.6g,%.6g,%.6g,%.6g)\n",
                V, g[0], g[1], g[2], H[0], H[1], H[2], H[3], H[4], H[5]);
    esa_pg_destroy(handle);
    return 0;
}
