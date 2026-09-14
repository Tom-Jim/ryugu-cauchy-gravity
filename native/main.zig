//! Zig binding to ESA polyhedral gravity via the C ABI in esa_bridge.h.
//! Cube smoke proves the link; full mesh bake uses the C++ CLI that shares
//! the same ABI (`ryugu_gradient_bake_cpp`).

const std = @import("std");
const c = @cImport({
    @cInclude("esa_bridge.h");
});

pub fn main() !void {
    const vertices = [_]f64{
        -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1,
        -1, -1, 1,  1, -1, 1,  1, 1, 1,  -1, 1, 1,
    };
    const faces = [_]u32{
        1, 3, 2, 0, 3, 1, 0, 1, 5, 0, 5, 4, 0, 7, 3, 0, 4, 7,
        1, 2, 6, 1, 6, 5, 2, 3, 6, 3, 7, 6, 4, 5, 6, 4, 6, 7,
    };
    const handle = c.esa_pg_create(&vertices, 8, &faces, 12, 1.0, 1);
    if (handle == null) return error.EsaCreateFailed;
    defer c.esa_pg_destroy(handle);

    const p = [_]f64{ 0, 0, 0 };
    var V: f64 = 0;
    var g = [_]f64{ 0, 0, 0 };
    var H = [_]f64{ 0, 0, 0, 0, 0, 0 };
    if (c.esa_pg_eval(handle, &p, &V, &g, &H) != 0) return error.EsaEvalFailed;

    std.debug.print(
        "zig↔ESA ok: V={e} g=({e},{e},{e}) H6=({e},{e},{e},{e},{e},{e})\n",
        .{ V, g[0], g[1], g[2], H[0], H[1], H[2], H[3], H[4], H[5] },
    );
}
