const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const root_module = b.createModule(.{
        .root_source_file = b.path("main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
    });
    root_module.addIncludePath(b.path("."));
    root_module.addObjectFile(b.path("bridge-build/libesa_pg_bridge.a"));
    root_module.addObjectFile(b.path("esa-build/libtetgen_lib.a"));
    root_module.addObjectFile(b.path("esa-build/_deps/yaml-cpp-build/libyaml-cpp.a"));
    root_module.addObjectFile(b.path("esa-build/_deps/spdlog-build/libspdlog.a"));

    const exe = b.addExecutable(.{
        .name = "ryugu_esa_bind_smoke",
        .root_module = root_module,
    });
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    const run_step = b.step("run", "Run Zig↔ESA C ABI smoke test");
    run_step.dependOn(&run.step);
}
