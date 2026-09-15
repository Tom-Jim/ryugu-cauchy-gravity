//! Link the vendored ESA polyhedral-gravity library (analytic uniform-density
//! Hessian) that the C++ bakes already use. `bakes/build` is produced by
//! `bun run bakes:build` (and `bakes/esa-build` by `bun run esa:configure`).

use std::path::{Path, PathBuf};

fn main() {
    let root: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crate lives in <project>/bakes/rtfp")
        .to_path_buf();
    let bridge = root.join("bakes/build");
    let esa = root.join("bakes/esa-build");

    for dir in [
        bridge.clone(),
        esa.clone(),
        esa.join("_deps/yaml-cpp-build"),
        esa.join("_deps/spdlog-build"),
    ] {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }

    println!("cargo:rustc-link-lib=static=esa_pg_bridge");
    println!("cargo:rustc-link-lib=static=tetgen_lib");
    println!("cargo:rustc-link-lib=static=yaml-cpp");
    println!("cargo:rustc-link-lib=static=spdlog");
    println!("cargo:rustc-link-lib=dylib=c++");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", bridge.join("libesa_pg_bridge.a").display());
}
