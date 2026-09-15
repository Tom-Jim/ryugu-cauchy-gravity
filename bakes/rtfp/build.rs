//! Link the vendored ESA polyhedral-gravity library (analytic uniform-density
//! Hessian) that the C++ bakes already use. `bakes/build` is produced by
//! `bun run bakes:build` (and `bakes/esa-build` by `bun run esa:configure`).
//!
//! Everything here is behind the `esa` cargo feature: the RT-FP split and the
//! Carlson jump-surface solver do not need the library, so the default build
//! has no native dependencies at all and a bare checkout still compiles.

use std::path::{Path, PathBuf};

fn main() {
    if std::env::var_os("CARGO_FEATURE_ESA").is_none() {
        println!("cargo:rerun-if-changed=build.rs");
        return;
    }

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
    println!(
        "cargo:rerun-if-changed={}",
        bridge.join("libesa_pg_bridge.a").display()
    );
}
