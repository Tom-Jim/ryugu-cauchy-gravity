//! Validate every WGSL source and optionally link the ESA reference library.
//! `make cpp` produces `build/c++/libesa_pg_bridge.a` and its dependencies.
//!
//! Everything here is behind the `esa` cargo feature: the RT-FP split and the
//! Carlson jump-surface solver do not need the library, so the default build
//! has no native dependencies at all and a bare checkout still compiles.

use std::path::{Path, PathBuf};

fn validate_wgsl(dir: &Path) {
    for entry in std::fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("cannot read WGSL directory {}: {error}", dir.display()))
    {
        let path = entry.expect("WGSL directory entry").path();
        if path.is_dir() {
            validate_wgsl(&path);
        } else if path.extension().and_then(|value| value.to_str()) == Some("wgsl") {
            println!("cargo:rerun-if-changed={}", path.display());
            let source = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("cannot read WGSL {}: {error}", path.display()));
            let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|error| {
                panic!("WGSL parse failed for {}: {error}", path.display())
            });
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|error| {
                panic!("WGSL validation failed for {}: {error}", path.display())
            });
        }
    }
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    validate_wgsl(&root.join("src/wgsl"));

    if std::env::var_os("CARGO_FEATURE_ESA").is_none() {
        println!("cargo:rerun-if-changed=src/rust/build.rs");
        return;
    }

    let bridge = root.join("build/c++");

    for dir in [
        bridge.clone(),
        bridge.join("_deps/polyhedral_gravity-build/src/polyhedralGravity"),
        bridge.join("_deps/yaml-cpp-build"),
        bridge.join("_deps/spdlog-build"),
    ] {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }

    println!("cargo:rustc-link-lib=static=esa_pg_bridge");
    println!("cargo:rustc-link-lib=static=polyhedralGravity_lib");
    println!("cargo:rustc-link-lib=static=tetgen_lib");
    println!("cargo:rustc-link-lib=static=yaml-cpp");
    println!("cargo:rustc-link-lib=static=spdlog");
    match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("macos") => println!("cargo:rustc-link-lib=dylib=c++"),
        Ok("linux") => println!("cargo:rustc-link-lib=dylib=stdc++"),
        _ => {}
    }

    println!("cargo:rerun-if-changed=src/rust/build.rs");
    println!(
        "cargo:rerun-if-changed={}",
        bridge.join("libesa_pg_bridge.a").display()
    );
}
