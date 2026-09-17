# Contributing

## Boundaries

Keep the five computation pipelines independent. Shared code may manage WebGPU devices, buffers, progress, RHGF records, and error handling, but an algorithm must select its own runtime input builder and WGSL entry points.

Keep rendering unified. Solver-specific code ends when it emits an RHGF v5 face-scalar record; Bevy mesh updates, color windows, and face painting must remain shared.

The Bun server is static-only. Numerical work belongs in Rust/WASM or WGSL, never in TypeScript or JavaScript.

## Layout

- `src/rust/compute/source/`: rebuild solver inputs from GLB and density TOML
- `src/rust/compute/browser/`: browser WebGPU execution
- `src/rust/compute/native/`: native reference and CLI support
- `src/rust/viewer/frontend/`: Vue-facing state and persistence
- `src/rust/viewer/render/`: shared Bevy rendering
- `src/wgsl/compute/`: algorithm kernels
- `src/wgsl/viewer/`: shared viewer kernels
- `src/web/`: Vue, Bun static server, and build scripts
- `src/c++/`: optional CMake-managed reference bridge

Do not add nested Cargo manifests, precomputed pipeline buffers, result records, or server-side solver endpoints.

## Checks

```sh
make check
make test
```

For browser packaging:

```sh
make install
make build
make serve
```

For the optional ESA bridge:

```sh
make cpp
cargo run --release --features esa --bin rtfp-bake -- --selftest
```

Keep files scoped by responsibility. Prefer established crates and packages over local infrastructure when they reduce code without hiding the numerical method.
