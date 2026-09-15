# Contributing

Thanks for looking at this project. It is a numerical-methods comparison suite:
four independent solvers evaluate the gravity-gradient tensor of the same
density field over the same 196,608-face shape model of asteroid (162173)
Ryugu, and the viewer renders all four through one identical display path.
Because the output of the project is a comparison, the following rules matter
more here than they would in a typical application.

## Ground rules

1. **Comparability is the contract.** Every solver must write the *same* RHGF v5
   record over the *same* face ordering, at the *same* observation height, on
   the *same* mass budget. If a change breaks any of those four, it is a
   regression even if the new numbers look nicer.
2. **Parallelism for the GPU solvers lives in WGSL.** RT-FP and Carlson express
   every parallel step as a WGSL compute kernel; do not add host-side threading
   (`rayon`, `std::thread::spawn` in the Rust host, worker pools) to them. If a
   GPU solver is too slow, the answer is a better kernel, not a thread pool. The
   two C++ reference bakes are host programs and predate this rule; new work
   should not extend that pattern.
3. **Prefer a library to hand-rolled code** for anything that already has a
   mature implementation (linear algebra, BVH traversal, image encoding,
   elliptic integrals, CMake targets, HTTP serving).
4. **State the tolerance you actually achieved.** Every Carlson bake prints the
   representation tolerance it reached. Never quote a design tolerance as if it
   were the measured one.

## Repository layout

```
bakes/rtfp/          Rust bake host: RT-FP and Carlson solvers, WGSL kernels
bakes/mascon/        C++ mascon (voxel direct-sum) bake
bakes/werner/        C++ Werner / ESA polyhedral bake
bakes/common/        Thin C++ bridge onto the ESA polyhedral-gravity library
src/server/          Bun development server and one endpoint per solver
src/server/solvers/  One file per algorithm (boot / status / start)
src/viewer/          Bevy + WebGPU viewer (Rust, compiled to WASM)
src/web/             Single-page control surface
assets/density/      Density field definitions (TOML)
assets/models/       Rendered shape model (the raw OBJ stays outside the repo)
assets/records/      Finished per-face ‖H‖_F records, one per solver
tools/               Optional helper scripts
docs/images/         Figures referenced by the README
```

## Development setup

Required:

* Rust (stable) with the `wasm32-unknown-unknown` target
* [Bun](https://bun.sh) 1.2 or newer
* `wasm-pack`
* CMake 3.20+ and a C++20 compiler

The Hayabusa2 shape models are not versioned. Place the 200k-face OBJ at
`../Ryugu_wasm/assets/models/SHAPE_SFM_200k_v20180804.obj` relative to the
repository root, or point `BAKE_OBJ` at it.

## Common commands

```sh
bun run dev            # build the WASM viewer, then serve on http://127.0.0.1:3000
bun run serve          # serve only (viewer must already be built)
bun run rtfp:build     # release build of the Rust bake host
bun run bakes:build    # build the C++ bakes (Werner, mascon)
bun run esa:configure  # configure and build the vendored ESA reference library
```

## Tests

```sh
# Unit tests for the Rust solvers and the leapfrog kernels.
cd bakes/rtfp && cargo test --release

# Numerical self-test: closed-form checks, the analytic surface form against a
# brute-force direct sum, the Carlson constant-density identity, and Carlson
# against RT-FP on the real mesh. Requires a GPU.
./target/release/rtfp-bake --selftest --mode cauchy --normalize total_mass
```

Tests that need the external Ryugu OBJ skip themselves with a note when the file
is absent, so a bare checkout still runs green.

## Style

* Rust: `cargo fmt` and `cargo clippy -- -D warnings` must be clean, in both the
  viewer crate and `bakes/rtfp`.
* TypeScript: `bun run typecheck`.
* WGSL: keep kernels behind the `#[repr(C)]` structs in `bakes/rtfp/src/gpu.rs`;
  the host and the shader must stay byte-compatible.
* Documentation and comments: **standard academic English**, conclusion first.
  Prose in the repository is English; no other language is accepted in source,
  comments, or docs.

## Pull requests

* One logical change per pull request.
* Run the test suite and paste the relevant self-test lines in the description.
* If a change moves a reported number, say which number moved and by how much.
* Do not commit `target/`, `pkg/`, `node_modules/`, bake checkpoint files or
  progress logs; `.gitignore` already covers them. Do commit the finished
  `assets/records/*_faces.bin` records, since the README figures depend on them.

## License

By contributing you agree that your contribution is licensed under the terms in
[LICENSE](LICENSE) (MIT).
