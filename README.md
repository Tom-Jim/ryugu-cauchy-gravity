# Ryugu Cauchy Gravity

<p align="center">
  <a href="https://github.com/Tom-Jim/ryugu-cauchy-gravity"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-source-181717?style=flat-square&logo=github&logoColor=white"></a>
  <a href="https://www.rust-lang.org/"><img alt="Rust 2024" src="https://img.shields.io/badge/Rust-2024-000000?style=flat-square&logo=rust&logoColor=white"></a>
  <a href="https://bevy.org/"><img alt="Bevy 0.19" src="https://img.shields.io/badge/Bevy-0.19-74c0fc?style=flat-square"></a>
  <a href="https://bun.sh/"><img alt="Bun" src="https://img.shields.io/badge/Bun-1.4.2-fbf0df?style=flat-square&logo=bun&logoColor=black"></a>
  <a href="https://www.w3.org/TR/webgpu/"><img alt="WebGPU" src="https://img.shields.io/badge/WebGPU-WASM%20%2B%20native-005a9c?style=flat-square"></a>
  <a href="https://www.w3.org/TR/WGSL/"><img alt="WGSL" src="https://img.shields.io/badge/shader-WGSL-0f766e?style=flat-square"></a>
  <a href="https://vuejs.org/"><img alt="Vue" src="https://img.shields.io/badge/Vue-3-42b883?style=flat-square&logo=vuedotjs&logoColor=white"></a>
  <a href="https://github.com/Tom-Jim/ryugu-cauchy-gravity/actions"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/Tom-Jim/ryugu-cauchy-gravity/ci.yml?branch=main&style=flat-square&label=CI"></a>
</p>

<p align="center">
  <a href="https://tom-jim.github.io/ryugu-cauchy-gravity/"><img alt="Open the WebGPU viewer" src="https://img.shields.io/badge/OPEN_WEBGPU_VIEWER-0f766e?style=for-the-badge"></a>
</p>

## Preview

The five algorithms use one observation surface, one RHGF face-record format,
one scalar colour window and one Bevy renderer. The images are solver
comparisons, not separate rendering pipelines.

| Carlson · Cauchy · 16 m | RT-FP · Cauchy · 16 m |
| :---: | :---: |
| [![Carlson Cauchy](docs/images/carlson-cauchy-16m.png)](docs/images/carlson-cauchy-16m.png) | [![RT-FP Cauchy](docs/images/rtfp-cauchy-16m.png)](docs/images/rtfp-cauchy-16m.png) |
| Mascon · Cauchy · 16 m | Werner · uniform · 16 m |
| [![Mascon Cauchy](docs/images/mascon-cauchy-16m.png)](docs/images/mascon-cauchy-16m.png) | [![Werner uniform](docs/images/werner-uniform-16m.png)](docs/images/werner-uniform-16m.png) |

| CarlsonAlpha · fractional Cauchy · 16 m |
| :---: |
| [![CarlsonAlpha fractional Cauchy](docs/images/carlsonalpha-fractional-cauchy-16m.png)](docs/images/carlsonalpha-fractional-cauchy-16m.png) |

Low-altitude references:

| Carlson · Cauchy · 1 mm | RT-FP · Cauchy · 1 mm | Mascon · Cauchy · 1 mm |
| :---: | :---: | :---: |
| [![Carlson Cauchy at 1 mm](docs/images/carlson-cauchy-1mm.png)](docs/images/carlson-cauchy-1mm.png) | [![RT-FP Cauchy at 1 mm](docs/images/rtfp-cauchy-1mm.png)](docs/images/rtfp-cauchy-1mm.png) | [![Mascon Cauchy at 1 mm](docs/images/mascon-cauchy-1mm.png)](docs/images/mascon-cauchy-1mm.png) |

## Scope

This repository compares five independent gravity-gradient computations for a
196,608-face model of asteroid (162173) Ryugu:

1. Werner: uniform-density polyhedral reference.
2. Mascon: runtime voxelisation and Barnes-Hut point-mass evaluation.
3. RT-FP: Cauchy-density radial finite-part path.
4. Carlson: independent `alpha = 1` density-jump surface path.
5. CarlsonAlpha: general positive-alpha radial finite-part path.

The browser downloads only the GLB model, the two density TOML files and the
Rust/WASM/Vue static bundle. The server distributes files only; it performs no
numerical work. Every algorithm computes again after entry, with no shipped
precomputed solver record.

## Final mathematical conclusions

- The displayed scalar is the face-average of the Frobenius norm of the
  gravity-gradient tensor evaluated on the common offset observation surface.
- Werner, uniform RT-FP, uniform Carlson and uniform CarlsonAlpha are the
  common-density reference tracks and must agree up to floating-point and
  discretisation error.
- RT-FP Cauchy, Carlson Cauchy and Mascon Cauchy consume the same normalized
  `assets/density/cauchy.toml` field and total-mass convention.
- CarlsonAlpha fractional Cauchy and Mascon elliptic consume the same
  `assets/density/cauchy_elliptic.toml` field without silently replacing the
  exponents by `alpha = 1`.
- The browser Carlson `alpha = 1` implementation is a direct triangle
  density-jump surface integral. It does not use an `R_J` duplication loop,
  negative-parameter continuation, multipole approximation or near/far-field
  split.
- The special identity `R_J(x, y, z, z) = R_D(x, y, z)` is valid only when a
  future `R_J` implementation actually supplies `p = z`; it is not a license
  to rewrite the current Carlson surface path.
- No unverified Mobius remapping, two-iteration universal truncation, or
  Fukushima-style minimax replacement is used in the production path.

## Published validation

These are the repository's recorded face-wise comparisons at a 16 m
observation height. They compare the same density field and the same face
sampling convention.

| Pair | Density pairing | Median relative difference | Maximum | Faces over 5% |
| --- | --- | ---: | ---: | ---: |
| Carlson vs RT-FP | Cauchy | 0.115% | 24.6% | 56 |
| RT-FP vs Mascon | Cauchy | 0.297% | 96.2% | 204 |
| Carlson vs Mascon | Cauchy | 0.316% | 96.2% | 255 |
| CarlsonAlpha vs Mascon | fractional Cauchy | 0.304% | 96.3% | 203 |
| RT-FP vs Werner | uniform | 4.49e-7 | 3.5e-6 | 0 |
| Carlson vs Werner | uniform | 4.48e-7 | 3.4e-6 | 0 |
| CarlsonAlpha vs Werner | uniform | 4.49e-7 | 3.5e-6 | 0 |

The table is a solver-consistency statement, not a claim that the designed
density TOML is an observational inversion of Ryugu's interior.

## Architecture

```text
src/
├── rust/
│   ├── bin/                 Native CLI and self-checks
│   ├── compute/
│   │   ├── browser/         WebGPU coordination and dispatch
│   │   ├── native/          Native numerical reference code
│   │   └── source/          Runtime GLB and density construction
│   └── viewer/
│       ├── compute/         Viewer compute boundary
│       ├── frontend/        Session, temporary browser storage and Vue state
│       └── render/          Unified Bevy rendering
├── wgsl/compute/            Independent solver kernels
├── c++/                     Optional ESA reference bridge
└── web/                     Vue view, static server and build tools
```

There is one root `Cargo.toml`, one root `Makefile`, and one optional CMake
project under `src/c++`.

## Toolchain

- Rust and WGSL: Cargo and wasm-pack
- Vue and web tooling: Bun
- Optional ESA bridge: CMake
- Python is not required

```sh
make install
make build
make serve
make check
make test
make cpp
```

`make check` is source-only validation for formatting, Clippy and TypeScript.
The GitHub Actions workflow additionally builds the native crate, the WASM
viewer and the static GitHub Pages bundle.

## Density assets

- `assets/density/cauchy.toml`: normalized `alpha = 1` Cauchy stress field.
- `assets/density/cauchy_elliptic.toml`: fractional-Cauchy field with explicit
  positive exponents.
- `assets/models/ryugu.glb`: the only geometry input.

Generated `target/`, `pkg/`, `dist/` and `build/` directories are disposable.

## License

MIT
