# Ryugu Cauchy Gravity

Five independent gravity-gradient solvers for the 196,608-face Ryugu model, with one Rust/WGSL browser runtime and one Bevy renderer.

## Runtime model

The browser downloads only:

- `assets/models/ryugu.glb`
- `assets/density/cauchy.toml`
- `assets/density/cauchy_elliptic.toml`
- the Rust/WASM and Vue bundles

Rust parses the GLB and density descriptions after page entry, then constructs the selected solver input in memory. No result records, generated face buffers, Mascon trees, or other precomputed pipelines are shipped.

The computation branches are deliberately separate:

1. Werner: closed-form uniform-density polyhedral surface pipeline.
2. Mascon: runtime voxelization plus its own Barnes-Hut WGSL pipeline.
3. RT-FP: analytic near field plus ray finite-part remainder.
4. Carlson: density-jump surface pipeline.
5. CarlsonAlpha: general-alpha near field and radial remainder pipeline.

Every branch produces the same RHGF v5 face-scalar record. The branches then converge on one Bevy update path, one color-window implementation, and one exploded-mesh renderer.

## Source layout

```text
src/
├── rust/
│   ├── bin/                 Native CLI and self-checks
│   ├── compute/
│   │   ├── browser/         Browser WebGPU coordination
│   │   ├── native/          Native numerical reference code
│   │   └── source/          Runtime GLB/density input construction
│   └── viewer/
│       ├── compute/         Viewer-facing compute boundary
│       ├── frontend/        Session, persistence and Vue state
│       └── render/          Unified Bevy rendering
├── wgsl/
│   ├── compute/             Independent solver kernels
│   └── viewer/              Shared observation/scalar kernels
├── c++/
│   └── common/              Optional ESA reference bridge
└── web/
    ├── server/              Static-only Bun server
    ├── tools/               Bun build and staging scripts
    ├── app.ts               Vue view layer
    └── index.html
```

There is one root `Cargo.toml`, one root `Makefile`, and one optional CMake project under `src/c++`.

## Toolchain

- Rust and WGSL: Cargo, wasm-pack
- Vue and web tooling: Bun
- Optional C++ reference bridge: CMake
- Python is not required

Common commands:

```sh
make install
make build
make serve
make check
make test
make cpp
```

The server only distributes static files. All numerical work remains in Rust/WASM and WGSL in the browser.

## Preserved assets

Only the density descriptions, the Ryugu GLB, and the published README figures are source assets. Generated directories such as `target/`, `pkg/`, `dist/`, and `build/` are disposable.

## Published comparisons

| Solver / density | 1 mm | 16 m |
| --- | --- | --- |
| RT-FP / Cauchy | ![RT-FP Cauchy at 1 mm](docs/images/rtfp-cauchy-1mm.png) | ![RT-FP Cauchy at 16 m](docs/images/rtfp-cauchy-16m.png) |
| Carlson / Cauchy | ![Carlson Cauchy at 1 mm](docs/images/carlson-cauchy-1mm.png) | ![Carlson Cauchy at 16 m](docs/images/carlson-cauchy-16m.png) |
| Mascon / Cauchy | ![Mascon Cauchy at 1 mm](docs/images/mascon-cauchy-1mm.png) | ![Mascon Cauchy at 16 m](docs/images/mascon-cauchy-16m.png) |

Additional references:

- ![Werner uniform density at 16 m](docs/images/werner-uniform-16m.png)
- ![CarlsonAlpha fractional Cauchy at 16 m](docs/images/carlsonalpha-fractional-cauchy-16m.png)

## License

MIT
