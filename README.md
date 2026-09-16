# ryugu-cauchy-gravity

<p align="center">
  <a href="https://github.com/Tom-Jim/ryugu-cauchy-gravity"><img alt="GitHub repository" src="https://img.shields.io/badge/GitHub-Tom--Jim%2Fryugu--cauchy--gravity-181717?style=flat-square&logo=github&logoColor=white"></a>
  <a href="https://www.rust-lang.org/"><img alt="Rust 2024" src="https://img.shields.io/badge/Rust-2024-000000?style=flat-square&logo=rust&logoColor=white"></a>
  <a href="https://bevy.org/"><img alt="Bevy 0.19" src="https://img.shields.io/badge/Bevy-0.19-74c0fc?style=flat-square"></a>
  <a href="https://bun.sh/"><img alt="Bun" src="https://img.shields.io/badge/Bun-runtime-fbf0df?style=flat-square&logo=bun&logoColor=black"></a>
  <a href="https://www.w3.org/TR/webgpu/"><img alt="WebGPU" src="https://img.shields.io/badge/WebGPU-native%20%2B%20WASM-005a9c?style=flat-square"></a>
  <a href="https://www.w3.org/TR/WGSL/"><img alt="WGSL" src="https://img.shields.io/badge/shader-WGSL-0f766e?style=flat-square"></a>
  <a href="https://www.typescriptlang.org/"><img alt="TypeScript" src="https://img.shields.io/badge/TypeScript-server-3178c6?style=flat-square&logo=typescript&logoColor=white"></a>
</p>

<p align="center">
  <a href="https://github.com/Tom-Jim/ryugu-cauchy-gravity"><img alt="Browse the source on GitHub" src="https://img.shields.io/badge/GITHUB-SOURCE-181717?style=for-the-badge&logo=github&logoColor=white"></a>
  <a href="https://tom-jim.github.io/ryugu-cauchy-gravity/"><img alt="Open the interactive WebGPU demo" src="https://img.shields.io/badge/OPEN_THE_INTERACTIVE_WEBGPU_DEMO-0f766e?style=for-the-badge"></a><br>
  <sub>Browse the source or launch the browser demo and compare all five solvers.</sub>
</p>

## Preview

All solvers render the same record format through the same colour window,
at the same observation height, so the images below are directly comparable.

| **Carlson · Cauchy · 16 m** | **RT-FP · Cauchy · 16 m** |
| :---: | :---: |
| [![Carlson, Cauchy density, 16 m](docs/images/carlson-cauchy-16m.png)](docs/images/carlson-cauchy-16m.png) | [![RT-FP, Cauchy density, 16 m](docs/images/rtfp-cauchy-16m.png)](docs/images/rtfp-cauchy-16m.png) |
| **Mascon · Cauchy · 16 m** | **Werner · uniform · 16 m** |
| [![Mascon, Cauchy density, 16 m](docs/images/mascon-cauchy-16m.png)](docs/images/mascon-cauchy-16m.png) | [![Werner, uniform density, 16 m](docs/images/werner-uniform-16m.png)](docs/images/werner-uniform-16m.png) |

The generalized CarlsonAlpha path provides the fractional-Cauchy check against
the voxel solver on the same observation surface:

| **CarlsonAlpha · fractional Cauchy · 16 m** |
| :---: |
| [![CarlsonAlpha, fractional Cauchy density, 16 m](docs/images/carlsonalpha-fractional-cauchy-16m.png)](docs/images/carlsonalpha-fractional-cauchy-16m.png) |

At 16 m, Carlson and RT-FP retain the same smooth density structure, while
Mascon shows visible cell-scale shading. Werner is the uniform-density reference
that both exact solvers reproduce to `4 × 10⁻⁷`.

| **Carlson · Cauchy · 1 mm** | **RT-FP · Cauchy · 1 mm** |
| :---: | :---: |
| [![Carlson, Cauchy density, 1 mm](docs/images/carlson-cauchy-1mm.png)](docs/images/carlson-cauchy-1mm.png) | [![RT-FP, Cauchy density, 1 mm](docs/images/rtfp-cauchy-1mm.png)](docs/images/rtfp-cauchy-1mm.png) |
| **Mascon · Cauchy · 1 mm** | **Mascon failure mode** |
| [![Mascon, Cauchy density, 1 mm](docs/images/mascon-cauchy-1mm.png)](docs/images/mascon-cauchy-1mm.png) | At 1 mm the voxel record dissolves into per-cell speckle. This is a solver limitation at a standoff far below the cell size, not a rendering artefact. |

Five independent algorithms for the **gravity-gradient tensor** (the Hessian of the
gravitational potential) of a body with a **non-uniform interior**, compared on
one shape model of asteroid (162173) Ryugu.

The point of the repository is not one solver but the agreement between them.
All five evaluate the same 196,608-face mesh from the same observation surface
and write the same per-face record. The four `α = 1` records share one
total-mass budget; the fractional-Cauchy pair keeps the TOML weights unchanged
so Mascon and CarlsonAlpha use one common scale. The viewer renders every solver
through one identical display path, so a difference on screen is a difference
between solvers and nothing else.

| Solver | Method | Density | Where it runs |
| --- | --- | --- | --- |
| **Werner** | Closed-form polyhedral tensor (the ESA reference library) | uniform | host, multithreaded C++ |
| **Mascon** | Voxel direct sum over 192³ point masses | arbitrary | host, multithreaded C++ |
| **RT-FP** | Analytic near field plus a directional quadrature of the radial remainder | Cauchy, α=1 | GPU, WGSL |
| **Carlson** | Density jump surfaces over a star-cone decomposition, evaluated by the polyhedral surface integral | piecewise constant | GPU, WGSL |
| **CarlsonAlpha** | General-α radial finite part with Carlson-library verification | Cauchy, arbitrary positive α | GPU, WGSL |

## Results

Face-by-face relative difference `|a − b| / max(|a|, |b|)` over all 196,608
faces, observation surface 16 m above the terrain, Ryugu mass 4.5 × 10¹¹ kg.
`> 5 %` is the count of faces that disagree by more than five percent.

| Pair | Same density? | median | p90 | p99 | max | > 5 % |
| --- | --- | --- | --- | --- | --- | --- |
| **Carlson vs RT-FP** | yes (Cauchy field) | **0.115 %** | 0.28 % | 0.51 % | 24.6 % | 56 |
| RT-FP vs Mascon | yes (Cauchy field) | 0.297 % | 0.79 % | 1.51 % | 96.2 % | 204 |
| Carlson vs Mascon | yes (Cauchy field) | 0.316 % | 0.84 % | 1.61 % | 96.2 % | 255 |
| **CarlsonAlpha vs Mascon** | yes (fractional Cauchy field, α ≠ 1) | **0.304 %** | 0.81 % | 1.55 % | 96.3 % | 203 |
| **RT-FP vs Werner** | yes (uniform) | **4.49 × 10⁻⁷** | 1.1 × 10⁻⁶ | 1.8 × 10⁻⁶ | 3.5 × 10⁻⁶ | 0 |
| **Carlson vs Werner** | yes (uniform) | **4.48 × 10⁻⁷** | 1.1 × 10⁻⁶ | 1.8 × 10⁻⁶ | 3.4 × 10⁻⁶ | 0 |
| **CarlsonAlpha vs Werner** | yes (uniform) | **4.49 × 10⁻⁷** | 1.1 × 10⁻⁶ | 1.8 × 10⁻⁶ | 3.5 × 10⁻⁶ | 0 |

Read the two halves separately.

*In the uniform-density limit*, RT-FP and Carlson each reproduce the polyhedral
closed form to about 4 × 10⁻⁷ relative — one part in two million, i.e. the
round-off of the 32-bit record these results are stored in. Two formulations
that share no numerical machinery beyond the mesh agree with a third
implementation to the precision of the file format.

*On the varying field*, the two `α = 1` GPU solvers agree with each other to
0.12 % median, while the voxel solver sits 0.30 % away — a median difference two
and a half times larger from a solver that is also far heavier. The generalized
CarlsonAlpha path reproduces the same fractional-Cauchy Mascon record to 0.30 %
median, without falling back to `α = 1`. The tails differ much more than the
medians: Carlson and RT-FP stay inside 0.51 % for 99 % of `α = 1` faces, whereas
Mascon has about 0.1 % of faces beyond 5 %, with individual faces off by 96 %.
Mascon's error is not a smooth bias that a calibration could absorb; it is
concentrated where the local terrain is closest to a cell centre.

### The same comparison at 1 mm

Observation surface one millimetre above the terrain — the regime a lander, a
sampling manoeuvre or a low-altitude gravity-gradient survey occupies:

| Pair | Same density? | median | p90 | p99 | max | > 5 % |
| --- | --- | --- | --- | --- | --- | --- |
| **Carlson vs RT-FP** | yes | **0.36 %** | 0.94 % | 2.37 % | 38.5 % | 316 |
| RT-FP vs Mascon | yes | 40.6 % | 82.3 % | 98.3 % | 99.98 % | 186,322 |
| Carlson vs Mascon | yes | 40.7 % | 82.3 % | 98.3 % | 99.98 % | 186,351 |

Carlson and RT-FP lose almost nothing when the surface moves from
16 m down to 1 mm: the median disagreement grows from 0.12 % to 0.36 %, and 316
of 196,608 faces (0.16 %) exceed 5 %. Both stay usable in the regime a lander, a
sampling manoeuvre or a low-altitude gradient survey actually occupies.

Mascon does not. Its median disagreement against either GPU solver is 40.6 %,
and its computed field is on average 4.76× larger than RT-FP's on the same face.
A voxel direct sum is simply not defined at a standoff an order of magnitude
below its own cell size (5.253 m), and nothing about that number is a rendering
artefact: it is the difference between the two stored records.

### Height sweep

The fixed probe set from `--selftest` compares the Carlson tensor with the RT-FP
tensor at every height the viewer exposes. It uses 32 mesh vertices per height;
the error is the Frobenius relative difference between the two tensors.

| Standoff | Carlson vs RT-FP median | worst |
| --- | --- | --- |
| 1 mm | 1.05 × 10⁻² | 4.55 × 10⁻² |
| 1 m | 4.39 × 10⁻³ | 2.46 × 10⁻² |
| 4 m | 3.28 × 10⁻³ | 2.38 × 10⁻² |
| 8 m | 2.76 × 10⁻³ | 2.28 × 10⁻² |
| 16 m | 2.31 × 10⁻³ | 2.14 × 10⁻² |
| 32 m | 1.76 × 10⁻³ | 1.93 × 10⁻² |

This probe table is deliberately harsher than the full-record table above: its
fixed vertices include near-edge and near-vertex samples, while the record
average is taken over all 196,608 faces. The important check is the trend: the
two formulations remain consistent from 1 mm to 32 m, and neither develops the
near-field loss of meaning seen in the voxel field.

The practical reading is a standoff rule. A voxel model needs a handful of cells
between the observation point and the body — for this mesh, roughly 15–25 m at a
192³ grid and 24–40 m at 128³. Below that the grid has to be refined until its
cells are small compared with the standoff, which is the cost the two surface
methods do not pay at all.

## The density field

`assets/density/cauchy.toml` is a **designed stress case**, not an inversion
product, and its shape follows the algorithms rather than the other way round.

Carlson resolves the interior along each cone axis, so structure that varies
with *direction* is free to it and structure that varies with *radius* costs
slabs. RT-FP integrates the radial direction in closed form, so radial structure
is free to it and directional structure has to stay smooth on the scale of its
288-direction quadrature. The field therefore puts most of its variation in
direction — a 12-fold equatorial ore belt over a near-uniform matrix, with a
regolith deficit, polar deficits and a mid-latitude compensation — and carries
two classes of genuine discontinuity: a narrow core–mantle step and discrete ore
bodies and voids. A voxel grid can only average those into a cell mass.

The contrast is set by measurement, not by taste. Carlson's slab refinement
reaches its per-cone cap before it reaches the tolerance it requests, so its
representation error saturates rather than falling smoothly. Scaling the angular
layers alone, the worst Carlson-versus-RT-FP disagreement at a 1 mm standoff
moves from **4.55 × 10⁻²** at the shipped contrast to 6.38 × 10⁻² at 1.08× and
7.09 × 10⁻² at 1.54×, and the cross-solver self-test rejects anything at or above
6 × 10⁻². The shipped field is the strongest one the solver budget actually
supports. The discontinuities are not part of that trade-off — halving, widening
or keeping the core step leaves the worst within-cone variation unchanged at
5.76 × 10⁻² — so they are carried at full strength.

Two earlier layouts were measured and rejected: a purely radial stratification
(worst within-cone slab variation 1.19 × 10⁻¹, worse for every solver) and a
version of this layout at 1.54× the angular contrast (fails the self-test).

## Verification

`rtfp-bake --selftest` (or `bun run rtfp:selftest`) checks the closed-form
identities and cross-solver consistency independently of the record path. Every
line below is a measured number from the current revision.

| Check | Result |
| --- | --- |
| Cube tensor vs the ESA polyhedral library, far field | 2.5 × 10⁻¹⁶ |
| Cube tensor vs ESA, 1 mm above a face centre | 1.1 × 10⁻¹³ |
| Cube tensor vs ESA, 1 mm above a face corner | 1.7 × 10⁻¹⁴ |
| Cube tensor vs ESA, 1 mm above an edge | 6.1 × 10⁻¹⁵ |
| Cube tensor vs ESA, 1 mm above a corner vertex | 3.1 × 10⁻⁹ |
| Direct 24³ point-mass sum vs ESA, same body | 3.3 × 10⁻⁷ |
| Closed-form surface form vs ESA, real mesh, 40 samples | median 6.5 × 10⁻¹², worst 2.1 × 10⁻⁹ |
| Carlson, uniform density vs the polyhedral tensor | median 6.7 × 10⁻⁴, worst 3.3 × 10⁻³ |
| Carlson, Cauchy field vs RT-FP, 1 mm | median 1.05 × 10⁻², worst 4.55 × 10⁻² |
| Carlson, Cauchy field vs RT-FP, 16 m | median 2.31 × 10⁻³, worst 2.14 × 10⁻² |
| WGSL near-field kernel vs an f64 closed form | median 4.3 × 10⁻⁴, worst 3.3 × 10⁻³ |
| BVH traversal vs brute-force intersection | identical |
| WGSL rays + remainder vs an f64 brute force | median 3.0 × 10⁻⁷, p90 5.0 × 10⁻⁷ |

The two `10⁻³`-level analytic entries are the 32-bit storage of the kernel, not
the formulation: both are measured against a double-precision reference and
disappear if the same comparison is done before narrowing to `f32`.

The two cross-solver rows are sampled checks on a fixed probe set, not the full
mesh, so they read slightly worse than the all-face table at the top; both are
gated by the same `6 × 10⁻²` worst-case limit.

Three honest caveats, all visible in the table above:

* The WGSL ray/remainder path has rare outliers — 2 of 192 probe points exceeded
  5 % — on samples placed exactly on a face plane, where the ray hits a
  degenerate edge. The mesh-scale comparison used for the record is not affected
  at that rate (56 of 196,608 faces).
* Ryugu is **not** star-shaped from its centroid (signed volume over covered
  volume = 0.99898), so Carlson's cone decomposition is not exact. The residual
  is a fixed, measurable offset that the construction keeps out of the
  constant-density term; it is the reason Carlson's own tail is larger than
  RT-FP's.
* The shipped **Carlson** path implements the PDF's jump-surface/star-cone
  reduction for `α = 1`. The separate **CarlsonAlpha** path accepts general
  positive `α` through the radial finite-part quadrature and uses the `ellip`
  crate's `R_F/R_D/R_J/R_C` implementations as its f64 verification backend.
  This is deliberately narrower than claiming that every genus-one boundary
  reduction in the derivation has been symbolically completed for every `α`.

## Why the surface solvers

**They do not need a volumetric grid.** RT-FP, Carlson and CarlsonAlpha work
directly from the mesh and an analytic density, so the accuracy limit never
becomes "the cell is 5.25 m wide". The observation surface can be a millimetre
above the terrain — the regime a
lander, a sampling manoeuvre, or a low-altitude gravity-gradient survey actually
occupies — without the solver's own representation getting in the way.

**They are cheap where it counts.** The body is a surface, not a volume, and
the GPU paths spend their work on face and direction structure. Cost scales
with faces × observation points; there is no 192³ grid to fill, and no per-cell
bookkeeping whose cost grows as the cube of the desired resolution. Resolution
here is spent on the mesh, which is the thing that actually carries the shape
information.

**They are structurally independent of each other.** RT-FP splits the tensor into
a closed-form near field plus a directional quadrature of the density deviation;
Carlson reduces the `α = 1` problem to jump surfaces with no rays; CarlsonAlpha
integrates the general-α radial finite part. Sharing only the mesh and the input
field turns agreement into a real cross-check rather than a shared-code
round-trip.

**They are indifferent to how the density is parameterised.** RT-FP integrates
the radial direction in closed form, so fine *radial* structure costs nothing and
only the directional part is quadratured. Carlson represents density by its
discontinuities, so a rubble pile modelled as discrete units, an ore body, or a
void is expressed directly as jump surfaces rather than smeared onto a grid.

## Where they lose

* **Carlson pays for density contrast inside a cell.** Its error is set by how
  much the density varies across one cone slab, and the slab budget is finite:
  past a threshold the refinement saturates and the error stops falling. Every
  run prints the tolerance it reached and the worst within-slab variation, so the
  number is never assumed.
* **Carlson's cone decomposition assumes a star-shaped body.** The deviation is
  small (0.1 % on Ryugu) but it is a modelling choice, not a numerical accident.
* **RT-FP needs a radially integrable density law.** The shipped field is a sum
  of kernels with a closed-form radial integral; a general law would need a
  radial quadrature.
* **RT-FP carries a directional quadrature error.** 288 directions put it at the
  few × 10⁻⁴ level with rare degenerate outliers; doubling the directions is
  cheap but not free.
* **Neither is validated against measured gravity.** The density field shipped
  here is a *designed* stress case, not a Hayabusa2 inversion product. The
  agreement in the tables is between solvers; it says nothing about the real
  interior of Ryugu.

## Choosing a solver

| Situation | Use |
| --- | --- |
| Uniform density, shape-only study, published benchmark | **Werner / polyhedral** — closed form, fastest, most widely validated |
| Density required as a grid product, or coupling to a code that wants cell masses | **Mascon** — trivial to consume, but keep cells small relative to the closest observation distance |
| Smooth, low-order interior variation over a large body | **Mascon** or **spherical harmonics**; a grid or an expansion is the natural language |
| Strongly three-dimensional density, or evaluation within metres of the terrain | **RT-FP** |
| Piecewise-constant interior (units, voids, ore bodies), or an independent check on another solver | **Carlson** |
| Deep-space / far-field work over a long arc | **Spherical harmonics** — cheapest and standard, and the near-surface regime is far away by construction |
| Tesseroids and prisms | Standard for layered or gridded regional models where the density is constant per element; they inherit the voxel trade-off between element size and standoff |

## Running it

```sh
bun install
bun run dev            # build the WASM viewer, serve on http://127.0.0.1:3000
```

The viewer needs a browser with WebGPU and cross-origin isolation (both are
served by the development server).

The hosted static build is available at
[tom-jim.github.io/ryugu-cauchy-gravity](https://tom-jim.github.io/ryugu-cauchy-gravity/).
It is generated by `bun run pages:build`: `wasm-pack` writes the browser package
to `pkg/`, then `scripts/stage-pages.mjs` copies only the required runtime files
to the generated `dist/` directory. On pushes to `main`, GitHub Actions uploads
`dist/` as the Pages artifact and deploys it through the `github-pages`
environment. Neither `pkg/` nor `dist/` is committed to the repository.

Five algorithm tabs, one per solver, each with the same two-segment vertical
observation-height slider: 1 mm → 500 mm over the lower half of the track and
1 m → 32 m over the upper half. Moving the slider restarts the bake at the new
height. Mascon and CarlsonAlpha share the fractional-Cauchy density switch;
RT-FP and Carlson retain their Cauchy/uniform controls. The viewer compares each
record face by face with the reference that uses the same density model.

> A note on the uniform-density reference at extreme standoffs: the C++
> polyhedral bake is the slow path within a millimetre of the terrain, so the
> near-field comparison in this project is made against the two GPU solvers.

## Reproducing a record

```sh
OBJ=../Ryugu_wasm/assets/models/SHAPE_SFM_200k_v20180804.obj

# RT-FP, varying density, GPU
./target/release/rtfp-bake --obj "$OBJ" --out assets/records/rtfp_faces.bin \
  --order assets/records/.rtfp_order.bin --density assets/density/cauchy.toml \
  --mode cauchy --solver ray --directions 288 --standoff-mm 16000 \
  --normalize total_mass

# Carlson, same field and surface
./target/release/rtfp-bake --obj "$OBJ" --out assets/records/carlson_cauchy_faces.bin \
  --order assets/records/.carlson_cauchy_order.bin --density assets/density/cauchy.toml \
  --mode cauchy --solver carlson --standoff-mm 16000 --normalize total_mass

# Mascon, same field and surface
./bakes/build/ryugu_mascon_bake_cpp --obj "$OBJ" --out assets/records/mascon_faces.bin \
  --order assets/records/.mascon_order.bin --density assets/density/cauchy.toml \
  --grid 192 --standoff-mm 16000

# CarlsonAlpha and Mascon, fractional-Cauchy comparison field
./target/release/rtfp-bake --obj "$OBJ" \
  --out assets/records/carlsonalpha_elliptic_faces.bin \
  --order assets/records/.carlsonalpha_order.bin \
  --density assets/density/cauchy_elliptic.toml \
  --mode elliptic --normalize raw --solver carlson-alpha \
  --directions 288 --standoff-mm 16000
./bakes/build/ryugu_mascon_bake_cpp --obj "$OBJ" \
  --out assets/records/mascon_elliptic_faces.bin \
  --order assets/records/.mascon_elliptic_order.bin \
  --density assets/density/cauchy_elliptic.toml \
  --grid 192 --standoff-mm 16000

# Werner / ESA polyhedral, uniform density
./bakes/build/ryugu_gradient_bake_cpp --obj "$OBJ" --out assets/records/gradient_faces.bin \
  --order assets/records/.bake_order.bin --standoff-mm 16000

# Compare any two records
bun tools/compare-records.ts assets/records/carlson_cauchy_faces.bin \
  assets/records/rtfp_faces.bin
```

The original Cauchy and uniform records use the same total-mass convention
(`--normalize total_mass` resolves the density weights so that
∫ρ dV = 4.5 × 10¹¹ kg). The fractional-Cauchy comparison keeps the TOML weights
raw for both Mascon and CarlsonAlpha, so those two records also share one
density scale.

### Tests

```sh
cd bakes/rtfp && cargo test --release      # unit tests, plus the closed-form and
                                           # cross-solver checks when the mesh is present
bun run rtfp:selftest                      # same checks, verbose, needs a GPU
bun run typecheck                          # server and tools
```

## Repository layout

```
bakes/rtfp/          Rust bake host: RT-FP, Carlson, CarlsonAlpha and WGSL kernels
bakes/mascon/        mascon voxel direct sum (C++)
bakes/werner/        polyhedral closed form over the ESA reference library (C++)
pkg/                 generated wasm-pack browser package (not committed)
dist/                generated GitHub Pages static site (not committed)
src/server/          Bun development server, one endpoint per solver
src/viewer/          Bevy + WebGPU viewer (Rust, compiled to WASM)
src/web/             single-page control surface
assets/density/      the density field (TOML)
assets/records/      finished per-face records, one per solver and density
tools/               record comparison and other helpers
docs/images/         figures used by this README
```

## License

MIT — see [LICENSE](LICENSE). Contributions follow [CONTRIBUTING.md](CONTRIBUTING.md).
