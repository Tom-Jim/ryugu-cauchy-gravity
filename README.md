# ryugu-cauchy-gravity

Four independent solvers for the **gravity-gradient tensor** (the Hessian of the
gravitational potential) of a body with a **non-uniform interior**, compared on
one shape model of asteroid (162173) Ryugu.

The point of the repository is not one solver but the agreement between them.
All four evaluate the same 196,608-face mesh, from the same observation surface,
on the same total-mass budget, and write the same per-face record. The viewer
renders every solver through one identical display path, so a difference on
screen is a difference between solvers and nothing else.

| Solver | Method | Density | Where it runs |
| --- | --- | --- | --- |
| **Werner** | Closed-form polyhedral tensor (the ESA reference library) | uniform | host, multithreaded C++ |
| **Mascon** | Voxel direct sum over 192³ point masses | arbitrary | host, multithreaded C++ |
| **RT-FP** | Analytic near field plus a directional quadrature of the radial remainder | arbitrary | GPU, WGSL |
| **Carlson** | Density written as jump surfaces over a cone decomposition, evaluated as a surface integral | piecewise constant | GPU, WGSL |

## Results

Face-by-face relative difference `|a − b| / max(|a|, |b|)` over all 196,608
faces, observation surface 16 m above the terrain, Ryugu mass 4.5 × 10¹¹ kg.
`> 5 %` is the count of faces that disagree by more than five percent.

| Pair | Same density? | median | p90 | p99 | max | > 5 % |
| --- | --- | --- | --- | --- | --- | --- |
| **Carlson vs RT-FP** | yes (Cauchy field) | **0.108 %** | 0.27 % | 0.49 % | 22.4 % | 54 |
| RT-FP vs Mascon | yes (Cauchy field) | 0.294 % | 0.78 % | 1.50 % | 96.1 % | 206 |
| Carlson vs Mascon | yes (Cauchy field) | 0.312 % | 0.82 % | 1.58 % | 96.2 % | 252 |
| **RT-FP vs Werner** | yes (uniform) | **4.49 × 10⁻⁷** | 1.1 × 10⁻⁶ | 1.8 × 10⁻⁶ | 3.5 × 10⁻⁶ | 0 |
| **Carlson vs Werner** | yes (uniform) | **4.48 × 10⁻⁷** | 1.1 × 10⁻⁶ | 1.8 × 10⁻⁶ | 3.4 × 10⁻⁶ | 0 |

Read the two halves separately.

*In the uniform-density limit*, RT-FP and Carlson each reproduce the polyhedral
closed form to about 4 × 10⁻⁷ relative — one part in two million, i.e. the
round-off of the 32-bit record these results are stored in. Two formulations
that share no numerical machinery beyond the mesh agree with a third
implementation to the precision of the file format.

*On the varying field*, the two GPU solvers agree with each other to 0.11 %
median, while the voxel solver sits 0.29 % away — a median difference three
times larger from a solver that is also far heavier. The tails differ much more
than the medians: Carlson and RT-FP stay inside 0.5 % for 99 % of faces, whereas
mascon has 206 faces (0.1 %) beyond 5 %, with individual faces off by 96 %.
Mascon's error is not a smooth bias that a calibration could absorb; it is
concentrated where the local terrain is closest to a cell centre.

### The same comparison at 1 mm

Observation surface one millimetre above the terrain — the regime a lander, a
sampling manoeuvre or a low-altitude gravity-gradient survey occupies:

| Pair | Same density? | median | p90 | p99 | max | > 5 % |
| --- | --- | --- | --- | --- | --- | --- |
| **Carlson vs RT-FP** | yes | **0.365 %** | 0.95 % | 2.33 % | 36.0 % | 293 |
| RT-FP vs Mascon | yes | 40.4 % | 82.1 % | 98.3 % | 99.98 % | 186 268 |
| Carlson vs Mascon | yes | 40.6 % | 82.2 % | 98.3 % | 99.98 % | 186 308 |

The two grid-free solvers stay mutually consistent to a third of a percent
median. Mascon does not degrade gently: **94.7 % of faces** disagree with the
grid-free result by more than five percent, the median disagreement is 40 %, and
mascon's own maximum reading is 3.55 × 10⁻³ against 2.33 × 10⁻⁶ from the exact
solvers — a factor of 1 500. A 5.2 m cell cannot represent a field whose
observation point is 1 mm away from the boundary of the body.

Cost, on the same machine: RT-FP evaluates the whole mesh in about 20 s, Carlson
in about 12 min. The uniform-density reference bake is the one that becomes
impractical here — it advances through 2 688 of 196 608 faces in 90 s at 1 mm
against a full mesh in about a minute at 16 m — which is why the near-surface
comparison above is between the two GPU solvers and the voxel solver.

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
| WGSL near-field kernel vs an f64 closed form | median 4.3 × 10⁻⁴, worst 3.3 × 10⁻³ |
| BVH traversal vs brute-force intersection | identical |
| WGSL rays + remainder vs an f64 brute force | median 2.8 × 10⁻⁷, p90 4.7 × 10⁻⁷ |

The two `10⁻³`-level entries are the 32-bit storage of the analytic kernel, not
the formulation: both are measured against a double-precision reference and
disappear if the same comparison is done before narrowing to `f32`.

Two honest caveats, both visible in the table above:

* The WGSL ray/remainder path has rare outliers — 2 of 192 probe points exceeded
  5 % — on samples placed exactly on a face plane, where the ray hits a
  degenerate edge. The mesh-scale comparison used for the record is not
  affected at that rate (54 of 196,608 faces).
* Ryugu is **not** star-shaped from its centroid (signed volume over covered
  volume = 0.99898), so Carlson's cone decomposition is not exact. The residual
  is a fixed, measurable offset that the construction keeps out of the
  constant-density term; it is the reason Carlson's own tail is larger than
  RT-FP's.

## Why Carlson and RT-FP

**They do not need a volumetric grid.** Both work directly from the mesh and an
analytic density, so the accuracy limit never becomes "the cell is 5.2 m wide".
The observation surface can be a millimetre above the terrain — the regime a
lander, a sampling manoeuvre, or a low-altitude gravity-gradient survey actually
occupies — without the solver's own representation getting in the way.

**They are cheap where it counts.** The body is a surface, not a volume, and both
methods integrate over that surface. Cost scales with faces × observation points
and runs on the GPU; there is no 192³ grid to fill, and no per-cell bookkeeping
whose cost grows as the cube of the desired resolution. Resolution here is
spent on the mesh, which is the thing that actually carries the shape
information.

**They are structurally independent of each other.** RT-FP splits the tensor into
a closed-form near field plus a directional quadrature of the density deviation;
Carlson reduces the whole problem to a jump-surface integral with no rays and no
directions. Sharing only the mesh, they are a real cross-check: a coding error in
either would have to be mirrored in the other to stay hidden. That is what makes
the 0.11 % agreement on a varying density meaningful rather than circular.

**They are indifferent to how the density is parameterised.** RT-FP integrates
the radial direction in closed form, so fine *radial* structure costs nothing and
only the directional part is quadratured. Carlson represents density by its
discontinuities, so a rubble pile modelled as discrete units, an ore body, or a
void is expressed directly as jump surfaces rather than smeared onto a grid.

## Where they lose

* **Carlson pays for density contrast inside a cell.** Its error is set by how
  much the density varies across one cone slab. A high-frequency field needs many
  slabs, which costs memory. Every run prints the tolerance it reached and the
  worst within-slab variation, so the number is never assumed.
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

## Screenshots

All four pages render the same record format through the same colour window, at
the same observation height, so the images are directly comparable.

Carlson on the varying (Cauchy) field, 16 m. The panel reports its residual
against the same-density reference and against the uniform-density one:

![Carlson, Cauchy density, 16 m](docs/images/carlson-cauchy-16m.png)

RT-FP on the same field and the same surface:

![RT-FP, Cauchy density, 16 m](docs/images/rtfp-cauchy-16m.png)

Mascon on the same field: the voxel structure is visible in the shading, and the
record's own maximum is 18× the exact solvers' — a single face where the cell
approximation dominates:

![Mascon, Cauchy density, 16 m](docs/images/mascon-cauchy-16m.png)

Werner, uniform density, 16 m — the reference both exact solvers reproduce to
4 × 10⁻⁷:

![Werner, uniform density, 16 m](docs/images/werner-uniform-16m.png)

The same field and mesh one millimetre above the terrain. Carlson's panel
reports the residuals directly: 40.5 % median against the voxel record, and the
uniform-density reference is flagged as a height mismatch because that record
is still baked at 16 m.

![Carlson, Cauchy density, 1 mm](docs/images/carlson-cauchy-1mm.png)

![RT-FP, Cauchy density, 1 mm](docs/images/rtfp-cauchy-1mm.png)

Mascon at the same surface: the colour window had to switch to `asinh`, the
`std/mean` of the record jumps from 0.17 to 11.9, and the body dissolves into
per-cell speckle. This is the failure mode, not a rendering artefact.

![Mascon, Cauchy density, 1 mm](docs/images/mascon-cauchy-1mm.png)

## Running it

```sh
bun install
bun run dev            # build the WASM viewer, serve on http://127.0.0.1:3000
```

The viewer needs a browser with WebGPU and cross-origin isolation (both are
served by the development server).

Four algorithm tabs, one per solver, each with the same two-segment vertical
observation-height slider: 1 mm → 500 mm over the lower half of the track and
1 m → 32 m over the upper half. Moving the slider restarts the bake at the new
height. RT-FP and Carlson also carry a density switch (uniform ρ₀ vs the shipped
Cauchy field) and a live face-by-face comparison against the reference records.

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
  --order assets/records/.carlson_order.bin --density assets/density/cauchy.toml \
  --mode cauchy --solver carlson --standoff-mm 16000 --normalize total_mass

# Mascon, same field and surface
./bakes/build/ryugu_mascon_bake_cpp --obj "$OBJ" --out assets/records/mascon_faces.bin \
  --order assets/records/.mascon_order.bin --density assets/density/cauchy.toml \
  --grid 192 --standoff-mm 16000

# Werner / ESA polyhedral, uniform density
./bakes/build/ryugu_gradient_bake_cpp --obj "$OBJ" --out assets/records/gradient_faces.bin \
  --order assets/records/.bake_order.bin --standoff-mm 16000

# Compare any two records
bun tools/compare-records.ts assets/records/carlson_cauchy_faces.bin \
  assets/records/rtfp_faces.bin
```

Every solver is normalised to the same total mass (`--normalize total_mass`
resolves the density weights so that ∫ρ dV = 4.5 × 10¹¹ kg), so a residual is
never a bookkeeping difference.

### Tests

```sh
cd bakes/rtfp && cargo test --release      # 11 unit tests
bun run rtfp:selftest                      # closed-form + cross-solver, needs a GPU
bun run typecheck                          # server and tools
```

## Repository layout

```
bakes/rtfp/          Rust bake host: RT-FP and Carlson solvers, WGSL kernels
bakes/mascon/        mascon voxel direct sum (C++)
bakes/werner/        polyhedral closed form over the ESA reference library (C++)
src/server/          Bun development server, one endpoint per solver
src/viewer/          Bevy + WebGPU viewer (Rust, compiled to WASM)
src/web/             single-page control surface
assets/density/      the density field (TOML)
assets/records/      finished per-face records, one per solver
tools/               record comparison and other helpers
docs/images/         figures used by this README
```

## License

MIT — see [LICENSE](LICENSE). Contributions follow [CONTRIBUTING.md](CONTRIBUTING.md).
