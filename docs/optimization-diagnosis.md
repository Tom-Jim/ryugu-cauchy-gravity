# Numerical and performance diagnosis

This note records the implementation contract behind the three browser
diagnostics. Numerical targets below are acceptance gates, not measurements;
the rewritten code must be benchmarked separately.

## 1. Pareto sweep

The former Mascon V-shape mixed three effects: cold asset/pipeline setup in the
first sample, a representation-error floor from the voxel field, and dominated
small-theta samples whose extra tree work no longer improved that field. A true
Pareto plot must discard points for which another sample is both faster and
more accurate. The diagnostic now performs a one-point untimed warm-up and
draws only non-dominated samples.

The former RT-FP/Carlson plateaus were primarily angular-error floors. Prefixes
of a spherical rule are not spherical quadrature rules; they bias both total
weight and second moments. Every direction sweep now constructs a complete
rule. CarlsonAlpha formerly swept GL order while fixing only 64 directions, so
the angular floor hid radial convergence. Its sweep now couples GL4/8/16 with
32/64/128 directions.

Remaining limits are deliberately separated:

- ray/BVH: determinant gating, per-hit ULP merge tolerance, hit/interval/stack
  capacity (every overflow is fatal);
- radial: RC endpoint arithmetic for alpha=1, complete GL rules for general
  alpha;
- angular: complete direction rules only;
- timing: batch wall time per point, after warm-up, including submission and
  readback. It is not labelled as isolated shader time.

## 2. Near-surface residual

Uniform density is nearly exact because the polyhedral boundary formula is its
native representation. Continuous Cauchy fields add a direction-dependent
finite-part remainder. Their 1 mm--1 m error can therefore expose endpoint
motion, grazing rays, angular under-resolution and density approximation even
when the uniform tensor is correct.

The implemented fixes are:

- retain all non-convex ray intervals rather than a single star-shaped exit;
- bind hit merge tolerance to triangle scale, incidence and f32 ULP size;
- store the entry and exit triangle id for every interval;
- differentiate each endpoint exactly as
  `grad_S R=-R[n-(n.u)u]/(n.u)`;
- retain the spherical derivative residual instead of declaring it zero;
- use a cancellation-safe endpoint integrand and exact RC-backed alpha=1 beta
  derivative; and
- compare Cauchy candidates on a common 128-direction rule against a cached
  full-direction RT-FP reference. Fractional candidates use GL16 consistently.

Initial acceptance goals are RMS relative tensor residual below `2e-3` for
Cauchy and `5e-3` for fractional Cauchy over 1 mm--1 m, with no systematic
growth toward the surface. These goals are not yet reported as achieved.

## 3. Tensor consistency

Six-component storage enforces `H_ij=H_ji` by construction, so it cannot
independently measure antisymmetric accumulation. The former approximately
`2.5e-16` horizontal line was also below the precision of the f32 GPU output
and could not establish a machine-precision result.

The chart now reports the aggregate exterior harmonic residual

```text
sqrt(sum trace(H)^2) / sqrt(sum ||H||_F^2)
```

and censors values at f32 epsilon. An independent conservative-field test still
requires either nine unsymmetrized components or finite-difference curl loops.
Symmetric projection is applied before workgroup reduction so reduction order
cannot create a hidden displayed asymmetry.

## 4. Solver-specific implementation state

### Werner

Werner remains the uniform-density reference. Triangle invariants are uploaded
once, face work is distributed over 64 lanes per observation point, and the
six tensor components are reduced in workgroup memory. Its main unavoidable
cost is all-face evaluation; changing that would change the reference role.

### Mascon

The runtime voxel table and Barnes--Hut tree remain cached immutable GPU
buffers. The opening angle is the accuracy control. Smaller theta cannot repair
voxel representation error, so dominated samples are removed from the Pareto
frontier. Near nodes continue to use direct sums; far nodes use multipoles.

### RT-FP

RT-FP retains an independent direct `T*S` angular integral. It shares only the
ray/BVH interval infrastructure with Carlson. One 64-lane workgroup now owns an
observation point and divides the direction rule among its lanes; the former
single-invocation direction loop was a principal browser stall.

### Carlson

Carlson uses the continuous alpha=1 density and the hybrid spherical identity.
The fixed-endpoint beta derivative is exact and its angle difference uses RC
duplication. Moving endpoints use stored face ids. The nonzero derivative
residual is retained, so the method is not advertised as residue-free. The
all-face elementary near tensor uses a cancellation-safe piecewise `asinh`;
performing three RC duplications per face there was redundant and dominated
the runtime.

### CarlsonAlpha

CarlsonAlpha uses the same geometry derivative. Alpha=1 takes the RC exact
branch; other positive exponents integrate `2*alpha*sigma^2*D^(-alpha-1)` with
complete GL4/8/16 rules. Arbitrary real alpha is not falsely labelled genus
one. RF/RD/RJ remain implemented and independently callable, but no physical
quartic packet is dispatched until a complete real-principal generator exists.

## 5. Browser scheduling and memory

Analytic, inside-probe, BVH-ray, remainder and tensor-composition passes are
encoded into one command buffer. There is no intermediate CPU readback.
Immutable mesh, BVH, directions, kernels and pipelines are cached.

Endpoint face ids increase an interval slot from 8 to 16 bytes. At 288
directions and 16 slots, 1024 points would require 72 MiB in one storage
binding. Browser ray blocks are therefore capped at 512 points (36 MiB), below
common 64 MiB binding limits and with shorter UI-yield intervals. The remaining
readback is only the final scalar/tensor block plus the overflow word.

The next performance gate is timestamp-query profiling on adapters that expose
it. Until then, no shader-only speedup ratio is claimed from wall-clock charts.
