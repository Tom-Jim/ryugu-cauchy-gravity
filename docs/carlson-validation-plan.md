# Carlson validation and performance gates

No speedup or accuracy number is accepted from the pre-rewrite records. The
following gates are intentionally separated so geometry, quadrature and
special-function errors cannot cancel each other.

## Numerical gates

1. Compare GPU `RF/RD/RJ/RC` with the Rust `ellip` f64 bridge on valid arguments
   sampled logarithmically over the f32 normal range. Record domain rejection,
   iteration count, relative error and ULP error. Target: no silent non-finite
   result and relative error below `3e-6` away from singular limits.
2. Exercise exact identities independently: `RF(x,x,x)=x^-1/2`,
   `RD(x,x,x)=RJ(x,x,x,x)=x^-3/2`, `RC(x,x)=x^-1/2`, and
   `RJ(x,y,z,z)=RD(x,y,z)`.
3. Before adding any non-degenerate quartic production path, validate
   repeated-root and lower-degree limits and prove that the transformed radical
   is real on the complete endpoint interval with every `RJ` pole outside it.
4. Compare the `alpha=1` RC-backed fixed-endpoint beta derivative with central
   differences of the closed RT-FP radial formula and high-order f64 quadrature
   on every emitted ray interval.
5. Compare the hybrid spherical identity against direct high-order quadrature
   of `T_ij*S`. Include moving-endpoint cases and closed meshes with non-coplanar
   adjacent faces; a boundary-only result is not an acceptable reference.
6. Compare complete spherical rules only. Direction count, radial rule and BVH
   tolerances are refined independently.

## Near-surface acceptance targets

For common observation points and continuous density data:

- uniform paths versus f64 Werner: RMS relative tensor residual `<= 5e-6`;
- Cauchy at heights from 1 mm through 1 m: `<= 2e-3` against the f64
  finite-part reference, with no monotone growth toward the surface;
- fractional Cauchy over the same range: `<= 5e-3` initially, then refine the
  radial rule until the 8-to-16-node difference is below one quarter of the
  reported residual;
- exterior harmonic check: `|trace(H)|/||H||F <= 2e-5` in f32;
- any hit, interval, BVH-stack or Carlson-domain overflow is a failed sample,
  never a truncated result.

These are acceptance thresholds, not measured results.

## Performance gates

- Measure GPU timestamp time separately from asset construction, submission,
  readback and chart rendering. The UI chart keeps wall time per point as a
  separate user-visible metric.
- Record peak interval storage as
  `points * directions * (16 * sizeof(vec4<f32>) + sizeof(u32))` and keep ray
  browser blocks at 512 points (36 MiB at 288 directions). Direction tiling is
  still the next step if lower-memory adapters must be supported.
- The `alpha=1` Carlson branch must beat the former GL16 branch before it is
  called an optimization. RF/RD/RJ duplication is enabled only where a valid
  genus-one reduction removes more quadrature work than it adds.
- Reuse pipelines and immutable mesh/BVH/direction buffers. A later scratch
  arena may reuse per-block interval/output buffers, but only after abort and
  validation-error lifetimes are covered.

## Non-degenerate real-arc status

There is no production non-degenerate real-arc coefficient generator. The
earlier disconnected scaffold was removed because it neither supplied the
physical two-positive-quadratic map nor participated in a solver. Adding it
again requires the complete f64 principal-branch and adjacent-edge gates above,
plus elementary fallbacks for repeated roots and lower-degree curves.

The attempted purely local boundary generator is intentionally not a production
dependency: it cannot represent `T_ij*S dOmega` for arbitrary `S` on the closed
sphere without the interior derivative residual. The production hybrid path
performs the adjacent-edge cancellation analytically and retains that residual.
An adjacent-trace validator remains a required gate for any future explicit arc
packets; non-degenerate packets must not be dispatched until both incident
half-edges have passed it.
