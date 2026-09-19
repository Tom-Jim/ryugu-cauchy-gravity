# Carlson symmetric-integral and hybrid-residual contract

The production Carlson backend uses the real symmetric integrals `RF`, `RD`,
`RJ`, and `RC` with the domain conventions of Carlson (1995). Duplication
contracts normalized argument differences by a factor of four. Once those
differences are small, the remaining symmetric integral is evaluated by its
fixed Taylor polynomial about the common mean.

## Corrected derivative identity

For

```text
Q(s) = sqrt((s+x)(s+y)(s+z)),
```

direct differentiation gives

```text
d [(s+z)/Q] / ds
  = 1/Q
    - (s+z)/(2Q) * [1/(s+x) + 1/(s+y) + 1/(s+z)].
```

The former expression `1/Q - (s+z)^2/(2 Q^3)` is not an identity and must not
be used to generate reduction coefficients.

Eliminating the `(s+z)/(s+z)` factor gives the constant-coefficient form

```text
d [(s+z)/Q] / ds
  = -1/(2Q)
    - (z-x)/(2(s+x)Q)
    - (z-y)/(2(s+y)Q).
```

This is the form a future non-degenerate reduction must use; no generated
coefficient may depend on the integration variable. The production hybrid
path does not need or instantiate such a generator.

For a quadratic numerator `P(s)=a2*s^2+a1*s+a0`, the valid first step is

```text
P(s)/(s+p)
  = a2*s + (a1-a2*p) + (a0-a1*p+a2*p^2)/(s+p).
```

Consequently, after the endpoint map has produced the canonical Carlson
arguments, the third-kind coefficient is

```text
AJ = (2/3) * (a0-a1*p+a2*p^2).
```

The `RF` and `RD` coefficients depend on that endpoint map and its algebraic
boundary term; they are not universal functions of `a0,a1,a2,p` alone.
Moreover, an integral on `[0,infinity)` with an uncancelled quadratic numerator
is divergent. A finite-arc transformation must retain its endpoint terms and
cancellations before applying the canonical tail formulas.

## Quartic branch contract (not a production path)

If a tangent-half-angle boundary implementation is added, its quartic must be
coefficient-scaled before its discriminant is evaluated. A nonzero
discriminant alone does not select a usable real Carlson path: repeated roots,
lower degree, the sign of the radical on the complete interval and every `RJ`
pole must be classified together.

The discriminant is an algebraic-curve classifier, not a real-path domain
test. The concrete endpoint map must still prove that its radical is real and
that no `RJ` pole lies on the integration interval.

For the physical Cauchy ray problem the tangent-half-angle radical has extra
structure. With `K=sigma^-2+|p|^2` and
`N=beta1*(1-s^2)+2*beta2*s`,

```text
P4(s) = K*(1+s^2)^2 - N(s)^2
      = [sqrt(K)*(1+s^2)-N(s)] [sqrt(K)*(1+s^2)+N(s)].
```

Cauchy--Schwarz and `K>|p|^2` make both quadratic factors strictly positive
on every real arc. Thus the production real branch does not require four real
roots. DLMF 19.29.24--25 (two positive quadratics) is the relevant genus-one
classification. The repository keeps this as an acceptance contract rather
than shipping an unused four-affine-factor mapper.

## Requirements for a future finite real-interval endpoint map

Write the positive branch on an oriented interval as

```text
P4(t) = c product_i L_i(t),   L_i(t)=a_i+b_i t,   c>0,
```

with every `L_i` strictly positive at both endpoints. Since each factor is
affine, the endpoint test proves positivity on the complete interval. With
`X_i=sqrt(L_i(upper))`, `Y_i=sqrt(L_i(lower))`, and the DLMF endpoint
combinations `U_ij`, the first-kind term is

```text
integral dt/sqrt(P4) = 2 RF(U_12^2,U_13^2,U_23^2)/sqrt(c).
```

Ratio terms must reduce respectively to `RD` plus an algebraic endpoint term,
and to `RJ` plus `RC`, as in DLMF 19.29.7 and 19.29.8. A future third-kind map
must compute both equivalent DLMF 19.29.9 formulas for `U_alpha5^2`, reject
scale-aware disagreement, and repeat all domain checks after f64-to-f32
conversion.

## Global one-form obstruction and retained residual

Let `T_ij(u)=3*u_i*u_j-delta_ij`. The proposed local vector field

```text
A_ij = 1/2 [(e_i.u)(e_j cross u) + (e_j.u)(e_i cross u)]
```

is divergence-free on the sphere; it cannot satisfy `div_S A_ij=T_ij`.
The correct local primitive field is

```text
B_ij = -(1/6) grad_S T_ij,      div_S B_ij = T_ij.
```

For a scalar ray remainder `S`, integration by parts gives

```text
T_ij S dOmega = d(S i_Bij dOmega) - (B_ij.grad_S S) dOmega.
```

There is no smooth local one-form whose exterior derivative equals `T_ij*S`
for arbitrary `S` on the complete sphere: the integral of an exact two-form on
a closed sphere is zero, while choosing `S=T_ij` gives a strictly positive
integral. Therefore the second term is mandatory, not an implementation defect.

On a closed, consistently oriented polyhedral direction partition the trace of
the first term is the same one-form on both sides of every shared arc and its
two orientations cancel. Production consequently evaluates the exact hybrid
identity

```text
integral T_ij S dOmega
  = (1/6) integral grad_S(T_ij).grad_S(S) dOmega.
```

The ray pass stores the triangle id at both interval endpoints. The residual
kernel differentiates `R=(n.x-h)/(n.u)` as

```text
grad_S R = -R [n-(n.u)u]/(n.u),
```

and applies the Leibniz endpoint terms together with the fixed-endpoint density
derivative. Hence the implementation neither assumes that the residual
vanishes nor introduces a face-local gauge on a shared edge.

Any future explicit boundary decomposition must reject the complete packet set
unless every undirected edge has two opposite, ULP-equal traces. Such packets
are not dispatched by the hybrid production path, because all internal traces
cancel before quadrature.

General positive real density exponents are not all elliptic. CarlsonAlpha uses
Carlson forms only for terms that have actually reduced to genus one. Its exact
`alpha=1` fixed-endpoint derivative uses the degenerate `RC` branch, while other
exponents use complete cancellation-safe GL4/GL8/GL16 radial rules for
`partial_beta S`; moving-endpoint derivatives remain analytic.
