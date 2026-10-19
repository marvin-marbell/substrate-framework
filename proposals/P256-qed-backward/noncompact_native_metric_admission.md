# Original native full-ten Gram admission, uniform in all shifts

This producer supplies the previously missing quantitative membership for the
Sharp event23 **declared** box: temporal Schur `s>=.40`, `G00<=.43`, each
`Gjj<=.065`, and `sqrt((G00-s)/s)<=.002`. Parameters remain original
`a=1,m=.2,r=.12,v=.3,kappa=2,preferred=0,improved=False`. The theorem concerns
the flat uniform whole-heavy sector, not arbitrary nonuniform configurations.
It is not sample extrapolation, a diagonal constraint on native metrics, or a
new minimizer. `soft_budget.select_gram` proposes the cubic trial;
`original_charged_principal.original_frame(..., native_toron=True)` supplies the
existing actual full-ten native consumer.

## 1. Actual objective and residual distance

Reuse `bulk_global_gram.md`: fixed TOTAL half filling over all modes gives the
unrestricted all60 uniform-field lower bound, attained by Clifford fields,

`F(G)=kappa ||G||_F^2/2 - 2 average sqrt(x^T G x), G>=0`,

where `x0=m+r sum_j(1-cos p_j)`, `xj=v sin p_j`. No local selector, forced-two
occupation, or absolute-value replacement of the original heavy law is used.
The gradient at a positive trial is

`grad F(G)=kappa G-average xx^T/sqrt(x^T G x)`.

The negative square-root term is convex, so F is kappa-strongly convex in the
FULL symmetric-matrix Frobenius metric. Monotonicity and the PSD minimizer's
variational inequality imply, for any positive trial,

`||Gstar-Ghat||_F <= ||grad F(Ghat)||_F/kappa`.

This remains true if the minimizer initially lies on the cone boundary.
The resulting small distance proves it is interior here. Uniqueness and
continuum sign/permutation symmetry force the CONTINUUM metric to be cubic
diagonal. This does not constrain any shifted finite-grid minimizer.

## 2. Analytic full-ten integrand on a genuine complex polystrip

Take the actual cubic diagonal trial `Ghat=diag(g0,gs,gs,gs)`, positive, and
Clifford `h(z)=sqrt(g0)x0(z) beta+sqrt(gs) sum_j xj(z) gamma_j`. For
`z=p+iy`, `|yj|<=sigma`, the Hermitian part of beta h has lower bound

`delta=sqrt(g0)[m-3r(cosh sigma-1)]-v sqrt(3gs) sinh sigma`.

Indeed the real scalar Wilson part is at least the first bracket; beta gamma_j
are anti-Hermitian, and only the imaginary sine part contributes to the
Hermitian part, with anticommuting norm at most
`v sqrt(gs sum_j cos^2(pj) sinh^2(yj))<=v sqrt(3gs)sinh sigma`.
The imaginary scalar Wilson part contributes nothing. A certified positive
delta makes beta h strictly accretive, hence the least singular value of h is
at least delta throughout the CLOSED strip, not merely at sampled points.

Clifford algebra gives `h(z)^2=q(z)I`, `q=x^T Ghat x`. Every eigenvalue of h
has modulus at least delta, so `q!=0` and `|sqrt(q)|>=delta` everywhere.
The open polystrip is simply connected. Consequently q has a holomorphic log,
and the square root is selected by its positive value on the real torus.
Period translations preserve that branch: their quotient is a constant sign,
which is positive on the real torus. The square root and all ten integrands
are therefore holomorphic and 2pi-periodic. Positivity of delta on the closed
strip also gives a slightly larger open domain, legitimizing contour shifts
right up to its boundary. No assertion that q has positive real part is needed.

For the matrix integrand `f(z)=x(z)x(z)^T/sqrt(q(z))`, complex Frobenius norm
satisfies `||xx^T||_F=sum_mu |x_mu|^2`. The ACTUAL explicit strip bound is

`||f(z)||_F <= M = [(m+3r(1+cosh sigma))^2+3v^2 cosh^2 sigma]/delta`.

The certified consumer uses sigma=.5. Its proof applies to a tiny interval
around every decimal parameter, containing both exact decimals and native
binary64 BARE values; this pays parameter rounding without changing the law.

## 3. Exponential shifted-grid remainder, not mesh agreement

For the normalized continuum integral, simultaneous contour shifts in the
three coordinate directions give the matrix Fourier-coefficient bound

`||fhat(k)||_F <= M exp(-sigma |k|_1)`.

The normalized N-by-N-by-N trapezoid at ANY real offset aliases k=N ell and
adds only phase factors of modulus one. Thus, UNIFORMLY in the offset,

`||average_N,offset f-integral f||_F <= E_N`,

`E_N=M[((1+exp(-sigma N))/(1-exp(-sigma N)))^3-1]`.

This controls the full matrix, including all six offdiagonal coordinates.
All64 centres are a subset of these offsets; they require no separate sweep.
The error decreases with N. If an interval evaluation of the reference
midpoint grid of size Q bounds its full gradient norm by rQ, then

`||G_continuum-Ghat||_F <= (rQ+E_Q)/kappa`,

`||G_N,offset-Ghat||_F <= (rQ+E_Q+E_N)/kappa`.

The program encloses the reference evaluation using python-flint Arb, not
floating mesh agreement. Reflection makes each reference offdiagonal integral
EXACTLY zero; permutation makes all three spatial diagonals equal. It sums
positive interval contributions over the permutation/reflection-reduced
midpoint nodes. All ten gradient coordinates are accounted for: one temporal,
three spatial, six zero offdiagonals. Transcendentals, trial decimals, parameter
widths, sums, divisions, square roots, and strip/error arithmetic are ball
operations. Exported binary endpoints are rounded outward and their direction
is checked using Arb. The floating existing solver is only a trial generator;
its stopping tolerance is never an enclosure.

## 4. Explicit admitted box and Schur budget

For a certified distance d, write `G=[a,b^T;b,S]`. Frobenius accounting gives
`a in [g0-d,g0+d]`, each spatial diagonal in `[gs-d,gs+d]`,
`lambda_min(S)>=gs-d`, and `||b||^2<=d^2/2`. Therefore

`s=a-b^T S^-1 b >= sL=g0-d-d^2/[2(gs-d)]`,

`tilt<=d/sqrt(2(gs-d)sL)`.

The same Frobenius budget bounds every offdiagonal entry by `d/sqrt(2)`.
The CLI finds the first integer N where these sufficient inequalities fit the
EXACT declared box. This is the first N admitted by this bound, not a claim of
optimal physical onset. At and beyond it, the same exported G/Schur box holds
for EACH real shifted-grid minimizer, hence each retained all64 toron well.

## 5. Actual native consumer and reproducibility

The small shifted-grid consumer executes the existing full-ten minimizer,
actual `BulkQuantumLaw.onebody`, and actual TOTAL whole-half occupation. It
compares the original native spectrum's occupied sea with the Gram-attaining
sea and checks the actual all60 field stationarity. The nontrivial centre
(1,2,3) rejects a diagonal-only stationary substitute by its nonzero full-ten
offdiagonal gradient. An unpaid offdiagonal/Schur budget fails the declared
box. A mesh-only receipt with its analytic remainder removed is rejected.
These are consumer guards, not the global admission proof; sections 1–4 and
the interval receipt supply that proof.

Run with the archived Sharp event23 packet (its exact bytes and SHA256 are
bound into the output):

```
/home/administrator/substrate-framework/.venv/bin/python \
  noncompact_native_metric_admission.py --declared-packet /path/to/event23.json \
  --output /path/to/native-metric-receipt.json
```

Default reference Q64 uses 128-bit Arb and native size3. python-flint0.9.0 is
required in the scientific environment. Source hashes bind the inherited
original all60/whole-heavy supplier chain plus this pair and the actual native
principal pair. The output records input bytes, exact trial, certified full-ten
residual, strip constants, continuum enclosure, N_metric, uniform native
G/Schur bounds, rejected shortcuts, and actual native consumer. No charged
pole, real-time transport, nonuniform concentration, fixed-coupling continuum
UV theorem, or completed joined half-spin/scattering campaign is implied.

## 6. Observed admission and decision

The final archived attempt03 naturally exited0 in3.4267751260194927 seconds;
final real Pyright diagnostics were empty. Its Q64/Arb128 certificate reports:

| Bound | Certified value |
|---|---:|
| trial g0 | .4187253908699568 |
| trial gs | .0632221974274214 |
| full-ten reference gradient norm | <=1.836214803124742e-15 |
| strip coercivity, sigma=.5 | >=.03160497593259302 |
| full-matrix strip majorant M | <=40.384979337732446 |
| continuum quadrature error | <=3.0686523841788646e-12 |
| continuum distance to trial | <=1.5352442994909946e-12 |
| N_metric, uniform every real shift | **25** |
| G00, all N>=25 | [.41827388338567106,.4191768983542425] |
| each Gjj, all N>=25 | [.06277068994313571,.0636737049117071] |
| spatial least eigenvalue, all N>=25 | >=.06277068994400763 |
| temporal-spatial cross norm | <=.0003192640032783481 |
| temporal Schur | >=.4182722595473932 |
| tilt | <=.0019703436230670023 |

The sufficient bound at N24 has tilt<=.0032573287291253437 and does NOT
admit the declared box. No physical exclusion at N24 is claimed. Continuum
G00 lies in [.41872539086842153,.4187253908714921]; each continuum spatial
diagonal lies in [.06322219742588615,.06322219742895666]. Exact continuum
symmetry makes its actual offdiagonals zero, independently of the relaxed
distance-box values.

The native N3/centre(1,2,3) consumer has108 modes and54 occupied; its original
whole-half sea is-.7504729858891499 and the attained Gram sea is
-.75047298588915 (error1.1102230246251565e-16). Actual all60 field gradient
norm2.617378738287523e-16 and full-ten gradient norm2.722067100814323e-16 were
observed. Its temporal offdiagonal .0008028191107071212 is genuinely nonzero;
this small-grid observation is deliberately NOT substituted for admission.
All three adverse shortcuts were rejected at their actual assertions.

Attempt01 is retained: a zero-crossing Arb residual ball squared then square
rooted produced NaN from a tiny negative lower endpoint, causing honest
admission failure. The correction squares outward absolute residual endpoints;
the successful certificate does not discard a physical residual or suppress
the failure. Both natural exits/stdout/stderr are archived.

Decision delta: the exact Sharp event23 tighter family no longer needs a
conditional metric-membership assumption. Its combined paid onset remains
N>=max(25,108999)=108999 and T>=26.53508134094653 N^2; its H and UE constants
are unchanged. This admission does not establish downstream current-score
incidence factors or replace a receiver's own physical construction.

Two concrete successors are (1) select the now-admitted anchored even-loop
measure in the native two-receiver/recoil/current consumer at its paid onset,
and (2) derive diffusive marked Wilson-loop fourth/time moments on this SAME
box to lower the conservative photon/current constants. The first is selected
for the immediate joined construction; the second is an actual alternate
constant-improvement construction, not a review gate.

Evidence: `local://issue222-native-metric-attempt03-receipt.json`, archived
stdout/stderr and `local://issue222-native-metric-state-packet.json`. The
executed Python is unchanged since attempt03. This delivery paragraph was
updated afterward; the packet preserves both executed and published proof-note
hashes. Earlier successful attempt02 took4.268458423903212 seconds. Pyright
identified a possibly unbound loop-result variable; explicit initialization
removed the diagnostic and the final source rerun reproduced the same bounds.
