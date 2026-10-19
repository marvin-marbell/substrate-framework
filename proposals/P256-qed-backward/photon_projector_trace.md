# Exact primitive photon tensor and periodic contact representation

`angular_photon_spectrum.py` evaluates the continuous-transfer occupied-sea
response using the actual selected scalar `RotationBand.couplings()`.
`photon_projector_trace.py` compares it with independent full 4x4 matrix
occupied/empty eigenstate sums. This is an algebraic reduction of the same
quadratic response, not a different photon approximation.

## Positive spectral trace

With normalized four-component band directions a,b and real vertex
coefficients u_i,u_j, the derived Clifford projectors give

\[
\operatorname{tr}(P_+(a)u_i\!\cdot\!\Gamma P_-(b)u_j\!\cdot\!\Gamma)
=(1+a\!\cdot b)(u_i\!\cdot u_j)
 -(a\!\cdot u_i)(b\!\cdot u_j)
 -(a\!\cdot u_j)(b\!\cdot u_i).
\]

The 3x3 tensor is positive semidefinite: it is the Gram matrix of actual
occupied-to-empty current amplitudes. Both transfer orientations are retained,
with their actual positive energy sums. The existing spectral kernels,
subthreshold poles and residues therefore retain their normalization.
Positive normalized Brillouin weights enter selection and response together;
split-Gauss order n means n nodes per each of six axis segments, `(6n)^3`
points, not n³ points.

## A discriminating magnetic quadrature probe

The magnetic coefficient divides a contact-minus-pair cancellation by |q|².
A small quadrature error in the contact can dominate a low-q probe even when
the electric spectral derivative is stable. On the selected ma=.2 law at
physical matching momentum/mass .04, split-Gauss orders 8/10 gave direct
magnetic coefficients `.2340678268/.05811363975`.

The **same continuum integral** has a better-conditioned representation.
For every actual primitive straight link and its full midpoint path factor,
periodicity gives

\[
\int_p\operatorname{tr}(P_-\partial_i^2h_d)
=-\int_p\operatorname{tr}((\partial_iP_-)\partial_i h_d).
\]

The path factor multiplies each link contribution by `(path/steps)^2`.
The projector derivative is evaluated from the full band, not that isolated
link. For `P_-=(I-n·Gamma)/2`, the scalar form uses
`partial_i n=(partial_i d-n(n·partial_i d))/E`. The resulting zero-transfer
contact cancels the pair response pointwise rather than amplifying an
integration error by 1/|q|². It is not a subtraction of an observed negative
curvature or deletion of a contact term.

The diagnostic gives `.06965822501/.06962849981` with this representation.
It applies to the **periodic thermodynamic Brillouin integral**, not a discrete
finite-torus sum. The finite-torus `vacuum_limit` law remains unchanged.
Finite quadrature can still have longitudinal/shift errors; integration by
parts is not an exact arbitrary-transfer finite-grid Ward theorem.

## Actual independent matrix control and consumer

```sh
python proposals/P256-qed-backward/photon_projector_trace.py \
  --output proposals/P256-qed-backward/photon_periodic_contact_receipt.json
python proposals/P256-qed-backward/elastic_join.py \
  --quadrature split_gauss --orders 8 10 --mass-ratios .2 \
  --output proposals/P256-qed-backward/elastic_join_periodic_receipt.json
```

Both complete commands naturally exit zero. The independent matrix contact
uses the occupied/empty spectral projector derivative, not the scalar
Clifford derivative. Contact differences are below `2.72e-16`; spectral
static/derivative/change differences are below `2.08e-15`. The direct-link
versus by-parts finite-quadrature difference is recorded separately, not
required to vanish at an insufficient order.

The full nonforward same-sea rate is `1.67502696174e-5` at order8 and
`1.67503297366e-5` at order10; their relative difference is about `3.59e-6`.
The corresponding QED reference deviations are `.00233087/.00232942`:
quadrature stability does not erase the finite-model physical difference.

## Analytic infrared data, not a finite matching probe

`ContinuousSea.infrared()` differentiates the same primitive spectral integral
at exactly zero transfer. For transfer along axis a and transverse vertex
axis b, put \(R=|d|,\ n=d/R,\ u=-q_e\partial_b d/2\), and
\[
 W(t)=(1+n(p+t\hat a)\cdot n(p))u^2
       -2(n(p+t\hat a)\cdot u)(n(p)\cdot u).
\]
The straight-link transverse vertex has no transfer dependence when a differs
from b. Writing primes for a derivatives at zero transfer gives
\[
 \Delta\chi_E=\left\langle W(0)/R^3\right\rangle_p,\qquad
 \Delta\chi_B=-4\left\langle
 \frac{W''}{2R}-\frac{W'R'}{2R^2}
 +W(0)\left(\frac{R'^2}{4R^3}-\frac{R''}{4R^2}\right)
 \right\rangle_p.
\]
Here \(W'=-2(n'\cdot u)(n\cdot u)\) and
\(W''=-|n'|^2u^2-2(n''\cdot u)(n\cdot u)\).
Actual primitive first/second link derivatives supply R', R'', n', n'';
the by-parts contact cancels the zero-transfer pair term pointwise.
No small-q division, fitted intercept or subtracted observed curvature
defines these coefficients. Normalized BZ integration error remains separate.

The actual CLI `photon_infrared.py` executes q/m `.08,.04,.02,.01` and
compares both finite response coefficients and the full photon poles/residues
with these derivatives. Midpoint24/32/48 at ma=.4 and weighted8/10 at ma=.2
naturally exit zero. At midpoint48, ma=.4, the analytic coefficients are
`.0284327618767` electric and `.0527184927431` magnetic; the actual q=.004
pole speed differs from the infrared speed by `-6.67341e-7` and its canonical
source by `2.49917e-10`. The errors decrease with q². Treating the finite
q/m=.04 row as the exact infrared baseline fails the same electric and
magnetic convergence checks at every executed midpoint order.

The [inclusive join](elastic_join.md) uses
\(e_{\rm IR}=(\chi_E+\Delta\chi_E)^{-1/2}\) and
\(c_{\rm IR}^2=(\chi_B+\Delta\chi_B)/(\chi_E+\Delta\chi_E)\).
Its independently chosen finite matching probe and selected matter velocity
are not relabeled as exact infrared data.
