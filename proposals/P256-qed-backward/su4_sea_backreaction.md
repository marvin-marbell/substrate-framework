# Balanced occupied-sea SU(4) backreaction at a uniform selected saddle

`su4_sea_backreaction.py` calculates the **full 45-real-direction static,
uniform (`q=0`) colour-link Hessian**: three axes times all fifteen normalized
SU(4) generators. It includes the occupied sea, not just bare Higgs curvature.
It does not calculate the electromagnetic photon response, impose SU(4) Gauss
projection, or establish a quantum, finite-wavevector, global or nonlinear Higgs
phase. The balanced mean-colour construction below is a distinct microscopic
law from the one-cell `J=0` quantum Gauss model; no joining of their domains is
assumed.

## Microscopic law and the actually selected frame

The light species is a fundamental with the endpoint-averaged primitive link law
of `RotationBand`/`scalar_paths`. Its bare parameters are `a=1`, positive `ma`,
`r=v=1`, and `improved=True`. The heavy fundamental is electromagnetically neutral
and has the **whole one-body Hamiltonian**

\[
 h_H(p,U)=-\Lambda h_L(p,U),\qquad \Lambda>0.
\]

This reverses the onsite, nearest and improved two-step terms together. It is not
a sign flip of only a mass term. Each species fills its actual two lowest levels.
At identity links their occupied projectors are complementary:
`P_H=I-P_L`. Hence the combined local one-body density is `I` and every traceless
colour generator has zero mean. The implementation independently diagonalizes
`-Lambda*h` to measure that complement and its energy. It reports the local
Slater-sea colour covariance; the mean cancellation **does not set colour
variance to zero and is not Gauss projection**. No local fixed-occupation or
centre eigenstate is inferred from global half filling.

The quartic bare orbital potential is

\[
 V_0=\frac{\Lambda\kappa}{2}\|G\|_F^2,\qquad
 G_{\mu\nu}=\tfrac14\operatorname{Tr}(\Phi_\mu\Phi_\nu),
\]

whose bare minimum is zero. The program calls
`heavy_orbital_join.selected_saddle(order, ma, Lambda, kappa, heavy_ma=ma)`
with the same common positive Brillouin-zone weights. It uses the resulting
actual frame, rather than inserting an assumed `PhiGamma` response. The existing
saddle's same-sign heavy sea and the opposite heavy law used here have the same
integrated energy; their occupied colour projectors are different, and the
old same-sign population interpretation is not reused.

For any uniform unitary links the onsite trace vanishes and all hopping terms
have nonzero displacement. The continuous full-BZ trace therefore integrates
to zero, even when an individual `h(p,U)` has a nonzero trace. Ordered eigenvalues
give the pointwise identity

\[
 \sum_{o=1}^{2}E_{H,o}
 =-\Lambda\sum_{e=3}^{4}E_{L,e}
 =\Lambda\sum_{o=1}^{2}E_{L,o}-\Lambda\operatorname{Tr}h_L.
\]

Thus the continuous joint half-sea energy is `(Lambda+1)*E_light`. Finite weighted
quadratures can have trace-integration remainders. The receipt reports those,
pointwise non-Abelian traces, and the independently computed negative-heavy
finite differences, rather than imposing the scaling identity on those energies.

## Exact primitive link derivatives

`matrix_unit_basis()` has `Tr(Ba Bb)=4 delta_ab`. The generators are
`Ta=Ba/(2*sqrt(2))`, so `Tr(Ta Tb)=delta_ab/2`, checked in the JSON. Coordinates are
axis-major `(x,T0..T14),(y,T0..T14),(z,T0..T14)` and

\[
 U_i=\exp\!\left(ig\sum_a\theta_{i,a}T_a\right).
\]

The onsite scalar and each directed positive path's coefficient vector are
extracted from the existing `RotationBand` law. With
`F_d=sum_mu scalar_mu(d)*Phi_mu`, a single-axis path of length `l` contributes

\[
 e^{ip\cdot d}\frac{F_d U_i^l+U_i^l F_d}{2}+\mathrm{HC}.
\]

This is the ordered unitary endpoint law, not independent phases on the two
endpoints or a replacement linearized hopping law. At zero source,

\[
 V_{i,a}(p)=\sum_{d\parallel i}e^{ip\cdot d}
   \frac{ig l}{2}\{F_d,T_a\}+\mathrm{HC},
\]

\[
 C_{i a,i b}(p)=\sum_{d\parallel i}e^{ip\cdot d}
   \frac{-g^2 l^2}{4}\{F_d,T_aT_b+T_bT_a\}+\mathrm{HC}.
\]

Contacts across different axes are zero for the **actual single-axis paths**.
The two-step path therefore has factor two in the first derivative and factor
four in the contact. The calculation rejects a path outside this stated domain.

For the actual eigenvectors and eigenvalues of the `4x4` zero-link Hamiltonian,
with two occupied and two empty levels, the light half-sea Hessian is

\[
 (H_{\rm sea})_{AB}=\langle\operatorname{Tr}(P_o C_{AB})\rangle
 -2\operatorname{Re}\left\langle
 \sum_{o,e}\frac{(V_A)_{oe}(V_B)_{eo}}{E_e-E_o}\right\rangle.
\]

There is no hand-inserted gamma trace or response. A streamed occupied-to-empty
`2x2` feature per direction builds the weighted `45x45` Gram matrix. Contact
integration first accumulates the Fourier-weighted `4x4` occupied projectors and
then contracts constant anticommutators. Neither a whole-BZ stack of all
`45x4x4` derivative matrices nor a whole-BZ contact tensor is allocated.

## Bare restoring and a genuine instability threshold

The actual nearest-link bare alignment energy per cell is

\[
 E_{\rm Higgs}=\frac{\Lambda J_{\rm pref}}8
 \sum_i\|\Phi-U_i\Phi U_i^\dagger\|_{\rm HS}^2.
\]

The sum includes all four orbital fields. Its Hessian is
`Lambda*Jpref*Mbase`, where `Mbase` is three identical blocks of the matrix
returned as element zero by `gauge_mass(frame, Ta, alignment=1, coupling=g)`.
Consequently

\[
 H_{\rm total}=(\Lambda+1)H_{\rm sea}+\Lambda J_{\rm pref}M_{\rm base}.
\]

The receipt reports full matrices, contact/pair/bare/total spectra and raw
negative, positive and numerically zero eigenvalue counts. It forms the symmetric
inverse-root generalized matrix

\[
 K=M_{\rm base}^{-1/2}\frac{\Lambda+1}{\Lambda}
      H_{\rm sea}M_{\rm base}^{-1/2},\qquad
 J_{\rm critical}=\max(0,-\lambda_{\min}(K)).
\]

The raw generalized spectrum and its minimum are reported **before** the
nonnegative threshold definition. A strictly larger scalar `Jpref` suffices for
positive definiteness of this particular static `q=0` Hessian. The program shows
`Jcritical + max(0.1, 0.05*Jcritical)` as a separate explicit safe scalar
**parameter hypothesis**, not a dynamically selected stiffness. It never changes
the requested `Jpref` to conceal an instability. The default `Jpref=1` may be
unstable, which is a scientifically meaningful output, not a failed numerical
control and not a stopping condition for the larger construction.

Both sea and bare matrices scale as `g^2`, so the generalized threshold is
independent of that coupling normalization. A second coupling rebuilds the actual
derivatives and eigenspace Gram; the control enforces `g²`-normalized sea/bare
matrix agreement **and the raw generalized spectrum**, before clipping its
minimum to define `Jcritical`. Agreement of two clipped zero thresholds alone
does not test units.

## Discriminating controls and refinement

* Two full, dense, pairwise noncommuting three-axis colour directions use the
  actual exponential links and fresh `4x4` spectra of **both** `h` and `-Lambda*h`.
  The first direction is chosen from three fixed trigonometric candidates to
  resolve both analytic pieces, before any finite differences are calculated.
  Central second differences at `0.005` and `0.0025` report component energies,
  predicted curvature, errors and convergence, plus separate omitted-contact and
  omitted-pair predictions. Numerical cancellation has an explicitly reported
  roundoff allowance. The inert quartic energy is reported separately; central
  differences sum the actual sea and Higgs component differences to avoid adding
  and subtracting that irrelevant constant.
* A genuinely off-spin whole-colour conjugation moves the mass orbital field.
  It conjugates **both the frame and the links**, computes the resulting energy,
  and rebuilds all 45 directions of the sea Hessian in the original generator
  basis. The reported covariance is `H'=R H R^T` in all three axes. Omitting the
  link conjugation is a separate finite-source adverse control.
* The scalar identity control is intentionally not an SU(4) direction. A uniform
  identity link `exp(i theta_i I)` exactly shifts `p_i` by `theta_i/a` in the
  actual primitive Hamiltonian. Its bare Higgs energy is zero and its **continuous
  normalized full-BZ** sea Hessian is zero. A nonzero finite weighted-quadrature
  remainder is retained rather than subtracted. The program refines this control
  at all requested orders with the **same finest selected frame** so a changing
  saddle cannot masquerade as quadrature convergence. It does not assert exact
  finite-grid Ward cancellation or assume monotonic convergence.

The default midpoint orders are 24 and 32. `split_gauss` uses the common positive
weighted `brillouin_quadrature` directly, with recommended orders 8 and 10. The
JSON reports the selected-frame/Hessian/threshold changes and identity-control
remainders. These are finite numerical estimates, not a rigorous error bound on
the continuum integral.

## Portable execution

Use Python with the project's scientific dependencies installed; no private
machine paths or environment-specific notes are required:

```sh
python proposals/P256-qed-backward/su4_sea_backreaction.py \
  --orders 24 32 --light-ma 0.2 --Lambda 64 --kappa 1 \
  --g 0.7 --Jpref 1 --output su4_sea_backreaction_receipt.json

python proposals/P256-qed-backward/su4_sea_backreaction.py \
  --quadrature split_gauss --orders 8 10 --light-ma 0.2 --Lambda 64 \
  --g 0.7 --Jpref 1 --output su4_sea_backreaction_gauss_receipt.json
```

After inspecting a measured threshold, a separate run with an explicitly chosen
`--Jpref` above it tests that scalar-parameter hypothesis on the same law. The
CLI never automatically replaces the supplied value. `--steps` accepts two
strictly decreasing positive finite-difference steps; `--chunk-size` only changes
streaming allocation. Output contains SHA-256 hashes of the implementation,
companion note and directly relevant source owners. Exit status concerns the
implemented numerical controls; `all_supplied_Jpref_static_stable` separately
reports the physical sign outcome. Scientific execution belongs to the parent
integration boundary, not the worker's implementation handoff.

## Executed static colour response

Both displayed parent-owned commands naturally exited zero, in 20.08 s
(midpoint) and 123.79 s (`split_gauss`). The
[midpoint receipt](su4_sea_backreaction_receipt.json) and
[`split_gauss` receipt](su4_sea_backreaction_gauss_receipt.json) retain actual
implementation/source manifests, including the repaired implementation SHA256
`7354d5279bf8313a56add25bf0b2ec2f1f115da710d145b459016f53c1c29500`.
`all_numerical_controls_pass` and `all_supplied_Jpref_static_stable` are true
in both. At `Lambda=64`, `g=.7`, `Jpref=1`:

| Quadrature | single-light sea minimum | total 45-direction minimum | identity-U(1) Hessian norm |
|---|---:|---:|---:|
| Midpoint 24 | 0.000180797269 | 1.72608843589 | 6.606e-8 |
| Midpoint 32 | 0.000180799704 | 1.72608859524 | 1.448e-10 |
| Split Gauss 8 | 0.000180424053 | 1.72605989950 | 9.585e-6 |
| Split Gauss 10 | 0.000180829497 | 1.72609060440 | 7.576e-7 |

The sea Hessian itself is positive in these finite estimates, so the computed generalized `Jcritical` is zero; no compensating stiffness was silently inserted. The midpoint sea-Hessian change is `6.72e-9`. These observations support this saddle's static uniform curvature, including all colour directions; they do not prove its full quantum, finite-wavevector or nonlinear phase, or identify the separate fixed-orbit Gauss consumer as this law's quantum ground.

The alternate coupling is `.959`. Across all four orders the rebuilt
`g²`-normalized sea and bare relative residuals are at most `3.56e-16` and
`7.63e-17`; raw generalized-spectrum relative residuals are at most
`4.04e-15`. All unit-control checks pass. The explicit wrong-linear-scaling
alternative would give relative spectral mismatch
`abs(1-.7/.959)=.27007299` despite both clipped `Jcritical` values staying
zero; that arithmetic is an adverse unit witness, not another scientific run.
The receipt enum is exactly `split_gauss`, not `gauss`.

The [exact Haar provider](exact_haar_alignment.md) now also enters an executed
`Jpref=.2,R=2` [Gauss-photon EFT](quantum_su4_photon.md), with a
[same-H dressed charge carrier](gauss_photon_charge.md). This static uniform
sea Hessian does not establish that EFT as its quantum ground, nor select the
unrestricted [full-band covariant section](full_band_gauss_section.md) as a
phase. The existing proof and receipt-time source attribution are preserved;
this evidence update does not claim a rerun or completion of the campaign.

