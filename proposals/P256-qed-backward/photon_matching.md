# A positive-residue dressed photon and its primitive scattering response

The same microscopic sea that supplies Compton's virtual electron–hole states now supplies the photon energy and external-state normalization. This is a constructive photon theorem in the **Gaussian microscopic rotor plus exact quadratic free-sea response** approximation. It joins matter and photon feedback instead of identifying a pure-gauge stiffness with the coupled model's charge.

## 1. One microscopic normalization

Begin with the actual continuous-time rotor Hamiltonian of [code_phase.md](code_phase.md), in unit lattice spacing:

\[
H_\gamma=\frac{U_E}{2}\sum_l E_l^2-K\sum_p\cos(\operatorname{curl}\theta)_p.
\]

Its small-field action has electric stiffness `chi_E0=1/U_E` and magnetic stiffness `chi_B0=K`. The executable chooses `U_E=.09`, `K=1/.09`, hence both stiffnesses `11.111111...`. These are declared Hamiltonian coefficients. They are **not** substituted from a four-dimensional Euclidean Monte Carlo receipt. Nonlinear compact-photon corrections can subsequently improve this Gaussian approximation.

For the transverse real cosine `theta_y(x)=t cos(Qx)`, let `N=L^3`, `norm=N/2`, and `Qhat=2 sin(Q/2)`. The exact [Fourier sea](vacuum_limit.md) supplies its contact C and spectral channels `(Delta,w)`, including both +/-Q. Define

\[
\Pi_0=\frac1{\mathrm{norm}}\left(C-2\sum\frac w\Delta\right),\quad
\eta_0=\frac2{\mathrm{norm}}\sum\frac w{\Delta^3}>0,\quad
S=\chi_{B0}\widehat Q^2+\Pi_0.
\]

`Pi_0=delta chi_B Qhat²`. The uniform electric coefficient and this finite-Q transverse coefficient are different observables; the implementation does not exchange them.

## 2. Positive pole theorem and a controlled error bracket

Write `s=omega²` and let `g=min Delta>0`. Below the pair gap,

\[
D(s)=S-\chi_{E0}s-s\eta(s),\qquad
\eta(s)=\frac2{\mathrm{norm}}\sum\frac w{\Delta(\Delta^2-s)}.
\]

This is the analytically continued spectral response of the actual occupied-to-empty intermediate states. Subtracting the static pair value before evaluating frequency avoids repeated subtraction of two extensive quantities.

For `0<=s<g²`, spectral weights are nonnegative and

\[
\eta_0\le\eta(s)\le\frac{\eta_0}{1-s/g^2},\qquad
-D'(s)=\chi_{E0}+\frac2{\mathrm{norm}}\sum
\frac{w\Delta}{(\Delta^2-s)^2}>0.
\]

Assume `S>0` and

\[
b=\frac S{\chi_{E0}+\eta_0}<g^2.
\]

Then `D(0)>0`, `D(b)<=0`, and strict monotonicity gives a **unique positive subthreshold photon pole** `s_*`. Its residue is positive and the pole has the rigorous bracket

\[
\boxed{\frac S{\chi_{E0}+\eta_0/(1-b/g^2)}\le s_*\le b.}
\]

Thus the low-frequency photon is not assumed to be a prescribed oscillator: its energy is constructed from the coupled matter response, with a finite-error bound. Below the pair threshold this quadratic spectral kernel has no pair-production imaginary part.

At the pole define

\[
Z_*^{-1}=-D'(s_*),\qquad e_*=\sqrt{Z_*}=1/\sqrt{Z_*^{-1}}.
\]

The required-band API uses physical momentum `q=2pi n/(a L)` and physical frequency directly. The compact-angle residue is `R_theta=-D'(s_*)`. With `theta=a A` and cell volume `a^3`, the continuum-volume field residue is `R_A=R_theta/a`, so the field leg is `sqrt(a/R_theta)` and the signed current coupling is `band.charge*sqrt(a/R_theta)`. The cell-normalized link leg is `1/sqrt(R_theta)`; a cell-normalized physical-A leg is instead `1/(a sqrt(R_theta))`. These are distinct normalization conventions, not interchangeable names.

`transverse_spectrum(L, band, ...)` and `match_photon(L, band, ..., electric_coefficient=UE1, magnetic_coefficient=K1)` require the actual `RotationBand`; it owns a,m,r,v,e and the hopping law. `electric_coefficient` is Hamiltonian UE1, while the output `lagrangian_epsilon=1/UE1`. There is no `ma` band factory inside either API, no pole-level hopping override, and no legacy aliases. The CLI explicitly constructs its former unit-spacing improved band and UE1=.09,K1=1/.09. Its historical spacing-rescaled Compton workflow now explicitly constructs the physical band and uses `UE1_phys=UE1_unit/a`, `K1_phys=K1_unit/a`; therefore `omega_phys=omega_unit/a`, `R_theta,phys=a R_theta,unit`, and the continuum-volume field leg equals the old unit-spacing leg. The previously executed numbers below belong to their cited revisions; this API cutover requires its own parent integration run.

## 3. Executed photon construction and speed adjustment

`photon_matching.py` completed naturally. For `ma=.2`, mode1 and equal bare stiffnesses:

| L | omega (dimensionless) | Pair threshold | Canonical e_* |
|---|---:|---:|---:|
| 16 | .390601958 | .634508819 | .299318200 |
| 32 | .196284355 | .479657026 | .299390990 |
| 48 | .130974797 | .438885681 | .299393505 |

Every pole lies inside its derived bracket; residue, monotonic spectral expansion and omitted-term controls are exercised. Source hash: `84c4b9d4f26d599ef046f8340676c2a82e190acfc634e3d0eb558c3c574755f3`.

The same construction gives an explicit low-frequency velocity-matching operation. At the selected momentum, choose

\[
\chi_{E0}=\chi_{B0}+\delta\chi_B(Q)-\eta_0.
\]

This sets the quadratic-in-frequency pole to `omega=Qhat` while retaining the sea response. The L48 calculation gives `chi_E0=11.1405463`, `U_E=.0897622047`; the exact spectral pole has `omega/Qhat=.999970136`, inside the controlled frequency correction. This is parameter matching at the chosen probe, not a prediction of why the microscopic coefficients take that value. A zero-momentum matching theorem is the natural strengthened version.

## 4. The pole returns to actual bosonic Pauli scattering

`photon_matched_compton.py` creates the corresponding finite monochromatic quantum photon mode. Its energy is the computed pole, not `|k|`. A polarization insertion carries the computed continuum-volume physical-field leg. Matter vertices and the two-photon contact are the same primitive-link derivatives with charge owned by the required band; both external legs are rescaled together. The **actual** `FilledSeaConsumer` acts with bosonic Pauli words and occupation-state denominators, including vacuum-disconnected subtraction at forward scattering. The adverse bare leg is derived from the actual Hamiltonian coefficient as `sqrt(a UE1)`, not hardcoded `.3`. Both-gauge Ward substitution detects contact deletion that a single longitudinal/transverse orthogonality could conceal.

The independent comparator is the explicit band resolvent with the same measured photon energy, current, contact and external spinors. It is not used to generate the Pauli intermediate states. For `a=.2`, `m=1`, `ma=.2`, forward axis-x modes:

| L | Physical omega | Primitive forward squared response | Delete virtual pairs |
|---|---:|---:|---:|
| 16 | 1.95300979 | .0322157831 | .00775735514 |
| 32 | .981421773 | .0321404019 | .00447697971 |
| 48 | .654873987 | .0321391908 | .00258889068 |

The maximum amplitude comparison error is `1.11e-16`; both Ward substitutions remain below `5.21e-17`. At L48 using the bare `.3` source instead gives `.0324004062`, rather than `.0321391908`: a measurable normalization change. The corresponding soft forward Thomson expression is `4 e_*^4=.0321387878`, approached by the exercised softer mode.

This is **forward monochromatic connected tree electron response with dressed external photon legs**. It is not a complete radiative cross section: electron self-energy, vertex corrections and nonlinear compact-photon fluctuations remain subsequent coupled improvements. Restricting this finite mode calculation to its actual catalogued momenta prevents a constant mode energy from masquerading as a general dispersion relation. Nonforward dressed modes and their phase-space Jacobian are the next consumer extension.

The [actual finite bulk-frame join](bulk_frame_compton.md) measures raw same-field light-sea poles and residues, reconstructs the field-derived carrier with declared finite errors, and adds an actual primitive current-excited neutral Gauss test. Neither that join nor this Gaussian theorem supplies a physical charged Gauss gap, compact photon phase or a full radiative amplitude.

Correctness does not require these diagnostic effects to exceed fixed lower
sizes. At the selected cutoff family a=.001, a legitimate contact-deletion
Ward defect is6.2187e−9, below the former1e−8 gate; at weak g²=.0001 the bare
versus dressed response differs by only3.02e−5 relatively, below the former
1e−4 gate. Those incidental effect-size checks have been removed, not relaxed.
The actual full Pauli response still must equal its independent resolvent,
satisfy the complete same-pole Ward identities, and use a valid positive
spectral pole. Bare-leg/contact-deleted responses remain measured diagnostics.
This does not make an omitted contact or wrong photon normalization valid.

The same defect class existed in the Gaussian pole checks: contact/pair
contributions need not exceed one percent of a bare restoring force. Those
minimum-size gates have also been removed. Pole-equation error is now divided
by its positive static restoring scale in both reduced and independent raw
consumers, rather than compared with an absolute energy-unit threshold.
The actual weak-coupling `L=8,a=.001,UE1=3e-8,K1=3e10` path has
`omega=22.9610059429 < pair_gap=72.9859101204`, positive inverse residue and
relative pole error `7.47e-17`. Its complete current has Ward error
`1.65e-24`, while the omitted-contact current has `1.05e-9`.
`tests/test_photon_matching.py` protects acceptance of that legitimate pole
and checks independent response equality and the relative Ward distinction.

The remaining virtual-pair minimum-size gate was also wrong. For the actual
contact-dominated `a=1,m=.1,r=.1,v=.001,L=8,UE1=1e-6,K1=1` band, its
correction is only `1.74e-4` of the forward squared response, but the complete
Pauli kernel and Ward contraction are valid. That size gate is removed.
The real source exposed a second numerical defect before reaching it: the
nonlinear correction at the certified upper pole bound was smaller than a
bare-kernel subtraction ulp, reversing the root bracket. The equivalent
kernel is now evaluated as

\[
D(s)=D(0)(1-s/s_{\rm upper})
 -\frac{2s^2}{\mathrm{norm}}\sum\frac{w}{\Delta^3(\Delta^2-s)}.
\]

Both raw and reduced consumers solve the dimensionless fraction
`s/s_upper` on `[0,1]`. The upper endpoint has a manifestly nonpositive
kernel; no sign coercion, bracket enlargement or input special case is used.
Positive residue checks allow `R>=epsilon>0`: a real positive sea correction
can round away next to a very large bare epsilon without invalidating its
pole. The `UE1=1e-12,K1=1e6` recovery reaches that rounded-equality case.
The permanent contact-dominated regression checks both regimes using actual
raw fields and relative independent-current/Ward comparisons.

## 5. Reproduce and strengthen

```sh
python proposals/P256-qed-backward/photon_matching.py --output proposals/P256-qed-backward/photon_matching_receipt.json
python proposals/P256-qed-backward/photon_matched_compton.py --output proposals/P256-qed-backward/photon_matched_compton_receipt.json
```

The integration manifest records its own bound revision; it does not prove that later supplier receipts were included. The current theorem chain contains energetic code selection, sea-selected spinor algebra, a spectral photon pole, and the same primitive current scattering response. Strengthening it means enlarging the dynamical domain and making the matching increasingly self-consistent, rather than replacing its positive content with a list of restrictions.
