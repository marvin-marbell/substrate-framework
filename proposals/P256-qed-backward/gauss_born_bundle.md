# Covariant paired sea and a negative unrestricted quantum-field trial

`gauss_born_bundle.py` constructs a genuine field-dependent neutral Gauss section from the actual primitive Pauli operators. It then supplies a variational theorem for all sixty canonical coordinates per site. This is not a solved ground wavefunction, a Clifford-quartet selection theorem, a charged lower gap, or a three-dimensional phase certificate.

## 1. A covariant state, not mean-colour projection

At one site retain exactly two heavy and two light particles. The actual primitive even generators give `6⊗6=1⊕15⊕20`, with Casimirs `0,4,6`. Let `|S>` be its unique normalized singlet. For any separated rank-two occupied projector `P` of a traceless Hermitian `4×4` matrix, define

\[
 N_P=c_L^\dagger P c_L,\quad
 \Pi_P=\tfrac12N_P(N_P-I),\quad
 |\Omega(P)\rangle=\sqrt6\,\Pi_P|S\rangle.
\]

On the light two-particle sector, `N_P` has eigenvalues `0,1,2`, so this polynomial is the exact projector onto the filled occupied plane. The singlet's light marginal is `I_6/6`; consequently `||Ω||=1`. The heavy particles fill the complementary plane. No Slater-minor sign convention is imported: all bilinears and the singlet are built with existing primitive Pauli words.

For `g∈SU(4)`,

\[
 \Omega(gPg^\dagger)=R_H(g)R_L(g)\Omega(P).
\]

Thus a gauge-invariant field envelope times this section satisfies the actual local Gauss equation. With `δ_aP=i[T_a,P]` and the active-field convention `E_a=-iδ_a`, it obeys `(E_a-Q_H^a-Q_L^a)Ψ=0`. The implementation independently compares the analytic field derivative with actual primitive charge actions and finite group rotations.

The matter state alone is **not** a Gauss singlet: its total matter Casimir is four. Freezing its field dependence leaves a Gauss defect of norm two. This is precisely the missing distinction between cancelling a mean and constructing a physical quantum state.

## 2. Berry cancellation and quantum kinetic cost

The paired state can be identified, by the singlet's fixed Hodge map, with the vectorized rank-one projector onto the occupied two-particle Slater line. It is globally phase-defined on the Grassmannian:

\[
 \langle\Omega|d\Omega\rangle=0,\qquad
 \langle d_a\Omega|d_b\Omega\rangle
 =\operatorname{Tr}(d_aP\,d_bP).
\]

The two individual determinant-line curvatures need not vanish; their opposite occupied/complement curvatures cancel. The quantum metric does **not** cancel. For canonical raw matrices `Tr(B_aB_b)=4δ_ab`, and ordered eigenvalues of `Φ_0`,

\[
 \sum_{a=1}^{15}\|\partial_{u_a}\Omega\|^2
 =8\sum_{o=1,2;\ e=3,4}(E_e-E_o)^{-2}.
\]

At `Φ_0=bβ`, this is `8/b²`. The code checks the full metric against primitive state derivatives and independent finite field perturbations. These are colour/field-bundle facts, not a mechanical rotation experiment or a Lorentz theorem.

## 3. An admissible all-sixty-coordinate trial

Use at every site the paired state of `Φ_0`; take independent isotropic Gaussian densities of variance `σ²` for its fifteen canonical coordinates and `ε²` for the forty-five spatial-field coordinates. All widths are positive, and every coordinate remains unrestricted. Every colour link has the constant Haar wavefunction; electromagnetic electric flux is zero. Local light occupation is exactly two, so EM Gauss holds. The field envelope is gauge invariant; its derivative and the matter derivative supply colour Gauss. Hopping expectations vanish because every species' local occupation is fixed, not because hopping is omitted from the Hamiltonian.

For the whole-heavy law `H_H=-Λh_L`, with common positive onsite scalar `f0`, the matter energy at each field configuration is

\[
 (\Lambda+1) f_0(E_1+E_2).
\]

For ordered traceless eigenvalues write `E1=-s/2-x`, `E2=-s/2+x`, `E3=s/2-y`, `E4=s/2+y`, with `x,y≥0` and `x+y≤s`. Then `Tr Φ0²≤3s²`, proving

\[
 -(E_1+E_2)\ge2\rho_0/\sqrt3,\quad
 \mathbb E\rho_0=\sigma\sqrt2\,\Gamma(8)/\Gamma(15/2).
\]

This conservative bound is analytic, not a Monte Carlo energy estimate.

The projector's singularities at a middle-level crossing are integrable in the quadratic-form domain. The complex-Hermitian eigenvalue measure contains the squared Vandermonde factor. More explicitly, add the independent scalar trace mode: gaps and projectors are unchanged, and `Φ0=2σX` for standard `GUE4` density `exp(-Tr X²/2)`. If `S_N=E Σ_(i<j)(xi-xj)^(-2)`, Gaussian score integration by parts gives

\[
 N+4S_N=N^2-2N(N-1)+8S_N,
 \quad S_N=N(N-1)/4.
\]

For `N=4`, the occupied–empty pair sum is a subset of the six-pair sum, so the mean projector metric is at most `6/σ²`. The paired Berry connection is zero, eliminating cross terms with the real envelope. Hence the full field kinetic expectation per site is bounded by

\[
 \sqrt\Lambda\,[39/(8\sigma^2)+45/(8\epsilon^2)].
\]

Level crossings do not make this packet a smooth pointwise eigenvector everywhere; finite form energy, not pointwise smoothness, is the needed variational domain. Smooth approximations of the covariant packet give the same form limit.

## 4. Negative energy and nonvanishing invariant amplitude

Use the declared microscopic terms

- `sqrt(Λ)/2 ΣΠ²`;
- bare quartic `Λκ/2 ||G||F²`, `Gμν=Σ_a uμa uνa`, whose unique bare minimum is zero;
- positive alignment `ΛJ/8 Σ_bonds ||Φx-UΦyU†||HS²`;
- optional positive colour plaquette `ΛK4(1-ReTr Uloop/4)`;
- nonnegative electric energy and compact EM magnetic energy bounded below by `-KEM Nplaquette`.

Exact Gaussian moments give

\[
 Q=\tfrac\kappa2[255\sigma^4+90\sigma^2\epsilon^2+855\epsilon^4].
\]

For cubic bond and plaquette densities at most three, define

\[
 a=\tfrac2{\sqrt3}\sqrt2\,\Gamma(8)/\Gamma(15/2),\quad
 C=Q+45J(\sigma^2+3\epsilon^2)+6K_4-f_0a\sigma,
 \quad T=39/(8\sigma^2)+45/(8\epsilon^2).
\]

The photon magnetic mean in the zero-electric trial is zero. The actual full Hamiltonian therefore satisfies

\[
 E_0/V\le\Lambda C+\sqrt\Lambda T-f_0a\sigma.
\]

When `C<0`, sufficiently large `Λ` produces a negative energy density proportional to `Λ` in an **actual Gauss-compatible trial**, without replacing the quantum fields by a preselected orbit. The full ground obeys the corresponding variational upper bound. If the existing endpoint-path norm estimate is `|h|≤c_H Σρ_x`, the lower Hamiltonian estimate yields

\[
 \frac1V\sum_x\langle\rho_x\rangle
 \ge\max\left(0,
 \frac{-E_{\rm trial}/(\Lambda V)-3K_{\rm EM}/\Lambda}
 {c_H(1+1/\Lambda)}\right),\qquad
 \liminf_{\Lambda\to\infty}\frac1V\sum_x\langle\rho_x\rangle\ge-C/c_H>0.
\]

This excludes collapse to the bare zero-amplitude root in that parameter domain. It does not establish which four-field root wins, spatial order, a local amplitude bound, spontaneous gauge-symmetry breaking, or a charged mass gap. The theorem uses the existing full-CAR `S=+` qubit-code embedding on the graph; the executable checks the local primitive algebra, not a full many-site spectrum.

## Execution

```sh
python proposals/P256-qed-backward/gauss_born_bundle.py --output gauss_born_bundle_receipt.json
```

The CLI checks selected and genuinely deformed field matrices, all fifteen differential Gauss equations, finite primitive covariance, nonzero frozen/missing-heavy defects, individual nonzero and paired zero Berry curvature, the complete canonical metric and finite-difference refinement. Its trial constants are analytic bounds. Numerical full-field ground evolution, a nonlinear phase, and joined scattering are outside this calculation.

### Parent-owned execution and parameter boundary

[gauss_born_bundle_receipt.json](gauss_born_bundle_receipt.json) completed
naturally in 1.23 s, implementation SHA256
`ed0b037ba52293f60e311d09831903511db0daf3447038cd61bdef4ed2072515`.
The executable uses `a=1,m=.4,r=.2,v=.3,improved=False` and the existing
`uniform_data(...,size=3,stiffness=2,preferred=0)` supplier for the selected
control. This is **not** the `m=.2,r=.12,v=.3` bare band of the aligned photon
and dressed-carrier receipts; their numerical results are not glued to this
full-field trial.

Across selected/deformed controls, differential Gauss residuals are at most
`3.87e-16`, finite primitive covariance residuals at most `3.93e-16`, and
paired curvature residuals at most `7.73e-16`, while the frozen-state Gauss
defect is `2` and the missing-heavy defect is `1.581138830084`.
The canonical metric agrees with its inverse-gap formula, and halving the
derivative step reduces error by approximately four.

With `sigma=.2,epsilon=.1,kappa=1,J=.05,K4=.01,KEM=.02`, the analytic
leading coefficient is `C=-.397404412577834`, kinetic coefficient
`T=684.375`, and sufficient leading-balance scale `2965670.5400281516`.
At `Lambda=1e7`, the energy **upper** per site is `-1809861.2317550182`
and invariant mean-amplitude **lower** is `.010867866936389676`; at `1e6`
the trial upper remains positive. These are variational theorem constants,
not measured full-field ground energies or a charged lower gap.
The [full-band section](full_band_gauss_section.md) extends the covariant-state
question to occupied hopping bands on its declared positive-overlap chart,
not to a proved common full-field ground. Original supplier and receipt-time
proof hashes remain unchanged by this documentation update.
