# Forty-five quantum restoring shapes with a compact photon

`quantum_restoring_shapes.py` defines a **Gaussian shape EFT**, not the full
nonlinear sixty-field phase. It joins all 45 uniform restoring coordinates to
the actual `SpinorPlaquette` Pauli/code/integral-rotor Gauss domains for
`N=7,8,9`. Its reproducible quantum consumer is an explicitly labelled **coupled
variational trial**. An exact small Schur solve must not be mistaken for a
coupled ground-state solve. Issue 222 remains open; this is not full QED, a
three-dimensional quantum Higgs phase, or a continuum scattering construction.

## The potential is supplied by the recorded sea, not postulated radial rigidity

The supplier call is exactly

```python
uniform_data(RotationBand(a=1., m=.2, improved=True),
             3, 1., 0., matrix_unit_basis())
```

The underlying energy per cell is the occupied half-filled sea plus
`1/2 ||G(Phi)||_F^2`, with `G_mn=Tr(Phi_m Phi_n)/4`. The preferred Gram is zero:
the elastic term by itself prefers zero. The **sea** supplies the nonzero
stationary Clifford frame. This is the finite `3^3`, one-sea construction of
`local_orbital_stability`, distinct from any heavy/light, multi-sea selected
phase in `heavy_orbital_join`.

Let `B_a` be the actual matrix-unit basis, `Tr(B_a B_b)=4 delta_ab`, and
`Phi_mu=Phi_mu^0+sum_a c_mu,a B_a`. The program constructs

\[
 K_0=K_{\rm sea}(q=0)+K_{\rm elastic},\qquad
 O_{(\mu,a),t}=\tfrac14\operatorname{Tr}
 B_a i[B_t,\Phi^0_\mu].
\]

It uses the actual rank-15 tangent range `O`, takes its orthogonal complement,
and diagonalizes the **actual** `45 x 45` restriction. The output contains the
selected Gram, selected fields, every measured eigenvalue `k_a`, every normal
column `V`, all `delta Phi_mu,a`, and projection/stationarity residuals. It
rejects a nonpositive eigenvalue or an incorrectly sized orbit; no numerical
curvature or threshold is invented in this note. Coordinates are local
per-cell uniform field deviations; the Hessian supplier is a sea **density**,
not the total energy of 27 cells. The stiffness multiplying it below is an
explicit EFT parameter.

The orientation projection matches `orbital_gauge_locking.run`, but removing
physical orientation modes requires its **gauged-orientation hypothesis** or
an explicit local-slice restriction. This CLI quantizes compact electromagnetic
U1 links, not SU4 links or SU4 Gauss generators. It does not establish that the
15 orientation modes disappear in the global-SU4 nonlinear theory. Nor does
uniform positivity assert spatial stability or nonlinear quantum order.

## Actual primitive shape vertices

The immutable reference has `m=.8, r=.04, v=.08, UE=.6, K=.25, delta=1.2` and
`S=+`. If `g_0,g_s` are the measured selected Gram entries, the **bare**
coefficients are `m/sqrt(g_0), r/sqrt(g_0), v/sqrt(g_s)`. Inserting `Phi^0`
therefore recovers the original reference Hamiltonian. `RotationBand` supplies
the directed scalar paths. The sea uses its improved three-dimensional band;
the planar quantum reference uses its original nearest-neighbour band, with
the absent z Wilson onsite contribution removed. The program checks the
selected-field matter matrix against the immutable reference matrix.

For each of the 45 normals it differentiates these same scalar paths to obtain
`G_a=dH_matter/dxi_a`. All onsite diagonal densities, all arbitrary onsite
flavour transfers, and all 16 flavour transfers on each oriented physical link
are included. Every transfer is made by the actual root path,
`code.words.path_A(path).times_i(-1)` and `code.element` on raw two-word code
states, with the actual oriented integer electric shift. The builder checks
`[word,S]=0`, every original Gauss state, every accepted transfer's Gauss law,
and every sparse `G_a`'s Hermiticity. It never changes the supplier's `H`,
`onsite` or `terms`. Determinants are used only to embed onsite **trial vectors**
with the supplier's actual code phases, never to construct Hamiltonian entries.

For the onsite block `D_a` and four directed link blocks `T_la`, use

\[
 C_a=4\sum_{ij}|(D_a)_{ij}|+
       2\sum_{l=1}^{4}\sum_{ij}|(T_{la})_{ij}|.
\]

Each density has norm at most one; each projected primitive transfer times an
untruncated rotor shift has norm at most one. Thus `||G_a||<=C_a` on the
**untruncated** Gauss domain. The printed constants are coefficient bounds,
not sampled finite matrix norms. A planar reduction may leave some restoring
normal combinations matter-dark; those oscillators are still retained, not
replaced by a selected-only radial coordinate.

## Exact all-coordinate infinite-gap bound

For the entire coordinate space `xi in R^45`, define

\[
 H_\Lambda=H_{\rm ref}+
 \sum_{a=1}^{45}\left[
  \frac{\Pi_a^2}{2M_\Lambda}+
  \frac{\Lambda k_a\xi_a^2}{2}-\frac{\omega_a}{2}
 \right]+\sum_a\xi_aG_a,
 \quad M_\Lambda=\Lambda^{-1/2},\quad
 \omega_a=\Lambda^{3/4}\sqrt{k_a}.
\]

With `q_a=(2 M_Lambda omega_a)^(-1/2)` and oscillator lowering operator `a_a`,
put `b_a=a_a+q_a G_a/omega_a`. On the common form core,

\[
 \omega_a a_a^\dagger a_a+q_a(a_a+a_a^\dagger)G_a
 =\omega_a b_a^\dagger b_a-\frac{G_a^2}{2\Lambda k_a}.
\]

`G_a` acts on matter/rotors and commutes with `xi,Pi`, but **the different
G_a need not commute with one another or with H_ref**. Each `b_a^dag b_a` is
positive separately. Summing gives the exact quadratic-form lower bound

\[
 H_\Lambda\ge H_{\rm ref}-\delta_\Lambda,
 \qquad\delta_\Lambda=\frac{1}{\Lambda}
 \sum_a\frac{C_a^2}{2k_a}.
\]

There is no assertion of a simultaneous displaced vacuum. In particular, a
naive square completion that drops the kinetic term and forgets the normal
ordering subtraction would give the wrong bound. Bounded `G_a` and positive
`k_a` give the standard semibounded oscillator form extension.

A neutral reference state times the **common undisplaced vacuum** has zero
linear-coordinate expectation; its total trial energy is its reference
energy. Apply the reference `compact_charge_gap` argument: integer Gauss
offsets, nonwrapping untruncated compact rotor, retained boundary electric
energy, `r,v>=0`, onsite `M=m+2r>0`, spin-scalar U1 Wilson paths, and the neutral
onsite/photon product trial with zero hopping expectation. Its lower bound is

\[
 \Delta_{\rm ref,bound}=M+U_E/2-8\max(r,v)=0.54.
\]

The common pure-photon infimum cancels between the charged lower bound and the
neutral upper trial. Therefore

\[
 \boxed{\Delta_{\Lambda,\rm charge}\ge
 0.54-\frac{A}{\Lambda}},\qquad
 A=\sum_a\frac{C_a^2}{2k_a},\qquad
 \Lambda>\Lambda_*:=A/0.54.
\]

The threshold and all its inputs are calculated and published in JSON. Default
`--scales .5 2 8` means `Lambda/Lambda_*`, so the first case deliberately has no
positive certificate, while the other two have theorem margins `.27,.4725`.
These margins are algebraic consequences of the chosen ratios, not measured
infinite spectra. **General SU4 links do not inherit this scalar-U1 reference
bound.** No conclusion is exported to an arbitrary quantum SU4 vacuum.

## A real, adverse-capable coupled quantum trial

The represented oscillator space contains complete total-excitation shells
`K=0,1`: vacuum plus **every one** of the 45 single-excitation states. It is not
a tensor product of selected per-mode cutoffs. `QuantumShapes` applies the
sparse physical matrix and every sparse `G_a` without assembling a Kronecker
Hamiltonian. `--oscillator-cutoffs 1` is explicit; this implementation does not
silently pretend to include `K=2`.

Let `psi` be the actual physical onsite/photon trial (with charged boundary
balance in cell zero). In this complete shell space, form the orthonormal
trial subspace

\[
 |0\rangle\psi,\qquad
 |1_a\rangle\frac{G_a\psi}{\|G_a\psi\|}
\]

for nonzero images. Its exact projected Hamiltonian is an arrowhead. Writing
`E0=<psi,H_ref psi>`, `d_a=<phi_a,H_ref phi_a>+omega_a-E0` and
`b_a=q_a||G_a psi||`, solve stably for its energy decrease

\[
 s=\sum_a\frac{b_a^2}{d_a+s},\qquad E_{\rm trial}=E0-s.
\]

The scalar Schur equation avoids asking ARPACK to resolve a huge-frequency
oscillator spectrum. JSON reports the actual small Ritz residual, the **full
finite Hamiltonian residual**, energy expectation, shape populations,
coordinate moments and `xi_a G_a` correlations. It optimizes the deliberately
limited trial subspace, **not the entire finite shell Hamiltonian**. Its energy
is an upper bound on both the finite-shell ground and the untruncated ground
because the physical, nonwrapping cutoff state embeds in the untruncated
rotor space. An explicit conservative untruncated lower bound uses photon
energy `>=-K`; a stronger finite-domain lower bracket uses the supplier's
finite Holder rotor energy. Their gap to the trial is not advertised as a
residual-based ground certificate.

The adverse case deletes the strongest actual shape vertex, re-solves the
trial subspace, and prints both its loss of energy lowering and the action
change on the actual coupled state. Participation can be very small at these
large theorem-positive stiffnesses; a tiny but nonzero correlation is not a
claim of a strong-coupling quantum phase.

Coordinate moments are continuum wavefunction moments, not powers of
compressed coordinates. For the represented state `x_0, x_a`,

\[
 \langle\xi_a\rangle=2q_a\Re\langle x_0,x_a\rangle,
 \quad\langle\xi_a^2\rangle=q_a^2(1+2\|x_a\|^2),
 \quad\langle\xi_a\xi_b\rangle=2q_aq_b\Re\langle x_a,x_b\rangle
 \quad(a\ne b).
\]

The second moment includes excursions into `K=2` even though the state is in
`K<=1`. The omitted excursions of `(P xi_a P)^2` are printed as a positive
adverse moment defect. Normal coordinate covariance and electric-cutoff shell
probability are also reported.

## Coordinate-covariant time reversal and same-H Peierls vertices

Time reversal uses the actual supplier `local_T`, with `Phi_0` even and
`Phi_i` odd. In the normalized 60-coordinate basis its real map is
`F=diag(+,-,-,-) tensor Ad(local_T K)`. The normal-coordinate map is
`S=V^T F V`. The program verifies closure, `S^2=I`, covariance of the actual
Hessian/frequencies, invariance of the selected frame, and the actual
Pauli/Gauss law

\[
 \Theta\,G(z)\,\Theta^{-1}=G(Sz),\qquad
 \Theta\xi\Theta^{-1}=S\xi.
\]

Degenerate eigenbases may mix under `S`; **no fixed arbitrary G_a is forced
individually T-even**. This preserves basis covariance of all 45 coordinates.

Every reference link and every xi-dependent shape link receives its own
primitive Peierls phase `exp(i theta(eta_target-eta_source))`. Current and
contact come from its first and second derivatives **before** comparison with
any density commutator. The consumer separately checks both derivatives and
finite pure-gradient unitary covariance, and quantifies the defect when
xi-dependent current/contact are omitted. Oscillator/electric/magnetic/onsite
terms commute with the cell density; their omission from the numerical
commutator comparison avoids artificial huge-frequency cancellation error.

For this same complete finite Hamiltonian the exact operator identities are

\[
 J=i[\rho_\eta,H],\qquad C=-[\rho_\eta,[\rho_\eta,H]].
\]

For an **exact eigenstate**, when the appropriate reduced resolvent exists,
`<C>-2<J Q(H-E)^(-1) Q J>=0`. That is an operator-level consequence of pure
unitary gauge conjugation. This CLI's state is a trial with a reported full
residual; it does **not** calculate or pretend to observe the ground-state
resolvent cancellation, a scattering amplitude, or a continuum Ward limit.

## Reproduction and evidence boundary

From a Python environment containing the project's NumPy/SciPy dependencies:

```sh
python proposals/P256-qed-backward/quantum_restoring_shapes.py \
  --cutoffs 1 --oscillator-cutoffs 1 --scales .5 2 8 \
  --output /tmp/quantum-restoring-shapes.json
```

Larger physical domains are selected with `--cutoffs 1 2`; they are considerably
more expensive because each sector retains all occupations and all accepted
Gauss electric states. Defaults use one electric cutoff and no unbounded
iterative ground solver. JSON includes SHA-256 hashes of the consumer and all
named scientific suppliers. Exit success means its explicit construction and
adverse/identity checks completed, not that a quantum ground state converged.
The parent executed the complete default CLI; all construction, adverse and
identity checks completed. The receipt gives `k_min=8.04156336801237e-5`,
`A=593873.588844812` and `Lambda_*=1099765.90526817`. At twice the threshold
the neutral trial has full-H residual `.15920505647`, despite projected
residual `1.68e-16`. Its first eleven soft-mode variances sum to `13.703`,
versus selected-frame coefficient norm squared `2.986990`; the Gaussian state
has not demonstrated a narrow approximation to the nonlinear quartic root.
The [complete-shell consumer](quantum_shape_resolvent.md) uses the full
physical Schur operator and its own full-H/response acceptance, rather than
relabelling this trial as a ground state.

## Stable continuum moments at very small shell probability

For a normalized represented state
`psi=|0>x0+sum_a|1_a>x_a`, write `p_a=||x_a||^2` and
`P_one=sum_a p_a`. The image outside K1 under `xi_a` consists of
`sqrt(2)*q_a*x_a` at `|2_a>` and `q_a*x_b` at `|1_a,1_b>` for `b!=a`.
Its exact squared norm is therefore `q_a^2*(P_one+p_a)`.
`coordinate_observables` evaluates this positive sum directly rather than
subtracting nearly equal `P xi_a^2 P` and `(P xi_a P)^2` expectations.
A vacuum state legitimately has zero defect and zero one-shell participation.

The completed [strong-narrow full-Schur run](quantum_shape_resolvent.md)
exercises probabilities around `4.3e-21`, with actual outside-K1 norms around
`5.25e-23` and coordinate variance statistic `0.0017182570727851325`.
This repairs a floating-point diagnostic failure without changing the
kinetic mass power, shell Hamiltonian or full-H spectral acceptance.
