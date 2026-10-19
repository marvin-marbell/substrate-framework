# Separated insertion with a fixed remote reference: full physical accounting

## Same operator and domains

This extends `pair_full_field_charge_transfer` on the **same** Hamiltonian

\[
H=\tfrac{\sqrt\Lambda}{2}p_\Phi^2+u_4 C_U
 +\tfrac{u_1}{2}(p_0^2+p_1^2)-K\cos(A_0+A_1)
 +\Lambda(V_\kappa+JW/8-d\Gamma(h_H))+d\Gamma(h_L).
\]

No field mode, matter configuration, Haar representation or torus Fourier mode
is removed. The neutral fibre is NH4/NL4 (4900), and charged matter fibres are
NH4/NL3 or5 (3920), tensored with a genuine four-component external reference
at **R**. The Hamiltonian coefficients and coordinate measures are unchanged.
The reference changes right colour/EM Gauss constraints, not the Hamiltonian.
This closed graph has zero total EM source including the compensating reference.
It is not an isolated/infinite-volume charge construction.

The exact normalized neutral ground projector density is denoted by rho.
It exists by the supplier's coercive compact-resolvent result. Reflection gives
`<n_L>=2`, `<dGamma(P_L h_L)>=<H_L>/2`, and
`<dGamma(h_L P_L)>=<H_L>/2`. No finite packet is substituted for rho.

## Actual remote dressings and all Gauss generators

Write c_Li for the actual primitive marker-word ladders; they act on the complete
native Pauli fibres. U transforms as g_L U g_R†. For each independent path define

\[
f_{-,0}=e^{-iA_0},\quad f_{-,1}=e^{iA_1},\qquad f_{+,k}=f_{-,k}^*.
\]

The remote maps, with insertion at L and reference at R, are

\[
(A_{-,k}\psi)_j=f_{-,k}\sum_i(U^\dagger)_{ji}c_{Li}\psi,
\qquad
(A_{+,k}\psi)_j=f_{+,k}\sum_i U_{ij}c_{Li}^\dagger\psi.
\]

Boundary generators are T_a for a hole and -T_a^T for a particle, with
Tr(T_a T_b)=delta_ab/2. Norm operators are n_L and 4-n_L, respectively;
each exact-ground image has norm squared2, not1. Both null local occupations
are retained: empty for the hole and full for the particle.

Let D_UL=-i derivative along U(s)=exp(isT_a)U, and D_UR=-i derivative
along U(s)=U exp(-isT_a). Let J_Phi,x^a be **the full Cartesian field angular
momentum**, -i derivative of every Phi_mu,x along exp(isT_a)Phi exp(-isT_a).
The physical neutral colour identities are

\[
(D_{Ux}^a+J_{\Phi,x}^a)\psi=Q_x^a\psi.
\]

For the base dressing J_Phi A=0, and direct differentiation gives, for all30
generators,

\[
D_{UL}^a A=Q_{L,q}^a A-AQ_{L,0}^a,\qquad
D_{UR}^a A=(Q_{R,q}^a+Q_B^a)A-AQ_{R,0}^a.
\]

Therefore the **complete** colour Gauss operators intertwine; the Cartesian
angular momentum still acts on the neutral input. It does not disappear from
an energy calculation. Centre congruences are included by this group covariance.

For EM put p_j=-i partial_Aj. The neutral constraints are
`(p0-p1)psi=(n_L-2)psi` and the opposite right constraint. Both path dressings
have gradient charge q: `(p0-p1)A=q A`. Charged right source is `n_R-2-q`.
This proves both site constraints with the fixed compensating source at R.
These are the source orientations of `quantum_su4_photon`; there is one colour
link, not two independent SU4 kinetic operators.

Bounded smooth multiplication maps preserve the full form domain. Derivatives
of the unitary link and phase dressings are bounded. Quartic moments control
matter multiplication. Thus operator existence is not conditional on a packet
chart or a solved quantum ground.

## Full kinetic commutator and exact-ground trial cost

The Cartesian kinetic commutator is **zero** for the base maps. Every scalar
potential, alignment, magnetic multiplication and entire heavy bilinear
commutes with them. They are not claimed to commute with link kinetics.
For C_U=sum_a(D_UL^a)^2 and C_F=15/8,

\[
[u_4 C_U,A]\psi=u_4 C_F A\psi
 +2u_4\sum_a(D_{UL}^a A)D_{UL}^a\psi.
\]

Native CAR and unitarity give the particularly useful identity, for **both** signs,

\[
A^\dagger D_{UL}^a A=-Q_{\mathrm{light},L}^a.
\]

For path0 the phase commutator is `u1 A/2 + u1 q A p0`;
for path1 it is `u1 A/2 - u1 q A p1`. The light commutator is the transported
usual left-ladder commutator: `[H_L,c_Li]=-sum_j(h_L)ij c_j`, or its creation
counterpart. Consequently its normalized exact-ground contribution is
`-<H_L>/4` for both signs; the colour dressing does not alter this CAR result.

Use an **incoherent equal mixture** of the two individually physical path trial
densities, `rho_q=(A_q0 rho A_q0†+A_q1 rho A_q1†)/4`. There are no uncomputed
coherent cross terms. Using H rho=E_N rho and the actual EM Gauss identity gives

\[
E_{\mathrm{trial},q}-E_N
=-\langle H_L\rangle/4+u_4 C_F
-u_4\langle Q_{\mathrm{light},L}\cdot D_{UL}\rangle
+u_1/2-u_1\operatorname{Var}_\rho(n_L)/4.
\]

Here the EM equality uses `<n_L>=2`: for a hole
`q<W(n_L-2)>=-Var(n_L)`, and for a particle it is the same. Thus mixing the
parallel paths really removes the independent EM loop momentum, rather than
bounding it by an assumed loop ground. The remaining link cross term is real:
Q_light acts on matter, commutes with D_UL, and the invariant sum is Hermitian.

**The surviving full-Gauss obstruction is precisely**

\[
-u_4\langle Q_{\mathrm{light},L}\cdot Q_{\mathrm{total},L}\rangle
+u_4\langle Q_{\mathrm{light},L}\cdot J_{\Phi,L}\rangle.
\]

Replacing D_UL by Q_total alone deletes a genuine unbounded Cartesian angular
momentum. The receipt includes a native smooth section with all30 full Gauss
identities satisfied and a nonzero defect when the Cartesian part is omitted.
That section is only a differential witness, not rho. Field-coordinate moment
concentration does not on its own bound field angular momentum or its correlation
with Q_light. No bounded-matter estimate is advertised as a uniform charge bound.

## A valid finite-Lambda bound, not a uniform limiting one

The full local light Fock representation has Casimirs0,15/8,5/2,15/8,0 for
occupations0..4, so `sum Q_light,a² <=5/2`. Let T_N be the expectation of the
**sum of all positive kinetics** in rho. Retaining these terms in the supplier's
global lower multiplication inequality and comparing to its full-domain
variational upper gives

\[
T_N\le B_T(\Lambda)
=\Lambda\{E_{\mathrm{trial},N}^{\mathrm{upper}}/\Lambda
 -C_{\min}+C_L/\Lambda+K/\Lambda+C_Y\Lambda^{-4/3}\}.
\]

Numerical evaluation uses the exactly equivalent positive expression
`B_T=(3/4) Lambda expected_global_remainder_upper`, not a subtraction of
rounded total energies. At large Lambda that subtraction can erase the
entire growing trial excess; the source moment expression evaluates its
nonnegative summands before adding any global minimum.

All constants are those in `pair_full_field_concentration`; its
`ground_energy_lower_excess_divided_by_Lambda` is the negative last three
terms. Since `u4 <C_U> <= T_N`, Cauchy gives

\[
|u_4\langle Q_{\mathrm{light},L}\cdot D_{UL}\rangle|
\le\sqrt{(5/2)u_4 B_T(\Lambda)}.
\]

Together with its moment-controlled light cost C_light(Lambda), the rigorous
finite-graph variational statement is

\[
E_{q,\mathrm{ground}}-E_N
\le C_{\mathrm{light}}(\Lambda)+u_4(15/8)+u_1/2
 +\sqrt{(5/2)u_4 B_T(\Lambda)}.
\]

This is finite for every finite Lambda>=1 on the full domains. B_T=O(Lambda^5/6)
and this conservative upper cost grows O(Lambda^5/12). It does **not** provide a
uniform large-Lambda separated-reference cost, unlike the coincident R map.
Floating evaluations of analytic inequalities are not interval certificates.

## Materially different correction: an all-field reference twist

Test `B=U V_R`, `V_R=exp(i beta Phi_0,R)`, beta=.7. Replace U by B in each
colour coefficient. This is not a constant-field packet: it is a globally smooth,
unitary, all-field dependent dressing on the original R120 domain. It transforms
as g_L B g_R† and therefore also intertwines the complete Gauss constraints.
The derivative of V_R in the right field angular direction is essential; omitting
it produces a measured nonzero Gauss defect.

The full product rule is now

\[
[K,A_B]\psi=-\tfrac{\sqrt\Lambda}{2}
 \sum_{b=1}^{120}\{(\partial_b^2 A_B)\psi+2(\partial_b A_B)\partial_b\psi\}
 +u_4 C_F A_B\psi+2u_4\sum_a(D_{UL}^a A_B)D_{UL}^a\psi
 +\text{the same phase terms}.
\]

Only the15 Cartesian coordinates of Phi_0,R have nonzero new derivatives.
They are exact exponential Frechet derivatives, not guessed commutators in a
noncommuting field direction. Crucially, V_R is unitary and independent of U, so
`A_B† D_UL A_B=-Q_light,L` **exactly**, for either sign and any beta. Thus this
correction does not remove the surviving colour cross; it adds genuine
sqrt(Lambda)-weighted Cartesian derivative and cross terms. The producer tests
this invariance and the nonzero additional field derivatives at noncentral link,
nonzero phase and all120-field inputs. No unsupported assertion that the new
field cross cancels the old one in rho is made. This rejects a local reference
unitary twist as an algebraic cancellation mechanism, not all possible dressings.

The next design-changing requirement is control/cancellation of the **joint**
Q_light--Cartesian angular correlation (or a genuinely different covariant
transport involving the left field and its full kinetic product rule), rather
than an EM-loop fix or another unitary decoration at R. No global no-go follows.

## Reproduction and scope

Use the scientific Python environment, with BLAS threads1:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/administrator/substrate-framework/.venv/bin/python \
proposals/P256-qed-backward/pair_separated_charge_transfer.py \
--output proposals/P256-qed-backward/pair_separated_charge_transfer_receipt.json
```

The producer checks both signs and paths on complete native fibres at two
noncentral/allfield/nonzero-phase points. It verifies norm, all colour and EM
intertwining, entire heavy and light commutators, full link/phase kinetic product
rules on independent local jets, and discriminating adverse controls. Exact-ground
formulae are proved above, not inferred from those finite differential probes.
Parent owns final acceptance execution and integrated diagnostics.

Operator existence and a finite graph variational cost are established. A
uniform limiting separated cost remains unresolved. Neither is a spectral
charged-particle theorem, positive lower gap, physical spin, mobile band,
isolated/Coulomb excitation, bulk phase or campaign completion.
