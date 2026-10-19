# Explicit full-field neutral trial and boundary-average charge theorem

This removes the supplied `neutral_trial_coefficient` hypothesis in the
unrestricted-law section of `gauss_charge_intertwiner.md`. It does **not**
reinterpret that consumer's fixed-orbit `J=0` EFT as an unrestricted quantum
field calculation. The theorem here has all sixty independent canonical real
coordinates per vertex, genuine continuous Haar colour links, and untruncated
compact EM rotors. The executable evaluates analytic constants and actual
primitive Pauli-code algebra; it is not a sixty-field ground-spectrum solver.
The parent owns scientific CLI execution and source-bound numerical receipts.

The primitive inputs are `gauss_charge_intertwiner.py` SHA256
`861008cb7dea64d2b2e0cef513c62de1d55539a1527a5b6fea8b5b879ea956fa`
and `quantum_su4_gauss.py` SHA256
`9e91c25bc4bcbc5f2306b38d4716b4e3af4664fc22757aedda527c056aea096b`.
Neither source is edited. This is progress on the backward invention campaign,
not campaign completion, neutral-sector global dominance, a lower gap, a phase
or Lorentz theorem, a pure mechanical half-spin result, or canonical QED.

## 1. One finite-graph microscopic law

Let a finite graph have `V` vertices, distinct endpoints for each matter path,
`Nb` alignment bonds and `Np` simple primitive plaquettes. Parallel matter paths
are counted separately; a path and its Hermitian adjoint are counted **once**.
Each vertex has four unrestricted traceless Hermitian fields

\[
 \Phi_{x\mu}=\sum_{a=1}^{15}u_{x\mu a}B_a,\quad
 \operatorname{Tr}B_aB_b=4\delta_{ab},\quad
 \rho_x^2=\sum_{\mu a}u_{x\mu a}^2,\quad
 (G_x)_{\mu\nu}=\sum_a u_{x\mu a}u_{x\nu a}.
\]

The field Hilbert space is `L²(R^(60V),du)`. The field law is the **local sum**

\[
 T_\Phi={\sqrt\Lambda\over2}\sum_{x\mu a}\Pi_{x\mu a}^2,\qquad
 V_\Phi={\Lambda\kappa\over2}\sum_x\|G_x\|_F^2,
 \qquad \Lambda\ge1,\quad\kappa>0.
\]

There is no prescribed root, radius, Clifford orbit or oscillator shell.
The positive colour-covariant field alignment is

\[
 H_J={\Lambda J\over8}\sum_{(xy)\in\text{bonds}}\sum_\mu
 \|\Phi_{x\mu}-U_{xy}\Phi_{y\mu}U_{xy}^\dagger\|_{HS}^2,
 \qquad J\ge0.
\]

Each primitive colour link is a multiplication coordinate on `L²(SU(4),dU)`
with normalized genuine Haar measure. Gauge electric energies are nonnegative.
An optional positive colour plaquette term is
`Lambda*K4*(1-ReTr(Uloop)/4)`, `K4>=0`. Its value lies in `[0,2Lambda K4]`.
Compact EM electric energy is nonnegative and the magnetic term on each simple
plaquette is `-KEM*cos(theta_loop)`, or another inversion-symmetric term bounded
below by `-KEM` with zero constant-electric-state expectation. `KEM>=0` does
not scale with `Lambda`. A different magnetic trial mean must be included
explicitly in `C0`; bounded-below alone does not specify an upper trial energy.

There are two fundamental matter species, four modes per vertex, each globally
half-filled: `NH=NL=2V`. Heavy matter is EM-neutral, light local EM charge is
`N_L(x)-2`, and

\[
 H_H=-\Lambda h_H,\qquad H_L=+h_L.
\]

For either flavour, an onsite term is `c_x† f0 Phi_x0 c_x`, with real `f0`.
For an oriented primitive path `e:x->y`, let `U_e` be the ordered product of
its genuine colour links, and `z_e` its EM phase for light matter (`1` for
heavy). The endpoint-averaged path block is

\[
 M_e={z_e\over2}\sum_{\mu=0}^3 f_{e\mu}
       (\Phi_{x\mu}U_e+U_e\Phi_{y\mu}),\qquad
 h_e=c_x^\dagger M_e c_y+c_y^\dagger M_e^\dagger c_x.
\]

`scalar_paths(RotationBand)` is the sole owner of `f0,f_e`: Wilson/scalar
coefficients are real and kinetic/spatial coefficients imaginary. Improved
length-two paths are products of two primitive links, not independent shortcuts.
On the CLI's open box the existing three-axis onsite coefficient is retained,
including on boundary vertices; no degree-adjusted onsite law is inferred from
`PhysicalHamiltonian`'s different one-axis fixed-orbit consumer.

The physical space satisfies all fifteen local SU(4) Gauss generators, including
field adjoint rotations and left/right link derivatives, and compact EM Gauss.
The bosonic `PauliCode` remains the actual `S=+` code. Its neutral parity marker
has no colour or EM charge; no new matter spectator is added. Local primitive
algebra is evaluated with its actual directed boundary words.

For a volume-uniform family assume bounded matter path incidence norms,
`Nb/V<=b`, `Np/V<=p`, and fixed finite scalar coefficients. Length or number of
paths cannot grow silently with volume. These hypotheses are met by fixed
`RotationBand` nearest/optional length-two paths on the cubic boxes in the CLI.

## 2. An exact nonzero neutral Gauss state

At **each** vertex use the epsilon singlet in `wedge²4 tensor wedge²4 = 6 tensor6`:

\[
 \chi_x={1\over\sqrt6}\sum_{i<j,\ k<l}
       \epsilon_{ijkl}\,c_{Hi}^\dagger c_{Hj}^\dagger
                            c_{Lk}^\dagger c_{Ll}^\dagger|0\rangle.
\]

Only the six complementary-pair occupations occur, orthogonally. Thus its raw
norm squared is exactly six. The actual code's CAR and `det(SU4)=1` give
`(Q_H^a+Q_L^a)chi_x=0` for all fifteen generators. It is exactly two-heavy and
two-light, not a mean Slater cancellation. The CLI constructs each component
by actual `PauliCode` creation words; it checks the full eight-mode CAR,
all fifteen singlet generators, the exact number equations and the reduced
one-body density `I4/2` for each species.

Choose a fixed positive `sigma`. The normalized site field wavefunction is

\[
 \psi_\sigma(u)=(2\pi\sigma^2)^{-60/4}
                    e^{-\rho^2/(4\sigma^2)}.
\]

The **density** variance of each canonical coordinate is `sigma²`. The
wavefunction exponent is consequently `-rho²/(4sigma²)`, not
`-rho²/(2sigma²)`. This function is invariant under all orthogonal adjoint colour
rotations, so its field colour generator vanishes exactly. Let every colour
link be the normalized Haar constant `1`, and every EM link and external
neutral boundary port have electric flux zero. Link derivatives then vanish.
The product

\[
 \Psi_{\rm n,tr}=\bigotimes_x(\psi_\sigma\chi_x)
                      \otimes1_{\rm colour\ links}\otimes|E=0\rangle_{\rm EM}
\]

is therefore an exact nonzero neutral state in the full continuous Gauss
space. Light number two cancels the EM background locally. Gaussian tails give
finite field kinetic and every polynomial moment; no projection/nonzero-overlap
hypothesis or selected field orbit is needed.

### Exact Gaussian constants

For independent `u_(mu,a)~N(0,sigma²)`, a diagonal Gram entry obeys
`E G_(mu,mu)²=15*17sigma⁴=255sigma⁴`. A distinct-row entry obeys
`E G_(mu,nu)²=15sigma⁴`. The Frobenius square counts four diagonal and twelve
ordered off-diagonal entries, giving

\[
 \langle\|G\|_F^2\rangle=(4\cdot255+12\cdot15)\sigma^4
          =1200\sigma^4,\qquad \langle\rho^2\rangle=60\sigma^2.
\]

Because `partial_(mu,a) psi=-u_(mu,a)psi/(2sigma²)`, each canonical momentum
square has mean `1/(4sigma²)`. Hence

\[
 \langle T_\Phi\rangle_x={7.5\sqrt\Lambda\over\sigma^2},\qquad
 \langle V_\Phi\rangle_x=600\Lambda\kappa\sigma^4.
\]

For a bond with independent endpoint fields, cross means vanish for every link
value. `TrBB=4` gives

\[
 \left\langle\sum_\mu\|\Phi_{x\mu}-U\Phi_{y\mu}U^\dagger\|_{HS}^2\right\rangle
      =4(60+60)\sigma^2=480\sigma^2.
\]

Thus alignment costs `60Lambda J sigma²` per bond. The positive colour
plaquette cost is safely at most `2Lambda K4` per plaquette. Haar constants
have zero colour electric energy. For simple nontrivial EM plaquettes the
constant-electric-state magnetic expectation is exactly zero. Both species'
onsite means vanish: reduced density `I4/2` contracts a traceless field.
Every hopping term changes fixed local species numbers, so its expectation
vanishes as well, including the whole `-Lambda h_H`.

Since `sqrt(Lambda)<=Lambda`, the neutral-sector ground infimum satisfies

\[
 E_{0,\rm n}\le\Lambda V C_0,\qquad
 C_0={7.5\over\sigma^2}+600\kappa\sigma^4
                 +60J\sigma^2{N_b\over V}+2K_4{N_p\over V}.
\]

Taking density upper bounds `b,p` yields one `C0` independent of volume.
`sigma` is an explicit chosen trial width, not a measured ground variance.

For all three-dimensional open boxes, `b=p=3` is a common safe bound, so
`C0_uniform=7.5/sigma²+600kappa sigma⁴+180J sigma²+6K4`. Each positive
displacement in the actual `scalar_paths` can be incident at most twice per
vertex, giving `c_f_uniform=8(|f0|+2sum_d||f_d||)`, independent of every
box side length. The API reports these common cubic constants and their `R4`
in addition to the tighter graph-specific incidence/density constants.

## 3. A local-sum coercivity bound, uniform in volume and Lambda

For each vertex, `G_x` is a positive four-by-four Gram matrix with trace
`rho_x²`. Therefore `||G_x||²>=rho_x⁴/4`. This is applied **before** summing
over vertices; there is no aggregate-Gram law or lost factor of `V`.

For four fermion modes `||c_x† A c_y||<=4||A||op` by singular-value
decomposition and norm-one primitive transfers. `||Phi_mu||HS=2||u_mu||` gives

\[
 \|h_{0,x}\|\le8|f_0|\rho_x,\qquad
 \|h_e\|\le8\|f_e\|_2(\rho_x+\rho_y).
\]

Transport along an arbitrary primitive path has norm one on genuine Haar/EM
spaces; no compressed-`PUP` unitary identity is used. Define

\[
 c_{f,x}=8\left(|f^f_0|+\sum_{e\ni x}\|f_e^f\|_2\right),\qquad
 c_H=\max_xc_{H,x},\quad c_L=\max_xc_{L,x},\quad c=c_H+c_L.
\]

A path is incident once at each endpoint. Intermediate vertices along a
length-two path do not carry its field coefficient, although its link
transport uses their primitive links. The inequality is a direct-integral
operator/quadratic-form estimate, not an expectation-only mean-field law.
For `Lambda>=1`, `|H_H|+|H_L|<=Lambda*c*sum_x rho_x`.

The exact Young maximum is

\[
 cr\le{\kappa\over16}r^4+D,\qquad
 D={3\over4}\left({4\over\kappa}\right)^{1/3}c^{4/3}.
\]

Discarding only nonnegative terms gives

\[
 H\ge{\Lambda\kappa\over16}\sum_x\rho_x^4-\Lambda VD-K_{EM}N_p.
\]

Together with the explicit trial, any normalized neutral ground density matrix
`Gamma` satisfies

\[
 {1\over V}\operatorname{Tr}\Gamma\sum_x\rho_x^4
 \le R_4:={16\over\kappa}\left(C_0+
 {3\over4}(4/\kappa)^{1/3}(c_H+c_L)^{4/3}
                         +K_{EM}{N_p\over V}\right).
\]

The sharper last term has an additional `1/Lambda`; the displayed bound is
uniform for `Lambda>=1`. A neutral minimizer exists for the usual strictly
positive electric Laplacians on every compact colour/EM link: quartic field
confinement, positive canonical kinetic, compact gauge manifolds and finite
matter space give compact resolvent, also on the closed Gauss subspace. If
only nonnegative gauge terms are specified, attainment is a separate hypothesis;
the infimum/trial and coercivity bounds still hold, with `epsilon/(Lambda V)`
for epsilon-minimizers. The exact charged ground-ensemble formula below uses
an attained neutral ground. No global neutral dominance is proved.

This is a **volume-average** field moment. On an open boundary it does not give
`<rho_x⁴><=R4` at each individual vertex.

## 4. Derive the full-field particle-hole symmetry

Use a **unitary**, not antiunitary, CAR particle-hole operation on both species
`C c_(xf,i) C†=c_(xf,i)†`, together with the configuration pullbacks

\[
 U_{\rm colour}\mapsto U_{\rm colour}^*,\quad
 z_{EM}\mapsto z_{EM}^*,\quad E_{EM}\mapsto-E_{EM},\quad
 \Phi_0\mapsto-\Phi_0^T,\quad
 \Phi_j\mapsto+\Phi_j^T\ (j=1,2,3).
\]

On finite CAR Fock space this unitary is explicit; for the eight-mode primitive
probe take `parity * product_i(c_i+c_i†)`. The product alone sends `c_i` to
`-c_i†`; parity removes that minus. The CLI constructs all of these operators
from actual code words and checks unitarity, ladder covariance, local numbers
and the colour-generator automorphism. Global finite CAR admits the same
construction with the required parity factor. It stays in the `S=+` code;
PH of the two global half-filled species also preserves global matter parity.

Let `s=(-1,+1,+1,+1)`. The actual coefficient condition is

\[
 s_\mu f_{e\mu}=-f_{e\mu}^*,\qquad s_0 f_0=-f_0^*.
\]

This holds for **real** Wilson/scalar and **imaginary** kinetic path
coefficients, for both standard and improved `scalar_paths`. After transforming
fields, path links and EM phase,

\[
 M'_e={z_e^*\over2}\sum_\mu s_\mu f_{e\mu}
      (\Phi_{x\mu}^T U_e^*+U_e^*\Phi_{y\mu}^T)
                         =-M_e^{\dagger T}.
\]

The onsite block is minus its transpose too. Thus the entire one-body matrix
satisfies `h'=-h^T`. CAR now gives

\[
 C(c^\dagger h'c)C^\dagger=\operatorname{Tr}h'-c^\dagger h'^T c
                         =c^\dagger hc,
\]

because all onsite fields are traceless and off-diagonal blocks have no trace.
This separately preserves `h_H,h_L`, hence also the whole heavy coefficient
`-Lambda`. A general complex coefficient law need not have this symmetry;
the CLI exposes every path's coefficient phase and a wrong-spatial-sign
control. The theorem's `w=2` conclusion requires the displayed condition.

Transposition is a real orthogonal map on the raw fifteen-coordinate basis.
`rho²` is unchanged and `G'_(mu,nu)=s_mu s_nu G_(mu,nu)`, so the full quartic
and canonical kinetic are invariant. Alignment differences become signed
transposes of the original differences, preserving their HS norm. Haar measure
is preserved by the outer automorphism `U->U*`; colour electric Casimirs and
`ReTr Uloop` are unchanged. In the colour Gauss algebra the operation sends
`T->-T^T` on matter, adjoint field and link generators consistently. Thus it
maps the full physical domain to itself, not just centre/mean-charge constraints.
Compact EM link inversion and electric-flux inversion preserve electric and
magnetic terms; `N_L(x)-2` changes sign, as required by EM Gauss. Neutral
external ports remain in their zero-flux/trivial-representation sector.

For any actual neutral ground ensemble `Gamma`, average it with its PH image:
`Gamma_sym=(Gamma+P Gamma P†)/2`. It is still a stationary neutral ground
ensemble, even with degeneracy. At **every** vertex,

\[
 \operatorname{Tr}\Gamma_{sym}N_L(x)=2,
 \qquad\operatorname{Tr}\Gamma_{sym}(4-N_L(x))=2.
\]

This controls the charge denominator without translation or a homogeneous
periodic graph. It does **not** convert a spatial mean field moment into a
pointwise moment. PH also preserves the moment bound because it preserves
`rho_x`.

## 5. Exact physical boundary charge trial and commutator

Let `B` be declared actual external boundary ports of the **same open graph**.
The neutral domain fixes their colour representation to trivial and outgoing
EM flux to zero. At one port `x` the charged domain instead has fundamental
external colour for a hole, antifundamental for a particle, and outgoing EM
flux `q=-1/+1`. Give that external EM flux the specified energy
`UE_boundary*q²/2`. The external colour representation has no additional
Casimir energy in this law; if an external colour electric Hamiltonian is added,
its energy must be added to the bound. These are prescribed boundary data,
not extra matter modes or hidden dynamical spectators.

The actual code-word maps are

\[
 A_{x,-}=\sum_i c_{xLi}\otimes|i\rangle_4,\qquad
 A_{x,+}=\sum_i c_{xLi}^\dagger\otimes|\bar i\rangle_{\bar4}.
\]

Their external representations intertwine all fifteen colour Gauss generators,
including the field and link pieces, and EM Gauss holds because changing the
light charge and the flux at this same boundary vertex by `q` leaves every
internal electric field unchanged. Actual CAR yields

\[
 W_{x,-}=A_{x,-}^\dagger A_{x,-}=N_L(x),\qquad
 W_{x,+}=A_{x,+}^\dagger A_{x,+}=4-N_L(x).
\]

The maps commute/intertwine the full field kinetic/potential/alignment, whole
heavy Hamiltonian, colour electric/plaquette, and internal EM electric/magnetic
spectator dynamics. Field derivatives and genuine link coordinates are not
changed; only the light commutator and external EM energy remain.
For an intertwining spectator `O`, its normalized charged expectation is
`Tr Gamma W O / Tr Gamma W`, **occupation-weighted**, and need not equal
`Tr Gamma O`. Cancellation in the **total stationary Hamiltonian** is exact:

\[
 \Delta_{x,\pm}^{tr}:=
 {\operatorname{Tr}A_{x,\pm}\Gamma_{sym}A_{x,\pm}^\dagger H_{x,\pm}
       \over w_{x,\pm}}-E_{0,n}
 ={\operatorname{Tr}\Gamma_{sym}S_{x,\pm}\over w_{x,\pm}}
                                      +{UE_{boundary}\over2},\qquad w_{x,\pm}=2.
\]

Writing the full light one-body matrix as `h_(yz)` (with operator-valued
field/link entries commuting with matter), exact CAR gives

\[
 S_{x,-}=\sum_i c_{xi}^\dagger[H_L,c_{xi}]
                  =-\sum_z c_x^\dagger h_{xz}c_z,\qquad
 S_{x,+}=\sum_i c_{xi}[H_L,c_{xi}^\dagger]
                  =-\sum_z c_z^\dagger h_{zx}c_x.
\]

The potential trace term in `S+` is zero, since `Tr h_xx=0`. These are one row
and one column, not the whole light Hamiltonian. Their ground expectations are
real, although the row/column operators individually need not be Hermitian.
Only the onsite and paths **incident** to `x` enter. Unitary transport and
four norm-one transfers give the sharper directed-row estimate

\[
 |\operatorname{Tr}\Gamma_{sym}S_{x,\pm}|
 \le8|f_0|\langle\rho_x\rangle
       +4\sum_{e\ni x}\|f_e\|_2
                              (\langle\rho_x\rangle+\langle\rho_{y(e,x)}\rangle).
\]

The factor four here is for one directed row/column half; the factor eight
in the whole-H coercivity bound counts the path **and** its adjoint. If PH is
unavailable, the same exact formula holds for a stationary neutral ensemble
with explicit positive denominator `w_x`; no `w=2` may be substituted from
global half-filling alone.

### Resolve the open-boundary/local-moment issue by averaging

Define the nonnegative exact incidence coefficients

\[
 b_v(B)=8|f_0|\mathbf1_{v\in B}
 +4\sum_{e=(xy):\,v\in\{x,y\}}\|f_e\|_2
                                  (\mathbf1_{x\in B}+\mathbf1_{y\in B}).
\]

Sum the row estimates over the physical boundary set. Jensen and discrete
Holder with conjugate powers `4/3,4` give, for **each charge sign**,

\[
 \left|{1\over|B|}\sum_{x\in B}
              (\Delta_{x,\pm}^{tr}-UE_{boundary}/2)\right|
 \le {V^{1/4}R_4^{1/4}\over2|B|}
                           \left(\sum_v b_v(B)^{4/3}\right)^{3/4}.
\]

In fact the same right-hand side bounds the average of the absolute deviations.
No individual local moment or translational averaging is assumed. It also
implies that at least one boundary location has trial cost no greater than the
average upper bound for either sign; it does not make every location uniform.

Since `b_v(B)<=cL`, a simple sufficient family hypothesis is
`alpha=|B|/V>=alpha0>0`, giving the volume- and `Lambda`-uniform average bound

\[
 {1\over|B|}\sum_{x\in B}\Delta_{x,\pm}^{tr}
       \le{UE_{boundary}\over2}+{c_L\over2\alpha_0}R_4^{1/4}.
\]

For a fixed-thickness open slab of thickness one or two, every vertex is an
actual exposed boundary vertex: `B=V`. More generally bounded thickness gives
positive boundary density. For `B=V`, `b_v(B)=c_(L,v)` and the sharper bound is
`UE_boundary/2+cL R4^(1/4)/2`. This is the CLI default `(2,4,4)` slab. Both the
neutral ensemble and charged ports belong to that **same** graph and same law;
there is no periodic-ground/boundary-probe gluing.

A growing three-dimensional open cube has vanishing boundary density. The
exact Holder formula still applies, but these premises alone do not certify a
volume-uniform boundary average there. An interior insertion on a graph with
only remote external ports needs colour Wilson transport and EM dressing.
Those dressings generally change link electric energies and their commutators
with the light and field-alignment laws; their actual costs must be computed.
On a closed periodic graph a net charge also needs specified compensating
charge/flux data. None of those costs is replaced by `UE_boundary/2` here.
Consequently this is **not** a uniform bulk mass theorem.

## 6. Executable API and evidence boundary

`full_field_charge_bound.py` supplies:

- `open_box(shape)` and `primitive_edges(graph,band)`: actual open boundary,
  bonds/plaquettes, and all actual `scalar_paths` with adjoints counted once.
- `gaussian_constants(sigma)` and `gaussian_wick_control(sigma)`: density-width
  convention, exact Wick and canonical derivative constants.
- `incidence_coefficients(...)`: safe local whole-H coefficients.
- `theorem_constants(graph,light_band,heavy_band,...)`: explicit `C0`, Young
  constant, `R4`, exact boundary row counts, Holder charge-average upper and
  positive-density sufficient upper, plus side-length-independent cubic-family
  constants and all-boundary slab upper. Heavy and light bands may differ if
  each satisfies the real/imaginary PH coefficient condition.
- `primitive_singlet_and_ph()`: actual `S=+` eight-mode primitive singlet/CAR/PH
  controls; no canonical ladder replacement or spectator.
- `ph_coefficient_controls(band,PH,units)`: every actual scalar path, genuine
  unitary primitive path values, full-field transpose/Gram/alignment invariance,
  and second-quantized PH identity in actual code bilinears. Its deliberately
  wrong common-negative-transpose control discriminates nonzero kinetic terms.

The CLI requires `--output`. It checks the pinned primitive source hashes and
writes all source hashes, theorem hypotheses/constants and algebra controls.
For parent-owned execution, for example:

```sh
/home/administrator/substrate-framework/.venv/bin/python \
  proposals/P256-qed-backward/full_field_charge_bound.py \
  --shape 2 4 4 --output /tmp/full_field_charge_bound_receipt.json
```

`--improved` exercises the real length-two Wilson path coefficients as well.
Scalar coefficients, graph shape, `kappa,J,K4,KEM,sigma` and external EM electric
cost are CLI parameters. The finite primitive algebra probes do not numerically
sample the full `R^(60V)` ground, prove a selected Clifford phase, or provide a
positive lower charge gap. A positive upper trial cost has no such implication.

### Parent-owned executed constants

The [default receipt](full_field_charge_bound_receipt.json) and
[improved-path receipt](full_field_charge_bound_improved_receipt.json) record
actual primitive/Wick/PH controls from implementation SHA256
`b86f5fe62f2a838f2a5cd84e70c1b3e74e801aef91cd2dda6d4ce123e58541bf`.
Both use the same `(2,4,4)` graph, `V=32`, `B=V`, `sigma=.3`,
`kappa=1,J=.05,K4=.01,KEM=.02,UE_boundary=.01`. In **both** calculations,
`C0=88.75958333333335`; the zero hopping means in the neutral trial do not
change with improved paths.

| Path law | paths/species | `cH=cL` | `R4` | boundary-average trial upper |
|---|---:|---:|---:|---:|
| Default | 64 | `15.21110255093` | `3229.643852745` | `51.95129485249` |
| Improved | 96 | `21.2` | `4236.939068751` | `77.52738872469` |

The distinct uppers arise from the actual path/incidence coercivity constants.
These executions evaluate the analytic theorem and finite primitive algebra,
not a full-field eigenstate. Volume-uniform boundary averages require positive
boundary density; neither positive upper proves a bulk or lower charge gap.
The [paired covariant trial](gauss_born_bundle.md) and
[full-band section](full_band_gauss_section.md) strengthen separate state
constructions without changing that verdict. The source manifests retain the
derivation's receipt-time hash; this note's evidence update is not a rerun.
