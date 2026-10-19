# Boundary-dressed charge preserves the actual Gauss angular cost

This is a charge **upper-cost** construction, not a positive lower charge-gap
proof. The executable consumes the existing atomic SU(4)-Gauss Hamiltonian,
source SHA256 `9e91c25bc4bcbc5f2306b38d4716b4e3af4664fc22757aedda527c056aea096b`.
It preserves every native centre-allowed physical state and both link indices.
Its fixed simultaneous adjoint orbit and `J=0` remain declared EFT restrictions.
The unrestricted quartic field theorem below is a separate microscopic law,
not an interpretation of that atomic calculation as a 60-coordinate simulation.
No scientific execution was performed by the implementation worker; the parent
owns execution, source-bound receipts and the new independent-photon integration.

## 1. The actual bosonic-word charge operator

`PauliCode` describes bosonic edge qubits projected by the actual cycle word
`S=+1`. Modes are `LH=0..3`, `LL=4..7`, `RH=8..11`, `RL=12..15` and the
**neutral** parity marker is mode 16. In the normalized code occupation basis,
for an occupied right-light mode use the directed word

\[
 c_{Ri}: -i A_{Ri,16}\,n_{Ri},\qquad
 c_{Ri}^{\dagger}: -i A_{16,Ri}(1-n_{Ri}).
\]

Here each `A` is `code.words.path_A(code.words.path(source,target))`;
`code.element(mask,new_mask,word,sign=1)` evaluates both raw code bits, the
actual word phase and the target code phase. The occupation selector is
implemented by visiting the appropriate source masks. This is the same
charged boundary-path convention as `quantum_spinor_plaquette.creator_overlap`.
The marker toggles physical parity; it has no electromagnetic or colour charge.
There are no constructed canonical-CAR matrices, Slater minors or preassembled
exterior-power operators in this consumer.

The actual words give
\(\{c_i,c_j^\dagger\}=\delta_{ij}\) and \(\{c_i,c_j\}=0\).
The CLI checks the full sixteen-state right-light local Fock space, including
local occupations 0 through 4. Independently, on the complete two-cell neutral
matter domain it checks

\[
 C_{-,i}^{\dagger}C_{-,j}+C_{+,j}^{\dagger}C_{+,i}
   =\delta_{ij}\mathbf1,
\]

where each rectangular `C` was evaluated directly from these code words.

Define unnormalized right-boundary dressings

\[
 A_-\psi=\sum_{i=1}^4 c_{Ri}\psi\otimes|i\rangle_{4},\qquad
 A_+\psi=\sum_{i=1}^4 c_{Ri}^{\dagger}\psi\otimes|\bar i\rangle_{\bar4}.
\]

Actual CAR, not a mean-filling assumption, gives

\[
 A_-^{\dagger}A_-=N_R,\qquad A_+^{\dagger}A_+=4-N_R.
\]

On a fixed `N_R=n>0`, `A_-/sqrt(n)` is an isometry; it excludes `n=0`.
On `n<4`, `A_+/sqrt(4-n)` is an isometry; it excludes `n=4`.
For a general normalized state, the normalized image uses respectively
\(\sqrt{\langle N_R\rangle}\) or \(\sqrt{\langle4-N_R\rangle}\).
It is not an all-state norm-one map. Zero denominators are real null domains,
not grounds for a fake charged ground fallback.

## 2. All fifteen generators, both Gauss constraints, and the whole heavy H

Normalize the colour generators by `Tr(T_a T_b)=delta_ab/2`.
The actual even code-word units give
\([Q_R^a,c_i]=-\sum_j(T_a)_{ij}c_j\) and
\([Q_R^a,c_i^\dagger]=\sum_jc_j^\dagger(T_a)_{ji}\).
Thus for each of all fifteen generators,

\[
 (Q_{R,\mathrm{charged}}^a+Q_B^a)A_\pm
   =A_\pm Q_{R,\mathrm{neutral}}^a,
 \qquad Q_{L,\mathrm{charged}}^a A_\pm=A_\pm Q_{L,\mathrm{neutral}}^a.
\]

For holes `Q_B=T_a`; for particles `Q_B=-T_a*`.
These signs follow from fundamental versus antifundamental covariance, not
from the smaller charged-matter Casimir. In the existing tensor convention,
`Eleft` acts by `T_a.T` on the first colour-link index, and `Eright` by `-T_a`
on the second. The charge maps leave both indices, and the representation name,
unchanged. Consequently

\[
 K_G={1\over2I}\sum_a
 \big[(E_L^a-Q_L^a)^2+(E_R^a-Q_R^a-Q_B^a)^2\big],
 \qquad K_{G,c}A_\pm=A_\pm K_{G,0}
\]

for **every positive inertia**, including one tending to zero. This is an
operator identity, not an equality only of mean charges or Casimirs.
The script compares the existing expanded `orientation` with an independently
applied sum of squares, checks both full dressed Casimirs and sparse operator
intertwining for all fifteen generators. Actual wrong boundary-sign and omitted
boundary operators must have nonzero defects on deterministic physical vectors.

The heavy entire Hamiltonian consists of even heavy onsite and link-dressed
transfer words. Actual CAR makes each commute with every right-light charge
word. Their colour-link factors also commute with the boundary dressing, so

\[
 H_{H,c}A_\pm=A_\pm H_{H,0}
\]

for balanced whole-H sign `-Lambda` **or** same sign `+Lambda`.
This does not need complementary classical projectors or a dynamically proved
relation between heavy and light ground states. SU(4) electric energy is also
unchanged because its link representation and indices are unchanged.

**Expectation caveat.** Intertwining is not generally equality of normalized
charged and *unweighted* neutral heavy/angular means. For an intertwining `O`,

\[
 {\langle A\psi,O_cA\psi\rangle\over d}
   ={\langle\psi,W O_0\psi\rangle\over d},
 \quad W=A^\dagger A,\quad d=\langle W\rangle.
\]

The CLI reports this occupation-weighted comparator, its residual, and the
unweighted neutral mean separately. Correlations between local occupancy and
heavy/angular cost are not silently discarded.

## 3. Centre and electromagnetic Gauss law; independent loop extension

The physical fibre keeps `N_total_left mod 4 = centre(link_rep)`.
Removing right light leaves this left restriction unchanged. At right,
`N_R,total` loses one while boundary `4` contributes centre one. Adding right
light gains one while boundary `bar4` contributes centre three. Both preserve
the exact centre congruence. Neutral `(NH,NL)=(4,4)` therefore maps into actual
`(4,3)/B4` or `(4,5)/Bbar4`, with electromagnetic charge `q=-1/+1`.
No link representation cutover, local-filling truncation or missing-boundary
restriction is substituted for the actual fibre.

On the atomic tree, `e=nlight_left-2` is unchanged by right light charge.
The boundary field is `q`; the electric difference is exactly `UE1/2`.
Heavy occupations do not enter the electromagnetic count. The parity marker
likewise does not enter it.

For an independent compact loop index `ell`, tensor the same charge map with
`1_ell`. On two parallel paths take

\[
 e_0=\ell+n_{L,\mathrm{light}}-2,\qquad e_1=\ell.
\]

Since `A` changes neither `ell` nor `nlight_left`, both internal electric terms
intertwine. The loop shift `S|ell>=|ell+1>` commutes with `A`, as do all photon
magnetic operators acting only on the loop. This is exact on the infinite loop
and on any identical finite spectator cutoff. The script includes a deterministic
three-ell probe of the electric and magnetic identity; that probe is not an
infinite-cutoff photon spectral certificate. Light hopping on independent EM
paths still contributes to the light commutator, rather than being called a
spectator. The parent owns joining the new photon module to this consumer.

## 4. Primitive one-cell Casimir proof and its adverse complement

The small probe freezes a colour-singlet heavy occupation and retains all
sixteen right-light occupations. Its units are products of the actual charged
word matrices. Restricting the source to `N=2` gives dimension six and Casimir
`C2=5/2`. Actual code-word maps give

\[
 {A_-\over\sqrt2}:\wedge^2 4\longrightarrow4\otimes4,
 \qquad
 {A_+\over\sqrt2}:\wedge^2 4\longrightarrow\bar4\otimes\bar4.
\]

Each image is six-dimensional, intertwines all fifteen generators, and has
**total matter-plus-boundary** Casimir `5/2`. The ten-dimensional symmetric
complement has Casimir `9/2`; it is not in the charged image. Charged matter
alone has Casimir `15/8` in both maps. Applying that matter-only Casimir to the
image produces a nonzero defect from `5/2`: inferring an angular-cost saving or
charged collapse from `15/8 < 5/2` omits the physical charged boundary and is
incorrect. The CLI checks these operator identities in the primitive small
consumer without importing exterior-power Hamiltonian builders.

## 5. Actual neutral ground and charged Rayleigh upper consumer

The default invocation is portable:

```sh
python proposals/P256-qed-backward/gauss_charge_intertwiner.py \
  --output /tmp/gauss_charge_intertwiner.json
```

Use the project scientific Python environment when running it. Defaults are
`Lambda=1`, `I=2`, the genuine minimal33 Haar link cutoff, `NH=4`, `NL=4`, and
the current atomic light/heavy coefficient shape. `--lambdas 1 8 64` and
`--heavy-sign 1` expose the two whole-H hypotheses without changing the charge
construction. All native local light occupations 0..4 are retained and reported.

The CLI obtains the neutral state's two lowest returned levels by actual sparse
`eigsh(which='SA')` on the complete physical Hamiltonian, then QR-orthogonalizes
and performs the Hermitian projected Ritz refinement. It explicitly applies the
**full H** to each returned eigenvector and enforces the declared scaled residual
and orthogonality limits. It does not label a matter-only, Schur-only or projected
trial as the full-H ground. This is nevertheless a numerical ground estimate,
not a certified exact neutral lower energy.

For a normalized exact stationary neutral ground `psi`, put
`d_-=<N_R>` and `d_+=<4-N_R>`. The charged trial difference is

\[
 \Delta_{-,\mathrm{trial}}={UE1\over2}
  +{\sum_i\langle c_i^\dagger[H_L,c_i]\rangle\over d_-},
 \qquad
 \Delta_{+,\mathrm{trial}}={UE1\over2}
  +{\sum_i\langle c_i[H_L,c_i^\dagger]\rangle\over d_+}.
\]

This follows by applying the full charged H to `A psi`, using every exact
non-light intertwining identity and `H0 psi=E0 psi`. The executable independently
computes the charged full-H Rayleigh energy, the commutator through rectangular
maps and actual light Hamiltonian actions, and the onsite/hopping formula below.
Numerical comparison tolerance includes the enforced neutral full-H residual.

For a traceless onsite matrix, the real stationary numerators both equal

\[
 -\langle H_{L,\mathrm{onsite},R}+\tfrac12H_{L,\mathrm{hopping}}\rangle.
\]

At operator level the hole expression contains one directed hopping half;
the particle expression contains the opposite half and an onsite trace term.
Tracelessness deletes the latter. Stationarity of right occupation, since
all non-light H terms commute with it, deletes the imaginary current expectation
and makes the two directed expectations equal. The script reports the complex
commutator, onsite mean, full light hopping mean and stationarity residual rather
than assuming that every local count equals half filling. It reports the actual
occupancy probability distribution, `p0`, `p4` and both actual denominators.

The charged Rayleigh **energy** is variationally an upper bound to the charged
sector ground energy. With exact neutral `E0`, its difference is an upper bound
to the corresponding charge gap; it is never a positive lower gap proof. With
only a numerical neutral Ritz estimate the reported difference is a numerical
comparison, not a rigorous charge-gap bound, because an upper estimate of neutral
energy is not a certified neutral lower bound. No sign of the trial difference
is promoted to a theorem about positivity or collapse.

## 6. Unrestricted quartic field moment theorem

### Declared law and coefficient bounds

Let four **unrestricted** traceless Hermitian fields be

\[
 \Phi_\mu=\sum_{a=1}^{15}u_{\mu a}B_a,\quad
 \operatorname{Tr}B_aB_b=4\delta_{ab},\quad
 \rho^2=\sum_{\mu a}u_{\mu a}^2,\quad
 G_{\mu\nu}=\tfrac14\operatorname{Tr}(\Phi_\mu\Phi_\nu).
\]

All sixty real field coordinates are canonically quantized, with kinetic law
`Pi^2/(2M)` and `M=Lambda^(-1/2)`. The bare potential is
`Lambda*kappa*||G||_F^2/2`, `kappa>0`; its preferred field is zero, not an imposed
Clifford radius. Heavy matter is EM-neutral and its whole Hamiltonian is
`-Lambda h_H` or `+Lambda h_H`. Heavy even terms commute with light charge;
the proof does not require opposite-sign selection. Colour/U(1) path transport
is unitary on the unrestricted gauge Hilbert space (a projection may only reduce
its norm). Assume nonnegative field kinetic and electric terms and photon
magnetics bounded below by `-K_-`, uniformly in `Lambda`.

Write each unscaled flavour Hamiltonian as
`h_f(u)=sum_(mu,a,alpha) u_(mu,a) s^f_(alpha,mu,a) O_alpha`, including onsite
terms and **both** directed path adjoints, with `||O_alpha||<=1`. This expansion
exists for bounded fermion transfer operators and bounded unitary path matrix
entries on a fixed finite graph. Set

\[
 k^f_{\mu a}=\sum_\alpha|s^f_{\alpha,\mu a}|,\qquad
 C_f=\Big(\sum_{\mu a}(k^f_{\mu a})^2\Big)^{1/2}.
\]

Triangle and Cauchy-Schwarz give explicit coefficient bounds
`||Hheavy(u)||<=Lambda*C_H*rho`, `||Hlight(u)||<=C_L*rho`.
These finite constants depend on graph size and the declared scalar hopping and
onsite coefficients; they are not claimed uniform in graph volume.

For a graph with independent sixty-coordinate fields **at each vertex**, the
same displayed factor four is valid if the declared quartic law uses
`G=sum_x G_x`, `rho^2=sum_x rho_x^2`: this aggregate Gram is positive, has rank
at most four, and is invariant under every local colour rotation. This is an
explicit finite-graph law, not a hidden replacement of a local potential.
If instead the law is the local sum `sum_x ||G_x||^2`, the bound is
`rho^4/(4V)` for `V` vertices; use `kappa_eff=kappa/V` throughout the following
constants. A factor-four bound for the local sum with global rho would be wrong.
For one sixty-coordinate field block, `V=1` and the original constants apply.

### Quartic coercivity, Young constant and the moment bound

`G` is positive semidefinite, `Tr G=rho^2`, and has four rows. Thus

\[
 \|G\|_F^2\ge{\rho^4\over4},\qquad
 V_{\rm bare}\ge{\Lambda\kappa\over8}\rho^4.
\]

For `Lambda>=1`, set `C=C_H+C_L`. The explicit Young inequality is

\[
 C\rho\le{\kappa\over16}\rho^4+D,\qquad
 D={3\over4}\Big({4\over\kappa}\Big)^{1/3}C^{4/3}.
\]

It follows on the full Gauss domain, as a quadratic-form bound, that

\[
 H\ge {\Lambda\kappa\over16}\rho^4-\Lambda D-K_-.
\]

Assume an **actually constructible neutral Gauss trial** supplies
`E0<=C_trial*Lambda` for every `Lambda>=1`. Then a normalized neutral ground
(or a stationary neutral ground density matrix) satisfies

\[
 \langle\rho^4\rangle\le
 {16\over\kappa}\left(C_{\rm trial}+D+{K_-\over\Lambda}\right)
 \le R_4:={16\over\kappa}(C_{\rm trial}+D+K_-),
 \qquad \langle\rho\rangle\le R_4^{1/4}.
\]

`quartic_moment_bound` returns these constants when passed **proven**
`kappa,C_H,C_L,C_trial,K_-`. The atomic fixed-orbit receipt does not supply
`C_trial` for the unrestricted law. If the full ground is not known to be
neutral, this is a theorem about the neutral-sector infimum and its ground,
not an assertion of neutral-sector global dominance.

### An explicit nonzero neutral Gauss trial, not a mean Slater projection

For the symmetric two-cell graph with `NH=NL=4`, put **exactly** two heavy and
two light modes at each cell. A local SU(4) singlet is constructed by applying
actual charged creation words to the code vacuum with the invariant epsilon
contraction

\[
 |\chi_x\rangle\ \propto\!
 \sum_{i<j,\,k<l}\epsilon_{ijkl}
 c_{Hi}^{\dagger}c_{Hj}^{\dagger}c_{Lk}^{\dagger}c_{Ll}^{\dagger}|0\rangle.
\]

The six nonzero complementary-pair occupations are orthogonal, so the norm is
nonzero. Actual CAR gives the invariant epsilon transformation law. Each cell
has total colour centre zero; light number two supplies exact neutral U(1)
background cancellation, not merely cancellation of a mean density. The tensor
product of these local singlets is an exact colour singlet. All creation phases
in this definition are those of the bosonic code words. On a graph with more
vertices the same local construction has `NH=NL=2V`.

Choose an irreducible simultaneous field root `Phi_*` (commutant only the
centre), for example a simultaneous four-generator Clifford tuple as a centre
of a packet, **not** as a constraint on the quantum field. For fixed positive
width sigma let

\[
 f(\Phi)=\int dU\,
 \exp\left[-{1\over2\sigma^2}\sum_\mu
   \|\Phi_\mu-U\Phi_{*,\mu}U^\dagger\|_F^2\right].
\]

On independent local fields average independently at each vertex. This is
explicit Haar gauge averaging of a smooth wavepacket near the compact root
orbit. The integrand is strictly positive, so `f` is nonzero; it is smooth,
gauge invariant, square integrable and has finite canonical kinetic and every
polynomial field moment. Compactness of the root orbit provides Gaussian tail
bounds and differentiation under the integral. No unproved nonzero projection
of a coloured mean Slater state is used.

Take `Psi_trial = normalized f * product_x chi_x * 1_SU4links * |ell=0>_U1`.
The constant colour-link function is normalized under Haar measure and has
zero electric Casimir. Local singlets and invariant `f` satisfy every continuous
colour Gauss generator. The U(1) trial has exact zero internal and boundary
electric fields because light count is two at each vertex. Photon magnetics
have finite bounded expectation. This is a genuine full neutral Gauss trial.
For any finite graph it supplies finite constants

\[
 T_f=\tfrac12\langle\Pi^2\rangle_f,\quad
 Q_f={\kappa\over2}\langle\|G\|^2\rangle_f,\quad
 C_{\rm trial}=T_f+Q_f+(C_H+C_L)\langle\rho\rangle_f+K_{\rm abs},
\]

where `K_abs` bounds the absolute photon magnetic expectation (or include its
actual upper expectation). The constants are finite integrals of this explicit
normalized packet, independent of `Lambda`. The fixed width is enough:
`M^(-1)=sqrt(Lambda)<=Lambda`. Thus `Etrial<=C_trial*Lambda` for `Lambda>=1`.
The packet need not be semiclassically localized by the ground law, and no SSB,
Clifford ground selection or harmonic approximation is needed for this estimate.

### From exact boundary intertwining to O(1) charged trial cost

The charge words are independent of canonical field coordinates and photon
loop indices. They intertwine the full colour generators including the boundary
and commute with field differentiation. Thus every non-light field kinetic,
bare quartic, Gauss angular, colour electric, whole-heavy and photon spectator
operator intertwines even as the canonical inertia shrinks. This is the full
operator theorem, not a bare charged-matter Casimir argument.

For the light commutator, a conservative explicit bound is

\[
 \left\|\sum_i c_i^\dagger[H_L,c_i]\right\|,
 \left\|\sum_i c_i[H_L,c_i^\dagger]\right\|
 \le 8C_L\rho,
\]

since there are four norm-one charge words and each commutator norm is at most
`2||H_L||`. A sharper coefficient bound comes from the actual right onsite and
one directed hopping half. Given an actual occupation denominator lower bound
`d_-/d_+>=d_*>0`, the moment theorem therefore yields

\[
 |\Delta_{\pm,\rm trial}-UE1/2|
 \le {8C_L R_4^{1/4}\over d_*}=O(1)
 \quad (\Lambda\to\infty).
\]

A fixed `NL=4` sector alone does **not** imply `d_*=2` for a single state.
If the untruncated two-cell Hamiltonian is invariant under neutral left/right
reflection, symmetrize an actual stationary neutral ground density matrix as
`rho_sym=(rho+R rho R^dagger)/2`. Reflection gives
`R^dagger N_R R=N_L=4-N_R`, hence **exactly** `<N_R>=2` and `<4-N_R>=2`.
The reflected states stay actual neutral ground states, even in a degenerate
multiplet. Normalize the mixed charged trial `A rho_sym A^dagger` by its trace
two. It need not have `p0=0`; positive denominator is an ensemble fact, not a
local-state truncation. This reflection hypothesis must be checked for the
chosen graph, coefficient law and untruncated gauge domain; the CLI reports its
actual single-ground counts without assuming it.

The conclusion is uniform **upper-cost control** for charge trial states under
these explicit hypotheses. It supplies neither a positive lower charge gap nor
a full phase, continuum, infinite-volume electron or canonical QED construction.
It also does not prove charged collapse. The exact dressed boundary removes the
invalid premise that a smaller isolated charged-matter Casimir alone lowers the
physical Gauss angular energy.

## 7. Executed primitive intertwiners and charged trials

The parent-owned current CLI completed `Lambda=1,64`, naturally exiting zero; `gauss_charge_intertwiner_receipt.json` binds source SHA-256 `861008cb7dea64d2b2e0cef513c62de1d55539a1527a5b6fea8b5b879ea956fa`. All native right-light occupations `0..4` remain present. The actual whole-neutral-domain CAR residual is `3.11e-14`; all-fifteen-generator, angular, whole-heavy and spectator-loop intertwiners pass. Incorrect or omitted boundary generators produce angular defects greater than `0.57`, rather than vanishing controls.

| Lambda | neutral numerical ground energy | both charged trial Rayleigh differences | actual normalization weight |
|---|---:|---:|---:|
| 1 | -0.475184727962 | 0.317854865865 | 2 |
| 64 | -52.155417563885 | 0.407720654569 | 2 |

The neutral scaled full-H residual maxima are `5.61e-14` and `9.52e-10`; the independent charged energy formulas agree within `1.03e-12`. The trial numbers are **upper-cost comparisons**, not positive lower charged gaps. This finite fixed-orbit consumer does not execute the unrestricted quartic field theorem above.

