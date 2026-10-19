# Original-law physical binary-cube endpoint lift

## 1. Result and ownership

This construction gives an **exact physical proper-cubic orbit intertwiner** of
an original-law, full-rank metric-Clifford centre minimum, its complete colour
Gauss orbit, the entire fixed-number heavy/light CAR spaces, and an opposite
colour/EM reference. Its body-restored fundamental endpoint has the binary-cube
cocycle. An actual quarter-turn has fourth power `-I` on that endpoint; the
complete `NH=2V`, odd-light, opposite-reference operation has fourth power `+I`.

It is not a proof of localized mechanical spin one-half, an interacting charged
pole, a thermodynamic deconfined sector, or a continuous Noether embedding. In
particular, an internal cocycle alone is not physical spin. The advance here is
the explicit **physical** position/vector/link-index operation, its transport
between all 64 independently selected native Gram/centre classes, and the
previously essential reversed-axis seam gauge restoration. No old changed-law
continuum spin packet is recycled as a new original-law result.

The governing plan is `issue-222-continuing-backward-20260930`, revision
`381993ecb362c8c67dbc334aebbcb6093b64a7b78111b5a9e785d4746b9bbe3b`, template
`general`, revision
`89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`, work item
`campaign`, requirements `outcome` and `verification`. Only this `.md` and its
`.py` producer are owned here. Original frame selection, endpoint, current,
Gauss suppliers and their proofs are read-only.

`original_charged_principal.original_frame(..., native_toron=True)` owns the
full-ten-coordinate original Gram minimization. `BulkQuantumLaw.onebody` and
`toron_links` own the native endpoint and seam geometry;
`physical_spin_boundary.physical_cubic` owns position/vector/oriented-link
transport. `original_normal_gap_bound.peierls_derivatives` owns current/contact.
The geometry object's constructor initializes changed-27-selector metadata,
but neither its selected frame nor **any selector energy/leading functional**
is consumed. There is no added CAR species, isotropic-frame replacement,
angular/Fock/rotor cutoff, occupation mask, or inserted spin postulate.
The fundamental quantum CAR and compact gauge law remain assumptions of the
original quantum construction, not an Euler derivation of quantization.

## 2. Physical microscopic operation

Let `R` be any signed permutation matrix with determinant `+1`; there are exactly
24. Primitive sites are `x in (Z/NZ)^3`, with `N>=3`. Write `R4=diag(1,R)`.
The raw physical operation is

\[
 x\mapsto Rx,\qquad
 \Phi'_0(Rx)=\Phi_0(x),\qquad
 \Phi'_i(Rx)=R_{ij}\Phi_j(x).
\]

A positive link `(x,x+e_a)` goes to `(Rx,Rx+Re_a)`. If `Re_a=+e_b`, its colour
matrix and EM angle are copied to `(Rx,b)`. If `Re_a=-e_b`, they are copied to
`(R(x+e_a),b)` as `U^dagger` and `-A`, respectively. Matter is a spatial scalar:
`(P_R f)(Rx)=f(x)` with no internal spin action. This is precisely the existing
`physical_cubic` operation, not a gauge transformation or a redefinition of it.

The native endpoint has onsite `(m+3r/a) Phi0` and edge

\[
 h_{x,x+e_a}=e^{iA_{x,a}}\frac{F_a(x)U_{x,a}+U_{x,a}F_a(x+e_a)}2,
 \qquad F_a=-\frac{r\Phi_0+i v\Phi_a}{2a}.
\]

For a reversed link, `Phi'_b=-Phi_a`, so its new forward coefficient is the
adjoint of the old reverse coefficient. Consequently, for arbitrary full
fields, colour links and compact EM angles,

\[
 h(\Phi',U',A')=P_Rh(\Phi,U,A)P_R^\dagger. \tag{1}
\]

The same operation preserves the original scalar quartic field law, summed
alignment squares, summed colour/EM plaquette terms, flat field kinetic metric,
Haar link metrics and original full-number matter terms: all sums are permuted,
vector indices transformed orthogonally, and reversed link energies paired
with their adjoints. This symmetry does not use the changed local selector law.
It acts on all `60V` Cartesian field and `45V` colour-link coordinates and on
the whole compact EM space. It does not assert an arbitrary continuous rotation
symmetry of a finite cubic lattice.

Differentiating (1) for `A+t d` proves native current/contact covariance. The
rotated tangent `d'` uses exactly the same **oriented** angle transport as `A'`;
current is odd and contact is even under link reversal. No small-angle condition
on `A` is needed.

## 3. All native Gram/centre classes

For centre labels `c in Z4^3`, set `theta=(pi/2)c`. On native momenta, let
`x(p)=a*(Wilson mass, v sin(ap_1)/a, v sin(ap_2)/a, v sin(ap_3)/a)` as supplied
by the original coefficients. The original Gram objective is

\[
 f_c(G)=\frac1a\left[\frac\kappa2\operatorname{Tr}G^2
       -2\left\langle\sqrt{x(p+\theta/(Na))^TGx(p+\theta/(Na))}\right\rangle\right],
 \qquad \kappa=2.
\]

Its full symmetric-ten-direction Hessian is bounded below by `kappa` in
orthonormal symmetric coordinates. At a positive interior stationary minimum,
strict convexity makes `G_c` unique. A cubic signed permutation permutes the
native momentum set, rotates the three sine components and leaves the Wilson
mass unchanged. Reducing the rotated angle modulo `2pi` only permutes that
set again. Therefore

\[
 c_R=Rc\pmod4,\qquad G_{c_R}=R_4G_cR_4^T. \tag{2}
\]

This is an exact conditional statement at full-rank native minima, not a
floating proof that every volume is globally regular. The producer independently
solves all 64 full-ten minima and reports their stationary residuals, then
checks all `64*24` instances of (2). It does not solve one frame and copy it
isotropically to the other classes. The N5 numerical regularity observation
is not a theorem of N5 global quantum selection.

Let `B_c=sqrt(G_c)` be the positive matrix square root, and let the actual
selected frame be `Phi^c_mu=B_c[mu,nu] Gamma_nu`. The producer recovers
`Gamma=B_c^-1 Phi^c` from the dynamically selected frame and checks its full
Clifford anticommutators. This retains all mass/spatial and spatial/spatial
Gram cross terms. It is a change of body coordinates, not an additional field
constraint or a replacement energy law.

## 4. Deriving the SU4 compensator

For the recovered frame, `Gamma0` anticommutes with the three spatial Gamma
matrices, and `Gamma_a^2=I`. Thus the three bivectors

\[
 B_1=\Gamma_2\Gamma_3,\quad B_2=\Gamma_3\Gamma_1,\quad
 B_3=\Gamma_1\Gamma_2
\]

are traceless anti-Hermitian, satisfy `B_i^2=-I`, and
`B_i B_j=-epsilon_ijk B_k` for `i != j`. They commute with `Gamma0`.
Their commutators with the spatial Gamma matrices fix the vector generator:
for example `[Gamma1 Gamma2,Gamma1]=-2 Gamma2` and
`[Gamma1 Gamma2,Gamma2]=2 Gamma1`. There is no free generator normalization.

Write a physical rotation's unit-quaternion coordinates as `(w,q1,q2,q3)`.
They are extracted from **R**, with a deterministic sign convention. Define

\[
 S_R=wI+q_1B_1+q_2B_2+q_3B_3. \tag{3}
\]

The bivector identities imply unitarity, determinant one, and

\[
 S_R\Gamma_\mu S_R^\dagger=(R_4)_{\mu\nu}\Gamma_\nu. \tag{4}
\]

One can derive (3) without presupposing a spin action: put
`S=wI+q_i B_i` in `S Gamma_a S^dagger=R_ab Gamma_b`; the Clifford identities
produce the ordinary quaternion rotation polynomial and `w^2+|q|^2=1`.
Its two solutions differ by `-I`. Equation (4), rather than a half-angle rule
assigned to raw matter, is the defining assertion exercised in the producer.

The positive square root is covariant under `R4`, so (2) and (4) give

\[
 R_4\Phi^c=S_R\Phi^{c_R}S_R^\dagger. \tag{5}
\]

Restoring the physical fields to the target canonical body frame therefore
acts by `S_R^dagger` on a fundamental endpoint and by its complex conjugate on
an antifundamental reference. This restoration is a colour gauge operation
**following** the physical position/vector rotation, not its microscopic spin
law. With this row-index convention, `S_R S_T` lifts `TR`, whereas
`S_R^dagger S_T^dagger` lifts `RT`. The producer checks the latter order against
every group pair at every independently selected Gram class.

For a z-quarter-turn, the commutators above give
`S=(I+Gamma1 Gamma2)/sqrt(2)`. Its fourth power is `-I` because the square of
that bivector is `-I`. The coefficient is determined by (4); it is not added
by fiat. The 24 choices have a sign cocycle `epsilon(R,T) in {+1,-1}`:

\[
 S_R^\dagger S_T^\dagger=\epsilon(R,T)S_{RT}^\dagger.
\]

The all-pairs producer uses the actual target-centre frame for the second
restoration, verifies that the sign is independent of `c`, and checks the
cocycle associator for all `24^3` triples. This is the binary proper-cube
extension, not a claimed microscopic continuous Spin(3) symmetry.

## 5. The actual reversed-axis centre seam correction

The canonical centre connection has integer link exponent
`z_c(x,a)=c_a` if `x_a=N-1`, zero otherwise, and `U=i^z I`. Physical rotation
transports it by the exact orientation prescription of section 2 to `z_rot`.
In particular an axis reversal moves the seam to coordinate zero and negates
its exponent. It is generally **not** the canonical seam of `c_R`.

Solve the actual cochain equation

\[
 z_{c_R}(y,a)=k_R^c(y)+z_{\rm rot}(y,a)-k_R^c(y+e_a)\pmod4,
 \qquad k_R^c(0)=0. \tag{6}
\]

Both connections are flat and have the same torus holonomies, since physical
orientation transforms them by `c_R=Rc`. Hence their ratio is an exact Z4
cochain. Alternatively, the producer constructs it by primitive-edge traversal
and rejects any inconsistent cycle. The root fixes uniqueness on the connected
torus; all links, including the periodic links and reversed axes, are checked.
Set `kappa_R^c(y)=i^{k_R^c(y)}`. This belongs to SU4 because its determinant is
`kappa^4=1`, and has no action on adjoint Phi. The full local restoration is

\[
 K_R^c(y)=\kappa_R^c(y)S_R^\dagger,\qquad
 E_R^c=K_R^cP_R. \tag{7}
\]

Restoring physically rotated links by `K(y) U'(y,a) K(y+e_a)^dagger` now gives
the **actual canonical** target seam. The EM field remains physically rotated;
no compensating EM gauge or inverse reference seam is introduced.

Composition, for R after T, obeys

\[
 k_R^{Tc}(y)+k_T^c(R^{-1}y)=k_{RT}^c(y)\pmod4. \tag{8}
\]

Both sides solve the same actual seam equation, and both vanish at zero,
which every R fixes. Consequently there is no omitted residual constant Z4
factor in this rooted convention. Combining (7), (8) and the Clifford sign,

\[
 E_R^{Tc}E_T^c=\epsilon(R,T)E_{RT}^c. \tag{9}
\]

The producer checks all `64*24*24` seam and endpoint group compositions. At N5,
1161 of the 1536 rooted lifts have nonzero k somewhere. Omitting this cochain
still leaves an abstract spin-matrix cocycle, but it **fails the native
canonical-seam endpoint/current/contact covariance**, so is not the required
orbit intertwiner. The adverse control measures this failure, not just a
nonzero metadata flag.

## 6. Complete Gauss orbit, Fock and reference intertwiner

Do not restrict to centre-valued local gauge transformations. A point on the
complete colour orbit is

\[
 \Phi(x)=g_x\Phi^c g_x^\dagger,\quad
 U(x,a)=g_xU_c(x,a)g_{x+e_a}^\dagger,\quad g_x\in SU(4).
\]

After physical R and restoration (7), its new orbit coordinate is
`g'_y=S_R^dagger g_{R^-1y} S_R`; the centre cochain cancels in this adjoint
coordinate but **does not** cancel in the link seam restoration. In the
fundamental representation
`K_R(y) g_x = g'_y K_R(y)`. Thus `E_R` intertwines every local gauge element,
not merely one Cartan generator. Differentiating gives all 15 local colour
Gauss generators. In an orthonormal generator basis,

\[
 A_{ab}=2\operatorname{Tr}(T_a S_R^\dagger T_b S_R)
\]

is a real orthogonal 15-by-15 adjoint matrix. The producer checks its complete
basis reconstruction and orthogonality at all64-by24 frames.

The orbital/link pullback is unitary on the full flat-field/Haar-link domain.
Cotangent fields and left/right link electric generators are transported by
the same orthogonal/index/orientation transformation. The reference is moved
from `r` to `Rr` and transformed in `bar4`; holding its geometry fixed would
be a different, not here established, operation. For arbitrary gauge angles
`eta`, scalar light modes acquire `exp(i eta_x)` and the opposite EM reference
acquires `exp(-i eta_r)`. The original background is two light units per site.
Heavy matter remains EM neutral. The colour generator contains **both** heavy
and light matter and the reference; it is not a heavy-only physical sector.

Therefore the full constraints, in the source sign convention,

\[
 G_4(x)=L_\Phi(x)+\operatorname{div}E_4(x)-Q_H(x)-Q_L(x)-Q_{\rm ref}(x),
\]
\[
 G_1(x)=\operatorname{div}E_1(x)-(N_L(x)-2)-q_{\rm ref}(x)
\]

are mapped to their target-position counterparts (with the colour adjoint
matrix). Their common kernel is invariant. Background two is permuted, not
removed. The reference singlet contraction is invariant because
`(K tensor conjugate(K)) vec(I)=vec(I)`.

For any one-particle unitary E, the whole CAR implementation is the exterior
functor `Gamma(E)=direct_sum_n wedge^n E`. It satisfies, without occupation
cuts,

\[
 \Gamma(E)c^\dagger(f)\Gamma(E)^\dagger=c^\dagger(Ef),\quad
 \Gamma(E)d\Gamma(h)\Gamma(E)^\dagger=d\Gamma(EhE^\dagger).
\]

These identities follow on every antisymmetric n-particle tensor and therefore
on the **entire** fixed-NH and fixed-NL spaces, including non-Slater states.
Tensoring both species, the full orbital/link pullback and the moved dual
reference proves the complete Gauss orbit/Fock/reference intertwiner. This
is an exact Hilbert-space theorem, not a numerical simulation of an enormous
many-body wavefunction. The producer's complete local16 exterior-power and
actual fundamental/bar4 contraction checks are finite arithmetic controls of
these identities, not substitutes for the whole-space proof.

By (9), a cocycle sign acts on matter as
`epsilon^(NH+NL)` and on the one antifundamental reference as `epsilon`.
On the original charged sector `NH=2V`, `NL=2V+1`, their product is `+1`.
Specifically four physical quarter-turn restorations give

\[
 (-I)^{\wedge NH}=+I,\qquad (-I)^{\wedge NL}=-I,\qquad
 \overline{-I}=-I,\qquad (+I)(-I)(-I)=+I. \tag{10}
\]

The heavy contribution is an **ordinary** representation, not a claim of
zero heavy Berry connection, zero energy, no mean charge, or no charge
variance. Raw microscopic `P_R^4=I` and the actual physical fields and both
link systems also return. Equations (9)-(10) are therefore consistent with
that microscopic ordinary operation. Endpoint `-I` is a body/reference-relative
factor, not a negative sign for the whole physical closed operation.

A BO embedding can carry this exact group action only after the actual
normal-vacuum/auxiliary well subspace and the full-Gauss isometry are controlled.
Physical symmetry transports **well eigenspaces/projectors** between `c` and
`Rc`; it does not by itself prove that a selected scalar well vector has
trivial stabilizer character or justify fabricating wavefunctions. The parent
original-law BO construction supplies its own regularity/domain/normal-gap
premises. This producer neither modifies that theorem nor claims to simulate
its quantum full ground or validate it at N5.

## 7. Native controls and honest proof scope

`native_case` uses independent full-ten native source and target frames, all
4V native modes, actual canonical and physically rotated colour seams, and
independently sampled EM phases over the **entire compact circle**. It checks
(7) against native `h`, first Peierls derivative and second contact derivative.
The arbitrary directional tangent is physically transported with orientation.
It also checks arbitrary site-dependent noncentral SU4 orbit covariance and
the exact U1 current/contact commutator Ward identities, retaining background
and reference ownership in the theorem.

Two adverse constructions alter the same native consumer input, and reach
and fail the **identical** endpoint/current/contact covariance assertions:

1. Rotate positions and links but omit the spatial Phi vector indices.
2. Restore the body by S alone but omit the actual nontrivial reversed-axis
   seam cochain.

Cases whose cochain vanishes are reported as non-witnesses for the second
branch; no failed assertion is invented there. The default selected cases
include actual reversed-axis witnesses. Curvature is measured by all three
oriented compact plaquette phases, not by a linearized curl. Physical R
moves the fixed-position curvature pattern; a pure EM gauge leaves it
unchanged. This separates a physical operation from gauge restoration.

The separate pure-gauge/Berry control takes the actual full2V occupied native
sea frame. It applies an internal Clifford-compensator gauge path with **no**
site/vector-index rotation, then restores that gauge before computing polar
Berry overlaps. The restored full occupied frame is constant and its overlap
holonomy is `+I`. This does not say that bare Berry holonomy on an arbitrary
unrestored gauge path vanishes. It demonstrates exactly why a body-restored
internal path cannot supply the physical rotation claim. It is not the old
changed-law low-momentum packet.

### Executed design-discriminating observation

A scoped native-Python probe invoked all64 `gram_group_controls`, all twelve
default `native_case` constructions at each of `N=5, a=1/.001, seed222940`,
plus the complete-Fock/reference and restored-native-sea controls. It exited0.
It did not run a project test suite, formatter, linter, final type check or
full CLI. Full evidence is `artifact://1654`; the earlier one-case design
probe is `artifact://1653`. These are observations of the public APIs, not a
claim that the final receipt-writing CLI was exercised.

The final implementation SHA256 exercised was
`28c84681cb80503def00fece882b5daa1cc3de801ee430db2750e45f39fc7665`.
Entry/exit source/proof bindings were unchanged. This proof's observation
section was updated afterwards, so its final receipt binding must be refreshed
by the parent. Native Python was3.12.3, NumPy2.5.3, the project venv executable,
and all three declared BLAS/OpenMP thread variables were1.

Observed residuals at `a=1/.001`, respectively:

- all64 Gram covariance `4.5112e-16 / 2.9051e-16`;
- all64 native-frame covariance `1.0699e-14 / 7.5817e-15`;
- all36864 endpoint products `1.6012e-14 / 9.0851e-15`;
- exact integer rooted-seam and sign-associator residuals `0 / 0`;
- complete15 adjoint reconstruction/orthogonality `5.1657e-14 / 3.8014e-14`;
- maximum full10 stationarity `1.3323e-15 / 1.7716e-12`;
- maximum12-case endpoint/current/contact covariance
  `(2.3734e-14, 7.8997e-15, 1.3805e-14)` /
  `(1.7681e-14, 1.1037e-14, 2.0395e-14)`;
- the full noncentral local-SU4-orbit physical endpoint/current/contact
  covariance is at most `1.87698e-14` across both spacings;
- the fundamental quarter-turn fourth-power minus residual is `8.8818e-16`
  at both spacings; full sector parity is exactly `+1`;
- actual native250-mode restored Berry overlap loop plus residual
  `7.4314e-14 / 7.3281e-14`;
- five selected nontrivial-seam adverse witnesses per spacing.

For `c=(1,2,3)` and the z-quarter-turn, adverse defects at `a=1` are:

| Omitted ingredient | Endpoint | Current | Contact |
| --- | ---: | ---: | ---: |
| Phi spatial vector indices |2.38737398205|2.27797627854|3.86435864977|
| Rooted reversed-axis seam gauge |2.16705500995|1.99107361927|3.63727622597|

Every entry reaches and fails its matching covariance assertion with tolerance
`2e-8`. At `a=.001` all six assertions also fail (defects2.14 through4.96).
For that same actual phase sample, the physical fixed-position compact-
curvature change is27.34152380456 versus pure-gauge residual8.60015e-15.
The four raw physical turns in every selected case returned fields, both
link systems and scalar matter with zero observed residual.

These are native floating observations, not interval certificates or quantum
full-ground measurements. The producer includes source/proof hashes at entry
and exit and rejects changed bindings. Its final CLI and native typing are
parent-owned integration verification; no unobserved run is asserted here.
Before edits, native Pyright stdio symbols and a definition lookup exercised
the actual `physical_cubic` source, and the structural Python AST device matched
that construct. The worker had no mounted LSP device; native pull diagnostics
timed out after successful navigation. This gap was reported, not passed off
as a successful diagnostic result.

## 8. APIs, reproducibility, and remaining mechanical debt

Public construction APIs are `proper_cube_rotations`, `recover_gamma`,
`clifford_compensator`, `seam_cochain`, `make_lift`, and
`PhysicalRotationLift.onebody/fields/local`. `restore_links` restores actual
oriented links. `gram_group_controls`, `native_case`,
`whole_fock_reference_controls` and `restored_gauge_berry_control` are the
native discriminating consumers. `run` joins them without cuts. The immutable
lift records source/target centre labels, the actual site permutation, actual
rooted cochain and derived compensator.

From `proposals/P256-qed-backward`, the parent CLI is:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/administrator/substrate-framework/.venv/bin/python \
original_physical_rotation_lift.py --size 5 --spacing 1 \
--centres 000,100,123,222 --seed 222940 \
--output original_physical_rotation_lift_receipt.json
```

All64 Gram/group checks are unconditional parts of this invocation, irrespective
of the selected expensive native endpoint cases. The default four centres and
three rotations give 12 endpoint/current/contact cases. Source SHA256,
unchanged-entry/exit bindings, original bare parameters, native inputs, Python
executable/version, NumPy and thread environment are in the receipt. The
producer retains `.2/.12/.3` bare matter parameters and full selected Gram;
changing spacing is an explicit input, not a stealth photon/velocity retuning.

For **mechanical localized spin**, still missing are:

1. A controlled continuous mechanical generator with field and link momenta,
   colour boundary flux and an explicit decision about reference motion and
   ownership. The finite cube alone has no infinitesimal SO3 generator.
2. A genuine SO3 approximation and full-Gauss low-energy embedding with
   `W^dagger J_micro W -> L+S`, including angular field/link/reference terms,
   on localized mobile charged states rather than an internal band marker.
3. A deconfined finite-energy thermodynamic charged sector and controlled
   reference/boundary removal. The endpoint/reference-relative binary sign is
   not by itself a separated electron's interferometric `2pi` sign.

None of these premises is established by the all64 algebra or the native
current controls. Accordingly the proved deliverable stops at the exact
**physical binary-cube endpoint mechanism**, not a conditional Spin packet
relabelled as an original-law physical half-spin result. The joined electron-
photon campaign remains open.
