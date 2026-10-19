# Frozen actual-bulk-mean compact U1 measure with one Dirac determinant

## Scope and input ownership

This is a complete finite conditional Euclidean measure and an actual dense
Metropolis consumer, **not** a claim that the original interacting campaign is
complete. The active shared plan is
`issue-222-continuing-backward-20260930`, revision
`c182aa82519fba2d52c8112f1b29cda63c055871502bb2d3177c1ea337331624`, with template
`general` revision
`89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`.

The caller must name both a bulk receipt and one of its actual branches.
`bulk_frame_compton.read_bulk` verifies the supplier SHA256 bindings, the
original seed bytes, immutable-input statements, and the explicitly selected
**validation** stationary irreducible rank-15 finite-mesh local minimum.
`reconstruct_frame` reconstructs amplitudes and the simultaneous unitary from
those coordinate matrices. It produces the actual mean-owned `RotationBand`
parameters, not the bare defaults. Its Clifford/isotropy/intertwiner tolerance
is part of the returned reconstruction: the raw matrices and their reduced
band agree to this supplied numerical tolerance, not by a newly claimed exact
floating-point identity. This work does not make the mesh-8 frozen coordinates
stationary, choose a global ground, or promote a mesh result to all momenta.

Conditions for this consumer and its Wilson-deletion controls are a positive mass and spatial Wilson coefficient,
positive velocity and spacing, unimproved nearest-axis hopping, and primitive
charge magnitude `e=1`. The scalar U1 links are genuine elements of U1. Spatial
fermion boundaries and all photon boundaries are periodic; the time fermion
boundary is antiperiodic. One four-orbital light Grassmann field is integrated.
There is no second light flavour, heavy determinant, rooting, absolute-value
replacement or determinant of `D†D`. The frozen colour frame is reconstructed
from the receipt; SU4 links, heavy orbitals and normal fields are not sampled.

The source owners read for this construction are:

- `bulk_frame_compton.read_bulk/reconstruct_frame/raw_paths` and
  `local_orbital_stability.scalar_paths`;
- executing `rotation_band.RotationBand.couplings/current/contact/energy`;
- `photon_matching.match_photon`, including `theta=a A` at `e=1`, physical
  spacing, `epsilon=1/UE1` and bare speed `a sqrt(UE1 K1)`;
- `quantum_su4_photon.CompactGaussPhoton` electric energy and light-only Peierls
  current/contact, and `quantum_su4_gauss.PhysicalHamiltonian` charge ownership;
- `full_band_gauss_section.peierls_action`, whose directed primitive bilinears
  use the same actual scalar path coefficients. Its two-cell, two-oriented-path
  average is not silently substituted for a 3D nearest-axis link.

## Euclidean Clifford matrices from the executing band

Write `beta=MIXING`, `alpha_i=HOPPING_i`. These are the existing matrix-unit
operators of the band, not a separately installed gamma representation. Define

\[
 \Gamma_0=\beta,\qquad \Gamma_i=-i\beta\alpha_i,
 \qquad\Gamma_5=\Gamma_0\Gamma_1\Gamma_2\Gamma_3.
\]

The executed band identities imply Hermitian `Gamma_mu`,
`{Gamma_mu,Gamma_nu}=2 delta_mu_nu`, Hermitian unitary `Gamma5`, and
`{Gamma5,Gamma_mu}=0`. The program exposes the residuals. In the raw mean frame
these matrices are conjugated by the reconstructed unitary. A constant change
of basis cannot change the determinant or create a flavour.

For a forward covariant shift

\[
 (S_\mu\psi)(x)=U_\mu(x)\psi(x+\hat\mu),
\]

include the antiperiodic minus sign only on the temporal fermion seam. Each
`S_mu` is unitary. Put

\[
 \begin{split}
 W_i&={r\over a}\left[I-{S_i+S_i^\dagger\over2}\right],\quad
 A_i={v\Gamma_i\over2a}(S_i-S_i^\dagger),\\
 W_0&={1\over dt}\left[I-{S_0+S_0^\dagger\over2}\right],\quad
 A_0={\Gamma_0\over2dt}(S_0-S_0^\dagger),\\
 D[U]&=mI+\sum_{\mu=0}^3(W_\mu+A_\mu).
 \end{split}
\]

The temporal Wilson coefficient is **`1/dt`**. This is an explicit added
Euclidean regulator. It is not an existing term of the finite continuous-time
microscopic Hamiltonian. The spatial Wilson term and spatial kinetic term are
exactly the mean-owned unimproved band law.

## Positivity, one flavour and gauge covariance

Every Wilson term is positive semidefinite because

\[
 I-(S+S^\dagger)/2=(I-S)^\dagger(I-S)/2.
\]

Each `A_mu` is anti-Hermitian. Consequently

\[
 {D+D^\dagger\over2}=mI+\sum W_\mu\ \ge mI>0,
 \qquad D^\dagger=\Gamma_5D\Gamma_5.
\]

For every nonzero right eigenvector, `Re(lambda)>=m`; thus `D` is nonsingular.
Gamma5 Hermiticity makes its determinant real. Along
`D_s=mI+s sum(W+A)`, `0<=s<=1`, it remains real and nonsingular and starts at
`det(mI)>0`. Therefore **`det D>0` for one flavour**. Neither a paired species nor
an absolute value is needed. The implementation calls complex `slogdet(D)` and
rejects any sign different from `+1`; it never changes that sign.

For any site phases `g(x)=exp(i chi(x))`, including fully time-dependent ones,

\[
 U'_\mu(x)=g(x)U_\mu(x)g(x+\hat\mu)^*,\qquad
 D[U']=G D[U]G^\dagger.
\]

The photon plaquettes are invariant under the same transformation. This is
local covariance of the actual combined measure, not just a free-momentum
Ward identity.

The adverse controls deliberately separate the proof's obligations. Deleting
gauge links fails covariance; deleting the positive mass permits a genuine
constant zero mode when a temporal holonomy cancels antiperiodicity; deleting
one forward directed term breaks gamma5 Hermiticity and exposes a complex
single-flavour sign on the nontrivial probe. Deleting spatial Wilson, temporal
Wilson or temporal Dirac terms changes the determinant weight. These last
three deletions need not destroy positivity: calling any positive alternative
law the required source law would be the wrong conclusion. Contact/vertex
controls below detect their own physical changes.

## Exact temporal pole and the continuous-time band

At free spatial momentum `p`, let

\[
 m_s=m+{r\over a}\sum_i(1-\cos(ap_i)),\qquad
 p_v^2=\sum_i\left[{v\sin(ap_i)\over a}\right]^2.
\]

The free Euclidean symbol is

\[
 D(\omega,p)=\left[m_s+{1-\cos(\omega dt)\over dt}\right]I
 +i\Gamma_0{\sin(\omega dt)\over dt}
 +i\sum_i\Gamma_i{v\sin(ap_i)\over a}.
\]

Continue `omega=i E_dt`. Its zero determinant is equivalent to

\[
 \left[m_s+{1-\cosh(E_{dt}dt)\over dt}\right]^2
 -{\sinh^2(E_{dt}dt)\over dt^2}+p_v^2=0.
\]

Using `cosh²-sinh²=1` gives the exact physical pole

\[
 \boxed{\cosh(E_{dt}dt)=
 { (1+m_sdt)^2+1+dt^2p_v^2\over2(1+m_sdt)}}.
\]

An equivalent numerically stable expression is

\[
 E_{dt}={2\over dt}\operatorname{asinh}
 \left({dt\sqrt{m_s^2+p_v^2}\over2\sqrt{1+m_sdt}}\right).
\]

Thus `E_dt -> sqrt(m_s²+p_v²)=RotationBand.energy(p)` as `dt->0`, with leading
error `-m_s E dt/2`. The code checks the exact equation, a zero singular value
of the independently continued 4x4 symbol, and four refinements at three
nontrivial momenta including a spatial corner. Deleting the temporal Wilson
term instead gives `asinh(dt E)/dt`; it has a different finite-dt pole and
reintroduces the temporal doubler. Deleting spatial Wilson changes even the
continuous-time target, including the corner masses. No finite-dt Lorentz or
radiative statement follows from this transfer.

## Spatial current and contact are the actual Peierls owners

For one actual raw forward source coupling `T_i`, the Hamiltonian reverse
coupling is `T_i†`. Multiplication on the left by the raw mean `beta_raw` gives

\[
 F_i=\beta_{raw}T_i,
 \qquad B_i=\beta_{raw}T_i^\dagger.
\]

In the reduced band these are

\[
 F_i={v\Gamma_i-rI\over2a},\qquad
 B_i={-v\Gamma_i-rI\over2a}.
\]

**The reverse Dirac coefficient is not `F_i†`.** Confusing these would delete
the anti-Hermitian Dirac difference. With a nonzero compact phase `theta`,

\[
 \partial_\theta D_i=i(e^{i\theta}F_i-e^{-i\theta}B_i)
 =\beta_{raw}\partial_\theta H_i,
\]

\[
 \partial_\theta^2D_i=-(e^{i\theta}F_i+e^{-i\theta}B_i)
 =\beta_{raw}\partial_\theta^2H_i.
\]

At link-midpoint momenta the charge-divided spatial current and the physical
contact supplied by `RotationBand` are

\[
 J_i=v\cos(ap_{mid,i})\alpha_i+r\sin(ap_{mid,i})\beta,
\]

\[
 C=e^2a\sum_i\epsilon_i\epsilon'_i
 [-v\sin(ap_{mid,i})\alpha_i+r\cos(ap_{mid,i})\beta].
\]

Since `theta=e a A` and `q=-e`, the Hamiltonian vector-source derivative is
`e J_i=-q J_i`. The code compares the raw `D` vertex/contact with `beta_raw`
times both the raw path Hamiltonian and the source API matrices at two
nonzero-transfer input pairs. It also uses independent finite differences of
the raw compact-link block at three nonzero phases. It reports the raw/reduced
frame tolerance; these are one-body coefficients of the primitive bilinears,
not a newly run 4900-state primitive calculation. Deleting the Wilson vertex,
the contact, or the required left beta multiplication is discriminated.

## Rotor normalization is anisotropic and Hamiltonian-owned

Use exactly the Hamiltonian rotor convention

\[
 H_\gamma={UE1\over2}\sum_\ell E_\ell^2
           -K1\sum_{p\;spatial}\cos\theta_p,
 \qquad E_\ell\in\mathbb Z.
\]

With Haar integration `dtheta/(2pi)`, the **exact** single-link electric kernel
in time is

\[
 K_t(\phi)=\sum_{n\in\mathbb Z}e^{-UE1\,dt\,n^2/2}e^{in\phi}
 =\sqrt{{2\pi\over UE1\,dt}}
   \sum_{k\in\mathbb Z}
   \exp[-(\phi+2\pi k)^2/(2UE1\,dt)].
\]

Temporal links implement the local Gauss projection and turn the electric
angle differences into temporal plaquettes. The finite step is the usual
Hamiltonian electric/magnetic product regulator; its compact photon weight is

\[
 w_\gamma[U]=\prod_{p\;temporal}K_t(\theta_p)
             \exp[K1\,dt\sum_{p\;spatial}\cos\theta_p].
\]

Thus the temporal Villain quadratic coefficient is **`1/(UE1 dt)`**, and the
spatial cosine coefficient is **`K1 dt`**. There is no isotropic guessed beta.
Haar normalization is retained, with a positive image sum evaluated using
`logsumexp`. The truncation uses a uniform relative tail bound: for retained
`|k|<=M`, put `d=2pi(M+1)-pi`; the omitted sum divided by a retained image is
bounded by

\[
 2\left(1+{t\over2\pi d}\right)
 \exp[(\pi^2-d^2)/(2t)],\qquad t=UE1\,dt.
\]

The numerical implementation chooses this bound below `1e-14`. It exposes the
bound and checks Haar normalization and the first Fourier coefficient
`exp(-UE1 dt/2)`, which is the actual integer electric gap. Deleting either
photon contribution or replacing the temporal heat kernel by an isotropic
cosine changes the nontrivial combined weight. In physical spatial units,
`theta=a A`, `epsilon=1/UE1`, and the bare small-angle speed is
`a sqrt(UE1 K1)`; no extra `a` is inserted into the dimensionless rotor energy.
The dressed photon pole and its physical external leg normalization are not
recomputed by, or equated to, this finite compact chain.

## Neutral charge and the exact finite-dt transfer determinant

The primitive owners use light charge `n_light(x)-2`, not `n_light(x)`, and
keep heavy orbitals electromagnetically neutral. A relativistic-looking
positive determinant does not by itself settle this convention. Here it can
be resolved for the **conditional four-orbital light model**.

Choose a beta eigenbasis, group the two positive and two negative orbitals at
each spatial site, and write the actual spatial Peierls Hamiltonian as

\[
 h_t=\begin{pmatrix}W_t&F_t\\F_t^\dagger&-W_t\end{pmatrix},
 \quad W_t=mI+W_{space,t}>0.
\]

`F_t` includes the full spatial kinetic hopping, including nontrivial U1
links. Set `B_t=I+dt W_t`, `C_t=dt F_t`. In temporal gauge, `dt D` has equations

\[
 B_tx_t+C_ty_t-x_{t-1}=0,\qquad
 -C_t^\dagger x_t+B_ty_t-y_{t+1}=0.
\]

Elimination gives the exact one-particle transfer

\[
 \binom{x_t}{y_{t+1}}=T_t\binom{x_{t-1}}{y_t},\qquad
 T_t=\begin{pmatrix}
 B_t^{-1}&-B_t^{-1}C_t\\
 -C_t^\dagger B_t^{-1}&B_t+C_t^\dagger B_t^{-1}C_t
 \end{pmatrix}>0,\quad\det T_t=1.
\]

Positivity follows by the positive `B_t^{-1}` block and positive Schur
complement `B_t`. No flavour has been doubled: the transfer acts on the same
four one-particle orbitals. Let the remaining site-dependent temporal link
holonomy be `Omega_x`; the canonical fugacity is its inverse,
`z_x=Omega_x*`. Let `Z` repeat each `z_x` on its four beta-basis orbitals, and
`P=T_(Nt-1)...T_0`. The antiperiodic seam and block elimination give

\[
 \boxed{\det D=dt^{-4V}\left(\prod_t\det B_t\right)
       \left(\prod_x z_x^{-2}\right)\det(I+ZP)}.
\]

The Fock trace of the one-particle product is `det(I+ZP)`. Its prefactor
`prod_x z_x^-2` is exactly the neutral background in
`Tr[prod_x z_x^(n_light(x)-2) ...]`. **It is already included in `det D`.**
Multiplying the measure by another background phase would double-count it;
omitting it when comparing with the canonical `N` trace would change the
charge convention. The code compares complex signs and real log magnitudes
on three uniform temporal holonomies with nontrivial static spatial links,
and independently on time-dependent spatial links with site-dependent final
holonomies. The wrong `N` rather than `N-2` trace and deletion of the finite
Wilson normalization are explicitly exposed.

For a static background, `H_dt=-log(T)/dt` is Hermitian but is **not** `h` at
finite `dt`. Expanding the exact transfer gives

\[
 T=I-dt h+O(dt^2),\qquad H_{dt}\longrightarrow h.
\]

Also `log(det B)/dt -> Tr W=2 Ns(m+3r/a)`, independent of spatial links for
these lattice sizes. The finite-dt determinant normalization is genuinely
link dependent and is retained. Only its continuous-time limit can be
absorbed into a gauge-independent energy zero. Time-ordered products converge
to the frozen light Hamiltonian evolution; the same local background
convention survives. Temporal photon links integrate the local Gauss
constraint, which on a periodic spatial torus implies total light number
`2 Ns`. That is a constraint on one light species, not the addition of another
species or a different normal ordering. The static nontrivial-link transfer
and its normalization have four explicit numerical refinements in the CLI.

**Precise unresolved joined prerequisite:** the identity between this
conditional U1 light measure and the **unfrozen primitive neutral SU4/Pauli
Gauss trace** has not been constructed. It would have to retain the primitive
heavy sector, Pauli-code/centre restrictions, dynamical colour links and all
normal fields, and derive their conditional integration measure/Jacobian and
any induced interactions. The exact neutral determinant reduction above
settles the background convention for the frozen four-orbital U1 light model;
it does not eliminate those missing physical degrees of freedom. Therefore
`full_primitive_neutral_measure_join_claim=false` and finite-dt microscopic
Hamiltonian identity is false. This is a reachable complete conditional
construction, not a joined Hamiltonian identity disguised by sign handling.

## The finite combined measure and actual consumer

The sampled probability density relative to independent link Haar measure is

\[
 d\mu[U]={1\over\mathcal Z}\left[\prod_\ell{d\theta_\ell\over2\pi}\right]
 \exp\left[\sum_{p\;temporal}\log K_t(\theta_p)
 +K1\,dt\sum_{p\;spatial}\cos\theta_p+\log\det D[U]\right].
\]

Finite compactness, a strictly positive continuous determinant and positive
heat kernels give a finite positive partition function. `joint_log` returns
this actual log weight and its separately named photon/fermion terms.
`metropolis` proposes symmetric wrapped link-angle changes and accepts using
the difference of this combined weight, with a fresh exact dense determinant
at each proposal. Rejections restore the old link and weight. Both hot and
cold starts run with recorded independent seeds and finish naturally after
exactly the requested finite number of proposals. The consumer returns
initial observations, every retained sample, acceptance statistics, final
angles/hashes, and a final independently recomputed retained weight.

Observations are temporal/spatial plaquette cosines, actual 1x2 Wilson loops
in all six planes, conserved integer DeGrand-Toussaint compact dual cube currents, and
integer fluxes through every periodic two-plane. On extent-2 directions a
1x2 loop wraps that direction; it is still the explicitly computed contour,
not a large-distance Wilson loop. Neither topology observations nor a small
chain establish thermodynamic confinement, a Coulomb phase, equilibrated
hot/cold agreement or a continuum topological charge. Raw correlated samples
are reported without fictitious independent-sample uncertainties.

Supported consumer sizes have `NT,NX,NY,NZ>=2`, volume at most64, dense Dirac
dimension at most256, at most20000 total two-chain proposals, and
`1e-4<=UE1 dt<=20`. The bounded temporal probe requires
`dt*(m+6r/a+3v/a)<=2`. No large-volume or thermodynamic extrapolation is built
into these bounds. One sweep defaults to `4*volume` independent random
single-link proposals (with replacement); `--updates-per-sweep` can explicitly
request a smaller bounded run. `burn` counts discarded sweeps and `samples`
counts retained sweeps, not individual independent configurations.

## Reproducible command handoff

From the repository root, an affordable full conditional consumer is:

```sh
/home/administrator/substrate-framework/.venv/bin/python \
  proposals/P256-qed-backward/compact_dirac_measure.py \
  --bulk-receipt proposals/P256-qed-backward/periodic_bulk_frame_receipt.json \
  --branch environmental_Clifford_L3 \
  --ue1 0.4 --k1 0.7 --dt 0.25 --lattice 2 2 2 2 \
  --seed 222 --burn 2 --samples 4 \
  --output /tmp/issue222-compact-dirac-measure.json
```

For the same full construction controls with a shorter consumer, add
`--updates-per-sweep 8`; this changes the chain length, not the action, input,
current, determinant or transfer checks. The caller must preserve the command
and its resulting receipt if it is to be cited. All scientific source hashes,
input receipt bytes and original seed bytes are checked unchanged at the end.
Output cannot overwrite a bound source/input. SHA256 source bindings include
this derivation, the implementation and the imported supplier owners.

This assignment deliberately performs **no scientific run, build, lint, test
suite or formatter**. Its native Pyright LSP/structural-AST probes are editor
intelligence, not evidence that the scientific controls pass. The integration
owner must run the supplied actual CLI, observe its natural exit and signed
receipt, and review the physical unit factors. No numerical result for this
new measure is asserted here before that execution. All scientific completion
flags beyond the conditional construction are explicitly false: no charged-B,
mechanical spin, normal-field, compact phase, radiative or Lorentz completion.

Observed non-scientific verification: native Pyright1.1.414 language-server
symbols and references on `RotationBand`, definition resolution from the new
module to that owner, and final new-file version2 diagnostics `[]`, rooted at
`/home/administrator/substrate-222`. Positive Python AST queries matched the
owner's `couplings` and the new `joint_log` and `neutral_transfer_controls`
functions; the mounted AST adapter did not expose its parser version.
The real CLI `--help` imported successfully and exited0 in0.92s with all required
receipt/branch/UE1/K1/dt/lattice/seed/burn/samples options. None of those probes
executed a determinant, scientific control or Metropolis chain.
