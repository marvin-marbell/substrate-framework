# Native massive-fermion locality and contact-completed vector-current functional

Governing plan `issue-222-continuing-backward-20260930`, assignment revision `e92bd5a83b413ce8f0b7d3c450ace4174256cc06404c647352a8f61c27b13926`; pinned general template `89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`, work item `campaign`, requirements `outcome/verification`. Fundamental original heavy/light CAR and gauge quantization are assumptions. This construction owns the actual continuous-time body-frame fermion operator, not the finite-dt regulator in `compact_dirac_current` and not a quantum compact-photon phase.

## 1. Exact original source and domain

Use the fixed full positive real Gram matrix, without replacing it by an isotropic one,

\[
 \Phi_\nu=(\sqrt G)_{\nu\rho}\Gamma_\rho,
 \qquad \{\Phi_\nu,\Phi_\rho\}=2G_{\nu\rho}I_4.
\]

The native onsite coefficient is `m+3r/a`. On a nearest-axis periodic graph `N>=5`, `a>0`, `m>0`, `r>=0`, finite `v`, the actual endpoint hopping is

\[
 F_j=-{r\Phi_0+i v\Phi_j\over2a},\qquad
 h_{x,x+e_j}=F_j U_{x,j}e^{iA_{x,j}},\qquad h_{x+e_j,x}=h_{x,x+e_j}^\dagger.
\]

Here the fixed retained central seams are included in the unitary shifts. The effective compact-U1 spatial paths may otherwise be arbitrary. The fixed-metric theorem does not permit independently varying noncentral colour links or a noncommuting temporal SU4 connection: those would invalidate the body-frame fixed-beta coercivity premise. `BulkQuantumLaw.onebody(fields, links, phases)` is reused for its original endpoint-average law only; its added selector energy is never evaluated. The actual full-G original supplier is `original_charged_principal.original_frame(..., native_toron=True)`.

The authoritative prior source/proof bindings at construction are:

| Artifact | SHA256 |
|---|---|
| `original_uniform_charge_gap.py` | `8483a19b8135e6f96e8f92f3f2c3c5ae51b4b024ee83a1b7f12b089feb7697d8` |
| `original_uniform_charge_gap.md` | `79e218d9275e6f2169def603acfcbfa8b6175cb6218b9bd6ae8085a25eeaf6d8` |
| `original_slow_kinetic_bo.py` | `8ca4f73a2c8f412a1dab42af9a81e44ef714742799f8bfcf398847f12a0fc8d1` |
| `original_slow_kinetic_bo.md` | `8dd35c0b3aef56cb521891b9d431c8497600f90201945f0e1e3da5a68916e658` |
| `original_charged_principal.py` | `95242f66759648031475b4a47be0cb3f316fc9dafd283d560cb2726d9c4ddc7c` |
| `bulk_quantum_selection.py` | `0dcdb45b755cc805186e72d1db70cc32bf392fa1a35c2e42d8dd99ff43b8f24f` |
| `original_compact_charge_bound.py` | `b980bfb62ed9fa8de18eea39c855cba2833c287118496dbb0a4f08a0c436ccc7` |

Every CLI result carries fresh complete local imported-source/companion-proof SHA256 bindings, including these two new artifacts, and rejects a change during its run. A proof cannot contain its own hash without changing it; its exact runtime identity belongs in that result. Source identity is not scientific verification.

Set

\[
 s=G_{00}-G_{0s}G_s^{-1}G_{s0}>0,\quad
 \beta={\Phi_0-(G_s^{-1}G_{s0})_j\Phi_j\over\sqrt s},\quad
 \mu=m\sqrt s.
\]

For arbitrary bounded measurable compact spatial paths and real commuting site temporal connection, on the anti-periodic continuous-time circle of period `T`, including arbitrary U1 site-unitary interface kicks,

\[
 \mathcal D=\partial_t-i a_0(t)+h(A(t)),\qquad
 \operatorname{Re}\langle u,\beta\mathcal D u\rangle\ge\mu\|u\|_2^2.
\]

This is the prior full-Gram Schur-dual theorem, not a simultaneous diagonalization of mass and odd parts. The temporal connection and admitted kicks commute with beta. No bound on `a0` or derivative of a Brownian spatial path is needed.

## 2. Joint spatial/time Combes–Thomas estimate

Let `d_X` be physical nearest-axis shortest torus distance (including the seam) and `d_T` shortest time-circle distance. Choose a periodic real time weight `f`, `|f'|<=gamma`, and a real site weight `w` such that every native edge satisfies `|w_x-w_y|<=alpha a`. Spatial multiplication acts identically on the four internal components. Define

\[
 C_h=2\sum_{j=1}^3\|F_j\|,\qquad
 \xi=C_h\operatorname{expm1}(\alpha a),\qquad
 \gamma+\xi<\mu.
\]

For one directed edge term, conjugation changes the unitary shift by a multiplier of magnitude at most `expm1(alpha a)`. Each forward and reverse shift has norm one. Thus on the entire spatial Hilbert space, with no dimension factor,

\[
 E_w=e^wh(A)e^{-w}-h(A),\qquad \|E_w\|\le\xi.
\]

Neither a row-count estimate proportional to volume nor an open-boundary path is used. Also

\[
 B=\sqrt{G_{00}}(m+6r/a)+{|v|\over a}\sum_{j=1}^3\sqrt{G_{jj}},
 \qquad\|h(A)\|\le B.
\]

Since beta has norm one, the combined conjugation obeys

\[
 \mathcal D_{f,w}=e^{f+w}\mathcal D e^{-f-w}
 =\mathcal D-f'+E_w,
\]

\[
 \operatorname{Re}\langle u,\beta\mathcal D_{f,w}u\rangle
 \ge(\mu-\gamma-\xi)\|u\|_2^2,
 \qquad
 \boxed{\|\mathcal D_{f,w}^{-1}\|_{2\to2}\le{1\over\mu-\gamma-\xi}.}
\]

The site weight commutes with temporal U1 kicks; a periodic time weight preserves their unitary domain and AP seam. The first-order finite-spatial circle operator has index zero by its finite-dimensional monodromy boundary problem. Coercivity excludes a homogeneous AP solution, hence existence follows as well as uniqueness. Conjugation alone, without this inverse/domain argument, would not establish the bound.

### Two evaluation intervals, not an L2-to-delta shortcut

Fix `delta>0`, `T>=8delta`, and put `k=B+gamma+xi`. On a source-free interval both the homogeneous evolution and the adjoint homogeneous evolution in the kernel's source variable satisfy

\[
 e^{-k|t-u|}\|v(u)\|\le\|v(t)\|\le e^{k|t-u|}\|v(u)\|.
\]

The anti-Hermitian temporal term cancels in the derivative of squared norm; the non-Hermitian spatial perturbation costs at most xi. Kicks are unitary and preserve this estimate.

For points at time-circle separation at least `2delta`, select a length-delta target interval and a disjoint length-delta source interval that do not cross the other's source. Target evaluation of the solution to an L2 source on the second interval costs `exp(k delta)/sqrt(delta)`. Apply the L2 inverse bound, then duality to obtain the kernel row as an L2 function of its source. Apply the adjoint homogeneous source evaluation estimate on that second interval. Together these two actual evaluation intervals give

\[
 \|K_{f,w}(t,s)\|\le{e^{2k\delta}\over\delta(\mu-\gamma-\xi)}.
\]

For separation below `2delta`, move the target to separation `2delta` on its **same side** of the source and propagate back without crossing the source jump. This costs at most `exp(2k delta)`. `T>=8delta` leaves the required intervals on both arcs. Both one-sided diagonal values are included. Therefore

\[
 C={\exp[4(B+\gamma+\xi)\delta]\over\delta(\mu-\gamma-\xi)}.
\]

Take `f(u)=gamma d_T(u,s)` and `w_x=alpha d_X(x,y)`; both satisfy their Lipschitz premises on the periodic graph. Undoing the scalar weights gives the **four-by-four site block** bound

\[
 \boxed{\|K_{xy}(t,s)\|\le C\exp[-\gamma d_T(t,s)-\alpha d_X(x,y)].}
\]

C is independent of volume, period for all `T>=8delta`, and the admitted background. For a fixed lattice spacing, a concrete admissible choice is

\[
 \gamma=\mu/4,\quad\xi=\mu/4,\quad
 \alpha={1\over a}\log(1+\mu/(4C_h)),\quad
 \|\mathcal D_{f,w}^{-1}\|\le2/\mu.
\]

This is uniform along a native large-N/all64 family only if its actual full metrics have a uniform positive Schur lower bound and uniform B/Ch upper bounds. The parent's original BO/interior/full-Gram transfer supplies that hypothesis where proved. Four N5 sampled Grams are not a proof of it. The constants need not be uniform as `a` tends to zero.

### Identification of the actual kernel

Let `P(t)` be the exact ordered evolution from zero, `U=P(T)` and `Q=(I+U)^-1`. On one cut of the circle,

\[
 K(t,s)=\begin{cases}P(t)QP(s)^{-1},&t>s,\\
 P(t)(Q-I)P(s)^{-1},&t<s.\end{cases}
\]

At equality use the corresponding one-sided formula. Thus

\[
 K(s^+,s)-K(s^-,s)=I,\qquad K(T,s)=-K(0,s).
\]

The identity term is a distributional source jump, not a delta evaluated in L2. `time_evolution` returns the actual ordered evolution/prefixes/background phase and `temporal_green` supplies the same Q. The CLI first invokes `green_kernel_controls`, then checks an interior source jump and arbitrary-source AP endpoint, **before** comparing any norms to C. A constant identity matrix can satisfy a loose inequality but is rejected by this same source-jump accept assertion.

## 3. Local connected logdet cycles and all local contacts

Use the physical background-normalized fermion functional

\[
 W[A,a_0]=\log Z_F[A,a_0],\qquad
 Z_F=e^{-2i\int_0^T\sum_x a_{0x}dt}\det(I+U)>0.
\]

The positivity and normalization are those of the exact continuous-time prior theorem, not a centred temporal Wilson determinant. Its phase and every time/interface holonomy must remain bound to the source. Let `V_i` denote local variations of D at their own spacetime points. For a link, the exact native kth jet is

\[
 (\partial_{A_e}^k h)_{xy}=i^k h_{xy},\quad
 (\partial_{A_e}^k h)_{yx}=(-i)^k h_{yx};
\]

it is supported on the two endpoints and has norm `||F_j||` for every k>=1. In particular the reverse **first** jet is `-i h_yx`, not `+i h_yx`, and the local second jet is `-h_xy/-h_yx`. A temporal first jet is `-i P_x`; higher temporal jets vanish. Source units must be declared before selecting a common vertex norm L. For example, for a dimensionless temporal source `b_x=a a0_x`, its first vertex norm is `1/a`.

For spatial variations `tr(delta h)=0`. The continuous equal-time prescription for first variations is

\[
 \delta W=\int dt\,\operatorname{tr}[K(t^+,t)\,\delta h(t)]
 -i\int dt\sum_x[\operatorname{tr}K_{xx}(t^+,t)-2]\delta a_{0x}(t).
\]

Equivalently use the midpoint `K_mid=(K^++K^-)/2=K^+-I/2` with `delta D`: each site has four modes, so its half-identity trace is exactly the background two. It is not permissible to omit that contact and relabel the unnormalized determinant as the physical functional. The spatial first formula is independent of the one-sided choice because its jet has zero trace.

For distinct insertions and a source linear in those insertions, the connected nth derivative for n>=2 is the sum over `(n-1)!` cyclic orders, anchored at insertion one, of

\[
 (-1)^{n-1}\operatorname{tr}[K(z_1,z_2)V_2K(z_2,z_3)V_3\cdots K(z_n,z_1)V_1].
\]

The kernel argument ordering is essential; replacing it by a product of equal-time resolvents is not the actual current functional. Let `S_i` be the one/two-site support of vertex i. A block between two such supports has norm at most `2C exp[-gamma d_T-alpha d_X(S_i,S_j)]`. Closing the trace costs rank at most eight. Consequently each cycle is bounded by

\[
 8(2C)^n\prod_i\|V_i\|
 \exp\{-\sum_{i=1}^n[\gamma d_T(t_i,t_{i+1})+\alpha d_X(S_i,S_{i+1})]\}.
\]

This is actual connected exponential clustering of the **background-conditioned fermion logdet** with no V/T prefactor. It does not assert connected clustering after integrating the photon field.

### Contacts are local partition terms, not separated cycles

For nonlinear native link phases, partition the n differentiated labels into nonempty blocks B. A block contributes the true coincident local jet `D_B`; it vanishes unless the labels in a nontrivial block concern the same edge and time. For a partition with ell blocks, sum its `(ell-1)!` cyclic orders, with sign `(-1)^(ell-1)`, of `Tr(K D_B1 ... K D_Bell)`. The one-block trace has the first-variation prescription above. Thus, explicitly,

\[
 W_{12}=\operatorname{Tr}(K D_{12})-\operatorname{Tr}(K D_1K D_2),
\]

\[
 W_{123}=\operatorname{Tr}(K D_{123})
 -\operatorname{Tr}(K D_{12}K D_3+K D_{13}K D_2+K D_{23}K D_1)
 +\operatorname{Tr}(K D_1K D_2K D_3+K D_1K D_3K D_2).
\]

The nontrivial-block terms carry local temporal delta distributions and edge-equality constraints. They cannot be discarded, fitted into a false separated-point exponential, or replaced by an alternate Wilson seagull. Distributional derivatives and time-order boundary values are defined by the exact operator evolution. After integration against smooth bounded sources their contact delta measures have unit total mass in each collapsed variable.

### Uniform anchored factorial summability, including contacts

Use an edge's initial site as its anchor, with its type one of the three positive-axis links or the temporal site current. An endpoint lies within a of its anchor. Write

\[
 q(z,z')=\gamma d_T(t,t')+\alpha d_X(x,x'),\qquad
 S_X(b)=\left[1+{2\over\operatorname{expm1}(b a)}\right]^3.
\]

Uniformly in periodic N and T,

\[
 \sum_y e^{-b d_X(x,y)}\le S_X(b),\qquad
 \int_0^T e^{-c d_T(t,s)}ds\le2/c.
\]

For distinct linear vertices of norms at most L, remove one edge of a cycle to integrate along a spanning path. The support-to-anchor replacement costs at most `exp(2n alpha a)`. This yields

\[
 \sup_{z_1,\tau_1}\sum_{\tau_2,\ldots,\tau_n}\int dz_2\cdots dz_n\,|W_n^{linear}|
 \le (n-1)!\,8(2CL e^{2\alpha a})^n
 [4S_X(\alpha)(2/\gamma)]^{n-1}.
\]

For an exponentially diameter-weighted norm note that a closed cycle has q-length at least twice the diameter of its anchors. Spend half the exponent on that diameter, and integrate the remaining half-rates. Define

\[
 A_*=2CL e^{2\alpha a},\qquad M_*=4S_X(\alpha/2)(4/\gamma).
\]

For the full contact-completed derivative measure `H_n=delta^n W`, n>=2, the native higher link jets have the same norm bound L. Coincident labels collapse to the same edge anchor and time, so they create no extra diameter. A partition into ell blocks gives at most `8 A_*^ell M_*^(ell-1)` times `(ell-1)!`. With Stirling number `S(n,ell)`,

\[
 S(n,\ell)(\ell-1)!\le n!{n-1\choose\ell-1}.
\]

This inequality follows by sorting each block and encoding an ordered partition as a permutation with ell-1 cuts. Summing the partitions proves the finite **contact measure total-variation** bound

\[
 \boxed{\sup_{z_1,\tau_1}\sum_{\tau_2,\ldots,\tau_n}\int dz_2\cdots dz_n\,
 e^{\operatorname{diam}_q(z_1,\ldots,z_n)}d|H_n|
 \le8n! A_*(1+A_*M_*)^{n-1}.}
\]

The anchored integral sums sites and integrates time; it does not count volume as an external prefactor. This is a genuine exponentially weighted infinitesimal current-functional bound, with all native contact partitions retained. Its constants can be very large. It supplies local analytic response control for suitably small bounded source changes; it does **not** supply a finite-amplitude, gauge-invariant compact monopole-polymer activity or its smallness. Integer flux defects and arbitrary compact excursions are not small differentiated sources. Such a stability theorem cannot be inferred by renaming this norm a monopole bound.

## 4. Exact vector Ward identity and the time-interface seam

For a real periodic site gauge chi, `g_x(t)=exp(i chi_x(t))`, the actual source law transforms as

\[
 A'_{xy}=A_{xy}+\chi_x-\chi_y,\quad
 a'_{0x}=a_{0x}+\dot\chi_x,\quad h'=g h g^\dagger,
 \quad \mathcal D'=g\mathcal Dg^\dagger.
\]

At a jump of chi the transformed solution satisfies

\[
 u'(t^+)=g(t^+)g(t^-)^\dagger u'(t^-).
\]

The closing time seam uses the same rule. It is wrong to take only the product of exponentials of conjugated segment Hamiltonians: those products have unrelated intermediate gauge frames. Every interface kick and the closing one are required. Once retained, `U'=g(0)Ug(0)^dagger` and

\[
 K'(t,s)=g(t)K(t,s)g(s)^\dagger
\]

with the same one-sided AP/source conventions. The original background-two phase also includes actual temporal kick angles. For a periodic piecewise gauge their sum telescopes, including the closing seam. Integer U1 winding changes that phase by an integer multiple of `4pi`, hence not its physical exponential. Smooth-gauge covariance and measurable unitary-interface covariance are the same operator statement, not two different regulators.

Infinitesimally the actual native vector vertex is

\[
 \sum_e(\chi_x-\chi_y)\partial_{A_e}h=i[\chi,h].
\]

Using the physical W, including its background contact, exact gauge invariance gives

\[
 0=\int dt\left[\sum_e(\chi_x-\chi_y){\delta W\over\delta A_e}
 +\sum_x\dot\chi_x{\delta W\over\delta a_{0x}}\right],
\]

or `div(delta W/delta A)-partial_t(delta W/delta a0)=0` in the distributional convention fixed above. Differentiating it preserves the native local contact terms and the source jump. A bubble alone generally violates it. There is no anomaly from a temporal doubling regulator because no temporal difference operator is introduced.

The independent eight-mode benchmark additionally uses a static native tree Hamiltonian with link phase .37 and zero temporal connection. Shifting that phase is a genuine vector-gauge change, so its exact first and second logdet derivatives vanish. If the native contact is removed, the second derivative becomes the finite-temperature Duhamel susceptibility of a nonconstant Hermitian current. The full Gibbs density is faithful and the native current is nonzero and not scalar, so that susceptibility is strictly positive: this is not an accidental random-background witness. This adverse input reaches the same Ward accept assertion as the contact-completed input. The separate whole-CAR correlator benchmark retains unequal segment Hamiltonians and commuting temporal connections.

## 5. Canonical/local Gauss meaning and what is not being proved

The compact quantum operator remains exactly

\[
 H=u_e\sum_e p_e^2+K\sum_p(1-\cos\operatorname{curl}_p A)+d\Gamma h_G(A),
 \qquad u_e>0,\ K>0,
\]

with original background two and whole neutral `NL=2V`. Its local Haar projector imposes

\[
 (\operatorname{div}p)_x=n_x-2
\]

and not a sitewise two-particle truncation. The constant-site subgroup projects the degree-4V determinant onto its `2V` exterior coefficient:

\[
 \int {d\theta\over2\pi}e^{-2iV\theta}\det(I+e^{i\theta}U)
 =\operatorname{Tr}_{\wedge^{2V}}U.
\]

The remaining local twists impose the remaining local Gauss constraints on actual compact heat-kernel paths. The background-normalized determinant is positive on each admitted twist by the prior theorem; the actual positive photon heat kernel and exact Hamiltonian Trotter limit own that path measure. No imported isotropic Euclidean Wilson action is substituted.

A Ward identity remains valid under the gauge-invariant projection. However, connected derivatives of the **log of a projected/integrated trace** are not the conditional cycles of section 3: differentiating the measure creates additional covariances of conditional first derivatives. In particular a dynamical massless photon can mediate long-range current correlations even when the fermion conditional kernel is exponentially local. Neither canonical projection nor positive integration allows those extra correlations to be silently dropped.

A charged insertion needs its actual opposite static reference and temporal/spatial Wilson factors; the sector then has `NL=2V+q`, total Gauss zero. This artifact does not show that a carrier cannot bind to that reference, nor that its spin is a localized total Noether/mechanical spin. It does not change the parent's BO transfer, original angular hypotheses or any of the all64 centre blocks.

If bare parameters are declared `u_e=c_e g^2/(2a)` and `K=c_e/(g^2 a)`, then `c_gamma^2=2u_e K a^2=c_e^2` is a bare algebraic matching. `UE1=2u_e` in the original Gaussian API; those coefficients must not be interchanged. This locality theorem neither predicts that matching nor supplies radiative charge/speed matching. It proves no photon masslessness, deconfinement, compact-photon thermodynamic phase, periodic quantum ground, UV limit or inclusive cross section.

## 6. Consumer, adverse assertions and integration command

`original_massive_photon_locality.py` provides:

- `locality_constants(band, gram, frame, delta=...)`: exact full-G formulas evaluated numerically, with a strict coercivity budget.
- `ContinuousGreen`: exact ordered piecewise-constant continuous-time background, interior evaluations, both source sides and arbitrary-source AP identification, reusing the prior time/Green interfaces.
- `native_link_jets(h,x,y)`: actual compact endpoint first/current and second/contact jets.
- `current_cycle(green,times,vertices)`: one correctly time-ordered oriented cycle; an nth connected response sums its cyclic orders.
- `run(...)`/CLI: by default N5 at a=1 and .001, four full-ten-coordinate native Gram cases `(0,0,0)`, `(1,0,2)`, `(2,2,2)`, `(3,1,0)` for each spacing. Every native matrix has 500 modes. The full fields, retained centre seam, compact spatial/temporal phases and generated site gauges have source/input bindings.

The native consumers identify temporal Green first, measure every native neighbour/seam weight, actual spatial conjugated-h perturbation and beta-coercive minimum, actual site-block Green values (including farthest sites), native seam finite-difference current/contact jets, a separated two-current cycle, actual spatial gauge covariance, every piecewise temporal interface/closing seam, logdet invariance, all-edge vector Ward and physical determinant normalization.

The adverse inputs are a constant identity Green, an open-path spatial weight across the actual periodic seam, the wrong reverse native current sign, omitted time-interface kicks, omitted background-two phase, and omitted contact in the independent second Ward derivative. Each is passed to the actual physical accept assertion and must be rejected; passing a large loose bound alone cannot count as identification.

For each spacing, the noncubic full-G case also runs an independent entire eight-mode/256-state exterior-algebra trace. All number sectors 0 through 8 are retained. `exterior_hamiltonian` enforces actual Hermiticity. Its independent continuous-time Fock evolution is compared to the determinant, its actual time-ordered CAR correlator to the Green matrix, its one-sided CAR contractions to the source jump, and its connected local-current trace to the conditional logdet cycle. An exact 17-node global Haar Fourier rule is compared both to the full neutral 70-dimensional trace and to the exterior coefficient. The unprojected grandcanonical value must genuinely differ. This validates the **global number component**, not the entire local compact-photon Haar integral. No periodic quantum photon ground is computed.

Owner's final science command (after all suppliers have landed):

```sh
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/administrator/substrate-framework/.venv/bin/python \
/home/administrator/substrate-222/proposals/P256-qed-backward/original_massive_photon_locality.py \
--size 5 --spacings 1 .001 --output /tmp/issue222-original-massive-locality.json
```

A targeted single-case invocation uses `--centre 1 0 2 --spacings 1`. The parent completed the eight-native500/two-whole256 integration; the static mass-homotopy consumer below also completed its four native cases. These executions identify the declared conditional operators and adverse inputs, not a photon-integrated phase. No implementation-copy or forwarding tests are added.

Before editing, real Pyright JSON-RPC was launched at `/home/administrator/substrate-222` with the science Python. It returned the prior `time_evolution`, `temporal_green`, `exterior_hamiltonian` document symbols, resolved the actual `time_evolution` call to its native definition, and published no diagnostics for that representative source. The mounted `xd://lsp` device was unavailable, so the real server protocol was exercised directly. A positive Python ast-grep query `expm($ARG)` matched the actual continuous-time generator call and the prior independent CAR controls. Those setup probes are not post-edit verification of this artifact.

## Appendix A. Exact mass density and finite-amplitude conditional Hessian

This appendix retains the fixed positive **full** G, native endpoints, Schur
beta, central seams and commuting temporal U1 source of section 1. It does not
extend the fixed-beta premise to arbitrary original Cartesian-Phi histories
or independent noncommuting temporal SU4 fields. The auxiliary mass below is
an identity tool; the physical Hamiltonian remains its M=0 value. It is not
a light-only counterterm or a common-heavy/light mass cutover.

### A.1. Density and exact large-mass normalization

Set `s=Schur(G)`, `mu=m sqrt(s)`, and

\[
 c=(m+3r/a)\sqrt s,\quad
 O=(m+3r/a)(\Phi_0-\sqrt s\,\beta),\quad
 d_0=\|O\|=(m+3r/a)\sqrt{G_{00}-s}.
\]

Then `{beta,O}=0`, `Tr beta=0`, and

\[
 h_M(A)=h(A)+M\beta=(M+c)\beta+V(A),\qquad M\ge0.
\]

The diagonal of V is O and its offsite blocks are the actual phase-dressed
native F_j, without modified endpoints or removed modes. Coercivity improves
to `Re(beta D_M)>=mu+M`. For the exact positive physical determinant,

\[
 W_M=\log\!\left[e^{-2i\int\sum_x a_{0,x}}\det(I+U_M)\right],
 \quad {dW_M\over dM}=\int dt\sum_x r_M(x,t),
 \quad r_M=\operatorname{Tr}_4[\beta K_M(x,t^+;x,t)].
\]

Both source sides give the same r_M because their jump is I and
`Tr beta=0`. A localized real mass variation proves that this local density
is real; site-U1 covariance proves its exact anchored gauge invariance.
The background-two factor is retained and is independent of M.

Write `q=M+c`, and use the free covariant operator
`D0_q=partial_t-i a0+q beta` with the **same** temporal connection and
interfaces. Its site holonomy theta_x gives beta source factors
`1/(1+exp(-qT+i theta_x))` and
`exp(-qT-i theta_x)/(1+exp(-qT-i theta_x))`, each with multiplicity two.
For T>=T0>0,

\[
 \|K^0_q(t,s)\|\le D(q)e^{-q d_T(t,s)},\quad
 D(q)={1\over1-e^{-qT_0}},\quad
 |r^0_q-2|\le {4e^{-qT}\over1-e^{-qT}}.
\]

The first Born density vanishes **at each anchor**:
`Tr_4 beta(K0_q V K0_q)(x,t;x,t)=0`. The free kernels force the
intermediate site to x; O is beta-odd while the free spin factors are
beta-diagonal. The same cancellation holds for the first determinant trace
`Tr K0_q V`, and for its source derivatives. Offsite first/contact jets
cannot survive a one-hop closed walk.

Choose the preceding locality budget `xi=Ch expm1(alpha a)`,
`gamma+xi<mu`, and define

\[
 v_\alpha=d_0+Ch\,e^{\alpha a},\quad
 \kappa(q)={2D(q)\over q-\gamma},\quad z(q)=\kappa(q)v_\alpha.
\]

For z<1 the positive closed-walk bound, with the zero first Born term
removed **before** taking absolute values, is

\[
 |r_M-r^0_q|\le {4D(q)z(q)^2\over1-z(q)}=O(M^{-2}).
\]

The finite mass interval uses the two-evaluation-interval point bound.
Consequently the gauge-invariant real anchored energy density

\[
 \ell_A(x,t)=-\int_0^\infty[r_M(A;x,t)-2]\,dM
\]

converges absolutely, uniformly in anchor and admitted backgrounds.
The subtraction is exactly two, not a reference determinant with unknown
holonomy. To identify its normalization, the large-q loop expansion starts
at n=2: K0_q V is Hilbert-Schmidt on the finite spatial/time circle, its
first trace is zero, and

\[
 {|W_M-W^0_q|\over VT}
 \le4D(q)v_\alpha\sum_{n\ge2}{z(q)^{n-1}\over n}=O(q^{-1}).
\]

Equality to the actual monodromy determinant follows by its lambda V
variation and the Green resolvent, with the vanishing first trace fixing
the constant. The free physical determinant is
`Z0_q=product_x[2 cosh(qT)+2 cos(theta_x)]^2`; hence

\[
 \lim_{M\to\infty}(W_M-2MVT)=2cVT,\qquad
 W_0(A,a_0)=2cVT+\int dt\sum_x\ell_A(x,t).
\]

Canonical or local-Haar projection has not been interchanged with this
conditional identity.

### A.2. Integrable source Hessian, with all native contacts

Spatial sources are dimensionless compact link angles. The temporal
coordinate is `b=a a0`, with vertex `-i P_x/a`. Spatial kth jets are the
actual `i^k h_xy` and `(-i)^k h_yx`; second jets remain contact measures
at one edge/time. Let H_M be the total variation of the two-source
derivative of r_M, weighted by the diameter of its mass anchor and both
source anchors in `Q=gamma d_T+alpha d_X`, summed over native source types.
For the convergent high-mass walks put `u(q)=kappa(q)/a`. Distributing two
spatial jets, mixed jets and two temporal insertions gives

\[
 H_M\le {4D\over(1-z)^3}
 \left[z^2(4-3z+z^2)+4u z^2(3-3z+z^2)+2u^2\right].
\]

The free temporal-temporal three-kernel term is deliberately included;
it is already O(q^-2). A false O(M^-1) spatial contact is not introduced
from the identically zero first Born term. Choose

\[
 q_{\rm tail}=\max(c,\log2/T_0,\gamma+8v_\alpha,\gamma+4/a),\quad
 M_{\rm tail}=q_{\rm tail}-c.
\]

Then D<=2, z<=1/2, u<=1, and the high-mass integral is bounded by
`(16384 v_alpha^2+2048/a^2)/(q_tail-gamma)`.
For the finite mass interval define

\[
 \begin{split}
 gap_0&=\mu-\gamma-\xi,\\
 \delta_{\rm low}&=\min(T_0/8,[4(B+M_{\rm tail}+\gamma+\xi)]^{-1}),\\
 \bar C&=e/(\delta_{\rm low}gap_0),\\
 P&=4[1+2/\operatorname{expm1}(\alpha a/2)]^3(4/\gamma),\\
 L&=\max(\max_j\|F_j\|,1/a),\\
 H_{\rm low}&=16(2\bar C)^3L^2e^{6\alpha a}P^2
             +8(2\bar C)^2Le^{4\alpha a}P.
 \end{split}
\]

Here B is the genuine operator bound from section 1. The first term
accounts for both oriented three-kernel cycles; the second accounts for
the native two-kernel contact. Thus

\[
 \|\operatorname{Hess}\ell\|_{\rm anchored,diameter\,TV}
 \le H_\ell=M_{\rm tail}H_{\rm low}
       +{16384v_\alpha^2+2048/a^2\over q_{\rm tail}-\gamma}<\infty.
\]

This fixed-a/full-G constant is independent of N, T>=T0 and all admitted
background amplitudes. Absolute domination permits differentiating the
mass integral and finite compact-geodesic source changes. It neither
differentiates Brownian paths nor supplies small monopole activity.
No evenness or vanishing first source variation is assumed for a generic
tilted G.

### A.3. A deterministic positive measure, not only a TV constant

Use the low-mass positive point-kernel envelope
`b(z,z')=Cbar exp[-Q(z,z')]` (with the prior source-side/support constants)
and the actual one/two-site positive incidence matrices of each jet.
At anchor z0 the two oriented Hessian envelopes are the rank-four
contractions of `b(z0,a)v_j(a,b)b(b,c)v_k(c,d)b(d,z0) ds du`.
Add the contact envelope `4 b(z0,a)v_jj(a,b)b(b,z0) delta(s-u) ds du`
with its edge/type equality. Integrate these positive measures over the
finite mass interval. Above M_tail use the positive free-kernel/native
walk envelopes of A.2, omitting the cancelled first Born diagram before
absolute values and retaining free temporal contacts.

Their mass integral defines one background-independent positive measure
nu_(N,T) on relative source offsets/types, including contact atoms:

\[
 |\operatorname{Hess}\ell_{z_0}|\le
       \operatorname{translate}_{z_0}\nu_{N,T},\qquad
 \int e^{\operatorname{diam}_Q(0,y,\zeta)}\,d\nu_{N,T}\le H_\ell.
\]

Coefficients depend on geometry and fixed a/G, never on compact phases,
temporal connection or placement of a central seam. Unit phases have
already been removed under norms. Tonelli applies before anchors are
summed. This is a family on the respective finite torus/circle, not one
measure asserted on every continuum space. Both positive marginals have
the same exponential moment.

### A.4. Ward consequence and the unavoidable seam limitation

The anchored density is gauge invariant, so its Hessian annihilates a
pure source gauge vector in either slot. For a smooth probe, a temporal
cut gauge followed by a spatial rooted-tree gauge has local residual

\[
 \eta_{\rm res}(y,t_0+u)=
   \sum_p n_{y,p}(\operatorname{curl}\eta_A)_p(t_0)
   +\int_0^u E_\eta(y,t_0+s)\,ds,\quad
 E_\eta=\partial_t\eta_A-d(\eta_b/a).
\]

If the associated harmonic/seam residuals vanish, a filling with area
at most C_area(1+r/a)^2 gives

\[
 |W''[\eta,\eta]|\le
 2C_{\rm area}^2 H_\ell P_4(\alpha a)
       \int dt\sum_p|\operatorname{curl}\eta_A|^2+
 {8H_\ell\over e^2\gamma^2}\int dt\sum_e|E_\eta|^2,
 \quad P_4=\sup_{r\ge0}(1+r/a)^4e^{-\alpha r}.
\]

Translate/sum anchors **before** Cauchy-Schwarz. Filling area is paid
twice, once in the filling and once by its translated multiplicity.
For the electric term, translated time integration costs u^2 and
`sup_u u^2 exp(-gamma|u|)=4/(e^2 gamma^2)`. Splitting the two residual
terms supplies the displayed factor two. All contact atoms remain.

On a finite torus/circle the exact residual is local plus seam.
Its positive-envelope seam quadratic and cross terms must also be
retained; separating them by Young changes the bulk constants.
A nonzero temporal-holonomy probe cannot be removed by a periodic gauge.
This estimate therefore does not erase the finite-temperature Debye
response or prove an unrestricted Maxwell-relative compact phase.

### A.5. Executed static consumer and precise scope

`original_mass_density.py --output <receipt.json>` runs the actual N5
full-ten-coordinate Gram/500-mode native matrices at a=1,.001 and centres
000,102. Its exact static spectral source is compared to the ordered
monodromy Green source and a mass finite difference. Eighty rational
mass-quadrature nodes identify finite-endpoint logdet differences.
The independent entire two-site/256-state exterior trace checks the
mass insertion in every number sector; no sitewise two-particle rule is
used. Identity mass in place of beta and zero density must reach and
fail the physical accept assertions.

For the same-holonomy static spatial comparison on N>=5, one-hop loops
are absent, two-hop phases cancel, and no three-hop closed nearest-axis
walk exists. The first possibly changing phase loop has four hops.
With M_cap>=max(8Ch,2/T), its integrated conservative bound is
`VT (8192/3) Ch^4/M_cap^3`. This is not used as a holonomy-changing bound.

The four-case CLI completed with all controls accepted; finite-endpoint
integral errors were below 4.81e-11. The receipt binds actual source bytes,
selected inputs and runtime. Numerical evidence is static and conditional:
it does not verify the infinite endpoint, density Hessian on Brownian
paths, photon/local-Haar integration, normalized current-fiber activity
or a massless phase. The infinite endpoint and Hessian above are analytic
claims with these explicitly narrower hypotheses.

## Appendix B. Actual compact-current baseline: energy is not activity

`original_compact_current_diagnostic.py` identifies the original affine
photon baseline on attained coarse-current charts. At physical block time
`delta=(a/ce)g^p`, the unchanged native coefficients are
`ue=ce g²/(2a)`, `K=ce/(ag²)`, hence
`beta_s=K delta=g^(p-2)` and `alpha_t=1/(2ue delta)=g^(-2-p)`.
Spatial b_ij are principal plaquette angles; temporal b_0j contain every
integer electric image. The signed temporal transport and spatial
endpoint shift use the same connection convention.

The compact current is the integer three-cochain
`q=db/(2pi)` with `dq=0`; its dual is J=*q. Its total homology is zero
because q is real-exact on the periodic complex. This does not remove
paired winding-loop decompositions or harmonic flux labels.

For arbitrary real affine plaquette lifts x,y,

\[
 I(x,y)=\int_0^1[1-\cos((1-s)x+sy)]\,ds
 =1-\cos((x+y)/2)\operatorname{sinc}((y-x)/2),
 \quad \operatorname{sinc}z={\sin z\over z},
\]

and the exact sharp bound is
`I(x,y)>=[(1-cos x)+(1-cos y)]/6`.
Put z=(y-x)/2 and c=(x+y)/2. After subtraction the expression is affine
in cos c, so check cos c=+/-1. For the plus sign one needs
`2+cos z>=3 sinc z`: on [0,pi], the derivative of
`z(2+cos z)-3 sin z` is
`2 sin(z/2)[2 sin(z/2)-z cos(z/2)]>=0`;
outside that interval use `3/|z|<1<=2+cos z`.
The minus sign needs `2-cos z+3 sinc z>=0`, immediate on [0,pi]
and at least `1-3/pi` afterwards. Evenness and the zero limit complete
the proof. Opposite small endpoints saturate 1/6.

After summing intervals, each spatial endpoint appears twice:

\[
 S_{\rm baseline}\ge
 {\beta_s\over3}\sum_{\rm spatial}(1-\cos b)
 +{\alpha_t\over2}\sum_{\rm temporal}b^2
 \ge{2\beta_s\over3\pi^2}\|b\|^2
 \ge{\beta_s\over6}\|J\|^2,\qquad0<g\le1.
\]

The second inequality uses principal spatial angles and alpha_t>=beta_s.
The last uses `||d_2||²<=16` on the four-dimensional unit complex and
`db=2pi q`; **db is not zero** merely because link lifts have a curl.
Principal branches and retained images contribute its integer defect.

There is a useful exact current-coordinate expression. With forward
cochain differences,
`q_0ij=(d0 b_ij-d_i b_0j+d_j b_0i)/(2pi)`.
Subtract the start plaquette's constant 2pi lift inside its cosine.
The actual affine curvature is then

\[
 b_{ij}(k)+s(d_i b_{0j}-d_j b_{0i})
 =(1-s)b_{ij}(k)+s[b_{ij}(k+1)-2\pi q_{0ij}(k)].
\]

This is a local identity on the **actual** compatible chart, not a
fractional Dirac-sheet interpolation between current sectors. It retains
full winding barriers: x=0,y=2pi l with nonzero integer l has I=1
although both endpoint Wilson potentials are zero. An endpoint-only
upper replacement would therefore change the original photon law.

The portable consumer executed three N5/four-time-slice attained charts
with integer-image scales1,7,1000 (maximum actual magnitude2000). It
identifies quantization, current closure, total homology, the exact
current-coordinate interpolation, and all displayed energy inequalities.
Its opposite-small-endpoint and full-winding inputs reject stronger false
endpoint bounds; the attained nonzero q rejects db=0.
These are selected chart computations, not a truncation of the model's
integer image sums or an all-input numerical certificate.

The missing phase step is still the **normalized** complete restricted
ratio `Z_B(b)/Z_B(0)`, including outside currents, the actual closed-CAR
factor, all harmonic labels and normalized bridge residual. The baseline
energy cannot be divided by a guessed Gaussian partition function.
Connected loop/dipole susceptibility is also required; local correction
stability or one-point occupation alone does not supply it.

## Appendix C. Actual normalized compact preparation, including nonstationary packets

`original_compact_soft_response.py` is a portable **continuous one-plaquette
comparator** for the following exact full-domain preparation theorem.
It does not numerically solve the original periodic field/CAR ground.
The original Hamiltonian is H=ue sum E_e^2+Hrest, E_e=-i partial_Ae;
every term of Hrest commutes with scalar compact-A multipliers.

For an actual normalized full-Gauss finite-energy density rho, let w(A)
be its photon marginal after tracing **all** Cartesian, SU4, heavy/light
CAR and reference fibres. This is a positive Hilbert-space density,
not a positive unintegrated determinant assumption. It retains every
toron, rotor sector and reference; integral w=1.

Complete real smooth periodic, gauge- and harmonic-translation-invariant
phase functions in

\[
 a_\lambda(S,S)=\int w(|\nabla S|^2+\lambda S^2),\quad\lambda>0.
\]

The restriction is on a preparation multiplier, not H or its states.
Finite photon kinetic energy gives sqrt(w) in H1 and weighted-gradient
closure at marginal zeros; no division by w or density floor is used.
Weighted approximation gives the compact unitary exp(iS) and preserves
the physical finite-energy form domain and both Gauss laws.

For a real transverse zero-winding edge target delta, Riesz gives

\[
 a_\lambda(u_\delta,S)=\int w\,\delta\cdot\nabla S,\qquad
 D_\delta+\lambda P_\delta=\delta\cdot M\delta,\quad
 M\delta=\int w\nabla u_\delta.
\]

Here D is the complete weighted gradient norm squared and P the
weighted phase norm squared. Thus 0<=M<=I and Ddelta<=norm(delta)^2.
Harmonic q0 lies in M's exact kernel because the phase functions are
toron-independent; the harmonic rotors themselves are retained.
Uniform w gives M=0. Compact admissibility alone is not a nonzero cloud.

For stationary rho the exact energy increase under exp(iu_delta) is
ue Ddelta. Matter current/contact and all coordinate/CAR multiplication
observables commute with the phase; the electric expectation changes
by Mdelta, not by a fractional shift of the integer rotor spectrum.

### Nonstationary packet: retain the actual linear electric term

For an arbitrary actual finite-energy rho define the quadratic-form
current functional

\[
 J_\rho(S)=\tfrac12\operatorname{Tr}\rho
                 \sum_e\{E_e,\partial_eS\},\qquad
 |J_\rho(S)|\le\sqrt{K_\rho D(S)},\quad
 K_\rho=\operatorname{Tr}\rho\sum_eE_e^2.
\]

The exact energy change for a scalar phase is ue[D(S)+2Jrho(S)].
It is false to discard J for a moving/nonstationary packet.
Define a second complete Riesz solution

\[
 a_\lambda(u_0,S)=-J_\rho(S)
\]

and prepare exp[i(u0+udelta)]rho exp[-i(u0+udelta)]. Its **original-to-
prepared** energy obeys

\[
 {\Delta E\over u_e}
 =D_\delta-D_0-2\lambda P_0-2\lambda\langle u_0,u_\delta\rangle_w
 \le D_\delta+\tfrac{\lambda}{2}P_\delta
 \le\lVert\delta\rVert^2.
\]

The square completion is exact. A potentially extensive K_rho controls
existence of u0 but does not enter this upper energy budget.
The corrected baseline rho0=exp(iu0)rho exp(-iu0) lowers the original
energy by ue(D0+2lambda P0). Relative to that baseline the soft increment
is instead ue[Ddelta-2lambda inner(u0,udelta)]; it is **not** automatically
bounded by ue norm(delta)^2 at finite lambda.

The original-to-prepared electric shift is mean(grad u0)+Mdelta.
Only the increment relative to rho0 is Mdelta. Equal-time actual matter
current and any independently established instantaneous matter velocity
are unchanged; the construction neither creates a velocity nor controls
subsequent scattering by commutation alone.

At each fixed graph, lambda P0 tends to zero without a weighted Poincare
gap. If B is the closed weighted gradient and A=B*B, boundedness of J on
the gradient norm represents it by a finite vector j. Polar decomposition
gives lambda P0 as the spectral integral of lambda x/(x+lambda)^2,
bounded by1/4 and tending pointwise to zero. Consequently a finite lambda
can make the corrected-baseline increment at most
ue(1+epsilon)norm(delta)^2. That baseline depends on lambda; no fixed
lambda0 phase, overlap or infinite-volume representation is inferred.

The native canonical budget remains

\[
 u_e\sum_e\delta_e^2=\tfrac12\sum_k|\delta\Pi_{\rm can}(k)|^2.
\]

Smooth real transverse 1/k profiles have a finite three-dimensional
energy budget. The complete generated UV gradient is included in it.
Finite energy is not a shell-uniform nonzero response, a coordinate-
quadrature cloud or a velocity-matched electron dressing.

### Actual low-energy coherence supplier and its infrared limit

The original positive-bare whole-heavy BO theorem supplies an
EM-independent full-Gauss lift W with residual dLambda and complete
complement lower GammaLambda. It retains all64 offsets. With
norm(Hlight,theta)<=B(a)V, the exact compact trial
psi_beta proportional to exp[-beta sum_p(1-cos bp)/2],
beta=sqrt(K/ue), has energy at most Cgamma V,
Cgamma=3sqrt(2)d/a. The full compact-Haar two-copy covariance inequality,
I1/I0>=1-1/beta and integration by parts establish this bound; no
oscillator/Fourier cutoff or Gaussian covariance is assumed.

For ANY actual density with energy at most its sector ground plus Delta,
put

\[
 L_\Delta=[C_\gamma+2B]V+\Delta+6d_\Lambda,\qquad
 \epsilon_\Delta={L_\Delta\over KV}
                  +{6L_\Delta\over\Gamma_\Lambda+BV}.
\]

The full block comparison and [WWdagger,U(A)]=0 give
sum_p mean(1-cos bp)<=epsilonDelta V on the actual microscopic marginal;
its complement is not discarded and no magnetic cross term is hidden.
This coherence result does not require stationarity.

For a desired margin epsilon_*<1/4 with epsilon0<epsilon_*, the allowed
excess is explicitly

\[
 \Delta\le {\epsilon_*-\epsilon_0
              \over 1/(KV)+6/(\Gamma_\Lambda+BV)}.
\]

Here epsilon0 is the supplier at Delta=0. Arbitrary excess still gives
the coherence inequality, but need not give a positive response.

For a finite nonzero real transverse plane mode with
Q(k)^2=sum_j[2sin(akj/2)]^2, epsilonDelta<1/4 and
lambda(3V)<=epsilonDelta,

\[
 {\delta\cdot M\delta\over\lVert\delta\rVert^2}
 \ge { (1-4\epsilon_\Delta)^2
          \over 2+193\epsilon_\Delta/Q(k)^2}.
\]

The source test is a compact sine sum, not the prepared phase; its full
UV gradient noise remains in the denominator. If epsilonDelta<=1/8 and
Q^2>=193epsilonDelta, the response fraction is at least1/12.
The lower bound degrades as Q tends to zero at fixed g. A response on
the total norm of a 1/k profile can hide zero deep-IR shells. A fixed
reference is not moved by an inferred translation average.
These positive-branch BO constants are not silently reused on the new
signed common-mass branch.

### Executed whole-circle comparator and reached adverses

The four-edge contractible cycle has the entire rotor operator
4ue(-partial_phi^2)+K(1-cos phi), not a truncated electric basis.
At actual native N5 centre000/a1 coefficients and g=.3, the normalized
Mathieu ground gives response fraction .9981610311; K0's uniform
ground gives exactly zero. The scalar phase has lambda_cycle=lambda/4
and delta_edge=.37, so its full target norm squared is4delta^2.

The comparator solves f=psi*u rather than dividing its differential
equation by exponentially small density. Inward log-ground integration
retains relative barrier accuracy without flooring. The physical phase
gradient uses the derivative of the **represented cubic f**, not the
independently collocated velocity: the latter left a periodic-integral
error about2.92e-6; the represented phase gives below1.58e-8.
Whole-circle ground and actual prepared-energy identities agree below
1.24e-14; an independent periodic integration-by-parts response agrees
below9.33e-15.

A real nonstationary input psi exp[i(.83 sin phi)] has energy variance
.00127316. Naively applying the stationary phase gives increase
.00978620, exceeding the actual target budget .00185590 and reaching
the physical rejection. The current-corrected original-to-prepared
change is -.00669028. The corrected-baseline increment is .00186367,
legitimately above that budget at finite lambda, demonstrating why
the two reference energies must not be confused.

These are continuous finite-comparator observations, not original
full-field/CAR ground samples, a mobile charged packet, shell-uniform
transport, mass renormalization, physical Noether spin or scattering.

## Appendix D. Neutral canonical/global-Haar freezing at a positive body frame

`original_canonical_floquet.py` exercises this exact **fixed-positive-G
body** interface. It applies to all compact photon paths, not to arbitrary
Cartesian Phi or nonaligned SU4 fields. The Schur Hermitian-unitary beta
gives {beta,h}/2>=mu I, mu=m sqrt(SchurG)>0; the actual generator is
-h+i diag(a0). FullPhi=0 would give h0 and a holonomy pi zero determinant,
so no unrestricted microscopic Schur-gap claim is possible.

For a Floquet eigenvector, y=wdagger beta w obeys
y'<=-2mu norm(w)^2. Temporal closure commuting with beta leaves y unchanged.
Consequently every eigenvalue satisfies

\[
 |\lambda|\le e^{-\mu T}\quad\hbox{or}\quad
 |\lambda|\ge e^{\mu T}.
\]

This direct estimate permits nonnormal evolution and arbitrarily
noncommuting real physical time slices. Actual J-pseudo-unitarity pairs
lambda_small=1/conjugate(lambda_large), so there are exactly2V in each
group, counting algebraic multiplicity. All4V modes remain.

For A0=integral dt sum_x a0_x, detU=exp(4iA0). The physical vacuum
factor is Zvac=exp(-2iA0)product_large(lambda)>0. For any real global
holonomy alpha,

\[
 Z_F(\alpha)=Z_{\rm vac}
       \prod_{\rm large}\left|1+{e^{-i\alpha}\over\lambda}\right|^2.
\]

The background-two factor exp[-i(2A0+2Valpha)] is essential.
It cancels the large-eigenvalue phase, not an arbitrarily selected
occupation at each site or momentum.

With q=exp(-muT),

\[
 (1-q)^{4V}Z_{\rm vac}\le Z_F(\alpha)
                       \le(1+q)^{4V}Z_{\rm vac}.
\]

The complete NEUTRAL sector NL=2V is the global-Haar average Zcan.
For any selected alpha*, Zcan/ZF(alpha*) lies between exp(-eta_can)
and exp(eta_can), eta_can=4V log[(1+q)/(1-q)].
Charged NL=2V+1 selects a nonzero Fourier coefficient; the constant
vacuum has zero coefficient there. There is no charged point-freeze
or carrier threshold from this neutral inequality.

On the actual compatible atlas a global holonomy shift preserves photon
action and physical current. Wrapping site phases requires compensating
temporal-image changes n'_e=n_e-(d_s k)_e; raw image-integer events are
not invariant. Applying the positive pointwise inequality before every
remaining **body photon** Haar/bridge/outside-current/toron/flux integral
controls the complete restricted ratio ZB(b)/ZB(0) by exp(±2eta_can)
when only this global phase is fixed rather than averaged.
Uniform positive native Schur bounds and T_N=Delta N^2 make eta_can
tend to zero on the declared exhaustion.

The full Cartesian/whole-heavy original model is connected by its
separate normalized thermal/bounded-photon-observable comparison on
the permitted graph-dependent finite Lambda_N trajectory. That is not
a finite-Lambda multiplicative bound on rare microscopic current fibres.
The body ratio here still has no monopole activity or connected dipole
susceptibility estimate; global-Haar freezing is not a phase proof.

The same actual affine action admits a conditional GOOD-current coarse
Hessian bound, with q0ij=0, principal spatial endpoints below pi/4,
fixed harmonic/topological data and the genuine bridge-force covariance
retained. Magnetic and matter variances must both be included.
Independent anchored Ward gauges act only on the two source slots;
Brascamp-Lieb still uses the original longitudinal/static-mean Dirichlet
noise domain. This conditional bound cannot declare the cosine model,
the offset/Haar mixture or the bad-current sector globally convex.

The default CLI executes native N5 centres000/102, all500 computed
Floquet modes, noncommuting durations3+5=T8 and inhomogeneous real a0.
Its501 equally spaced alpha values resolve the ENTIRE degree500 CAR
polynomial exactly; this is a number projection, not a Hamiltonian cut.
It separately checks every0..8 sector of two eight-mode trees,
all256 CAR states each. The source exterior helper accepts Hermitian
Hamiltonians: h and real a0 are lifted separately before forming -H+iA.

Observed direct physical complex-logdet errors are below6.83e-13,
reciprocal-pair errors below3.73e-14 and Fourier/elementary-sector errors
below2.09e-16. The no-background determinant reaches the same physical
assertion and is rejected. Finite-precision budgets condition the actual
factors(1+zlambda), vacuum and positive pair separately; no small mode
is dropped or assigned an imposed reciprocal value. The budgets
2.45e-5/6.37e-5 remain below the original1e-4 discrimination cap.
These are float64 finite-T diagnostics, not outward eigenvalue intervals
or numerical T-infinity/periodic quantum-ground evidence.

