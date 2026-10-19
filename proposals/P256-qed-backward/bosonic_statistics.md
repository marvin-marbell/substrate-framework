# Primitive bosonic edge qubits → charged fermionic exchange

**Scope:** a quantum tensor-product qubit Hamiltonian, its derived charge hopping
algebra, and finite-sector dynamics. This supplies a statistics interface to the
same charged-band consumer; it supplies neither physical spin, a Dirac band,
Lorentz symmetry, nor a photon Coulomb phase. The complex quantum Hilbert-space
law and the selected Pauli couplings are fundamental inputs. The construction is
not a new discovery of the superfast encoding, and encoding an imported endpoint
Hamiltonian alone would not constitute a deeper physical root.

The executable `bosonic_statistics.py` starts from primitive Pauli matrices and
Kronecker products, **not** microscopic CAR. Its separate, labelled occupation
comparator is used only to measure the resulting transfer. `run()` reports exact
small matrices on a four-vertex star and square, coherent exchange pulses,
coherent many-body transfer, an odd-parity single-particle transfer, first and
second same-link phase derivatives, and the full finite-penalty extra sectors.
It also includes a charge-preserving microscopic loop-changing defect, for which
the low-energy dynamics are deliberately not an exact rewriting of a fermion
Hamiltonian. Numerical results require executing the report; the derivation below
specifies predictions rather than asserting an unexecuted measurement.

## 1. Primitive definition and all sign conventions

Let a connected simple graph have vertices `0,...,V-1`, `E` edges and bounded
maximum degree `d`. Put a two-dimensional **bosonic** Hilbert space on each edge:

\[
\mathcal H_b=\bigotimes_{e\in E}\mathbb C^2_e.
\]

Operators on different tensor factors commute. On a single factor,
`X Z = - Z X = -i Y`. Fix a local order of incident edges at each vertex.
The implementation sorts the undirected edges lexicographically and uses their
indices for every local order. Orient `A_ij` positively when `i<j`, so
`epsilon_ij=+1` for `i<j` and `epsilon_ji=-epsilon_ij`. This convention defines
an algebraic sign, not the electric-flux direction. Define

\[
B_i=\prod_{e\ni i}Z_e,\qquad
A_{ij}=\epsilon_{ij}X_{ij}
 \prod_{e<_i(ij)}Z_e\prod_{f<_j(ij)}Z_f,
\qquad n_i=\frac{I-B_i}{2}.
\tag{1}
\]

All Z tails exclude the edge carrying X. A repeated Z cancels. On a simple graph
an edge other than `ij` cannot be incident on both endpoints, but using symmetric
difference makes the support rule explicit. `B_i` has weight at most `d`, and
`A_ij` has weight at most `2d-1`; their support is the union of the two incident
edge stars. This is genuinely local on a bounded-degree graph.

These are chosen bosonic observables. We do not initially define a single-site
fermion annihilator or assume that qubit excitations already obey Fermi statistics.

### Deriving the algebra from Pauli support

Every A and B is a Hermitian involution. B's commute. A has exactly one X, so

\[
A_{ij}B_k=(-1)^{\delta_{ik}+\delta_{jk}}B_kA_{ij},
\qquad A_{ij}=-A_{ji}.
\tag{2}
\]

For different edges sharing vertex `i`, exactly **one** of the two local tails
contains the other's X edge: the later incident edge contains the earlier edge's
Z. Consequently there is one Pauli anticommutation, not zero or two. Disjoint
edges have no X/Z crossing even if their Z tails overlap. Thus

\[
A_{ij}A_{kl}=(-1)^{|\{i,j\}\cap\{k,l\}|}A_{kl}A_{ij}.
\tag{3}
\]

For the same edge the intersection size is two, as required for commutation.
No fermionic reference is needed for (1)–(3). Deleting the local Z tails removes
precisely the shared-vertex anticommutation; it changes the physical exchange,
not merely a convention.

## 2. Directed charge hopping and the sign that must not be copied blindly

Define a directed source `i` → target `j` operator

\[
T_{i\to j}=-i A_{ij}n_i(1-n_j)
 =-\frac{i}{4}A_{ij}(I-B_i)(I+B_j).
\tag{4}
\]

The rightmost projectors require an occupied source and empty target. By (2), A
flips exactly those two occupations. Direct multiplication gives

\[
T_{i\to j}^{\dagger}=T_{j\to i},\qquad
T_{i\to j}^{\dagger}T_{i\to j}=n_i(1-n_j),\qquad
T_{i\to j}T_{i\to j}^{\dagger}=n_j(1-n_i),
\tag{5}
\]

and `[n_i,T]=-T`, `[n_j,T]=+T`. Each directed legal hop has unit magnitude.
The Hermitian hopping combination is

\[
T_{i\to j}+T_{j\to i}=\frac{i}{2}A_{ij}(B_i-B_j).
\tag{6}
\]

A sign cross-check uses a **comparison**, not an implementation assumption:
if a reference writes `c_i=a_i+a_i†`,
`d_i=-i(a_i-a_i†)`, `B_i=-i c_i d_i=1-2a_i†a_i`, and
`A_ij=-i c_i c_j`, then

\[
a_i=c_i n_i,\quad a_j^{\dagger}=c_j(1-n_j),\quad
c_jc_i=-iA_{ij},\quad a_j^{\dagger}a_i=T_{i\to j}.
\]

This independently fixes (4) and (6). The displayed unnumbered hopping formula
in the HTML rendering of [Setia et al., *Superfast encodings for fermionic quantum
simulation*](https://arxiv.org/html/1810.05274) has the opposite sign for the stated
`A=-i c_i c_j` convention. We use the above derivation rather than transplanting
that displayed formula. Changing A's orientation sign requires changing all
associated formulas together; it does not justify mixing conventions.

Ordered pair creation is

\[
C_{ij}=-iA_{ij}(1-n_i)(1-n_j),
\]

which corresponds, in the separated comparison, to `a_j†a_i†`. It flips two
occupations and changes their total U(1) charge by `2q`; it is **not** included in
the charge-conserving Hamiltonian. It is useful for describing the parity algebra
and boundary strings, not for silently breaking charge conservation.

## 3. Exchange is a primitive bosonic calculation

Use a star with central vertex `0` and leaves `1,2,3`. Its edge ordering gives

\[
A_{01}=X_{01},\quad A_{02}=Z_{01}X_{02},\quad
A_{03}=Z_{01}Z_{02}X_{03}.
\]

Start with occupations `{1,2}`. The following six legal hops exchange the two
excitations and close the occupation path:

\[
1\to0,\ 0\to3,\ 2\to0,\ 0\to1,\ 3\to0,\ 0\to2.
\tag{7}
\]

Each star edge is traversed in both directions, so all scalar link phases and
orientation factors cancel. The ordered local Z tails leave a return amplitude
`-1`; undressed commuting-site hard-core boson hopping returns `+1`. Removing
those tails from the *edge-qubit* construction also returns `+1`. There is no
cycle stabilizer on a tree, so this exchange cannot be attributed to a hidden
loop-sector sign or a choice of reference occupation basis.

An analytic coherent-spectral prediction follows without using a CAR reference:
the star's six N=2 occupation configurations form a six-cycle in configuration
space. With uniform coefficient `-t`, zero on-site potential and zero link
phases, (7) makes the signed cycle anti-periodic. Its allowed momenta are
`(2k+1)pi/6`, so its spectrum is
`{-sqrt(3)t,-sqrt(3)t,0,0,+sqrt(3)t,+sqrt(3)t}`. Removing the tails makes the
cycle periodic, with momenta `2k pi/6` and spectrum
`{-2t,-t,-t,+t,+t,+2t}`. Thus the statistics changes coherent energies, not only
a labelled exchange history. The executable's nonuniform-coefficient/phase
probe is a stronger general transfer check rather than a fit to these special
uniform energies.

More generally, for distinct neighbors `j,k,l` of `i`, (3) and the moving
occupation projectors in (4) give

\[
T_{l\to i}T_{i\to k}T_{j\to i}
 =-T_{j\to i}T_{i\to k}T_{l\to i}.
\tag{8}
\]

Both sides act on the same initial occupations `{j,l}` and final occupations
`{i,k}`; all other occupation patterns vanish on both sides. Reversing the three
shared-vertex A's entails three anticommutations and hence one minus sign. This
is the hopping-algebra statistics criterion derived in
[Levin and Wen, *Fermions, strings, and gauge fields in lattice spin
models*](https://arxiv.org/pdf/cond-mat/0302460.pdf), Eq. (4). The executable report
checks this specific algebra and the closed path (7); the literature supplies a
lead, not a substitute for the present transfer.

`exchange_report` additionally applies actual coherent pulse unitaries

\[
U_{ij}=\exp[+i(\pi/2)(T_{i\to j}+T_{j\to i})]
\]

at the six successive edges. On each legal occupation the pulse supplies the
same factor `i` times its hop. Removing the common `i^6` pulse phase leaves the
same `-1`; the return probability is one. This exercises dynamics, rather than
merely inspecting a formal sign identity.

## 4. Loop identities, code dimension, and extra sectors

For a simple oriented cycle `gamma=(v_0,...,v_{L-1})`, define

\[
S_\gamma=i^L A_{v_0v_1}A_{v_1v_2}\cdots A_{v_{L-1}v_0}.
\tag{9}
\]

Reversal of its A product gives `(-1)^L`; complex conjugation of `i^L` supplies
the same sign. Thus S is Hermitian and squares to I. Each endpoint occurs twice
around the cycle, so S commutes with every B. Counting shared endpoints using
(3) also gives commutation with every A, hence with every hop. All S's commute.
They are **not** equal to I on the full bosonic Hilbert space.

Choose a spanning tree and its `r=E-V+1` chord cycles. Each cycle has its own
chord X support, so no nonempty combination can be a scalar or `-I`; these are
independent stabilizers. A cycle decomposition generates all remaining loop
identities. Consequently, for every sign vector `sigma in {+1,-1}^r`,

\[
P_\sigma=\prod_{\ell=1}^r\frac{I+\sigma_\ell S_\ell}{2},\qquad
\operatorname{rank}P_\sigma=2^{E-r}=2^{V-1},\qquad
\sum_\sigma P_\sigma=I.
\tag{10}
\]

In the desired `sigma=+1` code, every cycle product in (9) is I. For comparison,
substituting `A_ij=-i c_i c_j` in a cycle makes adjacent c's cancel and gives
`i^L(-i)^L=1`, reproducing exactly this identity. This comparison is restricted
to the code; applying it to all `2^E` qubit states is false.

Since each edge Z appears twice,

\[
\prod_iB_i=I,\qquad(-1)^{\sum_i n_i}=+1.
\tag{11}
\]

The rank-one joint occupation/loop projectors identify every allowed even
occupation pattern with one code state. Their independence follows from the
`V-1` independent incidence Z checks plus the `r` independent cycle checks,
which together have rank E. Directed hopping and ordered pair operations connect
all even configurations; projectors and these paths generate the matrix units.
The algebra is therefore the complete even-parity charge algebra, not just a
match of one-particle energies.

For an open path with L edges,

\[
A^{\rm path}_{v_0v_L}=i^{L-1}A_{v_0v_1}\cdots A_{v_{L-1}v_L}.
\tag{12}
\]

Contractible path choices agree in the `S=+1` sector by (9). In other sectors they
differ by the loop's Z2 sign. This is the explicit string/boundary content; a
single charged creation operator is not a local tensor-factor observable.

On the square `0→1→2→3→0`, the implementation's edges are
`(01,03,12,23)` and

\[
S=-X_{01}Y_{03}Y_{12}X_{23}.
\]

The four-step closed two-excitation exchange `0→1,2→3,1→2,3→0` returns `-1` in
`S=+1` but `+1` in `S=-1`. The latter is a loop-flux contamination, not a global
claim that those excitations lose their local star exchange statistics. It
explicitly shows why a reference transfer that omits the sector restriction is
wrong. Arbitrary scalar flux would multiply both by its Wilson phase; the star
exchange isolates the statistics independently of that flux.

## 5. Odd sectors and the single-electron consumer

A closed graph with (11) cannot contain a single excitation. Pretending that an
empty plus one-electron state lies in its even code would invalidate the transfer.
Two explicit remedies have distinct meanings:

1. **Fixed boundary parity background.** Set `B_b→-B_b` at a selected boundary
   vertex b, leaving A and S unchanged. The local algebra (2)–(3) is unchanged,
   but `prod B=-I`. The code now has the full odd-parity charge algebra, again
   dimension `2^(V-1)`. This is `BosonicEdgeModel(..., parity=-1,
   boundary_vertex=b)`. `run()` exercises its N=1 coherent transfer on the square.
   The changed boundary background is declared input, not a dynamically derived
   isolated-electron creation law. It can be understood as a fixed charge beyond
   the represented region. The graph must have an actual boundary for that
   interpretation; algebraically the sign twist also exists on a finite graph.
2. **Explicit spectator/boundary mode.** Add a boundary vertex r and an edge
   connecting it to the graph. Keep total parity even and fix `n_r=1` for a bulk
   odd sector; bulk hops commute with this condition. A single bulk electron then
   has an explicit occupied spectator. If charge injection is required, include
   its source and the boundary transfer as well, and preserve total charge and
   parity. One cannot inject a sole electron from an impossible odd-total vacuum:
   an initially occupied source plus spectator, for example, has even total
   number before and after a source→bulk transfer. Longer boundary transfers use
   the strings in (12), not a local bare single-particle creation operator.

For a filled negative band the total microscopic occupation parity must similarly
be fixed, with a declared boundary sector if needed. Normal-ordering the charge
does not alter this global parity accounting.

### Local even spin generators and the central parity identification

The rotation-band supplier selects four orbitals per spatial site in order
`(t,s)=(0,0),(0,1),(1,0),(1,1)`, with
`C_a=diag(1,-1) tensor sigma_a` and `B_band=sigma_1 tensor I`.
Its candidate internal generators are

\[
j_a=-\frac{i}{2}C_bC_c=\frac12 I_2\otimes\sigma_a
\quad ((a,b,c)\ \text{cyclic}),\qquad
[j_a,j_b]=i\epsilon_{abc}j_c.
\tag{12a}
\]

Those matrices are **supplied by that band construction**, not silently
postulated as physical spin by this statistics supplier. Their relevance to
physical rotations additionally requires the band relation
`D(R) h(p) D(R)†=h(Rp)` and the combined field action
`psi'(x)=D(R)psi(R^-1 x)`, with total generator `L+j`. At finite lattice spacing
the supplier only has cubic rotations; continuous covariance is an asymptotic
claim for the parent to test.

For the BKSF lift include the six internal edges among four orbitals at a site
(zero Hamiltonian coefficients on unused edges are allowed). Define local
**even** generators entirely from (4):

\[
\mathcal J_a(x)=
 \sum_\alpha(j_a)_{\alpha\alpha}n_{x\alpha}
 \sum_{\alpha\ne\beta}(j_a)_{\beta\alpha}
       T_{(x,\alpha)\to(x,\beta)}.
\tag{12b}
\]

Each term has bounded local edge-star support and commutes with every loop
stabilizer. `quadratic(j_a, modes=site_orbitals)` implements precisely (12b);
`bilinear(target,source,path)` also exposes the explicit string for a nonedge.
Within the desired code the complete even algebra established above gives

\[
[\mathcal Q(h),\mathcal Q(k)]P=\mathcal Q([h,k])P,
\quad
[\mathcal J_a,\mathcal J_b]P=i\epsilon_{abc}\mathcal J_cP.
\tag{12c}
\]

One direct justification uses the rank-one occupation projectors: each off-
diagonal matrix unit transfers one occupation with (4), its two orders differ
only when they share a mode, and (2)–(3), (9) supply exactly the intermediate-mode
Kronecker terms. Thus the single-particle Lie algebra lifts to the many-body
even algebra, not to a tensor product of independent spinors for each occupied
site without its fermionic signs.

Since `exp(-2pi i j_a)=-I_4`, diagonalize j_a (eigenvalues +1/2,+1/2,-1/2,-1/2).
An N_x-orbital exterior state accumulates the product of its N_x single-orbital
signs. The exact central relation on the code is consequently

\[
e^{-2\pi i\mathcal J_a(x)}P=(-1)^{N_x}P
 =\left(\prod_{\alpha=0}^3 B_{x\alpha}\right)P.
\tag{12d}
\]

This is central for on-site even observables, but not for a hop crossing the
site boundary. For a global rotation over all four-orbital sites, (11) gives

\[
e^{-2\pi i\sum_x\mathcal J_a(x)}P=
 \left(\prod_{x,\alpha}B_{x\alpha}\right)P=P
\tag{12e}
\]

in the closed even sector. Rotating only one separated **odd-occupation**
wavepacket region relative to its odd partner gives -1; rotating both gives
(-1)(-1)=+1. A region whose wavepacket straddles the cut is not a parity
eigenstate, so an unconditional scalar -1 would be false there. Orbital angular
momentum of an ordinary spatial packet supplies an integer 2pi factor; its
half-spin sign is the internal factor only after the physical covariance
requirement has been met.

After the negative-band doublet is actually supplied and filled, hole creation
is `b_alpha†=a_negative,alpha`. Its commutator with the quadratic spin generator
gives the hole matrix `j_h=-j^T`, which still satisfies the SU(2) algebra,
`j_h^2=3I/4` after summing axes, and `exp(-2pi i j_h)=-I`. It is equivalent to the
fundamental doublet by the SU(2) pseudoreality transformation `i sigma_2`. Removing
a negative-band particle gives charge `-q`; adding a positive-band particle
gives charge q. Filling both spin partners gives `det D(R)=1`. Thus a particle
addition and hole removal each change **relative** parity by one, while their
charge-anticharge pair preserves even global parity.

The constant global filled-sea parity must be retained. An entangled filled band
need not be an eigenstate of a sharply cut real-space region's microscopic
occupation parity. Its packet-relative half-spin sign requires controlled
quasiparticle modes or comparison to the matched vacuum's rotation response;
it is not an unconditional assertion that `prod_region B=-1` on that sea.
A one-orbital static spectator or the odd boundary sign twist above only fixes
the encoding's parity: **neither supplies the opposite partner's physical
half-spin rotation law**. Likewise, (12d) alone does not establish microscopic
SO(3) symmetry of the bosonic Pauli Hamiltonian, nor physical spin-statistics for
arbitrary internal matrices. These are code-level rotation-transfer identities
conditional on the actual band supplier, its filled state, and spatial covariance.

## 6. Same-link charge, source, current, and contact derivatives

For a directed edge define a complex coefficient w and link phase theta:

\[
H_{ij}(\theta)=w e^{i\theta}T_{i\to j}
 +w^*e^{-i\theta}T_{j\to i},\qquad
\theta=+qa A_{ij}^{\rm gauge}.
\tag{13}
\]

The gauge symbol in theta is a physical vector potential, **not** the statistics
operator `A_ij` in (1). The forward phase is `U†` for
`U=exp[-iqa A_gauge]`, matching the existing consumer's link convention. With
`R=exp[i sum_i chi_i n_i]`,

\[
R T_{i\to j}R^\dagger=e^{i(\chi_j-\chi_i)}T_{i\to j},\qquad
R H(\theta)R^\dagger=H(\theta+\chi_j-\chi_i).
\tag{14}
\]

All scalar derivatives remain in the same algebra:

\[
\partial_\theta^s H=w i^s e^{i\theta}T
 +w^*(-i)^s e^{-i\theta}T^\dagger.
\tag{15}
\]

The directed number current, electrical current, and vector-potential source are

\[
I_{i\to j}=\dot n_j=-\partial_\theta H,
\quad\dot n_i=-I_{i\to j},\quad
J_{i\to j}=q I_{i\to j},\quad
\frac{\partial H}{\partial A_{ij}^{\rm gauge}}=-aJ_{i\to j}.
\tag{16}
\]

Scalar coupling is `q A_0(i)n_i`. The sign in (16) follows directly from
`dot n=i[H,n]`, not a convention silently chosen after differentiating. The
second derivative in (15) supplies the same-link contact term. Since the code
isometry V does not depend on theta, `H_b(theta)V=V H_f(theta)` implies
`(partial_theta^s H_b)V=V(partial_theta^s H_f)` for **every** s. The report measures
this explicitly for s=1,2, besides the full complex-phase coherent dynamics.

For quantum link coupling take an independent rotor `[E,U†]=-U†`. Then

\[
\mathcal T_{i\to j}=T_{i\to j}\otimes U_{ij}^{\dagger},\qquad
G_i=\operatorname{div}E_i-n_i+b_i.
\tag{17}
\]

The source's outgoing E and its n both decrease by one; the target's incoming E
and n changes cancel with the reversed divergence sign. Thus `[G_i,Tcal]=0` at
both ends. Loop projectors commute with the matter part and the independent
rotor. The bare matter hop without the link shift fails this commutator. The
small report uses an explicitly **nonwrapping** E=-2,...,2 lower shift: its
commutator is exact, but it is not unitary at the endpoints and is not proof of a
compact rotor's infinite-Hilbert dynamics or of a photon phase. Charge q, rotor
quantum law and electromagnetic coupling remain inputs at this scope.

The statistical loop signs in (9) are a **separate Z2 sector**, whereas (17)
uses an independent electromagnetic U(1) link. This construction does not
identify statistical defects with Maxwell magnetic monopoles, or infer physical
spin from a charge-monopole field-angular-momentum formula. Neither the Z2
constraint nor its gap fixes the U(1) photon normalization or fine-structure
constant. It therefore does not inherit, evade, or solve a same-photon dyon's
electric/magnetic normalization problem by relabelling its field.

### Arbitrary bounded-degree charged spinor graph

Flatten modes `(x,alpha)` into graph vertices, with an edge whenever the chosen
single-particle matrix has a nonzero off-diagonal coefficient. Include on-site
component mixing as internal edges. For a fixed number s of components and z
spatial neighbors, degree is bounded by at most `s(z+1)-1`. Map every coefficient
`h_(y beta),(x alpha)` to (13) with the exact same spatial link `U_xy†`, independent
of alpha,beta; internal on-site edges carry U=I. Density is `q sum_alpha n_(x alpha)`.
Equations (1)–(17) therefore supply a local map for arbitrary complex charged
spinor hopping matrices, same-link current vertices and contact derivatives,
with no replacement of matrix entries by a Dirac guess. On-site diagonal and
density-density terms are polynomials in n and map directly. This statement
proves the **statistics/coupling algebra** only; it does not generate the
component matrix, its physical rotation law, mass, or photon dynamics.

Bounded degree does **not** make every fundamental cycle short. On an arbitrary
bounded-degree graph a spanning-tree chord cycle can be arbitrarily long. The
hop map is local regardless, but a Hamiltonian penalizing all those cycles need
not be uniformly local. On cell complexes with bounded plaquettes, local
plaquette penalties enforce contractible loops; noncontractible cycles remain
topological/boundary sectors unless prepared or separately fixed. Our finite
square penalty is short and local. We do not silently upgrade this finite proof
to a fully local all-topology code preparation theorem.

## 7. Actual finite bosonic Hamiltonian and its non-encoding corrections

The unprojected microscopic Hamiltonian is

\[
H_b=\sum_{ij}H_{ij}(\theta_{ij})+\sum_i\mu_i n_i
 +\Delta\sum_\ell\frac{I-S_\ell}{2}.
\tag{18}
\]

It acts on the entire `2^E` tensor-product space, not on a postselected matrix.
Every extra sign sector in (10) exists with its own matter dynamics and energy
shift `Delta * number_of_negative_cycles`. With no loop-mixing terms, (18)
commutes with every S: exact code transfer is then exact for **finite** Delta,
provided the state is in the selected sector. Energetic selection of the entire
code band additionally requires sector separation; a sufficient bound is
`Delta>2||H_matter||`. A large penalty does not project an arbitrary initially
contaminated pure state or thermal state into the code by unitary evolution.
The report prints full spectra of every sector, their union, and the separation
of the undesired-sector bottom from the complete desired-sector top.

To distinguish a real microscopic model from an exact encoding, add a local,
charge-conserving primitive defect on square edge `(0,3)`:

\[
D=X_{03}\,[n_0(1-n_3)+n_3(1-n_0)],\qquad
H_{b,g}=H_b+gD.
\tag{19}
\]

The bracket commutes with X, so D is Hermitian. `[N,D]=0`, but `{S,D}=0`, since S
contains `Y_03`. Thus `P D P=0`, `D^2=n_0(1-n_3)+n_3(1-n_0)` and D connects
physical matter occupations to a genuine peripheral loop sector. It can itself
be gauged with the same rotor if local Gauss invariance is required. No new CAR
law occurs in this perturbation.

For `|g|,||H_matter|| << Delta`, the leading low-energy operator is

\[
H_{\rm eff}=P H_{\rm matter}P
 -\frac{g^2}{\Delta}P D^2 P
 +O(g^2\|H_{\rm matter}\|/\Delta^2,g^4/\Delta^3).
\tag{20}
\]

The correction is an occupation-dependent bond interaction, **not** just a
universal energy shift. In the zero-hopping two-level special case, an initial
code state with `D^2=1` has exact peripheral leakage

\[
p_{\rm leak}(t)=\frac{4g^2}{\Delta^2+4g^2}
 \sin^2\!\left[\tfrac12\sqrt{\Delta^2+4g^2}\,t\right].
\tag{21}
\]

With the actual hopping present, the report evolves the full microscopic matrix
at `Delta=12,24,48`, `g=0.2`, `t=1.1`, observes leakage, compares its lowest code-
dimension eigenvalues to exact-code energies, and compares to (20). These are
finite microscopic deviations, not an exact equality claimed after projecting
away the defect. Suppression requires a controlled penalty hierarchy and sector
preparation; there is no claim of robustness for arbitrary perturbations.

A second adversary replaces every dressed A in the hopping by its undressed
oriented X. On the star this immediately changes exchange to `+1`, even though
occupation parity still works. On the square the resulting Hamiltonian can mix
loop sectors and its projected desired-code Hamiltonian differs from (18).
Penalizing loops alone cannot restore the missing local shared-vertex algebra.
The report measures both that difference and finite-penalty leakage.

## 8. Executable interfaces and remaining physical obligations

`BosonicEdgeModel(vertices, edges, parity=1, boundary_vertex=0)` exposes:

- `A_words`, `B_words`: primitive Pauli support and boundary signs;
- `A(i,j)`, `B[i]`, `n(i)`, `hop(i,j)`, `pair_creation(i,j)`, `path_A(path)`;
- `bilinear(target,source,path)`, `quadratic(matrix,modes)`, `region_parity(modes)`
  for the conditional local even many-body rotation lift;
- `cycles`, `stabilizers`, `sector_projector(signs)`, `occupation_projector(...)`,
  `state(...)`, `occupation_basis(number,signs)`;
- `loop_penalty(Delta)`, `hamiltonian(coefficients, phases, onsite, penalty)`;
- `gauged_hop`, `quantum_gauged_hop`, `link_derivative(...,order)`,
  `number_current`, and the explicitly adversarial `bare_X_transfer`.

Matrices are dense and exponentially sized, deliberately suitable for exact
small probes. `reference_alignment` is **comparison-only** and must not be used
as the microscopic constructor. It aligns the arbitrary phases of joint Pauli
projector states to a separated occupation comparator, then reports an actual
intertwiner residual. `coherent_report` compares full state evolution, not merely
spectra, and verifies the same-link first/second derivatives. Direct CLI:

```sh
/home/administrator/substrate-framework/.venv/bin/python proposals/P256-qed-backward/bosonic_statistics.py
```

The returned JSON includes all explicit inputs, primitive algebra residuals,
exchange amplitudes, full/coherent spectra and dynamics, negative controls,
finite-defect deviations, and separate acceptance booleans. It imports no
existing supplier to hide a fundamental charge-statistics assumption.

**Surviving debt:** why this particular local Pauli Hamiltonian and its hierarchy
are realized; a physically massive spin-1/2 charged band; stable photon phase;
relativistic rotation/spacetime behavior; parameter matching and the joined
Compton response. This slice materially supplies bosonic quantum dynamics with
fermionic charge exchange and a controlled exact-code/finite-defect distinction.
It does not close those remaining electron–photon obligations.
