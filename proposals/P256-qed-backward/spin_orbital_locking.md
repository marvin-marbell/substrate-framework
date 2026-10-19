# Physical spin from a selected bosonic orbital phase

This **conditional explanatory theorem** strengthens `rotation_band.md`: the
half-angle action is the uniquely derived connected **diagonal** rotation
subgroup of an ordered scalar/vector orbital field, not merely internal band
covariance. It combines spatial and compensating flavour actions. The complete
CLI has executed the stabilizer, cubic lifts, actual Pauli states and current.
The ordered phase remains a hypothesis; this does not assert quantum spontaneous
symmetry breaking.

## Hypotheses, owner, and units

1. Fundamental complex Hilbert-space quantum mechanics is assumed. The
   microscopic orbital variables are four unconstrained traceless Hermitian
   bosonic matrices, hence 60 real fields, and microscopic edge variables are
   ordinary bosonic qubits. Neither a fundamental spinor nor CAR matrices are
   inputs to this calculation.
2. Before ordering, microscopic four-flavour matter is a **spatial scalar**.
   There is an exact global flavour SU(4). The derived primitive charged operator
   from the edge-Pauli/neutral-marker construction transforms in its fundamental
   representation. This flavour symmetry and its matter representation are
   hypotheses of the microscopic law, not consequences of the Clifford energy
   inequality. The orbital matrices transform in the adjoint.
3. Under spatial rotations, \(A_0\) is a scalar and \((A_1,A_2,A_3)\) a vector.
   Isotropic scalar hopping/gradient coefficients, the quartic trace-Gram
   potential, and the same matter hopping respect the proper cubic group on the
   lattice. Their leading longwave continuum terms respect SO(3). There is no
   exact continuous coordinate rotation of a finite cubic lattice.
4. The uniform phase selected by `soft_budget.py` is locally stable, as in the
   local orbital theorem. Its frame obeys
   \[
   \{A_\mu,A_\nu\}=2G_{\mu\nu}I_4,\qquad
   G=\operatorname{diag}(g_0,g_s,g_s,g_s),\qquad g_0,g_s>0.
   \]
   Uniform selection and local stability do not prove a thermodynamic quantum
   ordered phase. All physical locking conclusions here are conditional on that
   phase being realized.

Use \(\hbar=1\). The lattice spacing \(a\) is a length, \(m\) and \(r/a\)
are energies, \(v\) is a velocity, and \(e\) a charge. Gram scales are
dimensionless. The CLI obtains the actual soft-selected SPD Gram matrix, rather
than fixing a hard orbital budget. It uses the same improved band coefficients,
with \(m_{\rm eff}=m\sqrt{g_0}\), \(r_{\rm eff}=r\sqrt{g_0}\), and
\(v_{\rm eff}=v\sqrt{g_s}\). Mass, charge, and the fundamental quantum law remain
inputs, not measured-electron predictions.

## Why the microscopic symmetry is present

Write \(G_{\mu\nu}=\tfrac14\operatorname{tr}(A_\mu A_\nu)\). A representative
orbital potential is
\[
 V=\frac\kappa2\|G-bI_4\|_F^2.
\]
SU(4) conjugation leaves every trace invariant. A physical vector rotation acts
as \(G\mapsto\operatorname{diag}(1,R)G\operatorname{diag}(1,R)^T\), so this
potential is invariant too. A scalar-coefficient gradient term
\[
 \frac J2\sum_{\mu=0}^3\sum_{k=1}^3
 \frac14\operatorname{tr}(\partial_kA_\mu\,\partial_kA_\mu)
\]
is invariant under global flavour conjugation and simultaneous rotations of
coordinates and vector components. Its lattice nearest-neighbour analogue is
exactly cubic, with the stated continuum symmetry. More general coefficients may
distinguish the scalar from the vector triplet without distinguishing vector
axes. The leading matter term \(\psi^\dagger(mA_0-ivA_i\partial_i)\psi\) has
these symmetries with spatially scalar \(\psi\), transforming fundamentally only
under flavour. The Wilson scalar correction is cubic and becomes irrelevant to
continuous covariance at longwave momenta. For the uniform field, the occupied
half-filled sea respects global flavour conjugation and the cubic momentum
permutations. These facts connect the selected energy law to the physical
symmetry being locked; a trace-Gram identity alone would not establish that link.

## Solve the stabilizer without supplying SU(2)

Choose fifteen traceless Hermitian matrix units \(H_a\), not spatial spin
matrices, and expand \(T=\sum_at_aH_a\). Let \(\Omega\) be a real
antisymmetric 3 by 3 matrix. There are **18 real unknowns**. Solve the real and
imaginary entries of
\[
 i[T,A_0]=0,\qquad i[T,A_i]+\Omega_{ij}A_j=0.
 \tag{1}
\]
Our sign convention is \(\Omega=-[\boldsymbol\theta]_\times\). The active
spatial rotation is \(R=\exp([\boldsymbol\theta]_\times)\) and the fundamental
flavour action \(U=\exp(-i\theta_kJ_k)\). Then the order transforms as
\(A'_i=R_{ij}UA_jU^\dagger\); differentiating this action gives exactly (1).
Also \(Uh(p)U^\dagger=h(Rp)\), fixing the physical rotation convention.

The CLI constructs the entire linear map in exact rational/complex arithmetic.
For the canonical selected Clifford representative it derives a rank-15 map,
with a three-dimensional kernel whose projection onto the three spatial
unknowns is invertible. Multiplying that kernel by the inverse projection
normalizes each spatial axis and **produces** \(J_x,J_y,J_z\). No SU(2)
generator is supplied to the solve. Positive scalar/vector Gram factors and
simultaneous flavour conjugation leave this rank and normalization unchanged.
The CLI also solves all coefficients afresh for an actual soft-selected scaled
frame conjugated by a generic SU(4) basis transformation.

Here is the analytic uniqueness argument behind the finite linear solve. Put
\(B_0=A_0/\sqrt{g_0}\), \(B_i=A_i/\sqrt{g_s}\). These four Clifford matrices
generate all \(M_4(\mathbb C)\). If two internal generators lift the same
\(\Omega\), their difference commutes with every \(B_\mu\). The commutant is
scalar, and tracelessness makes that difference zero. Thus there is at most one
lift for each spatial infinitesimal rotation. Solving (1) yields, equivalently,
\[
 J_k=-\frac i2B_iB_j\quad((i,j,k)\text{ cyclic}).
 \tag{2}
\]
Equation (2) is verified *after* solving, not inserted as a definition of
physical spin. It supplies a lift for every axis, proving the kernel is exactly
three-dimensional. The solved generators obey
\[
 [J_i,J_j]=i\epsilon_{ijk}J_k,\qquad
 \{J_i,J_j\}=\tfrac12\delta_{ij}I_4,\qquad
 \sum_iJ_i^2=\tfrac34I_4.
\]
Their axial eigenvalues are \(+1/2,-1/2\), each twice. On the fundamental
charged field the connected stabilizer is therefore two spin-half doublets,
with \(U(2\pi)=-I_4\) and \(U(4\pi)=I_4\). Its projection onto physical SO(3)
is the double cover Spin(3). The connected unbroken group is unique, not a
choice of arbitrary orbital basis.

For precision, the full adjoint-order stabilizer also contains the SU(4) centre
\(\{I,iI,-I,-iI\}\). The connected diagonal Spin(3) includes \(\pm I\), but
not \(\pm iI\); the latter add disconnected components. “Unique connected
stabilizer” does not mean the entire stabilizer has no extra central elements.

## Physical rotation is not a basis change or a Berry phase

A general flavour conjugation sends one ordered frame to another. It is a
microscopic symmetry but normally moves the vacuum along its orbital orbit.
A physical rotation of the selected phase must instead combine the coordinate
rotation with its **unique** compensating flavour action from (1), preserving
that phase. Choosing a display basis \(Q\) sends all fields, generators, band
states and currents through the same \(Q\), but is not itself a physical
rotation. The CLI uses a generic non-spin SU(4) \(Q\) to check this distinction
at the consumer.

The combined path ends at \((I_{\rm space},-I_{\rm flavour})\) after \(2\pi\).
Pure microscopic spatial \(2\pi\), acting on the declared spatial scalars and
bosonic vector fields without that compensator, ends at
\((I_{\rm space},I_{\rm flavour})\) and remains \(+1\). The diagonal Noether
generator is \(K=J_{\rm space}+Q_{\rm flavour}\), including orbital-field
contributions; in the fixed uniform background its matter action is \(L+J\).
The primitive rest-state consumer below exponentiates the flavour contribution,
and the current consumer checks the combined covariance. Neither is a
mechanically implemented rigid-space interferometer. The theorem identifies
the conditional unbroken symmetry, not an unmeasured rotation apparatus.

In particular, a positive rest projector can be **constant** under the entire
rotation path: the solved \(J_i\) commute with \(A_0\). Choose a constant
orthonormal frame \(C\) for that projector. Its Berry connection
\(iC^\dagger dC\) is zero and its Berry holonomy is the identity. Nevertheless
the combined symmetry acts inside the same constant subspace as
\(C^\dagger U(2\pi)C=-I_2\). There is no contradiction: these are different
operations. A rotating choice of basis without the endpoint sewing correction
would mislabel a gauge phase as Berry transport. **No physical minus sign is
inferred here from adiabatically rotating a constant positive projector.** The
minus sign follows from the actual unbroken symmetry acting on charged states.
The CLI emits this counterexample explicitly.

## Exact lattice group: 24 rotations and 48 inherited spin lifts

Enumerate all proper signed-permutation matrices: there are 24, the proper
cubic group. Exponentiate the **solved** generators for quarter turns around
three axes. Since the derived anticommutators are known, use
\[
 U(q)=q_0I_4-2i\,q_iJ_i,
 \qquad q_0^2+|\mathbf q|^2=1
\]
with the three generators \(q=(1,\mathbf e_i)/\sqrt2\). The quaternion product
is an exact representation consequence of the solved algebra, not an imported
spin action. Breadth-first closure gives 48 exact elements of
\(\mathbb Q(\sqrt2)\), the binary octahedral group. The CLI checks all
\(48^2=2304\) products, all 144 lift relations with the generating quarter
turns, 24 projected proper signed permutations, two lifts per rotation, and the
projection kernel \(\{I_4,-I_4\}\). The multiplication table is retained in
its output.

These are the spin lifts inherited from the connected continuum diagonal group.
If one instead asks for *every* SU(4) transformation preserving the adjoint order
above each cubic rotation, its four-element centre gives 96 elements. The
48-element result is not a false claim that this larger preimage has only 48.

For the actual same improved band, every inherited lift obeys
\[
 Uh_a(p)U^\dagger=h_a(Rp),\qquad
 Uj_i(p',p)U^\dagger=R_{ji}j_j(Rp',Rp),
 \tag{3}
\]
including the scalar Wilson/midpoint-link contributions. Signed permutations
commute with componentwise sine/cosine and the cubic mass function, which proves
(3). The CLI checks all 48 lifts at finite \(a\), not just a continuum surrogate.
For a noncubic rotation it separately reports the decreasing finite-lattice
defect at fixed physical mass, velocity and Wilson coefficient. Only the
continuum Hamiltonian has exact arbitrary-angle covariance.

## Finite gap, current and spin share the selected field basis

The coefficients that coupled to \(A_\mu\) before selection are unchanged.
With the selected Gram scales they become the effective \(m,r,v\) above. The
CLI reconstructs the microscopic Hamiltonian directly from those coefficients
and the selected/conjugated field, compares it with the same band, and derives
its compensating generators from that field's full real linear map. The same
basis transformation acts on \(h\), the finite-transfer current and the derived
spin generators. There is no separately chosen spin-frame map.

For \(m_{\rm eff}>0\), \(r_{\rm eff}\geq0\), the improved mass is
\[
 M_a(p)=m_{\rm eff}+\frac{r_{\rm eff}}a
       \sum_i[2\sin^2(ap_i/2)]^2\geq m_{\rm eff}.
\]
Hence \(E_a(p)\geq m_{\rm eff}\) and the positive/negative band separation is
at least \(2m_{\rm eff}\). The current is the existing actual midpoint-link
current, not a Gaussian or continuum replacement. It obeys
\[
 \widehat k_i j_i(p',p)=h_a(p')-h_a(p),\qquad
 \widehat k_i=\frac2a\sin[a(p'_i-p_i)/2],
\]
in the same field basis. The charge factor is the supplied \(q=-e\).

## Actual bosonic-code state consumer

The CLI constructs a four-flavour block with the same `EdgeWords`,
`block_terms` and `boundary_ladder` law as `pauli_compton.py`: four actual edge
qubits, four physical flavour vertices, and one **neutral hidden parity
marker**. It evaluates every 16 by 16 generator and current matrix by primitive
Pauli transfers on raw edge-bit states. No canonical CAR matrix builds these
operators. The tree block is the primitive full-parity consumer, not the full
spatial compact-rotor prototype.

The occupied negative rest doublet is filled by actual boundary creation
operators; positive-band creation and negative-band annihilation make electron
and hole states. The fundamental action lifts as
\(\widehat J_i=\sum_{\alpha\beta}c_\alpha^\dagger(J_i)_{\alpha\beta}c_\beta\).
Because the primitive code supplies the full charge algebra,
\[
 \exp(-2\pi i\widehat J_{\mathbf n})=(-1)^{N_{\rm physical}}.
\]
An actual even filled sea has \(N=2\) and transforms by \(+1\), an electron has
\(N=3\) and a hole \(N=1\), both transforming by \(-1\); an electron pair
\(N=4\) and empty state \(N=0\) give \(+1\). The CLI exponentiates the actual
Pauli-built generator on the **entire** physical 16-state space, checks these
vectors individually, and extracts spin-half Casimirs and normal-ordered
charges \(q\) and \(-q\) from the electron/hole sectors. It also checks
quarter-turn covariance of the full primitive current matrices, including
particle/hole mixing, and invariance of the actual filled rest sea.

The marker has occupation one exactly in odd physical sectors, so total
extended occupation is always even. Counting it as matter would incorrectly
turn every 2-pi action into \(+1\), and assigning it charge would spoil the
physical charge. Both wrong interpretations are adverse controls. This resolves
how a bosonic microscopic Hilbert space carries a charged spinorial sector;
it does not claim that every strictly local bosonic observable is a spinor.
The odd charged creation operator carries the boundary string already present
in the statistics supplier.

## Wrong frames have their actual smaller stabilizers

The exact same 18-variable solver is run on discriminating alternatives:

| Frame | Exact rank | Connected dimension | Spatial projection | Pure internal kernel |
|---|---:|---:|---:|---:|
| Selected cubic-isotropic metric Clifford | 15 | 3 | 3 | 0 |
| \(A_1=A_2=A_3=B_3\), with Clifford \(A_0=B_0\) | 14 | 4 | 1 | 3 |
| Three distinct spatial Clifford scales \((1,4/5,6/5)\) | 18 | 0 | 0 | 0 |
| Axial scales \((1,1,6/5)\) | 17 | 1 | 1 | 0 |

The completed CLI verifies these exact ranks and kernels. For the commuting
frame the internal commutant is su(2), independently of spatial so(2) rotations about
\((1,1,1)\). Its dimension four is **not** an isotropic spin-locking subgroup:
the spatial projection has dimension only one and is not uniquely lifted.
Three distinct nonzero Clifford scales allow no connected spatial rotations and
no pure internal generators. The axial frame retains only the diagonal so(2)
about its unscaled axis. In general (1) forces \([\Omega,G_s]=0\); unequal
spatial Gram eigenvalues therefore destroy precisely the unavailable physical
axes. Discrete rotations may remain even where the connected algebra is zero.

## Reproduction and evidence boundary

Run with the campaign Python interpreter:

```sh
python proposals/P256-qed-backward/spin_orbital_locking.py \
  --output proposals/P256-qed-backward/spin_orbital_locking_receipt.json
```

The default CLI includes exact rank/closure, actual soft-selected field solve,
all 48 same-band covariance consumers, finite-gap/current mapping, primitive
odd/even state action, current transformations, adverse stabilizers and the
constant-projector counterexample. It exits nonzero if any asserted consumer
fails. `--a`, `--m`, `--r`, `--v`, `--e`, and `--stiffness` name physical/input
parameters; their hypotheses are validated at the entry point.

The preserved completed receipt has `all_checks_pass=true`. Its selected-field
linear map has rank 15 and three physical lifts, residual `4.95e-15`.
The same microscopic-field/band map differs by `3.70e-15`; the exact
finite-transfer current Ward error is `2.21e-16`. The selected effective
gap is `1.13375082480`, and the particle/hole band separation bound
`2.26750164960`. These are model-input-dependent values, not measured electron
parameters. Exact cubic closure and every primitive odd/even state/current
control complete in the same run. The constant rest-projector Berry transport
is identity while the physical charged 2-pi action is minus identity.

Quantum ordering, asymptotic dressing and the full interacting electron/photon
continuum remain constructive directions, not conclusions of these checks.
