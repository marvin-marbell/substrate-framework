# Local SU(4) gauge locking: a changed microscopic orbital law

This is a **materially different constructive route**, not a reinterpretation
of `orbital_hydrodynamics`. That model has a global SU(4) and fifteen physical
orientation modes. Here SU(4) is local and has its own link rotors, electric
Casimir, Gauss constraints and Wilson magnetic Hamiltonian. At the selected
classical/occupied-sea saddle, the fifteen orientations are gauge coordinates
and all fifteen continuous gauge directions acquire positive quadratic masses.
Neither that statement nor the finite consumers below prove a quantum Higgs
phase, deconfinement, three-dimensional order, or a finished electron/photon
construction.

## 1. Primitive fields, links and Hamiltonian

Matter remains four **fundamental scalar flavour labels** per cell, realized by
the existing bosonic edge-Pauli transfers. It is not a supplied fundamental spin
representation. The four orbital fields are unrestricted traceless Hermitian
matrices, with all sixty real coordinates retained:

\[
 A_\mu(x)=z_{\mu a}(x)B_a,\quad
 \operatorname{Tr}(B_aB_b)=4\delta_{ab},\quad
 G_{\mu\nu}(x)=\tfrac14\operatorname{Tr}(A_\mu A_\nu).
\]

Keep the arbitrary-shape soft potential
\(V=\kappa\sum_x\|G(x)-g_*I\|_F^2/2\), \(\kappa,g_*>0\), and a
second-order orbital kinetic Hamiltonian \(\sum\pi_{\mu a}^2/(2M)\).
No SU(2) generator, anticommutator penalty, orthonormal frame restriction or
hard Gram budget chooses the shape. The local occupied sea and the soft
potential select the same representative supplied by `soft_budget.select_gram`:

\[
 F_\mu=\sqrt{g_\mu}\,\mathrm{SELECTED}_\mu,
 \qquad g=(g_s,g_v,g_v,g_v),\quad g_s,g_v>0.
\]

The executable obtains this from the actual improved `RotationBand` scalar
couplings on the recorded cubic \(3^3\) sea, not from guessed Gram values. This
is a selected flat-link saddle of the changed law; it is **not** a proof that
quantum fluctuating SU(4) links leave the original sea-selection theorem intact.

The arbitrary-link Wilson \(\pm m\) exclusion in `wilson_mass_geometry` uses
compact U(1) links, whose transport is spin-scalar and commutes with the
selected grading. **It is not a spectral-gap theorem for arbitrary SU(4)
links**, which need not commute with that grading. A pure local-frame
transformation of the flat vacuum transforms both its fields and links and
is exactly gauge equivalent, preserving its spectrum. General SU(4)
fluctuations are not of that form and require separate spectral bounds and
phase control. The full charged spectrum of this changed law is unproved.

On each oriented independent spatial link introduce \(L_{xy}\in SU(4)\),
\(L_{yx}=L_{xy}^\dagger\), with quantum Hilbert space \(L^2(SU(4))\).
The fundamental matrix entries are bounded multiplication operators. Write
\(T_a=B_a/(2\sqrt2)\), so \(\operatorname{Tr}(T_aT_b)=\delta_{ab}/2\).
The electric term and Wilson magnetic term are explicitly

\[
 H_{E4}=\frac{U_{E4}}2\sum_{\ell,a}(E^a_{\ell,L})^2,
 \qquad
 H_{B4}=K_4\sum_p\left(1-\frac14\operatorname{ReTr}
             \overrightarrow{\prod_{\ell\in\partial p}} L_\ell\right),
 \quad U_{E4},K_4>0.
\]

Left and right regular electric generators implement independent endpoint
rotations; their squared Casimirs agree. Their orientation convention is fixed
by the group action \(L_{xy}\mapsto G_xL_{xy}G_y^\dagger\), not by silently
replacing the two endpoints by one Abelian electric number. Orbital momenta
transform in the adjoint. SU(4) Gauss generators are the infinitesimal local
group action on incident links, orbital fields, fundamental matter **and the
reference representation below**; physical states are invariant under it.
The existing independent U(1) rotors, their electric/magnetic terms and Gauss
law with electromagnetic charge \(n_x-2\) are retained.

The covariant nearest-neighbour gradient is

\[
 H_{\nabla A}=\frac J8\sum_{\langle xy\rangle,\mu}
 \operatorname{Tr}\left[A_\mu(x)-L_{xy}A_\mu(y)L_{xy}^\dagger\right]^2,
 \qquad J>0. \tag{1}
\]

For every actual primitive directed path \(\gamma:y\to x\), including
improved two-link paths, use the **ordered** SU(4) transport \(L_\gamma\)
and existing U(1) transport \(u_\gamma\). The target-x/source-y hopping block is

\[
 C_{xy}=u_\gamma\sum_\mu s_\mu(\gamma)
 \frac{A_\mu(x)L_\gamma+L_\gamma A_\mu(y)}2. \tag{2}
\]

`scalar_paths()` extracts the \(s_\mu\) and onsite scalar from the existing
`RotationBand.couplings()`. Each matrix entry multiplies the **same**
\(-iA_{\mathrm{Pauli},\gamma}n_y(1-n_x)\) primitive transfer, with its
adjoint. Onsite terms are the same scalar times \(A_0(x)\). Products follow the
physical path; an independent flavour component is not assigned an unrelated
rotor path. In the quantum law SU(4) links commute with orbital coordinates
because they occupy different factors. The Hermitian adjoint, not a
commutative replacement of an ordered path product, defines the reverse term.

## 2. Covariance is derived at both consumers

Under \(c_x\mapsto G_xc_x\), \(A_\mu(x)\mapsto G_xA_\mu(x)G_x^\dagger\)
and \(L_{xy}\mapsto G_xL_{xy}G_y^\dagger\), intermediate factors in
\(L_\gamma\) cancel. Both summands of (2) therefore transform as
\(C_{xy}\mapsto G_xC_{xy}G_y^\dagger\); the difference in (1) transforms
by conjugation at x. The U(1) endpoint phases cancel separately. This gives
local covariance for arbitrary Hermitian shapes, not just the selected orbit.

`TwoCellPauli` uses the actual `quantum_spinor_plaquette.PauliCode`: normalized
raw two-word states in \(S=+\), eight active modes and **all 70 fixed-N=4
occupations**. The other eight plaquette matter modes are empty. Off-diagonal
matrix elements are obtained with `PauliCode.element()` and actual primitive
root-to-root paths; no canonical CAR matrix or wedge sign builds this matter
Hamiltonian. This fiber is not an SU(4)-Gauss projection and its external link
values are not a quantum rotor simulation.

For independent Hermitian local matrices \(K_0,K_1\), construct their actual
Pauli lifts \(\widehat K\). The transfer algebra gives
\([\widehat K,\widehat h]=\widehat{[K,h]}\); exponentiating therefore yields

\[
 \widehat{G h G^\dagger}
 =e^{i\widehat K}\widehat h e^{-i\widehat K},\quad
 G=\operatorname{diag}(e^{iK_0},e^{iK_1}). \tag{3}
\]

The CLI checks the infinitesimal identity and (3) independently on actual
Pauli matrices. Its fields include unrestricted non-Clifford Hermitian
perturbations, its link is noncentral, and the endpoint frames are independent
non-spin SU(4) transformations. The adverse control transforms both fields
but **omits the link transformation** and must fail covariance. The emitted
Hamiltonians and code-frame matrix make the consumer inspectable.

## 3. Exact quadratic Higgs matrix and all analytic multiplicities

At uniform \(F\), put
\(L=\exp(i g_{SU4}a V^aT_a)\). Expanding the actual energy (1) gives

\[
 H_{\nabla A}=\frac{a^2}2V^a\mathcal M_{ab}V^b+O(V^3),\qquad
 \boxed{\mathcal M_{ab}=\frac{Jg_{SU4}^2}4\sum_\mu
  \operatorname{Tr}([T_a,F_\mu]^\dagger[T_b,F_\mu])}. \tag{4}
\]

Here \(\mathcal M\) is the quadratic energy coefficient in the stated gauge
coordinate normalization. Interpreting its eigenvalues as propagating
frequency masses further requires a specified canonical time-kinetic
normalization (electric dynamics); changing that normalization rescales the
physical frequencies. No unreported frequency mass is inferred from (4).
For every real traceless Hermitian \(X=v^aT_a\),

\[
 v^a\mathcal M_{ab}v^b
 =\frac{Jg_{SU4}^2}4\sum_\mu\|[X,F_\mu]\|_{HS}^2\ge0.
\]

Vanishing requires X to commute with all four normalized Clifford matrices
\(\Gamma_\mu=F_\mu/\sqrt{g_\mu}\). Their products generate
\(M_4(\mathbb C)\), hence their common commutant is scalar. Traceless X is
therefore zero: (4) is strictly positive in **all fifteen** directions.
This is an exact algebraic statement for every positive selected Gram, not
an inference from fifteen rounded numerical eigenvalues.

Choose Hermitian Clifford blades of degree \(k=1,2,3,4\), normalized to
\(\operatorname{Tr}(T^2)=1/2\). For even k a blade anticommutes with the
Clifford generators *inside* its index set; for odd k it anticommutes with
those *outside*. An anticommuting direction contributes
\(\operatorname{Tr}[T,F_\mu]^\dagger[T,F_\mu]=2g_\mu\), so each mass is
\(Jg_{SU4}^2\sum_{\mu\in\mathcal A}g_\mu/2\). The cubic groups are:

| Degree | Gram sum | Multiplicity |
|---|---|---:|
| 1 | \(3g_v\) | 1 |
| 1 | \(g_s+2g_v\) | 3 |
| 2 | \(g_s+g_v\) | 3 |
| 2 | \(2g_v\) | 3 |
| 3 | \(g_s\) | 1 |
| 3 | \(g_v\) | 3 |
| 4 | \(g_s+3g_v\) | 1 |

Multiply every Gram sum by \(Jg_{SU4}^2/2\). Accidental coincidences add
multiplicities, never create zeros. At \(g_s=g_v=g\), the groups combine to
\(Jg_{SU4}^2g/2\) (4), \(Jg_{SU4}^2g\) (6),
\(3Jg_{SU4}^2g/2\) (4), \(2Jg_{SU4}^2g\) (1).

For \(F'=QFQ^\dagger\), set
\(R_{ab}=2\operatorname{Tr}(T_aQT_bQ^\dagger)\). Then
\(\mathcal M(F')=R\mathcal M(F)R^T\). The executable constructs the full
selected and generic-conjugated 15-by-15 matrices, checks this equation and
matches both spectra to all fifteen blade values. It also compares (4) with
the curvature of the **unexpanded** link exponential in (1).

## 4. Forty-five shapes, fifteen eaten coordinates, residual centre

The original 60-coordinate space decomposes into the injective map
\(X\mapsto i[X,F_\mu]\) (15 dimensions) and its orthogonal complement
(45 dimensions). In the global model all fifteen are genuine orbit modes.
In this local model they are gauge-redundant orientations: unitary gauge
removes them, and the corresponding continuous gauge fields have the positive
quadratic coefficients (4). This is the quadratic Higgs replacement of the
Goldstone branch, not a deletion of physical fluctuations from the old law.

The 45 transverse **shape** coordinates remain physical. Their restoring
curvatures come from the occupied sea plus the original soft potential, not
from a supplied spin matrix. The CLI computes the complete uniform 60-by-60
Hessian on the same selected finite \(3^3\) sea, checks its fifteen orbit
flats, and reports all 45 complement eigenvalues without clipping. This finite
saddle result does not prove positivity at every spatial momentum, stability
of the full quantum law, or thermodynamic order.

The stabilizer of four Clifford adjoints in SU(4) is exactly
\(Z_4=\{I,iI,-I,-iI\}\). The continuous gauge directions are all lifted,
but this discrete gauge group remains. Fundamental matter and centre-2
reference charges are nontrivial representations of it; their presence does
not by itself condense them or establish a deconfined Z4 phase. Centre fluxes,
confinement and quantum Higgs dynamics remain physical obligations.

## 5. A neutral electromagnetic sea is not an SU(4) singlet background

Two occupied fundamental modes per neutral cell transform in
\(\wedge^2\mathbf4=\mathbf6\). A centre element \(z=iI\) acts on this
pair as \(i^2=-1\): its SU(4) centre charge is **2 mod 4 per cell**, even
though its electromagnetic charge relative to \(n-2\) is zero. A scalar
EM background supplies no compensating colour charge. It cannot make a
link-free half-filled product state satisfy SU(4) Gauss.

One consistent microscopic choice adds a reference in
\(\overline{\mathbf6}\) per cell (equivalent to the real SU(4) six), with
centre charge -2=2 mod4. The orbital momenta, link electric fields, matter
and this reference all enter SU(4) Gauss. Matter \(\mathbf6\) tensor
reference \(\overline{\mathbf6}\) contains an invariant singlet, so the
centre mismatch can genuinely be cancelled. This is an added representation,
not a fabricated scalar colour density; electric flux/boundary sectors are
another possibility and must be specified if used.

The negative rest doublet of the selected \(F_0\) has a determinant pair
which is a singlet under the **derived combined** Spin(3). The CLI computes
that pair in the actual generic-conjugated frame, its exterior-square
reference action, the centre -1 action, and the derived spin annihilation.
A selected spin-singlet vector is **not an SU(4) singlet**. Freezing its colour
components without transforming the reference space would violate local
covariance. One can use a covariantly transforming reference and gauge
projection/entanglement into the \(6\otimes\bar6\) singlet; that projection
is an explicit remaining quantum construction, not performed in this CLI.
No false full SU(4)-Gauss ground-state simulation is claimed.

## 6. The code selector survives independent noncommuting coefficients

Retain the existing matched Pauli/U(1) paths, stabilizers and characters:

\[
 Q_\eta=\prod_{\ell:\eta_\ell=-1}Z_\ell,\qquad
 R_\eta=\exp\left(i\pi\sum_{\ell:\eta_\ell=-1}E_{U1,\ell}\right).
\]

Put every bounded SU(4) link-entry/path coefficient on its independent gauge
factor, and every bounded orbital coefficient on its orbital factor.
\(T_s=(Q_\eta R_\eta)\otimes I_{SU4,orbital,reference}\) commutes with
these coefficients, their Hamiltonians and colour charges. Each matched
Pauli path changes by \(\eta_\gamma\), each U(1) transport by the same
\(\eta_\gamma\); their product does not change. Different coefficient
operators need **not** commute with one another or with their auxiliary
Hamiltonian. SU(4) electric and Wilson terms are unchanged by T_s. Thus

\[
 T_s^\dagger H_sT_s=H_+
 +2K_{U1}\sum_{p:s_p=-1}C_p+\Delta N_{def},\qquad
 T_s^\dagger H_sT_s\ge H_++(\Delta-2K_{U1})N_{def}. \tag{5}
\]

The margin involves the U(1) plaquette magnetic coefficient, **not** an added
\(K_4\) penalty. This follows on finite graphs for bounded coefficients and
self-adjoint semibounded auxiliary Hamiltonians; the unbounded arbitrary
orbital coordinates require the closed-form/coercive hypotheses already
specified in `quantum_orbital_sector.md`, not an unqualified bounded-operator
assertion. SU(4) multiplication entries alone are bounded; their electric
Casimir is an unbounded self-adjoint auxiliary kinetic operator and need not
commute with them. The U(1) \(Q/R\) transformation also commutes with the
SU(4) Gauss projection when that projection is actually supplied.

As a finite adverse-capable consumer, the CLI places distinct bounded
noncommuting 2-by-2 matrices on every primitive path of the existing actual
U(1)-Gauss `CoupledSquare`, with a noncommuting auxiliary Hamiltonian. It
checks the complete sector identity and spectral comparison in (5). These
matrices prove the independent-coefficient algebra, not a substitute SU(4)
rotor vacuum. On the exact SU(4) rotor different multiplication entries
commute, but do not commute with the electric kinetic operator; noncommuting
Galerkin compressions and more general bounded auxiliary coefficients are
also covered by the same theorem when truncation is matched across sectors.

## 7. Bound the spin interpretation, rather than treating gauge as spin

`spin_orbital_locking.solve_selected_generators` solves the full adjoint/spatial
stabilizer map for the actual scaled/conjugated fields. Its unique traceless
internal compensators obey the derived Spin(3) algebra, Casimir 3/4 and
\(U(2\pi)=-I\). This remains an algebraic lift of **combined** coordinate
rotation and internal compensation; no SU(2) matrices selected the shape.
At finite cubic lattice spacing only proper cubic spatial rotations are exact
(the inherited lift is binary octahedral). Arbitrary continuous rigid rotations
require the continuum limit and cannot be silently assigned exact lattice
symmetry.

In the gauged law, internal local SU(4) is redundancy. A pure gauge
\(-I\) does **not** measure spin. For the Wilson-dressed relational bilinear
\(c_x^\dagger L_{xy}c_y\), a local centre change at x sends both
\(c_x\) and \(L_{xy}\) to their negatives, so the observable is unchanged.
The CLI explicitly checks this cancellation, as well as centre covariance
of the actual hopping block. A Wilson-line charged operator anchored to a
boundary transforms trivially under gauge transformations trivial at that
boundary; under a nontrivial boundary centre transformation it can carry a
fundamental centre charge. That is a **boundary/reference action**, not a
pure rigid-space rotation experiment and not a local gauge observable's
minus sign.

An operational half-integer-spin inference needs a specified physical
coordinate-rotation action on the charged, boundary-dressed/reference-coupled
state and apparatus, plus a comparison/interference experiment that survives
gauge quotient and dynamics. Boundary conditions decide whether a centre
action is physical or redundant. The compensating lift alone does not prove
that a 2pi rigid-space rotation acts nontrivially on gauge-invariant states,
that a stable electron exists, or that spin/statistics and scattering of the
full continuum theory have been derived.

## Execution and evidence ownership

The parent executed the default scientific CLI successfully:

```sh
python proposals/P256-qed-backward/orbital_gauge_locking.py \
  --output proposals/P256-qed-backward/orbital_gauge_locking_receipt.json
```

The fifteen bare Higgs **energy curvatures** range from 0.258219369813 to
1.62998778074 at J1 and SU(4) coupling .7; these are not dressed quantum
frequencies. The independent-coefficient sector identity residual is
2.03e−15. Independent source review accepted the local covariance, mass
matrix, centre/reference accounting and gauge-minus cancellation. Full
SU(4) Gauss projection and a three-dimensional quantum Higgs phase remain
unproved.

The JSON includes the exact source hashes, selected Gram/fields, full positive
15-direction mass matrices, blade multiplicities, conjugation and exact-link
curvature comparisons, unrestricted-shape actual Pauli covariance and omitted
link adversary, 45 uniform shape curvatures, centre/reference checks, and
independent noncommuting-coefficient sector comparison. `all_checks_pass` is
a finite construction receipt, not quantum phase or spin-experiment proof.
