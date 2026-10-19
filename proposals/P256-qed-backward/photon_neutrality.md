# Compact-photon neutrality on a general finite spatial graph

## Result and intended consumer

For **untruncated** compact U(1) link rotors on any finite connected spatial
graph, with strictly positive electric coefficients and a real bounded
Wilson potential, the bottom of every admissible complete Gauss-character
sector is at least the neutral photon ground energy:

\[
 H_{\mathrm{ph}}\big|_{q}\ \geq\ \epsilon_0 I.
\]

No estimate proportional to the magnetic coefficients is lost in comparing
these sectors. The reference energy itself depends on the magnetic potential;
“magnetic-independent comparison” does **not** mean a magnetic-independent
vacuum energy. This is a useful common-vacuum-energy cancellation in a
matter/photon operator estimate, **not** by itself a charged-fermion mass-gap
theorem. Matter onsite energies, hopping estimates, code sectors, and the
neutral variational state remain separate obligations of their actual owner.

For a noncyclic Fourier cutoff, a charged block remains bounded below by this
**continuum** neutral reference. Comparing it instead with the **cutoff**
neutral ground is a different proposition. The general continuum positivity
proof does not prove that proposition. This artifact supplies:

1. the untruncated operator theorem, with domains and complete boundary Gauss;
2. an exhaustive two-adjacent-plaquette noncyclic Fourier consumer;
3. a rationally certifiable search for a genuine finite-Fourier counterexample
   under more general real Wilson potentials, distinct from the fundamental
   positive plaquette potential;
4. a constructive finite quotient-grid approximation with exact integer U(1)
   Gauss characters, a proved neutral comparison, and a controlled continuum
   limit regardless of that search outcome;
5. a computable continuum-reference lower bound from a neutral Ritz block.

The runnable source is `photon_neutrality.py`. It does not construct a new
charged matter Hamiltonian and does not alter any shared quantum module.

## Units, orientations, and physical boundary owner

Set \(\hbar=1\). Link angles \(\theta_l\), Fourier fields \(E_l\), integer
charges, and loop windings are dimensionless. Every \(U_{E,l}\), magnetic
coefficient \(K_p\), and ground energy is in the same energy unit. Heat time
has inverse-energy units.

Let \(B\in\mathbb Z^{V\times E}\) have +1 at the oriented tail of a link and
-1 at its head. On the full vertex set,

\[
 G_v=(B E)_v,\qquad E_l=-i\partial_{\theta_l}.
\]

If matter/background charges are \(Q_v\), and fixed outgoing external boundary
fluxes contribute \(b_v\), the internal rotor character is

\[
 q_v=Q_v-b_v,\qquad \sum_v q_v=0.
\]

The external electric contribution is the separate scalar
\(C_b=\frac12\sum_{a\in\mathrm{external}}U_{E,a}b_a^2\). It is not silently
part of the internal photon ground energy. With fixed external flux and no
external angle-dependent coupling,

\[
 (H_{\mathrm{ph},q}+C_b)\ \geq\ (\epsilon_0+C_b)I.
\]

This statement does not allow an uncompensated total internal charge. For a
finite connected graph the image of the integer incidence map is exactly
\(\{q\in\mathbb Z^V:\sum q_v=0\}\), proved by an integer spanning-tree solve.
The diagonal vertex gauge U(1) is the kernel of the action on link angles;
its character must therefore be trivial. Fractional charges are not
characters of compact U(1). With only a subgroup of boundary gauge
transformations imposed, the corresponding subgroup characters are used;
that is not a justification for dropping a vertex equation from a claimed
**complete** Gauss law. Dynamical external links or pinned external angles
require their own enlarged operator/domain, not the scalar argument above.

## Untruncated operator, domain, and positivity

Use normalized Haar measure on \(\mathbb T^E\) and the Hilbert space
\(\mathcal H=L^2(\mathbb T^E)\). Define

\[
 H_E=-\frac12\sum_l U_{E,l}\partial_l^2,
 \qquad
 V(\theta)=-\sum_p K_p\cos(c_p\cdot\theta+\delta_p),
 \qquad Bc_p=0,
\]

with finitely many integer loop vectors, real coefficients/phases, and
\(U_{E,l}>0\). Positivity of \(K_p\) is not required for this theorem. More
generally a real bounded gauge-invariant multiplication operator suffices.
The associated closed lower-semibounded quadratic form has domain
\(H^1(\mathbb T^E)\). Since the electric metric is uniformly positive and
\(V\) is bounded, the self-adjoint operator domain is
\(H^2(\mathbb T^E)\); bounded perturbation does not change it. The resolvent is
compact. Smooth Wilson potentials give smooth eigenfunctions.

The free anisotropic heat kernel is the product of positive wrapped Gaussian
kernels. For every \(t>0\), Feynman–Kac and boundedness of the real potential
sandwich the interacting heat kernel between positive multiples of the free
kernel. Thus \(e^{-tH}\) is positivity improving. Its lowest eigenvalue is
simple and admits an a.e. strictly positive real normalized ground vector.
For the smooth Wilson potential the elliptic strong maximum principle makes
this vector strictly positive at every angle. This is an **angle-space**
statement on the untruncated torus; it is not Perron–Frobenius applied inside
an already-selected Fourier charge block.

For \(\alpha\in\mathbb T^V\), define

\[
 (T_\alpha\psi)(\theta)=\psi(\theta+B^T\alpha).
\]

These translations preserve both Sobolev domains, commute with \(H_E\), and
commute with \(V\) because \(Bc_p=0\). Each translated ground vector is again
a normalized strictly positive ground vector. Simplicity implies a scalar
multiple; positivity and normalization fix that scalar to +1. Consequently
\(T_\alpha\psi_0=\psi_0\): the ground lies in the **zero complete Gauss
character**, not merely in the zero total charge sum.

The reducing sector

\[
 \mathcal H_q=\{\psi:T_\alpha\psi=e^{iq\cdot\alpha}\psi\}
 =\overline{\operatorname{span}}\{e^{ie\cdot\theta}:e\in\mathbb Z^E,
                                      Be=q\}
\]

inherits the restricted form/operator domains. The min–max principle gives
\(H|_q\geq\epsilon_0\). In fact a fixed nonzero admissible \(q\) has strictly
larger lowest energy, because its bottom is attained and the unique global
ground cannot lie in two distinct characters. No uniform quantitative
charged-sector gap is inferred from that strictness.

## Exact integer cycles and the two-plaquette Fourier consumer

The open graph has vertices
\((0,0),(1,0),(2,0),(0,1),(1,1),(2,1)\), numbered 0 through 5. Its seven
oriented links are

\[
 (0,1),(1,2),(3,4),(4,5),(0,3),(1,4),(2,5).
\]

Choose tree links 2,3,4,5,6 and chord links 0,1. The primitive integer cycle
columns are

\[
 C_1=(1,0,-1,0,-1,1,0)^T,
 \quad C_2=(0,1,0,-1,0,-1,1)^T.
\]

They satisfy \(BC=0\), and the chord rows form the identity. For any
admissible integer \(q\), leaf elimination gives a unique tree field
\(a(q)\) with zero chords. **Every** integral solution of complete Gauss is

\[
 e=a(q)+Cn,\qquad n\in\mathbb Z^2.
\]

For a noncyclic Fourier cube \(|e_l|\leq R\), the true domain is
\(D_{q,R}=\{n:|a_l+(Cn)_l|\leq R\ \forall l\}\), not two independently
truncated loop rotors. In the neutral sector it is the hexagonal integer
region

\[
 |n_1|\leq R,\quad |n_2|\leq R,\quad |n_1-n_2|\leq R.
\]

The neutral dimensions are therefore 7,19,37 for \(R=1,2,3\), respectively.
A Wilson loop acts by its exact integer cycle shift if the target belongs
to the domain and is otherwise omitted. It NEVER wraps a field through
\(+R\) to \(-R\).

The CLI enumerates all \((2R+1)^7\) link configurations, partitions them by
**all six** integer Gauss charges, and verifies reconstruction by the exact
integer tree/cycle solver. Every nonempty character is compared, including
characters outside a chosen matter occupation range. Separately, it reports
all physically admitted four-orbital/site occupation patterns at
\(N=11,12,13\): half-fill background 2/site and root external flux
\(b=N-12\), with \(n_v=q_v+2+b\delta_{v0}\in\{0,1,2,3,4\}\). Dimensions
include the actual \(\prod_v\binom4{n_v}\) orbital multiplicities. No
occupation-wavefunction or matter-sector ground is prescribed.

The four fundamental-potential cases use \((U_E;K_1,K_2)\) equal to
\((0.6;0.25,0.25)\), \((0.0072;8/9,8/9)\), \((0.11;3,3)\), and
\((0.6;0.01,2)\). These cover weak/strong, campaign, and anisotropic magnetic
regimes. They are finite numerical discriminants; a nonnegative minimum in
these cases is not a theorem for all coefficients or all graphs.

## Why Fourier angle positivity cannot be imported

Already for one rotor at \(R=1\), the projected free heat kernel at angle
separation \(\pi\) is

\[
 k_{1,t}(\pi)=\frac{1-2e^{-tU_E/2}}{2\pi}<0
 \quad\text{for }0<t<2\log2/U_E.
\]

Thus the smallest nontrivial Fourier truncation loses the angle positivity
used above. This is a genuine counterexample to the **positivity shortcut**,
not a counterexample to the charged-versus-neutral ground comparison. The
consumer labels it accordingly. A second adverse control wraps all seven
fields modulo 3 at \(R=1\): complete integer Gauss changes while its mod-3
value does not. A Z3 link model therefore cannot be relabelled as the exact
compact-U(1) cutoff. Fractional characters and an uncompensated external
boundary flux are explicitly rejected.

There is an independent exact one-plaquette finite-cutoff positive control:
`compact_charge_gap.py` proves the Holder trace/ground comparison for the
noncyclic integer intersection. The new CLI imports only its tiny rotor
routines, checks all affine domains at R1/2/3, and records that source hash.
For the ground inequality alone one can further see the special one-loop
geometry without angle positivity. Write the oriented fields as \(n+a_l\)
with a common link cutoff and let \(U_\Sigma=\sum_lU_{E,l}\). The integer
shift of any vector by \(a_l\) fits the neutral interval because its
\(l\)-th field fits. Toeplitz Wilson matrix elements are shift invariant.
Then

\[
 \langle\psi,H_a\psi\rangle
 =\sum_l\frac{U_{E,l}}{U_\Sigma}
        \langle U_{a_l}\psi,H_0U_{a_l}\psi\rangle
 \geq \epsilon_{0,R}\|\psi\|^2.
\]

This even allows arbitrary real one-loop Wilson coefficients. The mapping
uses the **common one-dimensional neutral interval**; it is not available
by individually shifting links in a multiloop Gauss domain.

The separate R1 search allows all real cosine windings in the difference
hexagon, with rational integer coefficients and \(U_E=1/100\). A found
witness is re-certified in exact `Fraction` arithmetic: a rational threshold
\(\lambda\) has positive LDL pivots for
\(H_{0,R}-\lambda I\), while a rational charged trial vector has Rayleigh
quotient below \(\lambda\). This proves
\(E_{q,R}<E_{0,R}\) without relying on floating eigenvalue tolerances. R0
contains only the zero field and zero charge, so any such R1 witness is at
the smallest possible nontrivial cutoff. The counterexample below concerns
the **general real Wilson class**, not the positive two-fundamental-loop
subclass. A bounded search with no witness would report precisely that
bounded outcome, never universal absence.

### Parent-executed smallest-cutoff counterexample

The parent ran the decision probe to its natural exit 0 and preserved
`photon_neutrality_counterexample_receipt.json`, source SHA256
`3c5c44eb78d4e1877191506e362c84680fff9065fea164406ae45e77f028c88f`.
Trial 6 supplies R1, \(U_E=1/100\), and complete integer character
\(q=(0,-1,0,0,1,0)\). The neutral/charged dimensions are 7/6. All phases
vanish; the potential is \(-\sum_d K_d\cos(d\cdot\phi)\) with

| \(d\) | (0,1) | (0,2) | (1,-1) | (1,0) | (1,1) | (1,2) | (2,0) | (2,1) | (2,2) |
|---|---|---|---|---|---|---|---|---|---|
| \(K_d\) | -2 | -3 | 0 | -1 | -3 | 1 | -1 | -1 | 3 |

The observed floating ground energies are
\(E_{0,1}=-3.440583803585343\) and
\(E_{q,1}=-3.6456904340600973\), a difference of
\(-0.20510663047475441\). More importantly, the preserved **exact rational
certificate** proves strict order with

\[
 \lambda=-1771569/500000,\qquad
 \mathcal R_q=-28043796471897/7692314248600<\lambda.
\]

All seven LDL pivots of \(H_{0,1}-\lambda I\) are positive. In the charged
field ordering of the receipt, the rational trial vector is

\[
 (234371/500000,\ 33401/125000,\ -145311/500000,\
  -242047/500000,\ -190487/10^6,\ 594801/10^6).
\]

Its nonzero norm is included in the exact Rayleigh denominator. Thus the
general finite-Fourier neutral comparison really fails; it is not merely
unproved because its positivity shortcut fails. R0 has no charged field,
so R1 is the smallest cutoff possible on this graph. The true compact
continuum theorem survives, and the finite route below constructs the
correct comparison rather than suppressing this adverse result.

## A controlled reference even for a Fourier cube

A charged Fourier-Galerkin subspace is a true subspace of the continuum
sector, so \(E_{q,R}\geq\epsilon_0\), regardless of any cutoff-neutral
comparison. Let \(p_R\) be the neutral Ritz ground, \(v\geq\|V\|\), and
\(u=\min_lU_{E,l}\). In the discarded neutral complement at least one
integer field has magnitude \(R+1\); hence

\[
 QH Q\geq d_R Q,\qquad d_R=\tfrac12u(R+1)^2-v,
 \qquad \|PHQ\|\leq v.
\]

Block-form Cauchy–Schwarz and the smallest eigenvalue of the two-by-two
comparison matrix give the rigorous neutral reference

\[
 \ell_R=\max\left\{-v,
 \frac{p_R+d_R-\sqrt{(d_R-p_R)^2+4v^2}}2\right\}
 \leq\epsilon_0\leq p_R.
\]

Consequently every charged Galerkin block is at least \(\ell_R\). As
\(R\to\infty\), neutral Fourier polynomials are a form core,
\(p_R\downarrow\epsilon_0\), and the reference converges to the same vacuum
energy. The formula is a theorem; its CLI evaluation uses a floating
neutral Ritz energy and is labelled as such, not as interval-certified
arithmetic. It remains useful when only a finite neutral operator is
available, without falsely replacing the continuum reference by \(p_R\).

## Constructive finite route: covariant differences on the gauge quotient

The quotient angles are \(\phi=C^T\theta\in\mathbb T^c\),
\(c=E-V+1\). Because C is a primitive fundamental-cycle basis, its Haar
pushforward is normalized Haar and

\[
 (J_a f)(\theta)=e^{ia\cdot\theta}f(C^T\theta)
\]

is an isometry onto the complete Gauss-character sector with \(Ba=q\).
In this chart

\[
 E_l J_a f=J_a(a_l-iC_l\cdot\nabla_\phi)f,
 \quad
 H_q=\frac12\sum_lU_{E,l}(a_l-iC_l\cdot\nabla)^2+V(\phi).
\]

This is an exact reduced operator with form domain \(H^1(\mathbb T^c)\)
and operator domain \(H^2(\mathbb T^c)\). A tree graph has c=0 and a
one-dimensional character sector; the empty quotient is handled separately
by the same formula.

Let \(h=2\pi/M\), and sample the quotient torus at \(M^c\) sites. Define

\[
 \mathfrak h^h_q[u]
 =\sum_{x,l}\frac{U_{E,l}}{2h^2}
     |u_{x+C_l}-e^{-iha_l}u_x|^2
   +\sum_x V(hx)|u_x|^2.
\]

Use normalized grid measure, or equivalently ordinary Euclidean coordinates
with the same normalization in both sectors. The potential is the actual
real phase-dependent Wilson potential, not its Fourier-space hopping
surrogate. All kinetic edge weights are positive. The neutral quotient
graph is connected because the chord rows of C contain every coordinate
unit direction, so its heat semigroup is positivity improving and its
ground is unique and strictly positive. Zero cycle rows, including bridges,
produce the appropriate nonnegative charged scalar terms.

The elementary magnetic/diamagnetic inequality

\[
 ||u_{x+C_l}|-|u_x||\leq
     |u_{x+C_l}-e^{-iha_l}u_x|
\]

and reality of V give

\[
 \mathfrak h^h_q[u]\geq\mathfrak h^h_0[|u|],
 \qquad E^h_q\geq E^h_0.
\]

This is a genuine finite-dimensional comparison for arbitrary real Wilson
coefficients/phases and any strictly positive link electric metric. It
needs no Fourier-cube pointwise positivity assumption.

**Gauss is not modularized.** The finite space carries
\(T_\alpha u=e^{iq\cdot\alpha}u\) with its exact integer label. Equivalently,
cardinal Fourier reconstruction uses fields \(a+Cn\), all satisfying
\(B(a+Cn)=q\) exactly. One may take any finite direct sum of these labelled
spaces. The discrete electric Hamiltonian is an approximation of the
reduced differential operator, not the exact squared truncated link E and
not an exact finite-dimensional Weyl rotor algebra. Periodicity of the
**quotient coordinates** does not authorize wrapping an individual link
field. The stencil energies can alias when q itself grows with M; no
uniform-in-all-charges continuum limit or strict finite charged gap is
claimed.

Changing the integer representative by \(a\mapsto a+Ck\) gives the exact
unitary equivalence

\[
 H^h_{a+Ck}=M_k^*H^h_aM_k,
 \qquad (M_ku)_x=e^{ikhx}u_x.
\]

The consumer constructs both stencils independently and checks this
identity. It also reconstructs all principal Fourier modes and checks
complete integer Gauss, rather than only a mod-M version.
It checks an actual matrix plane-wave expectation against the independent
charged kinetic symbol, and compares grid energies with same-potential,
same-electric-metric noncyclic Ritz sequences at R2/4/6. These are evidence
consumers for the proved limit, not a finite sample promoted to convergence.

## Legitimate continuum limit

Fix an admissible integer q, its tree representative a, positive electric
coefficients, and a finite real Wilson potential while M increases. On a
fixed Fourier mode n the discrete kinetic symbol is

\[
 \sum_l\frac{U_{E,l}}{h^2}
      [1-\cos(h(a_l+(Cn)_l))]
 \longrightarrow\frac12\sum_lU_{E,l}(a_l+(Cn)_l)^2.
\]

The nonnegative error is bounded by
\(h^2\sum_lU_{E,l}(a_l+(Cn)_l)^4/24\). This is consistency on fixed modes,
not a uniform claim about modes at the cutoff.

For compactness choose the tree representative with zero chord fields.
The chord rows of C are coordinate unit vectors. On principal Fourier
indices \(|hn_j|\leq\pi\),

\[
 \frac{U_{E,j}}{h^2}[1-\cos(hn_j)]
 \geq \frac{2U_{E,j}}{\pi^2}n_j^2.
\]

Bounded discrete energy and bounded V therefore give a uniform H1 bound
on trigonometric reconstructions. Rellich compactness supplies strongly
L2-convergent subsequences. Fixed-mode consistency and Fatou give the
kinetic liminf inequality. Bounded finite Fourier multiplication converges
on low modes; the uniform potential norm controls the high-mode remainder,
so the aliased sampled potential has the same form limit. Trigonometric
polynomials are a form core and provide recovery sequences. Thus the
forms converge, and compactness/min–max gives convergence of the ground
and each fixed low eigenvalue to the exact sector operator. This argument
uses continuous finite Wilson potentials; arbitrary bounded measurable
potentials cannot be justified by point sampling alone.

The sampled neutral comparison consequently tends to the same untruncated
vacuum reference. Neither grid energies nor different graph sizes are
asserted to form a monotone sequence.

## Execution and evidence contract

Parent-owned scientific commands, from the project root:

```sh
/home/administrator/substrate-framework/.venv/bin/python proposals/P256-qed-backward/photon_neutrality.py --output /tmp/photon-neutrality.json
/home/administrator/substrate-framework/.venv/bin/python proposals/P256-qed-backward/photon_neutrality.py --mode cube --cutoffs 1 2 3
/home/administrator/substrate-framework/.venv/bin/python proposals/P256-qed-backward/photon_neutrality.py --mode counterexample --counterexample-trials 256
/home/administrator/substrate-framework/.venv/bin/python proposals/P256-qed-backward/photon_neutrality.py --mode grid --grid-sites 9 17 33
```

The full JSON is printed and optional `--output` writes the same data. The
record carries the exact source SHA256, graph, units, coefficients, domains,
character counts, operator dimensions, counterexamples/search bounds, and
adverse controls. The finite cube branch reports counterexamples rather
than suppressing them. The grid branch asserts its proved comparison and
actual complete-Gauss invariants. The parent-executed counterexample above
is an observed result on its pinned probe source. The exhaustive fundamental
cube and full grid branches remain parent-owned executions; this note
does not predeclare their outcomes.

Language probes used Pyright 1.1.414 stdio and ast-grep 0.45.3. Initial
symbols/references and representative diagnostics were exercised before
implementation; a positive structural query matched the established
rotor-domain constructor. The final rooted server explicitly loaded this
project's `pyproject.toml` and selected the project interpreter. Definition
and references resolved the new `grid_operator`; final version-2 diagnostics
were empty. A positive AST query parsed its full changed body. The actual
project-Python `--help` CLI naturally exited 0. The OMP `xd://lsp` adapter
is not mounted in this child; actual stdio is the fallback, not a claimed
OMP integration. Scientific executions remain parent-owned as above.
The governing campaign/template remain unchanged. This sub-result does not
merge, close, or promote the full electron-photon campaign.
