# Unrestricted quantum adjoint field as a microscopic mass supplier

This construction is a **distinct** microscopic supplier, not a reinterpretation
of the positive-halfline fixed-Clifford radial/filled-sea Hamiltonian. Its field
is an arbitrary traceless Hermitian four-flavour matrix

\[
 A_0=\sum_{a=1}^{15}\phi_a T_a,\qquad \phi\in\mathbb R^{15},\qquad
 \operatorname{Tr}(T_aT_b)=\delta_{ab}/2.
\]

All fifteen canonical coordinates are present. The numerical basis is the
complete tensor-Pauli basis divided by \(\sqrt8\), obtained from
`spin_orbital_locking.exact_units`; no three matrices or Clifford frame are
selected. Any orthogonal change of the complete Hermitian basis gives an
equivalent model. The actual four-flavour primitive matter code is the
`pauli_compton.EdgeWords` five-vertex tree with four edge qubits and one neutral
boundary parity marker. Both physical parities occur. The marker is neither
charged matter nor an appended spectator spin. `primitive_matrix` evaluates
all bilinears by actual edge-Pauli transfers; `boundary_ladder` supplies the
actual odd addition/removal maps. Their equality and full CAR are checked.

## Hamiltonian, domains and units

For physical occupation \(N=0,\ldots,4\), the matter space is the primitive
\(\wedge^N\mathbb C^4\), with dimensions \(1,4,6,4,1\). Define

\[
 Q_a=\sum_{ij}c_i^\dagger (T_a)_{ij}c_j,\qquad
 H_N=\sum_{a=1}^{15}\left({\Pi_a^2\over2M}
                         +{\kappa\phi_a^2\over2}\right)
       +\mu\sum_a\phi_aQ_a,
 \qquad M,\kappa>0,\quad \mu\in\mathbb R.
\]

The Hilbert space is \(L^2(\mathbb R^{15},d^{15}\phi)\otimes
\wedge^N\mathbb C^4\); \(\phi_a\) acts by multiplication and
\(\Pi_a=-i\partial_{\phi_a}\). Schwartz functions tensored with finite matter
vectors are a common invariant core. The harmonic oscillator form domain is
\(\{\psi:\partial_a\psi,\phi_a\psi\in L^2\text{ for every }a\}\). The linear
matrix potential is infinitesimally form- and operator-bounded relative to the
oscillator, so its form defines a lower-bounded self-adjoint Hamiltonian, with
the oscillator operator domain and compact resolvent. The oscillator has a
finite-dimensional internal factor; no ultraviolet field limit is being
claimed. Write \(E_N=\inf\operatorname{spec}H_N\).

Units use \(\hbar=1\). If a field coordinate has units \(F\), then
\(\Pi\) has units action/\(F\), \(M\) has units \(1/(\text{energy}\,F^2)\),
\(\kappa\) has units energy/\(F^2\), and \(\mu\) has units energy/\(F\).
The matrix \(A_0\) has field units; \(\mu A_0\), not \(A_0\) alone, has energy
units. \(\omega=\sqrt{\kappa/M}\) is an energy with \(\hbar=1\).
The bare potential \(\kappa\sum_a\phi_a^2/2\) has its unique minimum at zero:
no prescribed nonzero mass, amplitude, or ordered beta direction appears.
The microscopic coupling \(\mu\) generates the quantum dressing.

## Primitive Casimir and noncommuting-square lower bound

Completeness of an arbitrary trace-normalized su(4) basis gives

\[
 \sum_a (T_a)_{ij}(T_a)_{kl}
 ={1\over2}\left(\delta_{il}\delta_{jk}
                    -{1\over4}\delta_{ij}\delta_{kl}\right).
\]

Using the actual primitive CAR, or equivalently reducing this identity in its
verified bilinear representation, gives

\[
 \sum_a Q_a^2=C_N I,\qquad C_N={5N(4-N)\over8},\qquad
 C_1=C_3={15\over8},\quad C_2={5\over2}.
\]

For example \(\sum_{ij}c_i^\dagger c_jc_j^\dagger c_i=N(5-N)\), so
\(C_N=[N(5-N)-N^2/4]/2\). The CLI evaluates the Casimir and su(4) commutators
on every actual number sector, rather than merely inserting a CAR comparator.

Completing **each single-coordinate** square yields the form identity

\[
 H_N=\sum_a\left[{\Pi_a^2\over2M}
       +{\kappa\over2}\left(\phi_a+{\mu Q_a\over\kappa}\right)^2\right]
       -{\mu^2\over2\kappa}\sum_aQ_a^2.
\]

For each fixed \(a\), the finite Hermitian \(Q_a\) can be diagonalized, and
translation in that one coordinate makes its bracket an ordinary shifted
harmonic oscillator. It therefore has the operator-form lower bound
\(\omega/2\). Summing the inequalities is valid even though the different
\(Q_a\) do **not** commute. Thus

\[
 \boxed{E_N\ge E_0^{\rm osc}-{\mu^2 C_N\over2\kappa}},\qquad
 E_0^{\rm osc}={15\omega\over2}.
\]

This argument does not assume the existence of a simultaneous shifted
vacuum, nor assert saturation of the sum of lower bounds.

## Neutral trial and improved exact charge-gap theorem

Let \(v\) be any normalized primitive neutral vector, \(|0\rangle\) the
fifteen-oscillator vacuum, \(X=\sum_a\phi_aQ_a\), and

\[
 s={\mu^2\over2M\omega^3}.
\]

Since \(\langle0|\phi_a\phi_b|0\rangle=\delta_{ab}/(2M\omega)\),
\(\|X|0\rangle v\|^2=C_2/(2M\omega)\). The originally specified unnormalized
trial \(|0\rangle v-(\mu/\omega)X|0\rangle v\) has norm squared
\(1+sC_2\) and normalized energy

\[
 E_0^{\rm osc}-{\omega s C_2\over1+sC_2}.
\]

The cubic expectation of \(X\) is zero by oscillator inversion parity; this
uses no commutativity of the matter matrices. Subtracting that variational
upper bound from the charged lower bound gives the earlier strict interval
\(0<s<2/15\).

The stronger result optimizes the entire two-level Rayleigh space. Its two
orthonormal vectors for each neutral \(v\) are
\(|0\rangle v\) and \(X|0\rangle v/\sqrt{C_2/(2M\omega)}\). Different
neutral vectors remain orthogonal by the scalar Casimir. On their full
twelve-dimensional span the exact compression is

\[
 E_0^{\rm osc} I_{12}
 +\begin{pmatrix}0&\mu\sqrt{C_2/(2M\omega)}\\
                 \mu\sqrt{C_2/(2M\omega)}&\omega\end{pmatrix}\otimes I_6.
\]

The diagonal coupling on the first-shell vectors vanishes by the same odd
triple term. Consequently

\[
 E_2\le E_0^{\rm osc}+{\omega\over2}
                         \left(1-\sqrt{1+4sC_2}\right),
\]

and for either charged sector

\[
 \boxed{E_{1,3}-E_2\ge {\omega\over2}
       \left(\sqrt{1+4sC_2}-1\right)-\omega s C_1.}
\]

For \(s>0\), the right-hand side is positive exactly when
\(\sqrt{1+4sC_2}>1+2sC_1\), equivalently

\[
 \boxed{0<s<{C_2-C_1\over C_1^2}={8\over45}.}
\]

This is an **infinite-oscillator theorem**: a charged lower bound minus a
neutral variational upper. It is not an exact ground-energy formula, and it
does not follow from differences of finite Ritz energies. At
\(M=\kappa=1,\mu=.4\), \(s=.08\) and the derived bound is
\(0.0208203932499\). At zero coupling there is no strict gap. Outside the
strict interval this estimate has no positivity conclusion. The CLI preserves
the earlier trial as a comparator and checks the improved twelve-dimensional
compression using all six actual primitive neutral vectors.

If an explicitly **fixed** one-cell U(1) outward boundary flux is chosen as
\(N-2\), its electric energy \(U_E(N-2)^2/2\), \(U_E\ge0\), adds
\(U_E/2\) to either charged-neutral gap. This is only a constrained one-cell
flux cost, not a dynamical photon oscillator, Coulomb field, or photon spectrum.

## SU(4)-invariant quantum Ritz consumer

The CLI uses every oscillator number state with total excitation
\(\sum_an_a\le K\). Defaults are \(K=1,2,3,4\) and \(\mu=.2,.4\),
\(M=\kappa=1\). The oscillator dimensions are \(16,136,816,3876\).
A total-excitation cutoff preserves SU(4); fifteen independent per-axis
cutoffs would not.

Set \([T_b,T_a]=if_{bac}T_c\). Actual total generators are

\[
 J_b=Q_b+L_b,\qquad L_b=\sum_{ca}if_{bac}a_c^\dagger a_a,
 \qquad [L_b,\phi_a]=if_{bac}\phi_c.
\]

These commute with the scalar coupling and the oscillator number Hamiltonian.
The full sparse coordinate-covariance identities are checked on **all**
cutoff states, including the outer shell, alongside primitive matter
commutators. The CLI additionally reports deterministic full-H commutator
probes. A single lowest eigenvector is expanded under these actual total
SU(4) generators, with reorthogonalization, to recover its full ground orbit.
There is no reliance on single-vector `eigsh` degeneracy counting and no
spectator-spin multiplication. Ground residual, orbit closure,
orthonormality and total Casimir are reported. An independent lowest solve
with the recovered ground projector lifted probes the eigen-complement;
its overlap, residual and separation expose missed degenerate ground states.
The observed finite-cutoff ground dimensions must be \(4,6,4\) for
\(N=1,2,3\). This is a finite numerical multiplet statement, not a proof
that the exact infinite ground representation cannot change at larger
coupling. Sectors zero and four are included as the free-oscillator controls.

For every sector the receipt reports nested Ritz energy changes, charged
Ritz gaps, shell probabilities and the norm of the full Hamiltonian's
missing-shell action. These monitor convergence; shell leakage and Ritz
nesting are **not** rigorous numerical error bars on the charge-gap
difference. The exact theorem is separately labelled in each receipt row.

All fifteen field means and the full symmetrized \(15\times15\) covariance
are evaluated on the equally weighted recovered multiplet. Applying fields
into shell \(K+1\) computes genuine canonical moments, not the misleading
\((P_K\phi P_K)^2\) outer-shell moments. Kinetic, harmonic-potential and
coupling expectations check the Ritz energy decomposition. Bare field
variance \(1/(2M\omega)\), induced variance, nonzero oscillator excitation
and energy lowering distinguish a genuinely quantum dressed state from a
prescribed classical shift. SU(4)-averaged \(\langle\phi_a\rangle\) and
\(\langle A_0\rangle\) vanish, and covariance is isotropic; no ordered beta
frame is inferred.

Actual boundary-ladder residues are

\[
 Z_{\rm add}={1\over6}\sum_i\|V_3^\dagger
       (I_{\rm osc}\otimes c_i^\dagger)V_2\|_F^2,\qquad
 Z_{\rm remove}={1\over6}\sum_i\|V_1^\dagger
       (I_{\rm osc}\otimes c_i)V_2\|_F^2.
\]

Here \(V_N\) is the recovered actual ground frame. Per-flavour residues,
transition singular values and all-final-state weights are reported. The
all-final summed weight is exactly two in each direction by physical
occupation and CAR. Ground residues need not equal two: excited final
oscillator/matter states carry the remainder. These are finite one-cell
addition/removal matrix elements, not relativistic pole residues or an
asymptotic scattering transfer.

## Adverse single-field control

Retaining only
\(T=\operatorname{diag}(3,-1,-1,-1)/\sqrt{24}\) destroys the full15 Casimir
mechanism. Its actual sector weights are sums of occupied flavour weights.
For \(N=1,3\), the largest squared weight is \(9/24=3/8\). For \(N=2\),
it is \(4/24=1/6\). A single oscillator can be shifted exactly in each
weight eigenspace, so its sector ground energy is
\(\omega/2-\mu^2\max(Q_N^2)/(2\kappa)\), giving

\[
 \boxed{E_{1,3}^{\rm single}-E_2^{\rm single}
       =-{5\mu^2\over48\kappa}<0\quad(\mu\ne0).}
\]

The receipt derives those weights from the same actual primitive generators
and includes a cheap cutoff-32 one-oscillator diagonalization. Its finite
agreement is a control, while the displacement formula proves the exact
negative gap. The full15 positive result cannot be attributed to any generic
single field or tracelessness alone.

## Reproduction and boundary of this result

Run with a Python environment containing NumPy, SciPy and SymPy from the repository root:

```sh
python proposals/P256-qed-backward/quantum_adjoint_mass.py \
  --output proposals/P256-qed-backward/quantum_adjoint_mass_receipt.json
```

The parent executed all defaults successfully. At total-excitation cutoff4
(3876 oscillator states), the charge-one/three Ritz gaps are
0.0122609907794 for \(\mu=.2\), and 0.0466020296094 for \(\mu=.4\).
The corresponding **infinite-Hilbert-space certified lower bounds** are
0.0102225575052 and 0.0208203932499. Ritz differences are not certificates.
The adverse single-field infinite gaps are respectively
−0.00416666666667 and −0.0166666666667.

`--M`, `--kappa`, `--UE` and `--tolerance` expose the stated parameters.
JSON binds source hashes, separates theorem from Ritz evidence, and exits
nonzero when numerical checks fail. Independent source review accepted the
primitive Casimirs, noncommuting-coordinate bound, optimized neutral trial,
full-shell spectra and residue controls; scientific execution is parent-owned.

The neutral multiplet is six and the charged multiplets four in the finite
weak-coupling consumer; this is not a unique vacuum or a selected physical
spin doublet. There is no chosen Clifford/spacetime spin, Dirac propagation,
three-dimensional ordered phase, full photon join, electron-photon
scattering transfer, or closure of issue 222/full QED. The valuable result is
an unrestricted, zero-preferring quantum field whose complete adjoint
coupling provably supplies a positive one-cell charge gap in the stated
interval, with primitive code matrix elements and a reproducible finite
quantum-spectrum consumer.
