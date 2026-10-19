# Energetic code selection survives weak microscopic sector mixing

This extends [code_phase.md](code_phase.md), rather than replacing its full
matter/rotor mechanism by a projected Hamiltonian. The microscopic Hamiltonian
can now fail to commute with the code stabilizer: its physical ground state and
its genuinely low excited states still approach the desired code sector as the
mixing tends to zero. The theorem controls that approximation quantitatively.
The executable keeps **both** sectors of the actual gauge-constrained square.

## 1. The separation that the previous mechanism really supplies

Work on the physical Gauss domain, with its specified background, topology,
parity and electric boundary conditions. Let P be the joint all-positive code
projector and Q=I-P; these are not Gauss projectors. For the unperturbed full
bosonic Pauli/compact-rotor Hamiltonian H0 of `code_phase.py`, [P,H0]=0. Write

\[
 H_P=PH_0P|_{P\mathcal H},\quad H_Q=QH_0Q|_{Q\mathcal H},\qquad
 E_0=\min\sigma(H_P),\quad
 \gamma=\inf\sigma(H_Q)-E_0>0.
\]

The existing microscopic Q_eta R_eta conjugation gives

\[
 E_s\ge E_0+(\Delta-2K)N_{\rm def},
 \qquad \gamma\ge\Delta-2K
\]

on a nontrivial simply connected finite domain with at least one defect in each
complementary sector. Unequal local margins give the corresponding minimum
positive sum. Global unpenalized sectors must instead be included in P or
separated by a proved additional mechanism. The square has one loop, hence one
complementary sector. We use its **actual** gamma when computing the finite
bounds, while also checking the proved Delta-2K lower bound.

No claim that `max spectrum(H_P) < min spectrum(H_Q)` is needed or made. High
selected-sector matter/photon states may overlap defective bands. The relevant
separation is between the **ground energy**, or a specified genuinely low
energy window, and the bottom of the complementary spectrum.

The following ground-sector estimate also allows a degenerate P ground space.
For a unique ground-state continuation only, impose the additional, distinct
assumption that E0 is a simple isolated eigenvalue of H0 and define

\[
 g_0=\min(\sigma(H_0)\setminus\{E_0\})-E_0>0.
\]

An internal selected-sector excitation can set g0 smaller than gamma. The finite
consumer checks this rather than identifying these two gaps.

## 2. General bounded Gauss-preserving perturbation

Let H=H0+V, with V Hermitian, preserving the physical Gauss domain, and
`epsilon=||V||` its **spectral operator norm**, not a Frobenius norm. For any
normalized perturbed ground eigenstate psi=x+y, x=P psi, y=Q psi, its energy E
obeys the elementary variational inequalities

\[
 E_0-\epsilon\le E\le E_0+\epsilon.
\]

The upper bound follows by evaluating H on an unperturbed P ground state; the
lower bound follows from H0>=E0 and V>=-epsilon. In the Q equation,

\[
 (H_Q+QVQ-E)y=-QVPx.
\]

The operator on the left is bounded below by
`gamma-2 epsilon`, using E<=E0+epsilon and QVQ>=-epsilon. If
`2 epsilon < gamma`, it has a bounded inverse. Therefore

\[
 \|y\|\le {\epsilon\over\gamma-2\epsilon}\|x\|,
 \qquad
 \boxed{\|Q\psi\|^2\le
 {\epsilon^2\over(\gamma-2\epsilon)^2+\epsilon^2}.}
\]

The normalization denominator is retained: this is a probability bound, not
just an unnormalized amplitude estimate. Thus ground leakage is O(epsilon^2)
without exact microscopic code conservation. A sufficient uniform estimate
uses the existing proved margin Delta-2K in place of gamma.

For more economical constants one may use `a=||PVP||`, `b=||QVQ||`,
`c=||QVP||`: E<=E0+a and the denominator is `gamma-a-b`, while the numerator is
c. The useful condition is `a+b<gamma`, not an alleged separation of entire
P/Q bands. This permits large sector-internal dynamics when they do not close
the relevant bottom-of-spectrum separation.

### Ground identity is an additional, controlled statement

For the simple isolated E0 above, Weyl's variational eigenvalue estimates give

\[
 E_1(H)-E_0(H)\ge g_0-2\epsilon>0
\]

when `2 epsilon<g0`. The perturbed ground therefore remains unique and cannot
exchange its identity with another eigenstate. Let P0 be the unperturbed
one-dimensional ground projector. Applying the same block argument to
I-P0 gives the general fidelity-loss bound with denominator `g0-2 epsilon`.
For the purely off-diagonal perturbation below E<=E0, improving that denominator
to `g0-epsilon`:

\[
 1-|\langle\phi_0,\psi\rangle|^2
 \le {\epsilon^2\over(g_0-\epsilon)^2+\epsilon^2}.
\]

This is not an assertion that low-energy electrons or photons have a
volume-independent excitation gap: g0 and gamma have different jobs.

## 3. A particularly usable purely sector-changing perturbation

Suppose PVP=QVQ=0 and let c=||QVP||. A P-ground trial gives E<=E0, while the Q
resolvent has denominator at least gamma. Hence, for every ground eigenstate,

\[
 \boxed{w=\|Q\psi\|^2\le {c^2\over\gamma^2+c^2}},
 \qquad
 w\le {c^2\over(E_0+\gamma-E)^2+c^2}.
\]

The second version uses the measured energy. For the energy shift, normalized
x+y has expectation bounded below by

\[
 E_0\|x\|^2+(E_0+\gamma)\|y\|^2-2c\|x\|\|y\|.
\]

The minimum is the lower eigenvalue of the real 2-by-2 matrix with diagonal
entries E0,E0+gamma and off-diagonal entries -c. Consequently

\[
 \boxed{E_0-\rho(\gamma,c)\le E\le E_0},\qquad
 \rho={\sqrt{\gamma^2+4c^2}-\gamma\over2}
 ={2c^2\over\sqrt{\gamma^2+4c^2}+\gamma}
 \le {c^2\over\gamma}.
\]

These two inequalities hold at every c, but become useful small-leakage and
quadratic-energy-shift estimates at small c/gamma. They do not assert unique
ground continuation at large c; that needs the separate g0 condition.

The eliminated block also gives the exact Feshbach equation below the Q edge:

\[
 [H_P-E-PVQ(H_Q-E)^{-1}QVP]x=0,\qquad
 y=-(H_Q-E)^{-1}QVPx.
\]

The self-energy is positive before its subtraction, explaining the downward
energy shift. For V=lambda Z in the executable, with isolated unperturbed
normalized phi0, finite-matrix analytic perturbation theory or expansion of
this resolvent gives

\[
 E(\lambda)=E_0-\lambda^2
 \langle\phi_0,ZQ(H_Q-E_0)^{-1}QZ\phi_0\rangle+O(\lambda^4),
\]
\[
 w(\lambda)=\lambda^2
 \| (H_Q-E_0)^{-1}QZ\phi_0\|^2+O(\lambda^4).
\]

Here `S H(lambda) S=H(-lambda)` makes the isolated ground energy and code leakage
even functions, removing odd orders. The coefficients are computed directly
from the actual matrix, not fitted to the observed leakage curve. In this
example Z is a unitary bijection of the sectors, so both coefficients are
strictly positive. The CLI checks the small-amplitude coefficient and slopes.

## 4. Legitimate low-energy excited states are retained, not discarded

For any eigenstate of H at energy E<E0+gamma-b, the Q equation gives

\[
 \|Q\psi_E\|^2\le
 {c^2\over(E_0+\gamma-b-E)^2+c^2},\qquad b=\|QVQ\|.
\]

Thus every eigenstate at E<=E0+Lambda+epsilon with
`Lambda+b+epsilon<gamma` has a uniform version of this bound. This does **not**
cover overlapping high bands. For the executable b=0, so each low state's
actual energy supplies its denominator.

For completeness, let the first r unperturbed physical eigenvalues all be
selected-sector states below the complementary edge. Suppose the cluster has
an external separation

\[
 s=E_{r}(H_0)-E_{r-1}(H_0)>0
\]

with zero-based indices. At the fixed midpoint of those two levels, precisely
r perturbed levels remain below the midpoint when `2 epsilon<s`; ordered
energies shift by at most epsilon. This follows directly from min-max: the
r-th level cannot cross up and the next level cannot cross down. Degeneracies
**inside** the retained cluster are harmless; do not demand continuation of an
arbitrary eigenbasis there.

There is also an elementary subspace-retention bound. Let L0 project onto the
unperturbed r-level cluster. For each normalized perturbed low eigenvector,
the (I-L0) block has inverse separation at least `s-2 epsilon`; its probability
outside L0 is at most `d=epsilon^2/((s-2 epsilon)^2+epsilon^2)`.
For a matrix U of the r orthonormal new low eigenvectors,
`||(I-L0)U||^2 <= ||(I-L0)U||_F^2 <= r d`. Thus the smallest principal cosine
between the old and new low subspaces obeys

\[
 \cos^2\theta_{\max}\ge\max(0,1-rd).
\]

The executable checks this bound as well as level counting. It protects the
whole low subspace, including internal mixtures or degenerate eigenbases.

`sector_stability.py` retains every selected level strictly below the defective
edge, chooses this cluster's actual midpoint, and diagonalizes the complete
physical matrix after perturbation. It compares the count, ordered energies,
all individual outside-sector weights and the principal cosines between the
old and new low subspaces. No legitimate low selected state is erased by
imposing the code constraint at the Hamiltonian boundary.

## 5. A truly Gauss-permitted microscopic bosonic operator

The actual consumer imports `CoupledSquare` and `PauliWord` from the existing
files. It uses the same four primitive edge qubits, four **noncyclic** rotors,
complex nearest and multi-link transfers, magnetic term, electric term,
integer background (1,0,1,0), N=2, and physical raw-basis enumeration.

The added operator is **one bare code-qubit Z on edge zero**:

\[
 H(\lambda)=H_0+\lambda Z_0^{\rm code}.
\]

Every B_v is a diagonal Z word, so this Z commutes with occupations. It acts on
no rotor, so it commutes with E and every Gauss operator
`G_v=div E_v-(n_v-b_v)`. It is Hermitian, has norm one, and anticommutes with the
square loop stabilizer because that stabilizer has X/Y support on this edge.
Thus it is a genuine local microscopic bosonic perturbation that changes the
code sector while preserving the physical state domain, total charge and flux.
Its P/P and Q/Q blocks vanish. It is not the rotor rotation R and is not the
nonlocal many-edge sector intertwiner for a general sign assignment.

This choice avoids disguising the violation caused by an **undressed** bare X:
X on a single edge changes the endpoint occupations but not electric flux,
hence violates Gauss. The CLI applies that bare X to every physical raw state
and reports its nonzero Gauss violation rather than projecting it away.
`code_phase.py`'s alternative bare-X *transfer dressed with the appropriate
rotor and occupancy projectors* would also be admissible; its nonzero
cross-sector block is exactly why the present stability theorem is useful.
The new example needs no projection or alteration of that existing code.

Each Z insertion uses the existing physical-state insertion boundary. An
escape is an error, never a silently dropped matrix element. The raw physical
Hamiltonian retains P and Q, unlike the selected spectral tree representation
used for the Compton response in [pauli_compton.md](pauli_compton.md). This
supplies a controlled energetic justification for low-energy use of that
selected algebra; it does not yet compute perturbatively corrected Compton
amplitudes or prove infinite-volume scattering stability.

## 6. Strong-coupling adverse behavior has its own bound

The same executable also takes a genuinely strong lambda, without demanding
that it satisfy the weak norm or isolated-cluster hypotheses. Center H0 by a
scalar alpha and put `L=||H0-alpha I||`. For lambda>0, a trial vector in the
Z=-1 space gives `E-alpha<=-lambda+L`. The lower estimate on the actual ground
expectation is `E-alpha>=-L+lambda <Z>`. Therefore

\[
 t=\Pr(Z=+1)={1+\langle Z\rangle\over2}\le {L\over\lambda}.
\]

Since S anticommutes with Z, it connects only opposite Z spaces:
`|<S>|<=2 sqrt(t(1-t))<=2 sqrt(L/lambda)`. With the original code-sector
weight `w=(1-<S>)/2`,

\[
 \boxed{|w-\tfrac12|\le\sqrt{L/\lambda}.}
\]

Strong local sector mixing thus approaches half weight outside the original
sector: energetic protection is not immunity to arbitrary mixing. The CLI
chooses lambda=10000 max(L,gamma), checks this independent bound and substantial
leakage, and labels the weak norm/isolated-continuation estimates inapplicable.
The pure off-diagonal variational energy/leakage inequalities remain true but
need not be sharp there. This adverse case is not a failure of their small
coupling conclusion.

## 7. Parent execution and evidence boundary

From the repository root:

```sh
/home/administrator/substrate-framework/.venv/bin/python \
  proposals/P256-qed-backward/sector_stability.py \
  --cutoff 1 --magnetic 0.7 --electric 0.8 --delta 2.0 \
  --hopping 0.6 --improved 0.19 --phase 0.23 \
  --output proposals/P256-qed-backward/sector_stability_receipt.json
```

The consumer emits actual full spectra, energy shifts, leakage, exact resolvent
coefficients, Feshbach residuals, Gauss preservation/violation, retained low
levels, and the adverse strong-mixing result. Failed inequalities or controls
raise with the complete report. Weak amplitudes are chosen from the measured
gamma, g0 and low-cluster gap, not from a guessed whole-band gap. Optional
`--cutoff 2` changes the finite domain; it is not a continuum limit.

The parent executed the displayed finite-square CLI naturally to exit0.
The actual complementary ground gap is `gamma=2.03834455`, the isolated
ground gap `.802283713`, and six legitimate selected low states are retained.
For local mixing lambda `.00652833028`, outside-sector ground weight is
`6.36457293e-6`, below the pure-offdiagonal bound `1.02575724e-5`; the energy
shift is `-1.59728802e-5`. High selected/defective bands overlap, so this
consumer genuinely distinguishes ground-edge protection from a false
whole-band separation. All executed physical and spectral controls pass.
Source: `262e9b14564dbdf8bd4c50ba9f3d4c64c1649e7c10e15296bf729f546f623680`.

## 8. Constructive scope

The code-phase mechanism now selects not just an exactly invariant sector but
an **approximately selected robust low-energy algebra** under bounded,
Gauss-permitted microscopic violations. The new small parameter is the total
sector-mixing norm relative to the defect ground separation; no matter-hopping
norm enters the original separation proof. Preserving physical Gauss and
retaining legitimate low matter/photon excitations are independent of demanding
exact code conservation.

The supplied finite theorem does not derive the primitive bosonic Pauli law,
quantum postulates, Delta/K hierarchy, or small microscopic mixing. An extensive
sum of local perturbations can have a volume-growing total norm; a uniform
thermodynamic/local-stability theorem needs additional locality/gap structure,
not replacement of epsilon by a single-site coefficient. Nor is this a photon
Coulomb-phase, thermal preparation, dressed-charge matching or asymptotic
scattering proof. It is a positive, controlled extension on the **existing
coupled physical Hamiltonian**, directly testable at the full finite-spectrum
consumer, and leaves the joined electron-photon campaign active.
