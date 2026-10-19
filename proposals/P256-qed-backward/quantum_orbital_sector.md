# The same statistics selector with genuinely quantum orbital coefficients

The [exact energetic bosonic-code theorem](code_phase.md) does not require its
hopping coefficients to be fixed numbers. It survives their replacement by
quantum orbital operators, including operators that do **not** commute with the
orbital kinetic Hamiltonian. This joins one genuine quantum orbital Hamiltonian
to the same Pauli matter and physical photon rotors, rather than appending a
spectator oscillator or applying a static Born–Oppenheimer saddle.

The primitive quantum tensor law, Born rule, dynamics, scalar couplings and
specified graph remain model inputs. The theorem is exact on its stated domain;
the finite executable below is a consumer to be executed, not a recorded result.
It does not assert a quantum Clifford-selection or photon-phase theorem.

## 1. Construction and exact intertwiner

Retain the code words, occupation operators, admissible stabilizer signs,
Gauss law and matched code/physical-rotor paths of `code_phase.md`. Let
`H_phys,+` be the actual all-plus physical Gauss Hilbert space. Let `H_aux` be a
bosonic Hilbert space carrying a self-adjoint auxiliary Hamiltonian `H_a`.
For every directed path gamma use the **actual** primitive gauged transfer

\[
 D_\gamma=-iA_\gamma n_{\rm source}(1-n_{\rm target})U_\gamma.
\]

Here the code path and the physical rotor path are identical, apart from
internal tree links that carry no electromagnetic rotor. Auxiliary coefficients
`B_gamma` act only on `H_aux`; a complex coefficient is allowed provided its
adjoint term and domains/forms are included consistently. Define

\[
 H=H_E\otimes I+I\otimes H_a+V(n)\otimes I
   +\sum_\gamma\bigl(D_\gamma\otimes B_\gamma
                     +D_\gamma^\dagger\otimes B_\gamma^\dagger\bigr)
   -K\sum_p C_p\otimes I
   +\Delta\sum_p\frac{1-S_p}{2}\otimes I,
\quad K\geq0.
\]

There is no condition `[B_gamma,H_a]=0`, nor any need for the different
`B_gamma` to commute with one another. Tensor separation from the physical code
and rotors, plus the domain conditions below, is the relevant condition.
`V(n)` may include any bounded sector-independent occupation interaction on the
finite matter space. Extra auxiliary-dependent occupation terms are likewise
allowed under the same form/domain hypotheses, since `Q_eta` fixes occupations.

For an admissible sector `s`, choose the same link character eta as in the
fixed-coefficient theorem. The microscopic unitaries are

\[
 Q_\eta=\prod_{l:\eta_l=-1}Z_l,\qquad
 R_\eta=\exp\left(i\pi\sum_{l:\eta_l=-1}E_l\right),\qquad
 T_s=(Q_\eta R_\eta)\otimes I_{\rm aux}.
\]

Only physical spatial links occur in `R_eta`; internal tree edges can have
`eta=1`. On integral electric fields the rotor factor is a diagonal sign.
`Q_eta` fixes every occupation and `R_eta` every electric field, so both preserve
Gauss and the specified physical boundary conditions. They map the **actual**
all-plus domain onto sector `s`, not onto an assumed CAR matrix representation.

For every path, including improved and repeated-link paths when they define the
stated primitive transfer,

\[
 Q_\eta^\dagger A_\gamma Q_\eta=\eta_\gamma A_\gamma,
 \qquad R_\eta^\dagger U_\gamma R_\eta=\eta_\gamma U_\gamma,
 \qquad \eta_\gamma^2=1.
\]

Consequently `T_s` leaves each `D_gamma tensor B_gamma` unchanged. It also
leaves `I tensor H_a` unchanged, even when `B_gamma` drives genuine auxiliary
quantum transitions. The magnetic plaquette changes by
`R_eta^dagger C_p R_eta=s_p C_p`; the penalty equals `Delta N_def` in sector
`s`. Thus, on the domains/closed forms specified next,

\[
 \boxed{T_s^\dagger H_sT_s
       =H_+ +2K\sum_{p:s_p=-1}C_p\otimes I
              +\Delta N_{\rm def}I.}
\]

Physical rotor products are unitary before electric truncation and contractions
after the noncyclic compression used here. Therefore `-I<=C_p<=I`, giving

\[
 \boxed{T_s^\dagger H_sT_s\geq
 H_+ +(\Delta-2K)N_{\rm def}I.}
\]

When the Hamiltonians are bounded below this implies the corresponding bound
on the infima of spectra. If ground eigenstates exist, the undesired sector's
ground energy is at least `(Delta-2K) N_def` above the all-plus ground. With
compact resolvent the min–max principle gives the same bound on every ordered
eigenvalue. This is **not** separation of entire bands: higher all-plus levels
can overlap undesired-sector levels. No smallness assumption on the hopping or
orbital coupling is used in the comparison.

## 2. Domains: bounded, truncated and unbounded cases

### Bounded auxiliary coefficients

Take a finite graph, finite matter/code factors and either integral untruncated
rotors or the explicitly noncyclic finite electric domain. All primitive
`D_gamma` and `C_p` are bounded. Let every `B_gamma` be bounded and let `H_a` be
self-adjoint. The free tensor sum `H_0=H_E tensor I+I tensor H_a` is defined by
its joint spectral calculus: these factors strongly commute because they act
on different Hilbert spaces. A finite bounded symmetric perturbation defines
`H` on `D(H_0)`. `T_s` preserves that domain and commutes with `H_0`; hence the
identity above is an operator identity with the conjugated operator domain.
For ground-energy selection additionally assume `H_a` is bounded below.
If it is not, the identity can still be meaningful but a bottom at minus
infinity supplies no useful ground-sector conclusion.

This covers an infinite quantum oscillator with a bounded function such as
`t+g tanh(X)`: it need not commute with oscillator kinetic energy. A finite
orbital truncation makes all coefficients bounded and is an exact instance of
the theorem for that **truncated Hamiltonian**.

### Finite Galerkin truncations

Use the **same** auxiliary subspace and the same electric cutoff in every code
sector. Their projections commute with `T_s` because the auxiliary projection
acts on another factor and the electric projection is diagonal in `E`. Compress
the actual Hamiltonian/form, including every path and its adjoint. The same
unitary identity and margin hold exactly at each cutoff. No commutation between
the auxiliary projection and `H_a` is needed. Nested energy convergence needs a
nested compression of one specified Hamiltonian; replacing a compressed
quartic by the fourth power of a compressed coordinate changes its boundary
terms and does not supply that variational argument.

### Unbounded coefficients: closed-form extension

For ground selection take `H_a` semibounded and shift the free tensor sum to a
nonnegative closed form `h_0`, with form domain `Q_0`. Specify the auxiliary
coefficient forms and the physical path couplings on a dense form core
`D_c subset Q_0`. Require:

1. `D_c` and `Q_0` are invariant under the bounded physical stabilizers, sector
   projectors and `T_s`, with identical admissible physical domains in matched
   sectors. Primitive path shifts preserve the appropriate free electric form
   domain; finite electric truncations satisfy this automatically.
2. The total hopping form `w` is symmetric, extends continuously to `Q_0`, and
   obeys `|w[psi]|<=a h_0[psi]+b||psi||^2` with `a<1`. Sector-independent
   unbounded extra terms must be included in this same estimate or in the free
   semibounded form. These are **form** conditions, not an assertion that all
   formal operator products are self-adjoint on an unspecified intersection.
3. On the core the auxiliary coefficients really act on the auxiliary factor
   and the matched microscopic paths obey the displayed `Q/R` identities.
   In particular `T_s` maps the specified coefficient-form domains to themselves.

KLMN then defines the unique self-adjoint semibounded Hamiltonian represented
by the closed form `h_0+w` plus the bounded magnetic and penalty terms. The
core equality extends by form continuity to all of `Q_0`. Unitary conjugation
of closed forms and their associated operators proves the exact identity and
the bound. Its operator domain is the domain supplied by that representation
theorem, and the conjugated domain is `T_s^dagger D(H_s)`.

An alternative sufficient operator construction is a specified common invariant
core with a symmetric total interaction that is relatively `H_0`-bounded with
relative bound below one; Kato–Rellich gives the self-adjoint closure. Merely
writing "unbounded self-adjoint coefficients commute with rotors" is not this
construction. Tensor-factor spectral commutation does not by itself establish
self-adjointness, semiboundedness or a domain for their Hamiltonian sum.

### A concrete coercive quartic orbital family

Let `x in R^d`,

\[
 H_a=-\sum_a\frac{1}{2M_a}\partial_{x_a}^2+V(x),\qquad
 V(x)\geq c\|x\|^4-C,\quad c,M_a>0,
\]

and take a finite list of coefficients
`B_gamma(x)=t_gamma+sum_a g_gamma,a x_a`, including their complex conjugate
adjoints. For every epsilon>0,
`|B_gamma(x)|<=epsilon ||x||^4+C_epsilon`. Since each physical path is bounded,
the total matter coupling is infinitesimally form-bounded against the free
quartic Hamiltonian, with no restriction on finite `g_gamma,a`. Smooth compactly
supported orbital functions tensor finite-support rotor/code states in the
specified Gauss sector provide a common form core. Physical shifts preserve
finite rotor support and the electric quadratic form domain. `Q_eta R_eta`
changes only bounded phases on this core. The proof therefore applies to one
unbounded, genuinely coupled quantum Hamiltonian, not only to its finite
matrices.

The unrestricted [soft orbital Gram potential](soft_budget.md) is one such
family. With matrix-unit coordinates normalized by `tr(T_a T_b)/4=delta_ab`,
`tr G=||x||^2`, and

\[
 V_{\rm elastic}=\frac\kappa2\|G-g_0I_4\|_F^2
 \geq\frac\kappa8\|x\|^4-\kappa g_0\|x\|^2+2\kappa g_0^2.
\]

Adding a positive orbital kinetic term and letting these coordinates enter the
same primitive bosonic transfers meets the closed-form hypotheses on a finite
graph. If `UE>0` on every physical rotor, the confining orbital potential and
electric energy give compact resolvent on the finite-graph Gauss subspace; the
infinitesimal form perturbation preserves its compact form-domain embedding.
Hence normalizable ground states and ordered eigenvalue bounds exist there.
This lifts the **statistics selection** result to quantized soft orbitals. The
separate claim that the quantum ground is concentrated near the BO-selected
Clifford orbit still requires a dynamical/fluctuation argument; quantization
does not automatically make the BO minimizer an exact orbital eigenstate.

## 3. Finite primitive consumer, not matrix forwarding

`quantum_orbital_sector.py` uses `code_phase.CoupledSquare` with **both** actual
stabilizer sectors of its Gauss physical raw square, fixed `N=2`, background
`(1,0,1,0)`, four independent physical photon rotors and the primitive Pauli
edge words. The only discarded transfers are the explicit noncyclic electric
boundary shifts. A microscopic term leaving Gauss raises in `insert`; no
silent Gauss projection or matrix-CAR constructor is introduced.

The Hamiltonian is

\[
 H_L=H_{\rm square}(h=.6)\otimes I_L+I\otimes H_{a,L}
       +g D\otimes X_L,
 \qquad D=\frac{\partial H_{\rm square}}{\partial h}.
\]

`D` is assembled from the four actual `transfer_matrix(edge)` operators and
their original directed phases. An independent call to the actual primitive
Hamiltonian at `h+1` checks `H(h+1)-H(h)=D`. The two two-link and one three-link
improved transfers, their original coefficients, onsite terms, electric term,
magnetic term and code penalty remain in `H_square(h=.6)`. In particular this
is not the invalid replacement of the whole matter operator by its one-link
derivative.

The auxiliary Hamiltonian compresses a quantum oscillator with
`omega=1.3`, quartic coefficient `.08`:

\[
 H_a=\omega(N+1/2)+.08X^4,\qquad
 X=(a+a^\dagger)/\sqrt{2\omega},\qquad h=.6+gX.
\]

The full untruncated `H_a` does not commute with `X`; the report checks a
nonzero commutator in every finite consumer. A padded Fock space evaluates the
true `P_L X^4 P_L`, retaining excursions above the cutoff before returning.
This makes cutoffs `L=4,8,12` nested Galerkin consumers of the same Hamiltonian.
The default weak/strong couplings are `g=.15,.9`. The photon/code constants are
`K=.7, UE=.8, Delta=2`, so the exact defect margin is `.6` at **every** cutoff
and coupling. This constant is a proved bound, not a claimed measured gap.

The map derives the occupation-dependent phase from
`<n,-|Q_eta|n,+>` and combines it with the physical rotor sign `(-1)^E_0`.
It compares the whole coupled operator identity, its operator lower bound,
ordered spectra, actual ground gap, and the full spectrum against the union of
both independently constructed sector spectra. The map extends by `tensor I_L`,
not by substituting the two sector matrices for one another.

The actual all-plus ground supplies non-spectator evidence:

- orbital-versus-full-physical linear/von Neumann entanglement entropy;
- connected `X` covariances with the primitive hopping derivative, the physical
  photon plaquette cosine and electric energy;
- the coupled energy derivative `dE_0/dg=<D tensor X>`, compared with a centered
  ground-energy difference from the full sector Hamiltonian;
- retained top-Fock weight, expected truncated-CCR boundary defect `L p_top`,
  and probability that any physical electric field lies at its boundary.

These boundary quantities are reported, not silently suppressed and not
converted into an unproved continuum error bound. Small oscillator boundary
weights alone do not remove the separately fixed rotor cutoff. The script also
reports changes of the actual ground energy along the nested cutoff sequence.

Two adverse microscopic changes must violate the operator identity:

1. Multiply one actual quantum path by the code stabilizer `S`. This flips that
   path's coefficient only in the undesired sector and leaves its matched
   rotor transfer unchanged. It is a genuine wrong **sector path sign** and
   preserves Gauss, so it cannot be hidden by projection. A uniform sign change
   in both sectors would remain an allowed common coefficient and is correctly
   **not** used as a purported counterexample.
2. Pair the code `Z_0` with the rotor sign on edge 1 instead of edge 0. Its
   magnetic curl still flips, but its matter path signs no longer cancel.

## 4. Complete execution commands and interpretation

From the repository root, the parent can run:

```sh
/home/administrator/substrate-framework/.venv/bin/python proposals/P256-qed-backward/quantum_orbital_sector.py --output proposals/P256-qed-backward/quantum_orbital_sector_receipt.json
/home/administrator/substrate-framework/.venv/bin/python proposals/P256-qed-backward/quantum_orbital_sector.py --rotor-cutoff 2 --oscillator-cutoffs 4 8 12 --couplings .15 .9 --output proposals/P256-qed-backward/quantum_orbital_sector_rotor2_receipt.json
```

The first command executes all six default consumers. The second varies the
independent photon-electric boundary while preserving the same orbital/code
construction. The CLI emits a JSON report, source hashes and individual
physical checks; it exits nonzero if any claimed identity, gap bound,
non-spectator requirement or adverse discriminator fails. Both commands have
now naturally completed: all six default consumers and all six independent
rotor-cutoff-2 consumers pass their actual physical checks. The preserved
receipts record noncommuting dynamics, actual entanglement/covariances,
operator identities, measured spectra and cutoff boundaries.

The useful next physical join is to quantize the unrestricted soft-budget
orbital coordinates on the same decorated primitive graph, then control their
fluctuations in the current/photon consumer. The theorem has already removed
the potential incompatibility: noncommuting quantum hopping dynamics do not
weaken the energetic selector of the same bosonic statistics sector.
