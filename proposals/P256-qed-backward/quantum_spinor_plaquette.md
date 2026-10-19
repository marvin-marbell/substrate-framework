# Fully quantum four-orbital charged states and a compact plaquette photon

This is a **constructive finite planar model**, extending `quantum_compact_photon.py` from spinless matter to the actual scalar four-orbital `RotationBand` matrix-unit law. The complete default CLI has executed successfully on all six charged/neutral cutoff cases. It is not a continuum spin theorem, a three-dimensional phase, or a QED cross section.

## Reproducible consumer and ownership

From `proposals/P256-qed-backward`:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python quantum_spinor_plaquette.py \
  --output quantum_spinor_plaquette_receipt.json
```

Default electric cutoffs are **1 and 2**, with `m=0.8`, `r=0.04`, `v=0.08`, `K=0.25`, `UE=0.6`, `Delta=1.2`. `--check-minus` additionally diagonalizes the neutral discarded sector; without it the complete sparse sector-intertwining operator identity is still checked. `--cutoffs 1 2 3` strengthens the cutoff comparison. Scalar `--magnetic`, `--electric`, `--delta`, `--m`, `--r`, `--v`, and `--lanczos-steps` expose the actual model. Enlarging K requires `Delta>2K`; the improved charged-gap theorem below has **no magnetic-K loss**, so remains useful in the weaker-angle regime. That regime can require larger electric cutoffs.

The only new implementation is `quantum_spinor_plaquette.py`. Its microscopic builder imports `EdgeWords`/`PauliWord` indirectly from `pauli_compton.py`, hence the existing `primitive_pauli_words` law; the scalar onsite/link law imports `RotationBand` and its matrix-unit transitions from `rotation_band.py`. SciPy supplies sparse storage, low-energy eigensolves, conjugate-gradient resolvents, and a small **Lanczos tridiagonal**, never a dense full Hamiltonian spectrum. Deterministic starts are fixed. Failed identities, absent photon weight, absent charged creator residue, nonpositive charged gaps, unresolved multiplet structure, or unconverged resolvents raise and exit nonzero. JSON is written only after the complete campaign succeeds.

Units are `hbar=a=1`. `m,r,v,K,UE,Delta`, excitation frequency and source j have energy units; integer E, angle phi, B and occupations are dimensionless. Static B susceptibility has inverse-energy units. The scalar speed v is an energy in lattice units, not a measured light speed.

## Actual graph, code states and boundary conditions

Four open spatial cells are `(0,0),(1,0),(1,1),(0,1)`. Cell c has physical orbital vertices `4c,...,4c+3`. Three star/tree edges join each cell root `4c` to its other three orbitals. Four physical edges join successive cell roots around the square. One extra leaf, vertex 16, joins root 0 and is a **neutral parity marker**. The graph has 17 vertices, 17 edge qubits, and one independent cycle.

Physical N is 7, 8 or 9. Marker occupation is `N mod 2`, so the primitive graph's total even parity permits both physical parities. The marker is not an electron and contributes neither charge nor electric energy. It is T-even. This is an explicit boundary parity completion, not an assertion that the marker is a microscopic charged species.

Use the actual primitive occupation words `B_v` and the actual oriented cycle word S, with `S^2=1`. Remove the edge `(0,12)` to obtain a tree. Solve the occupation-bit constraints on that tree, map its bits back into the full edge ordering, and choose the smaller of the two S-related raw words q and q'. If `S|q>=a_q|q'>`, the normalized code state is

`|n,s> = (|q> + s a_q |q'>)/sqrt(2)`, for `s=+1,-1`.

Every matrix element is evaluated by applying the actual Pauli word to these **two raw code words** and taking their overlap. This is a compressed exact code basis, not a canonical CAR Hamiltonian. The tree occupation solver, projector normalization, code transitions, primitive path phases, and sparse Hermiticity are assertions on the runnable consumer.

The only electromagnetic links are four physical square edges, oriented `0->1->2->3->0` in cell labels. There is a fixed outward boundary electric flux `q_ext=N-8` at cell 0; the boundary field is not dynamical. Each cell has background occupation 2. The **complete** Gauss equations are

`E_c - E_(c-1) + q_ext delta_(c0) = n_c - 2`.

Thus their sum is exactly `q_ext=N-8`. Define `rho_c=n_c-2-q_ext delta_(c0)`, and use a single loop integer ell:

`(E0,E1,E2,E3) = (ell, ell+rho1, ell+rho1+rho2, ell-rho0)`.

The implementation generates physical occupation combinations, solves these affine equations, and intersects their four noncyclic cutoffs `|E_c|<=L`. It does not enumerate a brute-force qubit/rotor tensor product. Internal orbital edges have no separate electromagnetic rotor: these are co-located internal transitions; Gauss charge is cell occupation. The cutoffs are hard electric endpoints, **not cyclic clock wraparound**. The basis is nested as L increases.

## Same microscopic matter and nonlinear photon Hamiltonian

Let `beta=MIXING`, `alpha_x,alpha_y=HOPPING[0:2]` be the existing scalar orbital matrix-unit combinations. The honest two-direction nearest-neighbour reduction has

`h0=(m+2r) beta`, `T_d=-r beta/2-i v alpha_d/2`, for d=x,y.

The absent z-direction link and its Wilson onsite r beta are removed. The hopping remains an explicitly selected orbital law, not an emergent isotropy claim. All onsite and link entries become primitive endpoint transfers. For an occupied source a and empty destination b, the microscopic code action is

`-i A_path n_a(1-n_b)`.

The chosen path is the cell star path, the actual direct square edge, and the destination star path; it never silently substitutes the other route around the plaquette. The rotor shifts lower the oriented electric field along the matter path, with the reverse shift for reverse orientation. Onsite/internal-cell transitions do not shift E. The sparse builder rejects any transition escaping the full Gauss basis.

W is the actual Gauss loop shift `E_c -> E_c+1` for every c, zero at the hard boundary. B and H are

`B=(W-Wdag)/(2i)`,

`H = H_orbital[Pauli,rotor] + UE/2 (sum_c E_c^2 + q_ext^2) - K/2 (W+Wdag) + Delta(1-S)/2`.

The boundary energy is present in charged energy comparisons. The magnetic operator is the full nonlinear compact `-K cos(phi)`: no harmonic, Gaussian, frozen-gauge or factorized-vacuum substitution constructs H. The matter code states and electric fields are the actual state owners; the ground multiplets come from this H, not a supplied sea wavefunction.

## Exact selection of the code sector

Let Q be Z on the omitted physical edge `(0,12)` and R the diagonal rotor parity `(-1)^E3`. Q flips S. Q changes the sign of every primitive matter path using that edge; R changes the sign of its electric shift, cancelling the change. Both leave the electric/boundary energies unchanged. R reverses the sign of W. Hence the actual raw-word-derived unitary F=QR between code sectors satisfies

`Fdag H_- F = H_+ + Delta I + K(W+Wdag)`.

At a noncyclic cutoff `||W||<=1`, so

`inf spectrum(H_-)-inf spectrum(H_+) >= Delta-2K`.

This is an operator theorem for **each N and every finite cutoff**. `Delta>2K` therefore selects S=+ before using the projected consumer. The CLI also builds the entire neutral S=- sparse operator and checks the exact equality, and optionally measures its lowest energy. It does not infer the selection from a projected ground spectrum.

All actual primitive A paths commute with S; onsite occupations do too. B acts only on the electric loop. Thus neither B nor any hopping-phase derivative/current can couple the selected sector to the discarded sector. The zero cross-sector response entry in JSON is an **algebraically proven identity**, not a numerical spectrum observation.

## Constructive sufficient positive charged gap

Set `M=m+2r>0`. The onsite one-particle energies are eight copies each of `-M,+M`; therefore its minimum in N=8 is `-8M`, and its minimum in N=7 or 9 is `-7M`.

For one spatial link the four singular values of T sum to `2 max(r,v)` when `r,v>=0`. Local orbital rotations are justified by the derived matrix-unit code law. Each rotated two-orbital hopping channel, dressed with its **common noncyclic rotor contraction**, has norm at most its singular value: it connects orthogonal occupied-empty and empty-occupied subspaces. The four-link hopping norm is consequently at most `8 max(r,v)`.

The [integer-Gauss photon theorem](compact_charge_gap.md) compares every occupation's actual shifted-loop domain with the neutral photon block. A positive closed-walk expansion and four-factor Holder inequality give `Tr exp(-t H_a) <= Tr exp(-t H_0)`, hence `inf H_a >= epsilon_0`, even at the noncyclic finite cutoff. The neutral trial fills both negative onsite orbitals per cell and uses the **actual neutral photon ground vector**, not E=0. It has energy `-8M+epsilon_0`, since hopping changes cell occupations. Either charged sector has boundary energy `UE/2`. Therefore

`E_(7 or 9) - E_8 >= M + UE/2 - 8 max(r,v)`.

This improves the previous estimate by K and supplies a **uniform-in-cutoff, magnetic-independent** positive charged gap. The default bound is `.54`; the weak-angle example `UE=.0072,K=.8888888889` retains bound `.2436` whereas the old estimate was negative. The CLI reports the analytic condition, both actual charged gaps, and rejects a measured violation. Sparse eigenpair residuals and ARPACK selection do not constitute rigorous spectral enclosures; the operator theorem supplies the rigorous bound in its stated domain.

## Derived time reversal and charged Kramers states

The local orbital antiunitary follows from the existing two-label matrix units:

`T_local = (I_label tensor iY_label) complex_conjugation`.

Direct scalar identities give `U U*=-I4`, `U beta* Udag=beta`, and `U alpha_d* Udag=-alpha_d` for d=x,y. The actual link coupling, including its i, is consequently invariant. Electric fields and the fixed boundary flux are T-even. B is T-odd. No extra CAR/Dirac matrices are introduced as microscopic inputs.

To lift T to the actual code basis, propagate phases g_n using **actual primitive tree-edge transfers**, and check every omitted-cycle-edge transfer. The selected sector has the ordinary matrix-unit signs; the discarded sector has the real sign twist only on the omitted edge. A wedge-order sign is used solely as an intertwiner comparator, never to build H. If the physical orbital permutation sends n to p(n), and its product of local phases and reordering sign is t_n, then

`T |n,s> = [g_(p(n)) t_n / conjugate(g_n)] |p(n),s>` anti-linearly.

The marker remains unchanged and T-even. This yields, on the actual Pauli code/rotor basis,

`T^2=(-1)^N`, `T H T^-1=H`, `T B T^-1=-B`.

The executable checks the **entire sparse operator** for these identities. For odd N, every eigenstate has an orthogonal equal-energy T partner. The theorem guarantees at least a Kramers pair, not that its ground multiplet is exactly two-dimensional. The CLI finds one ground vector, constructs its actual antiunitary partner, checks orthogonality/eigen-residuals, then probes the **entire orthogonal complement** using a certified spectral-width deflation penalty. This rejects extra unresolved ground states without assuming a single-vector Krylov algorithm returns every member of a degenerate eigenspace. It is a charged finite Kramers/spinor join, not a continuum rotation/scattering theorem.

## Actual positive photon measure and same-Hamiltonian polarizability

For a ground vector g of energy E0 and the complete resolved ground multiplet projector P0, use

`b=(1-P0) B g`, `mu(A)=<b, 1_A(H-E0) b>`.

The exact finite spectral theorem makes mu a positive measure on strictly positive energies, provided b is nonzero and the ground multiplet is isolated. The CLI **requires** nonzero b; no photon signal is inferred from a supplied Gaussian. Its exact connected identities are

`Var_g(B) = ||b||^2 + (||P0 B g||^2-|<g,Bg>|^2)`,

`int epsilon dmu = <b,(H-E0)b> = <g,[B,[H,B]]g>/2`.

The second zero-energy term matters in the charged Kramers sector and is not mislabeled as an inelastic photon pole. Lanczos with two-pass reorthogonalization gives a positive quadrature of this **actual** sparse measure. Its weights sum to `||b||^2` and its first moment matches the direct sparse operator moment. JSON labels these poles as **Lanczos quadrature**, not an exact full spectrum. Increasing Lanczos steps changes their resolution; sum rules alone do not certify individual higher poles.

The static polarizability is computed separately by converged conjugate gradient on

`(H-E0+P0) x=b`, `chi_B(0)=2 Re<b,x>`.

Its residual is reported and unconverged or nonpositive results fail. Addition/removal sectors use the same H, B and numerical method; JSON reports their B weight and static-polarizability differences relative to N=8. These differences may be positive, negative or small; they are not forced to resemble a QED cross section. Lanczos and resolvent results concern this finite boundary/rotor model.

For the neutral vacuum the actual code-state/electric-field amplitudes are reshaped and traced to obtain matter/rotor purity. The two raw q words belonging to each occupation are orthogonal to those of every other occupation and do not depend on E, so this compressed trace equals the raw Pauli embedding trace. Purity below one demonstrates nonfactorization of the actual state; it includes Gauss-induced correlations and does not by itself distinguish kinematic from dynamical entanglement. Charged purity refers to one pure member of the pair, not its mixed density matrix. The CLI requires neutral nonfactorization.

## Actual boundary creator and adverse controls

At cell 0 each physical orbital has an actual primitive path to the marker. Addition projects `-i A_path` onto **both empty endpoints to both occupied endpoints**; removal projects it onto physical occupied/marker empty to physical empty/marker occupied. The marker toggles, keeping total primitive parity even. Boundary flux changes by the same physical charge, and all four internal E remain unchanged, exactly preserving the full Gauss constraints. This is a finite boundary Gauss string, not a local gauge-noninvariant electron constructor.

The implementation applies this actual word between the neutral and charged code bases. For all four boundary orbitals it reports creator norm, total overlap with the charged ground **multiplet**, and normalized overlap; it requires a nonzero combined ground residue. This is a measured finite charged-state spectral residue, not continuum LSZ normalization.

Adverse controls independently require rejection of:

- boundary flux shifted by one, against the complete Gauss equations;
- assigning an electromagnetic unit charge to the occupied odd-N marker;
- reversing an **actual admissible matter hop's** electric transport signs, with its initial/wrong fields and Gauss residual included;
- reversing a primitive oriented-cycle path sign, which changes S to -S and violates the selected code projector/path law;
- reversing an actual closing-edge matrix element, against the Pauli-derived phase intertwiner, reporting both its correct and wrong-sign residuals.

The primitive-phase intertwiner additionally checks tree transport consistency and actual closing-edge matrix elements over every physical occupation in the selected N sectors. Wrong paths cannot be hidden by a canonical matrix builder. In the neutral sector the marker is unoccupied, so its wrong-charge control is deliberately null; both charged sectors perform it on an occupied marker.

Cutoff diagnostics include shell probability, W/Wdag raising/lowering losses, nested ground-energy monotonicity, B weight/static changes and the residual of the smaller ground vector embedded into the larger **actual** Gauss basis. The latter bounds distance to the larger spectrum, not automatically its ground eigenvalue or a continuum error. Two cutoffs provide an honest convergence diagnostic, not a claim of already achieved cutoff convergence.

## Executed default result

`quantum_spinor_plaquette_receipt.json` preserves the naturally completed default run, including source SHA256, all six full Gauss sectors, adverse controls, creator residues and spectral diagnostics. At cutoff 2:

- Neutral E0 is `-7.07799787753`; either charge costs `1.17884803380`.
- Both charged ground multiplets have dimension 2, with the next complement gap `.293664285583`. The neutral ground is isolated by `1.22008518689`.
- The first neutral B-active pole is `1.22008518689`, weight `.489664303124`; charged small-weight quadrature nodes need not equal the exact complement gap.
- Neutral matter/rotor linear entropy is `.01199789745`; electric-shell probability is `5.8505e-5`, down from `.0260243` at cutoff 1.
- The actual boundary creation/removal image has nonzero residue in both charged ground multiplets. At cutoff 1 its normalized overlap is `.99073113274`.

The cutoff-1 to cutoff-2 neutral energy shift is `-.000239517478`; its embedded eigenvector residual is `.03146666055`. These are separate diagnostics: the residual is not silently relabeled as a ground-energy error. The historical receipt predates the magnetic-independent bound and tensor-solver API update; the integrated run binds current source.

The executed strong-hopping sequence (`m=.8,r=.4,v=1.2,K=8/9,UE=.0072`)
retains both charged multiplets when the sufficient hopping bound is negative.
At cutoffs4/6 the actual charged costs are `1.43998997109/1.43752684208`;
charged shell probabilities fall from `.022182` to `.0015434`. The first
neutral B-active poles are `.221464950730/.165846275068`, so photon frequency
convergence is not inferred from the smaller charged-gap change.
`quantum_spinor_strong_hopping_receipt.json` binds this actual execution.
The separate [quantum-geometric theorem](quantum_gauge_mass.md) proves a
positive `.5732` bound on the **untruncated** one-loop domain at these
couplings; its trial does not automatically belong to the compressed
per-edge Fourier domains used here.

## Next strengthening on this construction

After running this consumer, increase electric cutoffs and Lanczos depth until the charged gap/residue and photon static response stabilize, retaining shell and embedding diagnostics. Then replace the four-cell boundary by a larger connected open lattice with the same neutral parity completion, prove the multi-cycle Delta margin rather than assuming it, and test charged-state localization/transport and multiple spatial photon modes from that same operator. A continuum spin theorem would additionally require a controlled spatial rotation action and charged dispersion limit; the present Kramers result is one constructive obligation toward it, not its replacement.

## References and source boundaries

- Existing campaign laws: [`pauli_compton.py`](pauli_compton.py), [`rotation_band.py`](rotation_band.py), [`code_phase.py`](code_phase.py), [`quantum_compact_photon.py`](quantum_compact_photon.py), and [`quantum_orbital_sector.py`](quantum_orbital_sector.py). These supply primitive Pauli paths, scalar orbital units, compact Gauss shifts, nonlinear photon response conventions, and the same-current defect-sector mechanism.
- S. Bravyi and A. Kitaev, *Fermionic quantum computation* (2000), [arXiv:quant-ph/0003137](https://arxiv.org/abs/quant-ph/0003137): background on graph/edge encoding. Here encoded matrix elements are evaluated from the repository's actual primitive words.
- J. Kogut and L. Susskind, *Hamiltonian formulation of Wilson's lattice gauge theories*, Phys. Rev. D **11**, 395 (1975), [doi:10.1103/PhysRevD.11.395](https://doi.org/10.1103/PhysRevD.11.395): compact Hamiltonian electric/loop formulation, not a substitute for this boundary/Gauss construction.
- C. Lanczos, *An iteration method for the solution of the eigenvalue problem of linear differential and integral operators* (1950), [doi:10.6028/jres.045.026](https://doi.org/10.6028/jres.045.026): positive tridiagonal spectral quadrature. Individual returned quadrature nodes remain finite-algorithm approximations.
