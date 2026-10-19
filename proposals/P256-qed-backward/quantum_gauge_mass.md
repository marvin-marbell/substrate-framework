# Quantum-geometric charge mass without a heavy-hopping hypothesis

This is a constructive theorem for the **untruncated one-loop Gauss rotor**, followed by actual scalar consumers and an optional comparison with the existing coupled physical radial consumer. It replaces the sufficient `M>8 max(r,v)` assumption with the positive static Wilson gap. It does not assert the same new bound after a finite Fourier or occupation-dependent per-edge electric compression.

## Model, source and domain

Use `quantum_spinor_plaquette.py`'s four cells, four internal orbitals per cell, actual 17-edge `PauliCode`, `S=+1` sector and neutral parity marker. Its occupation phases are obtained with `PauliCode.intertwiner(N)` from the primitive edge words; no supplied CAR matrices construct the microscopic Hamiltonian. The one-body comparator and its derivative come from `wilson_mass_geometry.py`'s `plaquette_one_body(phi,m,r,v)`. The Clifford shape remains conditional on the orbital-law construction; this theorem does not supply a fundamental quantization law or quantize the other orbital coordinates.

Fix `m>0`, `r,v>=0`, `UE>0`, and finite `K>=0`. Units are `hbar=a=1`; `phi` and the radial amplitude `R` are dimensionless, `m,r,v,UE,K,kappa` have energy units and radial inertia has inverse-energy units.

For `N=7,8,9`, every occupation is retained and the Hilbert space is

\[
\bigwedge^N\mathbb C^{16}\otimes L^2(\mathbb T_\phi),\qquad
|\ell\rangle\longmapsto e^{i\ell\phi}/\sqrt{2\pi},\quad \ell\in\mathbb Z.
\]

The wedge is the proven comparator for the actual Pauli code, not a microscopic CAR postulate. The boundary electric flux is `b=N-8`; the marker has no gauge charge. Put `n_c=sum_{j=0}^3 n_{4c+j}`. The **actual** Gauss solution, in the coordinate `ell=E0`, is

\[
E_e=-i\partial_\phi+a_e,\qquad
(a_0,a_1,a_2,a_3)=(0,n_1-2,n_1+n_2-4,-n_0+2+b).
\]

Consequently `E_c-E_{c-1}+b delta_{c0}=n_c-2`. The photon term is

\[
H_{\rm ph}=\frac{U_E}{2}\sum_{e=0}^3E_e^2-K\cos\phi
             +\frac{U_E}{2}b^2.
\]

With this Fourier convention a source-cell-0 to target-cell-1 microscopic transfer lowers `ell` by one, and its one-body matrix element carries `exp(-i phi)`. Only edge 0 carries the fiber flux; placing it on the code cut edge without also transforming the connections is a different Hamiltonian.

The theorem uses periodic Sobolev form domains and all integer loop fields. The separate `Delta>2K` code-defect theorem continues to select the `S=+1` ground sector if that penalty is included. It is not needed to derive the geometric inequality inside `S=+1`.

## Static matter and global parallel sea

The Wilson proof in `wilson_mass_geometry.md` gives a gap `(-m,m)` for every flux and arbitrary nonnegative hopping strengths: the Hermitian Wilson mass block is positive, and the Schur argument excludes eigenvalues in that interval. Exact chirality pairs the eight occupied and eight unoccupied levels. Write their filled-sea energy as `Esea(phi)`. Pointwise the quadratic matter Hamiltonian obeys

\[
H_{{\rm mat},8}(\phi)\ge E_{\rm sea}(\phi),\qquad
H_{{\rm mat},7/9}(\phi)\ge E_{\rm sea}(\phi)+m.
\]

The neutral local cell density is exactly two because the constant chiral involution commutes with each cell-density matrix, maps the negative projector `P` to `1-P`, and each cell has four orbitals. Thus `<a_e>=0` for the neutral sea.

A normalized local eigensystem is **not** by itself a global periodic parallel Slater state. The fourth shared API, `parallel_sea_frame(h,dh)->(frame,dframe)`, supplies the actual globally periodic beta-reference polar occupied frame and derivative. The constant chiral constraint makes its determinant connection identically zero; no logarithmic phase branch or accumulated numerical parallel transport is used. Its construction and zero-holonomy proof belong to `wilson_mass_geometry.md`. The sea used below is its determinant multiplied by the occupation phases derived from the actual Pauli intertwiner. The CLI checks normalization, derivative, parallel connection, period and the connection sign directly in those coordinates. There is no assumption that the interacting ground factorizes into this sea and a rotor state.

## Exact fixed-amplitude quantum gap theorem

Define the scalar periodic operator

\[
H_s=-2U_E\partial_\phi^2+E_{\rm sea}(\phi)-K\cos\phi,
\qquad F_s=\inf\sigma(H_s).
\]

Let `Psi(phi)` be any charged vector-valued state and `f(phi)=||Psi(phi)||`. Each `a_e` is Hermitian diagonal, so the covariant Kato inequality gives

\[
|f'|\le\| (\partial_\phi+i a_e)\Psi\|\quad\text{a.e.}
\]

This follows from `f f'=Re<Psi,Psi'>`: the extra `i a_e` has zero real expectation. The weak statement extends through zeros by the usual norm regularization. Summing the four inequalities, using the static band gap and the actual boundary energy yields

\[
E_{7/9}\ge F_s+m+\frac{U_E}{2}.
\]

The scalar ground `chi_s` on the connected circle is normalized, real and strictly positive. Use only as a **variational trial**

\[
\Psi_{\rm tr}(\phi)=\chi_s(\phi)\psi_{\rm sea}(\phi).
\]

For every edge, the cross term containing `chi_s chi_s'` vanishes because the sea is normalized and the connection is Hermitian. Hence

\[
E_8\le F_s+\frac{U_E}{2}\int_{\mathbb T}\chi_s^2G\,d\phi,
\qquad G=\sum_e\|(\partial_\phi+i a_e)\psi_{\rm sea}\|^2.
\]

Both estimates use the **same** scalar energy; the magnetic potential cancels without losing `K`. Therefore

\[
\boxed{E_{7/9}-E_8\ge m+\frac{U_E}{2}
                       -\frac{U_E}{2}\sup_\phi G(\phi).}
\]

No kinetic/hopping triangle bound and no factorized interacting-ground assumption enter this result.

### Geometric identity and certified supremum bound

Let `P'=dP/dphi`, let `D_e` be the one-particle diagonal in `gauss_diagonals()` and include its scalar background in `a_e`. For the globally parallel determinant,

\[
g=\operatorname{Tr}(P P'^2),\quad
\operatorname{var}a_e=\operatorname{Tr}(P D_e(1-P)D_e),\quad
z_e=\operatorname{Tr}(P P'D_e),
\]

\[
\boxed{G_e=g+\operatorname{var}a_e-2\operatorname{Im}z_e.}
\]

The sign is for `partial+i a`: expanding `||psi'+i a psi||^2` gives `2 Re(i<psi',a psi>)=-2 Im z`. The determinant consumer independently computes the plus-sign covariant norm, not just another copy of the trace expression. A wrong-sign control is reported as discriminating only if the actual selected points have nonzero sensitivity; a symmetry that erases sensitivity is not called a successful adverse test.

The occupied/unoccupied Sylvester equation and their separation `>=2m` imply

\[
\|P'\|\le\frac{\|h'\|}{2m},\qquad
 g\le8\|P'\|^2\le\frac{2\|h'\|^2}{m^2}.
\]

Here `P'` is off-diagonal in the occupied/unoccupied split. The operator norm of the off-diagonal block equals `||P'||`; the separated positive Sylvester inverse follows from its decaying-semigroup integral, so no dimension-dependent entrywise estimate is being substituted.

The actual edge-0 hopping block is `T=(-r beta+i v alpha_x)/2`, up to the source orientation. Its largest singular value is `(r+v)/2`. In the Hermitian derivative matrix the two flux-dependent blocks have this same norm. In the neutral number-eight space the connections have norms `(0,2,4,2)`. Using `||x+y||^2<=2||x||^2+2||y||^2` gives the certified, unsampled bound

\[
G\le8g+2\sum_e\|a_e\|^2
\le\boxed{\frac{16\|h'\|^2}{m^2}+48}=G_{\max}.
\]

For `m=.8, r=.04, v=.08, UE=.0072`, `||h'||=.06`, so `Gmax=48.09` and the theorem yields

\[
E_{7/9}-E_8\ge .630476.
\]

This analytic arithmetic improves the old `.2436` sufficient bound, independently of `K`, **on the infinite-rotor domain only**. The sampled `G` and its scalar-ground-weighted average are useful diagnostics, not certified suprema. This conservative theorem can also be positive when `m+2r<=8 max(r,v)`; it is a sufficient positivity criterion, not a claim that every electric strength is gapped.

## Quantum radial extension: an actual two-dimensional scalar problem

Let

\[
H_R=-\frac{1}{2I}\partial_R^2+\kappa R^4,\quad
R>0,\quad\Psi(0,\phi)=0,\qquad I,\kappa>0.
\]

The full Hamiltonian is `H_R+R Hmatter(phi)+Hph`. There is no preferred nonzero amplitude in the bare potential. Multiplication by positive `R` leaves the occupied frame independent of `R`; its vanishing at the single boundary point does not require choosing an eigenframe there. The joint scalar operator and its mass-perturbed ground energy are

\[
H_{s,R}=H_R-2U_E\partial_\phi^2+R E_{\rm sea}(\phi)-K\cos\phi,
\qquad F_{\rm joint}(c)=\inf\sigma(H_{s,R}+cR).
\]

Kato's inequality applies to the radial kinetic energy as well as the four electric connections. A charged state's matter energy is `>=R(Esea+m)`, and the positive joint scalar neutral ground gives a neutral trial, with zero radial derivative of the sea and zero cross terms. Therefore

\[
\boxed{E_{7/9}-E_8\ge F_{\rm joint}(m)-F_{\rm joint}(0)
                       +\frac{U_E}{2}(1-G_{\max}).}
\]

This trial **does not** assume a product of a radial wave and a phase wave. `quantum_gauge_mass.py` constructs the coupled two-dimensional matrix `H_R tensor 1 + 1 tensor Hphase + PRP tensor Esea(phi)`; it solves this joint problem both at `c=0` and `c=m`, and reports phase/radial purity as a diagnostic rather than factorizing them.

### Strict mass and sufficiently weak electric coupling

The quartic potential confines the radial coordinate, and the scalar ground is strictly positive on the connected interior. For `c2>c1`, using the normalized ground of `H_{s,R}+c2 R` gives

\[
F_{\rm joint}(c_2)\ge F_{\rm joint}(c_1)
                 +(c_2-c_1)\langle R\rangle_{c_2}
>F_{\rm joint}(c_1).
\]

Strict monotonicity alone does not yet dominate the geometric subtraction at arbitrary `UE`. To prove existence at sufficiently weak `UE`, first set its phase kinetic coefficient to zero. If `f(s)` denotes the half-line ground energy of `H_R+sR`, define

\[
F_0(c)=\min_{\phi\in\mathbb T}
        \{f(E_{\rm sea}(\phi)+c)-K\cos\phi\}.
\]

For fixed `m>0`, `f(s+m)-f(s)>0`. This difference is continuous in `s`; the flux-dependent `Esea` ranges over a compact interval. Thus

\[
\delta=\min_\phi\{f(E_{\rm sea}(\phi)+m)-f(E_{\rm sea}(\phi))\}>0,
\qquad F_0(m)-F_0(0)\ge\delta.
\]

Positive phase kinetic energy gives `F_UE(c)>=F_0(c)`. Conversely localized smooth periodic phase trials, with radial ground trials at a minimizing phase, show `F_UE(c)->F_0(c)` as `UE` decreases to zero: choose localization width going to zero with `UE/width^2` going to zero. Consequently the joint mass difference remains positive with a compact uniform margin for sufficiently weak `UE`, whereas `UE(Gmax-1)/2` tends to zero. This proves a positive quantum charge mass for every fixed positive bare `m`, `kappa`, inertia and finite `K,r,v>=0` at sufficiently weak electric coupling. It is an existence theorem, not a numerical certified threshold or a preferred-radius ansatz.

## Computed consumers and noninterchangeable cutoffs

`--fourier-cutoffs` retains integer modes `-L,...,L` of the scalar **periodic** operator. `--phase-points` computes trapezoidal Fourier coefficients of the actual static sea and samples its geometric function. Scalar ground energies are quadrature Galerkin/Ritz diagnostics; potential quadrature error is not interval-certified. In particular subtracting two Ritz energies cannot certify a ground-energy difference, and a finite Fourier ground need not inherit exact pointwise positivity. The CLI states these limits and reports reconstructed-wave diagnostics.

`--radial-modes` and `--boxes` use `quantum_radial_spinor.RadialBasis`: Dirichlet sine functions with exact compressed `R` and `R^4` moments. `PR^4P` is used, not `(PRP)^4`. Their energies are finite-box/sine diagnostics, not exact half-line ground energies. A difference of upper estimates remains an estimate, even if its sequence stabilizes.

`--physical-cutoffs` additionally runs **the same** `quantum_radial_spinor.RadialSpinor` physical Hamiltonian in all three number sectors, using actual `SpinorPlaquette` Gauss occupations and its Pauli transitions. This bypasses only that module's CLI's old sufficient hopping restriction; it does not change its Hamiltonian. The output places the measured `E7-E8`, `E9-E8` alongside scalar diagnostics with the same radial basis. The physical per-edge cutoff induces different loop domains for different occupations. The geometric trial need not lie in those compressed domains and the scalar Kato trial comparison does not transfer to them: no new infinite-rotor bound is asserted there. These measured gaps may agree with the analytic infinite theorem only after separately controlled limits, not by declaration.

The direct determinant controls check selected flux points, actual intertwiner phases, normalization, parallel derivative, integer periodicity and the plus-sign connection formula. The fractional-loop adverse control has monodromy `-1`, so it is excluded from `L2(T)` even though formal Gauss differences alone would not detect a common fractional shift. Wrong boundary flux and charging the actual odd parity marker violate Gauss. There is no claim that such formal controls prove a thermodynamic, continuum or full-field phase.

## Parent execution commands

The parent executed all three scenarios below successfully. Commands use the project's NumPy/SciPy/SymPy environment.

```sh
/home/administrator/substrate-framework/.venv/bin/python proposals/P256-qed-backward/quantum_gauge_mass.py --output proposals/P256-qed-backward/quantum_gauge_mass_receipt.json
```

Exercise the removal of the heavy-hopping restriction (`m+2r=1.6`, `8 max(r,v)=9.6`) on the same scalar construction:

```sh
/home/administrator/substrate-framework/.venv/bin/python proposals/P256-qed-backward/quantum_gauge_mass.py --m .8 --r .4 --v 1.2 --boxes 5 6 --fourier-cutoffs 6 10 --phase-points 256 --output proposals/P256-qed-backward/quantum_gauge_mass_strong_hopping_receipt.json
```

Run the joined physical comparison with exactly shared weak parameters and radial sine bases. Its physical Fourier domain is still different, as stated above:

```sh
/home/administrator/substrate-framework/.venv/bin/python proposals/P256-qed-backward/quantum_gauge_mass.py --physical-cutoffs 1 --physical-radial-modes 2 4 --radial-modes 4 8 --fourier-cutoffs 6 10 --boxes 4 --output proposals/P256-qed-backward/quantum_gauge_mass_physical_receipt.json
```

The default uniform certified lower bound is 0.630476. The strong-hopping
scenario gives 0.5732, while the earlier sufficient hopping bound is
−7.9964. Its box6/radial16/Fourier10 joint scalar mass difference is
1.48882973395, a Ritz diagnostic rather than a certificate. The weak
physical comparison at box4/rotor cutoff1 gives gaps 1.17738432 and
1.25150035 with respectively two and four radial modes; this change
does not establish radial convergence. Independent source review accepted
the determinant geometry, covariant Kato comparison and radial weak-electric
existence proof without extending them to finite occupation-dependent cutoffs.

No merge, issue closure, continuum promotion, sampled-supremum certification or canonical origin claim is implied by these commands or this subtheorem.
