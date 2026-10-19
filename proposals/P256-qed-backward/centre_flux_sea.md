# Collective centre flux from the actual spatial heavy sea

## Domain and reproducible supplier

This is a frozen-field **finite open-cube discriminator**, not a thermodynamic gauge-phase certificate. The cube is exactly `full_field_charge_bound.open_box((2,2,2))`: eight vertices, twelve nearest bonds and six faces. Its seven-edge spanning tree is fixed to identity. The five remaining links independently take `1,i,-1,-i`, yielding all `4^5=1024` centre gauge classes. The receipt contains every class and all six oriented face fluxes, rather than just a proposed favourable pattern.

The shared field is obtained from `uniform_data(RotationBand(a=1,m=.2,r=.12,v=.3,improved=False),3,2,0,matrix_unit_basis())`. It is the same frame `Phi0=b beta`, `Phii=s alpha_i` used by the local Haar/centre suppliers; this is not a new optimized flux-dependent field saddle. The scalar coefficients are read from `RotationBand.couplings()` through the existing `scalar_paths` and `primitive_edges` suppliers. The onsite is the **full three-axis** coefficient `m+3r`, unchanged on the open boundary. Every actual available positive directed path and its adjoint is included. The endpoint coupling is

`h_xy = (sum_mu f_mu Phi_x,mu U_path + U_path sum_mu f_mu Phi_y,mu)/2`.

Heavy matter is EM-neutral. Its Hamiltonian is **whole** `-Lambda*h`, including onsite and hopping. At canonical half filling its energy is minus Lambda times the sum of the **highest sixteen** eigenvalues of the actual 32 by 32 h. A lower-band sign flip, one-axis two-path doubling, or hopping-only multiplier is not the same law. A bare plaquette comparison adds `K4 sum_faces(1-Re z_flux)`; alignment is zero for every centre link and the fixed field, and other frozen-field constants cancel.

Parent-owned final command, from this proposal directory:

```sh
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/administrator/substrate-framework/.venv/bin/python centre_flux_sea.py --output centre_flux_sea_receipt.json
```

`--heavy-scale` is Lambda in this static comparison; `--K4` varies the competition with the bare compact magnetic term. JSON binds this file, its proof note and actual coefficient/statistics/Pauli suppliers by portable basename and SHA256. It records the shared governing plan revision. No import, existing documentation or integration entry is modified.

## Hard positive controls and gap theorem

The gauge convention is `U_xy -> g_x U_xy g_y^dag`, so centre exponents transform as `k_xy -> k_xy+g_x-g_y (mod4)`. The supplier checks **every** class under a nonconstant centre gauge transformation, compares the complete one-body matrix with block-diagonal conjugation, and tree-fixes back to the original representative. The six fluxes distinguish all 1024 classes. Face cycles are `+i,+j,-i,-j`, with ordered faces `xy(z0,z1),xz(y0,y1),yz(x0,x1)`. Reverse cycles negate the exponent. The oriented Bianchi identity is

`F_yz(x1)-F_yz(x0)-F_xz(y1)+F_xz(y0)+F_xy(z1)-F_xy(z0)=0 (mod4)`.

The code verifies this identity and flux gauge invariance in every row; six arbitrary unconstrained fluxes would overcount the graph's cycle rank five. It also rotates each minimizing representative by every proper cubic rotation, retaining reversed edge orientation, and verifies the resulting gauge classes lie on the same energy orbit.

For every centre configuration the static gap follows analytically, not from the enumeration alone. Write

`h=beta M+Q`, `M=b[(m+3r)I-r/2 sum_i(S_i+S_i^dag)]`,

where each oriented open shift S_i is a contraction including its actual centre phase. Therefore `M>=bm I`, `[M,beta]=0`, and `{Q,beta}=0`. For `|lambda|<bm`, the beta-positive Schur block `M_+-lambda` is positive, and the beta-negative Schur complement is strictly negative. Inertia gives exactly sixteen positive and sixteen negative eigenvalues and excludes `(-bm,bm)` without requiring `[M,Q]=0` or small hopping. Half-filled occupied/empty cluster separation is at least `2bm`. The receipt verifies all 1024 spectra against this bound. It does **not** assume pairwise +/- degeneracy in a general complex flux background. Trace h is zero, so the heavy energy is also `-Lambda Tr|h|/2`.

The spectral filling comparison is tied to the **actual 4900-state Pauli section** `FullBandGaussSection`, whose transfer matrices are constructed from primitive marker-word code elements. At all four centre phases, the state is projected onto the top four heavy modes and the actual lifted heavy Hamiltonian expectation/eigen-residual are compared with minus the top-four eigenvalue sum. No canonical ladder builder is used. The existing two-cell owner retains only one active direction and hence onsite `m+r`. Its exact difference from the three-axis graph probe is explicitly `2r diag(Phi0,Phi0)`, checked entrywise as a matrix norm. The pair probe binds the spectral algebra and shared hopping; it does not silently replace the spatial cube by the owner's one-axis geometry. Nonzero light EM phases in the pair control do not multiply the heavy coupling.

## Analytic loop coefficient versus finite comparison

A useful local analytic discriminator is available without pretending that a hopping expansion is controlled at the physical coupling. Put `A=b(m+3r)`, `h(t)=A beta+tT`, `X=A{beta,T}`, and `Y=T^2`. Then `h(t)^2=A^2 I+tX+t^2Y`. From the analytic square-root series the exact coefficient of `t^4` in `-Tr|h(t)|/2` is

`Tr(Y^2)/(16 A^3) - Tr(X^2Y+XYX+YX^2)/(32 A^5) + 5 Tr(X^4)/(256 A^7)`.

Matrix trace cyclicity reduces the middle three terms, but the executable retains their noncommuting order explicitly. At fourth hopping order, the only centre-dependent closed walks on this cube are elementary plaquettes. The supplier decomposes this exact coefficient as a constant plus six `cos(pi F_face/2)` terms and rejects any residual above its tolerance. This is an **analytic finite-graph closed-loop coefficient**: unlike a fitted physical-energy plaquette ansatz, it is derived before evaluation. It supplies the sign and actual primitive coefficient of the first induced loop energy, with the whole-heavy scaling Lambda.

A simple uniform convergence condition is `q=2|t| ||T||/A+t^2 ||T||^2/A^2<1`. The code records the maximum actual hopping norm over the whole enumerated family, reports whether this condition holds at `t=1`, and supplies a smaller controlled homotopy scale. The bound `16 A q^5/(1-q)` controls the square-root tail **after the fourth power of `(tX+t^2Y)/A^2`**, using `|binom(1/2,n)|<=1`. It is deliberately not mislabeled as a remainder after fourth order in t: the retained square-root polynomial also contains t powers five through eight. Thus a fourth-order coefficient by itself does not certify physical t=1 flux minimization. The full 1024 spectral comparison is the finite-cube evidence there.

The receipt gives the complete sorted-flux sector comparison, all minimizing configurations, the next distinct total-energy gap, flat and uniform-pi heavy energies, the additional uniform-pi bare cost `12K4`, and their pairwise K4/Lambda crossing. That crossing is **only a two-pattern comparison**, not a certificate that no intermediate flux sector wins; the full enumeration with the requested K4 settles the finite graph. Individual flux gaps are not quantum tunnelling gaps or deconfinement gaps.

## Material consumer when flat flux fails

The implementation constructs an actual **period-two, 32-orbital magnetic Bloch consumer** for a minimizing cube representative and independently for the uniform-pi competitor; it would not stop at a nonflat finite minimum. The extension uses `U_i(x)=cube U_i(x with x_i=0)`, with the transverse coordinates reduced modulo two. It rebuilds onsite and every actual directed spatial coupling, including phases across supercell seams and all adjoints. Four independently specified supercell momenta expose full spectra and the same positive Wilson gap. This is a real candidate spatial consumer, not a relabeling of the old flat four-orbital RotationBand energy.

For uniform pi flux the extension is gauge-equivalent on the cube to `U_i(x)=(-1)^(sum_(j<i)x_j)`. Every oriented primitive face then has flux -1, including seams of the period-two extension. Magnetic translations obey `T_i T_j=-T_j T_i`. An ordinary simultaneous three-momentum four-band map cannot be retained. Physical translations combined with centre gauge operations can remain symmetries; a commuting supercell translation group yields the implemented 32-band map. Uniform pi flux is invariant under cubic axis permutations and orientation reversal modulo four, while representative links may require a compensating centre gauge transformation. The finite proper-rotation orbit control tests the selected-frame scalar/vector covariance at the energy level. If a more general nonflat pattern minimizes for another K4, the receipt retains its actual periodic supercell consumer and minimizing cubic orbit rather than assuming full cubic invariance of one representative.

**If nonflat flux wins**, the next material joined construction is to use the selected magnetic background's eigenprojectors, current/contact derivatives and magnetic translation labels for both light and heavy spatial consumers, then recompute charged propagation and same-current photon response on that background. The code already supplies the band matrix needed for these derivatives. It does not assume that the old flat four-band photon/Compton cartridge automatically survives a changed flux. If flat flux wins, the independent folded-band comparator instead binds this candidate back to the existing flat consumer. Neither case proves an infinite-volume minimizing field configuration: larger cells and quantum link/field fluctuation bounds remain a different problem.

## Evidence boundaries

Analytic results here are gauge counting, oriented flux identities, the static positive-mass/inertia theorem and the finite-graph fourth-order closed-loop expression with its explicitly delimited convergence bound. Actual Pauli pair residuals and all 1024 spectra are finite algebra/numerical evidence with source binding. Parent final CLI execution is separate from writer API probes.

Local four-centre minima and this collective discriminator do not establish a nonlinear spatial phase, deconfinement, an isolated quantum link quartet in the full angular Hamiltonian, a thermodynamic charged gap, light/photon transfer, emergent Lorentz or Dirac continuum, mechanical half-spin, or full QED. The diagonal orbital rotation action is not a pure microscopic rigid-space half-spin experiment. These distinctions remain explicit even when a nonflat sea preference is numerically large compared with fixed bare K4.

## Observed writer API discriminator, not parent final execution

At the shared bare coefficients, the complete 1024-class API probe found the **unique flat gauge class**, not uniform pi flux, as its finite-cube minimum. For heavy scale one and K4 zero, the energy was `-5.895679838087116`; the next class was higher by `0.0004709496602961494`. Uniform pi gave `-5.8929596686717005`, higher than flat by `0.0027201694154150857`. Every nonnegative bare K4 penalizes nonflat faces and leaves the flat class unchanged, so this fixed-field finite-cube preference survives arbitrary nonnegative K4 and positive heavy scale. This conclusion follows from the finite spectral comparisons; it is not an exact interval/rational or thermodynamic proof.

The exact fourth-order elementary-face coefficient was approximately `-0.000230632923992113` for each cosine, consistent with the flat preference. The physical hopping convergence bound was q approximately `1.19433`, so the fourth-order expansion alone is **not** a controlled physical-coupling energy proof. The static gap lower bound was `0.1296796650300664`, compared with the measured all-class minimum `0.2530506078145532`. The actual primitive pair eigen-residual was at most `7.69e-16`, and the explicit onsite-geometry coefficient control was zero.

The full local noncentral SU4 endpoint control had matrix residual `3.31e-16`; the flat period-two magnetic consumer reproduced the eight folded copies of the actual selected four-band spectrum within `1.53e-15`. The competing pi consumer was also built and exercised: all 24 oriented periodic faces, including seams, retained exponent two, and its sampled static gaps remained positive. It is retained as a genuine adverse consumer, even though it does not win this bare-law cube. Consequently this discriminator does not force a magnetic remapping of the current shared flat consumer, but a larger-cell/quantum phase conclusion still cannot be inferred from that result.

Tooling was exercised with native Pyright symbols and a definition query at repository root, resolving `primitive_edges` and its `scalar_paths` definition, and a positive structural AST query. Scoped diagnostics with the project interpreter reported zero errors/warnings; a first default-interpreter invocation could not resolve NumPy/SciPy and was corrected by selecting the actual scientific virtual environment. No parent final CLI run, integration or project-wide test is claimed here.

## Parent finite-cube receipt and energy-unit correction

The actual parent default CLI naturally exited zero into
`centre_flux_sea_receipt.json`. Its unique flat minimum, full-band gap to the
next class, competing pi bands, primitive residuals and Bloch-folding control
confirm the scoped finite-cube observations above; this is not an
infinite-volume or quantum-link ground-state conclusion.

Parent adverse invocations exposed two unit-dependent reporting defects:
at `Lambda=1e-8,K4=0` an absolute tolerance admitted all 1024 classes and
lost the next gap, while at `Lambda=1e8,K4=0` a raw cubic residual of
`2.384185791015625e-7` falsely rejected a valid symmetry comparison.
Classification now uses the actual unscaled top-sixteen energy `e` and exact
centre plaquette cost `c=sum(0,1,2,1)[F_face]` in the common positive units
`S=max(Lambda,K4)`: compare `(Lambda/S)e+(K4/S)c`, then restore the physical
gap by multiplication with S. The relative tolerance has no additive
physical-unit floor. Rotation validation separately checks unit-heavy
energies and exactly equal bare costs; the raw physical residual remains
reported with a distinct normalized control. The pi/flat crossing is
`(e_flat-e_pi)/12`, genuinely in K4/Lambda units.

The corrected actual small/large invocations naturally exited zero. Both
select only the flat class and have normalized next gap
`.0004709496602961494`; physical gaps are `4.709496602961494e-12` and
`47094.96602961494`. Unit-heavy cubic residual is `2.6645352591003757e-15`.
These source-bound receipts at code
`e55861de6a2cb5c5c074d7e8eaeee061542d82a04903ae730d1c2e20bca28985`
were inspected in a scoped comparison/provenance re-review. The executing
`pauli_compton.py` owner is now included in the receipt manifest.

`tests/test_centre_flux_sea.py` runs the actual CLI with both Hamiltonian
coefficients commonly rescaled by `1,1e-8,1e8`, checking identical minimizing
classes and correctly rescaled physical energies/gaps. This passed, including
nonzero K4, rather than merely asserting that the program did not throw.
Normalized comparison metadata preserves the unit meaning even when a
physical binary64 gap is too small to represent; it does not make this
finite numerical ranking an exact rational certificate.

