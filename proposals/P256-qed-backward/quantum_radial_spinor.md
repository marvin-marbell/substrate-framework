# A quantum radial origin for the actual charged spinor mass

## Construction and ownership

`quantum_radial_spinor.py` replaces the prescribed overall amplitude of the actual four-cell, 17-edge Pauli/compact-photon consumer by a positive quantum bosonic amplitude. It does **not** replace its microscopic matrix builder. The onsite and hopping parts are extracted from `SpinorPlaquette` using the already checked `compact_charge_gap.operator_parts`; the photon, Gauss intersection, neutral parity marker, boundary electric energy and code penalty remain the physical operators of that builder.

The conditional Clifford **shape** is inherited from the unrestricted 60-coordinate sea-selection theorem in `soft_budget.py` and its stability extensions. Only its common amplitude is quantized here. No argument below quantizes all 60 coordinates or proves the shape is unchanged by the full quantum field fluctuations.

In units `hbar=a=1`, choose dimensionless `R>=0`, radial inertia `M_R>0` with inverse-energy units, and `kappa>0` with energy units. On `L²((0,infinity),dR)` impose Dirichlet boundary condition at zero and use

`Hrad = -d²/(2 M_R dR²) + kappa R⁴`.

The bare bosonic potential has its unique minimum at zero: **no source, negative quadratic term, selected nonzero norm, or imported preferred R-star** occurs. Let `Hmat(m0,r0,v0)` be the actual onsite/hopping operator, and let `Hph = UE/2 sum E_c² - K/2(W+Wdag)`. The joined Hamiltonian is

`H_N,s = Hrad tensor I + R tensor Hmat_N,s + I tensor Hph_N + UE/2 (N-8)² + Delta(1-s)/2`.

Thus every onsite and hopping coefficient, not the photon coefficients, is multiplied by the same quantum coordinate. N=7,8,9 are the actual physical sectors. The odd-N neutral marker completes the edge-code parity and has zero electromagnetic charge. Boundary flux is `N-8`, and its energy is not dropped.

## Domain, coercivity and a normalizable ground state

For fixed electric cutoff the physical Hilbert space is finite-dimensional, with bounded self-adjoint Hmat and Hph. The closed form domain is

`D(q) = H¹_0((0,infinity); Hphysical) intersect {psi: R² psi in L²}`.

For every `epsilon>0`, `C R <= epsilon R⁴ + C_epsilon` on the half-line. Therefore the linear matter coupling is infinitesimally form-bounded relative to the positive quartic radial Hamiltonian. The Friedrichs form sum defines a self-adjoint, bounded-below H with this common form domain. Its operator domain is the Dirichlet quartic Schrödinger domain; the bounded physical matrix coefficients do not impose a hidden condition at a nonzero radius.

The Dirichlet kinetic is defined directly as a self-adjoint positive
Laplacian. No globally self-adjoint translation momentum on the half-line,
or unrestricted Weyl canonical pair for a positive R, is asserted.

Taking epsilon smaller than kappa gives a coercive lower bound by a positive radial kinetic plus positive quartic minus a constant. Compactness of the embedding on bounded intervals, with the quartic controlling the tail, gives compact resolvent. Every finite-cutoff sector consequently has a normalizable lowest eigenstate. This is not an assumed interacting wavefunction. Radial kinetic does **not** commute with R Hmat; in particular a radial oscillator cannot be appended as an unchanged spectator.

## Exact code selection survives the radial kinetic

Use the existing actual raw-word unitary `F_eta = Q_eta R_eta`: Q is the omitted-edge Pauli Z, and `R_eta=(-1)^E3` is electric parity (this R_eta is not the radial coordinate). It cancels the code-sector sign in every gauged matter path and reverses W while preserving all electric and boundary energies. Extend it by `Iradial tensor F_eta`. It is independent of the radial coordinate, so it commutes with the radial kinetic **and** radial multiplication. On the common domain,

`F_eta† H_N,- F_eta = H_N,+ + Delta I + K Iradial tensor (W+Wdag)`.

The noncyclic cutoff has `||W||<=1`, hence

`inf H_N,- - inf H_N,+ >= Delta - 2K`.

This is exact for each actual N. The CLI constructs both physical code sectors, checks their complete sparse intertwiner, checks the separate matter/photon identities, and probes the joined tensor identity including the noncommuting kinetic. The numerical tensor probe supplements, rather than replaces, the factor operator identities and proof. Delta>2K selects the plus ground sector; it is not a whole-band separation claim.

The real radial sine basis has ordinary conjugation. The physical antiunitary extends to `Ttotal = Iradial tensor Tphysical`, hence `Ttotal²=(-1)^N`, H is T-invariant and the actual loop B is T-odd. The existing solver constructs the exact odd-N partner and probes the entire orthogonal complement with low-rank deflation. It does not assume that a single-vector Krylov run enumerates a degenerate ground space.

## A quantum charged-mass theorem with no magnetic loss

Assume `r0,v0>=0`, `M0=m0+2r0>0`, and define

`C_hop=8 max(r0,v0)`, `c0=8M0`, `ccharge=7M0+C_hop`.

The actual local onsite spectrum is `(-M0,-M0,+M0,+M0)` per cell. Its many-body minimum is -8M0 in N8 and -7M0 in N7/N9. The existing actual singular-channel proof bounds the four gauged hops by C_hop: the transport is a contraction on the noncyclic electric domain, not a frozen link. Thus for charged sectors, as an operator on the physical space,

`Hmat_N >= -ccharge I`.

For the photon, the integer-Gauss finite-cutoff Holder theorem in `compact_charge_gap.md` supplies the common bound `Hph>=epsilon0 I`, where epsilon0 is the ground energy of the **actual** neutral rotor with equal electric fields. This bound holds for every K and every nonempty Gauss intersection, with the original noncyclic cutoff. No -K estimate is needed in the charged comparison.

Define the genuine half-line quantum energy

`F(c)=inf spectrum[-d²/(2M_R dR²)+kappa R⁴-cR]`.

Because R>=0, the charged quadratic-form inequality, retaining kinetic unchanged, is

`E7,E9 >= epsilon0 + UE/2 + F(ccharge)`.

For an upper bound on E8, fill both negative onsite modes at each actual cell, using the primitive Pauli-code intertwiner phases. Multiply by the **actual neutral photon eigenvector** and the normalizable scalar radial ground of F(c0). This state has two particles per cell, equal electric fields, zero Gauss residual and zero boundary charge. A spatial hop changes cell counts, so its expectation in this trial is zero. Therefore

`E8 <= epsilon0 + F(c0)`.

The product appears **only as a variational trial**; the interacting ground states are solved from the full H. Combining yields the constructive quantum mass bound

`E7-E8, E9-E8 >= UE/2 + F(ccharge)-F(c0)`.

For c2>c1, let psi1 be the positive normalized half-line scalar ground at c1. Since its support is not concentrated at R=0, `<R>_psi1>0`, and

`F(c2) <= F(c1) - (c2-c1)<R>_psi1 < F(c1)`.

Equivalently the simple scalar ground obeys Hellmann-Feynman `F'(c)=-<R>_c<0`. If `M0>C_hop`, then `ccharge<c0`, so the lower bound is strictly greater than UE/2. It is positive for **every** positive kappa and M_R, every positive K, and every physical electric cutoff, provided the independently required code-selection condition Delta>2K holds. Arbitrarily weak UE and larger K are supported; no small-K premise enters the mass theorem. This is a sufficient parameter region, not a necessity claim.

The coefficient inputs remain m0,r0,v0,kappa,M_R,UE,K,Delta. They are not independently derived microscopic constants. But m0 is now a scalar energy coupling, not the prescribed physical electron rest energy: the finite-boundary particle and hole masses are the interacting quantum **energy differences** E9-E8 and E7-E8, which need not coincide. Neither is equated here with an infinite-volume relativistic pole mass.

## Emergent amplitude and the adiabatic approximation

In the neutral onsite trial the effective scalar potential is `kappa R⁴-c0 R`. Its positive minimum is **derived from occupied-sea energy**:

`R_ad=(c0/(4kappa))^(1/3)`, `V(R_ad)=-3 kappa R_ad⁴`.

The bare bosonic potential itself still prefers zero. For large M_R, with the minimum far from the Dirichlet wall relative to its width, local expansion gives curvature `12 kappa R_ad²`, radial frequency `omega_R=sqrt(12 kappa R_ad²/M_R)` and width approximately `1/sqrt(2M_R omega_R)`. These are conditional adiabatic estimates for that scalar trial, not fixed inputs to H or substitutes for the measured radial density of the interacting consumer. For small inertia the radial amplitude is a broad quantum distribution. The exact quantum gap proof does not use the adiabatic approximation.

## Exact sine-Galerkin consumer and norm certification

The consumer uses `u_n(R)=sqrt(2/L)sin(n pi R/L)`, n=1..d, on [0,L], with Dirichlet endpoints. Kinetic is exactly diagonal. Moments are integrated analytically using sine-product cosine integrals:

`[PR^pP]_nm=L^p [I_p(n-m)-I_p(n+m)]`, `I_p(k)=integral_0^1 x^p cos(k pi x) dx`.

For nonzero integer k, writing a=k pi and sigma=(-1)^k,

- `I_1(k)=(sigma-1)/a²`;
- `I_2(k)=2sigma/a²`;
- `I_4(k)=4sigma/a²-24sigma/a⁴`;
- `I_p(0)=1/(p+1)`.

In particular the quartic is **PR⁴P, not (PRP)⁴**. The adverse control computes their nonzero operator difference. The tensor ordering is radial-major, and a separately formed product action verifies it; intentionally physical-major input is rejected. Existing adverse controls reject incorrect Gauss transport, boundary flux, marker charge and code path signs.

Sparse actual Hmat and Hph act only on the physical factor. A LinearOperator evaluates

`H X = Hrad X + (PRP) (Hmat X_transpose)_transpose + (Hph X_transpose)_transpose + scalar X`.

No giant Kronecker CSR or dense physical Hamiltonian is materialized. The certified deflation norm supplied to `solve_low(..., norm_bound=...)` is

`||Hrad_G||_2 + L(8M0+C_hop) + 2UE cutoff² + K + UE (N-8)²/2 + Delta(1-s)/2`.

The onsite norm is at most 8M0 across these sectors; `||PRP||<=L`; the electric diagonal has at most four fields of magnitude cutoff; and `||K(W+Wdag)/2||<=K`. This is a triangle bound on the actual compressed H, not a measured spectral radius. It allows the solver to deflate the whole ground multiplet without accessing nonexistent CSR attributes on the LinearOperator.

The mass proof also holds **exactly within each finite Galerkin model** with `F_G(c)=lambda_min(Hrad_G-c PRP)`. PRP is strictly positive. The CLI evaluates F_G and checks its corresponding bound against the actual coupled N7/8/9 energies. The neutral quantum trial energy is evaluated under that same coupled operator and must match `epsilon0+F_G(c0)`.

Each scalar F_G is a variational **upper** bound on the half-line F. The difference of two upper bounds is not a rigorous numerical bound on their half-line difference. JSON therefore calls the returned difference a `projected_model_charge_gap_bound`, and explicitly refuses a numerical half-line certificate. The analytical strict-positivity theorem remains rigorous independently of those approximations.

## Reproducible execution and physical observations

Run with the campaign interpreter from the proposal directory:

```sh
/home/administrator/substrate-framework/.venv/bin/python quantum_radial_spinor.py --output quantum_radial_spinor_receipt.json
```

Defaults use physical electric cutoff1 and radial dimensions4 and8 at box4: all six actual N7/8/9 coupled ground/response cases are executed. An additional actual coupled N8 run at box3 provides box sensitivity. `--modes`, `--boxes`, `--cutoff`, `--inertia`, `--quartic`, `--m0`, `--r0`, `--v0`, `--magnetic`, `--electric`, `--delta`, and `--lanczos-steps` expose the model. With a single box only the six cases are needed; a targeted parent smoke can use `--modes 4 --boxes 4 --lanczos-steps 12` while still exercising **all three coupled charge sectors**. Larger K with weaker UE is supported by setting Delta>2K; the mass condition concerns m0/r0/v0 alone.

The complete default CLI has naturally exited zero. Its source-bound receipt
contains all six actual coupled charge/neutral cases, the extra neutral box
consumer and the independent scalar Ritz diagnostics. At radial dimension 8,
box 4 and electric cutoff 1, the neutral energy is `-6.24355351632`; removal
and addition each cost `1.53185901704`. The actual neutral mean amplitude is
`1.43695175272`, while the bare quartic still has its minimum at zero.
The neutral photon/rest linear entropy is `.01282621675`; the outer-box
probability is `1.79e-7`. Radial mean/density, its nonzero coupled entropy,
the charged creator residues and same-H positive B response are recorded
without replacing the interacting state by the variational product.

The neutral energy decreases by `.08186006554` between dimensions 4 and 8.
That sizeable Galerkin change is retained as a refinement diagnostic, not
hidden behind the small outer-box probability or a claim of full convergence.

For each actual ground representative, the consumer reports radial reduced-density eigenvalues, basis probabilities, position density, mean and variance, near-zero/outer-box probabilities, radial/rest purity, and electric-field purity after tracing out **both radial and Pauli occupation degrees of freedom**. The embedding is the actual code-occupation/electric basis; Gauss correlations are included and are not mislabeled as independent photon Fock factors. Charged quantities refer to the chosen deterministic ground representative, not a thermal mixture of the Kramers pair. Basis probabilities are coefficients in a sine basis, not position bins.

The physical B loop is tensor-extended without modification. The same-H connected response uses the existing positive Lanczos quadrature and projected static resolvent, with direct equal-time/energy moment and double-commutator checks. It includes ground-multiplet elastic weight and does not fabricate an enumerated full-Fock spectral sum. Primitive neutral-marker boundary creation/removal is radial-preserving; its exact physical image is projected onto the charged ground multiplet to report actual addition/removal residues.

Nested radial dimensions at fixed L have an exact variational inclusion. The consumer checks ground-energy nonincrease and the embedded smaller-state residual in the larger H. The residual is distance-to-spectrum evidence, not a certified ground error. Different finite sine spaces at different boxes are **not** nested, even though the underlying continuum Dirichlet form domains are; box energy differences are therefore labeled sensitivity diagnostics, never forced monotonic. Scalar dimension32/64/96 values at every requested box provide inexpensive separate one-dimensional convergence diagnostics, not oscillator-only evidence in place of coupled runs.

## Retained constructive directions

This model supplies a genuinely quantum amplitude and an exact sufficient massive charged-gap mechanism to the inherited shape/current/photon consumer. Quantizing independent shape coordinates and their spatial gradients, controlling the selected-shape response after that extension, enlarging the actual physical lattice, obtaining propagating/localized charged excitations and multiple photon modes, and proving a 3D/continuum limit remain distinct constructive directions. The finite open plaquette theorem is not a claimed full QED origin or campaign completion.
