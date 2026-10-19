# Positive Wilson mass, actual Gauss fibers, and a parallel filled sea

This construction removes the **hopping-norm restriction from the static one-body gap**, not from every many-body charged-gap theorem. It uses the four orbital matrix units already selected by the conditional orbital law. No microscopic canonical CAR Hamiltonian is introduced. The comparison with an occupied one-body sea is justified by the actual Pauli code intertwiner, checked against the complete compressed Gauss matrix. A dynamical quantum orbital phase, the infinite-volume limit, and the full interacting charged gap remain distinct obligations.

## Strong mass theorem without commuting gauge derivatives

Let beta be a self-adjoint involution and let

`h = beta M + Q`, `M >= m I > 0`, `[M,beta]=0`, `{Q,beta}=0`,

with M and Q self-adjoint. The theorem applies directly to finite matrices and bounded operators on the stated lattice/link domains. No claim about unspecified unbounded-operator domains is required. Neither `[M,Q]=0` nor a restriction on `||Q||/m` is assumed.

In the beta-positive/negative decomposition,

`h-lambda = [[M_+-lambda, B], [Bdag, -M_--lambda]]`.

For `-m < lambda < m`, the upper diagonal A is strictly positive and invertible. Its lower Schur complement is

`S = -M_--lambda - Bdag A^-1 B <= -(m+lambda) I < 0`.

Both A and S are invertible. The exact block triangular factorization therefore makes `h-lambda` invertible. Thus

`spec(h) intersect (-m,m) = empty`.

The same congruence preserves inertia in finite dimension: the number of negative eigenvalues is exactly the beta-negative dimension. Arbitrarily large hopping and arbitrary link curvature cannot close this gap while the positive-mass hypotheses hold. Squaring h and dropping the noncommuting cross terms is neither necessary nor valid.

### Nearest and improved Wilson masses on actual link domains

For a unitary link translation U, or its actual Galerkin compression U_R, the compressed operator is a contraction. Consequently

`L_R = I - (U_R+U_Rdag)/2 >= 0`.

The nearest mass is `M=m I+r sum_i L_i`, with r nonnegative. It commutes with the constant orbital beta and is at least m even when translations in different directions fail to commute.

On the **untruncated unitary** domain the improved primitive mass is

`W_i = r [I-(U_i+U_idag)/2]^2`.

Expanding before compression gives coefficients `3r/2` onsite, `-r` on each oriented nearest link, and `r/4` on each oriented straight two-link path. Every path has its actual adjoint. On a contraction U_R the corresponding actual primitive two-link operator is

`W_primitive,R = r [3I/2-U_R-U_Rdag+(U_R^2+U_Rdag^2)/4]`

`= r L_R^2 + r [2I-U_R U_Rdag-U_Rdag U_R]/4 >= 0`.

The second term is a positive boundary defect. It vanishes for a unitary periodic shift but does **not** vanish on the open or noncyclic truncated domain. Squaring the compressed nearest operator silently changes the Hamiltonian. For straight open rectangular paths, the compressed forward two-link path is U_R squared; for other geometries the actual compressed path must be used, rather than assuming that identity. More generally, compression of a positive untruncated W is positive, independently of a chosen path factorization.

The executable constructs periodic and open 3D matrices from the existing `RotationBand(improved=True).couplings()`, with arbitrary nonuniform link phases, small m and large v. It keeps nearest and two-link paths and all adjoints. It reports the nonzero `[M,Q]`, the actual spectrum, mass lower bound, and the open-boundary defect. Its adverse compression control rejects the false equality `W_primitive,R=r L_R^2` by an operator difference, not by falsely claiming that the latter square is negative.

## Exact plaquette interfaces and Fourier orientation

`wilson_mass_geometry.py` exports:

- `plaquette_one_body(phi,m,r,v) -> (h,dh)`: complex 16 by 16 matrices, with dh the exact derivative with respect to phi;
- `sea_projector(h,dh) -> (P,dP)`: the negative spectral projector and its occupied-empty Sylvester derivative;
- `gauss_diagonals() -> ((D_0,b_0),...,(D_3,b_3))`: four 16 by 16 diagonal orbital matrices and scalar backgrounds;
- `parallel_sea_frame(h,dh) -> (frame,dframe)`: the canonical beta-reference polar valence frame and its exact derivative. The wedge of this frame is globally periodic and determinant-parallel; no scalar phase correction is required.

The first function uses **exactly the current `SpinorPlaquette` coefficients**: obtain `onsite3,links3` from `RotationBand(a=1,m=m,r=r,v=v).couplings()`, subtract the absent z-link onsite `r beta`, and use spatial links `(0,1,x)`, `(1,2,y)`, `(3,2,x)`, `(0,3,y)` with forward orbital coupling `links3[axis][1]dag`. These are not a new assumed tight-binding convention.

The physical cycle is `0 -> 1 -> 2 -> 3 -> 0`, with electric fields indexed in that order. The actual primitive transfer in this orientation lowers its electric field by one. For the Fourier convention

`|ell> -> exp(i ell phi)`, `ell=E_0`,

a row/column transition with `ell'-ell=s` has symbol phase `exp(i s phi)`. In particular, the actual source-cell0 to target-cell1 hop has s=-1 and phase `exp(-i phi)`, and the reverse has `exp(+i phi)`. All other forward links in the above planar coefficient list leave E0 unchanged. **Physical edge0 carries the flux in this gauge.** The code's omitted tree edge is physical edge3; that algebraic choice does not move the Fourier phase there.

### Electric offsets and the neutral marker

The complete actual Gauss solver gives

`E=(ell, ell+rho_1, ell+rho_1+rho_2, ell-rho_0)`

in N8, with `rho_c=n_c-2`. Let C_c be the one-body projector on all four orbitals of cell c. Then

`D=(0,C_1,C_1+C_2,-C_0)`, `b=(0,-2,-4,+2)`,

and `E_e=ell+dGamma(D_e)+b_e` exactly. For charged N, the current boundary convention adds `N-8` to b3, and leaves the other offsets unchanged. The parity marker carries **no electromagnetic charge**. The executable obtains its actual occupancy from the raw Pauli words and rejects marker mischarge in N7/N9, rather than trying to detect an absent marker in N8.

### Consumer-bound code-fiber verification

For every selected N and every nonzero matter entry of the full `SpinorPlaquette` Gauss Hamiltonian, the executable removes the known electric and magnetic pieces, retains its actual raw code element, and conjugates by phases g_n derived by `PauliCode.intertwiner(N)`. It then multiplies by `exp(i(E0_row-E0_column) phi)` and compares with the single-transfer symbol coefficient and the independent wedge-order intertwiner sign. It also checks the exact derivative using the same **observed row/column transport**, all Gauss offsets, and coverage of every nonzero off-diagonal one-body coefficient.

The microscopic H is always built by the actual primitive Pauli words. The wedge-order sign is a comparator after that build, not a canonical CAR matrix builder. Wrong Fourier orientation and the wrong phase on code-cut edge3 are separately rejected against the actual matrix. Finite electric truncation remains a compression with occupation-dependent boundaries: this check does **not** identify that finite Hamiltonian with a direct integral of independent phase fibers. The untruncated rotor Fourier representation is a different domain.

## Derivative and charge geometry without phased eigenvector differentiation

Diagonalize h into its negative and positive clusters. If V_- and V_+ are orthonormal cluster bases and epsilon denotes the actual eigenvalues, set

`X_ai = (V_+dag dh V_-)_ai / (epsilon_i^- - epsilon_a^+)`,

`P=V_- V_-dag`, `dP=V_+ X V_-dag + adjoint`.

These are the unique occupied-empty solution of

`h dP-dP h = -(dh P-P dh)`, `P dP+dP P=dP`.

No denominator involves two occupied states or two empty states. Degeneracies inside a cluster cause no eigenvector-phase singularity. The exported derivative uses neither finite differences nor a numerical phase choice. The runnable independent derivative control differences only the **projectors**, which remain smooth at those internal degeneracies.

The negative-positive spectral separation is at least 2m. The Sylvester integral gives

`||dP|| <= ||dh||/(2m)`.

For the normalized determinant-parallel Slater state psi,

`g=||partial psi||^2=Tr(P dP^2) <= N_occ ||dh||^2/(4m^2)`.

The CLI reports this metric, its bound, exact charge-offset means and derivatives, and the offset variances

`Var(dGamma(D))=Tr[P D (I-P) D]`.

Mean neutrality is not zero charge fluctuation. Those fluctuations remain available to the quantum gauge consumer.

## Global smooth, periodic, already-parallel determinant frame

### Constant derived chirality

Let `Gamma5=beta alpha_x alpha_y alpha_z`, formed from the four existing derived orbital matrices. It is a constant self-adjoint involution, anticommutes with beta and every alpha_i, and therefore anticommutes with h for **all link phases**, including curved/noncommuting translations. Scalar spatial Wilson masses commute with this local Gamma5. Hence

`Gamma5 P Gamma5=I-P`.

Because Gamma5 is local to each cell, the occupied trace on every four-orbital cell is exactly two. The occupied determinant curvature is zero: the complementary projector has the opposite Chern trace, while constant chiral conjugation gives the same trace. This is determinant Berry geometry, **not** the physical spin connection. Physical Spin3 is supplied by the prior orbital-spatial locking construction.

The mass-preserving homotopy

`h_t=m beta+t(W beta+Q)`, `0<=t<=1`,

stays gapped by the strong mass theorem. This establishes topological triviality. Topological triviality or zero curvature alone would not establish zero flat holonomies; the following explicit canonical frame supplies that stronger statement.

### Reference overlap from the exact Lyapunov identity

Take an orthonormal valence basis V with `Vdag h V=-A`, A strictly positive. Let `B=Vdag beta V` and `C=Vdag M V>0`. Since `{h,beta}=2M`,

`A B+B A=-2C`,

`B=-2 integral_0^infinity exp(-t A) C exp(-t A) dt < 0`.

Thus beta is strictly negative as a quadratic form on the valence space. For the fixed beta-negative reference E_ref, the nonzero overlap eigenvalues are those of

`Vdag E_ref E_refdag V=(I-B)/2 > I/2`.

The valence and reference dimensions agree by the Schur inertia theorem, so `E_refdag P E_ref` is invertible with spectrum at least 1/2. Therefore the canonical frame

`C_beta=P E_ref (E_refdag P E_ref)^(-1/2)`

exists globally, smoothly, and periodically wherever h does. Its derivative uses the analytic inverse-square-root divided difference

`f[ai,aj]=-1/[sqrt(ai) sqrt(aj) (sqrt(ai)+sqrt(aj))]`,

which also gives the correct equal-eigenvalue derivative. No finite-difference frame phases occur.

### Its determinant connection vanishes exactly

In fixed beta-positive/negative coordinates, write the valence graph as

`C_beta=[X;I] S`, `S=(I+Xdag X)^(-1/2)`.

Since Gamma5 is constant and anticommutes with beta, it has block form `[[0,U],[Udag,0]]` for a fixed unitary U. Chiral symmetry maps the valence space to its orthogonal complement, giving

`Xdag U+Udag X=0`, hence `Udag X=iZ`, with Z Hermitian.

Thus

`C_beta=[i U Z S; S]`, `S=(I+Z^2)^(-1/2)`.

For any parameter derivative,

`Tr(C_beta dag partial C_beta)=Tr(S^2 Z partial Z)+Tr(S^-1 partial S)`.

Both traces on the right are real: `S^2 Z` and `partial Z` are Hermitian, and `S^-1` and `partial S` are Hermitian. Orthonormality simultaneously makes the left trace purely imaginary. It is therefore **exactly zero**. The wedge of the canonical frame is a global periodic section with `<psi,partial psi>=0`. All determinant holonomies are one. There is no principal-log branch, scalar phase correction, or hardcoded numerical zero.

A parent-requested direct matrix probe at `m=.03,r=.27,v=1.7`, with `phi=0,.731,2.4,pi`, observed `|Tr(C_beta dag C_beta')| <= 3.55e-16`. This is scoped evidence for the canonical-frame decision, not a receipt for the full CLI or the many-body photon consumer.

## Runnable discriminants and honest execution boundary

From this directory, using the project's scientific Python:

```sh
/home/administrator/substrate-framework/.venv/bin/python wilson_mass_geometry.py --output wilson_mass_geometry_receipt.json
```

The default includes the actual complete N7/N8/N9 Gauss models at cutoff1 and the small-m/large-v open/periodic nearest/improved 3D consumers. A neutral-only integration smoke can select `--numbers 8`; it intentionally does not cover the odd-marker adverse check. `--cutoff`, `--shape`, and scalar parameters are explicit. Output is deterministic JSON with a supplier source hash, inputs, exact matrix residuals, observed spectra, geometry, and adverse controls. Assertions reject incorrect transport/compression and declared-hypothesis violations. Missing M positivity and beta-even instead of beta-odd hopping each give an actual zero-gap counterexample in the control.

The parent owns the full scientific executions and composed photon consumer. This artifact does not claim an unobserved default receipt. Its available strong result is the static positive-mass theorem and a concrete consumer-bound implementation of the derivative, charge, and canonical sea geometry, with the full interacting charged-gap implication left to the joined quantum gauge calculation.
