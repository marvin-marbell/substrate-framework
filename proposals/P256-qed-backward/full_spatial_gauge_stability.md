# Full finite-bulk field/link Hessian and Gauss-linear principal dynamics

`full_spatial_gauge_stability.py` is an executable scientific supplier, not a result receipt. The parent must execute it after the input and sources are stable. No numerical success, nonuniform stability, quantum phase or confinement is claimed by this document. A negative mode is an admissible scientific result, not an exception to suppress.

## 1. Actual input and supported law

The default immutable input is `periodic_bulk_frame_receipt.json`, SHA256

`4a34efc4053ffbdf42f74ff492960d55109a5a359f2c367a24b14af740872a62`.

The explicitly selected branch is `environmental_Clifford_L3`, candidate `validation_stationary_candidate`. Its actual coordinate pointer is `/same_law_seed_branches/0/validation_stationary_candidate/coordinate_values`; the program resolves the named branch rather than assuming this index. It loads all sixty numerical coordinates without Clifford projection, normalization, diagonalization of their Gram matrix or optimization. The other two actual same-law branches are evaluated by default, including the open-cube-seeded numerical U1 stratum. None is relabelled the global ground. `--candidate coarse_stationary_candidate` changes the explicit coordinate source; it does not transplant validation coordinates.

The executed owners are:

- `sea_selection.matrix_unit_basis`: all fifteen traceless Hermitian Cartesian matrices, `Tr(B_a B_b)=4 delta_ab`;
- `local_orbital_stability.scalar_paths(RotationBand)`: actual onsite scalar and directed path scalars, including their imaginary parts and Fourier sign;
- `periodic_bulk_frame.PeriodicFunctional`: the actual midpoint meshes, full global half-filling, uniform gradient and uniform60 Hessian;
- `local_orbital_stability.elastic_hessian` and `fourier_sea_hessian`: independent inherited elastic/fixed-link controls, with the latter used only after verifying its two-occupied-modes-per-block hypothesis;
- `cartesian_angular_quotient` and the Hamiltonian in `full_band_packet_bound.md`: Cartesian kinetic coefficient, single colour-link Casimir and the Lambda multiplier of the leading potential.

The bare input is unchanged: `a=1, m=.2, r=.12, v=.3, improved=False, kappa=2, g0=0, J=.2`. This supplier supports its three nearest positive-axis paths. It fails loudly if the executed owner supplies improved/longer paths; it does not pretend that a two-step path has one nearest link. There is no newly added colour plaquette, scalar orientation rotor, electromagnetic response or light-sea term. The calculation is the full spatial extension of the **leading quartic + alignment + whole-heavy potential**, not the full quantum Hamiltonian after integrating its other sectors.

At start the input digest must equal `--input-sha256`. Coordinate/path owner revisions must agree with the receipt. All inherited executable/proof hashes and the new executable/proof hashes are bound. Bound source/proof/input bytes are checked after calculation and immediately before output; a changed input or mixed source revision prevents a receipt. The governing plan is `issue-222-continuing-backward-20260930`, assigned revision `d66d469ecc0e4e487bb104dc866f7fd679ae0e868c1a5d3abf38c1ede747183b`, work item `campaign`, requirements `outcome/verification`, general-template revision `89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`.

## 2. Potential, coordinates and Fourier convention

Let `Phi_x,mu = sum_a c_x,mu,a B_a`, `T_a=B_a/sqrt(8)`, hence `Tr(T_a T_b)=delta_ab/2`. Each positive-axis colour link has one exponential coordinate `U_x,d=exp(i A_x,d,a T_a)`. There are `4*15 + 3*15 = 105` real coordinates per site, represented by a complex Fourier block at q, with the usual reality relation between q and -q.

For an actual directed path scalar vector `s_d`, define `F_x,d=sum_mu s_d,mu Phi_x,mu`. The source-owned endpoint extension is

\[
 h_{x,x+d}=\frac{F_{x,d}U_{x,d}+U_{x,d}F_{x+d,d}}2,
 \qquad h_{x+d,x}=h_{x,x+d}^{\dagger}.
\]

The onsite term is the executed onsite scalar times `Phi_x,0`. This transforms as `h_xy -> g_x h_xy g_y^dagger`, under `Phi_x -> g_x Phi_x g_x^dagger`, `U_xy -> g_x U_xy g_y^dagger`.

The unscaled potential **per cell** is

\[
 V=\frac1N\left[
 \frac\kappa2\sum_x\|C_x C_x^T\|_F^2
 +\frac J8\sum_{x,d,\mu}\|\Phi_{x,\mu}-U_{x,d}\Phi_{x+d,\mu}U_{x,d}^{\dagger}\|_{HS}^2
 -\sum_{\mathrm{largest}\;2N}\lambda(h)\right].
\]

The heavy sign is minus the globally largest half, not `-Tr|h|/2`, not a per-k filling assumption and not an onsite-only sign. The actual restoring Hamiltonian has `Lambda*V`.

Write angular momenta `k=a*k_physical`, `q=a*q_physical`, and use the ket convention `exp(i k.x)`. A coordinate variation has `delta z_x=z exp(i q.x)`; the vertex is `<k+q|delta h|k>`. The program executes `sea_selection.coefficients` through the periodic owner, so the background is exactly `h(k)=sum_mu x_mu(k) Phi_mu`.

## 3. Linear vertices and indispensable contact

For the field coordinate `(mu,b)` the endpoint average gives

\[
 V_{\mu b}(k,q)=\frac{x_\mu(k)+x_\mu(k+q)}2 B_b.
\]

For colour link `(d,a)`, differentiating `U=exp(i A)` at the identity gives

\[
 V_{da}(k,q)=\frac i2\left[
 e^{ik.d}\{F_d,T_a\}-e^{-i(k+q).d}\{F_d^{\dagger},T_a\}\right].
\]

These expressions include both orientations of each executed path. The backward phase involves `k+q`, not k. Their contraction with a local gauge column must yield `i[T_a h(k)-h(k+q) T_a]` and is explicitly checked.

The mixed second derivative with a field at **-q** and a link at **+q** is

\[
 W_{\mu b,da}(k;-q,q)=\frac i2\left[
 e^{ik.d}s_{d\mu}(B_bT_a+e^{-iq.d}T_aB_b)
 -e^{-ik.d}s_{d\mu}^*(T_aB_b+e^{-iq.d}B_bT_a)\right].
\]

Its opposite block is the Hermitian conjugate. Link/link derivatives on the same link have

\[
 W^{+}_{da,db}=-\frac14\{F_d,\{T_a,T_b\}\},
 \qquad
 W_{da,db}(k)=e^{ik.d}W^{+}_{da,db}+e^{-ik.d}(W^{+}_{da,db})^{\dagger}.
\]

There is no field/field onebody contact and no different-link contact for this nearest-link law. These are second derivatives, not guessed current-response counterterms. They follow from `dU=i T_a`, `d_a d_b U=-{T_a,T_b}/2`. The program integrates `-Tr P(k) W_ij` for the **same exact heavy occupation projector**. Mixed contact is complex at general q and is not replaced by a real q0 value.

Let `n_k,s` be the globally selected occupation bits, and `M_i=<u_k+q,t|V_i|u_k,s>`. The full cluster response is

\[
 H^{sea}_{ij}(q)=-\frac1N\sum_{k,s,t:n_{k+q,t}\ne n_{k,s}}
 \frac{n_{k+q,t}-n_{k,s}}{\epsilon_{k+q,t}-\epsilon_{k,s}}
 M_i^* M_j
 -\frac1N\sum_k\operatorname{Tr}[P_k W_{ij}(k;-q,q)].
\]

Both occupied/empty orientations appear. No same-cluster denominator, denominator floor, artificial spectral regulator or removal of an adverse crossing is used. A nonpositive global occupied/empty gap aborts this differentiable calculation. Exactly degenerate eigenvalues within either cluster are harmless. Numerical Hermitian averaging occurs only after checking the pre-averaged Hermiticity defect; it is not gauge projection.

The quartic field block is the inherited zero-preference elastic Hessian. Alignment at a uniform identity-link mean has linearization

\[
 \delta R_d=(1-e^{iq.d})\delta\Phi-i[\delta A_d,\Phi].
\]

With `D_a=coordinates(i[T_a,Phi])`, its Hessian is `J sum_d L_d^dagger L_d`, where `L_d=((1-exp(i q.d))*Id_60, -D on link d)`. This is the full `J/8` HS alignment: `Tr(B_a B_b)=4 delta_ab` converts it to `J/2` times the Cartesian squared linearized norm. The same field tangent is not added again as a separate rotor.

## 4. Actual gauge columns and off-stationarity Ward identity

The fifteen actual Fourier gauge columns are

\[
 G_a(q)=\begin{pmatrix}
 D_a\\(1-e^{iq_1})e_a\\(1-e^{iq_2})e_a\\(1-e^{iq_3})e_a
 \end{pmatrix}.
\]

The receipt retains their numerical entries, not just their rank. It also retains the full105 gradient and the full differentiated Ward RHS. Local gauge invariance gives

\[
 H(q)G(q)=-(d\tau)^T\nabla V,
\]

**not** `H G=0` away from a stationary mean. For the field part the RHS is the Cartesian coordinate representation of `i[T_a,gradient_field]`. In exponential link coordinates, BCH gives

\[
 \tau_A=t_x-t_y+\frac i2[t_x+t_y,A]+O(A^2),
\]

so the link RHS is `(1+exp(i q.d))/2` times the adjoint action on that link's actual gradient. Dropping this chart term can manufacture an incorrect Ward test at a general mean. No part of H is replaced by a projected matrix to obtain zero columns.

A generic, deliberately off-stationary mean made from the actual coordinates is used for the adverse control. The correct full105 Hessian must satisfy this identity while the same test must reject (i) omission of the mixed/link diamagnetic contact, (ii) the wrong heavy-sea curvature sign and (iii) the wrong linear colour-link vertex sign. Each wrong result must have Ward residual greater than `1e-5`; the correct Ward residual must be at most `3e-7`. Thus a missing contact cannot hide behind stationarity, a projected gauge subspace or a test of a manufactured zero.

## 5. Independent consumer probes and their scopes

The mandatory small real-space consumer uses a periodic `3^3` lattice, the actual endpoint fields, actual directed scalar paths, matrix-unit basis and actual SU4 generators. Its midpoint seam is periodic. It is not the reported coarse/fine bulk mesh.

It checks:

1. Actual real-space `h` against inherited `local_hamiltonian` at identity links, then against the diagonal Bloch Hamiltonian through a unitary spatial Fourier transform.
2. The full105 first onebody variation at oblique angular q=`2*pi*(1,1,0)/3`, phase `.37`, with nonzero field and link components. Real-space analytic derivatives are assembled by differentiating endpoints and exponentials, **not** by inverse-transforming the tested vertices.
3. The actual mixed/link second onebody variation against central finite differences of the independently assembled finite-link Hamiltonian. The second exponential derivative is present before taking traces.
4. A nonzero first energy variation at a generic off-stationary mean, and the real-wave second energy variation against both finite energy differences and the exact finite real-space spectral cluster formula with contact. The nonzero first direction has a uniform radial component plus the oblique wave; this avoids passing a first-energy test merely because a uniform mean integrates a pure nonzero-q wave to zero. Its mixed partner is a pure oblique wave, so the mixed second derivative is `Re(u^dagger H(q) v)/2`.
5. Finite nonuniform SU4 covariance of the **actual h**, and nontrivial infinitesimal wave vertices for generators 0, 7 and a deterministic mixed generator. The differentiated potential Ward uses all fifteen generators.
6. Independent q0 field60 curvature from `PeriodicFunctional.hessian`, and finite-q fixed-link field60 Fourier response from the inherited helper. The latter has an explicit verified two-per-k occupation scope; exact global filling in the new response is never replaced by that helper's narrower rule.

The two finite-difference steps are `2e-4` and `1e-4`. Required absolute errors are: first-h `<=2e-7`, second-h `<=2e-6`, first-energy `<=3e-7`, second-energy `<=3e-6`; Fourier/covariance/spectral identities `<=3e-7`. These are numerical gates for executed finite matrices, not interval-certified endpoints or derivative convergence theorems. A failed gate aborts output rather than returning fake fallback data.

## 6. Gauss-linear kinetic normal spectrum

The source kinetic is

\[
 T=\frac{\sqrt\Lambda}{2}\sum_{60}p_c^2+u_4\sum_{45}p_A^2,
 \qquad
 M=\operatorname{diag}\left(\Lambda^{-1/2}I_{60},\frac1{2u_4}I_{45}\right).
\]

There is one Cartesian field mass and one genuine link mass, no second field rotor. A potential Hessian or its pseudoinverse is never an inertia. Linear Gauss is `G^dagger p=0`; velocities are therefore M-horizontal: `G^dagger M v=0`.

Put `y=M^(1/2) delta z`, and choose orthonormal columns Z spanning the complement of `M^(1/2) G`. The actual principal restoring matrix is

\[
 K_{normal}(q)=Z^{\dagger}M^{-1/2}\,[\Lambda H(q)]\,M^{-1/2}Z.
\]

Its eigenvalues are squared oscillation frequencies, or negative squared frequencies/growth rates if unstable. All signed eigenvalues and per-mode potential curvatures are retained. Real oscillation frequencies and imaginary-frequency growth rates are reported separately; a root of the positive part is a display field, not replacement of the signed spectrum. The numerical reporting tolerance for *potential curvature per coordinate norm* is `2e-6`, so a large Cartesian kinetic coefficient does not erase a slower link-sector negative mode by a field-scale threshold.

The gauge rank is resolved in the actual unweighted coordinate columns first. Mass weighting is not allowed to reclassify the stabilizer. The resulting eigenvectors are checked for the actual Gauss mass-horizontal condition and the projected eigenproblem. `field_kinetic_participation` records the proportion of each mass-normalized eigenvector in the original sixty field coordinates, rather than guessing a mode's identity from its index.

The mass-horizontal residual gate is `2e-6`, the relative projected eigenproblem gate is `2e-10`, and the relative self-adjointness defect of the Lambda-scaled reduced matrix must be at most `1e-12` before numerical symmetrization. Its scale is not compared to an absolute unscaled-potential tolerance.

These are equilibrium modes only when the same full105 mean is actually stationary on the reported mesh. The full gradient residual, its field/link components, and `Lambda`-scaled norm are printed separately. At a nonstationary frozen coarse-grid mean the restricted matrix is a local quadratic/principal observation, not an earned equilibrium normal spectrum. The program does not reoptimize coordinates to hide the discrepancy.

At q0 the field stabilizer is `ker D`. An irreducible resolved mean has rank15, leaving90 horizontal coordinates. A numerical U1 mean has rank14 at q0, leaving91; at nonzero q the link part normally restores gauge rank15, leaving90. The missing q0 gauge column is a stabilizer of the background, not by itself a physical flat oscillator. Residual stabilizer action and the singular quotient still matter for quantization. The program reports the actual singular values, resolution, kernel, commutator residual and q0 link curvature in the resolved unbroken sector.

For an exactly commuting unbroken U1 generator, alignment has zero link mass. A constant unbroken link shifts Bloch momenta by its charges, so a continuum BZ integral is shift-invariant. A **finite** midpoint sum is not invariant under arbitrary continuous shifts: its q0 sea holonomy curvature is retained, compared across meshes and never forcibly set to zero. The receipt's numerical stabilizer does not certify exact commutation. Thus this branch exposes the massless/unbroken continuum mechanism and singular-stratum boundary without silently substituting irreducible field data or promising an exact finite-grid massless spectrum.

## 7. Quadrature, q grid and scaling limits

By default the same actual validation coordinates are frozen on coarse8 and finer16 midpoint BZ quadratures. The input candidate grid is16. Seven actual input-grid q indices are

`(0,0,0), (2,0,0), (2,2,0), (2,4,2), (8,0,0), (8,8,0), (8,8,8)`.

They represent zero, the softest axis wave shared by both quadratures, soft planar oblique, a distinct three-axis oblique wave, edge, face-edge and corner. q is angular `2*pi*index/input_grid`, physical q is divided by a. Every supplied q must belong to **both** quadratures and the selected input grid so the differentiated gauge Ward compares exact finite-mesh momentum permutations, not a shifted inconsistent quadrature. `--q-index` can request other compatible waves; q0 is always included first. A finer pair of quadratures can reach the softer input wave index1. The program reports H, its signed spectrum, occupation gap, gauge columns, contact norm, principal modes and full gradient at every evaluated mesh. It also reports actual coarse/finer matrix and spectrum differences and inherited energy cover bounds. Those energy bounds do not certify gradient/Hessian convergence.

For the unscaled potential H fixed and `u4` fixed, a pure Cartesian field mode has `omega^2 ~ Lambda^(3/2)` and `omega ~ Lambda^(3/4)`. A pure link restoring mode has `omega^2 ~ 2*u4*Lambda` and `omega ~ sqrt(Lambda*u4)`. A mixed mode need not have either simple exponent; its Gauss-horizontal subspace itself depends on M. The CLI recomputes spectra at `Lambda/4, Lambda, 4*Lambda` with u4 fixed and at `u4/2, u4, 2*u4` with Lambda fixed. Ordered eigenvalues can cross; these samples are not a matched-eigenbranch scaling proof. Unbroken/transverse soft modes and flat holonomies require their own limits. No all-mode `Lambda^.75` assertion is made.

## 8. Commands, expected gates and next boundary

From the repository root, the parent full scientific run is:

```sh
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/administrator/substrate-framework/.venv/bin/python \
  proposals/P256-qed-backward/full_spatial_gauge_stability.py \
  --input-receipt proposals/P256-qed-backward/periodic_bulk_frame_receipt.json \
  --input-sha256 4a34efc4053ffbdf42f74ff492960d55109a5a359f2c367a24b14af740872a62 \
  --branch environmental_Clifford_L3 --candidate validation_stationary_candidate \
  --grid 8 --validation-grid 16 --Lambda 1000000 --ue4 .3 \
  --output /tmp/issue222-full-spatial-gauge-stability.json
```

The selected-branch-only consumer, with a finer quadrature and the genuinely softer input wave, is:

```sh
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/administrator/substrate-framework/.venv/bin/python \
  proposals/P256-qed-backward/full_spatial_gauge_stability.py \
  --input-receipt proposals/P256-qed-backward/periodic_bulk_frame_receipt.json \
  --input-sha256 4a34efc4053ffbdf42f74ff492960d55109a5a359f2c367a24b14af740872a62 \
  --branch environmental_Clifford_L3 --no-all-branches \
  --grid 16 --validation-grid 32 --q-index 0,0,0 --q-index 1,0,0 \
  --q-index 1,1,1 --q-index 8,0,0 --q-index 8,8,8 \
  --Lambda 1000000 --ue4 .3 \
  --output /tmp/issue222-full-spatial-gauge-stability-finer.json
```

Meaningful expected gates: bound bytes must remain unchanged; exact cluster gaps must be positive; real-space first/second derivative and independent fixed-link controls must pass; off-stationary local Ward must pass while contact/sign deletions fail it; principal modes must satisfy mass-horizontal Gauss and their projected eigenproblem. Stationarity, nonuniform positive curvature and quadrature agreement are **observations to inspect**, not hardcoded success criteria. Negative modes and coarse-mesh stationarity loss remain in the output. `scientific_controls_pass` means only that the executed derivation/consumer checks passed; it is not a viability, ground, phase or confinement bit.

A successful finite-q principal calculation can inform the next materially joined construction. It cannot finish it. The next quantum-normal/flux join requires the actual moving normal ground bundle over these fields and links, singular-stratum treatment, inherited product/Haar measure and drift, Berry/Born terms, uniform excitation control and finite-volume/thermodynamic bounds, plus the **same actual input mean and law** at the discrete centre-flux/current boundary. Finite induced closed-loop flux ordering, a two-cell bound, a derived spin algebra and this finite-q quadratic spectrum do not imply a joined electron-photon quantum phase.
