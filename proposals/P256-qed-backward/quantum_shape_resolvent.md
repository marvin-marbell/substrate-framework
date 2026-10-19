# Complete quantum restoring-shape ground and response

`quantum_shape_resolvent.py` supplies the **actual numerically resolved ground of the complete total-excitation K=1 Hamiltonian**, rather than the 46-dimensional active-image trial. `quantum_restoring_shapes.py` remains unchanged as the variational comparator. Both use the same immutable compact-Gauss `SpinorPlaquette`, primitive Pauli transfers, selected sea, and all 45 restoring operators. No reference onsite-product trial is substituted for a reference-H ground.

## Domain and construction

The physical domain is every raw same-N state satisfying compact U1 Gauss, S=+, and the open integer electric cutoff. Default parameters are m=.8, r=.04, v=.08, UE=.6, K=.25, and the existing sector penalty delta=1.2. The bare planar coefficients are rescaled by the selected cubic Gram and checked against the actual reference onsite and hopping matrix. The shape supplier is the measured positive 45-coordinate restoring Hessian at the selected finite 3³ sea. The 15 orientation directions are removed by the existing restoring-slice assumption; this is **not** a new dynamical SU4 Gauss construction.

For each scale, the Hilbert space is the tensor product of the full physical space with oscillator vacuum and all 45 single excitations: dimension 46d. Its diagonal blocks are Hphys and Hphys+omega_a, and its off-diagonal blocks are q_a G_a. Vacuum normal ordering is retained. No oscillator intermediate is silently discarded within K=1.

First `solve_low` solves the actual compact reference H, including its full physical complement. Its measured residuals and numerical energy labels are preserved. The coupled solve uses

\[
 S(E)=H_{\rm phys}-E-\sum_{a=1}^{45}q_a^2G_a
 (H_{\rm phys}+\omega_a-E)^{-1}G_a.
\]

Below the first-shell spectrum, every inverse is positive and

\[
 S'(E)=-I-\sum_a q_a^2G_a(H_{\rm phys}+\omega_a-E)^{-2}G_a<0.
\]

Schur inertia identifies the lowest crossing with the lowest full-H eigenvalue. A numerical reference-energy minus displacement-bound bracket is used, with an explicitly reported residual margin. Physical-dimension `eigsh`, body warm starts, and Brent roots are used; each shifted physical inverse is CG with diagonal preconditioning and **measured** residuals. The reconstructed shell components are x_a=-q_a R_a G_a x_0. The entire normalized 46d vector is acted on by the entire H. The CLI rejects a result whose full absolute eigenresidual exceeds `--full-residual-target`; a small Schur or Ritz residual is never relabelled as a full residual.

For odd N, physical antiunitary T acts in each block and the derived shape-T matrix mixes all 45 shell coordinates. This reconstructs the actual coupled Kramers partner. The next Schur branch is resolved in the complete physical complement: at each energy the fresh lowest Schur state and its actual T partner are deflated, not a fixed reference ground or a truncated eigenstate list. The full next eigenstate is reconstructed and its full-H residual and ground leakage checked. Positive next-branch separation and next-root negative inertia are required before any rank or subgap-response claim. Ambiguity raises an error; there is no fake-ground fallback. These are numerical complete-domain eigenbranch observations, not interval-certified spectral bounds.

## Full external-current consumer

Peierls differentiation starts from the primitive sparse entry geometry, independently of density commutators. Labels below 16 are onsite and have zero angle. A link label supplies index `label//16-1`. The physical link orientation is `(0,1),(1,2),(3,2),(0,3)`, not the cycle orientation. Fourier sources use the exact straight-link integral, including the midpoint phase and sinc factor, in that orientation. A reverse transfer receives **minus the same complex source angle**, not the conjugated angle. The conjugated outgoing source is chosen explicitly at the external vertex. Full sparse block checks verify J_q†=J_-q.

One primitive Hadamard factor is built for each source or two-source contact and then applied to both the reference hopping and every xi-dependent G_a. Duplicate sparse geometric positions retain one common factor rather than multiplying an accidentally summed duplicate factor. Every vertex acts on the complete coupled 46d space.

The response solves

\[
 [Q(H-E-\omega)Q+gP]y=QJF,
 \qquad P=FF^\dagger,
\]

with the actual normalized coupled frame F. Its preconditioner includes all **46 full physical diagonal blocks**, including every single-excitation mode. Those physical inverse actions use tightly converged measured inner CG; a fixed positive common shift makes each preconditioning block SPD even when the unprojected vacuum response diagonal block is not. Projection on both sides and a positive P block retain an SPD preconditioner. There is no truncation to reference ground, low eigenstates, or active-image intermediates. Inner and outer solver residuals and iteration counts are recorded separately.

The pure-gradient check derives J and C from primitive Peierls phases, compares against independent density commutators, and solves the actual complete static resolvent. Its exact independent solution is -i Q rho F. Both the equation residual and difference from numerical CG are reported. The full contact-minus-pair matrix must cancel. The deleted-contact calculation leaves minus the actual pair, while frozen-to-P static intermediates leave the actual contact because the identical-source P pole terms cancel. Both damaged calculations must have norms larger than the error scale.

The nonforward probe uses two distinct Fourier momenta and a strictly subgap frequency based on the resolved coupled first complement. It retains both time orderings, the full two-source contact, and analytic ground-multiplet Born terms. Coarse/fine CG repeats demonstrate convergence. An independently recomputed unitary change of the actual doublet frame must conjugate the entire kernel; neutral N=8 instead exercises a ground-frame phase change. This is finite external **matter-source response with compact magnetic W fixed**, not a total dynamical electromagnetic scattering calculation.

## Widths and exact theorem scope

All coordinate means, nonzero single-excitation probabilities, xi-G correlations, and continuum-coordinate second moments of the represented wavefunction are reported. The second moments include excursions outside K=1; squaring a compressed coordinate operator would give the wrong width. `full_wavefunction_sigma_coordinate_variance` is the explicitly defined restoring-slice coordinate statistic `(1/60) sum Var(xi_a)`, not an assertion about a global nonlinear quartic root.

The existing completing-square theorem retains its exact domain:

\[
 \Delta_{\rm charge}\ge .54-A/\Lambda,
 \qquad A=\sum_a C_a^2/(2k_a).
\]

The existing construction gives A≈593873.588844812 and strict-positive threshold Lambda≈1099765.90526817; the CLI recomputes both from the supplier. Default factor 8 gives displacement bound .0675. This inequality is valid for the bounded linear Gaussian shape EFT with untruncated compact Gauss rotors. It does not prove a nonlinear 60-coordinate theory, a thermodynamic 3D phase, or a narrow quartic wavefunction. Small coupled wavefunction corrections or excitation probability do **not** narrow the full coordinate variance. Quantum Gauss J=0 and any separate global soft-orbital model remain distinct domains.

## Portable commands and receipts

From the repository root, with Python providing NumPy and SciPy:

```sh
python proposals/P256-qed-backward/quantum_shape_resolvent.py \
  --cutoffs 1 --numbers 7 8 9 --scales 8 \
  --full-residual-target 2e-8 --output quantum_shape_resolvent_receipt.json
```

A selected-sector or scale comparison uses the same complete solver:

```sh
python proposals/P256-qed-backward/quantum_shape_resolvent.py \
  --cutoffs 1 --numbers 7 --scales 2 8 \
  --eigen-maxiter 4000 --inner-maxiter 1000 --response-maxiter 2000 \
  --root-maxiter 60 --output quantum_shape_scale_receipt.json
```

Solver tolerances, limits, scale factors, particle numbers, electric cutoffs, source momenta, and frequency fraction are exposed. K=1 is the implemented exact finite oscillator domain, not a parameter claiming K>1 convergence. Smaller scales may be more expensive or fail the requested numerical spectral resolution; failure is explicit. Source SHA-256 values accompany JSON for this executable and every listed scientific supplier. Each Schur solve records all 45 shifted-inverse measurements; each response stage records complete outer-CG receipts and all 46 block preconditioner measurements.

Implementation-language diagnostics and structural queries are not scientific execution evidence. This contribution does not close the joined electron-photon campaign, claim full QED, or establish a photon S matrix.

## Executed complete ground and current

The default three-sector CLI naturally exited zero in 2670.89 seconds. `quantum_shape_resolvent_receipt.json` records executable SHA-256 `241dde7e15fe2faac7801c3676bd4a7beaecb58c809b1e882e939105cbe59527`, the loaded shape supplier, parameters and complete-action residuals. Subsequent LSP renames of a private linear-operator argument and local block variable do not change this historical source binding.

| Particle number | complete dimension | lowest energy | ground rank | measured complement spacing | nonforward kernel norm |
|---|---:|---:|---:|---:|---:|
| 7 | 844008 | -5.899535349417 | 2 | 0.302480294811 | 0.0000479549339940 |
| 8 | 1040796 | -7.078566283557 | 1 | 1.229278329831 | 0.0000223545156233 |
| 9 | 844008 | -5.899535349417 | 2 | 0.302480294811 | 0.0000479549339947 |

Both finite charged energy differences are `1.179030934139`. Maximum full ground and next-state residuals are `1.35e-11` and `6.10e-13`; the reconstructed odd-sector partners are orthogonal. Static Ward kernel norms are at most `7.90e-15`, whereas contact omission or frozen intermediates leave norms `0.0152222330` in odd sectors and `0.0102766057` in the neutral sector. The separate infinite Gaussian-EFT inequality gives `0.4725` at the executed `Lambda=8798127.242145365`.

The spectral-origin repair applies ARPACK to `H_eff(E)`, then subtracts `E` after convergence; it does not change the Schur operator or truncate its domain. Already measured Brent endpoint signs are reused. Earlier timeouts are not completed scientific evidence.

The field kinetic law already has `M_Lambda=Lambda^-1/2`, `omega_a=Lambda^(3/4)*sqrt(k_a)` and `q_a=(2*M_Lambda*omega_a)^-1/2`. The observed broad coordinate statistic `sigma^2=0.171825720369` is not caused by an incorrect kinetic power: the softest supplied curvature is `8.04156336801e-5`. The vacuum statistic alone is `0.171825707279`, despite coupled nonvacuum probability only `4.2951e-7`. Increasing `Lambda` narrows coordinates as `Lambda^-1/4` in variance; initiating a larger-scale calculation is not proof of its outcome or of anharmonic control.


## Executed strong-narrow regime and moment repair

The `--scales 800000000` complete three-sector command completed naturally
in 2458.90 s after the coordinate-moment repair. Its receipt is
`quantum_shape_narrow_receipt.json`, with resolver source
`487f0bd6d82479cc261ae76fef126e2397697fb02159d34506c027b845e23b36`
and shape supplier
`1875ef8c60766c3bd0b64bb8d34a1c21cfa38b60c0fa505171925edef2e92d57`.
The executed stiffness scale is `Lambda=879812724214536.5`.

| Particle number | Ground energy | Rank | Measured complement spacing | Full ground residual |
|---:|---:|---:|---:|---:|
| 7 | -5.898828586217483 | 2 | 0.302428130649461 | 1.347928314072e-12 |
| 8 | -7.077758360062544 | 1 | 1.228879288313028 | 1.520829394460e-15 |
| 9 | -5.898828586217479 | 2 | 0.302428130649454 | 1.348017152404e-12 |

The finite charged transfers are approximately `1.178929773845`; next-state
full-H residuals are below `9.0e-14`. All three coordinate variance statistics
are `sigma^2=0.0017182570727851325`, substantially narrower than the default
Gaussian state. Nonvacuum shape probabilities are only `4.30567e-21`
in the odd sectors and `4.92292e-21` in the neutral sector.

An earlier invocation failed in the post-solve coordinate diagnostic;
it did not publish a receipt and supplies no observed spectral numbers.
Subtracting two nearly equal coordinate moments lost the genuine outside-K1
norm at these probabilities. The repaired diagnostic evaluates the actual
grade-two image norm `q_a^2*(P_one+p_a)` directly, with observed maxima
`5.24684e-23` and `5.99868e-23`. Vacuum is a legal state, so nonzero
participation is not an acceptance requirement.

This verifies the narrowed **Gaussian K1 EFT** and its completed current
consumer. It does not quantify the omitted anharmonic/full60 potential,
restore removed orientation modes, or establish the full nonlinear phase.
