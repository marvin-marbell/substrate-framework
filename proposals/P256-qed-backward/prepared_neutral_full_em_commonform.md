# Original ALL55 EM six-mode common form

This new component is a pointwise ingredient of the selected moving-translation quantum Galerkin representation, not a global L2 energy, whole Hamiltonian propagator or closed physical rotation proof. Frozen providers and the shared prepared state are unchanged. Governing plan: `issue-222-fullfield-closedrotation-continuation-20261006`, assignment revision `668c7d8ea2ec9c71c964e9f16d3c75e8cd879e10433e72fbcad0c3b89c7da417`, scoped `prepared-neutral-full-field-outcome`.

## API and measure

`operator_matrices(model,data,sample,direction,sector,physical_point=None,q_shift=None,h=None,profile='source19',derivative_steps=None)` returns complex 6x6 `kinetic`, `potential`, `Gram` (also `gram`), `total`, coarse kinetic and observed FD errors. Sectors are `neutral`, `source19`, `SOFT`, or `charged` with an explicit profile. The original held coefficients are exactly `neutral_state` and `<profile>_normalized_source_state` from the shared cache. No pointwise normalization, filter, new Z, source-bank recapture or H_D body unitary is applied.

Use `sample=model.basis_at(qFoot,em,point=footPoint)`. The six multipliers are exactly the frozen interface: 1; sqrt(2)z; (2z²−1)/sqrt(2); (2|qFoot|²/sigma²−normal_dim)/sqrt(2normal_dim); sqrt(2)em[52]; and (2|em[:52]|²−52)/sqrt(104), where z=qFoot·normalized(direction)/sigma. Initial coefficients (1,0,0,0,0,0) therefore retain the identical once-prepared state.

Matrices are **unweighted factored point integrands**. Parent GLOBAL quadrature samples the original all-normal/all55 Gaussian and multiplies `cutoff_squared=cutoffFoot²` exactly once. The physical Haar half density rho(qPhysical)^−1/2 cancels the physical measure. For qPhysical=qFoot+s with s independent of EM, the translated Haar factor has no EM derivative; `physical_point` is deliberately not used to recapture SOURCE. The parent owns moving connection, Phi/Coulomb/matter/scalar terms and actual coefficient evolution. This is a variational representation, not proof of the whole G characteristic.

## Original operator

For each EM coordinate j, D_j is the shared original `directional_basis` jet contracted with the fixed global coefficient vector, plus the analytic multiplier derivative. That producer includes −em·dem over **all55**, including all three native harmonic Gaussian derivatives. The kinetic form is Σ_j w_j D_j†D_j, with w=omega/2 on all52 coexact modes and UE/N on the three harmonics. Potential is (Σ_52 omega_j em_j²/2)Gram. The original zero point is already present once in literal kinetic+potential; it is neither added nor subtracted. The kinetic trace is all55, not just the trial direction.

The bank retains actual SOURCE harmonic dependence through original theta/N, actual four Ref states and original plusBeta/minusBeta-conjugate functions. No coexact-only characteristic Gaussian replacement or preapplied changing-H_D phase occurs. Optional `derivative_steps=(coarse,fine)` accepts the actual shared all55 derivative arrays from `field_jets`, allowing reuse between the three sectors at a quadrature node.

## Executed evidence

The NEW CLI ran on the original cache SHA `10d61adb4a374a05e45338522abefcda260ed25036396b1bea26c48c57e4bee3` and manifest SHA `90a6d7f34617540042406a07c5d8690cecdc456b2d585ee150c152c7d2e07f9a`, original source_point q0/em. Natural exit 0, external wall 1298.739135213 seconds, empty stderr; stdout equals archived science JSON bytes. No old scientific CLI was rerun.

Two FD steps were 2^-18 and 2^-19. Independent literal packet finite variations evaluate the original bank at qFoot and em±h and multiply the complete all55 Gaussian ratio and six multipliers, rather than reusing derivative code. Fine maximum packet derivative errors were 2.39e−9 (neutral), 1.18e−8 (SOFT); all individual axes and source19 values are in the science JSON. Kinetic relative refinement changes were 0 (neutral), 3.56e−12 (source19), 1.40e−12 (SOFT). Total offdiagonal Frobenius norms are nonzero (neutral 19.38586943, SOFT 19.08591787); Hermitian residuals are approximately 2.5e−15. Actual held SOURCE harmonic derivative norms at the three harmonic axes are 0.9348231751, 0.4649092773, 0.4650580142 for both original banks. These are observed point controls, not certified global error bounds.

Private evidence (all local files):
- `local://issue222-full-em-commonform-science.json`
- `local://issue222-full-em-commonform-receipt.json`
- `local://issue222-full-em-commonform-stdout.txt` and `...-stderr.txt`
- `local://issue222-full-em-commonform-capsule.zip` (executed source bytes and exact two input files)
- `local://issue222-full-em-commonform-post-real-lsp.json`
- `local://issue222-full-em-commonform-typing-bridge.json`
- `local://issue222-full-em-commonform-current-api-smoke.json`

Actual Pyright stdio RPC was exercised before editing on the original shared producer (430 symbols, received empty diagnostics, definition/references), and after science on this new file (98 symbols, received empty diagnostics, definition/references). The xd LSP device is unavailable; direct actual Pyright is the evidence. Positive ast-grep queries matched the shared directional call before editing and the new operator declaration after science. A source-epoch Path typing issue was repaired only with runtime-identity `typing.cast`; the archived executed source SHA `e18430480fe00bfcc72d4bb32943077b29eda47f7c5979ba19c3e4f56551e840` and current SHA `ff4a205d714eba9dc68a86d19eaa8bf2af22fe058509a51edf4b48a7ab60fe09` have identical cast-erased ASTs. Current real sample/multiplier/source-epoch smoke also exited 0. No unobserved rerun of the full science is claimed.

Remaining scope: FD remainder is measured, not certified; parent must perform GLOBAL quadrature and full H−OmegaG/moving connection/coefficient evolution. Native harmonics remain unwrapped. No legal-N full-state closure, global L2 normalization from these point matrices, quantum-law derivation or Born/physical spin completion follows from this component.
