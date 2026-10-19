# Exact original functional Weyl weak forms

This compiler integrates the actual finite-Weyl/full-incoming-creator ABI over all52 coexact Gaussian coordinates. It does not replace the field functions by held point arrays. Normal Phi and the three free harmonic coordinates remain separate; this module alone makes no full-H/state/spin/scattering claim.

For Gaussian wave `exp(−y²/2)`, base factor `exp(i k.y)`, and original incoming packet g,

`a_g† exp(i k.y) = [sqrt(2) g.y − i(g.k)/sqrt(2)] exp(i k.y)`.

The phase-derivative term is retained. A right Fourier multiplication `exp(i h.y)` changes the phase without applying a new creator. Recreating the photon with phase `k+h` would solve a different operator-ordering problem.

`kernels` supplies all four base/base, base/incoming, incoming/base and incoming/incoming blocks. With d=k_right+h−k_left, scalar Gaussian characteristic is `exp(−|d|²/4)`. The two photon-polynomial means are `i g_right.(h−k_left)/sqrt(2)` and `i conj(g_left).(k_right+h)/sqrt(2)`; their product plus `vdot(g_left,g_right)` gives the two-creator block. Distinct complex packets and distinct phase sets are supported.

`oscillator_kernels` integrates the literal full quadratic coexact Hamiltonian `sum omega/2*(−partial_y²+y²)`, retaining its zero point once. It is not a fictitious harmonic theta potential. `directional_kernels` integrates the actual physical Gaussian derivative of the right factor multiplied by a finite Fourier flow; weak Hermitian/half-divergence symmetrization belongs to its caller. This permits literal sine Noether flow without linearization or point quadrature over EM.

`bilinear` contracts those scalar kernels with original physical fibre coefficients and channel maps. A full ordinary-Fock first operator can be inserted on the right; Phi second moments require the independently supplied BEFORE-compression Wick outside-second form. No eigenvalue clipping, physical whitening or input re-normalization is performed. Kernel distance nonnegativity removes floating cancellation in squared real phase distances, not physical signed residuals; interval certification is not claimed.

`linear_pack` combines fixed-frequency/packet/channel structures at changed q/theta coefficients. It preserves the literal incoming polynomial and column offsets. Changed Fourier or channel structure is rejected rather than silently zipped into a wrong derivative. This representation is not a claimed physical invariant particle module.

## Exercised independent mathematical control

A throwaway script exercised two active coexact directions with complex g and nonzero, distinct real Weyl phases; the remaining50 vacuum directions were analytically integrated. Independent direct16- and24-point-per-direction Gauss-Hermite integrations compared the full physical wave derivatives, Fourier multiplication, metric and oscillator forms. Maximum discrepancy was4.47e-15 for metric,1.40e-15 for shifted metric,3.84e-15 for the oscillator and1.86e-16 for the directional form. Literal `linear_pack.values` with an independent fibre operator and complex coefficient combination differed by3.26e-16. This is a positive mathematical ABI smoke, not a native52-dimensional quadrature or physical model result.

The whole-H consumer `prepared_neutral_functional_hamiltonian.py` uses these exact moments for the actual functional96 SOURCE/G/J/contact/incoming bank while retaining normal/theta quadrature and fullPhi trace debt explicitly. The all52 creator calculus applies to EVERY phase-dependent generated operator function. Whether the canonical screened SOURCE alone has zero coexact word is a separate producer-owned claim, never generalized to `B f(H_D)`, Peierls current or the whole operator family.
