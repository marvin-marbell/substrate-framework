# P253/0164 validation boundary

The original initial-state calculation was independently reviewed twice. Both reviews initially found the same incorrect attribution of a convective term to the *material*-acceleration difference. The corrected `derivation.md` distinguishes `D_t u_+−D_t u_−=−∇Δp/ρ` from the Eulerian `∂_t(u_+−u_−)`, which includes the negative convective contrast; both reviewers rechecked and closed their finding. One reviewer independently reconstructed the Gaussian moment and its Schwartz far-field expansion. The other verified the stationary medium, index germs, common-lifespan transport and vanishing-amplitude adverse control at `e=0`. The two unit-norm side waves justify `|U_side|≤2` and `0<|e|<1/2`. This is a scoped review of an initial/local classical result, not a stability or particle result.

From the repository root, the algebraic oracle exits successfully with:

```text
.venv/bin/python proposals/P253-euler-particle-mechanisms/attempts/0164-helical-medium-defect/helical_defect_octupole.py
Q_background: -pi**(3/2)*(6*lam - 1)*exp(-1/(4*lam))/(24*lam**(9/2))
Q_compact_cross: -13*sqrt(2)*pi**(3/2)/(4608*lam**(7/2))
Q_sidewave: 0 for every lambda and e by z parity
Q_total(lambda=25): -1.78081100143365e-5
```

The fixed-parameter extension was then independently challenged twice. The index reviewer checked the large-`delta` implicit-function germs `X_±=∓(0,e/2,−e/3)/delta+O(delta⁻²)`, opposite Jacobian signs and common smooth lifespan on the same `U_e+V_0±delta W` initial states. The pressure reviewer reconstructed the side-wave density odd in `z`, every zero-order stress parity cancellation and the unchanged strict negative cubic moment; neither executed the oracle. An author smoke using the actual SymPy vorticity and SciPy roots at `e=1/10,lambda=25,delta=100` exited with `X_+≈(−3.8330e−6,−5.00108e−4,3.33366e−4)`, `X_-≈(−3.8330e−6,5.00108e−4,−3.33366e−4)`, residual norms `1.12e−16`, and Jacobian determinants `−5.9990e6` / `+5.9990e6`. This numerical root check is not the existence proof; the invertible germ is. The pressure octupole is exact at this same fixed nonzero `e`, but no later-time shape or all-mode robustness is established.

The full repository schema validator exits successfully:

```text
env PYTHONPATH=src .venv/bin/python scripts/validate_repository.py
WORKFLOW VALID: 271 claims, 271 accepted, 14 proposals, 5 skills; MIGRATION QUEUE: 218 units, 0 pending, 0 partial
```

Both commands were rerun after the material-acceleration repair; the subsequent restoration of the originally reviewed `|U_side|≤2` bound and this receipt do not affect their algebraic execution. The oracle checks the moment coefficients, not the pressure-contrast prefactor `2ρδ`, which follows by subtracting the two actual quadratic stress tensors in the derivation. Numerical finite-Fourier Bloch samples elsewhere in this investigation changed with truncation; no all-mode stability verdict follows. Finite **initial** perturbation excess, nondegenerate index transport over common smooth lifespan and a nonzero initial exterior octupole delimit the seed construction; no persistent carrier, electric charge, electron, neutrino or quantum/relativistic bridge has been established.

The separate full-pressure WKB **ODE**, not Euler PDE, certificate now checks two parameters on the repaired half-sum basis. Its refactored script exited successfully:

```text
.venv/bin/python proposals/P253-euler-particle-mechanisms/attempts/0164-helical-medium-defect/certify_helical_wkb.py
e=3/10 global_matrix_norm_error_lt: 14.874932635380213...
e=3/10 true_wkb_trace_interval: [-66.82629536778945..., -7.326564826268599...]
e=3/10 certified_wkb_trace_below_minus_two: True
e=1/10 global_matrix_norm_error_lt: 0.039924320107161289...
e=1/10 true_wkb_trace_interval: [0.17538793230706056..., 0.33508521273570572...]
e=1/10 certified_one_ray_wkb_trace_between_minus_two_and_two: True
```

`floquet.md` gives the exact translation-periodic whole-space streamline, Euler WKB ray and analytic Peano/Cayley/global-error bounds. Two independent reviews initially found that `n=δx+δy` did **not** match the certified matrix, and one also caught the distinction between closure on the torus quotient and translation on `R³`. The derivation and script use `n=(δx+δy)/2` and explicitly retain the whole-space state. Both reviewers rechecked and closed their findings for the **original e=3/10** certificate. Two separate reviewers then independently checked the new e=1/10 bounds and scope without findings; they did not rerun the script. The interval calculations depend on `mpmath.iv` enclosing arithmetic and trigonometric values. Neither certificate evolves a localized Euler PDE perturbation or establishes all-mode stability.

`packet.md` supplies a separate mathematical transfer from that certified WKB ODE multiplier to the linearized Euler `L²(R³)` **operator norm**. Two independent reviewers first rejected its unsupported finite-time Leray remainder; after the exact identity `L_ext=−U·∇+(I−2P)∇U`, fixed-horizon oscillatory multiplier estimate, Duhamel bound and correct `(N,σ,m)` limit order were supplied, both rechecked and accepted at this scoped theorem. The fixed-time whole-space shortwave result in Shvydkoy, https://arxiv.org/pdf/math-ph/0412019, §4.2/formula (75), is an independent analytic cross-check; the compact-torus *equality* of essential radii is not imported. This is a mathematical PDE semigroup result with high-frequency initial data and continuous frequency limit, **not** a numerically evolved finite-frequency wavepacket, fixed eigenmode, nonlinear background/defect instability or #203 particle/quantum claim.

## Fixed `e=1/10` adverse material-polarization certificate

After `material.md` and `certify_helical_material.py` were integrated, the new interval command exited naturally:

```text
.venv/bin/python proposals/P253-euler-particle-mechanisms/attempts/0164-helical-medium-defect/certify_helical_material.py
material Cayley trace: [2.00003883474930394413971181182793546209326874, 2.00003883474930394413971181182793606112968986]
material global matrix norm error < 0.000012062687500888888888888888888888888888889898
true material trace interval: [2.00001470937430216636, 2.00006296012430572192]
e=1/10 certified material trace above two: True
```

The exact determinant is one by the `log h` identity; the interval Cayley determinant enclosed one to approximately `8·10⁻³⁴`. Parent independent RK4 at `N=400,800,1600` gave material traces `2.000038835434,2.000038834770,2.000038834729`, and direct six-dimensional BAS at `N=1600` gave `|ξ(T)|≈0.99378763546`, `|b(T)|≈1.00625119928` and transverse residual `<1.5·10⁻¹²`. The first proposed proof used the **false general** identity `(ξ×b)'=A(ξ×b)`; a counterexample invalidated it before acceptance. The repaired proof restricts to the invariant two-dimensional normal plane and uses the exact full-pressure identity `d log(|ξ||b|)/dt=−tr D`; the independent investigator repeated full BAS through 50 periods with covector/amplitude return errors below `1.5·10⁻¹¹`. The certificate remains conditional on `mpmath.iv` outward enclosures and the analytic Peano/Cayley bound. The pre-existing `packet.md` fixed-horizon whole-`R³` argument applies with a nonzero scaled-return covector; this is operator-norm growth, not a finite-frequency Euler simulation or nonlinear indexed-defect verdict. No independent revision-bound review of the new material-ray derivation has yet been credited.

After the metadata and documentation edits, `env PYTHONPATH=src .venv/bin/python scripts/validate_repository.py` exited `WORKFLOW VALID: 271 claims, 271 accepted, 14 proposals, 5 skills; MIGRATION QUEUE: 218 units, 0 pending, 0 partial`. These counts validate schema/workflow only; they do not promote a particle claim or certify the new analysis.

The author-only [chiral packet pressure control](chiral-pressure.md) uses the same `e=0` indexed Euler initial states, not a different microscopic action. Exact quadratic-stress subtraction isolates the four-state index-by-packet-chirality acceleration; the stress first moment separately checks its nonzero algebraic core-only field. During exploration, Gauss–Hermite orders 40/64/96 agreed on the far coefficient `3.6251230404704e-7`, packet Gauss–Legendre orders 32/48/64/80 stabilized the nonzero near-field coefficient, and directly differenced four full-stress increments matched the isolated `W*P` integral at `R=1.25,1.5`. The separated `W*P` term has a Gaussian-overlap bound, not a claimed numerical exact zero. These quadratures were transient author calculations, not a committed replay script or independent peer review; the supplement gives the fields, integral formulas, node prescription and outputs for reproduction. No Euler time evolution or physical chiral weak current has been established.
