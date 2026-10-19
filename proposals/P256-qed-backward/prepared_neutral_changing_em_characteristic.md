# Changing-quintic physicalG EM characteristic (partial action only)

This pair implements the literal native `NoetherGaussianBank.evaluate(y)` EM
characteristic on all 52 real coexact coordinates and all three harmonic
coordinates. It consumes the accepted once-seeded shared cache. It neither
reprepares nor refilters the state, resets a profile, changes the law, projects to
hard modes, replaces coexact flow by its mean, nor evolves quantum `H_EM`.
The original 81 oriented native sine-clover curls remain literal sine functions.

## Direction, wave convention, and ABI

For `G = -i(V·d + div(V)/2)` and `Hdrive = -Omega(t) G`, the selected
quintic clock is the existing `pulse_point(clock,t/clock)`. The forward
characteristic and wave transported *along that characteristic* satisfy

- `zdot = -Omega(t) V(z)`, with `z=(y,theta)` and
  `V=concat(provider.flow,provider.harmonic)`;
- `d log(halfdensity)/dt = +Omega(t) div(V)/2`;
- `psi1(z1)=halfdensity*psi0(z0)`.

Writing the wave as the common coexact Gaussian times a coefficient function
therefore gives the moving-endpoint coefficient factor
`exp((||y1||²-||y0||²)/2)*halfdensity`. Harmonic coordinates are not secretly
included in this coexact Gaussian. A consumer evaluating at a **fixed final
coordinate** must use the inverse characteristic to obtain `z0`; substituting
the forward endpoint into the initial function has the wrong direction.

`evolve_em(bank,clock,t0,t1,em0,steps=8)` returns:

- `em_endpoint`, `times`, `em_path`, and literal `provider_path`;
- `integrated_log_half_density`, `half_density`, `gaussian_ratio`, `factor`,
  and `factor_log`;
- `initial_coordinate_jacobian`, the full 55×55 forward variational map;
- `initial_log_half_density_gradient` and `initial_log_factor_gradient`,
  derivatives with respect to all 55 initial coordinates;
- `integrated_anchored_lambda_C = -integral Omega lambda_C dt` and
  `integrated_harmonic = -integral Omega harmonic dt`.

Every provider record includes nonzero harmonic and anchored `lambda_C` values,
raw edge flow, divergence, and gamma. These are carried for subsequent source-CAR
coupling, not replaced by zero or called an already coupled fibre evolution.
The analytic sine-coefficient Jacobian drives the variational ODE with the same
classical RK4 stages as the characteristic. The last three Jacobian columns are
retained even though this provider has no theta dependence. Creators/current
pullbacks can use the inverse variational map and the density derivatives;
no finite-Fock replacement is introduced here.

## Actual native execution

The executable loads the exact accepted cache NPZ
`10d61adb4a374a05e45338522abefcda260ed25036396b1bea26c48c57e4bee3`
and manifest
`90a6d7f34617540042406a07c5d8690cecdc456b2d585ee150c152c7d2e07f9a`.
Initial coordinates are the core's nonzero seed `222913`, all52 Gaussian draws
multiplied by `.07`, and harmonic theta `(.19,-.11,.23)`. The native clock is
566. The actual interval is `t0=.37T=209.42`, `t1=209.451`, with 8 stages of
steps and independent 16/32-step refinement runs. Omega changes over this
interval; it is not a frozen instantaneous step.

The successful scientific source is
`2c1c5500629601b33379131229f209eb22f162006ded1b2533a84ef738770870`.
Its completed CLI wrapper is `local://issue222-changing-em-attempt3-wrapper.json`,
science is `local://issue222-changing-em-science-attempt3.json`, and the exact
100 imported-source archive is `local://issue222-changing-em-sources-attempt3.zip`.
Natural exit was 0, wrapper wall 22.442651962 seconds, scientific wall
21.398908987 seconds, stdout 127667 characters and stderr empty. The wrapper
records the exact absolute Python/argv/cwd and both streams. All loaded source
bytes were stable during execution. No old scientific CLI was rerun.

Observed scale-sensitive witnesses:

| Witness | Observed result |
|---|---:|
| coefficient factor | 1.0000007217153195 |
| integrated log half-density | 1.0113117253314581e-11 |
| endpoint 16/32-step discrepancy / actual motion | 5.325959760659934e-12 |
| log-half-density refinement relative discrepancy | 2.0248169758109965e-10 |
| full55 Jacobian directional FD relative error | 4.080863593439734e-12 |
| log-factor initial gradient FD relative error | 1.6119628095709517e-08 |
| log(det Jacobian)+2 log(halfdensity) error | 8.424366853874618e-16 |
| frozen-Omega endpoint difference / actual motion | 6.107988302588233e-05 |
| mean-zero-flow endpoint difference / actual motion | 1 |
| missing-divergence relative factor signal | 1.011311725336572e-11 |

The tolerances compare flow/adverse differences to their own physical scale,
not an absolute `allclose` floor or a source-profile-specific threshold.
Refinement and FD are finite numerical estimates, not certified error bounds
or astronomical/legal-volume simulations. The local determinant identity
checks half-density/Lebesgue volume compensation; it is **not** a full-field
L2 quadrature, global state normalization, or a full-pulse norm measurement.

## Independent provider and actual native source witnesses

The literal native `primitive_flow` is evaluated independently at a nonzero
field chart point. Its raw81 edge coefficients are independently projected
with the native coexact vectors, harmonic mean, and correctly signed,
reference-anchored Gauss pseudoinverse. Maximum provider relative error is
1.0674436090061714e-13. All55 coordinate finite differences give relative
Jacobian error 5.0991580839863964e-12. Independent native projections and FD
divergence verify `gamma=i(Vy·y-div(V)/2)` with scaled error
3.370899571318659e-13, while the actual independent finite-Weyl Gaussian gamma
function gives relative error 2.899669938378761e-14. These are not mocked
provider echoes. Harmonic norm is 9.806161787854837e-06 and anchored lambda norm
5.270344206576326e-05, both explicitly nonzero.

For any differentiable matrix source `B(z)` and arbitrary two-bank wave `psi`,
the actual first-order product rule is
`L(B psi)-B Lpsi=(V·dB)psi`, where the Gaussian-divided operator has zeroth
term `div(V)/2-y·Vy`. The native witness evaluates all four original words,
448 full CAR rows, both neutral banks, and the original SOURCE19/SOFT profiles
with two distinct arbitrary smooth coefficient functions, not only a fixed
prepared coefficient or finite-Fock wave. The native N3 SOFT restriction is
exact zero in the accepted cache and is retained as such, not fabricated into
a nonzero second source profile. The two arbitrary neutral functions are
nonzero and independently varied.

The native canonical source has no coexact derivative but does have a harmonic
source difference. A harmonic-scaled FD is necessary to avoid subtracting
indistinguishably small source phases. Measured product-rule scaled error is
1.0720324995429376e-11; native full-direction versus harmonic-only source
jet discrepancy is 1.876397573727493e-09, with jet norm
6.482063802908544e-06. Common transport of `B psi` versus the commonly
transported factors agrees to 1.0616959051569058e-16. Incorrectly resetting B
to the endpoint physical source leaves relative signal
1.65443361742114e-09. The algebraic product rule is general; these FD values
are native point witnesses, not a global-L2/source-CAR time-flow theorem.

Two failed attempts are retained rather than retagged: attempt1 omitted the
required harmonic `theta/N` contribution in native A and the authoritative
provider rejected it; attempt2 used an unscaled source FD and failed its
relative refinement assertion. Attempt3 fixes the actual harmonic boundary
and uses a scale-aware source displacement without changing profiles/law.

## Remaining scope

This is **partial physicalG EM action only**. It does not include quantum
`H_EM`, harmonic kinetic evolution, coupled Phi/link/matter/Ref/Root dynamics,
source-CAR connections, neutral RIGHT/current time propagation, a complete
physical pulse, closed rotation, global field-L2/state norm, or legal-volume
unrepresented-mode bounds. Neither full physical trajectory nor closed-minus
nor Born transfer is paid by this pair. The parent consumer owns joining this
ABI to those distinct still-required actions without recapturing the initial
state/bank or resetting the actual source.

Current-source real Pyright RPC and positive AST evidence are stored in
`local://issue222-changing-em-post-real-lsp.json`; the preimplementation
provider RPC is `local://issue222-changing-em-pre-real-lsp.json`.
