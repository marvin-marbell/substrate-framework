# Original quantum EM common form on the paired characteristic

`prepared_neutral_changing_em_kinetic.py` advances the original positive EM/harmonic form on the **same once-seeded paired RAW/55EM/native-H_D/literal-scalar interval**. It does not replace preparation, normalize a profile at the selected point, construct a receiver, or propagate the full quantum Hamiltonian.

## Family and physical derivative

The selected boundary is clock 566, `t0=.37*566`, `t1=t0+.031`, eight CF4/RK4 steps. The immutable shared cache has SHA256 `10d61adb4a374a05e45338522abefcda260ed25036396b1bea26c48c57e4bee3`; manifest `90a6d7f34617540042406a07c5d8690cecdc456b2d585ee150c152c7d2e07f9a`. Initial neutral coefficients are the fixed 214-vector, and SOURCE19/SOFT use their fixed globally normalized 1728-vectors. No new Z, amplitude floor, bank, ground, filter, or pointwise division is introduced.

For each of all 55 initial coordinates, both symmetric finite-difference steps `2^-13` and `2^-14` are actually evaluated. Each member evolves the complete EM characteristic and evaluates physical

`a=(legs@y).reshape(V,3)+theta/N`.

The base `native_provider(t)` owns the RAW Phi/link trajectory. The varied light body is the full108 original `law.onebody(Phi,links,a)-Omega*(scalar_orbital+diag(repeat(lambda_C,4)))`. Base heavy/ref bodies are retained unchanged in the same `propagate_bodies` CF4 calculation. The observed heavy and reference unitary differences are exactly zero. This is **parameter-family independence**, not removal of their physical state phases.

Each initial coefficient function is evaluated through `model.basis_at(q0,emvar,initialpoint)`: the original SOURCE/harmonic functions, hard polynomials and all finite Noether partner functions remain intact. The complete initial55 Gaussian is restored by

`exp[-(emvar@emvar-embase@embase)/2]`.

The common coefficient keeps the base RAW half-density, literal scalar-Phi phase and the varied EM half-density and `exp(2i*sum(integrated anchored lambda_C))`. The massive-sea convenience expectation is not a second scalar counterterm. Heavy/reference transport and scalar-Phi phase are not phase-fitted away.

RAW Phi, RAW Haar density, section inverse gauge, qtarget, heavy/ref and scalar-Phi phase are exactly em-independent in this family: their producer has no em argument except the independent EM branch. Consequently their EM derivatives are zero, not suppressed numerical signals. The W carrier factor and its inverse coefficient multiply to one; they are not extra physical EM drive. The returned analytic EM half-density gradient is only the finite-step RK4 variational derivative, not an exact continuous-time/global remainder certificate.

Write `K_j=-i Ubase^dagger (dU/dem0_j)`. This uses full RAW108 U, without polar repair, Hermitian symmetrization or phase alignment. The complete 55x55 characteristic Jacobian converts initial to final physical derivatives: derivative rows are multiplied by `inverseJ.T`, and full K matrices are transformed by the same rule.

For each physical axis the form evaluates

`||dv+i dGamma(K)v||^2 = ||dv+i P dGamma(K)P v||^2 + <v, [P dGamma(K^dagger)dGamma(K)P-(P dGamma(K)P)^dagger(P dGamma(K)P)] v>`.

The second moment is the full Slater Wick product **before compression**, including both neutral filled seas (neutral RIGHT) and charged particle/hole sectors with Ref3/Ref1. No neutral-PH or charged-PH component is discarded by projecting first. The complex finite-FD K is used as measured; its non-Hermitian residual is reported.

## Original law and numerical scope

The weights are all52 `omega_j/2` and three harmonic `UE/N`. The original magnetic potential is separately `sum(omega_j*yfinal_j^2/2)*||v||^2`. Harmonic potential is absent in the original law. `HardFockBasis.zero_point=7.973272719851423` is the given all52 physical reference: it is already present in the original differential kinetic-plus-potential operator and is neither subtracted nor added again. A pointwise kinetic-plus-potential density is **not** an integrated eigenenergy or zero-point expectation.

All numbers below are pointwise densities in the constant base Gaussian/Haar factored convention, not profile-normalized energies or global field-L2 integrals. Refined results:

| State | Full kinetic | Wrong project-first kinetic | Lost PH | Original magnetic potential | Harmonic kinetic |
|---|---:|---:|---:|---:|---:|
| Xi neutral | 0.03644283009762071 | 0.036442829158602454 | 9.390182598515834e-10 | 0.0364431769256689 | 3.153721004100801e-8 |
| SOURCE19 BXi | 0.03604749845146501 | 0.03604749753870853 | 9.127564714374588e-10 | 0.036047766978891005 | 1.0570494406442913e-7 |
| SOFT BXi | 0.03587901967549152 | 0.03587901876709959 | 9.083919349397413e-10 | 0.035879248650709746 | 1.435547922049285e-7 |

Coarse-to-refined full kinetic differences are respectively `4.029970732122301e-10`, `3.985283492102809e-10`, and `3.966102030128482e-10`; each PH signal exceeds this observed full-form FD change. PH changes themselves are at most `3.58e-18`. Thus project-before-derivative is measurably adverse on this finite experiment, not an asserted certified bound. The complete generator refinement difference is `7.835425604512653e-10` Frobenius.

The full55 Jacobian condition is `1.000238812199501`; inverse residual `1.7306664252250995e-15`. Coarse/refined Jacobian FD errors are `2.9421087953799874e-12` and `5.8972788754261174e-12`. EM log-half-density gradient errors are `7.198439167013277e-17` and `1.5782958671037799e-16`. Maximum finite-FD non-Hermitian K residuals are `1.290056962170302e-10` and `2.458039275940652e-10`: refinement reaches floating-point cancellation in the tiny U derivative, not a presumed clean asymptotic rate. All55 axis decompositions, interference, scalar derivatives and second moments are in the science receipt.

## Exact execution and language evidence

The new scientific CLI naturally exited zero in **629.9610753927846 seconds** wrapper walltime, empty stderr; raw stdout equals the saved science file byte-for-byte. It archived all **106** actual loaded project sources, stable across the computation. The executed/current new source SHA256 is `44c1fa1d808c0ccc4adb045e6b1e5d7c5efd32ac85d9cca833f135108d9bc7e4`; the coupled parent binding is the actual return-export-enabled current source, not the first historical archive retagged.

Reproduction, from this project directory:

```sh
/home/administrator/substrate-framework/.venv/bin/python prepared_neutral_changing_em_kinetic.py --cache-npz <immutable-shared-npz> --cache-manifest <immutable-manifest> --output <science.json> --source-archive <sources.zip>
```

Own local artifacts:

- `local://issue222-changing-em-kinetic-execution.json`: actual absolute command, cwd, natural exit, exact streams and science. Full environment metadata was removed after sensitive-value discovery; `local://issue222-changing-em-kinetic-metadata-redaction.json` preserves original hashes and explicit sanitized rebinding without retaining secret-bearing bytes. No science was rerun.
- `local://issue222-changing-em-kinetic-science.json`: both full55 FD levels and all axis/state rows.
- `local://issue222-changing-em-kinetic-sources.zip`: full actual loaded source bytes.
- `local://issue222-changing-em-kinetic-stdout.txt` and `...-stderr.txt`: actual captured streams.
- `local://issue222-changing-em-kinetic-inputs.json` and `...-inputs.zip`: exact consumed NPZ/manifest/guard identities, archived manifest/guard; immutable shared NPZ retained at its original path.
- `local://issue222-changing-em-kinetic-pre-real-lsp.json`: actual pre-edit Pyright symbols/definition/references and diagnostics on parent/Wick suppliers.
- `local://issue222-changing-em-kinetic-post-real-lsp.json`: exact-current direct Pyright stdio RPC, 106 symbols, native definition/references, actual received empty diagnostics, stable source hash; six positive Python AST declarations. Positive structural AST searches also reached parent CF4 and new precompression Wick call. The xd LSP device gap remains disclosed; direct real RPC is the exercised server.

Governing plan `issue-222-fullfield-closedrotation-continuation-20261006`, assignment revision `668c7d8ea2ec9c71c964e9f16d3c75e8cd879e10433e72fbcad0c3b89c7da417`, pinned general template `89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`, scoped `prepared-neutral-full-field-outcome`. Historical assignment revisions inside frozen imports remain historical.

This advances actual original EM common-form evaluation only. Finite FD/CF4/RK4 remainders, globalL2/common-form integration, quantum Phi kinetic and Coulomb multipair/fullH propagation, whole pulse/physical2pi, Root/Gauss/legal-mode and Born obligations remain separate and unclaimed. No frozen H, preparation, G or old scientific loop was rerun.
