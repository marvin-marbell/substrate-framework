# Changing quintic mean-matter interval

This new slice evolves the full light108, heavy108 and Ref4 mean bodies, using unchanged `mean_matter_generators(model,data)` inputs:

\[
\dot U=-iA(t)U,\quad A_L=h_L-\Omega(t)M_L,\quad
A_H=-\Lambda h_{\rm bare}-\Omega(t)M_H,\quad A_R=-\Omega(t)M_R.
\]

`evolve_mean_matter(generators,Lambda,clock,t0,t1,steps=8)` returns `unitaries`, `endpoint_bodies`, `connection`, `pulse_endpoints`, `refinements` and `control_metrics`. The connection is the full pullback **`-U.conj().T @ A_end @ U`**, not `-A_end`. The latter is generally incorrect when the evolving bodies do not commute. No fresh ground/eigensolve, refilter, SOURCE-function recapture, spin fit, phase alignment, polar-unitary repair or law retuning occurs.

The reusable `propagate_bodies(body_provider,t0,t1,steps=8)` core takes a callback returning Hermitian matrices keyed `light`, `heavy`, `reference`. It returns full unitaries, endpoint bodies and the same proper connection. A caller may provide genuinely varying field-dependent bodies; the scientific execution described here provides **mean-only** bodies. Exporting a callback is not evidence of a coupled field trajectory.

## Approximation and boundary meaning

The integrator uses fourth-order commutator-free two-exponential Gauss time ordering. With `c1,c2=1/2∓sqrt(3)/6` and `a1,a2=(3∓2sqrt(3))/12`, each interval multiplies the early exponential on the right and the later exponential on the left. All matrices and heavy phases remain present.

This is explicitly a **finite-step approximation**, not exact full-pulse evolution. `U(t0)=I` means an interval propagator acting on a once-prepared boundary input. It does **not** mean the missing preparation-to-`t0` prefix has already evolved. The default actual native clock is 566; the executed interval is `[209.42,209.451]`, i.e. `t0=.37T`, `dt=.031`, using 8 steps and 16/32-step comparisons.

**Commonfield, field timeconnection and completing-square commonenergy are not included in this slice.** They remain required in the parent coupled consumer. This module is not a whole Xi/matched-BXi physical trajectory, not a full-loop minus proof, and not a replacement of fullcurrent/neutralRIGHT obligations.

## Actual native proof

The final CLI loaded the unchanged shared checkpoint, with NPZ SHA256 `10d61adb4a374a05e45338522abefcda260ed25036396b1bea26c48c57e4bee3` and manifest SHA256 `90a6d7f34617540042406a07c5d8690cecdc456b2d585ee150c152c7d2e07f9a`. No full Hamiltonian/Gram/preparation recompile or frozen scientific rerun was performed.

The actual pulse changed from `Omega=.0180954438025781` to `.01809765433728715`, with theta from `1.6776872839307815` to `1.6782482769540072`. Positive controls gave:

| full sector | wrong frozen Omega(start), Frobenius | wrong endpoint connection, Frobenius | unitarity defect |
| --- | ---: | ---: | ---: |
| light108 | 9.91386716338e-8 | 2.42779287067e-8 | 6.08174606746e-14 |
| heavy108 | 7.16911467842e-8 | .00222401463166 | 3.33161511468e-10 |
| Ref4 | 1.00942037413e-8 | 2.60208521437e-18 | 6.66133814775e-16 |

Ref4 is a commuting family; its wrong-endpoint connection branch therefore is not a discriminating control. Its analytic theta-integral error is `2.24208655282e-16`. The light sector independently discriminates both required wrong branches. Connection cancellation in the returned matrices is algebraic by construction; it is not an independent integration accuracy claim.

The native smoke applies these actual full unitaries through the existing orbit-transport API. Heavy occupied shape remains 108×54, light occupied counts remain 54 and 58, both light occupied/projector transports and heavy54/Ref4 transport errors are zero. The initial SOURCE-function bank is the same object with the same seed-orbitals digest, prepared14 is unchanged, and compiled H/G/coherent neutral coefficients/SOURCE19/SOFT maps remain hash-identical. No initial bank is rebuilt on transported orbitals.

## Honest refinement and heavy numerical debt

The observed 8→16 and 16→32 Frobenius differences are:

| sector | 8→16 | 16→32 | ratio |
| --- | ---: | ---: | ---: |
| light | 1.26121747339e-14 | 9.85743569059e-14 | .1279458 |
| heavy | 2.25750540459e-11 | 1.38531550936e-11 | 1.6295966 |
| Ref | 4.44089217501e-16 | 1.33226763541e-15 | .3333333 |

These native observations **do not show ratio16 convergence** and are not certified error bounds. Light/Ref differences are at roundoff scale. Native Lambda is 16777216 and the heavy interval exponent one-norm is 427318.845606771. Exact CF4 factors of Hermitian bodies are unitary, so the observed heavy unitarity defect is numerical, not a time-ordering truncation defect. The large heavy wrong-connection norm may contain amplification of numerical nonunitarity and must not be interpreted entirely as a physical noncommutation signal.

Separate constant-body controls have no time-ordering error: repeated subdivision differs from a single exponential by `5.17752598082e-15`, while the same constant CF4 factorization differs by `2.45082795844e-14`. These observations do not bound errors of the varying heavy calculation. No phase is floored, fitted or removed, and no numerical failure is silently repaired. Full heavy large-phase accuracy remains distinct from finite-step time-ordering debt.

An independent callback smoke used full108 noncommuting bodies with a DOP853 2×2 repeated-block reference. Actual core errors at 2/4/8 steps were `.00340515281053`, `.000205236338398`, `1.26959322897e-5`, giving ratios `16.5913738137` and `16.1655192951`; wrong endpoint connection norm was `12.5323371274`. This verifies factor ordering in a non-roundoff regime, not improved native-heavy accuracy or microscopic coupled closure.

## Reproduction and exact receipts

Run from `proposals/P256-qed-backward` with the project Python. Let `LOCAL` be the receipt directory named by the supplied local URIs:

```sh
/home/administrator/substrate-framework/.venv/bin/python prepared_neutral_changing_mean_frame.py \
  --cache-npz "$LOCAL/issue222-shared-photon-state.npz" \
  --cache-manifest "$LOCAL/issue222-shared-photon-state-manifest.json" \
  --output "$LOCAL/issue222-changing-mean-matter-final-science.json" \
  --matrices-output "$LOCAL/issue222-changing-mean-matter-final-matrices.npz"
```

Final source SHA256: `25c22e08bd7eacb446bcfb968c37e0e7349657bc7fad0aac1c814d96d2fbc4ec`.

The final natural exit was 0, external wall 18.4342521911 s, internal science wall 16.4385808290 s, stderr empty. The source epoch recorded 99 initially imported and 100 final imported project sources, with no same-path changes; the final source archive binds the imported set. Full arrays are preserved, not replaced by scalar metrics.

- `local://issue222-changing-mean-matter-final-science.json`, SHA256 `ad78ae6a1d6bd73d8cda4278b03261d389b3b5b720592a1d4538d70640693232`.
- `local://issue222-changing-mean-matter-final-matrices.npz`, SHA256 `6dd38f55d3b1e4050b39cdf15f98aabaed5731ccfcb526b7310ef003adc4e71b`.
- `local://issue222-changing-mean-matter-final-execution.json` plus final stdout/stderr contain exact argv, cwd, exit, wall and stream hashes.
- `local://issue222-changing-mean-matter-callback-smoke.json` contains the independent executed callback probe and reference parameters.
- `local://issue222-changing-mean-matter-final-real-lsp.json`: actual Pyright stdio symbols/definition/references, received `publishDiagnostics=[]`, 88 symbols, six AST declarations, source stable, exit0. Device LSP was unavailable; direct real RPC was used. A positive structural AST query separately located both core calls.
- `local://issue222-changing-mean-matter-final-receipt.json` and `local://issue222-changing-mean-matter-final-source-archive.tar.gz` bind final sources, this document, science, full matrices, streams, RPC and smoke evidence.

The earlier completed mean-only execution before the parent's callback request is preserved as `local://issue222-changing-mean-matter-v1-archive.tar.gz`, not relabeled as final source evidence. Governing plan remains `issue-222-fullfield-closedrotation-continuation-20261006` at revision `57838ca23dbf1424677c1c2f4e183784463faa3f759a85ec2a2b9722ff1fd9ff`; no formal campaign status or debt is changed by this scoped result.
