# Full-word original EM point operator

This NEW component retains the four physical Ref words separately for each plus/minus bank. It does not modify the frozen EM supplier or the parent integrator, prepare a new state, normalize any individual word, or supply a quantum evolution/global L2 result.

## Interface and coefficient binding

`operator_matrices_channels(model, sample, direction, sector, coefficient_matrix, physical_point=None, q_shift=None, h=None, trace_samples=4, seed=222107, derivative_steps=None)` accepts the ORIGINAL full bank rows: 214 neutral or 1728 charged. Its columns are channels. All returned matrices, including `total`, are **UNWEIGHTED**. `cutoff_squared` is metadata: the parent multiplies the original foot cutoff squared ONCE in GLOBAL quadrature. `trace_samples` and `seed` match the Phi signature but EM sums ALL55 deterministically.

The shared `coefficient_channels` builder belongs to `prepared_neutral_phi_word_channels.py`. Neutral columns are the exact cached neutral state and its actual plus-minus charge-label direction. Charged ordering is the exact cached normalized coherent SOURCE, then `+word1,+word2,+word3,-word0,-word1,-word2,-word3`. The seven unit-word directions are frozen `word_maps` acting on the original charge-masked neutral coefficients, divided by the SAME original cached GLOBAL SOURCE normalizer. The coherent first column replaces `+word0`, whose original beta is nonzero. No new Z, individual normalization, spin fit, whitening, filter or mutable source is introduced. Initial coefficient `(1,0,...)` is the original prepared state exactly; subsequent coefficients need not keep the raw quartet fixed.

For a channel matrix C, `V=sample[sector]@C`. The wave matrix is `V[a,c]*m[r]`, flattened **CHANNEL-major then six modes**: 2 physical rows by 12 columns for neutral and 448 physical rows by 48 columns for charged. The six multipliers are the unchanged original constant, linear/Hermite normal-direction modes, radial normal mode, first harmonic-coordinate mode, and radial 52-coexact mode. Each original full-bank derivative is contracted with C ONCE, not propagated or assigned energy separately by word.

## Literal operator and density

For each of the 55 original EM directions j,

`D_j W = (original_directional_basis_j @ C) * m + V * D_j m`.

The original producer already contains the derivative of `exp(-sum(em55**2)/2)`, including the THREE harmonic coordinates, ONCE. The analytic multiplier jets are reused from the frozen EM common-form supplier. Two producer FD steps are retained, h and h/2.

The full complex cross-word matrices are

`K = sum_j weight_j * (D_j W)^dagger (D_j W)`,

`G = W^dagger W`,

`V_EM = (sum_{j<52} omega_j * em_j**2 / 2) * G`.

Weights are `omega_j/2` for ALL52 coexact coordinates and `UE/N` for the THREE harmonic coordinates. `total=K+V_EM`. Literal Gaussian kinetic and potential retain zero point once; nothing is added or subtracted. Harmonic SOURCE/Peierls derivatives are present through the original full bank. The bank is evaluated at the FOOT, not rebuilt at the physical translated point. Mean Phi translation is EM-independent, so there is no additional EM Haar derivative. This component does not preapply H_D/body/scalar/W phases. Full Phi/Gauss/Wick, HC, matter H_D, scalar potential, GLOBAL quadrature, time connection and actual coefficients belong to the parent.

## Executed scientific proof

Actual CLI: project Python `/home/administrator/substrate-framework/.venv/bin/python`, cwd `/home/administrator/substrate-222/proposals/P256-qed-backward`, entrypoint `prepared_neutral_em_word_channels.py`, flags `--cache-npz .../issue222-shared-photon-state.npz --cache-manifest .../issue222-shared-photon-state-manifest.json --output .../issue222-em-word-channels-science.json --source-archive .../issue222-em-word-channels-source-inputs.zip`. The exact absolute argv is in `local://issue222-em-word-channels-execution.json`.

Natural exit **0**, actual wrapper wall **1275.4410900759976 s**, empty stderr, raw stdout byte-identical to science JSON. Executed/current source SHA256 `c4ee0dc6bed52546d8c67682dca446130936afc7970114c1b7451b6055734fb7`. Cache SHA256 `10d61adb4a374a05e45338522abefcda260ed25036396b1bea26c48c57e4bee3`, manifest `90a6d7f34617540042406a07c5d8690cecdc456b2d585ee150c152c7d2e07f9a`, science/stdout `a7cbd25d24da95ad568de7a28579703108bd35b4ce3e0d65433269742e8a59db`, source/input archive `7787084ce9cbddc4f353b6a465a8af04acaba23ca817a091cf7b125f8c846eb0`. Guard bytes were captured and archived BEFORE loader/native reads: SHA256 `5aeceaf4a7cb104346ebc5d1130e22bae6090b5cf96afcd297e02a6facc22128`. All 110 loaded-source bindings and three exact inputs are retained; preloaded sources stayed unchanged. Late-loaded sources are explicitly labelled, not claimed pre-read.

At the SAME frozen original q/em point, independent literal packets reevaluate the actual bank at `em±h e_j`, all six literal multipliers, and the FULL55 Gaussian ratio. Their variations independently test derivatives and complex cross-bilinear kinetic/potential matrix contractions at BOTH `2^-18` and `2^-19` for neutral, SOURCE19 and SOFT. Maximum cross-bilinear relative discrepancies: neutral `6.542458747967374e-11`, SOURCE19 `9.99835449817662e-12`, SOFT `7.470104403156606e-11`. Observed coarse/fine kinetic relative differences: SOURCE19 `5.669669267441588e-12`, SOFT `3.421285185032719e-12`, neutral zero (analytic neutral jet).

Plausible wrong physics is discriminated: dropping the Gaussian changes K by roughly its full norm; dropping harmonic kinetic changes K relatively by `8.646251724464297e-7` neutral, `2.9121395775348877e-6` SOURCE19, `4.0310067729598455e-6` SOFT. Freezing to the original coherent channel produces complex-bilinear errors `232.72754974461972` SOURCE19 and `66.02421596441577` SOFT. Literal harmonic AND coexact held-bank derivatives are nonzero for every charged channel. Full cross-word entries and interference are retained, not inferred merely from matrix dimensions.

An additional actual point-fibre discriminator (`local://issue222-em-word-channels-physical-action.json`, natural0, 13.66451773000881 s) uses the recorded constant-mode K and eight physical word vectors. It solves their point Gram for the projected K action, then removes the actual initial coherent PHYSICAL vector, rather than just deleting a coefficient row. Outside-coherent ratios are SOURCE19 `5.211390335407578e-8` and SOFT `2.881074720167713e-7`, with well-conditioned point Grams (19.7024 / 16.4481). These small but resolved point actions are NOT a global Hamiltonian trajectory or norm claim; the general matrix preserves them instead of compressing them away.

## Tooling, custody and limits

Actual Pyright 1.1.414 stdio RPC was exercised before editing on the frozen supplier and after science on this exact current source: document symbols, definition, references, and received `publishDiagnostics=[]`; post 121 symbols, six AST declarations, natural0. The mounted xd LSP device is unavailable; direct REAL Pyright RPC was used, not a simulated checker. Positive specialized Python AST queries found actual `kinetic_form` calls; current Python AST also parsed all six declarations. Receipts: `local://issue222-em-word-channels-pre-real-lsp.json`, `local://issue222-em-word-channels-post-real-lsp.json`, `local://issue222-em-word-channels-current-AST.json`.

Receipts and source/input archives are private0600 and serialize ONLY explicit nonsecret scientific metadata, never `os.environ` or credentials. Frozen sources and the running historical parent were not edited or rerun. This finite-jet/point/Galerkin construction leaves certified FD remainders, GLOBAL integration, stochastic full-Phi estimates, full domain/harmonic/legal-N transfer, quantum-law and microscopic parameter debts open. Fundamental campaign status remains OPEN. Parent owns scientific integration; PhysicalCarrierTransport alone owns final consumer integration/publication; WorldModelSync04 alone owns canonical plan/model writes.
