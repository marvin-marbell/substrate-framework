# Prepared-neutral moving-frame photon current

## Exercised outcome and boundary

`prepared_neutral_moving_frame_current.py` exports
`moving_current_control(model, moved, data, sample, light_unitary, reference_unitary)`.
It compares the actual moving provider with an independent fixed-seed pullback at all three literal photon packets (`gi`, `gfwd`, `gout`) and both unchanged profiles (`source19`, `SOFT`). Full Wick current graphs are evaluated **before** particle-hole compression. This is a **POINTWISE off-mean, locally frozen mean-frame** calculation; it is not a whole-field L2 current estimate, a full changing-pulse trajectory, U2π, or Born transfer.

The historical current norm squared `2.261381336887e-7`, with outside-PH part `2.216987466982e-7`, was pointwise. Neither that datum nor the new six point values establishes a whole-field cause.

## Same state, law, packet and physical primitive

The CLI loads the accepted immutable cache, without a constructor sea eigensolve, preparation, H/G compilation, or instantaneous ground-state replacement. It calls the primary owner's `construct_point_frame` once. Native bindings remain N3, UE=2^-20, c=0.125, Lambda=2^24, sigma=2^-13, radius=0.012. The primary uses local step 0.031 at pulse fraction 0.37 of its actual quintic clock T=566. Its returned normal point has norm `4.043781611738843e-5`; all52 coexact coordinates have norm `0.5050581796813881`; theta is `[0.19,-0.11,0.23]`. Omega is `0.0180954438025781` and Omega-dot `7.132041767163893e-5`. This is local frozen-mean transport, not execution of the full pulse.

The photon creator coefficients are exactly `data['packet_templates']`, independently checked against `packet_templates(model.law, model.photon_basis)`. ALL52 oscillator coefficients are `model.photon_basis.packets @ template`. Centre positions and outgoing time6 phase are unchanged. Source19 retains the same one eta for all geometries; the distinct continuous SOFT restriction is not renormalized, whitened, reset or replaced by a periodic mode.

For each packet g, the physical tangent is

\[
 a_g = \frac{\mathrm{em.legs}\,g_{52}}{\sqrt2}
     = \mathrm{em.vectors}\sqrt{UE/\omega}\,g_{52}.
\]

The actual `full_coordinate_light_operators` primitive supplies J using its explicit real/imaginary split of the complex tangent, at the actual real base A. The independent canonical leg formula agrees within `6.64e-19`. Actual light-H coefficient is **1**, not Lambda. Heavy has no U1 current; Ref has no EM onsite current.

Every graph uses the actual filtered point

\[
 n=\mathrm{sample.neutral}\,\mathrm{data.neutral\_state},\qquad
 v=B(q)n=\mathrm{source.source\_matrix}\,n.
\]

Here ||n||²=`0.9791275109127137` and its distance from `fibre.raw_neutral` is `0.010551867387869698`. Point coefficients are never defaulted to the raw two-bank vector, normalized, or re-prepared. The held initial function bank and frozen Xi remain owned by the cache/primary; evaluating moving physical SOURCE does not recapture that bank.

## Independent pullback and actual neutral RIGHT

On the fixed seed fibre, with supplied actual UL/UR,

\[
 J_m=U_L^\dagger J U_L,\quad S_m=U_L^\dagger S U_R,\quad
 T_m=U_R^\dagger T U_L.
\]

The implementation builds independent pulled-back physical marks, not just a transformed projected SOURCE row. It compares full108 current coefficients, all448 SOURCE rows, the full source-relative map and its actual point value, full Wick norm and unrepresented-PH norm against the actual moving provider. All comparisons use **relative 1e-10 tolerances**, without an absolute floor that could hide the tiny SOFT signal.

For Q=dGamma(J), full physical even-CAR gives

\[
 [Q,B]=B(JS,-TJ).
\]

The exact relative map therefore comes from modified full physical marks. If Pn/Pc denote the held neutral/charged fibres, its represented identity is

\[
 P_c[Q,B]P_n=(P_cQP_c)B-B(P_nQP_n)-P_cB(1-P_n)QP_n.
\]

The last term is actual neutral RIGHT, not a small projected RIGHT used as closure. In the plus channel it is built from `-empty† J Pplus S`; in the minus channel from `T (I-Pminus) J occupied_minus`, with the physical Ref ordering. The independent identity residual is at most `1.36e-19` for Source19 and `1.85e-30` for SOFT. The full relative commutator on the initial neutral determinants is a single light/Ref bilinear and is represented by these modified marks; the separate **LEFT current graph is not closed** in the charged fibre.

The full graph is

\[
 \|Qv\|^2=v^\dagger P_cQ^\dagger QP_cv,
\]

using `fibre.bilinear_product(J.conj().T,J)` before compression. Its omitted PH/multipair contribution is retained as the difference from `||fibre.onebody(J) @ v||²`; it is not clipped or discarded.

## Observed six-point discriminators

| Profile | Packet | Full current norm² | Outside-PH norm² | Exact relative norm² | Missing neutral-RIGHT norm |
|---|---|---:|---:|---:|---:|
| Source19 | gi | 1.039635049259126e-7 | 1.023944422227935e-7 | 3.423458189970556e-9 | 4.070638452713087e-5 |
| Source19 | gfwd | 1.039425040945058e-7 | 1.026322824359675e-7 | 2.650034559303505e-9 | 3.792484102290958e-5 |
| Source19 | gout | 1.027344976752774e-7 | 1.016857189323238e-7 | 2.398319674691530e-9 | 3.742070032850763e-5 |
| SOFT | gi | 1.908245567662707e-29 | 1.884585899784529e-29 | 6.838177689413452e-31 | 6.686486128777951e-16 |
| SOFT | gfwd | 1.908216499016064e-29 | 1.884558793685080e-29 | 6.836011922952175e-31 | 6.686473002461978e-16 |
| SOFT | gout | 1.859339802533100e-29 | 1.857614120492804e-29 | 4.577847712767993e-31 | 6.637198203265533e-16 |

Maximum moving/pullback current operator error is `8.91e-19`. SOURCE map error is `1.045e-15` for Source19 and `1.751e-26` for SOFT. Relative-map errors are at most `8.56e-20` / `1.17e-30`. Full-Wick/PH norm pullback errors are at most `5.30e-23` / `5.61e-45` respectively.

Three decision-bearing wrong controls are explicitly labelled:

1. **Project before current:** loses the outside-PH column above, approximately 98–99.9% of each full norm².
2. **Frozen matter/current instead of transport:** retaining original matter coefficients changes represented responses by `3.916e-7`–`5.704e-7` for Source19 and `1.521e-18`–`4.915e-18` for SOFT. Holding the original current coefficient in the moving coordinate basis changes responses by `3.893e-7`–`5.634e-7` / `9.857e-19`–`6.629e-18`, versus correct agreement near `1e-19` / `1e-30`. Comparing only total norm² would conceal much of this phase/vector difference.
3. **Wrong recaptured first-order projected substitute:** the intentionally wrong `Pc Q Pc B - B Pn Q Pn` omits the actual RIGHT term. Its errors are the final table column: 66–78% of the exact relative-vector norm for Source19 and roughly 81–98% for SOFT. “First order” refers to the projected generator action, not an actual instantaneous-ground computation or state reset. No h(q) eigensolve is performed.

These SOFT values are **restriction diagnostics**. The accepted SOFT periodic projection is exactly zero and missing spatial mass remains near one; tiny native restriction output is not legal-N SOFT admission.

## Reproducible execution and source intelligence

Final actual command, with cwd `/home/administrator/substrate-222/proposals/P256-qed-backward`:

```sh
/home/administrator/substrate-framework/.venv/bin/python \
  prepared_neutral_moving_frame_current.py \
  --cache /home/administrator/.omp/agent/sessions/-/2026-09-30T11-33-07-678Z_01a0f216-fbde-7735-a53d-dac838dc2a9a/local/issue222-shared-photon-state.npz \
  --manifest /home/administrator/.omp/agent/sessions/-/2026-09-30T11-33-07-678Z_01a0f216-fbde-7735-a53d-dac838dc2a9a/local/issue222-shared-photon-state-manifest.json
```

Natural exit **0**, empty stderr, process wall `24.045281562022865` seconds. Exact stdout, argv/runtime/cwd, point binding and all100 loaded project-source hashes are retained in `local://issue222-moving-frame-current-final-receipt.json`, SHA256 `836ad2d2a31e464a44f790fa7045e23f2e3792329cd454b94acaba6b69b60bdb`.

- Executed Python SHA256: `eed086fd4fc19df53bb01adbc9bdd602ab1188d871653f20ae3b00a64e62deb4`.
- Cache SHA256: `10d61adb4a374a05e45338522abefcda260ed25036396b1bea26c48c57e4bee3`.
- Manifest SHA256: `90a6d7f34617540042406a07c5d8690cecdc456b2d585ee150c152c7d2e07f9a`.
- Primary constructor source was unchanged across the run; its exact hash is in the receipt.
- Positive ast-grep query exercised actual primitive calls before editing. Final Python AST SHA256: `d1e8ce0c96801c87bf1f22fd48a3f3b27af5b164e6cbbff72908bce76422709c`.
- Actual Pyright 1.1.414 stdio JSON-RPC in the selected proposal root: representative pre-edit143 symbols/0 diagnostics; final77 symbols/0 diagnostics. xd-LSP unavailable; direct real stdio is the explicitly recorded substitute.
- Initial successful absolute-tolerance receipt is preserved separately as `local://issue222-moving-frame-current-initial-receipt.json`; the final source/run strengthens tolerances to relative scale on the tiny SOFT signal without changing the physics outputs.

## Changelog and remaining debt

2026-10-07: Add this independent photon-current/SOURCE-relative moving-frame pair, all52 primitive transport, actual filtered point coefficients, full precompression Wick/PH and neutralRIGHT discriminators. Frozen suppliers and sibling files are unchanged by this owner; no supplier scientific CLI, preparation, H/G compilation, formatter, build or project-wide suite was executed.

This leaf completes the assigned pointwise local-frame current discriminator only. Whole-field integration, unrepresented dynamics over time, legal-N SOFT admission, full physical closed rotation and Born comparison remain separate obligations. No campaign closure or law/parameter-debt reduction is claimed. Governing plan `issue-222-fullfield-closedrotation-continuation-20261006`, revision `57838ca23dbf1424677c1c2f4e183784463faa3f759a85ec2a2b9722ff1fd9ff`, pinned general `89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`, scoped work `prepared-neutral-full-field-outcome`.
