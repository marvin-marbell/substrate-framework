# Once-seeded moving-frame full CAR discriminator

This additive consumer uses the accepted immutable shared cache and the primary
`construct_point_frame` API. It does not construct a new Hamiltonian, diagonalize
a sea, compile a preparation/Gram, reset occupations, renormalize an input, or
alter the original frozen fourteen-component `f(H14_MC) tau` state. The governing
plan is `issue-222-fullfield-closedrotation-continuation-20261006`, revision
`57838ca23dbf1424677c1c2f4e183784463faa3f759a85ec2a2b9722ff1fd9ff`, pinned general
`89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`, work item
`prepared-neutral-full-field-outcome`. Campaign and full-law/parameter debt remain open.

## Contract and construction

`car_source_control(fibre, source, neutral_value, light_body, reference_body)`
accepts arbitrary complex physical light108 and Ref4 onebody matrices, including
native currents. `neutral_value` is explicitly the actual filtered point
`sample['neutral'] @ data['neutral_state']`, not `fibre.raw_neutral`.

The full112 mode order is the held/moved light orbital order followed by all four
reference modes. The neutral determinant masks contain light occupations54/58
and Ref occupations4/0, with their actual two point coefficients. The source
oneparticle matrix is built from the producer's independent literal maps:

\[
 B=\begin{pmatrix}0&S_+\\S_-&0\end{pmatrix},\qquad
 A=\operatorname{diag}(U_L^\dagger A_L U_L,A_R).
\]

The source maps are likewise expressed in the light orbital coordinates. The
actual producer already owns the original words, raw quartet, both complex
partners, all four references and global factor2. No adjoint/sign replacement of
`S_-`, second quartet factor, clipping, small-coefficient erasure or norm floor
is introduced here.

`apply_dgamma` enumerates actual sparse full-Fock determinant actions. For
`c_i† c_j`, annihilation acts first, with the parity of occupied lower modes
computed before each operation; diagonal entries are number operators. The
charged-basis projection restores the existing even-CAR basis phases. The
consumer computes both actions **before** any projection:

\[
 \mathrm{LEFT}=d\Gamma(A)d\Gamma(B)\Psi,\quad
 \mathrm{RIGHT}=d\Gamma(B)d\Gamma(A)\Psi,
\quad \mathrm{LEFT}-\mathrm{RIGHT}=d\Gamma([A,B])\Psi.
\]

It compares full sparse vectors with the direct full112 bilinear commutator,
reports LEFT/RIGHT and their unrepresented particle-hole components, and checks
SOURCE rows and full second moments against `FullSlaterFibre.onebody`,
`reference`, and `bilinear_product(A†, A)`. The numerical assertion tolerance is
`2e-11` on operation-scaled CAR/source and full Wick agreement; the observed
CAR residuals are near machine precision, substantially below that threshold.

The weaker expression `P_C A P_C B Psi - B P_N A P_N Psi` is reported only as a
control. SOURCE can return a neutral particle-hole component into the represented
charged fibre, so its RIGHT term is not generally `B P_N A P_N Psi`. The exact
`full_projected_relative` vector includes this contribution. A returned-neutral
branch with **no RIGHT at all** differs by the reported nonzero full RIGHT norm
and is rejected on every executed case.

## Separate heavy factor

`heavy_common_control` retains the full108 heavy occupied54 Slater factor:

\[
 m_H=\operatorname{Tr}(P_H A_H),\qquad
 v_H=\|(I-P_H)A_H P_H\|_F^2,
\]

with the independent full Slater second moment as an additional reported value.
On the exact Slater factor, the common LEFT and RIGHT heavy squared norm is
`||B Psi||² (|m_H|² + v_H)`. Heavy even bilinears commute with the literal
light/Ref SOURCE, so these contributions cancel algebraically in the relative
commutator. They are not discarded in the absolute norms and are **not a SOURCE
defect cause**. The stored transported heavy projector has numerical trace
`54.000000000324846`; no projection purification or clipping is used. The reported
second moments and positive-form variance therefore have ordinary floating-point
projector/unitary roundoff, especially at the large heavy A scale.

The executed mean Noether heavy variance was `0.09674321207504843`; the heavy
A mean was approximately `-338220758.3730366` and its reported positive-form
variance `3.355744895851178e-05`. An enormous common heavy mean does not imply an
enormous SOURCE-relative defect.

## Actual execution

Runtime: `/home/administrator/substrate-framework/.venv/bin/python`, cwd
`/home/administrator/substrate-222/proposals/P256-qed-backward`. Command:

```sh
/home/administrator/substrate-framework/.venv/bin/python prepared_neutral_moving_frame_car.py \
  --cache /home/administrator/.omp/agent/sessions/-/2026-09-30T11-33-07-678Z_01a0f216-fbde-7735-a53d-dac838dc2a9a/local/issue222-shared-photon-state.npz \
  --manifest /home/administrator/.omp/agent/sessions/-/2026-09-30T11-33-07-678Z_01a0f216-fbde-7735-a53d-dac838dc2a9a/local/issue222-shared-photon-state-manifest.json
```

Natural exit0; external wall `48.27884699194692` seconds; empty stderr; stdout
187280 characters of JSON. Full stdout/argv/cwd/exit/wall/executed-source binding:
`local://issue222-moving-frame-car-science-receipt.json`. The stdout binds all100
loaded project Python source hashes and both accepted input hashes:

- NPZ: `10d61adb4a374a05e45338522abefcda260ed25036396b1bea26c48c57e4bee3`.
- Manifest: `90a6d7f34617540042406a07c5d8690cecdc456b2d585ee150c152c7d2e07f9a`.
- This executed Python source: `a25eb196a65bc6e3d729feec0e5a332b12159cf1c4787cd86b4898b368d340c2`.

One primary local point construction supplies mean physical Noether M and
instantaneous mean A (`h_L - Omega M_L`, `-Lambda h_bare - Omega M_H`,
`-Omega M_ref`). The actual light coefficient is **1**, not Lambda. Both sources
use the unchanged raw combined quartet and all4Ref.

| Source/body | Full LEFT norm | Full RIGHT norm | Absolute CAR residual | Operation-scaled CAR residual |
|---|---:|---:|---:|---:|
| source19/M | 0.5340786257822573 | 0.4873987384718341 | 1.3317016794450465e-16 | 1.0666134191058831e-16 |
| source19/A | 25.855496741529173 | 26.03604393214677 | 8.161045154151074e-15 | 1.563245213463524e-16 |
| SOFT/M | 6.6965901566810275e-12 | 6.486889777176605e-12 | 1.306545517293473e-27 | 8.800181187392565e-17 |
| SOFT/A | 3.5135004790108053e-10 | 3.530885693430906e-10 | 7.807682871153583e-26 | 1.1055814706458031e-16 |

source19 point SOURCE norm² is `0.9773600786285985`; SOFT point SOURCE norm² is
`1.7824212072132008e-22`, retained literally. This small native periodic SOFT
point response is not corrected by retuning or renormalization and is not a
wholefield norm or legal-N creator-normalization claim.

For source19/M, absolute LEFT and RIGHT PH norms both equal
`0.48543003382126615`, but the relative PH norm is `6.979596365858807e-17`.
For source19/A the corresponding norms are `0.00878412216323362` and
`1.2394866240658585e-18`. SOFT likewise cancels its PH components to
`8.74465450767244e-28` (M) and `1.576696424035839e-29` (A).
Thus the large absolute neutral/charged PH graph is not, by itself, an explanation
of the historical Bnative1.83 SOURCE defect.

The represented charged RIGHT contribution missed by neutral-only compression
has norm `0.0437624559176636` for source19/M and `0.0007915355721060223` for
source19/A. Full Wick second-moment relative errors are at most approximately
`1.95e-16` in these four LEFT cases.

## Tool evidence and limits

Actual Pyright1.1.414 stdio was exercised before editing on
`FullSlaterFibre.source_vectors_from_marks`: symbols332, definition/references,
received diagnostics0. Final source-bound stdio receipt
`local://issue222-moving-car-final-lsp.json`: symbols109, definition/references
including the primary caller, explicitly received `publishDiagnostics` with0
entries, natural server exit0. There is no claim of an xd-LSP integration or a
Markdown language server. Positive structural queries found `FullSlaterFibre`
and the new typed `car_source_control`; final Python AST has10 declarations.
A supplementary throwaway complex6-mode CAR check gave commutator error
`2.9318694848232452e-15` and an exact number-operator check. No permanent test,
project validation, old scientific CLI, preparation or H/G compilation was run.

This is an actual full112 CAR calculation **on a point and locally frozen mean
unitary frame**, not a full changing-pulse trajectory, physical closed2pi
execution, wholefield estimate, or full native quantum-field/Fock evolution.
Root, Q, all512 histories, full Gauss, infinite photon/matter Fock, global
state/field bounds, law/parameter closure and the campaign goal remain open.

## Change record

2026-10-07: added the disjoint full112 determinant SOURCE-relative consumer,
complex-current ABI, exact weak/full PH comparison, and separate heavy
mean-plus-variance common-cancellation report. Frozen supplier files unchanged.
