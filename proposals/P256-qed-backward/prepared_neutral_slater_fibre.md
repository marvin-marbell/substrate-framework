# Fixed full108 Slater/reference variational fibre

`prepared_neutral_slater_fibre.py` supplies the light/reference weak-matrix component for A's **same original ambient Hamiltonian**, not a curve Hamiltonian or an active12 replacement. The governing plan is `issue-222-fullfield-closedrotation-continuation-20261006` at `7954ba4c18ed7d90fdc3e6cbd3bf75c3dc80d568cae4781d73dadd61e041a718`, template `general` at `89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`; `prepared-neutral-full-field` remains in progress. This pair does not change that plan or publish a campaign conclusion.

## Physical preparation and basis

The constructor receives the actual law, mean Cartesian Phi fields, mean SU4 links, reference site, original spatial profile and raw quartet. It constructs the **entire** negative cluster of `h0=law.onebody(mean_fields,mean_links,A=0)`, then consumes `positive_spectator_modes` for the two complete positive spin pairs at native `+/-qx`. Its default integer momenta are `+/-max(1,N//4)`; the legal divisible-by-four family uses exactly `+/-N/4`. The remaining positive complement completes a full orthonormal `4V` orbital frame. No source coefficient chooses or whitens this frame.

At N=3, the two orthonormal neutral determinants are

\[
 |n_+\rangle=|\mathrm{negative}_{54},\mathrm{Ref4}\rangle,
 \qquad |n_-\rangle=|\mathrm{negative}_{54}+R4,\mathrm{Ref0}\rangle,
 \qquad |\mathrm{RAW}\rangle=(|n_+\rangle+|n_-\rangle)/\sqrt2.
\]

The second bank has 58 actual occupied light orbitals, not a bare sea54/Ref0 state. Both banks have total number 58. The charged basis consists of the actual even-CAR products

\[
 |+,r,e\rangle=a_e^\dagger r_r|n_+\rangle
 \quad(e\in\mathrm{all54\ empty},\ r=0,1,2,3),
\]
\[
 |-,r,i\rangle=r_r^\dagger a_i|n_-\rangle
 \quad(i\in\mathrm{all58\ occupied},\ r=0,1,2,3).
\]

Thus there are 216 plus and 232 minus determinants, **448 charged states**, not cropped active12 source responses. Reference index is the outer tensor index. `descriptor()` exposes the actual occupation masks, CAR phases and labels; no hole sign is guessed or absorbed into a source coefficient. The plus bank is light55/Ref3 and the minus bank light57/Ref1, so the same total58 survives both SOURCE maps.

The orbitals and raw bank coefficients are fixed and read-only. Changing fields, links, A, theta or time does not recapture a new sea or new spectators. An actual coordinate Hamiltonian is `h(fields,links,A)`, not `h0` substituted as a stationarity shortcut.

## Actual SOURCE as a field function

`source_vectors_at(fields,links,theta,a,profile=None,quartet=None)` calls `canonical_bank_coordinate_marks` for **both original signed maps on the same law**. This retains `literal_coordinate_mark`, all108 primitive coefficients, the original signed per-edge route, both endpoint frames, all four Ref modes, the raw quartet `[1,.3i,-.2+.4i,.6]` and its original global2. It never constructs a charge-minus Hamiltonian.

Write the plus full108-by4 coefficient matrix as C and the minus full4-by108 coefficient matrix as D. With fixed empty frame E and occupied58 frame H, the exact charged coefficients are

\[
 c^+_{r,e}=(E^\dagger C)_{e,r},\qquad
 c^-_{r,i}=(DH)_{r,i}.
\]

`source_matrix` has shape448-by2 and acts on the neutral bank coefficients. `source_vectors[+1]` and `source_vectors[-1]` already contain the RAW factor `1/sqrt2`; their disjoint sum is `total_source`. No positive source norm is discarded:

\[
 \|B_+\mathrm{RAW}\|^2=\tfrac12\operatorname{Tr}[C^\dagger(I-P_{54})C],
 \quad
 \|B_-\mathrm{RAW}\|^2=\tfrac12\operatorname{Tr}[DP_{58}D^\dagger].
\]

The returned `bank_source_Gram` is the unweighted two-bank matrix `source_matrix^dagger source_matrix`. The returned `SOURCE_normalizer` is the **pointwise diagnostic** square root of the RAW total norm; it is not permission for pointwise normalization, input whitening or per-fibre receiver resetting. A field-state normalization must integrate the actual `B(q) Psi(q)` in the full ambient measure. The integration owner may use these exact coordinate-dependent vectors as SOURCE-created field functions; this API does not project B onto finite field polynomials.

An important producer boundary is explicit: the original plus map is quartet-linear, but the original minus map conjugates the quartet. For four real basis columns, physical reconstruction is `Cplus @ beta + Cminus @ beta.conj()`. Its Hermitian RAW norm matrix is `Gplus + Gminus.T`. Treating both maps as one shared complex-linear map is incorrect, even though a real basis probe alone cannot detect the error.

## Exact weak onebody and bilinear products

`onebody(A,sector)` returns `P_fibre dGamma(A) P_fibre` for any actual full108 complex operator. It retains the filled-sea trace. For the single-particle and single-hole blocks, respectively,

\[
 M_+(A)=\operatorname{Tr}(P_{54}A)I+A_{EE},
 \qquad M_-(A)=\operatorname{Tr}(P_{58}A)I-A_{HH}^{T}.
\]

`reference(R,sector)` includes the derived **full four-mode** reference action: Ref3 holes use `Tr(R)I-R.T`; Ref1 particles use R. These transposes follow the stated even-CAR basis, not a charge-conjugated H. Neutral entries are the whole-sea traces and Ref4/Ref0 traces. Cross-bank entries vanish for these light-number/reference-number-preserving operators.

`bilinear_product(A,B,sector,reference_A=None,reference_B=None)` returns the exact weak matrix of

\[
 (d\Gamma(A)+d\Gamma(R_A))(d\Gamma(B)+d\Gamma(R_B))
\]

**before projection**. For a Slater projector P,

\[
 \langle d\Gamma(A)d\Gamma(B)\rangle
 =\operatorname{Tr}(PA)\operatorname{Tr}(PB)
   +\operatorname{Tr}[PA(I-P)B].
\]

In orbital coordinates the extra term beyond the product of compressed single-particle matrices is

\[
 \operatorname{Tr}(A_{HE}B_{EH})I-B_{EH}A_{HE};
\]

for a single-hole sector it is

\[
 \operatorname{Tr}(A_{HE}B_{EH})I-(A_{HE}B_{EH})^{T}.
\]

For the minus bank H has58 modes and its empty complement has50. These are the actual intermediate particle-hole/multipair contributions, not a squared compressed onebody. The reference sector is already exhausted by its four holes/particles; the API includes both light/reference cross products. The formulas remain valid for complex, non-Hermitian A and B, as the independent exact-CAR control checks.

`slater_expectation(P,A,B=None)` exposes the same full Slater trace/Wick expression for a heavy occupied projector when the integration owner needs it. It does not identify the trial with an actual ground state.

## Original Coulomb and absolute Gauss

`coulomb(Green,UE,sector)` diagonalizes the positive original lattice Green `L+` and evaluates each full density mode with the exact bilinear product:

\[
 H_C=U_E\sum_{x,y}\rho_x\mathrm{Green}_{xy}\rho_y,
 \qquad \rho_x=n_x-2+(N_{\mathrm{Ref}}-4)\delta_{x,R}.
\]

It does **not** replace this by `sum(projected_rho @ projected_rho)`. No normal-order subtraction or extra electric1/2 is introduced. `gauss(sector)` evaluates the absolute global constraint `Nlight-2V+Nref-4`, not merely the SOURCE commutator. The original bare sea54/Ref0 branch has Gauss=-4 and is explicitly rejected. The new58/Ref0 bank and both charged outputs have Gauss0.

This is the light/reference ingredient only. Ambient Phi kinetic/Haar measure, colour .6chi cometric, original EM `UE Pi^2+(K/2)curl^2`, free harmonic `UE/N ptheta^2`, scalar/heavy law, coupled Root beta penalties and all512 histories remain integration-owner ingredients; this pair neither removes them nor replaces them with one-dimensional dynamics. The small ordinary-CAR control does not establish a full encoded-Root odd-CAR upgrade or an unrestricted auxiliary SOURCE bound.

## Consumer API

```python
from prepared_neutral_slater_fibre import FullSlaterFibre, lattice_green

fibre = FullSlaterFibre(law, mean_fields, mean_links, reference_site,
                       profile=physical_profile, quartet=raw_beta)
source_q = fibre.source_vectors_at(fields_q, links_q, theta_q, A_q)
H108_q = law.onebody(fields_q, links_q, A_q)
Hn_q = fibre.onebody(H108_q, 'neutral')
Hc_q = fibre.onebody(H108_q, 'charged')
Gc_q = fibre.onebody(full108_gauge_q) + fibre.reference(full4_Ref_gauge_q)
Gc2_q = fibre.bilinear_product(full108_gauge_q, full108_gauge_q,
                              reference_A=full4_Ref_gauge_q,
                              reference_B=full4_Ref_gauge_q)
HCc = fibre.coulomb(lattice_green(law), UE, 'charged')
```

`coordinate(...)` combines the actual coordinate H and SOURCE result, without modifying preparation. `current_rows(A,source_q)` supplies the full108 current, charged weak rows, SOURCE response rows, and the **full** response norm from `P dGamma(A)^dagger dGamma(A) P`; it separately reports the unrepresented response norm. `onebody_graph_diagnostic(A)` makes the static-mean basis's outside particle-hole/multipair complement visible. It is not an error-suppression or dynamics-completeness certificate. `overlap('neutral')` is I2 and `overlap('charged')` is I448. These are fibre overlaps, not the eventual ambient weak Gram containing q-dependent field functions and full chart measure.

## Executed scientific proof

The natural final CLI was

```
/home/administrator/substrate-framework/.venv/bin/python prepared_neutral_slater_fibre.py
```

Cwd: `/home/administrator/substrate-222/proposals/P256-qed-backward`. Exit0; empty stderr; walltime4.131389752s, scientific section2.944439946s. It creates no project receipts or state caches. `local://issue222-slater-fibre-final-wrapper.json` binds argv/cwd/walltime/source/stdout/stderr hashes; full output is `local://issue222-slater-fibre-final-stdout.json`. It froze all83 actually imported scientific Python sources and observed them unchanged. Inputs include N3/full108, the original .65 Gaussian x/y spatial profiles and their complex coherent combination, the original raw quartet/global2, Ref=(0,1,1), theta=(.21,-.13,.09), actual harmonic plus nonzero coexact A, and native +/-qx positive spinpairs. These are native illustrative UE=.0025 data, not execution at the astronomical legal UE.

Observed results:

| Quantity | Executed value |
|---|---:|
| Original x-profile full plus RAW norm squared | .4125000000000001 |
| Original x-profile full minus RAW norm squared | .6312503758774033 |
| Original y-profile full minus RAW norm squared | .41260224134820234 |
| Complex x+iy profile full minus RAW norm squared | .5219263086128025 |
| x-profile total RAW SOURCE norm squared | 1.0437503758774034 |
| Neutral HC minimum eigenvalue | .000982026168142467 |
| Charged HC minimum eigenvalue | .0009642665983516234 |
| Project-before-HC-square wrong-branch defect | .04018039834913483 |
| Outside-pair HC correction minimum eigenvalue | .0009603478846777874 |
| Charged absolute global Gauss error | 1.81e-14 |
| Neutral absolute global Gauss error | 2.07e-17 |
| Actual h(A,theta) neutral particle-hole norms squared | .00724316261 / .00661229105 |
| h0 stationary control norms squared | about4.3e-29 / 4.1e-29 |
| Full SOURCE current response norm squared | .3412089983453639 |
| Unrepresented SOURCE current response norm squared | .3335390761656437 |
| Actual nonuniform Phi/link SOURCE-function change | .0014355332136811096 |
| Wrong shared complex-linear SOURCE-map defect | .618527271331058 |

All three physical profile coefficient norms equal their full Slater norm formulas within floating roundoff: there is no historical .1866/.0108 source crop. The nonuniform Phi/link probe changes B(q) while the fixed full108 orbital hash stays unchanged. Nonstationary actual H/current outside response is positive and deliberately remains visible.

A new independent10-mode/1024-state exact CAR calculation checks random complex non-Hermitian light/reference onebodies and their products. Maximum weak-product error was1.86e-14, SOURCE even-CAR coefficient error0, and full Slater expectation error below9e-16. It rejected wrong hole transpose with defect20.90563 and wrong normal-order subtraction with defects7.63214/5.48557. These are substantive uncertain-sign/Wick controls, not source-text or wiring tests.

The first CLI failed because its **control** incorrectly demanded shared quartet-complex-linearity. The unchanged original minus producer conjugates beta. The failure is retained in `local://issue222-slater-fibre-science-wrapper.json` and its stderr; the repair changed the control to honor the original plus/minus maps, not the producer or law. An intermediate passing receipt `local://issue222-slater-fibre-current-wrapper.json` remains historical. A first post-LSP receipt recorded metadata-path Optional-type diagnostics; the final explicit filename-narrowing loop resolves them.

## Source freeze, tooling and scientific boundary

Final Python SHA256: `0005e551e8270c5634a19d658cb3c504dbb1f7deee799cb9d4f3471552612634`. Final scientific stdout SHA256: `8c6ee5074b488aa6621117f0c9c390dd2b76245d70815a437ccc78659d2ea17b`.

Real Pyright1.1.414 stdio LSP was exercised before implementation on the original `canonical_bank_coordinate_marks` symbols/definition/references with received diagnostics, and after source freeze on `FullSlaterFibre`. The final scoped receipt `local://issue222-slater-fibre-final-lsp.json` records332 symbols, definition/references, **received empty diagnostics**, natural server exit0 and27 positive Python AST declarations. Structural ast-grep positively matched the new class and the exercised `slater_expectation` calls. The mounted xdLSP device is unavailable; Markdown LSP is unavailable. Those are retained tooling gaps, not fabricated passes.

This deliverable is a **native SAME-law variational projected H/state fibre**, with exact finite source representation and exact preprojection bilinear weak products. It is not full Fock dynamics, full Gauss-kernel dynamics, a full512 runtime, an actual light/heavy ground state, a global filtered creator normalization, a Born/scattering result, or a physical closed-loop construction. The substantial outside current/multipair norm proves why those distinctions matter. No frozen SC19/q128/native supplier, sibling source, service or git state was modified or rerun.
