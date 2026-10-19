# Original quartet SOURCE function bank

This pair owns only the INITIAL SOURCE-function extension, not Hamiltonian/state preparation or the physical Noether generator. Governing plan: `issue-222-fullfield-closedrotation-continuation-20261006`, revision `57838ca23dbf1424677c1c2f4e183784463faa3f759a85ec2a2b9722ff1fd9ff`, general template `89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`, scoped `prepared-neutral-full-field` remains IN_PROGRESS. WorldModelSync04 alone publishes canonical status.

## Exact interface and order

`QuartetSourceFunctionBank(seed_fibre, profiles)` accepts an already prepared `FullSlaterFibre` and an insertion-ordered mapping of named, fixed complex profiles of shape `[V]`. It snapshots its existing orbitals/occupied projectors and profiles as read-only arrays, without constructing another fibre, diagonalizing, refiltering, preparing or resetting its sea. All four Ref columns and both charged sectors are retained. Native N3 has full108 light space, fixed54 negative sea, four fixed positive spectators for the Ref0 bank, and full charged448 = plus216 + minus232.

* `word_sources(fields, links, theta, a)` returns `[templates,4,448,2]` actual source coefficient functions. Fields `[V,4,4,4]`, links `[V,3,4,4]`, theta `[3]`, A `[V,3]`; `theta = N mean(A)` is the existing producer contract.
* `source_values(fields, links, theta, a, neutral_values)` accepts `[2,K]` neutral bank values and returns `[448, templates*4*K]`. Columns are **template outer, word middle, neutral column inner**, words `(I, Gamma23, Gamma31, Gamma12)`. With two templates and A17's K14 the shape is `[448,112]`. Charged rows retain the fibre's reference-index-outer, active-light-index-inner order and plus-then-minus slices.
* `raw_input_map(beta,K,selected_profile)` returns `[2,templates*4*K,K]`. Slice0 uses literal beta; slice1 uses its conjugate. `combine_values(values,beta,K,selected_profile)` applies those two maps to the appropriate charged row slices and returns `[448,K]` physical B_beta values.
* These are constant **input** maps, independent of fields and mean U_L/U_R. A single scalar coefficient matrix is not universally valid for mixed-bank neutral columns. A17's bank-pure columns permit selecting the correct slice columnwise; universal `combine_values` also handles mixed columns correctly.
* `physical_source(moving_fibre,reference_frame,fields,links,theta,a,selected_profile,raw_beta)` separately evaluates the current observable in owner-supplied moving orbitals and Ref4 frame. It multiplies plus marks on the right by U_R and minus marks on the left by U_R dagger, then uses the moving fibre's full coefficients. It does not change INITIAL function columns or reprepare any orbitals.
* `directional_values(path,neutral_values,step)` accepts a pure coordinate-path callback `path(s)->(fields,links,theta,a)` and returns function values and central-FD derivative at step/2 with native refinement estimate `norm(D_h/2-D_h)/3`. Neutral values are held fixed. The integrator adds `B times derivative(neutral_values)` by the product rule, using `source_values(..., derivative_neutral_values)`. The callback can provide the exact link-group path and harmonic relation. The FD estimate is **not** a rigorous analytic error enclosure.
* `descriptor(K)` returns portable shape/order/orbital/profile-hash metadata. Runtime receipts use the repository's portable complex serialization.

The unit quartet is a coordinate basis of the original raw quaternion coefficients, not whitening or preparation. Every unit producer retains the ORIGINAL global divisor2. Recombination uses RAW beta `(1,.3i,-.2+.4i,.6)` without normalization, fitting a scalar, or replacing a measured Gram by the identity. The minus map is conjugate-linear because the original producer takes the conjugate transpose after charge-CP fields/links. Neutral coefficients remain linear and are never conjugated with beta.

## SOURCE19 and packet geometry

`source19_native_profile(law,descriptor_path)` requires the exact descriptor SHA256 `020fb175c9231b607fb0e53b016cf97528af9065a634dedc7ddea82bdac3556c`. It decodes the nineteen stored complex weights and accumulates each at the native site `(d+1)%3`, preserving their values and any fold multiplicity, with no renormalization. Metadata includes `profileHash`, descriptor hash, offsets, weights and fold rule. This is a diagnostic native fold, not the huge legal SOURCE19 geometry.

The scientific control supplies named `source19` and `forward` profiles containing exactly the same eta. They remain distinct SOURCE/photon templates downstream; their pure SOURCE columns coincide because photon geometry has not yet been tensor-multiplied. Outgoing-forward `exp(-i k_x)` and free `exp(-i omega*6)` belong to the photon packet, **not** a SOURCE eta spatial shift or inferred rho_SOURCE translation. The four hard packet legs and their all52-Gaussian Weyl/Fock integration belong to the sibling photon core and A integrator.

A17's Gaussian Gamma prototype is not SOURCE19 or B's selected SOFT profile. The bank accepts arbitrary explicitly supplied fixed named profiles, including an external SOFT profile, but chooses no retune and invents no legal narrow profile. A17's frozen state has SHA256 `eb283954e7ebb13e1b5368deb058214443d9a839b8319161c0306afb2c8b53b6`; direct loading observes neutral metric norm exactly1. This module does not alter that NPZ, its once-prepared filter, SOURCE norm squared `.8718600287632787`, or RAWfn neutral norm squared `.8920522472031723`. The same coefficient state is to be embedded in EM vacuum by A, not replaced by the CLI's independent function-value probe.

## Quaternion orbit and full Slater RIGHT

Put E1=Gamma23, E2=Gamma31, E3=Gamma12 and L=(I,E1,E2,E3). Clifford anticommutation gives E3 E1=E2, E3 E2=-E1, E3 squared=-I. For physical spatial spin primitive S_z=-i E3/2, left word multiplication is `S_z L_j = sum_i L_i T_ij`, with

```
T = (i/2) * [[ 0, 0, 0, 1],
             [ 0, 0,-1, 0],
             [ 0, 1, 0, 0],
             [-1, 0, 0, 0]]
```

`quaternion_spin_word_action` derives T by matrix inner products and verifies this identity, not by projecting a mixed-beta scalar spin expectation. On the uniform physical source chart the same T maps **both** sets of unit-word functions. In raw input coordinates the PLUS spin action is T beta, whereas the MINUS spin action is -T beta: conjugate-linearity and T*=-T turn the latter into the same T action on unit-word output functions. Each reference column is retained; none is traced to a mixed expectation.

For Q=dGamma(A), exact CAR gives `[Q,Bplus(M)]=Bplus(A M)` and `[Q,Bminus(M)]=Bminus(-M A)`. `full_slater_relative_spin` calculates this on full108 before the charged448 projection. It compares against `Qcharged B - B Qneutral`, retaining the neutral RIGHT term and also `Pcharged B Qneutral_outside`; the latter is essential because the fixed Slater two-bank neutral fibre is not invariant under Q. In detail, the right-outside plus rows are `-empty† A Poccupied M`; minus rows are `M (I-Poccupied) A occupied`. Subtracting this outside RIGHT gives the exact projected CAR commutator. No assertion that Sigma_neutral=0 or total G_SOURCE equals intrinsic spin is used.

Full response norms use the existing exact preprojection Wick `bilinear_product(A†,A)`, not the square of compressed `onebody(A)`. Charged and neutral particle-hole outside debt remains separately reported. Recovering SOURCE-relative word action does not erase physical state/background PH flow.

## Executed scope and adverse evidence

The natural CLI imports the current `ImprovedPhiLaw`, canonical screened per-edge signed-path producer and full108 Slater fibre. Three points have actual nonuniform Phi, SU4 links, theta and A. RAW combined B_beta functions agree with the original pure producer to at most `1.17e-15`; functions change by about `.02326` and `.04409` while initial orbitals remain fixed. This is an exact function API, not a point-normalized source-polynomial fit.

Observed controls before final source-typing verification:

* all four individual word relative errors are below `2.8e-16`; mixed RAW recovery `4.53e-16`;
* full four-word source-relative span outside `O(1e-16)`, versus old combined-beta two-bank static spin outside `.49804` and wrong scalar mixed-spin projection defect `.49933`;
* wrong minus complex-linearity defect at least `1.42319` and omitting three Ref modes defect `1.24271`;
* retained neutral RIGHT outside norm `.126715`; background neutral PH norm squared `.804171` / `.690959`;
* full charged spin Wick norm squared `.969213`, charged PH outside `.727876`;
* actual Peierls current full Wick norm squared `.361022`, current PH outside `.353625`;
* initial functions are unchanged under supplied moving mean frames, while the physical observable differs by `.110196`;
* native derivative refinement estimate `1.19e-10`.

Final exact source/receipt/archive/LSP hashes and wall time live in `local://issue222-quartet-functions-capsule.json`; scientific stdout and natural-exit wrapper are linked there. A positive structural AST query and real Pyright LSP symbols/definition/references/diagnostics target this actual module. No project-wide test/lint/build was run; no sibling/A17/Slater/B source was edited.

This is native N3 function and full-Slater/Wick evidence plus analytic finite Clifford/CAR identities. It is NOT a legal-Xi preparation, selected legal hard-ID run, full512 realization, exact native Fock closure, full Gauss closure, photon/Weyl integration proof or physical U2pi/Noether closure. B alone owns Gphysical; intrinsic relative-word recovery is not an intrinsic-only substitute for orbital/Phi/link/EM/harmonic/Ref flow. Original quantum-law/microscopic-parameter debt remains open. No status or closure promotion follows.
