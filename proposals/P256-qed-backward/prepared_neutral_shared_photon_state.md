# Shared native photon/function Hamiltonian

`prepared_neutral_shared_photon_state.py` supplies A's common Hamiltonian, Gram,
state, SOURCE coefficient maps and photon operators to the physical rotation and
transport consumers. B owns the physical Noether generators and actual once-loop;
T owns the scattering computation. This file does not manufacture a spin-only
rotation or certify the original joined physical goal.

## Frozen preparation and original law

The input is the immutable A17 archive
`issue222-ambient-weak-state.npz`, SHA256
`eb283954e7ebb13e1b5368deb058214443d9a839b8319161c0306afb2c8b53b6`.
Its fourteen prepared neutral coefficients are embedded into the coexact vacuum,
unchanged. They remain the original `f(H14_MC) tau`, not a new preparation using
an exact-EM Hamiltonian. There is no refilter, input whitening, calibration,
normalization reset or new occupied-state diagonalization in the cache loader.

The native parameters remain N=3, FAR=(0,1,1), lambda=2^24, UE=2^-20,
c=0.125, chi=lambda^-1/2, sigma=2^-13 and tube radius 0.012. The full field chart
has 2430 normal and 405 gauge directions, with its full Jacobian, Haar density,
inverse cometric, cross terms and vertical action. All 52 real coexact modes and
three harmonic coordinates are retained. Harmonic coordinates are `theta=N mean A`.
The fourteen original antithetic Phi/theta nodes and four full-raw-direction
kinetic samples per node consume the original seed-222 random sequence.

The Hamiltonian is the original quartic plus lambda*1.25 potential, both alignment
terms and 0.7 colour-magnetic term, bare EM-neutral massive/heavy sector with 54
occupied states, improved full-108 light sector with all 54 negative states, and
R4 once. Ref energy is zero. EM is `UE Pi^2 + K curl(A)^2/2`, with
`c^2=2 UE K`; the Coulomb form is `UE rho Green rho` before PH projection.
The harmonic potential is `UE theta^2/Np`. Ordinary heavy/light/Ref SU(4) onsite
Gauss products are evaluated before projection, including their outside terms;
Root is not substituted by a scalar identity. Total nality remains 112 mod 4=0.
Full auxiliary Q, all-eight Root/Gauss-kernel and multipair evolution remain debts.

Norms must not be conflated:

- Original raw filtered neutral norm squared, **before** its one normalization:
  `0.8920522472031723`.
- Original A17 Gaussian SOURCE norm squared, **after neutral normalization but
  before SOURCE normalization**: `0.8718600287632787`.
- That Gaussian control is neither SOURCE19 nor SOFT.
- The extended prepared neutral metric norm is exactly `1.0`; the retained
  vacuum Gram block agrees with the original Gram with measured error `0.0`.

Exact all-52 Gaussian/Weyl integration changes the old MC vacuum H block:
Frobenius difference `2.3509402583912746`, maximum entry difference
`0.7503215670585632`. This is reported, not hidden by retagging or refiltering.

## Real function space, not an artificial closed Fock sector

The four complex hard packet directions are native +/-qx and +/-qy with ez
polarization. Total photon number <=2 supplies fifteen hard functions.
`prepared_neutral_weyl_hard_fock.py` evaluates true projected Weyl operators with
all-52 Debye factors before projection and generic Gaussian-function creators.

`prepared_neutral_quartet_functions.py` supplies the fixed INITIAL function bank
using the actual four original words I/Gamma23/Gamma31/Gamma12, global beta factor
2 once, and all Ref components. Moving physical SOURCE is a separate object.
There are exactly two spatial profiles: one SOURCE19 bank and distinct SOFT.
The two SOURCE19 receiver geometries share the same bank; they are not duplicate
columns producing an exact Gram null.

`prepared_neutral_noether_gaussian_partners.py` supplies all oriented primitive
four-sine clovers, edge-averaged velocity and anchored Coulomb compensation
`lambda_C=-L^+ D^dagger raw_EM`, Ref=0. It retains harmonic flow as well as
coexact flow. The scalar kernel bank has 97 functions: hard15, gamma, sin81.
No 52^r tensor allocation or finite-polynomial replacement is used.

The selected true EM partners on the actual prepared state are

- common EVEN `gamma S Psi`, where
  `gamma=i Y_coexact(y).y - (i/2) div_y Y_coexact(y)`;
- common ODD neutral flow times `S Psi`, including neutral Coulomb action and
  the full harmonic derivative of the neutral Gaussian;
- charged RELATIVE ODD `[rho lambda_C,S] Psi - i Y_harmonic d_theta S Psi`.

The last line is the literal Born-minus-Ref rule with the negative-bank sign,
not `Qc_projected S-S Qn_projected`. Its neutral-RIGHT outside correction is
retained. The bank appends neutral even2/odd2 and charged
even16/commonodd16/relativeodd16 to the base210/base1680 functions. Dimensions
are consequently **214 neutral / 1728 charged**, without null suppression,
jitter or a quotient pretending that unsupported functions vanished.

The exact conditional Gaussian norm of gamma is `0.1067530955172436`; its hard15
captured norm squared is `7.919737019098745e-33`, and its outside hard15 norm is
`0.1067530955172436`. The printed outside fraction rounds to 1; the measured
norms, not a false finite-Fock closure claim, are authoritative. Appending the
actual gamma function recovers this selected component with error 0. This does
not make iterated Noether/H evolution or all creator images finite-dimensional.

Gram kernels and base-base H use exact all-52 Gaussian/Weyl integrals.
Generated-partner H uses explicitly **128 antithetic all-52 nodes, seed 222710**,
with the native Phi/theta quadrature and finite-difference function jets.
These are finite-computation debts, not convergence certificates.

## Coordinates and coefficient-map ABI

`load_shared_native_cache(npz,manifest)` verifies acceptance and the NPZ digest,
restores the full stored seed occupations, masks, orbitals and projectors from
ordinary JSON plus NPZ, and returns `(model,data)`. It does not diagonalize a
new occupied state, prepare Psi again, or compile H. `model.compilecache()`
returns that same cache.

The generic model implements `basis_at`, `directional_basis`,
`first_order_weak` and `gauge_operators`. Point values omit only the shared
Gaussian/Haar-half factor; directional values include its full derivative and
the cutoff derivative exactly once. `conditional_values`,
`conditional_derivative`, `coupled_form` and `frozen_nodes` are available for B's
same-basis physical generator compilation. B supplies the actual dq/dem,
body/heavy/Ref and compensating gauge flow; no mean-matrix or intrinsic-only G
is inserted by this supplier.

Public `data` keys:

| Key | Meaning |
| --- | --- |
| `neutral_H`, `charged_H` | Weak Hamiltonian forms, not coordinate generators |
| `neutral_Gram`, `charged_Gram` | Same-function metric forms |
| `neutral_state` | Prepared `c_exact` coordinate vector, frozen vacuum embedding |
| `neutral_embedding`, `preparation_state14` | Exact original-state embedding and coefficients |
| `source19_map`, `SOFT_map` | Charged-by-neutral **coefficient maps**, never weak forms |
| `source19_word_maps`, `SOFT_word_maps` | Four word coefficient maps |
| `neutral_creators`, `charged_creators` | Four coefficient creation operators |
| `*_creator_full_norm_Grams` | Before-projection full creator norm forms |
| `*_creator_outside_Grams` | Full minus represented creator norm forms |
| `neutral_coordinates`, `charged_coordinates` | RAW-to-conditioned function-coordinate congruences |
| `charge_labels`, `charge_projectors` | Two charged bank labels and coefficient projectors |
| `photon_basis_labels`, `charged_photon_labels` | Hard occupation labels; generated columns use -1, not false vacuum labels |
| `photon_projector_coordinates` | Hard/source sector coefficient embeddings, **not** complete physical number projectors on infinite Noether tails |
| `packet_templates` | Incoming and two outgoing geometries; no third receiver outcome |

If functions are `F'=F D`, weak forms are `D^dagger H D` and coefficients
transform by `D^-1`. SOURCE maps transform as `D_c^-1 B D_n`, with PLUS beta and
MINUS beta-conjugate, including the new even/common-odd neutral columns.
Relative-odd charged columns are not mislabeled as additional SOURCE images.
The diagonal congruence conditions coordinates only; RAW profiles, physical
input amplitudes, raw forms and state remain available unchanged.

Minimum conditioned Gram eigenvalues are `0.03694706738660121` neutral and
`0.016047452717555154` charged. There is no jitter or clipping. H is unshifted;
its negative sea energies are not an error requiring a positivity fix. The
measured absolute Hermiticity residuals are `2.7560e-7` / `1.0486e-6` at a large
energy scale; relative residuals in the actual cache probe are
`5.8534e-17` / `6.9506e-17`, respectively.

## SOURCE19, SOFT and physical normalization

SOURCE19 uses descriptor SHA256
`020fb175c9231b607fb0e53b016cf97528af9065a634dedc7ddea82bdac3556c`,
folding `(d+X)%3` at X=(1,1,1), without shifting or renormalizing eta.
Its raw function norm squared after neutral normalization is
`1.009147553233165`; the one global SOURCE divisor is `1.0045633644689442`.
Raw plus/minus contributions are `0.4125006242131229` /
`0.5966469290200422`. There is no separate bank reset.

SOFT binds B's exact original compact C-infinity momentum bump: p0=(2e-7,0,0),
width=1e-7, scalar L2(dp)=1, inverse Fourier convention `(2pi)^-3/2` and sites
minus (1.5,1.5,1.5). Pure-profile receipt SHA256
`f79e7de3267d1599bff426d238a2a470bf10dedae5703427252b2def5419d654`
contains the unmodified real-space restriction. Its native raw profile norm is
`1.4844654914318392e-11`; raw SOURCE function norm squared is
`1.8490300325794218e-22`, with one global divisor `1.359790437008373e-11`.
The periodic N3 Fourier projection is **exactly zero**. Conditioning these tiny
restricted functions or normalizing the conditional native SOURCE state does
not restore the missing physical spatial mass. No A17 Gaussian replaces SOFT.

Native hard k is +/-2pi/3; the legal q0 is approximately 2pi/5 with distinct
positive mode IDs and split about `1.5860989597e-29`. The principal magnitude
gap is `0.837758040957278`. Native c=0.125 also differs from the frozen legal
rational `1130202988249777/9007199254740992` by
`0.00047773800550576784`. No retuning, legal-mode identification or physical
SOFT/Born acceptance follows from this native cache.

## Real consumer paths and source epochs

`with_common_boost` applies one phase to every function and adds the documented
common-frame energy `Omega_dot Y0.q/chi - Omega^2 ||Y0||^2/(2 chi)` without a C0
retag. `in_mean_interaction_frame` transports all light108, Ref4 and optional
heavy108 mean spinors once. The held INITIAL SOURCE bank is not recaptured;
`source_at` / `current_source_function` retain the separate moving physical B.

The actual cache probe on a frozen native node measured SOURCE map error
`4.8213e-16`, all new neutral SOURCE columns `6.2433e-16`, common-boost phase
errors <=`5.8361e-15`, held INITIAL mean-frame error `0`, and moving physical
SOURCE difference `0.0038331342`. Full-Gaussian/Haar/source directional finite
checks give absolute residuals `9.3904e-8` / `2.7383e-7`; this is native finite
precision/FD evidence, not a continuum theorem. Coulomb CAR recovery is
`4.0422e-19`, with neutral-RIGHT outside norm `1.8929e-5`. The actual SOURCE
Coulomb full/outside-PH graph norms squared are `2.4838e-6` / `1.5898e-8`.
The full-108 current full/outside-PH graph norms squared are `2.2614e-7` /
`2.2170e-7`: the large outside contribution is reported, not discarded.
The metric H-Xi graph norm squared is `5.8939859611385736e16` and unshifted Xi
energy is `-242714945.40244955`.

A separate owner off-mean/no-copy consumer run exited 0 in 17.914484 seconds.
Its actual full448 SOURCE relative errors were `4.8631e-16` / `6.3757e-16`, and
all four literal creator wavefunctions agreed within `1.230e-15`. That evidence
alone does not prove boosts, time evolution or a physical closed rotation.

The H CLI naturally exited 0 in `3935.4426019270904` seconds. Accepted NPZ:
`issue222-shared-photon-state.npz`, SHA256
`10d61adb4a374a05e45338522abefcda260ed25036396b1bea26c48c57e4bee3`.
Manifest SHA256:
`90a6d7f34617540042406a07c5d8690cecdc456b2d585ee150c152c7d2e07f9a`.
Its executed source is `94dd798b541614254894be8841c1cfcbb19234f55a59cc6e52f3bc207b7f8bc1`.
The manifest preserves argv/cwd, all six input hashes, and 96 loaded source
endpoint hashes plus exact source archives. No uninterrupted disk identity is
claimed merely because endpoints agree.

The first new H attempt naturally exited 1 after 3957.554766 seconds at its
source-epoch guard. No successful physics cache was published; the precise
changed source/cause was not preserved and is not attributed retrospectively.
The repaired publication boundary stores a private UNACCEPTED checkpoint and
source epochs first, rejects changed/unknown late source epochs, and publishes
an accepted manifest only after sealing. The public loader refuses unaccepted
checkpoints. A real late-import/change/reject/load smoke exercises that boundary.

A subsequent real public-copy probe exposed recursion when `copy.copy` probes
an uninitialized instance before `original` exists. The permanent correction
uses `object.__getattribute__(self,'original')` before ordinary delegation.
Current corrected source SHA256 is
`32ca37d7b8da5adccd298269e917f01c27b7a44346fdc30f8b0a2614b074bb98`.
All module AST except this bootstrap getter is identical to executed 94dd;
initialized-owner delegation is unchanged. This is an API repair, **not** a
new scientific H/G/preparation epoch. During B's run the on-disk intervention
94dd->32ca->94dd is disclosed; which getter B imported is unobserved. B did not
call copy/boost/mean-frame in that scientific path. After T's sealed run released
the source hold, the correction was reapplied for the canonical API probe.
Do not retag frozen H94dd, B or T execution receipts as current-source runs.

The canonical corrected-source API probe naturally exited 0 in
`129.41066513909027` seconds, with empty stderr and all 96 loaded source endpoint
hashes unchanged. Current-source real pyright RPC exited 0, with 430 symbols,
38 positive AST declarations, real definition/reference responses and zero
diagnostics. These are current 32ca API proofs, not reruns of frozen H94dd.

Current-source real pyright RPC, positive AST, canonical cache probe, sealed CLI,
source-seal smoke, copy-before failure, specific AST bridge and owner semantic
probe are separately bound in `issue222-shared-photon-result-capsule.json`.
There are no new permanent tests, and no old preparation/scientific reruns.

## Acceptance boundary

This supplies shared native weak forms, exact selected function/creator kernels,
and real SOURCE coefficient maps. It does **not** close legal-volume control,
physical once-2pi rotation, Born comparison, infinite Weyl/Fock propagation,
full PH/multipair or all-eight auxiliary-Q/Gauss-kernel debts. B's actual rotation
and T's actual scattering results remain independent decision-bearing evidence;
neither compile success nor a counterfactual spin sign is substituted for them.
