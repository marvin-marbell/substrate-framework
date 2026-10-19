# Primitive charge carriers in the complete aligned Gauss–photon EFT

`gauss_photon_charge.py` joins the actual primitive boundary charge words to
`CompactGaussPhoton`, rather than adjoining three spectator samples to the old
atomic Hamiltonian. This is a constructive finite-model theorem and a runnable
numerical consumer. The parent owns scientific executions and their receipts;
this note contains derivations and describes checks, not invented run results.

## Domain and common microscopic law

Each Peter–Weyl block has the five independent axes

\[
(m,a,b,z,\ell),\qquad \ell=-R,\ldots,R.
\]

Here `m` is the complete centre-allowed matter fibre, `a,b` are the two colour
link indices, `z` is the charged boundary index, and `ell` is an independent
compact electromagnetic loop momentum. The neutral boundary has dimension one;
the hole and particle boundaries have dimension four. The marker remains the
unchanged neutral Pauli-code parity marker. There is no boundary/loop index
identification and no fixed-image Hilbert-space restriction.

The shared suppliers own the following facts:

| Object | Actual supplier and retained meaning |
| --- | --- |
| Orbital frame | `uniform_data` and `RotationBand`, the same selected fixed simultaneous adjoint orbit and bare coefficients |
| Statistics and ladders | `PauliCode` and `BoundaryChargeIntertwiner.components[name]`, actual rectangular primitive Pauli-word transitions |
| Colour connection | One common `HaarLink`, continuous Haar/Peter–Weyl compression, not a group sample |
| Physical matter/colour fibre | `PhysicalHamiltonian`, `NH=4`, `NL=4` neutral and `NL=3,5` charged, every native centre-allowed state |
| Photon and hopping | `CompactGaussPhoton`, every independent `ell`, actual two-path light hopping and source-independent dynamical magnetic potential |
| Alignment | One common `ExactHaarAlignment` provider, exact continuous projected potential with coefficient `Lambda*Jpref/8` |
| Neutral spectrum | Existing residual-enforced `solve_ground`, including its independent measured ground-frame completion |
| Charged pole search | Complete charged `linear_operator()`, `eigsh(k=1,which='SA',v0=normalized_A_image)` and an independently evaluated full-H residual |

Default parameters are minimal Haar (`1,4,bar4`), `R=1`, `Lambda=1`, `I=2`,
`Jpref=.2`, `m=.2`, `r=.12`, `v=.3`, stiffness `2`, `UE4=.3`, `UE1=.4`, and
`K=.7`. The heavy whole-H sign is `-1`, as in the common supplier, and both
flavours use the same bare band. These are declared EFT couplings, not fitted
or derived electron parameters. `Jpref` is required positive: Ward checks must
exercise an active exact alignment term. `R=2` is exposed without changing the
colour, matter or boundary policy.

The colour paths obey the supplier's explicit flat-colour-loop constraint:
there is one independent colour connection and colour momentum. This is not a
construction of an unconstrained multi-link SU(4) gauge theory.

## The five-axis charge map and its genuine adjoint

Write the actual right primitive light annihilators as `c_i`, `i=1,...,4`.
The source has total light number four, but its **right** number is unrestricted
within `0,...,4`. For hole charge `q=-1` and particle charge `q=+1`, respectively,

\[
 A_- = \sum_i c_i\otimes|i\rangle,
 \qquad
 A_+ = \sum_i c_i^\dagger\otimes|\bar i\rangle.
\]

In each block the implementation obtains the rectangular `c_i` restrictions
from `BoundaryChargeIntertwiner(code,neutral.base,charged.base)` and applies them
to the matter axis alone. Both colour indices and `ell` are spectators. Boundary
basis vectors are inserted at the **fourth** axis, leaving `ell` fifth. The
adjoint contracts that boundary index and applies the actual rectangular ladder
adjoints to matter. It is not an occupation guess or a reshape of merged axes.

Primitive CAR and orthonormal boundary vectors give operator identities on the
entire complete neutral domain:

\[
 A_-^\dagger A_-=N_R=:W_-,\qquad
 A_+^\dagger A_+=4-N_R=:W_+.
\]

Thus the maps vanish in the genuine `N_R=0` hole or `N_R=4` particle sector.
They are not all-state unit isometries. For a normalized source `psi`, define
`w=<psi,W psi>` and `phi=A psi/sqrt(w)` when `w>0`. The code checks sparse
primitive operator norms, the full five-axis `A†A` action, and actual adjoint
pairing with an independent charged vector.

## All preserved operators, including active alignment and magnetic dynamics

The charged right boundary generators are `T^a` for a hole and `-conj(T^a)` for
a particle. Actual primitive CAR therefore gives all fifteen relations

\[
 (Q_R^a+Q_b^a)A=AQ_R^a,\qquad Q_L^a A=AQ_L^a.
\]

Both colour electric generators commute with `A` because neither colour index
changes. Consequently the actual left and right Gauss derivatives intertwine,
and so does their positive square

\[
 H_G=\frac1{2I}\sum_{a=1}^{15}
 \big[(Q_L^a-E_L^a)^2+(Q_R^a+Q_b^a-E_R^a)^2\big].
\]

This also proves that the supplier's expanded `orientation` action intertwines.
The code checks all thirty primitive generator relations as sparse rectangular
operator identities, all thirty lifted Gauss derivative actions, and independent
full five-axis Gauss-square/expanded-orientation agreement. Reversing or omitting
the boundary generators is an actual adverse operator, not a changed tolerance.

Every heavy bilinear is even and commutes with the light charge word. Its actual
onsite and Haar-link hopping terms therefore intertwine, including the heavy
whole-H multiplier. The complete heavy term is checked, not merely its onsite
energy. The colour electric Casimir also intertwines.

The exact alignment `P W(U) P` acts only on the common colour connection and
mixes only equal-centre matter fibres. The same restricted charge ladders apply
in those fibres, and the exact same projected provider is shared by all sectors.
Thus the full active alignment action intertwines. No product of separately
compressed coordinate multipliers replaces that provider. The consumer invokes
the actual model alignment action and checks its intertwining separately.

The dynamical magnetic action is `-K(S+S†)/2` with the supplier's nonwrapping
compressed `ell` shifts. Since `A` neither changes `ell` nor treats it as the
boundary, magnetic dynamics intertwines at every finite `R`. This assertion
needs no cyclic wrapping and makes no infinite-cutoff convergence claim.

For the two oriented internal EM fields, in supplier conventions,

\[
 E_0=\ell+N_L-2,\qquad E_1=\ell,
\]

and the complete electric energy is

\[
 H_E=UE4\,C_2(U)+\frac{UE1}{2}(E_0^2+E_1^2+q^2).
\]

The charge word acts at the **right** cell; `N_L` and both internal edge fields
are unchanged. Its matching charged boundary carries the new `q`. Hence

\[
 H_{E,q}A-AH_{E,0}=\frac{UE1\,q^2}{2}A.
\]

The code separately checks colour electric, each internal field, and the entire
EM electric action including the boundary constant. It does not charge heavy
occupations electromagnetically.

## Exact weighted Rayleigh identity and stationary light cost

Let `H_0` be the complete neutral Hamiltonian and `H_q` the complete charged one,
including active alignment, both hopping paths and dynamical magnetism. The only
nontrivial commutator besides the boundary constant is the light bilinear part
`h_L`. For any source with positive `w`, without assuming stationarity,

\[
 \langle\phi,H_q\phi\rangle
 =\frac{\langle\psi,W H_0\psi\rangle}{w}
  +\frac{UE1}{2}
  +\frac{\langle A\psi,(h_{L,q}A-Ah_{L,0})\psi\rangle}{w}.
\]

The generally complex terms on the right combine to a real result. The code
compares the full charged Rayleigh action to this identity, and reports ordinary
and occupation-weighted neutral energies distinctly.

For a neutral eigenstate `H_0 psi=E_0 psi`, the weighted total energy is `E_0`.
Write `h_R` for the right-cell light onsite action and `T+T†` for the **full
actual two-path** light hopping at zero external phases. The photon shifts and
Haar matrices in `T` are retained. Primitive CAR contracts the hole hopping
commutator to one directed term and the particle commutator to its adjoint. The
onsite contraction is `-h_R` for holes and `tr(h_R)-h_R` for particles; the
selected onsite matrix is traceless, so both give `-h_R`.

Only light hopping changes `N_R`; all other complete-H terms commute with it.
Stationarity of `N_R` implies `<T-T†>=0`. Therefore for either charge,

\[
 \boxed{
 \langle\phi,H_q\phi\rangle-E_0
 =\frac{UE1}{2}
  -\frac{\langle h_R\rangle+\tfrac12\langle T+T^\dagger\rangle}{w}
 }
\]

up to the explicitly measured neutral eigenresidual in a numerical run. The
code checks the complex contracted commutator, stationarity, weighted total
energy, independent right-onsite/half-full-hopping numerator and full Rayleigh
difference. It reuses `HamiltonianParts.onsite_action` on each independent loop
slot and `model.hop_action(flavour=...)`; it neither duplicates the microscopic
Hamiltonian nor reconstructs a directed hopping operator from `H±iJ`.

For any exactly intertwined operator `O_q A=A O_0`, normalized expectations obey

\[
 \langle\phi,O_q\phi\rangle
 =\frac{\langle\psi,W O_0\psi\rangle}{w},
\]

**not** generally `<psi,O_0 psi>`. Receipts retain both values for angular Gauss,
complete heavy, colour electric, alignment and dynamical magnetic terms. The
right-occupation distribution is measured on the actual coupled neutral ground
state, rather than frozen to `N_R=2`.

The Rayleigh cost is a variational **upper** energy relative to a numerical
neutral reference. Positivity of that upper cost does not prove a lower charged
mass gap. No such inference is made.

## Residual-confirmed complete-H particle poles and spectral moments

The neutral source comes from the existing full coupled `solve_ground`; its
receipt retains measured ground rank, full-H residuals and independent deflated
completion. For a degenerate measured neutral multiplet this consumer chooses
its first orthonormal member and says so. This is not a multiplet-averaged source.

For each charge the normalized primitive image initializes `eigsh` on the
**complete** charged Hamiltonian, not `A†H_q A` or a fixed-image Ritz truncation.
The returned `k=1` candidate is independently normalized and checked with the
actual full Hamiltonian. Its scaled residual must satisfy the declared tolerance,
its energy transfer from the neutral reference must be positive beyond the
residual allowance, and its actual creator residue must exceed the declared
source-activity threshold. A failed condition raises rather than inventing a
particle pole.

A source-initialized search can explore the cyclic source subspace and can miss
orthogonal states; numerical roundoff can also mix symmetry sectors. A positive
residue and small full-H residual identify a source-active full-H eigenstate,
not a certificate that it is the complete charged ground or globally lowest
active pole. Even `which='SA'` does not upgrade this one-search result to a ground
or gap certificate. A degenerate returned eigenvector captures only a rank-one
residue, not necessarily its entire eigenspace residue. No finite-Krylov
quadrature poles are substituted for residual-confirmed eigenstate poles here.

For the unnormalized primitive source `s=A psi`, the **total** moments are
computed directly without a spectral truncation:

\[
 \mu_0=\|s\|^2=w,\qquad
 \mu_1=\langle s,(H_q-E_0)s\rangle,
 \qquad
 \mu_2=\|(H_q-E_0)s\|^2.
\]

For a normalized residual-confirmed candidate `v`, define
`Z=|<v,s>|²`, `Delta=E_v-E_0`. The code reports `Z`, `Z/w`, the unseen zeroth
weight `w-Z`, the signed first-moment remainder `mu1-Z*Delta`, and an independent
projection/action on the unseen source. Zeroth and first partition identities
are enforced, the latter with an allowance tied to the measured full-H
candidate residual. The total first moment divided by `w` must equal the actual
normalized-image Rayleigh transfer. No unseen spectral weight is silently
assigned to the returned pole, and the remainder's first moment need not be
positive without separate spectral knowledge.

An optional `--fullground-receipt` compares against a separately executed parent
`quantum_su4_photon.py` receipt only after matching supplier source hashes, bare
and heavy-band parameters, cutoff, `R`, `Lambda`, `I`, and active `Jpref`. All
three complete-H sectors must be present. The comparison reports the parent
charged ground Ritz estimates, residuals, measured ranks/gaps and pole offset;
it does not transform numerical measured gaps into exact certificates. No
receipt supplied means the comparison is explicitly unavailable.

## Current/contact Ward checks and discriminating wrong operators

The actual `model.operator_evidence` executes on the neutral and both charged
models with positive `Jpref`. It checks the full-H pure-gradient covariance,
actual current/contact phase derivatives, density-current and density-contact
Ward commutators, angular factorization, loop-slot policy, heavy EM neutrality,
and its existing wrong-Gauss/heavy-charge/wrapping controls. Alignment and
magnetism remain dynamical and source-independent. External phases act only on
the two actual light hopping paths; no source is added to the magnetic term.

Three additional adversarial constructions expose concrete wrong implementations:

1. **Wrong boundary:** reversing or omitting `Q_b` in the actual Gauss square
   breaks the charge intertwiner. Both are compared to the correct full angular
   action on a nontrivial lifted primitive image.
2. **No `ell` shift:** a path-1 matter transfer increases `N_L` while decreasing
   `ell`, preserving the other edge `E0`. The actual path-1 current commutes with
   `E0`; replacing its `ell` shift by the identity produces a nonzero wrong-edge
   commutator and changes the actual light Hamiltonian. Path-0 current likewise
   preserves `E1`. Merely testing an EM density Ward relation would not expose
   this error, because that relation can survive the incorrect no-shift model.
3. **Unweighted normalization:** an explicit equal-probability source on native
   `N_R=1,3` at `ell=0` has ordinary `<N_L>=2`. Its normalized actual hole image
   has `<N_L>=1.5`; its particle image has `<N_L>=2.5`. The weighted neutral
   expectation agrees exactly, while the unweighted prescription is wrong by
   exactly `0.5`. This is an operator/meaning witness, not a threshold selected
   because a physical ground-state covariance happened to be large.

The consumer reconstructs the full Hamiltonian from actual supplier actions,
checks all preserved parts separately and never changes or monkeypatches a
shared module.

## Public APIs, commands and source binding

The reusable APIs are:

- `PhotonChargeIntertwiner(code, neutral, charged)` with `apply`, `adjoint`,
  `weight`, `normalized` and the actual primitive `components`.
- `PhotonHamiltonianParts(model)` with lifted `onsite_action`, actual `flavour`,
  `angular`, `colour_electric`, `em_electric`, `alignment`, `magnetic` and
  `reconstruction`.
- `operator_evidence`, `charge_trial`, `source_active_pole`,
  `compare_fullground`, `source_identity`; `gauss_generator`, `gauss_square`,
  `edge_field` expose the checked complete-domain operator conventions.

From the repository root, the full default execution is:

```sh
/home/administrator/substrate-framework/.venv/bin/python \
  proposals/P256-qed-backward/gauss_photon_charge.py \
  --output proposals/P256-qed-backward/gauss_photon_charge_receipt.json
```

The same-law cutoff comparison and operator-only smoke are:

```sh
/home/administrator/substrate-framework/.venv/bin/python \
  proposals/P256-qed-backward/gauss_photon_charge.py --R 1 2 \
  --output proposals/P256-qed-backward/gauss_photon_charge_r1_r2_receipt.json

/home/administrator/substrate-framework/.venv/bin/python \
  proposals/P256-qed-backward/gauss_photon_charge.py --operators-only
```

`--operators-only` does not produce charged poles or solve a neutral ground; it
is an explicitly labelled bounded smoke, not the complete scientific execution.
To compare independently executed complete-ground data, add
`--fullground-receipt PATH` pointing to an exact matching parent photon receipt.

Before orbital minimization, Haar construction or an eigensolve, the CLI captures
all loaded local supplier/source hashes and this proof note's hash. It checks
the inherited atomic source pin and the active aligned photon pin, then emits
a flushed source-bound preflight record to stderr. Final JSON includes the same
source identities, and completion fails if any source or proof note changed
during execution. An interrupted run's preflight record is provenance of an
attempt, not a completed scientific receipt.

Computational cost grows with the complete native matter fibre, both colour
indices, boundary dimension and `2R+1` loop slots. Larger Haar cutoffs and neutral
ground multiplicities require larger complete-H eigensolves. No state pruning,
fixed-image charged diagonalization, random group replacement or wrapping is
used to make a larger run appear tractable. Native Python/Pyright symbol and
reference probes and a positive AST structural query were exercised during
implementation; the OMP `xd://lsp` device was unavailable. Scientific CLIs and
post-integration diagnostics are reserved to the parent and are not claimed as
worker-observed results.

## Scope retained

This construction joins genuine primitive charge creation/annihilation,
continuous colour Gauss angular dynamics, active exact Haar alignment,
compact-loop electromagnetic dynamics, actual external matter current/contact,
and residual-checked particle-source eigenstate poles in one declared finite
EFT. Its theorem preserves operators and derives an occupation-weighted charged
trial cost; its CLI measures the actual coupled model.

It does not select a Dirac doublet, lower an emergent mass, prove an
infinite-cutoff limit, quantize the unrestricted nonlinear sixty-coordinate
orbital field, establish a three-dimensional photon phase, derive Lorentz
invariance, demonstrate mechanical spin, supply a photon S-matrix, or complete
the backward electron–photon campaign. Fundamental quantum dynamics, the fixed
selected orbital orbit, finite connection cutoffs, flat-colour-loop constraint
and declared bare couplings remain explicit microscopic/representation debt.

## Parent-owned coupled execution

The `--R 2 --fullground-receipt quantum_su4_photon_aligned_receipt.json`
invocation completed naturally in 322.20 s after the number-operator dtype
and heterogeneous-receipt type repairs. Its output is
`gauss_photon_charge_verified_receipt.json`, bound to source SHA256
`b0336874a2aee458aa8e4b621728b3a1d05c23f18f5adeb2bc98b10d630ec9d7`.
The shared aligned photon supplier remains
`ffadad919f74fe510484530037d4ae11b7860046fa384aa721839d2771d5c0bf`.
Parameters are `Lambda=1,inertia=2,Jpref=.2,m=.2,r=.12,v=.3,stiffness=2`,
`UE4=.3,UE1=.4,K=.7`, with whole-heavy sign `-1` and minimal colour cutoff.

| Charge | Complete-H source pole | Transfer from neutral | Normalized creator residue | Full-H residual |
|---:|---:|---:|---:|---:|
| -1 | -0.42633957456507476 | 0.284743263400302 | 0.9645065263318479 | 1.510561400916e-11 |
| +1 | -0.426339574565097 | 0.2847432634002797 | 0.9645065263318874 | 1.510553416428e-11 |

For both signs, the actual creator norm is approximately `2`, total first
transfer moment `0.6356052501195`, normalized first moment
`0.31780262505978`, and unseen zeroth weight `0.07098694733685`
(`0.03549347366843` after normalization). The unseen **signed** first
transfer moment is `0.0863317783623`; agreement with its directly projected
value does not turn it into a positive-gap certificate.

The separately executed full-ground receipt is source/parameter matched,
SHA256 `43ed66ee2726ba6ccad20c0d1d6688d6104c4e45a986fff221472563afc81771`.
The two source poles agree with its charged ground Ritz energies within
`7.35e-14`. This comparison is independent finite-domain evidence, not a
claim that source-initialized `k=1` alone proves spectral completeness.

`ReviewPhotonCharge` inspected the repaired source and derivation and found
no actionable defect. Its numerical inspection used the historical
`f585ffb087d8a641ab329b85fd7e7d0ec9fa9fd697c8b7d8ecca25b63b6fa3e2`
receipt; the fresh execution above is parent-owned, not reviewer-run.
All full-field, continuum, Dirac, Lorentz, scattering and mechanical-spin
limits in the preceding section remain open.
