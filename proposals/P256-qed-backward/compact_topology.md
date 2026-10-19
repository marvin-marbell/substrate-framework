# Compact-photon winding diagnosis and a frozen global kernel

The original long weak-coupling hot-start runs did **not** equilibrate. Their stable, tiny blocked errors were not evidence of convergence. The ordered-state photon/loop agreement remains scoped finite-volume evidence; it cannot by itself establish a phase.

## 1. Competing causes and direct observation

`compact_phase_refined.json` preserves the original 12 completed ensembles: `L=4,6,8`, beta `11.1111111111` or `.35`, ordered/random links, 600 burn sweeps, 2000 measurements, thinning2. At weak beta the L6 random chain retained monopole density `.00231481481481`, L8 `.0009765625`, while ordered chains had zero. Their larger Wilson loops disagreed severely. Integer global flux updates did not remove this discrepancy.

A constant quantized density suggested wrapping monopole worldlines but did not identify them. `compact_topology.py` reconstructs the actual signed dual current from the three-form plaquette boundary. The forward/backward offset matters: after dualization the current is shifted one link so outgoing-minus-incoming divergence is exactly zero. It decomposes that current into directed simple cycles and sums unwrapped steps to obtain integer winding. At intersections a decomposition need not be unique.

Replaying the **same** weak hot seeds through 600 preparation sweeps directly observes:

- L6, seed11351: two length6 cycles with opposite nonzero winding, 12 occupied dual links.
- L8, seed11353: two opposite nonzero-winding cycles, 16 occupied dual links.

A random strong-state positive control on the same topology map gives 490 occupied dual links (density `.478515625`). This distinguishes an empty observable from actual absence in the weak configurations.

## 2. A proposal, not a new gauge action

Write the raw plaquette curl as `d theta = f_principal+2pi n`, with integer Dirac sheets n. From the **preproduction snapshot** compute

\[
\delta\theta=-2\pi\,\Delta^{-1}d^\dagger n.
\]

The lattice inverse is Fourier-diagonal, with the true zero momentum excluded, not replaced by an absolute spectral floor. This Hodge field is a useful collective translation: it can remove the known monopole-sheet field while retaining the original compact variables and Wilson action. Its success is measured, not assumed.

Freeze the template before production. Propose either its positive or negative with equal probability; optionally translate the fixed template uniformly in periodic space. The resulting proposal distribution is independent of the current configuration and symmetric under reversal. Add the field, wrap link angles, and use the **actual original Wilson-action** Metropolis difference. Thus each fixed production kernel satisfies detailed balance with the original measure. Local heat baths and integer global flux proposals continue.

Recomputing the Hodge template from each current state and applying the same Metropolis rule would not establish this symmetry. The implementation forbids re-freezing a production kernel. The frozen-template law is a sampling repair, not a new physical mechanism or a guarantee of mixing across every unknown trap.

## 3. Executed repair and independent consumer checks

Both following CLIs completed naturally with Python3.12.3/NumPy2.5.3; source hash `0eeb4d6d4c69d79af65bada7d4196290eec298a32384e4060b57e75cfb77b67d`:

```sh
python proposals/P256-qed-backward/compact_topology.py \
  --output proposals/P256-qed-backward/compact_topology_receipt.json
python proposals/P256-qed-backward/compact_topology.py \
  --size 8 --seed 11353 \
  --output proposals/P256-qed-backward/compact_topology_l8_receipt.json
```

Defaults: beta11.111, burn600, measurements400, thinning2. Each production run attempted 800 global translations and accepted one. Both known winding pairs disappeared; the directly measured final dual current is zero. The fixed removing candidates lower the action by **568.0009** at L6 and **783.8627** at L8.

The important consumer is not that an update accepted. It is whether the formerly bad hot chain now reproduces the ordered estimates and independent loop predictions:

| L | Repaired hot connected helicity | Ordered helicity |
|---|---:|---:|
| 6 | 10.8567637 +/- .0003223 | 10.8563920 +/- .0002976 |
| 8 | 10.8563367 +/- .0001916 | 10.8561706 +/- .0002077 |

Differences are about .85 and .59 combined blocked standard errors. At L8 the repaired `R2,T3` Wilson loop is `.9200960 +/- .0001245`, versus the independent measured-stiffness Gaussian prediction `.9201908`; the bare-beta Gaussian gives `.9219475`. The fitted photon energy estimates remain finite-volume noisy correlator results, with invalid positive spectral fits reported as null. These uncertainties are blocked statistical estimates, not coverage of undetected metastability.

The renormalized pure-gauge coupling is conditional `e_R=1/sqrt(chi)`, approximately `.30350`, not the input bare `.3`. Using bare charge in a tree rate normalized to the measured photon stiffness would bias it by approximately `(beta/chi)^2-1=.0475`. Source, current and photon residue must be matched together.

## 4. Limits and next physical move

These are **pure four-dimensional Euclidean compact Wilson** ensembles. They do not prove a thermodynamic Coulomb phase, unquenched charged-matter matching, a fixed finite-regulator continuous-time Hamiltonian identity, or an electron. The sampler has repaired two directly diagnosed sibling traps without changing the action. A phase claim still requires additional domain/limit evidence.

The matter feedback calculation in [vacuum_feedback.md](vacuum_feedback.md) already demonstrates why measured pure-gauge stiffness is not automatically the dressed charge of the coupled model. Its finite-volume susceptibility has significant regulator/infrared dependence. The next work must control that limit and connect it to the same quantum Hamiltonian, not silently substitute the favorable ordered ensemble or a preset positive correction.
