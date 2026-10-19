# Primitive bosonic currents, full physical parity, and the filled-sea Compton join

This construction supplies full physical fermionic operators, charged Gauss-preserving states and connected Compton scattering from actual bosonic Pauli words. The [energetic code theorem](code_phase.md) selects its low-energy sector; [sea-energy minimization](sea_selection.md) and the [soft orbital law](soft_budget.md) select its spinor algebra. The current transfer is therefore an ingredient of a growing dynamical explanation, with quantum postulates and microscopic coefficients stated explicitly.

## 1. Local law and a full physical Fock space

`bosonic_statistics.py` supplies one code qubit per graph edge, local words

\[
B_i=\prod_{e\ni i}Z_e,\qquad n_i=(1-B_i)/2,
\]

and oriented A_ij: an edge X with ordered incident Z tails, A_ji=-A_ij. Shared-vertex A words anticommute. On the selected commuting loop sector, all

\[
S_C=i^{|C|}\prod_{(i,j)\in C}A_{ij}
\]

have eigenvalue +1. An endpoint word along a simple path of k edges is
`A_path=i^(k-1) product A_edges`. The graph identities and loop conditions make path-equivalent endpoint words agree on this sector.

The original closed encoding fixes total occupation parity because `product_i B_i=I`. An artificial second charged species is not a satisfactory cure. Instead attach **one neutral hidden boundary leaf a**. It is a code parity marker, not an electromagnetic particle, a negative-energy spectator, or another electron flavor. Define

\[
\gamma_i^{(1)}=A_{ai},\qquad \gamma_i^{(2)}=i A_{ai}B_i,
\qquad c_i= A_{ai}n_i,\qquad c_i^\dagger=A_{ai}(1-n_i).
\]

The endpoint/shared-vertex algebra gives `{gamma_i^r,gamma_j^s}=2 delta_ij delta_rs`; consequently full physical CAR follows from the primitive bosonic words. No CAR matrices construct these operators. Odd operators have boundary strings; even bilinears retain local paths:

\[
c_i^\dagger c_j=-i A_{ji}n_j(1-n_i).
\]

`product_physical B_i=B_a`. Thus the physical fermion parity is unrestricted, while the total code parity, including the neutral marker, remains even. With V physical modes the selected code dimension is exactly `2^V`, not `2^(V-1)`. Giving the marker electric charge would incorrectly double a physical creation operator's charge.

The actual four-physical-mode/tree CLI checks all mixed and lowering anticommutators on the complete 16-dimensional physical Fock space: residuals **0**. This is an algebraic consequence on the selected code domain, not an assertion that arbitrary bosonic Hamiltonians generate CAR.

## 2. Bounded microscopic spatial support

Each physical cubic cell has four orbitals and only the three internal star edges `0-alpha`. Spatial edges join orbital 0 of adjacent cells. Internal edges carry **no physical electromagnetic rotor**. Elementary spatial plaquettes generate the full cycle space on a simply connected open box; stars and the boundary leaf introduce no additional cycles.

An on-site band bilinear has a path of at most two edges, a nearest spatial bilinear at most three, and an improved straight-two-link bilinear at most four. All primitive degrees and Pauli support stay bounded with volume. This is not the potentially nonlocal fundamental-cycle penalty of a generic dense graph.

The actual symbolic Pauli CLI verifies ranks, signs of dependent products, Hermiticity, and commutation with **every** primitive A/B word:

| Cubic side | Physical modes | Code qubits | Plaquette count / independent rank | Selected dimension |
|---|---:|---:|---:|---:|
| 2 | 32 | 37 | 6 / 5 | 2^32 |
| 3 | 108 | 136 | 36 / 28 | 2^108 |
| 4 | 256 | 337 | 108 / 81 | 2^256 |

There are no frustrated `-I` dependencies. Selected maximum path-A weights are 11, 21, 22; maximum path length is four. These are exact finite symbolic checks plus the general bounded-degree/path argument, not dense simulation of a 337-qubit state.

The [coupled code-phase theorem](code_phase.md) supplies ground-sector selection when `Delta>2K`, without imposing the code projector as the microscopic law. Adding the boundary leaf preserves that theorem: it adds no cycle and its occupation is preserved by Q_eta and all physical Hamiltonian terms. Periodic space needs separately specified noncontractible sectors.

## 3. Genuine charged Gauss-physical creation

The CLI also constructs two physical matter cells, a neutral hidden leaf, and two actual noncyclic quantum electric rotors. One rotor is internal, the other is an electric boundary link. In raw dimensionless electric variables,

\[
G_0=E_{01}+E_{0b}-n_0,\qquad G_1=-E_{01}-n_1.
\]

Physical charge is `q(n0+n1)`; the leaf is excluded. A physical electron at cell 1 is prepared with the Pauli boundary creator and the **integer** string `U_01,lower U_0b,raise`. The complete 36-dimensional operator obeys `[G_i,D^dagger]=0` and `[Q,D^dagger]=q D^dagger`. From the vacuum it produces `E01=-1, E0b=+1, n1=1`; charge density and outgoing physical electric flux both equal **q=-0.303500410715123**.

The actual residuals are zero. Bare creation violates Gauss (commutator norm 4.24264); charging the marker violates the same-charge commutator (norm .858429). The reverse dressed annihilation recovers the physical vacuum exactly. This proves a finite physical charged string state, **not** Coulomb minimization or an interacting asymptotic packet.

Raw E is canonically conjugate to the link angle. For the standard vector-potential convention `theta=e a A`, `q=-e`, physical Maxwell electric field is `E_phys=q E_raw/a^2`; it is minus the momentum conjugate to the spatial vector potential. This sign explains why the raw outgoing flux +1 represents negative physical electron charge. Gauss, current and photon normalization must use the same convention.

## 4. Filled sea: where the negative poles actually come from

`FilledSeaConsumer(band, photons, (p,k,pf,kf))` diagonalizes the **selected** four-orbital band at the momentum blocks needed by the two photon insertions. Negative modes are occupied. External physical electrons are created by the primitive boundary operator; physical holes are removed from the filled sea. There is no fictitious charged spectator or imported fermion ladder matrix.

The mode graph is a tree and the calculation uses exact Pauli bit actions and sparse amplitude dictionaries. `primitive_pauli_words` is shared with the dense supplier; the complete dense convention control gives residual zero. Energy denominators are energies of actual intermediate occupation states, relative to this vacuum. They are not evaluated by calling the one-particle negative-band resolvent.

For a normal electron intermediate, denominators are `Ei+omega-E(ps)` or `Ei-omega_out-E(pu)`. Direct transfer into an occupied negative mode is Pauli-blocked. Negative Feynman poles are instead recovered from allowed time orderings that create an additional positive electron and a negative-band hole, then annihilate that pair. Their primitive transfer signs and many-body energies produce the required relative sign and denominator.

At forward scattering the code merges identical momentum blocks. Disconnected filled-vacuum current response is explicitly subtracted with the external overlap; it must not be mistaken for connected electron Compton scattering. Deleting virtual-pair states deletes them in that vacuum response too. The two-photon contact comes from the same link law as the first derivatives.

The calculation is a **finite spectral tree-response representation** of the local even-algebra model, not a claim that those spectral terms themselves have bounded microscopic spatial support, nor a full coupled-Hamiltonian time evolution. Full CAR establishes the change of representation; the spatial support proof is separate. Omitted spectator momentum blocks affect disconnected vacuum/radiative loops, not this selected connected tree amplitude.

## 5. Return to spin, opposite charge, scattering and frames

For the same primitive filled-sea states, the rest-mode lift of `RotationBand.generators` gives:

- electron generator equal to the positive-band generator;
- hole generator **minus the transpose** of the negative-band generator;
- vacuum rest-block spin zero;
- SU(2), Casimir `J^2=3/4`, and `exp(-2pi i J_z)=-I` for both physical electron and hole;
- normal-ordered electron charge q and hole charge -q.

The actual lift errors and charge errors are zero; SU(2)/Casimir errors are below `5e-16`, and the 2pi residual is `1.43e-15`. These are intrinsic rest-mode representations, not a stand-alone proof of full microscopic physical coordinate rotations. Magnetic response and selected moving-frame scattering are addressed by [rotation_band.md](rotation_band.md) and [moving_frame.md](moving_frame.md).

Five forward/nonforward/oriented tree inputs, with `a=.0125..05`, incoming momentum `.2` or `.7`, produce connected amplitudes matching the independent Wilson-link comparator below `1e-16` and both Ward substitutions below `7e-17`. The shared charge is negative; CLI `--charge-magnitude` denotes positive e. At `a=.025, incoming momentum=.2, theta=.7`, the full spin/polarization sum is `.0269319807241`; deleting real virtual pairs gives `.000778699392` and a nonforward Ward residual `.0115905564`. At forward scattering deleting pairs can still pass this Ward control while badly changing the amplitude: the rate/response control matters too.

[Moving-frame calculations](moving_frame.md) and [leading-soft quantum dressing](soft_quantum.md) use this same actual Pauli consumer. They extend the observational relationships, not the microscopic law-selection claim.

## 6. Reproduce and remaining debt

```sh
python proposals/P256-qed-backward/pauli_compton.py \
  --output proposals/P256-qed-backward/pauli_compton_receipt.json
```

The current observed receipt uses Python3.12.3/NumPy2.5.3 and source SHA256 `add1dead7ca5b9c2a811153ba312cb32742439bc3820ada17a47f9f872bb8850`. The integration manifest must additionally bind imported supplier files; a main-file hash alone is not the whole artifact.

The chosen primitive quantum law, scalar hopping geometry and elastic/rotor coefficients are the microscopic starting point. Energetic code selection, sea-selected spinor structure and the [spectral photon/scattering join](photon_matching.md) progressively replace previous endpoint assumptions with mechanisms. The next improvements are spatial/quantum orbital fluctuations, nonforward dressed modes and self-consistent radiative matter/photon feedback.
