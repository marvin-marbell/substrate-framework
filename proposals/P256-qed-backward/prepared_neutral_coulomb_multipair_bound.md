# Uniform ordinary-Fock Coulomb multipair bound

This is a new bounded operator calculation on the unchanged native N=3 law and original prepared-state cache. It is not a new preparation, exact encoded-Root proof, or whole quantum propagator.

## Literal law and space

The producer `FullSlaterFibre.coulomb` uses `H_C = U_E rho^T Lplus rho`, **without a factor 1/2**. Here `rho_x = n_light,x - 2 + (N_ref - 4) delta_xR`; heavy fermions are electromagnetically neutral. All108 light orbitals and the same four Ref modes are retained. The bound ranges over every ordinary-CAR occupation state in the relevant number sectors, including arbitrary particle-hole/multipair channels, not merely the compiled448 charged rows.

The periodic native Laplacian has smallest nonzero eigenvalue3, so its mathematical Moore-Penrose inverse is positive and has norm1/3. The actual floating Green eigenvalue extrema are retained in the receipt (tiny negative roundoff is not treated as a physical negative mode).

Since the densities commute, `0 <= H_C <= (U_E/3) sum_x rho_x^2`. Exact integer dynamic programming maximizes that sum under each conserved particle count:

| Sector | Light count | Ref count | Maximum density square |
| --- | ---: | ---: | ---: |
| Neutral plus | 54 | 4 | 104 |
| Neutral minus | 58 | 0 | 136 |
| Charged plus | 55 | 3 | 110 |
| Charged minus | 57 | 1 | 126 |

Witness occupation/density arrays and their actual Green energies are exported. The witness lists place the special Ref site last before the physical Green-energy permutation. Every correct witness has total charge0; omitting the original Ref background gives charge4 and is explicitly rejected.

## Dynamical meaning

Given self-adjoint unitary evolution of the original non-Coulomb common Hamiltonian, adding this bounded Coulomb operator gives the standard Duhamel norm estimate `||(U_full-U_withoutCoulomb) psi|| <= duration * ||H_C|| * ||psi||`. It is uniform in every photon/Phi wavefunction and all ordinary-Fock multipair channels. The calculation does not assert the missing full Hamiltonian/domain theorem or actually execute that full propagator.

With the literal native `U_E=2^-20`, the largest short-interval bound for the actual floating interval209.42 to209.451 is about1.34023e-6. The clock566 bound is0.02447001 and does **not** pay a1e-4 target. That is an insufficient bound, not an observed failure of the physical trajectory. No native1e-4 acceptance gate or parameter repricing is introduced.

For general N, `sin(pi/N)>=2/N` yields `||Lplus||<=N^2/16`; all ordinary occupations satisfy `sum rho^2<=4N^3+32`. Thus `U_E*N^2*(4N^3+32)/16` is a general uniform upper bound. Legal-N constants and encoded-Root/Gauss physical-state equivalence remain unpaid.

## Actual execution and provenance

The new CLI consumed cache SHA10d61adb4a374a05e45338522abefcda260ed25036396b1bea26c48c57e4bee3 and manifest90a6d7f34617540042406a07c5d8690cecdc456b2d585ee150c152c7d2e07f9a. It naturally exited0 in10.016386586s, with empty stderr, raw stdout equal to the scientific JSON, and stable loaded source bytes. No old H/G/preparation/loop scientific CLI was repeated. Executed/current source SHA5f346b2056b38181e116a7441c4cf87c6814958a0ec5a2f62fd7aa9cafa93962 matches actual post-run Pyright stdio RPC:41 symbols, two positive AST declarations, diagnostics received and empty. Full commands, exact rational bounds, source archive and stream/input hashes are bound by `local://issue222-coulomb-multipair-bound-execution.json` and the result capsule. Metadata never serializes the process environment.
