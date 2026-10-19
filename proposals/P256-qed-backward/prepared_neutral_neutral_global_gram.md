# Global neutral twelve-mode metric calibration

This pair evaluates the **unchanged original neutral wavefunction**, not a new filter, ground state, covariance, or normalization. It provides the uncut global Gaussian Gram for the new six-field-mode times two-original-bank trial span. The original first channel is the coherent neutral state; the second is its original plus/minus bank difference. The initial coefficient is1 in the first channel's mode0 and0 elsewhere, so the original function is preserved identically.

## Exact Gaussian integration

The frozen cache embeds only the original fourteen coefficients: two banks times seven normal polynomials, with the coexact hard vacuum and no generated neutral coefficients. The normal polynomials are `1` and `sqrt(2) directions^T q/sigma`, with the original six orthonormal directions. The original cached static boost has unit modulus. The Haar half-density cancels the physical field measure. Both number banks are orthogonal; the original Gaussian includes all55 EM variables, including the native harmonic packet.

The six enrichments are1, linear and quadratic Hermites along the actual Noether mean direction, the radial Hermite in **all2430 normal coordinates**, the first harmonic Gaussian Hermite, and the radial Hermite in all52 coexact variables. The Noether direction needs at most one extra active Gaussian coordinate beyond the original six. A Gauss-Hermite order4 rule in those seven variables integrates every degree-six polynomial exactly. The remaining2423 normal coordinates are integrated analytically through their chi-square mean and variance; they are not frozen or discarded. EM mode moments are integrated analytically. An independent order5 polynomial rule supplies a decision-bearing numerical control.

The resulting global Gram is positive: its eigenvalues range from1.05827062708 to1.10312663379. Its first diagonal entry, the original neutral wavefunction's uncut global norm-squared, is **1.08059389117056**, not the old conditional compiled-Gram value1. This is a changed metric artifact, not a retag of the old cache and not a changed state function.

## Compact support and scope

The original radius0.012 gives a flat cutoff for norm(q)<=0.006. Bounding the original linear polynomial and the sum of the six squared enrichments gives an explicit upper-incomplete-Gamma bound for the positive Gram loss from the compact cutoff. Its numerical evaluation is4.09682586306e-151. The bound formula is analytic; this floating evaluation is not a certified interval enclosure. It concerns the **neutral** enriched Gram only, not SOURCE tails, encoded Root/Gauss equivalence, Hamiltonian closure, legal-N, or global ground-state preparation.

A consumer may use the uncut neutral Gram as a controlled cutoff approximation, retaining this bound and comparing it openly to its Monte-Carlo metric estimate. No wavefunction, SOURCE globalZ, detector, or effect is renormalized. Stochastic Hamiltonian forms, word-channel charged metrics, and outside-span quantum evolution remain separate obligations.

## Actual execution

The new CLI naturally exited0 in23.335569435s, program time22.079229696s, empty stderr and raw stdout equal to scientific JSON. Order4 versus order5 Gram difference was4.78569931176e-14. Executed/current source SHA c0b1d9e7436748725c7e852d10ea27a210ea54c0644962cd07a31994cbddfab7 matches actual post-run Pyright stdio RPC:55 symbols, two positive AST declarations, diagnostics received and empty, server exit0. Source archive515045fcc567269a7751120ea4af29f2358d6363f1951c8d8a6c121c9fb3c582 and the exact command/input/stream evidence are recorded in `local://issue222-neutral-global-Gram-execution.json`. The execution is assigned to the explicit Mercury governance successor bf56a5dd; advice is not a scientific approval gate. No historical H/G/preparation/loop scientific CLI was repeated and no environment is serialized. The campaign remains OPEN.
