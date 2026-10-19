# Actual periodic mean on the finite centre-flux cube

`bulk_frame_flux.py` uses each explicit validation-mean coordinate array from
`periodic_bulk_frame_receipt.json` in the source-owned `onebody` graph operator.
The tested geometry is the open two-by-two-by-two cube, not a periodic bulk flux
insertion. All twelve links are actual SU(4) centre matrices. Seven tree links can
be fixed to identity, leaving five independent Z4 cycle labels: 1024 distinct
centre gauge classes. Face exponents obey the oriented cube Bianchi identity.

For each class the finite one-body Hermitian matrix has32 modes. The faithful
primitive exterior representation at fixed whole-heavy half filling gives
`E_H/Lambda = -sum(largest16 eigenvalues)`. This is not an absolute-value ansatz,
a per-momentum filling assumption, or a claimed 601080390-state N16 ED. Actual
4900-state two-cell Pauli controls test occupied energy/eigenvector and the
three-axis onsite correction to the pair owner. The all-N representation identity
belongs to the existing primitive matrix-unit derivation.

All means are inserted as their actual traceless-Hermitian matrices; no
Clifford/isotropy, gauge-intertwiner or old fixed environmental mean is substituted.
For any uniform mean each centre transport commutes with all four fields, so the
full J/8 alignment term is zero. The quartic scalar coefficient is common to every
flux class and is computed from the actual mean Gram matrix. No bare plaquette
term is added. Thus measured class differences are precisely induced whole-heavy
spectral differences at these fixed means. A positive numerical next-class gap
is a finite comparison in units of Lambda, not a thermodynamic gap theorem.

Every class exercises centre-gauge covariance, canonical tree recovery, Wilson
invariance, Bianchi, Hermiticity and trace zero. A genuinely noncentral site SU(4)
transformation jointly rotates fields and links and checks the same actual graph
operator, independently of selected-Clifford identities. Source/proof/input bytes
are bound before execution and rechecked before the receipt is accepted. Input
stationarity is explicitly attributed to its own receipt and is not recomputed by
this flux probe.

Four lifted centre link values on a single bond are gauge equivalent in the
scalar full-frame quotient. Closed-cycle Wilson values instead distinguish graph
configurations. Neither fact establishes a physical quartet, charged spin sector,
continuous-link minimum, deconfined Z4 phase or complete quantum electron-photon
construction.

Run with the project's scientific Python:

```
python proposals/P256-qed-backward/bulk_frame_flux.py \
  --output proposals/P256-qed-backward/bulk_frame_flux_receipt.json
```
