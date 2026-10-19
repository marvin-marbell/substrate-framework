# Compact photon pole and source corrections from the actual rotor

`compact_asymptotics.py` derives and executes a non-Gaussian extension of
[quantum_compact_photon](quantum_compact_photon.md). The invariant neutral
Gauss block is the actual noncyclic loop rotor, not an appended oscillator.
In units hbar=1,

`H=2 UE n²-K cos(phi)`, `B=sin(phi)`, `omega=2 sqrt(UE K)`, `v=2 UE/omega`.

Holding omega fixed and taking v small gives

\[
\epsilon_1=\omega-\frac{U_E}{2}-\frac{U_E^2}{4\omega}+O(U_E^3/\omega^2),\qquad
w_1=v-\frac34v^2+O(v^3),\qquad
w_3=\frac{27}{128}v^3+O(v^4).
\]

The third odd excitation has `epsilon_3=3 omega-3 UE+O(UE²/omega)`.
Here epsilon denotes excitation energy above the ground, and w is the
squared matrix element of the dimensionless B, not a field-strength residue
in SI units. The remainder is a local weak-angle expansion; winding and
finite electric cutoffs require separate control.

## Derivation and wrong Gaussian alternative

Write `phi=sqrt(v) X`, `X=a+a†`. The compact quartic is `-UE X⁴/24`;
the sextic first-order term and quartic second-order mixing together produce
the second pole coefficient `-1/4`. First-order corrections to both states
and the cubic term in B give the source coefficients `-3/4` and `27/128`.
The script evaluates the complete intermediate-state oscillator algebra in
exact SymPy arithmetic; it does not truncate the physical rotor to that
intermediate algebra dimension.

A harmonic Hamiltonian with the nonlinear observable sin(phi) instead gives
`w1=exp(-v) v=v-v²+...`, `w3=v³/6+...`, and unshifted odd harmonics. Thus
multiple B poles alone do not discriminate a non-Gaussian Hamiltonian;
the calculated energy and source coefficients do.

## Executed consumer

From the repository root:

```sh
python proposals/P256-qed-backward/compact_asymptotics.py \
  --output proposals/P256-qed-backward/compact_asymptotics_receipt.json
```

The complete run naturally exits zero. It uses electric cutoffs 20,32,48
and UE=.2,.1,.05,.025,.0125, with omega=2 and K=1/UE.
The exact neutral raw-Gauss model and direct rotor agree to `4.44e-16`.
At UE=.2 the measured first excitation is `1.89419189243`; at UE=.0125
it is `1.99373030670`. The second-order approximation improves every
selected first-order error. The final cutoff-32 to cutoff-48 change at
UE=.0125 is `1.14e-9`, and all coefficient/source/alternative controls pass.
The initial insufficient-cutoff receipt is retained separately as
`compact_asymptotics_coarse_receipt.json`, not overwritten into a success.

This supplies a quantum compact correction to the same local photon law.
The finite loop is not by itself a three-dimensional propagating photon or
a scattering normalization; those joins remain separate constructions.
