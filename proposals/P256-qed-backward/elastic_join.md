# Soft-selected matter, same-sea photon and nonforward inclusive rate

`elastic_join.py` connects the unrestricted [soft orbital law](soft_budget.md)
to the [continuous nonforward Pauli consumer](angular_photon.md) and its
[leading-soft quantum cloud](soft_quantum.md). Selection, band, photon sea,
current, recoil and source normalization are evaluated on one shared law.

## Joined owner and approximations

The actual normalized Brillouin quadrature minimizes the convex Gram energy.
Its isotropic selected frame gives `m_eff=m sqrt(g_m)`,
`r_eff=r sqrt(g_m)`, `v_eff=v sqrt(g_s)`. This same band and same weighted
sea produce the positive subthreshold photon poles and source residues.
The microscopic electric stiffness matches the **low-frequency response**
photon/matter speed at physical momentum/mass `.04`. This is an explicit
matching input, not an exact finite-frequency pole condition or a theorem
that the two speeds dynamically coincide for arbitrary inputs.
Magnetic stiffness is `1/.09`; the constructed code penalty is three times
that stiffness and retains the exact positive defect margin.

Physical mass one is a unit choice: `a=m_eff` retains the selected gap and
cutoff rather than overwriting them. The hard consumer uses its actual
incoming/outgoing pole energies, recoil root, group flux and radial energy-delta
Jacobian. The primitive Pauli current supplies both the hard amplitude and
the soft velocity coefficient. The independent rest-frame soft integral
checks that coefficient; the same quantum displacement law and detector
resolution then give the inclusive leading-soft veto factor.

The leading-soft cloud instead uses analytic transfer-zero coefficients from
the same `ContinuousSea`: `e_IR=1/sqrt(chi_e+DeltaChi_E(0))` and
`c_IR=sqrt((chi_b+DeltaChi_B(0))/(chi_e+DeltaChi_E(0)))`.
The finite matching probe does not define these infrared values. The receipt
keeps the actual infrared/matter velocity mismatch; for weighted order10,
ma=.2 it is `-9.94052e-8`, rather than being set identically to zero.

The approximation is a uniform orbital sea saddle, Gaussian rotor plus
quadratic sea response, tree charged response with dressed photon legs, and
leading-soft inclusive dressing. It does not silently become the full
nonlinear compact quantum state of the separate plaquette constructions.

## Executed results and reconstruction

```sh
python proposals/P256-qed-backward/elastic_join.py \
  --output proposals/P256-qed-backward/elastic_join_infrared_receipt.json
python proposals/P256-qed-backward/elastic_join.py \
  --quadrature split_gauss --orders 8 10 --mass-ratios .2 \
  --output proposals/P256-qed-backward/elastic_join_infrared_gauss_receipt.json
```

The complete current midpoint and weighted consumers execute all physical/adverse
checks. Historical earlier midpoint receipts retain the old direct-contact
quadrature representation. The [periodic contact theorem](photon_projector_trace.md)
now removes its low-q cancellation error without changing the primitive law.
For the current by-parts split-Gauss order10, bare ma=.2:

- Hard differential rate: `1.67503297366e-5`.
- Inclusive differential rate at soft upper energy `.01` and detector
  resolution `.001`: `1.67359900764e-5`.
- Relative hard-rate deviation from the matter-speed-unit QED comparator:
  `.00232941525`, not zero or a claimed global continuum error bound.
- Low-probe magnetic sea coefficient: `.06962849981`.

The order8/10 rate change is `3.59e-6` relative. Earlier source-bound
receipts and the pre-repair Gauss discrepancy remain available as diagnosis
and iteration evidence. Quantum compact corrections, dynamical orbital
amplitude, complete spatial ordering and additional photon modes are being
constructed from their actual operators, not appended as independent
unchanged spectra.
