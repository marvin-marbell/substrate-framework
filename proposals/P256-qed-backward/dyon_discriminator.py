#!/usr/bin/env python3
"""Test a same-photon charge-monopole statistics mechanism before joining it."""
from __future__ import annotations

import json

import numpy as np


def field_angular_momentum(electric: float, magnetic: float, separation: float, order: int) -> float:
    """Integrate J_z of separated point sources; electric at +d/2, magnetic at -d/2.

    Cylindrical Maxwell fields give rho S_phi = -eg d rho^2/(16 pi^2 re^3 rm^3).
    Integration includes the full infinite volume; coordinate singularities are
    endpoints of split quadratures, not deleted cores or fitted spin labels.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    u = (nodes + 1) * np.pi / 4
    wu = weights * np.pi / 4
    rho = separation * np.tan(u)
    drho = separation / np.cos(u) ** 2
    integral = 0.
    endpoints = (-np.pi / 2, -np.arctan(0.5), 0., np.arctan(0.5), np.pi / 2)
    for lower, upper in zip(endpoints[:-1], endpoints[1:]):
        v = (nodes + 1) * (upper - lower) / 2 + lower
        wv = weights * (upper - lower) / 2
        z = separation * np.tan(v)
        dz = separation / np.cos(v) ** 2
        re2 = rho[:, None] ** 2 + (z[None, :] - separation / 2) ** 2
        rm2 = rho[:, None] ** 2 + (z[None, :] + separation / 2) ** 2
        integrand = (rho[:, None] ** 3 * drho[:, None] * dz[None, :]
                     / (re2 * rm2) ** 1.5)
        integral += float(wu @ integrand @ wv)
    return -electric * magnetic * separation * integral / (8 * np.pi)


def run() -> dict:
    # Minimal Dirac product eg=2pi. No physical electron parameter is fitted.
    electric = float(np.sqrt(2 * np.pi))
    magnetic = float(2 * np.pi / electric)
    fields = []
    for order in (32, 64, 128, 256):
        for separation in (0.5, 1., 3.):
            value = field_angular_momentum(electric, magnetic, separation, order)
            fields.append({"quadrature_order": order, "separation": separation,
                           "J_z": value, "absolute_error_from_exact_field_integral": abs(value + 0.5)})
    assert max(row["absolute_error_from_exact_field_integral"] for row in fields[-3:]) < 1e-4
    # Orthogonal electric/magnetic duality preserves the canonical Maxwell energy
    # metric. Rotating the dyon to a purely electric source gives Q^2=e^2+g^2.
    electric_values = np.geomspace(0.03, 100., 2001)
    alpha = (electric_values ** 2 + (2 * np.pi / electric_values) ** 2) / (4 * np.pi)
    index = int(np.argmin(alpha))
    target_alpha = 1 / 137.035999177
    target_charge = float(np.sqrt(4 * np.pi * target_alpha))
    assert float(alpha.min()) >= 1 - 1e-12
    assert target_alpha < 0.01
    # A visible-charge projection onto a separate photon can be weak. The
    # orthogonal charge does not vanish: gapping/transfer of that sector is debt.
    composite_charge = float(np.hypot(electric, magnetic))
    visible_projection = target_charge / composite_charge
    hidden_charge = float(composite_charge * np.sqrt(1 - visible_projection ** 2))
    return {"scope": "Canonical single Maxwell photon, theta=0, tree-level source normalization; "
                     "minimal electric/magnetic constituents. Not a theorem excluding all dyons, "
                     "theta terms, charge screening, kinetic mixing or additional sectors.",
            "actual_field_integrals": fields,
            "exact_angular_momentum": "J=-eg/(4pi) direction; minimal |J|=1/2",
            "binding_debt": "No electrostatic cross attraction between pure electric and pure magnetic "
                            "sources. This calculation does not produce a localized bound electron.",
            "same_photon_coupling": {"analytic_minimum_alpha": 1.,
                "scanned_minimum_alpha": float(alpha[index]),
                "scanned_electric_at_minimum": float(electric_values[index]),
                "target_alpha": target_alpha,
                "minimum_to_target_ratio": 1 / target_alpha,
                "decision": "Half-angular-momentum alone cannot transfer to the weak-charge tree consumer "
                            "under this single-photon source map."},
            "material_alternative": {"visible_projection": visible_projection,
                "visible_charge": target_charge, "remaining_hidden_charge": hidden_charge,
                "decision": "Statistics may come from a genuinely distinct hidden topological sector; "
                            "derive its gapped dynamics and visible-current map, not relabel the same field."},
            "status": "Target discriminant and route selection; no completed particle mechanism"}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
