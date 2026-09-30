"""Conditional spinning-particle interface; exact closure and adverse probes.

Run directly with the scientific Python environment. No calculations at import,
no generated files. This is an operator/interface calculation, not a microscopic
origin of QED. Conventions and second-order endpoint map are in constraint.md.
"""
from __future__ import annotations

import json

import sympy as sp

# Exterior monomials are sorted tuples of generator indices, 0..3 and 4=psi5.
Grass = dict[tuple[int, ...], sp.Expr]
ETA = (1, -1, -1, -1)


def clean(poly: Grass) -> Grass:
    return {key: value for key, coeff in poly.items() if (value := sp.expand(coeff)) != 0}


def add(*polys: Grass) -> Grass:
    result: Grass = {}
    for poly in polys:
        for key, value in poly.items():
            result[key] = result.get(key, sp.S.Zero) + value
    return clean(result)


def scale(poly: Grass, coefficient: sp.Expr) -> Grass:
    return clean({key: coefficient * value for key, value in poly.items()})


def mul(left: Grass, right: Grass) -> Grass:
    result: Grass = {}
    for a, av in left.items():
        for b, bv in right.items():
            if set(a).intersection(b):
                continue
            sign = (-1) ** sum(i > j for i in a for j in b)
            key = tuple(sorted(a + b))
            result[key] = result.get(key, sp.S.Zero) + sign * av * bv
    return clean(result)


def derivative(poly: Grass, index: int) -> Grass:
    """Left Grassmann derivative, including its position sign."""
    return clean({key[:pos] + key[pos + 1:]: (-1) ** pos * coeff
                  for key, coeff in poly.items() if index in key
                  for pos in [key.index(index)]})


def coefficient_derivative(poly: Grass, symbol: sp.Symbol) -> Grass:
    return clean({key: sp.diff(coeff, symbol) for key, coeff in poly.items()})


def bracket(left: Grass, right: Grass, pi: tuple, field: sp.Matrix,
            q: sp.Symbol, left_parity: int) -> Grass:
    """Reduced graded bracket for constant F; {pi_mu,pi_nu}=-q F_mu_nu."""
    result: Grass = {}
    for mu in range(4):
        for nu in range(4):
            result = add(result, scale(mul(coefficient_derivative(left, pi[mu]),
                                           coefficient_derivative(right, pi[nu])),
                                       -q * field[mu, nu]))
    odd_brackets = tuple(-sp.I * sign for sign in ETA) + (sp.I,)
    for index, value in enumerate(odd_brackets):
        result = add(result, scale(mul(derivative(left, index), derivative(right, index)),
                                   (-1) ** (left_parity + 1) * value))
    return result


def serialized(poly: Grass) -> dict[str, str]:
    return {"1" if not key else " ".join(f"psi{index}" for index in key): str(value)
            for key, value in sorted(poly.items())}


def classical_probe() -> dict:
    pi = sp.symbols("pi0:4", real=True)
    m, q = sp.symbols("m q", real=True, nonzero=True)
    entries = sp.symbols("F01 F02 F03 F12 F13 F23", real=True)
    field = sp.zeros(4)
    for (mu, nu), value in zip(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)), entries):
        field[mu, nu], field[nu, mu] = value, -value
    psi: list[Grass] = [{(index,): sp.S.One} for index in range(5)]
    f: Grass = {}
    for mu in range(4):
        for nu in range(4):
            f = add(f, scale(mul(psi[mu], psi[nu]), field[mu, nu]))
    kinetic = sum(ETA[mu] * pi[mu] ** 2 for mu in range(4))
    supercharge = add(*(scale(psi[mu], pi[mu]) for mu in range(4)), scale(psi[4], -m))
    h_without_spin = {(): (kinetic - m ** 2) / 2}
    h = add(h_without_spin, scale(f, -sp.I * q / 2))
    qq = bracket(supercharge, supercharge, pi, field, q, 1)
    closure = add(qq, scale(h, 2 * sp.I))
    preservation = bracket(supercharge, h, pi, field, q, 1)
    wrong = add(qq, scale(h_without_spin, 2 * sp.I))
    wrong_preservation = bracket(supercharge, h_without_spin, pi, field, q, 1)
    kappa = sp.Symbol("kappa", real=True)
    a = kappa * q / (2 * m)
    deformed_q = add(supercharge, scale(mul(f, psi[4]), -sp.I * a))
    deformed_h = scale(bracket(deformed_q, deformed_q, pi, field, q, 1), sp.I / 2)
    naive_h = add(h_without_spin, scale(f, -sp.I * q * (1 + kappa) / 2))
    naive_deformation_residual = add(deformed_h, scale(naive_h, sp.S.NegativeOne))
    return {
        "field": "six independent constant F_mu_nu; no zero-field shortcut",
        "QQ_plus_2iH": serialized(closure),
        "QH": serialized(preservation),
        "missing_spin_closure_residual": serialized(wrong),
        "missing_spin_QH": serialized(wrong_preservation),
        "deformed_H_minus_naive_rescaled_spin_H": serialized(naive_deformation_residual),
        "pass": not closure and not preservation and bool(wrong) and bool(wrong_preservation)
                and bool(naive_deformation_residual),
    }


def gamma_matrices() -> tuple[sp.Matrix, ...]:
    """Exact Dirac representation, independent of the numerical reference."""
    zero = sp.zeros(2)
    pauli = (sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[0, -sp.I], [sp.I, 0]]),
             sp.diag(1, -1))
    gamma0 = sp.diag(1, 1, -1, -1)
    spatial = tuple(zero.row_join(sigma).col_join((-sigma).row_join(zero)) for sigma in pauli)
    return (gamma0,) + spatial


def is_zero(matrix: sp.Matrix) -> bool:
    return all(sp.expand(value) == 0 for value in matrix)


def operator_probe() -> dict:
    gamma = gamma_matrices()
    gamma5 = sp.I * gamma[0] * gamma[1] * gamma[2] * gamma[3]
    psi = tuple(sp.I * gamma5 * matrix / sp.sqrt(2) for matrix in gamma)
    psi5 = sp.I * gamma5 / sp.sqrt(2)
    clifford = all(is_zero(psi[mu] * psi[nu] + psi[nu] * psi[mu]
                          - (ETA[mu] if mu == nu else 0) * sp.eye(4))
                   for mu in range(4) for nu in range(4))
    clifford = clifford and is_zero(2 * psi5 * psi5 + sp.eye(4))
    clifford = clifford and all(is_zero(matrix * psi5 + psi5 * matrix) for matrix in psi)
    x, y, energy, pz, m, q, b = sp.symbols("x y E pz m q B", real=True)
    # A_2=-Bx, hence F_12=-B; P=i partial-q A and [P1,P2]=i q B.
    def momentum(mu: int, vector: sp.Matrix) -> sp.Matrix:
        if mu == 0:
            return energy * vector
        if mu == 1:
            return sp.I * vector.diff(x)
        if mu == 2:
            return sp.I * vector.diff(y) + q * b * x * vector
        return pz * vector

    def dirac(vector: sp.Matrix) -> sp.Matrix:
        result = sp.zeros(4, 1)
        for mu in range(4):
            result += gamma[mu] * momentum(mu, vector)
        return result

    def charge(vector: sp.Matrix) -> sp.Matrix:
        return sp.I * gamma5 * (dirac(vector) - m * vector) / sp.sqrt(2)

    seed = sp.Matrix([1 + x * y, x ** 2 + y, x + y ** 2, 1 + x ** 2 * y])
    sigma12 = sp.I * (gamma[1] * gamma[2] - gamma[2] * gamma[1]) / 2
    scalar_square = sum((ETA[mu] * momentum(mu, momentum(mu, seed))
                         for mu in range(4)), sp.zeros(4, 1)) - m ** 2 * seed
    expected_square = (scalar_square + q * b * sigma12 * seed) / 2
    actual_square = charge(charge(seed))
    wrong_residual = (actual_square - scalar_square / 2).applyfunc(sp.expand)
    # Compare spin states at the same Landau oscillator index, not orbital ground
    # states optimized separately by spin. These exact values discriminate g=2.
    mass, electron_q, field_b, n = sp.S.One, -sp.S.One, sp.Rational(1, 10), 0
    orbital = abs(electron_q * field_b) * (2 * n + 1)
    spin_energy = {str(s): sp.sqrt(mass ** 2 + orbital - electron_q * field_b * s)
                   for s in (1, -1)}
    missing_energy = sp.sqrt(mass ** 2 + orbital)
    kappa = sp.Symbol("kappa", real=True)
    t = -kappa * q * b * sigma12 / (2 * m)

    def anomalous_charge(vector: sp.Matrix) -> sp.Matrix:
        return sp.I * gamma5 * (dirac(vector) - m * vector - t * vector) / sp.sqrt(2)

    expected_anomalous = (scalar_square + q * b * sigma12 * seed
                          + t * dirac(seed) - dirac(t * seed)
                          - 2 * m * t * seed - t * t * seed) / 2
    anomaly_correct = is_zero(anomalous_charge(anomalous_charge(seed)) - expected_anomalous)
    return {
        "clifford_brackets_exact": clifford,
        "differential_Q_squared_minus_H_zero": is_zero(actual_square - expected_square),
        "missing_spin_residual": [str(value) for value in wrong_residual],
        "landau_inputs": {"m": "1", "q": "-1", "B": "1/10", "n": n, "pz": "0"},
        "same_orbital_spin_energies": {key: str(value) for key, value in spin_energy.items()},
        "spin_up_minus_down": str(spin_energy["1"] - spin_energy["-1"]),
        "missing_spin_both_energies": str(missing_energy),
        "anomalous_full_square_exact": anomaly_correct,
        "pass": clifford and is_zero(actual_square - expected_square)
                and not is_zero(wrong_residual) and spin_energy["1"] != spin_energy["-1"]
                and anomaly_correct,
    }


def endpoint_probe() -> dict:
    """Exact off-shell resolvent check including two-photon contact and endpoint.

    Exact rational off-shell matrix inputs avoid inversion on a mass-shell pole.
    This checks the operator identity underlying external-state amputation, not a
    second numerical Compton benchmark or a scattering prediction by itself.
    """
    gamma = gamma_matrices()
    m = sp.Rational(1)
    d = gamma[0] * sp.Rational(2) + gamma[1] * sp.Rational(1, 3) + gamma[2] * sp.Rational(1, 5)
    # Two field insertions with nonzero scalar product: contact cannot vanish.
    v1 = gamma[1] + gamma[2] / 3
    v2 = gamma[1] / 2 + gamma[3]
    identity = sp.eye(4)
    b0 = d + m * identity
    s0 = (d - m * identity).inv()
    g0 = (d * d - m ** 2 * identity).inv()
    k1, k2 = d * v1 + v1 * d, d * v2 + v2 * d
    contact = v1 * v2 + v2 * v1
    ordered = b0 * g0 * (k1 * g0 * k2 + k2 * g0 * k1) * g0
    seagull = -b0 * g0 * contact * g0
    endpoint = -(v1 * g0 * k2 + v2 * g0 * k1) * g0
    first_order = s0 * (v1 * s0 * v2 + v2 * s0 * v1) * s0
    correct = is_zero(ordered + seagull + endpoint - first_order)
    without_contact = is_zero(ordered + endpoint - first_order)
    without_endpoint = is_zero(ordered + seagull - first_order)
    return {
        "scope": "off-shell mixed resolvent coefficient; analytic general map in constraint.md",
        "full_second_order_equals_first_order": correct,
        "without_contact_equals_first_order": without_contact,
        "without_endpoint_equals_first_order": without_endpoint,
        "contact_matrix": [[str(value) for value in row] for row in contact.tolist()],
        "pass": correct and not without_contact and not without_endpoint,
    }


def run() -> dict:
    result = {"status": "conditional_interface_not_ontology", "classical": classical_probe(),
              "operator": operator_probe(), "endpoints": endpoint_probe()}
    result["pass"] = all(result[key]["pass"] for key in ("classical", "operator", "endpoints"))
    return result


def main() -> None:
    result = run()
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
