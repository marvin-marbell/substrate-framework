"""Exact periodic phase-reset CAR/fullE control, not ground-order evidence."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import sympy as sp

from local_orbital_stability import scalar_paths
from rotation_band import RotationBand
from jordan_ground_parent_loop_control_20261012 import car

ROOT = Path(__file__).resolve().parent
State = tuple[int, tuple[int, ...]]
Vector = dict[State, Any]


def simplify(value: Any) -> Any:
    return sp.simplify(sp.expand_complex(value))


def add(out: Vector, state: State, value: Any) -> None:
    out[state] = simplify(out.get(state, 0) + value)


def norm(v: Vector) -> Any:
    return simplify(sum(sp.conjugate(value) * value for value in v.values()))


def inner(a: Vector, b: Vector) -> Any:
    return simplify(sum(sp.conjugate(value) * b.get(state, 0) for state, value in a.items()))


def shift(v: Vector, delta: tuple[int, ...]) -> Vector:
    return {(mask, tuple(k[i] + delta[i] for i in range(4))): value for (mask, k), value in v.items()}


def kraus(v: Vector, orientation: int, high: bool) -> Vector:
    out: Vector = {}
    for (mask, k), value in v.items():
        for changed, sign in ((False, 1), (True, -1 if high else 1)):
            target = tuple(k[i] + (orientation if changed else 0) for i in range(4))
            phase = sp.exp(sp.I * sp.pi * sum(target) / 4) if high else 1
            add(out, (mask, target), value * sign * phase / (2 * sp.sqrt(2)))
    return out


def electric(v: Vector) -> Any:
    return simplify(sum(64 * sum(k_i * k_i for k_i in k) * sp.conjugate(value) * value
                        for (_, k), value in v.items()))


def hop(v: Vector, coefficient: Any, reverse: bool = False) -> Vector:
    out: Vector = {}
    for (mask, k), value in v.items():
        moved = car(mask, 6 if reverse else 2, 2 if reverse else 6)
        if moved is not None:
            target, sign = moved
            flow = (k[0] + (-1 if reverse else 1), k[1], k[2], k[3])
            add(out, (target, flow), coefficient * sign * value)
    return out


def hopping_energy(v: Vector, coefficient: Any) -> Any:
    return simplify(inner(v, hop(v, coefficient)) + inner(v, hop(v, coefficient, True)))


def gauss(v: Vector, base_mask: int) -> None:
    base = [(base_mask >> (4 * site) & 15).bit_count() for site in range(4)]
    for mask, k in v:
        rho = [(mask >> (4 * site) & 15).bit_count() - base[site] for site in range(4)]
        assert [k[edge] - k[(edge - 1) % 4] for edge in range(4)] == rho
        assert mask.bit_count() == 8


def uniform_local_pair(cfg: dict[str, Any]) -> dict[str, Any]:
    w = Fraction(0)
    for load, mass in ((1, cfg['active_m']), (63, cfg['auxiliary_m'])):
        _, paths = scalar_paths(RotationBand(a=cfg['a'], m=mass, r=cfg['light_r'], v=cfg['light_v'], improved=True))
        for displacement, coefficients in paths:
            length = sum(abs(int(v)) for v in displacement)
            w += load * length * sum((abs(Fraction(float(z.real))) + abs(Fraction(float(z.imag)))
                                      for z in coefficients), Fraction(0))
    assert w == Fraction(46425051219964815, 8796093022208)
    # Accepted high-only electric reset gives <1+16 Ec²> <= 1025+2048W.
    # This is a fixed-source actual-ground corollary, not a small-q bound.
    t_upper = Fraction(64) + 128 * w
    m1 = Fraction(256)
    m2_upper = 16384 * t_upper + 65536
    m0_lower = 4 / (t_upper + 4)
    band_upper = 128 * (t_upper + 4)
    band_mass_lower = 2 / (t_upper + 4)
    assert m1 * m1 / m2_upper == m0_lower
    assert m0_lower - m1 / band_upper == band_mass_lower
    # Exact NONNATIVE UV atomic and atomless measures have these same moments.
    # A local covariance lower still does not place any weight at energy O(q).
    uv_energy = m2_upper / m1
    uv_mass = m1 / uv_energy
    uniform_center = Fraction(12, 13) * uv_energy
    uniform_mass = m1 / uniform_center
    assert uv_mass == m0_lower and uv_mass < 1
    assert uniform_mass * uniform_center == m1
    assert uniform_mass * Fraction(13, 12) * uniform_center ** 2 == m2_upper
    assert uniform_mass < 1 and uniform_center / 2 > 1
    return {'W_native_all_literal_roles': str(w), 'Ec_second_moment_upper': str(t_upper),
            'pair_first_moment': str(m1), 'pair_second_moment_upper': str(m2_upper),
            'wholeP0_pair_covariance_lower': str(m0_lower), 'finite_positive_band_upper': str(band_upper),
            'finite_positive_band_mass_lower': str(band_mass_lower),
            'scope': 'Conditional all even N>=4 translation-averaged global-min family and SAME full-history fixed local source; not measured ground or pointwise momentum.',
            'nonnative_UV_same_moment_countermodels': {'atomic_energy': str(uv_energy), 'atomic_mass': str(uv_mass),
                                                       'atomless_uniform_center': str(uniform_center), 'atomless_mass': str(uniform_mass)},
            'positive_IR_order_or_atom_from_local_lower': False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    request: dict[str, Any] = json.loads(Path(args.request).read_text())
    before = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in request['pins']}
    assert before == request['pins']
    b = sp.symbols('B', real=True)
    functions = [(1 + sp.exp(sp.I * orientation * b)) / (2 * sp.sqrt(2)) for orientation in (1, -1)]
    functions += [(1 - sp.exp(sp.I * orientation * b)) / (2 * sp.sqrt(2)) for orientation in (1, -1)]
    completeness = simplify(sum(sp.conjugate(f) * f for f in functions))
    connection = simplify(sum(sp.conjugate(f) * sp.diff(f, b) for f in functions))
    gradient = simplify(sum(sp.conjugate(sp.diff(f, b)) * sp.diff(f, b) for f in functions))
    assert completeness == 1 and connection == 0 and gradient == sp.Rational(1, 2)
    assert 64 * 4 * gradient == 128
    cfg = json.loads((ROOT / request['configuration']).read_text())['input_descriptor']['original_micro_config_UNCHANGED']
    assert cfg['auxiliary_count'] == 63 and cfg['a'] == 1 / 64
    _, paths = scalar_paths(RotationBand(a=cfg['a'], m=cfg['active_m'], r=cfg['light_r'], v=cfg['light_v'], improved=True))
    raw = next(coefficients[0].real for displacement, coefficients in paths if list(displacement) == [1, 0, 0])
    # Gamma0=diag(+,+,-,-); selected spin2 diagonal native path coefficient.
    binary_coefficient = -Fraction(float(raw))
    coefficient = sp.Rational(binary_coefficient.numerator, binary_coefficient.denominator)
    pairs = [(0, 1), (2, 3), (0, 1), (2, 3)]
    base_mask = sum(1 << (4 * site + spin) for site, spins in enumerate(pairs) for spin in spins)
    moved = car(base_mask, 2, 6)
    assert moved is not None
    moved_mask, parity = moved
    rows = []
    for sign in (-1, 1):
        source: Vector = {(base_mask, (0, 0, 0, 0)): 1 / sp.sqrt(2),
                          (moved_mask, (1, 0, 0, 0)): sign / sp.sqrt(2)}
        branches = [kraus(source, orientation, high) for high in (False, True) for orientation in (1, -1)]
        gauss(source, base_mask)
        for branch in branches:
            gauss(branch, base_mask)
        assert norm(source) == 1 and simplify(sum(norm(v) for v in branches)) == 1
        delta_e = simplify(sum(electric(v) for v in branches) - electric(source))
        delta_t = simplify(sum(hopping_energy(v, coefficient) for v in branches) - hopping_energy(source, coefficient))
        target_t = sign * parity * coefficient * (sp.sqrt(2) / 2 - 1) / 2
        assert simplify(delta_t - target_t) == 0 and delta_e == 128
        incoming_cos = simplify(sp.re(inner(source, shift(source, (1, 1, 1, 1)))))
        reset_cos = simplify(sum(sp.re(inner(v, shift(v, (1, 1, 1, 1)))) for v in branches))
        assert incoming_cos == 0 and reset_cos == sp.Rational(1, 2)
        rows.append({'coherence_sign': sign, 'CAR_parity': parity, 'fullE_increment_exact': str(delta_e),
                     'charged_hop_increment_exact': str(delta_t), 'cos_before': str(incoming_cos), 'cos_after': str(reset_cos),
                     'physical_fullPhi_SU4_quantum_ground_source': False})
    assert simplify(sp.sympify(rows[0]['charged_hop_increment_exact']) + sp.sympify(rows[1]['charged_hop_increment_exact'])) == 0
    assert rows[0]['charged_hop_increment_exact'] != '0'
    receipt = {'kind': 'EXACT_PERIODIC_SOFT_MAGNETIC_KRAUS_NATIVE_CAR_FULL_E_METHOD_NOT_ORDER_OR_PHASE',
               'kraus': 'K0±=(1+U±1)/(2sqrt2), K1±=V(1-U±1)/(2sqrt2); V=exp(iπEc/4), c²=4',
               'completeness': str(completeness), 'connection_current': str(connection), 'gradient_budget': str(gradient),
               'full_form_electric_increment': '128 I', 'rows': rows,
               'operator_action': 'U→(1+U²)/2; charged Tt→[cos²(B/2)+exp(-iπ(c.t)/4)sin²(B/2)]Tt',
               'ground_variational_consequence': '128+<Δcharged hopping>ground≥0, not a defect probability/order lower without signed carrier correlation',
               'once_shared_fullE_not_64_independent_copies': True, 'wrong_independent_reset_cost': 64 * 128,
               'opposite_native_CAR_signs_same_flux_statistics_rejected_universal_matter_gain': True,
               'Phi_Haar_SU4_neutral_unchanged_analytical_multiplier_commutation': True,
               'uniform_local_pair_corollary_pending_independent_review': uniform_local_pair(cfg),
               'scope': 'Actual canonical CAR/native path coefficient and affine integer-flow POINT factors. Full-form identity analytic on accepted common physical domain; no full physical quantum source/ground computed.',
               'physical1_ground_order_phase_or_atom': False, 'native_ground_fullH_backend': False,
               'source_input_sha256_before': before,
               'source_input_sha256_after': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in request['pins']},
               'elapsed_seconds': time.monotonic() - started}
    assert receipt['source_input_sha256_after'] == before
    with Path(args.output).open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'fullE_increment': '128', 'opposite_matter_signs': True, 'ground_or_phase': False}))


if __name__ == '__main__':
    main()
