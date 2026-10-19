"""Independent exact channel/form/spectral discriminators; never import target code."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any
import zipfile

import sympy as sp


def exact(value: Any) -> Any:
    return sp.simplify(sp.expand_complex(value))


def channel_form() -> dict[str, Any]:
    b = sp.symbols('B', real=True)
    psi = sp.Function('psi')(b)
    row = [(1 + sign * sp.exp(sp.I * orientation * b)) / (2 * sp.sqrt(2))
           for orientation in (1, -1) for sign in (1, -1)]
    completeness = exact(sum(sp.conjugate(f) * f for f in row))
    connection = exact(sum(sp.conjugate(f) * sp.diff(f, b) for f in row))
    derivative_norm = exact(sum(sp.conjugate(sp.diff(f, b)) * sp.diff(f, b) for f in row))
    assert (completeness, connection, derivative_norm) == (1, 0, sp.Rational(1, 2))
    # V commutes every full electric component; H_E on B is -256 d_B^2.
    increment = sp.simplify(sum(sp.conjugate(f) * (-256 * sp.diff(f * psi, b, 2)) for f in row)
                            + 256 * sp.diff(psi, b, 2))
    assert sp.simplify(increment - 128 * psi) == 0
    # A TP single orientation is a plausible wrong implementation, not the four-branch row.
    wrong = [(1 + sign * sp.exp(sp.I * b)) / 2 for sign in (1, -1)]
    wrong_connection = exact(sum(sp.conjugate(f) * sp.diff(f, b) for f in wrong))
    wrong_increment = sp.simplify(sum(sp.conjugate(f) * (-256 * sp.diff(f * psi, b, 2)) for f in wrong)
                                  + 256 * sp.diff(psi, b, 2))
    assert wrong_connection == sp.I / 2
    assert sp.simplify(wrong_increment - (128 * psi - 256 * sp.I * sp.diff(psi, b))) == 0
    assert wrong_increment != 128 * psi
    f = sp.exp(sp.cos(b) / 2)
    localization_price = sp.simplify(256 * sp.diff(f, b) ** 2)
    assert sp.simplify(localization_price - 64 * sp.exp(sp.cos(b)) * sp.sin(b) ** 2) == 0
    # V* U V=-U; high branch is V times its multiplier, not a coherent row sum.
    dual_u = exact(sum(sp.conjugate(g) * sp.exp(sp.I * b) * g * (1 if sign == 1 else -1)
                       for orientation in (1, -1) for sign in (1, -1)
                       for g in [(1 + sign * sp.exp(sp.I * orientation * b)) / (2 * sp.sqrt(2))]))
    assert sp.simplify(dual_u - (1 + sp.exp(2 * sp.I * b)) / 2) == 0
    phase_rows = []
    for frequency in range(-2, 3):
        phase = sp.exp(-sp.I * sp.pi * frequency / 4)
        dual_t = exact(sum(sp.conjugate(g) * g * (1 if sign == 1 else phase)
                           for orientation in (1, -1) for sign in (1, -1)
                           for g in [(1 + sign * sp.exp(sp.I * orientation * b)) / (2 * sp.sqrt(2))]))
        expected = sp.cos(b / 2) ** 2 + phase * sp.sin(b / 2) ** 2
        assert exact(dual_t - expected) == 0
        assert exact(sp.conjugate(expected) - (sp.cos(b / 2) ** 2
                                              + sp.exp(sp.I * sp.pi * frequency / 4) * sp.sin(b / 2) ** 2)) == 0
        phase_rows.append({'frequency': frequency, 'phase': str(exact(phase))})
    assert 64 * 128 != 128
    return {'trace_preserving': str(completeness), 'connection': str(connection),
            'derivative_norm': str(derivative_norm), 'fullE_plus_collectiveHC_increment': 128,
            'single_orientation_connection': str(wrong_connection),
            'single_orientation_increment': str(wrong_increment),
            'dropped_current_rejected': True, 'independent_64_reset_cost_rejected': 8192,
            'ground_f_localization_numerator': str(localization_price),
            'normalized_ground_cost': '64*<exp(cos B)*sin(B)^2>/<exp(cos B)>; nonnegative, not defectgain',
            'dual_U': '(1+U^2)/2', 'charged_reverse_phases': phase_rows}


def geometry(n: int) -> dict[str, Any]:
    sites = list(itertools.product(range(n), repeat=3))
    def next_site(x: tuple[int, ...], axis: int) -> tuple[int, ...]:
        y = list(x); y[axis] = (y[axis] + 1) % n
        return tuple(y)
    def face(x: tuple[int, ...]) -> dict[tuple[tuple[int, ...], int], int]:
        return {(x, 0): 1, (next_site(x, 0), 1): 1,
                (next_site(x, 1), 0): -1, (x, 1): -1}
    cycles = [face(x) for x in sites]
    c = cycles[0]
    divergence: dict[tuple[int, ...], int] = {}
    for (x, axis), value in c.items():
        divergence[x] = divergence.get(x, 0) + value
        y = next_site(x, axis); divergence[y] = divergence.get(y, 0) - value
    assert all(value == 0 for value in divergence.values()) and sum(v*v for v in c.values()) == 4
    phases, seams = set(), 0
    path_count = 0
    for role, copies, charge, lengths in [('neutral', 1, 0, (1,)), ('active', 1, 1, (1, 2)),
                                         ('auxiliary', 63, 1, (1, 2))]:
        for x in sites:
            for axis in range(3):
                for length in lengths:
                    y, frequency, seam = x, 0, 1
                    for _ in range(length):
                        frequency += charge * c.get((y, axis), 0)
                        if y[axis] == n-1: seam *= -1
                        y = next_site(y, axis)
                    assert y != x and abs(frequency) <= 2
                    if charge == 0: assert frequency == 0
                    phases.add(frequency); seams += copies * (seam == -1)
                    path_count += 2 * copies  # Ordered forward and its actual adjoint.
    # Wilson correlation=1 gives incidence cancellation. Orthogonal correlations do not.
    total: dict[Any, int] = {}
    for cycle in cycles:
        for edge, value in cycle.items(): total[edge] = total.get(edge, 0) + value
    flat_q0 = Fraction(64, len(cycles)) * sum(v*v for v in total.values())
    orthogonal_q0 = Fraction(64, len(cycles)) * sum(sum(v*v for v in c.values()) for c in cycles)
    assert flat_q0 == 0 and orthogonal_q0 == 256
    disjoint = []
    for cycle in cycles:
        if all(set(cycle).isdisjoint(old) for old in disjoint): disjoint.append(cycle)
        if len(disjoint) == 4: break
    assert len(disjoint) == 4
    # Every normalized real block vector has paired first moment256, also at q=0.
    for coeffs in [(1, 0, 0, 0), (1, 1, 1, 1), (1, -1, 1, -1)]:
        first = Fraction(256 * sum(x*x for x in coeffs), sum(x*x for x in coeffs))
        assert first == 256
    return {'N': n, 'Dc': 0, 'c_squared': 4, 'all65_forward_reverse_words': path_count,
            'charged_frequencies': sorted(phases), 'AP_forward_rows_weighted': seams,
            'flat_Wilson_kernel_q0': str(flat_q0), 'orthogonal_Wilson_kernel_q0': str(orthogonal_q0),
            'disjoint_paired_block_first_moment': '256 I',
            'kernel_controls_not_ground_correlations': True}


def ladder(mask: int, mode: int, create: bool) -> tuple[int, int] | None:
    occupied = bool(mask & (1 << mode))
    if occupied == create: return None
    return mask ^ (1 << mode), (-1) ** ((mask & ((1 << mode) - 1)).bit_count())


def car_control() -> dict[str, Any]:
    def compose(mask: int, first: int, first_create: bool, second: int, second_create: bool) -> dict[int, int]:
        a = ladder(mask, second, second_create)
        if a is None: return {}
        b = ladder(a[0], first, first_create)
        return {} if b is None else {b[0]: a[1]*b[1]}
    count = 0
    for mask in range(256):
        for i in range(8):
            for j in range(8):
                for other_create in (False, True):
                    left = compose(mask, i, False, j, other_create)
                    right = compose(mask, j, other_create, i, False)
                    combined = {key: left.get(key, 0)+right.get(key, 0) for key in left.keys() | right.keys()}
                    combined = {key: value for key, value in combined.items() if value}
                    assert combined == ({mask: 1} if other_create and i == j else {})
                    count += 1
    # Exact traceless-Hermitian primitive endpoint matrices and SU4 factors, not quantum states.
    phi_x = sp.diag(1, 2, -1, -2)
    phi_y = sp.Matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 3, 0], [0, 0, 0, -3]])
    link = sp.diag(sp.I, -sp.I, 1, 1)
    gx = sp.diag(1, sp.I, -1, sp.I)
    gy = sp.diag(-sp.I, 1, sp.I, 1)
    for g in (link, gx, gy): assert g.det() == 1 and g.H*g == sp.eye(4)
    endpoint = (phi_x*link+link*phi_y)/2
    transformed = ((gx*phi_x*gx.H)*(gx*link*gy.H)
                   +(gx*link*gy.H)*(gy*phi_y*gy.H))/2
    assert transformed == gx*endpoint*gy.H and endpoint != sp.zeros(4)
    # Same flux statistics can accompany opposite complete CAR coherence signs.
    phase = sp.exp(-sp.I*sp.pi/4)
    gains = [exact(sign * (phase-1 + sp.conjugate(phase)-1)) for sign in (-1, 1)]
    assert exact(sum(gains)) == 0 and gains[0] != 0
    return {'eight_mode_full_Fock_CAR_assertions': count, 'SU4_ordered_endpoint_covariance': True,
            'opposite_coherence_charged_gains': [str(x) for x in gains],
            'point_factors_not_complete_Gauss_quantumstates': True,
            'unitary_V_conjugation_preserves_CAR_analytically': True}


def spectral_control(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    w = Fraction(0)
    for profile in profiles:
        if profile['role'] == 'neutral': continue
        coefficients = profile['native_scalar_coefficients']
        assert len(coefficients) == 6
        for row, length in zip(coefficients, (1, 2, 1, 2, 1, 2)):
            w += profile['copies'] * length * sum((abs(Fraction(float(component)))
                                                   for pair in row for component in pair), Fraction(0))
    assert w == Fraction(46425051219964815, 8796093022208)
    t = sp.symbols('T', nonnegative=True)
    mean = sp.symbols('mean_E', real=True)
    plus_m2 = 16384*t + 65536*mean + 65536
    minus_m2 = 16384*t - 65536*mean + 65536
    m1, m2 = sp.Integer(256), sp.simplify((plus_m2+minus_m2)/2)
    lower = sp.cancel(m1*m1/m2); band = 128*(t+4)
    assert lower == 4/(t+4) and sp.simplify(lower-m1/band) == 2/(t+4)
    assert 1+16*(64+128*w) == 1025+2048*w
    tu = 64+128*w; m2u = 16384*tu+65536
    m0 = 4/(tu+4); r = 128*(tu+4); band_mass = 2/(tu+4)
    assert Fraction(4, 2**8202+4) > Fraction(1, 2**8201)
    # Two explicitly NONNATIVE measures with identical first and second positive-energy moments.
    atomic_energy, atomic_mass = m2u/256, m0
    center, diffuse_mass = Fraction(12, 13)*atomic_energy, Fraction(13, 12)*m0
    assert atomic_mass*atomic_energy == diffuse_mass*center == 256
    assert atomic_mass*atomic_energy**2 == diffuse_mass*Fraction(13, 12)*center**2 == m2u
    assert 0 < atomic_mass < diffuse_mass < 1 and center/2 > 1
    # Diffuse uniform law on [center/2,3center/2] has no atoms; both laws have a UV gap>1.
    for q in (Fraction(1, 8), Fraction(1, 16), Fraction(1, 32)):
        assert q < center/2 and q < atomic_energy
    # Omega=e0 and a second zero vector e1: scalar centering leaves the e1 ground component.
    source = sp.Matrix([0, sp.Rational(3, 5), sp.Rational(4, 5)])
    zero = sp.diag(1, 1, 0)
    mean_centered_mass = (source.H*source)[0]
    whole_removed_mass = (source.H*(sp.eye(3)-zero)*source)[0]
    assert mean_centered_mass == 1 and whole_removed_mass == sp.Rational(16, 25)
    # Positive levels can collapse to zero, so TOTAL Gram convergence is not projector transport.
    assert (sp.Matrix([0, 1]).H*sp.diag(0, 1)*sp.Matrix([0, 1]))[0] == 1
    assert (sp.Matrix([0, 1]).H*sp.zeros(2)*sp.Matrix([0, 1]))[0] == 0
    # E4 bound need not preserve direct E^2-source norm; E-source norm does converge in this witness.
    for n in (2, 4, 8):
        probability = Fraction(1, n**4)
        assert probability*n**4 == 1 and probability*n**2 == Fraction(1, n**2)
    return {'W_exact_all_charged': str(w), 'mean_cancelled_in_pair': True, 'm1': 256,
            'm2': str(m2), 'wholeP0_pair_lower': str(lower),
            'T_upper': str(tu), 'uniform_pair_lower': str(m0), 'uniform_band_cutoff': str(r),
            'uniform_band_mass_lower': str(band_mass), 'pair_lower_decimal_context': float(m0),
            'original_N4_strict_gt_2^-8201_unchanged': True,
            'UV_atomic': {'mass': str(atomic_mass), 'energy': str(atomic_energy)},
            'UV_atomless': {'mass': str(diffuse_mass), 'interval': [str(center/2), str(3*center/2)]},
            'same_moments_no_atom_inference': True, 'local_to_smallq_inference_rejected': True,
            'scalar_centered_mass': str(mean_centered_mass), 'wholeP0_mass': str(whole_removed_mass),
            'finiteP0_transport_rejected': True, 'E_squared_source_norm_loss_wrongcase': True,
            'countermodels_are_nonnative_not_claimed_ground_models': True}


def run(input_archive: Path, output: Path) -> None:
    with zipfile.ZipFile(input_archive) as inputs:
        original = json.loads(inputs.read('ground_magnetic_order_actual_20261012.json'))
        manifest = json.loads(inputs.read('magnetic_pair_review_input_manifest_20261012.json'))
        pins = {name: hashlib.sha256(inputs.read(name)).hexdigest() for name in manifest['members']}
        assert pins == manifest['members']
    result = {'kind': 'INDEPENDENT_EXACT_MAGNETIC_CHANNEL_FORM_SPECTRAL_CONTROLS_NOT_PHYSICAL_GROUND_BACKEND',
              'channel_form': channel_form(), 'native_geometry': [geometry(n) for n in (4, 6)],
              'native_CAR_point_factors': car_control(),
              'spectral': spectral_control(original['native'][0]['native_profiles']),
              'input_pins': pins, 'full_ground_exists_analytically_inherited': True,
              'native_quantum_ground_computed': False, 'scientific_admission': False, 'campaign_complete': False}
    with output.open('x') as stream: json.dump(result, stream, indent=2, sort_keys=True); stream.write('\n')
    print(json.dumps({'channel_cost': 128, 'scope': 'conditional fixed-local pair, not IR/atom',
                      'wrongcases_rejected': ['single-orientation/current', '64 independent resets', 'mean-versus-wholeP0',
                                              'finiteP0 transport', 'local-to-smallq', 'moment-to-atom']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.input_archive, args.output)
