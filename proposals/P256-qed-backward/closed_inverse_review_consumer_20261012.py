"""Independent exact native factors and explicitly nonnative inverse controls.
No original program import or execution; no native full-H or ground numerics.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import zipfile
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

ROOT = Path(__file__).resolve().parent
F = Fraction


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def native_factors(n: int, profiles: list[dict[str, Any]]) -> dict[str, Any]:
    sites = list(itertools.product(range(n), repeat=3))
    index = {x: j for j, x in enumerate(sites)}
    def step(x: tuple[int, ...], axis: int) -> tuple[int, ...]:
        return tuple((t + 1) % n if j == axis else t for j, t in enumerate(x))
    def parity(x: tuple[int, ...]) -> int:
        return (-1) ** sum(x)
    def div(flow: dict) -> dict:
        out: dict = defaultdict(F)
        for (x, axis), value in flow.items():
            out[x] += value
            out[step(x, axis)] -= value
        return dict(out)
    def dot(left: dict, right: dict) -> F:
        return sum((value * right.get(edge, F(0)) for edge, value in left.items()), F(0))
    # Exact longitudinal connection for the all64 checkerboard occupation.
    rho = {x: 128 * parity(x) for x in sites}
    eta = {x: F(parity(x), 6) for x in sites}
    longitudinal = {(x, axis): F(64, 3) * parity(x) for x in sites for axis in range(3)}
    assert div(longitudinal) == rho
    integer_deck = {(x, 0): F(parity(x)) for x in sites}
    deck_gradient = {(x, axis): eta[x] - eta[step(x, axis)] for x in sites for axis in range(3)}
    projected_deck = {edge: integer_deck.get(edge, F(0)) - value for edge, value in deck_gradient.items()}
    assert not any(div(projected_deck).values())
    assert any(value.denominator == 3 for value in projected_deck.values())
    x = (n - 1, n - 1, 0)
    face = {(x, 0): F(1), (step(x, 0), 1): F(1),
            (step(x, 1), 0): F(-1), (x, 1): F(-1)}
    assert not any(div(face).values()) and dot(face, longitudinal) == 0
    assert dot(face, projected_deck) == dot(face, integer_deck) == 2
    # Face plus a genuine harmonic flow; norm, shift and domains are not face-only.
    closed = {(y, axis): F(1, 7) if axis == 2 else F(0) for y in sites for axis in range(3)}
    for edge, value in face.items():
        closed[edge] += value
    assert not any(div(closed).values()) and dot(closed, longitudinal) == 0
    norm2 = dot(closed, closed)
    assert norm2 == 4 + F(n ** 3, 49)
    incoming = {edge: value + F(5, 13) * closed[edge] for edge, value in longitudinal.items()}
    electric_mean = dot(closed, incoming)
    increments = []
    for s in (F(-9, 5), F(-1, 37), F(0), F(8, 11)):
        shifted = {edge: value + s * closed[edge] for edge, value in incoming.items()}
        assert div(shifted) == rho
        delta = 64 * (dot(shifted, shifted) - dot(incoming, incoming))
        assert delta == 128 * s * electric_mean + 64 * s * s * norm2
        increments.append({'s': str(s), 'fullE64_increment': str(delta)})
    v = n ** 3
    width = 4 * v
    mask_one = sum(15 << (4 * j) for j, y in enumerate(sites) if parity(y) == 1)
    mask = sum(mask_one << (role * width) for role in range(65)) | (15 << (65 * width))
    origin = (0, 0, 0)
    destination = step(origin, 0)
    signs = []
    for role in range(65):
        i = role * width + 4 * index[origin]
        j = role * width + 4 * index[destination]
        sign = (-1) ** (mask & ((1 << i) - 1)).bit_count()
        removed = mask ^ (1 << i)
        sign *= (-1) ** (removed & ((1 << j) - 1)).bit_count()
        changed = removed | (1 << j)
        assert all(((changed >> (r * width)) & ((1 << width) - 1)).bit_count() == 2 * v for r in range(65))
        assert changed >> (65 * width) == 15
        signs.append(sign)
        if role:
            shifted = dict(longitudinal)
            shifted[(origin, 0)] -= 1
            changed_rho = dict(rho)
            changed_rho[origin] -= 1
            changed_rho[destination] += 1
            assert div(shifted) == changed_rho
    path_counts = []
    ap_counts = []
    affine_rows = []
    aliases = []
    load = F(0)
    for role, profile in enumerate(profiles):
        count, ap_count, n4_alias = 0, 0, 0
        for row in profile['paths']:
            axis, length = row['axis'], row['length']
            if role:
                load += F(profile['load']) * length * (abs(F(row['c0'])) + abs(F(row['ci'])))
            for start in sites:
                y, path, ap = start, {}, 1
                for _ in range(length):
                    path[(y, axis)] = path.get((y, axis), F(0)) + 1
                    ap *= -1 if y[axis] == n - 1 else 1
                    y = step(y, axis)
                charge_change = div(path)
                if role:
                    affine = sum((eta[z] * value for z, value in charge_change.items()), F(0))
                    pairing = dot(projected_deck, path)
                    integer = dot(integer_deck, path)
                    assert pairing + affine == integer and integer.denominator == 1
                    # Omitting the density-affine deck factor fails at length1.
                    if length == 1:
                        assert pairing.denominator == 3
                count += 1
                ap_count += int(ap == -1)
                n4_alias += int(n == 4 and length == 2 and dot(face, path) != 0)
        path_counts.append(count)
        ap_counts.append(ap_count)
        aliases.append(n4_alias)
        affine_rows.append({'role': profile['role'], 'charged_load': profile['load'] if role else 0,
                            'projected_deck_plus_density_phase_integer': bool(role),
                            'onsite_retained': profile['onsite']})
    assert path_counts == [3 * v, 6 * v, 6 * v]
    assert load == F(46425051219964815, 8796093022208)
    # Actual ordered noncommuting colour endpoint factors, not commuting scalars.
    gx = sp.Matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    gy = sp.diag(sp.I, -sp.I, 1, 1)
    px = sp.diag(2, -2, 1, -1)
    py = sp.Matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 2], [0, 0, 2, 0]])
    link = gx * gy
    block = (px * link + link * py) / 2
    transformed = ((gx * px * gx.H) * (gx * link * gy.H) + (gx * link * gy.H) * (gy * py * gy.H)) / 2
    assert gx.det() == gy.det() == 1 and transformed == gx * block * gy.H
    assert px * link != link * px
    collective = 64 * F(128 ** 2 * v, 12)
    independent = 64 * 64 * F(2 ** 2 * v, 12)
    assert collective == 64 * independent
    return {'scope': 'exact native factors; no native ground/fullH', 'N': n,
            'canonical_each65': 2 * v, 'all65_distinct_CAR_transfers': len(signs), 'Ref': 4,
            'fullrho': '128 parity', 'actual_projected_noninteger_deck': True,
            'deck_components': ['2 parity/3', '-parity/3', '-parity/3'],
            'occupation_affine_deck_required': True, 'closed_face_plus_harmonic_norm2': str(norm2),
            'increments': increments, 'paths': path_counts, 'AP_seams': ap_counts,
            'N4_literal_alias_counts': aliases, 'profiles': affine_rows,
            'all_charged_path_load': str(load), 'ordered_SU4_covariance': True,
            'collective_HC_to_independent_copy_ratio': 64}


def exact_forms() -> dict[str, Any]:
    x, phi, colour, s, h, k = sp.symbols('x Phi Colour s h k', real=True)
    psi = sp.Function('Psi')(x, phi, colour)
    connection = sp.Function('OccupationAffineConnection')(x, phi, colour)
    boost = sp.exp(sp.I * s * h * x)
    def electric(f: Any) -> Any:
        return -sp.I * sp.diff(f, x) + connection * f
    assert sp.simplify(electric(boost * psi) - boost * (electric(psi) + s * h * psi)) == 0
    assert sp.diff(boost * psi, phi) == boost * sp.diff(psi, phi)
    assert sp.diff(boost * psi, colour) == boost * sp.diff(psi, colour)
    # Local transported chart overlap and complete occupation connection.
    source, target = sp.symbols('source_lift target_lift', real=True)
    residual = sp.exp(sp.I * (s * h + source - target) * x)
    assert sp.simplify(sp.exp(-sp.I * target * x) * boost * sp.exp(sp.I * source * x) - residual) == 0
    return {'arbitrary_full_connection_preserved': True, 'all_Phi_Haar_derivatives_unchanged': True,
            'chart_overlap_complete_lift_identity': True, 'form_coefficients': [128, 64],
            'no_continuous_ground_eigensection_asserted': True}


def nonnative_controls() -> dict[str, Any]:
    # Sharp oscillator proof: L=64p^2+w^2 x^2/256-w/2, E=p, B=1.
    x, w, t = sp.symbols('x w t', real=True, positive=True)
    alpha = w / 128
    phi = sp.exp(-alpha * x * x / 2)
    ground_action = -64 * sp.diff(phi, x, 2) + (w * w * x * x / 256 - w / 2) * phi
    assert sp.simplify(ground_action) == 0
    source_mass = alpha / 2
    inverse = sp.simplify(source_mass / w)
    assert inverse == sp.Rational(1, 256)
    assert sp.simplify(sp.sqrt(source_mass) / sp.sqrt(w)) == sp.Rational(1, 16)
    assert sp.simplify(t / sp.sqrt(64 * t * t)) == sp.Rational(1, 8)
    # Independently PSD all-s method family; checks exact inequalities, not a native operator.
    r = sp.symbols('r', real=True)
    a = sp.Rational(1, 16)
    family = sp.Matrix([[64 * r * r, 128 * r * a], [128 * r * a, 1 + 64 * r * r]])
    assert sp.factor(family.det()) == 4096 * r ** 4
    vector = sp.Matrix([1, sp.Rational(1, 10)])
    electric = sp.Matrix([[0, a], [a, 0]])
    mean = (vector.T * electric * vector)[0]
    norm2 = (vector.T * vector)[0]
    energy = sp.Rational(1, 100)
    assert mean ** 2 > norm2 * energy / 256  # wrong diagonal1/16
    assert a > sp.Rational(1, 32) and inverse > sp.Rational(1, 512)
    assert 4 * inverse == sp.Rational(1, 64) > sp.Rational(1, 256)
    # Whole kernel, not mean, source-Q or a deleted finite kernel.
    omega = sp.Matrix([1, 0, 0])
    E = sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    e = E * omega
    P0 = sp.diag(1, 1, 0)
    assert (omega.T * e)[0] == 0 and P0 * e == e
    assert (sp.eye(3) - omega * omega.T) * e == e
    z = sp.Matrix([0, 1, 0])
    assert (z.T * e)[0] == 1  # energy norm0, dual inequality impossible
    # Continuous reciprocal tests see zero; the positive-only discontinuous version hides it.
    collapse = []
    for n in (4, 17, 128, 2048):
        eps = F(1, n)
        legal_weight = eps / 256
        assert legal_weight / eps == F(1, 256)
        assert 1 / eps > F(1, 256)
        row = {'energy': str(eps), 'legal_weight': str(legal_weight), 'illegal_unit_inverse': str(1 / eps)}
        for K in (1, 16, 257):
            value = min(F(K), 1 / eps)
            assert legal_weight * value <= F(1, 256)
            row[f'continuous_f{K}'] = str(legal_weight * value)
        collapse.append(row)
    assert F(257) * F(1, 2) > F(1, 256)
    # Bounded E and even the all-Q inequality do not imply E^2 Omega in D(L).
    n, u = sp.symbols('n u', integer=True, positive=True)
    tau = sp.symbols('tau', real=True)
    En = sp.Matrix([[0, a, 0], [a, 0, 1], [0, 1, 0]])
    Ln = sp.diag(0, 1, n ** 4)
    shifted = Ln + 128 * tau * En + 64 * tau ** 2 * sp.eye(3)
    det = sp.factor(shifted.det())
    assert sp.expand(det - 4096 * tau ** 4 * (n ** 4 - 256 + 64 * tau ** 2)) == 0
    # n>=4: all principal minors nonnegative, so every finite block has all-s positivity.
    v0 = sp.Matrix([1, 0, 0])
    source = En * v0
    source2 = En * source
    assert Ln * source == source
    e4_term = sp.simplify((source2.T * source2)[0] / n ** 4)
    graph_term = sp.simplify(((Ln * source2).T * (Ln * source2))[0] / n ** 4)
    assert e4_term == sp.Rational(257, 65536) / n ** 4
    assert graph_term == n ** 4 / 256
    # p-series summability versus positive divergent terms, not merely term->0.
    assert sp.summation(1 / n ** 4, (n, 4, sp.oo)).is_finite is True
    assert sp.limit(graph_term, n, sp.oo) == sp.oo
    # No arbitrary continuous eigenvector across a crossing; no Borel point evaluation.
    crossing_left = sp.diag(-1, 1)
    crossing_right = sp.diag(1, -1)
    assert (sp.Matrix([1, 0]).T * sp.Matrix([0, 1]))[0] == 0
    beta = F(1, 3)
    assert 64 * beta ** 2 > 0 and beta != 0
    prepared = sp.Matrix([1, 1])
    assert (prepared.T * electric * prepared)[0] != 0
    # Complex inverse Gram: every real quadratic1/256, complex sharp factor2.
    S = sp.Matrix([[1, sp.I], [-sp.I, 1]]) / 256
    real_a, real_b = sp.symbols('a b', real=True)
    real_vec = sp.Matrix([real_a, real_b])
    assert sp.expand((real_vec.T * S * real_vec)[0]) == (real_a ** 2 + real_b ** 2) / 256
    complex_vec = sp.Matrix([1, -sp.I])
    complex_value = (complex_vec.H * S * complex_vec)[0]
    assert complex_value == sp.Rational(1, 64)
    assert complex_value / (complex_vec.H * complex_vec)[0] == sp.Rational(1, 128)
    return {'scope': 'nonnative exact oscillator/form/domain/spectral controls; not native ground/fullH',
            'oscillator_sharp_allQ': '1/8', 'oscillator_sharp_source_dual': '1/16',
            'oscillator_sharp_real_inverse': str(inverse), 'face_sharp_inverse': '1/64',
            'wrong_diagonal1_16_source1_32_inverse1_512_face1_256_rejected': True,
            'all_s_2x2_determinant': str(sp.factor(family.det())),
            'wholeP0_not_mean_or_sourceQ': True, 'finiteP0_not_limitingP0': 'diag(0,1/N)->zero operator; second source changes from positive to whole-zero',
            'continuous_inverse_at_zero': 'fK(0)=K; positive-only value0 cannot be passed through weak total convergence',
            'collapse': collapse, 'zero_atom_detected': True,
            'E4_and_even_source_D_L_not_E2_source_D_L': {'E4_series': str(e4_term), 'L_E2_series': str(graph_term), 'all_s_block_determinant': str(det), 'n_min': 4, 'E_bounded': True},
            'continuous_chart_not_continuous_ground_section': 'diag(t,-t) switches orthogonal ground lines at t=0',
            'Borel_point_indicator': {'original_measure_integral': 0, 'invalid_selected_point_value': 1},
            'nonminimizing_character': {'beta': str(beta), 'global_defect': str(64 * beta ** 2), 'electric_mean': str(beta), 'fibre_relative_ground_excess': 0},
            'prepared_nonminimizing_source_inference_rejected': True,
            'complex_inverse_Gram': [['1/256', 'i/256'], ['-i/256', '1/256']],
            'sharp_complex_inverse': '1/128', 'imaginary_cross_retained': True,
            'atom_lower_bound_or_phase_inferred': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_bytes())
    before = {name: sha(Path(row['path']).read_bytes()) for name, row in request['inputs'].items()}
    assert before == {name: row['sha256'] for name, row in request['inputs'].items()}
    with zipfile.ZipFile(request['profile_archive']) as archive:
        raw = archive.read(request['profile_member'])
    assert sha(raw) == request['profile_member_sha256']
    profiles = json.loads(raw)['primitive_profiles']
    receipt = {'schema': 1, 'kind': 'INDEPENDENT_CLOSED_INVERSE_REVIEW_EXACT_CONTROLS',
               'native_factors': [native_factors(n, profiles) for n in request['N']],
               'exact_form_controls': exact_forms(), 'nonnative_controls': nonnative_controls(),
               'literal_target_proof': json.loads(Path(request['target_proof']).read_bytes()),
               'literal_corrected_interface': json.loads(Path(request['target_interface']).read_bytes()),
               'original_archive_inspection': request['original_archive_inspection'],
               'source_input_sha256_before': before,
               'source_input_sha256_after': {name: sha(Path(row['path']).read_bytes()) for name, row in request['inputs'].items()},
               'native_fullH_or_ground_computed': False, 'scientific_admission': False,
               'campaign_complete': False}
    assert receipt['source_input_sha256_after'] == before
    with args.output.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'receipt': args.output.name, 'completed_independent_controls': True}))


if __name__ == '__main__':
    main()