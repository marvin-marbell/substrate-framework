"""Independent exact native/source controls, never a complete ground backend."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import zipfile
from typing import Any, cast

import sympy as sp


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def simplify(value: Any) -> Any:
    return sp.simplify(sp.expand_complex(value))


def symbolic_channel() -> dict[str, Any]:
    D = sp.symbols('D', real=True)
    psi = sp.Function('psi')(D)
    rows = [(1 + sign*sp.exp(sp.I*s*D))/(2*sp.sqrt(2))
            for s in (1, -1) for sign in (1, -1)]
    tp = simplify(sum(sp.conjugate(c)*c for c in rows))
    connection = simplify(sum(sp.conjugate(c)*sp.diff(c, D) for c in rows))
    jets = simplify(sum(abs(sp.diff(c, D))**2 for c in rows))
    delta = simplify(sum(sp.conjugate(c)*(-640*sp.diff(c*psi, D, 2)) for c in rows)
                     + 640*sp.diff(psi, D, 2))
    assert tp == 1 and connection == 0 and jets == sp.Rational(1, 2)
    assert simplify(delta-320*psi) == 0
    one = [sp.sqrt(2)*c for c in rows[:2]]
    single_connection = simplify(sum(sp.conjugate(c)*sp.diff(c, D) for c in one))
    assert single_connection == sp.I/2
    # Dropping the conjugate orientation keeps TP but leaves a linear E_d term.
    single_delta = simplify(sum(sp.conjugate(c)*(-640*sp.diff(c*psi, D, 2)) for c in one)
                            + 640*sp.diff(psi, D, 2))
    assert simplify(single_delta-320*psi+640*sp.I*sp.diff(psi, D)) == 0
    alpha, nu = sp.symbols('alpha nu', real=True)
    phase = sp.exp(-sp.I*alpha*nu)
    dual = simplify(sum(sp.conjugate(c)*c*(1 if i % 2 == 0 else phase)
                        for i, c in enumerate(rows)))
    expected = sp.cos(D/2)**2+phase*sp.sin(D/2)**2
    assert simplify(dual-expected) == 0
    U = sp.exp(sp.I*D)
    assert simplify(U*(sp.cos(D/2)**2-sp.sin(D/2)**2)-(1+U**2)/2) == 0
    assert 320*64 == 20480 != 320
    return {'TP': str(tp), 'connection': str(connection), 'jet_square': str(jets),
            'shared_fullE_cost': 320, 'wrong_64_independent_cost': 20480,
            'single_orientation_connection': str(single_connection),
            'single_orientation_increment': '320I+640E_d',
            'path_dual': 'cos(D/2)^2+exp(-i alpha nu)sin(D/2)^2',
            'Wilson_dual': '(1+U_d^2)/2'}


def native_profiles(inputs: dict[str, bytes]) -> list[dict[str, Any]]:
    original = json.loads(inputs['ground_magnetic_order_actual_20261012.json'])
    profiles = original['native'][0]['native_profiles']
    descriptor = json.loads(inputs['induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json'])['input_descriptor']
    for filename, symbol in [('rotation_band.py', 'couplings'), ('local_orbital_stability.py', 'scalar_paths')]:
        tree = ast.parse(inputs[filename])
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == symbol]
        assert len(functions) == 1
    # Source shape must retain the endpoint half and its actual adjoint, not a sea projector.
    tree = ast.parse(inputs['local_orbital_stability.py'])
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'local_hamiltonian')
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'conj'
               for n in ast.walk(fn))
    assert [p['copies'] for p in profiles] == [1, 1, 63]
    for p, radial in zip(profiles, descriptor['exact_radial_species']):
        coefficients = p['native_scalar_coefficients']
        assert len(coefficients) == (3 if p['role'] == 'neutral' else 6)
        rows = coefficients if p['role'] == 'neutral' else coefficients[::2]
        for row in rows:
            assert Fraction(float(row[0][0])) == Fraction(radial['c0'])
            imaginary = [pair[1] for pair in row[1:] if pair[1] != 0]
            assert len(imaginary) == 1 and Fraction(float(imaginary[0])) == Fraction(radial['ci'])
        if p['role'] != 'neutral':
            assert all(Fraction(float(row[0][0])) == -Fraction(radial['c0'])/4
                       for row in coefficients[1::2])
    return profiles


def geometry(n: int, profiles: list[dict[str, Any]]) -> dict[str, Any]:
    def step(x: tuple[int, ...], axis: int) -> tuple[int, ...]:
        return tuple((v+1) % n if i == axis else v for i, v in enumerate(x))
    def face(x: tuple[int, ...]) -> Counter:
        return Counter({(x, 0): 1, (step(x, 0), 1): 1, (step(x, 1), 0): -1, (x, 1): -1})
    a, b = face((0, 0, 0)), face((1, 0, 0))
    d = {e: a[e]-b[e] for e in set(a) | set(b) if a[e] != b[e]}
    divergence = Counter()
    for (x, axis), value in d.items():
        divergence[x] += value
        divergence[step(x, axis)] -= value
    assert not any(divergence.values())
    assert sum(a[e]*b[e] for e in a) == -1 and sum(v*v for v in d.values()) == 10
    histograms = {}; weighted_allocation = Fraction(0); affected_aliases = 0; ap_affected = 0
    for p in profiles:
        hist = Counter(); charged = p['role'] != 'neutral'
        for row_index, coeff in enumerate(p['native_scalar_coefficients']):
            axis = row_index//2 if charged else row_index
            length = row_index % 2+1 if charged else 1
            magnitude = sum((abs(Fraction(float(v))) for pair in coeff for v in pair), Fraction(0))
            touched_starts = 0
            for x in itertools.product(range(n), repeat=3):
                y = x; path = []; ap = 1
                for _ in range(length):
                    path.append((y, axis))
                    if y[axis] == n-1: ap *= -1
                    y = step(y, axis)
                nu = sum(d.get(edge, 0) for edge in path) if charged else 0
                for orientation in (1, -1): hist[str(orientation*nu)] += p['copies']
                # Translation phases multiply the COMPLETE ordered coefficient, including AP.
                z = sp.symbols('z', nonzero=True)
                forward = ap*z**(-nu)
                reverse = ap*z**nu
                assert forward*reverse == 1
                touched = any(edge in d for edge in path)
                if charged and touched:
                    touched_starts += 1
                    ap_affected += (2*p['copies'] if ap == -1 else 0)
                    # Four-colour CAR <=4||M||; reverse + endpoint half =>4(r_x+r_y).
                    weighted_allocation += 8*p['copies']*magnitude
                    affected_aliases += int(length == 2)
            assert touched_starts <= 8*length
        histograms[p['role']] = dict(sorted(hist.items()))
    assert histograms['auxiliary'] == {k: 63*v for k, v in histograms['active'].items()}
    assert histograms['active']['1'] == 14 and histograms['active']['2'] == 3
    return {'N': n, 'histograms': histograms, 'overlap': -1, 'd2': 10,
            'closed': True, 'support_edges': len(d), 'AP_touched_terms': ap_affected,
            'literal_length2_touched_starts_not_deduplicated': affected_aliases,
            'endpoint_allocation_exact': str(weighted_allocation),
            'canonical_species_occupancy': 2*n**3, 'Ref': 4}


def ordered_factor_control() -> dict[str, Any]:
    phi_x = sp.diag(1, 2, -1, -2)
    phi_y = sp.Matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 2], [0, 0, 2, 0]])
    U1 = sp.Matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    U2 = sp.diag(sp.I, -sp.I, 1, 1)
    assert U1.det() == U2.det() == 1 and U1.H*U1 == U2.H*U2 == sp.eye(4)
    ordered = (phi_x*U1*U2+U1*U2*phi_y)/2
    wrong_order = (phi_x*U2*U1+U2*U1*phi_y)/2
    assert ordered != wrong_order and ordered != phi_x*U1*U2
    assert ordered.trace() != sum(ordered)  # Trace is not the CAR/operator contraction.
    return {'actual_endpoint_half_retained': True, 'wrong_SU4_order_rejected': True,
            'wrong_single_endpoint_rejected': True, 'factor_is_not_quantumPhi_or_ground': True}


def car_sign_control() -> dict[str, Any]:
    def ladder(mask: int, mode: int, create: bool) -> tuple[int, int] | None:
        if bool(mask & (1 << mode)) == create: return None
        return mask ^ (1 << mode), (-1)**((mask & ((1 << mode)-1)).bit_count())
    edges = {}; weights = [1, 2, -1, -2]
    for mask in range(256):
        for colour in range(4):
            first = ladder(mask, 4+colour, False)
            second = ladder(first[0], colour, True) if first else None
            if second:
                edges[(second[0], mask)] = weights[colour]*cast(tuple[int, int], first)[1]*second[1]
    hermitian = dict(edges)
    for (a, b), value in edges.items(): hermitian[(b, a)] = value
    assert hermitian and all(a != b for a, b in hermitian)
    sector = {m for m in range(256) if m.bit_count() == 4}
    assert len(sector) == 70 and any(a in sector and b in sector for a, b in hermitian)
    for (a, b), value in hermitian.items():
        assert a.bit_count() == b.bit_count()
        assert (-1)**((a & 15).bit_count()+(b & 15).bit_count())*value == -value
    patterns = {''.join(map(str, p)):63**sum(p) for p in itertools.product((0, 1), repeat=4)}
    assert sum(patterns.values()) == 64**4 != 64
    assert 64*(1+1)**2 == 256 != 128 and 64*(1-1)**2 == 0
    return {'all256_columns': True, 'canonical_four_sector_dimension': 70,
            'trace0_nonzero_and_exact_local_parity_sign': True, 'patterns': patterns,
            'all_assignments': sum(patterns.values()), 'shared_equal_flow_energy': 256,
            'shared_opposite_flow_energy': 0, 'not_SU4_Gauss_ground_counterexample': True}


def run(archive: Path, output: Path) -> None:
    with zipfile.ZipFile(archive) as z:
        inputs = {name: z.read(name) for name in z.namelist()}
    manifest = json.loads(inputs.pop('contiguous_carrier_review_input_manifest_20261012.json'))
    assert {name:digest(data) for name, data in inputs.items()} == manifest['members']
    profiles = native_profiles(inputs)
    W = sum((p['copies']*(i % 2+1)*sum((abs(Fraction(float(v))) for pair in row for v in pair), Fraction(0))
             for p in profiles if p['role'] != 'neutral'
             for i, row in enumerate(p['native_scalar_coefficients'])), Fraction(0))
    assert W == Fraction(46425051219964815, 8796093022208)
    native = [geometry(n, profiles) for n in (4, 6, 8)]
    for g in native: assert Fraction(g['endpoint_allocation_exact']) <= 64*W
    # Inspect immutable original execution as data, not rerun its consumer.
    capsule = json.loads(inputs['wilson_ground_kernel_actual1_20261012_capsule.json'])
    request = json.loads(inputs['wilson_ground_kernel_actual1_20261012_request.json'])
    assert capsule['execution']['natural_exit'] and capsule['execution']['returncode'] == 0
    assert capsule['preexecution_input_hashes'] == capsule['postexecution_input_hashes']
    import io
    with zipfile.ZipFile(io.BytesIO(inputs['wilson_ground_kernel_actual1_20261012_preexecution.zip'])) as pre, zipfile.ZipFile(io.BytesIO(inputs['wilson_ground_kernel_actual1_20261012_executed.zip'])) as post:
        for name, expected in request['inputs'].items():
            assert digest(pre.read(name)) == expected
            assert pre.read(name) == post.read(name)
    recorded = json.loads(inputs['wilson_ground_kernel_actual1_20261012_result.json'])
    for g, old in zip(native, recorded['contiguous_geometry']):
        assert g['histograms'] == old['all_forward_adjoint_frequency_counts']
        assert g['AP_touched_terms'] == old['AP_touched_terms']
    envelope = 2048*W
    assert Fraction(recorded['first_gate']['ground_variational_intersection'][0]) == -envelope
    result = {'kind':'INDEPENDENT_EXACT_NATIVE_CONTIGUOUS_CARRIER_REVIEW_NOT_ADMISSION',
              'input_pins':manifest['members'], 'channel':symbolic_channel(), 'native':native,
              'ordered_source_factors':ordered_factor_control(), 'CAR':car_sign_control(),
              'W':str(W), 'absolute_price':str(envelope), 'signed_ground_interval':[str(-envelope), '320'],
              'sufficient_K_epsilon0':5120, 'positive_K_not_inferred_from_norm':True,
              'original_capsule_bytes_checked_without_execution':True,
              'native_fullH_backend':False, 'ground_moments_computed':False,
              'scientific_admission':False, 'campaign_complete':False}
    with output.open('x') as f: json.dump(result, f, indent=2, sort_keys=True); f.write('\n')
    print(json.dumps({'complete_shared_cost':320, 'path_profile_counts_match_original':True,
                      'controls':'single orientation, 64 reset copies, AP/order/endpoints, mixed species, CAR sign',
                      'scope':'finite conditional complete form and signed gate only'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.input_archive, args.output)
