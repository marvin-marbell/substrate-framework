"""Native raw-curl real contact and filtered spectral noninference, not ground order."""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from pathlib import Path
import time
from typing import Any

import sympy as sp

ROOT = Path(__file__).resolve().parent
Cell = tuple[int, int, int]
Edge = tuple[int, Cell]


def sha(name: str) -> str:
    with (ROOT / name).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def move(cell: Cell, axis: int, n: int, step: int = 1) -> Cell:
    values = [(cell[k] + (step if k == axis else 0)) % n for k in range(3)]
    return values[0], values[1], values[2]


def face(normal: int, origin: Cell, n: int) -> dict[Edge, int]:
    j, k = (normal + 1) % 3, (normal + 2) % 3
    return {(j, origin): 1, (k, move(origin, j, n)): 1,
            (j, move(origin, k, n)): -1, (k, origin): -1}


def divergence(flow: dict[Edge, int], n: int) -> dict[Cell, int]:
    result: dict[Cell, int] = {}
    for (axis, origin), amount in flow.items():
        endpoint = move(origin, axis, n)
        result[origin] = result.get(origin, 0) - amount
        result[endpoint] = result.get(endpoint, 0) + amount
    return {site: amount for site, amount in result.items() if amount}


def exact_zero(matrix: Any) -> bool:
    return all(sp.simplify(value) == 0 for value in matrix)


def geometry() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    roots = {4: sp.I, 6: (1 + sp.I * sp.sqrt(3)) / 2,
             8: (1 + sp.I) / sp.sqrt(2)}
    zero: Cell = (0, 0, 0)
    for n, root in roots.items():
        base = [face(normal, zero, n) for normal in range(3)]
        assert all(not divergence(flow, n) and sum(v * v for v in flow.values()) == 4
                   for flow in base)
        overlaps: list[list[list[tuple[Cell, int]]]] = [[[] for _ in range(3)] for _ in range(3)]
        for x, y, z in product(range(n), repeat=3):
            site: Cell = (x, y, z)
            translated = [face(normal, site, n) for normal in range(3)]
            for a in range(3):
                for b in range(3):
                    overlap = sum(value * translated[b].get(edge, 0) for edge, value in base[a].items())
                    if overlap:
                        overlaps[a][b].append((site, overlap))
        for momentum in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 0)):
            phases = [sp.simplify(root ** component) for component in momentum]
            d = sp.Matrix([value - 1 for value in phases])
            curl = sp.Matrix([[0, -d[2], d[1]], [d[2], 0, -d[0]], [-d[1], d[0], 0]])
            direct = sp.Matrix(3, 3, lambda a, b: sp.simplify(sp.Add(*[
                amount * root ** sum(momentum[k] * site[k] for k in range(3))
                for site, amount in overlaps[a][b]])))
            symbol = sp.simplify(curl * curl.conjugate().T)
            q2 = sp.simplify((d.conjugate().T * d)[0])
            assert exact_zero(direct - symbol)
            assert exact_zero(d.T * curl)
            assert exact_zero(symbol - (q2 * sp.eye(3) - d.conjugate() * d.T))
            if q2 != 0:
                p_long = sp.simplify(d.conjugate() * d.T / q2)
                p_trans = sp.eye(3) - p_long
                assert exact_zero(p_trans * p_trans - p_trans)
                assert exact_zero(symbol - q2 * p_trans)
                assert not exact_zero(symbol - 4 * sp.eye(3))
            else:
                assert exact_zero(symbol)
            records.append({'N': n, 'momentum_integer': list(momentum), 'qhat_squared_exact': str(q2),
                            'direct_raw_face_overlap_matches_symbol': True,
                            'symbol_exact': [[str(sp.simplify(symbol[a, b])) for b in range(3)] for a in range(3)],
                            'Gauss_closed': True, 'Bianchi_dT_C_zero': True})
        neighboring = face(0, (0, 1, 0), n)
        overlap = sum(value * neighboring.get(edge, 0) for edge, value in base[0].items())
        difference = dict(base[0])
        for edge, amount in neighboring.items():
            difference[edge] = difference.get(edge, 0) - amount
        assert overlap == -1 and sum(value * value for value in difference.values()) == 10
    return records


def real_contact_boundary() -> dict[str, Any]:
    h = sp.diag(0, 1, 3)
    omega = sp.Matrix([1, 0, 0])
    a = sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    b = sp.Matrix([[0, -sp.I, 1], [sp.I, 0, 0], [1, 0, 0]])
    columns = sp.Matrix.hstack(a * omega, b * omega)
    full = columns.conjugate().T * h * columns
    real = sp.Matrix([[1, 0], [0, 4]])
    double = (omega.conjugate().T * (a * (h * b - b * h) - (h * b - b * h) * a) * omega)[0]
    assert full[0, 1] == sp.I and double == 0
    coefficients = sp.Matrix([1, sp.I])
    actual = (coefficients.conjugate().T * full * coefficients)[0]
    wrong = (coefficients.conjugate().T * real * coefficients)[0]
    assert actual == 3 and wrong == 5
    return {'method_only_not_native': True, 'complex_Gram_exact': [[str(v) for v in full.row(i)] for i in range(2)],
            'symmetric_real_contact_does_not_determine_complex_Gram': True,
            'full_complex_moment': str(actual), 'wrong_deleted_imaginary_moment': str(wrong)}


def filtered_models() -> list[dict[str, Any]]:
    q, gamma = sp.symbols('q gamma', positive=True)
    energy = sp.symbols('E', positive=True)
    lo, hi = 32 * q, 96 * q
    density = gamma / 64
    atom_mass = gamma * q
    m1_atom = atom_mass * 64 * q
    m1_diffuse = sp.integrate(energy * density, (energy, lo, hi))
    inverse_e_diffuse = sp.integrate(energy * density / (16384 * q**2), (energy, lo, hi))
    inverse_e_atom = m1_atom / (16384 * q**2)
    filt = (128 * q / (energy + 128 * q))**2
    filtered_diffuse = sp.integrate(filt * density, (energy, lo, hi))
    assert sp.simplify(m1_diffuse - m1_atom) == 0
    assert sp.simplify(inverse_e_diffuse - gamma / 256) == 0
    assert sp.simplify(inverse_e_atom - gamma / 256) == 0
    assert sp.simplify(filtered_diffuse - 16 * gamma * q / 35) == 0
    records: list[dict[str, Any]] = []
    for q_value in (F(1, 8), F(1, 16), F(1, 32), F(1, 64)):
        g = F(3, 4)
        atom_ir = F(4, 9) * g * q_value
        diffuse_ir = F(16, 35) * g * q_value
        uv = (1 - g) * (q_value / (2 + q_value))**2
        assert uv < q_value**2 / 4 and atom_ir > uv and diffuse_ir > uv
        records.append({'q_method_not_native_grid': str(q_value), 'gamma_not_measured_ground': str(g),
                        'same_total_first_moment': str(256 * (1 - g) + 64 * g * q_value**2),
                        'atom_filtered_IR_mass': str(atom_ir), 'diffuse_filtered_IR_mass': str(diffuse_ir),
                        'same_filtered_UV_mass': str(uv), 'IR_electric_inverse_weight': str(g / 256),
                        'diffuse_IR_has_no_atom': True, 'atom_and_diffuse_satisfy_real_inverse_upper': True})
    return records


def source_tail_control() -> dict[str, Any]:
    records = [{'R': r, 'radial_eighth_moment': '1', 'degree_four_source_squared_tail': '1',
                'tail_probability': str(F(1, r**8))} for r in (2, 4, 8, 16)]
    a = sp.diag(1, 0)
    h = sp.Matrix([[2, 1], [1, 1]])
    assert (h - a).det() == 0 and (h * h - a * a).det() == -1
    return {'method_only_not_native': True, 'uniform_radial8_does_not_give_uniform_source_tail': records,
            'positive_operator_power_order_inference_rejected': True,
            'higher_graph_may_hold_with_actual_native_commutator_proof': True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    request = json.loads(args.request.read_text())
    before = {name: sha(name) for name in request['pins']}
    assert before == request['pins']
    records = geometry()
    result = {'kind': 'NATIVE_CONTIGUOUS_RAW_CURL_REAL_CONTACT_AND_FILTERED_SPECTRAL_COUNTERCONTROL_NOT_GROUND_ORDER',
              'geometry': records, 'real_contact_complex_Gram_boundary': real_contact_boundary(),
              'fixed_normalization_spectral_models': filtered_models(), 'source_tail_control': source_tail_control(),
              'candidate_phase_Gram_only': {'C': '(1-gamma)I+gamma11T',
                  'real_paired_contact_symbol': '256(1-gamma)I+64gamma qhat^2 P_T',
                  'actual_ground_phase_Gram_computed': False},
              'source_IR_model_rescaling': 'sigma(dv)=mu(q dv)/q',
              'contact_scope': 'Symmetric REAL paired ground form for real spatial coefficients; imaginary complex momentum cross terms not eliminated.',
              'IR_linear_Faraday_model_only': 'mu_E(dE)=E^2/(16384q^2) mu_B_IR(dE), only on the deliberately linear IR model branch, not nonlinear UV source.',
              'native_actual_ground_or_fullH_evolution_computed': False,
              'physical1_phase_stiffness_or_atom': False, 'original_SOURCE_adopted': False,
              'campaign_complete': False, 'elapsed_seconds': time.monotonic() - started,
              'source_input_sha256_before': before,
              'source_input_sha256_after': {name: sha(name) for name in request['pins']}}
    assert result['source_input_sha256_before'] == result['source_input_sha256_after']
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
