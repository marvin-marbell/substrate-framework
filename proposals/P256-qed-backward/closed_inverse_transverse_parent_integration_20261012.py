from __future__ import annotations

import argparse
import hashlib
import json
import time
from fractions import Fraction as F
from pathlib import Path
from typing import Any

from history_operator_reset_parent_integration_20261012 import epoch

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def native_closed_form_controls() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n in (4, 6, 8):
        sites = [(x, y, z) for x in range(n) for y in range(n) for z in range(n)]
        edges = [(x, axis) for x in sites for axis in range(3)]
        index = {edge: i for i, edge in enumerate(edges)}

        def shift(x: tuple[int, int, int], axis: int) -> tuple[int, int, int]:
            out = list(x)
            out[axis] = (out[axis] + 1) % n
            return out[0], out[1], out[2]

        def face(axis: int, other: int) -> list[F]:
            x = (n - 1, n - 1, 0)
            c = [F(0) for _ in edges]
            for edge, sign in (((x, axis), 1), ((shift(x, axis), other), 1),
                               ((shift(x, other), axis), -1), ((x, other), -1)):
                c[index[edge]] += sign
            return c

        def divergence(vector: list[F]) -> dict[tuple[int, int, int], F]:
            out = dict.fromkeys(sites, F(0))
            for (x, axis), value in zip(edges, vector):
                out[x] += value
                out[shift(x, axis)] -= value
            return out

        ca, cb = face(0, 1), face(1, 2)
        h = [a / 3 + 2 * b / 7 for a, b in zip(ca, cb)]
        assert not any(divergence(h).values())
        assert sum(c * c for c in ca) == 4
        # Full affine electric POINT component: one shared raw flow, not a
        # separately prepared physical quantum vector or independently copied E.
        incoming = [F(64) if edge == ((n - 1, n - 1, 0), 0) else F(0) for edge in edges]
        electric = [k + c / 3 for k, c in zip(incoming, ca)]
        rho = divergence(electric)
        norm2 = sum(v * v for v in h)
        eh = sum(c * e for c, e in zip(h, electric))
        energy = 64 * sum(e * e for e in electric)
        increments = []
        for s in (F(-7, 3), F(-1, 11), F(2, 5)):
            shifted = [e + s * c for e, c in zip(electric, h)]
            assert divergence(shifted) == rho
            direct = 64 * sum(e * e for e in shifted) - energy
            target = 128 * s * eh + 64 * s * s * norm2
            assert direct == target
            assert direct != 128 * s * eh + 32 * s * s * norm2
            increments.append({'s': str(s), 'energy_increment': str(direct)})
        rows.append({'N': n, 'scope': 'EXACT_PERIODIC_NATIVE_GRAPH_AFFINE_E_POINT_FACTOR_NOT_QUANTUM_SOURCE',
                     'mixed_closed_face_norm2': str(norm2), 'shared_fullE_rho_max': str(max(abs(v) for v in rho.values())),
                     'increments': increments, 'same_rho_after_real_chart_shift': True,
                     'fullPhi_SU4_or_fullH_source_executed': False})
    return rows


def spectral_composition() -> dict[str, Any]:
    # Exact NONNATIVE spectral controls. The fixed native-source normalization is
    # not changed; these show what the proposed operator UPPER can and cannot say.
    inverse_bound = F(1, 128)
    kappa = F(3)
    rows = []
    for q in (F(1, 4), F(1, 16), F(1, 64)):
        velocity = F(2)
        residue = F(1, 256)
        atomic_inverse = residue / velocity
        atomic_mass = q * residue
        atomless_inverse = inverse_bound
        atomless_mass = kappa * q / 256
        assert atomic_inverse <= inverse_bound and atomless_inverse <= inverse_bound
        assert atomic_mass <= kappa * q / 128 and atomless_mass <= kappa * q / 128
        for order in (0, 1, 2, 3):
            atomic_moment = residue * velocity ** order
            atomless_moment = kappa ** (order + 1) / (128 * (order + 2))
            assert atomic_moment <= kappa ** (order + 1) / 128
            assert atomless_moment <= kappa ** (order + 1) / 128
        # A gapped source saturates the inverse cap and nevertheless has no
        # small-q spectral contribution; upper saturation is not an IR lower.
        massive_energy, massive_mass = F(256), F(2)
        assert massive_mass / massive_energy == inverse_bound
        assert kappa * q < massive_energy
        rows.append({'q': str(q), 'atomic_rescaled_mass': str(residue),
                     'atomless_rescaled_mass': str(atomless_mass / q),
                     'massive_lowband_mass': '0', 'zero_source_lowband_mass': '0'})

    # Continuous TOTAL-measure approximants force zero electric source mass in
    # the limiting whole kernel. Their inverse weights need not be conserved.
    collapse = []
    for n in (4, 16, 64):
        energy, mass = F(1, n), F(1, 256 * n)
        assert mass / energy == F(1, 256)
        for cutoff in (1, 8, 128):
            assert mass * min(F(cutoff), 1 / energy) <= F(1, 256)
        collapse.append({'energy': str(energy), 'mass': str(mass),
                         'inverse_mass': '1/256', 'total_mass_limit': '0'})
    assert F(1) * 128 > F(1, 256)  # False mass1 zero atom fails f_K(0)=K.

    # Whole-ground projection differs from mean subtraction even for mean0.
    ground_source = (F(0), F(1), F(0))
    ground_projection = ground_source[:2]
    assert sum(x * x for x in ground_projection) == 1
    assert ground_source[0] == 0

    # For a two-state all-character METHOD H_s, det=256s^2(g-64a^2+256s^2).
    # Native face ||c||^2=4 gives inverse a^2/g<=1/64. An arbitrary positive
    # selected H alone does not imply this discriminant requirement.
    gap, amplitude = F(64), F(1)
    assert amplitude * amplitude / gap == F(1, 64)
    assert gap - 64 * amplitude * amplitude == 0
    wrong_gap, tiny_shift = F(1), F(1, 16)
    determinant_wrong = 256 * tiny_shift ** 2 * (wrong_gap - 64 + 256 * tiny_shift ** 2)
    assert determinant_wrong < 0

    # Gaussian-integer entries are represented exactly in these small complex
    # operations. The cross term, rather than a pinned eigenvalue, rejects the
    # wrong complex bound inferred from a real quadratic estimate.
    gram = ((1, 1j), (-1j, 1))
    vector = (1, -1j)
    weight = sum(complex(vector[i]).conjugate() * gram[i][j] * vector[j]
                 for i in range(2) for j in range(2))
    norm2 = sum(abs(v) ** 2 for v in vector)
    assert weight.imag == 0 and weight.real > norm2
    matrix_measures = compressed_matrix_measure_controls()
    # Velocity inverse weight can escape to v=0 with ZERO mass in the limit.
    escape = [{'v': str(F(1, n)), 'mass': str(F(1, 128 * n)), 'inverse_weight': '1/128'} for n in (4, 16, 64)]
    return {'scope': 'EXACT_NONNATIVE_SPECTRAL_METHOD_AND_NONINFERENCE_CONTROLS',
            'atomic_atomless_massive_zero': rows, 'continuous_inverse_collapse': collapse,
            'velocity_inverse_escape_without_mass_atom': escape,
            'imaginary_cross_weight': str(weight.real), 'imaginary_cross_norm2': str(norm2),
            'noncommuting_compressed_positive_measures': matrix_measures,
            'wholeP0_mean0_contrary_source_mass': '1',
            'all_character_method_face_inverse': '1/64',
            'selected_positive_H_without_all_character_floor_rejected': True,
            'upper_or_inverse_saturation_does_not_force_IR_mass_or_atom': True,
            'native_ground_or_spectrum_computed': False}


def compressed_matrix_measure_controls() -> dict[str, Any]:
    def multiply(a: list[list[F]], b: list[list[F]]) -> list[list[F]]:
        return [[sum((a[i][k] * b[k][j] for k in range(2)), F(0))
                 for j in range(2)] for i in range(2)]

    def psd(a: list[list[F]]) -> bool:
        return a[0][1] == a[1][0] and a[0][0] >= 0 and a[1][1] >= 0 and a[0][0] * a[1][1] >= a[0][1] ** 2

    # B has rows (1,0),(1,1),(0,1), divided by32, in an energy1/2/3
    # Hilbert space. Underlying spectral projections commute, their compressed
    # positive matrix measures need not. Every entry below is exact rational.
    rows = ((F(1), F(0)), (F(1), F(1)), (F(0), F(1)))
    nu = [[[v[i] * v[j] / 1024 for j in range(2)] for i in range(2)] for v in rows]
    total = [[sum((a[i][j] for a in nu), F(0)) for j in range(2)] for i in range(2)]
    cap = [[(F(1, 256) if i == j else F(0)) - total[i][j] for j in range(2)] for i in range(2)]
    assert psd(cap)
    ab, ba = multiply(nu[0], nu[1]), multiply(nu[1], nu[0])
    commutator = [[ab[i][j] - ba[i][j] for j in range(2)] for i in range(2)]
    assert any(v for row in commutator for v in row)
    bands = []
    for omega in (F(1), F(2), F(3)):
        mass = [[sum((F(k + 1) * nu[k][i][j] for k in range(3) if k + 1 <= omega), F(0))
                 for j in range(2)] for i in range(2)]
        slack = [[(omega / 256 if i == j else F(0)) - mass[i][j] for j in range(2)] for i in range(2)]
        assert psd(mass) and psd(slack)
        bands.append({'omega': str(omega), 'source_matrix': [[str(v) for v in row] for row in mass]})
    return {'inverse_matrix': [[str(v) for v in row] for row in total],
            'compressed_commutator': [[str(v) for v in row] for row in commutator],
            'positive_source_bands': bands, 'compressed_commutativity_rejected': True,
            'native_spectral_measure_computed': False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    request: dict[str, Any] = json.loads(Path(args.request).read_text())
    before = {name: digest(ROOT / name) for name in request['pins']}
    assert before == request['pins']
    payloads: dict[str, dict[str, Any]] = {}
    bindings = {}
    for label, spec in request['epochs'].items():
        payloads[label], bindings[label] = epoch(spec)
    scalar = json.loads((ROOT / request['scalar_interface']).read_text())
    transverse = json.loads((ROOT / request['transverse_interface']).read_text())
    assert transverse['scalar_interface'] == request['scalar_interface']
    assert scalar['finite']['face_inverse_upper'] == '1/64'
    assert F(scalar['finite']['real_inverse_upper']['numerator'], scalar['finite']['real_inverse_upper']['denominator']) == F(1, 256)
    assert F(transverse['source_maps']['complex_B_norm_squared']['numerator'], transverse['source_maps']['complex_B_norm_squared']['denominator']) == F(1, 128)
    assert transverse['source_maps']['total_source_extension_claimed'] is False
    assert transverse['residue_velocity']['lower_mass_atom_phase_stiffness_commoncone_inferred'] is False
    reviewed = {}
    for label, spec in request['reviews'].items():
        proof = json.loads((ROOT / spec['proof']).read_text())
        assert digest(ROOT / spec['proof']) == spec['proof_sha256']
        assert proof['verdict'] == spec['verdict']
        if label == 'inverse':
            assert proof['mathematical_target_defects_found'] == []
            assert proof['target_pins'][scalar['proof']] == digest(ROOT / scalar['proof'])
        else:
            assert proof['blocking_internal_findings'] == []
            assert proof['target']['sha256'] == digest(ROOT / transverse['proof'])
        reviewed[label] = {'verdict': proof['verdict'], 'proof_sha256': spec['proof_sha256']}
    modes = sum(row['all_modes'] for row in payloads['transverse']['native_graph_Fourier'])
    assert modes == sum(n ** 3 for n in (4, 6, 8))
    assert payloads['inverse']['native_numerical_ground_or_fullH_backend'] is False
    assert payloads['transverse']['native_ground_or_fullH_backend'] is False
    receipt = {'schema': 1, 'kind': 'INDEPENDENT_INVERSE_TRANSVERSE_NATIVE_FACTOR_AND_SPECTRAL_COMPOSITION_NOT_PHASE',
               'epochs': bindings, 'members_compared': sum(row['members_compared'] for row in bindings.values()),
               'finite_native_exact_modes_from_immutable_scientist_control': modes,
               'native_closed_form': native_closed_form_controls(),
               'spectral_method': spectral_composition(),
               'independent_proof_reviews': reviewed,
               'same_history_inverse_and_transverse_proof_review_pending': False,
               'transverse_scalar_dependency_composed_not_rebound': True,
               'finiteP0_identified_with_limitingP0': False,
               'native_fullH_or_ground_backend': False, 'pointwise_selected_p_or_IR_atom_claimed': False,
               'physical1_phase_stiffness_or_photon': False, 'original_SOURCE_adopted': False,
               'campaign_complete': False, 'source_input_sha256_before': before,
               'source_input_sha256_after': {name: digest(ROOT / name) for name in request['pins']},
               'elapsed_seconds': time.monotonic() - started}
    assert receipt['source_input_sha256_after'] == before
    with Path(args.output).open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'members_compared': receipt['members_compared'], 'modes': modes,
                      'native_quantum_ground': False, 'phase_or_atom': False}))


if __name__ == '__main__':
    main()
