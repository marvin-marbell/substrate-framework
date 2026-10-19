"""Bound original wave epochs; exercise filtered contact and boundary realization prices."""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile

import sympy as sp

ROOT = Path(__file__).resolve().parent


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def verify_epoch(prefix: str) -> dict[str, Any]:
    cap = json.loads((ROOT / f'{prefix}_capsule.json').read_text())
    assert cap['returncode'] == 0 and cap['natural_exit']
    assert cap['sources_inputs_unchanged']
    files = cap.get('files', {})
    pre_name = cap.get('preexecution_zip', files.get('preexecution.zip'))
    exe_name = cap.get('executed_zip', files.get('executed.zip'))
    receipt_name = cap.get('receipt', files.get('receipt.json'))
    assert isinstance(pre_name, str) and isinstance(exe_name, str) and isinstance(receipt_name, str)
    assert sha((ROOT / pre_name).read_bytes()) == cap['preexecution_sha256']
    assert sha((ROOT / exe_name).read_bytes()) == cap['executed_sha256']
    assert sha((ROOT / receipt_name).read_bytes()) == cap['receipt_sha256']
    source_pins = cap.get('source_input_sha256')
    with zipfile.ZipFile(ROOT / pre_name) as pre, zipfile.ZipFile(ROOT / exe_name) as exe:
        if source_pins is None:
            assert prefix in {'magnetic_boundary_actual3_20261011', 'magnetic_boundary_slab_actual1_20261011'}
            source_pins = json.loads(pre.read('magnetic_boundary_preexecution_manifest.json'))['source_input_sha256']
        input_directory = 'inputs/' if prefix in {'filtered_flux_parent_actual1_20261011', 'phase_arc_parent_actual1_20261011'} else ''
        for member in pre.namelist():
            assert pre.read(member) == exe.read(member), (prefix, member)
        for member, pin in source_pins.items():
            assert sha(pre.read(input_directory + member)) == pin, (prefix, member)
        assert exe.read(receipt_name) == (ROOT / receipt_name).read_bytes()
        historical_live_difference = [name for name in source_pins
            if not name.startswith('external/') and (ROOT / name).is_file()
            and sha((ROOT / name).read_bytes()) != source_pins[name]]
    return {'epoch': prefix, 'natural_exit': True, 'seconds': cap['elapsed_seconds'],
            'pre_members_compared': len(pre.namelist()),
            'historical_live_differences_not_rebound': historical_live_difference,
            'receipt': receipt_name, 'receipt_sha256': cap['receipt_sha256']}


def verify_review(spec: dict[str, Any]) -> dict[str, Any]:
    exit_record = json.loads((ROOT / spec['exit']).read_text())
    assert exit_record['returncode'] == 0 and exit_record['natural_exit']
    assert exit_record[spec['stable_field']]
    with zipfile.ZipFile(ROOT / spec['pre']) as pre, zipfile.ZipFile(ROOT / spec['exe']) as exe:
        for name in pre.namelist():
            assert pre.read(name) == exe.read(spec['member_map'].get(name, name)), (spec['name'], name)
        for name, pin in spec['source_pins'].items():
            assert sha(pre.read(spec['input_directory'] + name)) == pin, (spec['name'], name)
        assert exe.read(spec['receipt']) == (ROOT / spec['receipt']).read_bytes()
        assert exe.read(spec['exit']) == (ROOT / spec['exit']).read_bytes()
        compared = len(pre.namelist())
    return {'review': spec['name'], 'natural_exit': True,
            'seconds': exit_record['elapsed_seconds'], 'pre_members_compared': compared,
            'original_review_bytes_not_rebound': True}


def filtered_contact_boundary() -> dict[str, Any]:
    L = sp.diag(0, 2, 7)
    field = sp.Matrix([[1, 2, -1], [2, 0, 3], [-1, 3, 1]])
    P0 = sp.diag(1, 0, 0)
    Q0 = sp.eye(3) - P0
    omega = sp.Matrix([1, 0, 0])
    epsilon = sp.Rational(1, 3)
    resolvent = epsilon * (L + epsilon * sp.eye(3)).inv()
    A = resolvent * Q0 * field * P0
    operator = A + A.H
    source = operator * omega
    assert operator == operator.H and source == resolvent * Q0 * field * omega
    filtered_first = (source.H * L * source)[0]
    commutator = L * operator - operator * L
    double = operator * commutator - commutator * operator
    assert (omega.H * double * omega)[0] == 2 * filtered_first
    old_commutator = L * field - field * L
    old_double = field * old_commutator - old_commutator * field
    old_first = (omega.H * old_double * omega)[0] / 2
    assert filtered_first == sp.Rational(8, 49) + sp.Rational(7, 484)
    assert old_first == 15 and filtered_first != old_first
    assert operator != resolvent * field
    return {'kind': 'EXACT_METHOD_FILTERED_SOURCE_NOT_LOCAL_MULTIPLIER',
            'filtered_first_moment_exact': str(filtered_first),
            'unfiltered_first_moment_exact': str(old_first),
            'wrong_reused_local_contact_rejected': True,
            'wrong_one_sided_resolvent_as_selfadjoint_operator_rejected': True,
            'native_ground_or_phase_computed': False}


def boundary_realization_price(n: int) -> dict[str, Any]:
    sites = [(x, y, z) for x in range(n) for y in range(n) for z in range(n)]
    edges = [(xyz, axis) for xyz in sites for axis in range(3)
             if axis != 0 or xyz[0] < n - 1]
    pinned = [(xyz, axis) for xyz, axis in edges if axis in (1, 2) and xyz[0] in (0, n - 1)]
    dynamic = [edge for edge in edges if edge not in pinned]
    interior = [xyz for xyz in sites if 0 < xyz[0] < n - 1]
    row = {xyz: i for i, xyz in enumerate(interior)}
    DI = sp.zeros(len(interior), len(dynamic))
    for j, (xyz, axis) in enumerate(dynamic):
        target = list(xyz)
        target[axis] = (target[axis] + 1) % n
        target_xyz = (target[0], target[1], target[2])
        if xyz in row:
            DI[row[xyz], j] += 1
        if target_xyz in row:
            DI[row[target_xyz], j] -= 1
    laplacian = DI * DI.T
    inverse = laplacian.inv()
    assert laplacian * inverse == sp.eye(len(interior))
    rho_one = sp.Matrix([2 if xyz[0] < n // 2 else -2 for xyz in interior])
    rho_all = 64 * rho_one
    longitudinal = DI.T * inverse * rho_all
    assert DI * longitudinal == rho_all
    one_copy_C = (rho_one.T * inverse * rho_one)[0]
    full_C = (rho_all.T * inverse * rho_all)[0]
    assert full_C == 4096 * one_copy_C and full_C != 64 * one_copy_C
    k = sp.zeros(len(dynamic), 1)
    k[0] = 1
    q = k - DI.T * inverse * DI * k
    assert DI * q == sp.zeros(len(interior), 1)
    assert (q.T * longitudinal)[0] == 0
    increment = ((q + longitudinal).T * (q + longitudinal))[0]
    assert increment == (q.T * q)[0] + full_C
    # A boundary yz loop has nonzero raw cochain but NO dynamic electric coordinates.
    x, y, z = 0, 0, 0
    loop = [((x, y, z), 1), ((x, y + 1, z), 2),
            ((x, y, z + 1), 1), ((x, y, z), 2)]
    assert all(edge in pinned and edge not in dynamic for edge in loop)
    plane_phase = F(1, n * n)
    assert plane_phase != 0 and plane_phase.denominator != 1
    assert len(interior) == (n - 2) * n * n
    assert len(dynamic) == (3 * n - 5) * n * n
    quotient_dimension = len(dynamic) - len(interior)
    original_torus_dimension = 2 * n**3 + 1
    assert quotient_dimension != original_torus_dimension
    return {'N': n, 'interior_Gauss_rank': len(interior),
            'dynamic_U1_coordinates': len(dynamic), 'pinned_tangential_coordinates_absent': len(pinned),
            'reduced_slab_electric_dimension': quotient_dimension,
            'original_torus_electric_dimension': original_torus_dimension,
            'boundary_loop_phase_in_2pi_units_exact': str(plane_phase),
            'collective_Coulomb_exact': str(full_C),
            'wrong_independent_copy_Coulomb_exact': str(64 * one_copy_C),
            'whole_collective_HE_HC_orthogonal_identity': True,
            'boundary_loop_has_no_dynamic_E_generator': True,
            'original_torus_reducing_sector_claim_rejected': True,
            'magnetic_tension_or_spectral_inequivalence_claimed': False,
            'canonical_density_control_only_not_ground': True}


def run(request_path: Path, output: Path) -> None:
    request = json.loads(request_path.read_text())
    for name, pin in request['pins'].items():
        assert sha((ROOT / name).read_bytes()) == pin, name
    epochs = [verify_epoch(prefix) for prefix in request['epochs']]
    review_snapshots = [verify_review(spec) for spec in request['review_snapshots']]
    normalization = [{'kappa_exact': str(k), 'fixed_first_harmonic_contact_cap_exact': str(48 * (4 + k)**2),
                      'weak_source_contact_exact': str(192 * k**2)} for k in (F(1, 4), F(1, 8), F(1, 16))]
    assert all(F(row['fixed_first_harmonic_contact_cap_exact']) > 768 for row in normalization)
    result = {'kind': 'IMMUTABLE_SMOOTHED_PHASE_ARC_BOUNDARY_PARENT_CONSUMER_NOT_GROUND_OR_PHASE',
              'epochs': epochs, 'filtered_contact': filtered_contact_boundary(),
              'review_snapshots': review_snapshots,
              'boundary_realization': boundary_realization_price(4),
              'normalization': normalization,
              'original_SOURCE_adopted': False, 'fullphysical_ground_computed': False,
              'physical1_phase_stiffness_or_pole': False, 'campaign_complete': False}
    for name, pin in request['pins'].items():
        assert sha((ROOT / name).read_bytes()) == pin, name
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'receipt': output.name, 'epochs': len(epochs),
                      'original_periodic_adoption': False, 'ground_or_phase': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.request, args.output)
