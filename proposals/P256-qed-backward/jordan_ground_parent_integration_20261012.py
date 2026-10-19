"""Immutable loop/reset join and native fixed-local spectral budget, not an IR theorem."""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time
from typing import Any
import zipfile

import sympy as sp

from local_orbital_stability import scalar_paths
from rotation_band import RotationBand

ROOT = Path(__file__).resolve().parent


def load(name: str) -> dict[str, Any]:
    return json.loads((ROOT / name).read_text())


def sha(name: str) -> str:
    with (ROOT / name).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verified_epoch(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    # Original executors use both flat and observed/outputs-prefixed members.
    # File identity and archive-member identity are separately pinned.
    for key in ('pre', 'executed', 'receipt', 'capsule', 'exit'):
        assert sha(spec[key]) == spec[key + '_sha256'], (key, spec[key])
    with zipfile.ZipFile(ROOT / spec['pre']) as pre, zipfile.ZipFile(ROOT / spec['executed']) as exe:
        names = pre.namelist()
        assert len(names) == len(set(names)) and set(names) == set(spec['member_map'])
        for name in names:
            assert pre.read(name) == exe.read(spec['member_map'][name]), name
        assert len(set(spec['member_map'].values())) == len(names)
        raw = exe.read(spec['receipt_member'])
        assert raw == (ROOT / spec['receipt']).read_bytes()
        exit_raw = exe.read(spec['exit_member'])
        assert exit_raw == (ROOT / spec['exit']).read_bytes()
        run: dict[str, Any] = json.loads(exit_raw)
        assert run['natural_exit'] is True and run['returncode'] == 0
        payload: dict[str, Any] = json.loads(raw)
    capsule = load(spec['capsule'])
    assert capsule[spec['stability']['flag']] is True
    for owner in (capsule, payload):
        for before, after in (('source_input_sha256_before', 'source_input_sha256_after'),
                              ('original_input_pins_before', 'original_input_pins_after'),
                              ('request_sha256_before', 'request_sha256_after')):
            if before in owner:
                assert owner[before] == owner[after]
    return payload, {'members_compared': len(names), 'natural_exit': True,
                     'returncode': 0, 'elapsed_seconds': run['elapsed_seconds'],
                     'immutable_source_inputs_unchanged': True}


def strip_transfer_metadata(value: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in value.items()
            if k not in ('all_input_pins', 'inputs_unchanged_after')}


def native_budget(request: dict[str, Any], soft: dict[str, Any]) -> dict[str, Any]:
    cfg = load(request['configuration'])['input_descriptor']['original_micro_config_UNCHANGED']
    assert cfg['auxiliary_count'] == 63 and cfg['a'] == 1 / 64
    # Merge the literal roles per native path before taking the entry norm.
    # This alternative collection is valid here because their hopping profiles
    # coincide; their distinct onsite masses are NOT thereby interchangeable.
    vectors: dict[tuple[int, ...], list[tuple[F, F]]] = {}
    for mass, count in ((cfg['auxiliary_m'], 63), (cfg['active_m'], 1)):
        _, paths = scalar_paths(RotationBand(a=cfg['a'], m=mass, r=cfg['light_r'],
                                            v=cfg['light_v'], improved=True))
        for displacement, coefficients in paths:
            key = tuple(int(v) for v in displacement)
            if key not in vectors:
                vectors[key] = [(F(0), F(0)) for _ in coefficients]
            vectors[key] = [(old[0] + count * F(float(z.real)),
                             old[1] + count * F(float(z.imag)))
                            for old, z in zip(vectors[key], coefficients)]
    w = sum((sum(abs(v) for v in key) *
             sum((abs(a) + abs(b) for a, b in row), F(0))
             for key, row in vectors.items()), F(0))
    prices = soft['uniform_local_pair_corollary_pending_independent_review']
    assert str(w) == prices['W_native_all_literal_roles']
    t = (F(1025) + 2048 * w - 1) / 16
    m1, m2 = F(256), 16384 * t + 65536
    mass, radius = m1 * m1 / m2, 2 * m2 / m1
    band = mass - m1 / radius
    assert mass == 4 / (t + 4) and band == 2 / (t + 4)
    assert str(t) == prices['Ec_second_moment_upper']
    assert str(mass) == prices['wholeP0_pair_covariance_lower']
    assert str(radius) == prices['finite_positive_band_upper']
    assert str(band) == prices['finite_positive_band_mass_lower']
    # Positive UV atom and diffuse UV spectrum satisfy these same moments.
    atom_energy, atom_mass = m2 / m1, mass
    centre = F(12, 13) * atom_energy
    diffuse_mass = m1 / centre
    assert atom_mass * atom_energy == m1
    assert atom_mass * atom_energy * atom_energy == m2
    assert diffuse_mass * centre == m1
    assert diffuse_mass * F(13, 12) * centre * centre == m2
    assert 0 < atom_mass < diffuse_mass < 1 and centre / 2 > 1
    return {'W_exact': str(w), 'T_upper': str(t), 'm1': str(m1),
            'm2_upper': str(m2), 'wholeP0_local_covariance_lower': str(mass),
            'positive_energy_cutoff': str(radius), 'positive_band_mass_lower': str(band),
            'direct_same_L_moment_argument': request['same_L_argument'],
            'finite_P0_identified_with_limiting_P0': False,
            'fixed_local_not_spatial_q_lower': True,
            'UV_atom_and_atomless_same_moments': True,
            'native_actual_ground_or_evolution_computed': False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    request = load(args.request)
    before = {name: sha(name) for name in request['pins']}
    assert before == request['pins']
    payloads: dict[str, dict[str, Any]] = {}
    verified: dict[str, dict[str, Any]] = {}
    for role, spec in request['epochs'].items():
        payloads[role], verified[role] = verified_epoch(spec)
        assert sha(spec['metadata_receipt']) == spec['metadata_receipt_sha256']
    old, typed = payloads['loop_original'], payloads['loop_typed']
    assert strip_transfer_metadata(old) == strip_transfer_metadata(typed)
    assert payloads['loop_review'] == payloads['loop_review_typed']
    ratio = F(old['local_normalized_gate']['normalized_ratio_at_physical1'])
    assert ratio > 1 and old['local_normalized_gate']['physical1_neumann_gate'] is False
    half = payloads['halfloop_original']
    value = F(half['all1plus63_signed_form'])
    lower = F(half['all1plus63_paid_lower'])
    assert lower < value < 0 and half['not_positive_flux_lower'] is True
    soft = payloads['parent_soft']
    assert soft['full_form_electric_increment'] == '128 I'
    assert soft['connection_current'] == '0' and soft['gradient_budget'] == '1/2'
    gain = [sp.sympify(row['charged_hop_increment_exact']) for row in soft['rows']]
    assert len(gain) == 2 and gain[0] != 0 and sp.simplify(gain[0] + gain[1]) == 0
    budget = native_budget(request, soft)
    corrections = {name: load(name) for name in request['corrections']}
    # Actual pinned frozen verdicts remain distinct from parent qualifications.
    reviews = {name: load(name) for name in request.get('reviews', [])}
    after = {name: sha(name) for name in request['pins']}
    assert after == before
    result = {'kind': 'JOINED_LOOP_OPERATOR_AND_UNIFORM_LOCAL_MAGNETIC_PAIR_NOT_IR_PHASE',
              'verified_epochs': verified,
              'members_compared': sum(row['members_compared'] for row in verified.values()),
              'loop': {'physical1_local_ratio': str(ratio), 'sufficient_gate_passed': False,
                       'polarized_endpoint_signed_coefficient': str(value),
                       'polarized_endpoint_paid_negative_lower': str(lower),
                       'same_vector_diagonal_ground_energy_claim': False,
                       'unrestricted_full64_mixed_species_price_executed': False,
                       'fullH_resolvent_compression_requires_physical_form_compatible_Q': True},
              'complete_soft_form_increment': '128 I',
              'opposite_native_POINT_carrier_signs': True,
              'uniform_fixed_local_magnetic_pair': budget,
              'correction_artifacts': list(corrections), 'independent_reviews': list(reviews),
              'typed_mathematical_payload_exact_equal': True,
              'original_targets_or_reviews_rebound': False,
              'fullphysical_ground_phase_stiffness_photon_admitted': False,
              'original_SOURCE_adopted': False, 'unconditional_machine_attestation': False,
              'campaign_complete': False, 'source_input_sha256_before': before,
              'source_input_sha256_after': after, 'elapsed_seconds': time.monotonic() - started}
    with Path(args.output).open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'epochs': len(verified), 'members': result['members_compared'],
                      'local_covariance_lower': budget['wholeP0_local_covariance_lower'],
                      'IR_phase_or_atom': False}))


if __name__ == '__main__':
    main()
