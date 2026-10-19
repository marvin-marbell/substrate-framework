"""Independent finite quantum canonical/all-character response falsifiers.

No target module import or old producer/checker execution. This is a METHOD
control, not the native fullPhi/SU4 slab state or a native magnetic sign.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np
from scipy.linalg import eigh
from scipy.special import iv


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def frozen_evidence(root: Path, request: dict) -> dict:
    hashes = {}
    for name, pin in request['pins'].items():
        raw = (root / name).read_bytes()
        assert sha(raw) == pin, name
        hashes[name] = pin
    with zipfile.ZipFile(root / request['executed_archive']) as archive:
        equality = {}
        for name in request['original_archive_members']:
            equality[name] = archive.read(name) == (root / name).read_bytes()
            assert equality[name], name
        original_exit = json.loads(archive.read('magnetic_thermal_actual2_20261011_exit.json'))
        assert original_exit['returncode'] == 0 and original_exit['natural_exit']
    profiles = json.loads((root / request['profile']).read_text())['primitive_profiles']
    receipt = json.loads((root / request['receipt']).read_text())
    channels = []
    for native in receipt['native_N_controls']:
        n = native['response_operator']['N']
        mj = mk = Fraction(0)
        for profile in profiles:
            if profile['scale'] != 'charged':
                continue
            for path in profile['paths']:
                if path['axis'] != 2:
                    continue
                # Both boundary z edges: 2N^2 starts. |cos|<=1 is deliberately
                # broad; source is fixed radian amplitude, not 1/q or 1/N.
                base = 8 * profile['load'] * 2 * n * n * (
                    abs(Fraction(path['c0'])) + abs(Fraction(path['ci'])))
                mj += base * path['length']
                mk += base * path['length'] ** 2
        stored = native['exact_thermal_enclosures']
        assert str(mj) == stored['MJ_N'] and str(mk) == stored['MK_N']
        source = native['response_operator']
        assert source['fixed_amplitude'] == 1
        assert source['literal_all64_CAR_jet_channels_exercised'] == 512
        assert all(face['exact_zero_net'] for face in source['perturbation_curvature'])
        channels.append({'N': n, 'MJ_N': str(mj), 'MK_N': str(mk),
                         'literal_CAR_jet_channels_archived': 512,
                         'actual_guard_contains_zero': True,
                         'guard': stored['actual_native_Fsecond_enclosure']})
    return {'target_sha256': hashes, 'archive_members_equal': equality,
            'original_natural_exit': original_exit, 'independent_rational_bound_constants': channels,
            'old_entrypoints_executed': False}


def transfer(mask: int, annihilate: int, create: int) -> tuple[int, int] | None:
    if not mask & (1 << annihilate):
        return None
    sign = (-1) ** (mask & ((1 << annihilate) - 1)).bit_count()
    reduced = mask ^ (1 << annihilate)
    if reduced & (1 << create):
        return None
    sign *= (-1) ** (reduced & ((1 << create) - 1)).bit_count()
    return reduced | (1 << create), sign


def canonical_model(request: dict) -> tuple[np.ndarray, np.ndarray]:
    # Two distinguishable fermion species, each exactly one of two modes.
    # No grand-canonical trace or target CAR implementation is used.
    masks = [1 << a | 1 << (2 + b) for a in range(2) for b in range(2)]
    lookup = {mask: i for i, mask in enumerate(masks)}
    hops = [np.zeros((4, 4), dtype=complex) for _ in range(2)]
    densities = []
    for shift, hop in zip((0, 2), hops):
        densities.append(np.diag([float(bool(mask & (1 << shift))) - .5 for mask in masks]))
        for column, mask in enumerate(masks):
            action = transfer(mask, shift + 1, shift)
            if action is not None:
                target, sign = action
                assert ((target & 3).bit_count(), (target >> 2).bit_count()) == (1, 1)
                hop[lookup[target], column] = sign
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z = np.diag([1., -1.])
    ra, rb = densities
    rho = ra + rb
    hbase = (np.kron(z / 3, np.eye(4)) + .19 * np.kron(y, ra - rb)
             + request['collective_density_coefficient'] * np.kron(np.eye(2), rho @ rho))
    forward = np.kron(x, .7 * hops[0] + .4j * hops[1])
    return hbase, forward


def model_jets(base: np.ndarray, forward: np.ndarray, alpha: float,
               length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    term = np.exp(1j * length * alpha) * forward
    adjoint = term.conj().T
    return base + term + adjoint, 1j * length * (term - adjoint), -length**2 * (term + adjoint)


def thermal_response(base: np.ndarray, forward: np.ndarray, beta: float,
                     request: dict) -> dict:
    h, j, k = model_jets(base, forward, 0., request['path_length'])
    energies, u = eigh(h)
    shift = float(energies.min())
    p = np.exp(-beta * (energies - shift))
    z = float(p.sum())
    p /= z
    jt = u.conj().T @ j @ u
    kt = u.conj().T @ k @ u
    mean_internal = float(np.dot(p, jt.diagonal().real))
    contact = float(np.dot(p, kt.diagonal().real))
    centered = jt - mean_internal * np.eye(len(p))
    chi_internal = zero_internal = 0.
    positive_mass = negative_mass = 0.
    kms_error = 0.
    for n in range(len(p)):
        for m in range(len(p)):
            gap = float(energies[m] - energies[n])
            square = float(abs(centered[n, m])**2)
            if abs(gap) < 1e-10:
                value = beta * p[n] * square
                zero_internal += value
            else:
                value = (p[n] - p[m]) / gap * square
                if gap > 0:
                    positive_mass += p[n] * square
                    kms_error = max(kms_error, abs(p[m] * square - math.exp(-beta * gap) * p[n] * square))
                else:
                    negative_mass += p[n] * square
            assert value >= -1e-13
            chi_internal += value
    # Entire continuous character torus b in[0,1), normalized db. Model scalar
    # energy c cos(2pi b), source j0+j1 cos(2pi b); all b are integrated ONCE.
    # The remaining independent characters, if appended, have integrand1.
    c = request['character_energy_cosine']
    j0, j1 = request['character_current_offset'], request['character_current_cosine']
    d = request['degenerate_current']
    t = beta * c
    cos_mean = -float(iv(1, t) / iv(0, t))
    cos_second = .5 * (1 + float(iv(2, t) / iv(0, t)))
    fibre_variance = j1**2 * (cos_second - cos_mean**2)
    between_fibres = beta * fibre_variance
    # Exact two-fold degenerate ancilla: H8 tensor I2, J includes d sigma_x.
    # Its WHOLE zero block has beta*d^2, purely offdiagonal in sigma_z basis.
    degenerate_zero = beta * d**2
    chi = chi_internal + between_fibres + degenerate_zero
    second = contact - chi
    internal_variance = float(np.dot(p, (centered @ centered).diagonal().real))
    total_variance = internal_variance + fibre_variance + d**2
    cap = beta * total_variance
    assert 0 <= chi <= cap + 2e-12
    assert between_fibres > .01 and degenerate_zero > .01
    assert np.linalg.norm(h @ j - j @ h) > .1
    assert abs(contact) > .01

    def free_energy(alpha: float) -> float:
        values = eigh(model_jets(base, forward, alpha, request['path_length'])[0], eigvals_only=True)
        minimum = float(values.min())
        internal = minimum - math.log(float(np.exp(-beta * (values - minimum)).sum())) / beta
        # Restore all scalar shifts BEFORE all-b integration/differentiation.
        return (internal - math.log(2 * math.cosh(beta * alpha * d)) / beta
                + alpha * j0 - math.log(float(iv(0, beta * (c + alpha * j1)))) / beta)

    f0 = free_energy(0.)
    step = request['finite_difference_step']
    def second_difference(epsilon: float) -> float:
        return (free_energy(epsilon) - 2 * f0 + free_energy(-epsilon)) / epsilon**2
    direct = (4 * second_difference(step / 2) - second_difference(step)) / 3
    assert abs(direct - second) < request['finite_difference_tolerance']
    assert kms_error < 1e-12 and positive_mass > .01 and negative_mass > .01
    wrong_atlas_second = (2 * math.pi)**2 * second
    assert abs(wrong_atlas_second - second) > .1
    # Discarding omega0: includes both diagonal/internal variance and the
    # non-diagonal EXACT degenerate block plus between-character variance.
    entire_zero = zero_internal + degenerate_zero + between_fibres
    return {'beta': beta, 'matrix_dimension': 16, 'canonical_species_particles': [1, 1],
            'quantum_field_dimension': 2, 'degeneracy_dimension': 2,
            'all_character_integral': 'normalized Haar db on[0,1); exact I0 identity evaluated binary64',
            'noncommuting_H_J_norm': float(np.linalg.norm(h @ j - j @ h)),
            'global_mean_current': mean_internal + j0 + j1 * cos_mean,
            'contact': contact, 'chi_internal': chi_internal, 'chi_global': chi,
            'beta_connected_equal_time_variance': cap, 'quantum_inequality_slack': cap - chi,
            'Fsecond': second, 'direct_Fsecond_Richardson': direct,
            'direct_difference_error': abs(direct - second),
            'lost_classical_fibre_variance_response_error': between_fibres,
            'missed_degenerate_offdiagonal_zero_block_error': degenerate_zero,
            'missed_entire_zero_block_response_error': entire_zero,
            'omitted_contact_response_error': abs(contact),
            'wrong_radian_as_atlas_source_Fsecond': wrong_atlas_second,
            'wrong_source_normalization_error': abs(wrong_atlas_second - second),
            'signed_measure_positive_frequency_mass': positive_mass,
            'signed_measure_negative_frequency_mass': negative_mass,
            'KMS_pair_error': kms_error,
            'scope': 'INDEPENDENT_FINITE_TOY_NOT_NATIVE_FULLPHI_SU4_THERMAL_NUMBER'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    frozen = frozen_evidence(args.request.parent, request)
    base, forward = canonical_model(request)
    results = [thermal_response(base, forward, beta, request) for beta in request['beta']]
    result = {'kind': 'INDEPENDENT_RESPONSE_REVIEW_NOT_ADMISSION',
              'source_sha256': sha(Path(__file__).read_bytes()),
              'request_sha256': sha(args.request.read_bytes()),
              'frozen_evidence': frozen, 'controls': results,
              'analytic_all_b_identity': 'integral_0^1 exp[-beta(c+alpha*j1)cos(2pi*b)-beta*alpha*j0] db = exp(-beta*alpha*j0)*I0(beta*(c+alpha*j1))',
              'degenerate_block_identity': 'H=H8 tensor I2; alpha*d*sigma_x gives Zancilla=2cosh(beta*alpha*d), chi_zero=beta*d^2',
              'native_response_sign_stiffness_ground_pole_admitted': False}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'receipt': args.output.name, 'controls': len(results), 'native_admission': False}))


if __name__ == '__main__':
    main()
