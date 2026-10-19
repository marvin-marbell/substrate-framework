#!/usr/bin/env python3
"""Continuous-Haar centre wells: rigorous operator/trial bounds and finite probes.

The declared scalar law H=kappa*(-Delta_SU4)+J*W is NOT the matter-coupled
angular photon Hamiltonian. Four trial packets are not an isolated low band.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
import scipy
from scipy.linalg import eigh, expm
from scipy.optimize import minimize_scalar

from exact_haar_alignment import ExactHaarAlignment
from quantum_su4_gauss import (CUTOFFS, HaarLink, RotationBand, matrix_unit_basis,
                               norm, require_small, source_hashes, uniform_data)


CENTRE_DISTANCE_CONSTANT = 2. + np.pi ** 2 / 2.


def commutator_controls(frame, generators):
    """Full complex HS superoperator, plus the real canonical local Hessian."""
    units = np.eye(16, dtype=complex).reshape(16, 4, 4)
    images = np.array([sum(field @ (field @ unit - unit @ field)
                           - (field @ unit - unit @ field) @ field
                           for field in frame) for unit in units])
    operator = images.reshape(16, 16).T
    values = np.linalg.eigvalsh(operator)
    require_small(norm(operator - operator.conj().T), 'HS commutator Hermiticity', 2e-9)
    require_small(norm(operator @ np.eye(4).ravel()), 'scalar commutator kernel', 2e-9)
    if values[1] <= 1e-10:
        raise ValueError('This centre-well construction requires a scalar-only commutant')
    gram = np.array([[np.trace(a @ b).real / 4 for b in frame] for a in frame])
    clifford_error = max(norm(a @ b + b @ a - 2 * gram[i, j] * np.eye(4))
                         for i, a in enumerate(frame) for j, b in enumerate(frame))
    require_small(clifford_error, 'selected weighted Clifford algebra', 2e-9)
    require_small(norm(gram - np.diag(np.diag(gram))), 'selected orthogonal Clifford frame', 2e-9)
    scales = np.diag(gram)
    if min(scales) <= 0:
        raise ValueError('Four nonzero weighted Clifford fields required')
    predicted = []
    for mask in range(16):
        grade = mask.bit_count()
        anticommute = [j for j in range(4) if bool(mask & (1 << j)) == (grade % 2 == 0)]
        predicted.append(4 * sum(scales[j] for j in anticommute))
    require_small(norm(values - np.sort(predicted)), 'Clifford full superoperator spectrum', 2e-9)
    commutators = np.array([[field @ t - t @ field for t in generators] for field in frame])
    local = np.einsum('maij,mbij->ab', commutators.conj(), commutators).real
    local_values = np.linalg.eigvalsh(local)
    require_small(norm(local_values - values[1:] / 2), 'canonical local metric factor two', 2e-9)
    return dict(superoperator_eigenvalues=values.tolist(), scalar_kernel_dimension=1,
                ell_min=float(values[1]), ell_max=float(values[-1]),
                clifford_gram=gram.tolist(), clifford_anticommutator_residual=clifford_error,
                clifford_predicted_eigenvalues=sorted(float(v) for v in predicted),
                local_quadratic_matrix=local.tolist(), local_quadratic_eigenvalues=local_values.tolist(),
                canonical_trace_residual=norm(2 * np.einsum('aij,bji->ab', generators, generators)
                                              - np.eye(15)))


def alignment_value(frame, u):
    return float(sum(norm(field @ u - u @ field) ** 2 for field in frame))


def continuous_controls(frame, generators, controls, seed=222, samples=48):
    """Finite adverse probes of independently proved inequalities, not a proof."""
    rng = np.random.default_rng(seed)
    worst = dict(global_lower_violation=0., global_upper_violation=0.,
                 centre_distance_violation=0., local_lower_violation=0.,
                 local_upper_violation=0., determinant_residual=0.)
    near = []
    ell_min, ell_max = controls['ell_min'], controls['ell_max']
    for _ in range(samples):
        raw = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        u, _ = np.linalg.qr(raw)
        u[:, 0] /= np.linalg.det(u)
        q = 4. - abs(np.trace(u)) ** 2 / 4.
        w = alignment_value(frame, u)
        distance2 = min(norm(u - 1j ** z * np.eye(4)) ** 2 for z in range(4))
        worst['global_lower_violation'] = max(worst['global_lower_violation'], ell_min * q - w)
        worst['global_upper_violation'] = max(worst['global_upper_violation'], w - ell_max * q)
        worst['centre_distance_violation'] = max(worst['centre_distance_violation'],
                                                distance2 - CENTRE_DISTANCE_CONSTANT * q)
        worst['determinant_residual'] = max(worst['determinant_residual'], abs(np.linalg.det(u) - 1))
    direction = rng.normal(size=15)
    direction /= norm(direction)
    x0 = np.einsum('a,aij->ij', direction, generators)
    local0 = float(direction @ np.array(controls['local_quadratic_matrix']) @ direction)
    for radius in (1., .3, .1, .03, .01):
        x = radius * x0
        u = expm(1j * x)
        w = alignment_value(frame, u)
        quadratic = radius ** 2 * local0
        worst['local_lower_violation'] = max(worst['local_lower_violation'],
                                            (1 - radius ** 2 / 12) * quadratic - w)
        worst['local_upper_violation'] = max(worst['local_upper_violation'], w - quadratic)
        near.append(dict(canonical_radius=radius, W=w, quadratic=quadratic,
                         relative_quadratic_error=(quadratic - w) / quadratic,
                         rigorous_relative_error_upper=radius ** 2 / 12))
    for label, value in worst.items():
        require_small(value, label, 2e-9)
    return dict(seed=seed, generic_SU4_samples=samples, maximum_violations=worst,
                shrinking_centre_probe=near, interpretation='numerical checks of analytic bounds only')


def packet_bound(controls, kappa, alignment, radius) -> dict[str, Any]:
    """Exact inequalities for four disjoint H1 radial Haar packets, rho<=1."""
    if not (0 < radius <= 1):
        raise ValueError('Certified packet chart requires 0<rho<=1')
    sinc = float(np.sinc(radius / (2 * np.pi)))
    jacobian_lower = sinc ** 12
    inverse_metric_upper = sinc ** -2
    kinetic_upper = 136 * kappa / (jacobian_lower * radius ** 2)
    potential_upper = alignment * controls['ell_max'] * radius ** 2 / 2
    mu = np.array(controls['local_quadratic_eigenvalues'])
    harmonic_ground = float(np.sqrt(kappa * alignment * mu).sum())
    harmonic_gap = float(2 * np.sqrt(kappa * alignment * mu[0]))
    lower_harmonic_ground = jacobian_lower * np.sqrt(1 - radius ** 2 / 12) * harmonic_ground
    return dict(J=alignment, kappa=kappa, radius=radius,
                exponential_Haar_jacobian_lower=jacobian_lower,
                exponential_inverse_metric_upper=inverse_metric_upper,
                kinetic_expectation_upper=kinetic_upper, potential_expectation_upper=potential_upper,
                four_trial_Rayleigh_upper=kinetic_upper + potential_upper,
                fourth_full_rotor_eigenvalue_upper=kinetic_upper + potential_upper,
                each_centre_sector_ground_upper=kinetic_upper + potential_upper,
                compressed_fundamental_amplitude_interval=[1 - radius ** 2 / 16, 1.],
                compressed_clock_operator_error_upper=radius ** 2 / 16,
                summed_fundamental_leakage_squared_upper=radius ** 2 / 2,
                euclidean_local_harmonic_ground=harmonic_ground,
                euclidean_local_harmonic_gap=harmonic_gap,
                one_ball_Dirichlet_ground_lower=lower_harmonic_ground,
                one_ball_Dirichlet_second_eigenvalue_lower=lower_harmonic_ground
                    + jacobian_lower * np.sqrt(1 - radius ** 2 / 12) * harmonic_gap,
                interpretation='four variational packets; no isolated full-rotor quartet or tunnelling certificate')


def optimized_packet_bound(controls, kappa, alignment):
    objective = lambda radius: packet_bound(controls, kappa, alignment, float(radius))['four_trial_Rayleigh_upper']
    fit: Any = minimize_scalar(objective, bounds=(1e-6, 1.), method='bounded', options={'xatol':1e-12})
    if not fit.success:
        raise RuntimeError('Packet radius minimization failed')
    radius = float(fit.x) if objective(float(fit.x)) < objective(1.) else 1.
    result = packet_bound(controls, kappa, alignment, radius)
    eta = .5
    raw_probability = CENTRE_DISTANCE_CONSTANT * result['fourth_full_rotor_eigenvalue_upper'] / (
        alignment * controls['ell_min'] * eta ** 2)
    result['low_four_outside_chord_radius'] = eta
    result['low_four_outside_probability_upper'] = min(1., raw_probability)
    result['unclipped_outside_probability_upper'] = raw_probability
    return result


def finite_probe(cutoff, frame, kappa, alignments):
    link = HaarLink(cutoff)
    provider = ExactHaarAlignment(link, frame)
    w = provider.matrix.toarray()
    labels = np.concatenate([np.full(link.reps[n].shape[1] ** 2, link.centre[n]) for n in link.names])
    casimir = np.concatenate([np.full(link.reps[n].shape[1] ** 2, link.casimir[n]) for n in link.names])
    require_small(norm(w - w.conj().T), 'finite projected W Hermiticity', 2e-8)
    require_small(norm(w[labels[:, None] != labels[None, :]]), 'finite centre conservation', 2e-8)
    w_min = float(np.linalg.eigvalsh(w)[0])
    if w_min <= 0:
        raise AssertionError('Finite standalone-W locking obstruction did not occur')
    runs = []
    for alignment in alignments:
        h = alignment * w + np.diag(kappa * casimir)
        grounds = {}
        sectors, all_low = [], []
        for charge in range(4):
            indices = np.flatnonzero(labels == charge)
            if not len(indices):
                sectors.append(dict(charge=charge, represented=False))
                continue
            values, vectors = eigh(h[np.ix_(indices, indices)], check_finite=False)
            tolerance = 1e-8 * max(1., abs(float(values[0])))
            ground = values - values[0] <= tolerance
            grounds[charge] = (indices, vectors[:, ground])
            sectors.append(dict(charge=charge, represented=True, dimension=len(indices),
                                lowest_eigenvalues=values[:min(8, len(values))].tolist(),
                                ground_multiplicity=int(ground.sum()),
                                first_excitation_above_ground=float(values[~ground][0] - values[0])
                                    if np.any(~ground) else None))
            all_low.extend((float(value), charge) for value in values[:8])
        transport = []
        for charge in range(4):
            target = (charge + 1) % 4
            if charge not in grounds or target not in grounds:
                transport.append(dict(source_charge=charge, target_charge=target, represented=False))
                continue
            si, sv = grounds[charge]
            ti, tv = grounds[target]
            matrices = np.array([tv.conj().T @ (u[ti][:, si] @ sv) for row in link.U for u in row])
            strength = float(np.sum(np.abs(matrices) ** 2) / sv.shape[1])
            rank = int(np.linalg.matrix_rank(matrices.reshape(16 * tv.shape[1], sv.shape[1]), tol=1e-9))
            transport.append(dict(source_charge=charge, target_charge=target, represented=True,
                                  source_ground_dimension=sv.shape[1], target_ground_dimension=tv.shape[1],
                                  summed_Uij_squared_per_source_state=strength, stacked_transport_rank=rank,
                                  ideal_localized_one_state_strength=4.))
        ordered = sorted(all_low)
        single_per_charge = len(grounds) == 4 and all(v.shape[1] == 1 for _, v in grounds.values())
        energies = [sector['lowest_eigenvalues'][0] for sector in sectors if sector['represented']]
        runs.append(dict(J=alignment, sectors=sectors, fundamental_ground_transport=transport,
                         lowest_eight=[dict(energy=e, charge=q) for e, q in ordered[:8]],
                         one_ground_state_in_each_charge=single_per_charge,
                         centre_ground_energy_spread=max(energies) - min(energies),
                         projected_H_lower_bound_J_times_Wmin=alignment * w_min))
    return provider, dict(cutoff=cutoff, irreps=list(link.names), dimension=link.dimension,
                          standalone_W_Ritz_minimum=w_min, continuum_W_infimum=0.,
                          fixed_cutoff_lower_bound_diverges_linearly_with_J=True,
                          spectra=runs, interpretation='finite exact Haar Ritz probes, not continuum band proof')


def build(args):
    band = RotationBand(a=1., m=args.m, r=args.r, v=args.v, improved=False)
    data = uniform_data(band, 3, args.stiffness, 0., matrix_unit_basis())
    frame = np.asarray(data['frame'], dtype=complex)
    generators = HaarLink('minimal').T
    controls = commutator_controls(frame, generators)
    continuous = continuous_controls(frame, generators, controls)
    providers, probes = {}, []
    for cutoff in args.cutoffs:
        providers[cutoff], evidence = finite_probe(cutoff, frame, args.kappa, args.alignments)
        probes.append(evidence)
    nested = []
    for small_name, large_name in itertools.permutations(providers, 2):
        small, large = providers[small_name], providers[large_name]
        if small.link.dimension >= large.link.dimension or not set(small.link.names) <= set(large.link.names):
            continue
        indices = np.concatenate([large.link.offset[n] + np.arange(small.link.reps[n].shape[1] ** 2)
                                  for n in small.link.names])
        error = norm(large.matrix[indices][:, indices] - small.matrix)
        require_small(error, 'nested exact Haar W projections', 2e-8)
        nested.append(dict(smaller=small_name, larger=large_name, residual=error))
    packet = [optimized_packet_bound(controls, args.kappa, j) for j in args.packet_alignments]
    projection_obstructions = []
    for evidence in probes:
        for bound in packet:
            finite_lower = bound['J'] * evidence['standalone_W_Ritz_minimum']
            projection_obstructions.append(dict(cutoff=evidence['cutoff'], J=bound['J'],
                finite_all_states_energy_lower=finite_lower,
                continuum_fourth_eigenvalue_upper=bound['fourth_full_rotor_eigenvalue_upper'],
                margin=finite_lower - bound['fourth_full_rotor_eigenvalue_upper'],
                all_finite_states_above_four_continuum_levels=
                    finite_lower > bound['fourth_full_rotor_eigenvalue_upper']))
    here = Path(__file__).resolve()
    hashes = source_hashes()
    for filename in (here.name, here.with_suffix('.md').name,
                     'gauged_rotation_fractionalization.py', 'gauged_rotation_fractionalization.md'):
        path = here.parent / filename
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(model='Continuous-Haar selected-Clifford-frame scalar centre-well rotor',
                parameters={key:value for key,value in vars(args).items() if key != 'output'},
                selected_source='local_orbital_stability.uniform_data(size=3,preferred=0); unimproved RotationBand',
                selected_frame=dict(real=np.real(frame).tolist(), imaginary=np.imag(frame).tolist()),
                selected_gram_diagonal=np.diag(data['selected']['G']).tolist(),
                selected_gram_stationarity_residual=float(data['selected']['gradient_residual']),
                declared_quantum_law='H = kappa*(-Delta_SU4) + J*sum_mu ||[Phi_mu,U]||_HS^2 on L2(SU4,dHaar)',
                normalization='Tr(Ta Tb)=delta_ab/2; ds^2=2Tr(dX^2); -Delta=sum_a Ea^2; no second right Casimir',
                commutator_controls=controls,
                global_bounds=dict(W_lower='ell_min*(4-|TrU|^2/4)', W_upper='ell_max*(4-|TrU|^2/4)',
                                   centre_chord_distance_squared_constant=CENTRE_DISTANCE_CONSTANT,
                                   weaker_requested_constant=4 + np.pi ** 2,
                                   W_kernel_points=['I','iI','-I','-iI'],
                                   multiplication_infimum=0., multiplication_zero_L2_eigenvectors=0),
                continuous_controls=continuous, finite_cutoff_probes=probes,
                nested_projection_controls=nested, certified_disjoint_Haar_packets=packet,
                finite_projection_obstructions=projection_obstructions,
                evidenced_clock_replacement='exact four-centre Haar trial-space transport with bounded amplitude/leakage, not dynamical low-band replacement',
                open_obligations=['uniform full-rotor fifth-eigenvalue lower bound and four-band separation',
                                  'tunnelling and comparison of spectral-band transport with the constructed trial packets',
                                  'replace scalar Casimir law by actual matter-coupled angular photon Hamiltonian',
                                  'spatial Gauss-invariant dynamics and an operational deconfinement/rotation test'],
                excluded_claims=['derived fundamental quantum law','full60 quantum solution','mechanical half-spin',
                                 'deconfined Higgs/Z4 phase','quantum3D phase','canonical/full QED'],
                source_sha256=hashes, runtime=dict(numpy=np.__version__, scipy=scipy.__version__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoffs', nargs='+', choices=tuple(CUTOFFS), default=list(CUTOFFS))
    parser.add_argument('--alignments', nargs='+', type=float, default=[0., .2, 2., 20., 200.])
    parser.add_argument('--packet-alignments', nargs='+', type=float, default=[200., 1e4, 1e6, 1e8, 1e10])
    parser.add_argument('--kappa', type=float, default=1.)
    parser.add_argument('--m', type=float, default=.2)
    parser.add_argument('--r', type=float, default=.12)
    parser.add_argument('--v', type=float, default=.3)
    parser.add_argument('--stiffness', type=float, default=2.)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    numbers = [args.kappa, args.m, args.r, args.v, args.stiffness,
               *args.alignments, *args.packet_alignments]
    if not all(np.isfinite(value) for value in numbers):
        parser.error('All coefficients must be finite')
    if min(args.kappa, args.stiffness, *args.packet_alignments) <= 0 or min(args.alignments) < 0:
        parser.error('Positive kappa/stiffness/packet J and nonnegative probe J required')
    result = build(args)
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.write_text(payload + '\n')
    print(payload)


if __name__ == '__main__':
    main()
