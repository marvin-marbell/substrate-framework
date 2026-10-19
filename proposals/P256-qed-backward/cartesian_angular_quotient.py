#!/usr/bin/env python3
"""Cartesian-derived scalar angular principal form; not a full BO or charged theorem."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy.linalg import block_diag, expm

from full_band_gauss_section import FullBandGaussSection
from local_orbital_stability import uniform_data
from rotation_band import RotationBand
from unrestricted_field_minimum import (BARE, STIFFNESS, pair_clifford_reference,
                                       source_hashes as inherited_source_hashes)

ROOT = Path(__file__).resolve().parent
# Same HS commutator resolution as the inherited candidate receipt, NOT an
# eigenvalue floor. The numerical cube kernel is explicitly conditional.
COMMUTATOR_RESOLUTION = 1e-7


def source_hashes():
    names = ('cartesian_angular_quotient.py', 'cartesian_angular_quotient.md',
             'full_haar_field_packet.py', 'full_haar_field_packet.md',
             'quantum_su4_gauss.md', 'sea_selection.md',
             'gauss_heavy_bo.py', 'gauss_heavy_bo.md')
    return {**inherited_source_hashes(), **{
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}}


def check(value, label, tolerance=3e-8):
    if not np.isfinite(value) or value > tolerance:
        raise AssertionError(f'{label}: {value} > {tolerance}')


def relative_error(left, right):
    return float(np.linalg.norm(left - right) / max(1., np.linalg.norm(right)))


def adjoint_matrix(U, generators):
    """Ad_U in the source-owned Tr(T_a T_b)=delta_ab/2 coordinates."""
    return 2 * np.einsum('aij,bji->ab', generators,
                         U @ generators @ U.conj().T).real


def column_spaces(columns, tolerance=1e-9):
    """Image and its orthogonal complement; empty inputs retain all covectors."""
    if columns.shape[1] == 0:
        return np.zeros((15, 0)), np.eye(15), np.zeros(0)
    vectors, singular, _ = np.linalg.svd(columns, full_matrices=True)
    rank = int(np.sum(singular > tolerance))
    return vectors[:, :rank], vectors[:, rank:], singular


def tangent_metric(frame, basis, generators) -> dict[str, Any]:
    differential = np.array([1j * (t @ frame - frame @ t) for t in generators])
    # D maps the 15 angular velocities into the ACTUAL 60 Cartesian coordinates.
    D = (np.einsum('amij,bji->mba', differential, basis).real / 4).reshape(60, 15)
    G = D.T @ D
    hs = np.einsum('amij,bmji->ab', differential, differential).real / 4
    check(relative_error(G, hs), 'Cartesian versus HS/4 tangent metric')
    _, singular, right = np.linalg.svd(D, full_matrices=False)
    # ||dPhi||HS = 2 ||du||: use the inherited HS resolution on 2*singular.
    rank = int(np.sum(2 * singular > COMMUTATOR_RESOLUTION))
    positive, kernel = right[:rank].T, right[rank:].T
    inverse = (positive / singular[:rank]**2) @ positive.T
    resolved = (positive * singular[:rank]**2) @ positive.T
    kernel_residual = float(np.linalg.norm(D @ kernel))
    steps, errors, bad_errors = (2e-4, 1e-4), [], []
    direction = np.sin(.37 * np.arange(15) + .29)
    direction /= np.linalg.norm(direction)
    T = np.einsum('a,aij->ij', direction, generators)
    exact = np.einsum('a,amij->mij', direction, differential)
    for step in steps:
        plus, minus = expm(1j * step * T), expm(-1j * step * T)
        finite = (plus @ frame @ plus.conj().T - minus @ frame @ minus.conj().T) / (2 * step)
        errors.append(float(np.linalg.norm(finite - exact)))
        # Replacing T by B=sqrt(8)T must fail against the same primitive tangent.
        wrong_plus, wrong_minus = expm(1j * step * np.sqrt(8) * T), expm(-1j * step * np.sqrt(8) * T)
        wrong = (wrong_plus @ frame @ wrong_plus.conj().T - wrong_minus @ frame @ wrong_minus.conj().T) / (2 * step)
        bad_errors.append(float(np.linalg.norm(wrong - exact)))
    check(errors[-1] / max(1., np.linalg.norm(exact)), 'actual conjugation differential', 2e-8)
    if errors[0] > 1e-12 and errors[-1] > .35 * errors[0]:
        raise AssertionError('Conjugation central differences did not converge quadratically')
    if bad_errors[-1] <= .5 * np.linalg.norm(exact):
        raise AssertionError('Unnormalized primitive generator adverse control unresolved')
    return dict(D=D, G=G, resolved_G=resolved, inverse=inverse, positive=positive,
                kernel=kernel, singular=singular, rank=rank, kernel_residual=kernel_residual,
                tangent_controls=dict(steps=list(steps), central_difference_errors=errors,
                    wrong_B_instead_of_T_errors=bad_errors,
                    Cartesian_HS4_metric_residual=relative_error(G, hs),
                    discarded_kernel_Cartesian_tangent_norm=kernel_residual,
                    raw_to_resolved_G_norm=float(np.linalg.norm(G - resolved))))


def quotient_principal(metric, V, generators, *, Lambda, ue4) -> dict[str, Any]:
    """Return the scalar horizontal cometric, with its invariant-Haar form measure.

    Arrays remain arrays for a joined consumer. At singular points horizontal
    dimension is NOT asserted to be the dimension of a smooth quotient chart.
    """
    root = np.sqrt(Lambda)
    G, inverse, B, K = (metric[key] for key in ('G', 'inverse', 'positive', 'kernel'))
    R = adjoint_matrix(V, generators)
    check(np.linalg.norm(R.T @ R - np.eye(15)), 'actual adjoint orthogonality')
    vertical, S, vertical_singular = column_spaces(np.column_stack((K, R @ K)))
    Q = root * (inverse + R @ inverse @ R.T) + 2 * ue4 * np.eye(15)
    Qh = S.T @ Q @ S
    Mh = np.linalg.solve(Qh, np.eye(Qh.shape[0]))
    # Independent finite constrained minimum uses the positive orbit coordinates,
    # not an epsilon-regularized inverse of a 45-by-45 degenerate inertia.
    Mf = B.T @ G @ B / root
    M = np.asarray(block_diag(Mf, Mf, np.eye(15) / (2 * ue4)))
    A = np.column_stack((-B, R @ B, np.eye(15)))
    C = S.T @ A
    direct_Qh = C @ np.linalg.solve(M, C.T)
    check(relative_error(direct_Qh, Qh), 'direct product inertia dual versus quotient cometric')
    direction = np.sin(.41 * np.arange(S.shape[1]) + .17)
    direction /= np.linalg.norm(direction)
    multiplier = np.linalg.solve(Qh, direction)
    minimizer = np.linalg.solve(M, C.T @ multiplier)
    kkt = np.block([[M, -C.T], [C, np.zeros((len(direction), len(direction)))]])
    independent = np.linalg.solve(kkt, np.r_[np.zeros(len(M)), direction])[:len(M)]
    primal = float(minimizer @ M @ minimizer / 2)
    dual = float(direction @ multiplier / 2)
    check(relative_error(C @ minimizer, direction), 'constrained velocity map')
    check(relative_error(independent, minimizer), 'independent KKT constrained minimizer')
    check(abs(primal - dual) / max(1., abs(dual)), 'minimum metric versus dual form')
    covector = S @ direction
    p0, p1 = -covector, R.T @ covector
    # Minimum Cartesian gradients solving D^T grad=p directly validate the
    # inverse-orbit metric, without identifying the ambient and orbit Laplacians.
    D = metric['D']
    grad0 = np.linalg.lstsq(D.T, p0, rcond=COMMUTATOR_RESOLUTION / (2 * metric['singular'][0]))[0]
    grad1 = np.linalg.lstsq(D.T, p1, rcond=COMMUTATOR_RESOLUTION / (2 * metric['singular'][0]))[0]
    ambient = float(root * (grad0 @ grad0 + grad1 @ grad1) / 2 + ue4 * (covector @ covector))
    predicted = float(covector @ Q @ covector / 2)
    check(relative_error(D.T @ grad0, p0), 'first Cartesian covector constraint')
    check(relative_error(D.T @ grad1, p1), 'second Cartesian covector constraint')
    check(abs(ambient - predicted) / max(1., predicted), 'Cartesian dual kinetic form')
    wrong_sum_inertia = S.T @ ((G + R @ G @ R.T) / root + np.eye(15) / (2 * ue4)) @ S
    wrong_Qh = np.linalg.solve(wrong_sum_inertia, np.eye(len(direction)))
    additive_error = float(np.linalg.norm(wrong_Qh - Qh) / np.linalg.norm(Qh))
    if additive_error < .05:
        raise AssertionError('Additive-inertia adverse control unresolved')
    field_Qh = S.T @ (root * (inverse + R @ inverse @ R.T)) @ S
    bad_HS4 = float(np.linalg.norm(.25 * field_Qh - field_Qh) / np.linalg.norm(field_Qh))
    check(abs(bad_HS4 - .75), 'HS instead of HS/4 adverse magnitude')
    # The stabilizer of V under H-left/H-right is h intersect Ad_V h.
    # Compute it in the h-coordinate space (column_spaces is intentionally 15D).
    if K.shape[1]:
        _, intersection_singular, intersection_right = np.linalg.svd((np.eye(15) - (R @ K) @ (R @ K).T) @ K)
        intersection_rank = int(np.sum(intersection_singular > 1e-9))
        isotropy = K @ intersection_right[intersection_rank:].T
    else:
        isotropy = np.zeros((15, 0))
    # A smooth scalar differential at an isotropy point must also be fixed by
    # the connected slice isotropy, in addition to annihilating vertical orbits.
    slice_actions = []
    for coordinate in isotropy.T:
        t = np.einsum('a,aij->ij', coordinate, generators)
        infinitesimal = 2 * np.einsum('aij,bji->ab', generators,
                                     1j * (t @ generators - generators @ t)).real
        slice_actions.append(infinitesimal @ S)
    if slice_actions:
        _, slice_singular, slice_right = np.linalg.svd(np.concatenate(slice_actions), full_matrices=True)
        smooth_rank = int(np.sum(slice_singular > 1e-8))
        smooth_covectors = S @ slice_right[smooth_rank:].T
    else:
        smooth_covectors = S
    return dict(adjoint=R, horizontal_basis=S, vertical_basis=vertical,
                horizontal_cometric=Qh, horizontal_inertia=Mh, ambient_cometric=Q,
                vertical_singular_values=vertical_singular,
                horizontal_covector_dimension=S.shape[1],
                double_coset_orbit_dimension=vertical.shape[1],
                connected_point_isotropy_dimension=isotropy.shape[1],
                smooth_scalar_first_covector_basis=smooth_covectors,
                smooth_scalar_first_covector_dimension=smooth_covectors.shape[1],
                isotropy_caveat='Connected infinitesimal isotropy only; disconnected stabilizer invariance remains required. At enhanced isotropy the quotient has a slice/cone, not this many smooth coordinates.',
                controls=dict(constrained_minimum=primal, dual_velocity_metric=dual,
                    independent_KKT_residual=relative_error(independent, minimizer),
                    constraint_residual=relative_error(C @ minimizer, direction),
                    direct_product_dual_residual=relative_error(direct_Qh, Qh),
                    actual_Cartesian_dual_form=ambient, predicted_principal_form=predicted,
                    wrong_additive_inertia_relative_error=additive_error,
                    wrong_HS_without_division_by_four_field_relative_error=bad_HS4))


def gauge_controls(frame, V, generators, metric, *, Lambda, ue4):
    a = .37 * np.sin(.43 * np.arange(15) + .21)
    b = .29 * np.cos(.31 * np.arange(15) + .47)
    g0, g1 = (expm(1j * np.einsum('a,aij->ij', c, generators)) for c in (a, b))
    q0, q1, link = g0, g1, g0 @ V @ g1.conj().T
    relative = q0.conj().T @ link @ q1
    finite_error = float(np.linalg.norm(relative - V))
    check(finite_error, 'finite actual two-endpoint gauge quotient')
    R = adjoint_matrix(V, generators)
    common = a / np.linalg.norm(a)
    link_velocity = common - R @ common
    zero = -common + link_velocity + R @ common
    check(float(np.linalg.norm(zero)), 'common endpoint conjugation is gauge')
    unreduced = float(common @ metric['G'] @ common / np.sqrt(Lambda)
                      + link_velocity @ link_velocity / (4 * ue4))
    if unreduced <= 0:
        raise AssertionError('Common physical-rotor adverse control unresolved')
    z = 1j * np.eye(4)
    centre_field_error = float(np.linalg.norm(z @ frame @ z.conj().T - frame))
    # g0=z, g1=1 sends V to zV and leaves the scalar endpoint fields unchanged.
    check(centre_field_error, 'residual centre fixes actual frame')
    return dict(finite_two_endpoint_relative_link_error=finite_error,
                common_gauge_relative_velocity_norm=float(np.linalg.norm(zero)),
                gauge_orbit_unreduced_energy_if_misidentified_as_physical=unreduced,
                correct_quotient_cost_of_zero_relative_velocity=0.,
                residual_centre_field_error=centre_field_error,
                scalar_centre_links_same_gauge_orbit=True)


def clifford_spectrum(frame):
    amplitudes_squared = np.einsum('mij,mji->m', frame, frame).real / 4
    result = []
    for mask in range(1, 16):
        even = mask.bit_count() % 2 == 0
        anti = [bool(mask & (1 << mu)) == even for mu in range(4)]
        result.append(float(np.dot(anti, amplitudes_squared) / 2))
    return np.sort(result)


def serializable(value):
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return dict(real=value.real.tolist(), imag=value.imag.tolist())
        return value.tolist()
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def run(*, Lambda=1e6, ue4=.3, input_receipt=ROOT / 'unrestricted_field_minimum_receipt.json'):
    if not all(np.isfinite(x) and x > 0 for x in (Lambda, ue4)):
        raise ValueError('Positive finite Lambda and ue4 required')
    identities = source_hashes()
    input_receipt = Path(input_receipt).resolve()
    input_bytes = input_receipt.read_bytes()
    receipt = json.loads(input_bytes)
    expected_bare = {**BARE, 'stiffness': STIFFNESS, 'Jpref': .2}
    if receipt['bare_coefficients'] != expected_bare or not receipt['source_bindings_unchanged_during_run']:
        raise ValueError('Input must be the bound same-bare unrestricted-field receipt')
    cube = receipt['cube']['best_candidate']
    if cube['common_SU4_stabilizer_Lie_dimension'] != 1:
        raise ValueError('Input cube candidate is not the receipt-resolved U1 comparison')
    band = RotationBand(**BARE)
    model = FullBandGaussSection(band, heavy_scale=1.)
    basis, generators = model.basis, model.generators
    generator_norm = np.einsum('aij,bji->ab', generators, generators).real
    generator_error = float(np.linalg.norm(generator_norm - np.eye(15) / 2))
    check(generator_error, 'source-owned primitive Lie generator norm')
    check(float(np.linalg.norm(sum(t @ t for t in generators) - 15 / 8 * np.eye(4))),
          'actual fundamental Casimir 15/8')
    environmental = np.asarray(uniform_data(band, 3, STIFFNESS, 0., basis)['frame'], complex)
    pair, pair_proof = pair_clifford_reference(model, stiffness=STIFFNESS)
    cube_coordinates = np.asarray(cube['coordinate_values'], float)
    if cube_coordinates.shape != (4, 15) or not np.all(np.isfinite(cube_coordinates)):
        raise ValueError('Actual cube candidate must contain all 60 finite Cartesian coordinates')
    cube_frame = np.einsum('ma,aij->mij', cube_coordinates, basis)
    noncentral = expm(1j * np.einsum('a,aij->ij', .71 * np.sin(.37 * np.arange(15) + .29), generators))
    check(float(np.linalg.norm(noncentral.conj().T @ noncentral - np.eye(4))), 'actual SU4 unitarity')
    check(float(abs(np.linalg.det(noncentral) - 1)), 'actual SU4 determinant')
    centre_distance = float(min(np.linalg.norm(noncentral - 1j**k * np.eye(4)) for k in range(4)))
    if centre_distance < .1:
        raise AssertionError('Noncentral actual SU4 point control unresolved')
    reports = {}
    for name, frame, expected_rank in (('environmental_full_Clifford', environmental, 15),
                                        ('balanced_two_field_pair', pair, 12),
                                        ('open_cube_receipt_U1_candidate', cube_frame, 14)):
        metric = tangent_metric(frame, basis, generators)
        if metric['rank'] != expected_rank:
            raise AssertionError(f'{name}: resolved orbit rank {metric["rank"]} != {expected_rank}')
        spectrum_control = None
        if name != 'open_cube_receipt_U1_candidate':
            spectrum_control = relative_error(np.linalg.eigvalsh(metric['G']), clifford_spectrum(frame))
            check(spectrum_control, 'algebraic Clifford tangent metric spectrum')
        nonzero_G = metric['singular'][:metric['rank']]**2
        qlower = 2 * np.sqrt(Lambda) / float(max(nonzero_G)) + 2 * ue4
        qupper = 2 * np.sqrt(Lambda) / float(min(nonzero_G)) + 2 * ue4
        points = {}
        for label, V in (('centre', np.eye(4, dtype=complex)), ('noncentral', noncentral)):
            point = quotient_principal(metric, V, generators, Lambda=Lambda, ue4=ue4)
            eigenvalues = np.linalg.eigvalsh(point['horizontal_cometric'])
            check(max(0., qlower - float(eigenvalues[0])) / max(1., qlower), 'horizontal isotropic lower envelope')
            check(max(0., float(eigenvalues[-1]) - qupper) / max(1., qupper), 'horizontal isotropic upper envelope')
            point['horizontal_cometric_eigenvalues'] = eigenvalues
            point['gauge_controls'] = gauge_controls(frame, V, generators, metric, Lambda=Lambda, ue4=ue4)
            points[label] = point
        reports[name] = dict(frame=frame, Cartesian_coordinates=np.einsum('mij,aji->ma', frame, basis).real / 4,
            G_ab=metric['G'], resolved_G_ab=metric['resolved_G'], G_eigenvalues=np.linalg.eigvalsh(metric['G']),
            primitive_tangent_singular_values=metric['singular'],
            field_inertia_ab=metric['G'] / np.sqrt(Lambda),
            orbit_dimension=metric['rank'], kernel_dimension=15-metric['rank'],
            stabilizer_Lie_dimension=15-metric['rank'], kernel_basis=metric['kernel'],
            normal_Cartesian_dimension_per_endpoint=60-metric['rank'],
            stabilizer_status=('algebraic Clifford commutant' if name != 'open_cube_receipt_U1_candidate' else
                'receipt-resolution numerical U1 only; exact kernel of finite coordinate bytes not proved'),
            Clifford_G_spectrum_residual=spectrum_control,
            tangent_controls=metric['tangent_controls'], points=points,
            isotropic_envelopes=dict(cometric_lower=qlower, cometric_upper=qupper,
                Dirichlet_kappa_lower=qlower/2, Dirichlet_kappa_upper=qupper/2,
                horizontal_inertia_eigenvalue_lower=1/qupper, horizontal_inertia_eigenvalue_upper=1/qlower,
                scope='On horizontal scalar covectors; for cube conditional on the resolved U1 germ'))
    prior_hashes = receipt.get('source_sha256', {})
    mismatches = {name: dict(input_receipt=old, executing_source=identities.get(name))
                  for name, old in prior_hashes.items() if identities.get(name) != old}
    if source_hashes() != identities or input_receipt.read_bytes() != input_bytes:
        raise RuntimeError('Source/proof/input receipt changed during execution; refusing mixed-bound result')
    return serializable(dict(schema=1, source_sha256=identities,
        source_bindings_unchanged_during_run=True,
        input_receipt=dict(path=str(input_receipt), sha256=hashlib.sha256(input_bytes).hexdigest(),
            inherited_source_sha256=prior_hashes, inherited_to_executing_source_mismatches=mismatches,
            scope='Candidate coordinate bytes are inputs, not a rerun of its classical stationarity proof'),
        parameters=dict(Lambda=Lambda, ue4=ue4, bare=expected_bare),
        primitive_generator_norm_residual=generator_error,
        link_inertia=1/(2*ue4), link_Casimir_coefficient=ue4,
        noncentral_point=noncentral, noncentral_centre_chord_distance=centre_distance,
        pair_reference_derivation=pair_proof, frames=reports,
        scalar_quotient='H\\SU4/H with pushforward normalized Haar; full Clifford H=Z4 gives SU4/Z4',
        principal_form='1/2 xi^T [sqrt(Lambda)(G^+ + Ad_V G^+ Ad_V^T)+2 ue4 Id] xi; xi horizontal',
        limitations=['Fixed-mean tangential principal form only; no normal reduction/confinement theorem.',
            'No substitution of quotient Riemannian volume for invariant product-measure pushforward.',
            'No removal of matter Berry/Born terms, density drift, Gauss constraints or external boundaries.',
            'Four central scalar links are gauge equivalent, not a scalar physical quartet.',
            'Cube stabilizer is numerical/conditional, not an algebraic exact U1 proof.',
            'No charged gap, quantum ground, phase, physical spin or electron-photon conclusion.'],
        all_checks_pass=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--Lambda', type=float, default=1e6)
    parser.add_argument('--ue4', type=float, default=.3)
    parser.add_argument('--input-receipt', type=Path, default=ROOT / 'unrestricted_field_minimum_receipt.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(Lambda=args.Lambda, ue4=args.ue4, input_receipt=args.input_receipt)
    payload = json.dumps(result, indent=2, allow_nan=False)
    args.output.write_text(payload + '\n')
    print(payload)


if __name__ == '__main__':
    main()
