#!/usr/bin/env python3
"""Native finite-window, full-CAR two-photon connected scattering coefficient.

External photons are canonical creation/annihilation packets. This is not the
imposed-source frequency response and does not require a charged pole.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

from bulk_frame_compton import raw_paths
from noncompact_charged_packet_transport import (
    currents, divergence, graph, green, packet_seed, source_marks, theorem_bounds, weak_family,
)
from noncompact_maxwell_phase import native_jets
from noncompact_native_current import accept, adverse
from noncompact_photon_amputation_graph import native_graph_certificate
from noncompact_photon_packet_supplier import NativeMomentum, beam_field
from noncompact_photon_response import encoded
from noncompact_physical_rotation_join import positive_spin_frame
from original_charged_principal import original_frame
from rotation_band import RotationBand

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-synthesis-continuation-20261006'


def require(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)


def packet(law: Any, axis: int, speed: float, centre: np.ndarray) -> dict[str, Any]:
    """Two native curl eigenmodes: genuine finite frequency bandwidth."""
    modes = []
    central = max(1, round(law.size/5))
    for number, weight in ((central, 1.), (central+1, .25)):
        mode = np.eye(3, dtype=int)[axis]*number
        if law.size == 3 and number == central+1:
            mode = np.eye(3, dtype=int)[axis]+np.array([0, 0, 1])
        k = 2*math.pi*mode/law.size
        d = 1-np.exp(-1j*k)
        q = float(np.linalg.norm(d))
        polar = np.array([0., 0., 1.], complex)-d.conj()*d[2]/q**2
        polar /= np.linalg.norm(polar)
        modes.append(dict(mode=mode, k=k, polar=polar, omega=speed*q,
                          coefficient=weight*np.exp(-1j*k@centre), frequency_sign=1))
    norm = math.sqrt(sum(abs(row['coefficient'])**2 for row in modes))
    for row in modes:
        row['coefficient'] /= norm
    velocity = sum(abs(row['coefficient'])**2*speed*np.sin(row['k'])/(row['omega']/speed)
                   for row in modes)
    return dict(modes=modes, omega_min=min(row['omega'] for row in modes),
                omega_max=max(row['omega'] for row in modes), mean_group_velocity=velocity,
                external_state='a_dagger(packet) Omega with actual CCR Gram; no imposed drive')


def annihilation_packet(photon: dict[str, Any]) -> dict[str, Any]:
    return {**photon, 'modes': [{**row, 'mode': -row['mode'], 'k': -row['k'],
        'polar': row['polar'].conj(), 'coefficient': row['coefficient'].conjugate(),
        'frequency_sign': -1} for row in photon['modes']]}


def scattering_kernel(momentum: NativeMomentum, incoming: dict[str, Any],
                      outgoing: dict[str, Any], source: np.ndarray, duration: float,
                      nodes: int) -> dict[str, np.ndarray]:
    """Connected Schur/Wick open line, with BOTH particle and hole bands.

    For t>s, the connected charged-minus-neutral contraction is
    J2(t) Q J1(s) - J1(s) P J2(t). Both photon time histories contribute.
    The local mixed Peierls derivative enters with -i, not as a pole.
    """
    x, w = np.polynomial.legendre.leggauss(nodes)
    times, weights = (x+1)*duration/2, w*duration/2
    parts = {name: np.zeros_like(source) for name in ('contact', 'absorb_first',
             'emit_first', 'hole_absorb_first', 'hole_emit_first')}
    for time, weight in zip(times, weights):
        t = float(time)
        evolved = momentum.evolve(source, t)
        parts['contact'] += -1j*weight*momentum.evolve(
            momentum.contact(incoming, outgoing, t, evolved), -t)
        j_out_t = momentum.interaction_current(outgoing, t, source)
        j_in_t = momentum.interaction_current(incoming, t, source)
        for earlier, inner_weight in zip((x+1)*t/2, w*t/2):
            s, scale = float(earlier), weight*inner_weight
            j_in_s = momentum.interaction_current(incoming, s, source)
            j_out_s = momentum.interaction_current(outgoing, s, source)
            parts['absorb_first'] -= scale*momentum.interaction_current(
                outgoing, t, momentum.project(j_in_s, False))
            parts['emit_first'] -= scale*momentum.interaction_current(
                incoming, t, momentum.project(j_out_s, False))
            parts['hole_absorb_first'] += scale*momentum.interaction_current(
                incoming, s, momentum.project(j_out_t, True))
            parts['hole_emit_first'] += scale*momentum.interaction_current(
                outgoing, s, momentum.project(j_in_t, True))
    parts['full'] = sum(parts.values(), np.zeros_like(source))
    return parts


def literal_peierls(law: Any, h: np.ndarray, first: np.ndarray,
                    second: np.ndarray, x: float, y: float) -> csr_matrix:
    """Literal endpoint exponentials, independent of Taylor/contact code."""
    result = h.copy()
    for site in range(law.volume):
        for axis in range(3):
            target = int(law.target[site, axis])
            a, b = slice(4*site, 4*site+4), slice(4*target, 4*target+4)
            phase = x*first[site, axis]+y*second[site, axis]
            result[a, b] = h[a, b]*np.exp(1j*phase)
            result[b, a] = h[b, a]*np.exp(-1j*phase)
    return csr_matrix(result)


def exterior_ratio(momentum: NativeMomentum, h: np.ndarray, incoming: dict[str, Any],
                   outgoing: dict[str, Any], source: np.ndarray, detector: np.ndarray,
                   duration: float, steps: int, x: float, y: float,
                   subtract_neutral: bool = True, pre: np.ndarray | None = None,
                   post: np.ndarray | None = None, full_output: bool = False) -> Any:
    """Exact complete-sea bordered exterior determinant, no one-electron sea freeze.

    Z_charge/Z_neutral is the Schur complement. Keeping the occupied block
    explicitly makes omitting the neutral denominator a consumer-visible error.
    """
    law = momentum.law
    occupied = []
    for index in range(law.volume):
        for band in range(4):
            if momentum.occupied[index, band]:
                column = np.zeros((law.volume, 4, 1), complex)
                column[index, :, 0] = momentum.vectors[index, :, band]
                occupied.append(column)
    occ = np.concatenate(occupied, axis=2)
    prepared = occ if pre is None else occ+y*pre
    final = occ if post is None else occ+x*post
    initial = np.concatenate((prepared, source), axis=2)
    wave = momentum.to_position(initial).reshape(4*law.volume, -1)
    dt = duration/steps
    for step in range(steps):
        time = (step+.5)*dt
        f = beam_field(law, incoming, time, True)
        g = beam_field(law, outgoing, time, True)
        generator = literal_peierls(law, h, f, g, x, y)
        wave = expm_multiply(-1j*dt*generator, wave, traceA=-1j*dt*generator.diagonal().sum())
    wave = momentum.evolve(momentum.to_momentum(wave), -duration)
    filled, added = wave[:, :, :len(occupied)], wave[:, :, len(occupied):]
    sea_block = np.einsum('par,pas->rs', final.conj(), filled)
    left = np.einsum('par,pas->rs', detector.conj(), filled)
    right = np.einsum('par,pas->rs', final.conj(), added)
    direct = np.einsum('par,pas->rs', detector.conj(), added)
    schur = direct-left@np.linalg.solve(sea_block, right)
    if full_output:
        return added-np.einsum('par,rs->pas', filled, np.linalg.solve(sea_block, right))
    value = complex(np.trace(schur))
    return value if subtract_neutral else complex(np.linalg.det(sea_block))*value


def preparation_tangent(momentum: NativeMomentum, photon: dict[str, Any],
                        time: float) -> tuple[np.ndarray, float]:
    """Complete particle-hole Euclidean ground jet with actual lattice gaps."""
    occupied = []
    energies = []
    for index in range(momentum.law.volume):
        for band in range(4):
            if momentum.occupied[index, band]:
                column = np.zeros((momentum.law.volume, 4, 1), complex)
                column[index, :, 0] = momentum.vectors[index, :, band]
                occupied.append(column)
                energies.append(momentum.values[index, band])
    sea = np.concatenate(occupied, axis=2)
    result = np.zeros_like(sea)
    minimum = math.inf
    for row in photon['modes']:
        acted = momentum.interaction_current({'modes': [row]}, time, sea)
        coefficients = np.einsum('pba,pbr->par', momentum.vectors.conj(), acted)
        denominators = momentum.values[:, :, None]-np.asarray(energies)[None, None, :]+row['omega']
        positive = ~momentum.occupied
        minimum = min(minimum, float(denominators[positive].min()))
        require(minimum > 0, 'complete particle-hole plus real photon preparation gap')
        coefficients[positive] /= denominators[positive]
        coefficients[~positive] = 0
        result -= np.einsum('pab,pbr->par', momentum.vectors, coefficients)
    return result, minimum


def sea_columns(momentum: NativeMomentum) -> np.ndarray:
    columns = []
    for site in range(momentum.law.volume):
        for band in range(4):
            if momentum.occupied[site, band]:
                column = np.zeros((momentum.law.volume, 4, 1), complex)
                column[site, :, 0] = momentum.vectors[site, :, band]
                columns.append(column)
    return np.concatenate(columns, axis=2)


def current_integral(momentum: NativeMomentum, photon: dict[str, Any], wave: np.ndarray,
                     duration: float, nodes: int) -> np.ndarray:
    x, w = np.polynomial.legendre.leggauss(nodes)
    result = np.zeros_like(wave)
    for time, weight in zip((x+1)*duration/2, w*duration/2):
        result -= 1j*weight*momentum.interaction_current(photon, float(time), wave)
    return result


def prepared_reduced_output(data: dict[str, Any], duration: float, nodes: int
                            ) -> tuple[dict[str, np.ndarray], float]:
    """Verified reduced Schur operator, not an external-leg positive filter."""
    momentum = data['momentum']
    sea = sea_columns(momentum)
    pre, gp = preparation_tangent(momentum, data['outgoing'], 0.)
    post, gq = preparation_tangent(momentum, annihilation_packet(data['incoming']), duration)
    def X(wave: np.ndarray) -> np.ndarray:
        return np.einsum('par,rs->pas', pre, np.einsum('par,pas->rs', sea.conj(), wave))
    def Y(wave: np.ndarray) -> np.ndarray:
        return np.einsum('par,rs->pas', sea, np.einsum('par,pas->rs', post.conj(), wave))
    source = data['full_source']
    qsource = momentum.project(source, False)
    xu, yu = X(source), Y(source)
    a_xu = current_integral(momentum, data['incoming'], xu, duration, nodes)
    a_qu = current_integral(momentum, data['incoming'], qsource, duration, nodes)
    b_yu = current_integral(momentum, data['outgoing'], yu, duration, nodes)
    b_qu = current_integral(momentum, data['outgoing'], qsource, duration, nodes)
    real = scattering_kernel(momentum, data['incoming'], data['outgoing'], qsource, duration, nodes)
    parts = dict(real_time=momentum.project(real['full'], False),
        initial_ground=-momentum.project(a_xu, False)-X(a_qu),
        final_ground=-momentum.project(b_yu, False)-Y(b_qu),
        initial_photon_exchange=-X(yu)+Y(xu))
    parts['full'] = sum(parts.values(), np.zeros_like(source))
    return parts, min(gp, gq)


def initial_cloud_exchange(data: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    """Actual leading distinct-source charged cloud Gram; not forced to zero."""
    parts, _ = prepared_reduced_output(data, 0., 4)
    wave = parts['initial_photon_exchange']
    scalar = complex(np.vdot(data['full_detector'], wave))
    pre, _ = preparation_tangent(data['momentum'], data['outgoing'], 0.)
    post, _ = preparation_tangent(data['momentum'], annihilation_packet(data['incoming']), 0.)
    neutral = complex(np.vdot(post, pre))
    neutral_error = accept(abs(neutral), 'disjoint native photon cones neutral cloud momentum control', 3e-11)
    rejected = adverse(abs(scalar), 'complete distinct-source initial photon cloud coefficient', 3e-7)
    return dict(coefficient=encoded(scalar), output_norm=float(np.linalg.norm(wave)),
        neutral_exchange_coefficient=encoded(neutral), neutral_exchange_error=neutral_error,
        charged_exchange_not_factorized=True, missing_charged_cloud_rejected=rejected,
        actual_finite_UE_exchange_Gram_whitening='PhotonPacketSupplier-owned; leading cloud is not actual Gram'), wave


def preparation_contour(data: dict[str, Any], duration: float, steps: int) -> dict[str, Any]:
    """Full UNFILTERED input AND output source boundaries, with independent FD."""
    momentum, h = data['momentum'], data['h']
    nodes = max(8, min(32, steps//4))
    coarse, gap = prepared_reduced_output(data, duration, nodes)
    fine, _ = prepared_reduced_output(data, duration, 2*nodes)
    source, detector = data['full_source'], data['full_detector']
    pre, _ = preparation_tangent(momentum, data['outgoing'], 0.)
    post, _ = preparation_tangent(momentum, annihilation_packet(data['incoming']), duration)
    def independent(resolution: int) -> np.ndarray:
        return mixed_difference(lambda x, y: exterior_ratio(momentum, h, data['incoming'],
            data['outgoing'], source, detector, duration, resolution, x, y,
            pre=pre, post=post, full_output=True), 4e-4)
    fd, fd_fine = independent(steps), independent(2*steps)
    fd_error = float(np.linalg.norm(fd_fine-fd))
    quadrature = float(np.linalg.norm(fine['full']-coarse['full']))
    tol = max(3e-7, 4*fd_error+4*quadrature)
    match = accept(np.linalg.norm(fine['full']-fd_fine),
        'full unfiltered prepared Schur operator versus literal complete-exterior derivative', tol)
    scalar = complex(np.vdot(detector, fine['full']))
    missing_final_hole = adverse(abs(np.vdot(
        momentum.project(detector, True), fine['full'])), 'full unfiltered prepared recoil detector', tol)
    cloud, cloud_wave = initial_cloud_exchange(data)
    connected = fine['full']-cloud_wave
    connected_scalar = complex(np.vdot(detector, connected))
    projected = momentum.project(fine['full'], False)
    spin_frames = np.asarray([positive_spin_frame(data['frame'],
        p+np.asarray(data['centre'])*math.pi/(2*momentum.law.size), data['band'])
        for p in momentum.momenta])
    spin = np.sum(abs(np.einsum('pas,par->psr', spin_frames.conj(), projected))**2, axis=(0, 2))
    beta = math.ceil(math.log(1e14)/gap)
    tail = (np.linalg.norm(pre)+np.linalg.norm(post))*math.exp(-gap*beta)
    return dict(coefficient=encoded(scalar),
        exclusive_recoil_probability_leading_coefficient=abs(scalar)**2,
        positive_hard_charge_projection_probability_leading_coefficient=float(np.linalg.norm(projected)**2),
        physical_spin_resolved_positive_hard_coefficients=spin,
        physical_spin_chart_role='global calibrated observable chart; not a local Noether certificate',
        full_detector_negative_component_norm=float(np.linalg.norm(momentum.project(detector, True))),
        missing_final_negative_detector_rejected=missing_final_hole,
        initial_charged_cloud_exchange=cloud,
        initial_cloud_subtracted_coefficient=encoded(connected_scalar),
        exclusive_cloud_subtracted_probability_leading_coefficient=abs(connected_scalar)**2,
        physical_cloud_subtracted_amplitude_leading=encoded(data['constants']['UE']*connected_scalar),
        inclusive_POVM_lower_candidate='exclusive connected joint-source channel after ACTUAL exchange-Gram whitening',
        actual_exchange_Gram_whitened=False,
        real_time_only_coefficient=encoded(complex(np.vdot(detector, fine['real_time']))),
        initial_ground_branch_correction_norm=float(np.linalg.norm(fine['initial_ground'])),
        final_ground_branch_correction_norm=float(np.linalg.norm(fine['final_ground'])),
        initial_photon_exchange_cross_norm=float(np.linalg.norm(fine['initial_photon_exchange'])),
        missing_initial_ground_branch_rejected=adverse(np.linalg.norm(fine['initial_ground']),
            'full prepared contour coefficient', tol),
        missing_final_ground_branch_rejected=adverse(np.linalg.norm(fine['final_ground']),
            'full prepared contour coefficient', tol),
        missing_initial_photon_exchange_rejected=adverse(np.linalg.norm(fine['initial_photon_exchange']),
            'full prepared contour coefficient', tol),
        numerical_time_step_difference=fd_error, numerical_quadrature_difference=quadrature,
        independent_complete_exterior_error=match, numerical_error_budget=tol,
        complete_particle_hole_preparation_gap=gap, finite_Euclidean_preparation_time=beta,
        Euclidean_jet_tail_upper=float(tail),
        initial_unfiltered_source_hole_norm=float(np.linalg.norm(momentum.project(source, True))),
        actual_ground_photon_leading_branches_included=True,
        normalized_finite_UE_Dyson_remainder_certified=False,
        scope='full-source recoil contour with actual leading initial cloud subtraction; fixed harmonic fibre/body')


def mixed_difference(callback: Any, step: float) -> Any:
    return sum(s*t*callback(s*step, t*step) for s in (-1, 1) for t in (-1, 1))/(4*step**2)


def carrier_speed_squared(gram: np.ndarray) -> float:
    """mu times the mean Hessian of the actual native positive energy at rest."""
    spatial_schur = np.diag(gram)[1:]-gram[0, 1:]**2/gram[0, 0]
    return float(.3**2*np.mean(spatial_schur)+.2*.12*gram[0, 0])


def setup(size: int) -> dict[str, Any]:
    band = RotationBand(a=1., m=.2, r=.12, v=.3, improved=False)
    centre = (1, 2, 3)
    selected, frame = original_frame(band, size, np.asarray(centre)*math.pi/2, True)
    gram = np.asarray(selected['G'])
    # One N5-anchored bare photon family is fixed for EVERY resolution comparator.
    anchor = gram if size == 5 else np.asarray(original_frame(
        band, 5, np.asarray(centre)*math.pi/2, True)[0]['G'])
    speed = math.sqrt(carrier_speed_squared(anchor))
    constants = weak_family(.01, speed, 1., 'conservative')
    law, h, baseline = graph(size, centre, frame, band)
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    links = law.toron_links(np.asarray(centre)*math.pi/2)
    green_matrix, _, _ = green(law)
    eta, preparation = packet_seed(law, .8, np.array([1.2, 0., 0.]))
    origin = np.asarray(preparation['origin'])
    reference = int(law.index[tuple((origin+np.array([-2, -2, -2])) % size)])
    harmonic = np.zeros(3)
    marks = source_marks(law, fields, links, eta, reference, baseline, harmonic, green_matrix)
    alpha = np.array([1., .3j, -.2+.4j, .6]); alpha /= np.linalg.norm(alpha)
    momentum = NativeMomentum(law, h, frame, centre, harmonic)
    full_source = momentum.to_momentum(np.einsum('a,air->ir', alpha, marks))
    source = momentum.project(full_source, False)
    source_norm = float(np.linalg.norm(source))
    accept(abs(source_norm**2-2), 'complete bar4 native source initial CT norm2')
    source /= source_norm
    full_source /= source_norm
    transfer = 2*math.pi/size*np.array([1., -1., 0.])
    recoil_eta = eta*np.exp(1j*(law.sites-law.sites[reference])@transfer)
    recoil_marks = source_marks(law, fields, links, recoil_eta, reference, baseline, harmonic, green_matrix)
    full_detector = momentum.to_momentum(np.einsum('a,air->ir', alpha, recoil_marks))
    detector = momentum.project(full_detector, False)
    detector_norm = float(np.linalg.norm(detector))
    detector /= detector_norm
    full_detector /= detector_norm
    incoming = packet(law, 0, speed, origin)
    outgoing = annihilation_packet(packet(law, 1, speed, origin+np.array([0, 1, 0])))
    return dict(band=band, centre=centre, frame=frame, gram=gram, constants=constants,
                carrier_speed_squared=carrier_speed_squared(gram), family_anchor_N=5,
                law=law, h=h, momentum=momentum, eta=eta, reference=reference,
                source=source, detector=detector, full_source=full_source,
                full_detector=full_detector, incoming=incoming, outgoing=outgoing,
                transfer=transfer, origin=origin, source_norm=source_norm)


def conditional_remainder(data: dict[str, Any], duration: float,
                          coefficient_norm: float) -> dict[str, Any]:
    """Available SAME-family effective-state budgets, not a relative probability."""
    constants, law = data['constants'], data['law']
    charge = theorem_bounds(constants, law, data['eta'], data['reference'], duration)
    ue, tau = constants['UE'], constants['endpoint_norm_upper']
    omega_min = min(data['incoming']['omega_min'], data['outgoing']['omega_min'])
    omega_max = max(data['incoming']['omega_max'], data['outgoing']['omega_max'])
    # ||c_x^dagger T c_y||<=||T||_1<=4 tau, at EVERY photon/harmonic coordinate.
    # This deliberately finite-volume bound covers the actual ground, not Wick alone.
    current_norm = 8*tau*math.sqrt(3*law.volume)
    cj = current_norm**2
    debt = ue*cj/omega_min**3
    photon_residual = math.sqrt(ue*cj/omega_min)
    fmax = sum(abs(row['coefficient'])/math.sqrt(law.volume*row['omega'])
               for row in data['incoming']['modes'])
    onebody_current_norm = 6*tau*fmax
    commutator = 2*onebody_current_norm
    delta = law.volume*charge['neutral_excitation_density_upper']/2
    gentle = 2*math.sqrt(delta)+delta
    # Conservative multiband pull-through energy bound (triangle, no false eigenmode).
    depleted_excitation_energy = omega_max*debt+photon_residual*math.sqrt(debt)
    mixed_harmonic_energy = (depleted_excitation_energy
        +law.volume*charge['neutral_harmonic_energy_per_volume_upper']*debt)
    graph_admitted = law.size >= 5 and charge['admission']['photon_phase_volume_onset_met']
    graph_certificate = None
    preparation_error = exchange_error = source_amputation_error = None
    paid_graph_error = None
    if graph_admitted:
        graph_certificate = native_graph_certificate(law.size, duration, omega_min, omega_max,
                                                    source_data=data)
        require(graph_certificate['UE'] == ue and graph_certificate['K'] == constants['K'],
                'photon amputation graph uses the SAME native physical family')
        require(graph_certificate['actual_full_CAR_number'] == 2*law.volume,
                'photon graph retains the complete sea')
        boundary_error = graph_certificate['conditional_boundary_coefficient_error_upper']
        cloud_norm = graph_certificate['mixed_source_graph']['resolvent_packet_norm_upper']
        # One extra photon leg acts on the FULL CAR space, not only the one-body band.
        integrated_leg = duration*current_norm/math.sqrt(omega_min)
        preparation_error = 2*math.sqrt(2)*boundary_error*integrated_leg
        exchange_error = 4*boundary_error*cloud_norm+2*boundary_error**2
        source_amputation_error = (2*math.sqrt(2)*integrated_leg
            *graph_certificate['finite_time_amputation_source_error_upper'])
        paid_graph_error = preparation_error+exchange_error+source_amputation_error
    return dict(actual_coupled_charge_preparation=charge,
        actual_ground_current_variance_upper=cj, ground_current_bound_complete=True,
        current_bound_scope='finite N all-state bounded complete-CAR operator; scales with V',
        current_norm_upper=current_norm, photon_annihilation_debt_upper=debt,
        created_photon_Gram_lower=1., created_photon_Gram_upper_correction=debt,
        neutral_photon_preparation_residual_upper=photon_residual,
        normalized_neutral_photon_finite_time_error_upper=2*duration*photon_residual,
        joint_Gram_lower=2., joint_Gram_identity='fixed g: Gjoint=GB+Gram(a_g B_i Omega)>=2I',
        different_receiver_exchange_Gram_factorization=False,
        initial_photon_on_charge_annihilation_debt_upper=2*debt,
        evolved_charged_current_identity='J_g B_t Omega=B_t J_g Omega+[J_g,B_t]Omega',
        local_CAR_commutator_norm_upper=commutator,
        actual_Omega_sea_projection_delta_upper=delta,
        bounded_CAR_gentle_expectation_multiplier_upper=gentle,
        conditional_cumulant_norm=coefficient_norm,
        depleted_photon_excitation_energy_upper=depleted_excitation_energy,
        mixed_harmonic_kinetic_candidate_upper=mixed_harmonic_energy,
        mixed_harmonic_lower_bound_required='H-E0>=H_h-V epsilon_h in shifted coexact photon units',
        photon_source_amputation_graph=graph_certificate,
        photon_source_graph_admission=dict(admitted=graph_admitted, supplier_minimum_N=5,
            source_photon_phase_admitted=charge['admission']['photon_phase_volume_onset_met'],
            unavailable_scope='conditional finite-CAR diagnostic only; no native graph bound'),
        normalized_preparation_boundary_error_upper=preparation_error,
        normalized_initial_exchange_boundary_error_upper=exchange_error,
        normalized_source_amputation_error_upper=source_amputation_error,
        paid_graph_pieces_coefficient_error_upper=paid_graph_error,
        paid_graph_pieces_scope='conditional-cloud/source graph pieces only; NOT complete current-state flight/Dyson/inclusive error',
        physical_two_photon_leg_factor=ue,
        normalized_coefficient_remainder_certified=False,
        blocked_term='exact charged/neutral current-state flight remainder; actual harmonic expectation and receiver Gram; microscopic legal Lambda',
        leading_initial_photon_correction_order='UE, same as real-time two-current coefficient',
        observation_time=duration, photon_omega_min=omega_min,
        raw_state_comparison_is_not_relative_scattering_error=True)


def case(size: int, duration: float, nodes: int, independent: bool) -> dict[str, Any]:
    data = setup(size)
    law, h, momentum = data['law'], data['h'], data['momentum']
    source, detector = data['source'], data['detector']
    incoming, outgoing = data['incoming'], data['outgoing']
    coarse = scattering_kernel(momentum, incoming, outgoing, source, duration, nodes)
    fine = scattering_kernel(momentum, incoming, outgoing, source, duration, 2*nodes)
    coefficient = complex(np.vdot(detector, fine['full']))
    parts = {name: complex(np.vdot(detector, value)) for name, value in fine.items() if name != 'full'}
    quadrature = float(np.linalg.norm(fine['full']-coarse['full']))
    tolerance = max(2e-8, 4*quadrature)
    accept(quadrature, 'finite-window ordered photon quadrature', 2e-6)
    projected = momentum.project(fine['full'], False)
    inclusive_coefficient = float(np.linalg.norm(projected)**2)
    exclusive_coefficient = abs(coefficient)**2
    require(inclusive_coefficient > tolerance**2, 'nonforward inclusive connected coefficient is resolved')
    require(exclusive_coefficient > tolerance**2, 'recoil-marked connected coefficient is resolved')
    adverse_rows = {}
    for name in ('contact', 'absorb_first', 'emit_first', 'hole_absorb_first', 'hole_emit_first'):
        adverse_rows['missing_'+name] = adverse(abs(parts[name]), 'same complete-CAR cumulant', tolerance)
    adverse_rows['one_reference_colour'] = adverse(abs(coefficient-complex(np.vdot(
        detector[:, :, :1], fine['full'][:, :, :1]))), 'same complete-CAR cumulant', tolerance)
    # Derivative/current/contact have a separate native endpoint positive control.
    time = .37
    f = beam_field(law, incoming, time, True)
    first, _ = native_jets(law, h, f)
    exact = momentum.to_momentum(first@momentum.to_position(source).reshape(4*law.volume, 4))
    current_error = accept(np.linalg.norm(exact-momentum.current(incoming, time, source)),
                           'actual native photon current leg')
    onsite, paths = raw_paths(data['band'], data['frame'])
    del onsite
    endpoint_error = accept(max(np.linalg.norm(t-momentum.endpoints[int(np.flatnonzero(d)[0])]
        *np.exp(-1j*np.asarray(data['centre'])[int(np.flatnonzero(d)[0])]*math.pi/(2*size)))
        for d, t in paths), 'bulk raw path versus exact native endpoint')
    proof = None
    if independent:
        fd = mixed_difference(lambda x, y: exterior_ratio(momentum, h, incoming, outgoing,
            source, detector, duration, 64, x, y), 4e-4)
        fd_fine = mixed_difference(lambda x, y: exterior_ratio(momentum, h, incoming, outgoing,
            source, detector, duration, 128, x, y), 4e-4)
        fd_error = abs(fd_fine-fd)
        independent_tolerance = max(2e-7, 4*fd_error+4*quadrature)
        accept(abs(coefficient-fd_fine), 'complete exterior cumulant versus literal Peierls derivative', independent_tolerance)
        neutral_wrong = mixed_difference(lambda x, y: exterior_ratio(momentum, h, incoming, outgoing,
            source, detector, duration, 128, x, y, False), 4e-4)
        # At a nonforward recoil the neutral disconnected term may cancel exactly;
        # use the diagonal channel as the discriminating positive control.
        diagonal = scattering_kernel(momentum, incoming, annihilation_packet(incoming), source,
                                     duration, 2*nodes)
        diag_coefficient = complex(np.vdot(source, diagonal['full']))
        diagonal_wrong = mixed_difference(lambda x, y: exterior_ratio(momentum, h, incoming,
            annihilation_packet(incoming), source, source, duration, 128, x, y, False), 4e-4)
        neutral_rejection = adverse(abs(diag_coefficient-diagonal_wrong),
                                   'complete exterior cumulant versus literal Peierls derivative', independent_tolerance)
        proof = dict(literal_Peierls_mixed_derivative=encoded(fd_fine), coarse=encoded(fd),
                     ordered_cumulant=encoded(coefficient), time_step_difference=fd_error,
                     acceptance_tolerance=independent_tolerance, nonforward_unsubtracted=encoded(neutral_wrong),
                     neutral_diagonal_adverse=neutral_rejection)
        adverse_rows['missing_neutral_subtraction'] = neutral_rejection
    moved = momentum.to_position(source)
    density = np.sum(abs(moved)**2, axis=(1, 2))
    charge_current = currents(law, h, moved.reshape(4*law.volume, 4))
    derivative = momentum.to_position(-1j*np.einsum('pab,pbr->par', momentum.blocks, source))
    density_dot = 2*np.real(np.sum(moved.conj()*derivative, axis=(1, 2)))
    continuity = accept(np.linalg.norm(density_dot+divergence(law, charge_current)),
                        'same source/reference charge flux continuity')
    charge_velocity = charge_current.sum(axis=0)
    photon_velocity = incoming['mean_group_velocity']
    relative_velocity = float(np.linalg.norm(photon_velocity-charge_velocity))
    require(relative_velocity > 1e-7, 'actual incident relative flux is nonzero')
    incident_flux = relative_velocity/law.volume
    ue = data['constants']['UE']
    leg_factor = ue
    recoil_energy = float(np.vdot(detector, np.einsum('pab,pbr->par', momentum.blocks, detector)).real
                          -np.vdot(source, np.einsum('pab,pbr->par', momentum.blocks, source)).real)
    remainder = conditional_remainder(data, duration, float(np.linalg.norm(fine['full'])))
    spin_frames = np.asarray([positive_spin_frame(data['frame'],
        p+np.asarray(data['centre'])*math.pi/(2*size), data['band']) for p in momentum.momenta])
    spin_coefficients = np.einsum('pas,par->psr', spin_frames.conj(), projected)
    spin_probability = np.sum(abs(spin_coefficients)**2, axis=(0, 2))
    spin_error = accept(abs(float(spin_probability.sum())-inclusive_coefficient),
                        'physical positive-spin chart complete inclusive sum')
    return dict(N=size, a=1., volume=law.volume, modes=4*law.volume, occupied=2*law.volume,
        physical_band=dict(m=.2, r=.12, v=.3), true_Gram=data['gram'], centre=data['centre'],
        source_norm_squared=data['source_norm']**2, reference=law.sites[data['reference']],
        complete_reference_colours=4, source_filter=False, harmonic_comparator=np.zeros(3),
        incoming_photon=incoming, outgoing_photon=outgoing, physical_family=data['constants'],
        time=duration, energy_resolution=2*math.pi/duration, recoil_transfer=data['transfer'],
        charged_recoil_energy=recoil_energy, photon_frequencies_from_actual_curl=True,
        source_frequency_is_photon=False, photon_velocity=photon_velocity, charge_velocity=charge_velocity,
        actual_relative_velocity=relative_velocity, incident_flux_per_physical_area=incident_flux,
        flux_units='a=1; one normalized photon per volume; |vgamma-vcharge|/V; rate=P/T',
        coefficient=encoded(coefficient), parts={key: encoded(value) for key, value in parts.items()},
        inclusive_probability_leading_coefficient=inclusive_coefficient,
        exclusive_recoil_probability_leading_coefficient=exclusive_coefficient,
        physical_spin_resolved_inclusive_coefficients=spin_probability,
        physical_spin_chart_completeness_error=spin_error,
        physical_spin_chart_role='observable only; source and whole H remain unfiltered',
        physical_amplitude_leading=encoded(leg_factor*coefficient),
        log_physical_inclusive_probability_leading=2*math.log(leg_factor)+math.log(inclusive_coefficient),
        cross_section_leading_coefficient=inclusive_coefficient/(duration*incident_flux),
        quadrature_norm_difference=quadrature, coefficient_numerical_error_upper=tolerance,
        native_current_error=current_error, actual_bulk_endpoint_error=endpoint_error,
        source_flux_continuity_error=continuity, independent_full_CAR_proof=proof,
        coupled_ground_leading_preparation_contour=(preparation_contour(data, duration, 64)
            if independent else None),
        adverse=adverse_rows, coupled_inclusive_remainder=remainder,
        canonical_photon_units=photon_units(ue, incoming['omega_min']),
        speed_match=dict(c_eff_squared=data['carrier_speed_squared'], family_anchor_N=data['family_anchor_N'],
            photon_speed=data['constants']['bare_speed'],
            finite_N_carrier_speed_mismatch=math.sqrt(data['carrier_speed_squared'])-data['constants']['bare_speed'],
            role='one fixed N5 physical photon family; full-ten native Schur curvature and lattice velocities retained'),
        actual_coupled_nonzero_probability_certified=False,
        scope='executed native full-CAR finite-resolution leading connected cumulant; coupled coefficient bound not closed')


def photon_units(ue: float, omega: float) -> dict[str, Any]:
    """Vacuum matrix-element proof of the actual native canonical leg."""
    annihilator = np.diag(np.sqrt(np.arange(1, 4)), 1).astype(complex)
    creator = annihilator.conj().T
    vacuum = np.eye(4, dtype=complex)[:, 0]
    coordinate = math.sqrt(ue/omega)*(annihilator+creator)
    momentum = -1j*math.sqrt(omega/(4*ue))*(annihilator-creator)
    h = ue*momentum@momentum+omega**2/(4*ue)*coordinate@coordinate
    ccr = accept(np.linalg.norm((coordinate@momentum-momentum@coordinate)@vacuum-1j*vacuum),
                 'actual native photon coordinate/momentum vacuum CCR')
    free = accept(np.linalg.norm((h@creator-creator@h-omega*creator)@vacuum),
                  'actual native photon oscillator vacuum frequency')
    current = (coordinate@creator-creator@coordinate)@vacuum
    measured = float(np.vdot(vacuum, current).real)
    correct = math.sqrt(ue/omega)
    unit_error = accept(abs(measured/correct-1), 'actual photon creation current leg')
    wrong = adverse(abs(measured/math.sqrt(ue/(2*omega))-1), 'actual photon creation current leg')
    return dict(vacuum_CCR_error=ccr, free_frequency_error=free, current_leg_relative_error=unit_error,
        actual_coordinate_creation_element=measured, physical_two_leg_factor=ue,
        wrong_half_kinetic_convention_rejected=wrong)


def hashes() -> dict[str, str]:
    paths = {Path(__file__).resolve(), ROOT/'noncompact_inclusive_scattering_join.md'}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name and Path(name).resolve().parent == ROOT:
            paths.add(Path(name).resolve())
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[5, 7])
    parser.add_argument('--time', type=float, default=6.)
    parser.add_argument('--nodes', type=int, default=12)
    parser.add_argument('--independent', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(args.time > 0 and args.nodes >= 4 and all(n >= 3 for n in args.sizes),
            'finite native3D resolution inputs')
    before = hashes()
    rows = [case(n, args.time, args.nodes, args.independent) for n in args.sizes]
    require(hashes() == before, 'same source family unchanged throughout execution')
    result = dict(plan=PLAN, source_sha256=before,
        runtime=dict(executable=sys.executable, python=sys.version, numpy=np.__version__,
                     scipy=scipy.__version__, argv=sys.argv), results=rows,
        checks=True, coupled_scattering_closed=False)
    rendered = json.dumps(result, indent=2, allow_nan=False, default=encoded)
    args.output.write_text(rendered+'\n')
    print(rendered)


if __name__ == '__main__':
    main()
