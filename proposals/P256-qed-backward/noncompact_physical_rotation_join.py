#!/usr/bin/env python3
"""Spatial-scalar microscopic rotation, complete Gauss flux, and packet spin lift.

The finite cube action stays linear. D is the derived body-chart action AFTER
spatial/vector/link rotation, never an extra microscopic matter spin assignment.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'

import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
from scipy.special import gammaincc

from bulk_frame_compton import raw_h
from bulk_quantum_selection import BulkQuantumLaw
from noncompact_charged_packet_transport import (
    bundle_evolve, divergence, graph, green, packet_seed, source_gram, source_marks, weak_family,
)
from noncompact_mechanical_pulse import packet_marks
from noncompact_native_metric_admission import certificate
from original_charged_principal import OriginalSpatial, original_frame, portable
from original_intrinsic_angular_transfer import MASS, field_rotation_tangents
from original_physical_rotation_lift import (
    QUARTER_TURN, make_lift, recover_gamma, restore_links,
)
from original_total_angular_transfer import gauge_move
from physical_spin_boundary import native_current, physical_cubic
from rotation_band import RotationBand, spatial_rotation
from unrestricted_field_minimum import coordinates

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-synthesis-continuation-20261006'
TEMPLATE = '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'
SOURCES = (
    'noncompact_physical_rotation_join.py', 'noncompact_physical_rotation_join.md',
    'physical_spin_boundary.py', 'original_total_angular_transfer.py',
    'original_intrinsic_angular_transfer.py', 'original_physical_rotation_lift.py',
    'noncompact_mechanical_pulse.py', 'noncompact_charged_packet_transport.py',
    'noncompact_marked_metric.py', 'original_charged_principal.py',
    'bulk_frame_compton.py', 'rotation_band.py',
    'noncompact_native_metric_admission.py', 'noncompact_wilson_photon_envelope.py',
    'noncompact_photon_packet_supplier.py',
)


def accept(error: float, label: str, tolerance: float = 3e-8) -> float:
    value = float(error)
    if not np.isfinite(value) or value > tolerance:
        raise AssertionError(f'{label}: {value} > {tolerance}')
    return value


def rejected(error: float, label: str, tolerance: float = 3e-8) -> dict[str, Any]:
    try:
        accept(error, label, tolerance)
    except AssertionError as exc:
        return dict(defect=float(error), assertion=str(exc), rejected=True)
    raise AssertionError(f'adverse construction passed: {label}')


def spin_geometry(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full-ten Gram whitening; the half factor is fixed by physical vector action."""
    g = np.einsum('mij,nji->mn', frame, frame).real/4
    gamma = recover_gamma(g, frame)
    spins = np.asarray([-.5j*gamma[j]@gamma[k] for j, k in ((2, 3), (3, 1), (1, 2))])
    return g, gamma, spins


def physical_spin_intertwiner(frame: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Body restoration of a physical R. The principal lift has a sign seam.

    Apply positions, vector indices, photon vectors and oriented links FIRST.
    Use exp(-i angle axis.S) for a continuous lifted path, including its 2pi sign.
    A complete reference transforms by the conjugate matrix; its loop is +I.
    """
    _, gamma, _ = spin_geometry(frame)
    x, y, z, w = Rotation.from_matrix(rotation).as_quat(canonical=True)
    return (w*np.eye(4)-x*gamma[2]@gamma[3]-y*gamma[3]@gamma[1]-z*gamma[1]@gamma[2])


def rest_spin_frame(frame: np.ndarray) -> np.ndarray:
    """One anchored rest convention; no fresh degenerate spin basis at each p."""
    _, _, spins = spin_geometry(frame)
    values, modes = np.linalg.eigh(frame[0])
    rest = modes[:, values > 0]
    if rest.shape != (4, 2):
        raise ValueError('actual native rest fibre must have two positive modes')
    _, axes = np.linalg.eigh(rest.conj().T@spins[2]@rest)
    return rest@axes


def positive_spin_frame(frame: np.ndarray, momentum: np.ndarray,
                        band: RotationBand) -> np.ndarray:
    """Actual native positive cluster P+(h_Phi(p)), not an isotropic replacement.

    This is an external-spin/observable chart, NOT a filter of the CT source or H.
    The two spin labels use the same rest frame at every momentum.
    """
    rest = rest_spin_frame(frame)
    h = raw_h(band, frame, np.asarray(momentum))
    values, modes = np.linalg.eigh(h)
    if np.sum(values > 0) != 2 or np.min(abs(values)) <= 1e-12:
        raise ValueError('native positive spin chart requires the actual gap')
    occupied = modes[:, values > 0]
    projected = occupied@(occupied.conj().T@rest)
    gram = projected.conj().T@projected
    values, modes = np.linalg.eigh(gram)
    if values[0] <= 1e-12:
        raise ValueError('native positive chart lost overlap with its anchored rest fibre')
    return projected@((modes/np.sqrt(values))@modes.conj().T)


def packet_rotation(frame: np.ndarray, rotation: np.ndarray, momenta: np.ndarray,
                    wave: Any) -> np.ndarray:
    """D(R) psi(R^-1 p); wave is a continuum interpolation, not site interpolation."""
    d = physical_spin_intertwiner(frame, rotation)
    return np.asarray([d@wave(rotation.T@p) for p in momenta])


def mechanical_connection(law: Any, frame: np.ndarray, weight: np.ndarray,
                          axis: int = 2) -> tuple[np.ndarray, dict[str, Any]]:
    """Physical vector-index tangent projected with ALL field/link masses."""
    model = OriginalSpatial(law.band, law.size, law.basis,
                            coordinates(frame, law.basis).reshape(-1), np.zeros(3))
    tangent = field_rotation_tangents(frame, law.basis)[:, axis]
    fh = np.fft.fftn(weight.reshape((law.size,)*3), norm='ortho')
    coefficients = np.empty((law.size,)*3+(15,), complex)
    field_norm = link_norm = horizontal_norm = orthogonality = 0.
    for n in itertools.product(range(law.size), repeat=3):
        q = 2*math.pi*np.asarray(n)/law.size
        g = model.gauge_columns(q)
        y = tangent*fh[n]
        ell = np.linalg.solve(g.conj().T@(MASS[:, None]*g), g.conj().T@(MASS*y))
        residual = y-g@ell
        coefficients[n] = ell
        orthogonality = max(orthogonality, float(np.linalg.norm(g.conj().T@(MASS*residual))))
        field_norm += float(np.vdot(g[:60]@ell, g[:60]@ell).real)
        link_norm += float(np.vdot(g[60:]@ell, g[60:]@ell).real/.6)
        horizontal_norm += float(np.vdot(residual, MASS*residual).real)
    ec = np.asarray(np.fft.ifftn(coefficients, axes=(0, 1, 2), norm='ortho'),
                    dtype=np.complex128).reshape(law.volume, 15)
    accept(np.linalg.norm(ec.imag), 'actual real complete-Gauss rotation connection', 2e-7)
    ell = np.einsum('xa,aij->xij', ec.real, law.generators)
    accept(orthogonality, 'all15 mechanical mass-horizontal Gauss identity', 2e-7)
    return ell, dict(all_Gauss_generators=15, field_vertical_mass_norm_squared=field_norm,
        electric_link_flux_mass_norm_squared=link_norm,
        retained_horizontal_normal_mass_norm_squared=horizontal_norm,
        mass_horizontal_error=orthogonality)


def gauss_flux_consumer(law: Any, frame: np.ndarray) -> dict[str, Any]:
    """Derivatives of the genuine Wilson/Coulomb source, including reference."""
    rng = np.random.default_rng(22210641)
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    fields += .002*law.fields(rng.normal(size=(law.volume, 4, 15)))
    links = np.asarray([[expm(.03j*np.einsum('a,aij->ij', rng.normal(size=15), law.generators))
                         for _ in range(3)] for _ in law.sites])
    a = .04*rng.normal(size=(law.volume, 3))
    eta, _ = packet_seed(law, .85, np.asarray([.2, -.08, .04]))
    reference = 0
    marks = packet_marks(law, fields, links, a, eta, reference)
    weight = np.exp(-np.sum((law.sites-law.size//2)**2, axis=1)/3)
    ell, report = mechanical_connection(law, frame, weight)
    target = (1j*np.einsum('xij,axjr->axir', ell, marks.reshape(4, law.volume, 4, 4))
              -1j*np.einsum('axir,rj->axij', marks.reshape(4, law.volume, 4, 4), ell[reference])).reshape(marks.shape)
    step = 1e-5
    moved, field_only = [], []
    for sign in (-1, 1):
        gauge = np.asarray([expm(sign*step*1j*q) for q in ell])
        f, u, _ = gauge_move(law, fields, links, a, gauge, np.zeros(law.volume))
        moved.append(packet_marks(law, f, u, a, eta, reference))
        field_only.append(packet_marks(law, f, links, a, eta, reference))
    derivative = (moved[1]-moved[0])/(2*step)
    error = accept(np.linalg.norm(derivative-target), 'physical Gauss source/reference/flux derivative', 3e-7)
    field_derivative = (field_only[1]-field_only[0])/(2*step)
    source_term = 1j*np.einsum('xij,axjr->axir', ell, marks.reshape(4, law.volume, 4, 4)).reshape(marks.shape)
    flux_defect = np.linalg.norm(field_derivative-target)
    ref_defect = np.linalg.norm(source_term-target)
    # The ACTUAL physical Phi-vector tangent, decomposed into Gauss plus normal.
    # The normal includes opposite link flux; neither its derivative nor norm is zero.
    physical_df = np.zeros_like(fields)
    physical_df[:, 1] = -weight[:, None, None]*fields[:, 2]
    physical_df[:, 2] = weight[:, None, None]*fields[:, 1]
    vertical_df = 1j*(ell[:, None]@fields-fields@ell[:, None])
    vertical_du = 1j*(ell[:, None]@links-links@ell[law.target])
    horizontal_df = physical_df-vertical_df
    physical_values, horizontal_values = [], []
    for sign in (-1, 1):
        horizontal_links = np.asarray([[expm(-sign*step*vertical_du[x, j]@links[x, j].conj().T)@links[x, j]
                                        for j in range(3)] for x in range(law.volume)])
        physical_values.append(packet_marks(law, fields+sign*step*physical_df, links, a, eta, reference))
        horizontal_values.append(packet_marks(law, fields+sign*step*horizontal_df, horizontal_links, a, eta, reference))
    physical_derivative = (physical_values[1]-physical_values[0])/(2*step)
    horizontal_derivative = (horizontal_values[1]-horizontal_values[0])/(2*step)
    physical_error = accept(np.linalg.norm(physical_derivative-target-horizontal_derivative),
                            'actual physical Phi-index Noether equals Gauss plus normal/flux', 3e-7)
    report.update(full_source_Gauss_derivative_error=error,
        reference_connection_norm=float(np.linalg.norm(ell[reference])),
        omitted_electric_Wilson_flux=rejected(flux_defect, 'physical Gauss source/reference/flux derivative', 3e-7),
        omitted_completebar4_reference=rejected(ref_defect, 'physical Gauss source/reference/flux derivative', 3e-7),
        actual_physical_Noether_decomposition_error=physical_error,
        retained_horizontal_source_derivative_norm=float(np.linalg.norm(horizontal_derivative)),
        no_bare_matter_spin=True, reference_site=reference)
    return report


def screened_boundary(frame: np.ndarray, sizes: list[int]) -> dict[str, Any]:
    """Ordinary rigid plateau, NOT the nonclosed precompensated torque route."""
    g, _, _ = spin_geometry(frame)
    gs = float(np.trace(g[1:, 1:])/3)
    accept(np.linalg.norm(g-np.diag([g[0, 0], gs, gs, gs])), 'screening diagnostic actual isometric native frame', 2e-7)
    mass = .6*gs
    kappa = math.acosh(1+mass/4)
    rows = []
    for n in sizes:
        if n < 5 or n % 2 != 1:
            raise ValueError('screening torus requires oddN>=5')
        radius = n//4
        k = 2*math.pi*np.arange(n)/n
        q2 = 2-2*np.cos(k)
        window = np.empty(n)
        window[0] = 2*radius+1
        window[1:] = np.sin((2*radius+1)*k[1:]/2)/np.sin(k[1:]/2)
        pair = window[:, None]*window[None, :]
        denominator = mass+q2[:, None]+q2[None, :]
        phase = np.exp(1j*k*(n//2))
        reference_pair = pair*phase[:, None]*phase[None, :]
        source, reference_value = 0., 0j
        for i in range(n):
            source += window[i]*np.sum(pair/(denominator+q2[i]))
            reference_value += window[i]*phase[i]*np.sum(reference_pair/(denominator+q2[i]))
        source *= mass/n**3
        reference_value *= mass/n**3
        accept(abs(reference_value.imag), 'exact screened reference Fourier reality')
        local = abs(1-source)
        leakage = abs(float(reference_value.real))
        local_upper = min(1., 12*math.exp(-kappa*(radius+1)))
        ref_upper = min(1., 12*math.exp(-kappa*(n//2-radius)))
        accept(max(0., local-local_upper), 'true electric-flux screening local bound')
        accept(max(0., leakage-ref_upper), 'true electric-flux screening reference bound')
        rows.append(dict(size=n, rigid_plateau_radius=radius, source_coefficient=float(source),
                         fixed_RIGHT_coefficient=leakage, local_defect=float(local),
                         paid_local_upper=local_upper, paid_reference_upper=ref_upper,
                         actual_point_source_two_pi_bound=math.pi*(local+leakage)))
    target = 1e-3
    guard = math.ceil(math.log(24/target)/kappa)
    return dict(actual_gs=gs, screening_mass_squared=mass, exact_discrete_rate=kappa,
        ordinary_rigid_plateau_rows=rows, constructive_guard_cells=guard,
        local_plus_reference_generator_upper_at_guard=24*math.exp(-kappa*guard)/2,
        physical_two_pi_norm_upper_at_guard=2*math.pi*24*math.exp(-kappa*guard)/2,
        guard_scope='source-relative creator commutator; not extensive vacuum trace distance',
        coefficient_scope='actual N5-frame coefficient on independently sized screening graphs; not their true minimizing Grams',
        discarded_route='torque=f+Delta f/(.6gs) has a noninteger/nonclosed exterior2pi field curve')


def cubic_mobile_consumer(law: Any, selected: dict[str, Any], frame: np.ndarray,
                          centre: tuple[int, ...], time: float) -> dict[str, Any]:
    _, h, baseline = graph(law.size, centre, frame, law.band)
    eta, seed = packet_seed(law, .85, np.asarray([.2, -.08, .04]))
    reference = 0
    g, _, gd = green(law)
    harmonic = np.asarray([.03, -.02, .01])
    h = law.onebody(np.tile(frame, (law.volume, 1, 1, 1)), law.toron_links(np.pi/2*np.asarray(centre)), baseline+harmonic)
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    links = law.toron_links(np.pi/2*np.asarray(centre))
    marks = source_marks(law, fields, links, eta, reference, baseline, harmonic, g)
    rf, ru, ra, p = physical_cubic(law, fields, links, baseline+harmonic, QUARTER_TURN)
    rh = law.onebody(rf, ru, ra)
    covariance = accept(np.linalg.norm(rh-p@h@p.T), 'actual moving-source physical cubic covariance')
    values, modes = np.linalg.eigh(h)
    rv, rm = np.linalg.eigh(rh)
    evolved = bundle_evolve((values, modes), marks, time)
    scalar_rotated_marks = np.einsum('ij,ajr->air', p, marks)
    rotated_evolved = bundle_evolve((rv, rm), scalar_rotated_marks, time)
    packet_error = accept(np.linalg.norm(rotated_evolved-np.einsum('ij,ajr->air', p, evolved)),
                          'physical orbital moving-source covariance', 3e-7)
    lift = make_lift(law, np.asarray(centre), QUARTER_TURN, np.asarray(selected['G']), frame)
    restored_fields, restored_links = lift.fields(rf), restore_links(law, lift, ru)
    restored_h = law.onebody(restored_fields, restored_links, ra)
    lift_error = accept(np.linalg.norm(restored_h-lift.onebody(h)), 'actual spatial then Gauss body restoration', 3e-7)
    d = lift.compensator.conj().T
    internal = np.einsum('ij,ajr->air', np.kron(np.eye(law.volume), d), evolved)
    physical_body = np.einsum('xij,axjr->axir', lift.local,
                              np.einsum('ij,ajr->air', p, evolved).reshape(4, law.volume, 4, 4)).reshape(evolved.shape)
    internal_defect = np.linalg.norm(internal-physical_body)
    # Real coexact photon coordinates AND their electric tangent, oriented pullback.
    rng = np.random.default_rng(22210642)
    photon = .07*rng.normal(size=(law.volume, 3))
    electric = .09*rng.normal(size=(law.volume, 3))
    for vector in (photon, electric):
        chi = g@divergence(law, vector)
        vector -= chi[:, None]-chi[law.target]
        vector -= vector.mean(axis=0)
        accept(np.linalg.norm(divergence(law, vector)), 'actual coexact photon/electric native divergence')
    _, _, photon_r, _ = physical_cubic(law, fields, links, photon, QUARTER_TURN)
    _, _, electric_r, _ = physical_cubic(law, fields, links, electric, QUARTER_TURN)
    def photon_energy(a, e):
        curl = sum(np.sum((a[law.target[:, i], j]-a[:, j]-a[law.target[:, j], i]+a[:, i])**2)
                   for i, j in ((0, 1), (0, 2), (1, 2)))
        return float(np.sum(e*e)+curl)
    em_error = accept(abs(photon_energy(photon_r, electric_r)-photon_energy(photon, electric)),
                      'real quadratic photon coordinate/electric spatial rotation')
    direction = rng.normal(size=(law.volume, 3))
    j = law.onebody(fields, links, baseline+harmonic+1e-5*direction)
    j -= law.onebody(fields, links, baseline+harmonic-1e-5*direction)
    _, _, direction_r, _ = physical_cubic(law, fields, links, direction, QUARTER_TURN)
    jr = law.onebody(rf, ru, ra+1e-5*direction_r)-law.onebody(rf, ru, ra-1e-5*direction_r)
    current_error = accept(np.linalg.norm(jr-p@j@p.T)/(2e-5), 'same native Peierls photon-current rotation', 3e-7)
    projector = modes[:, values < 0]@modes[:, values < 0].conj().T
    ct_projector = np.eye(len(h))-projector
    metric = (source_gram(marks, projector)+source_gram(marks, ct_projector))/2
    accept(np.linalg.norm(metric-2*np.eye(4)), 'same unfiltered CT complete-reference initialG')
    ground_metric = source_gram(marks, projector)
    positive = marks-np.einsum('ij,ajr->air', projector, marks)
    hpositive = np.einsum('ij,ajr->air', h, positive)
    energy = np.einsum('air,bir->ab', positive.conj(), hpositive)
    metric_values, metric_modes = np.linalg.eigh(ground_metric)
    if metric_values[0] <= 1e-9:
        raise AssertionError('actual whole-sea complete-reference source metric singular')
    initial_inverse = (metric_modes/np.sqrt(metric_values))@metric_modes.conj().T
    normalized_energy = initial_inverse@energy@initial_inverse
    energy_error = accept(np.linalg.norm(normalized_energy-normalized_energy.conj().T),
                          'actual charged-minus-SAME-neutral full-CAR energy Hermitian form')
    # No final metric inverse. Full colour-reference and local boundary signs differ.
    _, _, spins = spin_geometry(frame)
    endpoint = expm(-2j*math.pi*spins[2])
    local_error = accept(np.linalg.norm(endpoint+np.eye(4)), 'derived physical local boundary2pi')
    complete_error = accept(np.linalg.norm(np.kron(endpoint, endpoint.conj())-np.eye(16)),
                            'full source completebar4 physical2pi is linear')
    # Execute four actual spatial quarter turns on THIS moving source, not just S.
    raw_f, raw_u, raw_a, raw_wave = fields.copy(), links.copy(), (baseline+harmonic).copy(), evolved.copy()
    body_f, body_u, body_a, body_wave = fields.copy(), links.copy(), (baseline+harmonic).copy(), evolved.copy()
    boundary_wave = evolved.copy()
    current_centre = np.asarray(centre)
    for _ in range(4):
        raw_f, raw_u, raw_a, raw_p = physical_cubic(law, raw_f, raw_u, raw_a, QUARTER_TURN)
        raw_wave = np.einsum('ij,ajr->air', raw_p, raw_wave)
        current_frame = body_f[0]
        current_gram = np.einsum('mij,nji->mn', current_frame, current_frame).real/4
        body_lift = make_lift(law, current_centre, QUARTER_TURN, current_gram, current_frame)
        pf, pu, body_a, body_p = physical_cubic(law, body_f, body_u, body_a, QUARTER_TURN)
        body_f, body_u = body_lift.fields(pf), restore_links(law, body_lift, pu)
        for wave in (body_wave, boundary_wave):
            moved_wave = np.einsum('ij,ajr->air', body_p, wave).reshape(4, law.volume, 4, 4)
            wave[:] = np.einsum('xij,axjr->axir', body_lift.local, moved_wave).reshape(wave.shape)
        body_wave = np.einsum('air,rj->aij', body_wave, body_lift.local[reference].conj().T)
        current_centre = (QUARTER_TURN@current_centre) % 4
    raw_loop = accept(np.linalg.norm(raw_wave-evolved), 'actual spatial moving-source four-quarter2pi linear loop')
    completed_loop = accept(np.linalg.norm(body_wave-evolved), 'actual Gauss restored fullbar4 four-quarter2pi loop', 3e-7)
    boundary_loop = accept(np.linalg.norm(boundary_wave+evolved), 'actual restored source-end four-quarter2pi sign', 3e-7)
    field_loop = accept(np.linalg.norm(raw_f-fields)+np.linalg.norm(raw_u-links)+np.linalg.norm(raw_a-baseline-harmonic),
                        'actual physical field/vector/link four-quarter2pi closure', 3e-7)
    return dict(size=law.size, centre=centre, time=time, initial_G=metric,
        source=seed, full4V_modes=len(h), whole_neutral_occupied=2*law.volume,
        reference_site=reference, reference_components=4, physical_cubic_H_error=covariance,
        actual_unreversed_H_moving_source_error=packet_error, body_restoration_H_error=lift_error,
        photon_coordinate_electric_rotation_error=em_error, same_native_current_rotation_error=current_error,
        internal_only=rejected(internal_defect, 'physical orbital moving-source covariance', 3e-7),
        local_boundary_two_pi_error=local_error, complete_reference_two_pi_error=complete_error,
        actual_spatial_moving_source_four_quarter_loop_error=raw_loop,
        actual_completed_Gauss_reference_four_quarter_loop_error=completed_loop,
        actual_source_end_body_four_quarter_minus_error=boundary_loop,
        actual_field_vector_link_four_quarter_closure_error=field_loop,
        omitted_reference_full_loop=rejected(np.linalg.norm(np.kron(endpoint, np.eye(4))-np.eye(16)),
                                             'full source completebar4 physical2pi is linear'),
        actual_ground_initial_G=ground_metric, charged_minus_neutral_energy_eigenvalues=np.linalg.eigvalsh(normalized_energy),
        charged_energy_Hermitian_error=energy_error,
        actual_real_photon_probe_energy=photon_energy(photon, electric), Green_diagonal=gd,
        scope='actual native3D whole-sea CAR comparator and real photon rotation, not a numerical coupled3D ground')


def continuum_packet_consumer(frame: np.ndarray, band: RotationBand) -> dict[str, Any]:
    """Actual band chart, full orbital pullback, moving packet-family spin labels."""
    g, gamma, spins = spin_geometry(frame)
    rest = rest_spin_frame(frame)
    rest_spins = np.asarray([rest.conj().T@s@rest for s in spins])
    gs = float(np.trace(g[1:, 1:])/3)
    b0, bs = math.sqrt(g[0, 0]), math.sqrt(gs)
    iso = np.asarray([b0*gamma[0], *(bs*gamma[j] for j in range(1, 4))])
    db = float(np.linalg.norm(frame-iso))
    axis = np.asarray([1., 2., -1.]); axis /= np.linalg.norm(axis)
    angle = .61
    rotation = spatial_rotation(axis, angle)
    d = physical_spin_intertwiner(frame, rotation)
    directions = np.asarray([[1., 2., -1.], [-2., 1., .5], [.4, -1., 2.]])
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    delta = 2e-4
    rows = []
    for radius in (.16, .08, .04, .02):
        momenta = radius*directions
        covariance = ward = spin_only = spin_map = casimir = 0.
        for p in momenta:
            h = raw_h(band, frame, p)
            covariance = max(covariance, float(np.linalg.norm(raw_h(band, frame, rotation@p)-d@h@d.conj().T, ord=2)))
            u = positive_spin_frame(frame, p, band)
            j2 = np.zeros_like(u)
            for j in range(3):
                ej = np.eye(3)[j]
                rp, rm = spatial_rotation(ej, delta), spatial_rotation(ej, -delta)
                up, um = positive_spin_frame(frame, rp@p, band), positive_spin_frame(frame, rm@p, band)
                du = (up-um)/(2*delta)
                ju = spins[j]@u-1j*du
                spin_map = max(spin_map, float(np.linalg.norm(ju-u@rest_spins[j], ord=2)))
                d2u = (up-2*u+um)/(delta*delta)
                j2 += spins[j]@spins[j]@u-2j*spins[j]@du-d2u
                tangent = np.cross(ej, p)
                dh = np.einsum('i,ijk->jk', tangent, native_current(band, frame, p, p))
                ward = max(ward, float(np.linalg.norm(spins[j]@h-h@spins[j]-1j*dh, ord=2)))
                spin_only = max(spin_only, float(np.linalg.norm(spins[j]@h-h@spins[j], ord=2)))
            casimir = max(casimir, float(np.linalg.norm(j2-.75*u, ord=2)))
        rho = band.v*bs*band.a**2*radius**3/6+band.r*b0*band.a**3*radius**4/24
        c = band.m+band.r*band.a*radius**2/2+band.v*radius
        upper = 2*rho+2*db*c
        accept(max(0., covariance-upper), 'paid actual native continuous physical covariance', 3e-8)
        rows.append(dict(momentum_radius=radius, covariance_operator_error=covariance,
            paid_covariance_upper=upper, full_L_plus_S_Ward_error=ward,
            internal_spin_only_Ward_defect=spin_only, native_orbital_aware_spin_intertwining_error=spin_map,
            native_centered_radial_J_squared_error=casimir))
    # An actual moving source family: envelope centre AND carrier rotate.
    radius = .08
    carrier = radius*np.asarray([.8, -.2, .1])
    position = np.asarray([2., -.7, .4])
    points = carrier[None]+radius*directions
    def wave(p):
        return np.exp(-np.dot(p-carrier, p-carrier)/(2*radius**2)-1j*np.dot(p, position))*positive_spin_frame(frame, p, band)
    transported = packet_rotation(frame, rotation, points, wave)
    internal = np.asarray([d@wave(p) for p in points])
    orbit_defect = np.linalg.norm(transported-internal)
    # Isotropic quadratic source gives exact spatial intertwining at every p.
    iso_rest = rest_spin_frame(iso)
    mu, speed, rw = band.m*b0, band.v*bs, band.r*b0
    def quadratic(p):
        mass = mu+band.a*rw*np.dot(p, p)/2
        energy = math.sqrt(mass*mass+speed*speed*np.dot(p, p))
        return math.sqrt((energy+mass)/(2*energy))*(iso_rest+speed*np.einsum('i,ijk->jk', p, gamma[1:])@iso_rest/(energy+mass))
    quadratic_error = max(float(np.linalg.norm(d@quadratic(rotation.T@p)-quadratic(p)@(iso_rest.conj().T@d@iso_rest))) for p in points)
    accept(quadratic_error, 'derived physical positive-packet orbital-spin intertwiner', 2e-7)
    moments = {n: radius**n*math.gamma((n+3)/2)/math.gamma(1.5) for n in (2, 3, 4, 6, 8)}
    gaussian_rms_remainder = speed*band.a**2*math.sqrt(moments[6])/6+rw*band.a**3*math.sqrt(moments[8])/24
    return dict(actual_full10_Gram=g, actual_frame_isotropic_distance=db, native_rows=rows,
        quadratic_orbital_spin_covariance_error=quadratic_error,
        moving_source_internal_only=rejected(orbit_defect, 'derived physical positive-packet orbital-spin intertwiner', 2e-7),
        packet_carrier=carrier, packet_position=position,
        Gaussian_density_moments=moments, Gaussian_all_momentum_RMS_lattice_remainder=gaussian_rms_remainder,
        Brillouin_tail_probability_upper=float(gammaincc(1.5, (math.pi/(band.a*radius))**2)),
        finite_positive_packet_energy_upper=mu+rw*band.a*moments[2]/2+speed*math.sqrt(moments[2]),
        interpretation='rest radial packet J2 approaches3/4; moving family rotates carrier/centre and has spin1/2 labels, not J2=3/4 about the fixed reference')


def all64_rotation_geometry(law: Any) -> list[dict[str, Any]]:
    """No well weights or scalar offsets replaced by an isometric anchor."""
    rows = []
    for centre in itertools.product(range(4), repeat=3):
        selected, frame = original_frame(law.band, law.size, np.asarray(centre)*math.pi/2, True)
        gram, gamma, spins = spin_geometry(frame)
        gs = float(np.trace(gram[1:, 1:])/3)
        iso = np.asarray([math.sqrt(gram[0, 0])*gamma[0],
                          *(math.sqrt(gs)*gamma[j] for j in range(1, 4))])
        d = np.stack([coordinates(1j*(t@frame-frame@t), law.basis).reshape(-1).real
                      for t in law.generators], axis=-1)
        physical = field_rotation_tangents(frame, law.basis)[:60].real
        ell = np.linalg.solve(d.T@d, d.T@physical)
        physical_normal = physical-d@ell
        colour = np.einsum('aj,aik->jik', ell, law.generators)
        rows.append(dict(centre=centre, actual_full10_Gram=gram,
            all15_field_inertia_min=float(np.linalg.eigvalsh(d.T@d)[0]),
            rest_physical_index_normal_remainder=float(np.linalg.norm(physical_normal)),
            actual_mechanical_connection_minus_standard_spin=float(np.linalg.norm(colour-spins)),
            frame_isotropic_distance=float(np.linalg.norm(frame-iso)),
            actual_original_scalar_value=selected['value'],
            exact_scalar_offset_owner='same scalar E_c retained; classical value is not its replacement'))
    return rows


def shared_history_contract(directory: Path) -> dict[str, Any]:
    """Consume worker-owned executed moments; never relabel two-leg echo as SO3."""
    manifest = json.loads((directory/'manifest.json').read_text())
    source = hashlib.sha256((ROOT/'noncompact_mechanical_pulse.py').read_bytes()).hexdigest()
    if source != manifest['source_sha256']:
        raise AssertionError('shared-clock source changed: request a new producer packet')
    receipt = json.loads((directory/manifest['successful_receipt']).read_text())
    payload_bytes = (directory/'attempt02-stdout.json').read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != receipt['stdout_sha256']:
        raise AssertionError('executed normal/heavy moment payload changed')
    payload = json.loads(payload_bytes)
    normals = payload['full_quantum_normal']
    packet_cases = payload['native_quantum_packet']['history_cases']
    joined = []
    for case in packet_cases:
        normal = next(row for row in normals if row['centre'] == [0, 0, 0] and row['Lambda'] == case['Lambda'])
        tau = normal['duration']
        accept(abs(case['duration']-tau), 'same source/normal/heavy/light/photon clock')
        accept(abs(case['full_H']['physical_time']-2*tau), 'ordinary H uses actual full rotation-comparison clock')
        if not case['ordinary_H_unreversed']:
            raise AssertionError('shared history reversed physical H')
        joined.append(dict(Lambda=case['Lambda'], tau=tau,
            normal_histories=normal['histories'], retained_affine_B_bound=normal['affine_B_norm_upper'],
            actual_full_H_error=case['full_H']['endpoint_operator_error'],
            full_reference_components=case['full_H']['full_reference_components'],
            whole_heavy_dimension=case['full_H']['whole_heavy_dimension'],
            heavy_unwrapped_bare_energy_phase=case['full_H']['heavy_unwrapped_bare_energy_phase'],
            actual_neutral_RIGHT_Gram=case['full_H']['RIGHT_Gram']))
    return dict(source_sha256=source, executed_proof_sha256=manifest['executed_proof_sha256'],
        successful_receipt=receipt, shared_clock='tau=Lambda^-1/32 for normal/heavy/light/photon and actual neutral RIGHT',
        executed_normal_packet_same_clock=joined,
        whole_heavy='all924 exterior states with actual Lambda h, not source-only or reversed H',
        normal='all11250 axes plus full affine B covariance majorant; no normal momentum discarded',
        scope=manifest['scope'], body_intertwiner_and_spatial_orbital_observable='this producer',
        no_claim_that_echo_is_a_single_SO3_curve=True)


def admitted_rotation_budget(law: Any, frame: np.ndarray, time: float = 1.) -> dict[str, Any]:
    """Actual a1 weak family: source/flux/Gram/lattice debts without huge arrays.

    Weighted microscopic normal/heavy constants are NOT replaced by this budget.
    """
    if law.band.a != 1.:
        raise ValueError('this joined admission uses the a1 supplier, not a cutoff-unit substitution')
    constants = weak_family(.01, .1, 1., 'tight-paid')
    cert: dict[str, Any] = certificate(64, 128)
    n = max(cert['N_metric'], constants['photon_phase_N_onset'])
    n += 1-n % 2
    sigma, carrier_norm = .02, .02
    support = math.ceil(8/sigma)
    relative_radius = 4*support
    tail = 3*(2*sigma/math.sqrt(math.pi)*math.exp(-(sigma*support)**2)+math.erfc(sigma*support))
    eta_l1 = ((1+math.sqrt(2*math.pi)/sigma)/(math.pi**.25/math.sqrt(sigma)))**3/math.sqrt(1-tail)
    ue, tau = constants['UE'], constants['endpoint_norm_upper']
    rho, mu, green_upper = constants['relative_Hessian_bound'], constants['mu_lower'], 13/16
    variance = math.sqrt(ue*math.sqrt(green_upper)/(constants['bare_speed']*(1-rho)))
    free_variance = math.sqrt(ue*math.sqrt(green_upper)/constants['bare_speed'])
    force = math.sqrt(math.sqrt(constants['CE']*4*tau/2)*green_upper)
    epsilon = 12*tau*(variance+free_variance)+2*ue*green_upper+3*ue*tau*tau/(mu*mu*n**3)
    harmonic_density = 2*constants['B_upper']+epsilon
    l1 = 2*eta_l1*math.exp(6*tau*time)
    first = math.sqrt(3)*(2*relative_radius+4*time*tau)
    second = 3*(2*relative_radius**2+8*time*tau*relative_radius+4*time*tau+8*time*time*tau*tau)
    electric = 2*math.sqrt(ue*harmonic_density)*first+ue*second/n**3
    source_error = time*((6*tau*variance+ue*(4*force+13/4))*l1+electric)/math.sqrt(2)
    seam_rate = math.log1p(mu/(24*tau))
    cp = 8*(constants['B_upper']+mu)/(math.pi*mu)
    seam_prefactor = (384*tau*(constants['B_upper']+mu)/(math.pi*mu*mu)
        +24*tau*cp*time*math.exp(6*tau*math.expm1(seam_rate)*time))/math.sqrt(2)
    seam_log = math.log(seam_prefactor)-seam_rate*(n//2-support-1)
    seam_error = math.exp(max(-700., seam_log))
    t_alias = math.exp(-.5*n)
    majorant = cert['analytic_strip']['matrix_gradient_integrand_HS_majorant']
    alias = majorant*(6*t_alias+2*t_alias**3)/(1-t_alias)**3
    distance = (cert['full_ten_gradient']['Frobenius_residual_upper']+cert['continuum_quadrature_error_upper']+alias)/2
    g0, gs = map(float, cert['trial_decimal'])
    gram_min = min(g0, gs)-distance
    frame_distance = distance/math.sqrt(gram_min)
    _, gamma, _ = spin_geometry(frame)
    iso = np.asarray([math.sqrt(g0)*gamma[0], *(math.sqrt(gs)*gamma[j] for j in range(1, 4))],
                     dtype=np.complex128)
    d0 = np.stack([coordinates(1j*(t@iso-iso@t), law.basis).reshape(-1).real
                   for t in law.generators], axis=-1)
    v0 = field_rotation_tangents(iso, law.basis)[:60].real
    dn, vn = float(np.linalg.norm(d0, ord=2)), float(np.linalg.norm(v0, ord=2))
    delta_d = math.sqrt(30)*frame_distance
    delta_a = (2*dn+delta_d)*delta_d
    delta_rhs = dn*frame_distance+delta_d*(vn+frame_distance)
    mass_min = (math.sqrt(float(np.linalg.eigvalsh(d0.T@d0)[0]))-delta_d)**2
    resolvent_linf = 15/mass_min
    ell_error = resolvent_linf*(delta_rhs+delta_a*resolvent_linf*dn*vn)
    geometry_error = 2*math.pi*math.sqrt(2)*ell_error
    kappa = math.acosh(1+.6*(gs-distance)/4)
    guard = math.ceil(math.log(24*math.pi/1e-3)/kappa)
    screen_error = 24*math.pi*math.exp(-kappa*guard)
    moment = lambda power: sigma*(math.gamma((power+3)/2)/math.gamma(1.5))**(1/power)+carrier_norm
    lattice = 2*time*(.3*math.sqrt(gs)*moment(6)**3/6+.12*math.sqrt(g0)*moment(8)**4/24)
    truncation = 4*math.sqrt(tail)
    explicit = 2*source_error+2*seam_error+geometry_error+screen_error+lattice+truncation
    if n//2 <= support+guard+relative_radius or explicit >= .01:
        raise AssertionError('joined effective rotation/source geometry is not admitted')
    omega_min, omega_max = .005, .01
    cj = 192*tau*tau*n**3
    photon_debt = ue*cj/omega_min**3
    photon_residual = math.sqrt(ue*cj/omega_min)
    photon = dict(normalized_link_volume_mode='sum_xj|f|²=1, physicala1 Peierls current',
        all_state_current_squared_upper=cj, omega_min=omega_min, omega_max=omega_max,
        external_photon_leg_scale=math.sqrt(ue), annihilation_debt=photon_debt,
        creation_Gram_lower=1., creation_Gram_upper_offset=photon_debt,
        photon_on_actual_charge_initial_depletion_upper=2*photon_debt,
        exact_charged_path_normalized_insertion_error=2*abs(time)*photon_residual,
        finite_prepared_photon_energy_upper=omega_max*(1+photon_debt)+photon_residual,
        coexact_non_cubic_projector_error_upper=(omega_max/constants['bare_speed'])**2/6,
        coexact_non_cubic_frequency_error_upper=constants['bare_speed']*(omega_max/constants['bare_speed'])**3/24,
        same_actual_Omega_and_fullbar4_RIGHT=True,
        no_unbounded_ladder_times_generic_source_norm_error=True,
        composition='exact charged path insertion bound; source-to-free comparison needs weighted ladder/current/normal jets, not added to the generic norm budget')
    return dict(a=1., photon_speed=constants['bare_speed'], UE=ue, K=constants['K'],
        same_family='weak_family(.01,.1,1,tight-paid)', N=n,
        inverse_temperature_lower=constants['photon_inverse_temperature_coefficient']*n*n,
        same_Omega_source_CT_initial_norm_squared=2., observation_time=time,
        source_Gaussian_sigma=sigma, carrier_norm=carrier_norm, finite_source_halfwidth=support,
        source_reference_coordinate_bound=relative_radius, rigid_plateau_guard=guard,
        paid_same_H_source_error_each_branch=source_error,
        all_shift_all64_Gram_distance_upper=distance, all_shift_alias_log_upper=math.log(12*majorant)-.5*n,
        paid_full15_flux_geometry_error=geometry_error, paid_reference_screen_two_pi_error=screen_error,
        paid_orbital_lattice_RMS_error=lattice, paid_source_spatial_truncation_error=truncation,
        paid_fixed_frame_seam_error=seam_error, seam_log_upper=seam_log,
        explicit_effective_bounded_rotation_error=explicit,
        finite_positive_carrier_energy_upper=.2*math.sqrt(g0)+.12*math.sqrt(g0)*moment(2)**2/2+.3*math.sqrt(gs)*moment(2),
        actual_neutral_well_weights='arbitrary actual CT weights; uniform allshift bound, no equal weights',
        same_Omega_photon_current_leg=photon,
        microscopic_normal_heavy_payment='additional fixed-N weighted source/angular graph-norm error; choose finite Lambda(N); coefficient not numerically supplied',
        full3D_coupled_ground_simulated=False, microscopic_total_J2_spectrum_claim=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=5)
    parser.add_argument('--spacing', type=float, default=1.)
    parser.add_argument('--time', type=float, default=.5)
    parser.add_argument('--screen-sizes', nargs='+', type=int, default=[17, 33, 65, 321])
    parser.add_argument('--mechanical-evidence', type=Path, default=Path('/tmp/issue222-mechanical-shared-clock-20261006'))
    args = parser.parse_args()
    if args.size < 5 or args.size % 2 != 1 or not 0 < args.spacing <= 1:
        raise ValueError('odd native size>=5 and fixed0<a<=1 required')
    band = RotationBand(a=args.spacing, m=.2, r=.12, v=.3, improved=False)
    law = BulkQuantumLaw(args.size, band=band)
    selected0, frame0 = original_frame(band, args.size, np.zeros(3), True)
    centre = (1, 2, 3)
    selected, frame = original_frame(band, args.size, np.asarray(centre)*math.pi/2, True)
    sources = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}
    result = dict(plan=PLAN, template=TEMPLATE,
        microscopic_matter_action='physical site permutation only; derived body lift follows physical index/Gauss flux',
        actual_Gauss_flux=gauss_flux_consumer(law, frame0),
        ordinary_rigid_screened_boundary=screened_boundary(frame0, args.screen_sizes),
        actual_mobile_physical_cube=cubic_mobile_consumer(law, selected, frame, centre, args.time),
        isometric_orbital_spin=continuum_packet_consumer(frame0, band),
        tilted_true_Gram_orbital_spin=continuum_packet_consumer(frame, band),
        all64_actual_rotation_geometry=all64_rotation_geometry(law),
        shared_normal_wholeheavy_history=shared_history_contract(args.mechanical_evidence),
        admitted_a1_bounded_rotation_budget=admitted_rotation_budget(law, frame0),
        source_hashes=sources, all_checks_pass=True, campaign_complete=False)
    print(json.dumps(portable(result), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
