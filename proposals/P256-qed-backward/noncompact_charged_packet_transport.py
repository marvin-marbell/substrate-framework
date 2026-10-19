#!/usr/bin/env python3
"""Unfiltered complete-reference CT packet in the actual harmonic free frame.

All 4V modes and the entire 2V sea remain in the Hamiltonian. Pauli exclusion,
not a source band filter, removes occupied creator components. The executable
measures native free/static-conditional states; the companion derivation pays
the coupled neutral-state, harmonic electric and reference obligations.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
import scipy

from bulk_quantum_selection import BulkQuantumLaw
from noncompact_maxwell_phase import phase_constants
from noncompact_marked_metric import spin_fields
from noncompact_wilson_photon_envelope import anchored_family, even_loop_family
from original_compact_charge_bound import chiral_basis, polar_derivative, sea_section
from original_charged_principal import original_frame, portable
from rotation_band import RotationBand

ROOT = Path(__file__).resolve().parent
PLAN_REVISION = '12df449908d0ef440fbcb78c8a4c7e43e378ebe84a465a8c7bd93556a76d257a'
TEMPLATE_REVISION = '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'
GREEN_3D_UPPER = 13 / 16


def accept(error: float, label: str, tolerance: float = 3e-8) -> float:
    error = float(error)
    if not math.isfinite(error) or error > tolerance:
        raise AssertionError(f'{label}: {error} > {tolerance}')
    return error


def require(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)


def reject(callback: Any, label: str) -> str:
    try:
        callback()
    except AssertionError as failure:
        return f'{label}: {failure}'
    raise AssertionError(f'Wrong construction passed the SAME assertion: {label}')


def source_hashes() -> dict[str, str]:
    names = ('noncompact_charged_packet_transport.py', 'noncompact_charged_packet_transport.md',
             'noncompact_maxwell_phase.py', 'noncompact_maxwell_phase.md',
             'noncompact_wilson_photon_envelope.py', 'noncompact_wilson_photon_envelope.md',
             'noncompact_native_metric_admission.py', 'noncompact_native_metric_admission.md',
             'bulk_quantum_selection.py', 'original_charged_principal.py',
             'original_compact_charge_bound.py', 'original_compact_charge_bound.md',
             'noncompact_marked_metric.py', 'noncompact_marked_metric.md',
             'noncompact_neutral_supplier.md', 'noncompact_microscopic_charge_window.md',
             'rotation_band.py')
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def weak_family(floor: float, speed: float, factor: float,
                phase_envelope: str = 'conservative') -> dict[str, Any]:
    require(math.isfinite(factor) and factor >= 1, 'one fixed positive weak-family multiplier')
    require(phase_envelope in ('conservative', 'universal-paid', 'tight-paid'), 'declared source-owned photon envelope')
    base = phase_constants(floor, speed)
    constants: dict[str, Any]
    if phase_envelope != 'conservative':
        tight = phase_envelope == 'tight-paid'
        if tight:
            require(floor <= .4, 'certified tight box supplies Schur floor .4, not a larger requested floor')
        envelope = even_loop_family(.4, speed, tight=True) if tight else anchored_family(floor, speed, tight=False)
        constants = dict(a=1., m=.2, r=.12, v=.3, kappa=2.,
            schur_lower=.4 if tight else floor,
            Gram_Frobenius_upper=base['Gram_Frobenius_upper'], mu_lower=envelope['mu'],
            Ch_upper=envelope['Ch'],
            B_upper=(.2+3*.12)*math.sqrt(envelope['G00_upper'])+envelope['Ch'],
            UE=envelope['UE'], K=envelope['K'],
            CE=envelope['CE'], CB=envelope['CB'], bare_speed=speed,
            electric_Hessian=1/(2*envelope['UE']), relative_Hessian_bound=envelope['rho'],
            photon_phase_N_onset=envelope['N0_seam'], photon_inverse_temperature_coefficient=envelope['tau'])
    else:
        constants = dict(base, photon_phase_N_onset=5,
            photon_inverse_temperature_coefficient=base['tau'])
    constants['phase_envelope'] = phase_envelope
    constants['endpoint_norm_upper'] = ((.12*math.sqrt(.43)+.3*math.sqrt(.065))/2
        if phase_envelope == 'tight-paid' else (.12+.3)*math.sqrt(base['Gram_Frobenius_upper'])/2)
    constants['native_metric_N_onset'] = 25 if floor <= .4 else None
    constants['UE'] /= factor
    constants['K'] *= factor
    constants['electric_Hessian'] *= factor
    constants['relative_Hessian_bound'] /= factor
    rho = constants['relative_Hessian_bound']
    constants['covariance_lower_ratio'] = 1/(1+rho)
    constants['covariance_upper_ratio'] = 1/(1-rho)
    constants['spectral_support_speed_upper'] = speed*(1+rho)/(1-rho)
    constants['weak_factor'] = factor
    return constants


def symbol(frame: np.ndarray, momenta: np.ndarray, angles: np.ndarray,
           size: int) -> np.ndarray:
    shifted = momenta + angles / size
    coefficient = np.column_stack((.2 + .12 * np.sum(1 - np.cos(shifted), axis=1),
                                   .3 * np.sin(shifted)))
    return np.einsum('xm,mab->xab', coefficient, frame)


def all64(size: int, band: RotationBand, floor: float) -> tuple[list[dict[str, Any]], dict[tuple[int, ...], Any]]:
    momenta = 2 * np.pi * np.indices((size, size, size)).reshape(3, -1).T / size
    reports: list[dict[str, Any]] = []
    frames = {}
    for centre in itertools.product(range(4), repeat=3):
        angles = np.asarray(centre) * np.pi / 2
        selected, frame = original_frame(band, size, angles, True)
        gram = np.asarray(selected['G'], dtype=float)
        schur = float(gram[0, 0] - gram[0, 1:] @ np.linalg.solve(gram[1:, 1:], gram[1:, 0]))
        require(schur >= floor - 2e-10, 'finite all64 native Schur floor (not an all-N certificate)')
        values = np.linalg.eigvalsh(symbol(frame, momenta, angles, size)).reshape(-1)
        require(int(np.sum(values < 0)) == 2 * size**3, 'whole 2V filled sea in EVERY true Gram well')
        sea = float(values[values < 0].sum())
        leading = float(np.sum(gram**2) + sea / size**3)
        error = accept(abs(leading - float(selected['value'])), 'actual true Gram/whole-sea leading well identity')
        reports.append(dict(centre=list(centre), true10Gram=gram, Schur=schur,
            mu=.2 * math.sqrt(schur), full4V_modes=4 * size**3, occupied=2 * size**3,
            canonical_dimension=str(math.comb(4 * size**3, 2 * size**3)),
            whole_filled_sea_energy=sea, native_leading_scalar_well_per_volume=leading,
            leading_well_identity_error=error, native_Gram_gradient_residual=selected['gradient_residual'],
            normal_and_other_scalar_offset=f'c_N,{centre}: retained by theorem; not computed by this comparator'))
        frames[centre] = (selected, frame)
    return reports, frames


def graph(size: int, centre: tuple[int, ...], frame: np.ndarray, band: RotationBand) -> tuple[Any, np.ndarray, np.ndarray]:
    law = BulkQuantumLaw(size, band=band)
    angles = np.asarray(centre) * np.pi / 2
    links = law.toron_links(angles)
    # Actual central SU4 seams are retained. A curl-free REAL U1 translation
    # redistributes the combined holonomy uniformly; no non-SU4 colour link.
    baseline = np.tile(angles / size, (law.volume, 1))
    baseline -= (law.sites == size - 1) * angles[None, :]
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    h0 = np.asarray(law.onebody(fields, links, baseline), dtype=complex)
    return law, h0, baseline


def green(law: Any) -> tuple[np.ndarray, np.ndarray, float]:
    k = 2 * np.pi * law.sites / law.size
    q2 = np.sum(abs(1 - np.exp(1j * k))**2, axis=1)
    inverse = np.zeros_like(q2)
    inverse[q2 > 1e-12] = 1 / q2[q2 > 1e-12]
    plane = np.exp(1j * law.sites @ k.T) / math.sqrt(law.volume)
    g = (plane * inverse[None, :]) @ plane.conj().T
    laplacian = (plane * q2[None, :]) @ plane.conj().T
    accept(float(np.linalg.norm(laplacian @ g - np.eye(law.volume) + 1 / law.volume)),
           'native 3D Moore-Penrose Green INCLUDING zero-mode quotient')
    return g.real, q2, float(inverse.mean())


def density_covariance(projector: np.ndarray, volume: int) -> np.ndarray:
    blocks = projector.reshape(volume, 4, volume, 4)
    number = np.array([np.trace(blocks[x, :, x, :]).real for x in range(volume)])
    return np.diag(number) - np.sum(abs(blocks)**2, axis=(1, 3))


def currents(law: Any, h: np.ndarray, packet: np.ndarray) -> np.ndarray:
    wave = packet.reshape(law.volume, 4, -1)
    result = np.zeros((law.volume, 3))
    for x in range(law.volume):
        for j in range(3):
            y = int(law.target[x, j])
            endpoint = h[4*x:4*x+4, 4*y:4*y+4]
            result[x, j] = -2 * float(np.einsum('ar,ab,br->', wave[x].conj(), endpoint, wave[y]).imag)
    return result


def divergence(law: Any, current: np.ndarray) -> np.ndarray:
    result = current.sum(axis=1).copy()
    for j in range(3):
        np.add.at(result, law.target[:, j], -current[:, j])
    return result


def packet_seed(law: Any, width: float, carrier: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    origin = np.full(3, law.size // 2)
    displacement = (law.sites-origin+law.size//2) % law.size-law.size//2
    squared = np.sum(displacement**2, axis=1)
    envelope = np.exp(-squared/(4*width**2))
    envelope[squared > (2*width)**2] = 0
    k0 = np.asarray(carrier, dtype=float)
    eta = envelope*np.exp(1j*displacement@k0)
    eta /= np.linalg.norm(eta)
    require(int(np.sum(envelope > 0)) > 1, 'genuine finite-width unfiltered source')
    return eta, dict(origin=origin, width=width, exact_seed_support_sites=int(np.sum(envelope > 0)),
        momentum=k0, carrier_fixed_in_lattice_units=True, creator_filter_applied=False, scalar_coefficients=eta,
        occupied_components='annihilated by complete filled-sea CAR, not deleted from source or H')


def evolve(values: np.ndarray, vectors: np.ndarray, wave: np.ndarray, time: float) -> np.ndarray:
    phase = np.exp(-1j*time*values).reshape((len(values),)+(1,)*(wave.ndim-1))
    return vectors@(phase*(vectors.conj().T@wave))


def bundle_evolve(data: tuple[np.ndarray, np.ndarray], marks: np.ndarray, time: float) -> np.ndarray:
    modes = marks.shape[1]
    propagated = evolve(*data, marks.transpose(1, 0, 2).reshape(modes, 16), time)
    return propagated.reshape(modes, 4, 4).transpose(1, 0, 2)


def source_gram(marks: np.ndarray, projector: np.ndarray) -> np.ndarray:
    unoccupied = marks-np.einsum('ij,ajr->air', projector, marks)
    return np.einsum('air,bir->ab', marks.conj(), unoccupied)


def source_marks(law: Any, fields: np.ndarray, links: np.ndarray, eta: np.ndarray,
                 reference: int, baseline: np.ndarray, a: np.ndarray, g: np.ndarray) -> np.ndarray:
    _, _, _, labels = spin_fields(fields)
    chi = g@divergence(law, baseline)
    longitudinal = np.exp(1j*(chi-chi[reference]))
    relative = law.sites-law.sites[reference]
    phase = longitudinal*np.exp(-1j*relative@a)
    wilson = np.tile(np.eye(4, dtype=complex), (law.volume, 1, 1))
    # Genuine coordinate-tree Wilson products from x to R; no toron is replaced.
    for x in range(law.volume):
        site = law.sites[x].copy()
        for j in range(3):
            while site[j] != law.sites[reference, j]:
                if site[j] < law.sites[reference, j]:
                    transport = links[law.index[tuple(site)], j]
                    site[j] += 1
                else:
                    site[j] -= 1
                    transport = links[law.index[tuple(site)], j].conj().T
                wilson[x] = wilson[x]@transport
    marks = np.einsum('x,xasp,xpr->axsr', eta*phase, labels, wilson).reshape(4, 4*law.volume, 4)
    accept(float(np.linalg.norm(np.einsum('air,bir->ab', marks.conj(), marks)-4*np.eye(4))),
           'actual coordinate HS quaternion metric before CAR')
    return marks


def harmonic_jet(law: Any, h: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    first, second = np.zeros_like(h), np.zeros_like(h)
    for x in range(law.volume):
        y = int(law.target[x, axis])
        xs, ys = slice(4*x, 4*x+4), slice(4*y, 4*y+4)
        first[xs, ys], first[ys, xs] = 1j*h[xs, ys], -1j*h[ys, xs]
        second[xs, ys], second[ys, xs] = -h[xs, ys], -h[ys, xs]
    return first, second


def mixture_covariance(projector: np.ndarray, extra: np.ndarray, volume: int) -> np.ndarray:
    """Fixed-number four-reference mixture, NOT Wick of its mixed onebody."""
    pb, kb = projector.reshape(volume, 4, volume, 4), extra.reshape(volume, 4, volume, 4)
    rho = np.trace(kb, axis1=1, axis2=3).diagonal().real
    return (density_covariance(projector, volume)+np.diag(rho)
            -2*np.real(np.sum(pb*kb.conj(), axis=(1, 3)))-np.outer(rho, rho))


def surface(law: Any, h: np.ndarray, data: tuple[np.ndarray, np.ndarray],
            marks: np.ndarray, alpha: np.ndarray, reference: int, times: list[float],
            g: np.ndarray, ue: float) -> tuple[dict[str, Any], np.ndarray]:
    values, vectors = data
    occupied = vectors[:, values < 0]
    require(occupied.shape[1] == 2*law.volume, 'entire 2V occupied native sea')
    projector = occupied@occupied.conj().T
    occupation_gram = occupied.conj().T@occupied
    initial_gram = source_gram(marks, projector)
    gram_error = accept(float(np.linalg.norm(initial_gram-2*np.eye(4))),
                        'unfiltered COMPLETE-reference initial CT Gram2I')
    source = np.einsum('a,air->ir', alpha, marks)
    b0 = source-projector@source
    norm = float(np.vdot(b0, b0).real)
    accept(abs(norm-2), 'coherent complex quaternion source norm2')
    neutral_covariance = density_covariance(projector, law.volume)
    accept(float(np.linalg.norm(neutral_covariance@np.ones(law.volume))), 'whole neutral canonical density Ward')
    records, exterior_errors = [], []
    initial_density = np.sum(abs(b0.reshape(law.volume, 4, 4))**2, axis=(1, 2))/2
    for time in times:
        bt = evolve(*data, b0, time)
        all_marks = bundle_evolve(data, marks, time)
        right_gram = source_gram(all_marks, projector)
        accept(float(np.linalg.norm(right_gram-initial_gram)), 'matched-H whole-sea RIGHT Gram conservation')
        sea = occupied*np.exp(-1j*values[values < 0]*time)[None, :]
        overlap = sea.conj().T@bt
        for r in range(4):
            weight = float(np.vdot(bt[:, r], bt[:, r]).real)
            if weight > 1e-12:
                complete_gram = np.block([[occupation_gram, overlap[:, r:r+1]/math.sqrt(weight)],
                    [overlap[:, r:r+1].conj().T/math.sqrt(weight), np.array([[1.]])]])
                exterior_errors.append(accept(float(np.linalg.norm(complete_gram-np.eye(2*law.volume+1))),
                    'actual full 2V+1 exterior CAR for each nonzero reference amplitude'))
        extra = bt@bt.conj().T/2
        charged = projector+extra
        rho = np.sum(abs(bt.reshape(law.volume, 4, 4))**2, axis=(1, 2))/2
        measured = np.array([np.trace(charged[4*x:4*x+4, 4*x:4*x+4]
                                      -projector[4*x:4*x+4, 4*x:4*x+4]).real for x in range(law.volume)])
        density_error = accept(float(np.linalg.norm(measured-rho)), 'whole canonical reference-traced excess density')
        current = currents(law, h, bt)/2
        whole_current = np.zeros_like(current)
        for x in range(law.volume):
            for j in range(3):
                y = int(law.target[x, j])
                whole_current[x, j] = -2*np.trace(h[4*x:4*x+4, 4*y:4*y+4]
                    @(charged[4*y:4*y+4, 4*x:4*x+4]-projector[4*y:4*y+4, 4*x:4*x+4])).imag
        current_error = accept(float(np.linalg.norm(whole_current-current)), 'same full Wilson+odd physical current')
        derivative = np.real(np.sum(bt.reshape(law.volume, 4, 4).conj()
            *(-1j*h@bt).reshape(law.volume, 4, 4), axis=(1, 2)))
        continuity = accept(float(np.linalg.norm(derivative+divergence(law, current))),
                            'same physical Peierls current and actual charge continuity')
        eps = 2e-5
        plus = evolve(*data, b0, time+eps).reshape(law.volume, 4, 4)
        minus = evolve(*data, b0, time-eps).reshape(law.volume, 4, 4)
        finite = (np.sum(abs(plus)**2, axis=(1, 2))-np.sum(abs(minus)**2, axis=(1, 2)))/(4*eps)
        finite_error = accept(float(np.linalg.norm(finite-derivative)), 'independent density time derivative', 2e-8)
        covariance = mixture_covariance(projector, extra, law.volume)
        ward = accept(float(np.linalg.norm(covariance@np.ones(law.volume))), 'fixed 2V+1 reference-mixture density Ward')
        total = rho.copy(); total[reference] -= 1
        accept(abs(float(total.sum())), 'same outside reference makes exact zero total Gauss charge')
        moment = np.sum(rho[:, None]*np.exp(2j*np.pi*law.sites/law.size), axis=0)
        records.append(dict(time=time, excess_density=rho, physical_link_current=current,
            density_derivative=derivative, continuity_error=continuity, finite_difference_error=finite_error,
            whole_sea_density_error=density_error, whole_sea_current_error=current_error,
            complete_source_Gram=right_gram, fixed_number_covariance_Ward_error=ward,
            reference_reduced_density=bt.T@bt.conj()/2, excess_charge=float(rho.sum()),
            circular_spatial_moment=moment, density_change_L1=float(np.sum(abs(rho-initial_density))),
            total_axis_current=current.sum(axis=0), full_charge_including_static_R=float(total.sum()),
            light_source_energy=float(np.trace(bt.conj().T@h@bt).real/2),
            Coulomb_source_excess_diagnostic=ue*(float(np.trace(g@(covariance-neutral_covariance)))+total@g@total),
            Coulomb_scope='exact canonical reference-mixture moments; not included in free/static-conditional propagation'))
    extra0 = b0@b0.conj().T/2
    correct_cov = mixture_covariance(projector, extra0, law.volume)
    explicit_cov = np.zeros_like(correct_cov)
    mean = initial_density
    for r in range(4):
        nr = float(np.vdot(b0[:, r], b0[:, r]).real)
        if nr > 1e-12:
            vr = b0[:, r]/math.sqrt(nr)
            wr = np.sum(abs(vr.reshape(law.volume, 4))**2, axis=1)
            explicit_cov += nr/2*(density_covariance(projector+np.outer(vr, vr.conj()), law.volume)+np.outer(wr, wr))
    explicit_cov -= np.outer(mean, mean)
    covariance_error = accept(float(np.linalg.norm(correct_cov-explicit_cov)), 'independent full Slater/ref-mixture density moments')
    fake_cov = density_covariance(projector+extra0, law.volume)
    wrong_wick = reject(lambda: accept(float(np.linalg.norm(fake_cov@np.ones(law.volume))),
        'fixed 2V+1 reference-mixture density Ward'), 'treat mixed onebody as a Slater projector')
    missing_ref = reject(lambda: accept(float(np.linalg.norm(source_gram(marks[:, :, :1], projector)-2*np.eye(4))),
        'unfiltered COMPLETE-reference initial CT Gram2I'), 'retain only one reference colour')
    require(records[-1]['density_change_L1'] > 1e-7, 'actual finite-width transported profile moves')
    witness = max(records, key=lambda row: float(np.linalg.norm(row['density_derivative'])))
    wrong_current = reject(lambda: accept(float(np.linalg.norm(witness['density_derivative']
        -divergence(law, witness['physical_link_current']))), 'same physical Peierls current and actual charge continuity'),
        'reverse the physical current')
    frozen = reject(lambda: accept(float(np.linalg.norm(divergence(law, witness['physical_link_current']))),
        'same physical Peierls current and actual charge continuity'), 'freeze the transported density')
    omitted_reference = reject(lambda: accept(float(records[-1]['excess_charge']),
        'same outside reference makes exact zero total Gauss charge'), 'omit the opposite reference charge')
    return dict(all4V_modes=len(h), neutral_number=2*law.volume, charged_number=2*law.volume+1,
        full_neutral_canonical_dimension=str(math.comb(4*law.volume, 2*law.volume)),
        full_charged_canonical_dimension=str(math.comb(4*law.volume, 2*law.volume+1)),
        initial_CT_Gram=initial_gram, initial_CT_Gram_error=gram_error, initial_complete_source_norm_squared=norm,
        whole_exterior_Gram_error=max(exterior_errors), reference_mixture_covariance_error=covariance_error,
        neutral_density_force_variances=np.diag(g@neutral_covariance@g), records=records,
        finite_time_circular_centroid_displacement=np.angle(records[-1]['circular_spatial_moment']
            /records[0]['circular_spatial_moment'])*law.size/(2*np.pi),
        source_filter_applied=False, full_reference_purification=True,
        adverse=dict(mixed_onebody_Wick=wrong_wick, missing_reference_colours=missing_ref,
            wrong_current=wrong_current, frozen_profile=frozen, omitted_reference_charge=omitted_reference)), projector


def ct_adversary(marks: np.ndarray, occupied: np.ndarray, chirality: np.ndarray) -> dict[str, Any]:
    # Legal full half-filled Slater occupying every coherent source colour.
    q, _ = np.linalg.qr(np.column_stack((marks[0], occupied)), mode='reduced')
    bad = q[:, :occupied.shape[1]]@q[:, :occupied.shape[1]].conj().T
    partner = chirality@(np.eye(len(bad))-bad)@chirality.conj().T
    g_bad, g_partner = source_gram(marks, bad), source_gram(marks, partner)
    error = accept(float(np.linalg.norm((g_bad+g_partner)/2-2*np.eye(4))),
                   'actual full canonical CT partner restores unfiltered Gram2I')
    wrong = reject(lambda: accept(float(np.linalg.norm(g_bad-2*np.eye(4))),
        'unfiltered COMPLETE-reference initial CT Gram2I'), 'omit CT with all coherent source modes occupied')
    return dict(occupied_whole_source_Gram=g_bad, actual_CT_partner_Gram=g_partner,
        CT_averaged_Gram_error=error, rejected_without_CT=wrong,
        scope='full 2V canonical adversary, not a low-energy or interacting-ground substitution')


def joint_ccr_consumer(law: Any, adversary: dict[str, Any]) -> dict[str, Any]:
    # Real cosine/sine oscillators of an actual native transverse travelling mode.
    mode = np.zeros((law.volume, 3), dtype=complex)
    mode[:, 1] = np.exp(2j*np.pi*law.sites[:, 0]/law.size)/math.sqrt(law.volume)
    accept(float(np.linalg.norm(divergence(law, mode))), 'actual coexact photon travelling mode')
    accept(abs(complex(np.vdot(mode, mode.conj()))), 'orthogonal native counterpropagating modes')
    inputs = [(0, 0), (1, 0), (0, 1)]
    outputs = inputs+[(2, 0), (1, 1), (0, 2)]
    g = np.array([1., 1j])/math.sqrt(2)
    creator = np.zeros((len(outputs), len(inputs)), dtype=complex)
    annihilator = np.zeros((1, len(inputs)), dtype=complex)
    for column, occupation in enumerate(inputs):
        for j in range(2):
            target = list(occupation); target[j] += 1
            creator[outputs.index((target[0], target[1])), column] += g[j]*math.sqrt(occupation[j]+1)
            if occupation[j]:
                annihilator[0, column] += g[j].conjugate()*math.sqrt(occupation[j])
    ccr = accept(float(np.linalg.norm(creator.conj().T@creator
        -annihilator.conj().T@annihilator-np.eye(len(inputs)))),
        'exact infinite-oscillator CCR on finite source polynomial domain')
    vacuum = np.array([1., 0., 0.], dtype=complex)
    travelling = np.array([0., g[0], g[1]])
    returning = travelling.conj()
    bad, partner = adversary['occupied_whole_source_Gram'], adversary['actual_CT_partner_Gram']
    # Full CT-paired correlated density; not a photon/free-sea product or ground.
    pairs = ((bad, travelling), (partner, vacuum), (partner, returning), (bad, vacuum))
    base, created, removed = np.zeros((4, 4), dtype=complex), np.zeros((4, 4), dtype=complex), np.zeros((4, 4), dtype=complex)
    bare_photon_norm = 0.
    for car, state in pairs:
        creation_norm = float(np.linalg.norm(creator@state)**2)
        annihilation_norm = float(np.linalg.norm(annihilator@state)**2)
        base += car/4
        created += creation_norm*car/4
        removed += annihilation_norm*car/4
        bare_photon_norm += creation_norm/4
    identity = accept(float(np.linalg.norm(created-base-removed)), 'actual full-reference fixed-g joint CCR Gram')
    require(float(np.linalg.eigvalsh(created-base).min()) >= -3e-8, 'fixed photon g joint Gram >= initial source Gram')
    factorized = bare_photon_norm*base
    wrong = reject(lambda: accept(float(np.linalg.norm(created-factorized)),
        'ACTUAL correlated joint source Gram'), 'factor travelling-photon norm from CT source norm')
    return dict(exact_source_CCR_error=ccr, actual_initial_source_Gram=base,
        actual_correlated_joint_Gram=created, actual_removed_photon_Gram=removed,
        bare_photon_creation_norm_squared=bare_photon_norm, joint_CCR_identity_error=identity,
        wrong_product_factorization=wrong, photon_domain='infinite oscillator; source degree<=1, exact creator image degree<=2; NO Hamiltonian truncation',
        scope='CT-paired non-ground full canonical native adversary; fixed-g quartet lower bound, not arbitrary mode/source tensor block')


def duhamel(h0: np.ndarray, h1: np.ndarray, data0: tuple[np.ndarray, np.ndarray],
            data1: tuple[np.ndarray, np.ndarray], marks: np.ndarray, alpha: np.ndarray,
            conditional_sea: np.ndarray, time: float) -> dict[str, Any]:
    source = np.einsum('a,air->ir', alpha, marks)
    target, free = evolve(*data1, source, time), evolve(*data0, source, time)
    delta = target-free
    error = float(np.linalg.norm(delta-conditional_sea@delta))/math.sqrt(2)
    nodes, weights = np.polynomial.legendre.leggauss(32)
    integral, upper = np.zeros_like(source), 0.
    for node, weight in zip(nodes, weights):
        s, factor = time*(float(node)+1)/2, time*float(weight)/2
        residual = (h1-h0)@evolve(*data0, source, s)
        integral += -1j*factor*evolve(*data1, residual, time-s)
        upper += factor*float(np.linalg.norm(residual-conditional_sea@residual))/math.sqrt(2)
    identity = accept(float(np.linalg.norm(delta-integral)), 'unfiltered full-reference exact Duhamel identity')
    require(error <= upper+3e-8, 'actual conditional CAR source norm below actual residual integral')
    mapped_gram = source_gram(bundle_evolve(data0, marks, time), conditional_sea)
    defect = float(np.linalg.norm(mapped_gram-2*np.eye(4)))
    wrong_metric = (reject(lambda: accept(defect, 'ACTUAL mismatched-H RIGHT Gram is not initial CT Gram', 1e-10),
        'force time-dependent RIGHT Gram2I') if defect > 1e-10 else 'not informative at this resolution')
    return dict(normalized_source_state_error=error, actual_CAR_residual_integral_upper=upper,
        source_identity_error=identity, initial_norm_squared=2., actual_mapped_RIGHT_Gram=mapped_gram,
        mapped_RIGHT_Gram_defect=defect, adverse_forced_RIGHT_metric=wrong_metric,
        scope='actual conditional h1 sea on BOTH sides; not the fully coupled photon ground')


def car_excess_consumer(law: Any, source: np.ndarray, free_sea: np.ndarray,
                        conditional_sea: np.ndarray) -> dict[str, Any]:
    """Check exact positive/hole CAR on the ACTUAL conditional whole sea."""
    u, v = source-free_sea@source, free_sea@source
    accept(abs(float(np.vdot(u, u).real)-2), 'matched free positive coefficient HS trace2')
    pu = conditional_sea@u
    ep, eh = float(np.linalg.norm(pu)), float(np.linalg.norm(v-conditional_sea@v))
    b = source-conditional_sea@source
    bp = u-pu
    norm = float(np.vdot(b, b).real)
    positive_norm = float(np.vdot(bp, bp).real)
    actual = (norm/2-1)*conditional_sea+b@b.conj().T/2
    positive = (positive_norm/2-1)*conditional_sea+bp@bp.conj().T/2
    target = np.sum(abs(u.reshape(law.volume, 4, 4))**2, axis=(1, 2))/2
    actual_density = np.array([np.trace(actual[4*x:4*x+4, 4*x:4*x+4]).real for x in range(law.volume)])
    positive_density = np.array([np.trace(positive[4*x:4*x+4, 4*x:4*x+4]).real for x in range(law.volume)])
    point = int(np.argmax(abs(actual_density-target)))
    sl = slice(4*point, 4*point+4)
    o0 = float(np.trace(conditional_sea[sl, sl]).real)
    cross = complex(np.vdot(u[sl], pu[sl]))
    nu_o = ep*ep*o0+cross-float(np.vdot(pu[sl], pu[sl]).real)
    rhs = (float(np.vdot(u[sl], u[sl]).real)-nu_o-cross.conjugate())/2
    identity = accept(abs(complex(positive_density[point])-rhs), 'exact full CAR positive-source excess identity')
    positive_upper = 5*ep/math.sqrt(2)
    hole_upper = 2*(2+math.sqrt(2))*eh
    positive_error = float(np.max(abs(positive_density-target)))
    complete_error = float(np.max(abs(actual_density-target)))
    require(positive_error <= positive_upper+3e-8, 'actual particle-annihilator CAR excess bound')
    require(complete_error <= positive_upper+hole_upper+3e-8, 'actual unfiltered positive+hole excess bound')
    return dict(point=point, actual_mapped_source_norm_squared=norm,
        actual_particle_annihilator_norm=ep, actual_hole_annihilator_norm=eh,
        positive_CAR_identity_error=identity, positive_density_error_max=positive_error,
        complete_density_error_max=complete_error, positive_density_error_upper=positive_upper,
        complete_density_error_upper=positive_upper+hole_upper,
        source_filtered=False, scope='algebraic decomposition only; actual static-conditional whole sea, not coupled Omega')


def theorem_bounds(constants: dict[str, Any], law: Any, eta: np.ndarray,
                   reference: int, time: float) -> dict[str, Any]:
    ue, speed, rho = constants['UE'], constants['bare_speed'], constants['relative_Hessian_bound']
    tau = constants['endpoint_norm_upper']
    mu = constants['mu_lower']
    sigma = math.sqrt(ue*math.sqrt(GREEN_3D_UPPER)/(speed*(1-rho)))
    sigma0 = math.sqrt(ue*math.sqrt(GREEN_3D_UPPER)/speed)
    fsum = 4*tau
    density = math.sqrt(constants['CE']*fsum/2)
    force = math.sqrt(density*GREEN_3D_UPPER)
    polar_total = 3*ue*tau*tau/(mu*mu)
    epsilon = 12*tau*(sigma+sigma0)+2*ue*GREEN_3D_UPPER+polar_total/law.volume
    # Keep the actual scalar toron sea potential in Kato/variational comparison.
    # Only the p_a moment uses its crude range <= 2VB, not occupation density.
    harmonic_energy_density = 2*constants['B_upper']+epsilon
    origin = np.full(3, law.size//2)
    displacement = (law.sites-origin+law.size//2) % law.size-law.size//2
    ell = int(np.max(np.sum(abs(displacement[abs(eta) > 0]), axis=1)))
    initial_l1 = 2*float(np.sum(abs(eta)))
    radius = math.ceil(ell+6*tau*math.expm1(1.)*time)
    z, n = math.exp(-2.), radius+1
    shell = (4*n*n/(1-z)+8*n*z/(1-z)**2+4*z*(1+z)/(1-z)**3+2/(1-z))
    tail = 2*math.exp(ell+6*tau*math.expm1(1.)*time-n)*math.sqrt(shell)
    polynomial_l1 = 2*(2*radius+1)**1.5+tail
    l1 = polynomial_l1
    if 6*tau*time <= math.log(l1/initial_l1):
        l1 = initial_l1*math.exp(6*tau*time)
    relative = law.sites-law.sites[reference]
    m1 = 2*np.sqrt(np.sum(abs(eta[:, None])**2*relative**2, axis=0))
    m2 = 2*np.sqrt(np.sum(abs(eta[:, None])**2*relative**4, axis=0))
    jet = 2*tau
    d1 = float(np.linalg.norm(m1+2*time*jet))
    d2 = float(np.sum(m2+2*time*jet*m1+2*(time*jet+time*time*jet*jet)))
    electric = 2*math.sqrt(ue*harmonic_energy_density)*d1+ue*d2/law.volume
    residual = (6*tau*sigma+ue*(4*force+13/4))*l1+electric
    normalized_error = time*residual/math.sqrt(2)
    eta_exc = epsilon/mu
    density_coefficient = (1+3/math.sqrt(2))*l1*math.sqrt(eta_exc)
    kappa = math.log1p(mu/(24*tau))
    cp = 8*(constants['B_upper']+mu)/(math.pi*mu)
    toron_prefactor = (384*tau*(constants['B_upper']+mu)/(math.pi*mu*mu)
        +24*tau*cp*time*math.exp(6*tau*math.expm1(kappa)*time))/math.sqrt(2)
    source_radius = int(np.max(abs(displacement[abs(eta) > 0])))
    cut_distance = max(0, law.size//2-source_radius-1)
    toron_error = toron_prefactor*math.exp(-kappa*cut_distance)
    onset = 2*(math.ceil(math.log(toron_prefactor/.005)/kappa)+source_radius+1)
    return dict(Green_diagonal_uniform_upper=GREEN_3D_UPPER, coexact_A_variance_upper=sigma*sigma,
        native_static_density_Kubo_coefficient=constants['CE'], complete_Wilson_fsum_coefficient=fsum,
        density_structure_factor_coefficient=density, density_force_L2_upper=force,
        harmonic_polar_trial_total_kinetic_upper=polar_total,
        neutral_excitation_density_upper=eta_exc, neutral_harmonic_energy_per_volume_upper=harmonic_energy_density,
        exact_initial_CT_Gram='2I under CT-even full-reference ideal-Clifford source/neutral CT density',
        source_norm_squared=2., packet_matrix_L1_time_upper=l1, harmonic_source_first_HS_derivative_upper=d1,
        weighted_unitary_L1_polynomial_upper=polynomial_l1, initial_source_L1_radius=ell,
        harmonic_source_Laplacian_HS_upper=d2, paid_harmonic_electric_residual_upper=electric,
        instantaneous_source_residual_upper=residual, finite_time_normalized_source_error_upper=normalized_error,
        density_comparison_coefficient_per_entry_l1=density_coefficient,
        density_Duhamel_coefficient_per_entry_l1=normalized_error*(2+normalized_error),
        fixed_frame_toron_CT_rate=kappa, fixed_frame_toron_prefactor=toron_prefactor,
        initial_source_coordinate_radius=source_radius, actual_cut_distance=cut_distance,
        paid_fixed_frame_toron_error_upper=toron_error, fixed_frame_pointwise_density_percent_onset=onset,
        positive_band_source_preparation=False, harmonic_cell_one_over_N_error=False,
        source_seam_omitted=False, seams='retained exactly inside h0(a); fixed-frame replacement still needs a paid seam tail',
        admission=dict(phase_and_Schur_supplier_hypotheses_required=True,
            source_norm_admitted_without_small_N_occupation_bound=True,
            finite_time_error_below_quarter=normalized_error <= .25,
            fixed_frame_density_percent_seam_admitted=toron_error <= .005,
            photon_phase_volume_onset_met=law.size >= constants['photon_phase_N_onset'],
            native_metric_volume_onset_met=(law.size >= constants['native_metric_N_onset']
                if constants['native_metric_N_onset'] is not None else None),
            coupled_ground_simulated=False, microscopic_N0_Lambda_admission_not_inferred=True))


def electric_consumer(law: Any, h: np.ndarray, data: tuple[np.ndarray, np.ndarray],
                      source: np.ndarray, fields: np.ndarray, links: np.ndarray,
                      phases: np.ndarray, relative: np.ndarray, time: float,
                      frame: np.ndarray, gram: np.ndarray) -> dict[str, Any]:
    plus, minus, _ = chiral_basis(frame, gram, law.volume)
    section = sea_section(h, plus, minus)
    records = []
    nodes, weights = np.polynomial.legendre.leggauss(24)
    for j in range(3):
        dj, ddj = harmonic_jet(law, h, j)
        dq = plus.conj().T@dj@minus
        du = polar_derivative(section, dq)
        dw = -minus@du.conj().T/math.sqrt(2)
        dtheta = float(np.trace(np.linalg.solve(section['Q'], dq)).imag)
        berry = accept(abs(complex(np.trace(section['occupied'].conj().T@dw))+.5j*dtheta),
                       'true native beta-polar principal-log Berry cancellation')
        first0 = -1j*np.repeat(relative[:, j], 4)[:, None]*source
        first = evolve(*data, first0, time)
        for node, weight in zip(nodes, weights):
            s, factor = time*(float(node)+1)/2, time*float(weight)/2
            first += -1j*factor*evolve(*data, dj@evolve(*data, source, s), time-s)
        eps = 2e-5
        waves = []
        for shift in (eps, -eps):
            shifted = phases.copy(); shifted[:, j] += shift
            hh = np.asarray(law.onebody(fields, links, shifted), dtype=complex)
            vv, qq = np.linalg.eigh(hh)
            mm = np.exp(-1j*shift*np.repeat(relative[:, j], 4))[:, None]*source
            waves.append(evolve(vv, qq, mm, time))
        finite = (waves[0]-waves[1])/(2*eps)
        derivative_error = accept(float(np.linalg.norm(finite-first)), 'actual harmonic source Frechet/electric derivative', 5e-8)
        wrong = reject(lambda: accept(float(np.linalg.norm(first-evolve(*data, first0, time))),
            'actual harmonic source Frechet/electric derivative', 5e-8), 'omit harmonic derivative of the propagator')
        occupied, empty = data[1][:, data[0] < 0], data[1][:, data[0] > 0]
        transitions = empty.conj().T@dj@occupied
        gap = data[0][data[0] > 0, None]-data[0][None, data[0] < 0]
        qgt = float(np.sum(abs(transitions/gap)**2))
        polar_qgt = float(np.vdot(du, du).real/4)
        accept(abs(polar_qgt-qgt), 'native full-sea polar/spectral pair-gap QGT', 3e-7)
        records.append(dict(axis=j, full_Wilson_derivative_HS_squared=float(np.vdot(dj, dj).real),
            full_Wilson_contact_HS_squared=float(np.vdot(ddj, ddj).real), cancelled_Berry_error=berry,
            entire_sea_QGT=qgt, polar_QGT=polar_qgt, actual_source_derivative_HS=float(np.linalg.norm(first)),
            electric_derivative_identity_error=derivative_error, adverse_missing_propagator_derivative=wrong))
    return dict(axes=records, vacuum_section='actual spatial-Schur beta, accretive polar/principal-log phase',
        complete_pair_gap_denominators=True, actual_scalar_Esea_kept=True)


def native_case(args: argparse.Namespace, size: int) -> dict[str, Any]:
    band = RotationBand(a=1., m=.2, r=.12, v=.3, improved=False)
    family, frames = all64(size, band, args.schur_lower)
    centre = tuple(args.centre)
    selected, frame = frames[centre]
    law, _, baseline = graph(size, centre, frame, band)
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    links = law.toron_links(np.asarray(centre)*np.pi/2)
    eta, preparation = packet_seed(law, args.width, np.asarray(args.carrier))
    reference_site = (np.asarray(preparation['origin'])+np.asarray(args.reference_offset)) % size
    reference = law.index[tuple(reference_site)]
    require(abs(eta[reference]) == 0, 'actual opposite reference is outside the compact source support')
    g, q2, diagonal = green(law)
    constants = weak_family(args.schur_lower, args.speed, args.weak_factor, args.phase_envelope)
    chi = g@divergence(law, baseline)
    expected = np.exp(-1j*(law.sites-law.sites[reference])@(np.asarray(centre)*np.pi/(2*size)))
    baseline_source_error = accept(float(np.linalg.norm(np.exp(1j*(chi-chi[reference]))-expected)),
        'actual Coulomb baseline phase, not a bare scalar packet')
    x = 2*np.pi*law.sites/size
    perturbation = args.phase_amplitude*np.column_stack((np.sin(x[:, 1]), np.cos(x[:, 2]), np.sin(x[:, 0]+x[:, 1])))
    accept(float(np.linalg.norm(divergence(law, perturbation))), 'static test background is genuinely coexact')
    alpha = np.array([1., .3j, -.2+.4j, .6], dtype=complex)
    alpha /= np.linalg.norm(alpha)
    times = np.linspace(0., args.time, args.samples).tolist()
    probes = [np.zeros(3), np.array([.43, -.29, .61])*np.pi/size,
              np.array([-np.pi/size, 0., 0.]), np.array([np.pi/size, 0., 0.])]
    reports, endpoint_data = [], []
    first_data, first_marks = None, None
    electric, adversary, joint = None, None, None
    for index, a in enumerate(probes):
        h0 = np.asarray(law.onebody(fields, links, baseline+a), dtype=complex)
        h1 = np.asarray(law.onebody(fields, links, baseline+a+perturbation), dtype=complex)
        data0, data1 = np.linalg.eigh(h0), np.linalg.eigh(h1)
        for h, data in ((h0, data0), (h1, data1)):
            accept(float(np.linalg.norm(h@data[1]-data[1]*data[0][None, :])), 'actual whole4V harmonic-frame eigenpairs')
        marks = source_marks(law, fields, links, eta, reference, baseline, a, g)
        free, sea0 = surface(law, h0, data0, marks, alpha, reference, times, g, constants['UE'])
        conditional, sea1 = surface(law, h1, data1, marks, alpha, reference, times, g, constants['UE'])
        control = duhamel(h0, h1, data0, data1, marks, alpha, sea1, args.time)
        mapped_source = evolve(*data0, np.einsum('a,air->ir', alpha, marks), args.time)
        car_control = car_excess_consumer(law, mapped_source, sea0, sea1)
        if index == 0:
            first_data, first_marks = data0, marks
            # Native chirality is the product of the recovered whitened four gammas.
            values_g, vectors_g = np.linalg.eigh(np.asarray(selected['G']))
            gammas = np.einsum('ij,jab->iab', (vectors_g/np.sqrt(values_g))@vectors_g.T, frame)
            chirality4 = gammas[0]@gammas[1]@gammas[2]@gammas[3]
            chirality = np.kron(np.eye(law.volume), chirality4)
            adversary = ct_adversary(marks, data0[1][:, data0[0] < 0], chirality)
            joint = joint_ccr_consumer(law, adversary)
            source = np.einsum('a,air->ir', alpha, marks)
            electric = electric_consumer(law, h0, data0, source, fields, links, baseline,
                law.sites-law.sites[reference], args.time, frame, np.asarray(selected['G']))
        if index == 1:
            phase = np.repeat(np.exp(-1j*(law.sites-law.sites[reference])@a), 4)
            assert first_data is not None and first_marks is not None
            naive = phase[:, None]*bundle_evolve(first_data, first_marks, args.time).transpose(1, 0, 2).reshape(len(h0), 16)
            exact = bundle_evolve(data0, marks, args.time).transpose(1, 0, 2).reshape(len(h0), 16)
            seam_wrong = reject(lambda: accept(float(np.linalg.norm(exact-naive)),
                'actual harmonic frame includes coordinate seams'), 'phase-only fixed-frame propagation')
        else:
            seam_wrong = None
        reports.append(dict(harmonic_coordinate=a, free=free, static_conditional=conditional,
            exact_source_Duhamel=control, actual_conditional_CAR_excess=car_control,
            adverse_unpaid_coordinate_seam=seam_wrong))
        if index >= 2:
            endpoint_data.append((h0, marks, free))
    wrap = np.repeat(np.exp(-2j*np.pi*law.sites[:, 0]/size), 4)
    wrap_h_error = accept(float(np.linalg.norm(endpoint_data[1][0]
        -wrap[:, None]*endpoint_data[0][0]*wrap.conj()[None, :])), 'actual harmonic-cell large-gauge Hamiltonian gluing')
    wrap_m_error = accept(float(np.linalg.norm(endpoint_data[1][1]
        -np.exp(2j*np.pi*reference_site[0]/size)*wrap[None, :, None]*endpoint_data[0][1])),
        'actual source/reference large-gauge gluing')
    wrap_density_error = accept(max(float(np.linalg.norm(r['excess_density']-s['excess_density']))
        for r, s in zip(endpoint_data[0][2]['records'], endpoint_data[1][2]['records'])),
        'actual harmonic-cell boundary charged density is gauge invariant')
    return dict(size=size, volume=law.volume, genuine_three_axes=True, centre=list(centre),
        native_true10_Gram=selected['G'], all64_true_wells=family, complete_reference_dimension=4,
        scalar_offsets_policy='all actual c_N,theta retained; no global neutral winner chosen',
        reference_site=reference_site, reference_sector_truncated=False, source_preparation=preparation,
        coherent_quaternion_alpha=alpha, real_link_centre_translation=baseline,
        longitudinal_baseline_source_phase_included=True, conditional_real_link_background=perturbation,
        Green_diagonal=diagonal, native_nonzero_momenta=int(np.sum(q2 > 1e-12)),
        harmonic_frame_consumers=reports, whole_CAR_CT_adversary=adversary, actual_harmonic_electric_consumer=electric,
        fixed_g_joint_CCR_consumer=joint,
        large_gauge_wrap=dict(H_error=wrap_h_error, complete_reference_source_error=wrap_m_error,
                             charged_density_error=wrap_density_error),
        analytic_parameter_bound=theorem_bounds(constants, law, eta, reference, args.time),
        added_selector_methods_executed=False, finite_energy_charged_source_is_band_Hamiltonian=False,
        coupled_ground_numerically_computed=False,
        baseline_Coulomb_source_phase_error=baseline_source_error,
        prediction_scope='coupled theorem compares to h0(a) conditional free sea averaged over ACTUAL harmonic marginal; no invented marginal')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[5])
    parser.add_argument('--centre', type=int, nargs=3, default=[1, 2, 3])
    parser.add_argument('--width', type=float, default=.8)
    parser.add_argument('--carrier', type=float, nargs=3, default=[1.2, 0., 0.])
    parser.add_argument('--time', type=float, default=1.)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--phase-amplitude', type=float, default=.03)
    parser.add_argument('--reference-offset', type=int, nargs=3, default=[-2, -2, -2])
    parser.add_argument('--schur-lower', type=float, default=.01)
    parser.add_argument('--speed', type=float, default=.1)
    parser.add_argument('--weak-factor', type=float, default=1.)
    parser.add_argument('--phase-envelope', choices=('conservative', 'universal-paid', 'tight-paid'), default='conservative')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    require(min(args.sizes) >= 5 and all(0 <= c < 4 for c in args.centre), 'genuine 3D admitted finite graph and true centre')
    require(args.width > 0 and args.time > 0 and args.samples >= 2 and args.phase_amplitude > 0,
            'finite nonzero packet width, time, conditional adverse and samples')
    require(any(args.reference_offset), 'actual distinct compensating reference')
    result = dict(plan='issue-222-synthesis-continuation-20261006', assigned_revision=PLAN_REVISION,
        template_revision=TEMPLATE_REVISION, source_hashes=source_hashes(),
        environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform()),
        family=weak_family(args.schur_lower, args.speed, args.weak_factor, args.phase_envelope),
        cases=[native_case(args, size) for size in args.sizes],
        scope='unfiltered CT-complete native harmonic-frame whole-CAR consumer; actual coupled state only in companion conditional derivation',
        no_Euclidean_decay_as_motion=True, no_isolated_pole=True, no_infinite_time_ballistic_claim=True)
    text = json.dumps(portable(result), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(text+'\n')
    print(text)


if __name__ == '__main__':
    main()
