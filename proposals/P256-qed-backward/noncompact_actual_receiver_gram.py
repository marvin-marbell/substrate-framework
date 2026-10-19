#!/usr/bin/env python3
"""Original static, coherent, full-bar4 receiver Gram and one-sided whitening."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
from typing import Any
from fractions import Fraction

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'

import numpy as np
import scipy
from scipy.linalg import eigh, expm
from flint import acb, arb, ctx

import noncompact_inclusive_scattering_join as scattering
from sea_selection import SELECTED

ROOT = Path(__file__).resolve().parent
FROZEN_SUPPLIER = 'e3ea529d1272f7e21224145fb5c7c943471ce2cc608cf034629d97d70e9c5752'
PLAN = 'issue-222-synthesis-continuation-20261006'


def require(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)


def cloud_gram(sea: np.ndarray, left: np.ndarray, right: np.ndarray,
               left_cloud: np.ndarray, right_cloud: np.ndarray) -> tuple[complex, dict[str, complex]]:
    """Exact full-CAR virtual-cloud overlap; source matrices already include /sqrt2.

    For G_ij, left_cloud=X_j and right_cloud=X_i: the photon indices cross.
    No source P projection and no reference trace of squared amplitudes.
    """
    lp, rp = sea.conj().T @ left, sea.conj().T @ right
    lq, rq = left-sea @ lp, right-sea @ rp
    parts = dict(neutral_bulk=complex(np.vdot(lq, rq)*np.vdot(left_cloud, right_cloud)),
                 particle_exchange=-complex(np.vdot(right_cloud.conj().T @ lq,
                                                     left_cloud.conj().T @ rq)),
                 negative_source_boundary=complex(np.vdot(left_cloud @ lp, right_cloud @ rp)))
    return sum(parts.values()), parts


def exterior_cloud(sea: np.ndarray, left: np.ndarray, right: np.ndarray,
                   left_cloud: np.ndarray, right_cloud: np.ndarray) -> dict[str, Any]:
    """Independent entire-sea bordered determinants, mixed exterior derivative."""
    def difference(epsilon: float) -> complex:
        total = 0j
        for s, t in itertools.product((-1, 1), repeat=2):
            bra = sea+s*epsilon*left_cloud
            ket = sea+t*epsilon*right_cloud
            for r in range(left.shape[1]):
                a = np.column_stack((bra, left[:, r]))
                b = np.column_stack((ket, right[:, r]))
                total += s*t*np.linalg.det(a.conj().T @ b)
        return complex(total/(4*epsilon**2))
    coarse, fine, finer = (difference(e) for e in (.01, .005, .0025))
    extrapolated = (4*finer-fine)/3
    tolerance = max(2e-7, 4*abs((4*fine-coarse)/3-extrapolated))
    return dict(coarse=coarse, fine=fine, finer=finer,
                extrapolated=extrapolated, tolerance=tolerance)


def inverse_sqrt(gram: np.ndarray) -> np.ndarray:
    values, vectors = eigh(gram)
    require(bool(values[0] > 0), 'actual receiver Gram is positive')
    return (vectors/np.sqrt(values)) @ vectors.conj().T


def physical_amplitude(gram: np.ndarray, overlaps: np.ndarray, input_norm: float) -> np.ndarray:
    """Output ONLY is whitened. Input is the original normalized physical prep."""
    require(input_norm > 0, 'physical preparation norm')
    return inverse_sqrt(gram) @ overlaps/input_norm


def _positive_lower(value: Any) -> str:
    from decimal import Decimal, ROUND_FLOOR, localcontext
    require(bool(value > 0), 'strictly positive Arb lower interval')
    with localcontext() as context:
        context.prec, context.rounding = 65, ROUND_FLOOR
        candidate = Decimal(value.lower().str(75, radius=False))
        for _ in range(4):
            candidate *= Decimal('0.999999999999999999999999999999999999999999999999999999999999')
            text = str(candidate)
            if arb(text) < value:
                return text
    raise ArithmeticError('outward positive rational conversion')


def whitening_enclosure(c: complex, g: complex, m0: complex, c_error: Any,
                        g_error: Any, m_error: Any, gram_radius: Any,
                        input_norm_upper: Any, bare_gram: tuple[Any, Any] = (1, 1)) -> dict[str, Any]:
    """Volume-rescaled original RAW 2x2 output-whitening disk.

    c=V*m_y/UE, g=V*G_yx/UE; errors include conditional kernel and actual
    ground/flight differences. Radius bounds ||G-diag(gx0,gy0)||.
    No independent receiver-column rescaling or input whitening is assumed.
    """
    ctx.prec = 256
    ec, eg, em, r, n = map(arb, map(str, (c_error, g_error, m_error, gram_radius, input_norm_upper)))
    a, b = map(arb, map(str, bare_gram))
    minimum = a.lower() if a.lower() <= b.lower() else b.lower()
    require(bool(r >= 0 and minimum > r and n > 0 and ec >= 0 and eg >= 0 and em >= 0),
            'positive actual raw metric and supplied physical error budgets')
    ac, ag, am = map(lambda z: abs(acb(z)), (c, g, m0))
    diagonal = r/(2*(minimum-r)**arb('1.5'))
    offdiagonal = 3*r/(4*(minimum-r)**arb('2.5'))
    divided = -1/(a.sqrt()*b.sqrt()*(a.sqrt()+b.sqrt()))
    pieces = dict(raw_flight=(1/b.sqrt()+diagonal)*ec,
                  bare_offGram=-divided*eg*am,
                  forward_flight=-divided*(ag+eg)*em,
                  diagonal_whitening=diagonal*ac,
                  offdiagonal_whitening=offdiagonal*(ag+eg)*(am+em))
    error = sum(pieces.values(), arb(0))
    reference_ball = acb(c)/b.sqrt()+divided*acb(g)*acb(m0)
    margin = abs(reference_ball)-error
    reference = complex(float(reference_ball.real), float(reference_ball.imag))
    return dict(reference=reference, reference_Arb=reference_ball.str(65),
                bare_raw_Gram=[a.str(65), b.str(65)], divided_difference=divided.str(65),
                input_norm_upper=n.str(65), error_components={k:v.str(65) for k,v in pieces.items()},
                disk_radius=error.str(65), margin=margin.str(65),
                normalized_rescaled_lower=_positive_lower(margin/n) if margin > 0 else None,
                rigorous_positive=bool(margin > 0),
                formula='(c/sqrt(gy0)+f[gy0,gx0]*g*m0)/n; RAW output ONLY; no C0 subtraction')


def native_consumer(core_descriptor: dict[str, Any] | None = None) -> dict[str, Any]:
    data = scattering.setup(5)
    momentum, law = data['momentum'], data['law']
    if core_descriptor is not None:
        # Embed the SAME compact core matrix in the literal native CAR consumer.
        # This is an N5 algebra bridge, not a measurement at huge legal N.
        number = lambda row: complex(row['real'], row['imag'])
        matrix = np.asarray([[number(row) for row in line]
                            for line in core_descriptor['source_normalized_creation_matrix']])
        offsets = np.asarray(core_descriptor['compact_displacement'], dtype=int)
        eta = np.asarray([number(row) for row in core_descriptor['source_eta']])
        profiles = np.asarray([float(arb(value).exp().mid()) for value in
                               core_descriptor['source_core_normalized_log_profile_intervals']])
        reference = np.array([0, law.size//2, law.size//2])
        theta = np.asarray(data['centre'])*math.pi/2
        transfer = 2*math.pi/law.size*np.array([1, -1, 0])
        wave = np.zeros((law.volume, 4, 4), complex)
        detector = np.zeros_like(wave)
        for offset, coefficient, profile in zip(offsets, eta, profiles):
            site = (data['origin']+offset) % law.size
            index = law.index[tuple(site)]
            phase = np.exp(-1j*theta @ (site-reference)/law.size)
            wave[index] = coefficient*profile*phase*matrix
            detector[index] = wave[index]*np.exp(1j*transfer @ (site-reference))
        data = {**data, 'full_source':momentum.to_momentum(wave),
                'full_detector':momentum.to_momentum(detector)}
    sea = scattering.sea_columns(momentum).reshape(4*law.volume, 2*law.volume)
    source = data['full_source'].reshape(4*law.volume, 4)
    recoil_source = data['full_detector'].reshape(4*law.volume, 4)
    forward = scattering.packet(law, 0, data['constants']['bare_speed'], data['origin']+np.array([1, 0, 0]))
    recoil = scattering.annihilation_packet(data['outgoing'])
    clouds = [scattering.preparation_tangent(momentum, scattering.annihilation_packet(ph), 0.)[0]
              .reshape(4*law.volume, 2*law.volume) for ph in (forward, recoil)]
    rows, derivatives, components = [], [], []
    for left, right, lcloud, rcloud in ((source, source, clouds[0], clouds[0]),
            (recoil_source, recoil_source, clouds[1], clouds[1]),
            (recoil_source, source, clouds[0], clouds[1])):
        value, terms = cloud_gram(sea, left, right, lcloud, rcloud)
        literal = exterior_cloud(sea, left, right, lcloud, rcloud)
        require(abs(value-literal['extrapolated']) < literal['tolerance'], 'full-CAR exterior identity')
        rows.append(value); components.append(terms); derivatives.append(literal)
    gxx, gyy, gyx = rows
    gxy, _ = cloud_gram(sea, source, recoil_source, clouds[1], clouds[0])
    require(abs(gxy-gyx.conjugate()) < 1e-12, 'full receiver Hermitian identity')
    compared, _ = scattering.prepared_reduced_output({**data, 'incoming':forward}, 0., 4)
    existing = complex(np.vdot(data['full_detector'], compared['initial_photon_exchange']))
    require(abs(existing-gyx) < 1e-12, 'static FORWARD-cloud full Mout/T0 identity')
    wrong_source, _ = cloud_gram(sea, recoil_source, source,
        scattering.preparation_tangent(momentum, scattering.annihilation_packet(data['incoming']), 0.)[0]
        .reshape(4*law.volume, 2*law.volume), clouds[1])
    omit_reference, _ = cloud_gram(sea, recoil_source[:, :1], source[:, :1], clouds[0], clouds[1])
    require(abs(wrong_source-gyx) > 1e-5, 'incoming origin is not the output forward receiver centre')
    require(abs(omit_reference-gyx) > derivatives[2]['tolerance'], 'complete bar4 is essential')
    require(abs(components[2]['negative_source_boundary']) > derivatives[2]['tolerance'],
            'full negative source boundary is essential')
    # A legal same-number occupied state, but not CT-real: creation Gram vanishes.
    occupied_basis = np.linalg.qr(np.column_stack((source, np.eye(4*law.volume))), mode='reduced')[0]
    occupied_sea = occupied_basis[:, :2*law.volume]
    blocked = source-occupied_sea @ (occupied_sea.conj().T @ source)
    occupied_source_norm = float(np.linalg.norm(blocked)**2)
    require(occupied_source_norm < 1e-24, 'arbitrary occupied source has zero creation-state Gram')
    bare = []
    qsource = momentum.project(data['full_source'], False)
    for duration in (0., 6.):
        ph = sum(a['coefficient'].conjugate()*b['coefficient']*np.exp(-1j*b['omega']*duration)
            for a in forward['modes'] for b in data['incoming']['modes']
            if np.array_equal(a['mode'], b['mode']))
        charge = complex(np.vdot(source, momentum.evolve(qsource, duration).reshape(4*law.volume, 4)))
        m0 = ph*charge
        bare.append(dict(time=duration, m_x0=m0, physical_offGram_correction=-.5*gyx*m0))
    gamma5 = np.linalg.multi_dot(list(SELECTED))
    gamma = np.kron(np.eye(law.volume), gamma5)
    # The source even product, not an arbitrary branch-anticommutator assumption.
    even_error = float(np.linalg.norm(gamma @ (source @ source.conj().T)-(source @ source.conj().T) @ gamma))
    require(even_error < 1e-12, 'source density is even under fixed chirality')
    source_state_norm = float(np.linalg.norm(source-sea @ (sea.conj().T @ source))**2)
    require(abs(source_state_norm-1) < 1e-12, 'single-plus reference-complete free CT source state metric')
    return dict(N=5, onebody_dimension=4*law.volume, entire_sea_dimension=2*law.volume,
        reference='PLUS NL+1 tensor bar4; no opposite-charge preparation', reference_dimension=4,
        leading_static_Gram=np.array([[gxx, gxy], [gyx, gyy]]), cloud_components=components,
        independent_exterior_derivatives=derivatives, full_Mout_T0_identity_error=abs(existing-gyx),
        omitted_reference_error=abs(omit_reference-gyx), wrong_origin_forward_error=abs(wrong_source-gyx),
        source_P_boundary_error=abs(components[2]['negative_source_boundary']),
        single_plus_source_state_norm_squared=source_state_norm, source_even_chirality_error=even_error,
        arbitrary_occupied_source_creation_Gram=occupied_source_norm,
        arbitrary_occupied_source_CAR_anticommutator=float(np.linalg.norm(source)**2), bare_forward=bare,
        actual_finite_UE_3D_Gram_measured=False,
        source_scope='certified compact far-core matrix native embedding' if core_descriptor else 'historical SameCurrent3c20 source')


def _sector(number: int) -> tuple[list[int], dict[int, int]]:
    states = [x for x in range(16) if x.bit_count() == number]
    return states, {x:i for i,x in enumerate(states)}


def _creator(index: int, number: int) -> np.ndarray:
    incoming, _ = _sector(number)
    outgoing, lookup = _sector(number+1)
    matrix = np.zeros((len(outgoing), len(incoming)), complex)
    for j, state in enumerate(incoming):
        if not state & (1 << index):
            matrix[lookup[state | (1 << index)], j] = (-1)**((state & ((1 << index)-1)).bit_count())
    return matrix


def coupled_consumer(cutoff: int = 5, electric: float = .005) -> dict[str, Any]:
    """Actual coupled finite quantum Γ comparison, all CAR/bar4, no Wick vacuum."""
    a = np.diag(np.sqrt(np.arange(1, cutoff)), 1)
    identity = np.eye(cutoff)
    annihilators = [np.kron(a, identity), np.kron(identity, a)]
    frequencies = [.31, .43]
    bosdim = cutoff**2
    hph = sum(w*b.T @ b for w,b in zip(frequencies, annihilators))
    h = np.kron(np.eye(bosdim), .2*SELECTED[0])
    for b, w, gamma in zip(annihilators, frequencies, SELECTED[1:3]):
        field = math.sqrt(electric/w)*(b+b.T)
        eigenvalues, eigenvectors = eigh(field)
        cosine = (eigenvectors*np.cos(eigenvalues)) @ eigenvectors.T
        sine = (eigenvectors*np.sin(eigenvalues)) @ eigenvectors.T
        h += np.kron(.12*(np.eye(bosdim)-cosine), SELECTED[0])+.3*np.kron(sine, gamma)
    h = h.reshape(bosdim, 4, bosdim, 4).transpose(0, 2, 1, 3)
    def lift(number: int, body: np.ndarray) -> np.ndarray:
        states, _ = _sector(number)
        dim = len(states)
        result = np.zeros((bosdim*dim, bosdim*dim), complex)
        # C_i^dagger C_j on the COMPLETE number sector.
        creation = [_creator(i, number-1) for i in range(4)]
        for i in range(4):
            for j in range(4):
                result += np.kron(body[:, :, i, j], creation[i] @ creation[j].conj().T)
        return result
    def full_h(number: int) -> np.ndarray:
        dim = len(_sector(number)[0])
        return np.kron(hph, np.eye(dim))+lift(number, h)+electric*(number-2)**2*np.eye(bosdim*dim)
    neutral_h, charged_h = full_h(2), full_h(3)
    energies, eigenvectors = eigh(neutral_h)
    omega = eigenvectors[:, 0]
    gamma5 = np.linalg.multi_dot(list(SELECTED))
    anticomm = float(np.linalg.norm(np.einsum('ab,pqbc->pqac', gamma5, h)
                       +np.einsum('pqab,bc->pqac', h, gamma5)))
    alpha = np.array([1., .3j, -.2+.4j, .6]); alpha /= np.linalg.norm(alpha)
    marker = np.einsum('a,aij->ij', alpha, SELECTED)
    source = sum(np.kron(_creator(i, 2), marker[i, :, None]) for i in range(4))/math.sqrt(2)
    # Source target order: charged CAR first, complete coherent bar4 second.
    B = np.kron(np.eye(bosdim), source)
    photon_ops = [np.kron(b, np.eye(16)) for b in annihilators]
    charged_source = B @ omega
    X = np.column_stack([operator.conj().T @ charged_source for operator in photon_ops])
    gram = X.conj().T @ X
    metric = float(np.vdot(charged_source, charged_source).real)
    require(abs(metric-1) < 2e-12, 'actual coupled single-plus source metric under CT')
    actual_cloud_gram = np.array([[np.vdot(photon_ops[j] @ charged_source,
        photon_ops[i] @ charged_source) for j in range(2)] for i in range(2)])
    boundary = np.zeros((2, 2), complex)
    for i, b in enumerate(annihilators):
        commutator = b @ b.T-b.T @ b
        boundary[i, i] = np.vdot(charged_source,
            np.kron(np.eye(bosdim)-commutator, np.eye(16)) @ charged_source)
    identity_error = float(np.linalg.norm(gram-(metric*np.eye(2)+actual_cloud_gram-boundary)))
    require(identity_error < 2e-12, 'actual cloud/CCR Gram with explicit finite-cutoff boundary')
    chvalues, chvectors = eigh(charged_h)
    duration = 6.
    U = (chvectors*np.exp(-1j*duration*(chvalues-energies[0]))) @ chvectors.conj().T
    propagated = np.kron(U, np.eye(4)) @ X[:, 0]/np.linalg.norm(X[:, 0])
    Y = X @ inverse_sqrt(gram)
    amplitude = Y.conj().T @ propagated
    require(bool(np.linalg.norm(Y.conj().T @ Y-np.eye(2)) < 2e-12), 'actual coherent Y orthonormal')
    probabilities = abs(amplitude)**2
    complement = 1-float(np.sum(probabilities))
    require(complement >= -2e-12, 'original two effects plus inclusive complement')
    incorrect = abs((X.conj().T @ propagated)[1])**2
    require(abs(incorrect-probabilities[1]) > 1e-8, 'dropping global off-Gram changes actual probability')
    by_reference = np.sum(Y[:, 1].conj().reshape(-1, 4)*propagated.reshape(-1, 4), axis=0)
    trace_probability = float(np.sum(abs(by_reference)**2))
    require(abs(trace_probability-probabilities[1]) > 1e-8, 'reference trace changes the declared coherent effect')
    r = float(np.linalg.norm(gram-np.eye(2), 2))
    f = inverse_sqrt(gram)
    sharp = .75*r*abs(gram[1, 0])/(1-r)**2.5
    require(abs(f[1, 0]+.5*gram[1, 0]) <= sharp+1e-14, 'offdiagonal divided-difference remainder')
    # Derive the leading coefficient, rather than fitting a finite-UE probability.
    # All terms through lambda² (lambda=sqrtUE) are represented exactly at cutoff>=4.
    require(cutoff >= 4, 'leading two-photon contour requires all states through three photons')
    body0 = np.kron(np.eye(bosdim), .2*SELECTED[0])
    body1 = sum(np.kron((b+b.T)/math.sqrt(w), .3*gamma)
                for b,w,gamma in zip(annihilators, frequencies, SELECTED[1:3]))
    body2 = sum(np.kron((b+b.T) @ (b+b.T)/w, .06*SELECTED[0])
                for b,w in zip(annihilators, frequencies))
    reshape_body = lambda value: value.reshape(bosdim, 4, bosdim, 4).transpose(0, 2, 1, 3)
    neutral0 = np.kron(hph, np.eye(6))+lift(2, reshape_body(body0))
    neutral1, neutral2 = lift(2, reshape_body(body1)), lift(2, reshape_body(body2))
    e0, v0 = eigh(neutral0)
    vacuum = v0[:, 0]
    reduced = (v0[:, 1:]/(e0[1:]-e0[0])) @ v0[:, 1:].conj().T
    omega1 = -reduced @ (neutral1 @ vacuum)
    e2 = float(np.vdot(vacuum, neutral1 @ omega1+neutral2 @ vacuum).real)
    omega2 = -reduced @ (neutral1 @ omega1+neutral2 @ vacuum)-.5*np.vdot(omega1, omega1)*vacuum
    basis_preps = [np.column_stack([op.conj().T @ (B @ state) for op in photon_ops])
                   for state in (vacuum, omega1, omega2)]
    charged0 = np.kron(hph, np.eye(4))+lift(3, reshape_body(body0))-e0[0]*np.eye(4*bosdim)
    charged1 = lift(3, reshape_body(body1))
    charged2 = lift(3, reshape_body(body2))+np.eye(4*bosdim)
    dimension = len(charged0)
    generator = np.zeros((3*dimension, 3*dimension), complex)
    for k in range(3):
        generator[k*dimension:(k+1)*dimension, k*dimension:(k+1)*dimension] = -1j*charged0
    generator[:dimension, dimension:2*dimension] = -1j*charged1
    generator[dimension:2*dimension, 2*dimension:] = -1j*charged1
    generator[:dimension, 2*dimension:] = -1j*charged2
    contour = expm(duration*generator)
    evolution = [np.kron(contour[:dimension, k*dimension:(k+1)*dimension], np.eye(4)) for k in range(3)]
    raw_leading = sum(np.vdot(basis_preps[i][:, 1], evolution[k] @ basis_preps[j][:, 0])
                      for i in range(3) for j in range(3) for k in range(3) if i+j+k == 2)
    bare_forward = complex(np.vdot(basis_preps[0][:, 0], evolution[0] @ basis_preps[0][:, 0]))
    leading_cloud = np.array([[np.vdot(photon_ops[j] @ (B @ omega1), photon_ops[i] @ (B @ omega1))
                              for j in range(2)] for i in range(2)])
    leading = complex(raw_leading-.5*leading_cloud[1, 0]*bare_forward)
    t0_raw = sum(np.vdot(basis_preps[i][:, 1], basis_preps[j][:, 0]) for i in range(3)
                 for j in range(3) if i+j == 2)
    require(abs(t0_raw-leading_cloud[1, 0]) < 2e-12, 'exact contour/T0 actual leading Gram identity')
    wrong_subtraction = raw_leading-t0_raw
    require(bool(abs(wrong_subtraction-leading) > .001), 'C6−C0 is not the physical output-whitened coefficient')
    raw_actual = X.conj().T @ (np.kron(U, np.eye(4)) @ X[:, 0])
    ec = abs(raw_actual[1]/electric-raw_leading)+1e-10
    eg = abs(gram[1, 0]/electric-leading_cloud[1, 0])+1e-10
    em = abs(raw_actual[0]-bare_forward)+1e-10
    enclosure = whitening_enclosure(complex(raw_leading), complex(leading_cloud[1, 0]), bare_forward,
                                    ec, eg, em, r+1e-12, math.sqrt(1+r)+1e-12)
    require(enclosure['rigorous_positive'], 'actual dense-ground error disk has a physical nonzero margin')
    require(abs(amplitude[1])/electric >= float(enclosure['normalized_rescaled_lower']),
            'actual coherent probability obeys one-sided enclosure')
    return dict(cutoff=cutoff, UE=electric, omega=frequencies, time=duration,
        entire_neutral_dimension=len(omega), charged_with_bar4_dimension=X.shape[0],
        actual_neutral_energy=float(energies[0]), actual_single_plus_source_norm_squared=metric,
        all_A_chirality_error=anticomm, actual_global_Gram=gram, actual_cloud_Gram=actual_cloud_gram,
        explicit_CCR_cutoff_boundary=boundary, exact_Gram_identity_error=identity_error,
        actual_normalized_input_norm=float(np.linalg.norm(X[:, 0])), actual_amplitudes=amplitude,
        actual_two_effect_probabilities=probabilities, actual_inclusive_complement=complement,
        omitted_whitening_probability=incorrect, ref_trace_new_observable_probability=trace_probability,
        actual_offdiagonal_whitening_remainder=float(abs(f[1, 0]+.5*gram[1, 0])),
        divided_difference_upper=sharp,
        derived_contour_raw_m_y1=raw_leading, derived_offGram_g1=leading_cloud[1, 0],
        actual_bare_m_x0=bare_forward, derived_physical_leading=leading,
        wrong_C6_minus_C0=wrong_subtraction, T0_leading_Gram_identity_error=abs(t0_raw-leading_cloud[1, 0]),
        measured_actual_amplitude_divided_UE=amplitude[1]/electric,
        actual_finite_ground_transfer_enclosure=enclosure, neutral_Rayleigh_energy_second_order=e2,
        scope='actual coupled finite Γ comparison, not native3D/fullPhi/actual harmonic measure')


def legal_gram_budget(descriptor: dict[str, Any], source_packet: dict[str, Any],
                      kernel_packet: dict[str, Any], weak_bits: int = 4096) -> dict[str, Any]:
    """Actual-Gamma metric and graph transfer with exact FAR source bindings.

    TOTAL b rescales the binary baseline UE0. No p527 or old probability is
    relabelled. Full microscopic source/current flight is owned separately.
    """
    ctx.prec = 256
    size, volume = int(descriptor['N']), int(descriptor['V'])
    D = size//2
    require(size == 5*2**96 and volume == size**3, 'exact legal mode/volume integers')
    require(descriptor['reference'] == [0, D, D] and descriptor['origin'] == [D, D, D],
            'actual FAR source reference, not near-R geometry')
    require(source_packet['L'] == D+2 and descriptor['source_coordinate_tree_word_length_max'] == D+2,
            'actual FAR colour tree word')
    require(source_packet['eta'] == descriptor['screened_eta'] == 18 and
            source_packet['core']['alpha_expression'] == descriptor['screened_alpha_expression'] == '2^371',
            'same finite Haar source, not a retuned source')
    require(weak_bits >= 0, 'fixed positive dyadic weak family')
    family = descriptor['physical_family']
    original_ue = Fraction(family['UE0_exact'])
    speed = Fraction(family['c'])
    require(speed == Fraction(1130202988249777, 9007199254740992), 'authoritative exact carrier speed')
    electric = original_ue/(1 << weak_bits)
    magnetic = speed*speed/(2*electric)
    if weak_bits == family['division_exponent']:
        require(Fraction(family['UE']) == electric and Fraction(family['K']) == magnetic,
                'same exact selected family, not an additional rescaling of p527')
    require(kernel_packet['descriptor']['N'] == size and
            kernel_packet['descriptor']['reference'] == [0, D, D],
            'same FAR coherent-kernel consumer')
    for axis, key in ((0, 'incoming'), (1, 'outgoing_annihilation')):
        rows = descriptor[key]
        require(len(rows) == 2 and {row['positive_mode_number'] for row in rows} == {2**96, 2**96+1},
                'exact distinct native mode IDs, never merged floating momenta')
        require(sum(Fraction(row['weight_squared']) for row in rows) == 1,
                'normalized exact native photon weights')
        for row in rows:
            signed = row['mode_id'][axis]
            ratio = row['rational_k_over_2pi']
            require(Fraction(ratio['numerator'], ratio['denominator']) == Fraction(signed, size),
                    'actual finite-volume photon phase ratio')
    def ball(value: Fraction | int) -> Any:
        f = Fraction(value)
        return arb(f.numerator)/arb(f.denominator)
    def upper(value: Any) -> str:
        from decimal import Decimal, ROUND_CEILING, localcontext
        require(bool(value > 0), 'positive outward graph upper bound')
        with localcontext() as context:
            context.prec, context.rounding = 60, ROUND_CEILING
            candidate = Decimal(value.upper().str(70, radius=False))*Decimal('1.000000000000000000000000000001')
        text = str(candidate)
        require(bool(arb(text) > value.upper()), 'outward rational graph rounding')
        return text
    number = lambda row: complex(row['real'], row['imag'])
    raw_matrix = np.asarray([[number(row) for row in line] for line in descriptor['source_raw_q_over_2_matrix']])
    raw_hs = sum((abs(acb(z))**2 for z in raw_matrix.ravel()), arb(0))
    require(bool(abs(raw_hs-1) < arb('1e-12')), 'physical q/2 has unit HS: raw source_matrix/2, not /sqrt2')
    log_c = arb(descriptor['source_raw_profile_log_c_interval'])
    profile = sum((abs(acb(number(row)))**2*(2*length*log_c).exp()
                   for row,length in zip(descriptor['source_eta'], descriptor['source_coordinate_tree_word_lengths'])), arb(0))
    raw_source_core = raw_hs*profile/2
    declared_core = arb(descriptor['screened_core_source_Gram_interval'])
    require(bool(abs(raw_source_core-declared_core) < arb('1e-12')), 'raw PLUS state metric equals half HS, not anticommutator')
    raw_g0 = declared_core/2
    raw_n0 = raw_g0.sqrt()
    require(bool(raw_g0 > arb('0.2')), 'actual fixed common chart is bounded')
    # All sources share ONE constant chart scalar. This preserves raw Lowdin Y
    # even when finite-Haar actual diagonal source metrics are not equal.
    delta_b = 16*arb(str(source_packet['core']['mark_operator_perturbation']))
    source_metric_radius = 4*delta_b+delta_b**2
    ue, c, V, N = ball(electric), ball(speed), arb(volume), arb(size)
    require(bool(c > arb('0.1') and c < 1), 'matched carrier speed interval')
    tau, mu, w = arb(1), arb('0.1'), arb('0.1')
    cj = 192*V
    excitation = 1000*ue.sqrt()+3020*ue
    leakage_square = V*excitation/2
    leakage = leakage_square.sqrt() if leakage_square < 1 else arb(1)
    coulomb = ue*V*N*N/4
    energy = 16*V+coulomb+3*ue/(2*V)
    sigma = (2*energy/ball(magnetic)).sqrt()
    jet = arb(2**112)
    j0, j1, j2 = arb(5), jet, jet
    p1 = jet/(2*mu)
    p2 = jet/(2*mu)+2*jet*jet/(mu*mu)
    decay = w+2*mu
    i0 = j0/decay
    i1 = (j1+4*p1*j0)/decay+2*jet*j0/decay**2
    i2 = (j2+8*p1*j1+(4*p2+32*p1*p1)*j0)/decay
    i2 += (4*jet*j1+(2*jet+16*p1*jet)*j0)/decay**2+8*jet*jet*j0/decay**3
    residuals = dict(initial_current_vertex=8*tau*(3*V).sqrt()*sigma,
        initial_full_sea=(cj.sqrt()+2*V*j0)*leakage,
        light=24*V*V*tau*sigma*i0, Coulomb=4*coulomb*V*i0,
        harmonic=4*(ue/V).sqrt()*energy.sqrt()*arb(3).sqrt()*V*i1+6*ue*i2)
    # Factor2 safely covers the two exact, distinct frequency components.
    delta_u = 2*sum(residuals.values(), arb(0))/w**arb('1.5')
    umax = 2*cj.sqrt()/w**arb('1.5')
    tmax = 20*V.sqrt()/(w.sqrt()*decay)
    g_pieces = dict(actual_resolvent=4*V*delta_u*(umax+tmax),
                    actual_full_sea_moment=8*V*leakage*tmax*tmax)
    off_error = sum(g_pieces.values(), arb(0))
    photon_metric_radius = 8*ue*umax*umax
    input_photon_offset = 4*ue*umax*umax
    r = source_metric_radius+photon_metric_radius
    n_upper = 1+source_metric_radius+input_photon_offset
    require(bool(r < arb('0.01')), 'actual full-source global Gamma receiver Gram invertible')
    # Q here is the neutral excitation Hamiltonian H_N-E_Omega, not Pauli Q.
    qhalf = cj.sqrt()/w
    qfull = 2*cj.sqrt()/w.sqrt()
    positive_u = qhalf+energy.sqrt()*umax
    exponents = source_packet['derivative_q6']['normalized_power_of_two_upper_exponents']
    require(len(exponents) == 7, 'whole source derivatives q0..6')
    source_jets = [arb(2)**int(power) for power in exponents]
    # Direct harmonic D jets plus a conservative possible native-frame chain.
    phi_first, phi_second = arb(2)**35*V.sqrt(), arb(2)**104*V.sqrt()
    d1 = 6*(D+1)+source_jets[1]*phi_first
    d2 = 12*(D+1)**2+12*(D+1)*source_jets[1]*phi_first
    d2 += source_jets[2]*phi_first**2+source_jets[1]*phi_second
    d1_time = d1+24*jet
    d2_time = d2+48*jet*d1+24*jet+288*jet*jet
    positive_bu = 2*positive_u+(ue/V).sqrt()*d1_time*umax
    field_u = (2/ball(magnetic)).sqrt()*positive_u
    l1 = arb(2**20)  # Paid finite-T<=6 polynomial window, not exp(VT).
    force = 2*N*V.sqrt()+arb('3.25')
    rb_parts = dict(light=6*tau*l1*field_u, Coulomb=ue*force*l1*umax,
        harmonic=2*(ue/V).sqrt()*d1_time*positive_u+ue/V*d2_time*umax)
    rb_u = sum(rb_parts.values(), arb(0))
    charged_q_bu = 2*qfull+4*umax+rb_u
    jet_multiplier = sum((math.comb(6,q)*source_jets[q]*(1+D)**(6-q) for q in range(7)), arb(0))
    metric_multiplier = (1+umax+qhalf+qfull+positive_u+positive_bu+charged_q_bu+rb_u)*jet_multiplier
    # Exact Fock creation graph: ||Hph^.5 a†g psi|| <=
    # ||g/sqrtω|| ||Hph psi|| + ||sqrtω g|| ||psi||.
    # Selected source commutes with Hph. Frequencies of these modes are <1.
    # A form bound Hph+Hh<=Q+S does not itself give ||Hph Omega||.
    # Bounded Gamma light/Coulomb terms and |Eactual| give the full norm 2S.
    ground_photon_full = 2*energy
    prepared_x_graph = 2*ground_photon_full/w.sqrt()+2
    prepared_input_graph = prepared_x_graph/(1-source_metric_radius).sqrt()
    prepared_output_graph = arb(2).sqrt()*prepared_x_graph/(1-r).sqrt()
    prepared_photon_number = prepared_input_graph**2*N/(4*c)
    metric_multiplier += (prepared_input_graph+prepared_output_graph)*jet_multiplier
    static = kernel_packet['static_receiver']
    leading_lower = arb(static['every_actual_normalized_nu_GLOBAL_leading_rescaled_lower'])
    minimum = 1-source_metric_radius
    # Finite-alpha full-quartet/source-map calibration is already paid by the
    # supplied GLOBAL disk. Add ONLY actual-photon metric vs conditional-metric
    # transfer, not a new trace or a second finite-alpha chart adjustment.
    primitive_upper = arb(2**32)
    metric_added = off_error*(1+source_metric_radius)/(2*minimum**2)
    metric_added += photon_metric_radius*primitive_upper/(2*(minimum-photon_metric_radius)**2)
    off_remainder = 3*photon_metric_radius*(primitive_upper+off_error)*(1+source_metric_radius)
    metric_added += off_remainder/(4*(minimum-photon_metric_radius)**arb('2.5')*minimum.sqrt())
    metric_added += input_photon_offset*primitive_upper/(2*minimum**2)
    joined_margin = leading_lower-metric_added
    coefficient_lower = _positive_lower(joined_margin/V) if joined_margin > 0 else None
    amplitude_lower = _positive_lower(ue*joined_margin/V) if joined_margin > 0 else None
    return dict(N=size, V=volume, reference=[0, D, D], source_origin=[D, D, D],
        coordinate_tree_word_max=D+2, recoil_reference_phase=-1,
        baseline_UE_exact=str(original_ue), weak_bits_total=weak_bits,
        UE_exact=str(electric), K_exact=str(magnetic), c_exact=str(speed),
        K_rule='c²/(2UE) EXACT; rational never cast to underflowing float',
        raw_source_core_Gram=raw_source_core.str(65), raw_photon_core_Gram=raw_g0.str(65),
        raw_input_core_norm=raw_n0.str(65), raw_q_over_2_HS_squared=raw_hs.str(65),
        normalized_chart='ONE constant common scalar n0core; no independent-column/fibre normalization',
        source_metric_radius_in_common_chart_upper=upper(source_metric_radius),
        native_Gamma_enclosure=dict(tau_upper='1', mu_lower='1/10', B_upper='2',
                                   nonzero_photon_omega_lower='1/10', current_ah_twojet_upper=str(2**112)),
        neutral_excitation_density_upper=upper(excitation), whole_sea_leakage_upper=upper(leakage),
        actual_current_state_residual_components={k:upper(v) for k,v in residuals.items()},
        actual_U_minus_conditional_T_Omega_norm_upper=upper(delta_u),
        actual_offGram_volume_rescaled_error_components={k:upper(v) for k,v in g_pieces.items()},
        actual_offGram_volume_rescaled_error_upper=upper(off_error),
        raw_offGram_volume_rescaled_error_upper=upper(raw_g0*off_error),
        actual_global_Gram_radius_upper=upper(r), raw_actual_global_Gram_radius_upper=upper(raw_g0*r),
        actual_photon_Gram_radius_upper=upper(photon_metric_radius),
        actual_input_photon_norm_upper_offset=upper(input_photon_offset),
        actual_input_norm_upper=upper(n_upper), raw_actual_input_norm_upper=upper(raw_n0*n_upper),
        source_branch='single PLUS NL+1/bar4; selected C-real neutral effectiveGamma ground',
        source_norm_metric='raw .5HS; ideal common-chart1, finite-Haar source metric deviation retained',
        coherent_nu='same actual normalized marginal; no vacuum/fibretrace replacement',
        actual_Gamma_offGram_transfer_positive=bool(off_error < arb('0.01')),
        global_kernel_consumer_reference=kernel_packet['natural_exit_receipt'],
        supplied_GLOBAL_finite_Haar_leading_lower=static['every_actual_normalized_nu_GLOBAL_leading_rescaled_lower'],
        actual_metric_added_rescaled_amplitude_error_upper=upper(metric_added),
        actual_Gamma_metric_joined_rescaled_margin=joined_margin.str(65),
        actual_Gamma_metric_joined_coefficient_lower=coefficient_lower,
        actual_Gamma_metric_conditional_flight_amplitude_lower=amplitude_lower,
        actual_Gamma_metric_conditional_flight_square_lower=
            _positive_lower(arb(amplitude_lower)**2) if amplitude_lower else None,
        actual_Gamma_graph=dict(U_norm_upper=upper(umax), neutral_Q_half_U_norm_upper=upper(qhalf),
            neutral_Q_U_norm_upper=upper(qfull), positive_Hph_harmonic_half_U_norm_upper=upper(positive_u),
            positive_Hph_harmonic_half_B_U_norm_upper=upper(positive_bu),
            charged_excitation_B_U_norm_upper=upper(charged_q_bu),
            RB_U_norm_upper=upper(rb_u), RB_U_components={k:upper(v) for k,v in rb_parts.items()},
            RB_U_positive_energy_half_norm_supplied=False,
            units='nativea1; normalized Ug BEFORE sqrtUE; Q=Hneutral−EactualOmega'),
        actual_Gamma_prepared_photon_graph=dict(
            Omega_Hph_norm_upper=upper(ground_photon_full),
            normalized_original_input_Hph_half_norm_upper=upper(prepared_input_graph),
            original_global_Y_Hph_half_norm_upper=upper(prepared_output_graph),
            normalized_original_input_total_photon_number_upper=upper(prepared_photon_number),
            softest_coexact_omega_lower=(4*c/N).str(65),
            exact_creation_graph=True, actual_Gamma_not_static_Wick=True,
            microscopic_original_Omega_requires_normal_graph_transfer=True),
        normal_metric_source_input=dict(source_q6_exponents=exponents,
            source_harmonic_first_upper=upper(d1_time), source_harmonic_second_upper=upper(d2_time),
            mixed_source_sixjet_multiplier_upper=upper(jet_multiplier),
            metric_weighted_source_multiplier_upper=upper(metric_multiplier),
            requested_transfer_error=str(electric/Fraction(volume*2**20))),
        microscopic_W_normal_source_transfer_paid=False,
        raw_static_flight_transfer_paid=False, actual_Ei_probability_certified=False)


def encoded(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, complex):
        return dict(real=value.real, imag=value.imag)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def hashes() -> dict[str, str]:
    result = scattering.hashes()
    for name in ('noncompact_actual_receiver_gram.py', 'noncompact_actual_receiver_gram.md'):
        path = ROOT/name
        if path.exists():
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', action='store_true')
    parser.add_argument('--coupled', action='store_true')
    parser.add_argument('--cutoff', type=int, default=5)
    parser.add_argument('--legal-descriptor', type=Path)
    parser.add_argument('--weak-bits', type=int, default=4096)
    parser.add_argument('--screened-packet', type=Path)
    parser.add_argument('--kernel-packet', type=Path)
    args = parser.parse_args()
    require(args.native or args.coupled or args.legal_descriptor, 'select scientific consumer')
    before = hashes()
    require(before['noncompact_photon_packet_supplier.py'] == FROZEN_SUPPLIER, 'frozen original observable supplier')
    result: dict[str, Any] = dict(plan=PLAN, source_sha256=before,
        runtime=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, executable=sys.executable))
    if args.native:
        result['native'] = native_consumer()
    if args.coupled:
        result['coupled'] = coupled_consumer(args.cutoff)
    if args.legal_descriptor:
        require(args.screened_packet and args.kernel_packet, 'same source and coherent kernel packets required')
        input_paths = dict(descriptor=args.legal_descriptor, source=args.screened_packet, kernel=args.kernel_packet)
        inputs = {key:path.read_bytes() for key,path in input_paths.items()}
        result['input_sha256'] = {key:hashlib.sha256(value).hexdigest() for key,value in inputs.items()}
        result['legal'] = legal_gram_budget(json.loads(inputs['descriptor']), json.loads(inputs['source']),
                                           json.loads(inputs['kernel']), args.weak_bits)
        if args.native:
            result['native_far_core'] = native_consumer(json.loads(inputs['descriptor']))
    require(before == hashes(), 'consumer used unchanged scientific sources')
    print(json.dumps(result, indent=2, default=encoded))


if __name__ == '__main__':
    main()
