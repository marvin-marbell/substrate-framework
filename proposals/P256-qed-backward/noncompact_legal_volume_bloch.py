"""Same-N sparse conditional scattering at N=5*2**96; no native grid allocation."""
from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
from flint import arb, ctx
import noncompact_harmonic_scattering_fiber as fiber
from sea_selection import SELECTED

ROOT = Path(__file__).resolve().parent
COMMON_N = 5 * 2**96
FROZEN_FIBER_SHA = '0efb1e4adfc231d70a37be2a6af9e8f5dfbcf539a9f1a9ef36fbf997f0bf3351'


def binary64_ball(value: float) -> Any:
    """The canonical binary64 parameter as an exact rational, not its decimal display."""
    exact = Fraction.from_float(value)
    return arb(exact.numerator)/exact.denominator


def positive_lower(value: Any) -> str:
    """A decimal rational proved strictly below the ball's lower endpoint."""
    if not value > 0:
        raise AssertionError(f'Nonpositive certified lower: {value}')
    with localcontext() as context:
        context.prec, context.rounding = 60, ROUND_FLOOR
        candidate = Decimal(value.lower().str(70, radius=False)) * Decimal('0.999999999999999999999999999999')
    text = str(candidate)
    if not 0 < arb(text) < value.lower():
        raise AssertionError('Decimal lower rounding did not remain outward')
    return text


def positive_upper(value: Any) -> str:
    """A decimal rational proved strictly above a positive ball's upper endpoint."""
    if not value > 0:
        raise AssertionError('Positive upper bound required')
    with localcontext() as context:
        context.prec, context.rounding = 60, ROUND_CEILING
        candidate = Decimal(value.upper().str(70, radius=False)) * Decimal('1.000000000000000000000000000001')
    text = str(candidate)
    if not arb(text) > value.upper():
        raise AssertionError('Decimal upper rounding did not remain outward')
    return text


def capped_decay(exponent: Any) -> Any:
    """Never replace a tiny positive exponential by binary64 zero."""
    cap = arb(2)**-1024
    if exponent >= arb(1024)*arb(2).log():
        return cap
    return (-exponent).exp()


def alias_factor(rho: Any) -> Any:
    # Algebraically ((1+rho)/(1-rho))**3-1, without subtracting close ones.
    return 2*rho*(3+rho*rho)/(1-rho)**3


def exact_modes(size: int, speed: float) -> dict[str, Any]:
    if size < 2000 or size % 10:
        raise ValueError('Even finite N multiple5, at least2000, required')
    ctx.prec = 256
    q0, delta = 2*arb.pi()/5, 2*arb.pi()/size
    origin, reference = size//2, size//2-2
    photons = []
    for axis, sign in ((0, 1), (1, -1)):
        rows = []
        for offset, ratio in ((0, Fraction(1, 1)), (1, Fraction(1, 4))):
            number = size//5+offset
            mode = [0, 0, 0]
            mode[axis] = sign*number
            # Reduce the exact integer centre phase BEFORE any floating conversion.
            phase_turns = Fraction(-sign*number*origin, size) % 1
            phase = ((phase_turns*2).denominator == 1)
            if not phase:
                raise AssertionError('Exact centre phase must be a parity sign')
            parity = 1 if phase_turns == 0 else -1
            wave = q0+offset*delta
            weight = arb(ratio.numerator)/ratio.denominator/(arb(17)/16).sqrt()
            omega = 2*binary64_ball(speed)*(wave/2).sin()
            endpoint_phase = wave if sign == -1 else arb(0)
            rows.append(dict(mode_id=mode, positive_mode_number=number,
                rational_k_over_2pi=dict(numerator=sign*number, denominator=size),
                exact_origin_phase_turns=str(phase_turns), origin_parity=parity,
                weight_interval=weight.str(65), weight_squared=str(ratio**2/Fraction(17, 16)),
                k_absolute_interval=wave.str(65), omega_interval=omega.str(65),
                origin_stripped_coefficient_phase_interval=endpoint_phase.str(65),
                global_coordinate_coefficient_rule='parity * weight * exp(i endpoint_phase)',
                frequency_sign=sign, polarization=[0, 0, 1]))
        photons.append(rows)
    if sum(Fraction(row['weight_squared']) for row in photons[0]) != 1:
        raise AssertionError('Physical photon norm must remain sum of distinct mode weights squared')
    return dict(N=size, V=size**3, origin=[origin]*3, reference=[reference]*3,
        outgoing_center=[origin, origin+1, origin], cut_distance=size//2-2,
        q0_interval=q0.str(65), delta_k_interval=delta.str(65),
        recoil_intervals=[delta.str(65), (-delta).str(65), '0'],
        incoming=photons[0], outgoing_annihilation=photons[1],
        distinct_exact_modes=True, float_momenta_may_coincide_but_modes_remain_orthogonal=True,
        physical_input='unwhitened a†g B_eta Omega/sqrt2, normalized by actual n',
        physical_output='STATIC global coherent X_out G_out^-1/2; not input-whitened',
        source_covariant_phase='exp[-i(theta/N+a_h).(d+2)]; a_star=-theta/N',
        reference_distance_from_origin=[2, 2, 2], no_O_N_reference_oscillation=True)


def represented_packets(speed: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """Intermediate q0 kernel only. The nonzero SAME-N delta debt is paid separately."""
    ctx.prec = 256
    q0 = float((2*arb.pi()/5).mid())
    omega = float((2*binary64_ball(speed)*(arb.pi()/5).sin()).mid())
    packets = []
    for axis, sign in ((0, 1), (1, -1)):
        rows = []
        for weight in (4/math.sqrt(17), 1/math.sqrt(17)):
            k = np.zeros(3)
            k[axis] = sign*q0
            coefficient = weight*complex(fiber.bounded_phase(np.asarray(q0))) if sign == -1 else weight
            rows.append(dict(k=k, omega=omega, coefficient=coefficient,
                polar=np.array([0., 0., 1.]), frequency_sign=sign))
        packets.append(dict(modes=rows))
    return packets[0], packets[1]


def compact_source() -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Extract the exact local 19-site formula, not an N5 scattering observation."""
    base = fiber.setup(5)
    data = fiber.fiber_data(base, -np.asarray(base['centre'])*math.pi/10)
    law, origin = base['law'], np.asarray(base['origin'])
    position = data['momentum'].to_position(data['full_source']).reshape(law.volume, 4, 4)
    site = int(law.index[tuple(origin)])
    matrix = position[site]*math.sqrt(2)/base['eta'][site]
    support = abs(base['eta']) > 0
    displacement, coefficients = law.sites[support]-origin, base['eta'][support]
    fiber.accept(np.linalg.norm(position[support]-coefficients[:, None, None]*matrix/math.sqrt(2)),
        'same compact fullbar4 source/tree at compensated reference')
    if len(displacement) != 19 or int(np.max(abs(displacement))) != 1:
        raise AssertionError('Declared centred 19-site source changed')
    return base, matrix, displacement, coefficients


def represented_mean(quadrature: int, duration: float, frame: np.ndarray,
                     matrix: np.ndarray, displacement: np.ndarray, eta: np.ndarray,
                     incoming: dict[str, Any], outgoing: dict[str, Any]) -> tuple[complex, float]:
    # Bilinearity sums four representative rows; it does NOT renormalize merged modes.
    first = dict(incoming['modes'][0], coefficient=sum(row['coefficient'] for row in incoming['modes']))
    second = dict(outgoing['modes'][0], coefficient=sum(row['coefficient'] for row in outgoing['modes']))
    probe = fiber.SparseBlochFiber(np.array([[.17, .39, .61], [1.1, 2.2, 3.3]]), frame, duration)
    independent = sum(probe.connected_pair(i, o) for i in incoming['modes'] for o in outgoing['modes'])
    bilinear_error = float(np.linalg.norm(independent-probe.connected_pair(first, second)))
    fiber.accept(bilinear_error, 'four distinct representative mode terms remain bilinear')
    count, total = quadrature**3, 0j
    for start in range(0, count, 2048):
        indices = np.arange(start, min(start+2048, count))
        points = 2*math.pi*np.column_stack(np.unravel_index(indices, (quadrature,)*3))/quadrature
        kernel = fiber.SparseBlochFiber(points, frame, duration)
        source = fiber.compact_fourier(points, displacement, eta)
        detector = fiber.compact_fourier(points+first['k']+second['k'], displacement, eta)
        block = kernel.connected_pair(first, second)
        trace = np.trace(matrix.conj().T@block@matrix, axis1=1, axis2=2)/2
        total += complex(np.sum(detector.conj()*source*trace))
    return total/count, bilinear_error


def momentum_lipschitz(budgets: dict[str, Any], speed: float, duration: float,
                       eta: np.ndarray, incoming: dict[str, Any]) -> dict[str, Any]:
    """Full reduced-word derivative along the entire finite-N splitting segment."""
    tau, mu = budgets['endpoint_norm_upper'], budgets['gap_lower']
    omega = .999*min(row['omega'] for row in incoming['modes'])
    field = sum(abs(row['coefficient']) for row in incoming['modes'])/math.sqrt(omega)
    current, contact = 6*tau*field, 6*tau*field*field
    den, dh = 2*mu+omega, 4*tau
    rho, dp = 1+speed/(2*omega), dh/mu
    limits = dict(P=1., Q=1., a=duration*current, b=duration*current,
        c=duration*contact+duration**2*current**2,
        X=current/den, Y=current/den, Y0=current/den)
    dj = current*rho
    dx = (2*dp*current+dj)/den+(2*dh+speed)*current/den**2
    da = duration*dj+duration**2*current*(2*dh+speed)
    dc = (duration*(2*rho*contact+duration*(2*dh+2*speed)*contact)
        +duration**2*(2*rho+4*duration*dh+2*duration*speed)*current**2)
    changes = dict(P=dp, Q=dp, a=da, b=da, c=dc, X=dx, Y0=dx,
        Y=dx+duration*(2*dh+speed)*current/den)
    eta_l1 = float(np.sum(abs(eta)))
    word_change = 2*eta_l1**2*fiber.word_error(limits, changes)
    # Detector argument changes by <=delta in each x/y direction; |d|_1<=2.
    detector_change = 4*eta_l1**2*fiber.word_norms(limits)
    return dict(coefficient_derivative_upper=1.01*(word_change+detector_change),
        primitive_norms=limits, primitive_derivatives=changes,
        omega_segment_lower=omega, maximum_h_shift_derivative=dh,
        source_detector_derivative_contribution=detector_change,
        segment='q_r=q0+r*delta,r=0,1; detector shift−recoil(delta,−delta); full P/Q, contact,bothhistories, X,YT,Y0',
        method='uniform first derivative on whole real momentum/frequency segment; not a finite difference or an unbounded Taylor remainder')


def legal_case(metric_payload: dict[str, Any], size: int = COMMON_N,
               quadrature: int = 80, duration: float = 6.) -> dict[str, Any]:
    if size < COMMON_N or size % 10 or not 0 < duration <= 6:
        raise ValueError('Common legal-volume successor requires even multiple5 N>=5*2**96,0<T<=6')
    base, matrix, displacement, eta = compact_source()
    constants = base['constants']
    speed, ue = float(constants['bare_speed']), float(constants['UE'])
    descriptor = exact_modes(size, speed)
    incoming, outgoing = represented_packets(speed)
    metric = metric_payload['certificate']
    g0, gs = map(float, metric['trial_decimal'])
    frame = np.array([math.sqrt(g0), *([math.sqrt(gs)]*3)])[:, None, None]*SELECTED
    value, bilinear_error = represented_mean(quadrature, duration, frame, matrix,
        displacement, eta, incoming, outgoing)
    # Analytic monotone N>=2000 metric envelope, not the observed C2000.
    budgets = fiber.bloch_certificate_budgets(metric, 2000, quadrature, duration,
        incoming, outgoing, displacement, eta)
    ctx.prec = 256
    lipschitz = momentum_lipschitz(budgets, speed, duration, eta, incoming)
    delta = 2*arb.pi()/size
    split_error = arb(str(lipschitz['coefficient_derivative_upper']))*delta
    native_rho = capped_decay(arb(str(budgets['complex_strip']))*size)
    native_error = arb(str(budgets['integrand_strip_majorant']))*alias_factor(native_rho)
    native_metric_distance = (arb(str(metric['full_ten_gradient']['Frobenius_residual_upper']))
        +arb(str(metric['continuum_quadrature_error_upper']))
        +arb(str(metric['analytic_strip']['matrix_gradient_integrand_HS_majorant']))*alias_factor(native_rho))/2
    if not native_metric_distance < arb(str(budgets['frame_distance_upper'])):
        raise AssertionError('Same-N full-ten native metric escaped monotone analytic envelope')
    relative = displacement+2
    first = np.sqrt(2*np.sum(abs(eta[:, None])**2*relative**2, axis=0))
    second = np.sqrt(2*np.sum(abs(eta[:, None])**2*relative**4, axis=0))
    scalar = fiber.theorem_bounds(constants, base['law'], base['eta'], base['reference'], duration)
    eh = scalar['neutral_harmonic_energy_per_volume_upper']
    # Two-photon homogeneity: V*seam_N is the unit-V prefactor times exp(-kappa d_N).
    seam_t = fiber.uniform_enclosure(budgets['endpoint_norm_upper'], budgets['gap_lower'],
        budgets['spectrum_norm_upper'], duration, incoming, outgoing, 1, 0., first, second, ue, eh)
    seam_0 = fiber.uniform_enclosure(budgets['endpoint_norm_upper'], budgets['gap_lower'],
        budgets['spectrum_norm_upper'], 0., incoming, outgoing, 1, 0., first, second, ue, eh)
    kappa_lower = arb(str(seam_t['kappa']))*arb('0.999999999999')
    seam_rho = capped_decay(kappa_lower*(size//2-2))
    seam_prefactor = 1.01*(seam_t['massive_seam_tail_raw_upper']+seam_0['massive_seam_tail_raw_upper'])
    seam_error = arb(str(seam_prefactor))*seam_rho
    reference_error = sum((arb(str(budgets[key])) for key in (
        'native_frame_coefficient_error_upper', 'continuum_quadrature_error_upper', 'arithmetic_coefficient_error_upper')), arb(0))
    lower_ball = arb(abs(value))-reference_error-native_error-split_error-seam_error
    rescaled_lower = positive_lower(lower_ball)
    coefficient_lower = positive_lower(arb(rescaled_lower)/(size**3))
    physical_lower = positive_lower(arb(ue)*arb(coefficient_lower))
    square_lower = positive_lower(arb(physical_lower)**2)
    physical_field = sum((arb(row['weight_interval'])/arb(row['omega_interval']).sqrt()
        for row in descriptor['incoming']), arb(0))/arb(size**3).sqrt()
    tau_upper = arb(str(budgets['endpoint_norm_upper']))*arb('1.000000000001')
    current_onebody = 2*tau_upper*physical_field
    current_fullcar = 8*tau_upper*(size**3)*physical_field
    contact_onebody = 2*tau_upper*physical_field**2
    contact_fullcar = 8*tau_upper*(size**3)*physical_field**2
    descriptor.update(dict(compact_displacement=displacement, source_eta=eta,
        source_matrix=matrix, relative_X_minus_XR=relative,
        source_absolute_sites=[[size//2+int(coordinate) for coordinate in row] for row in displacement],
        normalized_source_HS_first_axis=first, normalized_source_HS_second_axis=second,
        source_coordinate_radius=1, source_L1_radius=2, source_max_relative_coordinate=3,
        source_L1_relative_radius=8, source_norm_squared=2.,
        physical_family=dict(c=speed, UE=ue, K=constants['K'], no_retuning=True),
        beam_strip_UE_coordinate_field_upper=positive_upper(physical_field),
        onebody_current_zpolar_upper=positive_upper(current_onebody),
        fullCAR_current_zpolar_upper=positive_upper(current_fullcar),
        onebody_contact_upper=positive_upper(contact_onebody),
        fullCAR_contact_upper=positive_upper(contact_fullcar),
        harmonic_h_first_and_second_axis_upper=positive_upper(2*tau_upper),
        norm_units='photon coordinate strips sqrtUE; exactV retained, J_CAR proportional sqrtV, contact_CAR O(1)',
        tau=budgets['endpoint_norm_upper'], mu=budgets['gap_lower'], B=budgets['spectrum_norm_upper'],
        actual_full10_Gram_distance_interval=native_metric_distance.str(65),
        coefficient_positive_lower=coefficient_lower,
        physical_unwhitened_candidate_amplitude_lower=physical_lower))
    return dict(N=size, V=size**3, no_native_grid_enumeration=True, quadrature_points=quadrature**3,
        descriptor=descriptor, fixed_trial_metric=np.diag([g0, gs, gs, gs]),
        computed_representative_rescaled_coefficient=value,
        not_C2000_or_universal_C_infinity=True,
        finite_N_splitting_recoil_error_interval=split_error.str(65),
        finite_N_momentum_lipschitz=lipschitz,
        native_N_periodization_error_interval=native_error.str(65),
        same_N_seam_error_interval=seam_error.str(65),
        seam_unit_volume_prefactor_upper=seam_prefactor,
        seam_positive_exponential_bound=seam_rho.str(65),
        reference_error_interval=reference_error.str(65), inherited_analytic_budgets=budgets,
        metric_envelope_auxiliary_order=2000,
        actual_native_metric_distance_interval=native_metric_distance.str(65),
        every_actual_normalized_nu_coherent_rescaled_lower=rescaled_lower,
        every_actual_normalized_nu_coherent_coefficient_lower=coefficient_lower,
        physical_unwhitened_candidate_amplitude_lower=physical_lower,
        square_of_conditional_amplitude_lower=square_lower,
        amplitude_log10_interval=(arb(physical_lower).log()/arb(10).log()).str(65),
        conditional_square_log10_interval=(arb(square_lower).log()/arb(10).log()).str(65),
        bilinear_four_mode_term_error=bilinear_error,
        collapsed_and_renormalized_mode_adversary_norm_squared=25/17,
        binary64_square_underflows=float(arb(square_lower)) == 0.,
        actual_global_Gram_whitened_probability_certified=False,
        normalization='C_N=mean(f_N)/V, physicalcandidateUE*C_N, two orthogonal global photon modes each1/sqrtV; compact source/ref each1/sqrtV',
        scalar_moment_EH=eh, scalar_paid_Esea_retained=True,
        scalar_moment_scope='conservative supplied N5 bound monotone forN>=5; not actual legal-volume ground simulation',
        receiver_transfer='STATIC Gout^-1/2m(T)/n; baremx0≠1; leadingyUE*(my1−.5g1_yx mx0)/n0 still needs actual reference/cloud/flight calibration',
        scalar_phase_scope='selectedN meets owner-reported sufficient volume threshold; this consumer proves scattering coefficient only, not phase/normal/legalLambda admission')


def cloud_pair(kernel: Any, incoming: dict[str, Any], outgoing: dict[str, Any]) -> np.ndarray:
    """Full unfiltered R_xy(0), not a positive-final source chart."""
    zero = np.zeros(3)
    x0 = kernel.preparation(outgoing, zero, True)
    xi = kernel.preparation(outgoing, incoming['k'], True)
    y0 = kernel.preparation(incoming, zero, False, 0.)
    yo = kernel.preparation(incoming, outgoing['k'], False, 0.)
    return -xi@y0+yo@x0


def static_means(result: dict[str, Any], quadrature: int, duration: float) -> tuple[complex, complex, complex]:
    descriptor = result['descriptor']
    speed = descriptor['physical_family']['c']
    incoming, outgoing = represented_packets(speed)
    first = dict(incoming['modes'][0], coefficient=sum(row['coefficient'] for row in incoming['modes']))
    second = dict(outgoing['modes'][0], coefficient=sum(row['coefficient'] for row in outgoing['modes']))
    out_static = dict(second, coefficient=second['coefficient']*complex(
        fiber.bounded_phase(np.asarray(-duration*second['omega']))))
    forward_receiver = dict(first, coefficient=first['coefficient']*complex(
        fiber.bounded_phase(np.asarray(-first['k'][0]))))
    bare_photon = complex(fiber.bounded_phase(np.asarray(first['k'][0]-duration*first['omega'])))
    gram = result['fixed_trial_metric']
    frame = np.sqrt(np.diag(gram))[:, None, None]*SELECTED
    matrix = descriptor['source_matrix']
    displacement, eta = descriptor['compact_displacement'], descriptor['source_eta']
    totals = np.zeros(3, dtype=complex)
    count = quadrature**3
    for start in range(0, count, 2048):
        indices = np.arange(start, min(start+2048, count))
        points = 2*math.pi*np.column_stack(np.unravel_index(indices, (quadrature,)*3))/quadrature
        kernel = fiber.SparseBlochFiber(points, frame, duration)
        source = fiber.compact_fourier(points, displacement, eta)
        detector = fiber.compact_fourier(points+first['k']+second['k'], displacement, eta)
        final = kernel.info(first['k']+second['k'])
        free_final = (fiber.bounded_phase(-duration*final['E'])[:, None, None]*final['Q']
            +fiber.bounded_phase(duration*final['E'])[:, None, None]*final['P'])
        raw = kernel.connected_pair(first, out_static)+cloud_pair(kernel, first, out_static)
        gram_cloud = cloud_pair(kernel, forward_receiver, second)
        initial = kernel.info(np.zeros(3))
        bare_charge = fiber.bounded_phase(-duration*initial['E'])[:, None, None]*initial['Q']
        for index, block in enumerate((free_final@raw, gram_cloud, bare_charge)):
            trace = np.trace(matrix.conj().T@block@matrix, axis1=1, axis2=2)/2
            weights = detector.conj()*source if index < 2 else abs(source)**2*bare_photon
            totals[index] += np.sum(weights*trace)
    return complex(totals[0]/count), complex(totals[1]/count), complex(totals[2]/count)


def static_case(metric_payload: dict[str, Any], size: int, quadrature: int, duration: float) -> dict[str, Any]:
    """Three SAME-N coherent kernels and their GLOBAL-product disk; no input whitening."""
    result = legal_case(metric_payload, size, quadrature, duration)
    c_static, g_off, m_forward = static_means(result, quadrature, duration)
    descriptor, budgets = result['descriptor'], result['inherited_analytic_budgets']
    metric = metric_payload['certificate']
    eta, displacement = descriptor['source_eta'], descriptor['compact_displacement']
    speed = descriptor['physical_family']['c']
    tau, mu, B = descriptor['tau'], descriptor['mu'], descriptor['B']
    eta_l1 = float(np.sum(abs(eta)))
    duration_norm = result['finite_N_momentum_lipschitz']['primitive_norms']
    raw_norm = 2*eta_l1**2*fiber.word_norms(duration_norm)
    real_majorant = budgets['real_integrand_majorant']
    g0, gs = map(float, metric['trial_decimal'])
    distance = budgets['frame_distance_upper']
    root_lipschitz = 1/(math.sqrt(gs)+math.sqrt(gs-distance))
    h_change = math.sqrt(.92**2+3*.3**2)*root_lipschitz
    e, epsilon = budgets['declared_elementary_error'], 2.**-52
    eh = 8*B*e
    ee = B*eh/mu+100*epsilon*B*B/mu+e
    ep = budgets['primitive_arithmetic_errors']['P']
    e_free = 2*ep+2*duration*ee+2*e+300*epsilon
    e_out = (1+duration)*e+300*epsilon
    sigma = budgets['complex_strip']
    ctx.prec = 256
    cosh = math.nextafter(float(arb(str(sigma)).cosh().upper()), math.inf)
    sinh = math.nextafter(float(arb(str(sigma)).sinh().upper()), math.inf)
    bim = 1.001*(3*.12*math.sqrt(g0)+.3*math.sqrt(3*gs))*sinh
    strip_free = 1.01*math.nextafter(float((arb(str(bim))*duration).exp().upper()), math.inf)
    strip_l1 = float(np.sum(abs(eta)*np.exp(sigma*np.sum(abs(displacement), axis=1))))
    complex_B = 1.001*(math.sqrt(g0)*(.2+3*.12*(1+cosh))+.3*math.sqrt(3*gs)*cosh)
    projector = (1+complex_B/budgets['complex_strip_coercivity_lower'])/2
    bare_strip = 1.01*2*strip_l1**2*projector*strip_free
    ctx.prec = 256
    rho_q = (-arb(str(sigma))*quadrature).exp()
    rho_n = capped_decay(arb(str(sigma))*size)
    delta = 2*arb.pi()/size
    c_frame = budgets['native_frame_coefficient_error_upper']+1.01*raw_norm*duration*h_change*distance
    c_arithmetic = budgets['arithmetic_coefficient_error_upper']+1.01*raw_norm*(e_free+e_out+e_free*e_out)
    c_strip = budgets['integrand_strip_majorant']*strip_free
    c_derivative = (result['finite_N_momentum_lipschitz']['coefficient_derivative_upper']
        +1.01*raw_norm*duration*(4*tau+speed))
    g_frame = budgets['native_frame_coefficient_error_upper']
    g_arithmetic = budgets['arithmetic_coefficient_error_upper']+1.01*real_majorant*(4*e+300*epsilon)
    g_strip = budgets['integrand_strip_majorant']
    m_frame = 1.01*2*eta_l1**2*(h_change/mu+duration*h_change)*distance
    m_arithmetic = (1.01*2*eta_l1**2*(ep+duration*ee+2*e+300*epsilon)
        +2*eta_l1**2*(1000*epsilon+4*e+400*epsilon*quadrature**3))
    incoming, outgoing = represented_packets(speed)
    first = descriptor['normalized_source_HS_first_axis']
    second = descriptor['normalized_source_HS_second_axis']
    ue, energy = descriptor['physical_family']['UE'], result['scalar_moment_EH']
    contour = fiber.uniform_enclosure(tau, mu, B, duration, incoming, outgoing, 1, 0., first, second, ue, energy)
    contour0 = fiber.uniform_enclosure(tau, mu, B, 0., incoming, outgoing, 1, 0., first, second, ue, energy)
    kappa, cp = contour['kappa'], contour['projector_weight_upper']
    growth = 6*tau*math.expm1(kappa)
    weighted = sum(abs(scale)*math.prod(contour['primitives'][name]['weighted'] for name in word)
        for scale, word in fiber.WORDS)
    free_weight = math.exp(2*growth*duration)
    c_seam_prefactor = 1.01*(free_weight*contour['massive_seam_tail_raw_upper']
        +2*duration*12*tau*free_weight*weighted)
    g_seam_prefactor = 1.01*contour0['massive_seam_tail_raw_upper']
    m_seam_prefactor = 1.01*2*eta_l1**2*free_weight*(cp*4*12*tau/mu+cp*duration*12*tau)
    seam_rho = capped_decay(arb(str(kappa))*arb('0.999999999999')*(size//2-2))
    components = []
    for frame_error, arithmetic, strip, derivative, seam in (
        (c_frame, c_arithmetic, c_strip, c_derivative, c_seam_prefactor),
        (g_frame, g_arithmetic, g_strip, result['finite_N_momentum_lipschitz']['coefficient_derivative_upper'], g_seam_prefactor),
        (m_frame, m_arithmetic, bare_strip, 2*eta_l1**2*(1+duration*speed), m_seam_prefactor)):
        pieces = dict(frame=arb(str(frame_error)), arithmetic=arb(str(arithmetic)),
            quadrature=arb(str(strip))*alias_factor(rho_q), native_periodization=arb(str(strip))*alias_factor(rho_n),
            finite_N_splitting_recoil=arb(str(derivative))*delta, same_N_seam=arb(str(seam))*seam_rho)
        radius = sum(pieces.values(), arb(0))
        components.append((radius, {name:value.str(65) for name,value in pieces.items()}))
    ec, eg, em = (item[0] for item in components)
    reference = c_static-.5*g_off*m_forward
    global_radius = ec+arb('.5')*(arb(abs(g_off))*em+arb(abs(m_forward))*eg+eg*em)
    lower = positive_lower(arb(abs(reference))-global_radius)
    coefficient_lower = positive_lower(arb(lower)/(size**3))
    physical_lower = positive_lower(arb(descriptor['physical_family']['UE'])*arb(coefficient_lower))
    forward_modes = [dict(row, origin_stripped_coefficient_phase_interval=(-arb(row['k_absolute_interval'])).str(65),
        global_coordinate_coefficient_rule='parity * weight * exp(-i k_x_abs)') for row in descriptor['incoming']]
    descriptor.update(dict(static_forward_receiver_center=[size//2+1, size//2, size//2],
        static_forward_receiver_modes=forward_modes,
        static_outgoing_photon_phase_intervals=[
            (arb(row['k_absolute_interval'])-duration*arb(row['omega_interval'])).str(65)
            for row in descriptor['outgoing_annihilation']],
        output_source_recoil_phase_intervals=[
            (int(row[0]-row[1])*delta).str(65) for row in displacement],
        static_GLOBAL_coefficient_positive_lower=coefficient_lower,
        static_conditional_physical_amplitude_positive_lower=physical_lower))
    result['static_receiver'] = dict(raw_C_static_volume_rescaled=c_static,
        T0_offGram_volume_rescaled=g_off, bare_forward_m_x0=m_forward,
        calibrated_volume_rescaled_reference=reference,
        kernel_error_components=dict(C_static=components[0][1], offGram=components[1][1], bare_forward=components[2][1]),
        GLOBAL_product_disk_radius=global_radius.str(65),
        every_actual_normalized_nu_GLOBAL_leading_rescaled_lower=lower,
        every_actual_normalized_nu_GLOBAL_leading_coefficient_lower=coefficient_lower,
        physical_conditional_receiver_leading_amplitude_lower=physical_lower,
        square_of_conditional_leading_amplitude_lower=positive_lower(arb(physical_lower)**2),
        formula='integralCstatic−.5*(integralg1_yx)*(integralm_x0), NOTintegral(C−.5gm), notC_T−C_0, noinputwhitening',
        n0=1, n0_scope='bare CT norm2 and normalized distinct photon modes; not actual input norm n=1',
        fullCAR_N5_bridge_owner='SameCurrentScattering/PhotonPacketSupplier; identity receipt required separately',
        actual_ground_flight_probability_certified=False)
    return result


def far_right_case(metric_payload: dict[str, Any], screened: dict[str, Any],
                   size: int, quadrature: int, duration: float) -> dict[str, Any]:
    """Separated RIGHT, exact recoil phase, finite-eta core loss and full R-jets."""
    result = static_case(metric_payload, size, quadrature, duration)
    descriptor, near = result['descriptor'], result['static_receiver']
    distance, longest = size//2, size//2+2
    if int(screened['L']) != longest or screened['eta'] != 18:
        raise ValueError('Same-N far-RIGHT finite eta18/alpha source packet required')
    ctx.prec = 256
    eta = arb(18)
    if not (4*eta).exp() >= 8*longest:
        raise AssertionError('Derived finite centre eta no longer pays far path')
    log_c = (-2/((4*eta).exp()+1)).log1p()
    c_long = (longest*log_c).exp()
    c_distance = (distance*log_c).exp()
    if not c_long > arb('.75'):
        raise AssertionError('Far colour word lost its admitted nonzero core amplitude')
    relative = [[distance+int(row[0]), int(row[1]), int(row[2])]
        for row in descriptor['compact_displacement']]
    lengths = [sum(abs(coordinate) for coordinate in row) for row in relative]
    if max(lengths) != longest:
        raise AssertionError('Exact source/path geometry disagrees with screened source packet')
    weights = [arb(float(np.real(value)))**2+arb(float(np.imag(value)))**2
        for value in descriptor['source_eta']]
    weight_sum = sum(weights, arb(0))
    profile_sum = sum((weight*((2*(length-distance))*log_c).exp()
        for weight, length in zip(weights, lengths)), arb(0))/weight_sum
    s = c_distance**2*profile_sum
    g_b, g_x = s/2, s/4
    n_core = g_x.sqrt()
    # q/2 CT source Gram s/2; declared photon vectors have the extra /sqrt2.
    # Normalization of actual input and output-only G0 cancels the COMMON loss.
    profile_relative_error = (-4*log_c).expm1()
    source_delta = arb(str(screened['core']['mark_operator_perturbation']))*arb('1.000001')
    n_lower = arb(1)/arb(8).sqrt()
    # Four 4x4 marks: HS<=2*operator_norm per column, quartet map<=4*delta.
    # All norms are GLOBAL L2(nu) norms. No fibrewise input normalization.
    raw_source_map_error = 4*source_delta
    raw_photon_map_error = raw_source_map_error/arb(2).sqrt()
    if not c_long/2-raw_photon_map_error > n_lower:
        raise AssertionError('Finite-alpha full-quartet physical Gram lower is not admitted')
    input_matrix_error = (1+1/arb(2).sqrt())*raw_source_map_error/n_lower
    global_gram_error = 2*n_core*raw_photon_map_error+raw_photon_map_error**2
    # Positive-matrix inverse-square-root Lipschitz constant: 1/(2*n_lower^3).
    output_matrix_error = (raw_source_map_error/(arb(2).sqrt()*n_lower)
        +s.sqrt()/arb(2).sqrt()*global_gram_error/(2*n_lower**3))
    normalized_mark_error = max(input_matrix_error, output_matrix_error)
    norms = result['finite_N_momentum_lipschitz']['primitive_norms']
    operator_raw = arb(str(fiber.word_norms(norms)))*arb('1.01')
    operator_cloud = arb(str(2*norms['X']*norms['Y0']))*arb('1.01')
    mark_product_error = 2*arb(2).sqrt()*normalized_mark_error+normalized_mark_error**2
    profile_product_error = profile_relative_error*(2+profile_relative_error)
    eta_l1 = arb(str(float(np.sum(abs(descriptor['source_eta'])))))
    extra_c = operator_raw*mark_product_error+2*eta_l1**2*operator_raw*profile_product_error
    extra_g = operator_cloud*mark_product_error+2*eta_l1**2*operator_cloud*profile_product_error
    extra_m = mark_product_error+2*eta_l1**2*profile_product_error
    # Full quartet g*m product: use the whole g row and bare m vector, not |g00|.
    # Reference source maps have HS operator norm sqrt2; forward CAR overlap<=1.
    added_radius = extra_c+arb('.5')*(2*operator_cloud*extra_m
        +extra_g+extra_g*extra_m)
    radius = arb(near['GLOBAL_product_disk_radius'])+added_radius
    # Reduce exact 2pi*(D/N) BEFORE floating arithmetic or interpolation in delta.
    recoil_turns = Fraction(distance, size) % 1
    if recoil_turns != Fraction(1, 2):
        raise AssertionError('Selected even common-N far recoil is not the exact half turn')
    reference = -near['calibrated_volume_rescaled_reference']
    reference_abs = (arb(float(reference.real))**2+arb(float(reference.imag))**2).sqrt()
    lower = positive_lower(reference_abs-radius)
    coefficient_lower = positive_lower(arb(lower)/(size**3))
    ue0_exact = Fraction.from_float(descriptor['physical_family']['UE'])
    c_exact = Fraction.from_float(descriptor['physical_family']['c'])
    ue_exact = ue0_exact/(2**4096)
    k_exact = c_exact*c_exact/(2*ue_exact)
    ue = arb(ue_exact.numerator)/ue_exact.denominator
    physical_lower = positive_lower(ue*arb(coefficient_lower))
    max_axis = [distance+1, 1, 1]
    # Global smooth raw q/2 HS<=1; n_X>=1/sqrt8 gives normalized HS<=2.
    normalized_jets = [[positive_upper(arb(2)*(axis**order)) for order in range(7)] for axis in max_axis]
    baseline_shift = [distance-2, -2, -2]
    shift_norm = arb(sum(coordinate**2 for coordinate in baseline_shift)).sqrt()
    harmonic_energy_density = arb(str(result['scalar_moment_EH']))
    electric_dirichlet = ue*sum(axis*axis for axis in max_axis)/(size**3)
    harmonic_common_phase_residual = (2*shift_norm*(ue*harmonic_energy_density).sqrt()
        +ue*sum(coordinate**2 for coordinate in baseline_shift)/(size**3))
    descriptor.pop('source_norm_squared')
    sqrt_family_ratio = arb(2)**-2048
    for name in ('beam_strip_UE_coordinate_field_upper', 'onebody_current_zpolar_upper', 'fullCAR_current_zpolar_upper'):
        descriptor[name] = positive_upper(arb(descriptor[name])*sqrt_family_ratio)
    for name in ('onebody_contact_upper', 'fullCAR_contact_upper'):
        descriptor[name] = positive_upper(arb(descriptor[name])*sqrt_family_ratio**2)
    descriptor.update(dict(reference=[0, size//2, size//2],
        right_reference_geometry='far separated axial RIGHT, not the near baseline control',
        relative_X_minus_XR=relative, source_max_relative_coordinate=distance+1,
        source_L1_relative_radius=longest, source_axis_relative_upper=max_axis,
        source_covariant_phase='exp[-i(theta/N+a_h).(d+(N//2,0,0))]',
        reference_distance_from_origin=[distance, 0, 0], no_O_N_reference_oscillation=True,
        no_O_N_reference_oscillation_reason='common harmonic phase cancels pointwise; exact recoil half-turn factored BEFORE moderate quadrature',
        output_source_recoil_phase_intervals=[
            (arb.pi()+2*arb.pi()/size*(int(row[0])-int(row[1]))).str(65)
            for row in descriptor['compact_displacement']],
        source_coordinate_tree_word_lengths=lengths, source_coordinate_tree_word_length_max=longest,
        source_path_axes_order=[0, 1, 2], source_path_direction='from sourceX to RIGHT; all steps inside0..N−1, no central seam',
        source_harmonic_normalized_axis_jets_q0_to_q6=normalized_jets,
        normalized_source_HS_first_axis=[row[1] for row in normalized_jets],
        normalized_source_HS_second_axis=[row[2] for row in normalized_jets],
        mixed_harmonic_jet_rule='normalizedHS<=2*prod((axis_max_j)**q_j), totalq<=6; no small-R replacement',
        screened_eta=18, screened_alpha_expression=screened['core']['alpha_expression'],
        screened_q6=screened['derivative_q6'],
        source_core_norm_squared_interval=g_b.str(65),
        source_norm_scope='screened source not primitive CT2I; rawcoreG_B=s/2, photonG_X=s/4, actualsource normal debt required',
        screened_core_source_Gram_interval=g_b.str(65),
        screened_core_photon_Gram_interval=g_x.str(65),
        screened_core_input_norm_interval=n_core.str(65),
        conditional_photon_input_norm_lower_interval=n_lower.str(65),
        conditional_input_norm_admission='bare core plus finite-alpha proven; actual coupled G_B>=1/4 after normal/source debt is separately required',
        colour_common_core_amplitude_interval=c_distance.str(65), colour_longest_core_amplitude_interval=c_long.str(65),
        screened_source_normalization='physicaln0 and outputG0 give profile eta*c^length/sqrt(s); commonq/2 andc^D cancel only through declared normalizations',
        source_matrix_scope='kernel matrix HS²4; trace(M† R M)/2 uses normalized creation matrix M/sqrt2. Raw q/2 matrix is M/2, not M/sqrt2.',
        source_normalized_creation_matrix=np.asarray(descriptor['source_matrix'])/math.sqrt(2),
        source_raw_q_over_2_matrix=np.asarray(descriptor['source_matrix'])/2,
        source_raw_profile_log_c_interval=log_c.str(65),
        source_core_normalized_log_profile_intervals=[
            ((length-distance)*log_c-profile_sum.log()/2).str(65) for length in lengths],
        fixed_core_chart_rescaling='Btilde=Braw/sqrt(s/2); Xtilde=Xraw/sqrt(s/4), qtilde=qraw/sqrt(s/4). Thus Xtilde=a†Btilde (no residualsqrt2), idealcoreG0=I,n0tilde=1. Physicalq/n andY unchanged; finite-alpha actualG0/n deviations explicitly paid, never reset.',
        bare_core_metric_uniform_in_harmonic_fibre=True,
        bare_core_metric_authority='quaternion M†M lies in I+spin-bivector span; Tr(h0(a_h) M M†)=0, so Tr(Q_h M M†)=Tr(MM†)/2 at every momentum. Scalar source profile/phase preserves Parseval.',
        finite_alpha_global_normalization='global G0=integral X†X and global physical n; G0^-1/2 perturbation bound paid in L2(nu), not resetting each fibre to1',
        finite_alpha_global_Gram_error_interval=global_gram_error.str(65),
        finite_alpha_input_normalized_matrix_error_interval=input_matrix_error.str(65),
        finite_alpha_output_whitened_matrix_error_interval=output_matrix_error.str(65),
        physical_family=dict(c=str(c_exact), UE=str(ue_exact), K=str(k_exact),
            UE0_exact=str(ue0_exact), division_exponent=4096, exact_K_equals_c_squared_over_2UE=True,
            prior_short_R_family_not_transferred=True, actual_family_flight_admitted=False),
        harmonic_energy_density_moment_used_interval=harmonic_energy_density.str(65),
        source_template_authority='native exact G_N ideal orbit; uniform g_min1/40,B²<=2, not a fixed trial-template substitution',
        actual_coupled_harmonic_phase_residual_upper_interval=harmonic_common_phase_residual.str(65),
        harmonic_source_Dirichlet_upper_interval=electric_dirichlet.str(65),
        Coulomb_pair_energy_upper_interval=(arb(13)*ue/4).str(65),
        far_recoil_phase_turns=str(recoil_turns), far_recoil_bra_phase=-1,
        old_short_R_flight_retuning_not_transferred=True,
        static_GLOBAL_coefficient_positive_lower=coefficient_lower,
        static_conditional_physical_amplitude_positive_lower=physical_lower))
    far = dict(near)
    for name in ('raw_C_static_volume_rescaled', 'T0_offGram_volume_rescaled',
                 'calibrated_volume_rescaled_reference'):
        far[name] = -near[name]
    far.update(dict(GLOBAL_product_disk_radius=radius.str(65),
        every_actual_normalized_nu_GLOBAL_leading_rescaled_lower=lower,
        every_actual_normalized_nu_GLOBAL_leading_coefficient_lower=coefficient_lower,
        physical_conditional_receiver_leading_amplitude_lower=physical_lower,
        square_of_conditional_leading_amplitude_lower=positive_lower(arb(physical_lower)**2),
        n0=n_core.str(65), n0_scope='physical q/2 screened rawcore input, not one; conditional leading formula includes actualinput/output normalization',
        finite_eta_alpha_extra_kernel_radii=dict(C_static=extra_c.str(65), offGram=extra_g.str(65), bare_forward=extra_m.str(65)),
        finite_eta_alpha_added_GLOBAL_radius=added_radius.str(65),
        screened_profile_relative_error_interval=profile_relative_error.str(65),
        finite_alpha_normalized_mark_error_interval=normalized_mark_error.str(65),
        actual_ground_flight_probability_certified=False))
    near_bloch_keys = ('computed_representative_rescaled_coefficient',
        'every_actual_normalized_nu_coherent_rescaled_lower', 'every_actual_normalized_nu_coherent_coefficient_lower',
        'physical_unwhitened_candidate_amplitude_lower', 'square_of_conditional_amplitude_lower',
        'amplitude_log10_interval', 'conditional_square_log10_interval')
    near_bloch = {name:result.pop(name) for name in near_bloch_keys}
    result.update(dict(near_geometry_control_bloch=near_bloch, near_geometry_control_static_receiver=near,
        static_receiver=far, far_RIGHT_geometry_admitted_conditionally=True,
        old_short_R_family_flight_admission_withdrawn=True))
    return result


def source_hashes() -> dict[str, str]:
    result = fiber.hashes()
    for name in ('noncompact_legal_volume_bloch.py', 'noncompact_legal_volume_bloch.md'):
        result[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    if result['noncompact_harmonic_scattering_fiber.py'] != FROZEN_FIBER_SHA:
        raise AssertionError('Frozen N2000 supplier code changed')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metric-certificate', type=Path, required=True)
    parser.add_argument('--size', type=int, default=COMMON_N)
    parser.add_argument('--quadrature', type=int, default=80)
    parser.add_argument('--time', type=float, default=6.)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--static-receiver', action='store_true')
    parser.add_argument('--right-reference', choices=('far', 'near'), default='far')
    parser.add_argument('--screened-source-packet', type=Path)
    args = parser.parse_args()
    before, started = source_hashes(), time.perf_counter()
    metric_bytes = args.metric_certificate.read_bytes()
    screened_bytes = args.screened_source_packet.read_bytes() if args.screened_source_packet else None
    if args.right_reference == 'far':
        if screened_bytes is None:
            raise ValueError('Exact same-N finite screened source packet required for far RIGHT')
        result = far_right_case(json.loads(metric_bytes), json.loads(screened_bytes), args.size, args.quadrature, args.time)
    else:
        consumer = static_case if args.static_receiver else legal_case
        result = consumer(json.loads(metric_bytes), args.size, args.quadrature, args.time)
    if (before != source_hashes() or metric_bytes != args.metric_certificate.read_bytes()
            or (screened_bytes is not None and screened_bytes != args.screened_source_packet.read_bytes())):
        raise AssertionError('Exact source/metric/screened input changed during consumer execution')
    receipt = dict(plan=fiber.PLAN, pinned_plan_revision=fiber.PINNED_PLAN_REVISION,
        template_revision=fiber.TEMPLATE_REVISION, source_sha256=before,
        input=dict(N=args.size, quadrature=args.quadrature, duration=args.time,
            static_receiver=args.static_receiver or args.right_reference == 'far',
            right_reference=args.right_reference,
            screened_source_packet=str(args.screened_source_packet),
            screened_source_packet_sha256=hashlib.sha256(screened_bytes).hexdigest() if screened_bytes else None,
            metric_certificate=str(args.metric_certificate),
            metric_certificate_sha256=hashlib.sha256(metric_bytes).hexdigest()),
        runtime=dict(executable=sys.executable, python=sys.version, numpy=np.__version__,
            Arb_precision_bits=ctx.prec, elapsed_seconds=time.perf_counter()-started),
        result=result, checks=True)
    rendered = json.dumps(receipt, indent=2, allow_nan=False, default=fiber.encoded)
    args.output.write_text(rendered+'\n')
    print(rendered)


if __name__ == '__main__':
    main()
