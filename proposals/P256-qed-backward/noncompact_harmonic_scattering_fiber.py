#!/usr/bin/env python3
"""Full-sea conditional scattering and all-harmonic-fiber enclosure.

No harmonic marginal is constructed. The finite consumer evaluates arbitrary
fibers and the supplied bounds apply to every normalized actual marginal.
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

from noncompact_charged_packet_transport import (
    graph, green, harmonic_jet, source_marks, theorem_bounds,
)
from noncompact_inclusive_scattering_join import (
    annihilation_packet, exterior_ratio, mixed_difference, preparation_tangent,
    scattering_kernel, setup,
)
from noncompact_native_current import accept, adverse
from noncompact_photon_packet_supplier import NativeMomentum
from noncompact_photon_response import encoded

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-synthesis-continuation-20261006'
PINNED_PLAN_REVISION = 'ad6a94bfefcf9a2e3d10d60a51d344c6ac559f67a5dc41d46d7ae9f87be9f9ff'
TEMPLATE_REVISION = '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'
Term = tuple[float, tuple[str, ...]]
Jet = dict[tuple[int, int], list[Term]]


def multiply(left: Jet, right: Jet) -> Jet:
    result: Jet = {}
    for (i, j), first in left.items():
        for (k, l), second in right.items():
            if i+k <= 1 and j+l <= 1:
                result.setdefault((i+k, j+l), []).extend(
                    (a*b, u+v) for a, u in first for b, v in second)
    return result


def schur_words() -> list[Term]:
    """Exact xy jet of A-F(Gdag F)^(-1)Gdag A, including P M."""
    f: Jet = {(0, 0): [(1., ('P',))], (1, 0): [(1., ('a', 'P'))],
        (0, 1): [(1., ('b', 'P')), (1., ('X',))],
        (1, 1): [(1., ('c', 'P')), (1., ('a', 'X'))]}
    inverse: Jet = {(0, 0): [(1., ('P',))],
        (1, 0): [(-1., ('P', 'a', 'P'))],
        (0, 1): [(-1., ('P', 'b', 'P'))],
        (1, 1): [(1., ('P', 'a', 'P', 'b', 'P')),
            (1., ('P', 'b', 'P', 'a', 'P')),
            (-1., ('P', 'c', 'P')), (-1., ('P', 'a', 'X')),
            (-1., ('Y', 'b', 'P')), (-1., ('Y', 'X'))]}
    right: Jet = {(0, 0): [(1., ('P',))],
        (1, 0): [(1., ('P', 'a')), (1., ('Y',))],
        (0, 1): [(1., ('P', 'b'))],
        (1, 1): [(1., ('P', 'c')), (1., ('Y', 'b'))]}
    return [(1., ('c',))]+[(-scale, word) for scale, word in
        multiply(multiply(f, inverse), right)[(1, 1)]]


WORDS = schur_words()


def polynomial_action(operators: dict[str, np.ndarray], source: np.ndarray,
                      words: list[Term] | None = None) -> np.ndarray:
    result = np.zeros_like(source)
    for scale, word in WORDS if words is None else words:
        wave = source
        for name in reversed(word):
            wave = operators[name]@wave
        result += scale*wave
    return result


def scalar_jet_product(left: tuple[float, float, float],
                       right: tuple[float, float, float]) -> tuple[float, float, float]:
    a, da, dda = left
    b, db, ddb = right
    return a*b, da*b+a*db, dda*b+2*da*db+a*ddb


def polynomial_bounds(primitives: dict[str, dict[str, float]]) -> dict[str, float]:
    """Telescoping words: localized seam defects, ordinary derivative jets."""
    norm = seam = first = second = 0.
    for scale, word in WORDS:
        norm += abs(scale)*math.prod(primitives[name]['norm'] for name in word)
        jets = (1., 0., 0.)
        for name in word:
            p = primitives[name]
            jets = scalar_jet_product(jets, (p['norm'], p['first'], p['second']))
        first += abs(scale)*jets[1]
        second += abs(scale)*jets[2]
        for index, name in enumerate(word):
            seam += (abs(scale)*primitives[name]['seam']
                *math.prod(primitives[s]['norm'] for s in word[:index])
                *math.prod(primitives[s]['weighted'] for s in word[index+1:]))
    return dict(norm=norm, seam_prefactor=seam, first_axis=first, second_axis=second)


def beam_upper(beam: dict[str, Any], volume: int) -> float:
    return sum(abs(row['coefficient'])*float(np.max(abs(row['polar'])))
        /math.sqrt(volume*row['omega']) for row in beam['modes'])


def uniform_enclosure(tau: float, mu: float, B: float, duration: float,
                      incoming: dict[str, Any], outgoing: dict[str, Any],
                      volume: int, cut_distance: float,
                      source_first: np.ndarray, source_second: np.ndarray,
                      ue: float, harmonic_energy_density: float) -> dict[str, Any]:
    """Valid for every harmonic phase, not a sampled-fiber certificate.

    source_first/second are unfiltered source HS moment bounds, normalized by
    sqrt2, for each relative coordinate X_j-X_R,j. Bounds cover full R_xy M,
    conservatively also paying optional Q projection derivatives/seams.
    """
    if min(tau, mu, B, volume, ue) <= 0 or duration < 0 or cut_distance < 0:
        raise ValueError('positive same-frame gap/endpoint/family, nonnegative time and paid cut distance')
    kappa = math.log1p(mu/(24*tau))
    cp = 8*(B+mu)/(math.pi*mu)
    delta_h = 12*tau
    h1 = 2*tau
    growth = 6*tau*math.expm1(kappa)
    exponential = math.exp(2*growth*duration)
    fi, fo = beam_upper(incoming, volume), beam_upper(outgoing, volume)
    ji, jo, contact = 6*tau*fi, 6*tau*fo, 6*tau*fi*fo
    d1 = 1+2*duration*h1
    d2 = 1+6*duration*h1+4*duration**2*h1**2
    primitives: dict[str, dict[str, float]] = {}
    for name, field, current in (('a', fi, ji), ('b', fo, jo)):
        weighted_j = math.exp(kappa)*current
        delta_j = delta_h*field
        primitives[name] = dict(norm=duration*current,
            weighted=duration*exponential*weighted_j,
            seam=duration*exponential*(delta_j+2*delta_h*duration*weighted_j),
            first=duration*current*d1, second=duration*current*d2)
    weighted_contact = math.exp(kappa)*contact
    di = exponential*(delta_h*fi+2*delta_h*duration*math.exp(kappa)*ji)
    do = exponential*(delta_h*fo+2*delta_h*duration*math.exp(kappa)*jo)
    primitives['c'] = dict(norm=duration*contact+duration**2*ji*jo,
        weighted=duration*exponential*weighted_contact
            +duration**2*exponential**2*math.exp(2*kappa)*ji*jo,
        seam=duration*exponential*(delta_h*fi*fo+2*delta_h*duration*weighted_contact)
            +duration**2*(do*exponential*math.exp(kappa)*ji+jo*di),
        first=duration*contact*d1+duration**2*ji*jo*2*d1,
        second=duration*contact*d2+duration**2*ji*jo*(2*d2+2*d1*d1))
    primitives['P'] = dict(norm=1., weighted=cp, seam=cp*4*delta_h/mu,
        first=cp*4*h1/mu, second=cp*(4*h1/mu+32*h1*h1/(mu*mu)))
    for name, field, current, beam in (('X', fo, jo, outgoing),
                                      ('Y', fi, ji, incoming)):
        omega = min(row['omega'] for row in beam['modes'])
        prefactor = cp*cp/(mu+omega)
        weighted_j = math.exp(kappa)*current
        primitives[name] = dict(norm=current/(2*mu+omega),
            weighted=prefactor*weighted_j,
            seam=prefactor*(delta_h*field+8*delta_h*weighted_j/mu),
            first=prefactor*current*(1+8*h1/mu),
            second=prefactor*current*(1+24*h1/mu+96*h1*h1/(mu*mu)))
    result = polynomial_bounds(primitives)
    # Q on the measured output has the same weighted/seam/derivative debts as P.
    q = primitives['P']
    result['projected_seam_prefactor'] = (result['seam_prefactor']
        +q['seam']*sum(abs(s)*math.prod(primitives[n]['weighted'] for n in w)
            for s, w in WORDS))
    output_first = np.array([math.sqrt(2)*(result['first_axis']+q['first']*result['norm'])
        +result['norm']*moment for moment in source_first])
    output_second = np.array([math.sqrt(2)*(result['second_axis']
        +2*q['first']*result['first_axis']+q['second']*result['norm'])
        +2*(result['first_axis']+q['first']*result['norm'])*first
        +result['norm']*second for first, second in zip(source_first, source_second)])
    motion = duration*(2*math.sqrt(ue*harmonic_energy_density)*float(np.linalg.norm(output_first))
        +ue/volume*float(np.sum(output_second)))
    # Full M,D each have HS norm sqrt2. Q M is norm1; no source band filter.
    scalar_upper = 2*result['norm']
    raw_seam_upper = 2*result['projected_seam_prefactor']*math.exp(-kappa*cut_distance)
    seam_upper = min(raw_seam_upper, 2*scalar_upper)
    return dict(result, kappa=kappa, projector_weight_upper=cp,
        conditional_amplitude_absolute_upper=scalar_upper,
        conditional_inclusive_coefficient_upper=2*result['norm']**2,
        amplitude_reference_seam_error_upper=seam_upper,
        massive_seam_tail_raw_upper=raw_seam_upper,
        physical_amplitude_absolute_upper=ue*scalar_upper,
        physical_amplitude_seam_error_upper=ue*seam_upper,
        coefficient_output_first_derivative_upper=output_first,
        coefficient_output_second_derivative_upper=output_second,
        coefficient_output_Laplacian_upper=float(np.sum(output_second)),
        harmonic_motion_coefficient_error_upper=motion,
        physical_harmonic_motion_amplitude_error_upper=ue*motion,
        seam_distance_for_amplitude_error_001=max(0., math.log(
            max(1., 2*result['projected_seam_prefactor']/.001))/kappa),
        scalar_paid_harmonic_energy_density_upper=harmonic_energy_density,
        endpoint_norm_upper=tau, whole_spectrum_norm_upper=B, gap_lower=mu,
        current_in_upper=ji, current_out_upper=jo, contact_upper=contact,
        physical_link_phase='A_link dimensionless; A= sqrt(UE/omega)(a+a_dagger)',
        physical_two_photon_factor=ue,
        harmonic_marginal='arbitrary ACTUAL normalized marginal; none manufactured',
        harmonic_scalar_potential='Esea(a) retained in UE/V p_a^2+Esea(a)',
        motion_order='UE*[O(sqrt(UE))+O(UE/V)] at fixed jets,T,N and bounded scalar-paid energy density',
        primitives=primitives)


def fiber_data(base: dict[str, Any], harmonic: np.ndarray) -> dict[str, Any]:
    data = dict(base)
    law, frame, centre = base['law'], base['frame'], base['centre']
    _, _, baseline = graph(law.size, centre, frame, base['band'])
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    links = law.toron_links(np.asarray(centre)*math.pi/2)
    h = np.asarray(law.onebody(fields, links, baseline+harmonic), dtype=complex)
    momentum = NativeMomentum(law, h, frame, centre, harmonic)
    g, _, _ = green(law)
    alpha = np.array([1., .3j, -.2+.4j, .6]); alpha /= np.linalg.norm(alpha)
    marks = source_marks(law, fields, links, base['eta'], base['reference'], baseline, harmonic, g)
    relative = law.sites-law.sites[base['reference']]
    recoil = base['eta']*np.exp(1j*relative@base['transfer'])
    dmarks = source_marks(law, fields, links, recoil, base['reference'], baseline, harmonic, g)
    full = momentum.to_momentum(np.einsum('a,air->ir', alpha, marks))/math.sqrt(2)
    dfull = momentum.to_momentum(np.einsum('a,air->ir', alpha, dmarks))/math.sqrt(2)
    source, detector = momentum.project(full, False), momentum.project(dfull, False)
    accept(abs(float(np.linalg.norm(source))**2-1), 'every-fiber complete bar4 source CAR norm')
    accept(abs(float(np.linalg.norm(detector))**2-1), 'every-fiber complete bar4 detector CAR norm')
    data.update(h=h, momentum=momentum, harmonic=harmonic, full_source=full,
        full_detector=dfull, source=source, detector=detector, baseline=baseline,
        fields=fields, links=links, marks=marks, relative=relative)
    return data


def occupied_columns(momentum: NativeMomentum) -> np.ndarray:
    dimension = 4*momentum.law.volume
    columns = np.zeros((dimension, 2*momentum.law.volume), dtype=complex)
    count = 0
    for site in range(momentum.law.volume):
        for band in range(4):
            if momentum.occupied[site, band]:
                columns[4*site:4*site+4, count] = momentum.vectors[site, :, band]
                count += 1
    if count != columns.shape[1]:
        raise AssertionError('entire 2V occupied sea is mandatory')
    return columns


def conditional_operators(data: dict[str, Any], duration: float, nodes: int) -> dict[str, np.ndarray]:
    momentum, law = data['momentum'], data['law']
    dimension = 4*law.volume
    identity = np.eye(dimension, dtype=complex).reshape(law.volume, 4, dimension)
    a, b, contact, absorb, emit = (np.zeros_like(identity) for _ in range(5))
    x, w = np.polynomial.legendre.leggauss(nodes)
    for t, weight in zip((x+1)*duration/2, w*duration/2):
        time = float(t)
        ja = momentum.interaction_current(data['incoming'], time, identity)
        jb = momentum.interaction_current(data['outgoing'], time, identity)
        a -= 1j*weight*ja
        b -= 1j*weight*jb
        contact -= 1j*weight*momentum.evolve(momentum.contact(data['incoming'],
            data['outgoing'], time, momentum.evolve(identity, time)), -time)
        for s, inner in zip((x+1)*time/2, w*time/2):
            absorb -= weight*inner*momentum.interaction_current(data['outgoing'], time,
                momentum.interaction_current(data['incoming'], float(s), identity))
            emit -= weight*inner*momentum.interaction_current(data['incoming'], time,
                momentum.interaction_current(data['outgoing'], float(s), identity))
    occupied = occupied_columns(momentum)
    pre, _ = preparation_tangent(momentum, data['outgoing'], 0.)
    post, _ = preparation_tangent(momentum, annihilation_packet(data['incoming']), duration)
    return dict(P=occupied@occupied.conj().T,
        a=a.reshape(dimension, dimension), b=b.reshape(dimension, dimension),
        c=(contact+absorb+emit).reshape(dimension, dimension),
        c_contact=contact.reshape(dimension, dimension),
        c_absorb=absorb.reshape(dimension, dimension),
        c_emit=emit.reshape(dimension, dimension),
        X=pre.reshape(dimension, -1)@occupied.conj().T,
        Y=occupied@post.reshape(dimension, -1).conj().T)


def prepared_output(data: dict[str, Any], operators: dict[str, np.ndarray]) -> np.ndarray:
    momentum = data['momentum']
    full = polynomial_action(operators, data['full_source'].reshape(4*momentum.law.volume, 4))
    return full.reshape(momentum.law.volume, 4, 4)


def reference_jets(data: dict[str, Any], time: float) -> dict[str, Any]:
    """Literal canonical sea scalar and relative-reference derivative consumer."""
    law, momentum, h = data['law'], data['momentum'], data['h']
    occupied = occupied_columns(momentum)
    occ_position = momentum.to_position(occupied.reshape(law.volume, 4, -1)).reshape(4*law.volume, -1)
    sea_energy = float(momentum.values[momentum.occupied].sum())
    source = momentum.to_position(data['full_source']).reshape(4*law.volume, 4)
    records = []
    for axis in range(3):
        hj, _ = harmonic_jet(law, h, axis)
        sea_first = float(np.trace(occ_position.conj().T@hj@occ_position).real)
        rel = np.repeat(data['relative'][:, axis], 4)[:, None]
        first = np.exp(-1j*time*sea_energy)*(-1j*rel*source-1j*time*sea_first*source)
        eps = 2e-5
        waves, sea_values = [], []
        for sign in (1., -1.):
            shifted = data['harmonic'].copy(); shifted[axis] += sign*eps
            hh = np.asarray(law.onebody(data['fields'], data['links'], data['baseline']+shifted), complex)
            values = np.linalg.eigvalsh(hh)
            energy = float(values[values < 0].sum())
            sea_values.append(energy)
            waves.append(np.exp(-1j*time*energy)*np.exp(-1j*sign*eps*rel)*source)
        derivative = (waves[0]-waves[1])/(2*eps)
        error = accept(np.linalg.norm(first-derivative), 'full Esea/source/reference scalar jet', 2e-6)
        sea_error = accept(abs((sea_values[0]-sea_values[1])/(2*eps)-sea_first),
            'whole-sea Hellmann scalar derivative', 2e-7)
        omitted = np.exp(-1j*time*sea_energy)*(-1j*rel*source)
        # Omitting XR preserves a scalar coefficient but changes its electric jet.
        absolute = np.repeat(law.sites[:, axis], 4)[:, None]
        missing_reference = np.exp(-1j*time*sea_energy)*(-1j*absolute*source-1j*time*sea_first*source)
        records.append(dict(axis=axis, Esea_first=sea_first, scalar_jet_error=error,
            Hellmann_error=sea_error, dropped_Esea_jet_defect=float(np.linalg.norm(first-omitted)),
            dropped_reference_jet_defect=float(np.linalg.norm(first-missing_reference))))
    best = max(records, key=lambda row: row['dropped_Esea_jet_defect'])
    ref = max(records, key=lambda row: row['dropped_reference_jet_defect'])
    return dict(Esea=sea_energy, axes=records,
        dropped_Esea_rejected=adverse(best['dropped_Esea_jet_defect'], 'full Esea/source/reference scalar jet', 2e-6),
        dropped_reference_rejected=adverse(ref['dropped_reference_jet_defect'], 'full Esea/source/reference scalar jet', 2e-6),
        scalar_cancellation='only in SAME charged/neutral exterior ratio; not removed from harmonic Hamiltonian')


def hashes() -> dict[str, str]:
    paths = {Path(__file__).resolve(), ROOT/'noncompact_harmonic_scattering_fiber.md'}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename and Path(filename).resolve().parent == ROOT:
            paths.add(Path(filename).resolve())
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def marginal_enclosure(reference: complex, coefficient_upper: float,
                       uniform_error: float) -> dict[str, Any]:
    """A certificate for EVERY normalized actual marginal; no measure input."""
    if coefficient_upper < abs(reference) or uniform_error < 0:
        raise ValueError('reference coefficient lies in its certified uniform enclosure')
    radius = min(uniform_error, coefficient_upper+abs(reference))
    lower = max(0., abs(reference)-radius)
    return dict(mean_amplitude_disk_center=reference, mean_amplitude_disk_radius=radius,
        mean_amplitude_absolute_upper=coefficient_upper,
        mean_exclusive_probability_coefficient_lower=lower**2,
        mean_exclusive_probability_coefficient_upper=coefficient_upper**2,
        nonzero_lower_certified=lower > 0,
        measure='all normalized actual marginals; no replacement distribution')


def harmonic_motion_consumer(data: dict[str, Any], duration: float, nodes: int,
                             enclosure: dict[str, Any]) -> dict[str, Any]:
    """Probe the real conditional coefficient's kinetic product-rule cross term."""
    axis, eps = 0, 2e-4
    waves = []
    for shift in (-eps, 0., eps):
        harmonic = data['harmonic'].copy(); harmonic[axis] += shift
        fiber = fiber_data(data, harmonic)
        output = prepared_output(fiber, conditional_operators(fiber, duration, nodes))
        waves.append(fiber['momentum'].to_position(output).reshape(4*fiber['law'].volume, 4))
    minus, central, plus = waves
    first = (plus-minus)/(2*eps)
    second = (plus-2*central+minus)/eps**2
    momentum = data['law'].size  # Periodic harmonic Fourier probe, NOT an actual ground.
    commutator = -(np.exp(1j*momentum*eps)*plus-2*central
        +np.exp(-1j*momentum*eps)*minus)/eps**2-momentum**2*central
    paid = -second-2j*momentum*first
    error = accept(np.linalg.norm(commutator-paid), 'harmonic kinetic coefficient cross derivative', 3e-7)
    missing = adverse(np.linalg.norm(commutator+second),
        'harmonic kinetic coefficient cross derivative', 3e-7)
    accept(max(0., float(np.linalg.norm(first))
        -enclosure['coefficient_output_first_derivative_upper'][axis]), 'uniform coefficient first jet')
    accept(max(0., float(np.linalg.norm(second))
        -enclosure['coefficient_output_second_derivative_upper'][axis]), 'uniform coefficient second jet')
    return dict(axis=axis, harmonic_probe_momentum=momentum, finite_difference_step=eps,
        coefficient_first_derivative_HS=float(np.linalg.norm(first)),
        coefficient_second_derivative_HS=float(np.linalg.norm(second)),
        kinetic_product_rule_error=error, omitted_electric_cross_term_rejected=missing,
        actual_ground_replaced=False, scope='kinematic native coefficient probe, no harmonic marginal supplied')


def case(size: int, duration: float, nodes: int, independent_steps: int) -> dict[str, Any]:
    base = setup(size)
    law = base['law']
    fibers = [np.array([.173, -.291, .407])*math.pi/size,
        np.array([math.pi/size, .217, -.133]),
        np.array([-math.pi/size, .217, -.133]),
        -np.asarray(base['centre'])*math.pi/(2*size)]
    rows, outputs, datas = [], [], []
    for harmonic in fibers:
        data = fiber_data(base, harmonic)
        operators = conditional_operators(data, duration, nodes)
        fine_operators = conditional_operators(data, duration, 2*nodes)
        full = prepared_output(data, fine_operators)
        coarse = prepared_output(data, operators)
        quadrature = accept(np.linalg.norm(full-coarse), 'prepared conditional contour quadrature', 3e-6)
        coefficient = complex(np.vdot(data['full_detector'], full))
        positive_final_probe = complex(np.vdot(data['detector'], full))
        projected = data['momentum'].project(full, False)
        positive_only = polynomial_action(fine_operators, data['source'].reshape(4*law.volume, 4))
        tolerance = max(3e-7, 4*quadrature)
        negative_boundary = adverse(np.linalg.norm(full.reshape(4*law.volume, 4)-positive_only),
            'unfiltered preparation contour source boundary', tolerance)
        real_parts = scattering_kernel(data['momentum'], data['incoming'], data['outgoing'],
            data['source'], duration, 2*nodes)
        real_coefficient = complex(np.vdot(data['detector'], real_parts['full']))
        missing_vertices = {}
        for name in ('contact', 'absorb', 'emit'):
            wrong = dict(fine_operators)
            wrong['c'] = wrong['c']-fine_operators['c_'+name]
            defect = float(np.linalg.norm(full-prepared_output(data, wrong)))
            missing_vertices[name] = adverse(defect, 'same current/contact/both-histories contour', tolerance)
        q = np.asarray(base['centre'])*math.pi/(2*size)+harmonic
        source_position = data['momentum'].to_position(data['full_source']).reshape(4*law.volume, 4)
        stripped = np.exp(1j*np.repeat(data['relative']@q, 4))[:, None]*source_position
        rows.append(dict(harmonic=harmonic, combined_link_phase=q,
            conditional_prepared_amplitude=coefficient,
            positive_final_chart_probe=positive_final_probe,
            final_negative_source_boundary_rejected=adverse(abs(coefficient-positive_final_probe),
                'full final Euclidean detector boundary', tolerance),
            complete_contour_output_HS_squared=float(np.linalg.norm(full)**2),
            positive_output_inclusive_coefficient=float(np.linalg.norm(projected)**2),
            real_time_only_amplitude=real_coefficient, preparation_change=coefficient-real_coefficient,
            quadrature_error=quadrature, Esea=float(data['momentum'].values[data['momentum'].occupied].sum()),
            all_modes=4*law.volume, occupied=2*law.volume, complete_bar4_RIGHT=4,
            negative_source_boundary_rejected=negative_boundary,
            missing_vertices_rejected=missing_vertices,
            omitted_reference_colour_rejected=adverse(abs(coefficient-complex(np.vdot(
                data['full_detector'][:, :, :1], full[:, :, :1]))), 'complete bar4 RIGHT coefficient', tolerance)))
        outputs.append(stripped)
        datas.append(data)
    gauge_source = max(float(np.linalg.norm(wave-outputs[-1])) for wave in outputs)
    accept(gauge_source, 'exact theta/N plus harmonic local source covariance')
    witness = datas[0]
    detector_position = witness['momentum'].to_position(witness['full_detector']).reshape(4*law.volume, 4)
    correct = witness['momentum'].to_position(witness['full_source']).reshape(4*law.volume, 4)
    missing_theta = np.exp(1j*np.repeat(witness['relative']@(np.asarray(base['centre'])*math.pi/(2*size)), 4))[:, None]*correct
    missing_harmonic = np.exp(1j*np.repeat(witness['relative']@witness['harmonic'], 4))[:, None]*correct
    source_adverse = dict(omitted_theta_over_N=adverse(np.linalg.norm(missing_theta-correct),
        'actual covariant source input'), omitted_harmonic=adverse(np.linalg.norm(missing_harmonic-correct),
        'actual covariant source input'))
    phase_operators = conditional_operators(witness, duration, 2*nodes)
    coefficient_phase_adverse = {}
    for name, wrong_position in (('omitted_theta_over_N', missing_theta),
                                  ('omitted_harmonic', missing_harmonic)):
        if name == 'omitted_theta_over_N':
            phase = np.exp(1j*np.repeat(witness['relative']@(np.asarray(base['centre'])*math.pi/(2*size)), 4))
        else:
            phase = np.exp(1j*np.repeat(witness['relative']@witness['harmonic'], 4))
        wrong_source = witness['momentum'].to_momentum(wrong_position)
        wrong_detector = witness['momentum'].to_momentum(phase[:, None]*detector_position)
        wrong_output = polynomial_action(phase_operators, wrong_source.reshape(4*law.volume, 4))
        wrong_coefficient = complex(np.vdot(wrong_detector, wrong_output))
        coefficient_phase_adverse[name] = adverse(abs(wrong_coefficient
            -rows[0]['conditional_prepared_amplitude']), 'same covariant prepared coefficient', 3e-7)
    # Large gauge gluing includes the opposite reference, not only the light field.
    plus, minus = datas[1], datas[2]
    delta = fibers[1]-fibers[2]
    light_gauge = np.exp(-1j*np.repeat(law.sites@delta, 4))
    large_h = accept(np.linalg.norm(plus['h']-light_gauge[:, None]*minus['h']*light_gauge.conj()[None, :]),
        'actual periodic large-gauge H boundary')
    large_coefficient = accept(abs(rows[1]['conditional_prepared_amplitude']-rows[2]['conditional_prepared_amplitude']),
        'large-gauge complete source/ref/current/contact coefficient', 2e-7)
    # Literal independent full exterior determinant reaches the prepared coefficient.
    pre, _ = preparation_tangent(witness['momentum'], witness['outgoing'], 0.)
    post, _ = preparation_tangent(witness['momentum'], annihilation_packet(witness['incoming']), duration)
    def literal(steps: int) -> np.ndarray:
        return mixed_difference(lambda x, y: exterior_ratio(witness['momentum'], witness['h'],
            witness['incoming'], witness['outgoing'], witness['full_source'], witness['full_detector'],
            duration, steps, x, y, pre=pre, post=post, full_output=True), 4e-4)
    literal_coarse, literal_fine = literal(independent_steps), literal(2*independent_steps)
    exact = prepared_output(witness, conditional_operators(witness, duration, 2*nodes))
    literal_error = float(np.linalg.norm(literal_coarse-literal_fine))
    agreement = accept(np.linalg.norm(exact-literal_fine), 'independent literal Peierls/entire-sea exterior contour',
        max(3e-6, 2*literal_error))
    mu = .2*math.sqrt(float(base['gram'][0, 0]-base['gram'][0, 1:]@np.linalg.solve(
        base['gram'][1:, 1:], base['gram'][1:, 0])))
    endpoints = [base['h'][:4, 4*int(law.target[0,j]):4*int(law.target[0,j])+4] for j in range(3)]
    tau = max(float(np.linalg.norm(endpoint, 2)) for endpoint in endpoints)
    onsite = float(np.linalg.norm(base['h'][:4, :4], 2))
    B = onsite+6*tau
    support = abs(base['eta']) > 0
    seam_vertices = np.any((law.sites == 0)|(law.sites == size-1), axis=1)
    distances = np.abs(law.sites[support, None, :]-law.sites[None, seam_vertices, :])
    distances = np.minimum(distances, size-distances)
    distance = float(np.min(distances.sum(axis=2)))
    first = np.sqrt(2*np.sum(abs(base['eta'][:, None])**2*witness['relative']**2, axis=0))
    second = np.sqrt(2*np.sum(abs(base['eta'][:, None])**2*witness['relative']**4, axis=0))
    ue = float(base['constants']['UE'])
    scalar_moment = theorem_bounds(base['constants'], law, base['eta'], base['reference'], duration)
    enclosure = uniform_enclosure(tau, mu, B, duration, base['incoming'], base['outgoing'],
        law.volume, distance, first, second, ue,
        scalar_moment['neutral_harmonic_energy_per_volume_upper'])
    for row in rows:
        accept(max(0., abs(row['conditional_prepared_amplitude'])
            -enclosure['conditional_amplitude_absolute_upper']), 'analytic EVERY-fiber amplitude enclosure')
    seam_actual = max(abs(row['conditional_prepared_amplitude']
        -rows[-1]['conditional_prepared_amplitude']) for row in rows[:-1])
    unpaid = adverse(seam_actual, 'zero-seam phase-only coefficient equality', 2e-7)
    accept(max(0., seam_actual-enclosure['amplitude_reference_seam_error_upper']),
        'paid uniform current/contact/preparation seam enclosure')
    return dict(N=size, a=1., duration=duration, fibers=rows, enclosure=enclosure,
        actual_marginal_enclosure=marginal_enclosure(rows[-1]['conditional_prepared_amplitude'],
            enclosure['conditional_amplitude_absolute_upper'], enclosure['amplitude_reference_seam_error_upper']),
        scalar_moment_supplier=scalar_moment,
        coefficient_harmonic_motion_probe=harmonic_motion_consumer(witness, duration, 2*nodes, enclosure),
        source_gauge_covariance_error=gauge_source, source_adverse=source_adverse,
        coefficient_phase_adverse=coefficient_phase_adverse,
        large_gauge_H_error=large_h, large_gauge_coefficient_error=large_coefficient,
        literal_exterior_difference_error=literal_error, literal_polynomial_agreement=agreement,
        source_reference_scalar_jets=reference_jets(witness, .37),
        observed_seam_coefficient_difference=seam_actual, unpaid_seam_rejected=unpaid,
        actual_marginal_used=False, coupled_ground_simulated=False,
        nonzero_actual_marginal_lower_certified=False,
        scope='exact finite free-sea preparation jets plus uniform all-fiber enclosure; actual neutral replacement debt separate')


def bounded_sinc(value: np.ndarray) -> np.ndarray:
    """Taylor sinc on |x|<=8; no cancellation or vendor transcendental error."""
    if float(np.max(abs(value))) > 8:
        raise ValueError('Certified sinc domain |x|<=8')
    squared = value*value
    result = np.full_like(value, 1/math.factorial(81))
    for k in range(39, -1, -1):
        result = 1/math.factorial(2*k+1)-squared*result
    return result


def bounded_phase(value: np.ndarray) -> np.ndarray:
    """Periodic polynomial exp(ix); binary64 argument reduction explicitly paid."""
    if float(np.max(abs(value))) > 64:
        raise ValueError('Certified phase argument-reduction domain |x|<=64')
    angle = (value+math.pi) % (2*math.pi)-math.pi
    squared = angle*angle
    cosine = np.full_like(angle, 1/math.factorial(36))
    sine = np.full_like(angle, 1/math.factorial(37))
    for k in range(17, -1, -1):
        cosine = 1/math.factorial(2*k)-squared*cosine
        sine = 1/math.factorial(2*k+1)-squared*sine
    return cosine+1j*angle*sine


def entire_time_integral(delta: np.ndarray, duration: float) -> np.ndarray:
    z = delta*duration/2
    return duration*bounded_phase(z)*bounded_sinc(z)


def entire_ordered_integral(later: np.ndarray, earlier: np.ndarray,
                            duration: float) -> np.ndarray:
    """Integral_0^T dt integral_0^t ds exp(i later*t+i earlier*s)."""
    result = np.empty_like(later, dtype=complex)
    small = abs(earlier*duration) <= .25
    if np.any(~small):
        result[~small] = (entire_time_integral(later[~small]+earlier[~small], duration)
            -entire_time_integral(later[~small], duration))/(1j*earlier[~small])
    if np.any(small):
        z, y = later[small]*duration, earlier[small]*duration
        # Dimensionless moments I_m(z)=integral_0^1 t^m exp(izt)dt.
        moments = [np.zeros_like(z, dtype=complex) for _ in range(12)]
        close = abs(z) <= .5
        if np.any(close):
            term = np.ones(int(np.sum(close)), dtype=complex)
            for order in range(40):
                for degree in range(12):
                    moments[degree][close] += term/(degree+order+1)
                term *= 1j*z[close]/(order+1)
        if np.any(~close):
            phase = bounded_phase(z[~close])
            moments[0][~close] = bounded_phase(z[~close]/2)*bounded_sinc(z[~close]/2)
            for degree in range(1, 12):
                moments[degree][~close] = (phase-degree*moments[degree-1][~close])/(1j*z[~close])
        partial = np.zeros_like(z, dtype=complex)
        power = np.ones_like(z, dtype=complex)
        for degree in range(11):
            partial += power*moments[degree+1]/math.factorial(degree+1)
            power *= 1j*y
        result[small] = duration**2*partial
    return result


class SparseBlochFiber:
    """Continuous full4x4 momentum-shift kernel, with each photon sqrtV stripped."""
    def __init__(self, points: np.ndarray, frame: np.ndarray, duration: float):
        self.points, self.frame, self.duration = points, frame, duration
        self.identity = np.eye(4, dtype=complex)
        self.endpoint = -(.12*frame[0]+1j*.3*frame[3])/2
        self.cache: dict[tuple[float, ...], dict[str, Any]] = {}

    def info(self, shift: np.ndarray) -> dict[str, Any]:
        key = tuple(float(q) for q in shift)
        if key not in self.cache:
            p = self.points+shift
            phase = bounded_phase(p)
            coefficient = np.column_stack((.2+.12*np.sum(1-np.real(phase), axis=1), .3*np.imag(phase)))
            h = np.einsum('pm,mab->pab', coefficient, self.frame)
            energy_squared = np.trace(h@h, axis1=1, axis2=2).real/4
            energy = np.sqrt(energy_squared)
            if float(np.min(energy)) <= 0:
                raise AssertionError('Full occupied sea gap required')
            positive = (self.identity+h/energy[:, None, None])/2
            self.cache[key] = dict(h=h, E=energy, Q=positive, P=self.identity-positive)
        return self.cache[key]

    def vertex(self, row: dict[str, Any], shift: np.ndarray) -> np.ndarray:
        pz = self.points[:, 2]+shift[2]
        phase = bounded_phase(pz)
        # SAME native x/y-directed photons have literal z polarization.
        if np.linalg.norm(row['polar']-np.array([0., 0., 1.])) > 2e-12 or abs(row['k'][2]) > 1e-12:
            raise ValueError('Declared native x/y photon z-polarization surface only')
        return (1j*row['coefficient']/math.sqrt(row['omega'])
            *(phase[:, None, None]*self.endpoint-phase.conj()[:, None, None]*self.endpoint.conj().T))

    def integral(self, row: dict[str, Any], shift: np.ndarray) -> np.ndarray:
        inside, outside = self.info(shift), self.info(shift+row['k'])
        j = self.vertex(row, shift)
        result = np.zeros_like(j)
        for sign_out, po in ((1., outside['Q']), (-1., outside['P'])):
            for sign_in, pi in ((1., inside['Q']), (-1., inside['P'])):
                delta = sign_out*outside['E']-sign_in*inside['E']-row['frequency_sign']*row['omega']
                result += (-1j*entire_time_integral(delta, self.duration))[:, None, None]*(po@j@pi)
        return result

    def preparation(self, row: dict[str, Any], shift: np.ndarray, initial: bool,
                    time: float | None = None) -> np.ndarray:
        inside, outside = self.info(shift), self.info(shift+row['k'])
        j = self.vertex(row, shift)
        denominator = outside['E']+inside['E']+row['omega']
        if initial:
            return -(outside['Q']@j@inside['P'])/denominator[:, None, None]
        t = self.duration if time is None else time
        phase = bounded_phase(-t*denominator)
        return -(phase/denominator)[:, None, None]*(outside['P']@j@inside['Q'])

    def mixed(self, incoming: dict[str, Any], outgoing: dict[str, Any]) -> np.ndarray:
        zero = np.zeros(3)
        shift = incoming['k']+outgoing['k']
        first, final = self.info(zero), self.info(shift)
        phase = bounded_phase(self.points[:, 2])
        contact = (-incoming['coefficient']*outgoing['coefficient']
            /math.sqrt(incoming['omega']*outgoing['omega'])
            *(phase[:, None, None]*self.endpoint+phase.conj()[:, None, None]*self.endpoint.conj().T))
        result = np.zeros_like(contact)
        for so, po in ((1., final['Q']), (-1., final['P'])):
            for si, pi in ((1., first['Q']), (-1., first['P'])):
                delta = (so*final['E']-si*first['E']
                    -incoming['frequency_sign']*incoming['omega']-outgoing['frequency_sign']*outgoing['omega'])
                result += (-1j*entire_time_integral(delta, self.duration))[:, None, None]*(po@contact@pi)
        for early, late in ((incoming, outgoing), (outgoing, incoming)):
            middle = self.info(early['k'])
            j_early, j_late = self.vertex(early, zero), self.vertex(late, early['k'])
            for so, po in ((1., final['Q']), (-1., final['P'])):
                for sm, pm in ((1., middle['Q']), (-1., middle['P'])):
                    for si, pi in ((1., first['Q']), (-1., first['P'])):
                        later = so*final['E']-sm*middle['E']-late['frequency_sign']*late['omega']
                        earlier = sm*middle['E']-si*first['E']-early['frequency_sign']*early['omega']
                        result -= entire_ordered_integral(later, earlier, self.duration)[:, None, None]*(po@j_late@pm@j_early@pi)
        return result

    def connected_pair(self, incoming: dict[str, Any], outgoing: dict[str, Any]) -> np.ndarray:
        zero = np.zeros(3)
        ki, ko = incoming['k'], outgoing['k']
        first, final = self.info(zero), self.info(ki+ko)
        a0, ao = self.integral(incoming, zero), self.integral(incoming, ko)
        b0, bi = self.integral(outgoing, zero), self.integral(outgoing, ki)
        x0, xi = self.preparation(outgoing, zero, True), self.preparation(outgoing, ki, True)
        y0, yo = self.preparation(incoming, zero, False), self.preparation(incoming, ko, False)
        cloud0 = (-xi@self.preparation(incoming, zero, False, 0.)
            +self.preparation(incoming, ko, False, 0.)@x0)
        return (final['Q']@(self.mixed(incoming, outgoing)
            -ao@self.info(ko)['P']@b0-bi@self.info(ki)['P']@a0)@first['Q']
            -final['Q']@ao@x0-xi@a0@first['Q']
            -final['Q']@bi@y0-yo@b0@first['Q']-xi@y0+yo@x0-cloud0)


def sparse_plane_rows(size: int, speed: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """Preserve SAME normalized two-mode packets and their finite-N frequencies."""
    if size < 5 or size % 5:
        raise ValueError('Declared selected lower construction uses finite N multiples of5')
    packets = []
    for axis, conjugate in ((0, False), (1, True)):
        rows = []
        for number, coefficient in ((size//5, 1.), (size//5+1, .25)):
            k = np.eye(3)[axis]*2*math.pi*number/size
            omega = speed*abs(1-bounded_phase(np.asarray(k[axis])))
            if conjugate:
                k = -k
            weight = coefficient/math.sqrt(1+.25**2)
            # Remove the common packet/source origin phases, but keep outgoing offset.
            coefficient = weight*bounded_phase(np.asarray(-k[1])) if conjugate else weight
            rows.append(dict(k=k, omega=float(omega), coefficient=coefficient,
                polar=np.array([0., 0., 1.]), frequency_sign=-1 if conjugate else 1))
        packets.append(dict(modes=rows))
    return packets[0], packets[1]


def compact_fourier(points: np.ndarray, displacement: np.ndarray,
                    coefficients: np.ndarray) -> np.ndarray:
    return bounded_phase(-points@displacement.T)@coefficients


def sparse_bloch_mean(size: int, quadrature: int, duration: float, frame: np.ndarray,
                      source_matrix: np.ndarray, displacement: np.ndarray,
                      coefficients: np.ndarray, speed: float, chunk: int = 2048) -> complex:
    """Compute V*C_N by a moderate periodic Bloch quadrature, never native N^3 H."""
    incoming, outgoing = sparse_plane_rows(size, speed)
    transfer = 2*math.pi/size*np.array([1., -1., 0.])
    count = quadrature**3
    total = 0j
    for start in range(0, count, chunk):
        indices = np.arange(start, min(start+chunk, count))
        points = 2*math.pi*np.column_stack(np.unravel_index(indices, (quadrature,)*3))/quadrature
        kernel = SparseBlochFiber(points, frame, duration)
        source = compact_fourier(points, displacement, coefficients)
        for first in incoming['modes']:
            for second in outgoing['modes']:
                shift = first['k']+second['k']
                detector = compact_fourier(points+shift-transfer, displacement, coefficients)
                block = kernel.connected_pair(first, second)
                trace = np.trace(source_matrix.conj().T@block@source_matrix, axis1=1, axis2=2)/2
                total += complex(np.sum(detector.conj()*source*trace))
    return total/count


CONNECTED_WORDS: list[Term] = [(1., ('Q', 'c', 'Q')),
    (-1., ('Q', 'a', 'P', 'b', 'Q')), (-1., ('Q', 'b', 'P', 'a', 'Q')),
    (-1., ('Q', 'a', 'X')), (-1., ('X', 'a', 'Q')),
    (-1., ('Q', 'b', 'Y')), (-1., ('Y', 'b', 'Q')),
    (-1., ('X', 'Y')), (1., ('Y', 'X')),
    (1., ('X', 'Y0')), (-1., ('Y0', 'X'))]


def word_norms(limits: dict[str, float]) -> float:
    return sum(abs(scale)*math.prod(limits[name] for name in word)
        for scale, word in CONNECTED_WORDS)


def word_error(limits: dict[str, float], errors: dict[str, float]) -> float:
    total = 0.
    for scale, word in CONNECTED_WORDS:
        for index, name in enumerate(word):
            total += (abs(scale)*errors[name]
                *math.prod(limits[n]+errors[n] for n in word[:index])
                *math.prod(limits[n] for n in word[index+1:]))
    return total


def bloch_certificate_budgets(metric: dict[str, Any], size: int, quadrature: int,
                              duration: float, incoming: dict[str, Any],
                              outgoing: dict[str, Any], displacement: np.ndarray,
                              coefficients: np.ndarray) -> dict[str, Any]:
    """Analytic trapezoid + native full-ten frame + IEEE polynomial error proof."""
    from flint import arb, ctx
    ctx.prec = 128
    if size < 25 or size % 5 or not 0 < duration <= 6 or quadrature < 32:
        raise ValueError('Certified selected finite-N surface: N>=25 multiple5,0<T<=6,q>=32')
    g0, gs = map(float, metric['trial_decimal'])
    sigma = float(metric['analytic_strip']['sigma'])
    if sigma != .5 or not metric['parameters']['a'] == 1:
        raise ValueError('Exact original a1 native-metric certificate required')
    majorant = float(metric['analytic_strip']['matrix_gradient_integrand_HS_majorant'])
    decay = (-arb(str(sigma))*size).exp()
    native_gradient = arb(str(majorant))*(((1+decay)/(1-decay))**3-1)
    distance_ball = (arb(str(metric['full_ten_gradient']['Frobenius_residual_upper']))
        +arb(str(metric['continuum_quadrature_error_upper']))+native_gradient)/2
    distance = math.nextafter(float(distance_ball.upper()), math.inf)
    floor = gs-distance
    if floor <= 0:
        raise AssertionError('Actual full-ten native frame distance reached nonpositive metric')
    mu = .2*math.sqrt(g0-distance-distance**2/(2*floor))
    tau = (.12*math.sqrt(g0+distance)+.3*math.sqrt(gs+distance))/2
    B = .56*math.sqrt(g0+distance)+6*tau
    fi, fo = beam_upper(incoming, 1), beam_upper(outgoing, 1)
    ji, jo, contact = 6*tau*fi, 6*tau*fo, 6*tau*fi*fo
    oi = min(row['omega'] for row in incoming['modes'])
    oo = min(row['omega'] for row in outgoing['modes'])
    limits = dict(P=1., Q=1., a=duration*ji, b=duration*jo,
        c=duration*contact+duration**2*ji*jo, X=jo/(2*mu+oo),
        Y=ji/(2*mu+oi), Y0=ji/(2*mu+oi))
    root_lipschitz = 1/(math.sqrt(gs)+math.sqrt(floor))
    h_change = math.sqrt(.92**2+3*.3**2)*root_lipschitz
    endpoint_change = (.12+.3)*root_lipschitz/2
    di, do, dk = 6*endpoint_change*fi, 6*endpoint_change*fo, 6*endpoint_change*fi*fo
    dp = h_change/mu
    dx = ((2*dp*jo+do)/(2*mu+oo)+2*h_change*jo/(2*mu+oo)**2)
    dy0 = ((2*dp*ji+di)/(2*mu+oi)+2*h_change*ji/(2*mu+oi)**2)
    changes = dict(P=dp, Q=dp,
        a=duration*(di+2*duration*h_change*ji),
        b=duration*(do+2*duration*h_change*jo),
        c=duration*(dk+2*duration*h_change*contact)
            +duration**2*(do*ji+jo*di+4*duration*h_change*ji*jo),
        X=dx, Y=dy0+2*duration*h_change*ji/(2*mu+oi), Y0=dy0)
    eta_l1 = float(np.sum(abs(coefficients)))
    real_majorant = 2*eta_l1**2*word_norms(limits)
    frame_error = 2*eta_l1**2*word_error(limits,
        {name: distance*value for name, value in changes.items()})
    # Complex strip: beta-accretivity encloses both full Bloch clusters.
    cbar = .999*float(metric['analytic_strip']['coercivity_lower'])
    sigma_ball = arb(str(sigma))
    cosh = math.nextafter(float(sigma_ball.cosh().upper()), math.inf)
    sinh = math.nextafter(float(sigma_ball.sinh().upper()), math.inf)
    complex_B = 1.001*(math.sqrt(g0)*(.2+3*.12*(1+cosh))+.3*math.sqrt(3*gs)*cosh)
    imaginary_h = 1.001*(3*.12*math.sqrt(g0)+.3*math.sqrt(3*gs))*sinh
    projector = (1+complex_B/cbar)/2
    flight = math.nextafter(float((arb(str(2*imaginary_h*duration))).exp().upper()), math.inf)
    strip_phase = math.nextafter(float(sigma_ball.exp().upper()), math.inf)
    jci, jco, kc = strip_phase*ji, strip_phase*jo, strip_phase*contact
    strip_limits = dict(P=projector, Q=projector,
        a=duration*flight*jci, b=duration*flight*jco,
        c=duration*flight*kc+duration**2*flight**2*jci*jco,
        X=projector**2*jco/(2*cbar+oo),
        Y=projector**2*jci*flight/(2*cbar+oi),
        Y0=projector**2*jci/(2*cbar+oi))
    strip_l1 = float(np.sum(abs(coefficients)*np.exp(sigma*np.sum(abs(displacement), axis=1))))
    strip_majorant = 1.01*2*strip_l1**2*word_norms(strip_limits)
    def periodic_error(order: int) -> float:
        t = (-arb(str(sigma))*order).exp()
        value = arb(str(strip_majorant))*(((1+t)/(1-t))**3-1)
        return max(1e-300, float(value.upper()))
    # A priori binary64 Horner budgets, rather than vendor sin/cos comparison.
    u = arb(2)**-52
    gamma = 240*u/(1-240*u)
    elementary = (gamma*arb(8).sinh()/8
        +arb(8)**82/arb(math.factorial(83))
        +gamma*arb.pi().exp()+arb.pi()**38/arb(math.factorial(38))
        +arb(64)*32*u)
    if not elementary < arb('1e-10'):
        raise AssertionError('Polynomial/argument reduction elementary budget not certified')
    e, epsilon = 1e-10, float(u)
    eh = 8*B*e
    ee = B*eh/mu+100*epsilon*B*B/mu
    ee += e  # Also encloses the finite-N physical photon-frequency/phase construction.
    ep = (eh+B*ee/mu)/(2*mu)+100*epsilon*(1+B/mu)
    ej_i, ej_o, ek = (j*(2*e+100*epsilon) for j in (ji, jo, contact))
    ef = 4*duration*e+duration**2*ee
    eordered = 300*duration**2*e+4*duration**3*ee/3
    ea = 4*(duration*(2*ep*ji+ej_i)+ji*ef+100*epsilon*duration*ji)
    eb = 4*(duration*(2*ep*jo+ej_o)+jo*ef+100*epsilon*duration*jo)
    ec = (4*(duration*(2*ep*contact+ek)+contact*ef)
        +16*(duration**2*(3*ep*ji*jo+ej_i*jo+ji*ej_o)/2
            +ji*jo*eordered+400*epsilon*duration**2*ji*jo))
    ex = ((2*ep*jo+ej_o)/(2*mu+oo)+2*ee*jo/(2*mu+oo)**2
        +100*epsilon*jo/(2*mu+oo))
    ey0 = ((2*ep*ji+ej_i)/(2*mu+oi)+2*ee*ji/(2*mu+oi)**2
        +100*epsilon*ji/(2*mu+oi))
    errors = dict(P=ep, Q=ep, a=ea, b=eb, c=ec, X=ex,
        Y=ey0+(2*duration*ee+e)*ji/(2*mu+oi), Y0=ey0)
    forward = (2*eta_l1**2*word_error(limits, errors)
        +real_majorant*(1000*epsilon+4*e+400*epsilon*quadrature**3))
    return dict(frame_distance_upper=distance, native_frame_coefficient_error_upper=1.01*frame_error,
        continuum_quadrature_error_upper=periodic_error(quadrature),
        native_N_periodization_error_upper=periodic_error(size),
        arithmetic_coefficient_error_upper=1.01*forward,
        complex_strip=sigma, complex_strip_coercivity_lower=cbar,
        integrand_strip_majorant=strip_majorant, real_integrand_majorant=real_majorant,
        polynomial_elementary_error_upper=float(elementary.upper()), declared_elementary_error=e,
        endpoint_norm_upper=tau, spectrum_norm_upper=B, gap_lower=mu,
        arithmetic_domain='IEEEbinary64 basic operations/sqrt; explicit Horner trig/sinc,|phase|<=64,|sincarg|<=8',
        source_Gram_variation='G^-1/2 sqrt(G) SELECTED=SELECTED: quaternion labels unchanged exactly',
        primitive_arithmetic_errors=errors)


def sparse_lower_case(metric_payload: dict[str, Any], size: int, quadrature: int,
                      duration: float) -> dict[str, Any]:
    from sea_selection import SELECTED
    metric = metric_payload['certificate']
    base = setup(5)
    data = fiber_data(base, -np.asarray(base['centre'])*math.pi/10)
    law, origin = base['law'], np.asarray(base['origin'])
    site = int(law.index[tuple(origin)])
    source_position = data['momentum'].to_position(data['full_source']).reshape(law.volume, 4, 4)
    matrix = source_position[site]*math.sqrt(2)/base['eta'][site]
    support = abs(base['eta']) > 0
    displacement, coefficients = law.sites[support]-origin, base['eta'][support]
    accept(np.linalg.norm(source_position[support]
        -coefficients[:, None, None]*matrix/math.sqrt(2)), 'actual compensated compact source/tree/quaternion')
    g0, gs = map(float, metric['trial_decimal'])
    frame = np.array([math.sqrt(g0), *([math.sqrt(gs)]*3)])[:, None, None]*SELECTED
    speed = float(base['constants']['bare_speed'])
    incoming, outgoing = sparse_plane_rows(size, speed)
    value = sparse_bloch_mean(size, quadrature, duration, frame, matrix,
        displacement, coefficients, speed)
    budgets = bloch_certificate_budgets(metric, size, quadrature, duration,
        incoming, outgoing, displacement, coefficients)
    error = sum(budgets[key] for key in ('native_frame_coefficient_error_upper',
        'continuum_quadrature_error_upper', 'native_N_periodization_error_upper',
        'arithmetic_coefficient_error_upper'))
    volume = size**3
    relative = displacement+2
    first = np.sqrt(2*np.sum(abs(coefficients[:, None])**2*relative**2, axis=0))
    second = np.sqrt(2*np.sum(abs(coefficients[:, None])**2*relative**4, axis=0))
    scalar_energy = theorem_bounds(base['constants'], law, base['eta'], base['reference'], duration)
    scalar_energy_density = scalar_energy['neutral_harmonic_energy_per_volume_upper']
    # Both T and T0 clouds have their own full Euclidean seam debts.
    seam_t = uniform_enclosure(budgets['endpoint_norm_upper'], budgets['gap_lower'],
        budgets['spectrum_norm_upper'], duration, incoming, outgoing, volume,
        size//2-2, first, second, base['constants']['UE'], scalar_energy_density)
    # Exact zero-time cloud is separately enclosed, never a tiny-time surrogate.
    seam_zero = uniform_enclosure(budgets['endpoint_norm_upper'], budgets['gap_lower'],
        budgets['spectrum_norm_upper'], 0., incoming, outgoing, volume,
        size//2-2, first, second, base['constants']['UE'], scalar_energy_density)
    scaled_seam = volume*(seam_t['amplitude_reference_seam_error_upper']
        +seam_zero['amplitude_reference_seam_error_upper'])
    lower = abs(value)-error-scaled_seam
    if lower <= 0:
        raise AssertionError(f'Positive reference not admitted: |VC|={abs(value)}, errors={error}, seam={scaled_seam}')
    ue = float(base['constants']['UE'])
    return dict(N=size, native_grid_simulated=False, moderate_Bloch_quadrature=quadrature,
        quadrature_points=quadrature**3, fixed_body_trial_Gram=np.diag([g0, gs, gs, gs]),
        actual_native_full10_metric_enclosed=True, source_bar4_unfiltered=True,
        finite_N_rescaled_connected_amplitude=value,
        finite_N_compensated_coefficient=value/volume,
        rescaled_reference_error_upper=error, rescaled_connected_seam_error_upper=scaled_seam,
        rescaled_ALL_actual_marginal_amplitude_absolute_lower=lower,
        ALL_actual_marginal_coefficient_absolute_lower=lower/volume,
        physical_unwhitened_amplitude_absolute_lower=ue*lower/volume,
        physical_unwhitened_probability_lower=(ue*lower/volume)**2,
        budgets=budgets, physical_family=dict(c=speed, UE=ue, K=base['constants']['K']),
        scalar_moment_supplier=scalar_energy,
        normalization='one normalized two-mode incoming/outgoing photon: C_N=BlochMean_N/V, amplitudeUE*C_N; no fixed C_infinity',
        N_dependent_observable=dict(mode_numbers=[size//5, size//5+1], weights=[1/math.sqrt(1+.25**2), .25/math.sqrt(1+.25**2)],
            charge_recoil=2*math.pi/size*np.array([1., -1., 0.]), frequencies=[r['omega'] for r in incoming['modes']],
            source_width=.8, charge_carrier=[1.2, 0., 0.], observation_time=duration),
        observable='existing coherent GLOBAL fullbar4 receiver candidate; no I_harmonic/ref trace or channel change',
        actual_global_Gram_whitened_probability_certified=False,
        remaining_transfer='actual neutral prepared comparison, leading actual globalGram/offGram control, fullflight/legalLambda; NOT erased by conditional positivity')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=3)
    parser.add_argument('--time', type=float, default=2.)
    parser.add_argument('--nodes', type=int, default=8)
    parser.add_argument('--exterior-steps', type=int, default=32)
    parser.add_argument('--bloch-lower', action='store_true')
    parser.add_argument('--target-size', type=int, default=2000)
    parser.add_argument('--quadrature', type=int, default=80)
    parser.add_argument('--metric-certificate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.size < 3 or args.time <= 0 or args.nodes < 4 or args.exterior_steps < 4:
        raise ValueError('native finite resolution and positive flight time required')
    before = hashes()
    metric_bytes = None
    if args.bloch_lower:
        if args.metric_certificate is None:
            raise ValueError('Existing exact native-metric certificate required for Bloch lower')
        metric_bytes = args.metric_certificate.read_bytes()
        result_data = sparse_lower_case(json.loads(metric_bytes), args.target_size, args.quadrature, args.time)
    else:
        result_data = case(args.size, args.time, args.nodes, args.exterior_steps)
    result = dict(plan=PLAN, pinned_plan_revision=PINNED_PLAN_REVISION,
        template_revision=TEMPLATE_REVISION, source_sha256=before,
        input=dict(N=args.size, duration=args.time, nodes=args.nodes, exterior_steps=args.exterior_steps,
            bloch_lower=args.bloch_lower, target_size=args.target_size, quadrature=args.quadrature,
            metric_certificate=str(args.metric_certificate),
            metric_certificate_sha256=hashlib.sha256(metric_bytes).hexdigest() if metric_bytes else None),
        runtime=dict(executable=sys.executable, python=sys.version, numpy=np.__version__, scipy=scipy.__version__),
        result=result_data, checks=True)
    if hashes() != before:
        raise AssertionError('source changed during consumer execution; rerun stable scoped proof')
    if metric_bytes is not None and args.metric_certificate.read_bytes() != metric_bytes:
        raise AssertionError('Exact metric certificate changed during Bloch proof')
    rendered = json.dumps(result, indent=2, allow_nan=False, default=encoded)
    args.output.write_text(rendered+'\n')
    print(rendered)


if __name__ == '__main__':
    main()
