#!/usr/bin/env python3
"""Continuous-time native full-Gram locality, current contacts and vector Ward law.

No finite-dt Dirac inverse or alternate Wilson law is used. Numerical controls
are actual background matrices and an independent whole 256-dimensional CAR
trace, not a periodic quantum-photon ground state or a compact Coulomb phase.
The integration owner runs this CLI after all suppliers have landed.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Sequence, cast

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.linalg import block_diag, expm, expm_frechet

from bulk_quantum_selection import BulkQuantumLaw
from original_charged_principal import original_frame, portable
from original_compact_charge_bound import chiral_basis, metric_constants
from original_uniform_charge_gap import (
    car_action, exterior_hamiltonian, exterior_trace, green_kernel_controls,
    require, residual, source_hashes as uniform_hashes, temporal_green,
    time_evolution, tree_onebody,
)
from rotation_band import RotationBand
from unrestricted_field_minimum import BARE

Matrix = NDArray[np.complex128]
RealArray = NDArray[np.float64]
ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-continuing-backward-20260930'
PLAN_REVISION = 'e92bd5a83b413ce8f0b7d3c450ace4174256cc06404c647352a8f61c27b13926'
TEMPLATE_REVISION = '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'
DEFAULT_CENTRES = ((0, 0, 0), (1, 0, 2), (2, 2, 2), (3, 1, 0))


def source_hashes() -> dict[str, str]:
    result = uniform_hashes()
    for name in ('original_massive_photon_locality.py', 'original_massive_photon_locality.md'):
        result[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return result


def array_hash(value: NDArray[Any]) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def relative(value: Matrix | RealArray, reference: Matrix | RealArray) -> float:
    return float(np.linalg.norm(value-reference)/max(1., np.linalg.norm(reference)))


def rejected(callback: Any, label: str) -> str:
    try:
        callback()
    except AssertionError as error:
        return str(error)
    raise AssertionError(f'{label}: adverse input reached and passed the actual accept assertion')


def locality_constants(band: RotationBand, gram: RealArray, frame: Matrix,
                       *, delta: float, gamma_fraction: float = .25,
                       xi_fraction: float = .25) -> dict[str, float]:
    """Fixed-metric exact formulas; floating evaluation is not an interval proof."""
    mu = metric_constants(band, gram, 1)['mu']
    require(delta > 0 and gamma_fraction > 0 and xi_fraction > 0
            and gamma_fraction+xi_fraction < 1, 'strict weighted coercivity budget')
    hops = [-(band.r*frame[0]+1j*band.v*frame[j])/(2*band.a) for j in (1, 2, 3)]
    ch = 2*sum(float(np.linalg.norm(hop, ord=2)) for hop in hops)
    require(ch > 0, 'nonzero native hopping for the spatial example')
    gamma, xi = gamma_fraction*mu, xi_fraction*mu
    alpha = math.log1p(xi/ch)/band.a
    b = math.sqrt(float(gram[0, 0]))*(band.m+6*band.r/band.a)
    b += abs(band.v)/band.a*sum(math.sqrt(float(gram[j, j])) for j in (1, 2, 3))
    gap = mu-gamma-xi
    log_c = 4*(b+gamma+xi)*delta-math.log(delta*gap)
    return dict(mu=mu, Ch=ch, B=b, gamma=gamma, alpha=alpha, xi=xi,
                delta=delta, weighted_inverse_upper=1/gap, log_point_prefactor=log_c,
                point_prefactor=math.exp(log_c), minimum_time_period=8*delta)


class ContinuousGreen:
    """Exact piecewise-constant evolution; continuous evaluation between knots.

    K(t,s) has an explicit one-sided source prescription. This class has no
    temporal discretization: segment widths describe the input background.
    Unitary time interfaces are checked separately below on that same source.
    """
    def __init__(self, hamiltonians: Sequence[Matrix], temporal: RealArray,
                 durations: RealArray):
        self.hamiltonians = list(hamiltonians)
        self.temporal, self.durations = temporal, durations
        self.u, self.prefixes, self.background_phase = time_evolution(
            hamiltonians, temporal, durations)
        self.identity = np.eye(len(self.u), dtype=np.complex128)
        self.source = temporal_green(self.u, self.prefixes)[0]
        self.knots = np.r_[0., np.cumsum(durations)]
        self.period = float(self.knots[-1])
        self.cache: dict[float, Matrix] = {float(t): p for t, p in zip(self.knots, self.prefixes)}

    def fundamental(self, time: float) -> Matrix:
        require(0 <= time <= self.period, 'evaluation lies on the declared time circle')
        if time not in self.cache:
            segment = min(int(np.searchsorted(self.knots, time, side='right'))-1,
                          len(self.durations)-1)
            generator = -self.hamiltonians[segment]+1j*np.diag(np.repeat(self.temporal[segment], 4))
            self.cache[time] = expm((time-self.knots[segment])*generator)@self.prefixes[segment]
        return self.cache[time]

    def kernel(self, time: float, source: float, *, side: str = 'plus') -> Matrix:
        require(side in ('plus', 'minus'), 'one-sided source prescription')
        pt, ps = self.fundamental(time), self.fundamental(source)
        positive = time > source or (time == source and side == 'plus')
        middle = self.source if positive else self.source-self.identity
        return np.linalg.solve(ps.T, (pt@middle).T).T

    def identity_controls(self) -> dict[str, float]:
        # This identification is deliberately evaluated before any decay bound.
        kernels = temporal_green(self.u, self.prefixes)
        controls = green_kernel_controls(self.u, self.prefixes, kernels)
        s = .37*self.period
        jump = self.kernel(s, s)-self.kernel(s, s, side='minus')
        residual(relative(jump, self.identity), 'interior continuous Green source jump')
        endpoint = self.kernel(self.period, s)+self.kernel(0., s)
        residual(float(np.linalg.norm(endpoint)/max(1., np.linalg.norm(self.kernel(0., s)))),
                 'arbitrary-source antiperiodic endpoint')
        controls['interior_source_jump'] = relative(jump, self.identity)
        controls['arbitrary_source_AP'] = float(np.linalg.norm(endpoint)/max(1., np.linalg.norm(self.kernel(0., s))))
        return controls


def torus_distance(sites: NDArray[Any], origin: NDArray[Any], size: int,
                   spacing: float) -> RealArray:
    difference = np.abs(sites-origin)
    return cast(RealArray, spacing*np.minimum(difference, size-difference).sum(axis=-1))


def weight_acceptance(law: BulkQuantumLaw, weight: RealArray, alpha: float) -> float:
    defect = float(np.max(np.abs(weight[:, None]-weight[law.target])))
    residual(max(0., defect-alpha*law.band.a), 'spatial weight includes every native periodic seam', 1e-12)
    return defect


def native_link_jets(h: Matrix, x: int, y: int) -> tuple[Matrix, Matrix]:
    """Actual endpoint h_xy: first i h_xy / -i h_yx, second -h_xy/-h_yx."""
    first, second = np.zeros_like(h), np.zeros_like(h)
    sx, sy = slice(4*x, 4*x+4), slice(4*y, 4*y+4)
    first[sx, sy], first[sy, sx] = 1j*h[sx, sy], -1j*h[sy, sx]
    second[sx, sy], second[sy, sx] = -h[sx, sy], -h[sy, sx]
    residual(float(np.linalg.norm(first-first.conj().T)), 'native compact current Hermiticity')
    return first, second


def native_gauge_controls(law: BulkQuantumLaw, fields: Matrix, links: Matrix,
                          phases: RealArray, green: ContinuousGreen,
                          rng: np.random.Generator) -> dict[str, Any]:
    count = len(phases)
    chi = rng.uniform(-1.2, 1.2, (count, law.volume))
    gauges = [np.exp(1j*np.repeat(c, 4)) for c in chi]
    transformed = []
    for h, phi, c, g in zip(green.hamiltonians, phases, chi, gauges):
        gauged_phi = phi+c[:, None]-c[law.target]
        native = law.onebody(fields, links, gauged_phi)
        target = g[:, None]*h*g.conj()[None, :]
        residual(relative(native, target), 'actual native compact spatial gauge covariance')
        transformed.append(native)
    # Evaluate exp(conjugated generator) using exact unitary covariance of the
    # original segment exponential; the generator was checked on native h.
    correct, wrong = green.identity.copy(), green.identity.copy()
    prefix_residuals = []
    for i, g in enumerate(gauges):
        segment = np.linalg.solve(green.prefixes[i].T, green.prefixes[i+1].T).T
        gauged_segment = g[:, None]*segment*g.conj()[None, :]
        wrong = gauged_segment@wrong
        next_g = gauges[(i+1) % count]
        kick = next_g*g.conj()
        correct = kick[:, None]*(gauged_segment@correct)
        target = next_g[:, None]*green.prefixes[i+1]*gauges[0].conj()[None, :]
        error = relative(correct, target)
        residual(error, 'piecewise gauge path retains each unitary kick and closing seam')
        prefix_residuals.append(error)
    expected = gauges[0][:, None]*green.u*gauges[0].conj()[None, :]
    residual(relative(correct, expected), 'closed piecewise gauge monodromy conjugacy')
    wrong_failure = rejected(lambda: residual(relative(wrong, expected),
                            'closed piecewise gauge monodromy conjugacy'), 'omitted temporal kicks')
    source_new = np.linalg.solve(green.identity+correct, green.identity)
    expected_source = gauges[0][:, None]*green.source*gauges[0].conj()[None, :]
    residual(relative(source_new, expected_source), 'Green source transforms with the same gauge interface')
    sign, logabs = np.linalg.slogdet(green.identity+green.u)
    sign_new, logabs_new = np.linalg.slogdet(green.identity+correct)
    residual(abs(float(logabs_new-logabs)), 'physical logdet gauge invariance', 2e-8)
    residual(abs(complex(sign_new/sign)-1), 'physical determinant phase gauge invariance', 2e-8)
    # Exact infinitesimal vector Ward law includes all 3V edges, notably seams.
    h, c = green.hamiltonians[0], chi[0]
    ward = np.zeros_like(h)
    wrong_ward = np.zeros_like(h)
    for x in range(law.volume):
        for axis in range(3):
            y = int(law.target[x, axis])
            sx, sy = slice(4*x, 4*x+4), slice(4*y, 4*y+4)
            difference = c[x]-c[y]
            ward[sx, sy] += difference*1j*h[sx, sy]
            ward[sy, sx] -= difference*1j*h[sy, sx]
            wrong_ward[sx, sy] += difference*1j*h[sx, sy]
            wrong_ward[sy, sx] += difference*1j*h[sy, sx]
    repeated = np.repeat(c, 4)
    commutator = 1j*(repeated[:, None]-repeated[None, :])*h
    residual(relative(ward, commutator), 'exact native vector-gauge Ward vertex')
    wrong_vertex = rejected(lambda: residual(relative(wrong_ward, commutator),
                           'exact native vector-gauge Ward vertex'), 'wrong reverse current sign')
    return dict(site_gauge_sha256=array_hash(chi), native_h_covariance=True,
                all_temporal_kick_prefix_residuals=prefix_residuals,
                Green_source_gauge_residual=relative(source_new, expected_source),
                physical_logdet_gauge_residual=abs(float(logabs_new-logabs)),
                wrong_no_kicks_residual=relative(wrong, expected), wrong_no_kicks_failure=wrong_failure,
                exact_all_edge_Ward_residual=relative(ward, commutator),
                wrong_reverse_current_residual=relative(wrong_ward, commutator),
                wrong_reverse_current_failure=wrong_vertex,
                temporal_background='real commuting U1 site connection only; no SU4 temporal extension',
                background2_kick_phase='sum of actual chi jump differences including closing seam is zero')


def current_cycle(green: ContinuousGreen, times: Sequence[float],
                  vertices: Sequence[Matrix]) -> complex:
    """One oriented logdet cycle; the connected derivative sums cyclic orders."""
    product = green.identity.copy()
    for i in range(len(vertices)):
        product = product@green.kernel(times[i], times[(i+1) % len(times)])@vertices[(i+1) % len(times)]
    return complex((-1)**(len(times)-1)*np.trace(product))


def native_controls(band: RotationBand, size: int, centre: Sequence[int],
                    gram: RealArray, frame: Matrix, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    law = BulkQuantumLaw(size, band=band)
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    links = law.toron_links(np.pi/2*np.asarray(centre))
    phases = rng.uniform(-math.pi, math.pi, (3, law.volume, 3))
    durations = np.array([.11, .13, .16])*band.a
    temporal = rng.uniform(-.6, .6, (3, law.volume))/band.a
    temporal[0, 0] += (math.pi/2-2*float(np.sum(durations[:, None]*temporal)))/(2*durations[0])
    hs = [law.onebody(fields, links, phi) for phi in phases]
    green = ContinuousGreen(hs, temporal, durations)
    identities = green.identity_controls()
    wrong_identity = rejected(lambda: green_kernel_controls(green.u, green.prefixes,
                              [green.identity]*len(green.prefixes)), 'identity Green proxy')
    constants = locality_constants(band, gram, frame, delta=green.period/8)
    alpha, gamma, xi, mu = (constants[name] for name in ('alpha', 'gamma', 'xi', 'mu'))
    origin = np.zeros(3, dtype=int)
    distances = torus_distance(law.sites, origin, size, band.a)
    weight = alpha*distances
    neighbour_difference = weight_acceptance(law, weight, alpha)
    wrong_weight = alpha*band.a*np.abs(law.sites-origin).sum(axis=-1)
    wrong_seam = rejected(lambda: weight_acceptance(law, wrong_weight, alpha), 'open-path weight on periodic torus')
    repeat = np.repeat(weight, 4)
    multiplier = np.exp(repeat[:, None]-repeat[None, :])
    beta4 = chiral_basis(frame, gram, 1)[2]
    beta = np.kron(np.eye(law.volume), beta4)
    perturbations, coercive = [], []
    for h, a0 in zip(hs, temporal):
        weighted = multiplier*h
        perturbation = float(np.linalg.norm(weighted-h, ord=2))
        residual(max(0., perturbation-xi), 'actual native spatial conjugation below Ch expm1(alpha a)', 2e-8)
        real_beta = beta@(weighted-gamma*green.identity-1j*np.diag(np.repeat(a0, 4)))
        least = float(np.linalg.eigvalsh((real_beta+real_beta.conj().T)/2)[0])
        residual(max(0., mu-gamma-xi-least), 'actual native weighted full-Gram coercivity', 2e-8)
        perturbations.append(perturbation)
        coercive.append(least)
    point_rows = []
    source_site = law.index[tuple(origin)]
    for fraction in (.0, .19, .53, .91):
        t, s = fraction*green.period, .0
        kernel = green.kernel(t, s)
        dtime = min(abs(t-s), green.period-abs(t-s))
        norms = np.array([np.linalg.norm(kernel[4*x:4*x+4, 4*source_site:4*source_site+4], ord=2)
                          for x in range(law.volume)])
        logarithms = np.log(np.maximum(norms, np.finfo(float).tiny))+gamma*dtime+alpha*distances
        residual(max(0., float(logarithms.max())-constants['log_point_prefactor']),
                 'actual spatial and temporal point Green inequality')
        point_rows.append(dict(time=t, circle_distance=dtime, max_log_weighted_block=float(logarithms.max()),
                               min_nonzero_block=float(norms[norms > 0].min()),
                               farthest_site_distance=float(distances.max()),
                               farthest_site_actual_norm=float(norms[int(np.argmax(distances))])))
    # Native endpoint derivative on an actual seam, not a copied hopping law.
    seam = next(x for x, site in enumerate(law.sites) if site[0] == size-1)
    target = int(law.target[seam, 0])
    first, second = native_link_jets(hs[0], seam, target)
    epsilon = 1e-4
    plus, minus = phases[0].copy(), phases[0].copy()
    plus[seam, 0] += epsilon
    minus[seam, 0] -= epsilon
    hp, hm = law.onebody(fields, links, plus), law.onebody(fields, links, minus)
    first_error = relative((hp-hm)/(2*epsilon), first)
    second_error = relative((hp-2*hs[0]+hm)/epsilon**2, second)
    residual(first_error, 'native compact seam first current jet', 3e-8)
    residual(second_error, 'native compact seam local contact jet', 2e-6)
    sites = [seam, law.index[tuple((np.array([size//2]*3)) % size)]]
    currents = [first, native_link_jets(hs[2], sites[1], int(law.target[sites[1], 1]))[0]]
    times = [.07*green.period, .61*green.period]
    actual_cycle = current_cycle(green, times, currents)
    support1 = law.sites[[seam, target]]
    support2 = law.sites[[sites[1], int(law.target[sites[1], 1])]]
    support_distance = min(float(torus_distance(support2, site, size, band.a).min()) for site in support1)
    time_distance = min(abs(times[1]-times[0]), green.period-abs(times[1]-times[0]))
    log_cycle_bound = math.log(8.)+2*(math.log(2.)+constants['log_point_prefactor'])
    log_cycle_bound += sum(math.log(float(np.linalg.norm(vertex, ord=2))) for vertex in currents)
    log_cycle_bound -= 2*(gamma*time_distance+alpha*support_distance)
    residual(max(0., math.log(max(abs(actual_cycle), np.finfo(float).tiny))-log_cycle_bound),
             'actual separated native local-current connected cycle bound')
    gauge = native_gauge_controls(law, fields, links, phases, green, rng)
    sign, logabs = np.linalg.slogdet(green.identity+green.u)
    physical_phase = complex(sign*np.exp(-1j*green.background_phase))
    residual(abs(physical_phase-1), 'actual background-two positive determinant phase', 2e-8)
    omitted_background = rejected(lambda: residual(abs(complex(sign)-1),
                                  'actual background-two positive determinant phase', 2e-8), 'omitted background two')
    return dict(size=size, volume=law.volume, modes=len(green.u), spacing=band.a, centre=list(centre),
                full_native_Gram=gram, constants=constants, continuous_time_period=green.period,
                spatial_phase_sha256=array_hash(phases), temporal_gauge_sha256=array_hash(temporal),
                actual_h_sha256=[array_hash(h) for h in hs], segment_durations=durations,
                temporal_Green_identity=identities, wrong_identity_Green_failure=wrong_identity,
                actual_periodic_spatial_neighbour_weight_difference=neighbour_difference,
                wrong_open_path_seam_failure=wrong_seam, actual_spatial_conjugation_norms=perturbations,
                actual_weighted_beta_coercive_minima=coercive, actual_point_Green_rows=point_rows,
                native_seam_current=dict(first_jet_relative_error=first_error, contact_jet_relative_error=second_error,
                    current_norm=float(np.linalg.norm(first, ord=2)), local_contact_norm=float(np.linalg.norm(second, ord=2))),
                connected_two_current_cycle=actual_cycle, log_cycle_bound=log_cycle_bound,
                separated_current_support_distance=support_distance,
                exact_vector_gauge_interfaces=gauge, normalized_determinant_phase=physical_phase,
                normalized_logabs_determinant=float(logabs), omitted_background_failure=omitted_background)


def fock_operators(modes: int, basis: Sequence[int]) -> list[Matrix]:
    lookup = {state: i for i, state in enumerate(basis)}
    annihilators = []
    for mode in range(modes):
        operator = np.zeros((len(basis), len(basis)), dtype=complex)
        for column, state in enumerate(basis):
            action = car_action(state, mode, False)
            if action is not None:
                target, sign = action
                operator[lookup[target], column] = sign
        annihilators.append(operator)
    return annihilators


def second_exponential_jet(a: Matrix, first: Matrix, second: Matrix) -> tuple[Matrix, Matrix, Matrix]:
    evolution, derivative = expm_frechet(a, first)
    zero = np.zeros_like(a)
    augmented = np.block([[a, first, zero], [zero, a, first], [zero, zero, a]])
    n = len(a)
    twice_ordered = 2*expm(augmented)[:n, 2*n:3*n]
    contact = expm_frechet(a, second, compute_expm=False)
    return evolution, derivative, twice_ordered+contact


def full_car_benchmark(frame: Matrix, band: RotationBand, seed: int) -> dict[str, Any]:
    """Whole eight-mode exterior algebra: all 256 states, no occupancy cut."""
    rng = np.random.default_rng(seed)
    phases = np.array([.21, -.6, .9])
    hs = [tree_onebody(frame, band, float(phase)) for phase in phases]
    durations = np.array([.11, .13, .16])*band.a
    temporal = rng.uniform(-.5, .5, (3, 2))/band.a
    green = ContinuousGreen(hs, temporal, durations)
    identities = green.identity_controls()
    sectors = [exterior_hamiltonian(hs[0], n) for n in range(9)]
    basis = [state for _, states in sectors for state in states]
    require(len(basis) == 256 and len(set(basis)) == 256, 'entire eight-mode CAR algebra retained')
    operators = fock_operators(8, basis)
    fock_h = [np.asarray(block_diag(*(exterior_hamiltonian(h, n)[0] for n in range(9))),
                         dtype=np.complex128) for h in hs]
    numbers = np.array([[(state & 15).bit_count(), (state >> 4).bit_count()] for state in basis])
    prefixes = [np.eye(256, dtype=complex)]
    for h, a0, duration in zip(fock_h, temporal, durations):
        generator = -h+1j*np.diag(numbers@a0)
        prefixes.append(expm(duration*generator)@prefixes[-1])
    total = prefixes[-1]
    z = complex(np.trace(total))
    determinant = complex(np.linalg.det(green.identity+green.u))
    residual(abs(z-determinant)/max(1., abs(z)), 'independent full 256 CAR trace equals determinant', 2e-8)
    def fock_prefix(time: float) -> Matrix:
        segment = min(int(np.searchsorted(green.knots, time, side='right'))-1, len(durations)-1)
        generator = -fock_h[segment]+1j*np.diag(numbers@temporal[segment])
        return expm((time-green.knots[segment])*generator)@prefixes[segment]
    t, s = .67*green.period, .23*green.period
    pt, ps = fock_prefix(t), fock_prefix(s)
    end = np.linalg.solve(pt.T, total.T).T
    middle = np.linalg.solve(ps.T, pt.T).T
    left = [end@operator@middle for operator in operators]
    right = [operator.conj().T@ps for operator in operators]
    actual = np.array([[np.einsum('ij,ji->', a, b)/z for b in right] for a in left])
    predicted = green.kernel(t, s)
    residual(relative(actual, predicted), 'independent continuous-time full-CAR Green correlator', 2e-8)
    # One-sided equal-time contractions independently identify the delta jump.
    total_annihilators = [total@a for a in operators]
    total_creators = [total@a.conj().T for a in operators]
    plus = np.array([[np.einsum('ij,ji->', a, b.conj().T)/z for b in operators]
                     for a in total_annihilators])
    minus = -np.array([[np.einsum('ij,ji->', b, a)/z for b in total_creators]
                       for a in operators])
    residual(relative(plus-minus, np.eye(8, dtype=complex)), 'full-CAR one-sided source jump')
    residual(relative(plus, green.source), 'full-CAR equal-time plus prescription')
    v_t, contact = native_link_jets(hs[2], 0, 1)
    v_s = native_link_jets(hs[0], 0, 1)[0]
    sector_vt = np.asarray(block_diag(*(exterior_hamiltonian(v_t, n)[0] for n in range(9))),
                           dtype=np.complex128)
    sector_vs = np.asarray(block_diag(*(exterior_hamiltonian(v_s, n)[0] for n in range(9))),
                           dtype=np.complex128)
    mean_t = complex(np.trace(end@sector_vt@pt)/z)
    end_s = np.linalg.solve(ps.T, total.T).T
    mean_s = complex(np.trace(end_s@sector_vs@ps)/z)
    connected = complex(np.trace(end@sector_vt@middle@sector_vs@ps)/z-mean_t*mean_s)
    cycle = current_cycle(green, (t, s), (v_t, v_s))
    residual(abs(connected-cycle)/max(1., abs(connected)), 'full-CAR connected current equals continuous logdet cycle', 2e-8)
    # A static tree-link phase is pure vector gauge. Without the native contact,
    # its second derivative is a strictly positive finite-temperature Duhamel
    # susceptibility of a nonconstant Hermitian current, not a random witness.
    ward_h = tree_onebody(frame, band, .37)
    u, du, ddu, ddu_no_contact = (np.eye(8, dtype=complex), np.zeros((8, 8), dtype=complex),
                                 np.zeros((8, 8), dtype=complex), np.zeros((8, 8), dtype=complex))
    for h, a0, duration in ((ward_h, np.zeros(2), green.period),):
        jet, jet2 = native_link_jets(h, 0, 1)
        a = duration*(-h+1j*np.diag(np.repeat(a0, 4)))
        e, de, dde = second_exponential_jet(a, -duration*jet, -duration*jet2)
        pure = second_exponential_jet(a, -duration*jet, np.zeros_like(jet))[2]
        ddu_no_contact = pure@u+2*de@du+e@ddu_no_contact
        ddu = dde@u+2*de@du+e@ddu
        du, u = de@u+e@du, e@u
    qdu = np.linalg.solve(np.eye(8)+u, du)
    derivative = complex(np.trace(qdu))
    second = complex(np.trace(np.linalg.solve(np.eye(8)+u, ddu)-qdu@qdu))
    missing_contact = complex(np.trace(np.linalg.solve(np.eye(8)+u, ddu_no_contact)-qdu@qdu))
    residual(abs(derivative), 'exact current Ward first logdet derivative', 2e-8)
    residual(abs(second), 'exact current Ward contact plus bubble cancellation', 2e-8)
    wrong_contact = rejected(lambda: residual(abs(missing_contact),
                             'exact current Ward contact plus bubble cancellation', 2e-8), 'omitted current contact')
    # Full number projection: the 17-node Fourier rule is exact for degree 8.
    roots = np.exp(2j*np.pi*np.arange(17)/17)
    normalized = np.array([root**-4*np.linalg.det(np.eye(8)+root*green.u)
                           *np.exp(-1j*green.background_phase) for root in roots],
                          dtype=np.complex128)
    canonical = complex(normalized.mean())
    neutral_rows = np.flatnonzero(numbers.sum(axis=1) == 4)
    neutral_trace = complex(np.trace(total[np.ix_(neutral_rows, neutral_rows)])*np.exp(-1j*green.background_phase))
    exterior = exterior_trace(green.u, 4)*np.exp(-1j*green.background_phase)
    residual(abs(canonical-neutral_trace)/max(1., abs(neutral_trace)), 'background-two Fourier projection equals whole neutral CAR trace', 2e-8)
    residual(abs(exterior-neutral_trace)/max(1., abs(neutral_trace)), 'independent full-sector canonical trace', 2e-8)
    require(abs(complex(normalized[0])-canonical) > 1e-4, 'grandcanonical differs from actual neutral projection')
    residual(float(np.max(abs(normalized.imag))), 'physical determinant real on every global Haar node', 2e-8)
    require(float(normalized.real.min()) > 0, 'physical determinant positive on every global Haar node')
    return dict(modes=8, entire_CAR_dimension=256, exterior_sector_dimensions=[len(states) for _, states in sectors],
                spacing=band.a, continuous_time_period=green.period, temporal_Green_identity=identities,
                phases=phases, temporal_gauge_sha256=array_hash(temporal), full_CAR_trace=z,
                determinant_trace=determinant, independent_Green_correlator_relative_error=relative(actual, predicted),
                full_CAR_source_jump_relative_error=relative(plus-minus, np.eye(8, dtype=complex)),
                connected_full_CAR_current=connected, continuous_logdet_current_cycle=cycle,
                first_vector_Ward_derivative=derivative, second_vector_Ward_derivative_with_contact=second,
                static_Ward_phase=.37, static_Ward_h_sha256=array_hash(ward_h),
                wrong_omitted_contact_second_derivative=missing_contact, wrong_omitted_contact_failure=wrong_contact,
                native_contact_norm=float(np.linalg.norm(contact, ord=2)), canonical_neutral_NL=4,
                exact_Haar_global_nodes=17, canonical_trace=canonical, neutral_full_CAR_trace=neutral_trace,
                neutral_exterior_trace=exterior, grandcanonical_trace=complex(normalized[0]),
                phase_scope='classical prescribed compact backgrounds; not a quantum-photon spectrum')


def run(*, size: int = 5, spacings: Sequence[float] = (1., .001),
        centres: Sequence[Sequence[int]] = DEFAULT_CENTRES, seed: int = 2221020) -> dict[str, Any]:
    require(size >= 5 and bool(spacings) and bool(centres), 'native N>=5 and nonempty original input family')
    require(all(math.isfinite(a) and a > 0 for a in spacings), 'positive finite lattice spacings')
    require(all(len(c) == 3 and all(int(n) == n and 0 <= n < 4 for n in c) for c in centres), 'native retained centre labels')
    bindings = source_hashes()
    rows, benchmarks = [], []
    for ai, spacing in enumerate(spacings):
        band = RotationBand(**{**BARE, 'a': float(spacing)})
        for ci, centre in enumerate(centres):
            selection, frame = original_frame(band, size, np.pi/2*np.asarray(centre), True)
            gram = np.asarray(selection['G'], dtype=float)
            row = native_controls(band, size, centre, gram, frame, seed+100*ai+ci)
            row['full10_Gram_gradient_residual'] = selection['gradient_residual']
            rows.append(row)
            if ci == 1 or len(centres) == 1:
                benchmarks.append(full_car_benchmark(frame, band, seed+1000+ai))
    require(source_hashes() == bindings, 'every source and companion proof remains bound through all consumers')
    return cast(dict[str, Any], portable(dict(
        governing_plan=PLAN, assignment_plan_revision=PLAN_REVISION, template_revision=TEMPLATE_REVISION,
        source_sha256=bindings, source_bindings_unchanged=True,
        runtime=dict(executable=sys.executable, python=sys.version, platform=platform.platform(),
                     numpy=np.__version__, scipy=scipy.__version__,
                     BLAS_threads={n: os.environ.get(n) for n in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS')}),
        native_full500_background_rows=rows, independent_whole256_CAR_benchmarks=benchmarks,
        analytic_scope='Fixed-SPD-fullG continuous-time compact-U1 weighted inverse, point Green, connected local logdet cycles and local contacts; exact vector Ward and canonical Gauss meaning.',
        full_quantum_periodic_photon_ground_computed=False, compact_Coulomb_phase_proved=False,
        photon_masslessness_from_locality=False, physical_localized_spin_proved=False,
        fundamental_original_CAR_and_gauge_quantization_assumed=True,
        all_checks_pass=True, campaign_complete=False,
        limitations=['Finite native background matrices are numerical consumers, not uniform interval certificates.',
                     'Connected fermion-logdet derivatives at a fixed background are not connected currents after photon integration.',
                     'No finite-amplitude gauge-invariant monopole polymer norm or weak-coupling stability is inferred.',
                     'Canonical local Haar projection requires the actual compact heat-kernel operator; the benchmark checks its global number component only.',
                     'Original heavy/normal/full-Gauss BO transfer and all64-block uniform hypotheses remain parent-owned.',
                     'No photon masslessness, isolated charge, deconfinement, quantum periodic ground, physical spin, UV or inclusive scattering conclusion.'])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=5)
    parser.add_argument('--spacings', type=float, nargs='+', default=[1., .001])
    parser.add_argument('--centre', type=int, nargs=3, action='append')
    parser.add_argument('--seed', type=int, default=2221020)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(size=args.size, spacings=args.spacings,
                 centres=DEFAULT_CENTRES if args.centre is None else args.centre, seed=args.seed)
    text = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.output is None:
        print(text, end='')
    else:
        args.output.write_text(text)
        print(json.dumps(dict(output=str(args.output), all_checks_pass=result['all_checks_pass'],
                              native_rows=len(result['native_full500_background_rows']),
                              full_quantum_periodic_photon_ground_computed=False,
                              compact_Coulomb_phase_proved=False)))


if __name__ == '__main__':
    main()
