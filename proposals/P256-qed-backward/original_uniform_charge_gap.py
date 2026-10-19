#!/usr/bin/env python3
"""Native continuous-time compact-U1 uniform-gap theorem consumer.

The companion proof is a full effective-operator theorem. The quantum consumer
is an exact Gauss-eliminated tree, NOT a periodic quantum-photon calculation.
Time-background products are exact for declared piecewise-constant Hamiltonians.
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
from scipy.linalg import expm

from bulk_quantum_selection import BulkQuantumLaw
from original_charged_principal import original_frame, portable
from original_compact_charge_bound import chiral_basis, metric_constants, source_hashes as compact_hashes
from rotation_band import RotationBand
from sea_selection import SELECTED
from unrestricted_field_minimum import BARE

Matrix = NDArray[np.complex128]
RealArray = NDArray[np.float64]
ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-continuing-backward-20260930'
PLAN_REVISION = '381993ecb362c8c67dbc334aebbcb6093b64a7b78111b5a9e785d4746b9bbe3b'
TEMPLATE_REVISION = '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'


def source_hashes() -> dict[str, str]:
    identities = compact_hashes()
    for name in ('original_uniform_charge_gap.py', 'original_uniform_charge_gap.md'):
        identities[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return identities


def require(value: bool, label: str) -> None:
    if not value:
        raise AssertionError(label)


def residual(value: float, label: str, tolerance: float = 2e-9) -> None:
    require(math.isfinite(value) and value <= tolerance, label)


def sector_basis(modes: int, particles: int) -> list[int]:
    return [sum(1 << j for j in occupied)
            for occupied in itertools.combinations(range(modes), particles)]


def exterior_hamiltonian(h: Matrix, particles: int) -> tuple[Matrix, list[int]]:
    """Entire conserved CAR sector, including every occupation configuration."""
    basis = sector_basis(len(h), particles)
    lookup = {state: j for j, state in enumerate(basis)}
    result = np.zeros((len(basis), len(basis)), dtype=complex)
    for column, state in enumerate(basis):
        for j in range(len(h)):
            if not state & (1 << j):
                continue
            removed = state ^ (1 << j)
            sign_j = -1 if (state & ((1 << j)-1)).bit_count() % 2 else 1
            for i in range(len(h)):
                if removed & (1 << i):
                    continue
                sign_i = -1 if (removed & ((1 << i)-1)).bit_count() % 2 else 1
                result[lookup[removed | (1 << i)], column] += sign_i*sign_j*h[i, j]
    residual(float(np.linalg.norm(result-result.conj().T)), 'full CAR Hermiticity')
    return result, basis


def exterior_trace(u: Matrix, particles: int) -> complex:
    return complex(sum(np.linalg.det(u[np.ix_(occupied, occupied)])
                       for occupied in itertools.combinations(range(len(u)), particles)))


def tree_onebody(frame: Matrix, band: RotationBand, phase: float = 0.) -> Matrix:
    """One undirected tree edge, with the native endpoint and Wilson factors."""
    onsite = (band.m+band.r/(2*band.a))*frame[0]
    hop = -(band.r*frame[0]+1j*band.v*frame[1])/(2*band.a)*np.exp(1j*phase)
    return np.block([[onsite, hop], [hop.conj().T, onsite]])


def car_action(state: int, mode: int, create: bool) -> tuple[int, int] | None:
    occupied = bool(state & (1 << mode))
    if occupied == create:
        return None
    sign = -1 if (state & ((1 << mode)-1)).bit_count() % 2 else 1
    return state ^ (1 << mode), sign


def tree_correlator_controls(sectors: dict[int, tuple[Matrix, list[int]]],
                             mu: float, electric: float) -> dict[str, Any]:
    """Actual Wilson/reference creator correlator and full CAR-polynomial range."""
    spectra = {}
    for n, (matter, basis) in sectors.items():
        flux = np.array([(state & 15).bit_count()-2 for state in basis])
        spectra[n] = np.linalg.eigh(matter+np.diag(electric*flux**2))
    e0, vectors0 = spectra[4]
    ground = vectors0[:, 0]
    pivot = int(np.argmax(abs(ground)))
    source_state = sectors[4][1][pivot]
    source_modes = [j for j in range(8) if source_state & (1 << j)]
    rows = []
    for n in (3, 5):
        target_basis = sectors[n][1]
        lookup = {state: j for j, state in enumerate(target_basis)}
        single_images = []
        creators = []
        for mode in range(8):
            operator = np.zeros((len(target_basis), len(ground)), dtype=complex)
            for column, state in enumerate(sectors[4][1]):
                acted = car_action(state, mode, n == 5)
                if acted is not None:
                    target, sign = acted
                    operator[lookup[target], column] = sign
            creators.append(operator)
            single_images.append(operator@ground)
        # P_source first, then annihilate its four modes and create each target.
        # This explicitly evaluates a whole CAR polynomial, not a rank proxy.
        matrix_unit_images = np.zeros((len(target_basis), len(target_basis)), dtype=complex)
        for column, target_state in enumerate(target_basis):
            state, sign = source_state, 1
            for mode in source_modes:
                acted = car_action(state, mode, False)
                assert acted is not None
                state, factor = acted
                sign *= factor
            for mode in range(8):
                if target_state & (1 << mode):
                    acted = car_action(state, mode, True)
                    assert acted is not None
                    state, factor = acted
                    sign *= factor
            require(state == target_state, 'CAR matrix-unit target from source vacuum')
            matrix_unit_images[lookup[state], column] = sign*ground[pivot]
        full_rank = int(np.linalg.matrix_rank(matrix_unit_images))
        single_rank = int(np.linalg.matrix_rank(np.column_stack(single_images)))
        require(full_rank == len(target_basis), 'dense charged CAR-polynomial range in exact tree')
        require(single_rank < len(target_basis), 'single-creator-only coverage fails same full-range assertion')
        energies, vectors = spectra[n]
        # Mode zero is the left endpoint; its Wilson string to the right
        # reference is 1 in tree gauge but shifts E by its actual Gauss charge.
        image = creators[0]@ground
        weights = abs(vectors.conj().T@image)**2
        gaps = energies-float(e0[0])
        require(float(weights.sum()) > 1e-8, 'nonzero physical charged Wilson creator residue')
        active = weights > 1e-13
        residual(max(0., mu-float(gaps[active].min())), 'charged creator active spectral threshold')
        times = (0., 1., 4., 16.)
        correlations = [float(weights@np.exp(-t*gaps)) for t in times]
        weighted = [math.exp(mu*t)*c for t, c in zip(times, correlations)]
        require(all(c <= weighted[0]+1e-10 for c in weighted), 'physical charged correlation mu decay')
        rows.append(dict(q=n-4, UE1=electric, net_CAR_word_charge=n-4,
                         charged_body_dimension=len(target_basis), actual_CAR_polynomial_rank=full_rank,
                         with_all_reference_vectors_rank=4*full_rank,
                         single_creator_wrong_range_rank=single_rank,
                         neutral_source_pivot_amplitude=float(abs(ground[pivot])),
                         charged_ground_gap=float(gaps[0]),
                         active_creator_threshold=float(gaps[active].min()),
                         creator_spectral_weight_sum=float(weights.sum()),
                         times=list(times), Wilson_reference_creator_correlation=correlations,
                         mu_weighted_correlation=weighted))
    return dict(rows=rows, reference_site='right', creator_site='left',
                U1_Wilson='exp(i A_left,right), unitary; exactly 1 after tree gauge elimination',
                reference='opposite U1 and all four bar4/4 body vectors; SU4 regular-orbit lift retained',
                electric_flux_shift='left creator +1, left annihilator -1, compensating static reference at right',
                scope='Exact finite quantum tree correlations and range, not a periodic quantum EM spectrum')


def tree_quantum_controls(frame: Matrix, band: RotationBand, mu: float,
                          electric: Sequence[float]) -> dict[str, Any]:
    h = tree_onebody(frame, band)
    sectors = {n: exterior_hamiltonian(h, n) for n in (3, 4, 5)}
    rows = []
    require(bool(electric), 'at least one tree coupling')
    for ue in electric:
        require(math.isfinite(ue) and ue >= 0, 'nonnegative tree electric coefficient')
        energies = {}
        for n, (matter, basis) in sectors.items():
            n_left = np.array([(state & 15).bit_count() for state in basis])
            flux = n_left-2
            # Reference at the right vertex: -E=n_right-2-q, q=n-4.
            right_defect = -flux-((n-n_left)-2-(n-4))
            residual(float(np.max(abs(right_defect))), 'both exact tree Gauss equations')
            full = matter+np.diag(ue*flux**2)
            values, vectors = np.linalg.eigh(full)
            energies[n] = float(values[0])
        gap_plus = energies[5]-energies[4]
        gap_minus = energies[3]-energies[4]
        residual(max(0., mu-min(gap_plus, gap_minus)), 'exact tree uniform-mu charged gaps')
        rows.append(dict(UE1=ue, E_neutral=energies[4], E_plus=energies[5], E_minus=energies[3],
                         gap_plus=gap_plus, gap_minus=gap_minus,
                         margin_over_mu=min(gap_plus, gap_minus)-mu))
    correlations = tree_correlator_controls(sectors, mu, 1./band.a)
    covered = {str(row['q']): row['with_all_reference_vectors_rank'] for row in correlations['rows']}
    wrong_total_gauss = [n-4 for n in (3, 5)]
    require(all(q != 0 for q in wrong_total_gauss), 'omitted reference fails summed Gauss')
    return dict(rows=rows, full_CAR_sector_dimensions={str(n): len(s[1]) for n, s in sectors.items()},
                reference_dimension=4, charged_matrix_unit_range=covered,
                charged_Wilson_reference_correlators=correlations,
                omitted_reference_total_Gauss=wrong_total_gauss,
                electric_flux='E=n_left-2 exactly; no independent tree loop flux exists',
                rotor_cutoff_used=False, Fock_cutoff_used=False,
                graph='two vertices, one edge; no plaquette, hence magnetic term identically zero',
                scope='Exact full quantum U1 Gauss tree; SU4/reference orbit lift algebra only, not quantum colour diagonalization')


def time_evolution(hamiltonians: Sequence[Matrix], temporal: RealArray,
                   durations: RealArray) -> tuple[Matrix, list[Matrix], float]:
    modes = len(hamiltonians[0])
    volume = modes//4
    require(len(hamiltonians) == len(durations) == len(temporal), 'matched time segments')
    require(temporal.shape == (len(durations), volume), 'site temporal gauge, repeated four colours')
    u = np.eye(modes, dtype=complex)
    prefixes = [u.copy()]
    for h, a0, dt in zip(hamiltonians, temporal, durations):
        require(dt > 0, 'positive continuous-time segment')
        u = expm(dt*(-h+1j*np.diag(np.repeat(a0, 4))))@u
        prefixes.append(u.copy())
    phase = float(2*np.sum(durations[:, None]*temporal))
    return u, prefixes, phase


def temporal_green(monodromy: Matrix, prefixes: Sequence[Matrix]) -> list[Matrix]:
    """K(t,0) on the positive-time arc, including both one-sided endpoints."""
    source = np.linalg.solve(
        np.eye(len(monodromy), dtype=np.complex128)+monodromy,
        np.eye(len(monodromy), dtype=np.complex128))
    return [prefix@source for prefix in prefixes]


def green_kernel_controls(monodromy: Matrix, prefixes: Sequence[Matrix],
                          kernels: Sequence[Matrix]) -> dict[str, float]:
    """Identify the Dirac inverse, not merely a matrix below a loose bound."""
    require(len(kernels) == len(prefixes) and len(kernels) >= 2,
            'Green source and full period endpoints retained')
    source = kernels[0]
    before = -source@monodromy
    identity = np.eye(len(monodromy), dtype=np.complex128)
    jump = float(np.linalg.norm(source-before-identity)/math.sqrt(len(monodromy)))
    residual(jump, 'temporal Green identity source jump')
    boundary = float(np.linalg.norm(kernels[-1]+before)/max(1., np.linalg.norm(source)))
    residual(boundary, 'temporal Green antiperiodic boundary')
    evolution = max(float(np.linalg.norm(kernel-prefix@source)/
                          max(1., np.linalg.norm(kernel)))
                    for prefix, kernel in zip(prefixes, kernels))
    residual(evolution, 'temporal Green homogeneous evolution')
    return dict(source_jump=jump, antiperiodic_boundary=boundary,
                homogeneous_evolution=evolution)


def determinant_controls(hamiltonians: Sequence[Matrix], temporal: RealArray,
                         durations: RealArray, beta4: Matrix, mu: float,
                         h_bound: float) -> dict[str, Any]:
    u, prefixes, background_phase = time_evolution(hamiltonians, temporal, durations)
    volume = len(u)//4
    chirality4 = SELECTED[0]@SELECTED[1]@SELECTED[2]@SELECTED[3]
    chirality = np.kron(np.eye(volume), chirality4)
    beta = np.kron(np.eye(volume), beta4)
    coercive = []
    mass_odd_commutators = []
    for h, a0 in zip(hamiltonians, temporal):
        weighted = beta@(h-1j*np.diag(np.repeat(a0, 4)))
        real = (weighted+weighted.conj().T)/2
        least = float(np.linalg.eigvalsh(real)[0])
        residual(max(0., mu-least), 'arbitrary-time beta Euclidean coercivity')
        residual(float(np.linalg.norm(chirality@h+h@chirality)), 'native fixed chirality')
        coercive.append(least)
        odd = h-beta@real
        mass_odd_commutators.append(float(np.linalg.norm(real@odd-odd@real)))
    pseudo = float(np.linalg.norm(u.conj().T@chirality@u-chirality)/max(1., np.linalg.norm(u)**2))
    residual(pseudo, 'Hamiltonian evolution pseudounitarity')
    sign, logabs = np.linalg.slogdet(np.eye(len(u))+u)
    normalized_phase = complex(sign*np.exp(-1j*background_phase))
    residual(abs(normalized_phase-1), 'background-two normalized determinant positive phase')
    wrong_phase_defect = abs(complex(sign)-1)
    require(wrong_phase_defect > 1e-4, 'omitted background reaches and fails determinant positivity')
    actual_kernels = temporal_green(u, prefixes)
    kernel_identity = green_kernel_controls(u, prefixes, actual_kernels)
    # A constant identity meets the conservative decay inequality at the
    # native probe times but is not a Dirac Green function. The same physical
    # source-jump assertion must reject it.
    wrong_identity = [np.eye(len(u), dtype=np.complex128)]*len(prefixes)
    try:
        green_kernel_controls(u, prefixes, wrong_identity)
    except AssertionError as error:
        identity_kernel_failure = str(error)
    else:
        raise AssertionError('constant identity accepted as temporal Dirac inverse')
    total_time = float(durations.sum())
    gamma = mu/2
    delta = min(1., total_time/8)
    # Exact theorem constant, for beta>=8 delta; independent of dimension.
    log_constant = 4*(h_bound+gamma)*delta-math.log(delta*(mu-gamma))
    kernels = []
    elapsed = 0.
    for dt, kernel in zip(durations, actual_kernels[1:]):
        elapsed += float(dt)
        if elapsed >= total_time:
            continue
        kernel_norm = float(np.linalg.norm(kernel, ord=2))
        distance = min(elapsed, total_time-elapsed)
        log_weighted = math.log(max(kernel_norm, np.finfo(float).tiny))+gamma*distance
        residual(max(0., log_weighted-log_constant), 'uniform weighted pointwise temporal Green bound')
        kernels.append(dict(time=elapsed, circle_distance=distance, norm=kernel_norm,
                            log_weighted_norm=log_weighted, log_uniform_bound=log_constant))
    # Homotopy endpoint: h=mu beta, temporal holonomy pi, then mu->0.
    # The massless endpoint has a zero AP determinant, the positive endpoint not.
    holonomy = np.full((1, 1), math.pi/total_time)
    period = np.array([total_time])
    wrong_u, _, _ = time_evolution([np.zeros((4, 4), dtype=complex)], holonomy, period)
    positive_u, _, _ = time_evolution([mu*beta4], holonomy, period)
    zero_mass_wrong = float(np.linalg.svd(np.eye(4)+wrong_u, compute_uv=False)[-1])
    positive_mass_control = float(np.linalg.svd(np.eye(4)+positive_u, compute_uv=False)[-1])
    require(zero_mass_wrong < 1e-12 and positive_mass_control > 1e-6,
            'massless temporal-holonomy no-zero wrong control')
    return dict(segment_count=len(durations), continuous_time_period=total_time,
                generator='partial_tau - i A0 + h; exact piecewise-constant exp, no time derivative lattice',
                normalized_log_determinant=float(logabs), normalized_phase=normalized_phase,
                omitted_background_phase_defect=wrong_phase_defect,
                beta_coercive_minimum=min(coercive), pseudounitary_relative_residual=pseudo,
                actual_M_D_commutator_norms=mass_odd_commutators,
                temporal_Green_identity=kernel_identity,
                wrong_constant_identity_kernel_failure=identity_kernel_failure,
                gamma=gamma, delta=delta, dimension_independent_h_bound=h_bound,
                pointwise_Green=kernels, zero_mass_AP_smallest_singular=zero_mass_wrong,
                positive_mass_same_holonomy_smallest_singular=positive_mass_control)


def canonical_controls(frame: Matrix, band: RotationBand,
                       rng: np.random.Generator) -> dict[str, Any]:
    durations = np.full(5, .4*band.a)
    temporal = rng.uniform(-1, 1, (5, 2))/band.a
    hamiltonians = [tree_onebody(frame, band, float(p)) for p in rng.uniform(-math.pi, math.pi, 5)]
    u, _, phase = time_evolution(hamiltonians, temporal, durations)
    roots = np.exp(2j*np.pi*np.arange(17)/17)
    normalized: NDArray[np.complex128] = np.array(
        [z**-4*np.linalg.det(np.eye(8)+z*u)*np.exp(-1j*phase) for z in roots],
        dtype=np.complex128)
    residual(float(np.max(abs(normalized.imag))), 'all global holonomies positive neutral determinant')
    require(float(normalized.real.min()) > 0, 'strict positive determinant for all holonomies')
    canonical = complex(np.mean(normalized))
    exterior = exterior_trace(u, 4)*np.exp(-1j*phase)
    residual(abs(canonical-exterior)/max(1., abs(exterior)), 'exact half-filled Fourier projection equals full exterior trace')
    require(abs(normalized[0]-canonical) > 1e-3, 'grandcanonical is not canonical neutral trace')
    h = tree_onebody(frame, band)
    matter, basis = exterior_hamiltonian(h, 4)
    flux = np.array([(state & 15).bit_count()-2 for state in basis])
    ue, period = .7/band.a, 2*band.a
    exact = float(np.trace(expm(-period*(matter+np.diag(ue*flux**2)))).real)
    splitting = []
    for steps in (2, 4, 8, 16):
        dt = period/steps
        # Exact Hamiltonian Strang product. The electric factor is exactly the
        # Gaussian temporal-gauge average exp(i sigma Z (n_left-2)).
        half = expm(-dt*matter/2)
        transfer = half@np.diag(np.exp(-dt*ue*flux**2))@half
        trace = float(np.trace(np.linalg.matrix_power(transfer, steps)).real)
        splitting.append(dict(steps=steps, dt=dt, trace=trace, relative_error=abs(trace-exact)/exact))
    require(splitting[-1]['relative_error'] < splitting[0]['relative_error']/30,
            'controlled physical Hamiltonian Trotter convergence')
    path_phases = []
    for _ in range(16):
        steps = 8
        dt = period/steps
        kicks = math.sqrt(2*dt*ue)*rng.normal(size=steps)
        half = expm(-dt*h/2)
        evolution = np.eye(8, dtype=complex)
        for kick in kicks:
            evolution = half@np.diag(np.exp(1j*kick*np.r_[np.ones(4), np.zeros(4)]))@half@evolution
        sign, _ = np.linalg.slogdet(np.eye(8)+evolution)
        path_phases.append(abs(complex(sign*np.exp(-2j*kicks.sum()))-1))
    residual(max(path_phases), 'positive actual electric Hamiltonian temporal-gauge paths')
    return dict(exact_global_holonomy_nodes=17, neutral_NL=4, volume=2,
                canonical_half_trace=canonical, full_exterior_half_trace=exterior,
                grandcanonical_trace=complex(normalized[0]),
                normalized_determinant_minimum=float(normalized.real.min()),
                background_two_phase=phase, physical_tree_electric=ue,
                exact_neutral_quantum_partition=exact, physical_Hamiltonian_Strang_rows=splitting,
                temporal_Gaussian_path_phase_max=max(path_phases),
                approximation='Strang splitting only; exact CAR tree spectrum is separate. No Euclidean Wilson/time doubler.')


def run(*, size: int = 5, spacing: float = 1., centre: Sequence[int] = (2, 2, 2),
        all_centres: bool = False, seed: int = 2221005,
        electric: Sequence[float] = (0., .001, .1, 1., 100., 1e6),
        time_segments: int = 4) -> dict[str, Any]:
    require(size >= 5 and 0 < spacing <= 1 and time_segments >= 2, 'native N>=5, 0<a<=1 and time segments>=2')
    require(len(centre) == 3 and all(int(n) == n and 0 <= n < 4 for n in centre), 'actual Z4 centre label')
    hashes = source_hashes()
    band = RotationBand(**{**BARE, 'a': spacing})
    labels = list(itertools.product(range(4), repeat=3)) if all_centres else [tuple(int(n) for n in centre)]
    metrics = []
    selected_frame: Matrix | None = None
    selected_gram: RealArray | None = None
    for label in labels:
        selected, frame = original_frame(band, size, np.pi/2*np.array(label), True)
        gram = np.asarray(selected['G'], dtype=float)
        constants = metric_constants(band, gram, size**3)
        metrics.append(dict(centre=label, native_full10_Gram=gram,
                            full10_gradient=selected['gradient_residual'], mu=constants['mu']))
        if label == tuple(centre):
            selected_frame, selected_gram = frame, gram
    assert selected_frame is not None and selected_gram is not None
    frame, gram = selected_frame, selected_gram
    mu = metric_constants(band, gram, size**3)['mu']
    _, _, beta4 = chiral_basis(frame, gram, 1)
    h_bound = math.sqrt(float(gram[0, 0]))*(band.m+6*band.r/band.a)+abs(band.v)/band.a*sum(math.sqrt(float(gram[j, j])) for j in (1, 2, 3))
    rng = np.random.default_rng(seed)
    law = BulkQuantumLaw(size, band=band)
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    links = law.toron_links(np.pi/2*np.array(centre))
    spatial = rng.uniform(-math.pi, math.pi, (time_segments, law.volume, 3))
    hamiltonians = [law.onebody(fields, links, phases) for phases in spatial]
    durations = np.full(time_segments, 2*spacing/time_segments)
    temporal = rng.uniform(-.8, .8, (time_segments, law.volume))/spacing
    # Deliberately fix the background phase to pi/2, so omission is a strong
    # wrong control for every seed, not an accidental small-effect threshold.
    temporal[0, 0] += (math.pi/2-2*float(np.sum(durations[:, None]*temporal)))/(2*durations[0])
    time_controls = determinant_controls(hamiltonians, temporal, durations, beta4, mu, h_bound)
    time_controls.update(segment_durations=durations,
        spatial_phase_sha256=hashlib.sha256(spatial.tobytes()).hexdigest(),
        temporal_gauge_sha256=hashlib.sha256(temporal.tobytes()).hexdigest(),
        spatial_phase_range=[float(spatial.min()), float(spatial.max())],
        temporal_gauge_range=[float(temporal.min()), float(temporal.max())])
    canonical = canonical_controls(frame, band, rng)
    tree = tree_quantum_controls(frame, band, mu, electric)
    require(source_hashes() == hashes, 'source/proof bindings unchanged throughout consumer')
    return cast(dict[str, Any], portable(dict(
        governing_plan=PLAN, assignment_plan_revision=PLAN_REVISION, template_revision=TEMPLATE_REVISION,
        source_sha256=hashes, source_bindings_unchanged=True,
        native_runtime=dict(executable=sys.executable, python=sys.version, platform=platform.platform(),
                            numpy=np.__version__, scipy=scipy.__version__,
                            BLAS_threads={n: os.environ.get(n) for n in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS')}),
        inputs=dict(size=size, volume=size**3, spacing=spacing, centre=list(centre), seed=seed,
                    primitive_m=band.m, primitive_r=band.r, primitive_v=band.v,
                    time_segments=time_segments, tree_electric=list(electric)),
        native_metric_rows=metrics, all_sampled_centres_mu_min=min(row['mu'] for row in metrics),
        fixedmetric_mu=mu, Schur_beta=beta4,
        fulltorus_arbitrary_time_background=time_controls, canonical_and_physical_time_projection=canonical,
        exact_Gauss_full_CAR_tree=tree,
        exact_effective_theorem='For every fixed SPD G, UE1>0,K1>0: E_(2V+/-1, opposite static reference)-E_(2V, neutral)>=m sqrt(Schur G).',
        theorem_domain='Whole H1 compact-link torus and entire conserved original CAR sectors; all original SU4 orbit/reference Gauss lifting retained conditional on parent BO effective form.',
        theorem_not_a_finite_probe=True, full_quantum_periodic_EM_ground_computed=False,
        fundamental_quantum_law_assumed=True, all_checks_pass=True, campaign_complete=False,
        limits=['Native time matrices are piecewise-constant backgrounds, not a full quantum photon diagonalization.',
                'Quantum spectrum is exactly the two-site tree only; no tree magnetic plaquette exists.',
                'The full quantum effective theorem is proved in the companion, not inferred from finite spectra.',
                'BO normal transfer, heavy/well offsets and colour angular corrections remain parent-owned.',
                'No charged upper bound, thermodynamic deconfinement, isolated/mobile charge, mechanical spin, photon pole or UV closure.'])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=5)
    parser.add_argument('--spacing', '--a', type=float, default=1.)
    parser.add_argument('--centre', type=int, nargs=3, default=[2, 2, 2])
    parser.add_argument('--all-centres', action='store_true')
    parser.add_argument('--electric', type=float, nargs='+', default=[0., .001, .1, 1., 100., 1e6])
    parser.add_argument('--time-segments', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2221005)
    parser.add_argument('--output', type=Path, default=ROOT/'original_uniform_charge_gap_receipt.json')
    args = parser.parse_args()
    result = run(size=args.size, spacing=args.spacing, centre=args.centre, all_centres=args.all_centres,
                 seed=args.seed, electric=args.electric, time_segments=args.time_segments)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output), all_checks_pass=result['all_checks_pass'],
                          fixedmetric_mu=result['fixedmetric_mu'],
                          full_quantum_periodic_EM_ground_computed=False)))


if __name__ == '__main__':
    main()
