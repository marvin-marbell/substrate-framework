#!/usr/bin/env python3
"""Narrow prepared SOFT clocks and actual lattice spatial full-turn obstruction.

No endpoint spin sign is used as evidence that physical fields close.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import itertools
import json
import math
import os
from fractions import Fraction as F
from pathlib import Path
from typing import Any, cast

if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from improved_phi_band import ImprovedPhiLaw, selected_band
from original_total_angular_transfer import primitive_flow, scalar_orbital
from noncompact_quantitative_normal_budget import ceil_log2, sqrt_lower, sqrt_upper
from physical_carrier_transport import PrimitiveSector, native_onebody
from prepared_neutral_full_field import neutral_descriptor
from prepared_neutral_hessian import hessian_descriptor, termwise_exponents
from radial_band_noether import BETA, C, NSTAR, UE, batch_h, source_seed
from rotation_band import RotationBand
from sea_selection import coefficients
from spatial_noether_identification import anchored_coefficient, derived_spins, spatial_source_identification

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-fullfield-closedrotation-continuation-20261006'


def encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return dict(real=value.real.tolist(), imag=value.imag.tolist()) if np.iscomplexobj(value) else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return dict(real=value.real, imag=value.imag)
    raise TypeError(type(value).__name__)


def quintic_clock(outer_radius: int | float) -> int:
    """Outward integer clock for max(theta_dot)*R <= c/4.

    Binary64 rounding at the physical R gives an integer below the exact
    required clock. This strict rational upper enclosure of pi avoids that
    loss without changing the pulse or its speed cap.
    """
    pi_upper = F('3.14159265358979323846264338327950288419716939937511')
    required = 15*pi_upper*F(outer_radius)/C
    return -(-required.numerator//required.denominator)


def repriced_source_free_word(previous_clock: F, clock: int) -> dict[str, Any]:
    """Re-evaluate the source band phase; small relative clock change is not small phase."""
    import mpmath as mp
    band, frame, _ = selected_band()
    with mp.workdps(90):
        def real(value: float | F | int) -> Any:
            fraction = value if isinstance(value, F) else F(value)
            return mp.mpf(fraction.numerator)/fraction.denominator
        fields = [mp.matrix([[mp.mpc(real(float(z.real)), real(float(z.imag)))
                              for z in row] for row in phi]) for phi in frame]
        a, m, r, v = (real(float(getattr(band, key))) for key in ('a','m','r','v'))
        rows = []
        for point in (('0','0','0'), ('0.0000001','0','0'),
                      ('0.0000002','0','0'), ('0.0000003','0','0')):
            p = [mp.mpf(x) for x in point]
            delta = [2*mp.sin(a*x/2)**2 for x in p]
            mass = m+r/a*sum(x*x if band.improved else x for x in delta)
            h = mass*fields[0]
            h_minus = mass*fields[0]
            for j in range(3):
                h += v*mp.sin(a*p[j])/a*fields[j+1]
                h_minus -= v*mp.sin(a*p[j])/a*fields[j+1]
            energies, _ = mp.eighe(h)
            hole_energies, _ = mp.eighe(-h_minus.T)
            phases = []
            for charge, energy in ((1, energies[2]), (-1, hole_energies[2])):
                old = mp.exp(-1j*real(previous_clock)*energy)
                new = mp.exp(-1j*clock*energy)
                phases.append(dict(charge=charge, positive_source_energy=mp.nstr(energy,70),
                    old_phase=[mp.nstr(old.real,70),mp.nstr(old.imag,70)],
                    new_phase=[mp.nstr(new.real,70),mp.nstr(new.imag,70)],
                    wrong_retagged_old_phase_defect=mp.nstr(abs(new-old),70)))
            rows.append(dict(momentum=list(point), phases=phases))
        if max(float(row['phases'][0]['wrong_retagged_old_phase_defect']) for row in rows)<.001:
            raise ValueError('new clock phase control failed to distinguish an old-state retag')
    return dict(previous_clock=str(previous_clock), clock=clock,
        clock_increment=str(F(clock)-previous_clock), precision_decimal_digits=90,
        scope='actual selected native mean-band coefficient phase control, not an all64 state evolution',
        exact_particle_word='exp(-i T h_s(p)) on the positive Pauli-allowed cluster',
        exact_hole_word='exp(+i T h_s(-p)^T) on the negative-cluster hole',
        ordinary_phase_rule='retain exp(-i T E_ms,b), Eextra and all aliases at the SAME new T; no old-state reuse',
        rows=rows)


def packet_family(frame: np.ndarray, band: Any, scalar_box: dict[str, Any] | None = None) -> dict[str, Any]:
    """Actual C-infinity CAR source; one integral norm, no fibre renormalization.

    T is proportional to R, not assumed proportional to 1/p. The radius must
    also cover free flight, whose group speed is bounded by v^2 |p| / m.
    The q=6 product Sobolev bound pays the full clock in evolved derivatives.
    """
    nodes, wr = np.polynomial.legendre.leggauss(20)
    cosine, wu = np.polynomial.legendre.leggauss(12)
    azimuth = 2*math.pi*np.arange(24)/24
    records = []
    seed = source_seed(frame, band)
    m = band.m*float(np.linalg.norm(frame[0], 2))
    v = band.v*max(float(np.linalg.norm(f, 2)) for f in frame[1:])
    if scalar_box is not None:
        m = min(m, float(F.from_float(band.m)*sqrt_lower(F(scalar_box['g0_lower']))))
        v = max(v, float(F.from_float(band.v)*sqrt_upper(F(scalar_box['gs_upper']))))
    for scale in (1., .1, .01, .001, .0001):
        width = .001*scale
        centre = np.array([.002*scale, 0., 0.])
        rr, uu, aa = np.meshgrid(width*(nodes+1)/2, cosine, azimuth, indexing='ij')
        displacement = np.stack((rr*np.sqrt(1-uu*uu)*np.cos(aa), rr*np.sqrt(1-uu*uu)*np.sin(aa), rr*uu), axis=-1).reshape(-1, 3)
        points = centre+displacement
        weights = (wr[:, None, None]*width/2*rr**2*wu[None, :, None]*(2*math.pi/24)*np.ones_like(aa)).reshape(-1)
        s = np.sum(displacement**2, axis=1)/width**2
        amplitude = np.exp(-1/(1-s))
        scalar_norm = math.sqrt(float(np.sum(weights*amplitude**2)))
        amplitude /= scalar_norm
        h, _, _ = batch_h(points, frame, band)
        values, vectors = np.linalg.eigh(h)
        positive = vectors[:, :, 2:]
        projected = positive@positive.conj().transpose(0, 2, 1)@seed
        source = projected*amplitude[:, None, None]
        norm2 = float(np.einsum('x,xab,xab->', weights, source.conj(), source).real)
        source /= math.sqrt(norm2)
        p = float(np.linalg.norm(centre))+width
        inner = 2**27/scale
        outer = 2*inner
        # Smooth quintic angular clock has max theta_dot = (15/8)*2pi/T.
        duration = quintic_clock(outer)
        speed = (15/8)*2*math.pi*outer/duration
        free_speed = v*v*p/m
        travel = duration*free_speed
        radius_available = inner-travel
        # Material revision: Cauchy derivatives of the exact massive band
        # phase on a LOCAL complex ball. Subtract the real E(p) before using
        # |exp(-i T(E(z)-E(p)))|, instead of paying the irrelevant rest phase.
        # sqrt branch stays separated by m/(64v); |E(z)-E(p)| <=
        # 8 v^2(p+width)|z-p|/m there. rho makes T times this <=1/4.
        phase_radius = min(width/4, m/(64*v),
                           m/(32*duration*v*v*(p+width)))
        derivative = []
        for q in range(7):
            bound = 0.
            for a in range(q+1):
                for b in range(q-a+1):
                    c = q-a-b
                    eta = math.sqrt(8)*math.exp(4/3)*(64*max(a, 1)/width)**a
                    projector = math.factorial(b)*(16/.1294)**b
                    evolution = math.factorial(c)*math.e/phase_radius**c
                    bound += math.factorial(q)/(math.factorial(a)*math.factorial(b)*math.factorial(c))*eta*projector*evolution
            derivative.append(bound*np.linalg.norm(seed)/math.sqrt(norm2))
        tail = math.sqrt(3)*(math.sqrt(3)/max(radius_available, 1))**6*derivative[6]
        angular_tail = p*math.sqrt(3)*(math.sqrt(3)/max(radius_available, 1))**5*derivative[6]+tail
        # |sin(p)-p|<=|p|^3/6; two rotated endpoints. Range2 mass
        # sum(1-cos p_j)^2 <= |p|^4/4 at either endpoint.
        lattice = 2*v*p**3/6+2*band.r*np.linalg.norm(frame[0], 2)*p**4/4
        records.append(dict(scale=scale, width=width, centre=centre, support_p_upper=p,
            prepared_CAR_integral_norm2=norm2, normalized_source_integral=float(np.einsum('x,xab,xab->', weights, source.conj(), source).real),
            native_quadrature_nodes=len(points), scalar_norm=scalar_norm,
            inner_radius=inner, outer_radius=outer, full_angle=2*math.pi,
            duration=duration, control_speed=speed, free_group_speed_upper=free_speed,
            time_units='unscaled microscopic Hamiltonian time; hbar=1, a=1',
            total_speed_upper=speed+free_speed, free_travel_upper=travel,
            available_tail_radius=radius_available, derivative_q0_q6_upper=derivative,
            local_phase_Cauchy_radius=phase_radius,
            phase_derivative_revision='subtract rest phase on analytic complex ball; no T*restmass tail penalty',
            retained_tail_norm_upper=tail, supported_angular_error_upper=2*math.pi*angular_tail,
            lattice_covariance_upper=float(lattice), integrated_lattice_upper=float(duration*lattice),
            legal_support=outer<NSTAR/4, speed_legal=speed+free_speed<float(C)/2,
            tail_certificate_useful=tail<1, prepared_from_literal_negative_sea=True,
            ideal_meanPhi_only=True, full_field_rotation_claim=False))
    return dict(bump='exp(-1/(1-|p-p0|^2/width^2)); zero outside',
        normalization='one global L2 CAR norm on filled negative sea',
        tails_retained=True, arbitrary_packet_totalJ_sharp_claim=False,
        hypotheses=dict(p3_sine_defect_proved=True, clock_inverse_p_only_if_radius_inverse_p=True,
                        evolved_Sobolev_clock_bound_can_fail=True), records=records)


def field_trajectory(frame: np.ndarray, band: Any, amplitude: float, smooth: bool) -> dict[str, Any]:
    """Execute physical coordinate/vector/SU4/literal sine EM through 2pi.

    This is a native field integral curve, not a full-field quantum simulation.
    The source jet is on the same initial sea; neither endpoint is reprepared.
    """
    law = ImprovedPhiLaw(3, band)
    rng = np.random.default_rng(222106)
    f = np.tile(frame, (law.volume, 1, 1, 1))
    if smooth:
        profile = np.cos(2*math.pi*law.sites[:, 0]/law.size)
        direction = law.fields(np.tile(rng.normal(size=(1, 4, 15)), (law.volume, 1, 1)))
        f += amplitude*profile[:, None, None, None]*direction
    else:
        f += amplitude*law.fields(rng.normal(size=(law.volume, 4, 15)))
    u = np.asarray([[expm(1j*amplitude*np.einsum('a,aij->ij', rng.normal(size=15), law.generators)) for _ in range(3)] for _ in range(law.volume)])
    a = amplitude*rng.normal(size=(law.volume, 3))
    centre = np.ones(3)
    r = (law.sites-centre+law.size/2)%law.size-law.size/2
    velocity = np.cross(np.array([0., 0., 1.]), r)
    intrinsic = np.ones(law.volume)
    shapes = [f.shape, u.shape, a.shape]
    cuts = np.cumsum([int(np.prod(shape)) for shape in shapes])[:-1]
    def pack(parts: Any) -> np.ndarray:
        return np.concatenate([x.ravel() for x in parts])
    def unpack(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        parts = [x.reshape(shape) for x, shape in zip(np.split(y, cuts), shapes)]
        return parts[0], parts[1], cast(Any, parts[2]).real
    def rhs(theta: float, y: np.ndarray) -> np.ndarray:
        return pack(primitive_flow(cast(Any, law), *unpack(y), velocity, intrinsic))
    y0 = pack((f, u, a))
    solution = solve_ivp(rhs, (0., 2*math.pi), y0, rtol=2e-9, atol=2e-11, dense_output=True, max_step=.04)
    if not solution.success or solution.sol is None:
        raise AssertionError(solution.message)
    ff, uf, af = unpack(solution.y[:, -1])
    reference = f[law.index[(0, 1, 1)]].copy()
    eta = np.array([1., .7+.2j, .4-.1j])
    # Onceprepared actual light sea of the native restriction, not imposed spin.
    from physical_carrier_transport import native_onebody
    h = native_onebody(law, f, u, a)
    _, modes = np.linalg.eigh(h)
    occupied = modes[:, :6]
    allowed = np.eye(12)-occupied@occupied.conj().T
    coefficient = anchored_coefficient(law, f, u, a, eta, BETA, reference)
    norm = math.sqrt(float(np.trace(coefficient.conj().T@allowed@coefficient).real))
    coefficient /= norm
    step = 2e-4
    at = .41
    coefficients = []
    for k in (-2, -1, 0, 1, 2):
        fk, uk, ak = unpack(solution.sol(at+k*step))
        coefficients.append(anchored_coefficient(law, fk, uk, ak, eta, BETA, reference)/norm)
    m2, m1, zero, p1, p2 = coefficients
    first = (m2-8*m1+8*p1-p2)/(12*step)
    second = (-m2+16*m1-30*zero+16*p1-p2)/(12*step**2)
    ident = spatial_source_identification(law, f, u, a, eta, BETA, reference)
    final = anchored_coefficient(law, ff, uf, af, eta, BETA, reference)/norm
    defect = float(np.linalg.norm(final-coefficient))
    axes = ident['spins'][:, 2]
    selected = [law.index[tuple(site)] for site in ((0, 0, 0), (1, 0, 0), (1, 1, 0))]
    spin = np.zeros((12, 12), complex)
    for j, site in enumerate(selected):
        spin[4*j:4*j+4, 4*j:4*j+4] = axes[site]
    P = occupied@occupied.conj().T
    neutral_mean = float(np.trace(P@spin).real)
    neutral_variance = float(np.trace(P@spin@(np.eye(12)-P)@spin).real)
    car_endpoint = native_car_endpoint(law, solution.sol, unpack, velocity, occupied,
                                       coefficient, final, (f, u, a), (ff, uf, af))
    return dict(amplitude=amplitude, spatial_smoothing_revision=smooth,
        native_N=3, angle=2*math.pi, native_solver_success=True, rhs_evaluations=solution.nfev,
        Phi_endpoint_defect=float(np.linalg.norm(ff-f)), colour_endpoint_defect=float(np.linalg.norm(uf-u)),
        literal_EM_endpoint_defect=float(np.linalg.norm(af-a)), source_endpoint_defect=defect,
        physical_path_closed=defect<1e-7 and np.linalg.norm(ff-f)<1e-7,
        first_SOURCE_jet_norm=float(np.linalg.norm(first)), true_second_SOURCE_jet_norm=float(np.linalg.norm(second)),
        scalar_curvature_shortcut_defect=float(np.linalg.norm(second+.25*zero)),
        nonzero_SAME_initial_neutral_rightaction_norm=math.sqrt(max(0., neutral_mean**2+neutral_variance)),
        onceprepared_source_norm=norm, final_source_not_renormalized=True,
        onceprepared_CAR_physical_angular_endpoint=car_endpoint,
        orbital_velocity_norm=float(np.linalg.norm(velocity)),
        Phi_horizontal=ident['horizontal_Phi_norm'], colour_boundary=ident['retained_colour_link_boundary_norm'],
        derived_Ref_anchor=ident['retained_anchor_flux_norm'], literal_sine_EM=ident['retained_real_sine_EM_norm'],
        initial_centre0_algebraic_jet_reduction_error=ident['first_error'], fixed_reference=True,
        native_scope='actual field full-turn, initial finite CAR sea/source jet witness; not full-field quantum path',
        full_field_prepared_neutral_identity=None,
        debt='microscopic full-turn does not become closed by momentum narrowing; quantum same-neutral fullQ bounds must come from preparation owner')


def native_car_endpoint(law: Any, solution: Any, unpack: Any, velocity: np.ndarray,
                        occupied: np.ndarray, coefficient: np.ndarray, final: np.ndarray,
                        initial_fields: Any, final_fields: Any) -> dict[str, Any]:
    """Exact exterior functor of ordered finite-CAR angular transport.

    Occupied columns are transported by a unitary, never re-diagonalized or
    normalized. Full Phi integral-curve samples and literal microscopic current
    remain separate from this reduced CAR connection. This is not PNFF-v3.
    """
    from scipy import sparse
    selected = [law.index[tuple(site)] for site in ((0, 0, 0), (1, 0, 0), (1, 1, 0))]
    indices = np.concatenate([np.arange(4*x, 4*x+4) for x in selected])
    def transport(steps: int) -> tuple[np.ndarray, np.ndarray]:
        full = np.eye(12, dtype=complex)
        path = full.copy()
        ds = 2*math.pi/steps
        for j in range(steps):
            fields, links, em = unpack(solution((j+.5)*ds))
            axes = derived_spins(law, fields[selected])[0][:, 2]
            spin = sparse.block_diag(list(axes)).toarray()
            orbital = scalar_orbital(law, links, em, velocity)[np.ix_(indices, indices)]
            gauss = np.diag(np.repeat(np.einsum('xj,xj->x', velocity, em)[selected], 4))
            horizontal = orbital+gauss
            full = expm(1j*ds*(horizontal+spin))@full
            path = expm(1j*ds*horizontal)@path
        return full, path
    unitary, path = transport(128)
    low, lowpath = transport(64)
    sector = PrimitiveSector(10)
    omega = sector.neutral(occupied, 1)
    initial = sector.source(coefficient, 1)@omega
    transported_neutral = sector.neutral(unitary@occupied, 1)
    # This is Gamma(U) applied to the ONCE-PREPARED charged state, not a
    # fresh application of B(final fields) to the transported neutral.
    endpoint = sector.source(unitary@coefficient, 1)@transported_neutral
    insertion_endpoint = sector.source(final, 1)@transported_neutral
    # Apply the inverse ORDERED horizontal path to both B and the SAME neutral.
    relative_neutral = sector.neutral(path.conj().T@unitary@occupied, 1)
    relative_coefficient = path.conj().T@unitary@coefficient
    relative = sector.source(relative_coefficient, 1)@relative_neutral
    same_initial_neutral_shortcut = sector.source(relative_coefficient, 1)@omega
    cut = np.zeros(law.volume)
    cut[selected[0]] = 1.
    direction = cut[:, None]-cut[law.target]
    currents = []
    for snapshots, charged_state, neutral_state in ((initial_fields, initial, omega),
                                                    (final_fields, endpoint, transported_neutral)):
        hcurrent = cast(Any, native_onebody)(law, *snapshots, direction)
        current = sector.lift(sparse.block_diag((hcurrent, np.zeros((4, 4)))).toarray())
        currents.append(float((np.vdot(charged_state, current@charged_state)-
                               np.vdot(neutral_state, current@neutral_state)).real))
    return dict(native_dimension=sector.dimension, angular_steps=128,
        step_refinement_unitary_error=float(np.linalg.norm(unitary-low, 2)),
        step_refinement_horizontal_error=float(np.linalg.norm(path-lowpath, 2)),
        light_unitarity_error=float(np.linalg.norm(unitary.conj().T@unitary-np.eye(12))),
        initial_SOURCE_state_norm=float(np.linalg.norm(initial)),
        endpoint_SOURCE_state_norm=float(np.linalg.norm(endpoint)),
        actual_SOURCE_state_endpoint_difference=float(np.linalg.norm(endpoint-initial)),
        actual_field_SOURCE_insertion_state_endpoint_difference=float(np.linalg.norm(insertion_endpoint-initial)),
        field_SOURCE_insertion_vs_onceprepared_transport_defect=float(np.linalg.norm(insertion_endpoint-endpoint)),
        ordered_horizontal_subtracted_minus_SOURCE_defect=float(np.linalg.norm(relative+initial)),
        ignored_SAME_neutral_transport_defect=float(np.linalg.norm(relative-same_initial_neutral_shortcut)),
        actual_relative_cut_current_initial=currents[0],
        actual_relative_cut_current_endpoint=currents[1],
        actual_relative_cut_current_endpoint_difference=abs(currents[1]-currents[0]),
        neutral_columns_unitarily_evolved_not_reprepared=True,
        final_SOURCE_not_renormalized=True, field_quantum_normals_not_simulated=True,
        selected_PNFF_native_executed=False)


def coarse_highmode_discriminator(frame: np.ndarray, band: Any) -> dict[str, Any]:
    """Literal range2 SOURCE commutator for a highmode scalar-Phi normal.

    A quarter-turn is an exact cubic permutation, a positive control. A coarse
    action that rotates only the smooth source but leaves this highmode fixed
    misses the term below. At rest the Phi0 normal leaves the frame/source
    mark unchanged, so the CAR commutator is exactly C -> delta_h C.
    This is NOT asserted to be the evolved PNFF-v2 distribution.
    """
    law = ImprovedPhiLaw(3, band)
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    links = np.tile(np.eye(4, dtype=complex), (law.volume, 3, 1, 1))
    em = np.zeros((law.volume, 3))
    mode = np.sin(2*math.pi*law.sites[:, 0]/3)
    rotated_mode = np.sin(2*math.pi*law.sites[:, 1]/3)
    original = np.zeros_like(fields)
    rotated = np.zeros_like(fields)
    original[:, 0] = mode[:, None, None]*frame[0]
    rotated[:, 0] = rotated_mode[:, None, None]*frame[0]
    difference = law.onebody(rotated-original, links, em)
    baseline = law.onebody(fields, links, em)
    energies, basis = np.linalg.eigh(baseline)
    positive = basis[:, energies > 0]
    allowed = positive@positive.conj().T
    seed = source_seed(frame, band)
    rows = []
    for p in (3e-7, 3e-8, 0.):
        coefficient = np.einsum('x,ab->xab', np.exp(1j*p*law.sites[:, 1]), seed).reshape(4*law.volume, 4)
        coefficient = allowed@coefficient
        coefficient /= np.linalg.norm(coefficient)
        image = allowed@difference@coefficient
        rows.append(dict(momentum=p, highmode_relative_SOURCE_commutator_per_unit_amplitude=float(np.linalg.norm(image))))
    if min(r['highmode_relative_SOURCE_commutator_per_unit_amplitude'] for r in rows) < 1e-3:
        raise AssertionError('highmode SOURCE adverse lost its positive control')
    return dict(native_N=3, Phi0_highmode='sin(2pi x/3) versus exact quarter-turn sin(2pi y/3)',
                literal_range2_hamiltonian=True, full_negative_sea=True,
                rows=rows, relative_CAR_identity='[dGamma(delta_h),B(C)]=B(delta_h C)',
                highmodes_retained_in_H=True, total_quartic_bound_does_not_make_highmode_small=True,
                no_claim_this_adverse_is_evolved_PNFF=True)


def coarse_relative_requirements(packet: dict[str, Any]) -> dict[str, Any]:
    """Decision thresholds from exact SAME-neutral subtraction, not paid caps."""
    chosen = packet['records'][-1]
    duration = chosen['duration']
    radius = chosen['outer_radius']
    word = 3*NSTAR
    dimension = 75*(NSTAR**3+1)*(word+1)
    return dict(canonical_neutral='PNFF-v2 unfiltered xi; exact microscopic evolution',
        ordered_relative_defect='integral ||(D_theta B_theta - B_theta D_theta) xi_t|| dt',
        D_theta='U_coarse(theta)^* H U_coarse(theta) - H',
        neutral_rightaction='B_theta D_theta xi_t retained; no J xi=0 premise',
        required_source_weighted_uniform_moment_sectors=[
            'highmode/Phi covariant orbital and Cartesian-vector remainder',
            'SU4 curvature and continued Wilson endpoint/anchor',
            'literal sin(curl A), coexact and harmonic canonical EM/Gauss boundary',
            'derived Ref frame, guarded scalar psi1/psi0 curvature',
            'true first/second relative SOURCE words on xi_t',
            'fullQ and source-first/half-domain normal stress'],
        selected_packet=chosen, full_spatial_source_support=NSTAR**3,
        literal_word_upper=word, whole_source_coordinate_dimension=dimension,
        whole_source_coordinate_dimension_log2_upper=dimension.bit_length(),
        two_angular_coefficient_cost_log2_upper=ceil_log2((1+F(radius))**2),
        six_angular_coefficient_cost_log2_upper=ceil_log2((1+F(radius))**6),
        clock_log2_upper=ceil_log2(1+F(duration)),
        target_quantum_field_relative_error=1e-4,
        required_commutator_norm_uniform_upper=1e-4/duration,
        sufficient_field_moment_if_linear_radius_factor=1e-4/(duration*(1+radius)),
        independently_paid=False,
        failed_premise='total rho quartic moment bounds amplitude, not distance/highmode moment about selected orbit',
        initial_compact_support_is_not_uniform_time_bound=True,
        selected_N_scientific_state_native_simulated=False)


def physical_error_factors(packet: dict[str, Any]) -> dict[str, Any]:
    """Actual Duhamel multipliers; fullQ domains are not an absolute-small gate.

    Cq below are independent, globally source-normalized microscopic coefficient
    bounds. They must be supplied by the full-field SOURCE graph; no old static
    graph is inserted here. Every quantum moment is on the actual xi_t.
    """
    row = packet['records'][-1]
    T, R = row['duration'], row['outer_radius']
    return dict(physical_error_inequality='E <= Eband + Etail + Emetric + Erelative_source + Ewell',
        relative_source_integrand='||([H,B_theta] - i d_t B_theta - target_source_band_word) xi_t||',
        normalization='Cq are derivatives of the ONCE globally normalized source; if raw use Cq/Z_B',
        terms=[
            dict(name='normal_kinetic_first', operator='chi grad_metric(B).Pi xi_t',
                 multiplier=T, bound='T C1 sup_t chi||Pi xi_t||', formal_v3_Lambda_power='-1/16'),
            dict(name='normal_kinetic_second', operator='chi Delta_metric(B) xi_t/2',
                 multiplier=T/2, bound='T chi C2/2', formal_v3_Lambda_power='-1/2'),
            dict(name='microscopic_highmode_matter', operator='[dGamma(delta_h(Phi_high)),B] xi_t',
                 multiplier=T, bound='T C0 sup_t ||rho_high xi_t||', formal_v3_Lambda_power='-5/16'),
            dict(name='angular_highmode_SOURCE', operator='theta_dot Y_remainder B xi_t',
                 multiplier=2*math.pi*R, bound='2pi R C1 sup_t ||rho_high xi_t||', formal_v3_Lambda_power='-5/16'),
            dict(name='colour_EM_Gauss_anchor', operator='actual relative continued colour/literal EM/Gauss/Ref words',
                 multiplier=2*math.pi, bound='integral |theta_dot| ||relative boundary SOURCE word xi_t||',
                 formal_v3_Lambda_power=None),
            dict(name='physical_well_leakage', operator='[H,Pwell] and theta_dot[J,Pwell]',
                 multiplier=T, bound='T sup ||[H,Pwell]xi_t|| + 2pi sup ||[J,Pwell]xi_t|| + initial outside norm',
                 formal_v3_Lambda_power=None)],
        fullQ_absolute_norm_must_be_small=False,
        fullQ_coefficient_in_this_direct_Duhamel_inequality=0,
        fullQ_and_half_domains_still_required=True,
        gap_filter_or_graphmix='only add a fullQ error after deriving its actual gap/cutoff/pulse coefficient; none silently inherited',
        formal_v3_negative_powers_not_actual_uniform_moment_proof=True,
        v2_excess_normal_kinetic_first_Lambda_power='1/8',
        all64_ordinary_offsets_retained=True,
        well_commutator='[H,Pwell]=-(chi/2)(Delta_metric Pwell+2 grad_metric Pwell.Pi)+colour terms',
        source_and_angular_coefficients_independently_paid=False)


def cubic_metric_certificate(metric_receipt: Path, packet: dict[str, Any], band: Any) -> dict[str, Any]:
    """Separate exact isotropic scalar uncertainty from finite-N anisotropy.

    The frozen Gram leaf is only read. The new native control uses a fresh
    arbitrary positive full10 Gram, not its executed minimizer supplier.
    """
    raw = metric_receipt.read_bytes()
    if hashlib.sha256(raw).hexdigest() != '17d414bb6150115f8f9a7478d4e9af59ff235fb5b31626ebe547f93e686a3a35':
        raise ValueError('wrong frozen native-metric03 certificate bytes')
    cert = json.loads(raw)['certificate']
    if not cert['admission'] or cert['parameters']['kappa'] != 2 or cert['parameters']['improved']:
        raise ValueError('original-heavy full10 variational certificate required')
    box = cert['continuum_box']
    g0lo, g0hi = (F.from_float(x) for x in box['G00'])
    gslo, gshi = (F.from_float(x) for x in box['Gjj_each'])
    mlo, mhi = (F.from_float(x) for x in cert['parameters']['m'])
    _, rhi = (F.from_float(x) for x in cert['parameters']['r'])
    _, vhi = (F.from_float(x) for x in cert['parameters']['v'])
    # Elementary Taylor/geometric remainders give cosh(1/2)<1.13,
    # sinh(1/2)<.53. These rational bounds extend the strip from the
    # numerical trial to EVERY exact scalar minimizer in its frozen box.
    ch, sh = F(113, 100), F(53, 100)
    delta = sqrt_lower(g0lo)*(mlo-3*rhi*(ch-1))-vhi*sqrt_upper(3*gshi)*sh
    numerator = (mhi+3*rhi*(1+ch))**2+3*vhi**2*ch**2
    majorant = 2*F.from_float(cert['analytic_strip']['matrix_gradient_integrand_HS_majorant'])
    if delta <= 0 or numerator > majorant*delta or cert['analytic_strip']['sigma'] != .5:
        raise AssertionError('actual continuum minimizer strip extension failed')
    # sigma N >1024 log2 is extremely strict at this N. log2<1 suffices.
    if F(NSTAR, 2) <= 1024:
        raise AssertionError('legal alias cap not admitted')
    alias = F(1, 2**1024)
    gradient_alias = majorant*2*alias*(3+alias**2)/(1-alias)**3
    gram_distance = gradient_alias/2  # full Frobenius Hessian >=kappa=2.
    lower_spatial = F(3, 50)
    if gslo-gram_distance <= lower_spatial:
        raise AssertionError('actual finite-N square-root Gram gap failed')
    frame_distance = sqrt_upper(1/(4*lower_spatial))*gram_distance
    # Exact scalar calibration debt is NOT an anisotropic rotation error.
    light_v = F.from_float(band.v)
    cone_lower, cone_upper = light_v*sqrt_lower(gslo), light_v*sqrt_upper(gshi)
    cone_debt = max(abs(cone_lower-C), abs(cone_upper-C))
    rows = []
    for prepared in packet['records']:
        p = F(str(prepared['support_p_upper']))
        duration = F(math.ceil(prepared['duration'])+1)
        light_m, light_r = F.from_float(band.m), F.from_float(band.r)
        cov = 2*(light_m+light_r*p**4/4+light_v*p)*frame_distance
        spin = 4*(frame_distance+gram_distance)/lower_spatial
        vactual = light_v*sqrt_upper(gshi+gram_distance)
        actual_mass = light_m*sqrt_lower(g0lo-gram_distance)
        lattice = 2*vactual*p**3/6+light_r*sqrt_upper(g0hi+gram_distance)*p**4/2
        rows.append(dict(scale=prepared['scale'], exact_duration_upper=str(duration),
            finiteN_anisotropic_operator_covariance_upper=str(cov),
            integrated_finiteN_anisotropic_covariance_log2_upper=ceil_log2(duration*cov),
            rest_current_spin_alias_upper=str(spin),
            rest_spin_alias_fullturn_log2_upper=ceil_log2(F(7)*spin),
            current_positive_projector_alias_upper=str(4*cov/F(1294, 10000)),
            actual_scalar_velocity_upper=str(vactual), actual_scalar_mass_lower=str(actual_mass),
            actual_group_speed_upper=str(vactual*vactual*p/actual_mass),
            integrated_actual_scalar_lattice_upper=float(duration*lattice),
            actual_scalar_sine_quartic_plus_alias_error_upper=float(duration*(lattice+cov))))
    # Native positive control for the SAME variational functional: complete
    # cubature is closed under reflections/permutations, not a heavy-leaf rerun.
    grid = 2*math.pi*np.indices((9,)*3).reshape(3, -1).T/9-math.pi
    heavy = RotationBand(**neutral_descriptor()['heavy_parameters'])
    x = coefficients(heavy, grid)
    A = np.array([[.65, .11, -.07, .02], [.11, .42, .03, -.05],
                  [-.07, .03, .31, .04], [.02, -.05, .04, .27]])
    G = A@A.T+.05*np.eye(4)
    def objective(g: np.ndarray) -> float:
        return float(np.sum(g*g)-2*np.mean(np.sqrt(np.einsum('xi,ij,xj->x', x, g, x))))
    baseline = objective(G)
    invariance = 0.
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1., 1.), repeat=3):
            Q = np.zeros((4, 4)); Q[0, 0] = 1.
            for j, k in enumerate(perm):
                Q[1+j, 1+k] = signs[j]
            invariance = max(invariance, abs(objective(Q@G@Q.T)-baseline))
    direction = np.array([[.2, .3, -.1, .04], [.3, -.4, .03, .02],
                          [-.1, .03, .12, -.05], [.04, .02, -.05, .16]])
    z = np.einsum('xi,ij,xj->x', x, direction, x)
    r = np.sqrt(np.einsum('xi,ij,xj->x', x, G, x))
    norm2 = float(np.sum(direction*direction))
    hessian = 2*norm2+.5*float(np.mean(z*z/r**3))
    if invariance > 2e-13 or hessian < 2*norm2:
        raise AssertionError('full10 symmetry/strong-convexity positive control failed')
    return dict(metric_input=dict(path=str(metric_receipt), sha256=hashlib.sha256(raw).hexdigest()),
        functional='F(G)=kappa||G||F^2/2 -2 integral sqrt(x^T G x), full PSD4x4 Gram, kappa=2',
        Hessian_identity='d2F[Delta,Delta]=kappa||Delta||F^2 +(1/2)integral (x^T Delta x)^2/(x^T G x)^(3/2)',
        symmetry='continuum x0 even/cubic and xj=v sin(pj); every spatial signed permutation preserves measure and F',
        uniqueness='kappa-strong convexity plus frozen positive interior certificate: unique minimizer fixed by all48 signed permutations',
        continuum_exact_shape='G_infinity=diag(g0,gs,gs,gs), without setting g0 or gs to their numerical trial',
        scalar_box=dict(g0_lower=str(g0lo), g0_upper=str(g0hi), gs_lower=str(gslo), gs_upper=str(gshi)),
        actual_minimizer_strip_coercivity_lower=str(delta), actual_minimizer_strip_majorant_upper=str(majorant),
        all_real_shifts=True, all64=True, alias_positive_cap='2^-1024',
        gradient_alias_upper=str(gradient_alias), full10_G_N_minus_G_infinity_upper=str(gram_distance),
        scalar_trial_distance_is_not_anisotropy=True,
        scalar_cone_calibration_debt_upper=str(cone_debt),
        rest_spin='derived current spin -i[Phi_j,Phi_k]/(4gs), then actual positive rest projection; scalar g0/gs cancel',
        calibration_and_actual_well_centering_still_separate=True,
        native_control=dict(full10_arbitrary_nonisotropic_Gram=G, signed_permutations=48,
                            objective_invariance_error=invariance, hessian_direction=hessian,
                            strict_Frobenius_lower=2*norm2, frozen_supplier_rerun=False),
        legal_N_rotation_metric_rows=rows, full_field_normal_highmode_debt_paid=False)


def physical_em_noether(law: Any, maps: np.ndarray,
                        orbitals: np.ndarray, reference_site: int) -> dict[str, Any]:
    """Actual periodic sine-clover/Gauss generator, Gaussian first-degree chart.

    maps[i,x,k] is the PHYSICAL A coefficient of standardized oscillator q_i.
    Nonlinear sine and Peierls functions are integrated before projection.
    No eigenvalue normalization, integer fit, or rotation of torons is imposed.
    """
    count, volume, _ = maps.shape
    xyz = law.sites.astype(float)-(law.size-1)/2
    velocity = np.cross(np.array([0., 0., 1.]), xyz)
    dual = np.linalg.pinv(maps.reshape(count, -1).T)
    gradient = np.zeros((3*volume, volume))
    for x, axis in itertools.product(range(volume), range(3)):
        gradient[3*x+axis, x] = 1
        gradient[3*x+axis, int(law.target[x, axis])] -= 1
    green = np.linalg.pinv(gradient.T@gradient)
    previous = np.argsort(law.target, axis=0)
    phases, tangents, gauges = [], [], []
    for anchor in range(volume):
        for j, k in itertools.combinations(range(3), 2):
            for s, t in itertools.product((-1, 1), repeat=2):
                z = anchor
                word = np.zeros(count)
                for axis, sign in ((j, s), (k, t), (j, -s), (k, -t)):
                    edge = z if sign == 1 else int(previous[z, axis])
                    word += sign*maps[:, edge, axis]
                    z = int(law.target[z, axis]) if sign == 1 else edge
                if z != anchor:
                    raise AssertionError('physical clover word did not close')
                delta = np.zeros((volume, 3))
                coefficient = s*t/(4*law.band.a**2)
                for axis, direction, orientation in ((k, j, 1), (j, k, -1)):
                    for x in (anchor, int(previous[anchor, axis])):
                        y = int(law.target[x, axis])
                        speed = (velocity[x, direction]+velocity[y, direction])/2
                        delta[x, axis] -= law.band.a*.5*speed*coefficient*orientation
                phases.append(word)
                tangents.append(dual@delta.reshape(-1))
                lam = -green@gradient.T@delta.reshape(-1)
                gauges.append(lam-lam[reference_site])
    words = np.asarray(phases)
    coefficients_native = np.asarray(tangents)
    linear = coefficients_native.T@words
    sine = coefficients_native.T@(np.exp(-np.sum(words**2, axis=1)/4)[:, None]*words)
    generator = np.zeros((count+1, count+1), dtype=complex)
    generator[1:, 1:] = -.5j*(sine.T-sine)
    gauge_words = np.asarray(gauges)
    lambda_coefficients = np.zeros((count+1, count+1, volume))
    lambda_coefficients[0, 1:] = (words.T@(np.exp(-np.sum(words**2, axis=1)/4)[:, None]*gauge_words))/math.sqrt(2)
    lambda_coefficients[1:, 0] = lambda_coefficients[0, 1:]
    gauss_orbital = np.einsum('pqx,xa,xb->pqab', lambda_coefficients, orbitals, orbitals)
    # Exact first-degree Gaussian matrix elements of every literal Wilson edge.
    orbital = np.zeros((count+1, count+1, 2, 2), dtype=complex)
    continuum_gauss = np.zeros_like(orbital)
    identity = np.eye(count)
    q = np.zeros((count+1, count+1, count))
    for i in range(count):
        q[0, i+1, i] = q[i+1, 0, i] = 1/math.sqrt(2)
    for x in range(volume):
        for axis in range(3):
            y = int(law.target[x, axis])
            speed = (velocity[x, axis]+velocity[y, axis])/2
            word = maps[:, x, axis]
            weight = math.exp(-float(word@word)/4)
            wilson = np.zeros((count+1, count+1), dtype=complex)
            wilson[0, 0] = 1
            wilson[0, 1:] = wilson[1:, 0] = 1j*word/math.sqrt(2)
            wilson[1:, 1:] = identity-np.outer(word, word)/2
            wilson *= weight
            forward = np.outer(orbitals[x], orbitals[y])/(2j*law.band.a)*speed
            orbital += np.einsum('pq,ab->pqab', wilson, forward)
            orbital += np.einsum('qp,ba->pqab', wilson.conj(), forward.conj())
        lam = np.einsum('j,ij->i', velocity[x], maps[:, x])
        continuum_gauss += np.einsum('pqi,i,ab->pqab', q, lam,
                                    np.outer(orbitals[x], orbitals[x]))
    covariant_orbital = orbital.copy()
    orbital -= gauss_orbital
    # Independent actual microscopic zero-field orbital, never rescaled.
    links = np.tile(np.eye(4, dtype=complex), (volume, 3, 1, 1))
    folded = np.kron(orbitals, np.eye(4))
    microscopic = folded.conj().T@scalar_orbital(law, links, np.zeros((volume, 3)), velocity)@folded
    hermitian = float(np.linalg.norm(orbital-orbital.conj().transpose(1, 0, 3, 2)))
    if hermitian > 1e-10:
        raise AssertionError('actual EM/Gauss orbital matrix is not Hermitian')
    return dict(generator=generator, orbital_coefficients=orbital,
        covariant_orbital_coefficients=covariant_orbital,
        gauss_orbital_coefficients=gauss_orbital,
        gauss_lambda_coefficients=lambda_coefficients,
        continuum_lambda_orbital_coefficients=continuum_gauss,
        actual_vs_continuum_Gauss_difference=float(np.linalg.norm(gauss_orbital-continuum_gauss)),
        Gauss_fix='lambda=-(D^dagger D)^+ D^dagger primitive_da; anchor lambda(FAR)=0',
        reference_site=reference_site,
        linear_velocity=linear, sine_velocity=sine,
        full_sine_photon_vacuum_to_two_quantum_norm=float(np.linalg.norm((sine+sine.T)/2)/math.sqrt(2)),
        full_linear_photon_vacuum_to_two_quantum_norm=float(np.linalg.norm((linear+linear.T)/2)/math.sqrt(2)),
        vacuum_pair_formula='P2G|0>: diagonal i*Mii/sqrt2, offdiagonal i*(Mij+Mji)/2; NOT retained by vacuum+degree1',
        physical_maps=maps, clover_phase_words=words,
        clover_coordinate_velocity_coefficients=coefficients_native,
        clover_gauge_words=gauge_words,
        coexact_Gauss_projection_norm=float(np.linalg.norm(dual@np.stack(
            [(np.einsum('xj,xj->x', velocity, f)[:, None]
              -np.einsum('xj,xj->x', velocity, f)[law.target]).reshape(-1)
             for f in maps], axis=1))),
        zero_field_microscopic_orbital=microscopic,
        nonlinear_sine_generator_change=float(np.linalg.norm(sine-linear)),
        orbital_Hermiticity_error=hermitian,
        local_supported_seed='P_coexact[exp(-r²/(2*.65²)) e_x/e_y]; complete coexact basis retains every outside partner',
        full_photon_Fock_truncation_debt=True)


def physical_spin_quotient(law: Any, frame: np.ndarray,
                           phi_cometric: float) -> dict[str, Any]:
    """Derive the body spin from vector Phi flow and all15 Gauss columns."""
    from moving_physical_noether import physical_generators
    from unrestricted_field_minimum import coordinates
    gauge = np.column_stack([coordinates(1j*(t@frame-frame@t), law.basis).reshape(60).real
                             for t in law.generators])
    mass = np.full(60, 1/phi_cometric)
    inertia = gauge.T@(mass[:, None]*gauge)
    singular = np.linalg.svd(np.sqrt(mass[:, None])*gauge, compute_uv=False)
    rank = int(np.linalg.matrix_rank(np.sqrt(mass[:, None])*gauge))
    if rank != 15:
        raise ValueError('physical Phi gauge image is not the full15-dimensional image')
    spins = physical_generators(law, frame)
    residual, coefficients_actual = [], []
    for j, k in ((2, 3), (3, 1), (1, 2)):
        tangent = np.zeros_like(frame)
        tangent[j], tangent[k] = -frame[k], frame[j]
        vector = coordinates(tangent, law.basis).reshape(60).real
        coefficient = np.linalg.solve(inertia, gauge.T@(mass*vector))
        coefficients_actual.append(coefficient)
        residual.append(float(np.linalg.norm(np.sqrt(mass)*(gauge@coefficient-vector))))
    projected = np.einsum('ia,ajk->ijk', coefficients_actual, law.generators)
    if np.linalg.norm(projected-spins) > 1e-10:
        raise ValueError('weighted physical Phi quotient differs from supplier Gauss solve')
    algebra = max(float(np.linalg.norm(spins[j]@spins[k]-spins[k]@spins[j]-1j*spins[i]))
                  for i,j,k in ((0,1,2),(1,2,0),(2,0,1)))
    values, basis = np.linalg.eigh(law.band.m*frame[0])
    positive = basis[:, values > 0]
    rest = np.array([positive.conj().T@s@positive for s in spins])
    casimir = sum(s@s for s in rest)
    return dict(spins=spins, full_Phi_coordinates=60, gauge_rank=rank,
        normal_coordinates=60-rank, gauge_singular_values=singular,
        phi_cometric=phi_cometric, actual_mass_metric='I60/chi; no normal-only projection',
        vector_flow_before_Gauss_residual=residual,
        solved_Gauss_coefficients=np.asarray(coefficients_actual),
        supplier_api_sha256=hashlib.sha256(inspect.getsource(physical_generators).encode()).hexdigest(),
        su2_algebra_defect=algebra, physical_rest_spin_eigenvalues=np.linalg.eigvalsh(rest[2]),
        physical_rest_Casimir=casimir, rest_cluster_dimension=positive.shape[1],
        halfspin_is_output_not_input=True)






def joint_phi_colour_noether_symbol(model: Any, alpha: np.ndarray,
                                    velocity: np.ndarray) -> dict[str, Any]:
    """All105/all15 mass-Gauss split of the actual mean primitive vector."""
    from unrestricted_field_minimum import coordinates
    law = model.law
    volume = law.volume
    fields = np.tile(model.frame, (volume, 1, 1, 1))
    links = np.tile(np.eye(4, dtype=complex), (volume, 3, 1, 1))
    phi_flow, colour_flow, _ = primitive_flow(
        law, fields, links, np.zeros((volume, 3)), velocity, alpha)
    phi_coordinates = coordinates(phi_flow, law.basis).reshape(volume, 60).real
    colour_coordinates = (2*np.einsum(
        'xjkl,alk->xja', -1j*colour_flow, law.generators)).reshape(volume, 45).real
    raw = np.c_[phi_coordinates, colour_coordinates]
    gauge_phi = np.column_stack([
        coordinates(1j*(t@model.frame-model.frame@t), law.basis).reshape(60).real
        for t in law.generators])
    cometric = np.r_[np.ones(60), np.full(45, .6)]
    mass = 1/cometric
    fourier = np.exp(-2j*math.pi*(law.sites@law.sites.T)/law.size)/math.sqrt(volume)
    raw_fourier = fourier@raw
    parameters, horizontal = [], []
    for mode, vector in zip(law.sites, raw_fourier):
        q = 2*math.pi*mode/law.size
        gauge = np.vstack((gauge_phi, *[
            (1-np.exp(1j*q[j]))*np.eye(15) for j in range(3)]))
        coefficient = np.linalg.solve(
            gauge.conj().T@(mass[:, None]*gauge), gauge.conj().T@(mass*vector))
        parameters.append(coefficient)
        horizontal.append(vector-gauge@coefficient)
    coefficients = fourier.conj().T@np.asarray(parameters)
    gauss = np.einsum('xa,aij->xij', coefficients, law.generators)
    if np.linalg.norm(gauss-gauss.conj().swapaxes(-1, -2)) > 1e-10:
        raise ValueError('joint mass-Gauss generator is not Hermitian')
    return dict(raw_vector=raw, horizontal_vector=fourier.conj().T@np.asarray(horizontal),
        gauss_parameters=coefficients, gauss_matrices=gauss,
        full_coordinates=105, full_gauss_columns=15,
        mass_diagonal=mass, colour_cometric_over_chi=.6,
        scope='actual mean primitive Phi/colour symbol; no normal-H or whole-G promotion')


def global_raw_noether_core() -> dict[str, Any]:
    """Complete RAW operator law plus outside-chart controls; no quantum H-loop."""
    from scipy.linalg import expm, block_diag
    from original_total_angular_transfer import scalar_orbital
    from noncompact_screened_gauss_dressing import green, divergence
    law: Any = ImprovedPhiLaw(3)
    volume = law.volume
    rng = np.random.default_rng(22261)
    fields: Any = law.fields(rng.normal(size=60*volume))*1e6
    links: Any = np.asarray([[expm(1j*np.einsum('a,aij->ij',
        rng.normal(size=15), law.generators)) for _ in range(3)]
        for _ in range(volume)])
    phases = 80*rng.normal(size=(volume, 3))
    reference = law.index[(0, 1, 1)]
    alpha = np.ones(volume)
    alpha[reference] = 0
    sites = law.sites-1
    velocity = alpha[:, None]*np.column_stack((-sites[:, 1], sites[:, 0],
                                               np.zeros(volume)))
    maximum_velocity = float(np.max(abs(velocity)))
    a = law.band.a
    growth = 3*maximum_velocity/a+float(np.max(abs(alpha)))
    flow = primitive_flow(law, fields, links, phases, velocity, alpha)
    scaled_flow = primitive_flow(law, fields/1e6, links, phases, velocity, alpha)
    linearity = float(np.linalg.norm(flow[0]/1e6-scaled_flow[0]))
    phi_ratio = float(np.linalg.norm(flow[0])/np.linalg.norm(fields))
    em_bound = 6*maximum_velocity*math.sqrt(volume)/a
    colour_bound = 12*maximum_velocity/a
    left = -1j*flow[1]@links.conj().swapaxes(-1, -2)
    if (phi_ratio > growth or np.linalg.norm(flow[2]) > em_bound
            or np.max(np.linalg.norm(left, axis=(-2, -1))) > colour_bound
            or linearity > 1e-12):
        raise ValueError('actual RAW primitive violates its global completeness bounds')
    gauge = np.asarray([expm(1j*np.einsum('a,aij->ij',
        rng.normal(size=15), law.generators)) for _ in range(volume)])
    fg: Any = gauge[:, None]@fields@gauge[:, None].conj().swapaxes(-1, -2)
    ug: Any = np.asarray([[gauge[x]@links[x, j]@gauge[law.target[x, j]].conj().T
        for j in range(3)] for x in range(volume)])
    gauged = primitive_flow(law, fg, ug, phases, velocity, alpha)
    expected_phi = gauge[:, None]@flow[0]@gauge[:, None].conj().swapaxes(-1, -2)
    expected_link = np.asarray([[gauge[x]@flow[1][x, j]@
        gauge[law.target[x, j]].conj().T for j in range(3)] for x in range(volume)])
    orbital = scalar_orbital(law, links, phases, velocity)
    orbital_g = scalar_orbital(law, ug, phases, velocity)
    body_g = cast(np.ndarray, block_diag(*gauge))
    covariance = dict(
        Phi_relative=float(np.linalg.norm(gauged[0]-expected_phi)/np.linalg.norm(flow[0])),
        colour=float(np.linalg.norm(gauged[1]-expected_link)),
        EM=float(np.linalg.norm(gauged[2]-flow[2])),
        scalar_orbital=float(np.linalg.norm(orbital_g-body_g@orbital@body_g.conj().T)))
    gauge_phase = rng.normal(size=volume)
    phase_g = phases+gauge_phase[law.target]-gauge_phase[:, None]
    u1_flow = primitive_flow(law, fields, links, phase_g, velocity, alpha)
    u1_body = np.diag(np.repeat(np.exp(-1j*gauge_phase), 4))
    covariance['U1_EM'] = float(np.linalg.norm(u1_flow[2]-flow[2]))
    covariance['U1_scalar_orbital'] = float(np.linalg.norm(
        scalar_orbital(law, links, phase_g, velocity)-
        u1_body@orbital@u1_body.conj().T))
    covariance['scalar_orbital_Hermiticity'] = float(np.linalg.norm(orbital-orbital.conj().T))
    if max(covariance.values()) > 1e-9:
        raise ValueError('outside-chart RAW gauge covariance failed')
    green_matrix = green(law)[0]
    lambda_c = green_matrix@divergence(law, flow[2])
    lambda_c -= lambda_c[reference]
    em_horizontal = flow[2]+lambda_c[law.target]-lambda_c[:, None]
    coulomb_defect = float(np.linalg.norm(divergence(law, em_horizontal)))
    # At Phi=0,U=I the FULL gauge Gram has constant null directions.
    # RAW Noether is still defined: no inverse or singular-source frame is used.
    zero_phi = np.zeros_like(fields)
    identity_links = np.tile(np.eye(4, dtype=complex), (volume, 3, 1, 1))
    zero_gauge, _ = native_full_gauss_columns(law, zero_phi, identity_links)
    constant = np.tile(rng.normal(size=15), volume)
    zero_null = float(np.linalg.norm(zero_gauge@constant))
    zero_flow = primitive_flow(law, zero_phi, identity_links, phases, velocity, alpha)
    if (coulomb_defect > 1e-9 or zero_null > 1e-12
            or not all(np.isfinite(part).all() for part in zero_flow)):
        raise ValueError('RAW singular-Phi or exact Coulomb reduction control failed')
    return dict(
        scope='finite-N global RAW operator theorem with outside-chart native controls; not a fullH quantum trajectory',
        measure='Lebesgue Phi60V x Haar SU4^(3V) x Lebesgue coexact A and three harmonics; finite ALL matter/Root/Ref fibre',
        U1_reduction='exact original div=-Dstar, D=target-source; PiL=-D Green rho; lambdaC=Green div(Y), anchored FAR; +rho lambdaC ONCE',
        harmonic_coordinates='meanA=q, Hharm=UE/V p_q²; theta=Nq,p_theta=p_q/N gives UE/N p_theta²',
        operator='Graw=-i(Yraw dot derivative + div_measure(Yraw)/2)+Mraw',
        matrix_potential='actual Hermitian scalar matter orbitals and rho lambdaC, all heavy/Root/Ref actions; smooth in RAW fields, no assigned body spin',
        global_flow_bounds=dict(
            Phi='||Yphi|| <=(3 vmax/a+alpha_max)||Phi||',
            colour_left_Frobenius_per_link='<=12 vmax/a',
            EM='||Yem||<=6 vmax sqrt(V)/a', Phi_Gronwall='exp((3vmax/a+alpha_max)|t|)',
            compact_colour=True, projected_EM_is_bounded=True,
            no_Phi_Gauss_inverse_or_source_frame_required=True),
        exact_unitary_transport='Uraw(t)psi(X)=J_minus_t(X)^1/2 W_t(X)psi(F_minus_t(X)); W_t is ordered unitary transport of Mraw along the COMPLETE RAW flow',
        selfadjoint_core='Ccompact_smooth(Phi,A) tensor Csmooth(SU4 links) tensor ALL finite matter/Root/Ref; complete smooth flow and smooth Hermitian M preserve it; time convolution gives a generator graph core',
        Gauss_kernel='Coulomb constraint solved exactly; residual SU4 compact gauge averaging PGauss commutes Uraw(t) by RAW equivariance, including Phi=0; restricted Stone generator is selfadjoint with core PGauss Ccompact_smooth',
        regular_horizontal_identity='only on regular inverse-Gram chart, Graw=Ghorizontal+actual matter/Ref compensators modulo sym(lambda Gauss); adjoint divergence contraction vanishes',
        inverse_Gram_not_extrapolated_through_singular_Phi=True,
        Dirichlet_boundary='Graw restricted to a regular tube is NOT claimed selfadjoint; H_D-Omega G is the symmetric Dirichlet weak form, with vanishing test-function boundary term',
        native=dict(N=law.size, Phi_amplitude=1e6, unbounded_EM_norm=float(np.linalg.norm(phases)),
            Phi_growth_ratio=phi_ratio, Phi_growth_upper=growth, Phi_linearity_error=linearity,
            literal_sine_EM_norm=float(np.linalg.norm(flow[2])), EM_norm_upper=em_bound,
            full_SU4_and_U1_RAW_covariance=covariance, anchored_Coulomb_projection_defect=coulomb_defect,
            Phi_zero_full_Gauss_constant_null_error=zero_null,
            Phi_zero_RAW_flow_norms=[float(np.linalg.norm(part)) for part in zero_flow]),
        no_quantum_law_or_microscopic_parameter_debt_removed=True,
        not_claimed=['H_D whole-clock admission','SOURCE-relative scalar curvature/connection bridge',
                     'actual once-prepared quantum spatial 2pi loop','native or legalN spatial closure'])


def native_full_gauss_columns(law: Any, fields: np.ndarray, links: np.ndarray, *,
                              field_tangent: np.ndarray | None = None,
                              link_tangent: np.ndarray | None = None) -> tuple[Any, np.ndarray]:
    """All 15V gauge columns in the raw105/site left-trivialized convention.

    Return (CSR_D, mass105), with mass Phi=1 and colour=1/.6. With BOTH
    matrix tangents supplied return the exact directional derivative D',
    not D plus D'. The fixed mass is unchanged. Link tangents are dU;
    the ambient colour coordinate is dU U†/i. No inverse Gram is used.
    """
    from scipy import sparse
    from unrestricted_field_minimum import coordinates
    derivative = field_tangent is not None
    if derivative != (link_tangent is not None):
        raise ValueError('both field and link tangents are required for Dprime')
    phi = fields if field_tangent is None else field_tangent
    volume = law.volume
    gauge = sparse.lil_matrix((105*volume, 15*volume), dtype=float)
    identity = np.eye(15)
    for x in range(volume):
        gauge[105*x:105*x+60, 15*x:15*x+15] = np.column_stack([
            coordinates(1j*(t@phi[x]-phi[x]@t), law.basis).reshape(60)
            for t in law.generators])
        for j in range(3):
            y = int(law.target[x, j])
            rows = slice(105*x+60+15*j, 105*x+75+15*j)
            u = links[x, j]
            if link_tangent is None:
                transported = np.asarray([u@t@u.conj().T for t in law.generators])
            else:
                du = link_tangent[x, j]
                transported = np.asarray([
                    du@t@u.conj().T+u@t@du.conj().T for t in law.generators])
            adjoint = (2*np.einsum('bkl,alk->ba', transported, law.generators)).real.T
            if not derivative:
                gauge[rows, 15*x:15*x+15] += identity
            gauge[rows, 15*y:15*y+15] -= adjoint
    return gauge.tocsr(), np.r_[np.ones(60), np.full(45, 1/.6)]


def native_joint_noether_vector(law: Any, fields: np.ndarray, links: np.ndarray,
                                phases: np.ndarray, alpha: np.ndarray,
                                velocity: np.ndarray, reference_site: int) -> dict[str, Any]:
    """Literal finite-chart105+EM flow and full Gauss projection, not pair L."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    from unrestricted_field_minimum import coordinates
    volume = law.volume
    phi_flow, colour_flow, em_flow = primitive_flow(
        law, fields, links, phases, velocity, alpha)
    left_colour = -1j*np.einsum('xjab,xjcb->xjac', colour_flow, links.conj())
    raw = np.c_[coordinates(phi_flow, law.basis).reshape(volume, 60),
        (2*np.einsum('xjkl,alk->xja', left_colour, law.generators)).real.reshape(volume, 45)]
    gauge, mass = native_full_gauss_columns(law, fields, links)
    weights = np.tile(mass, volume)
    normal = gauge.T@sparse.diags(weights)@gauge
    rhs = gauge.T@(weights*raw.ravel())
    parameters = np.asarray(spsolve(normal.tocsc(), rhs), dtype=float)
    horizontal = (raw.ravel()-gauge@parameters).reshape(volume, 105)
    orthogonality = float(np.linalg.norm(gauge.T@(weights*horizontal.ravel())))
    if not np.isfinite(parameters).all() or orthogonality > 1e-8*(1+np.linalg.norm(rhs)):
        raise ValueError('native regular-chart full Gauss projection failed')
    matrices = np.einsum('xa,aij->xij', parameters.reshape(volume, 15), law.generators)
    divergence = np.zeros(volume)
    for x in range(volume):
        for j in range(3):
            divergence[law.target[x, j]] += em_flow[x, j]
            divergence[x] -= em_flow[x, j]
    laplacian = np.sum(4*np.sin(np.pi*law.sites/law.size)**2, axis=1).reshape((law.size,)*3)
    rhs_fourier = np.fft.fftn(divergence.reshape((law.size,)*3))
    em_lambda_fourier = np.zeros_like(rhs_fourier)
    np.divide(-rhs_fourier, laplacian, out=em_lambda_fourier, where=laplacian>0)
    em_lambda = np.fft.ifftn(em_lambda_fourier).real.ravel()
    em_lambda -= em_lambda[reference_site]
    em_horizontal = em_flow+em_lambda[law.target]-em_lambda[:, None]
    return dict(raw_vector=raw, horizontal_vector=horizontal,
        gauss_parameters=parameters.reshape(volume, 15), gauss_matrices=matrices,
        derived_Ref_internal=matrices[reference_site], mass_diagonal=mass,
        raw_EM_vector=em_flow, horizontal_EM_vector=em_horizontal,
        lambda_C=em_lambda, harmonic_EM_vector=np.mean(em_flow, axis=0),
        gauss_orthogonality_defect=orthogonality,
        scope='actual regular native full-coordinate symbol; covariant symmetric momentum and matter/Ref actions required for operator',
        no_reference_position_motion=True, literal_sine_clover=True,
        plus_rho_lambda_counted_once=True)


class Source108CarContraction:
    """Same supplier state, all Ref, arbitrary active70 and full harmonic kernel."""
    def __init__(self, view: Any):
        from physical_fullfield_scattering import gaussian_source_embedding
        from prepared_neutral_canonical_source import literal_car_transfers
        self.view = view
        self.supplied = gaussian_source_embedding(view, view.orbitals[:, 0])
        bundle = self.supplied['state_embedding']
        self.initial = bundle['initial_active_state']
        field = np.einsum('a,axbcir->xbcir',
                         self.supplied['raw_coefficients'], bundle['source_packet_fields'])
        if view.charge == -1:
            field = field.swapaxes(-1, -2)
        self.coefficients = np.zeros((27, 9, 9, 108, 4), complex)
        for x in range(27):
            self.coefficients[x, :, :, 4*x:4*x+4] = bundle['profile'][x]*field[x]
        self.creators = literal_car_transfers(view, view.charge)
        self.embedding = np.kron(view.orbitals, np.eye(4))
        self.root_factor = np.linalg.qr(self.initial.reshape(9, -1).T, mode='r').T
        self.kernel = bundle['harmonic_kernel']
        self.complement = bundle['spatial_complement_projector']
        self.spin_projector = bundle['source_spin_projector']
        self.baseline = self.states(self.coefficients)
        self.norm2 = self.inner(self.baseline, self.baseline).real
        if abs(self.norm2-self.supplied['full_raw_creator_norm2']) > 1e-10:
            raise ValueError('same SOURCE108/full-harmonic state reconstruction lost')

    def states(self, coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        projection = (self.embedding.conj().T if self.view.charge == 1
                      else self.embedding.T)
        active_coefficients = np.einsum(
            'ij,oabjr->oabir', projection, coefficients, optimize=True)
        active = np.zeros((27, 9, 6, 2, 224), complex)
        for i, reference in itertools.product(range(8), range(4)):
            moved = np.einsum('oab,bhri->oahri',
                active_coefficients[:, :, :, i, reference], self.initial, optimize=True)
            car = self.creators[i, reference]
            active += np.asarray((car@moved.reshape(-1, 70).T).T).reshape(active.shape)
        outside = np.einsum('oabjr,bt->ojrat',
            coefficients, self.root_factor, optimize=True).reshape(27, 27, 4, 4, 9, 9)
        return active.reshape(27, -1), outside

    def inner(self, left: tuple[np.ndarray, np.ndarray],
              right: tuple[np.ndarray, np.ndarray]) -> complex:
        active = np.einsum('od,op,pd->', left[0].conj(), self.kernel, right[0],
                           optimize=True)
        outside = np.einsum('oxiruv,op,xy,ij,pyjruv->', left[1].conj(),
            self.kernel, self.complement, self.spin_projector, right[1], optimize=True)
        return complex(active+outside)

    def action(self, body: np.ndarray, reference: np.ndarray) -> np.ndarray:
        return np.einsum('ij,oabjr,rs->oabis',
                        body, self.coefficients, reference, optimize=True)


def source108_car_noether_consumer(state_capsule: Path) -> dict[str, Any]:
    """Execute the full-source ordinary-CAR component; not the whole quantum loop."""
    from prepared_neutral_spatial_gauss_well import PreparedSpatialGaussQuantumSlice
    from prepared_neutral_canonical_source import CanonicalPreparedSpatialChargeTransfer
    import time
    start = time.perf_counter()
    model = PreparedSpatialGaussQuantumSlice(state_capsule=state_capsule)
    law = model.law
    alpha = np.ones(law.volume)
    alpha[model.reference_site] = 0
    radius = law.sites.astype(float)-1
    velocity = alpha[:, None]*np.column_stack(
        (-radius[:, 1], radius[:, 0], np.zeros(law.volume)))
    symbol = joint_phi_colour_noether_symbol(model, alpha, velocity)
    gauss = symbol['gauss_matrices']
    reference_generator = gauss[model.reference_site]
    body_generator = scalar_orbital(cast(Any, law),
        np.tile(np.eye(4, dtype=complex), (law.volume, 3, 1, 1)),
        np.zeros((law.volume, 3)), velocity)
    for x in range(law.volume):
        body_generator[4*x:4*x+4, 4*x:4*x+4] += gauss[x]
    if np.linalg.norm(body_generator-body_generator.conj().T) > 1e-10:
        raise ValueError('full108 body Noether component is not Hermitian')
    identity_body, identity_ref = np.eye(108), np.eye(4)
    rows = []
    for charge in (1, -1):
        view = CanonicalPreparedSpatialChargeTransfer(model, charge)
        source = Source108CarContraction(view)
        body = body_generator if charge == 1 else body_generator.T
        reference = reference_generator if charge == 1 else reference_generator.T
        first = source.action(body, identity_ref)-source.action(identity_body, reference)
        if charge == -1:
            first = -first
        second = (source.action(body@body, identity_ref)
            -2*source.action(body, reference)
            +source.action(identity_body, reference@reference))
        first_state, second_state = source.states(first), source.states(second)
        endpoints = []
        for angle in (math.pi/2, 2*math.pi):
            ub = expm(-1j*angle*body_generator)
            ur = expm(-1j*angle*reference_generator)
            transformed = source.states(source.action(
                ub.conj().T if charge == 1 else ub.T,
                ur if charge == 1 else ur.conj()))
            transformed_norm = source.inner(transformed, transformed).real
            overlap = source.inner(source.baseline, transformed)
            omitted_ref = source.states(source.action(
                ub.conj().T if charge == 1 else ub.T, identity_ref))
            omitted_defect = max(0., transformed_norm
                +source.inner(omitted_ref, omitted_ref).real
                -2*source.inner(transformed, omitted_ref).real)
            endpoints.append(dict(angle=angle,
                actual_same_neutral_right_SOURCE_plus=math.sqrt(max(0.,
                    source.norm2+transformed_norm+2*overlap.real)/source.norm2),
                actual_same_neutral_right_SOURCE_minus=math.sqrt(max(0.,
                    source.norm2+transformed_norm-2*overlap.real)/source.norm2),
                omitted_derived_Ref_internal_action_defect=math.sqrt(
                    omitted_defect/source.norm2)))
        rows.append(dict(charge=charge, full_raw_creator_norm2=source.norm2,
            true_CAR_SOURCE_first_norm=math.sqrt(max(0.,
                source.inner(first_state, first_state).real)/source.norm2),
            true_CAR_SOURCE_second_norm=math.sqrt(max(0.,
                source.inner(second_state, second_state).real)/source.norm2),
            endpoint=endpoints, full108_all4Ref_full27harmonic=True,
            arbitrary_active70_not_arbitrary_initial_photon=True,
            ordinary_CAR_not_full_auxiliary_Q=True))
    return dict(plan=PLAN, work_item='soft-spatial-closed-rotation',
        status='actual source-complete joint-Gauss/body CAR component; whole-H loop unpaid',
        native_profile='boundary stress alpha1 except fixedFAR0; not legal radius alias',
        state_capsule_sha256=hashlib.sha256(state_capsule.read_bytes()).hexdigest(),
        xi_sha256=hashlib.sha256(model.xi.tobytes()).hexdigest(),
        operator_signature=model.operator_signature(), native=rows,
        derived_ref_internal_generator=reference_generator,
        full105_all15_mass_Gauss=True, no_reference_position_motion=True,
        whole_physical_G_and_driven_H_not_claimed=True,
        no_global64_filtered_state_inherited=True,
        scientific_seconds=time.perf_counter()-start)


def doubled_geometry_successor(frozen_packet: Path) -> dict[str, Any]:
    """Reprice the exact evolved derivative provider, never rerun its frozen science."""
    data = frozen_packet.read_bytes()
    old = json.loads(data)['packet_family']['records'][-1]
    width, p = old['width'], old['support_p_upper']
    norm_ratio = old['derivative_q0_q6_upper'][0]/(math.sqrt(8)*math.exp(7/3))
    speed_ratio = old['free_group_speed_upper']/p
    rows = []
    for multiplier in (1, 2, 4, 8, 16):
        duration = quintic_clock(2*old['inner_radius']*multiplier)
        inner = old['inner_radius']*multiplier
        available = inner-duration*old['free_group_speed_upper']
        phase_radius = min(old['local_phase_Cauchy_radius'],
                           1/(32*duration*speed_ratio*(p+width)))
        derivatives = []
        for q in range(7):
            bound = 0.
            for a in range(q+1):
                for b in range(q-a+1):
                    c = q-a-b
                    eta = math.sqrt(8)*math.exp(4/3)*(64*max(a, 1)/width)**a
                    projector = math.factorial(b)*(16/.1294)**b
                    phase = math.factorial(c)*math.e/phase_radius**c
                    bound += math.factorial(q)*eta*projector*phase/(
                        math.factorial(a)*math.factorial(b)*math.factorial(c))
            derivatives.append(bound*norm_ratio)
        if multiplier == 1 and max(abs(a/b-1) for a, b in zip(
                derivatives, old['derivative_q0_q6_upper'])) > 1e-12:
            raise ValueError('frozen actual derivative provider reconstruction mismatch')
        tail = math.sqrt(3)*(math.sqrt(3)/available)**6*derivatives[6]
        angular = 2*math.pi*(p*math.sqrt(3)*(math.sqrt(3)/available)**5
                            *derivatives[6]+tail)
        covariance = old['integrated_lattice_upper']*(duration/old['duration'])
        peredge = math.pi**2/2*(128/inner+128/inner**2)
        rows.append(dict(multiplier=multiplier, inner_radius=inner,
            outer_radius=2*inner, duration=duration, phase_radius=phase_radius,
            actual_derivative_q0_q6=derivatives, retained_tail=tail,
            supported_angular_error=angular, integrated_covariance=covariance,
            original_peredge_loop_error=peredge,
            partial_geometric_error_sum=angular+covariance+peredge,
            headroom=1e-4-angular-covariance-peredge))
    return dict(frozen_input_sha256=hashlib.sha256(data).hexdigest(),
        selected=rows[1], family=rows, same_scalar_beta_state_law=True,
        repriced_source_free_word=repriced_source_free_word(F(old['duration'])*2,
                                                           rows[1]['duration']),
        fullfield_error_proved=False, old_geometry_receipt_not_retagged=True)


def run(native_receipt: Path | None = None, native_source: Path | None = None,
        metric_receipt: Path | None = None) -> dict[str, Any]:
    neutral = neutral_descriptor()
    candidate = dict(descriptor=hessian_descriptor(), termwise_graph_scaling=termwise_exponents(),
                     candidate_not_current_join=True, fullQ_branch_boundary_guard_source_costs_paid=False)
    # Capture shared input APIs before science; siblings may later refine their
    # own candidate without silently rebinding this executed receipt.
    api_hashes = {f'{fn.__module__}.{fn.__name__}':hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
                  for fn in (neutral_descriptor, hessian_descriptor, termwise_exponents,
                             selected_band, ImprovedPhiLaw, primitive_flow, scalar_orbital,
                             PrimitiveSector, native_onebody, derived_spins, batch_h, source_seed,
                             anchored_coefficient, spatial_source_identification,
                             coefficients, sqrt_lower, sqrt_upper, RotationBand)}
    band, frame, metadata = selected_band()
    scalar_box = None
    if metric_receipt is not None:
        frozen_box = json.loads(metric_receipt.read_text())['certificate']['continuum_box']
        scalar_box = dict(g0_lower=str(F.from_float(frozen_box['G00'][0])),
                          gs_upper=str(F.from_float(frozen_box['Gjj_each'][1])))
    packet = packet_family(frame, band, scalar_box)
    metric = None if metric_receipt is None else cubic_metric_certificate(metric_receipt, packet, band)
    native_input = None
    if native_receipt is None:
        trajectories = [field_trajectory(frame, band, .001, False),
                        field_trajectory(frame, band, .0001, True)]
    else:
        # Reuse the executed full-turn science, not its failed stdout wrapper.
        # Subsequent tail/serialization edits do not invalidate field physics.
        frozen = json.loads(native_receipt.read_text())
        original = frozen['source_sha256'] == 'fa6625da5cc5d12b46bdb92b59d85735f53e1271eae085d6444a3c0e4b3a61a9'
        prepared = frozen['source_sha256'] == '41d0884f556bd120124c5de4eb54df5bc5641962a3b44873069b7471c59994aa'
        if not (original or prepared):
            raise ValueError('not a source-bound executed full-turn probe')
        if prepared:
            if native_source is None or hashlib.sha256(native_source.read_bytes()).hexdigest() != frozen['source_sha256']:
                raise ValueError('executed onceprepared source capsule required')
            frozen_ast = ast.parse(native_source.read_text())
            current_ast = ast.parse(Path(__file__).read_text())
            class RuntimeIdentityCasts(ast.NodeTransformer):
                def visit_Call(self, node: ast.Call) -> ast.AST:
                    node = cast(ast.Call, self.generic_visit(node))
                    if isinstance(node.func, ast.Name) and node.func.id == 'cast' and len(node.args) == 2:
                        return node.args[1]
                    return node
            frozen_ast = cast(ast.Module, RuntimeIdentityCasts().visit(frozen_ast))
            current_ast = cast(ast.Module, RuntimeIdentityCasts().visit(current_ast))
            for name in ('field_trajectory', 'native_car_endpoint'):
                old = next(node for node in frozen_ast.body if isinstance(node, ast.FunctionDef) and node.name == name)
                current = next(node for node in current_ast.body if isinstance(node, ast.FunctionDef) and node.name == name)
                if ast.dump(old, include_attributes=False) != ast.dump(current, include_attributes=False):
                    raise ValueError(f'changed executed native physics {name}')
        for fn in (selected_band, ImprovedPhiLaw, primitive_flow, anchored_coefficient, spatial_source_identification,
                   *((scalar_orbital, PrimitiveSector, native_onebody, derived_spins) if prepared else ())):
            key = f'{fn.__module__}.{fn.__name__}'
            if frozen['consumed_api_sha256'][key] != hashlib.sha256(inspect.getsource(fn).encode()).hexdigest():
                raise ValueError(f'changed full-turn physics dependency {key}')
        trajectories = frozen['actual_fullturn_native_trajectories']
        native_input = dict(path=str(native_receipt), sha256=hashlib.sha256(native_receipt.read_bytes()).hexdigest(),
                            original_cli_exit=1 if original else 0,
                            science_written_before_stdout_serialization_failure=original,
                            executed_source_capsule=str(native_source) if prepared else None)
    return dict(plan=PLAN, work_item='soft-spatial-closed-rotation',
        status='tested narrowing branch; full-field spatial closure OPEN',
        physical_parameters=dict(N=NSTAR, V=NSTAR**3, UE=str(UE), c=str(C), K=str(C*C/(2*UE)),
                                 reference=(0, NSTAR//2, NSTAR//2), light=metadata['parameters'], heavy=metadata['heavy_parameters'], BETA=BETA),
        packet_family=packet, actual_fullturn_native_trajectories=trajectories,
        coarse_highmode_SOURCE_discriminator=coarse_highmode_discriminator(frame, band),
        coarse_relative_fullfield_requirements=coarse_relative_requirements(packet),
        physical_consumer_error_factors=physical_error_factors(packet),
        cubic_unique_metric_rotation_certificate=metric,
        correlated_Hessian_material_candidate=candidate,
        native_fullturn_input=native_input,
        canonical_neutral_functional=neutral,
        canonical_neutral_identity_sha256=hashlib.sha256(json.dumps(neutral, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        canonical_fullfield_native_state_executed=False,
        native_CAR_endpoint_observed=all('onceprepared_CAR_physical_angular_endpoint' in row for row in trajectories),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        consumed_api_sha256=api_hashes,
        unpaid=dict(fullQ=True, source_first_and_half_domains=True, allstate_quartic=True,
                    guard=True, angular=True, normal=True, fullfield_same_neutral_closed_loop=True),
        no_prior150_or2048_reserve_inherited=True, SC19_receiver_unchanged=True,
        hardCompton_softLorentz_claim=False)


def first_energy_source_clover(root_capsule: Path, prepared_receipt: Path,
                              source_root_binding: Path) -> dict[str, Any]:
    """Actual prepared first-energy inputs; remaining uniform Noether scope explicit."""
    from original_total_angular_transfer import oriented_step
    N, V = NSTAR, NSTAR**3
    radius = 2684354560000
    clock = quintic_clock(radius)
    eta_upper = F(33, 32)
    prepared_bytes = prepared_receipt.read_bytes()
    prepared = json.loads(prepared_bytes)
    initial_state = prepared['initial_energy']
    if (prepared['exit_code'] != 0 or not prepared['loaded_sources_unchanged']
            or initial_state['identity'] != 'PNFF-RAW64-coherent-Ref4-sea-Ref0-seaPlus4-ONE-globalfilter-v2'
            or initial_state['absolute_initial_energy_upper'] !=
            '8V Lambda for actual normalized Xi, each BplusXi/BminusXi, pairedSOURCE and truea† variants'):
        raise ValueError('actual coherent-bank unshifted initial-energy receipt required')
    norm_box = initial_state['norm_inputs']['each_SOURCE_relative']
    source_lower = F(int(norm_box['exact_numerator_hex'], 16),
                     int(norm_box['exact_denominator_hex'], 16))
    if source_lower != F(1, 4):
        raise ValueError('prepared SOURCE norm is relative to normalized actual Xi, not unfiltered seed')
    source_root_bytes = source_root_binding.read_bytes()
    source_root = json.loads(source_root_bytes)['results']
    if (source_root['root_sha256'] !=
            '8e107f1fecb8483200c1520b8afd0032a653c24bae5b81d1ecad36a81d0cdd9e'
            or source_root['both_original_map_cycle_commutator_max'] != 0
            or source_root['both_original_map_code_leakage_max'] != 0
            or source_root['beta_discarded']):
        raise ValueError('literal original SOURCE/Root106 centrality with retained beta required')
    root_bytes = root_capsule.read_bytes()
    root_hash = hashlib.sha256(root_bytes).hexdigest()
    if root_hash != '8f75458bb21bb86b77a2136ae55db8bca460dc336b8b4dd602e5268ca822c870':
        raise ValueError('exact global Root106 successor proof required')
    root = json.loads(root_bytes)['global_exterior']
    bn = {key: F(root['actual_BN_coefficients_per_V'][key]) for key in ('a','b','c')}
    # Actual Young residue is a*Lambda+b+c/Lambda, not a cancelled e_global.
    onset = 2**16
    if bn['a']+bn['b']/onset+bn['c']/onset**2 > 2**12:
        raise ValueError('actual Young residue exceeds the global energy cap')
    shift_coefficient = F(4)
    # exp(2)<8 and quintic totalvariation(Omega)/Omega_max=2.
    work_coefficient = 8*(8+2**12+2*shift_coefficient+1)
    if work_coefficient > 2**16:
        raise ValueError('global work-energy majorant fails')
    fourth_coefficient = 256*F(2**16)  # retained Lambda*kappa/512, kappa=2
    second_coefficient = sqrt_upper(fourth_coefficient)
    root_amplitude = sqrt_upper(F(2*2**16, 1)/F(132,5))
    if second_coefficient > 2**16 or root_amplitude > 2**7:
        raise ValueError('global quartic/Root-energy bridge fails')
    root_endpoint_prefactor = 2*16*sqrt_upper(F(V))/source_lower*2**7*(1+F(clock,16))
    root_endpoint_log = ceil_log2(root_endpoint_prefactor)-2**312
    # Full Pauli coefficient counting, not ordinary-CAR anticommutation:
    # six length<=2 paths, four Phi matrices and four body/reference entries.
    # With <sum r_x^2><=2^16 V, their current norm is <=2^22 V.
    # The same coefficient covers the scalar-orbital contribution at Omega R<=c/4.
    # HC is bounded by UE*N^2*(V+2); its commutator is bounded by twice its
    # norm times the light/orbital norm. The displayed common polynomial is
    # larger than both estimates for N>=3.
    current = F(2**24*N*N*V*V)
    coulomb_exchange = current
    initial = 4*C*V+2*UE*N*N*(V+2)+3*UE/N
    energy = 3*(sqrt_upper(initial+UE*coulomb_exchange*clock)
                +sqrt_upper(UE)*current*clock)**2
    # Point encoded creator norm<=16; Cauchy on the unwhitened eta sum.
    # ||DG(delta_birth-delta_ref)||<=N; harmonic gradient is included.
    gradient = 16*N*sqrt_upper(F(V))*eta_upper
    # |sinF-F|<=2|F|, clover C^2=2, H_EM<=2K_EM and integral|Omega|<8.
    error = 32*radius*gradient*sqrt_upper(8*UE*energy)/(C*source_lower)
    law = ImprovedPhiLaw(3)
    _, frame, _ = selected_band()
    fields = np.tile(frame[None], (law.volume, 1, 1, 1))
    links = np.tile(np.eye(4, dtype=complex), (law.volume, 3, 1, 1))
    velocity = np.column_stack((-law.sites[:, 1]+1, law.sites[:, 0]-1,
                                 np.zeros(law.volume)))
    native_radius = float(np.max(np.linalg.norm(velocity, axis=1)))
    rng = np.random.default_rng(22207)
    native = []
    for scale in (.05, .8, 8., 80.):
        phases = scale*rng.normal(size=(law.volume, 3))
        linear = np.zeros((law.volume, 3, 3))
        for x in range(law.volume):
            for j in range(3):
                for i in range(j+1, 3):
                    for s in (-1, 1):
                        for t in (-1, 1):
                            y, angle = x, 0.
                            for axis, sign in ((j,s),(i,t),(j,-s),(i,-t)):
                                y, _, delta = oriented_step(cast(Any, law), y, axis, sign, links, phases)
                                angle += delta
                            linear[x,j,i] += s*t*angle/4
                    linear[x,i,j] = -linear[x,j,i]
        linear_y = np.zeros_like(phases)
        curl2 = 0.
        for x in range(law.volume):
            for i in range(3):
                y = law.target[x,i]
                for j in range(3):
                    linear_y[x,i] -= ((velocity[x,j]+velocity[y,j])/2
                                      *(linear[x,j,i]+linear[y,j,i])/2)
                for j in range(i+1,3):
                    f = phases[x,i]+phases[law.target[x,i],j]-phases[law.target[x,j],i]-phases[x,j]
                    curl2 += f*f
        _, _, sine_y = primitive_flow(cast(Any, law), fields, links, phases, velocity, np.zeros(law.volume))
        ratio = float(np.linalg.norm(sine_y-linear_y)**2/(native_radius**2*curl2))
        if ratio > 8+1e-10:
            raise ValueError('literal full-clover first-energy majorant failed')
        native.append(dict(phase_scale=scale, literal_clover_error_square_ratio=ratio, upper=8))
    path_sum = [sum(int(displacement[axis])*float(np.sum(np.abs(scalar)))
                    for displacement, scalar in law.paths if displacement[axis])
                for axis in range(3)]
    if max(path_sum) > 2:
        raise ValueError('actual range-two path coefficients exceed current majorant')
    return dict(plan=PLAN, work_item='soft-spatial-closed-rotation',
        status='actual prepared initial-energy/norm joined; uniform Noether/current bridge remains distinct',
        law05=True, radius=radius, clock=clock,
        exact_current_upper=str(current), actual_path_derivative_sums=path_sum,
        exact_initial_energy_upper=str(initial), exact_driven_energy_upper=str(energy),
        exact_EM_gradient_upper=str(gradient), actual_prepared_fermionic_SOURCE_norm_lower=str(source_lower),
        exact_rotation_clover_error_upper=str(error), error_log2_upper=ceil_log2(error),
        rotation_target='1e-4', target_met_conditionally=error<F(1,10000),
        native_literal_full_clover=native, exact_SOURCE_Pi_cancellation=True,
        pulse_total_angle_upper=8, cubic_photon_moment_not_required_for_this_route=True,
        same_current_Born_threshold_not_paid=True,
        root_proof_input=dict(path=str(root_capsule), sha256=root_hash),
        prepared_initial_input=dict(path=str(prepared_receipt),
            sha256=hashlib.sha256(prepared_bytes).hexdigest(),
            identity=initial_state['identity'], unshifted_H_R=True, C_identity=0,
            e_global_is_filter_argument_only=True,
            actual_relative_SOURCE_norm_lower=str(source_lower),
            ordinary_floor_alias_Eextra_phases_retained=True),
        literal_source_root_binding=dict(path=str(source_root_binding),
            sha256=hashlib.sha256(source_root_bytes).hexdigest(),
            both_maps_commute_cycle_penalties=True, beta_noncommuting_retained=True),
        conditional_global_work=dict(initial_whole_H_upper='8 V Lambda',
            actual_BN_upper='2^12 V Lambda', work_energy_upper='2^16 V Lambda',
            driven_field_fourth_upper=f'{fourth_coefficient} V',
            driven_field_second_upper=f'{second_coefficient} V',
            unchanged_Root_Delta_lower='(132/5) V Lambda^(5/4)',
            root_amplitude_upper='2^7 Lambda^(-1/8)',
            physical_root_endpoint_error_log2_upper=root_endpoint_log,
            same_nonzero_beta_retained=True, P_block_is_proof_only=True,
            full_aux_fermionic_SOURCE_upper='16 sqrt(V)', no_beta_over_gap_inference=True),
        remaining_uniform_operator_input=
            'same full-Gauss generator equals symmetrized autonomous raw Noether modulo constraint',
        no_preparation_or_reference_hardware_changed=True)


def soft_native_restriction(size: int = 3) -> dict[str, Any]:
    """Restrict the unchanged scalar SOFT, never alias it to a native packet.

    The native periodic Fourier projection vanishes exactly: not even p=0
    lies in the selected compact support. Coordinate restriction is a
    different, deliberately lossy operation; its tiny norm is not reset.
    """
    width = 1e-7
    centre = np.array([2e-7, 0., 0.])
    sites = np.indices((size,)*3).reshape(3, -1).T
    displacement = sites-size/2
    def evaluate(order: int) -> np.ndarray:
        nodes, weights = np.polynomial.legendre.leggauss(order)
        r, weights = (nodes+1)/2, weights/2
        bump = np.exp(-1/(1-r*r))
        scalar_norm = math.sqrt(4*math.pi*width**3*float(weights@(r*r*bump*bump)))
        sinc = np.sinc(width*np.linalg.norm(displacement, axis=1)[:, None]*r[None, :]/math.pi)
        integral = sinc@(weights*r*r*bump)
        return ((4*math.pi*width**3/(2*math.pi)**1.5/scalar_norm)
                *integral*np.exp(1j*(displacement@centre)))
    profile, refined = evaluate(20), evaluate(80)
    momenta = 2*math.pi*np.fft.fftfreq(size)
    points = np.stack(np.meshgrid(momenta, momenta, momenta, indexing='ij'), axis=-1).reshape(-1, 3)
    supported = np.linalg.norm(points-centre, axis=1)<width
    if supported.any():
        raise ValueError('native periodic SOFT projection is not the asserted zero control')
    raw_norm = float(np.linalg.norm(profile))
    return dict(plan=PLAN, unchanged_scale=1e-4, momentum_centre=centre,
        width=width, support_p_upper=3e-7, native_size=size,
        scalar_definition='L2(dp)-normalized exp[-1/(1-|p-p0|^2/w^2)] on the ball, zero outside',
        inverse_Fourier_convention='eta(d)=(2pi)^(-3/2) integral phi_hat(p) exp(i p.d) dp',
        origin=size/2, sites=sites, coordinate_restriction=profile,
        coordinate_restriction_norm=raw_norm,
        restriction_refinement_difference=float(np.linalg.norm(profile-refined)),
        scalar_quadrature_orders=[20, 80], original_packet_radial_order=20,
        periodic_native_Fourier_projection_norm=0,
        periodic_zero_reason='all native momenta, including zero, are outside the UNCHANGED support',
        raw_profile_not_renormalized=True,
        missing_physical_spatial_L2_mass_estimate=str(1-F.from_float(raw_norm)**2),
        scope='native coordinate restriction ONLY; not periodization, legal SOFT state or finite-N quadrature proof',
        no_retuned_width_momentum_spin_or_p_fibre_normalization=True)


def native_noether_fourier_terms(law: Any, velocity: np.ndarray,
                                em_legs: np.ndarray, reference_site: int) -> dict[str, Any]:
    """Literal sine-clover and scalar-orbital terms for exact Gaussian weak forms.

    ``em_legs`` maps ALL coexact coordinates and three macro-theta coordinates
    to A. Harmonic mean flow is retained. The anchored Coulomb coefficient is
    attached to rho exactly once; it is not another reference position flow.
    """
    volume, edges = law.volume, 3*law.volume
    if em_legs.shape != (edges, 2*(volume-1)+3):
        raise ValueError('ALL coexact plus three macro-theta legs are required')
    a = float(law.band.a)
    loops: dict[tuple[int, int, int], list[tuple[np.ndarray, float]]] = {}
    for x in range(volume):
        for j, i in itertools.combinations(range(3), 2):
            terms = []
            for s, t in itertools.product((-1, 1), repeat=2):
                y = x
                row = np.zeros(edges, dtype=int)
                for axis, sign in ((j, s), (i, t), (j, -s), (i, -t)):
                    if sign == 1:
                        row[3*y+axis] += 1
                        y = int(law.target[y, axis])
                    else:
                        site = law.sites[y].copy()
                        site[axis] = (site[axis]-1)%law.size
                        y = law.index[tuple(site)]
                        row[3*y+axis] -= 1
                if y != x:
                    raise ValueError('literal Noether plaquette failed to close')
                terms.append((row, s*t/(4*a*a)))
            loops[x, j, i] = terms
    coefficients: dict[tuple[int, ...], np.ndarray] = {}
    for x in range(volume):
        for i in range(3):
            y = int(law.target[x, i])
            for j in range(3):
                if j == i:
                    continue
                sign = 1 if j<i else -1
                pair = (min(j, i), max(j, i))
                v = (velocity[x, j]+velocity[y, j])/2
                for anchor in (x, y):
                    for row, weight in loops[anchor, *pair]:
                        first = np.flatnonzero(row)[0]
                        orientation = 1 if row[first]>0 else -1
                        key = tuple(int(z) for z in orientation*row)
                        coefficient = coefficients.setdefault(key, np.zeros(edges))
                        coefficient[3*x+i] -= a*v*sign*weight*orientation/2
    gradient = np.zeros((edges, volume))
    for x in range(volume):
        for j in range(3):
            gradient[3*x+j, int(law.target[x, j])] += 1
            gradient[3*x+j, x] -= 1
    laplacian = gradient.T@gradient
    values, modes = np.linalg.eigh(laplacian)
    keep = values>1e-10
    green = (modes[:, keep]/values[keep])@modes[:, keep].T
    inverse_legs = np.linalg.pinv(em_legs[:, :-3])
    sine_terms = []
    for row, raw in sorted(coefficients.items()):
        if not np.any(raw):
            continue
        phase = np.asarray(row)@em_legs
        # Original divergence is outgoing-minus-incoming = -D*.
        coulomb = -green@(gradient.T@raw)
        coulomb -= coulomb[reference_site]
        dem = np.r_[inverse_legs@raw, law.size*np.mean(raw.reshape(volume, 3), axis=0)]
        sine_terms.append(dict(edge_phase=np.asarray(row), gaussian_phase=phase,
            raw_EM_coefficient=raw, coordinate_velocity=dem, lambda_C=coulomb))
    orbital_terms = []
    for x in range(volume):
        for j in range(3):
            y = int(law.target[x, j])
            scalar = (velocity[x, j]+velocity[y, j])/(4j*a)
            if scalar:
                orbital_terms.append(dict(source=x, target=y, axis=j,
                    gaussian_phase=em_legs[3*x+j], forward_scalar=scalar,
                    reverse_is_Hermitian_conjugate=True))
    return dict(sine_terms=sine_terms, orbital_terms=orbital_terms,
        all_coordinate_dimension=em_legs.shape[1], harmonic_flow_retained=True,
        weak_principal_form='-i/2 (<V,Y.dW>-<Y.dV,W>), with exact product Gaussian/Haar measure',
        Coulomb_charge_term='+rho.lambda_C ONCE; anchored lambda_C(FAR)=0',
        scalar_orbital_has_no_assigned_intrinsic_spin=True,
        no_reference_position_motion=True, literal_sine_not_linearized=True)


def shared_physical_noether_pullback(model: Any, sample: dict[str, Any],
                                    alpha: np.ndarray, velocity: np.ndarray) -> dict[str, Any]:
    """Pull back RAW all105+EM, not a Phi45 slice or a clipped emitter spin."""
    point = sample['point']
    flow = native_joint_noether_vector(model.law, point['phi'], point['links'],
        sample['a'], alpha, velocity, model.reference_site)
    pulled = np.linalg.solve(point['jacobian'], flow['raw_vector'].ravel())
    n = model.chart.normal_dim
    legs = model.em['legs']
    dem = np.r_[np.linalg.pinv(legs)@flow['horizontal_EM_vector'].ravel(),
                model.law.size*flow['harmonic_EM_vector']]
    body = scalar_orbital(model.law, point['links'], sample['a'], velocity)
    body += np.diag(np.repeat(flow['lambda_C'], 4))
    heavy = scalar_orbital(model.heavy_law, point['links'],
                          np.zeros_like(sample['a']), velocity)
    return dict(normal_velocity=pulled[:n], gauge_velocity=pulled[n:],
        EM_velocity=dem, body=body, reference=np.zeros((4, 4), complex),
        heavy_body=heavy, raw=flow,
        raw_pullback_error=float(np.linalg.norm(point['jacobian']@pulled-flow['raw_vector'].ravel())),
        no_intrinsic_only_substitution=True, Coulomb_count=1)


def shared_physical_noether_weak(model: Any, sample: dict[str, Any],
                                alpha: np.ndarray, velocity: np.ndarray,
                                sector: str) -> np.ndarray:
    """Generic same-basis physical Noether weak matrix for A's function bank."""
    data = shared_physical_noether_pullback(model, sample, alpha, velocity)
    return model.first_order_weak(sample, data['normal_velocity'], data['EM_velocity'],
        data['gauge_velocity'], data['body'], data['reference'], sector=sector,
        heavy_body=data['heavy_body'])


def closed_driven_form_admission() -> dict[str, Any]:
    """Common closed-form/time-flow admission for original H-Omega G_RAW.

    This does not make the compressed first-order Dirichlet G self-adjoint.
    H_D-Omega G is instead defined by its symmetric weak form on H1_0.
    """
    radius, clock = 2684354560000, 1008124871316564
    pi_upper = F('3.14159265358979323846264338327950288419716939937511')
    speed_ratio = 15*pi_upper*radius/(4*clock*C)
    relative = sqrt_upper(F(2))*speed_ratio
    if speed_ratio>F(1, 4) or relative>=F(1, 2):
        raise ValueError('same-current Maxwell Noether form admission failed')
    return dict(plan=PLAN, clock=clock, radius=radius, exact_speed_ratio_upper=str(speed_ratio),
        exact_EM_relative_form_bound_upper=str(relative),
        pulse='theta=2pi(10u^3-15u^4+6u^5); Omega=60pi/T u^2(1-u)^2; endpoints Omega=0',
        common_form_domain='H1(Phi x SU4-links x coexactA x harmonicA) intersect Phi4 and curlA2 forms, tensor ALL finite Root/matter/Ref',
        Phi_form_bound='|Omega<Gphi>| <= epsilon chi||Pphi||^2/2 + Omega^2 Cphi^2 ||Phi||^2/(2 epsilon chi); Phi2 absorbed by ORIGINAL positive Lambda Phi4 (alignment/plaquette alone scale Lambda^(5/4))',
        colour_form_bound='compact SU4 smooth bounded Y gives epsilon ORIGINAL .6chi colour kinetic plus finite Omega^2/chi constant',
        matter_form_bound='actual encoded scalar orbital and rho.lambda_C are finite Hermitian multiplication; at most linear Phi, absorbed by original quartic; beta1/16 stays bounded NONzero',
        EM_form_bound='literal |sinF|<=|F|, cloverC^2<=2: |Omega G_EM| <= sqrt2 speed/c (H_EM+H_C); longitudinal reconstruction and harmonic kinetic retained',
        construction='uniformly lower-bounded common CLOSED forms by KLMN; C1 Omega and uniformly equivalent form norms give the Kato/Kisynski unique unitary time propagator',
        strong_Gauss='original invariant forms restrict to physical Gauss kernel; RAW selfadjoint group/core is separately bound, not inferred from a regular chart',
        Dirichlet_comparison='same H1_0 original D on ALLclassunion; boundary-zero symmetric first-order weak form perturbation defines SA H_D-OmegaG, NOT SA DirGraw itself',
        common_boost='W=exp(i Omega Y0.q/chi), all histories; mean normal metric=I. Retain -iWdaggerWdot=(Omegadot/chi)Y0.q and common -Omega^2||Y0||^2/(2chi)',
        source_relative_cancellation='[chi P^2/2,B]boost includes Omega DB[Y0], cancelling -Omega DB[Y0]. Control residual chiP-OmegaY0 and actual offmean metric/connection, NEVER bare Pi graph',
        no_ground_return_or_heavy_spin_stationarity=True,
        no_quantum_law_or_parameter_debt_removed=True)


def native_noether_fourier_control() -> dict[str, Any]:
    """Actual adverse large-field verification of the exact Gaussian operator terms."""
    from prepared_neutral_ambient_weak_state import coexact_coordinates
    band, frame, _ = selected_band()
    law: Any = ImprovedPhiLaw(3, band)
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    links = np.tile(np.eye(4, dtype=complex), (law.volume, 3, 1, 1))
    xyz = law.sites-law.size/2
    radius = np.linalg.norm(xyz, axis=1)
    u = np.clip((radius-1)/.5, 0, 1)
    alpha = 1-(10*u**3-15*u**4+6*u**5)
    velocity = alpha[:, None]*np.column_stack((-xyz[:, 1], xyz[:, 0], np.zeros(law.volume)))
    em = coexact_coordinates(law, 2.**-20, .125)
    legs = np.column_stack((em['legs'], np.tile(np.eye(3), (law.volume, 1))/law.size))
    reference = law.index[(0, 1, 1)]
    terms = native_noether_fourier_terms(law, velocity, legs, reference)
    rng = np.random.default_rng(222260)
    rows = []
    for scale in (1., 600.):
        z = rng.standard_normal(legs.shape[1])*scale
        phases = (legs@z).reshape(law.volume, 3)
        exact = primitive_flow(law, fields, links, phases, velocity, alpha)[2]
        synthesized = np.zeros(3*law.volume)
        linear = np.zeros_like(synthesized)
        coulomb = np.zeros(law.volume)
        dem = np.zeros(legs.shape[1])
        for term in terms['sine_terms']:
            angle = term['gaussian_phase']@z
            synthesized += term['raw_EM_coefficient']*math.sin(angle)
            linear += term['raw_EM_coefficient']*angle
            coulomb += term['lambda_C']*math.sin(angle)
            dem += term['coordinate_velocity']*math.sin(angle)
        direct = native_joint_noether_vector(law, fields, links, phases, alpha, velocity, reference)
        orbital = np.zeros((4*law.volume, 4*law.volume), complex)
        for term in terms['orbital_terms']:
            x, y = term['source'], term['target']
            edge = term['forward_scalar']*links[x, term['axis']]*np.exp(1j*(term['gaussian_phase']@z))
            orbital[4*x:4*x+4, 4*y:4*y+4] += edge
            orbital[4*y:4*y+4, 4*x:4*x+4] += edge.conj().T
        errors = dict(literal_sine=float(np.linalg.norm(synthesized-exact.ravel())),
            anchored_Coulomb=float(np.linalg.norm(coulomb-direct['lambda_C'])),
            all_coordinate_reconstruction=float(np.linalg.norm(legs@dem-direct['horizontal_EM_vector'].ravel())),
            scalar_orbital=float(np.linalg.norm(orbital-scalar_orbital(law, links, phases, velocity))))
        if max(errors.values())>1e-9:
            raise ValueError('literal Noether Gaussian operator terms disagree with primitive law')
        rows.append(dict(scale=scale, errors=errors,
            wrong_linearized_clover_defect=float(np.linalg.norm(linear-exact.ravel())),
            actual_harmonic_velocity_norm=float(np.linalg.norm(dem[-3:]))))
    if rows[-1]['wrong_linearized_clover_defect']<.05:
        raise ValueError('large-field control failed to reject linearized clover')
    return dict(plan=PLAN, native_size=3, origin=law.size/2, Rin=1, Rout=1.5,
        native_ue=2.**-20, native_c=.125, legal_ue_not_substituted=True,
        legal_c_not_substituted=True, coordinates_match_immutable_A17_native=True,
        alpha=alpha, velocity=velocity, reference_site=reference,
        exact_operator_terms=terms, controls=rows,
        scope='exact original RAW EM sine-clover/scalarorbital coefficients for SAME all55 weak forms, not a quantum-state or closed-loop verdict')


def source_relative_bridge(config_path: Path) -> dict[str, Any]:
    """Bind the legal SOURCE-relative normal, Root, field and harmonic costs."""
    config = json.loads(config_path.read_bytes())
    inputs: dict[str, Any] = {}
    bindings = {}
    for name, item in config['inputs'].items():
        path = Path(item['path'])
        body = path.read_bytes()
        digest = hashlib.sha256(body).hexdigest()
        if digest != item['sha256']:
            raise ValueError(f'frozen bridge input changed: {name}')
        inputs[name] = json.loads(body)
        bindings[name] = dict(path=str(path), sha256=digest)
    geometry = inputs['geometry']['result']['result']['selected']
    first = inputs['first_energy']['result']['result']
    normal = inputs['connection']['data']
    root = inputs['root']
    root_noe = inputs['root_noether']['scientific_result']
    harmonic = inputs['harmonic']['legal']['coherent_harmonic_initial_graphs']
    clock, radius, N, V = 1008124871316564, 2684354560000, NSTAR, NSTAR**3
    if (geometry['duration']!=clock or normal['clock']!=str(clock)
            or not first['law05'] or root_noe['no_beta_over_gap_inference'] is not True
            or not harmonic['no_driven_graph_claim']):
        raise ValueError('bridge input clock/law/Root or initial-graph semantics disagree')
    admission = closed_driven_form_admission()
    energy = F(first['exact_driven_energy_upper'])
    force = F(2**32*V**3, N)
    ptheta = 8+force*(clock+8*radius)
    theta_kinetic = 2*UE/N*(8*clock+force*(F(clock*clock, 2)+8*radius*clock))
    theta_Noether = 128*N*radius/(C*sqrt_lower(F(V)))*sqrt_upper(UE*energy)
    theta_upper = 2+theta_kinetic+theta_Noether
    if theta_upper>=3:
        raise ValueError('SOURCE-relative driven harmonic graph does not close')
    # Only the source commutator is compared, not the bulk light Hamiltonian.
    harmonic_mean_error = F(2**15*(clock+8*radius), N)
    harmonic_kinetic_error = F(2**12*clock, N)*UE*(1+ptheta)
    gauss_spin_error = 2**12*F(root['allcentre_mean']['legal_a_minus_alpha_upper_with_EXACT_gs_box'])
    angular_alias = F(root['allcentre_mean']['angular_integrated_alias_upper'])
    source_log = int(normal['SOURCE_endpoint_Lipschitz_coefficient_log2'])
    connection_log = int(normal['chi_AL_y_Pboost_bridge_coefficient_log2'])
    curvature_log = int(normal['curvature_Haar_vertical_chi_bridge_coefficient_log2'])
    # Two relative adjoints are multiplication operators after the exact
    # neutral RIGHT subtraction. On original D, bound their jets rather than
    # silently charge a bare G^2 SOURCE state or invent a photon j3 gate.
    second_relative_graph_log = (2*source_log+18*int(normal['connection_C_log2_upper'])
        +8*890866+64+3*ceil_log2(F(V))+4*ceil_log2(F(radius)))
    log_lambda = 2**315
    log_clock = ceil_log2(F(1+clock))
    normal_errors = dict(endpoint=source_log-log_lambda//4,
        connection=connection_log+log_clock-log_lambda//8,
        curvature_Haar_Casimir=curvature_log+log_clock-log_lambda//2,
        whole_to_original_D=171348180+log_clock-log_lambda//8,
        actual_mean_light_variation=171348180+log_clock-log_lambda//4)
    if max(normal_errors.values())>=-100:
        raise ValueError('fixed-radius source/connection/normal costs exceed legal budget')
    root_error_log = int(first['conditional_global_work']['physical_root_endpoint_error_log2_upper'])
    if root_error_log>=-100:
        raise ValueError('physical retained-beta Root source cost exceeds the bridge budget')
    clover = F(first['exact_rotation_clover_error_upper'])
    finite_tiny = harmonic_kinetic_error+gauss_spin_error+angular_alias+clover
    # Five normal terms and one Root term each have this honest dyadic cap.
    tiny_cap = finite_tiny+F(6, 2**100)
    total = F.from_float(geometry['partial_geometric_error_sum'])+harmonic_mean_error+tiny_cap
    def exact(value: F) -> dict[str, str]:
        return dict(exact_numerator_hex=hex(value.numerator),
                    exact_denominator_hex=hex(value.denominator))
    return dict(plan=PLAN, work_item='soft-spatial-closed-rotation',
        input_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        frozen_inputs=bindings, clock=clock, closed_driven_form_admission=admission,
        physical_comparison='U_C^Omega(t) B_SOURCE Xi - B_free(t) U_N^Omega(t) Xi, SAME neutral RIGHT and ordinary source/free words; never neutral return',
        common_fast='nonlinear allheavy/allRoot fast state remains common; on proofP encoded EVEN heavy/current commute disjoint SOURCE L/Ref; no tensorI onLight',
        code_boundary=root_noe,
        physical_beta='1/16 nonzero retained; P beta P=0 onlyproof; Q cost CQ(1+T beta)supQ, fullAux CQ16sqrtV',
        Root_projection_equation='compare Ppsi_C with Bmean_PP Ppsi_N; G preservesP, offdiagonalH is beta, hence residuals beta Qpsi_C and Bmean_PP beta Qpsi_N. Use actual BOTH initial8V Lambda/driven Root bounds; do not drop Q beta P in an unprojected Heisenberg commutator',
        root_source_error_log2_upper=root_error_log,
        regular_union='original radius and original D3r/4, scalar cutoff ALLclassunion; no thinGamma/newRootdependent cutoff',
        light_frame='bare constant centre Light/Ref; every admitted centre uses FULL105 mass-horizontal lift A_L(0)=0; physical H/SOURCE/beta conjugated together, not Kato/Fphi basis recapture',
        normal_error_log2_upper=normal_errors,
        second_relative_Noether_graph=dict(coefficient_log2_upper=second_relative_graph_log,
            scope='both actual SOURCE adjoints on ORIGINAL regular D, allRef/DY/Gauss/clover derivatives and exact neutral RIGHT; bounded-state whole-to-D transfer first',
            reason='ad_G B=-i DY B+[M,B], hence ad_G^2 B has no field momentum left; regular mean/chart jets bound the coefficients',
            bare_G2_SOURCE_graph_not_asserted=True, no_cubic_photon_gate=True),
        nonlinear_connection='A_L(y)=O(Cconn^2|y|); chi A_L(y)Pboost via weighted Caccioppoli; heavyKato remainsfast, SOURCE Berry derivative NONzero',
        actual_harmonic_initial_graph=harmonic,
        driven_harmonic=dict(force_upper=exact(force), ptheta_norm_upper=exact(ptheta),
            theta_norm_upper=exact(theta_upper), kinetic_drift_upper=exact(theta_kinetic),
            Noether_harmonic_drift_upper=exact(theta_Noether), theta_proved_less_than=3,
            force_reason='only Light has theta force; Root/beta, heavy neutral, curlEM and rho.lambda_C independenttheta; whole quartic controls force, not frozen lightsea',
            harmonic_energy='UE/N ptheta^2, theta=N meanA, ptheta=sumPi/N',
            full_macro_SOURCE_phase_retained=True, no_unbounded_norm_transfer=True),
        source_only_mean_harmonic_error_upper=str(harmonic_mean_error),
        harmonic_SOURCE_kinetic_error_upper=exact(harmonic_kinetic_error),
        fullGauss_mean_spin_error_upper=str(gauss_spin_error),
        mean_spin_scope='actual frozen heavy means/full4 original g and fixedFAR tail; bind literal SOURCE Noether identification, not an assigned fundamental body spin',
        geometric_error_upper=geometry['partial_geometric_error_sum'],
        total_norm_error_upper=exact(total),
        total_norm_error_upper_float=float(np.nextafter(float(total), math.inf)),
        target=1e-4, target_met=total<F(1, 10000),
        native_N3_SOFT_not_claimed=True,
        conditional_scope='canonical original quantum law and chosen parameters; graph/source intertwiners are bound proof inputs, not native fullFock/512 quantum experiment',
        campaign_open=True, quantum_law_and_parameter_debt_remains=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-receipt', type=Path)
    parser.add_argument('--native-source', type=Path)
    parser.add_argument('--metric-receipt', type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--source108-car', action='store_true')
    modes.add_argument('--geometry-successor-input', type=Path)
    modes.add_argument('--first-energy-clover', action='store_true')
    modes.add_argument('--global-raw-noether-core', action='store_true')
    modes.add_argument('--soft-native-restriction', action='store_true')
    modes.add_argument('--native-noether-fourier', action='store_true')
    modes.add_argument('--source-relative-bridge', type=Path)
    parser.add_argument('--state-capsule', type=Path)
    parser.add_argument('--root-proof-capsule', type=Path)
    parser.add_argument('--prepared-receipt', type=Path)
    parser.add_argument('--source-root-binding', type=Path)
    args = parser.parse_args()
    if args.source108_car:
        if args.state_capsule is None:
            parser.error('--source108-car requires the exact --state-capsule')
        receipt = source108_car_noether_consumer(args.state_capsule)
    elif args.global_raw_noether_core:
        receipt = global_raw_noether_core()
    elif args.soft_native_restriction:
        receipt = soft_native_restriction()
    elif args.native_noether_fourier:
        receipt = native_noether_fourier_control()
    elif args.source_relative_bridge is not None:
        receipt = source_relative_bridge(args.source_relative_bridge)
    elif args.geometry_successor_input is not None:
        receipt = doubled_geometry_successor(args.geometry_successor_input)
    elif args.first_energy_clover:
        if (args.root_proof_capsule is None or args.prepared_receipt is None
                or args.source_root_binding is None):
            parser.error('--first-energy-clover requires --root-proof-capsule, '
                         '--prepared-receipt and --source-root-binding')
        receipt = first_energy_source_clover(args.root_proof_capsule,
                                            args.prepared_receipt, args.source_root_binding)
    else:
        receipt = run(args.native_receipt, args.native_source, args.metric_receipt)
    args.output.write_text(json.dumps(receipt, default=encode, indent=2)+'\n')
    if (args.source108_car or args.geometry_successor_input is not None
            or args.first_energy_clover or args.global_raw_noether_core
            or args.soft_native_restriction or args.native_noether_fourier
            or args.source_relative_bridge is not None):
        print(json.dumps(dict(output=str(args.output), result=receipt), default=encode))
    else:
        print(json.dumps(dict(output=str(args.output), status=receipt['status'],
            field_closed=[r['physical_path_closed'] for r in receipt['actual_fullturn_native_trajectories']],
            integrated_lattice=[r['integrated_lattice_upper'] for r in receipt['packet_family']['records']],
            tail_useful=[r['tail_certificate_useful'] for r in receipt['packet_family']['records']]), default=encode))


if __name__ == '__main__':
    main()
