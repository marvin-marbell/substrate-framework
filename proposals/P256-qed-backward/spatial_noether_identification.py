#!/usr/bin/env python3
"""Physical spatial/current little-group identification of the intrinsic Phi source."""
from __future__ import annotations
import argparse
import hashlib
import inspect
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, cast
import ast
from fractions import Fraction
if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'
import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
from bulk_quantum_selection import BulkQuantumLaw
from cartesian_noether_group import EPS
from improved_phi_band import C, NSTAR, ImprovedPhiLaw, physical_h, selected_band
from moving_physical_noether import physical_generators, source19_boundary_word, source19_nested_casimir
from noncompact_screened_gauss_dressing import frame_chart, centre_scalar, physical_raw_noether_geometry_majorant
from original_total_angular_transfer import primitive_flow, scalar_orbital, covariant_field_difference
from physical_carrier_transport import (PrimitiveSector, SITES, native_onebody,
    native_source_coefficient, native_source_chart, verify_cartesian_successor)
from improved_physical_compton import decode
from rotation_band import RotationBand
from sea_selection import SELECTED

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-synthesis-continuation-20261006'
MASS_LOWER = .129417988064
LOWK = .05


def check(value: float, label: str, tolerance: float = 3e-7) -> float:
    if not math.isfinite(value) or value > tolerance:
        raise AssertionError(f'{label}: {value} > {tolerance}')
    return float(value)


def adverse(value: float, label: str, tolerance: float = 3e-7) -> dict[str, Any]:
    if not math.isfinite(value) or value <= tolerance:
        raise AssertionError(f'undiscriminating adverse {label}: {value}')
    return dict(rejected=True, defect=float(value), tolerance=tolerance)


def collar_polynomial(radius: np.ndarray, inner: float = 48., outer: float = 64.) -> np.ndarray:
    """C6 supported weak rotation, not a rigid torus rotation."""
    u = np.clip((radius-inner)/(outer-inner), 0., 1.)
    primitive = sum((-1)**k*math.comb(6, k)*u**(7+k)/(7+k) for k in range(7))
    return np.where(radius <= inner, 1., np.where(radius >= outer, 0., 1.-12012*primitive))


def spatial_flow(law: Any, fields: np.ndarray, links: np.ndarray, potential: np.ndarray,
                 axis: int = 2, centre: np.ndarray | None = None,
                 inner: float = 48., outer: float = 64.) -> dict[str, Any]:
    """Literal covariant physical coordinate/vector flow, including real sine EM.

    The canonical EM Lie form adds -d(xi.A); its matter Gauss boundary is
    -Lambda*rho. BOTH pieces are retained, not an erased neutral EM image.
    """
    centre = np.zeros(3) if centre is None else np.asarray(centre)
    r = (law.sites-centre+law.size/2) % law.size-law.size/2
    weight = collar_polynomial(np.linalg.norm(r, axis=1)*law.band.a, inner, outer)
    velocity = np.cross(np.eye(3)[axis], r*law.band.a)*weight[:, None]
    orbital = -np.einsum('xj,xjmab->xmab', velocity,
                         covariant_field_difference(cast(BulkQuantumLaw, law), fields, links))
    vector = np.zeros_like(fields)
    for j, k in itertools.product(range(3), repeat=2):
        vector[:, 1+j] -= EPS[axis, j, k]*weight[:, None, None]*fields[:, 1+k]
    # Colour and EM curvature contractions do not depend on intrinsic axis.
    _, du, da = primitive_flow(cast(BulkQuantumLaw, law), fields, links, potential,
                              velocity, np.zeros(law.volume))
    lam = np.einsum('xj,xj->x', velocity, potential)
    canonical_da = da+lam[:, None]-lam[law.target]
    return dict(field=orbital+vector, orbital_field=orbital, vector_field=vector,
                links=du, sine_EM=da, canonical_EM=canonical_da, EM_Gauss_parameter=lam,
                velocity=velocity, cutoff=weight, radius=outer,
                fixed_reference_coordinate=(0, NSTAR//2, NSTAR//2), reference_cutoff=0.)


def derived_spins(law: Any, fields: np.ndarray) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Gauss-frame reduction AFTER physical vector/coordinate flow is defined."""
    axes, guards = [], []
    for field in fields:
        frame, guard = frame_chart(field)
        pole = frame@np.asarray(SELECTED)@frame.conj().T
        axes.append(physical_generators(cast(Any, law), pole)); guards.append(guard)
    return np.asarray(axes), guards


def anchored_coefficient(law: Any, fields: np.ndarray, links: np.ndarray, potential: np.ndarray,
                         profile: np.ndarray, beta: np.ndarray, reference_phi: np.ndarray,
                         anchor_link: np.ndarray | None = None, anchor_phase: float = 0.) -> np.ndarray:
    """Same birth source, with an independent FAR reference and continued Wilson word."""
    origin, corner, bulk_third = [law.index[tuple(site)] for site in SITES]
    f0 = frame_chart(fields[origin])[0]
    fr = frame_chart(reference_phi)[0]
    anchor_link = np.eye(4, dtype=complex) if anchor_link is None else anchor_link
    path = links[origin, 0]@links[corner, 1]
    total_path = path@anchor_link
    scalar = centre_scalar(f0.conj().T@total_path@fr, 18., law.generators)[0]
    labels = np.asarray([np.eye(4), *(SELECTED[j]@SELECTED[k]
                         for j, k in ((2, 3), (3, 1), (1, 2)))])/2
    mark = scalar*np.einsum('a,aij->ij', beta, f0@labels@fr.conj().T)
    transport = (np.eye(4), links[origin, 0], path)
    phase = (0., potential[origin, 0], potential[origin, 0]+potential[corner, 1])
    return np.concatenate([profile[j]*np.exp(1j*(phase[-1]+anchor_phase-phase[j]))*
                           u.conj().T@mark for j, u in enumerate(transport)])/math.sqrt(2)


def coefficient_jet(law: Any, fields: np.ndarray, links: np.ndarray, potential: np.ndarray,
                    profile: np.ndarray, beta: np.ndarray, reference_phi: np.ndarray,
                    df: np.ndarray, du: np.ndarray, da: np.ndarray,
                    anchor_du: np.ndarray | None = None, anchor_da: float = 0., step: float = 2e-5) -> np.ndarray:
    anchor_du = np.zeros((4, 4), complex) if anchor_du is None else anchor_du
    values = [anchored_coefficient(law, fields+sign*step*df, links+sign*step*du,
                 potential+sign*step*da, profile, beta, reference_phi,
                 np.eye(4)+sign*step*anchor_du, sign*step*anchor_da) for sign in (1, -1)]
    return (values[0]-values[1])/(2*step)


def gauss_family(law: Any, fields: np.ndarray, links: np.ndarray, axes: np.ndarray,
                 angle: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Complete onsite-tree SU4 action INCLUDING anchored Wilson continuation."""
    gauge = np.asarray([expm(1j*angle*s) for s in axes])
    phi = gauge[:, None]@fields@gauge[:, None].conj().transpose(0, 1, 3, 2)
    u = gauge[:, None]@links@gauge[law.target].conj().transpose(0, 1, 3, 2)
    third = law.index[tuple(SITES[2])]
    return phi, u, gauge[third]


def spatial_source_identification(law: Any, fields: np.ndarray, links: np.ndarray,
                                  potential: np.ndarray, profile: np.ndarray, beta: np.ndarray,
                                  reference_phi: np.ndarray, axis: int = 2) -> dict[str, Any]:
    physical = spatial_flow(law, fields, links, potential, axis)
    axes, guards = derived_spins(law, fields)
    lam = physical['cutoff'][:, None, None]*axes[:, axis]
    gauge_phi = 1j*(lam[:, None]@fields-fields@lam[:, None])
    gauge_links = 1j*(lam[:, None]@links-links@lam[law.target])
    third = law.index[tuple(SITES[2])]
    anchor_gauge = 1j*lam[third]
    zero_phi, zero_u, zero_a = np.zeros_like(fields), np.zeros_like(links), np.zeros_like(potential)
    coefficient = anchored_coefficient(law, fields, links, potential, profile, beta, reference_phi)
    chosen = [law.index[tuple(site)] for site in SITES]
    modes = np.concatenate([np.arange(4*x, 4*x+4) for x in chosen])
    orbital = scalar_orbital(cast(BulkQuantumLaw, law), links, potential, physical['velocity'])[np.ix_(modes, modes)]
    em_parameter = np.repeat(physical['EM_Gauss_parameter'][chosen], 4)
    def jet(df: np.ndarray, du: np.ndarray, da: np.ndarray,
            anchor_du: np.ndarray | None = None, anchor_da: float = 0.) -> np.ndarray:
        return coefficient_jet(law, fields, links, potential, profile, beta, reference_phi,
                               df, du, da, anchor_du, anchor_da)
    total = jet(physical['field'], physical['links'], physical['canonical_EM'],
                anchor_da=float(physical['EM_Gauss_parameter'][third]))
    pieces = dict(orbital_field=jet(physical['orbital_field'], zero_u, zero_a),
                  Phi_horizontal=jet(physical['vector_field']-gauge_phi, zero_u, zero_a),
                  colour_link_boundary=jet(zero_phi, physical['links']-gauge_links, zero_a, -anchor_gauge),
                  actual_EM=jet(zero_phi, zero_u, physical['canonical_EM'],
                                anchor_da=float(physical['EM_Gauss_parameter'][third])))
    gauge_jet = jet(gauge_phi, gauge_links, zero_a, anchor_gauge)
    spin_coefficient = np.concatenate([lam[x]@coefficient[4*j:4*j+4] for j, x in enumerate(chosen)])
    first_error = check(float(np.linalg.norm(-1j*(total-sum(pieces.values()))-spin_coefficient)),
                        'physical Noether reduction to actual intrinsic source spin')
    gauss_error = check(float(np.linalg.norm(-1j*gauge_jet-spin_coefficient)),
                        'full15 anchored reference Gauss/source differential Ward')
    eta = source19_boundary_word()
    return dict(physical=physical, spins=axes, lambda_colour=lam, guards=guards,
                coefficient=coefficient, total_jet=total, pieces=pieces,
                spin_coefficient=spin_coefficient, orbital=orbital, EM_parameter=em_parameter,
                first_error=first_error, fullGauss_jet_error=gauss_error,
                horizontal_Phi_norm=float(np.linalg.norm(physical['vector_field']-gauge_phi)),
                retained_colour_link_boundary_norm=float(np.linalg.norm(physical['links']-gauge_links)),
                retained_anchor_flux_norm=float(np.linalg.norm(anchor_gauge)),
                retained_real_sine_EM_norm=float(np.linalg.norm(physical['sine_EM'])),
                eta18_curvature=eta['centre_second_curvature'])


def charged_band_h(frame: np.ndarray, band: RotationBand, p: np.ndarray,
                   branch: int = 1) -> np.ndarray:
    """Primitive particle/hole dual, not an independently assigned Dirac law."""
    if branch == 1:
        return physical_h(p, frame, band)
    if branch == -1:
        return -physical_h(-np.asarray(p), frame, band).T
    raise ValueError('Physical charge branch must be +/-1')


def physical_rest_spin(frame: np.ndarray, band: RotationBand, branch: int = 1) -> dict[str, Any]:
    """Physical magnetic/current spin, independently of any gamma-spin assignment."""
    # Establish actual physical Phi-vector tangents before body/Gauss reduction.
    flow = np.asarray([-np.einsum('jk,kab->jab', EPS[i], frame[1:]) for i in range(3)])
    law = ImprovedPhiLaw(3, band)
    spins = physical_generators(law, frame)
    errors = [check(float(np.linalg.norm(1j*(spins[i]@frame[1:]-frame[1:]@spins[i])-flow[i])),
                    'physical vector transformation BEFORE SU4 spin lock') for i in range(3)]
    if branch == -1:
        spins = -spins.transpose(0, 2, 1)
    h0 = charged_band_h(frame, band, np.zeros(3), branch)
    values, vectors = np.linalg.eigh(h0)
    positive = vectors[:, values > 0]
    negative_projector = vectors[:, values < 0]@vectors[:, values < 0].conj().T
    if positive.shape != (4, 2):
        raise ValueError('actual physical rest cluster must have two modes')
    _, basis = np.linalg.eigh(positive.conj().T@spins[2]@positive)
    positive = positive@basis
    velocities = band.v*frame[1:]
    if branch == -1:
        velocities = velocities.transpose(0, 2, 1)
    speed = math.sqrt(float(np.trace(velocities[0]@velocities[0]).real/4))
    currents = []
    for j, k in ((1, 2), (2, 0), (0, 1)):
        current = -.25j*(velocities[j]@negative_projector@velocities[k]-
                         velocities[k]@negative_projector@velocities[j])/speed**2
        currents.append(positive.conj().T@current@positive)
    current_spin: np.ndarray = np.asarray(currents)
    intrinsic = np.asarray([positive.conj().T@s@positive for s in spins])
    mass = float(np.min(values[values > 0]))
    negative = vectors[:, values < 0]
    resolvent = (negative/(mass-values[values < 0]))@negative.conj().T
    magnetic: np.ndarray = np.asarray([-.5j*positive.conj().T@(velocities[j]@resolvent@velocities[k]-
                            velocities[k]@resolvent@velocities[j])@positive
                           for j, k in ((1, 2), (2, 0), (0, 1))])
    g_values = [float((2*mass/speed**2*np.trace(mu@s)/np.trace(s@s)).real)
                for mu, s in zip(magnetic, current_spin)]
    check(max(abs(g-2) for g in g_values), 'physical spectral-current magnetic moment g2')
    current_error = check(float(np.linalg.norm(current_spin-intrinsic)),
                          'physical magnetic/current spin equals Phi Noether/rest clamp')
    casimir = sum(s@s for s in current_spin)
    check(float(np.linalg.norm(casimir-.75*np.eye(2))), 'physical REST current Pauli-Lubanski Casimir')
    # W^mu=(0,m*S) at REST, never arbitrary-packet total J.
    return dict(branch=branch, spins=spins, rest_embedding=positive, current_spin=current_spin,
                vector_before_Gauss_error=max(errors), current_identity_error=current_error,
                rest_Casimir_error=float(np.linalg.norm(casimir-.75*np.eye(2))),
                Pauli_Lubanski_W2_over_minus_m2=casimir, mass=mass, speed=speed,
                magnetic_moment_per_charge=magnetic,
                g=g_values, velocity_vertices=velocities,
                photon_c_exact=f'{C.numerator}/{C.denominator}',
                executed_cone_error=abs(speed-float(C)))


def positive_embedding(frame: np.ndarray, band: RotationBand, p: np.ndarray,
                       rest: np.ndarray, branch: int = 1) -> np.ndarray:
    values, vectors = np.linalg.eigh(charged_band_h(frame, band, p, branch))
    occupied = vectors[:, values > 0]
    if occupied.shape != (4, 2) or np.min(abs(values)) < MASS_LOWER:
        raise ValueError('actual positive-energy cluster/gap guard failed')
    projected = occupied@(occupied.conj().T@rest)
    gram, modes = np.linalg.eigh(projected.conj().T@projected)
    if min(gram) <= .5:
        raise ValueError('anchored positive-energy chart overlap failed')
    return projected@((modes/np.sqrt(gram))@modes.conj().T)


def lowk_little_group(frame: np.ndarray, band: RotationBand, momentum: np.ndarray,
                      rotation: np.ndarray, branch: int = 1) -> dict[str, Any]:
    p = np.asarray(momentum); q = rotation@p
    if max(np.linalg.norm(p), np.linalg.norm(q)) > LOWK+1e-14:
        raise ValueError('physical coarse spatial SO3 only supported at |p|<=.05')
    rest = physical_rest_spin(frame, band, branch)
    vector = Rotation.from_matrix(rotation).as_rotvec()
    d = expm(-1j*np.einsum('a,aij->ij', vector, rest['spins']))
    h, hq = charged_band_h(frame, band, p, branch), charged_band_h(frame, band, q, branch)
    # |sin p-p|<=|p|^3/6; scalar mass anisotropy is bounded by direct derivatives
    # of the literal one/two-link polynomial, not silently set isotropic.
    radius = float(np.linalg.norm(p))
    vector_debt = 2*band.v*max(np.linalg.norm(f, 2) for f in frame[1:])*radius**3/6
    mass_debt = abs(band.mass(q)-band.mass(p))*float(np.linalg.norm(frame[0], 2))
    covariance_bound = vector_debt+mass_debt+64*np.finfo(float).eps
    covariance = float(np.linalg.norm(hq-d@h@d.conj().T, 2))
    check(covariance, 'actual improved spatial band covariance debt', covariance_bound)
    up = positive_embedding(frame, band, p, rest['rest_embedding'], branch)
    uq = positive_embedding(frame, band, q, rest['rest_embedding'], branch)
    wigner = uq.conj().T@d@up
    canonical_rest = expm(-1j*np.einsum('a,aij->ij', vector, rest['current_spin']))
    leakage = float(np.linalg.norm((np.eye(4)-uq@uq.conj().T)@d@up, 2))
    wigner_error = float(np.linalg.norm(wigner-canonical_rest, 2))
    projector_bound = 4*covariance_bound/MASS_LOWER
    check(max(leakage, wigner_error), 'actual lowk physical spinor/Wigner chart', projector_bound)
    return dict(branch=branch, momentum=p, rotated_momentum=q, physical_spinor=d, Wigner=wigner,
                canonical_rest_spinor=canonical_rest, covariance_error=covariance,
                covariance_upper=covariance_bound, positive_cluster_leakage=leakage,
                physical_spinor_error=wigner_error, projector_upper=projector_bound,
                physical_coordinate_action='D(R)psi(R^-1p); -i p cross d_p PLUS current-derived S',
                exact_torus_SO3_claim=False, lowk_cutoff=LOWK,
                noGauss_defect=float(np.linalg.norm(hq-h, 2)))


def aligned_neutral(sector: PrimitiveSector, heavy: ImprovedPhiLaw, fields: np.ndarray,
                    links: np.ndarray, origin: np.ndarray, branch: int) -> np.ndarray:
    _, orbital = np.linalg.eigh(native_onebody(heavy, fields, links, np.zeros((heavy.volume, 3))))
    state = sector.neutral(orbital[:, :6], branch)
    overlap = np.vdot(origin, state)
    if abs(overlap) < .5:
        raise ValueError('same neutral occupied cluster lost anchored overlap')
    return state*np.exp(-1j*np.angle(overlap))


def complex_array(payload: Any) -> np.ndarray:
    if isinstance(payload, dict) and set(payload) == {'real', 'imag'}:
        return np.asarray(payload['real'])+1j*np.asarray(payload['imag'])
    return np.asarray(payload)


def frozen_intrinsic_source_word(receipt: dict[str, Any], source_frame: np.ndarray,
                                 reference_frame: np.ndarray, matrix: np.ndarray,
                                 axis: int, branch: int) -> np.ndarray:
    """Use frozen115 Ki on the SAME full16 seed; never rerun its supplier."""
    frozen_frame = frame_chart(complex_array(receipt['physical_frame']))[0]
    stored = next(row for row in receipt['source_groups'] if row['branch'] == branch)
    generator = complex_array(stored['K'])[axis]
    basis_map = reference_frame@frozen_frame.conj().T
    bridge = reference_frame@source_frame.conj().T
    seed = bridge@matrix if branch == 1 else matrix.conj().T@bridge.conj().T
    old_seed = basis_map.conj().T@seed@basis_map
    word = basis_map@(generator@old_seed.reshape(16)).reshape(4, 4)@basis_map.conj().T
    return bridge.conj().T@word if branch == 1 else word@bridge


def native_spatial_identification(chart: dict[str, Any], frame: np.ndarray,
                                  band: RotationBand, branch: int,
                                  intrinsic_receipt: dict[str, Any]) -> dict[str, Any]:
    """Consume SAME shared quantum chart/source/current; no reprepare or renorm."""
    sector, law = chart['sector'], chart['law']
    profile, beta = chart['profile'], chart['raw_quartet']
    count = chart['node_count']
    heavy = ImprovedPhiLaw(3, RotationBand(a=1., m=.2, r=.12, v=.3, improved=False))
    fields_list, references = chart['field_snapshots'], chart['independent_reference_phi']
    max_first = max_gauss = max_second = max_leibniz = max_intrinsic = max_casimir = 0.
    orbital_images, EM_images, records, orbit_ops = [], [], [], []
    current_derivatives = []
    total_ref = sector.nref-(4 if branch == 1 else 0)
    charge = sector.density.sum(axis=1)+total_ref
    check(float(np.max(abs(charge))), 'actual anchored reference scalar charge compensation')
    ref_image = float(np.linalg.norm(sector.nref*chart['node_neutrals'][0]))
    for node, ((fields, links, potential), reference_phi, coefficient, omega) in enumerate(zip(
            fields_list, references, chart['coefficient_snapshots'], chart['node_neutrals'])):
        check(float(np.linalg.norm(anchored_coefficient(law, fields, links, potential, profile, beta,
                            reference_phi)-coefficient)), 'SAME actual source at fixed independent Ref birth')
        node_records = []
        literal_axis_words = []
        reference_frame = frame_chart(reference_phi)[0]
        for axis in range(3):
            row = spatial_source_identification(law, fields, links, potential, profile, beta, reference_phi, axis)
            lam = row['lambda_colour']; selected = [law.index[tuple(site)] for site in SITES]
            mode_spin = np.zeros((16, 16), complex)
            for j, x in enumerate(selected):
                mode_spin[4*j:4*j+4, 4*j:4*j+4] = lam[x]
                source_frame = frame_chart(fields[x])[0]
                intrinsic_word = frozen_intrinsic_source_word(intrinsic_receipt, source_frame,
                                      reference_frame, coefficient[4*j:4*j+4], axis, branch)
                physical_word = (lam[x]@coefficient[4*j:4*j+4] if branch == 1
                                 else -coefficient[4*j:4*j+4].conj().T@lam[x])
                max_intrinsic = max(max_intrinsic, check(float(np.linalg.norm(intrinsic_word-physical_word)),
                                     'physical spatial/current rest SOURCE equals FROZEN115 K full16 seed'))
            spin = sector.lift(mode_spin)
            l = np.zeros((16, 16), complex); l[:12, :12] = row['orbital']
            orbital = sector.lift(l)
            em = sector.lift(np.diag(np.r_[-row['EM_parameter'], np.zeros(4)]))
            # Vacuum density offsets are part of the actual scalar/EM boundary.
            em += sparse.eye(sector.dimension)*float(2*np.sum(row['EM_parameter'][::4]))
            B = sector.source(coefficient, branch)
            reference_mode = np.zeros((16, 16), complex)
            reference_mode[12:, 12:] = row['spins'][selected[2], axis]
            reference_spin = sector.lift(reference_mode)
            target = branch*sector.source(row['spin_coefficient'], branch)
            first = -1j*sector.source(row['total_jet'], branch)+(orbital+em)@B-B@(orbital+em)
            subtracted = (-1j*sector.source(sum(row['pieces'].values()), branch)+orbital@B-B@orbital+
                          em@B-B@em)
            err = check(float(np.linalg.norm((first-subtracted-target)@omega)),
                        'physical relative SOURCE equals115K on SAME neutral', 3e-6)
            # Nonzero neutral is measured, never replaced by Jn Omega=0.
            orbital_images.append(float(np.linalg.norm(orbital@omega)))
            EM_images.append(float(np.linalg.norm(em@omega)))
            step = .002
            state_values, neutral_word_values = [], []
            for sign in (1, -1):
                pf, pu, anchor = gauss_family(law, fields, links, lam, sign*step)
                cm = anchored_coefficient(law, pf, pu, potential, profile, beta, reference_phi, anchor)
                op = sector.source(cm, branch)
                moved = sparse.linalg.expm_multiply(1j*sign*step*spin, omega,
                             traceA=1j*sign*step*complex(spin.diagonal().sum()))
                state_values.append(op@moved)
                neutral_word_values.append(op@(spin@moved))
            second_state = -(state_values[0]-2*(B@omega)+state_values[1])/step**2
            derivative_cross = -1j*(neutral_word_values[0]-neutral_word_values[1])/(2*step)
            literal_second = second_state-2*derivative_cross+B@(spin@(spin@omega))
            literal_axis_words.append(literal_second)
            second_target = .25*(B@omega)
            second_error = check(float(np.linalg.norm(literal_second-second_target)),
                                 'literal physical Gauss-rest second SOURCE word with neutral derivatives', 8e-6)
            # Physical functional neutral response, not just assigned spin matrices.
            eps = 2e-5; flow = row['physical']
            moved_neutrals, moved_states, moved_currents = [], [], []
            region = np.zeros(law.volume); region[selected[0]] = 1
            direction = region[:, None]-region[law.target]
            for sign in (1, -1):
                pf = fields+sign*eps*flow['field']; pu = links+sign*eps*flow['links']
                pa = potential+sign*eps*flow['canonical_EM']
                neutral = aligned_neutral(sector, heavy, pf, pu, omega, branch)
                cm = anchored_coefficient(law, pf, pu, pa, profile, beta, reference_phi,
                         anchor_phase=sign*eps*float(flow['EM_Gauss_parameter'][selected[2]]))
                moved_neutrals.append(neutral); moved_states.append(sector.source(cm, branch)@neutral)
                moved_currents.append(sector.lift(native_onebody(law, pf, pu, pa, direction)))
            dn = -1j*(moved_neutrals[0]-moved_neutrals[1])/(2*eps)
            dc = -1j*(moved_states[0]-moved_states[1])/(2*eps)
            physical_neutral = dn+(orbital+em)@omega
            physical_charged = dc+(orbital+em)@(B@omega)
            leibniz = check(float(np.linalg.norm(physical_charged-B@physical_neutral-first@omega)),
                            'literal physical SOURCE Leibniz retains SAME nonzero neutral', 3e-6)
            current_derivatives.append(float((np.vdot(moved_states[0], moved_currents[0]@moved_states[0]).real-
                                             np.vdot(moved_states[1], moved_currents[1]@moved_states[1]).real)/(2*eps)))
            max_first=max(max_first, err, row['first_error']); max_gauss=max(max_gauss,row['fullGauss_jet_error'])
            max_second=max(max_second,second_error); max_leibniz=max(max_leibniz,leibniz)
            node_records.append(dict(axis=axis, full_physical_neutral_noise=float(np.linalg.norm(physical_neutral)),
                primitive_spin_neutral_noise=float(np.linalg.norm(spin@omega)),
                orbital_neutral_noise=orbital_images[-1], EM_neutral_noise=EM_images[-1],
                reference_spin_neutral_noise=float(np.linalg.norm(reference_spin@omega)),
                explicit_reference_SOURCE_spin_word_norm=float(np.linalg.norm(
                    (reference_spin@B-B@reference_spin)@omega)),
                charged_reference_charge_image_norm=float(np.linalg.norm(sector.nref*(B@omega))),
                physical_source_reduction_error=err, true_second_SOURCE_error=second_error,
                retained_horizontal_Phi_norm=row['horizontal_Phi_norm'],
                retained_colour_link_boundary_norm=row['retained_colour_link_boundary_norm'],
                retained_anchor_flux_norm=row['retained_anchor_flux_norm'],
                retained_sine_EM_norm=row['retained_real_sine_EM_norm'],
                noOrbital_defect=float(np.linalg.norm((orbital@B-B@orbital)@omega)),
                noEM_defect=float(np.linalg.norm((-1j*sector.source(row['pieces']['actual_EM'],branch)+
                                                 em@B-B@em)@omega))))
            if axis == 2:
                orbit_ops.append(orbital)
        max_casimir = max(max_casimir, check(float(np.linalg.norm(sum(literal_axis_words)-
                                           .75*(sector.source(coefficient, branch)@omega))),
                                           'literal physical relative SOURCE total Casimir', 2.4e-5))
        records.append(dict(node=node, axes=node_records))
    g_orb = sparse.block_diag(orbit_ops, format='csr')
    # Actual ordinary H includes quantum Phi/SU4/A kinetic, finite auxiliary,
    # reference/Coulomb and paired harmonic scalar offsets from the SAME chart.
    psi = chart['source']; h = chart['H']; hpsi=h@psi
    torque = float(2*np.vdot(g_orb@psi,hpsi).imag)
    torque_norm = float(np.linalg.norm(g_orb@hpsi-h@(g_orb@psi)))
    no_orb=max(a['noOrbital_defect'] for r in records for a in r['axes'])
    no_em=max(a['noEM_defect'] for r in records for a in r['axes'])
    intrinsic_total_false=adverse(torque_norm,'Phi-internal-only as TOTAL physical angular generator')
    return dict(branch=branch, SAME_chart_source_norm=float(np.linalg.norm(psi)),
        source_prepared_once=True, same_current_and_ordinary_H=True, quantum_field_nodes=count,
        full_primitive_dimension=sector.dimension, whole_quantum_dimension=len(psi),
        far_reference_fixed=chart['selected_reference_coordinate'], prototype_coordinate=chart['prototype_reference_coordinate'],
        original_reference_Phi_unchanged=True, reference_charge_boundary_image_norm=ref_image,
        physical_source_first_max_error=max_first, full15_source_jet_max_error=max_gauss,
        physical_identification_to_FROZEN115_K_error=max_intrinsic,
        literal_physical_SOURCE_Casimir_error=max_casimir,
        literal_physical_second_SOURCE_max_error=max_second, literal_Leibniz_max_error=max_leibniz,
        nonzero_orbital_neutral_noise=max(orbital_images), nonzero_EM_neutral_noise=max(EM_images),
        physical_current_derivative_max=max(abs(v) for v in current_derivatives),
        actual_current_derivatives=current_derivatives,
        ordinary_orbital_torque_expectation=torque, ordinary_orbital_torque_image_norm=torque_norm,
        canonical_quantum_field_neutral_noise=float(np.linalg.norm(chart['Gfield']@chart['omega'])),
        adverse=dict(Phi_internal_as_total=intrinsic_total_false,
            omitted_orbital=adverse(no_orb,'omitted actual spatial orbital SOURCE piece'),
            omitted_EM=adverse(no_em,'omitted actual real EM/boundary SOURCE piece',1e-9),
            omitted_reference=adverse(float(np.linalg.norm(psi)), 'omitted compensating anchored Ref charge')),
        records=records)


def geometric_identification_budget() -> dict[str, Any]:
    radius = 64; c=float(C); period=4*math.pi*radius/c
    # The C6 collar degree13 has seven coefficients. This deliberately loose
    # bound includes all six derivatives, local sites, source/path/canonical
    # factors. It multiplies existing SAME-field q6 normal graphs, not CT words.
    coefficient_cap=12012*64*math.factorial(13)
    coefficient_bits=math.ceil(math.log2(coefficient_cap))
    site_bits=math.ceil(math.log2((2*radius+3)**3))
    six_geometry_bits=6*(coefficient_bits+math.ceil(math.log2(radius+2))+16)+site_bits+64
    if six_geometry_bits >= 1024:
        raise AssertionError('physical stress/collar geometric q6 reserve exceeded')
    return dict(tube_inner=48,tube_outer=radius, C6_collar=True,
        reference_distance=NSTAR//2, reference_outside_support=True,
        physical_rotation_period=period, maximum_coordinate_speed_upper=c/2,
        clock_distinct_from_Compton_T6=True, physical_rotation_angle=2*math.pi,
        period_definition='4*pi*64/C; steady supported weak rotation, not a sudden T6 pulse',
        onebody_orbital_operator_geometry_upper=3*radius,
        native_whole_Pauli_orbital_operator_upper=10*3*radius,
        rest_source_axis_operator_upper=.5,
        native_light_number_density_upper=4, native_compensated_density_absolute_upper=6,
        spatial_support_site_count_upper=(2*radius+3)**3,
        geometric_q6_bits=six_geometry_bits, geometric_normal_stress_reserve_bits=1024,
        additional_Compton_source_word_reserve_bits=0,
        original_and_improved_normal_log2_upper=-32874+1024,
        coexact_EM_image_log2_upper=-47199+1024,
        original_and_improved_amplitude_words_unchanged=[-4970,-5334],
        native_fast_contour_not_a_lowk_clock=True,
        eta18_local_source=source19_nested_casimir(),
        raw_primitive_flow=physical_raw_noether_geometry_majorant())


def serialize(value: Any) -> Any:
    if isinstance(value,np.ndarray):
        return dict(real=value.real.tolist(),imag=value.imag.tolist()) if np.iscomplexobj(value) else value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,complex): return dict(real=value.real,imag=value.imag)
    raise TypeError(type(value).__name__)


def consumed_api_sha256() -> dict[str,str]:
    apis=[PrimitiveSector,native_onebody,native_source_coefficient,native_source_chart,
          primitive_flow,scalar_orbital,covariant_field_difference,physical_generators,frame_chart,
          centre_scalar,physical_h,selected_band,ImprovedPhiLaw,verify_cartesian_successor]
    return {f'{fn.__module__}.{fn.__name__}':hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() for fn in apis}


def verify_shared_native_capsule(path: Path) -> dict[str, str]:
    source = path.read_text()
    nodes = {node.name: node for node in ast.parse(source).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    hashes = {}
    for api in (PrimitiveSector, native_onebody, native_source_coefficient, native_source_chart):
        segment = ast.get_source_segment(source, nodes[api.__name__])
        if segment is None or segment.strip() != inspect.getsource(api).strip():
            raise ValueError(f'shared native scientific API changed: {api.__name__}')
        hashes[api.__name__] = hashlib.sha256(segment.strip().encode()).hexdigest()
    return hashes


def uniform_lattice_identification(frame: np.ndarray, band: RotationBand,
                                   metadata: dict[str, Any]) -> dict[str, Any]:
    cert = metadata['calibration']
    if cert['N'] != NSTAR or not cert['all64_centres'] or not cert['all_real_shifts']:
        raise ValueError('actual all64/allrealshift heavy metric certificate required')
    check(float(np.linalg.norm(np.asarray(cert['G'])-
                      np.einsum('mij,nji->mn', frame, frame).real/4)),
          'actual physical kinetic metric, never source Gram', 1e-13)
    if metadata['charged_gap']['microscopic_compensated_gap_lower_float'] < MASS_LOWER:
        raise ValueError('physical microscopic rest gap lower bound changed')
    frame_debt = float(Fraction(cert['frame_distance_upper']))
    vector_debt = 2*band.v*max(np.linalg.norm(f, 2) for f in frame[1:])*LOWK**3/6
    mass_debt = abs(band.r/band.a)*(band.a*LOWK)**4*float(np.linalg.norm(frame[0], 2))/(
                    4 if band.improved else 24)
    mass_cap = abs(band.m)+abs(band.r/band.a)*(band.a*LOWK)**2
    metric_debt = 2*(mass_cap+band.v*math.sqrt(3)*LOWK)*frame_debt
    total = vector_debt+mass_debt+metric_debt+64*np.finfo(float).eps
    speed_squared = band.v**2*float(cert['G'][1][1])
    current_debt = 128*(1+abs(band.v)+abs(band.m))*frame_debt/(MASS_LOWER*speed_squared)
    points = np.array([0., 2., 48., 52., 56., 60., 64., 65., float(NSTAR//2)])
    cutoff = collar_polynomial(points)
    if np.any(cutoff < -1e-11) or np.any(cutoff > 1+1e-11) or np.any(np.diff(cutoff)>1e-11):
        raise AssertionError('supported C6 spatial collar is not monotone/positive')
    check(float(abs(cutoff[-1])), 'actual far anchored Ref outside rotation support')
    return dict(all_momenta_norm_at_most=LOWK, all_spatial_rotations=True,
        actual_light_parameters=cert['parameters'], actual_heavy_parameters=cert['heavy_parameters'],
        sine_vector_covariance_upper=vector_debt, literal_mass_covariance_upper=mass_debt,
        actual_full10_metric_covariance_upper=metric_debt, operator_covariance_upper=total,
        physical_positive_cluster_upper=4*total/MASS_LOWER,
        physical_current_spin_identification_upper=current_debt,
        actual_microscopic_mass_lower=metadata['charged_gap']['microscopic_compensated_gap_lower'],
        executed_cone_calibration_upper=cert['continuum_cone_calibration_debt_upper'],
        cutoff_smoke_points=points, cutoff_smoke_values=cutoff,
        source_or_Compton_filter_inserted=False, dimensional_not_amplitude_debt=True)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--descriptor',type=Path,required=True)
    parser.add_argument('--cartesian-packet',type=Path,required=True)
    parser.add_argument('--cartesian-receipt',type=Path,required=True)
    parser.add_argument('--cartesian-correction',type=Path,required=True)
    parser.add_argument('--band-metadata',type=Path,required=True)
    parser.add_argument('--pre-identification-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    descriptor=decode(json.loads(args.descriptor.read_text()))
    if len(descriptor['compact_displacement']) != 19:
        raise ValueError('physical identification uses actualSC19, notRAW41/profile19cubed')
    packet=json.loads(args.cartesian_packet.read_text())
    if hashlib.sha256(args.cartesian_receipt.read_bytes()).hexdigest()!=packet['receipt']['sha256']:
        raise ValueError('frozen115 executed intrinsic receipt changed')
    intrinsic_receipt=json.loads(args.cartesian_receipt.read_text())
    capsule_bindings=verify_shared_native_capsule(args.pre_identification_source)
    correction=verify_cartesian_successor(dict(cartesian=args.cartesian_packet,
        cartesian_receipt=args.cartesian_receipt, cartesian_correction=args.cartesian_correction),packet)
    band,frame,calibration=selected_band()
    rest=physical_rest_spin(frame,band)
    anti_rest=physical_rest_spin(frame,band,-1)
    rotation=Rotation.from_rotvec(np.array([.41,-.32,.27])).as_matrix()
    momenta=[np.array([.025,-.018,.021]),np.array([.047,0.,0.]),np.array([.02,.03,-.032])]
    lowk=[lowk_little_group(frame,band,p,rotation,branch) for branch in (1,-1) for p in momenta]
    for row in lowk:
        row['noGauss_rejected']=adverse(row['noGauss_defect'],
                    'physical scalar-coordinate-only without Phi/Gauss spin',
                    max(3e-7,row['covariance_upper']))
    uniform=uniform_lattice_identification(frame,band,json.loads(args.band_metadata.read_text()))
    budget=geometric_identification_budget()
    bound=uniform['operator_covariance_upper']
    budget['physical_rotation_lattice_phase_upper']=budget['physical_rotation_period']*bound
    native=[native_spatial_identification(native_source_chart(frame,band,descriptor,branch),frame,band,branch,
                                         intrinsic_receipt)
            for branch in (1,-1)]
    result=dict(plan=PLAN, physical_rest_spin=rest, calibration=calibration,
        actual_anti_rest_spin=anti_rest,
        lowk_little_group=lowk, geometric_identification=budget,native=native,
        uniform_lowk_physical_identification=uniform, immutable_native_capsule_apis=capsule_bindings,
        approved_intrinsic_successor=correction,
        current_spin_is_physical_rest_not_arbitrary_totalJ=True,
        original_heavy_and_deeper_statistical_laws_unchanged=True,
        all64_E0kappa_ordinary_offsets_retained=True,
        source_pair_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                           for name in ('spatial_noether_identification.py','spatial_noether_identification.md')},
        consumed_api_sha256=consumed_api_sha256(),
        input_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in (args.descriptor,args.cartesian_packet,args.cartesian_receipt,
                                   args.cartesian_correction,args.band_metadata,args.pre_identification_source)})
    args.output.write_text(json.dumps(result,default=serialize,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),native_branches=2,
        physical_rest_Casimir_error=rest['rest_Casimir_error'],
        lowk_physical_spinor_error=max(row['physical_spinor_error'] for row in lowk),
        physical_SOURCE_first_error=max(row['physical_source_first_max_error'] for row in native),
        physical_SOURCE_second_error=max(row['literal_physical_second_SOURCE_max_error'] for row in native),
        physical_current_response=max(row['physical_current_derivative_max'] for row in native),
        additional_CT_word_bits=0)))


if __name__=='__main__':
    main()
