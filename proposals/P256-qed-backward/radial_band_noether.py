#!/usr/bin/env python3
"""Independent RAW41 actual-band CAR preparation and physical SOURCE domains.

No scattering receiver is changed. Finite Bloch/native calculations below are
explicit approximations; the legal Nstar source graphs are algebraic certificates.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
import math
from dataclasses import asdict
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

from improved_phi_band import ImprovedPhiLaw, physical_h, selected_band
from improved_physical_compton import decode
from moving_physical_noether import field_noether_galerkin
from noncompact_mechanical_pulse import radial_profile
from noncompact_original_source_flight import sqrt_upper, dyadic_upper
from noncompact_scattering_flight import compact_descriptor, normalized_flight_bound, free_conjugated
from noncompact_screened_gauss_dressing import frame_chart
from noncompact_quantitative_normal_budget import ceil_log2
from original_total_angular_transfer import scalar_orbital
from physical_carrier_transport import (
    PrimitiveSector, SITES, native_onebody, native_source_coefficient,
    primitive_gauss_source_probe,
)
from rotation_band import RotationBand
from spatial_noether_identification import (
    anchored_coefficient, derived_spins, frozen_intrinsic_source_word,
    gauss_family, physical_rest_spin, spatial_source_identification, lowk_little_group,
    uniform_lattice_identification,
)
from unrestricted_field_minimum import BARE, local_potential

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-synthesis-continuation-20261006'
NSTAR = 5*2**96
C = F(1130202988249777, 9007199254740992)
UE = F.from_float(1.5429442570866347e-148)/2**4096
BETA = np.array([1., .3j, -.2+.4j, .6])


def check(error: float, name: str, upper: float = 3e-7) -> float:
    error = float(error)
    if not np.isfinite(error) or error > upper:
        raise AssertionError(f'{name}: {error} > {upper}')
    return error


def adverse(defect: float, name: str, lower: float = 1e-9) -> dict[str, Any]:
    if not np.isfinite(defect) or defect <= lower:
        raise AssertionError(f'non-discriminating adverse {name}: {defect}')
    return dict(rejected=True, defect=float(defect), tolerance=lower)


def radial_profile_data(band: RotationBand) -> dict[str, Any]:
    sites = np.indices((41,)*3).reshape(3, -1).T
    eta, guard, _ = radial_profile(SimpleNamespace(size=41, band=band, sites=sites),
                                  np.full(3, 20), .25, 2.4, 10., 17.)
    displacement = sites-20
    halo = np.zeros((53,)*3, complex)
    halo[6:47, 6:47, 6:47] = eta.reshape((41,)*3)
    positions = np.indices(halo.shape).astype(float)-26
    def orbital(axis: int, value: np.ndarray) -> np.ndarray:
        out = np.zeros_like(value)
        velocity = np.cross(np.eye(3)[axis], positions.reshape(3, -1).T)
        for j in range(3):
            out += velocity[:, j].reshape(halo.shape)*(np.roll(value, -1, axis=j)-
                    np.roll(value, 1, axis=j))/(2j*band.a)
        return out
    first = np.asarray([orbital(j, halo) for j in range(3)])
    second = np.asarray([orbital(j, first[j]) for j in range(3)])
    guarded = guard*eta
    guard_norm = float(np.linalg.norm(guarded))
    if guard_norm <= .5:
        raise AssertionError('actual guarded RAW normalization floor failed')
    guarded /= guard_norm
    six_geometry = (3*(20+6)/band.a)**6
    if six_geometry >= 2**50 or math.sqrt(3)*26 >= 48:
        raise AssertionError('RAW six-halo/angular domain geometry failed')
    return dict(eta=eta, guard=guard, guarded_eta=guarded, displacement=displacement,
                halo_eta=halo, halo_orbital_first=first, halo_orbital_second=second,
                actual_count=int(np.count_nonzero(eta)), support_count=41**3,
                source_word=NSTAR//2+60, screen_eta=18, alpha=2**371,
                profile_radius=20, profile_N=41, six_halo_radius=math.sqrt(3)*26,
                guard_norm=guard_norm, guard_distance=float(np.linalg.norm(eta-guard*eta)),
                normalized_guard_distance=float(np.linalg.norm(eta-guarded)),
                scalar_orbital_norms=np.linalg.norm(first.reshape(3, -1), axis=1),
                scalar_nested_casimir_norm=float(np.linalg.norm(second.sum(axis=0))),
                rigid_core_eta18_scalar_second=-9/math.cosh(36)**2,
                rigid_core_eta18_relative_axis_curvature=9/(math.cosh(36)**2*math.tanh(36)),
                eta18_not_replaced_by_one=True,
                angular_q6_geometry_upper=six_geometry, angular_q6_upper=2**50,
                CY=2**64, receiver_source=False)


def batch_h(points: np.ndarray, frame: np.ndarray, band: RotationBand,
            branch: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Literal scalar mass formula, batched; RotationBand.mass itself is scalar."""
    p = np.asarray(points)
    delta = 2*np.sin(band.a*p/2)**2
    mass = band.m+band.r/band.a*np.sum(delta**2 if band.improved else delta, axis=1)
    mass_first = band.r*np.sin(band.a*p)*(2*delta if band.improved else 1)
    mass_second = band.r*band.a*(2*(np.sin(band.a*p)**2+delta*np.cos(band.a*p))
                               if band.improved else np.cos(band.a*p))
    phi = frame if branch == 1 else np.concatenate((-frame[:1].transpose(0, 2, 1),
                                                   frame[1:].transpose(0, 2, 1)))
    h = mass[:, None, None]*phi[0]+np.einsum('xj,jab->xab',
                                  band.v*np.sin(band.a*p)/band.a, phi[1:])
    velocity = mass_first[:, :, None, None]*phi[None, :1]
    velocity = velocity+band.v*np.cos(band.a*p)[:, :, None, None]*phi[None, 1:]
    contact = mass_second[:, :, None, None]*phi[None, :1]
    contact = contact-band.v*band.a*np.sin(band.a*p)[:, :, None, None]*phi[None, 1:]
    return h, velocity, contact


def source_seed(frame: np.ndarray, band: RotationBand) -> np.ndarray:
    law = ImprovedPhiLaw(3, band)
    return native_source_coefficient(law, np.tile(frame, (27, 1, 1, 1)),
        np.tile(np.eye(4), (27, 3, 1, 1)), np.zeros((27, 3)),
        np.array([1., 0., 0.]), BETA)[:4]


def radial_band_preparation(frame: np.ndarray, band: RotationBand,
                            profile: dict[str, Any], quadrature: int = 129) -> dict[str, Any]:
    """CAR creation above filled negative sea, not imposed spinor/filter hardware.

    P+ is the independently computed Pauli-blocking image of the local creation
    coefficient. Its use here analyses that state, not a new apparatus operation.
    """
    if quadrature < 65 or quadrature % 2 != 1:
        raise ValueError('odd Bloch comparison period >=65 required')
    q = quadrature
    base = np.zeros((q,)*3, complex)
    start = q//2-20
    base[start:start+41, start:start+41, start:start+41] = profile['eta'].reshape((41,)*3)
    eta_hat = np.fft.fftn(np.fft.ifftshift(base), norm='ortho').reshape(-1)
    momenta = 2*np.pi*np.fft.fftfreq(q, d=band.a)
    points = np.stack(np.meshgrid(momenta, momenta, momenta, indexing='ij'), axis=-1).reshape(-1, 3)
    m = source_seed(frame, band)
    # Exercise equality with the actual public scalar physical_h, not new gamma.
    for p in (np.zeros(3), np.array([.3, -.2, .4]), np.array([2., -1., .7])):
        check(np.linalg.norm(batch_h(p[None], frame, band)[0][0]-physical_h(p, frame, band)),
              'literal original/improved band batch identity', 2e-14)
    result = []
    for branch in (1, -1):
        seed = m if branch == 1 else m.conj()
        gram = np.zeros((4, 4), complex)
        first = squared = lowk = 0.
        current = np.zeros(3); current2 = np.zeros(3); contact = np.zeros(3)
        density = np.zeros((q,)*3)
        projected = np.empty((len(points), 4, 4), complex)
        hole_error = 0.
        gap = math.inf
        for offset in range(0, len(points), 32768):
            p = points[offset:offset+32768]
            h, v, z = batch_h(p, frame, band, branch)
            values, vectors = np.linalg.eigh(h)
            if np.any(values[:, 1] >= 0) or np.any(values[:, 2] <= 0):
                raise AssertionError('literal original/improved occupied/unoccupied split failed')
            gap = min(gap, float(np.min(values[:, 2])))
            occupied = vectors[:, :, :2]; positive = vectors[:, :, 2:]
            pm = np.einsum('xai,xbi,bc->xac', positive, positive.conj(), seed)
            blocked = np.einsum('xai,xbi,bc->xac', occupied, occupied.conj(), seed)
            hole_error = max(hole_error, float(np.max(np.linalg.norm(pm+blocked-seed, axis=(1, 2)))))
            weight = abs(eta_hat[offset:offset+len(p)])**2
            gram += np.einsum('x,xab,xac->bc', weight, pm.conj(), pm)
            hp = h@pm
            first += float(np.einsum('x,xab,xab->', weight, pm.conj(), hp).real)
            squared += float(np.einsum('x,xab,xab->', weight, hp.conj(), hp).real)
            pw = np.sum(abs(pm)**2, axis=(1, 2))*weight
            lowk += float(np.sum(pw[np.linalg.norm(p, axis=1) <= .05]))
            for j in range(3):
                vp = v[:, j]@pm
                current[j] += float(np.einsum('x,xab,xab->', weight, pm.conj(), vp).real)
                current2[j] += float(np.einsum('x,xab,xab->', weight, vp.conj(), vp).real)
                contact[j] += float(np.einsum('x,xab,xab->', weight, pm.conj(), z[:, j]@pm).real)
            projected[offset:offset+len(p)] = eta_hat[offset:offset+len(p), None, None]*pm
        norm2 = float(np.trace(gram).real)
        if norm2 <= 1/8 or np.min(np.linalg.eigvalsh(gram)) <= 0:
            raise AssertionError('allRef4 actual CAR norm/rank admission failed')
        # Inverse Fourier density of the actual Pauli-blocked state; its tail is
        # retained. Compact bare eta does NOT imply compact projected state.
        for ref in range(4):
            wave = np.fft.ifftn(projected[:, :, ref].reshape(q, q, q, 4), axes=(0, 1, 2), norm='ortho')
            density += np.sum(abs(wave)**2, axis=-1)/norm2
        density = np.fft.fftshift(density)
        coordinates = np.indices((q,)*3).astype(float)-q//2
        radius = np.sqrt(np.sum(coordinates**2, axis=0))
        check(abs(float(density.sum())-1), 'actual CAR Fourier density normalization', 1e-12)
        l1 = float(np.sum(abs(profile['eta'])))
        rigorous_lowk = min(1., 8*float(np.linalg.norm(m, 'fro')**2)*l1*l1*(.05/np.pi)**3)
        # Finite trigonometric support and the literal gap give an Nstar-vs-
        # continuum Riemann error without allocating Nstar. Finite129 remains
        # a labelled comparison, NOT assigned this much smaller legal-grid debt.
        derivative_majorant = 64*(1+41**3)*(1+3*20)*(1+np.linalg.norm(m)**2)*(1+8/.1294)**4
        legal_grid_debt = 6*math.pi*derivative_majorant/NSTAR
        result.append(dict(branch=branch, actual_CAR_norm_squared=norm2,
            allRef4_Gram=gram, allRef4_Gram_eigenvalues=np.linalg.eigvalsh(gram),
            global_normalizer=math.sqrt(norm2), actual_gap_sample_min=gap,
            actual_band_energy_mean=first/norm2, actual_band_energy_rms=math.sqrt(squared/norm2),
            actual_band_energy_second=squared/norm2,
            literal_current_mean=current/norm2, literal_current_rms=np.sqrt(current2/norm2),
            literal_contact_mean=contact/norm2, occupied_plus_unoccupied_identity_error=hole_error,
            actual_lowk_sample_weight=lowk/norm2, legal_lowk_spectral_weight_upper=rigorous_lowk,
            retained_density_outside_RAW_cube=float(np.sum(density[np.max(abs(coordinates), axis=0)>20])),
            retained_density_outside48=float(np.sum(density[radius>48])),
            retained_density_outside64=float(np.sum(density[radius>64])),
            band_projection_not_hardware=True, predefined_Dirac_spinor=False,
            finite_Bloch_period=q, native_Bloch_comparison_not_Nstar=True,
            legal_Nstar_quadrature_debt_upper=legal_grid_debt,
            gap_hypothesis_lower=.129417988064,
            ordinary_fullQ_or_W2_energy_inferred_from_band_mean=False))
    return dict(parameters=asdict(band), branches=result,
                actual_scalar_profile_spectral_lowk_sample=float(np.sum(abs(eta_hat[np.linalg.norm(points, axis=1)<=.05])**2)),
                full_BZ_retained=True, full_RAW_claimed_lowk=False)


def native_radial_source_chart(frame: np.ndarray, band: RotationBand,
                              profile: dict[str, Any], branch: int) -> dict[str, Any]:
    """Independent finite quantum RAW restriction, NEVER the SC19 constructor."""
    sector = PrimitiveSector(10 if branch == 1 else 6)
    law = ImprovedPhiLaw(3, band)
    heavy = ImprovedPhiLaw(3, RotationBand(**BARE))
    selected = [law.index[tuple(site)] for site in SITES]
    eta = profile['eta'].reshape((41,)*3)
    radial = np.array([eta[tuple(20+site)] for site in SITES])
    rng = np.random.default_rng(22212191)
    normal = law.fields(rng.normal(size=(law.volume, 4, 15)))
    normal /= np.linalg.norm(normal)
    nodes, modes = np.linalg.eigh(np.array([[0., 1.], [1., 0.]])/math.sqrt(2))
    p2 = modes.T@np.diag([.5, 1.5])@modes
    scales = (.02, .04, .07)
    ue, chi = .0025, .0004
    K = float(C)**2/(2*ue)
    angles = np.array([.71, -.46, .32])
    green = np.array([[5., -1., -4.], [-1., 2., -1.], [-4., -1., 5.]])/9
    incidence = np.array([[1., 0.], [-1., 1.], [0., -1.]])
    full_bonds = [(x, int(law.target[x, j])) for x in range(27) for j in range(3)]
    blocks, sources, omegas, ops, snapshots, coefficients, orbitals = [], [], [], [], [], [], []
    currents = [[], []]; contacts = [[], []]; offsets = []; boundary = []
    scalar_values = []
    for kappa in (-1, 1):
        for zphi, zlink, za in itertools.product(nodes, repeat=3):
            fields = np.tile(frame, (27, 1, 1, 1))+.003*zphi*normal
            links = np.asarray([[expm(1j*(.03+scales[1]*zlink)*(1+x+j)*law.generators[3])
                                 for j in range(3)] for x in range(27)])
            potential = np.tile(kappa*angles/3, (27, 1))
            potential[:, 2] += scales[2]*za*np.cos(2*math.pi*law.sites[:, 0]/3)
            h = native_onebody(law, fields, links, potential)
            values, basis = np.linalg.eigh(h)
            if values[5] >= 0 or values[6] <= 0:
                raise AssertionError('actual native light negative sea failed')
            occupied = basis[:, :6]
            omega = sector.neutral(occupied, branch)
            coefficient = native_source_coefficient(law, fields, links, potential, radial, BETA)
            B = sector.source(coefficient, branch)
            source = B@omega
            # Same primitive sea yields the actual light-band Pauli projector.
            P = occupied@occupied.conj().T
            allowed = (np.eye(12)-P) if branch == 1 else P
            car_norm2 = float(np.trace(coefficient.conj().T@allowed@coefficient).real)
            check(abs(np.linalg.norm(source)**2-car_norm2), 'actual primitive source/CAR band norm', 1e-12)
            refcharge = sector.nref-(4 if branch == 1 else 0)
            rho = sector.density.copy().astype(float); rho[:, 2] += refcharge
            check(float(np.max(abs(rho.sum(axis=1)))), 'actual whole-reference compensated charge')
            coulomb = ue*np.einsum('xi,ij,xj->x', rho, green, rho)
            missing_ref = ue*np.einsum('xi,ij,xj->x', sector.density, green, sector.density)
            boundary.append(sparse.diags(coulomb-missing_ref))
            scalar = float(sum(local_potential(fields, full_bonds, links.reshape(-1, 4, 4))))
            scalar -= float(np.linalg.eigvalsh(heavy.onebody(fields, links, np.zeros_like(potential)))[54:].sum())
            curl = []; magnetic = 0.
            for j in range(3):
                for k in range(j+1, 3):
                    curl.append(potential[:, j]+potential[law.target[:, j], k]-potential[law.target[:, k], j]-potential[:, k])
                    loop = links[:, j]@links[law.target[:, j], k]@links[law.target[:, k], j].conj().transpose(0, 2, 1)@links[:, k].conj().transpose(0, 2, 1)
                    magnetic += .7*float(np.sum(4-np.trace(loop, axis1=1, axis2=2).real))
            scalar += K/2*float(np.sum(np.asarray(curl)**2))+magnetic
            harmonic = ue/27*(kappa*.37)**2
            unwanted = (sector.nref != (4 if branch == 1 else 0)) & (sector.nref != (3 if branch == 1 else 1))
            H = sector.lift(sparse.block_diag((h, np.zeros((4, 4)))).toarray())
            H += sparse.diags(coulomb+scalar+harmonic+unwanted.astype(float))
            H += .7*(sector.identity-sector.cycle)/2
            blocks.append(H); sources.append(source); omegas.append(omega); ops.append(B)
            snapshots.append((fields, links, potential)); coefficients.append(coefficient); orbitals.append(occupied)
            scalar_values.append(scalar)
            offsets.append(dict(kappa=int(kappa), ordinary_scalar=scalar, harmonic=harmonic,
                                photon_curl_squared=float(np.sum(np.asarray(curl)**2))))
            for edge in range(2):
                cut = np.zeros(27); cut[selected[0 if edge == 0 else 2]] = 1
                direction = cut[:, None]-cut[law.target]
                jm = native_onebody(law, fields, links, potential, direction)
                zm = native_onebody(law, fields, links, potential, direction, direction)
                currents[edge].append(sector.lift(sparse.block_diag((jm, np.zeros((4, 4)))).toarray()))
                contacts[edge].append(sector.lift(sparse.block_diag((zm, np.zeros((4, 4)))).toarray()))
    count = len(blocks)
    H = sparse.block_diag(blocks, format='csr')
    for axis, coefficient in enumerate((chi/(2*scales[0]**2), .3*chi/(2*scales[1]**2), ue/(2*scales[2]**2))):
        factors = [sparse.csr_matrix(p2) if k == axis else sparse.eye(2) for k in range(3)]
        k3 = sparse.kron(sparse.kron(factors[0], factors[1]), factors[2])
        H += coefficient*sparse.kron(sparse.eye(2), sparse.kron(k3, sector.identity), format='csr')
    derivative = modes.T@np.array([[0., 1.], [-1., 0.]])@modes/(math.sqrt(2)*scales[0])
    d16 = np.kron(np.eye(2), np.kron(np.kron(derivative, np.eye(2)), np.eye(2)))
    Gfield = sparse.kron(sparse.csr_matrix(field_noether_galerkin(np.ones((count, 1)), d16[None])), sector.identity, format='csr')
    omega = np.concatenate(omegas)/math.sqrt(count)
    source = np.concatenate(sources)/math.sqrt(count)
    normalizer = float(np.linalg.norm(source)); source /= normalizer
    Bfull = sparse.block_diag(ops, format='csr')/normalizer
    check(np.linalg.norm(Bfull@omega-source), 'onceprepared radial primitive source identity')
    shift = float(np.max(np.asarray(abs(H).sum(axis=1))))+1
    positive_H = H+shift*sparse.eye(cast(tuple[int, int], H.shape)[0], format='csr')
    Qs = positive_H@source
    return dict(sector=sector, law=law, H=H, Q=positive_H, positive_shift=shift,
        source=source, omega=omega, B=Bfull, Gfield=Gfield, node_count=count,
        field_snapshots=snapshots, coefficient_snapshots=coefficients, node_neutrals=omegas,
        occupied_orbitals=orbitals, independent_reference_phi=[f[selected[2]].copy() for f, _, _ in snapshots],
        current=[sparse.block_diag(x, format='csr') for x in currents],
        contact=[sparse.block_diag(x, format='csr') for x in contacts],
        profile=radial, raw_quartet=BETA, source_global_norm=normalizer,
        ordinary_offsets=offsets, reference_scalar_boundaries=boundary,
        UE=ue, K=K, chi=chi, full_bonds=full_bonds,
        actual_SOURCE_first_graph=float(np.linalg.norm(Qs)),
        actual_SOURCE_half_form=math.sqrt(float(np.vdot(source, Qs).real)),
        actual_SOURCE_normalized_energy=float(np.vdot(source, H@source).real),
        native_is_RAW_restriction_not_whole68921=True,
        native_legal_N_or_gap_constants_transferred=False)


def aligned_light_neutral(chart: dict[str, Any], fields: np.ndarray, links: np.ndarray,
                          potential: np.ndarray, anchor: np.ndarray, branch: int) -> np.ndarray:
    h = native_onebody(chart['law'], fields, links, potential)
    values, orbitals = np.linalg.eigh(h)
    if values[5] >= 0 or values[6] <= 0:
        raise AssertionError('physical light sea occupied/unoccupied derivative guard')
    neutral = chart['sector'].neutral(orbitals[:, :6], branch)
    overlap = np.vdot(anchor, neutral)
    if abs(overlap) <= .99:
        raise AssertionError('same physical neutral differentiability overlap guard')
    return neutral*np.conj(overlap)/abs(overlap)


def radial_noether_certificate(chart: dict[str, Any], frame: np.ndarray,
                               band: RotationBand, profile: dict[str, Any], branch: int,
                               intrinsic_receipt: dict[str, Any]) -> dict[str, Any]:
    """True relative SOURCE words on actual LIGHT sea and retained radial jets."""
    sector, law = chart['sector'], chart['law']
    selected = [law.index[tuple(site)] for site in SITES]
    first_eta = profile['halo_orbital_first']
    second_eta = profile['halo_orbital_second']
    rows = []; current_response = []; orbital_noise = []; em_noise = []; neutral_noise = []
    max_first = max_second = max_K = max_leibniz = 0.
    no_right = no_flux = false_spin = 0.
    gauss_controls = []
    for node, ((fields, links, potential), coefficient, omega, refphi) in enumerate(zip(
            chart['field_snapshots'], chart['coefficient_snapshots'], chart['node_neutrals'], chart['independent_reference_phi'])):
        if node in (0, chart['node_count']//2):
            gauss_controls.append(primitive_gauss_source_probe(law, fields, links, potential, sector,
                                      chart['profile'], BETA, omega, branch))
        B = sector.source(coefficient, branch)
        node_rows = []
        for axis in range(3):
            ident = spatial_source_identification(law, fields, links, potential,
                                                  chart['profile'], BETA, refphi, axis)
            lam = ident['lambda_colour']
            spin_modes = np.zeros((16, 16), complex)
            for j, x in enumerate(selected):
                spin_modes[4*j:4*j+4, 4*j:4*j+4] = lam[x]
                kword = frozen_intrinsic_source_word(intrinsic_receipt, frame_chart(fields[x])[0],
                       frame_chart(refphi)[0], coefficient[4*j:4*j+4], axis, branch)
                target = lam[x]@coefficient[4*j:4*j+4] if branch == 1 else -coefficient[4*j:4*j+4].conj().T@lam[x]
                max_K = max(max_K, check(np.linalg.norm(kword-target), 'actual radial frozen115 K seed'))
            spin = sector.lift(spin_modes)
            orbital_modes = np.zeros((16, 16), complex); orbital_modes[:12, :12] = ident['orbital']
            orbital = sector.lift(orbital_modes)
            em = sector.lift(np.diag(np.r_[-ident['EM_parameter'], np.zeros(4)]))
            em += sector.identity*float(2*np.sum(ident['EM_parameter'][::4]))
            Leta = np.array([first_eta[(axis, *tuple(26+site))] for site in SITES])
            L2eta = np.array([second_eta[(axis, *tuple(26+site))] for site in SITES])
            cmL = anchored_coefficient(law, fields, links, potential, Leta, BETA, refphi)
            cmL2 = anchored_coefficient(law, fields, links, potential, L2eta, BETA, refphi)
            spinL = np.concatenate([lam[x]@cmL[4*j:4*j+4] for j, x in enumerate(selected)])
            eta_word = -branch*sector.source(cmL, branch)
            first = -1j*sector.source(ident['total_jet'], branch)+(orbital+em)@B-B@(orbital+em)+eta_word
            boundary = (-1j*sector.source(sum(ident['pieces'].values()), branch)+(orbital+em)@B-B@(orbital+em))
            intrinsic = branch*sector.source(ident['spin_coefficient'], branch)
            max_first = max(max_first, check(np.linalg.norm((first-boundary-eta_word-intrinsic)@omega),
                                             'radial physical first SOURCE and retained eta', 3e-6))
            step = .002
            moved_states = []; neutral_words = []
            for sign in (1, -1):
                t = sign*step
                pf, pu, anchor = gauss_family(law, fields, links, lam, t)
                moved_profile = chart['profile']-1j*t*Leta-t*t/2*L2eta
                cm = anchored_coefficient(law, pf, pu, potential, moved_profile, BETA, refphi, anchor)
                op = sector.source(cm, branch)
                moved = expm_multiply(1j*t*spin, omega, traceA=1j*t*complex(spin.diagonal().sum()))
                moved_states.append(op@moved); neutral_words.append(op@(spin@moved))
            second_state = -(moved_states[0]-2*(B@omega)+moved_states[1])/step**2
            cross = -1j*(neutral_words[0]-neutral_words[1])/(2*step)
            second = second_state-2*cross+B@(spin@(spin@omega))
            target2 = .25*(B@omega)-2*(sector.source(spinL, branch)@omega)+sector.source(cmL2, branch)@omega
            max_second = max(max_second, check(np.linalg.norm(second-target2),
                                              'literal radial SOURCE second with actual eta/neutral', 8e-6))
            no_right = max(no_right, float(np.linalg.norm(B@(spin@omega))))
            no_flux = max(no_flux, float(np.linalg.norm(sector.source(ident['pieces']['actual_EM'], branch)@omega)))
            fixed = physical_rest_spin(frame, band)['spins'][axis]
            fixed_coefficient = np.concatenate([fixed@coefficient[4*j:4*j+4] for j in range(3)])
            false_spin = max(false_spin, float(np.linalg.norm((sector.source(fixed_coefficient-ident['spin_coefficient'], branch))@omega)))
            eps = 2e-5; flow = ident['physical']
            physical_neutrals = []; physical_states = []; physical_currents = []
            region = np.zeros(law.volume); region[selected[0]] = 1
            direction = region[:, None]-region[law.target]
            for sign in (1, -1):
                pf = fields+sign*eps*flow['field']; pu = links+sign*eps*flow['links']
                pa = potential+sign*eps*flow['canonical_EM']
                neutral = aligned_light_neutral(chart, pf, pu, pa, omega, branch)
                moved_profile = chart['profile']-1j*sign*eps*Leta
                cm = anchored_coefficient(law, pf, pu, pa, moved_profile, BETA, refphi,
                               anchor_phase=sign*eps*float(flow['EM_Gauss_parameter'][selected[2]]))
                state = sector.source(cm, branch)@neutral
                physical_neutrals.append(neutral); physical_states.append(state)
                onebody = native_onebody(law, pf, pu, pa, direction)
                physical_currents.append(sector.lift(sparse.block_diag((onebody, np.zeros((4, 4)))).toarray()))
            dn = -1j*(physical_neutrals[0]-physical_neutrals[1])/(2*eps)
            dc = -1j*(physical_states[0]-physical_states[1])/(2*eps)
            jn = dn+(orbital+em)@omega
            jc = dc+(orbital+em)@(B@omega)
            max_leibniz = max(max_leibniz, check(np.linalg.norm(jc-B@jn-first@omega),
                                                'actual LIGHT sea radial SOURCE Leibniz', 3e-6))
            neutral_noise.append(float(np.linalg.norm(jn)))
            orbital_noise.append(float(np.linalg.norm(orbital@omega))); em_noise.append(float(np.linalg.norm(em@omega)))
            response = float((np.vdot(physical_states[0], physical_currents[0]@physical_states[0]).real-
                              np.vdot(physical_states[1], physical_currents[1]@physical_states[1]).real)/(2*eps))
            current_response.append(response)
            node_rows.append(dict(axis=axis, retained_radial_first_SOURCE_norm=float(np.linalg.norm(eta_word@omega)),
                retained_radial_second_SOURCE_norm=float(np.linalg.norm(sector.source(cmL2, branch)@omega)),
                nonzero_neutral_rightaction=float(np.linalg.norm(spin@omega)),
                actual_physical_light_neutral_noise=float(np.linalg.norm(jn)),
                actual_current_response=response, retained_colour_link_flux=ident['retained_colour_link_boundary_norm'],
                retained_anchor_flux=ident['retained_anchor_flux_norm'], retained_literal_sine_EM=ident['retained_real_sine_EM_norm']))
        rows.append(dict(node=node, axes=node_rows))
    # Full ordinary H, not a Gamma/nearest-Wilson energy or source-value proxy.
    source = chart['source']; H = chart['H']; Q = chart['Q']
    normalized = check(abs(np.linalg.norm(source)-1), 'radial normalized SAME source')
    field_noise = float(np.linalg.norm(chart['Gfield']@chart['omega']))
    e = float(np.vdot(source, Q@source).real)
    first_graph = float(np.linalg.norm(Q@source))
    check(max(0., e-first_graph), 'true SOURCE energy-half/first graph inequality', 1e-9)
    copied_source_defect = float(np.linalg.norm(chart['profile']/np.linalg.norm(chart['profile'])-np.array([1., 0., 0.])))
    return dict(branch=branch, source_norm_error=normalized, actual_light_sea=True,
        native_nodes=chart['node_count'], primitive_dimension=sector.dimension,
        full_quantum_dimension=len(source), true_physical_first_SOURCE_error=max_first,
        true_radial_second_SOURCE_error=max_second, frozen115_full16_seed_error=max_K,
        actual_LIGHT_neutral_Leibniz_error=max_leibniz,
        actual_normalized_original_or_improved_SOURCE_first_graph=first_graph,
        actual_normalized_SOURCE_half_form=math.sqrt(e),
        actual_normalized_ordinary_SOURCE_energy=chart['actual_SOURCE_normalized_energy'],
        nonzero_field_neutral_noise=field_noise, nonzero_physical_neutral_noise=max(neutral_noise),
        nonzero_orbital_neutral_noise=max(orbital_noise), nonzero_EM_neutral_noise=max(em_noise),
        actual_radial_current_response_max=max(abs(x) for x in current_response),
        actual_radial_cut_current=[float(np.vdot(source, j@source).real) for j in chart['current']],
        actual_radial_contact=[float(np.vdot(source, j@source).real) for j in chart['contact']],
        gauss_controls=gauss_controls, ordinary_offsets=chart['ordinary_offsets'],
        native_parameters=dict(UE=chart['UE'], K=chart['K'], chi=chart['chi']),
        native_is_labeled_finite_RAW_restriction=True, whole_RAW_Nstar_not_allocated=True,
        adverse=dict(ignored_neutral=adverse(no_right, 'ignored actual light sea rightaction'),
            omitted_EM_flux=adverse(no_flux, 'omitted actual radial EM flux'),
            predefined_spin=adverse(false_spin, 'predefined constant spin replaces actual Phi frame'),
            copied_SC19=adverse(copied_source_defect, 'copied source19 point profile'),
            Wilson_energy_proxy=adverse(abs(first_graph-.17344423080868282), 'Wilson comparison used as fullQ graph')),
        records=rows)


def raw_graph_certificate(descriptor: dict[str, Any], raw: dict[str, Any],
                          original: dict[str, Any], improved: dict[str, Any],
                          profile: dict[str, Any], band: RotationBand) -> dict[str, Any]:
    """Retained high SOURCE tails and source-oriented all-state quartic domain.

    Original RAW first graphs are admitted, not re-executed. Changed improved
    photon/neutral data and literal range-two terms are pulled through afresh.
    """
    if raw['N'] != NSTAR or raw['source_profile']['tree_max'] != NSTAR//2+60:
        raise ValueError('actual distinct RAW same-N word required')
    if descriptor['N'] != NSTAR or descriptor['physical_family']['division_exponent'] != 4096:
        raise ValueError('actual totalb4096 Nstar family required')
    d = compact_descriptor(descriptor)
    d.update(tau=F(8), mu=F(1294, 10000), B=F(2))
    body = normalized_flight_bound(d, F(6))
    owner = improved['bounds']
    neutral = owner['renewed_actual_neutral']
    values = {k:F(v) for k,v in neutral['graph_values'].items()}
    shift = F(neutral['photon_form_shift_upper'])
    omega = F(neutral['omega_min_lower'])
    j_ground = F(neutral['current_ground_norm_upper'])
    volume = NSTAR**3
    mode_weight = sqrt_upper(F(2*volume)/omega)
    number = UE*mode_weight**2*j_ground**2/omega**2
    birth_norm = sqrt_upper(1+number)
    birth_first = C*sqrt_upper(F(12))*birth_norm+sqrt_upper(UE)*mode_weight*j_ground
    cd, a4 = F(owner['CD_changed']), F(owner['A4_unchanged'])
    birth_rho2 = sqrt_upper(2/a4*(birth_norm*birth_first+(cd+3)*birth_norm**2))
    jets = tuple(F(4)*2**int(q) for q in raw['source_q6_upper_exponents'])
    jets = free_conjugated(jets, F(6), F(16))
    k0, k1, k2 = jets[:3]
    word = NSTAR//2+60; dimension = 75*(41**3+1)*(word+1)
    literal_paths = []
    for q in range(7):
        coefficient = 3*F.from_float(band.v)+6*F.from_float(band.r)
        coefficient += F(3, 2)*F.from_float(band.r)*2**q
        if coefficient > 8*2**q:
            raise AssertionError('literal range-two q<=6 termwise coefficient envelope failed')
        literal_paths.append(dict(order=q, exact_path_coefficient=str(coefficient),
                                  envelope=str(F(8)*2**q)))
    improved_graphs = {}
    for label, n0, q0, rho2 in (('birth', birth_norm, birth_first, birth_rho2),
                              ('cloud', values['n'], values['Q_full'], values['rho2'])):
        half = sqrt_upper(n0*q0+shift*n0*n0)
        kinetic = sqrt_upper(2*n0*q0)+n0
        rho = sqrt_upper(n0*rho2)
        harmonic_first = 2*sqrt_upper(UE/volume)*sqrt_upper(F(3))*k0*word*half
        harmonic_second = 3*UE/volume*k0*word**2*n0
        terms = dict(source_on_actual_Q=k0*q0,
            chi_Pi_product=2*k1*kinetic, all_field_laplacian=dimension*k2*n0,
            literal_changed_Phi_endpoint=36*k0*rho,
            entire_actual_Coulomb=(body['entire_Coulomb_neutral_upper']+body['entire_Coulomb_charged_upper'])*k0*n0,
            harmonic_first=harmonic_first, harmonic_second=harmonic_second,
            actual_ordinary_shift=shift*k0*n0)
        first = cast(F, sum(terms.values()))
        norm = k0*n0
        improved_graphs[label] = dict(norm=ceil_log2(dyadic_upper(norm)),
            first=ceil_log2(dyadic_upper(first)), half=ceil_log2(dyadic_upper(sqrt_upper(norm*first))),
            termwise_log2_upper={k:ceil_log2(dyadic_upper(v)) for k,v in terms.items()},
            exact_first_sum=str(first))
    admitted = original['selected_source_flight']['sources']['RAW41']['graphs']
    original_graphs = {label:{k:int(g[k]['log2_upper']) for k in ('norm','first','half')}
                       for label,g in admitted.items()}
    if [original_graphs[x]['first'] for x in ('birth', 'cloud')] != [972,1617]:
        raise ValueError('original accepted RAW first-source identity changed')
    # Physical first/two-source words use at most B^(3)/B^(4) in Q B_j.
    # Six SOURCE vector jets use B^(6); no unsupported Q B^(6) (B^(8)) claim.
    jetlogs = [ceil_log2(j) for j in jets]
    angular_costs = {str(q):jetlogs[q+2]-jetlogs[2]+q*(64+8)+8
                     for q in range(3)}
    if max(angular_costs.values()) > 1024:
        raise AssertionError('new angular Q-source word geometric reserve exceeded')
    domains = {}
    for law, graphs in (('original', original_graphs), ('improved', improved_graphs)):
        records = {}
        for label, graph in graphs.items():
            records[label] = {}
            for q in range(3):
                first_log = graph['first']+angular_costs[str(q)]
                norm_log = graph['norm']+jetlogs[q]-jetlogs[0]+q*(64+8)+8
                half_log = (first_log+norm_log+1)//2
                cutoff = 64834
                records[label][str(q)] = dict(SOURCE_first_log2_upper=first_log,
                    SOURCE_norm_log2_upper=norm_log,SOURCE_half_log2_upper=half_log,
                    retained_high_SOURCE_norm_log2_upper=first_log-cutoff,
                    retained_high_SOURCE_half_log2_upper=first_log-cutoff//2,
                    invariant_spectral_cutoff_log2=cutoff,
                    high_source_kept_not_filter=True)
        domains[law] = records
    root_extra = 0  # All normalized eta/guard profiles have the SAME unit l2 cap.
    return dict(N=NSTAR,V=volume,UE_exact=str(UE),c_exact=str(C),b_total=4096,
        original_admitted_RAW_graphs=original_graphs, improved_recomputed_RAW_graphs=improved_graphs,
        original_field_and_all_state_quartic_Young_form=True,
        improved_changed_quartic_A4=str(a4), improved_changed_CD=str(cd),
        original_heavy_parameters=BARE, full_all64_E0kappa_offsets_and_AHnull_retained=True,
        raw_profile_q6_exponents=raw['source_q6_upper_exponents'],
        physical_CY=2**64, physical_six_SOURCE_vector_jet_upper_log2=jetlogs[6]+6*(64+8)+8,
        physical_angular_Q_SOURCE_cost_bits=angular_costs, SOURCE_domain_records=domains,
        guarded_profile_normalization=profile['guard_norm'],
        guard_preparation_distance_retained=profile['guard_distance'],
        guard_changes_preparation_not_hardware=True,
        guard_all_normalized_l2_geometry_bound_same=True,
        profile_angular_six_geometry=profile['angular_q6_geometry_upper'],
        literal_improved_paths_q0_q6=literal_paths,
        graph_half_derived_by_Cauchy_Schwarz_not_RBU_inference=True,
        co_moving_clock_inserted=False, Compton_source_word_added_bits=0,
        source19_receiver_and_probability_unchanged=True,
        additional_normal_aggregate_bits=root_extra,
        joined_normal_aggregate_log2_upper=-31850+root_extra,
        contained_EM_log2_upper=-46175+root_extra,
        EM_not_paid_twice=True,
        ordinary_SOURCE_Q_not_nearest_Wilson_comparison=True,
        q6_SOURCE_vector_domain_not_Q_of_sixth_word=True)


def soft_radial_preparation(frame: np.ndarray, band: RotationBand,
                            band_metadata: dict[str, Any], radial_nodes: int = 20) -> dict[str, Any]:
    """A DISTINCT prepared compact-momentum bump, never filtered RAW/SC19.

    The radial bump is centred at (.02,0,0), has radius .01, and is C-infinity
    at its support boundary. Thus every actual momentum has norm below .03.
    """
    centre = np.array([.02, 0., 0.]); width = .01; time = 6.
    xr, wr = np.polynomial.legendre.leggauss(radial_nodes)
    xu, wu = np.polynomial.legendre.leggauss(16)
    azimuth = 2*np.pi*np.arange(32)/32
    radius = width*(xr+1)/2
    rr, uu, aa = np.meshgrid(radius, xu, azimuth, indexing='ij')
    displacement = np.stack((rr*np.sqrt(1-uu*uu)*np.cos(aa),
                             rr*np.sqrt(1-uu*uu)*np.sin(aa), rr*uu), axis=-1).reshape(-1, 3)
    points = centre+displacement
    weights = (wr[:, None, None]*width/2*rr**2*wu[None, :, None]*
               (2*np.pi/32)*np.ones_like(aa)).reshape(-1)
    s = np.sum(displacement**2, axis=1)/width**2
    amplitude = np.exp(-1/(1-s))
    scalar_norm = math.sqrt(float(np.sum(weights*amplitude**2)))
    amplitude /= scalar_norm
    amplitude_gradient = -2*displacement/(width**2*(1-s[:, None])**2)*amplitude[:, None]
    m = source_seed(frame, band)
    rest = physical_rest_spin(frame, band)
    certified = uniform_lattice_identification(frame, band, band_metadata)
    mass = rest['mass']; velocity_bound = band.v*max(np.linalg.norm(f, 2) for f in frame[1:])
    cutoff = float(np.max(np.linalg.norm(points, axis=1)))
    if cutoff >= .03 or cutoff > .05:
        raise AssertionError('new SOFT support leaves the actual lowk ball')
    phase_operator_bound = (velocity_bound*.03**3/6+
        band.r*np.linalg.norm(frame[0], 2)*(.03**4/4 if band.improved else .03**2/2))
    phase_operator_bound += certified['actual_full10_metric_covariance_upper']
    records = []
    for branch in (1, -1):
        seed = m if branch == 1 else m.conj()
        h, velocities, _ = batch_h(points, frame, band, branch)
        values, vectors = np.linalg.eigh(h)
        negative, positive = vectors[:, :, :2], vectors[:, :, 2:]
        P = np.einsum('xai,xbi->xab', positive, positive.conj())
        A = P@seed*amplitude[:, None, None]
        norm2 = float(np.einsum('x,xab,xab->', weights, A.conj(), A).real)
        if norm2 <= 1/8:
            raise AssertionError('SOFT actual occupied-sea global SOURCE norm floor failed')
        A /= math.sqrt(norm2)
        gradients = []
        for j in range(3):
            # Actual spectral Frechet derivative of P+, including both signs.
            vertex = np.einsum('xai,xab,xbj->xij', negative.conj(), velocities[:, j], positive)
            divided = vertex/(values[:, None, 2:]-values[:, :2, None])
            dP = np.einsum('xai,xij,xbj->xab', negative, divided, positive.conj())
            dP += dP.conj().transpose(0, 2, 1)
            gradients.append((dP@seed*amplitude[:, None, None]+
                              (P@seed)*amplitude_gradient[:, j, None, None])/math.sqrt(norm2))
        gradients = np.asarray(gradients)
        energy = values[:, 2]
        group_velocity = np.stack([np.einsum('xai,xab,xbi->x', positive.conj(),
                           velocities[:, j], positive).real/2 for j in range(3)], axis=1)
        current = np.array([np.einsum('x,xab,xab->', weights, A.conj(),
                           velocities[:, j]@A).real for j in range(3)])
        position0 = np.array([np.einsum('x,xab,xab->', weights, A.conj(),
                                     1j*gradients[j]).real for j in range(3)])
        evolved = np.exp(-1j*time*energy)[:, None, None]*A
        evolved_gradients = np.asarray([np.exp(-1j*time*energy)[:, None, None]*
                      (gradients[j]-1j*time*group_velocity[:, j, None, None]*A) for j in range(3)])
        positionT = np.array([np.einsum('x,xab,xab->', weights, evolved.conj(),
                                     1j*evolved_gradients[j]).real for j in range(3)])
        transport_error = check(np.linalg.norm(positionT-position0-time*current),
                                'actually evolved SOFT SOURCE/current position identity', 2e-11)
        orbital = []
        total = []
        spins = physical_rest_spin(frame, band, branch)['spins']
        for axis in range(3):
            angular = np.cross(np.eye(3)[axis], points)
            L = -1j*np.einsum('xj,jxab->xab', angular, gradients)
            orbital.append(float(np.einsum('x,xab,xab->', weights, L.conj(), L).real))
            J = L+spins[axis]@A
            total.append(float(np.einsum('x,xab,xab->', weights, J.conj(), J).real))
        # Compare time evolution with the CURRENT-derived linear Hamiltonian,
        # not an imported Dirac spinor. Actual band preparation remains A.
        phi = frame if branch == 1 else np.concatenate((-frame[:1].transpose(0, 2, 1),
                                                       frame[1:].transpose(0, 2, 1)))
        linear = band.m*phi[0]+np.einsum('xj,jab->xab', band.v*points, phi[1:])
        le, lu = np.linalg.eigh(linear)
        evolved_linear = (lu*np.exp(-1j*time*le)[:, None, :])@lu.conj().transpose(0, 2, 1)@A
        continuum_error = math.sqrt(float(np.einsum('x,xab,xab->', weights,
                        (evolved-evolved_linear).conj(), evolved-evolved_linear).real))
        check(continuum_error, 'SOFT actual-vs-current-linearization evolution', time*phase_operator_bound+1e-12)
        # A Sobolev tail bound for the actual projected/evolved SOURCE, not bare
        # eta alone. Product derivatives retain P+ and the exact time evolution.
        derivative = []
        for q in range(7):
            bound = 0.
            for a in range(q+1):
                for b in range(q-a+1):
                    c = q-a-b
                    eta_bound = math.sqrt(8)*math.exp(4/3)*(64*max(a, 1)/width)**a
                    projector_bound = math.factorial(b)*(16/.1294)**b
                    evolution_bound = math.factorial(c)*(time*16)**c
                    coefficient = math.factorial(q)/(math.factorial(a)*math.factorial(b)*math.factorial(c))
                    bound += coefficient*eta_bound*projector_bound*evolution_bound
            derivative.append(bound*np.linalg.norm(seed)/math.sqrt(norm2))
        inner = 2**24; outer = 2**25; angle = 2**-28
        tail = math.sqrt(3)*(math.sqrt(3)/inner)**6*derivative[6]
        angular_tail = .03*math.sqrt(3)*(math.sqrt(3)/inner)**5*derivative[6]+tail
        control_speed = outer*angle/time+float(np.linalg.norm(current))
        if control_speed >= float(C)/2:
            raise AssertionError('SOFT supported physical control exceeds c/2')
        records.append(dict(branch=branch, actual_CAR_norm_squared=norm2,
            source_globally_normalized=True, prepared_from_filled_literal_negative_band=True,
            predefined_spinor_or_projection_hardware=False,
            actual_current=current, actual_position0=position0, actual_positionT6=positionT,
            actual_displacement=positionT-position0, actual_transport_identity_error=transport_error,
            actual_micro_vs_current_linearization_error=continuum_error,
            evolution_operator_upper=time*phase_operator_bound,
            actual_band_energy_mean=float(np.einsum('x,x,xab,xab->', weights, energy, A.conj(), A).real),
            orbital_axis_squared=orbital, actual_total_J_squared=sum(total),
            arbitrary_totalJ_not_identified_as_rest_Casimir=True,
            source_momentum_support_max=cutoff, full_prepared_momentum_weight_in_lowk=True,
            retained_spatial_tail_norm_upper=tail,
            supported_Noether_angular_tail_upper=angular_tail,
            supported_rotation_difference_upper=angle*angular_tail,
            spatial_tail_not_cut_or_postselected=True,
            support_inner=inner, support_outer=outer, physical_rotation_angle=angle,
            physical_rotation_time=time, physical_coordinate_speed_upper=control_speed,
            reference_Nstar_over2_outside_support=True,
            exact_radial_bump='exp(-1/(1-|p-p0|^2/.01^2)) on |p-p0|<.01; zero outside',
            derivative_q0_q6_norm_upper=derivative,
            native_radial_momentum_nodes=len(points), nodes_are_Bloch_twist_cubature_not_Nstar_allocation=True,
            actual_lowk_claim=True,
            original_photon_cone_match_claimed=bool(band.improved),
            current_rest_Casimir_error=rest['rest_Casimir_error'], measured_g=rest['g'],
            current_identification_uniform_upper=certified['physical_current_spin_identification_upper'],
            actual_microscopic_gap_lower=certified['actual_microscopic_mass_lower'],
            full10_metric_operator_debt=certified['actual_full10_metric_covariance_upper'],
            conditional_probability_density_upper=8*np.linalg.norm(seed)**2*(4*math.pi*width**3/3)/(2*math.pi)**3,
            normal_transfer_log2_upper=-31850+2048))
    # Independent primitive Root implementation of the same supported-band CAR.
    primitive = [soft_primitive_car(frame, band, branch) for branch in (1, -1)]
    # Uniform in real Bloch twist: normalized grid sums of a C-infinity bump.
    # A fixed continuum scalar amplitude is used, then ONE joint source norm;
    # there is no per-fibre normalization or postselection.
    momentum_measure = (2*math.pi)**3
    scalar_integral_lower = (4*math.pi/3)*(width/2)**3*math.exp(-8/3)/momentum_measure
    scalar_integral_upper = (4*math.pi/3)*width**3*math.exp(-2)/momentum_measure
    source_integral_lower = scalar_integral_lower/8
    lattice_sum_error = 6*math.pi/NSTAR*(256/width+16/.1294)*np.linalg.norm(m)**2
    relative_grid_error = lattice_sum_error/source_integral_lower
    if relative_grid_error >= 1e-12:
        raise AssertionError('legal Nstar SOFT SOURCE normalization quadrature admission failed')
    return dict(distinct_SOFT_not_RAW_or_SC19=True, parameters=asdict(band),
                momentum_centre=centre, momentum_radius=width,
                all_momentum_norm_upper=.03, scalar_preparation_norm=scalar_norm,
                branches=records, primitive_supported_Bloch_source=primitive,
                legal_Nstar_source_integral_lower=source_integral_lower,
                scalar_integral_lower=scalar_integral_lower,scalar_integral_upper=scalar_integral_upper,
                legal_uniform_twist_SOURCE_grid_error_upper=lattice_sum_error,
                legal_uniform_twist_SOURCE_relative_grid_error_upper=relative_grid_error,
                exact_preparation='w((2*pi*n+kappa)/N-p0) in mechanical momentum; one joint global CAR norm',
                paired_noncompact_kappa_and_harmonic_reference_phase_retained=True,
                per_fibre_or_reference_renormalization=False)


def soft_primitive_car(frame: np.ndarray, band: RotationBand, branch: int) -> dict[str, Any]:
    """Actual full16-mode Root CAR sea/source; three supported Bloch cells."""
    sector = PrimitiveSector(10 if branch == 1 else 6)
    points = np.array([[.02, 0., 0.], [.018, .002, 0.], [.023, -.001, -.003]])
    h, vertices, _ = batch_h(branch*points, frame, band)
    law = ImprovedPhiLaw(3, band)
    colour = [expm(1j*.04*(j+1)*law.generators[3+j]) for j in range(3)]
    reference = expm(.037j*law.generators[11])
    seed = source_seed(frame, band)
    weights = np.array([1., math.exp(-.08), math.exp(-.19)])
    weights /= np.linalg.norm(weights)
    onebody = sparse.block_diag([u@a@u.conj().T for u,a in zip(colour,h)]).toarray()
    values, modes = np.linalg.eigh(onebody)
    occupied = modes[:, :6]
    omega = sector.neutral(occupied, branch)
    coefficient = np.concatenate([weights[j]*u@seed@reference.conj().T for j,u in enumerate(colour)])
    B = sector.source(coefficient, branch)
    source = B@omega
    norm = float(np.linalg.norm(source)); source /= norm; B /= norm
    P = occupied@occupied.conj().T
    allowed = np.eye(12)-P if branch == 1 else P
    norm_error = check(abs(norm**2-np.trace(coefficient.conj().T@allowed@coefficient).real),
                       'SOFT actual primitive occupied/unoccupied CAR norm')
    H = sector.lift(sparse.block_diag((onebody, np.zeros((4, 4)))).toarray())
    H += .7*(sector.identity-sector.cycle)/2
    physical_spins = physical_rest_spin(frame, band)['spins']
    spins = []
    for axis in range(3):
        spin = sparse.block_diag([u@physical_spins[axis]@u.conj().T for u in colour]+[np.zeros((4, 4))]).toarray()
        spins.append(sector.lift(spin))
    true_second = np.zeros_like(source)
    neutral_noise = []
    for S in spins:
        true_second += S@(S@(B@omega))-2*S@(B@(S@omega))+B@(S@(S@omega))
        neutral_noise.append(float(np.linalg.norm(S@omega)))
    second_error = check(np.linalg.norm(true_second-.75*source), 'SOFT true primitive nested SOURCE Casimir')
    # A real small supported rotation over T6, not an internal-only endpoint.
    from scipy.spatial.transform import Rotation
    angle = 2**-28; time = 6.; speed = angle/time
    R = Rotation.from_rotvec(np.array([0., 0., angle])).as_matrix()
    rotated = points@R.T
    hp, _, _ = batch_h(branch*rotated, frame, band)
    d = expm(-1j*angle*physical_spins[2])
    rotated_h = sparse.block_diag([u@a@u.conj().T for u,a in zip(colour,hp)]).toarray()
    conjugated = sparse.block_diag([u@d@a@d.conj().T@u.conj().T for u,a in zip(colour,h)]).toarray()
    covariance = float(np.linalg.norm(rotated_h-conjugated, 2))
    micro = sector.lift(sparse.block_diag((rotated_h, np.zeros((4, 4)))).toarray())
    micro += .7*(sector.identity-sector.cycle)/2+speed*spins[2]
    free = expm_multiply(-1j*time*H, source, traceA=-1j*time*complex(H.diagonal().sum()))
    moved = expm_multiply(-1j*angle*spins[2], free,
                         traceA=-1j*angle*complex(spins[2].diagonal().sum()))
    conH = sector.lift(sparse.block_diag((conjugated, np.zeros((4, 4)))).toarray())
    conH += .7*(sector.identity-sector.cycle)/2+speed*spins[2]
    residual = float(np.linalg.norm((micro-conH)@moved))
    check(residual, 'SOFT actual moving micro/current Hamiltonian covariance', 10*covariance+1e-12)
    bdot_word = -speed*(spins[2]@B-B@spins[2])
    bdot_image = float(np.linalg.norm(bdot_word@omega))
    currents = []
    for axis in range(3):
        vm = sparse.block_diag([u@v[axis]@u.conj().T for u,v in zip(colour,vertices)]+[np.zeros((4, 4))]).toarray()
        J = sector.lift(vm)
        currents.append(float((np.vdot(source, J@source)-np.vdot(omega,J@omega)).real))
    if max(neutral_noise) <= 1e-8 or max(abs(x) for x in currents) <= 1e-6:
        raise AssertionError('SOFT primitive neutral/current observation is vacuous')
    observers = [lowk_little_group(frame,band,p,R,branch) for p in points]
    return dict(branch=branch, actual_momenta=points, full_primitive_dimension=sector.dimension,
        allRef4_raw_input=True, onceprepared_global_norm=norm,
        actual_primitive_CAR_norm_error=norm_error, true_nested_SOURCE_Casimir_error=second_error,
        actual_nonzero_neutral_spin_noise=neutral_noise, actual_relative_current=currents,
        physical_coordinate_momenta_rotated=rotated, physical_rotation_angle=angle,
        actual_moving_micro_equation_residual=residual, lattice_covariance_error=covariance,
        actual_little_group_spinor_error=max(x['physical_spinor_error'] for x in observers),
        actual_little_group_projector_upper=max(x['projector_upper'] for x in observers),
        literal_minus_i_Bdot_image=bdot_image, neutral_rightaction_retained=True,
        reference_fixed=True, native_is_conditional_meanPhi_supported_Bloch_witness=True,
        native_not_full_quantum_field_clock_or_Nstar=True)


def soft_domain_ledger(raw_graph: dict[str, Any]) -> dict[str, Any]:
    """Distinct extended source/weak rotation costs; no SC19 reserve borrowing."""
    inner, outer = 2**24, 2**25
    word = 3*NSTAR
    raw_dimension = 75*(41**3+1)*(NSTAR//2+61)
    soft_dimension = 75*(NSTAR**3+1)*(word+1)
    dimension_cost = ceil_log2(F(soft_dimension, raw_dimension))
    source_cost = dimension_cost+16+8  # new word<=6 oldword; two field jets.
    CY_log = 64+4*(25-6)
    # Six primitive-flow coefficients plus active volume, word and source chart.
    extra_geometry = 6*(CY_log-64)+3*(25-6)+48+128
    if extra_geometry >= 2048:
        raise AssertionError('SOFT actual stress/source geometry reserve exceeded')
    records = {}
    for law,key in (('original','original_admitted_RAW_graphs'),('improved','improved_recomputed_RAW_graphs')):
        records[law] = {}
        for label,g in raw_graph[key].items():
            first = g['first']+source_cost
            records[law][label] = dict(SOURCE_first_log2_upper=first,
                SOURCE_half_log2_upper=(g['norm']+8+first+1)//2,
                retained_high_half_log2_upper=first-64834//2,
                cutoff_log2=64834, actual_high_source_retained=True)
    return dict(distinct_source='SOFT compactmomentum / extendedspatial source',
        full_spatial_support_bound=NSTAR**3, source_word_upper=word,
        word_ratio_to_RAW_upper=6, actual_source_q6_added_word_bits=[8*q for q in range(7)],
        source_l2_global_normalization_not_l1_replacement=True,
        dimension_added_bits=dimension_cost, SOURCE_first_added_bits=source_cost,
        supported_rotation_inner=inner, supported_rotation_outer=outer,
        physical_CY_log2_upper=CY_log, additional_geometry_bits=extra_geometry,
        additional_normal_stress_reserve_bits=2048,
        original_and_improved_SOURCE_domains=records,
        joined_root_normal_log2_upper=-31850+2048,
        contained_EM_log2_upper=-46175+2048, EM_not_paid_twice=True,
        original_heavy_allstate_quartic_and_statistical_hypotheses_retained=True,
        all64_E0kappa_and_AHnull_retained=True,
        SOFT_inserts_no_Compton_pulse=True, additional_Compton_word_bits=0,
        SC19_original_improved_words_unchanged=[-4970,-5334],
        new_SOFT_costs_not_assumed_paid_by_SC19_150bits=True)


def radial_receipt_contract(receipt: dict[str, Any], descriptor: dict[str, Any],
                            band_metadata: dict[str, Any]) -> dict[str, Any]:
    """Recompute cheap scientific selectors/debts from the completed real receipt."""
    p = receipt['profile']
    if (p['profile_N'],p['profile_radius'],p['actual_count'],p['screen_eta'],p['source_word']) != (41,20,68921,18,NSTAR//2+60):
        raise ValueError('terminal radial consumer is not the actual distinct RAW41 family')
    if p['six_halo_radius'] >= 48 or p['CY'] != 2**64 or p['alpha'] != 2**371:
        raise ValueError('terminal RAW halo/CY/alpha contract changed')
    family = receipt['physical_family']
    if family['N'] != NSTAR or family['total_weak_bits'] != 4096 or F(family['UE']) != UE or F(family['c']) != C:
        raise ValueError('terminal radial physical family mismatch')
    if int(descriptor['N']) != NSTAR or descriptor['physical_family']['division_exponent'] != 4096:
        raise ValueError('joined descriptor mismatch')
    if not (band_metadata['calibration']['all64_centres'] and band_metadata['calibration']['all_real_shifts']):
        raise ValueError('actual band all64/realshift metadata missing')
    native = receipt['actual_original_and_improved_native']
    if {(x['improved'],x['branch']) for x in native}!={(i,b) for i in (False,True) for b in (1,-1)}:
        raise ValueError('actual original/improved radial particle/hole evidence missing')
    for x in native:
        check(x['true_physical_first_SOURCE_error'],'terminal physical RAW sourcefirst',3e-6)
        check(x['true_radial_second_SOURCE_error'],'terminal literal RAW sourcesecond',8e-6)
        check(x['actual_LIGHT_neutral_Leibniz_error'],'terminal actual light neutral',3e-6)
        if min(x['nonzero_field_neutral_noise'],x['nonzero_EM_neutral_noise'],x['actual_radial_current_response_max'])<=0:
            raise ValueError('terminal radial current/neutral observations erased')
    soft = receipt['soft_radial_preparation']
    if len(soft)!=2 or not all(x['distinct_SOFT_not_RAW_or_SC19'] for x in soft):
        raise ValueError('distinct actuallyprepared original/improved SOFT missing')
    for x in soft:
        for b in x['branches']:
            if b['source_momentum_support_max']>.05 or not b['full_prepared_momentum_weight_in_lowk']:
                raise ValueError('prepared SOFT leaves physical lowk support')
            if b['physical_coordinate_speed_upper']>=float(C)/2:
                raise ValueError('supported SOFT clock exceeds physical c/2')
            check(b['actual_transport_identity_error'],'terminal actually evolved soft current',2e-11)
            if np.linalg.norm(b['actual_displacement'])<=0 or b['retained_spatial_tail_norm_upper']<=0:
                raise ValueError('soft motion/tails replaced by zero')
    recomputed = soft_domain_ledger(receipt['large_Nstar_SOURCE_domains'])
    if recomputed != receipt['soft_SOURCE_and_geometry_ledger']:
        raise ValueError('terminal SOFT source/geometry ledger does not recompute')
    return dict(profile='RAW41 distinct preparation; SOFT distinct actual lowk preparation',
        actual_native_branches=4, scoped_continuum='SOFT improved actual c; RAW broadBZ, original photon cone mismatch retained',
        original_RAW_first_logs=[972,1617],
        improved_RAW_first_logs=[receipt['large_Nstar_SOURCE_domains']['improved_recomputed_RAW_graphs'][k]['first'] for k in ('birth','cloud')],
        SOFT_source_added_bits=recomputed['SOURCE_first_added_bits'],
        SOFT_normal_stress_added_bits=2048, joined_root_log2=recomputed['joined_root_normal_log2_upper'],
        contained_EM_log2=recomputed['contained_EM_log2_upper'],
        additional_Compton_word_bits=0, source19_receiver_or_probability_retagged=False)


def encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return dict(real=value.real.tolist(), imag=value.imag.tolist())
        return value.tolist()
    if isinstance(value, complex): return dict(real=value.real, imag=value.imag)
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, F): return str(value)
    raise TypeError(type(value).__name__)


def consumed_api_sha256() -> dict[str, str]:
    apis = (PrimitiveSector,native_onebody,native_source_coefficient,primitive_gauss_source_probe,
            ImprovedPhiLaw,RotationBand,physical_h,selected_band,radial_profile,
            frame_chart,anchored_coefficient,derived_spins,frozen_intrinsic_source_word,
            gauss_family,spatial_source_identification,physical_rest_spin,lowk_little_group,uniform_lattice_identification,
            field_noether_galerkin,scalar_orbital,local_potential,sqrt_upper,
            compact_descriptor,normalized_flight_bound,free_conjugated)
    return {f'{f.__module__}.{f.__name__}':hashlib.sha256(inspect.getsource(f).encode()).hexdigest()
            for f in apis}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--descriptor',type=Path,required=True)
    parser.add_argument('--raw-receipt',type=Path,required=True)
    parser.add_argument('--original-source-receipt',type=Path,required=True)
    parser.add_argument('--improved-band-receipt',type=Path,required=True)
    parser.add_argument('--intrinsic-receipt',type=Path,required=True)
    parser.add_argument('--spatial-packet',type=Path,required=True)
    parser.add_argument('--band-metadata',type=Path,required=True)
    parser.add_argument('--quadrature',type=int,default=129)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    inputs = {k:v for k,v in vars(args).items() if isinstance(v,Path) and k!='output'}
    before = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.values()}
    descriptor = decode(json.loads(args.descriptor.read_text()))
    raw = json.loads(args.raw_receipt.read_text())
    original = json.loads(args.original_source_receipt.read_text())
    improved = json.loads(args.improved_band_receipt.read_text())
    metadata = json.loads(args.band_metadata.read_text())
    intrinsic = json.loads(args.intrinsic_receipt.read_text())
    spatial = json.loads(args.spatial_packet.read_text())
    for name, value in spatial['owned_source_pair_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=value:
            raise ValueError('frozen Spatial source bytes changed')
    light,frame,calibration = selected_band()
    original_band = RotationBand(**BARE)
    profile = radial_profile_data(original_band)
    graphs = raw_graph_certificate(descriptor,raw,original,improved,profile,light)
    soft = [soft_radial_preparation(frame,band,metadata) for band in (original_band,light)]
    band_records = []
    native = []
    for band in (original_band,light):
        band_records.append(radial_band_preparation(frame,band,profile,args.quadrature))
        for branch in (1,-1):
            chart = native_radial_source_chart(frame,band,profile,branch)
            record = radial_noether_certificate(chart,frame,band,profile,branch,intrinsic)
            record['improved'] = band.improved
            native.append(record)
    profile_summary = {k:v for k,v in profile.items() if k not in (
        'eta','guard','guarded_eta','displacement','halo_eta','halo_orbital_first','halo_orbital_second')}
    result = dict(plan=PLAN,profile=profile_summary,physical_family=dict(N=NSTAR,V=NSTAR**3,
        total_weak_bits=4096,UE=str(UE),c=str(C),K=str(C*C/(2*UE)),source='RAW41 NOT SOURCE19'),
        calibration=calibration,actual_radial_band_preparations=band_records,
        actual_original_and_improved_native=native,large_Nstar_SOURCE_domains=graphs,
        soft_radial_preparation=soft,soft_SOURCE_and_geometry_ledger=soft_domain_ledger(graphs),
        scoped_REST_current_observer=[{k:v for k,v in physical_rest_spin(frame,band).items()
            if k in ('mass','speed','g','rest_Casimir_error','current_identity_error','executed_cone_error')}
            for band in (original_band,light)],
        complete_source19_TWO_receiver_or_P_relabelled=False,completed_supplier_leaf_rerun=False,
        source_pair_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ('radial_band_noether.py','radial_band_noether.md')},
        consumed_api_sha256=consumed_api_sha256(),input_sha256=before)
    result['terminal_radial_contract'] = radial_receipt_contract(result,descriptor,metadata)
    after = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.values()}
    if before != after: raise RuntimeError('radial scientific inputs changed in flight')
    args.output.write_text(json.dumps(result,default=encode,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),native_branches=len(native),
        max_SOURCE_first_error=max(x['true_physical_first_SOURCE_error'] for x in native),
        max_SOURCE_second_error=max(x['true_radial_second_SOURCE_error'] for x in native),
        actual_light_sea=True,extra_Compton_word_bits=0)))


if __name__ == '__main__':
    main()
