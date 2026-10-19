#!/usr/bin/env python3
"""Compact full-coordinate Gauss packet: analytic variational bound, not a ground."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[name] = '1'

import numpy as np
from scipy.linalg import expm
from scipy.special import beta, gammaln

from full_band_gauss_section import FullBandGaussSection, source_hashes as section_hashes
from local_orbital_stability import uniform_data
from quantum_su4_gauss import norm, require_small
from rotation_band import RotationBand
from sea_selection import matrix_unit_basis


def source_hashes():
    root = Path(__file__).resolve().parent
    return {**section_hashes(), **{name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ('full_band_packet_bound.py', 'full_band_packet_bound.md',
                         'gauss_born_bundle.md', 'full_field_charge_bound.md')}}


def bump(value, radius, power=2):
    """Real compact H1 amplitude; value is squared distance, not distance."""
    return max(0., 1. - value / radius**2)**power


def haar_link_normalizer(eta, order):
    """Deterministic SU4 Weyl integral estimate, NOT a uniform certificate.

    Exact Z is the displayed integral in the proof. Rescaling retains its tiny
    eta^15 value without subtracting nearly equal traces at the identity.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    limit = 2 * np.arcsin(eta / 2)
    xyz = np.stack(np.meshgrid(nodes, nodes, nodes, indexing='ij'), axis=-1)
    angles = limit * np.concatenate((xyz, -xyz.sum(axis=-1, keepdims=True)), axis=-1)
    q = np.sum(4 * np.sin(angles / 2)**2, axis=-1)
    vandermonde = np.ones(q.shape)
    for i in range(4):
        for j in range(i + 1, 4):
            vandermonde *= 4 * np.sin((angles[..., i] - angles[..., j]) / 2)**2
    integrand = np.maximum(0., 1. - q / eta**2)**4 * vandermonde
    w = np.einsum('i,j,k->ijk', weights, weights, weights)
    return float(limit**3 * np.sum(w * integrand) / (24 * (2 * np.pi)**3))


def log_seed_density(fields, U, phases, anchor_fields, anchor_U, anchor_phases,
                     R, eta, tau, link_Z):
    """Log normalized seed with the EXACT Haar link normalizer.

    The physical density is the gauge Haar average specified in the proof.
    Log form avoids underflow of the 120-dimensional compact normalizer.
    """
    d = np.einsum('xmij,aji->xma', fields - anchor_fields, matrix_unit_basis()).real / 4
    wrapped = (np.asarray(phases) - anchor_phases + np.pi) % (2 * np.pi) - np.pi
    values = [bump(float(np.sum(d**2)), R), bump(norm(U - anchor_U)**2, eta),
              *(bump(float(a*a), tau) for a in wrapped)]
    if min(values) <= 0:
        return -np.inf
    log_field_Z = 60 * np.log(np.pi) + 120 * np.log(R) + gammaln(5) - gammaln(65)
    phase_Z = tau * beta(.5, 5) / (2 * np.pi)
    return float(2 * np.log(values).sum() - log_field_Z - np.log(link_Z)
                 - 2 * np.log(phase_Z))


def anchor_probe(model, fields, U, phases, kappa, J):
    h = model.onebody(fields, U, phases)
    grams = np.einsum('xmij,xnji->xmn', fields, fields).real / 4
    W = norm(fields[0] - U @ fields[1] @ U.conj().T)**2
    elastic = kappa * float(np.sum(grams**2)) / 2
    alignment = J * W / 8
    heavy = -float(np.linalg.eigvalsh(h[0])[4:].sum())
    light = float(np.linalg.eigvalsh(h[1])[:4].sum())
    return dict(elastic=elastic, exact_full_field_W=W, alignment=alignment,
                whole_heavy_occupied_coefficient=heavy, light_occupied_energy=light,
                leading_coefficient=elastic + alignment + heavy)


def packet_point_controls(model, fields, U, phases):
    """Actual primitive Gauss and energy probe, not a uniform certificate."""
    section = model.state(fields, U, phases)
    psi = section.state
    colour, em, energies = [], [], []
    for site in range(2):
        for a, t in enumerate(model.generators):
            df = np.zeros_like(fields)
            df[site] = 1j * (t @ fields[site] - fields[site] @ t)
            du = 1j * t @ U if site == 0 else -1j * U @ t
            colour.append(norm(model.derivative(section, fields, U, phases, df, du)
                               - 1j * (model.matter.Q[site][a] @ psi)))
        direction = np.array([1., -1.]) * (1 if site == 0 else -1)
        em.append(norm(model.derivative(section, fields, U, phases, dphases=direction)
                       - 1j * ((model.number[1][site] - 2 * model.identity) @ psi)))
    for species, sign in ((0, -1), (1, 1)):
        operator = sign * model.lift(section.onebody[species], species)
        eigenvalue = float(sign * section.spectra[species][0][section.spectra[species][2]].sum())
        residual = norm(operator @ psi - eigenvalue * psi)
        energies.append(dict(occupied_energy=eigenvalue, full_primitive_H_residual=residual))
        require_small(residual, 'full occupied packet primitive energy', 3e-8)
    require_small(max(colour + em), 'packet colour and EM Gauss', 3e-8)
    return dict(projection_weight=section.weight, all30_colour_Gauss=colour,
                both_EM_Gauss=em, occupied_energies=energies)


def packet_bound(model, fields, U, phases, *, kappa=2., J=.2, ue4=.3,
                 ue1=.4, magnetic=.7, R=.001, eta=.001, tau=.001) -> dict[str, Any]:
    """Uniform chart/energy constants; no sampled extrema enter the proof."""
    if not all(np.isfinite(x) for x in (kappa, J, ue4, ue1, magnetic, R, eta, tau)):
        raise ValueError('Finite law coefficients and radii required')
    if min(kappa, ue4, ue1, R, eta, tau) <= 0 or min(J, magnetic) < 0:
        raise ValueError('Positive stiffness/electric/radii and nonnegative J/K required')
    if eta >= .1 or tau >= .1:
        raise ValueError('The certified injective near-identity support requires eta,tau<.1')
    section = model.state(fields, U, phases)
    if norm(fields[0] - fields[1]) > 1e-12:
        raise ValueError('This compact formula requires identical endpoint anchor fields')
    probe = anchor_probe(model, fields, U, phases, kappa, J)
    local_gap0 = min(s[1][-1] for s in section.local)
    global_gap0 = min(s[-1] for s in section.spectra)
    onsite = max(abs(x) for x in model.onsite)
    path = max(float(np.linalg.norm(x)) for x in model.paths)
    endpoint = max(norm(np.einsum('m,mij->ij', p, fields[x]))
                   for p in model.paths for x in range(2))
    field_dh = 2 * (onsite + path)
    delta = field_dh * R + endpoint * (eta + tau)
    local_gap, global_gap = local_gap0 - 4 * R, global_gap0 - 2 * delta
    if min(local_gap, global_gap) <= model.tolerance:
        raise ValueError('Failed uniform Weyl gap inequality')
    raw_change = 96 * R / local_gap + 16 * delta / global_gap
    raw_lower = np.sqrt(section.weight) - raw_change
    if raw_lower <= np.sqrt(model.tolerance):
        raise ValueError('Failed uniform positive-overlap inequality sqrt(w0)>D_raw')
    wmin = float(raw_lower**2)
    rho = norm(fields) / 2
    grams = np.einsum('xmij,xnji->xmn', fields, fields).real / 4
    gram_bound = norm(grams) + 2 * rho * R + R**2
    alignment_root = np.sqrt(probe['exact_full_field_W']) + 2 * np.sqrt(2) * R + 2 * norm(fields[1]) * eta
    elastic_bound = kappa / 2 * gram_bound**2
    alignment_bound = J / 8 * alignment_root**2
    heavy_bound = probe['whole_heavy_occupied_coefficient'] + 4 * delta
    C = float(elastic_bound + alignment_bound + heavy_bound)
    # Projector derivative <=2||dh||/gap; actual exterior-projector derivatives
    # <=N||dP||. Two local sqrt(6) factors cost at most 6, never frozen.
    dfield = 2 / np.sqrt(wmin) * (48 / local_gap + 16 * field_dh / global_gap)
    endpoint_support = endpoint + 2 * path * R
    dlink = 2 / np.sqrt(wmin) * (16 * endpoint_support / (np.sqrt(2) * global_gap))
    dphase = 2 / np.sqrt(wmin) * (4 * endpoint_support / global_gap)
    field_fisher = 5120 / R**2
    jacobian_lower = float((np.sin(eta) / eta)**12)
    link_fisher = 8 / eta**2 * np.pi**15 / ((3 / 4)**4 * jacobian_lower)
    phase_fisher = 6 / tau**2
    field_metric = 120 * dfield**2
    colour_metric = 15 * dlink**2
    em_metric = 2 * dphase**2
    T = float((field_fisher + field_metric) / 2)
    light_upper = probe['light_occupied_energy'] + 4 * delta
    B = float(ue4 * (link_fisher + colour_metric)
              + ue1 / 2 * (phase_fisher + em_metric) + magnetic + max(0., light_upper))
    threshold = None
    if C < 0:
        y = (T + np.sqrt(T*T + 4 * (-C) * B)) / (2 * (-C))
        threshold = float(max(1., y*y))
    return dict(anchor=probe, radii=dict(R=R, eta=eta, tau=tau),
        anchor_projection_weight=section.weight, endpoint_HS_bound=endpoint,
        owner_onsite=model.onsite, owner_path_norm=path, field_onebody_derivative_bound=field_dh,
        onebody_uniform_change=delta, local_anchor_gap=local_gap0, global_anchor_gap=global_gap0,
        certified_local_gap=local_gap, certified_global_gap=global_gap,
        raw_uniform_change=raw_change, certified_projection_weight=wmin,
        gram_Frobenius_upper=gram_bound, elastic_upper=elastic_bound,
        alignment_HS_root_upper=alignment_root, alignment_upper=alignment_bound,
        whole_heavy_upper=heavy_bound, C_Lambda=C,
        section_derivative_bounds=dict(field_each=dfield, colour_each=dlink, phase_each=dphase),
        envelope_fisher=dict(fields=field_fisher, colour=link_fisher, phases=phase_fisher),
        section_metric_upper=dict(fields=field_metric, colour=colour_metric, phases=em_metric),
        Haar_exponential_Jacobian_lower=jacobian_lower, T_sqrt_Lambda=T, B_constant=B,
        bound='E0 <= Etrial <= C*Lambda + T*sqrt(Lambda) + B; Lambda>=1',
        strict_negative_for_Lambda_greater_than=threshold,
        negative_energy_at_four_times_threshold=(None if threshold is None else
            float(C * 4 * threshold + T * np.sqrt(4 * threshold) + B)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    for name, default in (('m', .2), ('r', .12), ('v', .3), ('stiffness', 2.),
                          ('Jpref', .2), ('ue4', .3), ('ue1', .4), ('magnetic', .7),
                          ('field-radius', .001), ('link-radius', .001), ('phase-radius', .001)):
        parser.add_argument('--' + name, type=float, default=default)
    args = parser.parse_args()
    identities = source_hashes()
    band = RotationBand(a=1., m=args.m, r=args.r, v=args.v, improved=False)
    model = FullBandGaussSection(band, heavy_scale=1.)
    fields = np.tile(uniform_data(band, 3, args.stiffness, 0., model.basis)['frame'], (2, 1, 1, 1)).astype(complex)
    generator = np.einsum('a,aij->ij', .13 * np.sin(np.arange(15) * .37 + .29), model.generators)
    U, phases = expm(1j * generator), np.array([.137, -.081])
    first = anchor_probe(model, fields, U, phases, args.stiffness, args.Jpref)
    if first['leading_coefficient'] >= 0:
        # A material onsite-only anchor change, never a bare-coefficient change.
        scale = (abs(model.onsite[0]) / args.stiffness)**(1 / 3)
        fields = np.zeros_like(fields)
        fields[:, 0] = scale * np.diag([-1., -1., 1., 1.])
        route = 'material_onsite_only_anchor_after_selected_leading_obstruction'
    else:
        route = 'selected_frame_survives_actual_full_field_leading_probe'
    bound = packet_bound(model, fields, U, phases, kappa=args.stiffness, J=args.Jpref,
                         ue4=args.ue4, ue1=args.ue1, magnetic=args.magnetic,
                         R=args.field_radius, eta=args.link_radius, tau=args.phase_radius)
    if bound['C_Lambda'] >= 0:
        raise ValueError('Uniform packet leading coefficient is not negative; no negative-energy receipt')
    # A deformed point is an independent actual-section consumer, not the
    # source of any uniform gap, overlap, or energy estimate.
    deformation = np.einsum('xma,aij->xmij', .25 * args.field_radius / np.sqrt(120)
                           * np.sin(np.arange(120).reshape(2, 4, 15) * .43 + .17), model.basis)
    controls = packet_point_controls(model, fields + deformation, U, phases)
    normalizers = [dict(order=n, estimate=haar_link_normalizer(args.link_radius, n)) for n in (12, 20)]
    if any(z['estimate'] <= 0 for z in normalizers):
        raise AssertionError('Positive-control Haar Weyl quadrature failed')
    report = dict(status='normalized_compact_full_coordinate_Gauss_trial_bound_NOT_ground_or_phase',
        source_sha256=identities, parameters={k: v for k, v in vars(args).items() if k != 'output'},
        route=route, first_same_law_anchor_probe=first, quantitative_trial=bound,
        packet=dict(field_coordinates=120, colour_measure='normalized genuine SU4 Haar',
            phase_measure='two normalized compact circles', scalar_density='SU4^2 x U1^2 Haar average of normalized compact product seed',
            scalar_amplitude='positive square root of averaged density', section='actual 4900-state FullBandGaussSection.state',
            field_normalizer='pi^60 R^120 Gamma(5)/Gamma(65)',
            each_phase_normalizer='tau B(1/2,5)/(2*pi)',
            link_normalizer='exact 3-angle SU4 Weyl integral in proof; quadrature below is diagnostic only'),
        Haar_normalizer_quadrature_NOT_certificate=normalizers, actual_section_controls=controls,
        limits=['two cells, one genuine colour link, two compact parallel EM paths; not a 3D theorem',
                'Cartesian field kinetic replaces rather than adds a selected-orbit inertia rotor',
                'uniform estimates are analytic; finite controls and cubature do not certify extrema',
                'negative upper trial energy is not a solved full ground, Clifford phase, charged lower gap or mechanical spin',
                'large conservative Lambda threshold; no claim that Lambda=1 has a full-field negative ground'])
    if source_hashes() != identities:
        raise RuntimeError('Scientific source changed during execution; refusing mixed-source receipt')
    payload = json.dumps(report, indent=2, allow_nan=False)
    args.output.write_text(payload + '\n')
    print(payload)


if __name__ == '__main__':
    main()
