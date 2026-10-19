#!/usr/bin/env python3
"""All-centre/all-momentum original-source strong normal-gap bounds.

The analytic Schur theorem uses exact positive constants. Their floating
estimates here are diagnostics, not outward-certified onset parameters.
No full quantum ground or low-energy operator embedding is computed.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'

import numpy as np
from scipy.linalg import null_space

from bulk_quantum_selection import BulkQuantumLaw
from local_orbital_stability import periodic_grid
from original_charged_principal import (
    OriginalSpatial, charge_form, check, original_frame, plaquette_hessian,
    source_hashes as principal_hashes,
)
from rotation_band import RotationBand
from sea_selection import matrix_unit_basis, rank_report
from unrestricted_field_minimum import BARE, STIFFNESS, coordinates

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-continuing-backward-20260930'
PLAN_REVISION = '11e9251a60596401163092e5b39958a6ea55866301fed44e3b9fcd0a0370e31b'


def source_hashes():
    return {**principal_hashes(), **{
        name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('original_normal_gap_bound.py', 'original_normal_gap_bound.md')
    }}


def leading_norm_bound(band, G) -> dict[str, Any]:
    """Uniform in q and centre shift, for the native metric-Clifford source."""
    maximum_x_squared = (abs(band.m)+6*abs(band.r)/band.a)**2+3*(band.v/band.a)**2
    gap = 2*abs(band.m)*math.sqrt(float(np.linalg.eigvalsh(G)[0]))
    if gap <= 0:
        raise ValueError('A positive original whole-heavy cluster gap is required')
    F_squared_sum = (3*band.r**2*G[0, 0]+band.v**2*np.trace(G[1:, 1:]))/band.a**2
    vertex_trace_bound = 60*maximum_x_squared+30*F_squared_sum
    contact_norm_bound = math.sqrt(5400*(band.r**2+band.v**2)/band.a**2+450*F_squared_sum)
    elastic_norm_bound = 6*(STIFFNESS/band.a)*float(np.linalg.eigvalsh(G)[-1])
    return dict(heavy_gap_lower=gap, elastic_norm_upper=elastic_norm_bound,
        sea_norm_upper=float(vertex_trace_bound/gap), contact_norm_upper=contact_norm_bound,
        total=float(elastic_norm_bound+vertex_trace_bound/gap+contact_norm_bound),
        scope='analytic upper formula evaluated in floating arithmetic; no interval onset certificate')


def reduced_matrices(model, q, strong, leading=None):
    G = model.gauge_columns(q)
    form = charge_form(G, 1., .3)
    if form['rank'] != 15:
        raise AssertionError('The regular Clifford frame must retain all15 Gauss tangents')
    Z = null_space((np.sqrt(form['mass'])[:, None]*G).conj().T, rcond=1e-12)
    if Z.shape != (105, 90):
        raise AssertionError('Complete physical normal dimension is90, not a subblock')
    root = np.sqrt(form['a'])
    transform = root[:, None]*Z
    B = transform.conj().T@strong@transform
    B = (B+B.conj().T)/2
    A = None if leading is None else transform.conj().T@leading@transform
    if A is not None:
        A = (A+A.conj().T)/2
    check(float(np.linalg.norm(strong@G)), 'actual all-momentum strong gauge Hessian', 3e-8)
    return B, A


def peierls_derivatives(law, fields, links, phases, direction):
    """Exact current/contact of the native endpoint, including full Gram."""
    first = np.zeros((4*law.volume, 4*law.volume), complex)
    second = np.zeros_like(first)
    for x in range(law.volume):
        sx = slice(4*x, 4*x+4)
        for axis, scalar in law.paths:
            y = law.target[x, axis]
            sy = slice(4*y, 4*y+4)
            fx = np.einsum('m,mij->ij', scalar, fields[x])
            fy = np.einsum('m,mij->ij', scalar, fields[y])
            edge = (fx@links[x, axis]+links[x, axis]@fy)/2*np.exp(1j*phases[x, axis])
            current = 1j*direction[x, axis]*edge
            contact = -direction[x, axis]**2*edge
            first[sx, sy] += current
            first[sy, sx] += current.conj().T
            second[sx, sy] += contact
            second[sy, sx] += contact.conj().T
    return first, second


def native_current_controls(band, size, frame, angles):
    law = BulkQuantumLaw(size, band=band)
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    links = law.toron_links(angles)
    rng = np.random.default_rng(222987)
    phases = rng.uniform(-.5, .5, (law.volume, 3))
    gauge = rng.uniform(-.3, .3, law.volume)
    direction = gauge[:, None]-gauge[law.target]
    h = law.onebody(fields, links, phases)
    current, contact = peierls_derivatives(law, fields, links, phases, direction)
    Q = np.diag(np.repeat(gauge, 4))
    first_commutator = Q@h-h@Q
    ward = float(np.linalg.norm(current-1j*first_commutator))
    contact_ward = float(np.linalg.norm(contact+Q@first_commutator-first_commutator@Q))
    check(band.a*ward, 'aH full native current Ward')
    check(band.a*contact_ward, 'aH full native contact Ward')
    missing_contact = band.a*float(np.linalg.norm(Q@first_commutator-first_commutator@Q))
    if missing_contact <= 1e-5:
        raise AssertionError('The omitted-contact adverse input did not reach the Ward assertion')
    moved = phases+direction
    full_gauge = np.diag(np.exp(1j*np.repeat(gauge, 4)))
    covariance = float(np.linalg.norm(law.onebody(fields, links, moved)-full_gauge@h@full_gauge.conj().T))
    check(band.a*covariance, 'aH native compact-U1 covariance')
    return dict(dimension=len(h), current_Ward=ward, contact_Ward=contact_ward,
        compact_U1_covariance=covariance, omitted_contact_aH_defect=missing_contact,
        native_full_Gram_retained=True, arbitrary_compact_phases=True,
        scope='actual native multiplication current/contact; not a quantum scattering or photon pole')


def run(*, size=5, spacing=1., centres=None) -> dict[str, Any]:
    if size < 5 or not np.isfinite(spacing) or not 0 < spacing <= 1:
        raise ValueError('N>=5 unaliased quadratic source grid and0<a<=1 required')
    identities = source_hashes()
    band = RotationBand(**{**BARE, 'a': spacing})
    basis = matrix_unit_basis()
    labels = list(itertools.product(range(4), repeat=3)) if centres is None else [tuple(c) for c in centres]
    if not labels or any(len(c) != 3 or any(not 0 <= n < 4 for n in c) for c in labels):
        raise ValueError('Centre triples must contain three integers0..3')
    if len(set(labels)) != len(labels):
        raise ValueError('Duplicate centre classes do not enlarge the native class')
    rows = []
    for label in labels:
        angles = np.pi/2*np.asarray(label)
        selected, frame = original_frame(band, size, angles, True)
        value = coordinates(frame, basis).reshape(-1)
        model = OriginalSpatial(band, size, basis, value, angles)
        check(band.a*float(np.linalg.norm(model.gradient)), 'aH actual full10 source stationarity')
        richness = rank_report(model.x*band.a)
        if richness['quadratic_rank'] != 10:
            raise AssertionError('All10 finite shifted-grid equality directions must be retained')
        parts0 = model.hessian_parts(np.zeros(3))
        leading0 = parts0['local']+parts0['sea']+parts0['contact']
        strong0 = parts0['alignment']+plaquette_hessian(np.zeros(3))
        B0, A0 = reduced_matrices(model, np.zeros(3), strong0, leading0)
        vals0, vecs0 = np.linalg.eigh(B0)
        tolerance = 1e-9*max(1., float(np.max(abs(vals0))))
        kernel = vecs0[:, abs(vals0) <= tolerance]
        if kernel.shape[1] != 45 or np.any(vals0 < -tolerance):
            raise AssertionError('q0 must have exactly45 uniform normals and45 stiff normals')
        assert A0 is not None
        alpha = float(np.linalg.eigvalsh(kernel.conj().T@A0@kernel)[0])
        if alpha <= 0:
            raise AssertionError('The source uniform45 normal Hessian is not positive')
        bound = leading_norm_bound(band, selected['G'])
        check(max(0., float(np.linalg.norm(A0, ord=2))-bound['total']), 'analytic uniform leading norm upper')
        b_min = float(vals0[vals0 > tolerance].min())
        B_max = float(vals0.max())
        q_count = 0
        for index in periodic_grid(size):
            if not np.any(index):
                continue
            q = 2*np.pi*index/size
            strong = model.alignment(q)+plaquette_hessian(q)
            B, _ = reduced_matrices(model, q, strong)
            values = np.linalg.eigvalsh(B)
            if values[0] <= 1e-9*max(1., float(values[-1])):
                raise AssertionError('Nonzero q stiff physical form must be resolved positive')
            b_min = min(b_min, float(values[0]))
            B_max = max(B_max, float(values[-1]))
            q_count += 1
        L = bound['total']
        schur_t = 2*(L+2*L**2/alpha)/b_min
        rows.append(dict(centre=list(label), native_Gram=selected['G'].tolist(),
            physical_energy_per_cell=selected['value'], full10_gradient=selected['gradient_residual'],
            aH_full_field_link_gradient=band.a*float(np.linalg.norm(model.gradient)),
            source_quadratic_rank=richness['quadratic_rank'], whole_heavy_gap_lower=bound['heavy_gap_lower'],
            physical_uniform_normal_count=45, total_physical_normal_dimension=90*size**3,
            all_nonzero_momenta_checked=q_count, alpha_estimate=alpha,
            b_all_momenta_estimate=b_min, stiff_norm_upper_estimate=B_max,
            leading_norm_upper_formula=bound, sufficient_Lambda_estimate=schur_t**4,
            soft_frequency_lower_prefactor_estimate=math.sqrt(alpha/2),
            stiff_frequency_lower_prefactor_estimate=math.sqrt(b_min/2),
            zero_point_upper_formula='45V sqrt(L Lambda^(1/2)+Bmax Lambda^(3/4))',
            Gram_isotropic_debt=float(np.linalg.norm(selected['G']-np.diag(
                [selected['G'][0, 0], *[np.trace(selected['G'][1:, 1:])/3]*3])))))
    native_rows = []
    for label in ((0, 0, 0), (1, 0, 0), (2, 2, 2)):
        if label not in labels:
            continue
        angles = np.pi/2*np.asarray(label)
        _, frame = original_frame(band, size, angles, True)
        native_rows.append(dict(centre=list(label), **native_current_controls(band, size, frame, angles)))
    if source_hashes() != identities:
        raise RuntimeError('Normal-gap source/proof dependencies changed during execution')
    return dict(source_sha256=identities, source_bindings_unchanged=True,
        governing_plan=PLAN, assignment_plan_revision=PLAN_REVISION,
        inputs=dict(size=size, spacing=spacing, stiffness=STIFFNESS/spacing,
            centre_classes=len(labels), exhaustive_centres=len(labels)==64),
        kinetic_law='field Lambda^(-1/2)/2, colour .3 Lambda^(-1/2); original heavy Lambda, J/K4 Lambda^(5/4)',
        rows=rows, native_current_controls=native_rows,
        analytic_theorem='Positive q0 uniform alpha, positive stiff b and ||H0||<=L imply a full finite-graph normal gap at the displayed Schur threshold; proof constants exact, receipt estimates floating.',
        all_checks_pass=True, quantum_ground_computed=False, full_low_energy_embedding_proved=False,
        campaign_complete=False,
        limits=['Numerical alpha/b and onset are not interval certificates or measured quantum gaps.',
            'Centre-class normal positivity is not finite-N globality against noncentral/nonuniform competitors.',
            'Complete scalar well/physical Gauss operator embedding remains in the separate BO derivation.',
            'Native finite fullmetric current is not silently replaced by isotropic RotationBand.',
            'No thermodynamic Coulomb/Z4 phase, charged lower mass, mechanical half-spin or radiative/UV closure.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=5)
    parser.add_argument('--spacing', type=float, default=1.)
    parser.add_argument('--centre', type=int, nargs=3, action='append')
    parser.add_argument('--output', type=Path, default=ROOT/'original_normal_gap_bound_receipt.json')
    args = parser.parse_args()
    result = run(size=args.size, spacing=args.spacing, centres=args.centre)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output), all_checks_pass=True,
        classes=result['inputs']['centre_classes'], full_low_energy_embedding_proved=False)))


if __name__ == '__main__':
    main()
