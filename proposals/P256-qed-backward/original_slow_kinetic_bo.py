#!/usr/bin/env python3
"""Native full-cluster/Gauss/normal jets for the analytic slow-kinetic BO theorem.

This exercises the actual source, not an infinite-dimensional vacuum solver.
Deterministic shrinking normal directions are not samples of a quantum ground.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

if __name__ == '__main__':
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'

import numpy as np
from scipy.linalg import expm, null_space

from bulk_quantum_selection import BulkQuantumLaw
from original_charged_principal import (
    OriginalSpatial, charge_form, check, original_frame, portable,
)
from original_normal_gap_bound import peierls_derivatives, source_hashes as normal_hashes
from original_uniform_charge_gap import source_hashes as charge_hashes
from rotation_band import RotationBand
from unrestricted_field_minimum import BARE, STIFFNESS, coordinates

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-continuing-backward-20260930'
ASSIGNMENT = '381993ecb362c8c67dbc334aebbcb6093b64a7b78111b5a9e785d4746b9bbe3b'


def source_hashes() -> dict[str, str]:
    return {**normal_hashes(), **charge_hashes(), **{
        name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('original_slow_kinetic_bo.py', 'original_slow_kinetic_bo.md')
    }}


def whole_heavy(h, number) -> dict[str, Any]:
    values, vectors = np.linalg.eigh(h)
    cut = len(values)-number
    if not 0 < cut < len(values):
        raise ValueError('A genuine whole occupied/empty cluster is required')
    gap = float(values[cut]-values[cut-1])
    if gap <= 0:
        raise ValueError('Actual original whole-heavy cluster is not separated')
    occupied = vectors[:, cut:]
    return dict(values=values, vectors=vectors, cut=cut, gap=gap,
        projector=occupied@occupied.conj().T, energy=float(-values[cut:].sum()))


def normal_directions(model, law, rng):
    """Actual mass-horizontal q0 soft and nonzero-q stiff directions."""
    soft_basis = null_space(model.D.T, rcond=1e-12)
    if soft_basis.shape != (60, 45):
        raise AssertionError('All45 uniform physical shape directions must remain')
    soft = soft_basis@rng.normal(size=45)
    soft /= np.linalg.norm(soft)*np.sqrt(law.volume)
    soft_fields = np.tile(np.einsum('ma,aij->mij', soft.reshape(4, 15), law.basis),
                          (law.volume, 1, 1, 1))
    q = 2*np.pi*np.array([1, 0, 0])/law.size
    G = model.gauge_columns(q)
    form = charge_form(G, 1., .3)
    Z = null_space((np.sqrt(form['mass'])[:, None]*G).conj().T, rcond=1e-12)
    if Z.shape != (105, 90):
        raise AssertionError('The complete nonzero-q physical space is90-dimensional')
    vector = np.sqrt(form['a'])*(Z@(rng.normal(size=90)+1j*rng.normal(size=90)))
    wave = np.exp(1j*law.sites@q)
    real = (wave[:, None]*vector[None, :]).real
    norm = np.sqrt(np.sum(form['mass'][None, :]*real**2))
    real /= norm
    vector /= norm
    previous = np.array([[law.index[tuple((site-np.eye(3, dtype=int)[axis]) % law.size)]
                          for axis in range(3)] for site in law.sites])
    theta = real[:, 60:].reshape(law.volume, 3, 15)
    balance = (np.einsum('fa,xf->xa', model.D, real[:, :60])
        +(theta.sum(axis=1)-theta[previous, np.arange(3)].sum(axis=1))/.6)
    check(float(np.linalg.norm(balance)), 'complete native real-space mass Gauss direction')
    stiff_fields = np.einsum('xma,aij->xmij', real[:, :60].reshape(law.volume, 4, 15), law.basis)
    generators = np.einsum('xea,aij->xeij', theta, law.generators)
    return dict(soft_fields=soft_fields, stiff_fields=stiff_fields, generators=generators,
        q=q, vector=vector, Gauss_balance=float(np.linalg.norm(balance)))


def deformed_links(links, generators, amplitude):
    return np.array([[expm(1j*amplitude*generators[x, axis])@links[x, axis]
                      for axis in range(3)] for x in range(len(links))])


def field_colour_jet(law, fields, links, field_direction, generators):
    """Exact source first/second jets of Phi+t*dPhi, exp(itT)U."""
    first = np.zeros((4*law.volume, 4*law.volume), complex)
    second = np.zeros_like(first)
    for x in range(law.volume):
        sx = slice(4*x, 4*x+4)
        first[sx, sx] += law.onsite*field_direction[x, 0]
        for axis, scalar in law.paths:
            y = law.target[x, axis]
            sy = slice(4*y, 4*y+4)
            fx = np.einsum('m,mij->ij', scalar, fields[x])
            fy = np.einsum('m,mij->ij', scalar, fields[y])
            dx = np.einsum('m,mij->ij', scalar, field_direction[x])
            dy = np.einsum('m,mij->ij', scalar, field_direction[y])
            T, u = generators[x, axis], links[x, axis]
            du, ddu = 1j*T@u, -T@T@u
            d1 = (dx@u+fx@du+du@fy+u@dy)/2
            d2 = (fx@ddu+ddu@fy+2*dx@du+2*du@dy)/2
            first[sx, sy] += d1
            first[sy, sx] += d1.conj().T
            second[sx, sy] += d2
            second[sy, sx] += d2.conj().T
    return first, second


def jet_controls(model, law, fields, links, direction) -> tuple[dict[str, Any], dict[str, Any]]:
    source = whole_heavy(law.onebody(fields, links), 2*law.volume)
    dphi, generators = direction['stiff_fields'], direction['generators']
    first, second = field_colour_jet(law, fields, links, dphi, generators)
    cut, vectors, values = source['cut'], source['vectors'], source['values']
    transition = vectors[:, cut:].conj().T@first@vectors[:, :cut]
    denominator = values[cut:, None]-values[None, :cut]
    derivative = (vectors[:, cut:]@(transition/denominator)@vectors[:, :cut].conj().T)
    derivative += derivative.conj().T
    contact = float(np.trace(source['projector']@second).real)
    sea_curvature = float(-2*np.sum(abs(transition)**2/denominator))
    c = coordinates(fields, law.basis).reshape(law.volume, 4, 15)
    dc = coordinates(dphi, law.basis).reshape(law.volume, 4, 15)
    gram = c@c.transpose(0, 2, 1)
    dgram = dc@c.transpose(0, 2, 1)+c@dc.transpose(0, 2, 1)
    ddgram = 2*dc@dc.transpose(0, 2, 1)
    local = STIFFNESS/law.band.a*float(np.sum(dgram*dgram)+np.sum(gram*ddgram))
    full_curvature = local+sea_curvature-contact
    parts = model.hessian_parts(direction['q'])
    leading = parts['local']+parts['sea']+parts['contact']
    predicted = float((law.volume/2*np.vdot(direction['vector'], leading@direction['vector'])).real)
    curvature_error = abs(full_curvature-predicted)/max(1., abs(predicted))
    check(curvature_error, 'full4V whole-heavy jet versus original Fourier105 H0', 3e-8)
    if law.band.a*abs(contact) <= 1e-5:
        raise AssertionError('The omitted-contact adverse jet did not reach the source assertion')
    step = 1e-5
    plus = whole_heavy(law.onebody(fields+step*dphi,
        deformed_links(links, generators, step)), 2*law.volume)
    minus = whole_heavy(law.onebody(fields-step*dphi,
        deformed_links(links, generators, -step)), 2*law.volume)
    finite = (plus['projector']-minus['projector'])/(2*step)
    derivative_error = float(np.linalg.norm(finite-derivative))/max(1., np.linalg.norm(derivative))
    check(derivative_error, 'actual whole-heavy separated-projector derivative', 3e-6)
    rho = source['projector'][:4, :4]
    means = np.einsum('ij,aji->a', rho, law.generators).real
    variance = np.array([np.trace(rho@T@T-rho@T@rho@T).real for T in law.generators])
    if variance.sum() <= .1:
        raise AssertionError('Actual heavy variance cannot be erased by opposite light means')
    return dict(actual_whole_heavy_occupation=2*law.volume, actual_heavy_gap=source['gap'],
        whole_heavy_energy=source['energy'], full_jet_curvature=full_curvature,
        Fourier105_curvature=predicted, independent_curvature_relative_error=curvature_error,
        dropped_contact_aH_defect=law.band.a*abs(contact),
        projector_derivative_HS_norm=float(np.linalg.norm(derivative)),
        projector_derivative_relative_error=derivative_error,
        heavy_Slater_metric=float(np.sum(abs(transition/denominator)**2)),
        per_site_heavy_colour_mean=means, per_site_heavy_variances=variance,
        opposite_means_do_not_cancel_variance=float(2*variance.sum()),
        scope='actual full4V matrix/whole2V heavy cluster; no per-Bloch filling imposed on perturbed fields'), source


def run(*, size=5, spacing=1., lambdas=(1e12, 1e16, 1e20)) -> dict[str, Any]:
    if size < 5 or not np.isfinite(spacing) or not 0 < spacing <= 1:
        raise ValueError('N>=5 and0<a<=1 required for this source diagnostic')
    if not lambdas or any(not np.isfinite(L) or L <= 0 for L in lambdas):
        raise ValueError('Positive finite Lambda samples required')
    identities = source_hashes()
    band = RotationBand(**{**BARE, 'a': spacing})
    law = BulkQuantumLaw(size, band=band)
    rng = np.random.default_rng(2229102)
    rows = []
    for label in ((0, 0, 0), (1, 0, 0), (2, 2, 2)):
        angles = np.pi/2*np.asarray(label)
        selected, frame = original_frame(band, size, angles, True)
        model = OriginalSpatial(band, size, law.basis, coordinates(frame, law.basis).reshape(-1), angles)
        check(band.a*float(np.linalg.norm(model.gradient)), 'source fullfield/link native stationarity')
        fields = np.tile(frame, (law.volume, 1, 1, 1))
        links = law.toron_links(angles)
        direction = normal_directions(model, law, rng)
        jets, heavy = jet_controls(model, law, fields, links, direction)
        phases = rng.uniform(-np.pi, np.pi, (law.volume, 3))
        gauge = rng.uniform(-.5, .5, law.volume)
        phase_direction = gauge[:, None]-gauge[law.target]
        light0 = law.onebody(fields, links, phases)
        current0, contact0 = peierls_derivatives(law, fields, links, phases, phase_direction)
        wrong_photon_heavy = whole_heavy(light0, 2*law.volume)
        photon_heavy_defect = float(np.linalg.norm(wrong_photon_heavy['projector']-heavy['projector']))
        if photon_heavy_defect <= 1e-4:
            raise AssertionError('Incorrect EM-coupled heavy input must change the actual occupied section')
        scales = []
        for Lambda in lambdas:
            soft, stiff = Lambda**(-5/16), Lambda**(-7/16)
            moved_fields = fields+soft*direction['soft_fields']+stiff*direction['stiff_fields']
            moved_links = deformed_links(links, direction['generators'], stiff)
            actual_heavy = whole_heavy(law.onebody(moved_fields, moved_links), 2*law.volume)
            light = law.onebody(moved_fields, moved_links, phases)
            current, contact = peierls_derivatives(law, moved_fields, moved_links, phases, phase_direction)
            Q = np.repeat(gauge, 4)
            commutator = (Q[:, None]-Q[None, :])*light
            ward = float(band.a*np.linalg.norm(current-1j*commutator))
            contact_ward = float(band.a*np.linalg.norm(contact+(Q[:, None]-Q[None, :])*commutator))
            check(ward, 'nonuniform full-source native current Ward')
            check(contact_ward, 'nonuniform full-source native contact Ward')
            scales.append(dict(Lambda=Lambda, actual_unscaled_whole_heavy_gap=actual_heavy['gap'],
                heavy_excited_gap_times_Lambda=Lambda*actual_heavy['gap'],
                heavy_projector_change=float(np.linalg.norm(actual_heavy['projector']-heavy['projector'])),
                light_aH_change=float(band.a*np.linalg.norm(light-light0)),
                current_aH_change=float(band.a*np.linalg.norm(current-current0)),
                contact_aH_change=float(band.a*np.linalg.norm(contact-contact0)),
                current_aH_Ward=ward, contact_aH_Ward=contact_ward,
                deterministic_soft_amplitude=soft, deterministic_stiff_amplitude=stiff,
                common_connection_asymptotic_factor=Lambda**(-1/16),
                scope='shrinking actual normal direction; not a computed vacuum moment or quantum eigenvalue'))
        rows.append(dict(centre=list(label), native_full10_Gram=selected['G'],
            aH_native_full_stationarity=band.a*float(np.linalg.norm(model.gradient)),
            mass_horizontal_real_Gauss_error=direction['Gauss_balance'], jet_controls=jets,
            wrong_EM_coupled_heavy_projector_defect=photon_heavy_defect, scales=scales))
    if source_hashes() != identities:
        raise RuntimeError('Full BO source/proof dependencies changed during execution')
    return cast(dict[str, Any], portable(dict(source_sha256=identities, source_bindings_unchanged=True,
        governing_plan=PLAN, assignment_plan_revision=ASSIGNMENT,
        inputs=dict(size=size, volume=size**3, spacing=spacing, lambdas=list(lambdas)),
        rows=rows, all_checks_pass=True, quantum_ground_computed=False,
        native_N_globality_certified=False,
        analytic_scope='Generalized norm-resolvent fullGauss theorem at fixeda and sufficientlylargefiniteN; existenceN0, all64 scalar well offsets, no hard cuts.',
        low_energy_operator='full native metric light CAR and compactU1 photon on exact reference/Gauss domains',
        limits=['This CLI tests fullcluster/normal/source jets, not the infinite-dimensional isometry or exact scalar vacuum numerically.',
            'Deterministic normal directions are not quantum ground samples.',
            'No nativeN5 globality, uniformN/a estimate, isolated charge/pole, thermodynamic photon phase, mechanical spin or QED closure.'],
        campaign_complete=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=5)
    parser.add_argument('--spacing', type=float, default=1.)
    parser.add_argument('--lambdas', type=float, nargs='+', default=[1e12, 1e16, 1e20])
    parser.add_argument('--output', type=Path, default=ROOT/'original_slow_kinetic_bo_receipt.json')
    args = parser.parse_args()
    result = run(size=args.size, spacing=args.spacing, lambdas=args.lambdas)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output), all_checks_pass=True,
        quantum_ground_computed=False, native_N_globality_certified=False)))


if __name__ == '__main__':
    main()
