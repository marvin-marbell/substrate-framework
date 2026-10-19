#!/usr/bin/env python3
"""Certified original a1 full-PSD Gram admission, uniform in native toron shifts."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from flint import arb, ctx

from bulk_global_gram import source_hashes as global_source_hashes
from original_charged_principal import OriginalUniform, original_frame
from bulk_quantum_selection import BulkQuantumLaw
from rotation_band import RotationBand
from sea_selection import coefficients, matrix_unit_basis
from soft_budget import select_gram
from unrestricted_field_minimum import BARE, STIFFNESS, coordinates

ROOT = Path(__file__).resolve().parent
DECLARED = dict(s_floor=.4, G00_upper=.43, Gjj_upper=.065, tilt_upper=.002)


def upper(value):
    """Outward binary endpoint, with its direction checked by Arb itself."""
    result = float(value.upper())
    while not arb(result) >= value:
        result = math.nextafter(result, math.inf)
    return result


def lower(value):
    result = float(value.lower())
    while not arb(result) <= value:
        result = math.nextafter(result, -math.inf)
    return result


def interval(value):
    return [lower(value), upper(value)]


def shifted_error(n, sigma, majorant):
    t = (-sigma*n).exp()
    return majorant*(((1+t)/(1-t))**3-1)


def bounds(g0, gs, distance):
    # Frobenius norm pays both copies of each temporal/spatial cross term.
    spatial_floor = gs-distance
    if not spatial_floor > 0:
        return None
    cross_squared = distance**2/2
    schur = g0-distance-cross_squared/spatial_floor
    if not schur > 0:
        return None
    tilt = (cross_squared/(spatial_floor*schur)).sqrt()
    return dict(G00=interval(g0+arb(0, upper(distance))),
                Gjj_each=interval(gs+arb(0, upper(distance))),
                spatial_eigenvalue_lower=lower(spatial_floor),
                temporal_spatial_cross_norm_upper=upper(distance/arb(2).sqrt()),
                offdiagonal_entry_abs_upper=upper(distance/arb(2).sqrt()),
                Schur_lower=lower(schur), tilt_upper=upper(tilt),
                Frobenius_distance_upper=upper(distance))


def admitted(box, declared):
    return box is not None and (box['Schur_lower'] >= declared['s_floor']
        and box['G00'][1] <= declared['G00_upper']
        and box['Gjj_each'][1] <= declared['Gjj_upper']
        and box['tilt_upper'] <= declared['tilt_upper'])


def certificate(quadrature, precision):
    if quadrature < 8 or quadrature % 2 or precision < 80:
        raise ValueError('Even quadrature>=8 and precision>=80 required')
    ctx.prec = precision
    n = quadrature
    p = np.stack(np.meshgrid(*([2*np.pi*(np.arange(n)+.5)/n]*3),
                             indexing='ij'), axis=-1).reshape(-1, 3)
    band = RotationBand(**BARE)
    selected = select_gram(coefficients(band, p), STIFFNESS, 0.)
    # The existing solver proposes a trial only. Every subsequent proof uses
    # these explicitly recorded decimals and independently enclosed residuals.
    trial = [str(float(selected['G'][0, 0])), str(float(selected['G'][1, 1]))]
    g0, gs = map(arb, trial)
    # Contains both exact decimal physical parameters and executing binary64
    # BARE values. This also pays parameter rounding in native coefficient calls.
    m, r, v = (arb(s, '1e-16') for s in ('.2', '.12', '.3'))
    kappa = arb(2)
    nodes = [arb.pi()*(2*j+1)/n for j in range(n//2)]
    cosines = [q.cos() for q in nodes]
    sinsq = [q.sin()**2 for q in nodes]
    temporal, spatial_sum = arb(0), arb(0)
    terms = 0
    for i, j, k in itertools.combinations_with_replacement(range(n//2), 3):
        multiplicity = 1 if i == k else (3 if i == j or j == k else 6)
        x0 = m+r*(3-cosines[i]-cosines[j]-cosines[k])
        kinetic = v**2*(sinsq[i]+sinsq[j]+sinsq[k])
        radius = (g0*x0**2+gs*kinetic).sqrt()
        temporal += multiplicity*x0**2/radius
        spatial_sum += multiplicity*kinetic/radius
        terms += 1
    count = (n//2)**3
    residual0 = kappa*g0-temporal/count
    residuals = kappa*gs-spatial_sum/(3*count)
    # Full symmetric-matrix gradient: six offdiagonals vanish EXACTLY by
    # midpoint reflection; the three spatial diagonals agree by permutation.
    # Squaring a zero-crossing ball can acquire a tiny negative lower end.
    # Square outward positive absolute endpoints instead of taking that sqrt.
    residual = (arb(upper(abs(residual0)))**2+
                3*arb(upper(abs(residuals)))**2).sqrt()
    sigma = arb('.5')
    coercivity = g0.sqrt()*(m-3*r*(sigma.cosh()-1))-v*(3*gs).sqrt()*sigma.sinh()
    if not coercivity > 0:
        raise AssertionError('Analytic polystrip coercivity not certified')
    xnorm_squared = (m+3*r*(1+sigma.cosh()))**2+3*v**2*sigma.cosh()**2
    majorant = xnorm_squared/coercivity
    continuum_error = shifted_error(n, sigma, majorant)
    continuum_distance = (residual+continuum_error)/kappa
    continuum_box = bounds(g0, gs, continuum_distance)
    metric_n = None
    box = None
    for native_n in range(3, 10000):
        distance = (residual+continuum_error+shifted_error(native_n, sigma, majorant))/kappa
        box = bounds(g0, gs, distance)
        if admitted(box, DECLARED):
            metric_n = native_n
            break
    if metric_n is None or not admitted(continuum_box, DECLARED):
        raise AssertionError('Declared box not reached; sharpen the certificate')
    previous = bounds(g0, gs, (residual+continuum_error+
                              shifted_error(metric_n-1, sigma, majorant))/kappa)
    return dict(parameters=dict(a=1, m=interval(m), r=interval(r), v=interval(v),
                                kappa=2, preferred=0, improved=False),
        trial_decimal=trial, trial_supplier='soft_budget.select_gram, proposal only',
        Arb_precision_bits=precision, certified_quadrature=n,
        symmetry_reduced_interval_terms=terms, full_ten_gradient=dict(
            temporal=interval(residual0), spatial_each=interval(residuals),
            six_offdiagonals='exact zero by midpoint sign reflections',
            Frobenius_residual_upper=upper(residual)),
        analytic_strip=dict(sigma=.5, coercivity_lower=lower(coercivity),
                            matrix_gradient_integrand_HS_majorant=upper(majorant)),
        continuum_quadrature_error_upper=upper(continuum_error),
        continuum_box=continuum_box, N_metric=metric_n,
        all_real_shifts_all_N_ge_N_metric_box=box,
        bound_at_N_metric_minus_one=previous,
        previous_bound_admits=admitted(previous, DECLARED),
        declared_box=DECLARED, admission=True)


def native_consumer(size, centre, cert):
    band = RotationBand(**BARE)
    angles = np.asarray(centre)*np.pi/2
    solution, frame = original_frame(band, size, angles, True)
    basis = matrix_unit_basis()
    model = OriginalUniform(band, size, basis, angles)
    value = coordinates(frame, basis).reshape(-1)
    energy, gradient = model.evaluate(value)
    law = BulkQuantumLaw(size, band=band)
    h = law.onebody(np.tile(frame, (law.volume, 1, 1, 1)), law.toron_links(angles))
    eigenvalues = np.linalg.eigvalsh(h)
    radii = np.sqrt(np.einsum('pi,ij,pj->p', model.x, solution['G'], model.x))
    actual_whole = -float(eigenvalues[2*law.volume:].sum())/law.volume
    gram_sea = -2*float(radii.mean())
    native_error = abs(actual_whole-gram_sea)
    if native_error > 2e-10 or np.linalg.norm(gradient) > 2e-9:
        raise AssertionError('Actual native whole-heavy consumer failed')
    G = solution['G']
    full_gradient = 2*G-np.einsum('pi,pj,p->ij', model.x, model.x, 1/radii)/len(radii)
    diagonal_only = np.diag(np.diag(G))
    wrong_radii = np.sqrt(np.einsum('pi,ij,pj->p', model.x, diagonal_only, model.x))
    wrong_gradient = 2*diagonal_only-np.einsum('pi,pj,p->ij',
                                           model.x, model.x, 1/wrong_radii)/len(radii)
    offdiagonal_defect = float(np.linalg.norm(wrong_gradient-np.diag(np.diag(wrong_gradient))))
    if offdiagonal_defect <= 1e-8:
        raise AssertionError('Adverse shift failed to reject diagonal-only stationarity')
    rejected_forgery = dict(cert)
    # Close meshes without a remainder are deliberately not an admission input.
    rejected_forgery.pop('analytic_strip')
    if validate_certificate(rejected_forgery):
        raise AssertionError('Mesh-agreement-only shortcut accepted')
    wrong_box = bounds(arb(cert['trial_decimal'][0]), arb(cert['trial_decimal'][1]), arb('.01'))
    if admitted(wrong_box, DECLARED):
        raise AssertionError('Offdiagonal/tilt budget omission accepted')
    trial = np.diag([float(cert['trial_decimal'][0]), *([float(cert['trial_decimal'][1])]*3)])
    return dict(size=size, centre=list(centre), native_modes=len(eigenvalues),
        native_total_occupied=2*law.volume, native_whole_half_sea=actual_whole,
        Gram_attaining_sea=gram_sea, native_Gram_energy_error=native_error,
        actual_uniform_energy=energy, actual_full60_gradient_norm=float(np.linalg.norm(gradient)),
        actual_full10_gradient_norm=float(np.linalg.norm(full_gradient)), Gram=G.tolist(),
        distance_to_certified_trial=float(np.linalg.norm(G-trial)),
        diagonal_only_offdiagonal_gradient_defect=offdiagonal_defect,
        shortcuts_rejected=['diagonal-only native minimizer', 'unpaid offdiagonal/Schur tilt',
                            'mesh agreement without certified continuum remainder'],
        scope='Actual small native shifted whole-heavy Hamiltonian; not the global proof')


def validate_certificate(value):
    return (value.get('admission') is True and 'analytic_strip' in value
            and 'full_ten_gradient' in value and 'continuum_quadrature_error_upper' in value
            and admitted(value.get('all_real_shifts_all_N_ge_N_metric_box'), DECLARED))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declared-packet', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--quadrature', type=int, default=64)
    parser.add_argument('--precision', type=int, default=128)
    parser.add_argument('--native-size', type=int, default=3)
    parser.add_argument('--centre', type=int, nargs=3, default=[1, 2, 3])
    args = parser.parse_args()
    payload = args.declared_packet.read_bytes()
    declaration = json.loads(payload)['conditional_tight_loop_family']
    if {k: declaration[k] for k in DECLARED} != DECLARED:
        raise ValueError('Exact Sharp event23 declared box required')
    hashes = {**global_source_hashes(), **{name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('original_charged_principal.py', 'original_charged_principal.md',
                     'noncompact_native_metric_admission.py', 'noncompact_native_metric_admission.md')}}
    cert = certificate(args.quadrature, args.precision)
    consumer = native_consumer(args.native_size, args.centre, cert)
    if not validate_certificate(cert):
        raise AssertionError('Full admission proof missing')
    if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest for name, digest in hashes.items()):
        raise RuntimeError('Executed suppliers changed')
    if args.declared_packet.read_bytes() != payload:
        raise RuntimeError('Declared packet changed')
    result = dict(topic_key='native-full-ten-metric-admission-a1', certificate=cert,
        actual_native_consumer=consumer, source_sha256=hashes,
        input=dict(declared_packet=str(args.declared_packet),
                   declared_packet_sha256=hashlib.sha256(payload).hexdigest(),
                   quadrature=args.quadrature, precision=args.precision,
                   native_size=args.native_size, centre=args.centre),
        source_and_input_unchanged=True, all_checks_pass=True,
        limitations=['Original fixed a1 flat uniform sector only; no arbitrary nonuniform Gram admission.',
                     'No physical charged pole, real-time transport, or joined campaign completion.'])
    text = json.dumps(result, indent=2, allow_nan=False)+'\n'
    args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
