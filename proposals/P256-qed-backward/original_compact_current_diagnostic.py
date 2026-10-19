#!/usr/bin/env python3
"""Actual affine-cosine/retained-image current energy; no normalized phase claim."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any
if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[key] = '1'
ROOT = Path(__file__).resolve().parent
import numpy as np
from bulk_frame_compton import reconstruct_frame
from original_charged_principal import original_frame
from original_quantum_mass_diagnostic import source_hashes as mass_source_hashes
from rotation_band import RotationBand
from unrestricted_field_minimum import BARE


def difference(value, axis):
    return np.roll(value, -1, axis=axis)-value


def principal(value):
    return np.angle(np.exp(1j*value))


def cosine_integral(x, y):
    return 1-np.cos((x+y)/2)*np.sinc((y-x)/(2*np.pi))


def accept(error, label, tolerance=2e-10):
    if not np.isfinite(error) or error > tolerance:
        raise AssertionError(f'{label}: {error} > {tolerance}')
    return float(error)


def chart_case(seed, image_amplitude, beta_s, alpha_t):
    shape = (4, 5, 5, 5)
    rng = np.random.default_rng(seed)
    # A0 is the signed coarse temporal connection; its group transport and spatial
    # endpoint shift use this same convention. No unmatched ordinary-a0 kick is used.
    A = rng.uniform(-np.pi, np.pi, (4, *shape))
    images = rng.integers(-2, 3, (3, *shape))*image_amplitude
    faces = {}
    raw_spatial = {}
    integrals = []
    for mu, nu in itertools.combinations(range(4), 2):
        raw = difference(A[nu], mu)-difference(A[mu], nu)
        if mu == 0:
            faces[mu, nu] = raw+2*np.pi*images[nu-1]
        else:
            faces[mu, nu] = principal(raw)
            raw_spatial[mu, nu] = raw
            # Actual affine spatial lifts between both coarse slices; every retained
            # electric image contributes its integer curl throughout the interval.
            image_curl = difference(images[nu-1], mu)-difference(images[mu-1], nu)
            y = np.roll(raw, -1, axis=0)+2*np.pi*image_curl
            actual = cosine_integral(raw, y)
            endpoint_lower = (1-np.cos(raw)+1-np.cos(y))/6
            accept(float(np.maximum(0, endpoint_lower-actual).max()), 'exact affine cosine endpoint lower')
            integrals.append(actual)
    q = {}
    quantization_error = 0.
    for mu, nu, rho in itertools.combinations(range(4), 3):
        derivative = (difference(faces[nu, rho], mu)-difference(faces[mu, rho], nu)
                      +difference(faces[mu, nu], rho))/(2*np.pi)
        quantization_error = max(quantization_error, float(np.max(abs(derivative-np.rint(derivative)))))
        q[mu, nu, rho] = np.rint(derivative)
    accept(quantization_error, 'retained-image integer compact current', 2e-9)
    closure = (difference(q[1, 2, 3], 0)-difference(q[0, 2, 3], 1)
               +difference(q[0, 1, 3], 2)-difference(q[0, 1, 2], 3))
    accept(float(np.max(abs(closure))), 'exact closed current d q=0')
    current_norm2 = float(sum(np.sum(value*value) for value in q.values()))
    # Discrete exactness forces zero total homology; outside currents are not dropped.
    homology_error = max(float(np.max(abs(value.sum(axis=key)))) for key, value in q.items())
    accept(homology_error, 'real-exact current total homology')
    curvature_identity_error = 0.
    for (mu, nu), actual in zip(raw_spatial, integrals):
        # Exact actual interpolation, not an endpoint-only replacement: its winding
        # is the mixed-cube current. A constant start-face 2pi lift is invisible.
        x = faces[mu, nu]
        y = np.roll(x, -1, axis=0)-2*np.pi*q[0, mu, nu]
        curvature_identity_error = max(curvature_identity_error,
            float(np.max(abs(actual-cosine_integral(x, y)))))
    accept(curvature_identity_error, 'actual current-coordinate affine-cosine identity')
    b_norm2 = float(sum(np.sum(value*value) for value in faces.values()))
    temporal_energy = alpha_t/2*sum(float(np.sum(faces[0, nu]**2)) for nu in range(1, 4))
    actual_energy = beta_s*sum(float(np.sum(value)) for value in integrals)+temporal_energy
    sampled_lower = beta_s/3*sum(float(np.sum(1-np.cos(value))) for value in raw_spatial.values())+temporal_energy
    curvature_lower = 2*beta_s/(3*np.pi**2)*b_norm2
    current_lower = beta_s/6*current_norm2
    for left, right, label in ((actual_energy, sampled_lower, 'actual cosine versus sampled lower'),
                              (sampled_lower, curvature_lower, 'compact principal curvature lower'),
                              (curvature_lower, current_lower, 'actual closed current energy')):
        accept(max(0., right-left)/max(1., right), label)
    assert current_norm2 > 100
    return dict(seed=seed, sampled_image_scale=image_amplitude, maximum_absolute_retained_image=int(np.max(abs(images))),
                current_coordinate_affine_identity_error=curvature_identity_error, quantization_error=quantization_error,
                closure_error=float(np.max(abs(closure))), total_homology_error=homology_error,
                b_norm2=b_norm2, current_norm2=current_norm2, actual_affine_photon_energy=actual_energy,
                sampled_lower=sampled_lower, curvature_lower=curvature_lower, current_lower=current_lower,
                compact_angles_sha256=hashlib.sha256(A.tobytes()).hexdigest(),
                retained_images_sha256=hashlib.sha256(images.tobytes()).hexdigest(),
                db_equals_zero_rejected=True,
                scope='selected attained charts; image amplitudes are inputs, not physical truncations')


def native_source_hashes():
    return mass_source_hashes() | {Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'original_compact_current_diagnostic_receipt.json')
    args = parser.parse_args()
    before = native_source_hashes()
    bare = RotationBand(**{**BARE, 'a': 1.}, e=1.)
    selection, frame = original_frame(bare, 5, np.zeros(3), True)
    band, _, _ = reconstruct_frame(bare, frame)
    g, p = .3, .75
    ue, K, delta = band.v*g*g/2, band.v/(g*g), g**p/band.v
    beta_s, alpha_t = K*delta, 1/(2*ue*delta)
    accept(abs(beta_s-g**(p-2)), 'native spatial coefficient')
    accept(abs(alpha_t-g**(-2-p)), 'native temporal coefficient')
    epsilon = 1e-3
    integral = float(cosine_integral(-epsilon, epsilon))
    ratio = integral/(2*(1-np.cos(epsilon)))
    accept(abs(ratio-1/6), 'sharp affine-cosine opposite-endpoint limit', 2e-8)
    assert integral < (2*(1-np.cos(epsilon)))/5
    winding = float(cosine_integral(0., 2*np.pi))
    accept(abs(winding-1), 'retained full-winding interval')
    assert winding > .9 and 1-np.cos(2*np.pi) == 0
    rows = [chart_case(2221034+i, amplitude, beta_s, alpha_t) for i, amplitude in enumerate((1, 7, 1000))]
    result: dict[str, Any] = dict(plan='issue-222-continuing-backward-20260930',
        assignment_revision='ae3462dd2e6af3ce4e30b03a879237c98cec4e531c209079dfb62a43af7bfd89',
        proof_sha256='aa37da0bff5e0b632dcffbc0cd9ebfa2d97bf31697b282928254ba0936d97a36',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), source_hashes=before,
        native_Gram=selection['G'].tolist(), native_gradient=float(selection['gradient_residual']),
        original_photon=dict(g=g, p=p, ue=ue, K=K, delta=delta, beta_s=beta_s, alpha_t=alpha_t),
        runtime=dict(python=platform.python_version(), executable=sys.executable, numpy=np.__version__),
        inputs=dict(spatial_N=5, time_slices=4, all_harmonic_coordinates_retained=True),
        opposite_endpoint_ratio=ratio, full_winding_integral=winding,
        wrong_endpoint_only_upper_rejected=True, rows=rows, all_checks=True,
        scope='actual native-photon affine-cosine baseline/current chart only; no CAR/bridge integration, normalized current-fiber denominator, susceptibility, phase or original finiteLambda Gibbs computation')
    assert before == native_source_hashes(), 'native source changed during compact-current check'
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), all_checks=True, cases=len(rows),
                         maximum_absolute_retained_image=max(row['maximum_absolute_retained_image'] for row in rows), normalized_current_activity_proved=False)))


if __name__ == '__main__':
    main()
