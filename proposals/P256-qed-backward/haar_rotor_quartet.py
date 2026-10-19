#!/usr/bin/env python3
"""Global continuous-Haar scalar-rotor quartet and spectral transport certificate.

No Peter--Weyl cutoff enters this certificate. The scalar Casimir law is declared,
not the actual matter-coupled angular Hamiltonian or a derived quantum law.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
import scipy

from haar_centre_rotor import commutator_controls
from quantum_su4_gauss import (HaarLink, RotationBand, matrix_unit_basis,
                               source_hashes, uniform_data)

IMMUTABLE_ROTOR_SHA256 = '690fd7df0f47959479db91cfda1d6e561df229da390d46656e719eb10d8fb841'
DIMENSION = 15
# This is a numerical outward display guard, not a change to the Hamiltonian.
DISPLAY_GUARD = 1e-10


class CertificateRejected(ValueError):
    """The supplied parameters do not meet this certificate's accept boundary."""


def positive_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise CertificateRejected(f'{name} must be positive and finite')


def complex_fraction(value):
    return Fraction(float(value.real)), Fraction(float(value.imag))


def complex_product(a, b):
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def exact_commutator(a, b):
    result = []
    for i in range(4):
        for j in range(4):
            real, imag = Fraction(0), Fraction(0)
            for k in range(4):
                p = complex_product(a[i][k], b[k][j])
                q = complex_product(b[i][k], a[k][j])
                real += p[0] - q[0]
                imag += p[1] - q[1]
            result.append((real, imag))
    return result


def row_norm(matrix: list[list[Fraction]]) -> Fraction:
    return max(sum((abs(value) for value in row), Fraction(0)) for row in matrix)


def fraction_upper(value: Fraction) -> float:
    result = float(value)
    if Fraction(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def verified_quadratic(frame, generators, controls) -> dict[str, Any]:
    """Exact dyadic residual enclosure of the actual 15-coordinate quadratic.

    Eigensolver outputs are witnesses, not trusted eigenvalue assertions. Exact
    Fraction arithmetic checks M-V D V^t, V^t V-I, and the generator Gram matrix.
    Canonicalizing the concrete generator basis is included in the enclosure.
    """
    if frame.shape != (4, 4, 4) or generators.shape != (15, 4, 4):
        raise CertificateRejected('The actual four-field / fifteen-generator frame is required')
    if not (np.all(np.isfinite(frame)) and np.all(np.isfinite(generators))
            and all(np.array_equal(field, field.conj().T) for field in frame)
            and all(np.array_equal(t, t.conj().T) for t in generators)):
        raise CertificateRejected('Finite exactly Hermitian concrete fields/generators are required')
    fields = [[[complex_fraction(z) for z in row] for row in field] for field in frame]
    basis = [[[complex_fraction(z) for z in row] for row in t] for t in generators]
    # Rounded diagonal generators can have a tiny scalar trace. Remove it
    # exactly before canonicalization; commutators are unchanged by this.
    for t in basis:
        scalar = tuple(sum((t[i][i][k] for i in range(4)), Fraction(0)) / 4
                       for k in range(2))
        for i in range(4):
            t[i][i] = t[i][i][0] - scalar[0], t[i][i][1] - scalar[1]
    images = [[exact_commutator(field, t) for t in basis] for field in fields]
    exact_m = [[sum((images[f][a][k][0] * images[f][b][k][0]
                     + images[f][a][k][1] * images[f][b][k][1]
                     for f in range(4) for k in range(16)), Fraction(0))
                for b in range(15)] for a in range(15)]
    gram = [[2 * sum((basis[a][i][j][0] * basis[b][i][j][0]
                     + basis[a][i][j][1] * basis[b][i][j][1]
                     for i in range(4) for j in range(4)), Fraction(0))
             for b in range(15)] for a in range(15)]
    gram_error = row_norm([[gram[a][b] - int(a == b) for b in range(15)] for a in range(15)])
    values, vectors = np.linalg.eigh(np.asarray(controls['local_quadratic_matrix']))
    d = [Fraction(float(value)) for value in values]
    v = [[Fraction(float(value)) for value in row] for row in vectors]
    orth_error = row_norm([[sum((v[k][a] * v[k][b] for k in range(15)), Fraction(0))
                            - int(a == b) for b in range(15)] for a in range(15)])
    residual = row_norm([[exact_m[a][b] - sum((v[a][k] * d[k] * v[b][k]
                                              for k in range(15)), Fraction(0))
                          for b in range(15)] for a in range(15)])
    if max(gram_error, orth_error) >= Fraction(1, 10):
        raise CertificateRejected('Canonical generator or oscillator witness is ill-conditioned')
    # Q=V(V^tV)^(-1/2). The factor three bounds the polar-basis correction.
    enclosure = (residual + 3 * max(abs(value) for value in d) * orth_error
                 + row_norm(exact_m) * gram_error / (1 - gram_error))
    lower = [math.nextafter(float(value - enclosure), -math.inf) for value in d]
    upper = [math.nextafter(float(value + enclosure), math.inf) for value in d]
    if min(lower) <= 0:
        raise CertificateRejected('Positive actual local quadratic spectrum was not certified')
    return dict(eigenvalue_lower=lower, eigenvalue_upper=upper,
                eigenvalue_witness=values.tolist(), orthogonal_witness=vectors.tolist(),
                exact_dyadic_residual_row_norm_upper=fraction_upper(residual),
                exact_dyadic_orthogonality_row_norm_upper=fraction_upper(orth_error),
                exact_dyadic_generator_Gram_row_norm_upper=fraction_upper(gram_error),
                canonical_quadratic_enclosure_radius=fraction_upper(enclosure),
                Gaussian_definition='Q=V(V^t V)^(-1/2); T0=T-Tr(T)I/4; canonical generators=T0 G^(-1/2); A=Q diag(sqrt(J*mu_upper/kappa)) Q^t',
                lower_constraint_Gaussian='A_lower=Q diag(sqrt(c*J*mu_lower/kappa)) Q^t',
                interpretation='Exact rational witness enclosure; the display eigensolver itself is not a proof')


def choose_radii(kappa: float, alignment: float) -> tuple[float, float]:
    """Vary chart radius, not the bare law; balance geometry against IMS cost."""
    ratio_log = math.log(kappa) - math.log(alignment)
    natural_width = math.exp(ratio_log / 4)
    geometry_width = math.exp(ratio_log / 8)
    outer = max(45 * natural_width, 1.155 * geometry_width)
    return 2 * outer / 3, outer


def quartet_bound(quadratic: dict[str, Any], ell_min: float, kappa: float,
                  alignment: float, inner: float | None = None,
                  outer: float | None = None) -> dict[str, Any]:
    """A global min--max/IMS certificate, with an explicit spectral polar map."""
    positive_finite(kappa, 'kappa')
    positive_finite(alignment, 'J')
    positive_finite(ell_min, 'ell_min')
    if (inner is None) != (outer is None):
        raise CertificateRejected('Supply both inner and outer radii, or neither')
    if inner is None:
        inner, outer = choose_radii(kappa, alignment)
    assert inner is not None and outer is not None
    if not (math.isfinite(inner) and math.isfinite(outer) and 0 < inner < outer <= 1):
        raise CertificateRejected('The disjoint certified charts require 0<inner<outer<=1')
    lower = np.asarray(quadratic['eigenvalue_lower'], dtype=float)
    upper = np.asarray(quadratic['eigenvalue_upper'], dtype=float)
    if lower.shape != (DIMENSION,) or upper.shape != (DIMENSION,) or not (
            np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
            and np.all(lower > 0) and np.all(upper >= lower)):
        raise CertificateRejected('Fifteen positive ordered curvature enclosures are required')
    scale = math.sqrt(kappa) * math.sqrt(alignment)
    harmonic_upper = float(np.sqrt(upper).sum()) * scale
    harmonic_lower = float(np.sqrt(lower).sum()) * scale
    harmonic_gap_lower = 2 * math.sqrt(float(lower.min())) * scale
    # Exact dyadic polynomial endpoints plus outward conversion preserve the
    # bound direction even when a nonzero chart correction rounds to one.
    outer_squared = Fraction(outer) ** 2
    sinc_lower = math.nextafter(float(1 - outer_squared / 24), -math.inf)
    jacobian_lower = math.nextafter(float(Fraction(sinc_lower) ** 12), -math.inf)
    inverse_metric_upper = math.nextafter(float(Fraction(sinc_lower) ** -2), math.inf)
    potential_lower_factor = math.nextafter(float(1 - outer_squared / 12), -math.inf)
    derivative_bound = math.pi / (2 * (outer - inner))
    ims_cost = kappa * derivative_bound ** 2
    # The same global centre coercivity, with a conservative display guard.
    distance_constant = 2 + math.pi ** 2 / 2
    outside_chord_lower_squared = inner ** 2 * (1 - inner ** 2 / 24) ** 2 / 2
    outside_energy_lower = alignment * ell_min * outside_chord_lower_squared / distance_constant
    local_ground_lower = jacobian_lower * math.sqrt(potential_lower_factor) * harmonic_lower
    local_second_lower = jacobian_lower * math.sqrt(potential_lower_factor) * (
        harmonic_lower + harmonic_gap_lower)
    # Form the square-root factors before division: J/kappa may overflow
    # even when the actual Gaussian precision and energies are representable.
    alpha_min = math.sqrt(alignment) * math.sqrt(float(upper.min())) / math.sqrt(kappa)
    positive_finite(alpha_min, 'Gaussian precision')
    # t=alpha_min/2; product (1-t/alpha_i)^(-1/2) <= 2^(15/2).
    tail_log_upper = DIMENSION * math.log(2) / 2 - alpha_min * inner ** 2 / 2
    if tail_log_upper >= 0:
        raise CertificateRejected('Chernoff tail does not certify positive truncated normalization')
    # Never silently turn an underflowed positive tail into zero.
    tail_upper = math.exp(max(tail_log_upper, math.log(np.finfo(float).tiny)))
    tail_upper = math.nextafter(tail_upper * (1 + DISPLAY_GUARD), math.inf)
    packet_fraction_lower = math.nextafter(float(1 - Fraction(tail_upper)), -math.inf)
    norm_lower = math.nextafter(
        float(Fraction(jacobian_lower) * Fraction(packet_fraction_lower)), -math.inf)
    if not (0 < norm_lower <= 1):
        raise CertificateRejected('Positive Haar packet normalization was not certified')
    raw_upper = inverse_metric_upper / jacobian_lower * (
        harmonic_upper + ims_cost * tail_upper / packet_fraction_lower)
    lambda4_upper = raw_upper * (1 + DISPLAY_GUARD)
    lambda1_lower = min(local_ground_lower, outside_energy_lower) * (1 - DISPLAY_GUARD) - ims_cost * (1 + DISPLAY_GUARD)
    lambda5_lower = min(local_second_lower, outside_energy_lower) * (1 - DISPLAY_GUARD) - ims_cost * (1 + DISPLAY_GUARD)
    separation = lambda5_lower - lambda4_upper
    if not all(math.isfinite(value) for value in (scale, lambda1_lower, lambda4_upper, lambda5_lower)):
        raise CertificateRejected('Certificate arithmetic exceeds the supported finite numeric range')
    if lambda1_lower <= 0 or separation <= 0:
        raise CertificateRejected('This cutoff design does not prove a positive globally isolated quartet')
    leakage_squared = (lambda4_upper - lambda1_lower) / (lambda5_lower - lambda1_lower)
    leakage_squared = math.nextafter(leakage_squared * (1 + DISPLAY_GUARD), math.inf)
    if not 0 <= leakage_squared < 1:
        raise CertificateRejected('The energy bound does not prove an invertible trial-to-spectral polar map')
    projector_error = math.sqrt(leakage_squared)
    # Stable evaluation of sqrt(2(1-sqrt(1-delta^2))).
    polar_error = math.sqrt(2 * leakage_squared / (1 + math.sqrt(1 - leakage_squared)))
    trial_clock_error = outer ** 2 / 4
    transport_error = 2 * polar_error + trial_clock_error
    transport_nonzero = transport_error < 1
    return dict(J=alignment, kappa=kappa, inner_radius=inner, outer_radius=outer,
                radius_selection='adaptive balance unless explicit radii supplied; no change to the scalar bare law',
                geometry=dict(Haar_jacobian_lower=jacobian_lower, Haar_jacobian_upper=1.,
                              inverse_metric_lower=1., inverse_metric_upper=inverse_metric_upper,
                              potential_quadratic_lower_factor=potential_lower_factor),
                anisotropic_Gaussian=dict(alpha_min=alpha_min,
                    oscillator_ground_upper=harmonic_upper, oscillator_ground_lower=harmonic_lower,
                    oscillator_first_gap_lower=harmonic_gap_lower,
                    Chernoff_t=alpha_min / 2, Chernoff_log_tail_upper=tail_log_upper,
                    tail_probability_upper=tail_upper,
                    normalized_local_Haar_integral_lower=norm_lower,
                    normalized_local_Haar_integral_upper=1.,
                    exact_Haar_norm_squared='c_H * integral(chi^2*g^2*j dx); c_H=1/Vol_SU4; never replaced by the Euclidean norm'),
                global_IMS=dict(number_of_centre_cutoffs=4, number_of_outside_cutoffs=1,
                    radial_derivative_upper=derivative_bound, uniform_kinetic_error_upper=ims_cost,
                    outside_nearest_centre_chord_squared_lower=outside_chord_lower_squared,
                    outside_energy_lower=outside_energy_lower,
                    local_ground_lower=local_ground_lower, local_constrained_second_lower=local_second_lower,
                    codimension=4, constraint='integral(g_lower(x)*chi_z(x)*psi(z exp(iX)) dx)=0 for z=0..3'),
                eigenvalue_certificate=dict(lambda1_lower=lambda1_lower,
                    lambda4_upper=lambda4_upper, lambda5_lower=lambda5_lower,
                    quartet_gap_lower=separation, quartet_energy_spread_upper=lambda4_upper - lambda1_lower,
                    full_continuous_rotor=True, finite_Peter_Weyl_cutoff_used=False,
                    isolated_rank_four_spectral_projector=True, one_state_per_centre_charge=True),
                spectral_comparison=dict(trial_to_high_spectral_leakage_squared_upper=leakage_squared,
                    projector_operator_norm_error_upper=projector_error,
                    canonical_polar_isometry_error_upper=polar_error,
                    isometry='V=P S (S P S)^(-1/2), restricted to the actual four-packet space',
                    actual_eigenbasis='V f_q, with f_q=(1/2)*sum_z i^(q*z)*f_z; each actual spectral charge block is one-dimensional',
                    trial_clock_bundle_operator_error_upper=trial_clock_error,
                    actual_spectral_clock_bundle_operator_error_upper=transport_error,
                    actual_transport_invertibly_controlled=transport_nonzero,
                    actual_bundle_smallest_singular_value_lower=max(0., 1 - transport_error),
                    actual_bundle_outside_band_leakage_squared_upper=(2 * transport_error - transport_error ** 2)
                        if transport_nonzero else 1.,
                    bundle='multiplication by U on L2(SU4) tensor C4; clock diag(1,i,-1,-i) tensor I4',
                    charge_convention='C psi(U)=psi(iU); charge-q=i^q; U maps q to q+1 mod4'))


def adverse_controls(quadratic, ell_min, kappa, recovery_alignment):
    """Reject meaningful boundary failures, then recover the accepted actual law."""
    natural_width = math.exp((math.log(kappa) - math.log(recovery_alignment)) / 4)
    cases = [dict(name='zero kinetic coefficient', kappa=0., alignment=recovery_alignment),
             dict(name='negative alignment', kappa=kappa, alignment=-1.),
             dict(name='nonfinite alignment', kappa=kappa, alignment=math.inf),
             dict(name='overlapping chart convention', kappa=kappa, alignment=recovery_alignment, inner=.8, outer=1.1),
             dict(name='zero transition width', kappa=kappa, alignment=recovery_alignment, inner=.1, outer=.1),
             dict(name='one radius without its partner', kappa=kappa, alignment=recovery_alignment, inner=.1),
             dict(name='thin annulus loses global IMS separation', kappa=kappa, alignment=recovery_alignment,
                  inner=30 * natural_width, outer=30.001 * natural_width),
             dict(name='weak law with broad geometry', kappa=kappa, alignment=kappa),
             dict(name='tiny core loses Gaussian normalization', kappa=kappa, alignment=recovery_alignment,
                  inner=1e-3 * natural_width, outer=.1)]
    evidence = []
    for case in cases:
        kwargs = {key: value for key, value in case.items() if key != 'name'}
        try:
            quartet_bound(quadratic, ell_min, **kwargs)
        except (CertificateRejected, OverflowError) as error:
            evidence.append(dict(case=case['name'], rejected=True, reason=str(error)))
        else:
            raise AssertionError(f"Adverse parameter was accepted: {case['name']}")
    recovery = quartet_bound(quadratic, ell_min, kappa, recovery_alignment)
    if not recovery['spectral_comparison']['actual_transport_invertibly_controlled']:
        raise CertificateRejected('Recovery must certify nonzero actual spectral transport, not merely a quartet')
    # A finite strong law must survive the receipt boundary even if J/kappa
    # itself is outside binary64. Energy bounds scale with the whole law;
    # the Gaussian packet precision does not.
    extreme = quartet_bound(quadratic, ell_min, 1e-150, 1e160)
    scaled = quartet_bound(quadratic, ell_min, 1e-140, 1e170)
    json.dumps(extreme, allow_nan=False)
    json.dumps(scaled, allow_nan=False)
    for certificate in (extreme, scaled):
        geometry = certificate['geometry']
        if not (0 < geometry['Haar_jacobian_lower'] < 1
                and geometry['inverse_metric_upper'] > 1
                and 0 < geometry['potential_quadratic_lower_factor'] < 1
                and 0 < certificate['anisotropic_Gaussian']['normalized_local_Haar_integral_lower'] < 1):
            raise AssertionError('A nonzero Haar chart lost its outward geometric/normalization bounds')
    for key in ('lambda1_lower', 'lambda4_upper', 'lambda5_lower', 'quartet_gap_lower'):
        if not math.isclose(scaled['eigenvalue_certificate'][key],
                            1e10 * extreme['eigenvalue_certificate'][key], rel_tol=1e-10):
            raise AssertionError(f'Finite strong-law energy homogeneity failed: {key}')
    transport_key = 'actual_spectral_clock_bundle_operator_error_upper'
    if not math.isclose(extreme['anisotropic_Gaussian']['alpha_min'],
                        scaled['anisotropic_Gaussian']['alpha_min'], rel_tol=1e-10):
        raise AssertionError('Finite strong-law Gaussian precision changed under common rescaling')
    return dict(rejections=evidence, positive_recovery_J=recovery_alignment,
                positive_recovery_gap_lower=recovery['eigenvalue_certificate']['quartet_gap_lower'],
                positive_recovery_transport_error_upper=recovery['spectral_comparison'][transport_key],
                finite_extreme_ratio_recovery=dict(
                    kappa=1e-150, J=1e160,
                    alpha_min=extreme['anisotropic_Gaussian']['alpha_min'],
                    quartet_gap_lower=extreme['eigenvalue_certificate']['quartet_gap_lower'],
                    common_energy_rescaling=1e10,
                    actual_transport_error_upper=extreme['spectral_comparison'][transport_key]))


def build(args):
    here = Path(__file__).resolve()
    immutable_hash = hashlib.sha256((here.parent / 'haar_centre_rotor.py').read_bytes()).hexdigest()
    if immutable_hash != IMMUTABLE_ROTOR_SHA256:
        raise CertificateRejected('Immutable Haar rotor source binding changed; review the proof before accepting')
    band = RotationBand(a=1., m=args.m, r=args.r, v=args.v, improved=False)
    data = uniform_data(band, 3, args.stiffness, 0., matrix_unit_basis())
    frame = np.asarray(data['frame'], dtype=complex)
    generators = np.asarray(HaarLink('minimal').T)
    controls = commutator_controls(frame, generators)
    quadratic = verified_quadratic(frame, generators, controls)
    # In an HS-orthonormal Hermitian traceless basis the full complex
    # commutator superoperator is exactly 2*Mbar. This encloses its global
    # coercivity too, without trusting rounded Clifford eigenvalues.
    ell_min = 2 * min(quadratic['eigenvalue_lower']) * (1 - DISPLAY_GUARD)
    certificates = [quartet_bound(quadratic, ell_min, args.kappa, j, args.inner_radius, args.outer_radius)
                    for j in args.alignments]
    if not any(c['spectral_comparison']['actual_transport_invertibly_controlled'] for c in certificates):
        raise CertificateRejected('At least one declared J must control actual spectral transport below unit error')
    recovery_j = max(args.alignments)
    adverse = adverse_controls(quadratic, ell_min, args.kappa, recovery_j)
    hashes = source_hashes()
    for filename in (here.name, here.with_suffix('.md').name, 'haar_centre_rotor.py', 'haar_centre_rotor.md'):
        hashes[filename] = hashlib.sha256((here.parent / filename).read_bytes()).hexdigest()
    # For ds^2=2Tr(dX^2), the SU4 Riemannian volume is 2^(15/2)*(2*pi)^9/6.
    haar_volume = 2 ** (15 / 2) * (2 * math.pi) ** 9 / 6
    return dict(model='Certified actual continuous-Haar scalar centre-well rotor quartet',
                parameters={key: value for key, value in vars(args).items() if key != 'output'},
                declared_quantum_law='H=kappa*(-Delta_SU4)+J*sum_mu ||[Phi_mu,U]||_HS^2 on L2(SU4,dHaar)',
                normalization='Tr(Ta Tb)=delta_ab/2; ds^2=2Tr(dX^2); exactly one left OR right Casimir',
                selected_source='uniform_data(unimproved RotationBand,3,stiffness,0,matrix_unit_basis())',
                selected_frame=dict(real=np.real(frame).tolist(), imaginary=np.imag(frame).tolist()),
                generator_input=dict(real=np.real(generators).tolist(), imaginary=np.imag(generators).tolist()),
                selected_gram_diagonal=np.diag(data['selected']['G']).tolist(),
                actual_commutator_controls=controls, verified_actual_quadratic=quadratic,
                certified_global_ell_min=ell_min,
                Haar_normalization=dict(Riemannian_volume=haar_volume, chart_density_constant=1 / haar_volume,
                                        convention='Probability Haar; all energy/transport quotients cancel c_H exactly'),
                global_quartet_certificates=certificates, adverse_parameter_controls=adverse,
                arithmetic=dict(exact_dyadic_matrix_verification=True, scalar_display_guard=DISPLAY_GUARD,
                    interpretation='Analytic inequalities with exact rational quadratic witness enclosure and conservative binary64 display rounding; not a formal proof-assistant artifact'),
                remaining_obligations=['Resolve tunnelling splittings/amplitudes beyond the certified energy-spread upper bound',
                    'Transfer to actual matter-coupled angular photon/Gauss Hamiltonian, not this scalar Casimir law',
                    'Construct covariant endpoint frames and spatial Gauss-invariant propagation/deconfinement',
                    'Solve unrestricted full-field quantum phase and control the joined spacetime limit',
                    'Establish an operational mechanical rotation experiment; pure gauge/global spatial 2pi remains +1'],
                excluded_claims=['derived fundamental quantum origin', 'Dirac electron', 'mechanical half-spin',
                                 'deconfined spatial Z4 phase', 'full-field quantum solution', 'canonical/full QED'],
                source_sha256=hashes, runtime=dict(numpy=np.__version__, scipy=scipy.__version__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alignments', nargs='+', type=float, default=[1e10, 1e18, 1e26])
    parser.add_argument('--kappa', type=float, default=1.)
    parser.add_argument('--m', type=float, default=.2)
    parser.add_argument('--r', type=float, default=.12)
    parser.add_argument('--v', type=float, default=.3)
    parser.add_argument('--stiffness', type=float, default=2.)
    parser.add_argument('--inner-radius', type=float)
    parser.add_argument('--outer-radius', type=float)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    numbers = [args.kappa, args.m, args.r, args.v, args.stiffness, *args.alignments]
    if not all(math.isfinite(value) for value in numbers):
        parser.error('All law coefficients must be finite')
    if min(args.kappa, args.stiffness, *args.alignments) <= 0:
        parser.error('Positive kappa, stiffness and alignments required')
    try:
        result = build(args)
    except (CertificateRejected, OverflowError, FloatingPointError) as error:
        parser.error(str(error))
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.write_text(payload + '\n')
    print(payload)


if __name__ == '__main__':
    main()
