#!/usr/bin/env python3
"""Reference photon energy-form controls; no microscopic K-log continuation claim."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.special import ive

ROOT = Path(__file__).resolve().parent
SOURCE_BYTES_AT_IMPORT = {name: (ROOT/name).read_bytes() for name in (
    'covariant_photon_energy_control.py', 'covariant_induced_photon.py',
    'sea_selection.py', 'rotation_band.py')}
from covariant_induced_photon import Parameters, WilsonConsumer, phi_frame


def rotor_global_bounds(beta: float, m: int) -> dict[str, float]:
    """All integer electric sectors, including m of order sqrt(beta)."""
    if beta < 1:
        raise ValueError('The proved global bound requires beta >= 1')
    pi = m/math.sqrt(beta)
    quadratic = pi*pi/2
    error = ((25/24)*pi*pi+pi**4/24)/beta
    return dict(pi=pi, quadratic=quadratic,
                lower=quadratic-pi**4/(24*beta),
                upper=quadratic+(25/24)*pi*pi/beta,
                absolute_error_upper=error,
                normalized_nearest_sector_slope_error_upper=(
                    abs(pi+.5/math.sqrt(beta))**3/6+abs(pi)+1/math.sqrt(beta))/beta)


def native_rotor_control() -> dict[str, Any]:
    rows = []
    for beta in (1., 4., 16., 128., 4096., 1e6):
        integers = sorted({0, 1, 2, 3, 4, *(
            round(x*math.sqrt(beta)) for x in (.25, .5, 1., 2., 4., 8.))})
        i0 = float(ive(0, beta))
        for m in integers:
            im = float(ive(m, beta))
            ip = float(ive(m+1, beta))
            if min(i0, im, ip) <= 0:
                raise ArithmeticError('Actual Bessel coefficient is not positive representable')
            actual = math.log(i0)-math.log(im)
            slope = (math.log(im)-math.log(ip))*math.sqrt(beta)
            free_slope = (m+.5)/math.sqrt(beta)
            bound = rotor_global_bounds(beta, m)
            guard = 128*np.finfo(float).eps*(1+abs(actual)+abs(bound['lower'])+abs(bound['upper']))
            slope_guard = 128*np.finfo(float).eps*math.sqrt(beta)*(1+abs(math.log(im))+abs(math.log(ip)))
            if actual < bound['lower']-guard or actual > bound['upper']+guard:
                raise ArithmeticError('Native Bessel energy violates analytic global enclosure')
            if abs(slope-free_slope) > bound['normalized_nearest_sector_slope_error_upper']+slope_guard:
                raise ArithmeticError('Native Bessel commutator slope violates enclosure')
            rows.append(dict(beta=beta, electric_integer=m, actual_energy=actual,
                             actual_normalized_slope=slope, free_normalized_slope=free_slope,
                             floating_comparison_guard=float(guard),
                             slope_floating_comparison_guard=float(slope_guard), **bound))
    return dict(rows=rows,
        proof=[
            'r_nu=I_nu(beta)/I_(nu-1)(beta) lies between beta/[nu-1/2+sqrt(beta^2+(nu+1/2)^2)] and beta/[nu-1/2+sqrt(beta^2+(nu-1/2)^2)], nu>=1. The lower bound follows directly from the three-term recurrence and the upper bound at nu+1.',
            'Writing S_m=sum_(nu=1)^m asinh((nu-1/2)/beta), each log-ratio correction is in [0,nu/beta^2], hence S_m<=H_beta(m)<=S_m+m(m+1)/(2beta^2).',
            'For Phi_m=integral_0^m asinh(x/beta) dx, concavity gives Phi_m<=S_m<=Phi_m+m(m+1)/(48beta^3). Integrating 1-1/sqrt(1+u^2)<=u^2/2 gives m^2/(2beta)-m^4/(24beta^3)<=Phi_m<=m^2/(2beta).',
            'For all integer m and beta>=1: |H_beta(m)-Pi^2/2|<=beta^-1[(25/24)Pi^2+Pi^4/24], Pi=m/sqrt(beta). This is not a fixed-m asymptotic or a low-electric-sector truncation.',
            'The unitary nearest-electric-sector commutator divided by g=beta^-1/2 differs from the quadratic slope by at most g^2[|Pi+g/2|^3/6+|Pi|+g]. Negative m follows by H(-m)=H(m).'],
        ratio_bound_source='https://arxiv.org/html/2608.03633v2',
        scope='Exact scalar compact electric convolution. Combining with magnetic multiplication and charged SF to bound the normalized joint K-log jets remains a separate obligation.')


def canonical_active_energy_control(p: Parameters) -> dict[str, Any]:
    gamma, frame = phi_frame(p)
    rows = []
    grid = np.linspace(-math.pi, math.pi, 9)
    for spacing, mass in itertools.product(p.spacings, p.masses):
        consumer = WilsonConsumer(gamma, spacing, mass)
        if not 0 <= consumer.m0 < 1:
            raise ValueError('The log8 proof requires 0<=m0<1')
        maximum = max(consumer.energy(np.array(q)) for q in itertools.product(grid, repeat=3))
        bound = math.log(7+consumer.m0)
        if maximum > bound+1e-12:
            raise ArithmeticError('Actual Wilson energy exceeds the exact global upper bound')
        rows.append(dict(spacing=spacing, physical_mass=mass, bare_m0=consumer.m0,
                         actual_grid_maximum=maximum, exact_global_upper=bound,
                         strict_log8_margin=math.log(8)-bound))
    return dict(frame=frame, rows=rows,
        proof='Put b0=1+m0 and t=sum(1-cos p_j). By Cauchy, sum sin^2 p_j<=2t-t^2/3. Thus 2cosh E<=b0+t+(1+2t-t^2/3)/(b0+t). Its derivative has positive numerator b0^2+2b0-1+(4/3)b0*t+(2/3)t^2. Its maximum at t=6 is (7+m0)+(7+m0)^-1, hence E<=log(7+m0)<log8. B<=8 and F<=3 alone were not used as an energy proof.',
        active_SF_relative_bound='At fixed normalized Phi, B>=1, B<=8, ||F||<=3, ||T1||,||T1^-1||<=128. A real coexact link homotopy with max angle delta has ||dT1||<=240 delta. The positive-square-root Sylvester equation and the actual free E<log8 give ||sqrt(T1(A))sqrt(T1(flat))^-1-I||<=4096 delta. For delta<=1/6, det(B)^1/2 and exterior sectors up to 4V give ||SF(A)SF(flat)^-1-I||<=expm1(16400 V delta). This is an SF endpoint bound, not a uniform microscopic log(K) bound across its null space.')


def gaussian_energy_form_control(p: Parameters, grade: int = 5, initial_energy: float = .25) -> dict[str, Any]:
    if grade < 0 or initial_energy < 0 or p.spatial_size % 2:
        raise ValueError('Nonnegative grade/energy and an even spatial box are required')
    omega = 2*math.asinh(math.sqrt(3))
    c = omega/(math.pi*math.sqrt(3))
    c2 = 13/(4*math.pi**2*c*c)
    photon_degree = 2*grade
    energy = initial_energy+2*grade*math.log(8)+photon_degree*omega
    a_norm = math.sqrt(2*c2*energy+2*math.sqrt(c2))
    pi_norm = math.sqrt(2*(math.sinh(omega)/omega)*energy+2*math.sinh(omega))
    curl_norm = 4*a_norm
    bound = max(a_norm, pi_norm, curl_norm)
    if bound >= 64:
        raise ArithmeticError('Actual requested energy-form word exceeds the declared B=64')
    return dict(photon_maximum_energy=omega, dispersion_lower_coefficient=c,
        finite_box_inverse_energy_square_sum_upper=c2,
        initial_reference_energy=initial_energy, maximum_grade=grade,
        maximum_photon_field_degree=photon_degree, maximum_intermediate_energy=energy,
        link_A_operator_norm_upper=a_norm, link_Pi_operator_norm_upper=pi_norm,
        plaquette_curl_operator_norm_upper=curl_norm, common_word_factor=64,
        proof=[
            'For each nonzero finite-box Fourier mode E=2asinh(|qhat|/2)>=c|p|, c=2asinh(sqrt3)/(pi sqrt3). An infinity-norm shell r contains at most 26r^2 modes. Summing r=1,...,L/2 proves V^-1 sum E^-2<=13/(4 pi^2 c^2), with no fitted speed or missing soft modes.',
            'With the actual Gaussian covariance <A A>=1/(2sinh E) and <Pi Pi>=sinh E/2, annihilation energy estimates and creation CCR give ||A_link P_[Hph<=e]||^2<=2 C2 e+2sqrt(C2), and ||Pi_link P||^2<=2(sinh Omega/Omega)e+2sinh Omega. Coexact projectors have diagonal at most one.',
            'Each linear photon field changes free energy by at most Omega. Each active bilinear changes fermion energy by at most 2log8. A product through grade n with at most 2n photon fields is therefore controlled by e_initial+2n log8+2n Omega. Curl is bounded by the triangle inequality over its four translated link fields.',
            'These are operator energy-form bounds, not pointwise coordinate confinement. They do not assert that a Euclidean chart-tail probability is a real-time leakage bound.'],
        scope='Exact Gaussian positive photon transfer and free canonical fermion reference. The joint normalized microscopic K-log jet estimates and actual prepared Gauss state admission must be supplied independently.')


def reference_linear_log_jet_control(p: Parameters) -> dict[str, Any]:
    """The actual first log derivative of SF(gA) Kph SF(gA)^dagger."""
    energy = gaussian_energy_form_control(p, 8, .25)
    omega = energy['photon_maximum_energy']
    slots = 4*p.spatial_size**3
    coefficient = 2*1026*8*math.exp(omega/2)
    log_h1 = math.log10(coefficient)+math.log10(slots)+math.log10(64)
    fifth = 5*(-p.beta_power/2+math.log10(p.time_slices)+log_h1)-math.lgamma(6)/math.log(10)
    return dict(maximum_exterior_slots=str(slots), linear_log_jet_coefficient=coefficient,
        norm_H1_on_reference_energy_core_log10_upper=log_h1,
        pure_fifth_H1_Dyson_word_log10_upper=fifth,
        reference_energy_form_through_terminal_grade8=energy,
        proof=[
            'At g=0, the actual free T1 has spectrum in [1/8,8]. The real link derivative obeys ||T1_prime||<=240 B_field. The positive-root Sylvester equation therefore gives ||sqrt(T1)_prime sqrt(T1)^-1||<=960 B_field, enclosed by 1024 B_field. Exterior differentiation costs at most 4V slots; the det(B)^1/2 derivative costs at most 6V B_field, enclosed together by 1026*(4V)*B_field.',
            'Write R1=SF_prime SF0^-1. For K(g)=SF(gA) Kph SF(gA)^dagger, K1=R1 K0+K0 R1^dagger. The neutral normalization derivative is zero: R1 is linear in a nonzero coexact Gaussian field, and every free photon-ground diagonal matrix element vanishes.',
            'For strictly positive reference K0, Phi(X)=integral_0^infinity sqrt(K0)(K0+s)^-1 X sqrt(K0)(K0+s)^-1 ds is unital completely positive. H1=-Phi(K0^-1/2 K1 K0^-1/2). Stinespring contraction also applies on a finite free-energy core because its projection commutes with every integral Kraus operator.',
            'Whitening R1 costs at most 8 for its single fermion bilinear, since each actual signed free band energy is below log8, and exp(Omega/2) for its single linear photon field. Both boundary terms give ||H1 P_energy||<=2*1026*8*exp(Omega/2)*(4V)*B_field. No entrywise Schur bound was substituted for an operator norm.',
            'Five H1 insertions act on the unit-normalized reference source. The ordered time-simplex volume is Nt^5/5!, giving the reported pure fifth-word norm bound. The grade8 energy-form core covers every intermediate field and signed-band transition. This is not the sum of all mixed H2..H5 remainder words.'],
        scope='Exact first derivative of the Gaussian-photon/active-SF comparison transfer. No derivative is taken through the null space of microscopic Haar K. Higher joint log jets, source-boundary derivatives, and their microscopic common-range transfer remain separate obligations.')


def run(p: Parameters) -> dict[str, Any]:
    inputs = dict(parameters=asdict(p), maximum_grade=5, maximum_terminal_word_grade=8,
                  initial_reference_energy=.25, linear_SF_coefficient_upper=1026,
                  physical_scope='Gaussian reference energy-form and exact scalar compact rotor')
    encoded = json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()
    return dict(input_descriptor=inputs, input_descriptor_sha256=hashlib.sha256(encoded).hexdigest(),
                source_at_import={k:hashlib.sha256(v).hexdigest() for k,v in SOURCE_BYTES_AT_IMPORT.items()},
                source_bytes_at_import={k:v.decode() for k,v in SOURCE_BYTES_AT_IMPORT.items()},
                rotor=native_rotor_control(), active=canonical_active_energy_control(p),
                photon_energy_form=gaussian_energy_form_control(p),
                reference_linear_log_jet=reference_linear_log_jet_control(p),
                microscopic_joint_admission_proved=False,
                unproved_obligations=['Normalized charged/photon joint K-log derivatives on the actual common range, including source halfsteps',
                                      'Full interacting Gauss prepared source Grams and actual neutral-to-charged spectral cutoff',
                                      'Inclusive charged connected receiver Born lower after the finite fifth Duhamel and microscopic comparison errors'])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    result = run(Parameters(kappa_power=400))
    result['runtime_seconds'] = time.monotonic()-started
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(source=result['source_at_import'], input_sha256=result['input_descriptor_sha256'],
                         rotor_sectors=len(result['rotor']['rows']),
                         energy_form=result['photon_energy_form'],
                         microscopic_joint_admission_proved=False,
                         runtime_seconds=result['runtime_seconds'])))


if __name__ == '__main__':
    main()
