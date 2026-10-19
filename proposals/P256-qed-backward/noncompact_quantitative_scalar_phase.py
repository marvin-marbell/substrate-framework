#!/usr/bin/env python3
"""Quantitative original a1 scalar phase, including the full microscopic domain.

Finite gigantic parameters are represented by exact power expressions, never by
an N^3 grid or a materialized power-tower integer. Native small-box consumers are
separate diagnostics; the admission constants use rational/analytic proofs.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy.linalg import block_diag, expm

from bulk_flat_holonomy_bound import assess
from bulk_quantum_selection import BulkQuantumLaw
from local_orbital_stability import periodic_grid
from noncompact_native_metric_admission import certificate, validate_certificate
from original_charged_principal import original_frame
from rotation_band import RotationBand
from sea_selection import SELECTED, gram
from unrestricted_field_minimum import BARE

ROOT = Path(__file__).resolve().parent
N_PHASE = 2**96
G_FLOOR = Fraction(1, 20)
REDUCIBLE_CONTINUUM_GAP = Fraction(1, 2**88)
REDUCIBLE_FINITE_GAP = Fraction(1, 2**89)
SOURCES = ('noncompact_quantitative_scalar_phase.py',
           'noncompact_quantitative_scalar_phase.md', 'bulk_flat_holonomy_bound.py',
           'bulk_flat_holonomy_bound.md', 'original_slow_kinetic_bo.md',
           'noncompact_native_metric_admission.py', 'noncompact_native_metric_admission.md',
           'bulk_quantum_selection.py', 'rotation_band.py', 'sea_selection.py',
           'original_charged_principal.py', 'unrestricted_field_minimum.py')


def rational(value: Fraction | int) -> str:
    return str(value)


def power_expression(coefficient: int, inner_exponent: int, sign: int = 1) -> dict[str, Any]:
    """Exact value 2^(sign*coefficient*2^(2^inner_exponent))."""
    if coefficient < 1 or inner_exponent < 1 or sign not in (-1, 1):
        raise ValueError('Positive integer tower data and sign +/-1 required')
    return dict(base=2, exponent_sign=sign, exponent_coefficient=coefficient,
                exponent_tower=dict(base=2, inner_base=2,
                                    inner_exponent=inner_exponent),
                exact_expression=f'2^({sign*coefficient}*2^(2^{inner_exponent}))')


def phase_bounds(size: int) -> dict[str, Any]:
    """Closed-form positive admission; no minimization or grid in this function.

    The universal height majorant is deliberately enormous. Its precise
    quantified polynomial and resultant arithmetic are in the paired proof.
    """
    if size < N_PHASE:
        raise ValueError(f'Global phase needs N >= {N_PHASE}; N_metric=25 is not N_phase')
    volume = size**3
    variables = 100000*volume**2
    exponent = 100*variables**2
    regular_radius = Fraction(1, 2**20*volume**3)
    A4 = Fraction(1, 4*volume)
    BH = Fraction(22*volume, 5)
    # rho0^3 >= 2 BH/A4, with rho0 >=1; no rounded cube root.
    rho0 = 1+36*volume**2
    assert Fraction(rho0**3) >= 2*BH/A4
    return dict(a=1, N=size, V=volume, N_phase=N_PHASE,
        proof_kind='analytic rational Fourier/Clifford plus explicit polynomial CAD root-height bound',
        all_64_native_full10_metrics=True, centre_offsets_retained=True,
        flat=dict(Gram_Frobenius_norm_upper=3, grid_error_prefactor_upper=10,
                  Gram_eigenvalue_lower=rational(G_FLOOR),
                  reducible_continuum_gap_per_site=rational(REDUCIBLE_CONTINUUM_GAP),
                  reducible_finite_gap_per_site=rational(REDUCIBLE_FINITE_GAP),
                  slack_defect_denominator=4096, quadratic_richness_inverse_upper=2**16,
                  normalized_relation_squared_upper_factor=2**27,
                  monomial_relation_admission=rational(Fraction(1, 2**24))),
        regular=dict(ambient_tube_radius=rational(regular_radius),
                     source_field_gauge_Gram_lower='1/40',
                     tube_field_gauge_Gram_lower='1/160',
                     whole_heavy_gap_lower='1/20', tube_whole_heavy_gap_lower='1/40',
                     metric='raw Hermitian field HS/4 and link HS; actual Ta=Ba/sqrt(8)'),
        full_domain=dict(quantified_variable_upper=variables, initial_degree_upper=8,
                         initial_integer_coefficient_bits_upper=100*variables,
                         initial_polynomial_count_upper=2*variables**4,
                         projection_stages_upper=variables,
                         projection_degree_and_coefficient_bits_upper_expression=f'2^(2^{exponent})',
                         height_inner_exponent=exponent,
                         Lambda_phase=power_expression(8, exponent),
                         global_quadratic_restoration=power_expression(8, exponent+1, -1),
                         statement='For Lambda>=Lambda_phase, U_Lambda-vstar >= c_phase*dist(y,Z_N)^2 on the entire field/link domain; Z_N includes every globally minimizing retained centre orbit.',
                         core_barrier='Lambda*c_phase*epsilon^2 outside ambient-distance epsilon cores of Z_N',
                         all_64_core_barrier='The same lower bound holds outside the union of epsilon cores of all64 native centre-class orbits; nonminimal offsets are never erased.'),
        coercivity=dict(A4=rational(A4), BH_upper=rational(BH), rho0=rho0,
                        statement='V0 >= A4*rho^4-BH*rho; rho>=rho0 implies U_Lambda-vstar >= A4*rho^4/2',
                        charged_light_note='Use BL=2*NL*(11/10), C0=BL*rho0, C1=2*BL/A4 on the complete light/reference sector'))


def analytic_bloch_metric_enclosure(size: int, cert: dict[str, Any]) -> dict[str, Any]:
    """Full10 all-centre enclosure at the legal N; no N^3 enumeration."""
    if size < N_PHASE or not validate_certificate(cert):
        raise ValueError('Global legal N and certified full10 native supplier required')
    if Fraction(cert['analytic_strip']['sigma']) != Fraction(1, 2):
        raise ValueError('This rational exponential-tail majorant uses sigma=1/2')
    majorant = Fraction(cert['analytic_strip']['matrix_gradient_integrand_HS_majorant'])
    if majorant >= 64:
        raise ValueError('Native analytic-strip majorant exceeds the paid value64')
    # e^(-N/2)<=2^(-floor(N/2)); N>=256 makes the exact 3D tail
    # ((1+t)/(1-t))^3-1 <=32t. Bound its gradient contribution
    # by2^-100 rather than materializing 2^(N/2).
    assert size >= 256
    native_error = Fraction(1, 2**100)
    assert 64*32*Fraction(1, 2**128) <= native_error
    residual = Fraction(cert['full_ten_gradient']['Frobenius_residual_upper'])
    continuum_error = Fraction(cert['continuum_quadrature_error_upper'])
    distance = (residual+continuum_error+native_error)/2
    declared = Fraction(1, 2**38)
    if distance >= declared:
        raise ValueError('Interval supplier fails the declared large-N metric enclosure')
    g0, gs = map(Fraction, cert['trial_decimal'])
    return dict(N=size, allocated_grid_points=0, full10=True,
                trial_decimal=cert['trial_decimal'],
                Frobenius_distance_upper=rational(declared),
                diagonal_intervals=dict(temporal=[rational(g0-declared), rational(g0+declared)],
                                        spatial_each=[rational(gs-declared), rational(gs+declared)]),
                each_offdiagonal_interval=[rational(-declared), rational(declared)],
                all64_shift_descriptors=[dict(centre=list(theta),
                    momentum_shift='pi*centre/(2N)', exact_minimizer='G_Ntheta, not replaced by trial')
                    for theta in itertools.product(range(4), repeat=3)],
                native_gradient_tail_upper=rational(native_error),
                proof='Native full-matrix strong convexity plus certified continuum residual and analytic real-shift Bloch tail',
                offsets='Every actual native classical and exact scalar quantum offset remains retained, not rounded to zero')


def analytic_controls() -> dict[str, Any]:
    """Check decision-changing inequalities using integer/rational arithmetic."""
    r, v, temporal = Fraction(3, 25), Fraction(3, 10), Fraction(14, 25)
    # Bounds obtained from the distinct Fourier characters in the paired proof.
    richness = (1/r**4+3*16/v**4+6/v**4+
                6/(2*v**2*temporal**2))
    assert richness < 2**16
    assert 4/G_FLOOR**2*2**16 < 2**27
    assert Fraction(1, 16*6**3) > Fraction(1, 4096)
    assert Fraction(4096, 2**24) < Fraction(1, 2)
    assert Fraction(20, N_PHASE) < REDUCIBLE_CONTINUUM_GAP/2
    assert Fraction(1, 2**48*2**27*4096) > REDUCIBLE_CONTINUUM_GAP
    assert Fraction(1, 10000) > REDUCIBLE_CONTINUUM_GAP
    assert Fraction(11, 10)**2 > Fraction(23, 25)**2+3*v**2
    assert 8*Fraction(11, 10)**2 < 3**3
    # CD <=10, using pi<=22/7 and smax<=3.
    assert 4*Fraction(22, 7)**2*(9*r**2+3*v**2)*6 < 100
    # Robust majorants also enclose the executing binary rational parameters.
    rlo, vlo, clo = Fraction(119, 1000), Fraction(299, 1000), Fraction(556, 1000)
    rup, vup, mup = Fraction(121, 1000), Fraction(301, 1000), Fraction(201, 1000)
    robust_richness = 1/rlo**4+54/vlo**4+3/(vlo**2*clo**2)
    assert robust_richness < 2**16
    assert (mup+6*rup)**2+3*vup**2 < Fraction(11, 10)**2
    assert 4*Fraction(22, 7)**2*(9*rup**2+3*vup**2)*6 < 100
    assert Fraction(1, 16)-clo < Fraction(-1, 4)
    assert 4*Fraction(1, 5)**2*G_FLOOR > Fraction(1, 20)**2
    result: dict[str, Any] = dict(richness_inverse_majorant=rational(richness),
        binary_parameter_robust_richness_majorant=rational(robust_richness),
        monomial_Gram_row_error_upper=rational(Fraction(4096, 2**24)),
        finite_separation_debt=rational(Fraction(20, N_PHASE)),
        N_phase_exact=N_PHASE, all_analytic_integer_rational_inequalities=True)
    # Exercise symbolic gigantic parameters without constructing a power tower.
    bounds = phase_bounds(N_PHASE)
    assert bounds['full_domain']['Lambda_phase']['exponent_sign'] == 1
    assert bounds['full_domain']['global_quadratic_restoration']['exponent_sign'] == -1
    # Exercise the exact full-projection determinant majorant at bounded
    # depths. The proof's induction, not these examples, covers everyM.
    projection_rows = []
    for variables in (1, 2, 4, 8):
        degree, bits = 8, 100*variables
        logcount = (2*variables**4).bit_length()
        initial_a = max(degree.bit_length(), bits.bit_length(),
                        logcount.bit_length(), variables.bit_length())
        a = initial_a
        for _ in range(variables):
            degree, bits, logcount = (2*degree**2,
                16*degree**2*(bits+logcount+variables+1),
                4*logcount+16*degree**2)
            a = 3*a+6
            assert max(degree.bit_length(), bits.bit_length(),
                       logcount.bit_length()) <= a+1
        assert a < 2**(100*variables**2)
        projection_rows.append(dict(variables=variables,
            final_degree_bit_length=degree.bit_length(),
            final_coefficient_bit_bound_bit_length=bits.bit_length(),
            induction_upper=a))
    result['exact_projection_majorant_controls'] = projection_rows
    try:
        phase_bounds(25)
    except ValueError:
        result['N_metric_as_N_phase_shortcut_rejected'] = True
    else:
        raise AssertionError('Metric admission improperly promoted to global phase')
    return result


def native_controls(size: int, cert: dict[str, Any]) -> dict[str, Any]:
    """Actual native whole-heavy/reducer/full-field controls, not proof floats."""
    band = RotationBand(**BARE)
    law = BulkQuantumLaw(size, band=band)
    reference_size = 8
    reference = 2*np.pi*periodic_grid(reference_size)/reference_size
    pauli = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    rng = np.random.default_rng(222960)
    reducible = np.array([block_diag(np.einsum('a,aij->ij', rng.normal(size=3)*.25, pauli),
                                    np.einsum('a,aij->ij', rng.normal(size=3)*.25, pauli))
                          for _ in range(4)])
    reducible[0] += np.diag([.1, .1, -.1, -.1])
    blocks = [([0, 1], [.7, -.4, 1.2]), ([2, 3], [-.7, .4, -1.2])]
    reduced = assess(reducible, blocks, size, band, 2, reference, reference_size, native=True)
    cartan = np.array([np.diag(row-row.mean())*.12 for row in rng.normal(size=(4, 4))])
    cartan[0] = np.diag([5., 1., 1., -7.])
    angles = np.array([[.5, -.3, .7], [-.2, .6, -.5], [.8, -.9, .1]])
    angles = np.vstack([angles, -angles.sum(axis=0)])
    cartan_blocks = [([i], row.tolist()) for i, row in enumerate(angles)]
    transferred = assess(cartan, cartan_blocks, size, band, 2, reference, reference_size, native=True)
    if transferred['forced_block_error'] <= 1e-5:
        raise AssertionError('Whole-half occupation-transfer adverse boundary not reached')
    rows = []
    for centre in itertools.product(range(4), repeat=3):
        solution, frame = original_frame(band, size, np.asarray(centre)*np.pi/2, True)
        h = law.onebody(np.tile(frame, (law.volume, 1, 1, 1)),
                        law.toron_links(np.asarray(centre)*np.pi/2))
        eigenvalues = np.linalg.eigvalsh(h)
        actual = float(np.sum(gram(frame)**2)-eigenvalues[2*law.volume:].sum()/law.volume)
        error = abs(actual-solution['value'])
        if error > 2e-9:
            raise AssertionError('Native full10 whole-heavy centre consumer disagrees')
        rows.append(dict(centre=list(centre), energy_per_site=actual,
                         full10_Gram=solution['G'].tolist(), native_error=error,
                         offdiagonal_norm=float(np.linalg.norm(solution['G']-np.diag(np.diag(solution['G']))))))
    if max(row['offdiagonal_norm'] for row in rows) < 1e-7:
        raise AssertionError('All64 adverse native control failed to expose offdiagonal data')
    # No uniform/Clifford restriction: perturb every site, every field and link.
    fields = np.tile(np.sqrt([.42, .063, .063, .063])[:, None, None]*SELECTED,
                     (law.volume, 1, 1, 1))
    perturbation = np.einsum('xma,aij->xmij', rng.normal(size=(law.volume, 4, 15))*.08,
                            law.basis)
    fields += perturbation
    links = np.array([[expm(1j*np.einsum('a,aij->ij', rng.normal(size=15)*.1, law.generators))
                       for _ in range(3)] for _ in range(law.volume)])
    h = law.onebody(fields, links)
    vals, vectors = np.linalg.eigh(h)
    projector = vectors[:, 2*law.volume:]@vectors[:, 2*law.volume:].conj().T
    elastic = float(np.sum(law.grams(fields)**2))
    actual_v0 = elastic-float(vals[2*law.volume:].sum())
    polynomial_v0 = elastic-float(np.trace(projector@h).real)
    W, P = law.link_defects(fields, links)
    rho = float(np.sqrt(np.trace(law.grams(fields), axis1=1, axis2=2).sum()))
    coercive = rho**4/(4*law.volume)-(22*law.volume/5)*rho
    if abs(actual_v0-polynomial_v0) > 2e-8 or actual_v0 < coercive-2e-8 or min(W, P) <= 0:
        raise AssertionError('Full-domain actual projector/endpoint/coercivity control failed')
    if not validate_certificate(cert):
        raise AssertionError('Native full10 interval supplier not admitted')
    return dict(native_size=size, native_dimension=4*law.volume,
                actual_whole_occupation=2*law.volume, retained_centre_count=len(rows),
                all64_native_centres=rows, reducible_noncentral=reduced,
                Cartan_occupation_transfer=transferred,
                full_nonuniform=dict(actual_V0=actual_v0, projector_polynomial_V0=polynomial_v0,
                                     quartic_coercivity_lower=coercive, rho=rho, W=W, P=P,
                                     V1=W/40+7*P/10),
                scope='Actual small native consumer diagnostics; global large-N admission uses analytic bounds, not these floating spectra')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=N_PHASE)
    parser.add_argument('--native-size', type=int, default=3)
    args = parser.parse_args()
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}
    cert: dict[str, Any] = certificate(64, 128)
    if cert['continuum_box']['spatial_eigenvalue_lower'] <= .06:
        raise AssertionError('Continuum G floor required by Fourier-Clifford argument absent')
    if cert['all_real_shifts_all_N_ge_N_metric_box']['spatial_eigenvalue_lower'] <= .06:
        raise AssertionError('Every-real-shift native G floor absent')
    report: dict[str, Any] = dict(topic_key='quantitative-original-whole-heavy-scalar-phase-a1',
                  phase=phase_bounds(args.size), analytic_controls=analytic_controls(),
                  native_metric_certificate=cert, actual_native_controls=native_controls(args.native_size, cert),
                  legal_N_analytic_Bloch_metrics=analytic_bloch_metric_enclosure(args.size, cert),
                  source_sha256=hashes, all_checks_pass=True,
                  achievement='Existence-only N0 replaced by exact N_phase=2^96 and a closed-form finite full-domain Lambda/core-restoration majorant.',
                  limitations=['The full-domain algebraic height majorant is intentionally astronomical, not an efficient simulation prescription.',
                               'No scalar quantum well energy is substituted by a harmonic vacuum; all64 exact offsets remain source-owned.',
                               'This supplies scalar phase admission, not the joined particle/photon/scattering outcome.'])
    if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest for name, digest in hashes.items()):
        raise RuntimeError('Executed scalar suppliers changed during run')
    report['source_and_input_unchanged'] = True
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output), N_phase=N_PHASE,
                         retained_centres=report['actual_native_controls']['retained_centre_count'],
                         all_checks_pass=True)))


if __name__ == '__main__':
    main()
