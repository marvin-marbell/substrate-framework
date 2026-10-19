#!/usr/bin/env python3
"""All-Haar/all-phase full-coordinate Gauss packet, not a phase or ground solution."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.linalg import expm
from scipy.special import gammaln

from full_band_gauss_section import FullBandGaussSection
from full_band_packet_bound import bump, packet_point_controls
from local_orbital_stability import uniform_data
from quantum_su4_gauss import norm, require_small
from rotation_band import RotationBand
from unrestricted_field_minimum import (pair_clifford_reference,
                                       source_hashes as reference_source_hashes)


def source_hashes():
    root = Path(__file__).resolve().parent
    return {**reference_source_hashes(), **{
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ('full_haar_field_packet.py', 'full_haar_field_packet.md')}}


def log_field_seed_density(fields, anchor_fields, radius, basis):
    """Normalized 120-ball seed density; its SU4^2 average is the scalar packet.

    Both normalized Haar and both normalized phase scalar factors are exactly 1.
    The average is an exact compact integral, not a finite gauge sample.
    """
    coordinates = np.einsum('xmij,aji->xma', fields - anchor_fields, basis).real / 4
    amplitude = bump(float(np.sum(coordinates**2)), radius)
    if amplitude <= 0:
        return -np.inf
    log_z = 60 * np.log(np.pi) + 120 * np.log(radius) + gammaln(5) - gammaln(65)
    return float(2 * np.log(amplitude) - log_z)


def anchor_constants(model, fields):
    """Read the owner law, requiring an identical symmetric temporal anchor."""
    if model.scale != 1. or min(model.onsite) <= 0:
        raise ValueError('Require the shared positive-onsite, whole-heavy-scale-one law')
    if norm(fields[0] - fields[1]) > 1e-11:
        raise ValueError('Require identical endpoint anchor fields')
    temporal = np.linalg.eigvalsh(fields[0, 0])
    b = float(np.mean(np.abs(temporal)))
    if norm(temporal - np.array([-b, -b, b, b])) > 1e-10:
        raise ValueError('Require a symmetric rank-two temporal anchor, not arbitrary-field positivity')
    # Outward margins protect the displayed strict inequalities from roundoff;
    # the proof uses exact matrix norms, not numerical U/A extrema.
    margin = 1e-10 * max(1., norm(fields), max(model.onsite))
    endpoint = [sum(abs(p[mu]) * float(np.linalg.norm(fields[x, mu], 2))
                    for mu in range(4))
                for p in model.paths for x in range(2)]
    return dict(b_lower=b - margin, b_upper=b + margin,
                b_anchor=b, outward_margin=margin,
                onsite_min=float(min(model.onsite)), onsite_max=float(max(model.onsite)),
                heavy_onsite=float(model.onsite[0]), light_onsite=float(model.onsite[1]),
                path_norm=float(max(np.linalg.norm(p) for p in model.paths)),
                path_component_abs=np.max(np.abs(np.asarray(model.paths)), axis=0),
                endpoint_operator_upper=float(max(endpoint) + margin),
                rho_anchor=norm(fields) / 2,
                gram_anchor=norm(np.einsum('xmij,xnji->xmn', fields, fields).real / 4))


def full_haar_bound(model, fields, *, radius, kappa=2., J=.2, ue4=.3, ue1=.4,
                    magnetic=.7) -> dict[str, Any]:
    """Uniform inequalities on every U in SU4 and every A in T^2.

    No matter eigenvalues or finite U/A probes enter these constants.
    """
    if not all(np.isfinite(v) for v in (radius, kappa, J, ue4, ue1, magnetic)):
        raise ValueError('Finite radius and bare coefficients required')
    if min(radius, kappa, ue4, ue1) <= 0 or min(J, magnetic) < 0:
        raise ValueError('Positive radius/stiffness/electrics and nonnegative J/K required')
    a = anchor_constants(model, fields)
    R, p = radius, a['path_norm']
    temporal_lower = a['b_lower'] - 2 * R
    endpoint_upper = a['endpoint_operator_upper'] + np.sqrt(2) * p * R
    m0 = a['onsite_min'] * temporal_lower - endpoint_upper
    if temporal_lower <= model.tolerance or m0 <= model.tolerance:
        raise ValueError('Failed all-Haar positive-anticommutator inequality c*(b-2R)>F+sqrt(2)*p*R')
    emax = a['onsite_max'] * (a['b_upper'] + 2 * R) + endpoint_upper
    local_gap, global_gap = 2 * temporal_lower, 2 * m0
    beta_lower = m0 / emax
    cos2 = (1 + beta_lower) / 2
    weight_lower = cos2**8
    if weight_lower <= model.tolerance:
        raise ValueError('Failed quantitative actual filled-plane overlap certificate')
    # Ky Fan tests the onsite positive four-plane: off-diagonal hopping has
    # zero trace there. Tracelessness gives |Tr(P_rank2 dPhi)|<=||dPhi||HS.
    heavy_upper = -a['heavy_onsite'] * (4 * a['b_lower'] - 2 * np.sqrt(2) * R)
    light_upper = -a['light_onsite'] * (4 * a['b_lower'] - 2 * np.sqrt(2) * R)
    gram_upper = a['gram_anchor'] + 2 * a['rho_anchor'] * R + R**2
    elastic_upper = kappa / 2 * gram_upper**2
    # W <= 2 sum_(x,mu) Tr Phi_xmu^2: at equal fields this is
    # precisely 4 sum_mu Tr Phi_mu^2, never a centre-link estimate.
    W_upper = 8 * (a['rho_anchor'] + R)**2
    alignment_upper = J * W_upper / 8
    C = float(elastic_upper + alignment_upper + heavy_upper)
    if C >= 0:
        raise ValueError('Failed strictly negative all-Haar uniform leading-energy inequality')
    sqrt_weight = np.sqrt(weight_lower)
    dh_components = 2 * a['onsite_max'] * np.array([1., 0., 0., 0.]) + a['path_component_abs']
    # Actual filled rank-four projector: ||dPi||<=sqrt(4)||dP||.
    # Positive Sylvester formula: ||dP||<=||dh||/gap, without a factor two.
    # The paired local epsilon state has ||dchi||<=2||dP_local||.
    field_derivatives = (4 / local_gap * np.array([1., 0., 0., 0.])
                         + 4 * dh_components / global_gap) / sqrt_weight
    colour_derivative = 4 * endpoint_upper / (np.sqrt(2) * global_gap * sqrt_weight)
    phase_derivative = endpoint_upper / (global_gap * sqrt_weight)
    field_metric = float(30 * np.sum(field_derivatives**2))
    colour_metric = float(15 * colour_derivative**2)
    phase_metric = float(2 * phase_derivative**2)
    field_fisher = 5120 / R**2
    T = float((field_fisher + field_metric) / 2)
    # Scalar colour/EM Fisher vanishes exactly. Uniform phases also give
    # <cos(A0+A1)>=0 since the actual section is pointwise normalized.
    B = float(ue4 * colour_metric + ue1 / 2 * phase_metric + max(0., light_upper))
    y = (T + np.sqrt(T*T + 4 * (-C) * B)) / (2 * (-C))
    threshold = float(max(1., y*y))
    log_z = float(60 * np.log(np.pi) + 120 * np.log(R) + gammaln(5) - gammaln(65))
    return dict(radius=R, anchor={key: (value.tolist() if isinstance(value, np.ndarray) else value)
                                for key, value in a.items()},
        temporal_absolute_eigenvalue_lower=temporal_lower, local_gap_lower=local_gap,
        all_U_A_endpoint_operator_upper=endpoint_upper,
        anticommutator_m0=m0, onebody_operator_Emax=emax, global_gap_lower=global_gap,
        compressed_beta_lower=beta_lower, each_principal_cosine_squared_lower=cos2,
        each_species_filled_overlap_squared_lower=cos2**4,
        actual_Pauli_joint_projection_weight_lower=weight_lower,
        gram_Frobenius_upper=gram_upper, elastic_upper=elastic_upper,
        exact_full_W_upper=W_upper, alignment_upper=alignment_upper,
        heavy_KyFan_onsite_upper=heavy_upper, light_KyFan_onsite_upper=light_upper,
        C_Lambda=C, field_onebody_derivatives_each_component=dh_components.tolist(),
        section_derivatives=dict(field_each_component=field_derivatives.tolist(),
                                 colour_each=colour_derivative, phase_each=phase_derivative),
        section_metric_upper=dict(fields=field_metric, colour=colour_metric, phases=phase_metric),
        scalar_envelope_fisher=dict(fields=field_fisher, colour=0., phases=0.),
        magnetic_exact_packet_expectation=0., field_log_normalizer=log_z,
        scalar_amplitude_log_upper=-log_z / 2,
        trial_total_rho_support_upper=float(np.sqrt(2) * (a['rho_anchor'] + R)),
        T_sqrt_Lambda=T, B_constant=B,
        bound='E0 <= Etrial <= C*Lambda + T*sqrt(Lambda) + B; Lambda>=1',
        strict_negative_for_Lambda_greater_than=threshold,
        negative_energy_at_four_times_threshold=float(C * 4 * threshold + T * np.sqrt(4 * threshold) + B))


def select_radius(model, fields, *, initial_radius=.2, radius=None, **law):
    """Reject a wide packet, then repair its FIELD support, never gauge support."""
    failure = None
    try:
        first = full_haar_bound(model, fields, radius=initial_radius, **law)
    except ValueError as error:
        first, failure = None, str(error)
    if radius is not None:
        selected = full_haar_bound(model, fields, radius=radius, **law)
        return selected, dict(initial_radius=initial_radius, first_rejection=failure,
                              first_valid_bound=first, route='explicit_certified_field_radius')
    candidates, rejected = [], []
    # Selecting among certified scalar inequalities is not U/A sampling.
    for candidate in np.linspace(.005, .08, 76):
        try:
            candidates.append(full_haar_bound(model, fields, radius=float(candidate), **law))
        except ValueError as error:
            rejected.append(dict(radius=float(candidate), reason=str(error)))
    if not candidates:
        raise ValueError('No legal negative all-Haar packet in the certified radius selection; no narrow-gauge fallback')
    selected = min(candidates, key=lambda b: b['strict_negative_for_Lambda_greater_than'])
    return selected, dict(initial_radius=initial_radius, first_rejection=failure,
        first_valid_bound=first, route='all_Haar_all_phase_field_support_repair',
        candidate_rule='76 deterministic field radii .005:.001:.080; each accepted only by analytic uniform inequalities',
        valid_candidate_count=len(candidates), rejected_candidates=rejected,
        no_U_or_phase_samples_in_certificate=True)


def exact_word_matrix_units(model):
    """Re-evaluate actual two-word code amplitudes with exact fourth roots.

    No rounding of measured transfers and no supplied wedge signs occur.
    The only normalization is the exact two-term code projection factor 1/2.
    """
    code, matter = model.code, model.matter
    result, float_builder_error = [], 0.
    def projected(mask):
        first = code.state(mask, sign=1)[0][0]
        second, phase = code.S.action(first)
        return ((first, 1 + 0j), (second, phase))
    for species in range(2):
        # Discover physical mode labels from the owner's actual diagonal units.
        occupations = np.array([[(mask >> bit) & 1 for mask in matter.masks]
                                for bit in range(16)])
        modes = [next(bit for bit in range(16)
                      if np.array_equal(occupations[bit], model.units[species][i, i].diagonal().real))
                 for i in range(8)]
        units = {}
        for i in range(8):
            for j in range(8):
                rows, columns, amplitudes = [], [], []
                word = (None if i == j else
                        code.words.path_A(code.words.path(modes[j], modes[i])).times_i(-1))
                for column, mask in enumerate(matter.masks):
                    if not (mask >> modes[j]) & 1 or (i != j and (mask >> modes[i]) & 1):
                        continue
                    new = mask if i == j else mask ^ (1 << modes[j]) ^ (1 << modes[i])
                    amplitude = 1 + 0j
                    if word is not None:
                        target = dict(projected(new))
                        numerator = 0j
                        for raw, phase in projected(mask):
                            out, path_phase = word.action(raw)
                            numerator += target.get(out, 0j).conjugate() * path_phase * phase
                        amplitude = numerator / 2
                        if amplitude not in (1, -1, 1j, -1j):
                            raise AssertionError('Exact raw Pauli word escaped a unit code transition')
                    rows.append(matter.index[new]); columns.append(column); amplitudes.append(amplitude)
                units[i, j] = sparse.coo_matrix((amplitudes, (rows, columns)),
                    shape=model.identity.shape, dtype=complex).tocsr()
                float_builder_error = max(float_builder_error,
                                          norm(units[i, j] - model.units[species][i, j]))
        result.append(units)
    require_small(float_builder_error, 'exact-root Pauli words versus actual owner units', 3e-8)
    return result, float_builder_error


def primitive_number_polynomial_controls(model, section):
    """Actual Pauli matrix units, not supplied Slater signs or CAR builders."""
    exact_units, builder_error = exact_word_matrix_units(model)
    maximum, cross, adjoints = 0., 0., 0.
    zero = sparse.csr_matrix(model.identity.shape, dtype=complex)
    for species in range(2):
        units = exact_units[species]
        for (i, j), left in units.items():
            adjoints = max(adjoints, norm(left.getH() - units[j, i]))
            for (k, ell), right in units.items():
                expected = (units[i, ell] if j == k else zero) - (units[k, j] if i == ell else zero)
                maximum = max(maximum, norm(left @ right - right @ left - expected))
    for left in exact_units[0].values():
        for right in exact_units[1].values():
            cross = max(cross, norm(left @ right - right @ left))
    if max(maximum, cross, adjoints) != 0.:
        raise AssertionError('Exact fourth-root primitive matrix-unit algebra failed')
    initial_occupation = []
    for species in range(2):
        local = sparse.csr_matrix(model.identity.shape, dtype=complex)
        for site, (pminus, _) in enumerate(section.local):
            block = np.zeros((8, 8), complex)
            block[4*site:4*site+4, 4*site:4*site+4] = np.eye(4) - pminus if species == 0 else pminus
            local = local + model.lift(block, species)
        initial_occupation.append(norm(local @ section.initial - 4 * section.initial))
    rng = np.random.default_rng(222690)
    vector = rng.normal(size=model.matter.dimension) + 1j * rng.normal(size=model.matter.dimension)
    vector /= norm(vector)
    polynomial = []
    for species, projector in enumerate(section.projectors):
        number = model.lift(projector, species)
        filled = model.filled(number, 4, vector)
        polynomial.append(dict(idempotence=norm(model.filled(number, 4, filled) - filled),
                               occupied_number=norm(number @ filled - 4 * filled)))
    require_small(max(maximum, cross, adjoints, *initial_occupation,
                      *(v for row in polynomial for v in row.values())),
                  'actual Pauli matrix-unit/initial/filled-number representation', 3e-8)
    return dict(all_8192_same_species_matrix_unit_commutators=maximum,
                all_4096_cross_species_commutators=cross, all128_unit_adjoints=adjoints,
                exact_fourth_root_word_arithmetic=True, exact_words_vs_float_owner_units=builder_error,
                initial_local_complementary_filled_occupation_residuals=initial_occupation,
                actual_degree_four_number_polynomial_controls=polynomial,
                supplied_Slater_signs=False)


def principal_angle_controls(model, fields, U, phases, bound):
    section = model.state(fields, U, phases)
    B = np.zeros((8, 8), complex)
    for site, (pminus, _) in enumerate(section.local):
        B[4*site:4*site+4, 4*site:4*site+4] = np.eye(4) - 2 * pminus
    observations = []
    for species, (values, vectors, occupied, _, _) in enumerate(section.spectra):
        sign = 1 if species == 0 else -1
        v = vectors[:, occupied]
        beta = v.conj().T @ (sign * B) @ v
        anticommutator = (section.onebody[species] @ B + B @ section.onebody[species]) / 2
        observations.append(dict(anticommutator_min=float(np.linalg.eigvalsh(anticommutator)[0]),
            compressed_beta_min=float(np.linalg.eigvalsh(beta)[0]),
            principal_cosine_squared_min=float(np.linalg.eigvalsh((np.eye(4) + beta) / 2)[0]),
            global_gap=float(values[4] - values[3])))
    if (section.weight + 1e-10 < bound['actual_Pauli_joint_projection_weight_lower'] or
        min(row['principal_cosine_squared_min'] for row in observations) + 1e-10 <
            bound['each_principal_cosine_squared_lower']):
        raise AssertionError('Actual primitive consumer contradicts its analytic overlap bound')
    return section, dict(observed_projection_weight=section.weight, species=observations,
                         finite_point_NOT_uniform_certificate=True)


def amplitude_bounds(bound, model, *, kappa=2., magnetic=.7):
    """Finite two-sided energy and invariant-amplitude consequences, not order."""
    Lambda = 4 * bound['strict_negative_for_Lambda_greater_than']
    energy_upper = bound['C_Lambda'] * Lambda + bound['T_sqrt_Lambda'] * np.sqrt(Lambda) + bound['B_constant']
    c = 8 * (max(abs(x) for x in model.onsite) + max(np.linalg.norm(p) for p in model.paths))
    d, alpha = (Lambda + 1) * c, Lambda * kappa / 8
    def young(a):
        return 3 * d**(4 / 3) / (4**(4 / 3) * a**(1 / 3))
    lower = (-energy_upper - magnetic) / d
    return dict(evaluation_Lambda=Lambda, evaluated_energy_upper=float(energy_upper),
        energy_lower=float(-2 * young(alpha) - magnetic),
        ground_or_minimizing_sequence_total_rho_mean_lower=float(max(0., lower)),
        asymptotic_total_rho_mean_lower=float(-bound['C_Lambda'] / c),
        ground_or_minimizing_sequence_sum_rho_fourth_moment_upper=float(
            max(0., (energy_upper + 2 * young(alpha / 2) + magnetic) / (alpha / 2))),
        rho_definition='rho_x=sqrt(sum_mu Tr Phi_xmu^2/4); invariant volume sum, not local order')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    for name, default in (('m', .2), ('r', .12), ('v', .3), ('stiffness', 2.),
                          ('Jpref', .2), ('ue4', .3), ('ue1', .4), ('magnetic', .7),
                          ('initial-field-radius', .2)):
        parser.add_argument('--' + name, type=float, default=default)
    parser.add_argument('--field-radius', type=float, default=None)
    parser.add_argument('--anchor-frame', choices=('environmental', 'pair-stationary'),
                        default='environmental')
    args = parser.parse_args()
    identities = source_hashes()
    band = RotationBand(a=1., m=args.m, r=args.r, v=args.v, improved=False)
    model = FullBandGaussSection(band, heavy_scale=1.)
    if args.anchor_frame == 'pair-stationary':
        frame, frame_proof = pair_clifford_reference(model, stiffness=args.stiffness)
    else:
        frame = uniform_data(band, 3, args.stiffness, 0., model.basis)['frame']
        frame_proof = dict(scope='environmental three-axis Clifford family; not a pair field minimum')
    fields: NDArray[np.complex128] = np.tile(frame, (2, 1, 1, 1)).astype(np.complex128)
    law = dict(kappa=args.stiffness, J=args.Jpref, ue4=args.ue4, ue1=args.ue1, magnetic=args.magnetic)
    bound, repair = select_radius(model, fields, initial_radius=args.initial_field_radius,
                                  radius=args.field_radius, **law)
    deformation = np.einsum('xma,aij->xmij', .55 * bound['radius'] / np.sqrt(120)
                           * np.sin(np.arange(120).reshape(2, 4, 15) * .43 + .17), model.basis)
    generator = np.einsum('a,aij->ij', 2.3 * np.sin(np.arange(15) * .37 + .29), model.generators)
    U, phases = expm(1j * generator), np.array([2.137, -1.081])
    deformed = fields + deformation
    section, angles = principal_angle_controls(model, deformed, U, phases, bound)
    controls = packet_point_controls(model, deformed, U, phases)
    representation = primitive_number_polynomial_controls(model, section)
    report = dict(status='normalized_all_Haar_all_phase_full_coordinate_Gauss_trial_NOT_ground_or_phase',
        source_sha256=identities, parameters={k: v for k, v in vars(args).items() if k != 'output'},
        law_owner='unchanged FullBandGaussSection.onebody; unchanged full_band_packet_bound quadratic form',
        primitive_dimension=model.matter.dimension, NH=4, NL=4, field_coordinates=120,
        anchor_frame=dict(name=args.anchor_frame, proof=frame_proof,
                          fields_real=fields.real.tolist(), fields_imag=fields.imag.tolist()),
        packet=dict(scalar_density='SU4^2 Haar average of normalized field-only compact 120-ball density',
            field_normalizer='pi^60 R^120 Gamma(5)/Gamma(65)',
            colour_scalar='constant 1 on genuine normalized SU4 Haar',
            EM_scalars='constant 1 on both normalized circles; full T^2',
            matter='actual FullBandGaussSection.state, including heavy whole-negative top4/light bottom4',
            normalization='exact Fubini and pointwise section norm; no quadrature surrogate'),
        field_support_repair=repair, quantitative_trial=bound,
        primitive_point_controls=controls, actual_number_polynomial_representation=representation,
        finite_principal_angle_control=angles,
        invariant_amplitude_bounds=amplitude_bounds(bound, model, kappa=args.stiffness, magnetic=args.magnetic),
        actual_point=dict(fields_real=deformed.real.tolist(), fields_imag=deformed.imag.tolist(),
                         link_real=np.real(U).tolist(), link_imag=np.imag(U).tolist(), phases=phases.tolist()),
        limits=['two cells/one colour link/two parallel EM paths, not a 3D theorem',
                'Cartesian kinetic replaces selected-orbit inertia; no second rotor kinetic',
                'uniform proof uses operator inequalities, not finite U/A extrema',
                'scalar colour/EM Fisher is zero; all section colour/EM/Berry costs remain',
                'negative energy and invariant average amplitude do not prove a full phase, charged lower gap or physical half-spin'])
    if source_hashes() != identities:
        raise RuntimeError('Scientific source changed during execution; refusing mixed-source receipt')
    payload = json.dumps(report, indent=2, allow_nan=False)
    args.output.write_text(payload + '\n')
    print(payload)


if __name__ == '__main__':
    main()
