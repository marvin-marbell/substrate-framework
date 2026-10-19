#!/usr/bin/env python3
"""Source-owned periodic whole-heavy sea, all 60 uniform Cartesian fields.

Finite midpoint meshes and their local Hessians are not a quantum phase or a
thermodynamic/global minimum certificate. Run at the parent verification boundary.
"""
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
from scipy.linalg import expm, null_space, orth

from full_band_gauss_section import FullBandGaussSection
from local_orbital_stability import scalar_paths, uniform_data
from rotation_band import RotationBand
from sea_selection import SELECTED, coefficients, h_from_couplings, matrix_unit_basis
from unrestricted_field_minimum import (
    ALIGNMENT, BARE, STIFFNESS, check, coordinates, matrices,
    pair_clifford_reference, refine, source_hashes as inherited_source_hashes,
)

ROOT = Path(__file__).resolve().parent
PLAN_REVISION = '193fececfc2e36129e9a33ee7119d1b3ccaee663c13cfb45e34e9b3880e3885e'
CURVATURE_TOLERANCE = 2e-6
RANK_RCOND = 1e-7
GAP_TOLERANCE = 1e-10


def source_hashes():
    names = ('periodic_bulk_frame.py', 'periodic_bulk_frame.md',
             'local_orbital_stability.py', 'sea_selection.py', 'soft_budget.py', 'rotation_band.py')
    return {**inherited_source_hashes(), **{
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}}


def midpoint_grid(size, a):
    angles = -np.pi + (np.arange(size) + .5) * (2 * np.pi / size)
    return np.stack(np.meshgrid(angles, angles, angles, indexing='ij'), axis=-1).reshape(-1, 3) / a


class PeriodicFunctional:
    """Fixed TOTAL half filling; no absolute-value or per-momentum filling guess."""

    dimension, shape = 60, (4, 15)

    def __init__(self, band, size, basis):
        self.band, self.size, self.basis = band, size, basis
        self.momenta = midpoint_grid(size, band.a)
        self.x = coefficients(band, self.momenta)
        self.count = len(self.x)
        self.elementary = np.eye(60).reshape(60, 4, 15)

    def fields(self, value):
        return matrices(np.asarray(value).reshape(self.shape), self.basis)

    def seed(self, frame):
        return coordinates(frame, self.basis).reshape(-1)

    def spectral(self, value):
        frame = self.fields(value)
        h = np.einsum('pm,mij->pij', self.x, frame)
        eigenvalues, vectors = np.linalg.eigh(h)
        occupied = np.zeros(eigenvalues.shape, dtype=bool)
        # This fast path is justified by the sign separation, not spectral pairing.
        if np.all(eigenvalues[:, 1] < 0) and np.all(eigenvalues[:, 2] > 0):
            occupied[:, 2:] = True
        else:
            indices = np.argsort(eigenvalues.reshape(-1), kind='stable')[-2*self.count:]
            occupied.reshape(-1)[indices] = True
        gap = float(np.min(eigenvalues[occupied]) - np.max(eigenvalues[~occupied]))
        return eigenvalues, vectors, occupied, gap

    def elastic_gradient(self, value):
        c = np.asarray(value).reshape(self.shape)
        gram = c @ c.T
        return STIFFNESS / 2 * float(np.sum(gram**2)), (2 * STIFFNESS * gram @ c).reshape(-1)

    def evaluate(self, value):
        values, vectors, occupied, _ = self.spectral(value)
        projector = (vectors * occupied[:, None, :]) @ vectors.conj().transpose(0, 2, 1)
        trace = np.einsum('pij,aji->pa', projector, self.basis).real
        elastic, gradient = self.elastic_gradient(value)
        gradient -= (self.x.T @ trace / self.count).reshape(-1)
        sea = -float(values[occupied].sum()) / self.count
        return elastic + sea, gradient

    def hessian_parts(self, value):
        values, vectors, occupied, gap = self.spectral(value)
        if gap <= GAP_TOLERANCE:
            raise ValueError(f'No separated TOTAL occupied/empty cluster: {gap}')
        c = np.asarray(value).reshape(self.shape)
        gram = c @ c.T
        d = self.elementary
        dg = d @ c.T + c @ d.transpose(0, 2, 1)
        elastic = (2 * STIFFNESS * (dg @ c + gram @ d)).reshape(60, 60).T
        sea = np.zeros((60, 60))
        # Only empty-to-occupied matrix elements enter. Equal eigenvalues INSIDE
        # either cluster never appear in a denominator, even at Clifford degeneracy.
        for start in range(0, self.count, 256):
            stop = min(start + 256, self.count)
            u, e, occ = vectors[start:stop], values[start:stop], occupied[start:stop]
            transition = np.einsum('pie,aij,pjo->paeo', u.conj(), self.basis, u)
            mask = (~occ[:, :, None]) & occ[:, None, :]
            denominator = e[:, None, :] - e[:, :, None]
            weight = np.zeros_like(denominator)
            np.divide(1., denominator, out=weight, where=mask)
            feature = np.einsum('pm,paeo->mapeo', self.x[start:stop],
                                transition * np.sqrt(weight)[:, None]).reshape(60, -1)
            sea -= 2 * (feature.conj() @ feature.T).real / self.count
        return (elastic + elastic.T) / 2, (sea + sea.T) / 2

    def hessian(self, value):
        elastic, sea = self.hessian_parts(value)
        return elastic + sea


def occupation_report(functional, value) -> dict[str, Any]:
    values, _, occupied, gap = functional.spectral(value)
    positives = np.sum(values > 0, axis=1)
    negatives = np.sum(values < 0, axis=1)
    counts = occupied.sum(axis=1)
    return dict(whole_heavy_top_half=-float(values[occupied].sum()) / functional.count,
                total_occupied_modes=int(occupied.sum()), momentum_blocks=functional.count,
                occupation_counts_by_momentum_histogram={str(n): int(np.sum(counts == n)) for n in range(5)},
                two_positive_two_negative_each_momentum=bool(np.all(positives == 2) and np.all(negatives == 2)),
                top_two_each_momentum=bool(np.all(counts == 2)),
                occupied_empty_global_gap=gap,
                direct_top_two_gap_min=float(np.min(values[:, 2] - values[:, 1])),
                minimum_top_two_eigenvalue=float(np.min(values[:, 2])),
                maximum_bottom_two_eigenvalue=float(np.max(values[:, 1])),
                onebody_eigenvalue_min=float(values.min()), onebody_eigenvalue_max=float(values.max()),
                pairing_defect_max=float(np.max(abs(values + values[:, ::-1]))),
                occupation_rule='largest 2*N eigenvalues of all 4*N modes; positive heavy sign is minus h')


def ward_controls(functional, value, gradient, hessian, orbits):
    frame = functional.fields(value)
    generators = functional.basis / np.sqrt(8)
    gradient_fields = matrices(gradient.reshape(4, 15), functional.basis)
    gradient_orbits = np.array([coordinates(1j*(t @ gradient_fields-gradient_fields @ t),
                                              functional.basis).reshape(-1) for t in generators]).T
    acceleration = np.array([coordinates(-(t @ (t @ frame-frame @ t)
                                          -(t @ frame-frame @ t) @ t), functional.basis).reshape(-1)
                             for t in generators]).T
    first = gradient @ orbits
    differentiated = hessian @ orbits - gradient_orbits
    second = np.einsum('ai,ab,bi->i', orbits, hessian, orbits) + gradient @ acceleration
    angle = .23
    generator = np.einsum('a,aij->ij', .17*np.sin(np.arange(15)*.47+.21), generators)
    unitary = expm(1j * angle * generator)
    rotated = functional.seed(unitary @ frame @ unitary.conj().T)
    before, _ = functional.evaluate(value)
    after, rotated_gradient = functional.evaluate(rotated)
    expected_gradient = coordinates(unitary @ gradient_fields @ unitary.conj().T,
                                    functional.basis).reshape(-1)
    covariance = float(np.linalg.norm(rotated_gradient-expected_gradient))
    errors = dict(first_derivative_max=float(np.max(abs(first))),
                  differentiated_gradient_identity_norm=float(np.linalg.norm(differentiated)),
                  conjugation_curve_second_derivative_max=float(np.max(abs(second))),
                  finite_common_conjugation_energy_error=float(abs(after-before)),
                  finite_common_conjugation_gradient_covariance_error=covariance)
    for name, error in errors.items():
        check(error, f'common SU4 Ward {name}', 3e-7)
    return dict(**errors, controls_scope='all15 common generators; H*tau=L_T*gradient, not H*tau=0 off stationarity',
                no_local_gauge_or_quantum_Ward_claim=True)


def quadrature_report(functional, value):
    frame = functional.fields(value)
    _, paths = scalar_paths(functional.band)
    norms = np.array([np.linalg.norm(phi, 2) for phi in frame])
    lipschitz = np.zeros(3)
    for displacement, scalars in paths:
        lipschitz += 2 * abs(np.asarray(displacement)) * float(abs(scalars) @ norms)
    cover = float(np.pi / functional.size * lipschitz.sum())
    occupation = occupation_report(functional, value)
    positive_lower = occupation['minimum_top_two_eigenvalue'] - cover
    negative_upper = occupation['maximum_bottom_two_eigenvalue'] + cover
    continuum_sign_cover = positive_lower > 0 and negative_upper < 0
    return dict(angular_Hamiltonian_Lipschitz_per_axis=lipschitz.tolist(),
                Hamiltonian_operator_norm_cover_upper=cover,
                general_total_half_filling_energy_quadrature_error_upper=2*cover,
                separated_two_per_block_energy_quadrature_error_upper=cover,
                sharper_bound_applicable=bool(continuum_sign_cover),
                continuum_two_positive_two_negative_sufficient_cover=bool(continuum_sign_cover),
                continuum_positive_eigenvalue_lower=positive_lower,
                continuum_negative_eigenvalue_upper=negative_upper,
                bound_scope='exact path triangle/Weyl inequalities; numerical coefficients are floating evaluations, not interval-certified endpoints',
                gradient_Hessian_convergence_not_certified=True)


def algebra_report(functional, value, tolerance):
    c = np.asarray(value).reshape(4, 15)
    frame = functional.fields(value)
    generators = functional.basis / np.sqrt(8)
    orbits = np.array([coordinates(1j*(t @ frame-frame @ t), functional.basis).reshape(-1)
                       for t in generators]).T
    singular = np.linalg.svd(orbits, compute_uv=False)
    rank_tolerance = RANK_RCOND * max(1., float(singular[0]))
    rank = int(np.sum(singular > rank_tolerance))
    # Use the SAME absolute threshold for the orbit and its normal complement.
    rcond = rank_tolerance / float(singular[0]) if singular[0] else RANK_RCOND
    orbit_basis = orth(orbits, rcond=rcond)
    normal = null_space(orbits.T, rcond=rcond)
    lie_basis = null_space(orbits, rcond=rcond)
    complex_basis = np.concatenate((np.eye(4, dtype=complex)[None], functional.basis))
    commutators = np.array([t @ frame-frame @ t for t in complex_basis]).reshape(16, -1).T
    commutant_singular = np.linalg.svd(commutators, compute_uv=False)
    commutant_tolerance = RANK_RCOND * max(1., float(commutant_singular[0]))
    commutant_rcond = (commutant_tolerance / float(commutant_singular[0])
                      if commutant_singular[0] else RANK_RCOND)
    commutant = null_space(commutators, rcond=commutant_rcond)
    energy, gradient = functional.evaluate(value)
    hessian = functional.hessian(value)
    spectrum = np.linalg.eigvalsh(hessian)
    orbit_spectrum = np.linalg.eigvalsh(orbit_basis.T @ hessian @ orbit_basis)
    normal_spectrum = np.linalg.eigvalsh(normal.T @ hessian @ normal)
    occupation = occupation_report(functional, value)
    stationary = np.linalg.norm(gradient) <= tolerance
    normal_positive = bool(len(normal_spectrum) and normal_spectrum[0] > CURVATURE_TOLERANCE)
    negative = int(np.sum(spectrum < -CURVATURE_TOLERANCE))
    gapped = occupation['occupied_empty_global_gap'] > GAP_TOLERANCE
    local_minimum = bool(stationary and gapped and normal_positive and negative == 0)
    if not stationary:
        classification = 'not stationary on this finite mesh'
    elif negative:
        classification = 'finite-mesh stationary saddle'
    elif local_minimum:
        classification = 'finite numerical local minimum modulo common SU4 orbit'
    else:
        classification = 'finite stationary semidefinite quadratic evidence; extra flats unresolved'
    elastic, _ = functional.elastic_gradient(value)
    return dict(grid=functional.size, energy_per_cell=float(energy), elastic_per_cell=elastic,
                uniform_identity_link_alignment_per_cell=0., **occupation,
                coordinate_values=c.tolist(), field_Gram=(c @ c.T).tolist(),
                gradient_norm=float(np.linalg.norm(gradient)), gradient_max=float(np.max(abs(gradient))),
                actual_finite_mesh_stationary=bool(stationary), stationarity_tolerance=tolerance,
                Hessian_matrix=hessian.tolist(), Hessian_eigenvalues=spectrum.tolist(),
                Hessian_scope='all60 real traceless-Hermitian UNIFORM coordinates; not nonuniform modes',
                negative_Hessian_count=negative,
                flat_Hessian_count=int(np.sum(abs(spectrum) <= CURVATURE_TOLERANCE)),
                curvature_reporting_tolerance=CURVATURE_TOLERANCE,
                common_SU4_stabilizer_Lie_dimension=int(lie_basis.shape[1]),
                common_complex_commutant_dimension=int(commutant.shape[1]), orbit_dimension=rank,
                stabilizer_singular_values=singular.tolist(), stabilizer_rank_tolerance=rank_tolerance,
                stabilizer_nullspace_residual=float(np.linalg.norm(orbits @ lie_basis)),
                complex_commutant_singular_values=commutant_singular.tolist(),
                complex_commutant_rank_tolerance=commutant_tolerance,
                complex_commutant_nullspace_residual=float(np.linalg.norm(commutators @ commutant)),
                orbit_Hessian_eigenvalues=orbit_spectrum.tolist(),
                normal_Hessian_eigenvalues=normal_spectrum.tolist(),
                normal_Hessian_min=float(normal_spectrum[0]),
                orbit_Hessian_action_norm=float(np.linalg.norm(hessian @ orbit_basis)),
                orbit_normal_Hessian_cross_norm=float(np.linalg.norm(orbit_basis.T @ hessian @ normal)),
                Ward=ward_controls(functional, value, gradient, hessian, orbits),
                quadrature=quadrature_report(functional, value),
                finite_numerical_local_minimum_modulo_orbit=local_minimum,
                classification=classification, no_physical_inertia_or_pseudoinverse_mass=True)


def primitive_controls(band, basis, frame):
    momentum = np.array([.19, -.31, .47]) / band.a
    onsite, paths = scalar_paths(band)
    def fourier(fields, sign=1):
        result = onsite * fields[0].astype(complex)
        for displacement, scalars in paths:
            term = np.einsum('m,mij->ij', scalars, fields)
            term *= np.exp(sign*1j*band.a*np.dot(momentum, displacement))
            result += term + term.conj().T
        return result
    primitive = band.h_from_couplings(momentum)
    selected = np.einsum('m,mij->ij', coefficients(band, momentum[None])[0], SELECTED)
    errors = dict(primitive_Fourier_vs_scalar_paths=float(np.linalg.norm(primitive-fourier(SELECTED))),
                  primitive_Fourier_vs_coefficients=float(np.linalg.norm(primitive-selected)),
                  changed_frame_Fourier_vs_coefficients=float(np.linalg.norm(fourier(frame)-np.einsum(
                      'm,mij->ij', coefficients(band, momentum[None])[0], frame))),
                  changed_frame_vs_existing_reframe_owner=float(np.linalg.norm(
                      fourier(frame)-h_from_couplings(band, momentum[None], frame)[0])),
                  basis_HS_normalization=float(np.linalg.norm(np.einsum('aij,bji->ab', basis, basis)-4*np.eye(15))),
                  basis_trace=float(np.linalg.norm(np.trace(basis, axis1=1, axis2=2))),
                  basis_Hermiticity=float(np.linalg.norm(basis-basis.conj().transpose(0, 2, 1))))
    for name, error in errors.items():
        check(error, name, 1e-11)
    wrong = float(np.linalg.norm(primitive-fourier(SELECTED, sign=-1)))
    if wrong <= .05:
        raise AssertionError('Nontrivial-momentum Fourier control failed to reject a wrong phase sign')
    return dict(momentum=momentum.tolist(), **errors, wrong_Fourier_phase_sign_error=wrong,
                wrong_Fourier_phase_sign_rejected=True,
                coefficient_owner='sea_selection.coefficients(RotationBand); paths from local_orbital_stability.scalar_paths')


def derivative_controls(functional, value):
    # Off-stationarity all-coordinate probe keeps a sea-gradient sign error visible.
    value = np.asarray(value) + .017*np.sin(np.arange(60)*.43+.31)
    energy, gradient = functional.evaluate(value)
    elastic_hessian, sea_hessian = functional.hessian_parts(value)
    hessian = elastic_hessian + sea_hessian
    _, elastic_gradient = functional.elastic_gradient(value)
    radial = value / np.linalg.norm(value)
    column = int(np.argmax(np.linalg.norm(sea_hessian, axis=0)))
    directions = [('radial', radial), ('sea_curvature_column', np.eye(60)[column])]
    for phase in (.17, .83):
        direction = np.sin(np.arange(60)*.71+phase)
        directions.append((f'full60_phase_{phase}', direction/np.linalg.norm(direction)))
    rows = []
    wrong_gradient_error, wrong_hessian_error = 0., 0.
    for name, direction in directions:
        for eps in (4e-5, 2e-5):
            ep, gp = functional.evaluate(value+eps*direction)
            em, gm = functional.evaluate(value-eps*direction)
            fd = (ep-em)/(2*eps)
            fd_hessian = (gp-gm)/(2*eps)
            gradient_error = float(abs(fd-gradient @ direction))
            hessian_error = float(np.linalg.norm(fd_hessian-hessian @ direction))
            check(gradient_error, 'all60 analytic gradient central difference', 3e-7)
            check(hessian_error, 'all60 cluster Hessian projector derivative', 3e-6)
            rows.append(dict(direction=name, step=eps, gradient_error=gradient_error, Hessian_error=hessian_error))
            if name == 'radial':
                wrong_gradient_error = float(abs(fd-(2*elastic_gradient-gradient) @ direction))
            if name == 'sea_curvature_column':
                wrong_hessian_error = float(np.linalg.norm(fd_hessian-(elastic_hessian-sea_hessian) @ direction))
    if min(wrong_gradient_error, wrong_hessian_error) <= 1e-4:
        raise AssertionError('Derivative controls failed to reject the plausible wrong heavy-sea sign')
    return dict(probe_energy_per_cell=energy, probe_coordinate_values=value.reshape(4, 15).tolist(),
                selected_finite_differences=rows,
                wrong_sea_gradient_sign_error=wrong_gradient_error,
                wrong_sea_Hessian_sign_error=wrong_hessian_error, wrong_sea_sign_rejected=True,
                off_stationary_Ward=ward_controls(functional, value, gradient, hessian,
                    np.array([coordinates(1j*(t @ functional.fields(value)-functional.fields(value) @ t),
                                          functional.basis).reshape(-1)
                              for t in functional.basis/np.sqrt(8)]).T))


def receipt_seed(path, identities):
    payload = path.read_bytes()
    receipt = json.loads(payload)
    expected = {**BARE, 'stiffness': STIFFNESS, 'Jpref': ALIGNMENT}
    if receipt.get('bare_coefficients') != expected:
        raise ValueError('Open-cube receipt does not identify the same bare coefficients')
    if receipt.get('source_sha256', {}).get('sea_selection.py') != identities['sea_selection.py']:
        raise ValueError('Open-cube coordinates have a different matrix_unit_basis owner revision')
    value = np.asarray(receipt['cube']['best_candidate']['coordinate_values'], dtype=float)
    if value.shape != (4, 15) or not np.all(np.isfinite(value)):
        raise ValueError('Actual open-cube receipt must supply finite all60 uniform coordinates')
    return value.reshape(-1), dict(path=str(path.resolve()), sha256=hashlib.sha256(payload).hexdigest(),
                                  coordinate_pointer='/cube/best_candidate/coordinate_values',
                                  borrowed_results=False, role='initial coordinates only; all bulk properties recomputed')


def run(*, grid=8, validation_grid=16, maxiter=2000, stationarity_tolerance=2e-8,
        cube_receipt=ROOT/'unrestricted_field_minimum_receipt.json'):
    identities = source_hashes()
    basis = matrix_unit_basis()
    band = RotationBand(**BARE)
    coarse = PeriodicFunctional(band, grid, basis)
    fine = PeriodicFunctional(band, validation_grid, basis)
    environmental = uniform_data(band, 3, STIFFNESS, 0., basis)['frame']
    model = FullBandGaussSection(band, heavy_scale=1.)
    pair, pair_proof = pair_clifford_reference(model, stiffness=STIFFNESS)
    cube, input_binding = receipt_seed(cube_receipt, identities)
    seeds = [('environmental_Clifford_L3', coarse.seed(environmental)),
             ('balanced_pair_two_field_reference', coarse.seed(pair)),
             ('actual_open_cube_receipt_coordinates', cube)]
    primitive = primitive_controls(band, basis, coarse.fields(cube))
    derivatives = derivative_controls(coarse, coarse.seed(environmental))
    branches, rejected = [], []
    for name, initial in seeds:
        branch: dict[str, Any] = dict(name=name,
            initial_on_coarse=algebra_report(coarse, initial, stationarity_tolerance),
            initial_on_validation=algebra_report(fine, initial, stationarity_tolerance))
        try:
            coarse_value, optimization = refine(coarse, initial, maxiter, stationarity_tolerance)
            branch['coarse_stationary_candidate'] = algebra_report(coarse, coarse_value, stationarity_tolerance)
            branch['coarse_refinement'] = optimization
            branch['coarse_candidate_frozen_on_validation'] = algebra_report(fine, coarse_value, stationarity_tolerance)
            fine_value, fine_optimization = refine(fine, coarse_value, maxiter, stationarity_tolerance)
            branch['validation_stationary_candidate'] = algebra_report(fine, fine_value, stationarity_tolerance)
            branch['validation_refinement'] = fine_optimization
            branch['validation_derivative_controls'] = derivative_controls(fine, fine_value)
        except ValueError as error:
            rejected.append(dict(name=name, reason=str(error)))
        branches.append(branch)
    completed = [row for row in branches if 'validation_stationary_candidate' in row]
    if not completed:
        raise ValueError(f'No separated actual stationary candidate survived both meshes: {rejected}')
    for branch in completed:
        for key in ('coarse_stationary_candidate', 'validation_stationary_candidate'):
            candidate = branch[key]
            if not candidate['actual_finite_mesh_stationary']:
                raise AssertionError(f'{branch["name"]}: unearned stationarity on {candidate["grid"]}')
            if not candidate['two_positive_two_negative_each_momentum']:
                raise AssertionError(f'{branch["name"]}: final candidate left the two-positive/two-negative regime')
    ranking = lambda key: [dict(name=row['name'], energy_per_cell=row[key]['energy_per_cell'],
                               gradient_norm=row[key]['gradient_norm'], classification=row[key]['classification'])
                          for row in sorted(completed, key=lambda row: row[key]['energy_per_cell'])]
    final_identities = source_hashes()
    if final_identities != identities:
        raise RuntimeError('Source changed during bulk run; refusing mixed-revision receipt')
    if hashlib.sha256(cube_receipt.read_bytes()).hexdigest() != input_binding['sha256']:
        raise RuntimeError('Input open-cube receipt bytes changed during bulk run')
    return dict(schema=1, source_sha256=final_identities, input_receipt_binding=input_binding,
                source_bindings_unchanged_during_run=True, input_receipt_unchanged_during_run=True,
                governing_plan='issue-222-continuing-backward-20260930', assignment_plan_revision=PLAN_REVISION,
                bare_coefficients={**BARE, 'stiffness': STIFFNESS, 'g0': 0., 'Jpref': ALIGNMENT},
                field_domain='all60 uniform real traceless-Hermitian coordinates; identity links; no Gram/Clifford restriction',
                scalar_potential_per_cell='kappa/2 ||C C^T||_F^2; g0=0; uniform Jpref alignment is exactly zero',
                occupation='minus largest 2*N of all4*N eigenvalues, not minus Tr|h|/2 and not forced block filling',
                mesh=dict(coarse=grid, validation=validation_grid,
                          rule='equal-weight midpoint BZ cube; seam phase (-1)^(n+1), even antiperiodic/odd periodic'),
                optimization=dict(maxiter=maxiter, actual_stationarity_tolerance=stationarity_tolerance,
                                  Hessian_pseudoinverse_not_a_physical_mass=True),
                pair_seed_proof=pair_proof, pair_seed_proof_scope='pair family only; not a bulk energy or stationarity claim',
                primitive_positive_control=primitive, coarse_derivative_controls=derivatives,
                same_law_seed_branches=branches, rejected_refinements=rejected,
                coarse_ranking=ranking('coarse_stationary_candidate'),
                frozen_validation_ranking=ranking('coarse_candidate_frozen_on_validation'),
                refined_validation_ranking=ranking('validation_stationary_candidate'),
                independent_initial_competitors_attempted=len(seeds),
                all_three_branches_completed=len(completed) == len(seeds),
                all_checks_pass=True,
                all_checks_pass_scope='executed controls and accepted candidates; rejected branches remain explicit',
                distinctions=['Finite-mesh stationary residuals are recomputed separately on each grid.',
                              'Only positive normal Hessian and stationarity earn a finite numerical local minimum modulo orbit.',
                              'No nonuniform spatial Hessian, unrestricted global minimum, interval certification, or quantum phase is supplied.',
                              'Lie/commutant changes are recomputed; prior centre, spin, scalar-inertia or photon receipts do not transfer.',
                              'Lipschitz finite quadrature controls energy, not numerical gradient/Hessian or phase convergence.',
                              'Global common SU4 Ward controls do not prove local gauge/EM Ward identities.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--grid', type=int, default=8)
    parser.add_argument('--validation-grid', type=int, default=16)
    parser.add_argument('--maxiter', type=int, default=2000)
    parser.add_argument('--stationarity-tolerance', type=float, default=2e-8)
    parser.add_argument('--cube-receipt', type=Path, default=ROOT/'unrestricted_field_minimum_receipt.json')
    args = parser.parse_args()
    if args.grid < 3 or args.validation_grid <= args.grid or args.maxiter < 1:
        raise ValueError('Need grid>=3, a strictly finer validation grid, and positive iterations')
    if not 0 < args.stationarity_tolerance <= 1e-7:
        raise ValueError('Actual stationarity tolerance must be positive and <=1e-7')
    result = run(grid=args.grid, validation_grid=args.validation_grid, maxiter=args.maxiter,
                 stationarity_tolerance=args.stationarity_tolerance, cube_receipt=args.cube_receipt)
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(payload+'\n')
    print(payload)


if __name__ == '__main__':
    main()
