#!/usr/bin/env python3
"""Same-law Cartesian field competition; finite candidates are not global minima.

Run only at the parent-owned scientific verification boundary.  The pair varies
all 120 real field coordinates.  The cube varies all 60 uniform-field coordinates
and retains the actual 32-band, top-sixteen whole-heavy occupation.
"""
from __future__ import annotations

import argparse
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
from numpy.typing import NDArray
from scipy.linalg import block_diag, expm, null_space
from scipy.optimize import minimize

from centre_flux_sea import onebody as cube_onebody, tree_edges, source_hashes as cube_source_hashes
from full_band_gauss_section import FullBandGaussSection
from full_band_packet_bound import anchor_probe, packet_bound, source_hashes as packet_source_hashes
from full_field_charge_bound import open_box, primitive_edges
from local_orbital_stability import scalar_paths, uniform_data
from pauli_compton import EdgeWords
from rotation_band import RotationBand
from sea_selection import SELECTED, matrix_unit_basis

BARE: dict[str, Any] = dict(a=1., m=.2, r=.12, v=.3, improved=False)
STIFFNESS, ALIGNMENT = 2., .2
SOURCE_NAMES = ('unrestricted_field_minimum.py', 'unrestricted_field_minimum.md')


def source_hashes():
    root = Path(__file__).resolve().parent
    return {**packet_source_hashes(), **cube_source_hashes(),
            **{name: hashlib.sha256((root / name).read_bytes()).hexdigest()
               for name in SOURCE_NAMES}}


def check(error, label, tolerance=3e-8):
    if not np.isfinite(error) or error > tolerance:
        raise AssertionError(f'{label}: {error}')


def coordinates(fields, basis):
    return np.einsum('...mij,aji->...ma', fields, basis).real / 4


def matrices(values, basis):
    return np.einsum('...ma,aij->...mij', values, basis)


def local_potential(fields, bonds, links):
    """The existing zero-preference quartic and J/8 full-field W, unchanged."""
    grams = np.einsum('xmij,xnji->xmn', fields, fields).real / 4
    elastic = STIFFNESS / 2 * float(np.sum(grams**2))
    alignment = ALIGNMENT / 8 * sum(float(np.linalg.norm(
        fields[x] - u @ fields[y] @ u.conj().T)**2)
        for (x, y), u in zip(bonds, links))
    return elastic, alignment


class Functional:
    """Linear source-owned h and exact projector derivatives on the actual domain."""

    def __init__(self, geometry, band, pair_owner, basis):
        self.geometry, self.band, self.pair_owner, self.basis = geometry, band, pair_owner, basis
        self.volume = 2 if geometry == 'pair' else 8
        self.shape = (2, 4, 15) if geometry == 'pair' else (4, 15)
        self.dimension = int(np.prod(self.shape))
        self.graph = open_box((2, 1, 1) if geometry == 'pair' else (2, 2, 2))
        self.links = tuple(np.eye(4, dtype=complex) for _ in self.graph.bonds)
        elementary = np.eye(self.dimension).reshape((self.dimension,) + self.shape)
        self.directions = np.array([self.owner_h(self.fields(direction)) for direction in elementary])
        self.elementary = elementary
        check(np.linalg.norm(self.directions - self.directions.conj().transpose(0, 2, 1)),
              'source-owned onebody derivative Hermiticity')

    def fields(self, value):
        result = matrices(np.asarray(value).reshape(self.shape), self.basis)
        return result if self.geometry == 'pair' else np.tile(result, (8, 1, 1, 1))

    def seed(self, frame):
        value = coordinates(frame, self.basis)
        return np.tile(value, (2, 1, 1)).reshape(-1) if self.geometry == 'pair' else value.reshape(-1)

    def owner_h(self, fields, links=None):
        links = self.links if links is None else links
        if self.geometry == 'pair':
            return self.pair_owner.onebody(fields, links[0], (0., 0.))[0]
        return cube_onebody(self.graph, self.band, fields, links)

    def spectral(self, value):
        h = np.einsum('a,aij->ij', np.asarray(value).reshape(-1), self.directions)
        eigenvalues, vectors = np.linalg.eigh(h)
        half = len(eigenvalues) // 2
        occupied = vectors[:, half:]
        return h, eigenvalues, vectors, occupied @ occupied.conj().T

    def elastic_gradient(self, value):
        c = np.asarray(value).reshape(self.shape)
        g = c @ c.swapaxes(-1, -2)
        factor = 1 if self.geometry == 'pair' else 8
        gradient = factor * 2 * STIFFNESS * g @ c
        elastic = factor * STIFFNESS / 2 * float(np.sum(g**2))
        aligned = 0.
        if self.geometry == 'pair':
            difference = c[0] - c[1]
            aligned = ALIGNMENT / 2 * float(np.sum(difference**2))
            gradient[0] += ALIGNMENT * difference
            gradient[1] -= ALIGNMENT * difference
        return elastic, aligned, gradient.reshape(-1)

    def evaluate(self, value):
        _, eigenvalues, _, p = self.spectral(value)
        elastic, alignment, gradient = self.elastic_gradient(value)
        sea = -float(eigenvalues[len(eigenvalues)//2:].sum())
        gradient -= np.einsum('ij,aji->a', p, self.directions).real
        return elastic + alignment + sea, gradient

    def hessian(self, value):
        _, eigenvalues, vectors, _ = self.spectral(value)
        half = len(eigenvalues) // 2
        gap = eigenvalues[half] - eigenvalues[half - 1]
        if gap <= 1e-10:
            raise ValueError('No separated top-half cluster for the Hessian')
        c = np.asarray(value).reshape(self.shape)
        d = self.elementary
        g = c @ c.swapaxes(-1, -2)
        dg = d @ c.swapaxes(-1, -2) + c @ d.swapaxes(-1, -2)
        factor = 1 if self.geometry == 'pair' else 8
        elastic = factor * 2 * STIFFNESS * (dg @ c + g @ d)
        if self.geometry == 'pair':
            delta = d[:, 0] - d[:, 1]
            elastic[:, 0] += ALIGNMENT * delta
            elastic[:, 1] -= ALIGNMENT * delta
        result = elastic.reshape(self.dimension, self.dimension).T
        transitions = np.einsum('ie,aij,jo->aeo', vectors[:, :half].conj(),
                                self.directions, vectors[:, half:])
        denominators = eigenvalues[half:][None, :] - eigenvalues[:half, None]
        features = (transitions / np.sqrt(denominators)[None]).reshape(self.dimension, -1)
        result -= 2 * (features.conj() @ features.T).real
        return (result + result.T) / 2

    def global_lower_bound(self):
        """Algebraic coercive Ky Fan bound, not equality or a global-minimum receipt."""
        k = np.einsum('aij,bji->ab', self.directions, self.directions).real
        # A row-sum bound avoids interpreting an approximate maximal eigenvalue
        # as a certified upper endpoint; the margin is a floating-point receipt.
        upper = float(np.max(np.sum(np.abs(k), axis=1)))
        upper += 1e-10 * max(1., upper)
        n = len(self.directions[0])
        quartic = (STIFFNESS / (8 * self.volume) if self.geometry == 'pair'
                   else self.volume * STIFFNESS / 8)
        linear = np.sqrt(n * upper / 4)
        radius = (linear / (4 * quartic))**(1 / 3)
        return dict(trace_quadratic_row_sum_upper=upper, quartic_rho_fourth_lower=quartic,
                    heavy_linear_norm_bound=float(linear), minimizing_norm=float(radius),
                    energy_lower=-3 * quartic * float(radius**4),
                    theorem='F >= a*rho^4-b*rho >= -3*a*(b/(4*a))^(4/3)',
                    numeric_status='floating evaluation with margin; algebraic bound is exact in source coefficients',
                    equality_not_claimed=True)


def algebra_report(functional, value) -> dict[str, Any]:
    c = np.asarray(value).reshape(functional.shape)
    all_fields = functional.fields(value)
    frame = all_fields[0] if functional.geometry == 'cube' else all_fields
    generators = functional.basis / np.sqrt(8)
    commutators = np.array([t @ frame - frame @ t for t in generators], dtype=np.complex128)
    real_columns = np.concatenate((commutators.real.reshape(15, -1),
                                   commutators.imag.reshape(15, -1)), axis=1).T
    singular = np.linalg.svd(real_columns, compute_uv=False)
    tolerance = 1e-7 * max(1., float(singular[0]))
    rank = int(np.sum(singular > tolerance))
    orbits = np.array([coordinates(1j * (t @ frame - frame @ t), functional.basis).reshape(-1)
                       for t in generators]).T
    normal = null_space(orbits.T, rcond=1e-7)
    hessian = functional.hessian(value)
    spectrum = np.linalg.eigvalsh(hessian)
    normal_values = np.linalg.eigvalsh(normal.T @ hessian @ normal)
    _, eigenvalues, _, _ = functional.spectral(value)
    half = len(eigenvalues) // 2
    energy, gradient = functional.evaluate(value)
    elastic, alignment, _ = functional.elastic_gradient(value)
    return dict(energy=energy, elastic=elastic, alignment=alignment,
                whole_heavy_top_half=-float(eigenvalues[half:].sum()),
                coordinate_values=c.tolist(), field_Gram=(c @ c.swapaxes(-1, -2)).tolist(),
                gradient_norm=float(np.linalg.norm(gradient)), gradient_max=float(np.max(abs(gradient))),
                occupied_empty_cluster_gap=float(eigenvalues[half]-eigenvalues[half-1]),
                onebody_eigenvalues=eigenvalues.tolist(),
                common_SU4_stabilizer_Lie_dimension=15-rank,
                common_complex_commutant_dimension=16-rank,
                stabilizer_singular_values=singular.tolist(), stabilizer_rank_tolerance=tolerance,
                stabilizer_scope=('all fields at both sites; connected identity link forces a common gauge'
                                  if functional.geometry == 'pair' else 'uniform fields and identity links'),
                orbit_dimension=rank, orbit_Hessian_action_norm=float(np.linalg.norm(hessian @ orbits)),
                Hessian_eigenvalues=spectrum.tolist(), normal_Hessian_eigenvalues=normal_values.tolist(),
                normal_Hessian_min=float(normal_values[0]),
                negative_Hessian_count=int(np.sum(spectrum < -2e-6)),
                flat_Hessian_count=int(np.sum(abs(spectrum) <= 2e-6)),
                Hessian_scope='actual full Cartesian functional at a separated top-half cluster')


def derivative_controls(functional, value):
    value = np.asarray(value).reshape(-1)
    _, gradient = functional.evaluate(value)
    hessian = functional.hessian(value)
    rows = []
    for phase in (.17, .83):
        direction = np.sin(np.arange(functional.dimension) * .71 + phase)
        direction /= np.linalg.norm(direction)
        eps = 2e-5
        ep, gp = functional.evaluate(value + eps * direction)
        em, gm = functional.evaluate(value - eps * direction)
        energy_error = abs((ep-em)/(2*eps) - gradient @ direction)
        hessian_error = float(np.linalg.norm((gp-gm)/(2*eps)-hessian @ direction))
        check(energy_error, 'Ky Fan analytic gradient versus central derivative', 2e-7)
        check(hessian_error, 'cluster Hessian versus projector-gradient derivative', 2e-6)
        rows.append(dict(step=eps, phase=phase, gradient_error=float(energy_error), Hessian_error=hessian_error))
    fields = functional.fields(value)
    check(np.linalg.norm(functional.owner_h(fields)-functional.spectral(value)[0]),
          'cached primitive coefficient tensor versus actual onebody owner')
    return rows


def refine(functional, initial, maxiter, tolerance):
    """L-BFGS estimates, then actual residual repair; no optimizer-success shortcut."""
    result = minimize(functional.evaluate, np.asarray(initial), jac=True, method='L-BFGS-B',
                      options=dict(ftol=1e-15, gtol=tolerance/10, maxiter=maxiter, maxls=50))
    value, repairs = repair_only(functional, result.x, tolerance)
    energy, gradient = functional.evaluate(value)
    escapes = []
    for round_number in range(3):
        eigenvalues, vectors = np.linalg.eigh(functional.hessian(value))
        if eigenvalues[0] >= -2e-6:
            break
        direction = vectors[:, 0]
        choices = [value + sign * .03 * direction for sign in (-1, 1)]
        seed = min(choices, key=lambda x: functional.evaluate(x)[0])
        trial = minimize(functional.evaluate, seed, jac=True, method='L-BFGS-B',
                         options=dict(ftol=1e-15, gtol=tolerance/100, maxiter=maxiter, maxls=50))
        if trial.fun >= energy - 1e-10:
            break
        escapes.append(dict(round=round_number, negative_curvature=float(eigenvalues[0]),
                            energy_before=float(energy), energy_after=float(trial.fun)))
        # Repair the actual residual again after every material descent.
        value, repaired = repair_only(functional, trial.x, tolerance)
        repairs.extend(repaired)
        energy, gradient = functional.evaluate(value)
    return value, dict(optimizer_message=str(result.message), optimizer_success=bool(result.success),
                       optimizer_iterations=int(result.nit), stationarity_repairs=repairs,
                       deterministic_negative_curvature_escapes=escapes,
                       acceptance='actual gradient norm and cluster gap, not optimizer status')


def repair_only(functional, initial, tolerance):
    value = np.asarray(initial).copy()
    repairs = []
    for iteration in range(50):
        energy, gradient = functional.evaluate(value)
        residual = float(np.linalg.norm(gradient))
        if residual <= tolerance:
            return value, repairs
        eigenvalues, vectors = np.linalg.eigh(functional.hessian(value))
        direction = -vectors @ ((vectors.T @ gradient) / np.maximum(abs(eigenvalues), 1e-6))
        length = np.linalg.norm(direction)
        if length > .1:
            direction *= .1 / length
        for reduction in range(35):
            step = 2.**(-reduction)
            candidate = value + step * direction
            next_energy, next_gradient = functional.evaluate(candidate)
            if (next_energy <= energy + 1e-4 * step * gradient @ direction or
                    (abs(next_energy-energy) <= 2e-14 * max(1., abs(energy))
                     and np.linalg.norm(next_gradient) < .9 * residual)):
                value = candidate
                repairs.append(dict(iteration=iteration, residual_before=residual,
                                    residual_after=float(np.linalg.norm(next_gradient)), step=step))
                break
        else:
            raise ValueError(f'Post-descent stationarity repair stalled: {residual}')
    raise ValueError('Post-descent actual stationarity repair exhausted its deterministic iterations')


def pair_clifford_reference(model, *, stiffness: float) -> tuple[NDArray[np.complex128], dict[str, Any]]:
    """Exact minimum only in the stated balanced two-field Clifford family."""
    if not np.isfinite(stiffness) or stiffness <= 0:
        raise ValueError('Reference requires positive finite quartic stiffness')
    c = float(model.onsite[0])
    p = model.paths[0]
    if np.max(abs(p[2:])) > 1e-12 or abs(p[0].imag) > 1e-12 or abs(p[1].real) > 1e-12:
        raise ValueError('Reference requires the actual unimproved x path')
    t, d = float(abs(p[1])), float(abs(p[0]))
    radius = ((c**4+t**4)/stiffness)**(1/3)
    b, s = np.sqrt(np.array([c*c, t*t])/(stiffness*radius))
    frame = np.zeros_like(SELECTED, dtype=np.complex128)
    frame[0], frame[1] = b*SELECTED[0], s*SELECTED[1]
    if radius <= d*b:
        raise ValueError('Clifford family formula left its actual positive cluster regime')
    return frame, dict(stiffness=float(stiffness), temporal_amplitude=float(b), x_spatial_amplitude=float(s),
                       radius=float(radius), Wilson_split=float(d*b),
                       exact_restricted_minimum=-3*float(radius),
                       expression='kappa*(b^4+s^4)-4*sqrt(c^2*b^2+t^2*s^2)',
                       scope='uniform balanced anticommuting Phi0,Phi1; Phi2=Phi3=0; R>d*b',
                       not_full_global_minimum=True)


def seed_set(functional, selected, clifford):
    rows = [('selected_environmental_frame', functional.seed(selected)),
            ('two_field_Clifford_reference', functional.seed(clifford))]
    mass = np.zeros_like(SELECTED)
    c = (functional.pair_owner.onsite[0] if functional.geometry == 'pair'
         else scalar_paths(functional.band)[0])
    mass[0] = (c/STIFFNESS)**(1/3)*SELECTED[0]
    rows.append(('onsite_only', functional.seed(mass)))
    diagonal = np.zeros((4, 15)); diagonal[:, 12:] = .19*np.cos(
        np.arange(12).reshape(4, 3)*.61+.23)
    frame = matrices(diagonal, functional.basis)
    rows.append(('commuting_traceless_seed', functional.seed(frame)))
    for number, phase in enumerate((.31, 1.17)):
        full = .07*np.sin(np.arange(functional.dimension)*.43+phase)
        rows.append((f'all_direction_deterministic_seed_{number}', .8*functional.seed(selected)+full))
    return rows


def covariance_controls(functional, value):
    fields = functional.fields(value)
    groups = [expm(1j*np.einsum('a,aij->ij', .17*np.sin(np.arange(15)*.47+.21+x),
                              functional.basis/np.sqrt(8))) for x in range(functional.volume)]
    transformed = np.array([g @ f @ g.conj().T for g, f in zip(groups, fields)])
    links = [groups[x] @ u @ groups[y].conj().T
             for (x, y), u in zip(functional.graph.bonds, functional.links)]
    unitary = np.asarray(block_diag(*groups), dtype=np.complex128)
    h = functional.owner_h(fields)
    changed_h = functional.owner_h(transformed, links)
    covariance = float(np.linalg.norm(changed_h-unitary @ h @ unitary.conj().T))
    before = local_potential(fields, functional.graph.bonds, functional.links)
    after = local_potential(transformed, functional.graph.bonds, links)
    scalar = float(np.linalg.norm(np.array(before)-after))
    check(covariance, 'actual independent-endpoint SU4 onebody covariance')
    check(scalar, 'same local scalar potential independent-endpoint covariance')
    return dict(onebody_covariance_residual=covariance, scalar_potential_covariance_residual=scalar,
                full_local_gauge_transform_exercised=True,
                transformed_cube_fields_not_artificially_restricted_to_uniform=True)


def primitive_pair_energy(model, h):
    """Actual S=+ Pauli words in the whole four-of-eight heavy sector (70 states)."""
    # Exact heavy endpoint mode order in FullBandGaussSection.__init__.
    modes = tuple(range(4)) + tuple(range(8, 12))
    masks = [sum(1 << modes[i] for i in occupied)
             for occupied in itertools.combinations(range(8), 4)]
    index = {mask: k for k, mask in enumerate(masks)}
    actual = np.zeros((70, 70), complex)
    words = {(i, j): model.code.words.path_A(
        model.code.words.path(modes[j], modes[i])).times_i(-1)
        for i in range(8) for j in range(8) if i != j}
    for column, mask in enumerate(masks):
        for i, j in zip(*np.nonzero(h)):
            if not (mask >> modes[j]) & 1:
                continue
            if i == j:
                actual[column, column] -= h[i, j]
            elif not (mask >> modes[i]) & 1:
                new = mask ^ (1 << modes[i]) ^ (1 << modes[j])
                actual[index[new], column] -= h[i, j]*model.code.element(mask, new, words[i, j])
    values, vectors = np.linalg.eigh(actual)
    predicted = -float(np.linalg.eigvalsh(h)[4:].sum())
    check(abs(values[0]-predicted), 'whole heavy Pauli ground versus top-four occupation')
    check(np.linalg.norm(actual-actual.conj().T), 'actual whole-heavy Pauli Hermiticity')
    residual = float(np.linalg.norm(actual @ vectors[:, 0]-predicted*vectors[:, 0]))
    check(residual, 'actual whole-heavy Pauli ground eigenvector')
    return dict(primitive_dimension=70, actual_energy=float(values[0]), top_four_prediction=predicted,
                eigen_residual=residual, chart_overlap_not_used=True,
                owner='actual PauliCode.element S=+ raw-word amplitudes; no CAR matrix builder')


def primitive_cube_coefficients(h):
    """Actual 33-vertex tree words: full one- and two-particle spectra, not N=16 ED."""
    n = len(h)
    graph = open_box((2, 2, 2))
    if n != 4 * graph.volume:
        raise ValueError('Actual 32-mode cube coefficient matrix required')
    internal = [(4*x, 4*x+j) for x in range(8) for j in (1, 2, 3)]
    roots = [(4*graph.bonds[e][0], 4*graph.bonds[e][1]) for e in tree_edges(graph)]
    words = EdgeWords(n+1, internal + roots + [(0, n)])
    transfers = {(i, j): words.path_A(words.path(j, i)).times_i(-1)
                 for i in range(n) for j in range(n) if i != j}
    spectra = []
    one = np.linalg.eigvalsh(h)
    for number in (1, 2):
        occupations = list(itertools.combinations(range(n), number))
        states = [words.tree_basis_state(occ+((n,) if number % 2 else ())) for occ in occupations]
        index = {state: k for k, state in enumerate(states)}
        actual = np.zeros((len(states), len(states)), complex)
        for column, (state, occupied) in enumerate(zip(states, occupations)):
            actual[column, column] = sum(h[j, j] for j in occupied)
            for j in occupied:
                for i in np.flatnonzero(h[:, j]):
                    if i in occupied:
                        continue
                    target, amplitude = transfers[int(i), j].action(state)
                    actual[index[target], column] += h[i, j]*amplitude
        expected = np.array(sorted(sum(one[list(occ)]) for occ in occupations))
        error = float(np.linalg.norm(np.linalg.eigvalsh(actual)-expected))
        check(error, 'actual cube Pauli exterior spectrum', 2e-7)
        check(np.linalg.norm(actual-actual.conj().T), 'actual cube Pauli exterior Hermiticity')
        spectra.append(dict(particle_number=number, dimension=len(states), spectrum_residual=error))
    return dict(owner='bosonic_statistics primitive_pauli_words via EdgeWords; real boundary parity marker',
                primitive_spectra=spectra, top_sixteen_whole_heavy=-float(one[16:].sum()),
                filling_scope='N=16 follows the faithful primitive exterior representation, not executed many-body ED',
                onebody_pairing_not_assumed=True)


def quantum_reference(model, fields, classical_energy):
    """Promote only a chart-certified uniform reference, never all found candidates."""
    probe = anchor_probe(model, fields, np.eye(4), (0., 0.), STIFFNESS, ALIGNMENT)
    check(abs(probe['leading_coefficient']-classical_energy), 'packet anchor same classical functional')
    rejected = []
    for radius in (.001, .0003, .0001, .00003, .00001):
        try:
            bound = packet_bound(model, fields, np.eye(4), (0., 0.),
                                 kappa=STIFFNESS, J=ALIGNMENT, R=radius, eta=radius, tau=radius)
        except ValueError as error:
            rejected.append(dict(radius=radius, reason=str(error)))
            continue
        if bound['C_Lambda'] >= 0:
            rejected.append(dict(radius=radius, reason='uniform leading upper not negative'))
            continue
        section = model.state(fields, np.eye(4), (0., 0.))
        primitive = -model.lift(section.onebody[0], 0)
        heavy = -float(np.linalg.eigvalsh(section.onebody[0])[4:].sum())
        residual = float(np.linalg.norm(primitive @ section.state-heavy*section.state))
        check(residual, 'actual quantum packet heavy eigenstate')
        physical = model.controls('improved_two_field_Clifford_packet', fields, np.eye(4), (0., 0.))
        physical_checks = {name: physical[name] for name in (
            'SU4_differential_Gauss_each_site_all15', 'EM_differential_Gauss_each_site',
            'finite_SU4_covariance_each_site', 'finite_EM_covariance_each_site',
            'whole_species_energies', 'primitive_current_vs_onebody_errors',
            'primitive_contact_vs_onebody_errors', 'source_current_Ward_each_site',
            'source_contact_Ward_each_site', 'frozen_wavefunction_Gauss_defect',
            'missing_heavy_Gauss_defect')}
        return dict(candidate='two-field Clifford reference only', bound=bound,
                    primitive_heavy_eigen_residual=residual, rejected_radii=rejected,
                    actual_physical_packet_checks=physical_checks,
                    physical_domain='SU4^2 and U1^2 equivariant full 120-coordinate packet; normalized Haar',
                    source_proof='full_band_packet_bound.md; packet_bound uniform gap/overlap and Fisher inequalities',
                    not_a_quantum_phase=True)
    raise ValueError(f'No certified quantum packet for improved Clifford reference: {rejected}')


def analyse_geometry(functional, selected, clifford, maxiter, tolerance):
    seed_reports, candidates, rejected = [], [], []
    for name, seed in seed_set(functional, selected, clifford):
        # Seeds with an accidental half-cluster degeneracy still have a lawful
        # Ky Fan energy; their initial Hessian is deliberately not asserted.
        energy, gradient = functional.evaluate(seed)
        _, values, _, _ = functional.spectral(seed)
        half = len(values)//2
        seed_reports.append(dict(name=name, energy=energy, gradient_norm=float(np.linalg.norm(gradient)),
                                 gradient_max=float(np.max(abs(gradient))),
                                 gap=float(values[half]-values[half-1])))
        try:
            refined, receipt = refine(functional, seed, maxiter, tolerance)
            report = algebra_report(functional, refined)
            report.update(name=name, refinement=receipt)
            candidates.append((refined, report))
        except ValueError as error:
            rejected.append(dict(name=name, reason=str(error)))
    if not candidates:
        raise ValueError(f'No actual stationary candidate for {functional.geometry}: {rejected}')
    best, best_report = min(candidates, key=lambda row: row[1]['energy'])
    checks = derivative_controls(functional, best)
    covariance = covariance_controls(functional, best)
    selected_value = functional.seed(selected)
    selected_derivatives = derivative_controls(functional, selected_value)
    # The one-axis consumer sees no Phi2/Phi3 hopping: deleting them is an exact
    # same-law competitor, not a redefinition of its domain or bare potential.
    selected_trim = selected.copy()
    selected_trim[2:] = 0
    selected_energy = functional.evaluate(selected_value)[0]
    trim_energy = functional.evaluate(functional.seed(selected_trim))[0]
    result: dict[str, Any] = dict(geometry=functional.geometry, coordinate_dimension=functional.dimension,
                  field_domain=('all independent traceless Hermitian fields at both sites' if functional.geometry=='pair'
                                else 'all traceless Hermitian uniform fields; not the full 480-coordinate cube'),
                  whole_heavy_occupation=4 if functional.geometry=='pair' else 16,
                  onebody_dimension=8 if functional.geometry=='pair' else 32,
                  same_local_scalar_potential=True, deterministic_seeds=seed_reports,
                  stationary_candidates=[report for _, report in candidates], rejected_refinements=rejected,
                  best_candidate=best_report, selected_to_best_improvement=selected_energy-best_report['energy'],
                  selected_gradient_checks=selected_derivatives, best_derivative_checks=checks,
                  best_covariance=covariance, global_coercive_lower_bound=functional.global_lower_bound(),
                  axis_deletion_competitor=dict(selected_energy=selected_energy, trimmed_energy=trim_energy,
                                               difference=selected_energy-trim_energy,
                                               deleted_axes_enter_onebody=functional.geometry=='cube',
                                               is_single_axis_obstruction_only=functional.geometry=='pair'),
                  scope='deterministic finite stationary candidates and Hessians; no global minimum from seed search')
    if functional.geometry=='pair':
        result['selected_gradient_by_site_and_orbital'] = functional.evaluate(selected_value)[1].reshape(2,4,15).tolist()
        result['selected_unused_axis_gradient_norm'] = float(np.linalg.norm(
            functional.evaluate(selected_value)[1].reshape(2,4,15)[:,2:]))
        result['primitive_best'] = primitive_pair_energy(functional.pair_owner, functional.owner_h(functional.fields(best)))
        result['primitive_selected'] = primitive_pair_energy(functional.pair_owner, functional.owner_h(functional.fields(selected_value)))
    else:
        result['primitive_best'] = primitive_cube_coefficients(functional.owner_h(functional.fields(best)))
    return result


def run(*, maxiter=2000, stationarity_tolerance=2e-8):
    identities = source_hashes()
    basis = matrix_unit_basis()
    hs = np.einsum('aij,bji->ab', basis, basis).real
    traces = np.trace(basis, axis1=1, axis2=2)
    check(np.linalg.norm(hs-4*np.eye(15)), 'actual Cartesian basis normalization')
    check(np.linalg.norm(traces), 'actual Cartesian basis trace zero')
    check(np.linalg.norm(basis-basis.conj().transpose(0,2,1)), 'actual Cartesian basis Hermiticity')
    band = RotationBand(**BARE)
    model = FullBandGaussSection(band, heavy_scale=1.)
    selected = np.asarray(uniform_data(band, 3, STIFFNESS, 0., basis)['frame'], complex)
    reference, reference_proof = pair_clifford_reference(model, stiffness=STIFFNESS)
    pair = Functional('pair', band, model, basis)
    cube = Functional('cube', band, model, basis)
    ref_value = pair.seed(reference)
    reference_receipt = algebra_report(pair, ref_value)
    check(abs(reference_receipt['energy']-reference_proof['exact_restricted_minimum']),
          'actual whole-heavy reference versus derived restricted minimum')
    check(reference_receipt['gradient_norm'], 'two-field reference actual full-coordinate stationarity')
    owner_probe = anchor_probe(model, pair.fields(ref_value), np.eye(4), (0.,0.), STIFFNESS, ALIGNMENT)
    check(abs(owner_probe['leading_coefficient']-reference_receipt['energy']), 'same scalar potential owner binding')
    pair_result = analyse_geometry(pair, selected, reference, maxiter, stationarity_tolerance)
    cube_result = analyse_geometry(cube, selected, reference, maxiter, stationarity_tolerance)
    quantum = quantum_reference(model, pair.fields(ref_value), reference_receipt['energy'])
    onsite, paths = scalar_paths(band)
    _, edges = primitive_edges(cube.graph, band)
    final_identities = source_hashes()
    if final_identities != identities:
        raise RuntimeError('A source changed during the consumer run; no mixed-revision science receipt')
    return dict(schema=1, source_sha256=final_identities, source_bindings_unchanged_during_run=True,
                governing_plan='issue-222-continuing-backward-20260930',
                assignment_plan_revision='d7cfa8de60401582b6b901673280c7baeb7ca13556cfcf4f4ee7b80016cb94f1',
                bare_coefficients={**BARE, 'stiffness':STIFFNESS, 'Jpref':ALIGNMENT},
                field_domain=dict(basis_owner='sea_selection.matrix_unit_basis', directions_per_matrix=15,
                                  matrices_per_site=4, coordinates_per_site=60, identity_direction_permitted=False,
                                  basis_trace_residual=float(np.linalg.norm(traces)),
                                  basis_HS_normalization_residual=float(np.linalg.norm(hs-4*np.eye(15))),
                                  constraints='traceless Hermitian only; no Gram budget or Clifford restriction'),
                scalar_potential='kappa/2 sum_x ||G_x||_F^2 + Jpref/8 sum_bond,mu ||Phi_xmu-U Phi_ymu Udag||_HS^2',
                primitive_coefficients=dict(pair_onsite=model.onsite[0], cube_onsite=onsite,
                                            pair_x_path_real=model.paths[0].real.tolist(),
                                            pair_x_path_imag=model.paths[0].imag.tolist(),
                                            cube_path_count=len(edges), cube_sites=8, cube_bonds=len(cube.graph.bonds),
                                            actual_directed_displacements=[[int(k) for k in d] for d,_ in paths]),
                restricted_pair_proof=reference_proof, restricted_pair_actual=reference_receipt,
                pair=pair_result, cube=cube_result, quantum_reference=quantum,
                distinctions=['Selected uniform_data is a periodic three-axis environmental trial, not the pair minimum.',
                              'The cube uses onsite m+3r and all twelve actual edges, separately from pair onsite m+r.',
                              'A nonzero common stabilizer cannot inherit exactly-four-centre Higgs or angular claims.',
                              'No full unrestricted classical global minimizer, quantum phase, or thermodynamic result is claimed.',
                              'No light, boundary, or photon fibre is removed by this classical heavy functional.',
                              'Numerical cube candidates are not promoted to quantum energy uppers.'],
                all_checks_pass=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--maxiter', type=int, default=2000)
    parser.add_argument('--stationarity-tolerance', type=float, default=2e-8)
    args = parser.parse_args()
    if args.maxiter < 1 or not 0 < args.stationarity_tolerance <= 1e-7:
        raise ValueError('Positive iterations and actual stationarity tolerance <=1e-7 required')
    result = run(maxiter=args.maxiter, stationarity_tolerance=args.stationarity_tolerance)
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(payload+'\n')
    print(payload)


if __name__ == '__main__':
    main()
