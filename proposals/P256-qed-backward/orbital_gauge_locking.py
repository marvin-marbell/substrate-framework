#!/usr/bin/env python3
"""Changed local SU(4) adjoint-Higgs law; finite Pauli covariance, not a gauge ground simulation."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from scipy.linalg import block_diag, expm, null_space

from code_phase import CoupledSquare
from local_orbital_stability import (fourier_sea_hessian, scalar_paths,
                                     uniform_data)
from quantum_spinor_plaquette import PauliCode
from rotation_band import RotationBand
from sea_selection import matrix_unit_basis
from spin_orbital_locking import solve_selected_generators


def norm(matrix):
    return float(np.linalg.norm(matrix))


def complex_json(matrix):
    matrix = np.asarray(matrix)
    return dict(real=matrix.real.tolist(), imaginary=matrix.imag.tolist())


def adjoint(frame, unitary):
    return unitary @ frame @ unitary.conj().T


def gauge_mass(frame, generators, alignment, coupling):
    """Ta normalization Tr(Ta Tb)=delta_ab/2; no mass eigenvalue repair."""
    changes = np.array([t @ frame - frame @ t for t in generators])
    flattened = changes.reshape(15, -1)
    mass = alignment * coupling**2 / 4 * (flattened.conj() @ flattened.T)
    return mass, changes


def analytic_masses(g, alignment, coupling):
    """Clifford blades: even degree anticommutes inside; odd degree outside."""
    rows = []
    for degree in range(1, 5):
        for subset in itertools.combinations(range(4), degree):
            anticommutes = subset if degree % 2 == 0 else tuple(i for i in range(4) if i not in subset)
            rows.append(dict(degree=degree, subset=list(subset),
                             anticommutes_with=list(anticommutes),
                             mass_squared=float(alignment * coupling**2 / 2 * sum(g[i] for i in anticommutes))))
    return rows


def gradient_energy(left, right, link, alignment):
    delta = left - adjoint(right, link)
    return float(alignment / 8 * np.vdot(delta, delta).real)


def gauge_dressed_block(left, right, link, u1, scalars):
    # target x, source y: each summand is independently Gx-covariant/Gy-contravariant.
    return u1 * np.einsum('m,mij->ij', scalars,
                          (left @ link + link @ right) / 2)


class TwoCellPauli:
    """Complete fixed-N sector of eight active modes in the actual plaquette code.

    The other eight matter modes are empty. No SU4 Gauss projection is asserted.
    Matrix entries come from raw two-word PauliCode.element, not wedge signs.
    """
    def __init__(self, number=4):
        self.code = PauliCode()
        self.masks = [sum(1 << i for i in occupied)
                      for occupied in itertools.combinations(range(8), number)]
        self.index = {mask: i for i, mask in enumerate(self.masks)}
        self.number = number

    def lift(self, matrix):
        result = np.zeros((len(self.masks), len(self.masks)), complex)
        for column, mask in enumerate(self.masks):
            result[column, column] = sum(matrix[i, i] for i in range(8) if mask >> i & 1)
            for source in range(8):
                if not mask >> source & 1:
                    continue
                for target in range(8):
                    if mask >> target & 1 or abs(matrix[target, source]) < 1e-15:
                        continue
                    roots = (4 * (source // 4), 4 * (target // 4))
                    path = tuple(dict.fromkeys((source, *roots, target)))
                    word = self.code.words.path_A(path).times_i(-1)
                    if not word.commutes(self.code.S):
                        raise AssertionError('Dressed primitive transfer escaped the code')
                    new = mask ^ (1 << source) ^ (1 << target)
                    result[self.index[new], column] += matrix[target, source] * self.code.element(mask, new, word)
        return result


def two_cell_hamiltonian(fields, link, u1, onsite, scalars):
    hopping = gauge_dressed_block(fields[0], fields[1], link, u1, scalars)
    result = np.asarray(block_diag(onsite * fields[0, 0], onsite * fields[1, 0]))
    result[:4, 4:] = hopping
    result[4:, :4] = hopping.conj().T
    return result


def pauli_covariance(frame, basis, band) -> dict[str, Any]:
    # Independent non-spin local frames and a genuinely noncentral SU4 link.
    k0 = np.einsum('a,aij->ij', .21 * np.sin(np.arange(15) + .2), basis)
    k1 = np.einsum('a,aij->ij', .19 * np.cos(np.arange(15) + .7), basis)
    generators = np.asarray(block_diag(k0, k1))
    local_frames = [expm(1j * k0), expm(1j * k1)]
    link = expm(1j * np.einsum('a,aij->ij', .17 * np.sin(np.arange(15) + 1.1), basis))
    u1 = np.exp(.37j)
    # Arbitrary Hermitian shapes, not merely conjugated Clifford orbit points.
    fields = np.array([frame + .07 * basis[:4], frame + .05 * basis[4:8]])
    transformed = np.array([adjoint(fields[x], local_frames[x]) for x in range(2)])
    changed_link = local_frames[0] @ link @ local_frames[1].conj().T
    onsite, paths = scalar_paths(band)
    nearest = [(d, s) for d, s in paths if tuple(d) == (1, 0, 0)]
    if len(nearest) != 1:
        raise AssertionError('Expected one actual positive-x nearest path')
    scalars = nearest[0][1]
    h = two_cell_hamiltonian(fields, link, u1, onsite, scalars)
    hp = two_cell_hamiltonian(transformed, changed_link, u1, onsite, scalars)
    absent = two_cell_hamiltonian(transformed, link, u1, onsite, scalars)
    code = TwoCellPauli(4)
    H, Hp = code.lift(h), code.lift(hp)
    K = code.lift(generators)
    F = expm(1j * K)
    expected = F @ H @ F.conj().T
    one_frame = np.asarray(block_diag(*local_frames))
    lie_test = 1j * (generators @ h - h @ generators)
    # This checks the algebra enabling exponentiation on actual Pauli matrices.
    lie_residual = norm(1j * (K @ H - H @ K) - code.lift(lie_test))
    # A local centre change reverses this link; the Wilson-dressed observable cancels it.
    centre_fields = fields.copy()
    centre_frame = np.asarray(block_diag(-np.eye(4), np.eye(4)))
    centre_link = -link
    centre_h = two_cell_hamiltonian(centre_fields, centre_link, u1, onsite, scalars)
    probe_left = np.array([1., .2j, -.3, .4j]); probe_right = np.array([.3j, .7, .2, -.1j])
    bare_bilinear = complex(np.vdot(probe_left, link @ probe_right))
    dressed_centre = complex(np.vdot(-probe_left, centre_link @ probe_right))
    return dict(number=4, active_modes=8, dimension=len(code.masks),
                primitive_owner='quantum_spinor_plaquette.PauliCode.element on actual raw S=+ two-word states',
                hilbert_scope='Full fixed N=4 eight-active-mode code fiber; external U1/SU4 link values, NOT SU4-Gauss ground space',
                local_generators=complex_json(generators), su4_link=complex_json(link),
                u1_link=complex_json(np.array(u1)), primitive_hamiltonian=complex_json(H),
                transformed_primitive_hamiltonian=complex_json(Hp),
                local_frame_on_code=complex_json(F),
                one_body_covariance_residual=norm(hp - one_frame @ h @ one_frame.conj().T),
                actual_pauli_lie_algebra_residual=lie_residual,
                actual_pauli_covariance_residual=norm(Hp - expected),
                absent_link_transform_defect=norm(code.lift(absent) - expected),
                gradient_covariance_residual=abs(gradient_energy(fields[0], fields[1], link, 1.) -
                                                 gradient_energy(transformed[0], transformed[1], changed_link, 1.)),
                pure_local_centre_covariance_residual=norm(centre_h - centre_frame @ h @ centre_frame.conj().T),
                wilson_dressed_centre_cancellation=abs(bare_bilinear - dressed_centre),
                hermiticity_residual=norm(H-H.conj().T))


def wedge_two(unitary):
    """Background/reference representation only; never the Pauli matter builder."""
    pairs = list(itertools.combinations(range(4), 2))
    return np.array([[np.linalg.det(unitary[np.ix_(target, source)])
                      for source in pairs] for target in pairs])


def reference_representation(frame, basis) -> dict[str, Any]:
    spin, solved = solve_selected_generators(frame, basis)
    negative = np.linalg.eigh(frame[0])[1][:, :2]
    pairs = list(itertools.combinations(range(4), 2))
    selected = np.array([np.linalg.det(negative[list(pair), :]) for pair in pairs])
    spin6 = []
    for t in spin:
        # Analytic exterior-square differential, including the diagonal entries.
        matrix = np.zeros((6, 6), complex)
        for column, (i, j) in enumerate(pairs):
            for row, (k, l) in enumerate(pairs):
                matrix[row, column] = (t[k, i] * (l == j) + (k == i) * t[l, j]
                                       - t[k, j] * (l == i) - (k == j) * t[l, i])
        spin6.append(matrix)
    centre = wedge_two(1j * np.eye(4))
    return dict(representation='conjugate antisymmetric 6 reference; centre charge -2=2 mod4',
                selected_rest_pair=complex_json(selected), dimension=6,
                centre_i_action=complex_json(centre),
                centre_charge_two_residual=norm(centre + np.eye(6)),
                selected_spin_singlet_residual=max(norm(t @ selected) for t in spin6),
                derived_spin_casimir_residual=norm(sum(t @ t for t in spin) - .75 * np.eye(4)),
                derived_combined_lift_2pi_residual=norm(expm(-2j * np.pi * spin[2]) + np.eye(4)),
                actual_stabilizer_solve=solved,
                warning='An EM scalar background alone has no SU4 centre charge. Fixing a colour vector without transforming its reference Hilbert space violates local gauge covariance.')


def noncommuting_sector_consumer() -> dict[str, Any]:
    """Q_eta R_eta identity with independent bounded operator coefficients."""
    model = CoupledSquare(cutoff=1)
    H0, S, C, matter = model.operators()
    onsite = np.diag([sum(model.onsite[v] for v in model.words.occupation(q))
                      for q, _ in model.raw])
    free = H0 - matter + onsite
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.diag([1., -1.])
    auxiliary = .31 * Z + .27 * X
    H = np.kron(free, np.eye(2)) + np.kron(np.eye(model.dimension), auxiliary)
    coefficients = []
    for i, transfer in enumerate(model.transfers):
        coefficient = transfer.coefficient * (np.eye(2) + .2 * np.cos(i + .3) * X
                                               + .17 * np.sin(i + .7) * Z + .11j * Y)
        coefficients.append(coefficient)
        D = model.transfer_matrix(transfer.path)
        term = np.kron(D, coefficient)
        H += term + term.conj().T
    Q = model.pauli_matrix(model.Q)
    R = np.diag([(-1.)**fields[0] for _, fields in model.raw])
    T = Q @ R
    plus, _ = model.sector_basis(1)
    V = np.kron(plus, np.eye(2)); mapped = np.kron(T, np.eye(2)) @ V
    difference = mapped.conj().T @ H @ mapped - V.conj().T @ H @ V
    expected = np.kron(2 * model.K * plus.conj().T @ C @ plus, np.eye(2)) + model.delta * np.eye(V.shape[1])
    path_residual = max(norm(T.conj().T @ model.transfer_matrix(t.path) @ T - model.transfer_matrix(t.path))
                        for t in model.transfers)
    return dict(actual_raw_gauss_dimension=model.dimension,
                auxiliary_dimension=2, independent_path_count=len(coefficients),
                coefficient_pair_noncommutation=max(norm(a @ b-b @ a) for a in coefficients for b in coefficients),
                coefficient_kinetic_noncommutation=max(norm(a @ auxiliary-auxiliary @ a) for a in coefficients),
                matched_path_QR_residual=path_residual,
                full_noncommuting_sector_identity_residual=norm(difference-expected),
                predicted_sector_margin=model.delta - 2 * model.K,
                measured_comparison_minimum=float(np.linalg.eigvalsh(difference)[0]),
                scope='Independent bounded operator coefficient consumer of the existing U1 Gauss square, not an SU4 rotor or Higgs vacuum simulation')


def run(stiffness=1., preferred=1., alignment=1., coupling=.7):
    if min(stiffness, preferred, alignment, coupling) <= 0:
        raise ValueError('Require positive kappa, preferred Gram, J and SU4 coupling')
    basis = matrix_unit_basis()
    generators = basis / (2 * np.sqrt(2))
    band = RotationBand(a=1., m=.2, e=1., improved=True)
    data = uniform_data(band, 3, stiffness, preferred, basis)
    frame = data['frame']; g = np.diag(data['selected']['G'])
    mass, changes = gauge_mass(frame, generators, alignment, coupling)
    analytic = analytic_masses(g, alignment, coupling)
    eigenvalues = np.linalg.eigvalsh(mass)
    predicted = np.sort([row['mass_squared'] for row in analytic])
    q = expm(1j * np.einsum('a,aij->ij', .23 * np.cos(np.arange(15) + .9), basis))
    conjugated = adjoint(frame, q)
    mass_q, _ = gauge_mass(conjugated, generators, alignment, coupling)
    adjoint_coordinates = np.einsum('aij,bji->ab', generators, adjoint(generators, q)).real * 2
    hessian = fourier_sea_hessian(data, np.zeros(3, int), 3, basis) + data['elastic']
    orbit = data['orbit']; complement = null_space(orbit.T)
    restoring = np.linalg.eigvalsh(complement.T @ hessian @ complement)
    point = np.sin(np.arange(15) + .4); point /= np.linalg.norm(point)
    vector = np.einsum('a,aij->ij', point, generators)
    epsilon = 1e-4
    link = expm(1j * coupling * epsilon * vector)
    quadratic_curvature = 2 * gradient_energy(frame, frame, link, alignment) / epsilon**2
    covariance = pauli_covariance(conjugated, basis, band)
    reference = reference_representation(conjugated, basis)
    sector = noncommuting_sector_consumer()
    checks = dict(soft_selected_stationarity=data['selected']['gradient_residual'] < 1e-10,
                  positive_all_fifteen=float(eigenvalues[0]) > 1e-10,
                  analytic_clifford_multiplicities=norm(eigenvalues-predicted) < 1e-10,
                  conjugated_mass_spectrum=norm(np.linalg.eigvalsh(mass_q)-predicted) < 1e-10,
                  mass_adjoint_covariance=norm(mass_q-adjoint_coordinates @ mass @ adjoint_coordinates.T) < 1e-10,
                  fifteen_eaten_fortyfive_shapes=orbit.shape[1] == 15 and complement.shape[1] == 45,
                  actual_uniform_shapes_restoring=float(restoring[0]) > 1e-10,
                  uniform_orbit_is_flat=norm(hessian @ orbit) < 1e-9,
                  exact_gradient_quadratic=abs(quadratic_curvature-float(point @ mass.real @ point)) < 1e-7,
                  actual_primitive_covariance=covariance['actual_pauli_covariance_residual'] < 1e-9,
                  actual_primitive_lie_algebra=covariance['actual_pauli_lie_algebra_residual'] < 1e-9,
                  absent_link_control_detects_failure=covariance['absent_link_transform_defect'] > 1e-3,
                  gradient_covariance=covariance['gradient_covariance_residual'] < 1e-10,
                  centre_is_not_measurable_spin=covariance['wilson_dressed_centre_cancellation'] < 1e-12,
                  reference_has_required_centre_charge=reference['centre_charge_two_residual'] < 1e-12,
                  selected_reference_is_spin_singlet=reference['selected_spin_singlet_residual'] < 1e-9,
                  noncommuting_coefficients_exercised=sector['coefficient_pair_noncommutation'] > 1e-3 and sector['coefficient_kinetic_noncommutation'] > 1e-3,
                  bounded_coefficients_preserve_code=sector['full_noncommuting_sector_identity_residual'] < 1e-9)
    return dict(status='changed_local_SU4_adjoint_Higgs_quadratic_route_not_quantum_phase_proof',
                microscopic_law=dict(orbital_fields='Four arbitrary traceless Hermitian adjoints per cell; fundamental scalar flavour matter',
                                     potential='kappa/2 ||G(A)-preferred I||_F^2, G_mn=Tr(A_m A_n)/4',
                                     su4_links='Local L_xy in SU4, Hilbert L2(SU4) per independent link',
                                     electric='UE4/2 sum_links,a (E_left^a)^2 (electric Casimir)',
                                     magnetic='K4 sum_plaquettes (1-ReTr ordered L_plaquette/4)',
                                     gradient='J/8 sum_nearest,mu Tr[(A_mu(x)-L_xy A_mu(y) L_xy^dag)^2]',
                                     hopping='u1_xy sum_mu s_mu(path) [A_mu(x)L_path+L_path A_mu(y)]/2; same primitive Pauli transfer and adjoint',
                                     su4_gauss='left/right electric generators - matter adjoint charge - orbital adjoint charge - reference charge =0, with oriented signs fixed by local group action',
                                     u1_gauss='existing integral-rotor divergence and n-2 electromagnetic charge'),
                parameters=dict(kappa=stiffness, preferred=preferred, J=alignment, gSU4=coupling, selected_cube=3),
                selected_gram=data['selected']['G'].tolist(), selected_fields=complex_json(frame),
                conjugation=complex_json(q), conjugated_fields=complex_json(conjugated),
                generator_trace_normalization=norm(np.einsum('aij,bji->ab', generators, generators)-np.eye(15)/2),
                selected_gauge_mass_matrix=complex_json(mass), conjugated_gauge_mass_matrix=complex_json(mass_q),
                selected_mass_eigenvalues=eigenvalues.tolist(), analytic_blade_masses=analytic,
                analytic_cubic_multiplicities=[dict(sum_gram=s, multiplicity=m) for s, m in
                    [('3gv', 1), ('gs+2gv', 3), ('gs+gv', 3), ('2gv', 3), ('gs', 1), ('gv', 3), ('gs+3gv', 1)]],
                common_commutant_traceless_rank=int(np.linalg.matrix_rank(changes.reshape(15, -1))),
                gauge_mass_conjugation_residual=norm(mass_q-adjoint_coordinates @ mass @ adjoint_coordinates.T),
                gradient_curvature=quadratic_curvature, predicted_gradient_curvature=float(point @ mass.real @ point),
                uniform_shape_eigenvalues=restoring.tolist(), uniform_orbit_flat_residual=norm(hessian @ orbit),
                pauli_consumer=covariance, centre_reference=reference, code_selector=sector,
                limitations=['This adds an SU4 gauge law; orbital_hydrodynamics remains a global-SU4 model with physical orientation modes.',
                             'Positive quadratic vector masses do not prove a quantum Higgs phase, deconfined Z4, three-dimensional order or stable dressed particles.',
                             'Uniform 45-shape positivity is for the recorded finite selected sea, not the full quantum SU4-Gauss vacuum.',
                             'No full SU4 Gauss quantum ground state has been built or diagonalized; links in the Pauli consumer are external values.',
                             'The arbitrary compact-U1-link Wilson ±m exclusion does not apply to general SU4 links: they need not commute with the selected grading. Pure frame-transformed vacuum links are gauge equivalent; general SU4 fluctuations need separate spectral bounds.',
                             'The combined Spin3 lift is a selected-frame algebraic construction; exact lattice rotations are cubic only.',
                             'A pure local -I gauge transformation cancels in Wilson dressing. Operational spin requires a boundary/reference rotation action and a physical experiment, not this algebraic minus sign.'],
                checks=checks, all_checks_pass=all(checks.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stiffness', type=float, default=1.)
    parser.add_argument('--preferred', type=float, default=1.)
    parser.add_argument('--alignment', type=float, default=1.)
    parser.add_argument('--g-su4', type=float, default=.7)
    args = parser.parse_args()
    result = run(args.stiffness, args.preferred, args.alignment, args.g_su4)
    names = ('orbital_gauge_locking.py', 'soft_budget.py', 'sea_selection.py',
             'local_orbital_stability.py', 'spin_orbital_locking.py',
             'quantum_spinor_plaquette.py', 'code_phase.py', 'rotation_band.py')
    result['source_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + '\n')
    print(payload)
    return 0 if result['all_checks_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
