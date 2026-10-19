#!/usr/bin/env python3
"""Quantum half-line amplitude coupled to the actual Gauss/Pauli compact photon."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator

import compact_charge_gap as gap
import quantum_spinor_plaquette as spinor


@dataclass(frozen=True)
class RadialBasis:
    modes: int
    box: float
    inertia: float
    quartic: float

    def __post_init__(self):
        if self.modes < 2 or not all(math.isfinite(x) and x > 0 for x in
                                     (self.box, self.inertia, self.quartic)):
            raise ValueError('Require modes>=2 and finite positive box, inertia, quartic')

    def moment(self, power):
        """Exact sine integrals PR^pP, not powers of the compressed coordinate."""
        if power not in (1, 2, 4):
            raise ValueError('Supported exact moments are 1,2,4')
        n = np.arange(1, self.modes + 1)
        def cosine(k):
            result = np.full(k.shape, 1 / (power + 1), dtype=float)
            nonzero = k != 0
            a = np.pi * k[nonzero]
            parity = np.where(k[nonzero] % 2, -1., 1.)
            if power == 1:
                result[nonzero] = (parity - 1) / a**2
            elif power == 2:
                result[nonzero] = 2 * parity / a**2
            else:
                result[nonzero] = 4 * parity / a**2 - 24 * parity / a**4
            return result
        return self.box**power * (cosine(n[:, None] - n[None, :])
                                   - cosine(n[:, None] + n[None, :]))

    def kinetic(self):
        return np.diag((np.pi * np.arange(1, self.modes + 1) / self.box)**2
                       / (2 * self.inertia))

    def hamiltonian(self):
        return self.kinetic() + self.quartic * self.moment(4)

    def scalar_ground(self, coefficient):
        values, vectors = eigh(self.hamiltonian() - coefficient * self.moment(1),
                               subset_by_index=(0, 0))
        state = vectors[:, 0]
        return float(values[0]), state, float(state @ self.moment(1) @ state)

    def functions(self, coordinates):
        return np.sqrt(2 / self.box) * np.sin(np.pi * np.asarray(coordinates)[:, None]
                                            * np.arange(1, self.modes + 1) / self.box)


def radial_bounds(basis, parameters):
    c0 = 8 * parameters.mass
    charged = 7 * parameters.mass + parameters.hopping_bound
    f0, state, mean0 = basis.scalar_ground(c0)
    fc, _, meanc = basis.scalar_ground(charged)
    if not fc > f0 or not mean0 > 0 or not meanc > 0:
        raise AssertionError('Projected positive-coordinate monotonicity failed')
    return {'modes': basis.modes, 'box': basis.box, 'c_neutral': c0,
            'c_charged': charged, 'F_neutral_Ritz': f0, 'F_charged_Ritz': fc,
            'neutral_mean_R': mean0, 'charged_mean_R': meanc,
            'projected_model_charge_gap_bound': parameters.electric / 2 + fc - f0,
            'half_line_scope': 'Each F is a variational upper bound; their difference is NOT a certified half-line numerical gap.'}, state


class HermitianAction(LinearOperator):
    """A Hermitian tensor action with the public SciPy subclass interface."""
    def __init__(self, dimension, action):
        self.action = action
        super().__init__(dtype=np.dtype(complex), shape=(dimension, dimension))

    def _matvec(self, x):
        return self.action(x)

    def _rmatvec(self, x):
        return self.action(x)

    def _matmat(self, vectors):
        return np.column_stack([self.action(vectors[:, j]) for j in range(vectors.shape[1])])


class RadialSpinor:
    """Radial-major tensor ordering, sparse physical factor, no Kronecker CSR."""
    def __init__(self, physical, basis, parameters):
        self.physical, self.basis = physical, basis
        self.N = physical.N
        self.dimension = basis.modes * physical.dimension
        onsite, hopping, photon, self.decomposition_error = gap.operator_parts(physical)
        self.matter, self.photon = onsite + hopping, photon
        self.coordinate, self.radial = basis.moment(1), basis.hamiltonian()
        self.scalar = physical.UE / 2 * physical.boundary**2 + physical.delta * (1 - physical.sign) / 2
        # A certified triangle bound, including the largest possible electric energy.
        self.certified_norm_bound = (float(np.linalg.norm(self.radial, 2))
            + basis.box * (8 * parameters.mass + parameters.hopping_bound)
            + 2 * physical.UE * physical.cutoff**2 + physical.K + self.scalar)
        self.H = HermitianAction(self.dimension, self.action)
        self.B = HermitianAction(self.dimension, self.apply_B)

    def action(self, vector):
        x = np.asarray(vector).reshape(self.basis.modes, self.physical.dimension)
        matter_x = (self.matter @ x.T).T
        result = self.radial @ x + self.coordinate @ matter_x
        result += (self.photon @ x.T).T + self.scalar * x
        return result.reshape(-1)

    def apply_B(self, vector):
        x = np.asarray(vector).reshape(self.basis.modes, self.physical.dimension)
        return (self.physical.B @ x.T).T.reshape(-1)

    def apply_T(self, vector):
        x = np.asarray(vector).reshape(self.basis.modes, self.physical.dimension)
        result = np.empty_like(x)
        result[:, self.physical.Tperm] = x.conjugate() * self.physical.Tphase
        return result.reshape(-1)

    def controls(self):
        rng = np.random.default_rng(913 + self.N)
        x = rng.normal(size=self.dimension) + 1j * rng.normal(size=self.dimension)
        y = rng.normal(size=self.dimension) + 1j * rng.normal(size=self.dimension)
        x /= np.linalg.norm(x); y /= np.linalg.norm(y)
        hx = self.H @ x
        symmetry = np.linalg.norm(self.H @ self.apply_T(x) - self.apply_T(hx))
        square = np.linalg.norm(self.apply_T(self.apply_T(x)) - (-1)**self.N * x)
        hermitian = abs(np.vdot(y, hx) - np.vdot(self.H @ y, x))
        wrong_quartic = self.basis.moment(4) - np.linalg.matrix_power(self.coordinate, 4)
        quartic_error = float(np.linalg.norm(wrong_quartic, 2))
        # Consumer tensor order must agree with a separately formed product action.
        a = rng.normal(size=self.basis.modes); a /= np.linalg.norm(a)
        b = rng.normal(size=self.physical.dimension) + 1j * rng.normal(size=self.physical.dimension)
        b /= np.linalg.norm(b)
        expected = (np.outer(self.radial @ a, b)
                    + np.outer(self.coordinate @ a, self.matter @ b)
                    + np.outer(a, self.photon @ b + self.scalar * b)).reshape(-1)
        ordered = np.outer(a, b).reshape(-1)
        order_error = float(np.linalg.norm(self.H @ ordered - expected))
        wrong_order = np.outer(a, b).T.reshape(-1)
        wrong_order_error = float(np.linalg.norm(self.H @ wrong_order - expected))
        if max(symmetry, square, hermitian, order_error) > 2e-9 or min(quartic_error, wrong_order_error) < 1e-5:
            raise AssertionError('Radial tensor/antiunitary adverse control failed')
        return {'H_T_residual': float(symmetry), 'T_square_residual': float(square),
                'Hermiticity_bilinear_residual': float(hermitian),
                'tensor_product_action_residual': order_error,
                'wrong_tensor_order_residual': wrong_order_error,
                'wrong_compressed_fourth_power_norm': quartic_error,
                'radial_kinetic_coordinate_commutator_norm': float(np.linalg.norm(
                    self.basis.kinetic() @ self.coordinate - self.coordinate @ self.basis.kinetic(), 2)),
                'physical_Gauss_controls': self.physical.adverse_controls()}

    def code_sector_control(self, minus):
        """Tensor the actual Q_eta R_eta map, including noncommuting kinetic."""
        plus = self.physical
        physical_report = spinor.defect_intertwiner(plus, minus.physical)
        rows, phases = [], []
        for mask, fields in plus.raw:
            target = dict(plus.code.state(mask, -1))
            coefficient = sum(np.conjugate(target.get(q, 0))
                              * (-1)**(q >> plus.code.Q_edge & 1) * amplitude
                              for q, amplitude in plus.code.state(mask, 1))
            rows.append(minus.physical.index[mask, fields])
            phases.append(coefficient * (-1.0)**fields[3])
        transform = sparse.coo_matrix((phases, (rows, np.arange(plus.dimension))),
                                      shape=plus.H.shape).tocsr()
        matter_error = spinor.max_abs(transform.conjugate().T @ minus.matter @ transform - self.matter)
        photon_error = spinor.max_abs(transform.conjugate().T @ minus.photon @ transform
                                     - self.photon - plus.K * (plus.W + plus.W.conjugate().T))
        rng = np.random.default_rng(147 + self.N)
        x = rng.normal(size=(self.basis.modes, plus.dimension)) + 1j * rng.normal(
            size=(self.basis.modes, plus.dimension))
        x /= np.linalg.norm(x)
        transformed = (transform @ x.T).T.reshape(-1)
        lhs = (transform.conjugate().T @ (minus.H @ transformed).reshape(
            self.basis.modes, plus.dimension).T).T.reshape(-1)
        rhs = self.H @ x.reshape(-1) + plus.delta * x.reshape(-1)
        rhs += plus.K * ((plus.W + plus.W.conjugate().T) @ x.T).T.reshape(-1)
        tensor_error = float(np.linalg.norm(lhs - rhs))
        if max(matter_error, photon_error, tensor_error) > 2e-9:
            raise AssertionError('Quantum radial kinetic broke exact code-sector intertwining')
        return {**physical_report, 'matter_intertwiner_residual': matter_error,
                'photon_intertwiner_residual': photon_error,
                'full_radial_tensor_identity_residual': tensor_error,
                'kinetic_reason': 'I_radial tensor Q_eta R_eta is independent of radial coordinate.'}


def partial_traces(model, vector):
    physical, basis = model.physical, model.basis
    x = vector.reshape(basis.modes, physical.dimension)
    rho_radial = x @ x.conjugate().T
    masks = list(dict.fromkeys(m for m, _ in physical.raw))
    fields = list(dict.fromkeys(f for _, f in physical.raw))
    mi, fi = {m: i for i, m in enumerate(masks)}, {f: i for i, f in enumerate(fields)}
    rows = np.array([mi[m] for m, _ in physical.raw])
    cols = np.array([fi[f] for _, f in physical.raw])
    rho_photon = np.zeros((len(fields), len(fields)), dtype=complex)
    for a in range(basis.modes):
        matrix = sparse.coo_matrix((x[a], (rows, cols)), shape=(len(masks), len(fields))).tocsr()
        rho_photon += (matrix.conjugate().T @ matrix).toarray()
    purity_r = float(np.vdot(rho_radial, rho_radial).real)
    purity_ph = float(np.vdot(rho_photon, rho_photon).real)
    mean = float(np.trace(rho_radial @ basis.moment(1)).real)
    mean2 = float(np.trace(rho_radial @ basis.moment(2)).real)
    grid = np.linspace(0, basis.box, 129)
    functions = basis.functions(grid)
    density = np.einsum('ri,ij,rj->r', functions, rho_radial, functions).real
    # Exact represented interval probability via a polynomial-free sine quadrature.
    nodes, weights = np.polynomial.legendre.leggauss(max(64, 4 * basis.modes))
    def probability(low, high):
        points = low + (nodes + 1) * (high - low) / 2
        phi = basis.functions(points)
        return float((high - low) / 2 * np.dot(weights,
                     np.einsum('ri,ij,rj->r', phi, rho_radial, phi).real))
    probability_norm = probability(0, basis.box)
    shell = np.array([any(abs(e) == physical.cutoff for e in f) for _, f in physical.raw])
    if abs(probability_norm - 1) > 2e-8 or min(purity_r, purity_ph) <= 0:
        raise AssertionError('Actual radial/field partial trace normalization failed')
    return {'radial_purity': purity_r, 'radial_physical_linear_entropy': 1 - purity_r,
            'photon_purity_after_tracing_radial_and_matter': purity_ph,
            'photon_rest_linear_entropy': 1 - purity_ph,
            'radial_density_eigenvalues': np.linalg.eigvalsh(rho_radial).tolist(),
            'mean_R': mean, 'variance_R': mean2 - mean**2,
            'radial_basis_probabilities': np.real(np.diagonal(rho_radial)).tolist(),
            'radial_probability_normalization': probability_norm,
            'near_zero_probability': probability(0, .1 * basis.box),
            'outer_box_probability': probability(.8 * basis.box, basis.box),
            'electric_cutoff_shell_probability': float(np.sum(abs(x[:, shell])**2)),
            'loop_raise_loss': float(1 - np.linalg.norm((physical.W @ x.T).T)**2),
            'radial_density_samples': [{'R': float(r), 'density': float(d)} for r, d in zip(grid, density)],
            'trace_scope': 'Actual radial x Pauli-occupation x electric embedding; includes Gauss correlations, not a free Fock factorization.'}


def creator_residues(neutral, neutral_vector, charged, charged_ground):
    reports = []
    p0, pc = neutral.physical, charged.physical
    x = neutral_vector.reshape(neutral.basis.modes, p0.dimension)
    for orbital in range(4):
        image = np.zeros((neutral.basis.modes, pc.dimension), dtype=complex)
        source, target = (spinor.MARKER, orbital) if pc.N == 9 else (orbital, spinor.MARKER)
        word = p0.code.words.path_A(p0.code.words.path(source, target)).times_i(-1)
        for column, (mask, fields) in enumerate(p0.raw):
            if bool(mask >> orbital & 1) == (pc.N == 9):
                continue
            new = mask ^ (1 << orbital)
            row = pc.index.get((new, fields))
            if row is None:
                raise AssertionError('Actual neutral-marker creator escaped charged Gauss domain')
            image[:, row] += p0.code.element(mask, new, word) * x[:, column]
        flat = image.reshape(-1)
        norm = float(np.vdot(flat, flat).real)
        residue = float(np.linalg.norm(charged_ground.conjugate().T @ flat)**2)
        reports.append({'orbital': orbital, 'creator_norm_squared': norm,
                        'charged_ground_multiplet_residue': residue,
                        'normalized_ground_overlap': residue / norm if norm > 1e-14 else 0.})
    if sum(r['charged_ground_multiplet_residue'] for r in reports) <= 1e-9:
        raise AssertionError('Actual radial-preserving boundary creator has no charged ground residue')
    return reports


def nested_diagnostic(small, vector, energy, large, large_energy):
    if small.physical is not large.physical or small.basis.box != large.basis.box:
        raise ValueError('Nested diagnostic requires the same physical factor and radial box')
    x = np.zeros((large.basis.modes, large.physical.dimension), dtype=complex)
    x[:small.basis.modes] = vector.reshape(small.basis.modes, small.physical.dimension)
    image = x.reshape(-1)
    increase = large_energy - energy
    if increase > 2e-6:
        raise AssertionError('A nested sine-Galerkin space raised the coupled ground energy')
    return {'ground_energy_change': increase,
            'embedded_small_state_large_eigen_residual': float(np.linalg.norm(large.H @ image - energy * image)),
            'scope': 'Residual controls distance to the larger spectrum, not half-line ground error.'}


def run(args):
    parameters = gap.GapParameters(electric=args.electric, magnetic=args.magnetic,
                                  delta=args.delta, m=args.m0, r=args.r0, v=args.v0)
    parameters.validate()
    if parameters.mass <= parameters.hopping_bound:
        raise ValueError('Constructive sufficient theorem requires M0>8max(r0,v0)')
    if args.lanczos_steps < 2 or args.cutoff < 1:
        raise ValueError('Require Lanczos steps>=2 and electric cutoff>=1')
    modes = sorted(set(args.modes))
    boxes = sorted(set(args.boxes))
    bases = [RadialBasis(n, boxes[-1], args.inertia, args.quartic) for n in modes]
    code = spinor.PauliCode()
    def build_physical(number, sign):
        return spinor.SpinorPlaquette(code, number, args.cutoff, magnetic=parameters.magnetic,
                                     electric=parameters.electric, delta=parameters.delta,
                                     m=parameters.m, r=parameters.r, v=parameters.v, sign=sign)
    physical = {n: build_physical(n, 1) for n in (7, 8, 9)}
    physical_minus = {n: build_physical(n, -1) for n in (7, 8, 9)}
    previous, records, comparisons = {}, [], []
    for basis in bases:
        bound, radial_trial = radial_bounds(basis, parameters)
        models, solutions, by_n = {}, {}, {}
        for n in (8, 7, 9):
            model = RadialSpinor(physical[n], basis, parameters)
            models[n] = model
            controls = model.controls()
            controls['exact_quantum_code_sector'] = model.code_sector_control(
                RadialSpinor(physical_minus[n], basis, parameters))
            energies, ground, low = spinor.solve_low(model, norm_bound=model.certified_norm_bound)
            energy = float(energies[0])
            solutions[n] = (energy, ground)
            record = {'N': n, 'radial_modes': basis.modes, 'box': basis.box,
                      'electric_cutoff': args.cutoff, 'physical_dimension': physical[n].dimension,
                      'dimension': model.dimension, 'ground_energy': energy,
                      'certified_H_norm_bound': model.certified_norm_bound,
                      'operator_decomposition_residual': model.decomposition_error,
                      'controls': controls, 'low_spectrum': low,
                      'partial_traces': partial_traces(model, ground[:, 0]),
                      'B_response': spinor.response(model, energy, ground, args.lanczos_steps)}
            if n in previous:
                pm, pv, pe = previous[n]
                comparisons.append({'N': n, 'small_modes': pm.basis.modes, 'large_modes': basis.modes,
                                    **nested_diagnostic(pm, pv, pe, model, energy)})
            previous[n] = model, ground[:, 0], energy
            records.append(record); by_n[n] = record
        neutral_energy, neutral_ground = solutions[8]
        trial_physical, epsilon, _ = gap.neutral_trial(physical[8], parameters)
        trial = np.outer(radial_trial, trial_physical).reshape(-1)
        actual_trial_energy = float(np.vdot(trial, models[8].H @ trial).real)
        target_trial = epsilon + bound['F_neutral_Ritz']
        if abs(actual_trial_energy - target_trial) > 2e-8:
            raise AssertionError('Actual quantum neutral trial does not realize the rigorous upper bound')
        by_n[8]['actual_neutral_trial'] = {'energy': actual_trial_energy, 'target': target_trial,
                                         'residual': abs(actual_trial_energy - target_trial),
                                         'epsilon0': epsilon, 'radial_bounds': bound}
        for n in (7, 9):
            energy, ground = solutions[n]
            measured_gap = energy - neutral_energy
            margin = bound['projected_model_charge_gap_bound']
            charged_lower = epsilon + parameters.electric / 2 + bound['F_charged_Ritz']
            if measured_gap < margin - 2e-6 or energy < charged_lower - 2e-6:
                raise AssertionError('Actual coupled charged ground violates the projected rigorous bound')
            by_n[n]['physical_mass_energy_difference'] = measured_gap
            by_n[n]['projected_charge_gap_bound'] = margin
            by_n[n]['charged_absolute_lower_bound'] = charged_lower
            by_n[n]['creator_residues'] = creator_residues(models[8], neutral_ground[:, 0], models[n], ground)
    # Independent box change is an actual coupled neutral run, NOT a spectator.
    box_records = []
    for box in boxes[:-1]:
        basis = RadialBasis(modes[-1], box, args.inertia, args.quartic)
        model = RadialSpinor(physical[8], basis, parameters)
        energies, ground, low = spinor.solve_low(model, norm_bound=model.certified_norm_bound)
        box_records.append({'N': 8, 'box': box, 'modes': basis.modes,
                            'ground_energy': float(energies[0]), 'low_spectrum': low,
                            'partial_traces': partial_traces(model, ground[:, 0]),
                            'controls': model.controls(),
                            'comparison_to_largest_box_energy': float(energies[0]) - previous[8][2],
                            'scope': 'Different finite sine spaces are not nested; this is a box sensitivity diagnostic.'})
    scalar = []
    for box in boxes:
        for n in sorted(set(modes + [32, 64, 96])):
            row, _ = radial_bounds(RadialBasis(n, box, args.inertia, args.quartic), parameters)
            scalar.append(row)
    return {'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'parameters': vars(args) | {'output': str(args.output) if args.output else None},
            'model': 'Dirichlet half-line quantum radial kinetic + pure quartic + R Hmat + actual compact photon',
            'units': 'hbar=a=1; dimensionless R; inertia inverse-energy; quartic,m0,r0,v0,K,UE,Delta energy',
            'adiabatic_neutral_amplitude': (8 * parameters.mass / (4 * args.quartic))**(1/3),
            'adiabatic_scope': 'Derived minimum of quartic minus occupied-sea energy; never a preferred minimum of the bare bosonic potential.',
            'exact_code_margin': parameters.delta - 2 * parameters.magnetic,
            'half_line_theorem': 'UE/2+F(7M0+8max(r0,v0))-F(8M0)>0; exact for every positive K, cutoff, inertia and quartic with Delta>2K and M0>8max(r0,v0).',
            'cases': records, 'nested_radial_diagnostics': comparisons,
            'coupled_neutral_box_diagnostics': box_records, 'one_dimensional_Ritz_diagnostics': scalar,
            'scope': 'Only a quantum radial amplitude of conditionally selected Clifford shape; neither all 60 field coordinates nor the 3D/continuum theory are quantized here.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modes', nargs='+', type=int, default=[4, 8])
    parser.add_argument('--boxes', nargs='+', type=float, default=[3., 4.])
    parser.add_argument('--cutoff', type=int, default=1)
    parser.add_argument('--inertia', type=float, default=1.)
    parser.add_argument('--quartic', type=float, default=.5)
    parser.add_argument('--m0', type=float, default=.8)
    parser.add_argument('--r0', type=float, default=.04)
    parser.add_argument('--v0', type=float, default=.08)
    parser.add_argument('--magnetic', type=float, default=.25)
    parser.add_argument('--electric', type=float, default=.6)
    parser.add_argument('--delta', type=float, default=1.2)
    parser.add_argument('--lanczos-steps', type=int, default=24)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(args)
    payload = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(payload + '\n')
    print(payload)


if __name__ == '__main__':
    main()
