#!/usr/bin/env python3
"""Unrestricted 60-coordinate Gauss trial and boundary-average charge upper bound.

Analytic full-Hilbert-space theorem constants; primitive-code and scalar-path
controls are finite exact algebra probes, not a full-field spectrum simulation.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy import sparse
from scipy.linalg import expm

from gauss_charge_intertwiner import rectangular_ladder
from local_orbital_stability import scalar_paths
from quantum_spinor_plaquette import PauliCode
from quantum_su4_gauss import norm, require_small, sparse_sum
from rotation_band import RotationBand
from sea_selection import matrix_unit_basis

INTERTWINER_SOURCE = '861008cb7dea64d2b2e0cef513c62de1d55539a1527a5b6fea8b5b879ea956fa'
GAUSS_SOURCE = '9e91c25bc4bcbc5f2306b38d4716b4e3af4664fc22757aedda527c056aea096b'


@dataclass(frozen=True)
class PathEdge:
    """One directed primitive path PLUS its adjoint, not two separately counted edges."""
    x: int
    y: int
    coefficients: tuple[complex, complex, complex, complex]
    displacement: tuple[int, int, int]

    @property
    def coefficient_norm(self):
        return float(np.linalg.norm(self.coefficients))


@dataclass(frozen=True)
class OpenBox:
    shape: tuple[int, int, int]
    sites: tuple[tuple[int, int, int], ...]
    bonds: tuple[tuple[int, int], ...]
    plaquettes: int
    boundary: tuple[int, ...]

    @property
    def volume(self):
        return len(self.sites)


def open_box(shape):
    """Open primitive cubic box; every exposed vertex is an actual boundary port.

    A fixed-thickness slab has positive port density as lateral volume grows.
    The port carries a prescribed external colour representation and EM flux;
    it is not an added matter spectator or an internal electric link.
    """
    shape = tuple(int(n) for n in shape)
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError('An open box has three positive integral side lengths')
    sites = tuple((x, y, z) for x, y, z in itertools.product(*(range(n) for n in shape)))
    index = {site: x for x, site in enumerate(sites)}
    bonds = []
    for x, site in enumerate(sites):
        for axis in range(3):
            target = list(site); target[axis] += 1
            if target[axis] < shape[axis]:
                bonds.append((x, index[target[0], target[1], target[2]]))
    plaquettes = sum((shape[i] - 1) * (shape[j] - 1) * shape[k]
                     for i, j, k in ((0, 1, 2), (0, 2, 1), (1, 2, 0)))
    boundary = tuple(x for x, site in enumerate(sites)
                     if any(site[i] in (0, shape[i] - 1) for i in range(3)))
    return OpenBox(shape, sites, tuple(bonds), plaquettes, boundary)


def primitive_edges(graph, band):
    """Read actual scalar_paths; length-two paths use both primitive links.

    Retain RotationBand's onsite scalar unchanged on the open boundary: this
    theorem does not silently replace it by a degree-dependent mass term.
    """
    onsite, paths = scalar_paths(band)
    index = {site: x for x, site in enumerate(graph.sites)}
    edges = []
    for x, site in enumerate(graph.sites):
        for displacement, coefficients in paths:
            target = tuple(int(site[i] + displacement[i]) for i in range(3))
            if target in index:
                edges.append(PathEdge(x, index[target],
                                      (complex(coefficients[0]), complex(coefficients[1]),
                                       complex(coefficients[2]), complex(coefficients[3])),
                                      (int(displacement[0]), int(displacement[1]), int(displacement[2]))))
    return float(onsite), tuple(edges)


def incidence_coefficients(volume, onsite, edges):
    """Safe whole-H bounds: |h| <= sum_x c_x rho_x (quadratic forms)."""
    coefficients = np.full(volume, 8 * abs(onsite), dtype=float)
    for edge in edges:
        if not (0 <= edge.x < volume and 0 <= edge.y < volume) or edge.x == edge.y:
            raise ValueError('Primitive path endpoints must be distinct graph vertices')
        coefficients[edge.x] += 8 * edge.coefficient_norm
        coefficients[edge.y] += 8 * edge.coefficient_norm
    return coefficients


def gaussian_constants(sigma):
    """sigma^2 is the DENSITY variance of each independent canonical real u."""
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Gaussian density width must be finite and positive')
    return dict(coordinates_per_site=60, density_variance=sigma**2,
                rho_squared=60 * sigma**2, gram_squared=1200 * sigma**4,
                canonical_Pi_squared=15 / sigma**2,
                kinetic_sqrt_Lambda_coefficient=7.5 / sigma**2,
                alignment_HS_squared_per_bond=480 * sigma**2,
                quartic_Lambda_kappa_coefficient=600 * sigma**4,
                alignment_Lambda_J_per_bond=60 * sigma**2)


def theorem_constants(graph, light_band, heavy_band, *, kappa=1., J=0., K4=0.,
                      KEM=0., sigma=.3, boundary_electric=1.):
    """Construct C0, full local-sum moment bound, and physical boundary average.

    Lambda>=1. KEM bounds each compact EM magnetic term below by -KEM;
    simple primitive plaquettes have exactly zero neutral E=0 trial mean.
    J,K4,KEM and boundary_electric are nonnegative; no neutral dominance assumed.
    """
    if (not all(np.isfinite(v) for v in (kappa, J, K4, KEM, boundary_electric))
            or kappa <= 0 or min(J, K4, KEM, boundary_electric) < 0):
        raise ValueError('kappa>0 and finite nonnegative gauge/alignment coefficients required')
    V = graph.volume
    if V < 1 or not graph.boundary:
        raise ValueError('A nonempty graph and declared physical boundary ports are required')
    g = gaussian_constants(sigma)
    onsite_L, edges_L = primitive_edges(graph, light_band)
    onsite_H, edges_H = primitive_edges(graph, heavy_band)
    c_L_sites = incidence_coefficients(V, onsite_L, edges_L)
    c_H_sites = incidence_coefficients(V, onsite_H, edges_H)
    cL, cH = float(np.max(c_L_sites)), float(np.max(c_H_sites))
    bond_density, plaquette_density = len(graph.bonds) / V, graph.plaquettes / V
    C0 = (g['kinetic_sqrt_Lambda_coefficient'] + kappa * g['quartic_Lambda_kappa_coefficient']
          + J * g['alignment_Lambda_J_per_bond'] * bond_density + 2 * K4 * plaquette_density)
    young = .75 * (4 / kappa)**(1 / 3) * (cH + cL)**(4 / 3)
    R4 = 16 / kappa * (C0 + young + KEM * plaquette_density)
    # Cubic bounds are independent of every side length: Nb/V,Np/V <= 3;
    # each declared positive-displacement scalar path has at most two endpoints
    # incident at a vertex, including genuine length-two improved paths.
    _, paths_L = scalar_paths(light_band)
    _, paths_H = scalar_paths(heavy_band)
    uniform_cL = 8 * (abs(onsite_L) + 2 * sum(float(np.linalg.norm(f)) for _, f in paths_L))
    uniform_cH = 8 * (abs(onsite_H) + 2 * sum(float(np.linalg.norm(f)) for _, f in paths_H))
    uniform_C0 = (g['kinetic_sqrt_Lambda_coefficient'] + kappa * g['quartic_Lambda_kappa_coefficient']
                  + 3 * J * g['alignment_Lambda_J_per_bond'] + 6 * K4)
    uniform_D = .75 * (4 / kappa)**(1 / 3) * (uniform_cH + uniform_cL)**(4 / 3)
    uniform_R4 = 16 / kappa * (uniform_C0 + uniform_D + 3 * KEM)
    # Exact row/column commutator counts, including a path once at each endpoint.
    B = np.zeros(V, dtype=float); B[list(graph.boundary)] = 1
    row_coefficients = 8 * abs(onsite_L) * B
    for edge in edges_L:
        coefficient = 4 * edge.coefficient_norm * (B[edge.x] + B[edge.y])
        row_coefficients[edge.x] += coefficient
        row_coefficients[edge.y] += coefficient
    average_upper = (boundary_electric / 2 + R4**.25 * V**.25
                     * float(np.sum(row_coefficients**(4 / 3)))**.75
                     / (2 * len(graph.boundary)))
    boundary_density = len(graph.boundary) / V
    return dict(domain='full R^(60V), genuine Haar links, untruncated compact EM rotors',
                Lambda_domain='Lambda >= 1', shape=list(graph.shape), volume=V,
                bond_count=len(graph.bonds), plaquette_count=graph.plaquettes,
                bond_density=bond_density, plaquette_density=plaquette_density,
                physical_boundary_ports=list(graph.boundary), boundary_density=boundary_density,
                light_path_count=len(edges_L), heavy_path_count=len(edges_H),
                light_incidence_coefficients=c_L_sites.tolist(),
                heavy_incidence_coefficients=c_H_sites.tolist(), cH=cH, cL=cL,
                sigma=sigma, kappa=kappa, J=J, K4=K4, KEM=KEM,
                boundary_electric=boundary_electric, gaussian=g,
                neutral_trial_coefficient_C0=C0, young_constant_D=young,
                volume_average_rho_fourth_upper=R4,
                cubic_volume_uniform_C0=uniform_C0, cubic_volume_uniform_cH=uniform_cH,
                cubic_volume_uniform_cL=uniform_cL, cubic_volume_uniform_R4=uniform_R4,
                cubic_fixed_density_boundary_upper=(boundary_electric / 2
                    + uniform_cL * uniform_R4**.25 / (2 * boundary_density)),
                cubic_all_boundary_slab_uniform_upper=(boundary_electric / 2
                    + uniform_cL * uniform_R4**.25 / 2),
                charge_row_sum_coefficients=row_coefficients.tolist(),
                PH_ensemble_occupation_denominator_both_signs=2.,
                boundary_average_charge_trial_upper=average_upper,
                positive_density_safe_upper=boundary_electric / 2 + cL * R4**.25 / (2 * boundary_density),
                all_vertices_are_boundary=bool(len(graph.boundary) == V),
                all_ports_average_upper=boundary_electric / 2 + cL * R4**.25 / 2,
                limits=['upper trial energy, not a lower charge gap',
                        'individual local moments are not inferred from volume average',
                        'sparse boundary and interior dressings have no uniform bulk certificate',
                        'no fixed-orbit, oscillator-shell, phase, Lorentz or canonical QED theorem'])


def primitive_singlet_and_ph():
    """Use the unchanged S=+ PauliCode, including its actual neutral marker.

    Modes 0:4 and 4:8 are local heavy/light for the singlet. For the independent
    one-body coefficient probe they label the two endpoints of one species.
    No canonical ladder matrices or spectator matter construct these controls.
    """
    code = PauliCode()
    masks = list(range(256))
    ladders = [rectangular_ladder(code, masks, masks, i, False) for i in range(8)]
    identity = sparse.eye(256, format='csr')
    car = max(norm(ladders[i] @ ladders[j].conjugate().T + ladders[j].conjugate().T @ ladders[i]
                   - (identity if i == j else 0)) for i in range(8) for j in range(8))
    nilpotent = max(norm(ladders[i] @ ladders[j] + ladders[j] @ ladders[i])
                    for i in range(8) for j in range(8))
    require_small(car, 'primitive eight-mode CAR')
    require_small(nilpotent, 'primitive eight-mode nilpotency')
    vacuum = np.zeros(256, complex); vacuum[0] = 1
    singlet = np.zeros(256, complex)
    for i, j in itertools.combinations(range(4), 2):
        k, ell = tuple(v for v in range(4) if v not in (i, j))
        labels = (i, j, k, ell)
        sign = (-1)**sum(labels[a] > labels[b] for a in range(4) for b in range(a + 1, 4))
        component = vacuum
        for orbital in reversed((i, j, 4 + k, 4 + ell)):
            component = ladders[orbital].conjugate().T @ component
        singlet += sign * component
    raw_norm_squared = float(np.vdot(singlet, singlet).real)
    require_small(abs(raw_norm_squared - 6), 'epsilon singlet norm')
    singlet /= np.sqrt(raw_norm_squared)
    generators = matrix_unit_basis() / np.sqrt(8)
    units = {(i, j): ladders[i].conjugate().T @ ladders[j] for i in range(8) for j in range(8)}
    q = [sparse_sum(t[i, j] * (units[i, j] + units[4 + i, 4 + j])
                    for i in range(4) for j in range(4)) for t in generators]
    singlet_gauss = max(norm(operator @ singlet) for operator in q)
    require_small(singlet_gauss, 'local epsilon singlet all15 colour Gauss')
    numbers = [sparse_sum(units[4 * f + i, 4 * f + i] for i in range(4)) for f in range(2)]
    number_residual = max(norm(operator @ singlet - 2 * singlet) for operator in numbers)
    require_small(number_residual, 'exact two-heavy/two-light local number')
    onebody_mean = max(abs(np.vdot(singlet, units[4 * f + i, 4 * f + j] @ singlet)
                           - (.5 if i == j else 0)) for f in range(2) for i in range(4) for j in range(4))
    require_small(onebody_mean, 'singlet reduced one-body density')
    # For eight modes, product(c+c^dagger) maps c to -c^dagger; parity removes -.
    parity = sparse.diags([(-1)**mask.bit_count() for mask in masks], format='csr', dtype=float)
    PH = parity
    for c in ladders:
        PH = PH @ (c + c.conjugate().T)
    unitary = norm(PH.conjugate().T @ PH - identity)
    ph_car = max(norm(PH @ c @ PH.conjugate().T - c.conjugate().T) for c in ladders)
    ph_number = max(norm(PH @ n @ PH.conjugate().T - (4 * identity - n)) for n in numbers)
    ph_gauss = max(norm(PH @ operator @ PH.conjugate().T + sparse_sum(
        t[j, i] * (units[i, j] + units[4 + i, 4 + j])
        for i in range(4) for j in range(4))) for t, operator in zip(generators, q))
    for residual, name in ((unitary, 'PH unitary'), (ph_car, 'actual PH ladders'),
                           (ph_number, 'PH local number'), (ph_gauss, 'PH Gauss automorphism')):
        require_small(residual, name)
    report = dict(embedding='unchanged PauliCode S=+; actual marker path words',
                  full_local_eight_mode_dimension=256, local_6_tensor_6_dimension=36,
                  actual_word_CAR_residual=car, actual_word_nilpotency_residual=nilpotent,
                  epsilon_singlet_raw_norm_squared=raw_norm_squared,
                  singlet_all15_Gauss_residual=singlet_gauss,
                  exact_local_number_residual=number_residual,
                  singlet_reduced_density_residual=float(onebody_mean),
                  PH_unitarity_residual=unitary, PH_ladder_residual=ph_car,
                  PH_population_residual=ph_number, PH_Gauss_automorphism_residual=ph_gauss)
    return report, PH, units


def ph_coefficient_controls(band, PH, units):
    """Actual scalar paths plus actual Pauli bilinears; no frozen field restriction."""
    basis = matrix_unit_basis()
    gram_basis = np.einsum('aij,bji->ab', basis, basis).real
    metric = norm(gram_basis - 4 * np.eye(15))
    require_small(metric, 'canonical raw TrBB=4 metric')
    coordinates = np.sin(np.arange(120).reshape(2, 4, 15) * .317 + .41) / 7
    fields = np.einsum('xma,aij->xmij', coordinates, basis)
    signs = np.array([-1., 1., 1., 1.])
    transformed = signs[None, :, None, None] * fields.transpose(0, 1, 3, 2)
    transposition = np.einsum('aij,bji->ab', basis.transpose(0, 2, 1), basis).real / 4
    orthogonal = norm(transposition @ transposition.T - np.eye(15))
    G = coordinates @ coordinates.transpose(0, 2, 1)
    transformed_coordinates = np.einsum('ma,xma->xma', np.broadcast_to(signs[:, None], (4, 15)),
                                        coordinates @ transposition)
    Gp = transformed_coordinates @ transformed_coordinates.transpose(0, 2, 1)
    quartic = abs(float(np.sum(G**2) - np.sum(Gp**2)))
    rho = abs(float(np.sum(coordinates**2) - np.sum(transformed_coordinates**2)))
    onsite, paths = scalar_paths(band)
    phase_residual = max([abs(-onsite + np.conjugate(onsite))] +
                        [norm(signs * scalars + scalars.conjugate()) for _, scalars in paths])
    require_small(phase_residual, 'real mass / imaginary kinetic PH coefficients')
    algebra_residual = primitive_residual = alignment = 0.
    adverse = 0.
    path_reports = []
    for displacement, scalars in paths:
        # Product of genuine SU4 primitive link VALUES, not compressed PUP.
        U = np.eye(4, dtype=complex)
        for step in range(int(np.sum(abs(displacement)))):
            U = U @ expm(1j * np.einsum('a,aij->ij',
                np.cos(np.arange(15) * .19 + step * .31) / 13, basis))
        z = np.exp(.37j * int(np.sum(displacement)))
        def onebody(phi, link, em):
            M = em * sum(scalars[mu] * (phi[0, mu] @ link + link @ phi[1, mu]) / 2
                         for mu in range(4))
            return np.block([[onsite * phi[0, 0], M], [M.conj().T, onsite * phi[1, 0]]])
        h = onebody(fields, U, z)
        hp = onebody(transformed, U.conjugate(), z.conjugate())
        residual = norm(hp + h.T)
        algebra_residual = max(algebra_residual, residual)
        H = sparse_sum(h[i, j] * units[i, j] for i in range(8) for j in range(8))
        Hp = sparse_sum(hp[i, j] * units[i, j] for i in range(8) for j in range(8))
        primitive_residual = max(primitive_residual, norm(PH @ Hp @ PH.conjugate().T - H))
        aligned = fields[0] - np.einsum('ij,mjk,kl->mil', U, fields[1], U.conj().T)
        aligned_p = transformed[0] - np.einsum('ij,mjk,kl->mil', U.conjugate(), transformed[1], U.T)
        alignment = max(alignment, abs(norm(aligned)**2 - norm(aligned_p)**2))
        wrong_fields = -fields.transpose(0, 1, 3, 2)
        wrong = norm(onebody(wrong_fields, U.conjugate(), z.conjugate()) + h.T)
        adverse = max(adverse, wrong)
        path_reports.append(dict(displacement=[int(v) for v in displacement],
                                 scalar_real=[float(v) for v in scalars.real],
                                 scalar_imag=[float(v) for v in scalars.imag],
                                 coefficient_norm=float(np.linalg.norm(scalars)),
                                 PH_onebody_residual=residual,
                                 wrong_all_fields_negative_transpose_defect=wrong))
    for residual, name in ((orthogonal, 'transpose real-coordinate isometry'),
                           (quartic, 'full quartic PH invariance'), (rho, 'PH rho invariance'),
                           (algebra_residual, 'PH endpoint-averaged path identity'),
                           (primitive_residual, 'actual-code PH whole h identity'),
                           (alignment, 'PH full alignment invariance')):
        require_small(residual, name, 2e-9)
    if band.v != 0 and adverse < 1e-5:
        raise AssertionError('Wrong spatial PH sign failed to discriminate the kinetic paths')
    return dict(onsite_scalar=float(onsite), paths=path_reports,
                raw_basis_metric_residual=metric, transpose_coordinate_orthogonality_residual=orthogonal,
                coefficient_phase_residual=phase_residual, full_quartic_PH_residual=quartic,
                rho_PH_residual=rho, alignment_PH_residual=alignment,
                actual_code_whole_h_PH_residual=primitive_residual,
                endpoint_path_PH_residual=algebra_residual,
                adverse_wrong_spatial_sign_defect=adverse)


def gaussian_wick_control(sigma):
    """Exact degree-four Wick contraction, independent of theorem's closed formula."""
    diagonal = sum(3 if a == b else 1 for a in range(15) for b in range(15))
    off_diagonal = sum(a == b for a in range(15) for b in range(15))
    gram = (4 * diagonal + 12 * off_diagonal) * sigma**4
    kinetic = sum(1 / (4 * sigma**2) for _ in range(60)) / 2
    expected = gaussian_constants(sigma)
    require_small(abs(gram - expected['gram_squared']), 'Wick Gram constant')
    require_small(abs(kinetic - expected['kinetic_sqrt_Lambda_coefficient']), 'Gaussian derivative constant')
    return dict(diagonal_Wick_integer=diagonal, off_diagonal_Wick_integer=off_diagonal,
                gram_Wick_integer=4 * diagonal + 12 * off_diagonal,
                gram_squared=gram, kinetic_sqrt_Lambda_coefficient=kinetic,
                density_not_wavefunction_variance=True)


def source_hashes():
    here = Path(__file__).resolve().parent
    names = ('full_field_charge_bound.py', 'full_field_charge_bound.md',
             'gauss_charge_intertwiner.py', 'gauss_charge_intertwiner.md',
             'quantum_su4_gauss.py', 'quantum_spinor_plaquette.py',
             'local_orbital_stability.py', 'rotation_band.py', 'sea_selection.py', 'pauli_compton.py')
    return {name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shape', type=int, nargs=3, default=(2, 4, 4))
    parser.add_argument('--a', type=float, default=1.)
    parser.add_argument('--m', type=float, default=.4)
    parser.add_argument('--r', type=float, default=.2)
    parser.add_argument('--v', type=float, default=.3)
    parser.add_argument('--improved', action='store_true')
    parser.add_argument('--kappa', type=float, default=1.)
    parser.add_argument('--J', type=float, default=.05)
    parser.add_argument('--K4', type=float, default=.01)
    parser.add_argument('--KEM', type=float, default=.02)
    parser.add_argument('--sigma', type=float, default=.3)
    parser.add_argument('--boundary-electric', type=float, default=.01)
    args = parser.parse_args()
    if not np.isfinite(args.a) or args.a <= 0:
        parser.error('--a must be finite and positive')
    if not all(np.isfinite(v) for v in (args.m, args.r, args.v)):
        parser.error('matter scalar coefficients must be finite')
    hashes = source_hashes()
    if hashes['gauss_charge_intertwiner.py'] != INTERTWINER_SOURCE or hashes['quantum_su4_gauss.py'] != GAUSS_SOURCE:
        raise AssertionError('Assigned primitive source revisions changed; review before scientific execution')
    band = RotationBand(args.a, m=args.m, r=args.r, v=args.v, improved=args.improved)
    graph = open_box(args.shape)
    primitive, PH, units = primitive_singlet_and_ph()
    report = dict(source_sha256=hashes,
                  theorem=theorem_constants(graph, band, band, kappa=args.kappa, J=args.J,
                      K4=args.K4, KEM=args.KEM, sigma=args.sigma,
                      boundary_electric=args.boundary_electric),
                  gaussian_wick=gaussian_wick_control(args.sigma),
                  primitive_singlet_and_PH=primitive,
                  actual_scalar_path_PH=ph_coefficient_controls(band, PH, units),
                  evidence_scope='exact algebra/analytic constants; no full-field numerical ground solver',
                  conclusion='boundary-average charged trial upper cost; not a positive lower gap')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
