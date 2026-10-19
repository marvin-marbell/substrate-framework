#!/usr/bin/env python3
"""Physical rotations, full primitive charged Gauss boundary, and native EM spin.

The finite microscopic rotation is linear. The alternative low-momentum
Noether/current construction is an effective-band theorem, not a quantum-phase
or microscopic mechanical-spin claim. No inverse reference seam is installed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

from bulk_frame_compton import raw_h, raw_paths, reconstruct_frame
from bulk_quantum_selection import BulkQuantumLaw, source_hashes as quantum_hashes
from full_band_gauss_section import FullBandGaussSection, source_hashes as gauss_hashes
from pair_full_field_charge_transfer import charged_map, lift
from quantum_su4_gauss import norm, sparse_sum
from rotation_band import spatial_rotation

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-continuing-backward-20260930'
PLAN_REVISION = '3d185c8c5f60f10464cbba570c02dc8688192106456ee6177cc8c1fdc18d07c9'
CYCLES = ((1, 2, 0), (2, 0, 1), (0, 1, 2))


def check(value: float, label: str, tolerance: float = 2e-8) -> None:
    if not np.isfinite(value) or value > tolerance:
        raise AssertionError(f'{label}: {value} > {tolerance}')


def adverse(value: float, label: str, minimum: float = 1e-6) -> None:
    if not np.isfinite(value) or value < minimum:
        raise AssertionError(f'{label}: adverse defect {value} < {minimum}')


def source_hashes() -> dict[str, str]:
    names = ('physical_spin_boundary.py', 'physical_spin_boundary.md',
             'bulk_frame_compton.py', 'bulk_frame_compton.md',
             'pair_full_field_charge_transfer.py', 'pair_full_field_charge_transfer.md',
             'gauged_rotation_fractionalization.py', 'gauged_rotation_fractionalization.md')
    return {**quantum_hashes(), **gauss_hashes(), **{
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}}


def physical_cubic(law: BulkQuantumLaw, fields: np.ndarray, links: np.ndarray,
                   phases: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rotate positions/vector indices/oriented links; matter has NO spin action."""
    targets = np.array([law.index[tuple((rotation @ site) % law.size)] for site in law.sites])
    outfields = np.empty_like(fields)
    outfields[targets, 0] = fields[:, 0]
    outfields[targets, 1:] = np.einsum('ij,xjkl->xikl', rotation, fields[:, 1:])
    outlinks, outphases = np.empty_like(links), np.empty_like(phases)
    for x in range(law.volume):
        for axis in range(3):
            newaxis = int(np.flatnonzero(rotation[:, axis])[0])
            sign = int(rotation[newaxis, axis])
            if sign == 1:
                start = targets[x]
                outlinks[start, newaxis] = links[x, axis]
                outphases[start, newaxis] = phases[x, axis]
            else:
                start = targets[law.target[x, axis]]
                outlinks[start, newaxis] = links[x, axis].conj().T
                outphases[start, newaxis] = -phases[x, axis]
    permutation = np.zeros((4 * law.volume, 4 * law.volume))
    for x, y in enumerate(targets):
        permutation[4*y:4*y+4, 4*x:4*x+4] = np.eye(4)
    return outfields, outlinks, outphases, permutation


def phase_current(law: BulkQuantumLaw, fields: np.ndarray, links: np.ndarray,
                  phases: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Exact derivative of the native EM links, including endpoint Phi and colour."""
    current = np.zeros((4 * law.volume, 4 * law.volume), complex)
    for x in range(law.volume):
        for axis, scalar in law.paths:
            y = law.target[x, axis]
            left = np.einsum('m,mij->ij', scalar, fields[x])
            right = np.einsum('m,mij->ij', scalar, fields[y])
            coupling = (left @ links[x, axis] + links[x, axis] @ right) / 2
            change = 1j * direction[x, axis] * np.exp(1j * phases[x, axis]) * coupling
            sx, sy = slice(4*x, 4*x+4), slice(4*y, 4*y+4)
            current[sx, sy] += change
            current[sy, sx] += change.conj().T
    return current


def spatial_controls(law: BulkQuantumLaw) -> dict[str, Any]:
    rng = np.random.default_rng(222810)
    fields = law.uniform_fields() + law.fields(.04 * rng.normal(size=(law.volume, 4, 15)))
    links = np.array([[expm(1j * np.einsum('a,aij->ij', .13*rng.normal(size=15), law.generators))
                       for _ in range(3)] for _ in law.sites])
    phases = .2 * rng.normal(size=(law.volume, 3))
    direction = rng.normal(size=phases.shape)
    h = law.onebody(fields, links, phases)
    current = phase_current(law, fields, links, phases, direction)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    rf, ru, rp, permutation = physical_cubic(law, fields, links, phases, rotation)
    _, _, rd, _ = physical_cubic(law, fields, links, direction, rotation)
    covariance = norm(law.onebody(rf, ru, rp) - permutation @ h @ permutation.T)
    current_covariance = norm(phase_current(law, rf, ru, rp, rd) - permutation @ current @ permutation.T)
    leading_before, leading_after = law.leading(fields, links), law.leading(rf, ru)
    leading_error = abs(leading_before['actual_leading_energy'] - leading_after['actual_leading_energy'])
    step = 1e-5
    current_fd = norm((law.onebody(fields, links, phases + step*direction)
                       - law.onebody(fields, links, phases - step*direction))/(2*step) - current)
    wrong_fields = rf.copy()
    targets = np.argmax(permutation[::4, ::4], axis=1)
    wrong_fields[:, 1:] = fields[targets, 1:]
    missing_vector_defect = norm(law.onebody(wrong_fields, ru, rp) - permutation @ h @ permutation.T)
    for label, value in (('physical native cubic covariance', covariance),
                         ('physical native current covariance', current_covariance),
                         ('whole changed-law cubic functional', leading_error),
                         ('native EM source finite difference', current_fd)):
        check(value, label)
    adverse(missing_vector_defect, 'omitted spatial Phi indices')
    def plaquette_xy(theta: np.ndarray) -> np.ndarray:
        return np.array([theta[x, 0]+theta[law.target[x, 0], 1]
                         -theta[law.target[x, 1], 0]-theta[x, 1] for x in range(law.volume)])
    curvature_before, curvature_after = plaquette_xy(phases), plaquette_xy(rp)
    moved_curvature = norm(curvature_after-curvature_before)
    eta = .3*rng.normal(size=law.volume)
    gauge_phases = phases+eta[:, None]-eta[law.target]
    gauge_curvature_error = norm(plaquette_xy(gauge_phases)-curvature_before)
    adverse(moved_curvature, 'physical spatial rotation is not a pure gauge operation')
    check(gauge_curvature_error, 'pure gauge leaves actual EM curvature unchanged')
    # Four actual physical quarter turns, never an internal half-angle operation.
    f, u, p = fields.copy(), links.copy(), phases.copy()
    total = np.eye(4 * law.volume)
    for _ in range(4):
        f, u, p, perm = physical_cubic(law, f, u, p, rotation)
        total = perm @ total
    loop_errors = dict(fields=norm(f-fields), colour_links=norm(u-links),
                       EM_links=norm(p-phases), scalar_matter=norm(total-np.eye(len(total))))
    for name, value in loop_errors.items():
        check(value, 'physical 2pi ' + name)
    # Actual U1 curvature: a periodic probe has zero total flux, not zero local B.
    em = np.zeros((law.volume, 3))
    em[:, 1] = .07*np.sin(2*np.pi*law.sites[:, 0]/law.size)
    curl = np.array([em[law.target[x, 0], 1]-em[x, 1] for x in range(law.volume)])
    em_response = norm(law.onebody(law.uniform_fields(), law.identity_links(), em)
                       - law.onebody(law.uniform_fields(), law.identity_links(), np.zeros_like(em)))
    colour_only_em_curl = norm(np.zeros_like(curl))
    adverse(norm(curl), 'actual U1 plaquette magnetic field')
    adverse(em_response, 'actual native U1 magnetic response')
    return dict(spatial_rotation=rotation.tolist(), full_native_dimension=len(h),
                arbitrary_noncentral_colour_links=True, complete_vector_position_link_rotation=True,
                matter_action='site permutation only; no inserted spin matrix',
                cubic_hamiltonian_covariance=covariance, cubic_current_covariance=current_covariance,
                whole_selector_elastic_alignment_colour_plaquette_covariance=leading_error,
                native_EM_current_finite_difference=current_fd,
                omitted_Phi_spatial_indices_defect=missing_vector_defect,
                full_physical_four_quarter_turn_errors=loop_errors,
                physical_rotation_changes_fixed_position_U1_curvature=moved_curvature,
                pure_EM_gauge_U1_curvature_error=gauge_curvature_error,
                actual_U1_plaquette_B=curl.tolist(), actual_U1_total_flux=float(curl.sum()),
                actual_U1_native_hamiltonian_response=em_response,
                colour_only_configuration_EM_curl=colour_only_em_curl,
                magnetic_ownership='B_EM is oriented compact U1 plaquette curl/a^2; not the external colour B4 reference')


def local_primitive(matter: Any, matrix: np.ndarray, site: int, species: int = 1) -> Any:
    return sparse_sum(matrix[i, j]*matter.units[site, species, i, j]
                      for i in range(4) for j in range(4)).tocsr()


def primitive_boundary_controls(law: BulkQuantumLaw, spin: np.ndarray) -> dict[str, Any]:
    model = FullBandGaussSection(law.band, heavy_scale=1.)
    fields = np.tile(law.frame, (2, 1, 1, 1))
    fields += law.fields(.003*np.sin(np.arange(law.volume*60)*.37).reshape(law.volume, 4, 15))[:2]
    link = expm(1j*np.einsum('a,aij->ij', .07*np.cos(np.arange(15)*.43), model.generators))
    phases = np.array([.17, -.11])
    section = model.state(fields, link, phases)
    derivative_checks, sectors = [], []
    tangents = []
    for site in range(2):
        t = np.einsum('a,aij->ij', .23*np.sin(np.arange(15)*.31+.2+site), model.generators)
        df = np.zeros_like(fields)
        df[site] = 1j*(t @ fields[site]-fields[site] @ t)
        du = 1j*t @ link if site == 0 else -1j*link @ t
        dpsi = model.derivative(section, fields, link, phases, df, du)
        q = local_primitive(model.matter, t, site, 0) + local_primitive(model.matter, t, site, 1)
        derivative_checks.append(norm(dpsi-1j*q @ section.state))
        tangents.append((site, t, dpsi))
    for charge in (-1, 1):
        matter, _, _, mapping, boundary = charged_map(model, charge)
        identity = sparse.eye(matter.dimension, format='csr')
        ib = sparse.eye(4, format='csr')
        gauss, omitted_reference = 0., 0.
        for site in range(2):
            for a in range(15):
                q = sparse.kron(matter.Q[site][a], ib, format='csr')
                b = sparse.kron(identity, sparse.csr_matrix(boundary[a]), format='csr') if site == 1 else sparse.csr_matrix(q.shape)
                gauss = max(gauss, norm((q+b) @ mapping-mapping @ model.matter.Q[site][a]))
                if site == 1:
                    omitted_reference = max(omitted_reference, norm(q @ mapping-mapping @ model.matter.Q[site][a]))
        image = (mapping @ section.state).reshape(matter.dimension, 4)
        image /= norm(image)
        # Physical orbit of the bond about the fixed right colour/EM reference.
        # Contracting n.Phi uses exactly the owner's native x path in its moved
        # Cartesian direction; this is a cubic graph rotation, not a new law.
        quarter = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        directed_fields = fields.copy()
        direction = np.array([1., 0., 0.])
        orientations = []
        boundary_h_error = boundary_state_error = boundary_current_error = 0.
        baseline_dh = model.onebody_derivative(fields, link, phases, np.zeros_like(fields),
                                                np.zeros_like(link), np.array([1., 0.]))[1]
        baseline_current = lift(matter, baseline_dh, 1) @ image
        baseline_operators = [lift(matter, section.onebody[species], species) for species in range(2)]
        rotated_image = image
        for _ in range(4):
            directed_fields[:, 1:] = np.einsum('ij,xjkl->xikl', quarter, directed_fields[:, 1:])
            direction = quarter @ direction
            orientations.append(direction.tolist())
            native_fields = directed_fields.copy()
            native_fields[:, 1] = np.einsum('i,xijk->xjk', direction, directed_fields[:, 1:])
            rotated_section = model.state(native_fields, link, phases)
            rotated_image = (mapping @ rotated_section.state).reshape(matter.dimension, 4)
            rotated_image /= norm(rotated_image)
            boundary_state_error = max(boundary_state_error, norm(rotated_image-image))
            for species in range(2):
                boundary_h_error = max(boundary_h_error,
                                        norm(lift(matter, rotated_section.onebody[species], species)-baseline_operators[species]))
            rotated_dh = model.onebody_derivative(native_fields, link, phases, np.zeros_like(native_fields),
                                                   np.zeros_like(link), np.array([1., 0.]))[1]
            boundary_current_error = max(boundary_current_error,
                                          norm(lift(matter, rotated_dh, 1) @ rotated_image-baseline_current))
        for label, value in (('held-reference physical bond Hamiltonian', boundary_h_error),
                             ('held-reference physical charged state loop', boundary_state_error),
                             ('held-reference physical native source current', boundary_current_error)):
            check(value, label)
        differential, omitted_field_link = 0., 0.
        for site, t, dpsi in tangents:
            coordinates = 2*np.einsum('aij,ji->a', model.generators, t).real
            q = local_primitive(matter, t, site, 0) + local_primitive(matter, t, site, 1)
            b = np.einsum('a,aij->ij', coordinates, boundary) if site == 1 else np.zeros((4, 4))
            raw = (mapping @ section.state).reshape(matter.dimension, 4)
            derivative = (mapping @ dpsi).reshape(matter.dimension, 4)
            generator = q @ raw + raw @ b.T
            differential = max(differential, norm(derivative-1j*generator))
            omitted_field_link = max(omitted_field_link, norm(generator))
        em_gauss, em_differential = 0., 0.
        nright = 4-model.matter.light_left
        for site in range(2):
            source = model.matter.light_left-2 if site == 0 else nright-2
            target = matter.light_left-2 if site == 0 else matter.light_number-matter.light_left-2-charge
            em_gauss = max(em_gauss, norm(sparse.kron(sparse.diags(target, dtype=float), ib) @ mapping-mapping @ sparse.diags(source, dtype=float)))
            gradient = np.array([1., -1.])*(1 if site == 0 else -1)
            dpsi = model.derivative(section, fields, link, phases, dphases=gradient)
            raw = mapping @ section.state
            em_differential = max(em_differential, norm(mapping @ dpsi-1j*np.repeat(target, 4)*raw))
        # Global connected locking lift. It is NOT a physical spatial protocol.
        qspin = sparse_sum(local_primitive(matter, spin[2], site, species)
                           for site in range(2) for species in range(2))
        bspin = spin[2] if charge == -1 else -spin[2].T
        matter_loop = expm_multiply(-2j*np.pi*qspin, image)
        reference_loop = expm(-2j*np.pi*bspin)
        full_loop = matter_loop @ reference_loop.T
        restored_error, matter_minus = norm(full_loop-image), norm(matter_loop+image)
        # Gauge-invariant field-dependent spin density, not a fixed Pauli marker.
        magnetic_covariance, magnetic_frozen_defect = 0., 0.
        for site in range(2):
            gs = float(np.trace(law.frame[2] @ law.frame[2]).real/4)
            operator = -1j*(fields[site, 1] @ fields[site, 2]-fields[site, 2] @ fields[site, 1])/(4*gs)
            t = model.generators[6]
            df = 1j*(t @ fields[site]-fields[site] @ t)
            dop = -1j*(df[1] @ fields[site, 2]+fields[site, 1] @ df[2]
                      -df[2] @ fields[site, 1]-fields[site, 2] @ df[1])/(4*gs)
            q = local_primitive(matter, t, site, 0)+local_primitive(matter, t, site, 1)
            op = local_primitive(matter, operator, site)
            magnetic_covariance = max(magnetic_covariance, norm(local_primitive(matter, dop, site)-1j*(q @ op-op @ q)))
            magnetic_frozen_defect = max(magnetic_frozen_defect, norm(q @ op-op @ q))
        for label, value in (('all30 colour Gauss intertwiners', gauss),
                             ('complete field link reference differential Gauss', differential),
                             ('both EM Gauss intertwiners', em_gauss),
                             ('complete compact EM derivative', em_differential),
                             ('full pure gauge centre loop plus', restored_error),
                             ('odd matter compensator minus', matter_minus),
                             ('field-dependent primitive magnetic covariance', magnetic_covariance)):
            check(value, label)
        adverse(omitted_reference, 'omitted fundamental boundary')
        adverse(omitted_field_link, 'omitted field/link angular derivatives')
        adverse(magnetic_frozen_defect, 'frozen internal magnetic marker is not gauge invariant')
        sectors.append(dict(charge=charge, boundary_representation='4' if charge == -1 else 'bar4',
                            full_Pauli_matter_dimension=matter.dimension, with_reference_dimension=4*matter.dimension,
                            colour_Gauss_operator_error=gauss, field_link_reference_Gauss_derivative_error=differential,
                            EM_Gauss_operator_error=em_gauss, EM_full_derivative_error=em_differential,
                            omitted_colour_reference_defect=omitted_reference,
                            omitted_field_link_derivative_defect=omitted_field_link,
                            actual_odd_matter_compensator_minus_error=matter_minus,
                            full_gauge_restored_loop_plus_error=restored_error,
                            held_reference_after_colour_only_loop_distance_from_initial=norm(matter_loop-image),
                            field_dependent_magnetic_Gauss_covariance_error=magnetic_covariance,
                            frozen_magnetic_marker_colour_Gauss_defect=magnetic_frozen_defect))
        sectors[-1]['held_reference_physical_spatial_protocol'] = dict(
            bond_orientations=orientations, native_primitive_hamiltonian_covariance_error=boundary_h_error,
            charged_section_linear_spatial_action_error=boundary_state_error,
            native_EM_current_along_rotated_bond_error=boundary_current_error,
            full_physical_two_pi_plus_error=norm(rotated_image-image),
            reference='right endpoint is the fixed spatial pivot; both colour and EM reference held, no inverse seam or internal spin action',
            scope='four exact cubic graph rotations with physically moved bond and Phi indices; a finite section counterexample, not continuous Noether symmetry or quantum-ground transport')
    check(max(derivative_checks), 'neutral all-field Gauss tangents')
    return dict(native_owner='FullBandGaussSection/PrimitiveMatter/charged_map actual edge-Pauli words',
                neutral_full_dimension=model.matter.dimension, neutral_Gauss_tangent_errors=derivative_checks,
                arbitrary_all120_fields_noncentral_SU4_and_two_EM_paths=True, sectors=sectors,
                reference_ownership='colour 4/bar4 and compensating EM charge -charge at right; static spatial scalar reference, no assigned intrinsic mechanical spin',
                field_link_domain='all Cartesian R120, genuine SU4 Haar link, two compact U1 phases; pointwise section identities, not a quantum eigenstate',
                generator_identity='Gcolour=Lfield+Elink-Qmatter-Qreference; GEM=Ediv-(Nlight-2)-qreference; no omitted heavy charge',
                inference='The matter half-angle with held colour reference alone is not a Gauss-preserving closed physical rotation. Completing the gauge loop gives +1.')


def native_current(bare: Any, frame: np.ndarray, pout: np.ndarray, pin: np.ndarray) -> np.ndarray:
    _, paths = raw_paths(bare, frame)
    result = np.zeros((3, 4, 4), complex)
    for displacement, coupling in paths:
        axis = int(np.flatnonzero(displacement)[0])
        phase = np.exp(1j*bare.a*np.dot((pin+pout)/2, displacement))
        result[axis] += 1j*bare.a*(phase*coupling-phase.conjugate()*coupling.conj().T)
    return result


def gordon_noether_controls(law: BulkQuantumLaw) -> dict[str, Any]:
    band, unitary, reconstruction = reconstruct_frame(law.band, law.frame)
    gs = float(law.gram[1, 1])
    spin = np.array([-1j*(law.frame[j+1] @ law.frame[k+1]-law.frame[k+1] @ law.frame[j+1])/(4*gs)
                     for j, k, _ in CYCLES])
    beta = law.frame[0]/np.sqrt(law.gram[0, 0])
    alpha = law.frame[1:]/np.sqrt(gs)
    rest = unitary @ np.column_stack([band.spinor(np.zeros(3), label) for label in range(2)])
    srest = np.array([rest.conj().T @ s @ rest for s in spin])
    cspin = band.v**2/band.m
    cinertia = cspin+band.a*band.r
    minertia = 1/cinertia
    gpredicted = 2*minertia*cspin
    axis = np.array([1., 2., -1.]); axis /= norm(axis)
    angle = .61
    rotation = spatial_rotation(axis, angle)
    def positive_quadratic(p: np.ndarray) -> np.ndarray:
        mass = band.m+band.a*band.r*np.dot(p, p)/2
        energy = float(np.sqrt(mass**2+band.v**2*np.dot(p, p)))
        factor = np.sqrt((energy+mass)/(2*energy))
        return factor*(rest+band.v*np.einsum('i,ijk->jk', p, alpha) @ rest/(energy+mass))

    _, rest_labels = np.linalg.eigh(srest[2])
    spin_up = rest_labels[:, 1]
    d = expm(-1j*angle*np.einsum('i,ijk->jk', axis, spin))
    pbase = np.array([.23, -.31, .17])
    pavgbase, qbase = np.array([.17, -.11, .09]), np.array([.13, .07, -.19])
    rows = []
    for scale in (.08, .04, .02, .01):
        pavg, q = scale*pavgbase, scale*qbase
        pin, pout = pavg-q/2, pavg+q/2
        incoming = unitary @ np.column_stack([band.spinor(pin, label) for label in range(2)])
        outgoing = unitary @ np.column_stack([band.spinor(pout, label) for label in range(2)])
        raw = native_current(law.band, law.frame, pout, pin)
        projected = np.array([outgoing.conj().T @ j @ incoming for j in raw])
        gordon = cinertia*pavg[:, None, None]*np.eye(2)
        gordon = gordon.astype(complex)
        for i in range(3):
            for j, k, a in CYCLES:
                if i == a:
                    gordon[i] += -1j*cspin*(q[j]*srest[k]-q[k]*srest[j])
        gordon_error = norm(projected-gordon)
        # Extract moment from an actual finite-transfer EM current, not from S algebra.
        qx = .2*scale
        pinx, poutx = np.array([-qx/2, 0., 0.]), np.array([qx/2, 0., 0.])
        uin = unitary @ np.column_stack([band.spinor(pinx, label) for label in range(2)])
        uout = unitary @ np.column_stack([band.spinor(poutx, label) for label in range(2)])
        jy = uout.conj().T @ native_current(law.band, law.frame, poutx, pinx)[1] @ uin
        basis = 1j*qx*srest[2]
        cmeasured = float(np.vdot(basis, jy).real/np.vdot(basis, basis).real)
        gmeasured = 2*minertia*cmeasured
        p = scale*pbase
        h = raw_h(law.band, law.frame, p)
        hp = raw_h(law.band, law.frame, rotation @ p)
        covariance = norm(hp-d @ h @ d.conj().T)
        tangent = np.cross(axis, p)
        dh = np.einsum('i,ijk->jk', tangent, native_current(law.band, law.frame, p, p))
        s_axis = np.einsum('i,ijk->jk', axis, spin)
        noether_defect = norm(s_axis @ h-h @ s_axis-1j*dh)
        h2 = band.m*beta + band.v*np.einsum('i,ijk->jk', p, alpha) + band.a*band.r*np.dot(p, p)*beta/2
        h2r = band.m*beta + band.v*np.einsum('i,ijk->jk', rotation @ p, alpha) + band.a*band.r*np.dot(p, p)*beta/2
        h2_error = norm(h2r-d @ h2 @ d.conj().T)
        pure_spin_defect = norm(s_axis @ h2-h2 @ s_axis)
        dh2 = band.v*np.einsum('i,ijk->jk', tangent, alpha)
        noether_h2 = norm(s_axis @ h2-h2 @ s_axis-1j*dh2)
        positive = positive_quadratic(p)
        mass2 = band.m+band.a*band.r*np.dot(p, p)/2
        energy2 = float(np.sqrt(mass2**2+band.v**2*np.dot(p, p)))
        tangent_z = np.cross(np.array([0., 0., 1.]), p)
        dpositive = (np.sqrt((energy2+mass2)/(2*energy2))*band.v
                     * np.einsum('i,ijk->jk', tangent_z, alpha) @ rest/(energy2+mass2))
        jpositive = spin[2] @ positive-1j*dpositive
        positive_Jz_error = norm((jpositive-.5*positive) @ spin_up)
        positive_eigen_error = norm(h2 @ positive-energy2*positive)
        j_squared = np.zeros(4, complex)
        positive_rotation_action_error = 0.
        for a in range(3):
            tangent_a = np.cross(np.eye(3)[a], p)
            dpositive_a = (np.sqrt((energy2+mass2)/(2*energy2))*band.v
                           * np.einsum('i,ijk->jk', tangent_a, alpha) @ rest/(energy2+mass2))
            ja_positive = spin[a] @ positive-1j*dpositive_a
            positive_rotation_action_error = max(positive_rotation_action_error,
                                                   norm(ja_positive-positive @ srest[a]))
            j_squared += ja_positive @ srest[a] @ spin_up
        positive_J_squared_error = norm(j_squared-.75*positive @ spin_up)
        check(positive_rotation_action_error, 'positive packet full spatial derivative action')
        check(positive_J_squared_error, 'localized positive packet mechanical J squared')
        angular_step = 1e-4
        zplus = spatial_rotation(np.array([0., 0., 1.]), angular_step) @ p
        zminus = spatial_rotation(np.array([0., 0., 1.]), -angular_step) @ p
        native_positive = unitary @ np.column_stack([band.spinor(p, label) for label in range(2)])
        native_derivative = unitary @ np.column_stack([
            (band.spinor(zplus, label)-band.spinor(zminus, label))/(2*angular_step)
            for label in range(2)])
        native_packet_Jz_error = norm((spin[2] @ native_positive-1j*native_derivative
                                       -native_positive @ srest[2]) @ spin_up)
        check(positive_Jz_error, 'localized positive packet Jz half eigenvalue')
        check(positive_eigen_error, 'packet actually in positive quadratic band')
        rows.append(dict(momentum_scale=scale, projected_native_current_Gordon_error=gordon_error,
                         measured_current_spin_coefficient=cmeasured, measured_g_with_inertial_mass=gmeasured,
                         native_non_cubic_rotation_defect=covariance, native_continuous_J_commutator_defect=noether_defect,
                         isotropic_quadratic_h_covariance_error=h2_error, full_L_plus_S_quadratic_Noether_error=noether_h2,
                         spin_only_quadratic_Noether_defect=pure_spin_defect,
                         actual_raw_h_quadratic_remainder=norm(h-h2),
                         positive_packet_quadratic_Jz_half_error=positive_Jz_error,
                         positive_packet_quadratic_energy_eigen_error=positive_eigen_error,
                         positive_packet_quadratic_J_squared_error=positive_J_squared_error,
                         positive_packet_spatial_derivative_action_error=positive_rotation_action_error,
                         native_positive_packet_Jz_half_defect=native_packet_Jz_error))
        check(h2_error, 'quadratic physical band rotation')
        check(noether_h2, 'quadratic full Noether L+S')
        adverse(pure_spin_defect, 'spin only is not mechanical J', 1e-7)
    for key in ('projected_native_current_Gordon_error', 'native_continuous_J_commutator_defect',
                'native_non_cubic_rotation_defect', 'actual_raw_h_quadratic_remainder',
                'native_positive_packet_Jz_half_defect'):
        ratios = [rows[i][key]/rows[i+1][key] for i in range(3)]
        if min(ratios) < 6.5:
            raise AssertionError(f'{key}: expected cubic small-momentum decrease, found {ratios}')
    check(abs(rows[-1]['measured_g_with_inertial_mass']-gpredicted), 'native current g limit', 2e-6)
    adverse(abs(gpredicted-2), 'unmodified assumed g=2')
    # A spatial wavepacket is moved, not merely its four-component internal label.
    centre = np.array([.07, -.04, .02]); width = .09
    sample = np.array([.03, .08, -.02])
    def envelope(p: np.ndarray) -> float:
        return float(np.exp(-np.dot(p-centre, p-centre)/(2*width**2)))
    correct = d @ positive_quadratic(rotation.T @ sample) @ spin_up*envelope(rotation.T @ sample)
    spin_only = d @ positive_quadratic(sample) @ spin_up*envelope(sample)
    spatial_packet_defect = norm(correct-spin_only)
    adverse(spatial_packet_defect, 'spin-only packet is not spatial rotation')
    # Constant projector Berry transport is adversarial and remains +I.
    berry = np.eye(2, dtype=complex)
    for _ in range(64):
        left, _, right = np.linalg.svd(rest.conj().T @ rest)
        berry = left @ right @ berry
    berry_error = norm(berry-np.eye(2))
    single_loop = expm(-2j*np.pi*spin[2])
    pair_loop_error = norm(np.kron(single_loop, single_loop)-np.eye(16))
    check(berry_error, 'constant rest projector Berry identity')
    check(norm(single_loop+np.eye(4)), 'effective full packet 2pi minus')
    check(pair_loop_error, 'two effective carriers global integer')
    pair_Jz = np.kron(srest[2], np.eye(2))+np.kron(np.eye(2), srest[2])
    pair_Jz_spectrum = np.linalg.eigvalsh(pair_Jz)
    check(norm(pair_Jz_spectrum-np.array([-1., 0., 0., 1.])), 'two positive spatial packets integer Jz')
    # Inertial mass is independently measured from the same raw onebody spectrum.
    step = 2e-4
    energy = lambda p: float(np.linalg.eigvalsh(raw_h(law.band, law.frame, np.asarray(p)))[-1])
    inverse_measured = (energy([step, 0., 0.])+energy([-step, 0., 0.])-2*energy([0., 0., 0.]))/step**2
    check(abs(inverse_measured-cinertia), 'raw native inertial curvature', 2e-6)
    return dict(spin=spin, report=dict(actual_frame_reconstruction=reconstruction,
                field_commutator_spin_owner='-i[Phi_j,Phi_k]/(4 G_S); scalar G_S is gauge invariant',
                native_current_owner='raw endpoint Phi couplings differentiated with respect to physical U1 link source',
                projected_native_current_expansion='j_i/charge=(v^2/m+a*r) P_i I - i(v^2/m) epsilon_ijk q_j S_k + O((|P|+|q|)^3)',
                spin_current_coefficient=cspin, inertial_inverse=cinertia, actual_raw_inertial_inverse=inverse_measured,
                inertial_mass=minertia, derived_g=gpredicted,
                derived_g_formula='2/(1+m*a*r/v^2), using the actual frame-derived m,r,v',
                physical_EM_moment_coefficient=band.charge*cspin,
                physical_EM_moment='mu_EM=charge*(v^2/m) S; H_B=-mu_EM dot curl(A_U1)',
                low_momentum_rows=rows, spatial_packet_spin_only_defect=spatial_packet_defect,
                localized_positive_packet='radial Gaussian times positive_quadratic(p) spin_up; Jz eigenvalue +1/2, spatially localized Schwartz packet in effective continuum band',
                effective_pair_Jz_spectrum=pair_Jz_spectrum.tolist(),
                full_effective_generator='J=-i p cross grad_p+S on L2(R3,C4); h2=m beta+v alpha.p+(a*r/2)p^2 beta',
                microscopic_generator='spatial scalar-matter orbital + Cartesian vector-Phi orbital/index + colour/U1 link orbital + scalar reference orbital; no inserted matter intrinsic spin',
                effective_single_two_pi_minus_error=norm(single_loop+np.eye(4)),
                effective_two_carrier_two_pi_plus_error=pair_loop_error,
                constant_rest_projector_Berry_identity_error=berry_error,
                constant_rest_projector_invariance=norm(d @ rest @ rest.conj().T @ d.conj().T-rest @ rest.conj().T),
                limit='fixed-a low-momentum rotational defect O(a^2 v |p|^3+a^3 r |p|^4); quadratic band has exact continuum Noether J; no exact SO3 on finite lattice',
                scope='same-source finite-current/magnetic moment and continuum-band mechanical generator, not a proved embedding into the full quantum physical charged Hilbert space'))


def run(size: int = 3) -> dict[str, Any]:
    identities = source_hashes()
    law = BulkQuantumLaw(size)
    spatial = spatial_controls(law)
    alternative = gordon_noether_controls(law)
    primitive = primitive_boundary_controls(law, alternative['spin'])
    if source_hashes() != identities:
        raise RuntimeError('An owner source or proof changed during execution')
    return dict(source_sha256=identities, source_bindings_unchanged=True,
                governing_plan=PLAN, governing_plan_revision=PLAN_REVISION, work_item='campaign',
                quantum_supplier='explicit changed local27 fundamental/antifundamental CAR selector law; original itinerant-heavy fork not replaced',
                size=size, finite_physical_rotations=spatial, primitive_complete_Gauss_boundary=primitive,
                material_alternate_current_and_Noether_fork=alternative['report'],
                checked_route_failure='A matter/selected-frame half-angle alone with scalar held reference is not a closed complete-Gauss physical rotation. Actual physical cubic 2pi and restored pure gauge loops are +1; constant-projector Berry is I.',
                spin_claim_levels=dict(internal_algebra='exact in Clifford limit, supplier-dependent',
                                       EM_magnetic_moment='derived from actual native low-q U1 current, including Wilson inertial debt',
                                       projective_symmetry='effective selected-band/odd-charge lift only; not established localized microscopic symmetry fractionalization',
                                       mechanical_spin='exact Noether half-spin for the isotropic quadratic continuum-band fork; full microscopic charged-carrier mechanical half-spin unproved'),
                exact_remaining_premises=[
                    'A thermodynamic full-Gauss charged sector with deconfined colour/Z4 and EM dressing, localized relative to a physically specified reference, and a finite-energy mobile positive charged pole.',
                    'An intertwining/limit theorem for the microscopic spatial Noether generator on that charged sector, including field/link/reference/selector angular terms, identifying its local action with continuum L+S rather than cancellation by boundary/reference torque.',
                    'Uniform volume and continuum control sufficient to factorize two distant charged rotations. Fixed-volume field concentration does not establish this or spontaneous symmetry breaking.',
                    'Joining that sector to the actual compact photon/current quantum measure and the original-law or explicitly changed-law quantum ground; no Lorentz/radiative or campaign closure claimed.'],
                all_checks_pass=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=3)
    parser.add_argument('--output', type=Path, default=ROOT/'physical_spin_boundary_receipt.json')
    args = parser.parse_args()
    result = run(args.size)
    text = json.dumps(result, indent=2)
    args.output.write_text(text+'\n')
    print(text)


if __name__ == '__main__':
    main()
