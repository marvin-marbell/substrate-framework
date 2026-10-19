#!/usr/bin/env python3
"""Original full-Gram physical binary-cube endpoint lift, not mechanical spin.

Only the native geometry/endpoint methods of BulkQuantumLaw are consumed; its
changed 27-selector energy and frame are never used. See the companion proof.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from bulk_quantum_selection import BulkQuantumLaw
from original_charged_principal import original_frame, portable, source_hashes as principal_hashes
from original_normal_gap_bound import peierls_derivatives
from physical_spin_boundary import physical_cubic
from rotation_band import RotationBand
from unrestricted_field_minimum import BARE

RealArray = NDArray[np.float64]
Matrix = NDArray[np.complex128]
IntArray = NDArray[np.int64]
ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-continuing-backward-20260930'
PLAN_REVISION = '381993ecb362c8c67dbc334aebbcb6093b64a7b78111b5a9e785d4746b9bbe3b'
TEMPLATE_REVISION = '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'
QUARTER_TURN = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int64)


def check(error: float, label: str, tolerance: float = 2e-8) -> None:
    if not np.isfinite(error) or error > tolerance:
        raise AssertionError(f'{label}: {error} > {tolerance}')


def defect(matrix: NDArray[Any]) -> float:
    return float(np.linalg.norm(matrix))


def reject_same_assertion(error: float, label: str) -> dict[str, Any]:
    """An adverse branch must reach and fail the very same covariance check."""
    try:
        check(error, label)
    except AssertionError as failure:
        return dict(defect=error, rejected=True, assertion=str(failure))
    raise AssertionError(f'{label}: adverse input did not fail the covariance assertion')


def source_hashes() -> dict[str, str]:
    names = ('original_physical_rotation_lift.py', 'original_physical_rotation_lift.md',
             'original_normal_gap_bound.py', 'original_normal_gap_bound.md',
             'physical_spin_boundary.py', 'physical_spin_boundary.md')
    return {**principal_hashes(), **{
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}}


def proper_cube_rotations() -> list[IntArray]:
    rotations = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            rotation = np.zeros((3, 3), dtype=np.int64)
            rotation[np.arange(3), permutation] = signs
            if round(float(np.linalg.det(rotation))) == 1:
                rotations.append(rotation)
    rotations.sort(key=lambda rotation: tuple(rotation.reshape(-1)))
    return rotations


def rotation4(rotation: IntArray) -> RealArray:
    result = np.eye(4)
    result[1:, 1:] = rotation
    return result


def recover_gamma(gram: RealArray, frame: Matrix) -> Matrix:
    """Whiten the ACTUAL full-ten-coordinate selected frame, not isotropic copies."""
    values, vectors = np.linalg.eigh(gram)
    if values[0] <= 0:
        raise ValueError('Physical rotation lift requires a positive full native Gram')
    inverse_root = (vectors / np.sqrt(values)) @ vectors.T
    gamma = np.einsum('mn,nij->mij', inverse_root, frame)
    error = max(defect(gamma[m] @ gamma[n] + gamma[n] @ gamma[m]
                       - (2*np.eye(4) if m == n else np.zeros((4, 4))))
                for m in range(4) for n in range(4))
    check(error, 'actual whitened native Clifford algebra')
    return gamma


def clifford_compensator(rotation: IntArray, gamma: Matrix) -> Matrix:
    """Solve the vector conjugation in the native Clifford bivector algebra.

    Quaternion coordinates describe the physical R; the matrices and coefficient
    in this lift are fixed by the Clifford commutators, not an added spin label.
    S_R Gamma_a S_R^dagger = (diag(1,R) Gamma)_a. This S is anti-multiplicative;
    the actual body-restoration action S_R^dagger is multiplicative up to sign.
    """
    x, y, z, w = Rotation.from_matrix(rotation).as_quat(canonical=True)
    spin = (w*np.eye(4) + x*gamma[2] @ gamma[3]
            + y*gamma[3] @ gamma[1] + z*gamma[1] @ gamma[2])
    check(defect(spin @ spin.conj().T - np.eye(4)), 'derived SU4 compensator unitary')
    check(abs(complex(np.linalg.det(spin)) - 1), 'derived SU4 compensator determinant')
    check(defect(np.einsum('ij,mjk,kl->mil', spin, gamma, spin.conj().T)
                 - np.einsum('mn,nij->mij', rotation4(rotation), gamma)),
          'derived SU4 Clifford vector conjugation')
    return spin


def site_targets(law: BulkQuantumLaw, rotation: IntArray) -> IntArray:
    return np.array([law.index[tuple((rotation @ site) % law.size)] for site in law.sites],
                    dtype=np.int64)


def canonical_seam(law: BulkQuantumLaw, centre: IntArray) -> IntArray:
    return np.where(law.sites == law.size-1, centre[None, :], 0).astype(np.int64) % 4


def rotate_seam(law: BulkQuantumLaw, seam: IntArray, rotation: IntArray) -> IntArray:
    """Exact integer counterpart of physical_cubic's oriented colour links."""
    targets = site_targets(law, rotation)
    result = np.empty_like(seam)
    for axis in range(3):
        newaxis = int(np.flatnonzero(rotation[:, axis])[0])
        sign = int(rotation[newaxis, axis])
        start = targets if sign == 1 else targets[law.target[:, axis]]
        result[start, newaxis] = sign*seam[:, axis]
    return result % 4


def seam_cochain(law: BulkQuantumLaw, centre: IntArray, rotation: IntArray) -> IntArray:
    """Unique Z4 gauge k(y) with k(0)=0 restoring the ACTUAL rotated seams.

    z_target(y,a)=k(y)+z_rotated(y,a)-k(y+e_a) (mod 4).
    The output phase is kappa(y)=i**k(y); reversed axes are not discarded.
    """
    rotated = rotate_seam(law, canonical_seam(law, centre), rotation)
    target = canonical_seam(law, (rotation @ centre) % 4)
    root = law.index[(0, 0, 0)]
    cochain = np.full(law.volume, -1, dtype=np.int64)
    cochain[root] = 0
    queue = deque([root])
    while queue:
        x = queue.popleft()
        for axis in range(3):
            y = int(law.target[x, axis])
            expected = int((cochain[x]+rotated[x, axis]-target[x, axis]) % 4)
            if cochain[y] == -1:
                cochain[y] = expected
                queue.append(y)
            elif cochain[y] != expected:
                raise AssertionError('Actual rotated centre seam is not gauge-cohomologous')
    if np.any(cochain < 0):
        raise AssertionError('Native torus seam solver did not visit every primitive site')
    residual = (cochain[:, None]+rotated-cochain[law.target]-target) % 4
    if np.any(residual):
        raise AssertionError('Actual Z4 seam equation failed on a primitive link')
    return cochain


@dataclass(frozen=True)
class PhysicalRotationLift:
    """Full-site endpoint intertwiner, with no dense permutation allocation."""
    rotation: IntArray
    source_centre: IntArray
    target_centre: IntArray
    targets: IntArray
    cochain: IntArray
    compensator: Matrix

    @property
    def local(self) -> Matrix:
        return (1j**self.cochain)[:, None, None]*self.compensator.conj().T

    def onebody(self, matrix: Matrix) -> Matrix:
        """K_R P_R h P_R^dagger K_R^dagger, all 4V native modes retained."""
        volume = len(self.targets)
        inverse = np.argsort(self.targets)
        blocks = matrix.reshape(volume, 4, volume, 4)[inverse][:, :, inverse, :]
        local = self.local
        return np.einsum('xij,xjyk,ylk->xiyl', local, blocks, local.conj(),
                         optimize=True).reshape(4*volume, 4*volume)

    def fields(self, fields: Matrix) -> Matrix:
        """Body restoration AFTER physical_cubic, not instead of physical R."""
        local = self.local
        return np.einsum('xij,xmjk,xlk->xmil', local, fields, local.conj())


def make_lift(law: BulkQuantumLaw, centre: IntArray, rotation: IntArray,
              gram: RealArray, frame: Matrix) -> PhysicalRotationLift:
    return PhysicalRotationLift(rotation, centre, (rotation @ centre) % 4,
                                site_targets(law, rotation), seam_cochain(law, centre, rotation),
                                clifford_compensator(rotation, recover_gamma(gram, frame)))


def restore_links(law: BulkQuantumLaw, lift: PhysicalRotationLift, links: Matrix) -> Matrix:
    local = lift.local
    return np.einsum('xij,xajk,xalk->xail', local, links, local[law.target].conj())


def gram_group_controls(law: BulkQuantumLaw) -> tuple[dict[str, Any], dict[tuple[int, ...], dict[str, Any]]]:
    """Independent original native minimization at all64 classes; every group pair."""
    rotations = proper_cube_rotations()
    classes = list(itertools.product(range(4), repeat=3))
    data: dict[tuple[int, ...], dict[str, Any]] = {}
    for centre in classes:
        selected, frame = original_frame(law.band, law.size, np.pi/2*np.asarray(centre), True)
        data[centre] = dict(selected=selected, frame=frame,
                            gamma=recover_gamma(selected['G'], frame))
    indices = {tuple(rotation.reshape(-1)): i for i, rotation in enumerate(rotations)}
    product = np.array([[indices[tuple((r @ t).reshape(-1))] for t in rotations]
                        for r in rotations], dtype=np.int64)
    spins = np.array([[clifford_compensator(r, data[c]['gamma']) for r in rotations]
                      for c in classes])
    class_index = {c: i for i, c in enumerate(classes)}
    cochains = np.array([[seam_cochain(law, np.asarray(c), r) for r in rotations]
                         for c in classes])
    targets = np.array([site_targets(law, r) for r in rotations])
    gram_error = frame_error = spin_cocycle_error = gauss_adjoint_error = 0.
    seam_cocycle_error = 0
    nontrivial_seams = 0
    signs = np.empty((24, 24), dtype=np.int64)
    for ci, c in enumerate(classes):
        centre = np.asarray(c)
        for ri, r in enumerate(rotations):
            target = data[tuple((r @ centre) % 4)]
            r4 = rotation4(r)
            gram_error = max(gram_error, defect(target['selected']['G']
                             - r4 @ data[c]['selected']['G'] @ r4.T))
            spin = spins[ci, ri]
            frame_error = max(frame_error, defect(np.einsum('mn,nij->mij', r4, data[c]['frame'])
                - np.einsum('ij,mjk,kl->mil', spin, target['frame'], spin.conj().T)))
            rotated_generators = np.einsum('ij,ajk,kl->ail', spin.conj().T,
                                         law.generators, spin)
            adjoint = 2*np.einsum('aij,bji->ab', law.generators, rotated_generators).real
            gauss_adjoint_error = max(gauss_adjoint_error, defect(adjoint.T @ adjoint-np.eye(15)),
                defect(np.einsum('ab,aij->bij', adjoint, law.generators)-rotated_generators))
            nontrivial_seams += int(np.any(cochains[ci, ri]))
            for ti, t in enumerate(rotations):
                # R after T: the second S is extracted from the actual T-centre frame.
                tci = class_index[tuple((t @ centre) % 4)]
                pi = int(product[ri, ti])
                composed = spins[tci, ri].conj().T @ spins[ci, ti].conj().T
                direct = spins[ci, pi].conj().T
                sign = 1 if defect(composed-direct) <= defect(composed+direct) else -1
                spin_cocycle_error = max(spin_cocycle_error, defect(composed-sign*direct))
                if ci == 0:
                    signs[ri, ti] = sign
                elif signs[ri, ti] != sign:
                    raise AssertionError('Binary-cube sign spuriously depends on the native Gram class')
                pulled = cochains[ci, ti][np.argsort(targets[ri])]
                residual = (cochains[tci, ri]+pulled-cochains[ci, pi]) % 4
                seam_cocycle_error = max(seam_cocycle_error, int(np.max(residual)))
    associator_error = max(abs(int(signs[ri, ti]*signs[product[ri, ti], ui]
                                - signs[ti, ui]*signs[ri, product[ti, ui]]))
                          for ri in range(24) for ti in range(24) for ui in range(24))
    for error, label in ((gram_error, 'all64 physical native Gram covariance'),
                         (frame_error, 'all64 actual native frame compensation'),
                         (spin_cocycle_error, 'all64 binary-cube endpoint cocycle'),
                         (float(seam_cocycle_error), 'all64 actual rooted Z4 seam group law'),
                         (float(associator_error), 'binary-cube cocycle associativity'),
                         (gauss_adjoint_error, 'complete15 colour Gauss adjoint intertwiner')):
        check(error, label)
    return dict(classes=64, proper_rotations=24, gram_pairs=64*24, group_pairs=64*24*24,
                all64_full10_selected=True, maximum_stationarity=max(
                    float(data[c]['selected']['gradient_residual']) for c in classes),
                Gram_covariance=gram_error, actual_frame_covariance=frame_error,
                endpoint_cocycle=spin_cocycle_error, rooted_seam_cocycle=seam_cocycle_error,
                associator=associator_error, all15_Gauss_adjoint=gauss_adjoint_error,
                nontrivial_rooted_seam_lifts=nontrivial_seams,
                binary_sign_table=signs, centres=[dict(centre=c, G=data[c]['selected']['G'])
                                               for c in classes]), data


def plaquettes(law: BulkQuantumLaw, phases: RealArray) -> Matrix:
    return np.array([np.exp(1j*(phases[:, a]+phases[law.target[:, a], b]
                    - phases[law.target[:, b], a]-phases[:, b]))
                     for a, b in ((0, 1), (1, 2), (2, 0))]).T


def native_case(law: BulkQuantumLaw, centre: IntArray, selected: dict[str, Any],
                frame: Matrix, rotation: IntArray, rng: np.random.Generator) -> dict[str, Any]:
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    links = law.toron_links(np.pi/2*centre)
    # Full compact circle, not a small-angle or zero-curvature restriction.
    phases = rng.uniform(-np.pi, np.pi, (law.volume, 3))
    direction = rng.normal(size=phases.shape)
    lift = make_lift(law, centre, rotation, selected['G'], frame)
    rf, ru, rp, _ = physical_cubic(law, fields, links, phases, rotation)
    _, _, rd, _ = physical_cubic(law, fields, links, direction, rotation)
    rfields = lift.fields(rf)
    rlinks = restore_links(law, lift, ru)
    target_selected, target_frame = original_frame(law.band, law.size,
                                                  np.pi/2*lift.target_centre, True)
    target_fields = np.tile(target_frame, (law.volume, 1, 1, 1))
    target_links = law.toron_links(np.pi/2*lift.target_centre)
    h = law.onebody(fields, links, phases)
    current, contact = peierls_derivatives(law, fields, links, phases, direction)
    target_h = law.onebody(target_fields, target_links, rp)
    target_current, target_contact = peierls_derivatives(law, target_fields, target_links, rp, rd)
    expected = dict(endpoint=lift.onebody(h), current=lift.onebody(current),
                    contact=lift.onebody(contact))
    errors = dict(endpoint=law.band.a*defect(target_h-expected['endpoint']),
                  current=law.band.a*defect(target_current-expected['current']),
                  contact=law.band.a*defect(target_contact-expected['contact']),
                  actual_vector_fields=defect(rfields-target_fields),
                  actual_oriented_colour_seams=defect(rlinks-target_links),
                  target_Gram=defect(target_selected['G']-rotation4(rotation)
                                    @ selected['G'] @ rotation4(rotation).T))
    for name, value in errors.items():
        check(value, 'native physical '+name+' covariance')
    # Both adverse branches reach the identical three covariance assertions.
    wrong_fields = np.empty_like(fields)
    wrong_fields[lift.targets] = fields
    wrong_fields = lift.fields(wrong_fields)
    wrong_vector_current, wrong_vector_contact = peierls_derivatives(
        law, wrong_fields, rlinks, rp, rd)
    wrong_vector = dict(endpoint=law.onebody(wrong_fields, rlinks, rp),
                        current=wrong_vector_current, contact=wrong_vector_contact)
    vector_adverse = {name: reject_same_assertion(law.band.a*defect(actual-expected[name]),
                      'native physical '+name+' covariance')
                      for name, actual in wrong_vector.items()}
    # Constant spin restoration without the necessary reversed-axis seam gauge.
    spin = lift.compensator.conj().T
    wrong_seam_links = np.einsum('ij,xajk,kl->xail', spin, ru, spin.conj().T)
    wrong_seam_current, wrong_seam_contact = peierls_derivatives(
        law, target_fields, wrong_seam_links, rp, rd)
    wrong_seam = dict(endpoint=law.onebody(target_fields, wrong_seam_links, rp),
                      current=wrong_seam_current, contact=wrong_seam_contact)
    seam_adverse = {}
    for name, actual in wrong_seam.items():
        seam_error = law.band.a*defect(actual-expected[name])
        seam_adverse[name] = (reject_same_assertion(seam_error,
            'native physical '+name+' covariance') if np.any(lift.cochain)
            else dict(defect=seam_error, rejected=False,
                 reason='this particular seam is already canonical; no adverse witness claimed'))
    # Complete local SU4 orbit, not only the centre or a frozen field marker.
    gauges = np.array([expm(1j*np.einsum('a,aij->ij', .17*rng.normal(size=15), law.generators))
                       for _ in law.sites])
    gf = np.einsum('xij,xmjk,xlk->xmil', gauges, fields, gauges.conj())
    gu = np.einsum('xij,xajk,xalk->xail', gauges, links, gauges[law.target].conj())
    gauged_h = law.onebody(gf, gu, phases)
    expected_gauge = np.einsum('xij,xjyk,ylk->xiyl', gauges,
                    h.reshape(law.volume, 4, law.volume, 4), gauges.conj(), optimize=True).reshape(h.shape)
    gauge_error = law.band.a*defect(gauged_h-expected_gauge)
    check(gauge_error, 'native full local SU4 orbit endpoint covariance')
    orbit_fields, orbit_links, orbit_phases, _ = physical_cubic(
        law, gf, gu, phases, rotation)
    orbit_fields = lift.fields(orbit_fields)
    orbit_links = restore_links(law, lift, orbit_links)
    orbit_current, orbit_contact = peierls_derivatives(
        law, orbit_fields, orbit_links, orbit_phases, rd)
    gauge_current, gauge_contact = peierls_derivatives(law, gf, gu, phases, direction)
    orbit_matrices = dict(endpoint=law.onebody(orbit_fields, orbit_links, orbit_phases),
                          current=orbit_current, contact=orbit_contact)
    orbit_source = dict(endpoint=gauged_h, current=gauge_current, contact=gauge_contact)
    orbit_errors = {name: law.band.a*defect(actual-lift.onebody(orbit_source[name]))
                    for name, actual in orbit_matrices.items()}
    for name, value in orbit_errors.items():
        check(value, 'native full colour orbit physical '+name+' covariance')
    gauge_scalar = rng.normal(size=law.volume)
    gradient = gauge_scalar[:, None]-gauge_scalar[law.target]
    ward_current, ward_contact = peierls_derivatives(law, fields, links, phases, gradient)
    charge = np.repeat(gauge_scalar, 4)
    commutator = (charge[:, None]-charge[None, :])*h
    ward = law.band.a*defect(ward_current-1j*commutator)
    contact_ward = law.band.a*defect(ward_contact+(charge[:, None]-charge[None, :])*commutator)
    check(ward, 'native complete U1 Gauss current')
    check(contact_ward, 'native complete U1 Gauss contact')
    curvature_moves = defect(plaquettes(law, rp)-plaquettes(law, phases))
    gauge_curvature = defect(plaquettes(law, phases+gradient)-plaquettes(law, phases))
    check(gauge_curvature, 'pure gauge compact curvature unchanged')
    reject_same_assertion(curvature_moves, 'physical rotation is pure gauge curvature control')
    # Four actual physical turns return positions, fields, links, phases, raw scalar matter.
    f, u, p = fields, links, phases
    total_targets = np.arange(law.volume)
    for _ in range(4):
        f, u, p, _ = physical_cubic(law, f, u, p, QUARTER_TURN)
        total_targets = site_targets(law, QUARTER_TURN)[total_targets]
    raw_loop = dict(fields=defect(f-fields), colour=defect(u-links), EM=defect(p-phases),
                    matter_permutation=int(np.max(abs(total_targets-np.arange(law.volume)))))
    for name, value in raw_loop.items():
        check(float(value), 'raw scalar physical fourturn '+name)
    return dict(centre=centre, target_centre=lift.target_centre, rotation=rotation,
                dimension=4*law.volume, full_compact_phase_range=(-np.pi, np.pi),
                errors=errors, nontrivial_seam_sites=int(np.count_nonzero(lift.cochain)),
                cochain=lift.cochain, omitted_vector_adverse=vector_adverse,
                omitted_seam_adverse=seam_adverse, full_local_SU4_covariance=gauge_error,
                full_colour_orbit_physical_covariance=orbit_errors,
                U1_Gauss_current=ward, U1_Gauss_contact=contact_ward,
                physical_fixed_position_curvature_change=curvature_moves,
                pure_gauge_curvature_error=gauge_curvature, raw_scalar_fourturn=raw_loop)


def whole_fock_reference_controls(law: BulkQuantumLaw, gram: RealArray, frame: Matrix) -> dict[str, Any]:
    spin = clifford_compensator(QUARTER_TURN, recover_gamma(gram, frame)).conj().T
    loop = np.linalg.matrix_power(spin, 4)
    check(defect(loop+np.eye(4)), 'one fundamental physical endpoint quarterturn fourth minus')
    # The exterior functor is exact for EVERY occupation, not an odd-state mask.
    wedge_errors = []
    for number in range(5):
        occupations = list(itertools.combinations(range(4), number))
        exterior = np.array([[np.linalg.det(loop[np.ix_(left, right)])
                              for right in occupations] for left in occupations])
        wedge_errors.append(defect(exterior-(-1)**number*np.eye(len(occupations))))
    check(max(wedge_errors), 'entire local16 CAR exterior powers of endpoint loop')
    reference_loop = loop.conj()  # bar4; it also has the opposite EM charge.
    check(defect(np.kron(loop, reference_loop)-np.eye(16)), 'fundamental opposite-reference loop plus')
    singlet = np.eye(4).reshape(-1)/2
    singlet_error = defect(np.kron(spin, spin.conj()) @ singlet-singlet)
    check(singlet_error, 'actual fundamental/bar4 reference singlet intertwiner')
    heavy_number = 2*law.volume
    light_number = 2*law.volume+1
    total_sign = (-1)**(heavy_number+light_number+1)
    if total_sign != 1:
        raise AssertionError('Full heavy/light/reference fourturn must be ordinary')
    return dict(fundamental_fourturn_minus=defect(loop+np.eye(4)),
                full_local_CAR_occupations=(0, 1, 2, 3, 4), exterior_power_errors=wedge_errors,
                NH=heavy_number, NL=light_number, heavy_loop_sign=(-1)**heavy_number,
                odd_light_loop_sign=(-1)**light_number, opposite_reference_loop_sign=-1,
                full_Gauss_fourturn_sign=total_sign, reference_singlet=singlet_error,
                whole_fixed_NL_Fock_dimension=str(math.comb(4*law.volume, light_number)),
                proof_not_simulation='Exterior functor identities on whole fixed-number Fock, no occupation cuts')


def restored_gauge_berry_control(law: BulkQuantumLaw, centre: IntArray, frame: Matrix,
                                gram: RealArray) -> dict[str, Any]:
    """Native whole-heavy occupied frame; internal gauge loop with body restoration.

    This is deliberately NOT a physical rotation: no site/vector/link-index action.
    Bare Berry holonomy along an arbitrary gauge path need not be identity.
    """
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    links = law.toron_links(np.pi/2*centre)
    h = law.onebody(fields, links)
    values, vectors = np.linalg.eigh(h)
    occupied = vectors[:, :2*law.volume]
    spin = clifford_compensator(QUARTER_TURN, recover_gamma(gram, frame))
    previous = occupied
    holonomy = np.eye(2*law.volume, dtype=complex)
    maximum_restore = 0.
    for turn in range(1, 5):
        gauge = np.linalg.matrix_power(spin, turn)
        moved = np.einsum('ij,xjn->xin', gauge, occupied.reshape(law.volume, 4, -1))
        restored = np.einsum('ij,xjn->xin', gauge.conj().T, moved).reshape(occupied.shape)
        maximum_restore = max(maximum_restore, defect(restored-occupied))
        left, _, right = np.linalg.svd(restored.conj().T @ previous)
        holonomy = left @ right @ holonomy
        previous = restored
    berry_error = defect(holonomy-np.eye(len(holonomy)))
    check(maximum_restore, 'actual whole native sea gauge restoration')
    check(berry_error, 'restored native sea Berry overlap loop plus')
    return dict(occupied_modes=2*law.volume, native_whole_half_gap=float(values[2*law.volume]
                    - values[2*law.volume-1]), restored_frame_error=maximum_restore,
                restored_Berry_holonomy_plus=berry_error, physical_positions_rotated=False,
                Phi_vector_indices_rotated=False, bare_unrestored_Berry_identity_claimed=False)


def run(*, size: int = 5, spacing: float = 1., seed: int = 222940,
        centres: tuple[tuple[int, int, int], ...] = ((0, 0, 0), (1, 0, 0), (1, 2, 3), (2, 2, 2))) -> dict[str, Any]:
    if size < 3 or not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('Native physical endpoint needs N>=3 and positive finite spacing')
    identities = source_hashes()
    law = BulkQuantumLaw(size, band=RotationBand(**{**BARE, 'a': spacing}))
    group, data = gram_group_controls(law)
    rng = np.random.default_rng(seed)
    rows = []
    rotations = (QUARTER_TURN, np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.int64),
                 np.diag([-1, -1, 1]).astype(np.int64))
    for c in centres:
        centre = tuple(int(v) % 4 for v in c)
        for rotation in rotations:
            rows.append(native_case(law, np.asarray(centre), data[centre]['selected'],
                                    data[centre]['frame'], rotation, rng))
    if not any(row['omitted_seam_adverse']['endpoint']['rejected'] for row in rows):
        raise AssertionError('Selected native physical cases must include a wrong-seam adverse witness')
    origin = data[(0, 0, 0)]
    fock = whole_fock_reference_controls(law, origin['selected']['G'], origin['frame'])
    berry = restored_gauge_berry_control(law, np.zeros(3, dtype=np.int64), origin['frame'],
                                        origin['selected']['G'])
    final_hashes = source_hashes()
    if final_hashes != identities:
        raise AssertionError('Original physical rotation source/proof bindings changed during execution')
    return dict(status='exact_physical_binary_cube_endpoint_mechanism_NOT_localized_mechanical_half_spin',
                plan=PLAN, plan_revision=PLAN_REVISION, template_revision=TEMPLATE_REVISION,
                source_sha256=identities, source_bindings_unchanged=True,
                runtime=dict(python=platform.python_version(), executable=sys.executable,
                             numpy=np.__version__, BLAS_threads={v: os.environ.get(v) for v in
                                 ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}),
                inputs=dict(size=size, spacing=spacing, seed=seed, centres=centres, bare=BARE,
                            original_frame_full10=True, Gamma_from_native_full_Gram=True),
                all64_group=group, native_cases=rows, whole_Fock_reference=fock,
                restored_pure_gauge_Berry_control=berry,
                unchanged_owners=dict(original_energy_selection_retained=True,
                    changed27_selector_energy_executed=False, extra_CAR_or_spin_postulate=False,
                    all60V_fields_all45V_colour_links_full_compact_EM_full_Gauss=True),
                proof_scope=dict(physical_discrete_rotation_orbit_intertwiner=True,
                    rooted_seam_and_binary_cube_cocycle=True, whole_Fock_reference_Gauss_intertwiner=True,
                    quantum_full_ground_or_BO_embedding_numerically_simulated=False,
                    mechanical_Noether_continuum_localized_half_spin_proved=False,
                    campaign_complete=False),
                remaining_debt=[
                    'Control a continuous mechanical rotation generator including field/link momenta, colour boundary flux and reference ownership.',
                    'Prove an SO3 approximation and W^dagger J_micro W -> L+S on a localized mobile full-Gauss charged sector.',
                    'Establish deconfined finite-energy thermodynamic charged states and their reference/boundary removal; discrete endpoint cocycle alone is insufficient.'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=5)
    parser.add_argument('--spacing', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=222940)
    parser.add_argument('--centres', default='000,100,123,222')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    supplied = args.centres.split(',')
    if any(len(c) != 3 or any(v not in '0123' for v in c) for c in supplied):
        parser.error('--centres requires comma-separated Z4 triples, e.g. 000,100,123,222')
    centres: tuple[tuple[int, int, int], ...] = tuple(
        (int(c[0]), int(c[1]), int(c[2])) for c in supplied)
    result = run(size=args.size, spacing=args.spacing, seed=args.seed, centres=centres)
    payload = json.dumps(portable(result), indent=2)
    if args.output:
        args.output.write_text(payload+'\n')
    else:
        print(payload)


if __name__ == '__main__':
    main()
