#!/usr/bin/env python3
"""Original Cartesian field angular transfer, not full localized particle spin."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Any, cast

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import null_space

from bulk_quantum_selection import BulkQuantumLaw
from original_charged_principal import OriginalSpatial, original_frame, portable
from original_physical_rotation_lift import (
    QUARTER_TURN, check, clifford_compensator, make_lift, proper_cube_rotations,
    recover_gamma, seam_cochain,
)
from original_slow_kinetic_bo import source_hashes as bo_hashes
from rotation_band import RotationBand
from unrestricted_field_minimum import BARE, coordinates

Matrix = NDArray[np.complex128]
RealArray = NDArray[np.float64]
ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-continuing-backward-20260930'
ASSIGNMENT = '54e3034987e0918a06b1734de362727e1c0aeff914790ef8f21e68b6c6024810'
TEMPLATE = '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'
AXES = ((2, 3), (3, 1), (1, 2))
MASS = np.r_[np.ones(60), np.full(45, 1/.6)]


def source_hashes() -> dict[str, str]:
    return {**bo_hashes(), **{name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('original_intrinsic_angular_transfer.py', 'original_intrinsic_angular_transfer.md',
                     'original_physical_rotation_lift.py', 'original_physical_rotation_lift.md')}}


def field_rotation_tangents(frame: Matrix, basis: Matrix) -> Matrix:
    """Actual Phi-vector-index infinitesimals; no spin law is assigned to matter."""
    vectors = []
    for j, k in AXES:
        change = np.zeros_like(frame)
        change[j], change[k] = -frame[k], frame[j]
        vectors.append(np.r_[coordinates(change, basis).reshape(-1), np.zeros(45)])
    return np.asarray(vectors, dtype=np.complex128).T


def angular_connection(model: OriginalSpatial, q: RealArray, vectors: Matrix,
                       mass: RealArray = MASS) -> Matrix:
    gauge = model.gauge_columns(q)
    return np.linalg.solve(gauge.conj().T@(mass[:, None]*gauge),
                           gauge.conj().T@(mass[:, None]*vectors)).astype(np.complex128)


def shared_chiral_bases(gamma: Matrix) -> tuple[Matrix, Matrix]:
    # One common fibre convention for EVERY source/target. Fresh diagonalization
    # of a degenerate chiral eigenspace can arbitrarily change a Slater phase.
    values, vectors = np.linalg.eigh(gamma[0]@gamma[1]@gamma[2]@gamma[3])
    return vectors[:, values > 0].astype(np.complex128), vectors[:, values < 0].astype(np.complex128)


def positive_whole_frame(h: Matrix, bases: tuple[Matrix, Matrix]) -> tuple[Matrix, dict[str, float]]:
    cp = np.kron(np.eye(len(h)//4), bases[0])
    cm = np.kron(np.eye(len(h)//4), bases[1])
    left, singular, right = np.linalg.svd(cp.conj().T@h@cm)
    polar = left@right
    occupied = (cp+cm@polar.conj().T)/np.sqrt(2)
    values, vectors = np.linalg.eigh(h)
    cut = len(h)//2
    if values[cut-1] >= 0 or values[cut] <= 0 or singular[-1] <= 0:
        raise AssertionError('native whole heavy cluster must have a strict half-filled gap')
    actual = vectors[:, cut:]@vectors[:, cut:].conj().T
    projector_error = float(np.linalg.norm(occupied@occupied.conj().T-actual))
    check(projector_error, 'canonical frame is actual whole positive heavy cluster')
    return occupied.astype(np.complex128), dict(whole_cluster_projector_error=projector_error,
        smallest_chiral_singular=float(singular[-1]), whole_heavy_occupation=cut)


def native_angular_case(law: BulkQuantumLaw, centre: tuple[int, ...],
                        selected: dict[str, Any], frame: Matrix) -> dict[str, Any]:
    gram = np.asarray(selected['G'], dtype=np.float64)
    gamma = recover_gamma(gram, frame)
    spins = np.asarray([-1j/2*gamma[j]@gamma[k] for j, k in AXES], dtype=np.complex128)
    coefficients = 2*np.einsum('aij,kji->ak', law.basis/np.sqrt(8), spins).real
    vectors = field_rotation_tangents(frame, law.basis)
    angles = np.pi/2*np.asarray(centre)
    model = OriginalSpatial(law.band, law.size, law.basis, coordinates(frame, law.basis).reshape(-1), angles)
    # The connection is determined from the physical rotation tangent and all
    # fifteen Gauss generators before any comparison with Clifford bivectors.
    gs = float(np.trace(gram[1:, 1:])/3)
    isotropic = np.diag([gram[0, 0], gs, gs, gs])
    isometric_error = float(np.linalg.norm(gram-isotropic))
    analytic_errors, euclidean_defects, no_flux_defects = [], [], []
    generic_connections = []
    for q in 2*np.pi*law.sites/law.size:
        connection = angular_connection(model, q, vectors)
        generic_connections.append(float(np.linalg.norm(connection)))
        if centre in ((0, 0, 0), (2, 2, 2)):
            d2 = float(np.sum(np.abs(1-np.exp(1j*q))**2))
            expected = coefficients*gs/(gs+d2/.6)
            error = float(np.linalg.norm(connection-expected))
            check(error, 'physical angular connection with complete colour-link mass')
            analytic_errors.append(error)
            wrong = angular_connection(model, q, vectors, np.ones(105))
            euclidean_defects.append(float(np.linalg.norm(wrong-expected)))
            no_flux_defects.append(float(np.linalg.norm(coefficients-expected)))
    q = 2*np.pi*np.array([1., 0., 0.])/law.size
    gauge = model.gauge_columns(q)
    horizontal = null_space((np.sqrt(MASS)[:, None]*gauge).conj().T)/np.sqrt(MASS)[:, None]
    cross = float(np.linalg.norm(gauge.conj().T@(MASS[:, None]*horizontal)))
    check(cross, 'mechanical normal colour connection vanishes at the base point')
    mass_min = float(np.linalg.eigvalsh(model.D.T@model.D)[0])
    if mass_min <= 0:
        raise AssertionError('all fifteen field Gauss masses must be positive')
    # Independent actual whole sea, including its nonzero colour means. Only
    # bivector angular charges have zero means; no mean/variance is discarded.
    h = law.onebody(np.tile(frame, (law.volume, 1, 1, 1)), law.toron_links(angles))
    values, modes = np.linalg.eigh(h)
    occupied = modes[:, 2*law.volume:]
    projector = occupied@occupied.conj().T
    bivector_means = []
    for j, k in itertools.combinations(range(4), 2):
        generator = -1j/2*gamma[j]@gamma[k]
        density = projector[:4, :4]
        bivector_means.append(float(np.trace(density@generator).real))
    check(max(abs(v) for v in bivector_means), 'whole-heavy bivector angular means')
    row: dict[str, Any] = dict(centre=list(centre), native_full10_Gram=gram,
        source_stationarity=law.band.a*float(np.linalg.norm(model.gradient)),
        source_isometric_Gram_error=isometric_error, all15_field_Gauss_mass_min=mass_min,
        mass_horizontal_dimension=horizontal.shape[1], base_point_normal_colour_connection=cross,
        actual_whole_heavy_gap=float(values[2*law.volume]-values[2*law.volume-1]),
        heavy_bivector_angular_means=bivector_means, all_momenta_checked=law.volume,
        general_full15_angular_connection_norm_range=[min(generic_connections), max(generic_connections)])
    if analytic_errors:
        check(isometric_error, 'native invariant centre has isometric Gram')
        check(float(np.linalg.norm(model.D@coefficients-vectors[:60])), 'Cartesian rotation is derived colour bivector tangent')
        if max(euclidean_defects) <= 1e-4 or max(no_flux_defects) <= 1e-4:
            raise AssertionError('incorrect metric and omitted flux must be genuine adverse witnesses')
        casimir = float(np.linalg.norm(np.einsum('aij,ajk->ik', spins, spins)-.75*np.eye(4)))
        check(casimir, 'derived rest doublet spin Casimir')
        mu = law.band.m*np.sqrt(gram[0, 0])
        speed = law.band.v*np.sqrt(gs)
        wilson = law.band.r*np.sqrt(gram[0, 0])
        row.update(isometric_angular_connection_error=max(analytic_errors),
            wrong_Euclidean_mass_defect=max(euclidean_defects), omitted_colour_link_flux_defect=max(no_flux_defects),
            screening_length_in_cells=1/np.sqrt(.6*gs), derived_Casimir_error=casimir,
            source_rest_mass=mu, source_matter_speed=speed,
            same_current_free_Foldy_gyro=2*speed**2/(speed**2+mu*wilson*law.band.a))
    return row


def heavy_vacuum_character_controls(law: BulkQuantumLaw,
                                    data: dict[tuple[int, ...], tuple[dict[str, Any], Matrix]],
                                    bases: tuple[Matrix, Matrix]) -> dict[str, Any]:
    errors, parities = [], set()
    for centre, (selected, frame) in data.items():
        gamma = recover_gamma(selected['G'], frame)
        for rotation in proper_cube_rotations():
            spin = clifford_compensator(rotation, gamma).conj().T
            cochain = seam_cochain(law, np.asarray(centre), rotation)
            parities.add(int(cochain.sum()) % 2)
            # det(site permutation)^2=1. This is the full occupied Slater-frame
            # transformation, NOT an assertion of two particles at every site.
            character = (-1)**int(cochain.sum())*np.linalg.det(bases[0].conj().T@spin@bases[0])**law.volume
            errors.append(abs(character-1))
    check(max(errors), 'whole-heavy point character with common chiral convention')
    if parities != {0}:
        raise AssertionError('rooted three-dimensional centre seams have even total exponent')
    rows = []
    for centre in ((0, 0, 0), (1, 0, 0), (1, 2, 3), (2, 2, 2)):
        selected, frame = data[centre]
        target = tuple((QUARTER_TURN@np.asarray(centre)) % 4)
        _, target_frame = data[target]
        source_h = law.onebody(np.tile(frame, (law.volume, 1, 1, 1)), law.toron_links(np.pi/2*np.asarray(centre)))
        target_h = law.onebody(np.tile(target_frame, (law.volume, 1, 1, 1)), law.toron_links(np.pi/2*np.asarray(target)))
        source, first = positive_whole_frame(source_h, bases)
        destination, second = positive_whole_frame(target_h, bases)
        lift = make_lift(law, np.asarray(centre), QUARTER_TURN, selected['G'], frame)
        inverse = np.argsort(lift.targets)
        transported = np.einsum('xij,xjk->xik', lift.local, source.reshape(law.volume, 4, -1)[inverse]).reshape(4*law.volume, -1)
        overlap = destination.conj().T@transported
        phase, logabs = np.linalg.slogdet(overlap)
        check(abs(phase-1), 'actual whole-heavy Slater physical rotation character')
        check(abs(float(logabs)), 'actual whole-heavy Slater overlap has unit modulus')
        rows.append(dict(centre=list(centre), target_centre=list(target), source=first, target=second,
                         whole_Slater_phase_error=abs(phase-1), whole_Slater_log_modulus=float(logabs)))
    return dict(classes=len(data), rotations=24, character_error=max(errors), seam_sum_parities=sorted(parities),
                native_whole_positive_Slater_rows=rows, common_chiral_fibre_convention=True,
                auxiliary_trivial_stabilizer_character='analytic Kato covariance plus simple well/positive quadratic trial overlap; not quantum-vacuum numerics')


def constant_connection_counterexample() -> dict[str, Any]:
    rows = []
    connection, spin = .7, .5
    for value in (1e8, 1e16, 1e24):
        chi, omega = value**(-1/2), value**(3/8)
        variance = chi/(2*omega)
        distance = np.sqrt(-2*np.expm1(-connection**2*spin**2*variance/2))
        if not distance < 1e-3:
            raise AssertionError('normal phase trial must be norm-close for this adverse probe')
        rows.append(dict(Lambda=value, exact_ground_norm_distance=distance,
                         bare_image_momentum=0., exact_ground_momentum=connection*spin))
    return dict(operator='chi(p-.7 S)^2/2+Lambda^(5/4)x^2/2 on full R; S=.5',
                rows=rows, implication='norm closeness alone does not transfer unbounded momentum',
                actual_source_repair='mass-horizontal normal colour connection is zero at the base point; graded heavy/normal residuals and momentum resolvents required')


def run(*, size: int = 5, spacing: float = 1.) -> dict[str, Any]:
    if size < 5 or not np.isfinite(spacing) or not 0 < spacing <= 1:
        raise ValueError('Native N>=5 and finite0<a<=1 required')
    identities = source_hashes()
    band = RotationBand(**{**BARE, 'a': spacing})
    law = BulkQuantumLaw(size, band=band)
    data: dict[tuple[int, ...], tuple[dict[str, Any], Matrix]] = {}
    for centre in itertools.product(range(4), repeat=3):
        selected, frame = original_frame(band, size, np.pi/2*np.asarray(centre), True)
        data[centre] = selected, frame
    anchor, anchor_frame = data[(0, 0, 0)]
    bases = shared_chiral_bases(recover_gamma(anchor['G'], anchor_frame))
    rows = [native_angular_case(law, centre, *data[centre])
            for centre in ((0, 0, 0), (1, 0, 0), (1, 2, 3), (2, 2, 2))]
    character = heavy_vacuum_character_controls(law, data, bases)
    counterexample = constant_connection_counterexample()
    if source_hashes() != identities:
        raise RuntimeError('Angular source/proof dependencies changed during execution')
    return cast(dict[str, Any], portable(dict(governing_plan=PLAN, assignment_revision=ASSIGNMENT,
        template_revision=TEMPLATE, source_sha256=identities, source_bindings_unchanged=True,
        inputs=dict(size=size, spacing=spacing, original_whole_heavy=2*law.volume,
                    full_field_coordinates=60*law.volume, full_SU4_Gauss_generators=15*law.volume),
        Cartesian_angular_rows=rows, whole_heavy_vacuum_character=character,
        unbounded_momentum_counterexample=counterexample, all_checks_pass=True,
        analytic_statement='smoothed full-physical low-energy Cartesian intrinsic-angular compression, fixeda/existence-largefiniteN, graded momentum proof in companion; O(Lambda^-3/40)',
        quantum_normal_vacuum_computed=False, native_N_globality_certified=False,
        full_localized_mechanical_half_spin_proved=False,
        limits=['Intrinsic Cartesian Phi angular component, not total microscopic continuous Noether J.',
                'No periodic quantum photon ground, reference removal, deconfinement, localized charged pole or full QED closure.',
                'Native diagnostics do not calculate infinite-dimensional auxiliary well states.'], campaign_complete=False)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=5)
    parser.add_argument('--spacing', type=float, default=1.)
    parser.add_argument('--output', type=Path, default=ROOT/'original_intrinsic_angular_transfer_receipt.json')
    args = parser.parse_args()
    result = run(size=args.size, spacing=args.spacing)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), all_checks_pass=result['all_checks_pass'],
                         full_localized_mechanical_half_spin_proved=False)))


if __name__ == '__main__':
    main()
