#!/usr/bin/env python3
"""Finite-volume compact-U1 effective charged energy bound, with native controls.

The exact quadratic-form theorem is in the companion proof. Numerical controls
are one-body/Slater diagnostics, not a quantum EM diagonalization or BO transfer.
The integration owner runs this CLI and final type checks after delivery.
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
from pathlib import Path
from typing import Any, Sequence, cast

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.linalg import expm, svd

from bulk_quantum_selection import BulkQuantumLaw
from original_centre_charge import source_hashes as centre_hashes
from original_charged_principal import (
    OriginalSpatial, charge_controls, charge_form, check, original_frame,
    pair_reference, portable,
)
from original_normal_gap_bound import peierls_derivatives, source_hashes as normal_hashes
from rotation_band import RotationBand
from sea_selection import SELECTED, matrix_unit_basis
from unrestricted_field_minimum import BARE, STIFFNESS, coordinates

Matrix = NDArray[np.complex128]
RealArray = NDArray[np.float64]
Centre = tuple[int, int, int]
ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-continuing-backward-20260930'
PLAN_REVISION = '11e9251a60596401163092e5b39958a6ea55866301fed44e3b9fcd0a0370e31b'
TEMPLATE_REVISION = '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'
CENTRES = cast(tuple[Centre, ...], tuple(itertools.product(range(4), repeat=3)))


def source_hashes() -> dict[str, str]:
    """Bind every loaded local source and every available companion proof."""
    identities = {**normal_hashes(), **centre_hashes()}
    names = {'original_compact_charge_bound.py', 'original_compact_charge_bound.md',
             'bulk_quantum_selection.md'}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename is not None:
            path = Path(filename).resolve()
            if path.parent == ROOT and path.suffix == '.py':
                names.add(path.name)
    for name in tuple(names | set(identities)):
        if name.endswith('.py') and (ROOT/Path(name).with_suffix('.md')).is_file():
            names.add(str(Path(name).with_suffix('.md')))
    identities.update({name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                       for name in sorted(names)})
    return identities


def metric_constants(band: RotationBand, gram: RealArray, volume: int) -> dict[str, float]:
    """Exact formulas evaluated numerically; no outward interval certificate."""
    if band.m <= 0 or band.r < 0 or band.improved or volume <= 0:
        raise ValueError('Need m>0, r>=0, original nearest-axis law and finite V>0')
    if gram.shape != (4, 4) or not np.all(np.isfinite(gram)):
        raise ValueError('Need a finite full4x4 Gram matrix')
    check(float(np.linalg.norm(gram-gram.T)), 'symmetric native Gram')
    if np.linalg.eigvalsh(gram)[0] <= 0:
        raise ValueError('The exact effective theorem requires positive definite Gram')
    b = np.linalg.solve(gram[1:, 1:], gram[1:, 0])
    schur = float(gram[0, 0]-gram[0, 1:]@b)
    if schur <= 0:
        raise ValueError('Spatial Gram Schur complement must be strictly positive')
    mu = band.m*math.sqrt(schur)
    sf = float((3*band.r**2*gram[0, 0]+band.v**2*np.trace(gram[1:, 1:]))/band.a**2)
    return dict(spatial_Schur=schur, mu=mu, SF=sf,
        neutral_metric_upper=volume*sf/(4*mu**2),
        discarded_Wilson_aligned_mu=band.m*math.sqrt(float(gram[0, 0]))
            -abs(band.v)*float(np.sum(abs(gram[0, 1:])))/(band.a*math.sqrt(float(gram[0, 0]))))


def chiral_basis(frame: Matrix, gram: RealArray, volume: int) -> tuple[Matrix, Matrix, Matrix]:
    """Spatial-orthogonal beta, not an isotropic replacement of the native frame."""
    b = np.linalg.solve(gram[1:, 1:], gram[1:, 0])
    residual = frame[0]-np.einsum('i,ijk->jk', b, frame[1:])
    schur = float(gram[0, 0]-gram[0, 1:]@b)
    beta = residual/math.sqrt(schur)
    chirality = SELECTED[0]@SELECTED[1]@SELECTED[2]@SELECTED[3]
    check(float(np.linalg.norm(beta@beta-np.eye(4))), 'Schur beta square')
    check(float(np.linalg.norm(beta@frame[1:]+frame[1:]@beta)), 'beta orthogonal to all spatial fields')
    check(float(np.linalg.norm(chirality@frame+frame@chirality)), 'fixed Gamma5 fullmetric anticommutation')
    values, vectors = np.linalg.eigh(chirality)
    plus = vectors[:, values > 0]
    if plus.shape != (4, 2):
        raise AssertionError('Fixed chiral grading must have two modes of each sign')
    minus = beta@plus
    check(float(np.linalg.norm(plus.conj().T@minus)), 'chiral basis orthogonality')
    return cast(Matrix, np.kron(np.eye(volume), plus)), np.kron(np.eye(volume), minus), beta


def wilson_matrix(law: BulkQuantumLaw, links: Matrix, phases: RealArray) -> Matrix:
    """W=sum(I-Re T_i), using actual centre links and compact U1 phases."""
    result = 3*np.eye(law.volume, dtype=complex)
    for x in range(law.volume):
        for axis in range(3):
            y = law.target[x, axis]
            z = np.trace(links[x, axis])/4
            check(float(np.linalg.norm(links[x, axis]-z*np.eye(4))), 'central native link needed for fixed-metric theorem')
            hop = z*np.exp(1j*phases[x, axis])/2
            result[x, y] -= hop
            result[y, x] -= hop.conjugate()
    return result


def sea_section(h: Matrix, plus: Matrix, minus: Matrix) -> dict[str, Any]:
    """Represent the full2V Slater by its frame and global principal-log phase.

    Im Tr Log Q is the sum of principal arguments of Q's eigenvalues, NOT
    principal arg det Q; accretivity keeps every eigenvalue in the right half-plane.
    """
    q = plus.conj().T@h@minus
    mass = (q+q.conj().T)/2
    if np.linalg.eigvalsh(mass)[0] <= 0:
        raise ValueError('Principal-log section requires strict global accretivity')
    left, singular, right = svd(q, full_matrices=False)
    unitary = left@right
    occupied = (plus-minus@unitary.conj().T)/math.sqrt(2)
    theta = float(np.angle(np.linalg.eigvals(q)).sum())
    return dict(Q=q, M=mass, D=(q-q.conj().T)/(2j), U=unitary,
        singular=singular, right=right, occupied=occupied, theta=theta,
        Slater_phase=complex(np.exp(.5j*theta)))


def polar_derivative(section: dict[str, Any], dq: Matrix) -> Matrix:
    """Exact Sylvester derivative, including occupied/empty degeneracies."""
    u, right, singular = section['U'], section['right'], section['singular']
    rhs = u.conj().T@dq-dq.conj().T@u
    rotated = right@rhs@right.conj().T
    x = right.conj().T@(rotated/(singular[:, None]+singular[None, :]))@right
    return u@x


def section_derivative_controls(h: Matrix, section: dict[str, Any], plus: Matrix,
                                minus: Matrix, current: Matrix, mu: float) -> dict[str, float]:
    dq = plus.conj().T@current@minus
    du = polar_derivative(section, dq)
    occupied, u = section['occupied'], section['U']
    dw = -minus@du.conj().T/math.sqrt(2)
    theta_derivative = float(np.trace(np.linalg.solve(section['Q'], dq)).imag)
    raw_connection = complex(np.trace(occupied.conj().T@dw))
    cancelled = raw_connection+.5j*theta_derivative
    g_polar = float(np.vdot(du, du).real/4)
    values, vectors = np.linalg.eigh(h)
    half = len(values)//2
    transitions = vectors[:, half:].conj().T@current@vectors[:, :half]
    denominator = values[half:, None]-values[None, :half]
    g_spectral = float(np.sum(abs(transitions/denominator)**2))
    hs_upper = float(np.vdot(current, current).real/(8*mu**2))
    check(abs(cancelled), 'globally cancelled determinant Berry connection')
    check(abs(g_polar-g_spectral)/max(1., g_spectral), 'Slater polar versus actual spectral QGT')
    check(max(0., g_spectral-hs_upper)/max(1., hs_upper), 'exact per-direction QGT bound')
    return dict(raw_Berry_connection=raw_connection.imag,
        log_phase_derivative=theta_derivative, cancelled_Berry_residual=abs(cancelled),
        Slater_QGT=g_spectral, polar_QGT=g_polar, HS_gap_upper=hs_upper)


def u1_section_controls(law: BulkQuantumLaw, fields: Matrix, links: Matrix,
                        phases: RealArray, h: Matrix, section: dict[str, Any],
                        plus: Matrix, minus: Matrix, rng: np.random.Generator,
                        mu: float) -> dict[str, Any]:
    gauge = rng.uniform(-math.pi, math.pi, law.volume)
    direction = gauge[:, None]-gauge[law.target]
    current, contact = peierls_derivatives(law, fields, links, phases, direction)
    generator = np.repeat(gauge, 4)
    commutator = (generator[:, None]-generator[None, :])*h
    ward = float(law.band.a*np.linalg.norm(current-1j*commutator))
    contact_ward = float(law.band.a*np.linalg.norm(contact+
        (generator[:, None]-generator[None, :])*commutator))
    check(ward, 'native analytic compactU1 current Ward')
    check(contact_ward, 'native analytic compactU1 contact Ward')
    omitted_contact = float(law.band.a*np.linalg.norm(contact))
    moved_h = law.onebody(fields, links, phases+direction)
    gauge4 = np.exp(1j*generator)
    gauge2 = np.exp(1j*np.repeat(gauge, 2))
    covariance = float(law.band.a*np.linalg.norm(moved_h-gauge4[:, None]*h*gauge4.conj()[None, :]))
    check(covariance, 'fulltorus native U1 endpoint covariance')
    moved = sea_section(moved_h, plus, minus)
    predicted_frame = gauge4[:, None]*section['occupied']*gauge2.conj()[None, :]
    frame_error = float(np.linalg.norm(moved['occupied']-predicted_frame))
    check(frame_error, 'canonical whole Slater gauge frame covariance')
    phase_error = abs(moved['theta']-section['theta'])
    check(phase_error, 'global principal-log phase gauge invariance')
    overlap = (gauge4[:, None]*section['occupied']).conj().T@moved['occupied']
    determinant, lognorm = np.linalg.slogdet(overlap)
    slater_error = abs(np.exp(.5j*(moved['theta']-section['theta']))*determinant
                       -np.exp(-2j*gauge.sum()))
    check(float(slater_error), 'EXACT per-site U1 background2 equivariance')
    check(abs(float(lognorm)), 'normalized whole Slater gauge overlap')
    density = np.trace((section['occupied']@section['occupied'].conj().T)
        .reshape(law.volume, 4, law.volume, 4), axis1=1, axis2=3).diagonal().real
    check(float(np.max(abs(density-2))), 'whole sea density2 per site')
    periodic = phases.copy()
    periodic[0, 0] += 2*math.pi
    periodic_h = law.onebody(fields, links, periodic)
    check(float(law.band.a*np.linalg.norm(periodic_h-h)), 'actual compact phase2pi periodicity')
    periodic_sea = sea_section(periodic_h, plus, minus)
    periodic_phase = abs(periodic_sea['Slater_phase']-section['Slater_phase'])
    check(float(periodic_phase), 'global analytic Slater phase periodicity')
    derivatives = []
    for axis in range(3):
        edge_direction = np.zeros_like(phases)
        edge_direction[0, axis] = 1
        vertex, edge_contact = peierls_derivatives(law, fields, links, phases, edge_direction)
        row = section_derivative_controls(h, section, plus, minus, vertex, mu)
        _, scalar = law.paths[axis]
        f = np.einsum('m,mij->ij', scalar, fields[0])
        edge_hs = float(np.vdot(f, f).real)
        check(abs(float(np.vdot(vertex, vertex).real)-2*edge_hs)*law.band.a**2,
              'native endpoint current HS factor2')
        row.update(axis=axis, native_F_HS_squared=edge_hs,
                   contact_HS_squared=float(np.vdot(edge_contact, edge_contact).real))
        derivatives.append(row)
    return dict(current_aH_Ward=ward, contact_aH_Ward=contact_ward,
        omitted_contact_aH_defect=omitted_contact, gauge_aH_covariance=covariance,
        Slater_frame_covariance=frame_error, global_log_phase_covariance=phase_error,
        exact_background2_Slater_equivariance=float(slater_error),
        per_site_density_error=float(np.max(abs(density-2))),
        compact2pi_Slater_phase_error=float(periodic_phase), analytic_edge_derivatives=derivatives,
        scope='Whole2V Slater exterior identity proves local Gauss equivariance; density alone does not.')


def gauss_controls(model: OriginalSpatial, law: BulkQuantumLaw, fields: Matrix,
                   links: Matrix, phases: RealArray, h: Matrix,
                   rng: np.random.Generator) -> dict[str, Any]:
    directions = rng.normal(size=(law.volume, 15))
    gauge = np.array([expm(.17j*np.einsum('a,aij->ij', d, law.generators)) for d in directions])
    moved_fields = np.einsum('xai,xmij,xbj->xmab', gauge, fields, gauge.conj())
    moved_links = gauge[:, None]@links@gauge[law.target].conj().transpose(0, 1, 3, 2)
    moved_h = law.onebody(moved_fields, moved_links, phases)
    predicted = np.einsum('xai,xiyj,ybj->xayb', gauge,
        h.reshape(law.volume, 4, law.volume, 4), gauge.conj()).reshape(h.shape)
    covariance = float(law.band.a*np.linalg.norm(moved_h-predicted))
    check(covariance, 'actual all15/site SU4 endpoint covariance')
    rows = []
    for q in (np.zeros(3), 2*np.pi*np.array([1, 0, 0])/law.size):
        columns = model.gauge_columns(q)
        form = charge_form(columns, 1., .3)
        if form['rank'] != 15:
            raise AssertionError('Regular fullmetric frame must retain all15 SU4 Gauss columns')
        reference = pair_reference(model, q, form, np.array([1, 0, 0]))
        rows.append(dict(q=q, complete_inhomogeneous_Gauss=charge_controls(columns, form),
                         opposite_static_reference=reference))
    return dict(native_local_SU4_aH_covariance=covariance, full15_reference_rows=rows,
        centre_stabilizer_neutral_character=complex((1j)**(4*law.volume)),
        centre_stabilizer_doped_plus_bar4_character=complex((1j)**(4*law.volume+1-1)),
        reference_U1_charge_plus=-1, reference_U1_charge_minus=1,
        scope='Exact covariance/reference/Gauss algebra controls, not a zero angular energy or BO isometry.')


def native_case(model: OriginalSpatial, law: BulkQuantumLaw, frame: Matrix,
                gram: RealArray, angles: RealArray, phases: RealArray,
                rng: np.random.Generator) -> dict[str, Any]:
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    links = law.toron_links(angles)
    h = law.onebody(fields, links, phases)
    plus, minus, _ = chiral_basis(frame, gram, law.volume)
    constants = metric_constants(law.band, gram, law.volume)
    section = sea_section(h, plus, minus)
    q, mass, d = section['Q'], section['M'], section['D']
    reconstructed = plus@q@minus.conj().T+minus@q.conj().T@plus.conj().T
    check(float(law.band.a*np.linalg.norm(h-reconstructed)), 'fixed chiral native whole matrix')
    wilson = wilson_matrix(law, links, phases)
    expected_mass = math.sqrt(constants['spatial_Schur'])*np.kron(
        law.band.m*np.eye(law.volume)+law.band.r/law.band.a*wilson, np.eye(2))
    mass_error = float(law.band.a*np.linalg.norm(mass-expected_mass))
    check(mass_error, 'exact nonisotropic Schur-beta positive Wilson mass')
    wilson_min = float(np.linalg.eigvalsh(wilson)[0])
    mass_min = float(np.linalg.eigvalsh(mass)[0])
    check(max(0., -wilson_min), 'arbitrary compact phases Wilson positivity')
    check(law.band.a*max(0., constants['mu']-mass_min), 'allphase positive Schur mass')
    values, vectors = np.linalg.eigh(h)
    half = 2*law.volume
    if np.sum(values < 0) != half or np.sum(values > 0) != half:
        raise AssertionError('Actual whole negative/positive occupations must each equal2V')
    check(float(law.band.a*np.max(abs(values+values[::-1]))), 'actual fixed Gamma5 spectral plusminus pairs')
    smallest = float(np.min(abs(values)))
    check(law.band.a*max(0., constants['mu']-smallest), 'noncommuting mass Schur gap')
    spectral_projector = vectors[:, :half]@vectors[:, :half].conj().T
    canonical_projector = section['occupied']@section['occupied'].conj().T
    check(float(np.linalg.norm(spectral_projector-canonical_projector)), 'actual whole occupied canonical Slater projector')
    identities = u1_section_controls(law, fields, links, phases, h, section, plus, minus, rng, constants['mu'])
    colour = gauss_controls(model, law, fields, links, phases, h, rng)
    sum_f = sum(float(np.vdot(np.einsum('m,mij->ij', scalar, frame),
                                  np.einsum('m,mij->ij', scalar, frame)).real)
                for _, scalar in law.paths)
    check(law.band.a**2*abs(sum_f-constants['SF']), 'native fullmetric SF normalization')
    return dict(actual_matrix_dimension=len(h), actual_whole_negative_occupation=half,
        actual_Esea=float(values[:half].sum()), pointwise_particle_cost=float(values[half]),
        pointwise_hole_cost=float(-values[half-1]), actual_onebody_gap=smallest,
        Wilson_minimum=wilson_min, positive_M_minimum=mass_min,
        positive_M_aH_identity_error=mass_error,
        M_D_aH_squared_commutator=float(law.band.a**2*np.linalg.norm(mass@d-d@mass)),
        global_principal_log_phase=section['theta'], naive_principal_arg_det_Q=float(np.angle(np.linalg.det(section['U']))),
        U1_current_Slater_controls=identities, full_SU4_reference_controls=colour,
        arbitrary_compact_phase_range=[float(phases.min()), float(phases.max())])


def run(*, size: int = 5, spacing: float = 1., centres: Sequence[Sequence[int]] | None = None,
        all_centres: bool = False, phase_samples: int = 1, electric: float | None = None,
        magnetic: float | None = None, seed: int = 2221002) -> dict[str, Any]:
    if size < 5 or not np.isfinite(spacing) or not 0 < spacing <= 1 or phase_samples < 1:
        raise ValueError('Need native N>=5,0<a<=1 and at least one arbitrary compact phase sample')
    if all_centres and centres is not None:
        raise ValueError('Choose explicit centres or --all-centres, not both')
    hashes = source_hashes()
    band = RotationBand(**{**BARE, 'a': spacing})
    frames: dict[Centre, tuple[dict[str, Any], Matrix, dict[str, float]]] = {}
    for label in CENTRES:
        selected, frame = original_frame(band, size, np.pi/2*np.asarray(label), True)
        constants = metric_constants(band, np.asarray(selected['G']), size**3)
        frames[label] = selected, frame, constants
    mu_min = min(item[2]['mu'] for item in frames.values())
    metric_max = max(item[2]['neutral_metric_upper'] for item in frames.values())
    chosen_electric = mu_min/(2*metric_max) if electric is None else electric
    chosen_magnetic = .4/spacing if magnetic is None else magnetic
    if not np.isfinite(chosen_electric) or chosen_electric <= 0 or not np.isfinite(chosen_magnetic) or chosen_magnetic <= 0:
        raise ValueError('True interacting quantum compact photons require UE1>0 and K1>0')
    worst_gap = min(CENTRES, key=lambda c: frames[c][2]['mu'])
    adverse_metric = max(CENTRES, key=lambda c: float(np.linalg.norm(
        frames[c][0]['G']-np.diag(np.diag(frames[c][0]['G'])))))
    if centres is None:
        labels = list(CENTRES) if all_centres else list(dict.fromkeys(
            [(0, 0, 0), (1, 0, 0), worst_gap, adverse_metric]))
    else:
        labels = [cast(Centre, tuple(c)) for c in centres]
        if not labels or any(len(c) != 3 or any(not isinstance(n, (int, np.integer)) or not 0 <= n < 4 for n in c) for c in labels):
            raise ValueError('Centre classes must be nonempty triples of integers0..3')
        if len(set(labels)) != len(labels):
            raise ValueError('Duplicate centre classes do not enlarge evidence')
    basis = matrix_unit_basis()
    law = BulkQuantumLaw(size, band=band)
    rng = np.random.default_rng(seed)
    rows = []
    for label in labels:
        selected, frame, constants = frames[label]
        angles = np.pi/2*np.asarray(label)
        model = OriginalSpatial(band, size, basis, coordinates(frame, basis).reshape(-1), angles)
        stationarity = float(band.a*np.linalg.norm(model.gradient))
        check(stationarity, 'actual native full10 field/link stationarity')
        fields = np.tile(frame, (law.volume, 1, 1, 1))
        zero_h = law.onebody(fields, law.toron_links(angles))
        zero_values = np.linalg.eigvalsh(zero_h)
        endpoint_error = float(band.a*np.max(abs(zero_values-np.sort(model.values.reshape(-1)))))
        check(endpoint_error, 'actual native endpoint versus original whole Bloch spectrum')
        controls = [native_case(model, law, frame, selected['G'], angles,
            rng.uniform(-math.pi, math.pi, (law.volume, 3)), rng) for _ in range(phase_samples)]
        rows.append(dict(centre=list(label), native_full10_Gram=selected['G'],
            full10_gradient=selected['gradient_residual'], aH_full_stationarity=stationarity,
            endpoint_aH_Bloch_error=endpoint_error, constants=constants,
            neutral_energy_upper_excess=chosen_electric*constants['neutral_metric_upper'],
            charged_minus_neutral_lower=constants['mu']-chosen_electric*constants['neutral_metric_upper'],
            native_arbitrary_phase_controls=controls))
    if source_hashes() != hashes:
        raise RuntimeError('Native source/proof dependencies changed during execution')
    all_metric_rows = [dict(centre=list(c), **frames[c][2]) for c in CENTRES]
    return cast(dict[str, Any], portable(dict(source_sha256=hashes, source_bindings_unchanged=True,
        governing_plan=PLAN, assignment_plan_revision=PLAN_REVISION,
        template_revision=TEMPLATE_REVISION,
        native_runtime=dict(executable=sys.executable, python=sys.version,
            platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__,
            source_directory=str(ROOT), entrypoint='CLI' if __name__ == '__main__' else 'Python API',
            BLAS_threads={name: os.environ.get(name) for name in
                ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS')}),
        inputs=dict(size=size, volume=size**3, spacing=spacing, native_stiffness=STIFFNESS/spacing,
            m=band.m, r=band.r, v=band.v, UE1=chosen_electric, K1=chosen_magnetic,
            electric_choice='min_all64_mu/(2 max_all64_neutral_metric_upper)' if electric is None else 'explicit user coefficient',
            phase_samples=phase_samples, seed=seed, centres=[list(c) for c in labels]),
        all64_exact_formula_floating_controls=all_metric_rows,
        all64_mu_min=mu_min, all64_neutral_metric_upper_max=metric_max,
        all64_common_offset_mass_lower=mu_min-chosen_electric*metric_max,
        worst_Schur_gap_centre=list(worst_gap), adverse_fullmetric_centre=list(adverse_metric),
        rows=rows, all_checks_pass=True, quantum_ground_computed=False,
        full_rotor_or_Fock_cutoff_used=False, original_BO_transfer_proved=False,
        interacting_effective_operator='UE1 sum pA^2+K1 sum(1-cos curl A)+dGamma(native fullmetric HL(A)); exact fulltorus physical Gauss domains',
        theorem_scope='Exact finite-volume effective-form theorem for every SPD native Gram and every compact A; computed formulas are floating diagnostics, not interval certificates.',
        microscopic_corollary='Conditional on parent full-Gauss BO effective-operator transfer, including common scalar/centre offsets and controlled neutral/angular remainder.',
        tooling=dict(LSP='Native Pyright1.1.414 stdio documentSymbol and original_frame definition exercised before editing',
            AST='Mounted ast-grep positive Python function query exercised',
            gap='xd://lsp device absent; parent owns final native types/CLI'),
        campaign_complete=False,
        limits=['No numerical full quantum EM diagonalization or quantum ground was performed.',
            'No thermodynamic deconfinement, mechanical spin, isolated/mobile charge or QED closure.',
            'Finite-volume metric cost is extensive and the sufficient electric coefficient is not uniform in N or cutoff.',
            'Full SU4 covariance and static reference domain lifting do not imply zero colour angular/Born-Huang energy.',
            'A bare one-body gap alone is not a charged-minus-neutral ground-energy theorem.'])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=5)
    parser.add_argument('--spacing', '--a', type=float, default=1.)
    parser.add_argument('--centre', type=int, nargs=3, action='append')
    parser.add_argument('--all-centres', action='store_true')
    parser.add_argument('--phase-samples', type=int, default=1)
    parser.add_argument('--electric', type=float)
    parser.add_argument('--magnetic', type=float)
    parser.add_argument('--seed', type=int, default=2221002)
    parser.add_argument('--output', type=Path, default=ROOT/'original_compact_charge_bound_receipt.json')
    args = parser.parse_args()
    result = run(size=args.size, spacing=args.spacing, centres=args.centre,
        all_centres=args.all_centres, phase_samples=args.phase_samples,
        electric=args.electric, magnetic=args.magnetic, seed=args.seed)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output), all_checks_pass=result['all_checks_pass'],
        native_classes=len(result['rows']), all64_common_offset_mass_lower=result['all64_common_offset_mass_lower'],
        quantum_ground_computed=False, original_BO_transfer_proved=False)))


if __name__ == '__main__':
    main()
