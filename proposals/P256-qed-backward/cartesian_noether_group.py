#!/usr/bin/env python3
"""Cartesian SO3 field orbit, primitive reference clamp, and full source words."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'

import numpy as np
from scipy import sparse
from scipy.linalg import eigh, expm
from scipy.spatial.transform import Rotation

from bosonic_original_law_join import NativeOriginalModel, reference_sector_embedding
from bulk_quantum_selection import BulkQuantumLaw
from local_orbital_stability import scalar_paths
from moving_physical_noether import physical_generators, source19_boundary_word, source19_nested_casimir
from noncompact_native_current import accept, adverse
from noncompact_physical_rotation_join import physical_spin_intertwiner
from original_charged_principal import original_frame
from rotation_band import RotationBand
from sea_selection import matrix_unit_basis

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-synthesis-continuation-20261006'
EPS = np.zeros((3, 3, 3), int)
for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    EPS[i, j, k] = 1
    EPS[j, i, k] = -1


def cartesian_rotations() -> np.ndarray:
    """Exact second-moment Haar rule on bosonic constant and R_ij functions."""
    result = []
    for order in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            rotation = np.eye(3)[list(order)]*np.asarray(signs)[:, None]
            if np.linalg.det(rotation) > 0:
                result.append(rotation)
    return np.asarray(result)


def integer_orientation() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """L_i=-i physical Cartesian vector Lie derivative, not a spin postulate."""
    rotations = cartesian_rotations()
    functions = np.column_stack((np.ones(len(rotations)), math.sqrt(3)*rotations.reshape(-1, 9)))
    accept(np.linalg.norm(functions.T@functions/len(rotations)-np.eye(10)),
           'integer Cartesian SO3 Haar metric')
    generators = np.zeros((3, 10, 10), complex)
    # Phi_j -> -epsilon_ijk Phi_k, so -i*d Phi_j = +i*epsilon_ijk Phi_k.
    for i, j, k, a in itertools.product(range(3), repeat=4):
        generators[i, 1+3*k+a, 1+3*j+a] = 1j*EPS[i, j, k]
    return rotations, functions, generators


def physical_orbit(rotation: np.ndarray, frame: np.ndarray,
                   color: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Actual Cartesian Phi and parallel transported source/reference frames.

    color=C is the body-to-source Wilson transport (Transport's U_0x).
    Holding C fixed is covariant here because Phi_x=Cdagger Phi_body C.
    Unrelated endpoints instead require their transported C_R, not a frozen C.
    """
    color = np.eye(4, dtype=complex) if color is None else color
    body = frame.copy()
    body[1:] = np.einsum('ij,jab->iab', rotation, frame[1:])
    lift = physical_spin_intertwiner(frame, rotation).conj().T
    accept(np.linalg.norm(np.asarray([lift@f@lift.conj().T for f in frame])-body),
           'actual Cartesian Phi orbit and physical lift')
    return dict(body=body, source=color.conj().T@body@color,
                reference=body, lift=lift, color=color)


def source_M(rotation: np.ndarray, frame: np.ndarray, quartet: np.ndarray,
             color: np.ndarray | None = None) -> np.ndarray:
    """Raw fixed quartet/2, allRef4; color is left U_0x, not input whitening."""
    law = BulkQuantumLaw(3, band=RotationBand(a=1., m=.2, r=.12, v=.3, improved=False))
    spins = physical_generators(law, frame)
    m0 = (quartet[0]*np.eye(4)+2j*np.einsum('a,aij->ij', quartet[1:], spins))/2
    pose = physical_orbit(rotation, frame, color)
    u, c = pose['lift'], pose['color']
    return c.conj().T@u@m0@u.conj().T


def source_matrix(rotation: np.ndarray, frame: np.ndarray, seed_matrix: np.ndarray,
                  color: np.ndarray | None = None, branch: int = 1) -> np.ndarray:
    """Full16 matrix seed, no quartet projection. Plus seed=C*M_actual at I."""
    pose = physical_orbit(rotation, frame, color)
    c, u = pose['color'], pose['lift']
    transported = u@np.asarray(seed_matrix).reshape(4, 4)@u.conj().T
    if branch == 1:
        return c.conj().T@transported
    if branch == -1:
        return transported@c
    raise ValueError('Charged source branch must be +/-1')


def seed_coefficients(group: dict[str, Any], actual_matrix: np.ndarray) -> np.ndarray:
    """Identity source matrix -> coefficients E_ab, row-major a*4+b."""
    seed = (group['color']@actual_matrix if group['branch'] == 1
            else actual_matrix@group['color'].conj().T)
    return np.asarray(seed).reshape(16)


def source_embedding(group: dict[str, Any], rotation: np.ndarray) -> np.ndarray:
    """Evaluate actual primitive Pauli source basis at arbitrary SO3 orientation."""
    matrices = np.asarray([source_matrix(rotation, group['frame'], e, group['color'], group['branch'])
                           for e in np.eye(16, dtype=complex).reshape(16, 4, 4)])
    return group['creator_vectors']@matrices.reshape(16, 16).T


def source_word(group: dict[str, Any], actual_matrix: np.ndarray,
                axes: tuple[int, ...]) -> np.ndarray:
    """Quantum SOURCE word at the24 exact poses. Second words are literal jets."""
    coefficients = seed_coefficients(group, actual_matrix)
    if len(axes) == 2:
        return group['literal_second_values'][axes]@coefficients
    for axis in reversed(axes):
        coefficients = group['K'][axis]@coefficients
    return group['points']@coefficients


def _primitive_units(model: Any, destination: tuple[int, ...],
                     origin: tuple[int, ...]) -> np.ndarray:
    return np.asarray([[model.tree_bilinear(i, j) for j in origin] for i in destination])


def _source_operator(units: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.einsum('ij,ijab->ab', matrix, units)


def _function_embedding(functions: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.einsum('pa,pcm->acm', functions.conj(), values)/len(functions)


def cartesian_source_group(frame: np.ndarray, color: np.ndarray | None = None,
                           branch: int = 1, model: Any | None = None) -> dict[str, Any]:
    """Derive source-invariant quantum L, actual primitive Ref, and K=L-Ref.

    The sixteen matrix-valued source functions are U E_ab Udagger, hence
    bosonic l=0 or1. A 24-pose exact Haar rule extracts their quantum embedding.
    All derivatives/ref actions are executed on actual edge-Pauli states.
    """
    if branch not in (-1, 1):
        raise ValueError('Physical charged/anticharged branch must be +/-1')
    resolved_model: NativeOriginalModel = NativeOriginalModel() if model is None else model
    color = np.eye(4, dtype=complex) if color is None else color
    law = BulkQuantumLaw(3, band=RotationBand(a=1., m=.2, r=.12, v=.3, improved=False))
    rest_spins = physical_generators(law, frame)
    rotations, functions, bosonic = integer_orientation()
    occupied = resolved_model.reference_modes if branch == 1 else resolved_model.bulk_modes
    neutral = resolved_model.state(occupied)
    destination, origin = ((resolved_model.bulk_modes, resolved_model.reference_modes) if branch == 1
                           else (resolved_model.reference_modes, resolved_model.bulk_modes))
    units = _primitive_units(resolved_model, destination, origin)
    creators = np.column_stack([unit@neutral for unit in units.reshape(16, resolved_model.dimension, resolved_model.dimension)])
    base_units = np.eye(16, dtype=complex).reshape(16, 4, 4)
    values, lie_values, ref_values, ref_fields, coefficient_values = [], [], [], [], []
    vector_errors, covariance_errors = [], []
    for rotation in rotations:
        pose = physical_orbit(rotation, frame, color)
        u = pose['lift']
        sref = np.asarray([u@s@u.conj().T for s in rest_spins])
        ssource = color.conj().T@sref@color
        solved = physical_generators(law, pose['body'])
        vector_errors.append(accept(np.linalg.norm(solved-sref), 'Phi-derived pose spin field'))
        covariance_errors.append(accept(np.linalg.norm(sref-np.einsum('ia,ajk->ijk', rotation, rest_spins)),
                                        'literal Cartesian reference spin covariance'))
        matrices = np.asarray([color.conj().T@u@e@u.conj().T if branch == 1
                               else u@e@u.conj().T@color for e in base_units])
        psi = creators@matrices.reshape(16, 16).T
        derivatives = []
        refs = []
        trefs = np.asarray([resolved_model.encode_even(s, resolved_model.reference_modes) for s in sref])
        for i in range(3):
            ds = (ssource[i]@matrices-matrices@sref[i] if branch == 1
                  else sref[i]@matrices-matrices@ssource[i])
            derivatives.append(creators@ds.reshape(16, 16).T)
            refs.append(trefs[i]@psi)
        values.append(psi); lie_values.append(derivatives); ref_values.append(refs)
        ref_fields.append(trefs); coefficient_values.append(matrices)
    values = np.asarray(values)
    lie_values = np.asarray(lie_values)
    ref_values = np.asarray(ref_values)
    ref_fields = np.asarray(ref_fields)
    coefficients = _function_embedding(functions, values)
    embedding = coefficients.reshape(10*resolved_model.dimension, 16)
    metric_error = accept(np.linalg.norm(embedding.conj().T@embedding-np.eye(16)),
                          'actual primitive fullRef4 source invariant metric')
    reconstruction = accept(np.linalg.norm(np.einsum('pa,acm->pcm', functions, coefficients)-values),
                            'bosonic integer Wigner l<=1 actual source functions')
    L, reference, K = [], [], []
    field_derivatives = []
    ref_embeddings = []
    for i in range(3):
        derivative = np.einsum('ab,bcm->acm', bosonic[i], coefficients)
        derivative_values = np.einsum('pa,acm->pcm', functions, derivative)
        accept(np.linalg.norm(derivative_values-lie_values[:, i]), 'actual Phi/vector Lie source action')
        ref_embedding = _function_embedding(functions, ref_values[:, i])
        l = embedding.conj().T@derivative.reshape(-1, 16)
        r = embedding.conj().T@ref_embedding.reshape(-1, 16)
        k = l-r
        accept(np.linalg.norm(embedding@k-(derivative-ref_embedding).reshape(-1, 16)),
               'exact physical clamped source invariant span')
        L.append(l); reference.append(r); K.append(k)
        field_derivatives.append(derivative_values)
        ref_embeddings.append(ref_embedding)
    L, reference, K = np.asarray(L), np.asarray(reference), np.asarray(K)
    field_derivatives = np.asarray(field_derivatives).transpose(1, 0, 2, 3)
    second_values = []
    cross_values = {}
    curvature_wrong = []
    # Execute the TRUE differential second word and the derivative of the Ref
    # coefficient. This is not an assertion on squared first compressions only.
    for i in range(3):
        for j in range(3):
            lj_coeff = np.einsum('ab,bcm->acm', bosonic[j], coefficients)
            lij = np.einsum('pa,ab,bcm->pcm', functions, bosonic[i], lj_coeff)
            derivative_ref = sum(1j*EPS[i, j, k]*ref_fields[:, k] for k in range(3))
            applied = (lij-np.einsum('pab,pbm->pam', derivative_ref, values)
                       -np.einsum('pab,pbm->pam', ref_fields[:, j], field_derivatives[:, i])
                       -np.einsum('pab,pbm->pam', ref_fields[:, i], field_derivatives[:, j])
                       +np.einsum('pab,pbc,pcm->pam', ref_fields[:, i], ref_fields[:, j], values, optimize=True))
            cross_values[i, j] = applied
            if i == j:
                second_values.append(applied)
                accept(np.linalg.norm(applied-.25*values), 'true physical second SOURCE axis word', 2e-8)
            elif EPS[i, j].any():
                k = int(np.flatnonzero(EPS[i, j])[0])
                missing = applied+np.einsum('pab,pbm->pam', derivative_ref, values)
                curvature_wrong.append(float(np.linalg.norm(missing-applied)))
    second_error = accept(np.linalg.norm(sum(second_values)-.75*values),
                          'true nested physical SOURCE Casimir', 2e-8)
    commutator_errors = []
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        expected = 1j*(field_derivatives[:, k]-ref_values[:, k])
        commutator_errors.append(accept(np.linalg.norm(cross_values[i, j]-cross_values[j, i]-expected),
                                        'actual multiaxis field-Ref clamped Lie algebra', 2e-8))
        accept(np.linalg.norm(L[i]@reference[j]-reference[j]@L[i]-1j*reference[k]),
               'derived fullGauss reference covariance')
        accept(np.linalg.norm(reference[i]@reference[j]-reference[j]@reference[i]-1j*reference[k]),
               'primitive reference SU2 subalgebra')
    fake_internal = []
    for p, rotation in enumerate(rotations):
        source_generator = (color.conj().T@rest_spins[0]@color if branch == 1 else rest_spins[0])
        fake = resolved_model.encode_even(source_generator, destination)
        fake_internal.append(float(np.linalg.norm(fake@values[p]-(lie_values[p, 0]-ref_values[p, 0]))))
    # Wrong right-action is checked directly on the true source field, not a
    # dimension/symbol test. Integer L alone has a different Casimir/holonomy.
    wrong_right = adverse(np.linalg.norm(sum(np.einsum('pa,ab,bcm->pcm', functions, bosonic[i]@bosonic[i], coefficients)
                                            for i in range(3))-.75*values),
                          'true nested physical SOURCE Casimir')
    wrong_curvature = adverse(max(curvature_wrong), 'actual multiaxis field-Ref clamped Lie algebra')
    wrong_static = adverse(max(fake_internal), 'actual Phi/vector Lie source action')
    # A frozen reference has zero [L,Sref] coefficient derivative and cannot
    # obey [K,K]=i epsilon K: the uncancelled term is 2i epsilon Sref.
    fixed = np.asarray([resolved_model.encode_even(s, resolved_model.reference_modes) for s in rest_spins])
    def frozen_word(i: int, j: int) -> np.ndarray:
        lj = np.einsum('ab,bcm->acm', bosonic[j], coefficients)
        lij = np.einsum('pa,ab,bcm->pcm', functions, bosonic[i], lj)
        return (lij-fixed[j]@field_derivatives[:, i]-fixed[i]@field_derivatives[:, j]
                +fixed[i]@fixed[j]@values)
    wrong_fixed_ref = adverse(np.linalg.norm(frozen_word(0, 1)-frozen_word(1, 0)-
                                            1j*(field_derivatives[:, 2]-fixed[2]@values)),
                              'actual multiaxis field-Ref clamped Lie algebra')
    return dict(L=L, reference=reference, K=K, embedding=embedding, rotations=rotations,
        points=values, color=color, rest_spins=rest_spins, branch=branch, frame=frame,
        creator_vectors=creators, literal_second_values=cross_values,
        primitive_model=resolved_model, neutral=neutral, coefficient_values=np.asarray(coefficient_values),
        controls=dict(metric_error=metric_error, integer_Wigner_reconstruction_error=reconstruction,
            literal_vector_tangent_error=max(vector_errors), reference_covariance_error=max(covariance_errors),
            true_second_SOURCE_Casimir_error=second_error, multiaxis_commutator_error=max(commutator_errors),
            missing_reference_rightaction_rejected=wrong_right,
            missing_reference_field_curvature_rejected=wrong_curvature,
            fake_internal_rotation_rejected=wrong_static,
            wrong_frozen_reference_rejected=wrong_fixed_ref))


def relative_source_word(B: Any, charged: Any, neutral: Any,
                         axes: tuple[int, ...]) -> Any:
    """Literal ordered SOURCE word, preserving the SAME nonzero neutral action."""
    result = B
    for axis in reversed(axes):
        result = charged[axis]@result-result@neutral[axis]
    return result


def _ordered_word(generators: np.ndarray) -> np.ndarray:
    result = np.eye(generators.shape[1], dtype=complex)
    for axis, angle in ((0, math.pi), (1, math.pi), (0, -math.pi), (1, -math.pi)):
        result = expm(-1j*angle*generators[axis])@result
    return result


def photon_vectors() -> tuple[np.ndarray, np.ndarray]:
    """Bosonic vacuum plus actual Cartesian one-photon vector, integer spin."""
    generators = np.zeros((3, 4, 4), complex)
    for i, j, k in itertools.product(range(3), repeat=3):
        generators[i, 1+k, 1+j] = 1j*EPS[i, j, k]
    return generators, np.diag([1.5, 2.5, 2.5, 2.5])


def projected_photon_phase(q: np.ndarray) -> np.ndarray:
    """Exact Fock matrix elements BEFORE vacuum/one-photon compression."""
    factor = math.exp(-float(q@q)/2)
    result = np.zeros((4, 4), complex)
    result[0, 0] = 1
    result[0, 1:] = 1j*q; result[1:, 0] = 1j*q
    result[1:, 1:] = np.eye(3)-np.outer(q, q)
    return factor*result


def _reference_controls(group: dict[str, Any]) -> dict[str, Any]:
    model, spins = group['primitive_model'], group['rest_spins']
    generators: np.ndarray = np.asarray([model.encode_even(s, model.reference_modes) for s in spins])
    casimir = sum(s@s for s in generators)
    embedding = reference_sector_embedding(model)
    rows = []
    for count, projector in enumerate(embedding['projectors']):
        values, vectors = eigh(projector)
        selected = vectors[:, values > .5]
        spectrum = np.linalg.eigvalsh(selected.conj().T@casimir@selected)
        rows.append(dict(reference_count=count, primitive_dimension=len(spectrum),
                         Casimir_values=sorted(set(np.round(spectrum, 10).tolist()))))
        if count in (0, 4):
            accept(np.linalg.norm(casimir@projector), 'actual neutral Ref0/4 bosonic return')
    lie = max(accept(np.linalg.norm(generators[i]@generators[j]-generators[j]@generators[i]-1j*generators[k]),
                     'primitive allsector reference SU2', 2e-8)
              for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)))
    return dict(rows=rows, all_sector_Lie_error=lie,
                finite_unwanted_reference_penalty=17., reference_hard_filter=False)


def full_gauss_controls(group: dict[str, Any], frame: np.ndarray) -> dict[str, Any]:
    model, color = group['primitive_model'], group['color']
    basis = matrix_unit_basis()
    gb = expm(.19j*basis[3]-.13j*basis[9]); gr = expm(.11j*basis[6]+.07j*basis[12])
    physical = expm(model.encode_even(.19j*basis[3]-.13j*basis[9], model.bulk_modes)
                    +model.encode_even(.11j*basis[6]+.07j*basis[12], model.reference_modes))
    destination, origin = ((model.bulk_modes, model.reference_modes) if group['branch'] == 1
                           else (model.reference_modes, model.bulk_modes))
    units = _primitive_units(model, destination, origin)
    quartet = np.array([1., .3j, -.2+.4j, .6])
    rotation = Rotation.from_rotvec(np.array([.31, -.23, .17])).as_matrix()
    matrix = source_M(rotation, frame, quartet, color)
    if group['branch'] == -1:
        matrix = matrix.conj().T
    before = _source_operator(units, matrix)
    transformed = gb@matrix@gr.conj().T if group['branch'] == 1 else gr@matrix@gb.conj().T
    omitted_ref = gb@matrix if group['branch'] == 1 else matrix@gb.conj().T
    after = _source_operator(units, transformed)
    ward = accept(np.linalg.norm(after-physical@before@physical.conj().T),
                  'actual primitive fullSU4 source/reference Gauss Ward', 2e-8)
    body = physical_orbit(rotation, frame, color)
    source_frame = gb@body['source']@gb.conj().T
    ref_frame = gr@body['reference']@gr.conj().T
    transformed_color = gr@color@gb.conj().T
    # color is body/ref -> source, so Cdagger Phi_ref C = Phi_source.
    frame_error = accept(np.linalg.norm(source_frame-transformed_color.conj().T@ref_frame@transformed_color),
                         'noncentral Wilson Phi/ref fullGauss covariance')
    angles = np.array([.27, -.19])
    gauge = expm(1j*(angles[0]*sum(model.n(i) for i in model.bulk_modes)
                     +angles[1]*sum(model.n(i) for i in model.reference_modes)))
    u1 = accept(np.linalg.norm(gauge@before@gauge.conj().T-
                              np.exp(1j*group['branch']*(angles[0]-angles[1]))*before),
                'actual primitive real EM source/reference boundary Ward', 2e-8)
    return dict(full15_source_Ward_error=ward, noncentral_Wilson_frame_error=frame_error,
                real_EM_boundary_Ward_error=u1,
                omitted_reference_Gauss_rejected=adverse(np.linalg.norm(
                    _source_operator(units, omitted_ref)-physical@before@physical.conj().T),
                    'actual primitive fullSU4 source/reference Gauss Ward'))


def _native_body(group: dict[str, Any], frame: np.ndarray,
                 improved: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    model = group['primitive_model']
    color = group['color']
    band = RotationBand(a=1., m=.2, r=.12, v=.4990360130634525 if improved else .3, improved=improved)
    onsite, paths = scalar_paths(band)
    pose0 = int(np.argmin(np.linalg.norm(group['rotations']-np.eye(3), axis=(1, 2))))
    source_basis = group['points'][pose0]
    phi = color.conj().T@frame@color
    photon_l, photon_h = photon_vectors()
    loop_z = np.diag([1., -1.]); loop_x = np.array([[0., 1.], [1., 0.]])
    loop_h = 31*(np.eye(2)-loop_z)/2+.13*loop_x
    kappas = np.asarray([[.17, -.11, .08], [-.17, .11, -.08]])
    size = 16*4*2*2
    field_inertia = 2*float(np.sum(np.einsum('mij,mji->m', frame[1:], frame[1:]).real))
    rotor = sum(l@l for l in group['L'])/(2*field_inertia)
    scalar = 32*float(np.sum((np.einsum('mij,nji->mn', frame, frame).real/4)**2))
    harmonic = .015/27*float(kappas[0]@kappas[0])
    h = np.kron(rotor+scalar*np.eye(16), np.eye(16))
    h += np.kron(np.eye(16), np.kron(np.kron(.07*photon_h, np.eye(2)), np.eye(2)))
    h += np.kron(np.eye(64), np.kron(loop_h, np.eye(2)))
    h += (harmonic+.015/3)*np.eye(size)
    onsite_operator = model.encode_even(onsite*phi[0], model.bulk_modes)
    h += np.kron(source_basis.conj().T@onsite_operator@source_basis, np.eye(16))
    currents = np.zeros((3, size, size), complex)
    contacts = np.zeros_like(currents)
    link = expm(.08j*matrix_unit_basis()[4]-.06j*matrix_unit_basis()[11])
    lengths = []
    for displacement, coefficients in paths:
        axis = int(np.flatnonzero(displacement)[0]); length = int(displacement[axis])
        fx = np.einsum('m,mij->ij', coefficients, phi)
        fy = .91*fx
        transport = np.linalg.matrix_power(link, length)
        endpoint = (fx@transport+transport@fy)/2
        actual = model.encode_even(endpoint, model.bulk_modes)
        compressed = source_basis.conj().T@actual@source_basis
        q = np.zeros(3); q[axis] = .23*length
        phase = projected_photon_phase(q)
        fibre = np.diag(np.exp(1j*length*kappas[:, axis]))
        forward = np.kron(compressed, np.kron(np.kron(phase, loop_z), fibre))
        h += forward+forward.conj().T
        currents[axis] += 1j*length*(forward-forward.conj().T)
        contacts[axis] -= length**2*(forward+forward.conj().T)
        lengths.append(length)
    # Entire ROOT reference occupations/loop sectors are checked separately;
    # this exact charged source span has reference3 (plus) or1 (minus).
    return h, currents, contacts, dict(improved=improved, actual_path_lengths=lengths,
        field_inertia=field_inertia, original_quartic_scalar=scalar,
        harmonic_offset_pairs=kappas.tolist(), ordinary_scalar_offsets_retained=True,
        auxiliary_defect=.13, auxiliary_penalty=31., source_invariant_dimension=size,
        current_norms=[float(np.linalg.norm(j)) for j in currents],
        contact_norms=[float(np.linalg.norm(j)) for j in contacts],
        actual_projected_CCR_phase_before_truncation=True)


def intrinsic_clock(time: float, generators: np.ndarray,
                    field_l: np.ndarray) -> dict[str, Any]:
    """Four C2 quintic pulses, unitary convention exp(-i theta K).

    L=-i derivative means operator conjugation rotates Phi with the opposite
    parameter and reversed Cartesian row-matrix order, not a torus motion.
    """
    core = np.eye(len(generators[0]), dtype=complex)
    field = np.eye(len(field_l[0]), dtype=complex)
    phi_rotation = np.eye(3)
    active_axis, speed = 0, 0.
    for segment, (axis, angle) in enumerate(((0, math.pi), (1, math.pi),
                                            (0, -math.pi), (1, -math.pi))):
        u = min(1., max(0., (time-1.5*segment)/1.5))
        theta = angle*(10*u**3-15*u**4+6*u**5)
        if 0 < u < 1:
            active_axis, speed = axis, angle*(30*u**2-60*u**3+30*u**4)/1.5
        core = expm(-1j*theta*generators[axis])@core
        field = expm(-1j*theta*field_l[axis])@field
        phi_rotation = phi_rotation@Rotation.from_rotvec(-theta*np.eye(3)[axis]).as_matrix()
    return dict(core=core, field=field, Phi_rotation=phi_rotation, axis=active_axis, speed=speed)


def native_history(group: dict[str, Any], frame: np.ndarray,
                   improved: bool) -> dict[str, Any]:
    _, _, field_l = integer_orientation()
    photon_l, _ = photon_vectors()
    environment = 4*2*2*10
    field_neutral = np.asarray([np.kron(np.eye(16), field_l[i]) for i in range(3)])
    em_neutral = np.asarray([np.kron(np.kron(photon_l[i], np.eye(4)), np.eye(10))
                             for i in range(3)])
    full_neutral = field_neutral+em_neutral
    full_charged = [sparse.kron(sparse.csr_matrix(group['K'][i]), sparse.eye(environment))+
                    sparse.kron(sparse.eye(16), sparse.csr_matrix(full_neutral[i])) for i in range(3)]
    # The physical field-vector rotation is INTRINSIC Phi only. Real EM angular
    # generators remain alive, but are explicitly removed on both source legs.
    neutral = [sparse.csr_matrix(full_neutral[i]-em_neutral[i]) for i in range(3)]
    charged = [full_charged[i]-sparse.kron(sparse.eye(16), sparse.csr_matrix(em_neutral[i]))
               for i in range(3)]
    quartet = np.array([1., .3j, -.2+.4j, .6])
    spins = group['rest_spins']
    m0 = (quartet[0]*np.eye(4)+2j*np.einsum('a,aij->ij', quartet[1:], spins))/2
    if group['branch'] == -1:
        m0 = m0.conj().T
    source = sparse.kron(sparse.csr_matrix(m0.reshape(16, 1)), sparse.eye(environment)).tocsr()
    intrinsic = sum(relative_source_word(source, charged, neutral, (i, i)) for i in range(3))
    relative_error = accept(sparse.linalg.norm(intrinsic-.75*source),
                            'literal SOURCE Casimir with nonzero SAME-neutral Phi field', 2e-8)
    field_state = np.zeros(10, complex); field_state[[0, 1, 5]] = [1., .3j, -.2+.4j]
    field_state /= np.linalg.norm(field_state)
    ph_state = np.array([0., 1., .3j, .2]); ph_state /= np.linalg.norm(ph_state)
    neutral_state = np.kron(np.kron(np.kron(ph_state, np.array([1., 0.])),
                                    np.ones(2)/math.sqrt(2)), field_state)
    prepared = source@neutral_state
    neutral_noise = max(float(np.linalg.norm(k@neutral_state)) for k in neutral)
    em_noise = max(float(np.linalg.norm(k@neutral_state)) for k in em_neutral)
    if min(neutral_noise, em_noise) <= 1e-3:
        raise AssertionError('SAME neutral Phi or EM variance was silently erased')
    missing_neutral = adverse(float(np.linalg.norm(sum(k@k@prepared for k in charged)-intrinsic@neutral_state)),
                              'literal SOURCE Casimir with nonzero SAME-neutral Phi field')
    h, currents, contacts, body = _native_body(group, frame, improved)
    hfull = sparse.kron(sparse.csr_matrix(h), sparse.eye(10), format='csr')
    hfull += sparse.kron(sparse.eye(256), sparse.csr_matrix(sum(l@l for l in field_l)/
                                                          (2*body['field_inertia'])), format='csr')
    # These cross terms are REQUIRED: F(M0;R)*f_l(R) has a SINGLE field angle.
    # Omitting them would fake two independent tensor-Hermite Phi rotors.
    for i in range(3):
        hfull += sparse.kron(sparse.csr_matrix(np.kron(group['L'][i], np.eye(16))),
                             sparse.csr_matrix(field_l[i])/body['field_inertia'], format='csr')
    records = []
    clock_ward_errors = []
    physical_rotation = None
    for t in np.linspace(0., 6., 9):
        pulse = intrinsic_clock(float(t), group['K'], field_l)
        current_word, field_word, physical_rotation = pulse['core'], pulse['field'], pulse['Phi_rotation']
        body_vector = sparse.linalg.expm_multiply(-1j*t*hfull, prepared,
                                                 traceA=-1j*t*complex(hfull.diagonal().sum()))
        tensor = body_vector.reshape(16, 4, 2, 2, 10)
        rotated = np.einsum('ab,ij,bplkj->aplki', current_word, field_word, tensor)
        body_state = body_vector.reshape(256, 10)
        body_currents = np.asarray([sum(np.vdot(body_state[:, a], j@body_state[:, a]).real
                                        for a in range(10)) for j in currents])
        # Spatial sites/Ref remain fixed; intrinsic Phi rotation does NOT rotate
        # the huge torus or relabel physical current axes as R*j.
        records.append(dict(time=t, intrinsic_Phi_vector_rotation=physical_rotation.tolist(),
            actual_fixed_spatial_axis_current=body_currents.tolist(),
            source_norm=float(np.linalg.norm(rotated)),
            ordinary_energy=float(np.vdot(body_vector, hfull@body_vector).real),
            propagated_source_sha256=hashlib.sha256(rotated.tobytes()).hexdigest()))
        if pulse['speed']:
            # Exercise the actual moving state equation at interior times, not
            # merely the four copied finite matrices or endpoint calibrations.
            unitary = sparse.kron(sparse.csr_matrix(np.kron(current_word, np.eye(16))),
                                   sparse.csr_matrix(field_word), format='csr')
            state = rotated.reshape(-1)
            moving_h_state = unitary@(hfull@(unitary.conj().T@state))
            theta_term = pulse['speed']*(charged[pulse['axis']]@state)
            derivative = -1j*theta_term-1j*(unitary@(hfull@body_vector))
            clock_ward_errors.append(accept(np.linalg.norm(1j*derivative-moving_h_state-theta_term),
                                             'actual intrinsic moving ordinary equation', 2e-7))
    assert physical_rotation is not None, 'The fixed nine-node physical clock is nonempty'
    charged_loop = _ordered_word(group['K'])
    neutral_loop_error = accept(np.linalg.norm(_ordered_word(field_l)-np.eye(10)),
                               'bosonic intrinsic SO3/neutral Ref0or4 ordered return')
    charged_loop_error = accept(np.linalg.norm(charged_loop+np.eye(16)),
                                'actual charged clamped four-axis holonomy minus')
    field_return = accept(np.linalg.norm(physical_rotation-np.eye(3)),
                          'actual Cartesian Phi ordered SO3 return')
    integer_wrong = adverse(np.linalg.norm(_ordered_word(group['L'])+np.eye(16)),
                            'actual charged clamped four-axis holonomy minus')
    ref_counts = _reference_controls(group)
    scalar = source19_boundary_word()
    return dict(branch=group['branch'], body=body, charged_quantum_dimension=16*environment,
        neutral_quantum_dimension=environment, source_reference_columns=4,
        actual_raw_quartet=[str(v) for v in quartet], source_norm=float(np.linalg.norm(prepared)),
        SAME_neutral_Phi_Noether_noise_norm=neutral_noise, actual_subtracted_EM_noise_norm=em_noise,
        true_relative_SOURCE_Casimir_error=relative_error, missing_neutral_rejected=missing_neutral,
        ordered_charged_minus_error=charged_loop_error, ordered_neutral_plus_error=neutral_loop_error,
        actual_Phi_SO3_return_error=field_return, omitted_ref_integer_holonomy_rejected=integer_wrong,
        native_ordinary_control='intrinsic H_theta=R_K H0 R_Kdagger; Hdrive=H_theta+sum theta_dot K_i; SAME positive clock0..6',
        spatial_Ref_position_fixed=True, spatial_torus_rotation_claimed=False,
        common_EM_Noether_subtracted_on_both_source_legs=True,
        full_Phi_rotor_kinetic_cross_terms_retained=True,
        quintic_C2_four_segment_clock=True,
        interior_moving_ordinary_equation_error=max(clock_ward_errors),
        reference_full_Pauli_sectors=ref_counts,
        eta18_local_boundary_curvature_retained=scalar['centre_second_curvature'],
        missing_eta_curvature_rejected=scalar['missing_curvature_rejected'],
        native_scope='bosonic integer INTRINSIC Cartesian Phi orientation and exact primitive source-invariant subspace, coupled actual projected CCR/current plus both flux sectors and paired harmonic fibres; not unrestricted Phi vacuum or a rigid spatial lattice SO3',
        records=records)


def serialize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return dict(real=value.real.tolist(), imag=value.imag.tolist())
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--moving-packet', type=Path, required=True)
    parser.add_argument('--moving-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    moving_packet = json.loads(args.moving_packet.read_text())
    moving_bytes = args.moving_receipt.read_bytes()
    if hashlib.sha256(moving_bytes).hexdigest() != moving_packet['receipt']['sha256']:
        raise ValueError('Moving source receipt differs from its frozen packet')
    moving = json.loads(moving_bytes)
    if moving['selected']['source_valued_intrinsic_Casimir']['source_count'] != 19:
        raise ValueError('Cartesian group requires actualSC19, never profile19^3/RAW41')
    band = RotationBand(a=1., m=.2, r=.12, v=.3, improved=False)
    _, frame = original_frame(band, 5, np.zeros(3), False)
    color = expm(.17j*matrix_unit_basis()[3]-.11j*matrix_unit_basis()[8])
    model = NativeOriginalModel()
    groups = [cartesian_source_group(frame, color, branch, model) for branch in (1, -1)]
    native = [native_history(group, frame, improved) for group in groups for improved in (False, True)]
    # Four pi-sized segments vs the previous1/16 pulse amplify a q6 clock by
    # at most2^(9*6). Uniform source/form coefficients remain conditional on
    # SAME co-moving ordinary graphs; frozen static preparation is not reset.
    reserve = 54
    original_error = moving['selected']['moving_source_current_secondword_transfer_log2_upper']+reserve
    improved_error = moving['improved']['new_moving_source_current_word_transfer_log2_upper']+reserve
    if max(original_error, improved_error) >= -4922:
        raise AssertionError('Cartesian group q6/clock reserve exceeds physical moving target')
    result = dict(plan=PLAN, source_pair='cartesian_noether_group.py/.md',
        physical_frame=frame, source_color=color,
        source_groups=[dict(branch=g['branch'], L=g['L'], reference=g['reference'], K=g['K'],
                            controls=g['controls'], source_embedding_dimension=list(g['embedding'].shape),
                            fullGauss=full_gauss_controls(g, frame)) for g in groups],
        native=native, transfer=dict(original_moving_word_log2_upper=original_error,
            independently_improved_word_log2_upper=improved_error, q6_four_segment_clock_reserve_bits=reserve,
            source='actualSC19/allRef4/rawfixedquartet', frozen_preparation_not_reset=True,
            ordinary_H_E0kappa_all64_and_normal_hypotheses_retained=True,
            group_orbit_is_not_unrestricted_Phi_phase=True,
            local_eta_boundary=source19_nested_casimir()),
        input_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in (args.moving_packet, args.moving_receipt)},
        source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                       for name in ('cartesian_noether_group.py', 'cartesian_noether_group.md')})
    args.output.write_text(json.dumps(result, default=serialize, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output), branches=len(groups), histories=len(native),
        original_word_log2=original_error, improved_word_log2=improved_error,
        true_SOURCE_Casimir_error=max(g['controls']['true_second_SOURCE_Casimir_error'] for g in groups),
        multiaxis_holonomy_error=max(n['ordered_charged_minus_error'] for n in native))))


if __name__ == '__main__':
    main()
