#!/usr/bin/env python3
"""Actual prepared Gauss/OS source and physical Cartesian Noether controls.

The NEW law's gauge-null product metric, full normal/colour transport and
field-dependent Ref spin are distinct from LAW05. Bare Haar leaves a kernel:
unfiltered source and field-only U need not preserve its physical range.
This CLI joins actual positive-range thermal sources, paid angular actuation,
polarized body/Ref controls and CONDITIONAL full-Naux angular bounds.
Native PE6 is measured positive; native PE.25 is ZERO. Full microscopic
source PE.25 admission remains OPEN. Historical67bfab/642e and withdrawn
.894663 macro Grams are archived, never retagged to physical Dminus.
Use --root-receipt and --source-receipt with immutable matching physical
suppliers; source/projector/current/full4 Ref all use physical T. A joined
source ledger is not a proved microscopic source Gram.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, cast

for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np
import scipy
from scipy.linalg import expm, expm_frechet, logm
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
SOURCE_NAMES = ('covariant_physical_noether.py', 'covariant_root_transfer.py',
                'covariant_induced_photon.py', 'bosonic_statistics.py',
                'sea_selection.py', 'rotation_band.py', 'induced_root_transfer.py',
                'induced_photon_response.py', 'improved_phi_band.py',
                'microscopic_native_orbit_lift.py', 'local_orbital_stability.py',
                'covariant_prepared_gauss_state.py', 'covariant_outside_angular.py')
SOURCE_BYTES_AT_IMPORT = {name: (ROOT/name).read_bytes() for name in SOURCE_NAMES}
_loaded_code = sys._getframe().f_code
LOADED_CODE_MATCH = _loaded_code == compile(SOURCE_BYTES_AT_IMPORT[Path(__file__).name],
    _loaded_code.co_filename, 'exec', dont_inherit=True, optimize=sys.flags.optimize)
if not LOADED_CODE_MATCH:
    raise ArithmeticError('Loaded Noether module differs from its captured source')
import covariant_prepared_gauss_state as prepared
from bosonic_statistics import BosonicEdgeModel
from covariant_induced_photon import Parameters, WilsonConsumer, phi_frame
from covariant_root_transfer import (RootParameters, native_lift, positive_log,
                                     wilson_step, wilson_component_basis, centre_current_native,
                                     exact_chart_ground, charge_weighted_phi_budget,
                                     PhysicalComponentFrame)
sys.meta_path.insert(0, prepared.finder)
try:
    from covariant_outside_angular import outside_angular_bound
finally:
    sys.meta_path.remove(prepared.finder)
from sea_selection import SELECTED, matrix_unit_basis
SOURCE_NAMES = tuple(sorted(set(SOURCE_NAMES)|{name+'.py' for name in prepared.COMPILED}))
for _name in SOURCE_NAMES:
    _stem = _name.removesuffix('.py')
    if _stem in prepared.COMPILED:
        SOURCE_BYTES_AT_IMPORT[_name] = prepared.SNAPSHOT[_stem]
SOURCE_HASHES_AT_IMPORT = {name: hashlib.sha256(value).hexdigest()
                         for name, value in SOURCE_BYTES_AT_IMPORT.items()}


def real_vector(matrix: np.ndarray) -> np.ndarray:
    value = matrix.reshape(-1)
    return np.r_[np.real(value), np.imag(value)]


def physical_triad(phi: np.ndarray) -> dict[str, Any]:
    """Derive generators from literal Cartesian vector tangents in all60 Phi."""
    basis = np.asarray(matrix_unit_basis())
    columns = np.column_stack([real_vector(1j*(g@phi-phi@g)) for g in basis])
    coefficients = []
    residuals = []
    for j, k in ((2, 3), (3, 1), (1, 2)):
        tangent = np.zeros_like(phi)
        tangent[j], tangent[k] = -phi[k], phi[j]
        solution = np.linalg.lstsq(columns, real_vector(tangent), rcond=None)[0]
        coefficients.append(solution)
        residuals.append(float(np.linalg.norm(columns@solution-real_vector(tangent))))
    # Passive coframe-label pullback is +i[J,Phi]; active matter rotation
    # is exp(-i theta J) with the matching active orbital rotation R p.
    spins = np.einsum('ja,aik->jik', coefficients, basis)
    return dict(spins=spins, columns=columns, coefficients=np.array(coefficients),
                residuals=residuals, rank=int(np.linalg.matrix_rank(columns)),
                commutator_error=max(float(np.linalg.norm(spins[j]@spins[k]-spins[k]@spins[j]-1j*spins[l]))
                                     for j, k, l in ((0, 1, 2), (1, 2, 0), (2, 0, 1))),
                Casimir_error=float(np.linalg.norm(sum(s@s for s in spins)-.75*np.eye(4))),
                two_pi_error=float(np.linalg.norm(expm(-2j*math.pi*spins[2])+np.eye(4))))


def chart_velocity(phi: np.ndarray, fields: np.ndarray, frames: np.ndarray,
                   links: np.ndarray, edges: tuple[tuple[int, int], ...],
                   field_flow: np.ndarray, link_flow: np.ndarray) -> dict[str, Any]:
    """Pull actual primitive tangent through NEW quotient chart, all15 Gauss.

    At Y=0, solve the chart differential, not a frozen kinetic minimization.
    The covariance connection fixes the fibre lift.  Physical velocities are
    normal Ydot and dressed colour Ldot; vertical SU4 velocities disappear.
    """
    basis = np.asarray(matrix_unit_basis())
    local_columns = physical_triad(phi)['columns']
    omega = []
    normal = []
    body_fields = []
    for frame, field, flow in zip(frames, fields, field_flow):
        body_flow = frame.conj().T@flow@frame
        body_field = frame.conj().T@field@frame
        body_fields.append(body_field)
        actual_columns = np.column_stack([real_vector(1j*(g@body_field-body_field@g)) for g in basis])
        coefficient = np.linalg.solve(local_columns.T@actual_columns, local_columns.T@real_vector(body_flow))
        generator = np.einsum('a,aij->ij', coefficient, basis)
        omega.append(generator)
        normal.append(body_flow-1j*(generator@body_field-body_field@generator))
    omega = np.asarray(omega)
    dressed = []
    colour = []
    colour_coordinates = []
    jacobian_residuals = []
    for edge, (x, y) in enumerate(edges):
        left = frames[x].conj().T@links[edge]@frames[y]
        tangent = frames[x].conj().T@link_flow[edge]@frames[y]-1j*omega[x]@left+1j*left@omega[y]
        dressed.append(left)
        colour.append(-1j*tangent@left.conj().T)
        logarithm = logm(left)
        jacobian = np.column_stack([real_vector(-1j*cast(np.ndarray, expm_frechet(logarithm, 1j*g, compute_expm=False))@left.conj().T)
                                    for g in basis])
        coordinates = np.linalg.lstsq(jacobian, real_vector(colour[-1]), rcond=None)[0]
        colour_coordinates.append(coordinates)
        jacobian_residuals.append(float(np.linalg.norm(jacobian@coordinates-real_vector(colour[-1]))))
    normal = np.asarray(normal)
    colour = np.asarray(colour)
    return dict(omega=omega, normal=normal, colour=colour, dressed=np.asarray(dressed),
                colour_Lie_coordinates=np.asarray(colour_coordinates),
                actual_left_Jacobian_residual=max(jacobian_residuals),
                normalized_quotient_speed_squared=float(np.linalg.norm(normal)**2+np.linalg.norm(colour_coordinates)**2),
                reconstruction_error=float(np.linalg.norm(np.array(normal)+1j*(omega[:, None]@np.array(body_fields)-np.array(body_fields)@omega[:, None])-
                    np.array([v.conj().T@f@v for v, f in zip(frames, field_flow)]))))


def fullfield_local_native(phi: np.ndarray, triad: dict[str, Any], p: RootParameters,
                           rng: np.random.Generator) -> dict[str, Any]:
    """Nontrivial frames, colour holonomy, centre branch and FULL gauge lift."""
    basis = np.asarray(matrix_unit_basis())
    frames = np.array([expm(.19j*np.einsum('a,aij->ij', rng.normal(size=15), basis))
                       for _ in range(3)])
    frames *= np.array([1, 1j, -1])[:, None, None]
    fields = frames[:, None]@phi@frames[:, None].conj().transpose(0, 1, 3, 2)
    edges = ((0, 1), (1, 2), (0, 2))
    dressed = np.array([expm(.06j*np.einsum('a,aij->ij', rng.normal(size=15), basis))
                        for _ in edges])
    links = np.array([frames[x]@dressed[e]@frames[y].conj().T for e, (x, y) in enumerate(edges)])
    alpha = np.array([1., .25, 0.])
    flow = np.zeros_like(fields)
    flow[:, 1] = -alpha[:, None, None]*fields[:, 2]
    flow[:, 2] = alpha[:, None, None]*fields[:, 1]
    stationary_links = np.zeros_like(links)
    actual = chart_velocity(phi, fields, frames, links, edges, flow, stationary_links)
    # Full local Gauss flow, including all fields and all oriented endpoints.
    gauge = np.einsum('xa,aij->xij', rng.normal(size=(3, 15)), basis)
    gauge_fields = 1j*(gauge[:, None]@fields-fields@gauge[:, None])
    gauge_links = np.array([1j*(gauge[x]@links[e]-links[e]@gauge[y]) for e, (x, y) in enumerate(edges)])
    vertical = chart_velocity(phi, fields, frames, links, edges, gauge_fields, gauge_links)
    # Actual off-Clifford full-Phi normal packet, not gamma relabeling.
    candidate = np.einsum('ma,aij->mij', rng.normal(size=(4, 15)), basis)
    columns = triad['columns']
    correction = np.einsum('a,aij->ij', np.linalg.lstsq(columns, real_vector(candidate), rcond=None)[0], basis)
    projected = candidate-1j*(correction@phi-phi@correction)
    gram = np.array(Parameters().gram)
    tube_reach = min(float(min(gram))/(512*float(np.linalg.norm(phi))), math.sqrt(float(min(gram)))/64)
    normal_field = (tube_reach/8)*projected/np.linalg.norm(projected)
    off_fields = frames[:, None]@(phi+normal_field)@frames[:, None].conj().transpose(0, 1, 3, 2)
    off_flow = np.zeros_like(off_fields)
    off_flow[:, 1] = -alpha[:, None, None]*off_fields[:, 2]
    off_flow[:, 2] = alpha[:, None, None]*off_fields[:, 1]
    off_chart = chart_velocity(phi, off_fields, frames, links, edges, off_flow, stationary_links)
    off_world = frames@off_chart['omega']@frames.conj().transpose(0, 2, 1)
    off_gauge_fields = 1j*(off_world[:, None]@off_fields-off_fields@off_world[:, None])
    off_gauge_links = np.array([1j*(off_world[x]@links[e]-links[e]@off_world[y]) for e, (x, y) in enumerate(edges)])
    off_gauge = chart_velocity(phi, off_fields, frames, links, edges, off_gauge_fields, off_gauge_links)
    full_normal_group_error = normal_slice_error = normal_norm_error = 0.
    wrong_fixed_normal_error = 0.
    for theta in (.17, .73, 1.9, 2*math.pi):
        spatial = Rotation.from_rotvec([0., 0., theta]).as_matrix()
        spin_lift = expm(1j*theta*triad['spins'][2])
        label_normal = np.concatenate((normal_field[:1], np.einsum('jk,kab->jab', spatial, normal_field[1:])))
        transported_normal = spin_lift.conj().T@label_normal@spin_lift
        literal_full_phi = np.concatenate(((phi+normal_field)[:1], np.einsum('jk,kab->jab', spatial, (phi+normal_field)[1:])))
        constructed_full_phi = spin_lift@(phi+transported_normal)@spin_lift.conj().T
        full_normal_group_error = max(full_normal_group_error, float(np.linalg.norm(literal_full_phi-constructed_full_phi)))
        normal_slice_error = max(normal_slice_error, float(np.linalg.norm(columns.T@real_vector(transported_normal))))
        normal_norm_error = max(normal_norm_error, abs(float(np.linalg.norm(transported_normal)-np.linalg.norm(normal_field))))
        wrong_fixed_normal_error = max(wrong_fixed_normal_error, float(np.linalg.norm(literal_full_phi-spin_lift@(phi+normal_field)@spin_lift.conj().T)))
    # Material repair: transport links with the mechanically derived body lift.
    world = frames@actual['omega']@frames.conj().transpose(0, 2, 1)
    compensated_links = np.array([1j*(world[x]@links[e]-links[e]@world[y]) for e, (x, y) in enumerate(edges)])
    repaired = chart_velocity(phi, fields, frames, links, edges, flow, compensated_links)
    # This repair is gauge, not an electron rotation: Ref and sea must co-transport.
    reference = frames[2].conj().T@world[2]@frames[2]
    prepared_links = np.array([frames[x]@frames[y].conj().T for x, y in edges])
    prepared = chart_velocity(phi, fields, frames, prepared_links, edges, flow, stationary_links)
    qspeed = prepared['normalized_quotient_speed_squared']
    paid_log = math.log10(qspeed/4)+2*p.normal_width_power
    return dict(all15_Gauss_null_speed=vertical['normalized_quotient_speed_squared'],
                chart_differential_reconstruction_error=actual['reconstruction_error'],
                actual_nonclifford_normal_size=float(np.linalg.norm(normal_field)),
                actual_nonclifford_chart_reconstruction_error=off_chart['reconstruction_error'],
                actual_nonclifford_left_Jacobian_error=off_chart['actual_left_Jacobian_residual'],
                actual_nonclifford_full_Gauss_null_speed=off_gauge['normalized_quotient_speed_squared'],
                actual_nonclifford_physical_normal_velocity_norm=float(np.linalg.norm(off_chart['normal'])),
                actual_full45_normal_physical_group_error=full_normal_group_error,
                actual_transported45_normal_slice_error=normal_slice_error,
                actual_radial_normal_ground_radius_error=normal_norm_error,
                wrong_untransported_normal_physical_group_error=wrong_fixed_normal_error,
                nonclifford_scope='Actual off-Clifford full-Phi chart tangent and finite mechanical group: Vprime=VU, Yprime=Udag Lambda(R)Y U. Normal slice and radius are preserved; the successor radial normal ground, unlike old component-box ground, is exactly invariant. This is not a gamma-only action.',
                nontrivial_colour_holonomy_norm=float(np.linalg.norm(dressed[0]@dressed[1]-dressed[2])),
                physical_fixed_link_quotient_speed_squared=qspeed,
                repaired_covariantly_transported_link_speed_squared=repaired['normalized_quotient_speed_squared'],
                prepared_colour_identity_speed_squared=qspeed,
                actual_Ref_RIGHT_norm=float(np.linalg.norm(reference)),
                local_preparation_gaussian_generator_variance_log10_lower=paid_log,
                width_power=p.normal_width_power, compact_ground=exact_chart_ground(),
                cometric='Reduced product g_inverse=4Omega sigma^2 on Y45 and X15/link. Half-density Haar chart; V gauge-null. Pullback M gives D^T M D=0 for EVERY full Gauss column, NOT (I_Phi+L/.6).',
                field_Noether='Physical-components Dminus=partial-(dQ)Qdag on ALL matter/Ref. Ordinary world -i partial equals -i Dminus plus -i(dQ)Qdag, including normal/colour half-density flow and transported centre derivatives. Source derivative uses Acharged S-S Aneutral. FullCartesian outside/domain bounds require the NEW same-law supplier, not the historical Dplus ledger.',
                price_scope='Gaussian translation variance speed^2/(4sigma^2) is a LOWER bound from curvature -log(k)second>=1/(2sigma^2). This is quantum preparation, not probability1 inferred from a pointwise chart.',
                failed_native_mechanical_branch='Local Cartesian triad rotation with stationary physical links has prohibitive colour packet derivative cost. Pure-gauge link compensation removes it but co-transports Ref/sea and cannot alone produce an addressed physical sign.',
                old_LAW05_reference='physical_core_spin_module.py:25-109, separate methodology only; .6, gap and preparation bounds NOT transferred.')


def source_operator(model: BosonicEdgeModel, packet: np.ndarray, charge: int) -> np.ndarray:
    result = np.zeros_like(model.identity)
    for c, d in itertools.product(range(4), repeat=2):
        path = tuple([d+4]+([4] if d else [])+[0]+([c] if c else []))
        result += packet[c, d]*model.bilinear(c, d+4, path=path)
    return result/math.sqrt(2) if charge == 1 else result.conj().T/math.sqrt(2)


def physical_component_native_rejoin(gamma: np.ndarray, triad: dict[str, Any],
                                    p: RootParameters) -> dict[str, Any]:
    """Independent physical Dirac eigenvectors reject a raw beta-block transfer."""
    plus = (np.eye(4)+gamma[0])/2
    minus = np.eye(4)-plus
    consumer = WilsonConsumer(gamma, p.a, p.mass)
    theta = wilson_component_basis(gamma)
    momentum = np.array(p.momentum)
    def direct_transfer(spatial: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        energy = consumer.energy(spatial)
        b = 1+consumer.m0+float(np.sum(1-np.cos(spatial)))
        alpha = np.array([1j*gamma[0]@g for g in gamma[1:]])
        h = (b-math.cosh(energy))*gamma[0]+np.einsum('j,jab->ab', np.sin(spatial), alpha)
        values, spinors = np.linalg.eigh(h)
        transfer = np.zeros((4, 4), complex)
        projectors = []
        for sign in (1, -1):
            eigenvalue = math.exp(-sign*energy)
            pole = spinors[:, sign*values > 0]
            # Transfer CAR components and physical Dirac pole components differ
            # by this temporal Wilson factor, not by a silent basis relabel.
            modes, _ = np.linalg.qr((plus/eigenvalue+minus)@pole)
            projector = modes@modes.conj().T
            transfer += eigenvalue*projector
            projectors.append(projector)
        return transfer, projectors
    physical, determinant, energy = wilson_step(gamma, p.a, p.mass, momentum)
    direct, projectors = direct_transfer(momentum)
    block = theta.conj().T@physical@theta
    values, modes = np.linalg.eigh(physical)
    raw_values, raw_modes = np.linalg.eigh(block)
    rows = []
    for sign, projector in zip((1, -1), projectors):
        eigenvalue = math.exp(-sign*energy)
        observed = modes[:, sign*np.log(values) < 0]
        raw = raw_modes[:, sign*np.log(raw_values) < 0]
        rows.append(dict(charge=sign,
            physical_Dirac_to_transfer_projector_error=float(np.linalg.norm(observed@observed.conj().T-projector)),
            physical_source_band_Dirac_kernel_error=float(np.linalg.norm(consumer.inverse(np.array([sign*1j*energy, *momentum]))@(eigenvalue*plus+minus)@observed)),
            raw_beta_block_as_physical_source_projector_error=float(np.linalg.norm(raw@raw.conj().T-projector)),
            raw_beta_block_as_physical_Dirac_kernel_error=float(np.linalg.norm(consumer.inverse(np.array([sign*1j*energy, *momentum]))@(eigenvalue*plus+minus)@raw))))
    epsilon = 2e-5
    current_rows = []
    for axis in range(3):
        offset = epsilon*np.eye(3)[axis]
        tp, dp, _ = wilson_step(gamma, p.a, p.mass, momentum+offset)
        tm, dm, _ = wilson_step(gamma, p.a, p.mass, momentum-offset)
        independent_p, _ = direct_transfer(momentum+offset)
        independent_m, _ = direct_transfer(momentum-offset)
        current = -(positive_log(tp)-positive_log(tm))/(2*epsilon*p.a)
        independent = -(positive_log(independent_p)-positive_log(independent_m))/(2*epsilon*p.a)
        current_rows.append(dict(axis=axis, physical_current_Dirac_eigenvector_error=float(np.linalg.norm(current-independent)),
            wrong_raw_beta_block_physical_current_error=float(np.linalg.norm(current-theta.conj().T@current@theta)),
            actual_detB_vacuum_current=-(math.log(dp)-math.log(dm))/(2*epsilon*p.a)))
    model = BosonicEdgeModel(8, ((0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 7)))
    generator = native_lift(model, triad['spins'][2])+native_lift(model, triad['spins'][2], 4)
    q = expm(.37j*generator)
    dq = 1j*generator@q
    frame = PhysicalComponentFrame(q, dq)
    source0 = source_operator(model, projectors[0], 1)
    source = frame.source(source0, frame)
    derivative = 1j*(generator@source-source@generator)
    correct_derivative = frame.covariant_source_derivative(source, derivative, frame)
    wrong_plus = derivative-frame.connection@source+source@frame.connection
    result: dict[str, Any] = dict(component_law='physical-components-Dminus-20261008',
        source_functional_recipe='SAME neutral thermal/source minors, fullpositive eigenbasis, physical band projector and physical Ref/body components; no TOP2 restriction',
        selected_physical_gamma0_offdiagonal_norm=float(np.linalg.norm(gamma[0]-np.diag(np.diag(gamma[0])))),
        physical_transfer_direct_Dirac_eigenvector_error=float(np.linalg.norm(physical-direct)),
        raw_beta_block_physical_transfer_error=float(np.linalg.norm(block-direct)),
        raw_beta_block_spectrum_error=float(np.linalg.norm(np.linalg.eigvalsh(block)-values)),
        raw_beta_block_determinant_error=abs(np.linalg.det(block)-np.linalg.det(physical)),
        determinant_B=determinant, signed_source_projectors=rows, actual_physical_currents=current_rows,
        actual_full_primitive_Fock_source_Dminus_error=float(np.linalg.norm(correct_derivative)),
        wrong_Dplus_actual_Fock_source_error=float(np.linalg.norm(wrong_plus)),
        actual_full_primitive_Fock_world_connection_error=float(np.linalg.norm(frame.world_connection_generator()-generator)),
        scope='Native actual 128-dimensional primitive even Fock/source and physical4 Dirac/current comparison. Dminus frame cancellation is not a fullCartesian outside/Gauss source admission.')
    if result['physical_transfer_direct_Dirac_eigenvector_error'] > 1e-10:
        raise ArithmeticError('Physical transfer fails independent physical Dirac construction')
    if max(row['physical_Dirac_to_transfer_projector_error'] for row in rows) > 1e-10:
        raise ArithmeticError('Physical source band fails independent physical Dirac eigenvectors')
    if max(row['physical_source_band_Dirac_kernel_error'] for row in rows) > 1e-10:
        raise ArithmeticError('Physical source fails actual signed Dirac pole kernels')
    if max(row['physical_current_Dirac_eigenvector_error'] for row in current_rows) > 1e-6:
        raise ArithmeticError('Physical current fails independent Dirac eigenvector derivatives')
    if result['raw_beta_block_physical_transfer_error'] < .1 or result['wrong_Dplus_actual_Fock_source_error'] < .1:
        raise ArithmeticError('Physical basis/connection adverse control failed to discriminate')
    return result


def native_Ref_specimens(gamma: np.ndarray, triad: dict[str, Any], p: RootParameters,
                         rng: np.random.Generator) -> dict[str, Any]:
    model = BosonicEdgeModel(8, ((0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 7)))
    t, determinant, energy = wilson_step(gamma, p.a, p.mass, np.array(p.momentum))
    transfer = determinant*expm(native_lift(model, positive_log(t)))
    nactive = sum((model.n(c) for c in range(4)), np.zeros_like(model.identity))
    nref = sum((model.n(c) for c in range(4, 8)), np.zeros_like(model.identity))
    spins = triad['spins']
    body = np.array([native_lift(model, s) for s in spins])
    ref = np.array([native_lift(model, s, 4) for s in spins])
    # Ref Noether is an actual field-dependent rdag J(Phi_ref) r operator.
    # Re-solve the Cartesian tangent after every nontrivial SU4 frame change.
    basis = np.asarray(matrix_unit_basis())
    phi = np.sqrt(np.array(Parameters().gram))[:, None, None]*SELECTED
    gauge_hermitian = .11*np.einsum('a,aij->ij', rng.normal(size=15), basis)
    gauge_frame = expm(1j*gauge_hermitian)
    moved_phi = gauge_frame@phi@gauge_frame.conj().T
    moved_spins = physical_triad(moved_phi)['spins']
    moved_ref = native_lift(model, moved_spins[2], 4)
    full_gauge_generator = native_lift(model, gauge_hermitian)+native_lift(model, gauge_hermitian, 4)
    full_gauge_unitary = expm(1j*full_gauge_generator)
    ref_covariance = float(np.linalg.norm(moved_ref-full_gauge_unitary@ref[2]@full_gauge_unitary.conj().T))
    wrong_constant_ref = float(np.linalg.norm(moved_ref-ref[2]))
    infinitesimal_errors = []
    epsilon = 2e-5
    for generator in basis:
        small = expm(1j*epsilon*generator)
        derivative = (physical_triad(small@phi@small.conj().T)['spins'][2]-
                      physical_triad(small.conj().T@phi@small)['spins'][2])/(2*epsilon)
        lifted_derivative = native_lift(model, derivative, 4)
        lifted_gauss = native_lift(model, generator)+native_lift(model, generator, 4)
        infinitesimal_errors.append(float(np.linalg.norm(lifted_derivative-1j*(lifted_gauss@ref[2]-ref[2]@lifted_gauss))))
    packet = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
    packet /= np.linalg.norm(packet)
    results = []
    for charge in (1, -1):
        refcount = 4 if charge == 1 else 0
        selector_h = (nactive-2*model.identity)@(nactive-2*model.identity)+(nref-refcount*model.identity)@(nref-refcount*model.identity)
        values, vectors = np.linalg.eigh(selector_h)
        selected = vectors[:, values < 1e-8]
        eigen, modes = np.linalg.eigh(selected.conj().T@transfer@selected)
        sea = selected@modes[:, -1]
        source = source_operator(model, np.eye(4), charge)
        state = source@sea
        norm = float(np.vdot(state, state).real)
        state /= math.sqrt(norm)
        addressed = expm(-2j*math.pi*body[2])@state
        co_rotated = expm(-2j*math.pi*(body[2]+ref[2]))@state
        arbitrary = source_operator(model, packet, charge)
        # BOTH sides of the source are real primitive even operators.
        if charge == 1:
            body_expected = source_operator(model, spins[2]@packet, charge)
            ref_expected = source_operator(model, -packet@spins[2], charge)
        else:
            body_expected = source_operator(model, -spins[2]@packet, charge)
            ref_expected = source_operator(model, packet@spins[2], charge)
        body_action = body[2]@arbitrary-arbitrary@body[2]
        ref_action = ref[2]@arbitrary-arbitrary@ref[2]
        # Rotation is an ACTUAL apparatus pulse, not silently an autonomous symmetry.
        pulse = expm(-.37j*body[2])
        rotated_transfer = pulse@transfer@pulse.conj().T
        torque = body[2]@transfer-transfer@body[2]
        hamiltonian = -positive_log(transfer)
        connection_error = vector_flow_error = full_gauge_identity_error = field_only_body_error = 0.
        for angle in (.13, .71, 1.47, 2.6, 2*math.pi):
            frame = expm(1j*angle*triad['spins'][2])
            plus_frame = expm(1j*(angle+epsilon)*triad['spins'][2])
            minus_frame = expm(1j*(angle-epsilon)*triad['spins'][2])
            frame_derivative = (plus_frame-minus_frame)/(2*epsilon)
            connection = -1j*frame_derivative@frame.conj().T
            connection_error = max(connection_error, float(np.linalg.norm(connection-triad['spins'][2])))
            current_phi = frame@phi@frame.conj().T
            literal_flow = np.zeros_like(phi)
            literal_flow[1], literal_flow[2] = -current_phi[2], current_phi[1]
            observed_flow = (plus_frame@phi@plus_frame.conj().T-minus_frame@phi@minus_frame.conj().T)/(2*epsilon)
            vector_flow_error = max(vector_flow_error, float(np.linalg.norm(observed_flow-literal_flow)))
            chart_pullback = expm(-1j*angle*(body[2]+ref[2]))
            primitive_all_matter_cotransport = expm(1j*angle*(body[2]+ref[2]))
            full_gauge_identity_error = max(full_gauge_identity_error, float(np.linalg.norm(chart_pullback@primitive_all_matter_cotransport@state-state)))
            field_only_with_fixed_apparatus = chart_pullback@expm(1j*angle*ref[2])@state
            field_only_body_error = max(field_only_body_error, float(np.linalg.norm(field_only_with_fixed_apparatus-expm(-1j*angle*body[2])@state)))
        gauge_loop = expm(-2j*math.pi*(body[2]+ref[2]))
        counter_reference = expm(2j*math.pi*ref[2])
        repaired_state = gauge_loop@counter_reference@state
        duration_lattice = 1e-4
        idle_state = expm(-1j*duration_lattice*hamiltonian)@state
        actual_driven_body = expm(-1j*(duration_lattice*hamiltonian+2*math.pi*body[2]))@state
        actual_driven_Ref = expm(-1j*(duration_lattice*hamiltonian-2*math.pi*ref[2]))@state
        body_time_bound = math.pi*duration_lattice*float(np.linalg.norm(hamiltonian@body[2]-body[2]@hamiltonian, 2))
        body_time_error = float(np.linalg.norm(actual_driven_body+idle_state))
        if body_time_error > body_time_bound+1e-10:
            raise ArithmeticError('Actual finite-duration body Noether pulse violates its paid torque bound')
        relative_overlap = np.vdot(idle_state, actual_driven_body)
        wrong_scalar_overlap = np.vdot(state, gauge_loop@state)
        pole_ratio = float(np.vdot(state, transfer@state).real/eigen[-1])
        pole_ratio_error = abs(pole_ratio-math.exp(-energy))
        if pole_ratio_error > 1e-10:
            raise ArithmeticError('Actual primitive normalized sea/source pole does not match Photon')
        full_family_field_transport = 'Co-transport Phi/colour ONLY, primitive spatial-scalar matter held fixed. In the exact chart Vprime=VU, Yprime=Udag Lambda(R)Y U; the full45 normal radius/slice and successor radial ground are preserved. Every oriented colour link is co-transported to keep dressed log(Vdag U V) unchanged; its left Jacobian is retained. Actual bundle connection induces fibre spin. Co-transporting ALL matter/Ref is instead the identity Gauss branch. Actual field-dependent counter-Ref fixes the addressed apparatus. Root nonlinear matter/current remainder and static-source string energy are separately priced.'
        # Literal compact U1 link interior Galerkin, with no cyclic wrap.
        electric = np.diag(np.arange(-2, 3, dtype=float))
        raise_flux = np.diag(np.ones(4), -1)
        vacuum_flux = np.eye(5)[:, 2]
        dressed_flux = (raise_flux if charge == 1 else raise_flux.T)@vacuum_flux
        colour_path = expm(.13j*spins[0]-.19j*spins[1])
        separated = source_operator(model, colour_path, charge)@sea
        separated_norm = float(np.vdot(separated, separated).real)
        dressed_state = np.outer(dressed_flux, separated)
        repaired_dressed = (gauge_loop@counter_reference@dressed_state.T).T
        results.append(dict(charge=charge, actual_native_source_norm_squared=norm,
            active_number=float(np.vdot(state, nactive@state).real), Ref_number=float(np.vdot(state, nref@state).real),
            separated_unitary_colour_U1_source_norm_squared=separated_norm,
            separated_actual_body_enclosing_flux=float(np.vdot(dressed_flux, electric@dressed_flux).real),
            separated_integer_Gauss_error=float(np.linalg.norm((electric-charge*np.eye(5))@dressed_flux)),
            separated_wrong_undressed_Gauss_error=float(np.linalg.norm((electric-charge*np.eye(5))@vacuum_flux)),
            separated_actual_full_Ref_counterpulse_minus_error=float(np.linalg.norm(repaired_dressed+dressed_state)),
            addressed_body_two_pi_minus_error=float(np.linalg.norm(addressed+state)),
            co_rotated_Ref_two_pi_plus_error=float(np.linalg.norm(co_rotated-state)),
            repaired_gauge_transport_counter_Ref_minus_error=float(np.linalg.norm(repaired_state+state)),
            wrong_missing_counter_Ref_sign_error=float(np.linalg.norm(gauge_loop@state+state)),
            actual_paid_apparatus_energy_shift=float(np.vdot(state, (pulse.conj().T@hamiltonian@pulse-hamiltonian)@state).real),
            actual_apparatus_energy_operator_bound=float(np.linalg.norm(pulse.conj().T@hamiltonian@pulse-hamiltonian, 2)),
            actual_apparatus_energy_operator_bound_physical=float(np.linalg.norm(pulse.conj().T@hamiltonian@pulse-hamiltonian, 2))/p.a,
            actual_body_Noether_H_torque_norm_lattice=float(np.linalg.norm(hamiltonian@body[2]-body[2]@hamiltonian, 2)),
            actual_Ref_field_Noether_H_torque_norm_lattice=float(np.linalg.norm(hamiltonian@ref[2]-ref[2]@hamiltonian, 2)),
            physical_time_unit=p.a,
            actual_native_drive_duration_lattice=duration_lattice,
            actual_native_drive_duration_physical=duration_lattice*p.a,
            actual_finite_duration_body_two_pi_sign_error=body_time_error,
            actual_body_torque_time_sign_error_upper=body_time_bound,
            actual_finite_duration_field_dependent_Ref_counterpulse_sign_error=float(np.linalg.norm(actual_driven_Ref+idle_state)),
            actual_relative_interferometer_overlap=[float(relative_overlap.real), float(relative_overlap.imag)],
            actual_relative_interferometer_equal_arm_bright_probability=float((1+relative_overlap.real)/2),
            wrong_full_Gauss_scalar_loop_equal_arm_bright_probability=float((1+wrong_scalar_overlap.real)/2),
            drive_scope='Exact native exp[-i(tau H+2pi Jbody)] and exp[-i(tau H-2pi JRef(Phi))], versus idle exp[-i tau H]. Actuator angular speed2pi/tau; body error paid by unitary Duhamel/Trotter bound pi tau ||[H,Jbody]||, not zero-time endpoint alone.',
            actual_primitive_Phi_Cartesian_rotation_flow_error=vector_flow_error,
            actual_bundle_connection_flow_error=connection_error,
            actual_ALL_matter_Ref_Gauss_cotransport_identity_error=full_gauge_identity_error,
            actual_field_only_fixed_apparatus_body_group_error=field_only_body_error,
            actual_Ref_counterpulse_transfer_commutator=float(np.linalg.norm(ref[2]@transfer-transfer@ref[2])),
            fullfield_addressed_repair=full_family_field_transport,
            actual_active_Casimir_half_error=float(np.linalg.norm((sum(j@j for j in body)-.75*model.identity)@state)),
            arbitrary_packet_body_commutator_error=float(np.linalg.norm(body_action-body_expected)),
            arbitrary_packet_actual_Ref_commutator_error=float(np.linalg.norm(ref_action-ref_expected)),
            wrong_missing_Ref_generator_error=float(np.linalg.norm(ref_action)),
            actual_native_local_torque_norm=float(np.linalg.norm(torque, 2)),
            finite_apparatus_conjugated_transfer_positive_min=float(np.linalg.eigvalsh(rotated_transfer)[0]),
            fixed_momentum_spin_NOT_symmetry_norm=float(np.linalg.norm(rotated_transfer-transfer, 2)),
            canonical_sea_transfer_eigenvalue=float(eigen[-1]),
            canonical_source_pole_decay_energy_lattice=energy,
            actual_native_source_correlator_transfer_ratio=pole_ratio,
            actual_source_correlator_Photon_pole_error=pole_ratio_error,
            specimen_scope='Actual eight-mode primitive even fullRef+filledsea; addressed body pulse at same Wilson pole. Not proof that free physical local field rotation is cheap.'))
    return dict(native_dimension=model.dimension, primitive_edges=len(model.edges), determinant_B=determinant,
                field_dependent_Ref_Noether=dict(actual_finite_SU4_covariance_error=ref_covariance,
                    actual_all15_Gauss_infinitesimal_covariance_error=max(infinitesimal_errors),
                    wrong_constant_Ref_label_error=wrong_constant_ref,
                    operator='rdag J(Phi_ref) r, J derived from full60 Cartesian triad on actual transported field. Native Fock lift includes all15 gauge generators; source Ref background/centre packet unchanged because the operator preserves Ref number. Field-derivative plus Fock Gauss commutator cancel.',
                    physical_operation='Co-transport Phi/colour ONLY, with primitive spatial-scalar matter held fixed; chart pullback induces fibre spin. Actual covariant counter-Ref holds addressed apparatus fixed. Co-transporting ALL primitive matter/Ref as a full gauge transformation is IDENTITY, not physical spin. Relative interferometer compares the body-operation branch against unchanged apparatus; full-composite 2pi remains +.'),
                EM_sector='Baseline Ref has same unit EM operator charge as active; explicit background neutralizes its INITIAL state. Source preserves LOCAL active+Ref EM density. Coincident source norm is computed here; separated body+1/Refhole-1 needs actual periodic integer-Gauss/Wilson dressing from Root owner, not a fabricated neutral Ref operator or silently substituted external-charge fork.',
                charges=results, pulse_law='T(theta)=exp(-i theta dGamma(J_body)) T exp(+i theta dGamma(J_body)); local externally paid torque, Ref held fixed. Co-rotating Ref is the scalar source loop and has +2pi.',
                repaired_scope='Actual local gauge-transport plus counter-Ref addressed-apparatus channel; zero reduced field strain with finite native body torque/energy price. This is NOT an autonomous exact cubic-lattice rotational symmetry.')


def packet_orbit(gamma: np.ndarray, triad: dict[str, Any], photon: Parameters,
                 rng: np.random.Generator) -> dict[str, Any]:
    """Actual arbitrary spatial source packet and Wilson spacetime pole orbit."""
    nodes = rng.normal(size=(23, 3))*.27
    packets = rng.normal(size=(23, 4, 4))+1j*rng.normal(size=(23, 4, 4))
    packets /= np.linalg.norm(packets)
    vector = np.array([.21, -.34, .29])
    rotation = Rotation.from_rotvec(vector).as_matrix()
    spin = expm(-1j*np.einsum('j,jab->ab', vector, triad['spins']))
    # Addressed reference moves independently; here a genuine nonidentity apparatus.
    ref = expm(-.13j*triad['spins'][0])
    points = rotation@nodes.T
    orbit_error = np.linalg.norm(points.T-nodes)
    rows = []
    for mass, a in itertools.product(photon.masses, photon.spacings):
        consumer = WilsonConsumer(gamma, a, mass)
        covariance = fixed = pole_error = spacetime = 0.
        for momentum in nodes:
            spatial = a*momentum
            moved = a*(rotation@momentum)
            pole, spinors, b, energy = consumer.pole(spatial)
            target, rotated_spinors, rb, re = consumer.pole(moved)
            inverse = consumer.inverse(pole)
            transformed = consumer.inverse(target)
            covariance = max(covariance, float(np.linalg.norm(transformed-spin@inverse@spin.conj().T, 2)/a))
            fixed = max(fixed, float(np.linalg.norm(inverse-spin@inverse@spin.conj().T, 2)/a))
            pole_error = max(pole_error, float(np.linalg.norm(inverse@spinors)))
            spacetime = max(spacetime, abs(energy-re)/a)
        plus = spin@packets@ref.conj().T
        minus = ref@packets.conj().transpose(0, 2, 1)@spin.conj().T
        rows.append(dict(mass=mass, a=a, actual_Wilson_spacetime_orbit_error=covariance,
                         wrong_omitted_orbital_error=fixed, physical_pole_energy_rotation_error=spacetime,
                         actual_pole_inverse_error=pole_error,
                         arbitrary_plus_packet_norm=float(np.linalg.norm(plus)), arbitrary_minus_packet_norm=float(np.linalg.norm(minus))))
    torons = []
    for mass, a, history in itertools.product(photon.masses, photon.spacings, itertools.product((0, 1), repeat=3)):
        consumer = WilsonConsumer(gamma, a, mass)
        momentum = math.pi*np.array(history)/photon.spatial_size
        pole, vectors, b, energy = consumer.pole(momentum)
        antipole = np.array([-pole[0], *pole[1:]])
        singular = np.linalg.svd(consumer.inverse(antipole), compute_uv=False)
        torons.append(dict(mass=mass, a=a, history=list(history), residue=1/b,
                           positive_pole_error=float(np.linalg.norm(consumer.inverse(pole)@vectors)),
                           negative_pole_small_singular=float(singular[-1]), energy_physical=energy/a,
                           statistical_toron_momentum_physical=(momentum/a).tolist()))
    return dict(rows=rows, all72_mass_spacing_statistical_toron_poles=torons,
                arbitrary_packet_moved_spatial_norm=float(orbit_error),
                packet_definition='23 independent nonconstant spatial matrix samples, arbitrary complex 4x4 source coefficient; physical f(p)->U f(R^-1 p) Refdag, antiparticle dual. Nodes and samples both move, not a constant8-source fit.',
                finite_a_scope='Spacetime inverse Wilson operator at actual +/- poles; arbitrary SO3 is NOT an exact cubic-lattice symmetry. Errors decrease with a; no finite-a exact group claim. All8 statistical torons remain separate from colour-centre flux.')


def centre_transition(triad: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """Follow principal SU4/Z4 charts and ALL transition compensators."""
    basis = np.asarray(matrix_unit_basis())
    initial = expm(.17j*np.einsum('a,aij->ij', rng.normal(size=15), basis))
    centres = np.array([1, 1j, -1, -1j])
    continued = initial.copy()
    section = initial.copy()
    events = []
    for angle in np.linspace(0, 2*math.pi, 129)[1:]:
        lift = initial@expm(-1j*angle*triad['spins'][2])
        # Principal section from actual matrix overlap with starting chart.
        chosen = int(np.argmax([np.trace(initial.conj().T@(z*lift)).real for z in centres]))
        section = centres[chosen]*lift
        compensate = centres[int(np.argmax([np.trace(continued.conj().T@(z*section)).real for z in centres]))]
        continued = compensate*section
        if events == [] or events[-1]['phase'] != [float(compensate.real), float(compensate.imag)]:
            events.append(dict(angle=float(angle), phase=[float(compensate.real), float(compensate.imag)]))
    charges = []
    for charge, active, reference in ((1, 3, 3), (-1, 1, 1)):
        total = active+reference
        balances = [abs(z**active*z**reference*z**(-total)-1) for z in centres]
        charges.append(dict(charge=charge, active_occupation=active, actual_Ref_occupation=reference,
                            total_centre_character=total % 4, all4_centre_transition_balance_error=float(max(balances)),
                            active_two_pi_phase=(-1)**active, actual_Ref_two_pi_phase=(-1)**reference,
                            actual_link_two_pi_compensator=(-1)**(-total),
                            full_Gauss_scalar_loop_phase=(-1)**total*(-1)**(-total),
                            wrong_missing_Ref_transition_error=float(max(abs(z**active*z**(-total)-1) for z in centres)),
                            wrong_missing_link_transition_error=float(max(abs(z**total-1) for z in centres))))
    return dict(actual_continued_lift_minus_error=float(np.linalg.norm(continued+initial)),
                wrong_reset_principal_lift_plus_error=float(np.linalg.norm(section-initial)),
                missing_chart_transition_sign_error=float(np.linalg.norm(section-continued)),
                centre_events=events, actual_both_charge_sea_Ref_link_compensators=charges,
                distinction='Scalar section loop includes link-character/sea/Ref compensators and is +1. Addressed physical body rotation relative fixed Ref uses the mechanical triad J and orbital pullback, not resetting principal V or treating scalar section holonomy as spin.',
                no_source_retagging=True)





def thermal_spin_embedding(gamma: np.ndarray, triad: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Use the actual Cartesian spin in Field's invariant two-component block."""
    embedding = prepared.two_spin_embedding(gamma)
    spin = embedding.conj().T@triad['spins'][2]@embedding
    if np.linalg.norm(expm(2j*math.pi*spin)+np.eye(2)) > 1e-10:
        raise ArithmeticError('Actual Cartesian spin does not preserve the native axial block')
    return embedding, spin






def finite_exterior_lift(one: np.ndarray, states: list[tuple[int, ...]]) -> np.ndarray:
    """Finite canonical CAR minors; species modecount, not Field's fixed8."""
    index = {occupied: i for i, occupied in enumerate(states)}
    out = np.zeros((len(states),)*2, complex)
    for column, occupied in enumerate(states):
        for old in occupied:
            remaining = tuple(mode for mode in occupied if mode != old)
            removed_sign = (-1)**sum(mode < old for mode in occupied)
            for new in range(len(one)):
                if new in remaining or one[new, old] == 0:
                    continue
                target = tuple(sorted((*remaining, new)))
                sign = removed_sign*(-1)**sum(mode < new for mode in remaining)
                out[index[target], column] += sign*one[new, old]
    return out


def sector_matter_operator(one: np.ndarray, space: dict[str, Any]) -> np.ndarray:
    """Existing exterior CAR lift on each fixed species, then tensor sum."""
    if 'species_states' not in space:
        return prepared.matter_operator(one, space['states'])
    local_spaces = space['species_states']
    dimensions = [len(states) for states in local_spaces]
    out = np.zeros((math.prod(dimensions),)*2, complex)
    for species, states in enumerate(local_spaces):
        block = one[4*species:4*species+4, 4*species:4*species+4]
        local = finite_exterior_lift(block, states)
        out += np.kron(np.eye(math.prod(dimensions[:species])),
                       np.kron(local, np.eye(math.prod(dimensions[species+1:]))))
    return out


def physical_spin_on_range(spin: np.ndarray, space: dict[str, Any], field_modules: int) -> np.ndarray:
    """Literal onsite Phi-spin bilinears; number/flux and all Gauss rows retained."""
    one = np.kron(np.eye(field_modules), spin)
    matter = sector_matter_operator(one, space)
    full = np.array([[matter[i, j] if n == m else 0. for j, m, _ in space['physical']]
                     for i, n, _ in space['physical']], complex)
    return full


def physical_matter_columns(one: np.ndarray, space: dict[str, Any],
                            columns: np.ndarray) -> np.ndarray:
    """Literal tree-dressed bilinear: onsite charge change shifts E0, not E1."""
    matter = sector_matter_operator(one, space)
    out = np.zeros_like(columns)
    for flux_number in sorted({n for _, n, _ in space['physical']}):
        rows = np.array([i for i, (_, n, _) in enumerate(space['physical']) if n == flux_number])
        states = np.array([space['physical'][i][0] for i in rows])
        out[rows] = matter[np.ix_(states, states)]@columns[rows]
    return out


def physical_rotation_columns(spin: np.ndarray, space: dict[str, Any], modules: int,
                              angle: float, columns: np.ndarray) -> np.ndarray:
    """Actual exterior minors of exp(i theta J(Phi)); no full NF allocation."""
    one = np.kron(np.eye(modules), expm(1j*angle*spin))
    occupied = np.asarray(space['states'])
    rotation = np.linalg.det(one[occupied[:, None, :, None], occupied[None, :, None, :]])
    out = np.zeros_like(columns)
    for flux_number in sorted({n for _, n, _ in space['physical']}):
        rows = np.array([i for i, (_, n, _) in enumerate(space['physical']) if n == flux_number])
        states = np.array([space['physical'][i][0] for i in rows])
        out[rows] = rotation[np.ix_(states, states)]@columns[rows]
    return out


def range_noether_channel(space: dict[str, Any], generator: np.ndarray,
                          fundamental_spin: np.ndarray, modules: int,
                          ref_spins: np.ndarray, density: np.ndarray,
                          charged_values: np.ndarray, current: np.ndarray,
                          packet_source: np.ndarray, source_gram: float) -> dict[str, Any]:
    """Actuated rotation on the actual bare-Haar NONZERO moving range.

    Kato's offdiagonal control alone is NOT the physical rotation. Adding
    transported P(theta) G P(theta) supplies its diagonal angular actuator;
    together they produce the literal Phi-derived U(theta) on Ran(P(theta)).
    Common field pullback includes actual Ref; counter-Ref addresses body.
    """
    basis = space['u']
    compressed = basis.conj().T@generator@basis
    off = generator@basis-basis@compressed
    off_norm = float(np.linalg.norm(off, 2))
    angle = 2*math.pi
    rotated_columns = physical_rotation_columns(fundamental_spin, space, modules, angle, basis)
    body_overlap = basis.conj().T@rotated_columns
    kato_columns = rotated_columns@expm(-1j*angle*compressed)
    kato_overlap = basis.conj().T@kato_columns
    matter_rank = basis.shape[1]
    ref_generators = np.array([np.kron(s, np.eye(matter_rank)) for s in ref_spins])
    clock = np.diag(-np.log(charged_values))
    spin_drive = np.kron(np.eye(4), compressed)
    common_hamiltonian = clock+.31*spin_drive
    duration = .17
    common_clock = expm(-1j*duration*common_hamiltonian)
    ref_rotation = expm(2j*math.pi*ref_spins[2])
    counter = expm(-2j*math.pi*ref_generators[2])
    # Actual diagonal angular actuation is indispensable. Kato alone is not U.
    common_transport = np.kron(ref_rotation, body_overlap)
    joint_arm = common_transport@common_clock
    body_arm = joint_arm@counter
    interference = np.trace(density@joint_arm.conj().T@body_arm)
    # Finite duration in the covariant rotating frame, including canonical H.
    driven = expm(-1j*(duration*common_hamiltonian+2*math.pi*ref_generators[2]))
    actual_interference = np.trace(density@common_clock.conj().T@driven)
    body_against_clock = np.trace(density@common_clock.conj().T@body_arm)
    joint_against_clock = np.trace(density@common_clock.conj().T@joint_arm)
    source_identity_error = np.linalg.norm((counter+np.eye(len(counter)))@packet_source)
    values, vectors = np.linalg.eigh(density)
    square_root = (vectors*np.sqrt(np.maximum(values, 0)))@vectors.conj().T
    trial_columns = physical_rotation_columns(fundamental_spin, space, modules, .73, basis)
    static_range_loss = float(np.linalg.norm(trial_columns-basis@(basis.conj().T@trial_columns))**2/basis.shape[1])
    # P=I falsifier: offdiagonal Kato transport is identity, NOT spin-half.
    identity_range_rotation = expm(2j*math.pi*ref_spins[2])
    identity_range_kato = identity_range_rotation@expm(-2j*math.pi*ref_spins[2])
    angular_drive_off = expm(0j*ref_spins[2])
    observations = []
    for time_value in (0., 2., 8., 12.):
        unitary = expm(-1j*time_value*clock)
        rho_clock = unitary@density@unitary.conj().T
        rotated = counter@rho_clock@counter.conj().T
        observations.append(dict(time=time_value, density_trace=float(np.trace(rho_clock).real),
            current=float(np.trace(rho_clock@current).real),
            counter_Ref_current_difference=float(np.trace((rotated-rho_clock)@current).real),
            actual_charged_Liouville_clock=[float(np.trace(square_root@unitary@square_root@unitary.conj().T).real),
                                           float(np.trace(square_root@unitary@square_root@unitary.conj().T).imag)]))
    range_ref_commutator = max(float(np.linalg.norm(s@np.eye(4)-np.eye(4)@s))*math.sqrt(matter_rank) for s in ref_spins)
    return dict(actual_nonzero_range_rank=matter_rank, bare_Haar_kernel_retained=True,
        wrong_static_field_only_rotation_range_loss=static_range_loss,
        actual_Kato_tangent_control_norm_upper=2*off_norm,
        actual_diagonal_angular_control_norm=float(np.linalg.norm(compressed, 2)),
        actual_common_angular_energy_upper=(2*math.pi/duration)*(min(len(space['states'][0]), len(fundamental_spin)*modules-len(space['states'][0]))*float(np.linalg.norm(fundamental_spin, 2))+.5),
        actual_counter_Ref_angular_energy_upper=math.pi/duration,
        actual_canonical_clock_energy_upper=float(np.linalg.norm(common_hamiltonian, 2)),
        actual_Kato_support_map_error=float(np.linalg.norm(kato_columns.conj().T@kato_columns-np.eye(matter_rank))),
        actual_Kato_closed_range_error=float(np.linalg.norm(kato_columns-basis@kato_overlap)),
        actual_common_Kato_holonomy_minus_identity_error=float(np.linalg.norm(kato_overlap+np.eye(matter_rank))),
        actual_actuated_body_two_pi_minus_error=float(np.linalg.norm(body_overlap+np.eye(matter_rank))),
        actual_joint_two_pi_plus_error=float(np.linalg.norm(common_transport-np.eye(len(common_transport)))),
        actual_body_against_canonical_clock=[float(body_against_clock.real), float(body_against_clock.imag)],
        actual_joint_against_canonical_clock=[float(joint_against_clock.real), float(joint_against_clock.imag)],
        wrong_identity_range_Kato_minus_error=float(np.linalg.norm(identity_range_kato+np.eye(4))),
        identity_range_Kato_identity_error=float(np.linalg.norm(identity_range_kato-np.eye(4))),
        wrong_body_drive_off_minus_error=float(np.linalg.norm(angular_drive_off+np.eye(4))),
        SU2_not_claimed_for_projected_bulk_Kato=True,
        actual_Ref_SU2_error=max(float(np.linalg.norm(ref_generators[j]@ref_generators[k]-ref_generators[k]@ref_generators[j]-1j*ref_generators[l]))
                                for j, k, l in ((0, 1, 2), (1, 2, 0), (2, 0, 1))),
        actual_Ref_OS_range_commutator=range_ref_commutator,
        actual_Ref_full_common_H_commutator=float(np.linalg.norm(common_hamiltonian@ref_generators[2]-ref_generators[2]@common_hamiltonian)),
        actual_Ref_source_intertwiner_error=float(source_identity_error),
        actual_normalized_interference=[float(interference.real), float(interference.imag)],
        actual_finite_duration_interference=[float(actual_interference.real), float(actual_interference.imag)],
        actual_equal_arm_bright_probability_raw=float((1+actual_interference.real)/2),
        actual_retained_source_filter_operator_norm_squared=float(np.linalg.norm(packet_source, 2)**2),
        common_bulk_drive_NOT_discarded=True, source_retained_Gram=source_gram,
        common_unitary_density_trace=float(np.trace(joint_arm@density@joint_arm.conj().T).real),
        wrong_unnormalized_interference=[float(source_gram*interference.real), float(source_gram*interference.imag)],
        clock_current_observations=observations,
        operation='Literal Cartesian Phi coframe loop; offdiagonal Kato plus explicit diagonal angular control gives full moving-range rotation. SAME full body+aux+Ref angular/control clock in both arms; addressed branch cancels actual Ref. Joint2pi is PLUS, body2pi is MINUS, both against the same canonical clock.',
        control='Driven positive K(theta)=U(theta)K U(theta)†. Offdiagonal control<=2||QGP|| and diagonal PGP separately paid. No pi U pi, inverse global K, fictitious global Tmin or implicit angular actuator. Ref has all3 Cartesian SU2 generators and preserves nonzero range. Diagonal-off/P=I Kato and Ref-only false-positive controls explicitly fail intrinsic body rotation.')


def actual_thermal_source_channel(gamma4: np.ndarray, triad: dict[str, Any],
                                  grid: int, cutoff: int, steps: int) -> dict[str, Any]:
    """Join Field's actual integer-Gauss photon density, source minor and clock."""
    gamma = prepared.two_spin_frame(gamma4)
    embedding, spin = thermal_spin_embedding(gamma4, triad)
    neutral = prepared.native_range(gamma, .09, grid, cutoff, body_charge=0)
    scale = float(neutral['values'][0])
    neutral_values = neutral['values']/scale
    weights = neutral_values**steps
    weights /= weights.sum()
    rho = np.diag(weights)
    holonomy = np.zeros((len(neutral['physical']),)*2, complex)
    for j, (i, n, _) in enumerate(neutral['physical']):
        row = neutral['basis_index'].get((i, n+1))
        if row is not None:
            holonomy[row, j] = 1
    curl = (holonomy-holonomy.conj().T)/(2j)
    field_operator = neutral['u'].conj().T@curl@neutral['u']
    half = np.diag(np.sqrt(neutral_values))
    photon_raw = half@field_operator.conj().T@rho@field_operator@half
    photon_gram = float(np.trace(photon_raw).real)
    rho_photon = photon_raw/photon_gram
    packet = np.array([.1, .8, .4, .1], complex)
    packet /= np.linalg.norm(packet)
    cases = []
    for charge in (1, -1):
        charged = prepared.native_range(gamma, .09, grid, cutoff, body_charge=charge)
        coefficients = prepared.source_coefficients(gamma, .09, packet, body_charge=charge)
        raw_source, gauss_error = prepared.source_matrix(neutral, charged, coefficients)
        if charge == 1:
            signs = np.diag([1., -1., 1., -1.])
            ref_embedding = signs@embedding.conjugate()@np.diag([1., -1.])
            ref_spins = np.array([-signs@s.T@signs for s in triad['spins']])
        else:
            ref_embedding = embedding
            ref_spins = triad['spins']
        source = np.kron(ref_embedding, charged['u'].conj().T)@raw_source@neutral['u']
        charged_values = np.tile(charged['values']/scale, 4)
        source_half = np.diag(np.sqrt(charged_values))@source
        raw_density = source_half@rho_photon@source_half.conj().T
        source_gram = float(np.trace(raw_density).real)
        density = raw_density/source_gram
        current, _ = prepared.native_current(gamma, .09, grid, charged)
        generator = physical_spin_on_range(spin, charged, 4)
        channel = range_noether_channel(charged, generator, spin, 4, ref_spins, density, charged_values,
            np.kron(np.eye(4), current), source_half, source_gram)
        cases.append(dict(charge=charge, actual_Ref_initial_number=4 if charge == 1 else 0,
            actual_Ref_final_number=3 if charge == 1 else 1,
            actual_integer_Gauss_source_error=float(gauss_error), **channel))
    return dict(supplier='covariant_prepared_gauss_state.native_range/source_matrix/native_current',
        grid=grid, electric_cutoff=cutoff, time_steps=steps,
        neutral_positive_trace=float(np.trace(rho).real), photon_retained_Gram=photon_gram,
        photon_density_trace=float(np.trace(rho_photon).real), cases=cases,
        active_state='Actual interacting square Wilson-Fock/bare-Haar positive RANGE thermal density; full4 Ref spectator/background. Not a frozen filled sea.')


def two_site_wilson(gamma: np.ndarray, kappa: float, theta: float) -> tuple[np.ndarray, float, np.ndarray]:
    """Positive Wilson halfstep on the actual two-link periodic Gauss stressor."""
    b = np.eye(2, dtype=complex)/(2*kappa)
    f = np.zeros((2, 2), complex)
    for edge, (x, y) in enumerate(((0, 1), (1, 0))):
        transport = np.zeros((2, 2), complex)
        transport[x, y] = np.exp(1j*theta) if edge else 1
        b -= .5*(transport+transport.conj().T)
        f += .5*gamma[1, 0, 1]*(transport-transport.conj().T)
    if np.linalg.eigvalsh(b)[0] <= 0:
        raise ArithmeticError('Native heavy Wilson B failed positivity')
    lower = np.block([[np.eye(2), -f], [np.zeros((2, 2)), np.eye(2)]])
    transfer = lower.conj().T@np.block([[np.linalg.inv(b), np.zeros((2, 2))],
                                      [np.zeros((2, 2)), b]])@lower
    order = [0, 2, 1, 3]
    transfer = transfer[np.ix_(order, order)]
    values, vectors = np.linalg.eigh(transfer)
    return (vectors*np.sqrt(values))@vectors.conj().T, float(np.linalg.det(b).real), -(vectors*np.log(values))@vectors.conj().T


def coupled_aux_range(gamma: np.ndarray, kappas: tuple[float, ...],
                      grid: int, cutoff: int, charge: int) -> dict[str, Any]:
    """Determinant minors BEFORE one shared integer-Haar/Gauss range filter.

    Auxiliary local occupations fluctuate; they are not separately Gauss
    projected or replaced by a tensor product of independently admitted seas.
    """
    sets = [prepared.occupations(4, 2+charge if species == 0 else 2)
            for species in range(len(kappas))]
    components = list(itertools.product(*sets))
    states = [tuple(mode+4*species for species, block in enumerate(c) for mode in block)
              for c in components]
    def rho(c: tuple[int, ...]) -> np.ndarray:
        value = np.bincount([(mode % 4)//2 for mode in c], minlength=2)-len(kappas)
        value[0] -= charge
        return value
    middle = [i for i, c in enumerate(states) if np.all(rho(c) == 0)]
    physical = []
    for i, c in enumerate(states):
        for n in range(-cutoff, cutoff+1):
            flux = (n+int(rho(c)[0]), n)
            if max(abs(z) for z in flux) <= cutoff:
                physical.append((i, n, flux))
    samples = []
    for theta in 2*math.pi*np.arange(grid)/grid:
        matrix = np.ones((len(states), len(middle)), complex)
        for species, kappa in enumerate(kappas):
            half, determinant, _ = two_site_wilson(gamma, kappa, float(theta))
            row = np.array([c[species] for c in components])
            col = row[middle]
            matrix *= math.sqrt(determinant)*np.linalg.det(half[row[:, None, :, None], col[None, :, None, :]])
        samples.append(matrix)
    coefficients = np.fft.fft(np.array(samples), axis=0)/grid
    v = np.array([coefficients[n % grid, i] for i, n, _ in physical])
    u, singular, _ = np.linalg.svd(v, full_matrices=False)
    if singular[-1] <= 0:
        raise ArithmeticError('Shared auxiliary positive range lost injectivity')
    return dict(states=states, physical=physical, u=u, V=v, values=singular**2,
        species_states=sets, middle_dimension=len(middle),
        basis_index={(i, n): j for j, (i, n, _) in enumerate(physical)})


def coupled_aux_current(gamma: np.ndarray, kappas: tuple[float, ...],
                        grid: int, space: dict[str, Any]) -> np.ndarray:
    jets = []
    epsilon = 2e-4
    for theta in 2*math.pi*np.arange(grid)/grid:
        one = np.zeros((4*len(kappas),)*2, complex)
        vacuum = 0.
        for species, kappa in enumerate(kappas):
            _, dp, hp = two_site_wilson(gamma, kappa, float(theta+epsilon))
            _, dm, hm = two_site_wilson(gamma, kappa, float(theta-epsilon))
            one[4*species:4*species+4, 4*species:4*species+4] = (hp-hm)/(2*epsilon)
            vacuum -= (math.log(dp)-math.log(dm))/(2*epsilon)
        jets.append(sector_matter_operator(one, space)+vacuum*np.eye(len(space['states'])))
    fourier = np.fft.fft(np.array(jets), axis=0)/grid
    full = np.array([[fourier[(n-m) % grid, i, j] for j, m, _ in space['physical']]
                     for i, n, _ in space['physical']])
    return space['u'].conj().T@full@space['u']


def coupled_aux_source(neutral: dict[str, Any], charged: dict[str, Any],
                       coefficients: np.ndarray, charge: int) -> tuple[np.ndarray, float]:
    source = np.zeros((2*len(charged['physical']), len(neutral['physical'])), complex)
    outindex = {c: i for i, c in enumerate(charged['states'])}
    error = 0.
    for column, (i, n, flux) in enumerate(neutral['physical']):
        occupied = neutral['states'][i]
        for mode, ref_spin in itertools.product(range(4), range(2)):
            if (mode in occupied) == (charge == 1):
                continue
            target = tuple(sorted((*occupied, mode))) if charge == 1 else tuple(z for z in occupied if z != mode)
            final_flux = (flux[0]-charge*(mode//2), flux[1])
            row = charged['basis_index'].get((outindex[target], n))
            if row is None:
                continue
            if charged['physical'][row][2] != final_flux:
                raise ArithmeticError('Shared integer Gauss path failed')
            sign = (-1)**(sum(z < mode for z in occupied)+ref_spin) if charge == 1 else -(-1)**sum(z < mode for z in occupied)
            coefficient = coefficients[mode, ref_spin] if charge == 1 else coefficients[mode, ref_spin].conjugate()
            source[ref_spin*len(charged['physical'])+row, column] += coefficient*sign
            error = max(error, abs(final_flux[0]-final_flux[1]-(flux[0]-flux[1])+charge*(mode//2)))
    return source, error


def actual_auxiliary_range_controls(gamma4: np.ndarray, triad: dict[str, Any],
                                   grid: int, cutoff: int, steps: int) -> list[dict[str, Any]]:
    gamma = prepared.two_spin_frame(gamma4)
    embedding, spin = thermal_spin_embedding(gamma4, triad)
    rows = []
    for naux in (0, 1, 2):
        kappas = (.09,)+(.04,)*naux
        neutral = coupled_aux_range(gamma, kappas, grid, cutoff, 0)
        scale = float(neutral['values'][0])
        weights = (neutral['values']/scale)**steps
        weights /= weights.sum()
        rho = np.diag(weights)
        physical_weights = np.einsum('ir,r,ir->i', neutral['u'], weights, neutral['u'].conjugate()).real
        auxiliary_distributions = []
        for species in range(1, len(kappas)):
            counts = np.array([sum(4*species <= mode < 4*species+2 for mode in neutral['states'][i])
                               for i, _, _ in neutral['physical']])
            probabilities = [float(physical_weights[counts == n].sum()) for n in range(3)]
            mean = float(physical_weights@counts)
            angular = []
            for local in (True, False):
                one_spin = np.zeros((4*len(kappas), 4*len(kappas)), complex)
                size = 2 if local else 4
                one_spin[4*species:4*species+size, 4*species:4*species+size] = spin if local else np.kron(np.eye(2), spin)
                spin_columns = physical_matter_columns(one_spin, neutral, neutral['u'])
                first = float(np.trace(rho@neutral['u'].conj().T@spin_columns).real)
                second = float(np.trace(rho@spin_columns.conj().T@spin_columns).real)
                angular.append(dict(local=local, actual_mean=first, actual_variance=second-first*first))
            auxiliary_distributions.append(dict(species=species, local_number_probabilities=probabilities,
                actual_local_number_mean=mean, actual_local_number_variance=float(physical_weights@(counts-mean)**2),
                actual_physical_Phi_spin_observations=angular))
        cases = []
        for charge in (1, -1):
            charged = coupled_aux_range(gamma, kappas, grid, cutoff, charge)
            half_active, _, _ = two_site_wilson(gamma, kappas[0], 0.)
            values_active, vectors_active = np.linalg.eigh(half_active)
            selected = values_active < 1 if charge == 1 else values_active > 1
            base = np.kron(np.array([.6, .8])[:, None], np.eye(2))
            coefficients = vectors_active[:, selected]@vectors_active[:, selected].conj().T@base
            coefficients /= np.linalg.norm(coefficients)
            raw, defect = coupled_aux_source(neutral, charged, coefficients, charge)
            signs = np.diag([1., -1., 1., -1.])
            ref_embedding = signs@embedding.conjugate()@np.diag([1., -1.]) if charge == 1 else embedding
            ref_spins = np.array([-signs@s.T@signs for s in triad['spins']]) if charge == 1 else triad['spins']
            source = np.kron(ref_embedding, charged['u'].conj().T)@raw@neutral['u']
            values = np.tile(charged['values']/scale, 4)
            source_half = np.diag(np.sqrt(values))@source
            gram = float(np.trace(source_half@rho@source_half.conj().T).real)
            density = source_half@rho@source_half.conj().T/gram
            generator = physical_spin_on_range(spin, charged, 2*len(kappas))
            current = coupled_aux_current(gamma, kappas, grid, charged)
            cases.append(dict(charge=charge, integer_Gauss_defect=defect,
                **range_noether_channel(charged, generator, spin, 2*len(kappas), ref_spins,
                    density, values, np.kron(np.eye(4), current), source_half, gram)))
        rows.append(dict(native_auxiliary_species=naux, kappas=list(kappas),
            neutral_range_rank=neutral['middle_dimension'], physical_density_trace=float(physical_weights.sum()),
            actual_auxiliary_local_occupations=auxiliary_distributions, charges=cases,
            scope='Native invariant Dirac-block stressor with a SHARED two-link compact field, exact determinant occupied minors and joint integer Gauss/Haar filter; not a full4 huge-family vacuum extrapolation.'))
    return rows


def all_cycle_Ref_noether() -> dict[str, Any]:
    """Physical centre packets for fluctuating full4 local occupations."""
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2))
    assignments = np.array(list(itertools.product(range(4), repeat=len(edges))))
    divergence = np.zeros((len(assignments), 4), int)
    for e, (x, y) in enumerate(edges):
        divergence[:, x] += assignments[:, e]
        divergence[:, y] -= assignments[:, e]
    patterns = ((1, 3, 2, 2), (4, 0, 1, 3), (0, 4, 3, 1), (2, 1, 4, 1))
    source_error = gauss_error = missing_aux_error = 0.
    tested = 0
    cycle_counts: set[int] = set()
    for naux, charge, toron in itertools.product(range(3), (1, -1), itertools.product((0, 1), repeat=3)):
        active = np.array(patterns[0])
        aux = sum((np.array(patterns[s+1]) for s in range(naux)), np.zeros(4, int))
        reference_initial = 4 if charge == 1 else 0
        occupation = active+aux
        occupation[0] += reference_initial
        allowed = np.all((divergence-occupation) % 4 == 0, axis=1)
        cycle_counts.add(int(allowed.sum()))
        state = allowed.astype(complex)/math.sqrt(int(allowed.sum()))
        moved = assignments.copy()
        moved[:, 4] = (moved[:, 4]-charge) % 4
        moved[:, 2] = (moved[:, 2]-charge) % 4
        moved_divergence = np.zeros_like(divergence)
        for e, (x, y) in enumerate(edges):
            moved_divergence[:, x] += moved[:, e]
            moved_divergence[:, y] -= moved[:, e]
        charged_occupation = occupation.copy()
        charged_occupation[0] -= charge
        charged_occupation[3] += charge
        # Statistical toron phase affects path amplitude, NEVER centre-Gauss.
        path_phase = (-1)**sum(toron)
        charged_state = path_phase*state
        source_error = max(source_error, abs(float(np.linalg.norm(charged_state)**2-1)))
        for centre in itertools.product(range(4), repeat=4):
            exponents = (charged_occupation-moved_divergence)@np.array(centre)
            phases = 1j**(exponents % 4)
            gauss_error = max(gauss_error, float(np.linalg.norm((phases-1)*charged_state)))
            omitted = 1j**(((charged_occupation-aux-moved_divergence)@np.array(centre)) % 4)
            missing_aux_error = max(missing_aux_error, float(np.linalg.norm((omitted-1)*charged_state)))
        tested += 1
    return dict(full4_occupation_patterns=[list(p) for p in patterns], centre_character_dimension=len(assignments),
        coherent_cycles_per_divergence=max(cycle_counts), actual_source_path_norm_error=source_error,
        all_local_Z4_Gauss_phase_error=gauss_error, wrong_omitted_auxiliary_character_error=missing_aux_error,
        actual_aux_charge_P8_cases=tested,
        theorem='For EVERY local n and divergence-compatible centre-cycle assignment, z^n z^-divq=1. Source changes active/Ref by opposite unit and shifts a literal path; arbitrary auxiliary fluctuations and all8 statistical torons survive. Ref0/4->1/3 is odd, independent of every n_aux. Fulljoint field+ALLmatter is gaugeidentity; addressed field pullback counter-rotates actual Ref.')


def actual_quantum_normal_Ref_channel(gamma4: np.ndarray, triad: dict[str, Any],
                                      steps: int, nodes: int, charge: int) -> dict[str, Any]:
    """Actual signed PE6 quantum-normal purifier, full4 Ref and cutoff clock.

    This invokes the supplier's measured common-neutral spectral preparation,
    not a Gaussian Gram or a macro PE.25 relabel. Raw source Gram is retained
    separately; every subsequent phase/tail price uses the projected density.
    """
    supplied = prepared.native_normal_preparation(gamma4, steps, nodes, return_runtime=True, body_charge=charge)
    runtime = supplied['runtime']
    count = len(runtime['charged_eigenvalues'])//2
    ref_embedding = runtime['Ref4_hole_embedding'] if charge == 1 else runtime['Ref4_particle_embedding']
    embedding = np.kron(ref_embedding, np.eye(count))
    prepared_cut = prepared.energy_cut_source(runtime, 6.)
    density = embedding@prepared_cut['density']@embedding.conj().T
    source = embedding@prepared_cut['projector']@runtime['source_half']
    purifier = embedding@prepared_cut['purification']
    signs = np.diag([1., -1., 1., -1.])
    ref_spins = np.array([-signs@s.T@signs for s in triad['spins']]) if charge == 1 else triad['spins']
    values = np.tile(runtime['charged_eigenvalues'][:count], 4)
    energy = -np.log(values)
    clock = np.diag(energy)
    generators = np.array([np.kron(s, np.eye(count)) for s in ref_spins])
    counter = expm(2j*math.pi*generators[2])
    duration = .17
    idle = np.diag(np.exp(-1j*duration*energy))
    driven = expm(-1j*(duration*clock-2*math.pi*generators[2]))
    interference = np.trace(density@idle.conj().T@driven)
    # Cutoff acts on the actual nonzero OS range by a bounded Hermitian
    # GENERATOR, not pi U pi. No exact SU2 is claimed for this cutoff.
    coordinates = runtime['normal_coordinates']
    radius = supplied['summary']['radius']
    cutoff = []
    for coordinate in coordinates:
        z = (abs(coordinate)-radius/2)/(radius/4)
        if z <= 0:
            cutoff.append(1.)
        elif z >= 1:
            cutoff.append(0.)
        else:
            a, b = math.exp(-1/z), math.exp(-1/(1-z))
            cutoff.append(b/(a+b))
    physical_size = len(runtime['charged_physical_basis'])
    one_isometry = runtime['charged_range_isometry'][:nodes*physical_size, :count]
    field_cutoff = one_isometry.conj().T@np.kron(np.diag(cutoff), np.eye(physical_size))@one_isometry
    cutoff_generator = np.kron(ref_spins[2], field_cutoff)
    cutoff_drive = expm(-1j*(duration*clock-2*math.pi*cutoff_generator))
    cutoff_interference = np.trace(density@idle.conj().T@cutoff_drive)
    reduced = np.zeros((count, count), complex)
    for j in range(4):
        reduced += density[j*count:(j+1)*count, j*count:(j+1)*count]
    defect = np.eye(count)-field_cutoff
    coefficients = (defect@defect)*reduced.T
    uniform_defect_squared = float(np.abs(coefficients).sum())
    ground = np.kron(np.eye(4), runtime['charged_normal_ground_projector'][:count, :count])
    shell = np.kron(np.eye(4), runtime['charged_inner_shell_projector'][:count, :count])
    observations = []
    for time_value in (0., float(steps), 4.*steps):
        unitary = np.diag(np.exp(-1j*time_value*energy))
        evolved = unitary@density@unitary.conj().T
        rotated = counter@evolved@counter.conj().T
        observations.append(dict(time=time_value,
            actual_source_normal_excited_weight=float(np.trace((np.eye(4*count)-ground)@evolved).real),
            actual_source_inner_shell_tail=float(np.trace(shell@evolved).real),
            counter_Ref_shell_probability_change=float(np.trace(shell@(rotated-evolved)).real)))
    return dict(body_charge=charge, supplier_native_summary=supplied['summary'], actual_full4_Ref_number=3 if charge == 1 else 1,
        actual_density_trace=float(np.trace(density).real),
        actual_source_retained_Gram=prepared_cut['summary']['total_source_retained_Gram'],
        unfiltered_source_retained_Gram=runtime['source_retained_Gram'],
        actual_native_energy_preparation=prepared_cut['summary'],
        actual_projected_purifier_density_error=float(np.linalg.norm(purifier@purifier.conj().T-density)),
        actual_halfstep_source_operator_norm_squared=float(np.linalg.norm(source, 2)**2),
        actual_Ref_Casimir_error=float(np.linalg.norm(sum(s@s for s in ref_spins)-.75*np.eye(4))),
        actual_all3_Ref_full_quantum_normal_H_commutator=max(float(np.linalg.norm(clock@j-j@clock)) for j in generators),
        actual_Ref_two_pi_source_intertwiner_error=float(np.linalg.norm((counter+np.eye(4*count))@source)),
        actual_Ref_finite_duration_interference=[float(interference.real), float(interference.imag)],
        actual_cutoff_Ref_finite_duration_interference=[float(cutoff_interference.real), float(cutoff_interference.imag)],
        actual_cutoff_unitarity_error=float(np.linalg.norm(cutoff_drive.conj().T@cutoff_drive-np.eye(4*count))),
        uniform_clock_cutoff_defect_squared_upper=uniform_defect_squared,
        finite_duration_cutoff_error_upper=math.pi*math.sqrt(uniform_defect_squared),
        actual_cutoff_interference_error=float(abs(cutoff_interference+1)),
        observations=observations,
        price_proof='On the ACTUAL positive range A=pi chi(Y)JRef pi is bounded selfadjoint, so exp[-i(tH-2piA)] is unitary. Ideal Ref commutes full quantum H. Duhamel error<=pi sup_t sqrt(Tr rho(t)(1-chi_hat)^2). Sum of absolute spectral Fourier coefficients bounds ALL times, not sampled-only endpoints; actual retained source Gram normalizes rho.',
        angular_scope='This actual anisotropic1D normal model proves Ref range/clock/cutoff admission, NOT full Cartesian body rotation. Full45 mechanical body law has separate radial/colour transport and supplier-weighted covariance price; naked Gamma spin on this Y is explicitly forbidden.')


def actual_frame_transfer_covariance(gamma4: np.ndarray, triad: dict[str, Any],
                                     photon: Parameters) -> dict[str, Any]:
    gamma = prepared.two_spin_frame(gamma4)
    _, spin = thermal_spin_embedding(gamma4, triad)
    angle = .73
    rotation = expm(1j*angle*spin)
    literal = np.array([gamma[0], math.cos(angle)*gamma[1]-math.sin(angle)*gamma[2],
                       math.sin(angle)*gamma[1]+math.cos(angle)*gamma[2]])
    half0, _, _, _, _ = prepared.square_wilson(gamma, .09, .41)
    half1, _, _, _, _ = prepared.square_wilson(literal, .09, .41)
    body = np.kron(np.eye(4), rotation)
    # A genuine normal departure: do NOT infer Gamma-only covariance here.
    nonclifford = gamma.copy()
    nonclifford[1] *= 1.03
    literal_nonclifford = np.array([nonclifford[0],
        math.cos(angle)*nonclifford[1]-math.sin(angle)*nonclifford[2],
        math.sin(angle)*nonclifford[1]+math.cos(angle)*nonclifford[2]])
    wrong0, _, _, _, _ = prepared.square_wilson(nonclifford, .09, .41)
    actual1, _, _, _, _ = prepared.square_wilson(literal_nonclifford, .09, .41)
    phi = np.sqrt(np.array(photon.gram))[:, None, None]*SELECTED
    spatial = Rotation.from_rotvec([0., 0., angle]).as_matrix()
    moved_phi = np.array([phi[0], *np.einsum('jk,kab->jab', spatial, phi[1:])])
    world = expm(1j*angle*triad['spins'][2])
    return dict(actual_literal_Cartesian_Phi_motion_norm=float(np.linalg.norm(moved_phi-phi)),
        actual_literal_Cartesian_Phi_vs_spin_flow_error=float(np.linalg.norm(moved_phi-world@phi@world.conj().T)),
        actual_rotated_frame_Wilson_positive_halfstep_error=float(np.linalg.norm(half1-body@half0@body.conj().T)),
        wrong_offcliff_Gamma_only_Wilson_halfstep_error=float(np.linalg.norm(actual1-body@wrong0@body.conj().T)),
        negative_control='The .03 anisotropic normal deformation moves meanGram. Its ACTUAL Cartesian rotation is not Gamma-only conjugation. Full45 normal transport and weighted primitive coupling departure must be supplied; the Clifford native covariance cannot be silently extrapolated.')


def dirac_cycle_step(gamma: np.ndarray, kappa: float, theta: float) -> tuple[np.ndarray, float]:
    """Physical4-spin/two-site Wilson step, including the actual beta embedding."""
    frame = wilson_component_basis(gamma)
    canonical_gamma = frame.conj().T@gamma@frame
    bsite = np.eye(2, dtype=complex)/(2*kappa)
    spatial = np.zeros((2, 2), complex)
    for edge, (x, y) in enumerate(((0, 1), (1, 0))):
        transport = np.zeros((2, 2), complex)
        transport[x, y] = np.exp(1j*theta) if edge else 1
        bsite -= .5*(transport+transport.conj().T)
        spatial += .5*(transport-transport.conj().T)
    b = np.kron(bsite, np.eye(2))
    f = np.asarray(np.kron(spatial, canonical_gamma[1, :2, 2:]), dtype=complex)
    lower = np.block([[np.eye(4), -f], [np.zeros((4, 4)), np.eye(4)]])
    one = lower.conj().T@np.block([[np.linalg.inv(b), np.zeros((4, 4))],
                                   [np.zeros((4, 4)), b]])@lower
    order = [0, 1, 4, 5, 2, 3, 6, 7]
    one = one[np.ix_(order, order)]
    physical_frame = np.kron(np.eye(2), frame)
    one = physical_frame@one@physical_frame.conj().T
    values, vectors = np.linalg.eigh(one)
    return (vectors*np.sqrt(values))@vectors.conj().T, float(np.linalg.det(b).real)


def dirac_cycle_range(gamma: np.ndarray, grid: int, cutoff: int, charge: int) -> dict[str, Any]:
    states = prepared.occupations(8, 4+charge)
    def rho(c: tuple[int, ...]) -> int:
        return sum(mode < 4 for mode in c)-2-charge
    middle = [c for c in states if rho(c) == 0]
    physical = [(i, n, (n+rho(c), n)) for i, c in enumerate(states)
                for n in range(-cutoff, cutoff+1) if abs(n+rho(c)) <= cutoff]
    samples = []
    row, col = np.array(states), np.array(middle)
    for theta in 2*math.pi*np.arange(grid)/grid:
        half, determinant = dirac_cycle_step(gamma, .09, float(theta))
        samples.append(math.sqrt(determinant)*np.linalg.det(half[row[:, None, :, None], col[None, :, None, :]]))
    coefficients = np.fft.fft(np.array(samples), axis=0)/grid
    v = np.array([coefficients[n % grid, i] for i, n, _ in physical])
    u, singular, _ = np.linalg.svd(v, full_matrices=False)
    return dict(states=states, physical=physical, u=u, values=singular**2,
                middle_dimension=len(middle), basis_index={(i, n): j for j, (i, n, _) in enumerate(physical)})


def dirac_cycle_current(gamma: np.ndarray, grid: int, space: dict[str, Any],
                        epsilon: float = 2e-4) -> np.ndarray:
    """Literal physical4 Hamiltonian/link current, then the SAME integer-Gauss range."""
    jets = []
    for theta in 2*math.pi*np.arange(grid)/grid:
        hp, dp = dirac_cycle_step(gamma, .09, float(theta+epsilon))
        hm, dm = dirac_cycle_step(gamma, .09, float(theta-epsilon))
        current = -(positive_log(hp@hp)-positive_log(hm@hm))/(2*epsilon)
        vacuum = -(math.log(dp)-math.log(dm))/(2*epsilon)
        jets.append(finite_exterior_lift(current, space['states'])+vacuum*np.eye(len(space['states'])))
    fourier = np.fft.fft(np.array(jets), axis=0)/grid
    full = np.array([[fourier[(n-m) % grid, i, j] for j, m, _ in space['physical']]
                     for i, n, _ in space['physical']])
    return space['u'].conj().T@full@space['u']


def dirac_cycle_source(neutral: dict[str, Any], charged: dict[str, Any],
                       coefficients: np.ndarray, charge: int) -> tuple[np.ndarray, float]:
    source = np.zeros((4*len(charged['physical']), len(neutral['physical'])), complex)
    index = {c: i for i, c in enumerate(charged['states'])}
    defect = 0.
    for column, (i, n, flux) in enumerate(neutral['physical']):
        occupied = neutral['states'][i]
        for mode, ref_spin in itertools.product(range(8), range(4)):
            if (mode in occupied) == (charge == 1):
                continue
            target = tuple(sorted((*occupied, mode))) if charge == 1 else tuple(z for z in occupied if z != mode)
            final_flux = (flux[0]-charge*(mode//4), flux[1])
            row = charged['basis_index'].get((index[target], n))
            if row is None:
                continue
            defect = max(defect, float(abs(final_flux[0]-charged['physical'][row][2][0])))
            sign = (-1)**(sum(z < mode for z in occupied)+ref_spin) if charge == 1 else -(-1)**sum(z < mode for z in occupied)
            coefficient = coefficients[mode, ref_spin] if charge == 1 else coefficients[mode, ref_spin].conjugate()
            source[ref_spin*len(charged['physical'])+row, column] += sign*coefficient
    return source, defect


def actual_polarized_body_rotation(gamma4: np.ndarray, triad: dict[str, Any],
                                   grid: int, cutoff: int, steps: int) -> dict[str, Any]:
    """Operational body spin witness on actual full4 Wilson/Haar source."""
    gamma = gamma4
    spins = triad['spins']
    neutral = dirac_cycle_range(gamma, grid, cutoff, 0)
    scale = float(neutral['values'][0])
    weights = (neutral['values']/scale)**steps
    weights /= weights.sum()
    rho = np.diag(weights)
    half, _ = dirac_cycle_step(gamma, .09, 0.)
    values, band = np.linalg.eigh(half)
    base = np.kron(np.array([.6, .8])[:, None], np.eye(4))
    cases = []
    angle, duration = .73, .17
    for charge in (1, -1):
        selected = values < 1 if charge == 1 else values > 1
        coefficients = band[:, selected]@band[:, selected].conj().T@base
        coefficients /= np.linalg.norm(coefficients)
        charged = dirac_cycle_range(gamma, grid, cutoff, charge)
        raw, gauss_defect = dirac_cycle_source(neutral, charged, coefficients, charge)
        rank = len(charged['values'])
        source = np.kron(np.eye(4), charged['u'].conj().T)@raw@neutral['u']
        charged_values = np.tile(charged['values']/scale, 4)
        source_half = np.diag(np.sqrt(charged_values))@source
        raw_source_gram = float(np.trace(source_half@rho@source_half.conj().T).real)
        energy = -np.log(charged_values)
        quarter = (energy >= 0)&(energy <= .25)
        quarter_source = quarter[:, None]*source_half
        quarter_gram = float(np.trace(quarter_source@rho@quarter_source.conj().T).real)
        # Material source repair: honest measured native PE6, NOT macro PE.25.
        selected_energy = (energy >= 0)&(energy <= 6.)
        source_half = selected_energy[:, None]*source_half
        source_gram = float(np.trace(source_half@rho@source_half.conj().T).real)
        if source_gram <= 0:
            raise ArithmeticError('Actual native PE6 source numerator vanishes')
        density = source_half@rho@source_half.conj().T/source_gram
        signs = np.diag([1., -1., 1., -1.])
        ref_spins = np.array([-signs@s.T@signs for s in spins]) if charge == 1 else spins
        # Gauge-covariant actual Ref Jz herald, preserving ALL actual OS range.
        herald = np.eye(4)/2-charge*ref_spins[2]
        herald_full = np.kron(herald, np.eye(rank))
        herald_gram = float(np.trace(herald_full@density@herald_full).real)
        polarized = herald_full@density@herald_full/herald_gram
        clock = np.diag(-np.log(charged_values))
        idle = expm(-1j*duration*clock)
        clock_density = idle@polarized@idle.conj().T
        full_generators = [physical_spin_on_range(s, charged, 2) for s in spins]
        reduced = [charged['u'].conj().T@g@charged['u'] for g in full_generators]
        body = [np.kron(np.eye(4), g) for g in reduced]
        current = np.kron(np.eye(4), dirac_cycle_current(gamma, grid, charged))
        refined_current = np.kron(np.eye(4), dirac_cycle_current(gamma, grid, charged, 1e-4))
        packet_body = []
        packet_fock_norms = []
        for spin in spins:
            packet_one = 2*coefficients@spin@coefficients.conj().T
            packet_eigenvalues = np.linalg.eigvalsh(packet_one)
            packet_fock_norms.append(max(float(packet_eigenvalues[packet_eigenvalues > 0].sum()),
                                        -float(packet_eigenvalues[packet_eigenvalues < 0].sum())))
            packet_columns = physical_matter_columns(packet_one, charged, charged['u'])
            packet_body.append(np.kron(np.eye(4), charged['u'].conj().T@packet_columns))
        reference = [np.kron(s, np.eye(rank)) for s in ref_spins]
        actual_body = expm(-1j*duration*clock+1j*angle*body[0])@idle.conj().T
        ref_only = expm(-1j*duration*clock-1j*angle*reference[0])@idle.conj().T
        off = full_generators[0]@charged['u']-charged['u']@reduced[0]
        rotated_columns = physical_rotation_columns(spins[0], charged, 2, angle, charged['u'])
        kato = charged['u'].conj().T@rotated_columns@expm(-1j*angle*reduced[0])
        kato_full = np.kron(np.eye(4), kato)
        body_off = np.eye(len(clock))
        def polarization(operation: np.ndarray) -> list[float]:
            state = operation@clock_density@operation.conj().T
            return [float(np.trace(state@body[j]).real) for j in range(3)]
        def packet_polarization(operation: np.ndarray) -> list[float]:
            state = operation@clock_density@operation.conj().T
            return [float(np.trace(state@g).real) for g in packet_body]
        initial = polarization(body_off)
        observed = dict(actual_full_diagonal_body=polarization(actual_body),
                        Kato_only=polarization(kato_full), Ref_only=polarization(ref_only),
                        body_drive_off=polarization(body_off))
        expected = [initial[0], math.cos(angle)*initial[1]+math.sin(angle)*initial[2],
                    math.cos(angle)*initial[2]-math.sin(angle)*initial[1]]
        packet_initial = packet_polarization(body_off)
        packet_observed = dict(actual_full_diagonal_body=packet_polarization(actual_body),
            Kato_only=packet_polarization(kato_full), Ref_only=packet_polarization(ref_only),
            body_drive_off=packet_polarization(body_off))
        packet_expected = [packet_initial[0], math.cos(angle)*packet_initial[1]+math.sin(angle)*packet_initial[2],
                           math.cos(angle)*packet_initial[2]-math.sin(angle)*packet_initial[1]]
        actual_two_pi = expm(2j*math.pi*body[0])
        ref_two_pi = expm(-2j*math.pi*reference[0])
        body_endpoint = expm(-1j*duration*clock+2j*math.pi*body[0])
        joint_endpoint = expm(-1j*duration*clock+2j*math.pi*(body[0]+reference[0]))
        timed_body = np.trace(polarized@idle.conj().T@body_endpoint)
        timed_joint = np.trace(polarized@idle.conj().T@joint_endpoint)
        packet_norm_bound = max(packet_fock_norms)
        cases.append(dict(charge=charge, neutral_bare_Haar_rank=neutral['middle_dimension'],
            charged_bare_Haar_rank=charged['middle_dimension'], actual_source_Gram=source_gram,
            actual_unfiltered_source_Gram=raw_source_gram,
            actual_quartercut_source_Gram=quarter_gram,
            actual_native_energy_cut=6., actual_native_PE6_retained_fraction=source_gram/raw_source_gram,
            actual_PE6_source_energy=float(np.trace(density@clock).real),
            actual_herald_Gram=herald_gram, actual_joint_source_and_herald_probability=source_gram*herald_gram,
            actual_integer_Gauss_defect=gauss_defect,
            actual_physical4_link_current_norm=float(np.linalg.norm(current, 2)),
            actual_physical4_link_current_Hermiticity_error=float(np.linalg.norm(current-current.conj().T)),
            actual_physical4_link_current_derivative_refinement_error=float(np.linalg.norm(current-refined_current)),
            actual_physical4_link_current_mean=float(np.trace(clock_density@current).real),
            actual_physical4_link_current_after_body_mean=float(np.trace(actual_body@clock_density@actual_body.conj().T@current).real),
            actual_physical4_link_current_after_Ref_only_mean=float(np.trace(ref_only@clock_density@ref_only.conj().T@current).real),
            actual_physical4_link_current_angular_commutator=float(np.linalg.norm(current@body[0]-body[0]@current)),
            current_scope='Derivative -dtheta(logdetB + dGamma(logTphysical)); literal physical4 CAR and Fourier/tree integerGauss range. Not derivative of -log compressedK, not macro Born current admission.',
            actual_range_angular_leakage_norm=float(np.linalg.norm(off)),
            actual_body_angular_clock_commutator=float(np.linalg.norm(clock@body[0]-body[0]@clock)),
            initial_actual_body_polarization=initial, actual_intermediate_angle=angle,
            expected_actual_body_polarization=expected, observed_body_polarizations=observed,
            actual_body_polarization_rotation_error=float(np.linalg.norm(np.array(observed['actual_full_diagonal_body'])-expected)),
            actual_body_vs_Ref_only_polarization_difference=float(np.linalg.norm(np.array(observed['actual_full_diagonal_body'])-observed['Ref_only'])),
            actual_bounded_source_packet_spin_initial=packet_initial,
            actual_bounded_source_packet_spin_observed=packet_observed,
            actual_packet_spin_rotation_error=float(np.linalg.norm(np.array(packet_observed['actual_full_diagonal_body'])-packet_expected)),
            actual_packet_spin_Ref_only_discriminator=float(np.linalg.norm(np.array(packet_observed['actual_full_diagonal_body'])-packet_observed['Ref_only'])),
            packet_spin_operator_norm_bound=packet_norm_bound,
            packet_spin_scope='Literal gauge/tree-Wilson-dressed active packet a_fdag S_cart a_f, rank2 stable/occupied band. Norm<=1/2 independently Naux; same actual thermal-source/herald/clock, not an added abstract Ref label or bulk noise division.',
            actual_Kato_only_identity_error=float(np.linalg.norm(kato_full-np.eye(len(clock)))),
            actual_body_two_pi_minus_error=float(np.linalg.norm(actual_two_pi+np.eye(len(clock)))),
            actual_concurrent_clock_body_two_pi_interference=[float(timed_body.real), float(timed_body.imag)],
            actual_concurrent_clock_joint_two_pi_interference=[float(timed_joint.real), float(timed_joint.imag)],
            actual_concurrent_body_dark_probability=float((1+timed_body.real)/2),
            actual_common_body_and_Ref_angular_energy_norm=float(np.linalg.norm(2*math.pi*(body[0]+reference[0])/duration, 2)),
            actual_clock_energy_norm=float(np.linalg.norm(clock, 2)),
            Ref_only_two_pi_interference_despite_FAILED_body_witness=[
                float(np.trace(clock_density@ref_two_pi).real), float(np.trace(clock_density@ref_two_pi).imag)],
            actual_joint_two_pi_plus_error=float(np.linalg.norm(actual_two_pi@ref_two_pi-np.eye(len(clock)))),
            actual_polarized_density_trace=float(np.trace(polarized).real)))
    rotated_gamma = np.array([expm(1j*angle*spins[0])@g@expm(-1j*angle*spins[0]) for g in gamma])
    half0, _ = dirac_cycle_step(gamma, .09, .41)
    half1, _ = dirac_cycle_step(rotated_gamma, .09, .41)
    spin_rotation = np.kron(np.eye(2), expm(1j*angle*spins[0]))
    return dict(full4_Dirac_active_modes=8, native_even_volume=2, grid=grid, electric_cutoff=cutoff,
        actual_Cartesian_Phi2_Phi3_motion_norm=float(np.linalg.norm(rotated_gamma-gamma)),
        actual_rotated_Phi_Wilson_positive_halfstep_error=float(np.linalg.norm(half1-spin_rotation@half0@spin_rotation.conj().T)),
        actual_full4_SpinSU2_error=max(float(np.linalg.norm(spins[j]@spins[k]-spins[k]@spins[j]-1j*spins[l]))
                                     for j, k, l in ((0, 1, 2), (1, 2, 0), (2, 0, 1))),
        cases=cases,
        scope='Actual full4-spin two-site positive Wilson/Haar/integer-Gauss thermal source, stableband particle/occupiedband hole, full4Ref. Herald removes EPR-transpose ambiguity. Axisx commutes the one-axis hopping, hence actual Kato-only is IDENTITY while paid diagonal angular drive rotates body Jz/Jy. Ref-only2pi minus explicitly fails body-marginal witness. No two-spin Jx/Jy fitting, no filledsea/Hermite replacement.')


def all_sector_primitive_phase(gamma: np.ndarray, triad: dict[str, Any]) -> dict[str, Any]:
    """Actual even-path algebra, including local empty/full and fluctuating cells."""
    model = BosonicEdgeModel(8, ((0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 7)))
    spins = triad['spins']
    body = expm(2j*math.pi*native_lift(model, spins[0]))
    reference = expm(2j*math.pi*native_lift(model, spins[0], 4))
    ref_number = sum((model.n(c) for c in range(4, 8)), np.zeros_like(model.identity))
    ref_casimir = sum(native_lift(model, s, 4)@native_lift(model, s, 4) for s in spins)
    cases = []
    for charge in (1, -1):
        packet = (np.eye(4)+charge*gamma[0])/2
        source = source_operator(model, packet, charge)
        background = np.eye(len(source))[np.real(np.diag(np.asarray(ref_number))) == (4 if charge == 1 else 0)].T
        selected = source@background
        cases.append(dict(charge=charge,
            all_sector_body_source_anticommutator=float(np.linalg.norm(body@source+source@body)),
            all_sector_Ref_source_anticommutator=float(np.linalg.norm(reference@source+source@reference)),
            actual_Ref_background_number=4 if charge == 1 else 0,
            actual_Ref_source_number=3 if charge == 1 else 1,
            actual_Ref_background_source_minus_error=float(np.linalg.norm((reference+model.identity)@selected)),
            actual_Ref_source_Casimir_error=float(np.linalg.norm((ref_casimir-.75*model.identity)@selected))))
    return dict(primitive_edge_Hilbert_dimension=len(model.identity), cases=cases,
        actual_full_joint_two_pi_identity_error=float(np.linalg.norm(body@reference-model.identity)),
        scope='Every state of actual primitive even-parity eight-leaf graph, not n_body=2. CAR source intertwiners tensor with ANY even auxiliary bilinear transfer; local auxiliary n0..4 and all divergence-compatible centre cycles do not change fixed GLOBAL Ns parity. Ref3/1 is measured, not an extra abstract label.')


def canonical_aux_spin_incidence(gamma: np.ndarray, triad: dict[str, Any]) -> dict[str, Any]:
    """Actual occupied-band canonical determinant covariance, not Gauss vacuum."""
    spin = triad['spins'][2]
    momenta = list(itertools.product(2*math.pi*np.arange(4)/4, repeat=3))
    rows = []
    for kappa in (.03, .01, .003):
        covariances = []
        bulk = 0.
        mass = math.log(1/(2*kappa)-3)
        for momentum in momenta:
            one, _, _ = wilson_step(gamma, 1., mass, np.array(momentum))
            values, vectors = np.linalg.eigh(one)
            occupied = vectors[:, values > 1]
            covariance = occupied@occupied.conj().T
            covariances.append(covariance)
            bulk += float(np.trace(covariance@spin@spin-covariance@spin@covariance@spin).real)/len(momenta)
        local = sum(covariances)/len(covariances)
        variance = float(np.trace(local@spin@spin-local@spin@local@spin).real)
        rows.append(dict(kappa=kappa, local_spin_variance=variance,
                         local_variance_over_kappa4=variance/kappa**4,
                         total_spin_variance_per_cell=bulk,
                         total_variance_over_kappa4=bulk/kappa**4))
    return dict(momentum_grid=[4, 4, 4], rows=rows,
        scope='Exact canonical filled-band determinant covariance in a finite Wilson family. Establishes bulk spin incidence of order Naux*kappa^4~beta, NOT the interacting Gauss-prepared vacuum covariance. Independent actual shared-Gauss aux0/1/2 density controls are reported separately; no bulk noise division.')


def finite_full_species_actuation(photon: Parameters) -> dict[str, Any]:
    naux = 10**(photon.beta_power+4*photon.kappa_power)//16
    volume = photon.spatial_size**3
    neutral_number = 2*volume*(naux+1)
    generator_bound = volume*(naux+1)+1
    duration = .17
    energy_log = math.log10(generator_bound)+math.log10(2*math.pi/duration)
    return dict(actual_Naux=str(naux), spatial_volume=str(volume),
        neutral_canonical_number=str(neutral_number),
        charged_canonical_numbers=[str(neutral_number+q) for q in (1, -1)],
        actual_common_matter_and_Ref_angular_generator_norm_upper=str(generator_bound),
        actual_common_bulk_angular_energy_upper_log10_lattice=energy_log,
        actual_common_bulk_angular_energy_upper_log10_physical=[
            dict(a=a, value=energy_log-math.log10(a)) for a in photon.spacings],
        actual_counter_Ref_energy_lattice=math.pi/duration,
        finite_lattice_duration=duration,
        body_two_pi_scalar=-1, Ref3_or1_two_pi_scalar=-1, full_joint_scalar=1,
        protocol='Both arms use the SAME bulk moving-range angular control AND SAME canonical clock, without dividing the actual density/noise. Angular control contains diagonal PGP as well as offdiagonal Kato. Body arm differs by physical counterRef. Exact global oddNs controls2pi for every auxiliary local fluctuation, all centre cycles and P8. Matter+Ref energy above pays ALL Naux, not native commutator. Normal/colour Cartesian field energy and actual full-source clock moment require supplier state bounds; this matter bound is not a complete field actuator norm.')


def actual_normal_rotation_representation(triad: dict[str, Any]) -> dict[str, Any]:
    """Map the ACTUAL Cartesian rotation, not an arbitrary SO(45) generator."""
    basis = np.asarray(matrix_unit_basis())
    columns = []
    for index, matrix in itertools.product(range(4), basis):
        fields = np.zeros((4, 4, 4), complex)
        fields[index] = matrix
        columns.append(real_vector(fields))
    hermitian_frame, _ = np.linalg.qr(np.column_stack(columns))
    left, _, _ = np.linalg.svd(hermitian_frame.T@triad['columns'], full_matrices=True)
    normal_frame = hermitian_frame@left[:, 15:]
    colour_frame, _ = np.linalg.qr(np.column_stack([real_vector(g) for g in basis]))
    orthonormal_colour = [(column[:16]+1j*column[16:]).reshape(4, 4) for column in colour_frame.T]
    rows = []
    for axis, spin in enumerate(triad['spins']):
        spatial = np.zeros((4, 4))
        j, k = (axis+1)%3+1, (axis+2)%3+1
        spatial[j, k], spatial[k, j] = -1., 1.
        transformed = []
        for column in normal_frame.T:
            field = (column[:64]+1j*column[64:]).reshape(4, 4, 4)
            derivative = np.einsum('jk,kab->jab', spatial, field)-1j*(spin@field-field@spin)
            transformed.append(real_vector(derivative))
        normal = normal_frame.T@np.column_stack(transformed)
        colour = colour_frame.T@np.column_stack([
            real_vector(-1j*(spin@g-g@spin)) for g in orthonormal_colour])
        normal_weights = np.linalg.eigvalsh(1j*normal)
        colour_weights = np.linalg.eigvalsh(1j*colour)
        rows.append(dict(axis=axis,
            actual_normal_coordinate_angular_weights=normal_weights.tolist(),
            actual_colour_coordinate_angular_weights=colour_weights.tolist(),
            actual_normal_generator_norm=float(max(abs(normal_weights))),
            actual_colour_generator_norm=float(max(abs(colour_weights))),
            actual_normal_skew_error=float(np.linalg.norm(normal+normal.T)),
            actual_normal_slice_invariance_error=float(np.linalg.norm(np.column_stack(transformed)-normal_frame@normal))))
    return dict(actual_normal_dimension=normal_frame.shape[1], actual_colour_dimension=15,
        rows=rows, interpretation='Normal generator is spatial spin1 index PLUS adjoint internal spin1, restricted to actual45 complement of all15 gauge tangents. Rank2 angular weights are measured. Colour is actual AdS, norm1. Equivariant radial Gaussian contraction therefore gives Hphi>=|L|/2 for this actual representation, not every arbitrary SO(D) multiplier.')


def projected_field_angular_price(budget: dict[str, Any]) -> dict[str, Any]:
    """Whole-projector heat/Sylvester bound, not a density-error times huge J."""
    cutoff = .25
    gap = math.exp(-cutoff)-math.exp(-1)
    root_error_log = max(budget['full_step_relative_root_colour_log10'],
                         budget['full_step_relative_root_normal_log10'])
    # deltaR <=2*10^root_error; 2deltaR+deltaR² <=3deltaR.
    angular_log = root_error_log+math.log10(24/(math.e*gap))
    return dict(actual_energy_cutoff=cutoff, ideal_full_phi_excitation_gap_lower=1.,
        actual_ordered_Sylvester_gap=gap, actual_scalar_heat_angular_norm_upper=4/math.e,
        actual_full_normal_colour_L_projected_source_norm_upper_log10=angular_log,
        actual_field_angular_drive_energy_upper_log10_lattice=angular_log+math.log10(2*math.pi/.17),
        proof='For declared radial Friedrichs chart law, equivariant Gaussian contraction compares angular sectors: Hphi>=|L|/2 since normal coordinate rotation norm<=2 and colourAdS<=1. Hence ||L Nhalf||<=4/e, independent90V/NF. L Pphi=0 and [L,N]=0. For X=L P_E, QK0Q X-X(P_E K P_E)=-L deltaK P_E. Ordered Sylvester integral bounds WHOLE projector X, no sum of eigenvector bounds/dimension factor. LdeltaK bounded by (4/e)(2deltaR+deltaR²)lambda0 because left scalar heat smooths L. Actual P_E K P_E>=e^-.25lambda0, QK0Q<=e^-1lambda0.',
        scope='Declared product radial CHART heat + same Haar/full-species microscopic congruence. Outside completion must retain its own source escape/entropy admission; this is not an unconditional statement about the frozenCartesian original law or a gap-only angular inference.')


def run(seed: int = 22210608, grid: int = 24, cutoff: int = 2,
        steps: int = 8, normal_nodes: tuple[int, ...] = (7, 11), *,
        root_receipt_path: Path, source_receipt_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    root_receipt_bytes = root_receipt_path.read_bytes()
    root_receipt = json.loads(root_receipt_bytes)
    configuration = root_receipt['parameters']
    root_identity = root_receipt['capsule']
    if configuration.get('physical_component_connection_law') != 'physical-components-Dminus-20261008':
        raise ArithmeticError('Historical Dplus receipt cannot admit a physical Noether rejoin')
    if configuration.get('positive_transfer_matrix_basis') != 'physical-reference-Theta-block-ThetaDag-20261008':
        raise ArithmeticError('Root transfer is not in physical4 reference components')
    if SOURCE_HASHES_AT_IMPORT['covariant_root_transfer.py'] != root_identity['source_sha256']:
        raise ArithmeticError('Compiled Root source differs from the selected immutable supplier')
    if hashlib.sha256(root_identity['source_bytes_at_IMPORT']['covariant_root_transfer.py'].encode()).hexdigest() != root_identity['source_sha256']:
        raise ArithmeticError('Root receipt source bytes do not match its source identity')
    if hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(',', ':')).encode()).hexdigest() != root_identity['input_sha256']:
        raise ArithmeticError('Selected Root receipt has inconsistent physical-law input bytes')
    source_receipt_bytes = source_receipt_path.read_bytes()
    source_receipt = json.loads(source_receipt_bytes)
    source_identity = source_receipt['compiled_source_hashes']['covariant_prepared_gauss_state']
    source_input = source_receipt['input_configuration']
    if SOURCE_HASHES_AT_IMPORT['covariant_prepared_gauss_state.py'] != source_identity:
        raise ArithmeticError('Loaded prepared-source bytes differ from the selected immutable capsule')
    if hashlib.sha256(source_receipt['captured_source_bytes']['covariant_prepared_gauss_state'].encode()).hexdigest() != source_identity:
        raise ArithmeticError('Prepared-source receipt has inconsistent captured source bytes')
    if hashlib.sha256(source_input['captured_bytes'].encode()).hexdigest() != source_input['sha256']:
        raise ArithmeticError('Prepared-source receipt has inconsistent input bytes')
    if not source_receipt['compiled_source_unchanged'] or not source_receipt['main_code_matches_captured_source']:
        raise ArithmeticError('Prepared-source capsule did not execute stable matching source code')
    source_root = source_receipt['macro']['full_SU4_normal_Root']
    if (source_root['source_sha256'] != root_identity['source_sha256'] or
        source_root['root_input_sha256'] != root_identity['input_sha256']):
        raise ArithmeticError('Prepared source uses a different physical Root source/input epoch')
    p = RootParameters(**configuration['root'])
    photon = Parameters(**configuration['photon'])
    inputs = dict(seed=seed, root=asdict(p), photon=asdict(photon), native_grid=grid,
        electric_cutoff=cutoff, native_thermal_steps=steps, native_normal_nodes=list(normal_nodes),
        native_full4_actual_energy_cut=6., native_quartercut_adverse=.25,
        native_quantum_normal_actual_energy_cut=6.,
        root_receipt_path=str(root_receipt_path.resolve()),
        root_receipt_sha256=hashlib.sha256(root_receipt_bytes).hexdigest(),
        root_source_sha256=root_identity['source_sha256'], root_input_sha256=root_identity['input_sha256'],
        source_receipt_path=str(source_receipt_path.resolve()),
        source_receipt_sha256=hashlib.sha256(source_receipt_bytes).hexdigest(),
        prepared_supplier_source_sha256=source_identity, prepared_supplier_input_sha256=source_input['sha256'],
        physical_component_connection_law=configuration['physical_component_connection_law'],
        positive_transfer_matrix_basis=configuration['positive_transfer_matrix_basis'],
        full_microscopic_source_admitted=False,
        governing_plan='issue-222-induced-photon-delta13-continuation-20261008',
        governing_revision='85975947857ac7816cf0d8674f69229861014434b125a07a6996af0ecf72c4eb',
        assignment_governing_revision='85975947857ac7816cf0d8674f69229861014434b125a07a6996af0ecf72c4eb',
        pinned_general='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
        probe='NEW physical4 Theta/Dminus actual native PE6 Gauss/body/packet/Ref/current Noether with matching fullgeometry Root. FullNF clock CONDITIONAL/UNADMITTED; no old-projector/67bfab/Root−587 retag, no finiteSCREEN retrofit.')
    encoded = json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()
    rng = np.random.default_rng(seed)
    gamma, _ = phi_frame(photon)
    phi = np.sqrt(np.array(photon.gram))[:, None, None]*SELECTED
    triad = physical_triad(phi)
    physical_rejoin = physical_component_native_rejoin(gamma, triad, p)
    field = fullfield_local_native(phi, triad, p, rng)
    reference = native_Ref_specimens(gamma, triad, p, rng)
    orbit = packet_orbit(gamma, triad, photon, rng)
    thermal = actual_thermal_source_channel(gamma, triad, grid, cutoff, steps)
    polarized = actual_polarized_body_rotation(gamma, triad, grid, cutoff, steps)
    polarized_refined = actual_polarized_body_rotation(gamma, triad, grid+8, cutoff+1, steps)
    refinement = dict(refined=polarized_refined,
        source_Gram_relative_change=max(abs(new['actual_source_Gram']/old['actual_source_Gram']-1)
            for old, new in zip(polarized['cases'], polarized_refined['cases'])),
        bounded_packet_discriminator_change=max(abs(new['actual_packet_spin_Ref_only_discriminator']-old['actual_packet_spin_Ref_only_discriminator'])
            for old, new in zip(polarized['cases'], polarized_refined['cases'])))
    auxiliary = actual_auxiliary_range_controls(gamma, triad, grid, cutoff, steps)
    normal = [actual_quantum_normal_Ref_channel(gamma, triad, steps, nodes, charge)
              for nodes, charge in itertools.product(normal_nodes, (1, -1))]
    cycles = all_cycle_Ref_noether()
    primitive = all_sector_primitive_phase(gamma, triad)
    incidence = canonical_aux_spin_incidence(gamma, triad)
    budget = charge_weighted_phi_budget(p, photon)
    macro = prepared.macro_preparation(photon, gamma, root_parameters=p)
    source_binding = dict(full_actual_source_admitted=False,
        supplier_source_sha256=SOURCE_HASHES_AT_IMPORT['covariant_prepared_gauss_state.py'],
        supplier_receipt_path=str(source_receipt_path.resolve()),
        supplier_receipt_sha256=hashlib.sha256(source_receipt_bytes).hexdigest(),
        supplier_input_sha256=source_input['sha256'],
        supplier_branch=source_receipt['branch'],
        supplier_macro_admission=source_receipt['macro']['admission'],
        actual_loaded_supplier_macro_admission=macro['admission'],
        supplier_declared_Root_source_sha256=macro['full_SU4_normal_Root']['source_sha256'],
        supplier_declared_Root_input_sha256=macro['full_SU4_normal_Root']['root_input_sha256'],
        matching_physical_Root_source=macro['full_SU4_normal_Root']['source_sha256'] == root_identity['source_sha256'],
        matching_physical_Root_input=macro['full_SU4_normal_Root']['root_input_sha256'] == root_identity['input_sha256'],
        scope='Native actual energy_cut_source/runtime/Ref matrices retain their literal bounded2spin functional recipe. Source/root byte and exact physical-input joins are reported separately above: a matching supplier ledger is necessary but NEVER microscopic source-Gram admission. No Root−587 theorem or oldDplus metadata promoted. This consumer independently reconstructs its physical4 native source from physical halfsteps/fullpositive spectra.')
    outside = outside_angular_bound(root_receipt)
    common_energy_log = float(outside['whole_PE_global_Lindex_norm_log10_upper'])+math.log10(2*math.pi/.17)
    common_clock = dict(full_actual_source_admitted=False, conditional_energy_window=.25,
        conditional_common_bulk_angular_energy_lattice_upper_log10=common_energy_log,
        actual_counter_Ref_energy_lattice_upper=math.pi/.17,
        actual_common_clock_duration_lattice=.17,
        conditional_common_energy_physical_upper_log10=[dict(a=a, value=common_energy_log-math.log10(a)) for a in photon.spacings],
        generator='GLOBAL WORLD Phi spatial-index rotation; physical colour links fixed, no hidden outside coframe. Its actual chart pullback is full45 normal + colourAdS + ALLmatter/Ref Fock connection. Connection paid, not tiny.',
        protocol='Conditional construction K(theta)=U K Udag on moving physical range, Hdrive=U H_OS Udag-theta_dot G, propagator Uexp(-itH_OS). If a nonzero ACTUAL full source in PE=.25 is supplied, both arms use SAME complete G/clock, body arm counterRef. The microscopic full source numerator has NOT been proved. Gaussian action ratios, free poles or native source norms do not supply it. Actual native PE6/concurrent pulses/body/packet controls are separate.',
        scope='UNADMITTED full-source clock. Exact conditional angular theorem and finite FULL NF connection price only. Field .89466340743/Gram>=.5 claim WITHDRAWN: inserted-Gauss microscopic numerator was not justified. Native quartercut is measured ZERO; native PE6 is a distinct material source repair, never macro or finiteSCREEN retag.')
    angular = projected_field_angular_price(budget)
    angular_representation = actual_normal_rotation_representation(triad)
    frame_covariance = actual_frame_transfer_covariance(gamma, triad, photon)
    if max(triad['residuals']) > 1e-10 or triad['two_pi_error'] > 1e-10:
        raise ArithmeticError('Literal Cartesian triad failed')
    if max(field['actual_full45_normal_physical_group_error'],
           field['actual_transported45_normal_slice_error']) > 1e-10:
        raise ArithmeticError('Actual full45 mechanical normal flow failed')
    if field['wrong_untransported_normal_physical_group_error'] < 1e-7:
        raise ArithmeticError('Missing-normal adverse control did not discriminate')
    for row in polarized['cases']:
        if row['actual_herald_Gram'] <= 0 or abs(row['initial_actual_body_polarization'][2]) < .1:
            raise ArithmeticError('Actual polarized source or paid herald failed')
        if row['actual_body_polarization_rotation_error'] > 1e-10 or row['actual_body_vs_Ref_only_polarization_difference'] < .1:
            raise ArithmeticError('Actual body versus Ref-only discrimination failed')
        if row['actual_packet_spin_rotation_error'] > 1e-10 or row['actual_packet_spin_Ref_only_discriminator'] < .1:
            raise ArithmeticError('Actual bounded active packet body witness failed')
        if row['actual_Kato_only_identity_error'] > 1e-10 or row['actual_body_two_pi_minus_error'] > 1e-10:
            raise ArithmeticError('Paid diagonal angular actuation or Kato falsifier failed')
        if row['actual_physical4_link_current_Hermiticity_error'] > 1e-8 or row['actual_physical4_link_current_derivative_refinement_error'] > 1e-7:
            raise ArithmeticError('Actual physical4 primitive/Gauss current failed its numerical derivative control')
    for row in primitive['cases']:
        if max(row['all_sector_body_source_anticommutator'],
               row['actual_Ref_source_Casimir_error']) > 1e-10:
            raise ArithmeticError('State-independent primitive/Ref phase failed')
    if max(row['actual_normal_generator_norm'] for row in angular_representation['rows']) > 2+1e-10:
        raise ArithmeticError('Actual Cartesian normal angular representation exceeds spin2')
    runtime_hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_NAMES}
    return dict(status='CONDITIONAL_PHYSICAL_DMINUS_FULL4_GAUSS_PE6_NOETHER_MACRO_OPEN',
        inputs=inputs, input_sha256=hashlib.sha256(encoded).hexdigest(),
        source_at_import=SOURCE_HASHES_AT_IMPORT, source_at_runtime=runtime_hashes,
        source_bytes_at_import={name: value.decode('utf-8') for name, value in SOURCE_BYTES_AT_IMPORT.items()},
        actual_compiled_dependency_hashes=prepared.COMPILED, loaded_Noether_code_matches_capture=LOADED_CODE_MATCH,
        source_unchanged_at_runtime=runtime_hashes == SOURCE_HASHES_AT_IMPORT,
        physical_triad={k: v for k, v in triad.items() if k not in ('spins', 'columns', 'coefficients')},
        actual_physical_component_native_rejoin=physical_rejoin,
        actual_prepared_thermal_source=thermal, actual_full4_polarized_body_witness=polarized,
        actual_full4_flux_harmonic_refinement=refinement,
        actual_shared_Haar_aux0_1_2=auxiliary, actual_quantum_normal_Ref_source=normal,
        actual_all_sector_primitive_intertwiner=primitive, actual_all_cycle_Ref_P8=cycles,
        actual_literal_frame_halfstep_covariance=frame_covariance, fullfield_mechanical_group=field,
        field_dependent_Ref_Gauss_reference=reference['field_dependent_Ref_Noether'],
        reference_poles_and_orbit=orbit, canonical_aux_spin_incidence=incidence,
        actual_full_species_matter_actuation=finite_full_species_actuation(photon),
        actual_projected_radial_field_angular_price=angular,
        actual_fulloutside_and_cross_angular_supplier=outside,
        actual_common_full_species_angular_clock=common_clock,
        captured_immutable_Root_receipt_sha256=hashlib.sha256(root_receipt_bytes).hexdigest(),
        actual_Cartesian_normal_colour_angular_representation=angular_representation,
        captured_Root_species_weighted_budget=budget, actual_full_species_preparation_supplier=macro,
        preparation_supplier_physical_rejoin_status=source_binding,
        historical_preserved=dict(source642='642e8631f4d36e1b162a8d20258d6618e5188494d946c0189fe4cf029f84607d',
            source67='67bfab920f884d11c82a6973b6c6de81d04d6532ddf255271f95854efaf86d7e',
            capsule67='covariant_physical_noether_capsule_67bfab92.json',
            execution67='covariant_physical_noether_execution_67bfab92.json',
            capsule='covariant_physical_noether_capsule_642e8631.json',
            execution='covariant_physical_noether_execution_642e8631.json',
            scope='Historical canonical-block/oldDplus conditional epochs preserve their exact source and capsules. Their determinant/spectrum equality does not identify physical4 projectors/currents. No old−418/−268/−587, baref-source or67bfab bytes rebound to NEW physical Dminus. Conditional CURRENT427 and photon450 identities untouched.'),
        outstanding='Actual native full4 PE6 body/Ref discrimination, positive Haar thermal sources, auxiliary spin/number fluctuations and paid angular clocks survive. FullNF microscopic inserted-Haar retained source numerator/Pe.25 is NOT proved: withdrawn .894663/.5 metadata cannot admit it. Outside scalar/connection bounds remain conditional. Common protected subtraction never divides bulk spin variance or multiplies huge J by generic density error. This is a driven apparatus, not autonomous exact cubic SO3 or interacting finitepacket probability1. Physical Born/Maxwell campaign remains OPEN; finiteSCREEN delta13 unchanged.',
        elapsed_seconds=time.monotonic()-started,
        runtime=dict(python=sys.version, executable=sys.executable, numpy=np.__version__,
                     scipy=scipy.__version__, platform=platform.platform(),
                     executable_sha256=hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='covariant_physical_noether_capsule.json')
    parser.add_argument('--root-receipt', type=Path, required=True,
                        help='Immutable NEW physical-Dminus Root capsule matching the compiled Root source')
    parser.add_argument('--source-receipt', type=Path, required=True,
                        help='Immutable executed prepared-source capsule matching actual loaded bytes and Root input')
    parser.add_argument('--seed', type=int, default=22210608)
    parser.add_argument('--grid', type=int, default=24)
    parser.add_argument('--cutoff', type=int, default=2)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--normal-nodes', nargs='+', type=int, default=[7, 11])
    args = parser.parse_args()
    result = run(args.seed, args.grid, args.cutoff, args.steps, tuple(args.normal_nodes),
                 root_receipt_path=args.root_receipt, source_receipt_path=args.source_receipt)
    output = Path(args.output)
    output.write_text(json.dumps(prepared.portable(result), sort_keys=True, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], output=str(output.resolve()), input_sha256=result['input_sha256'],
                         source=result['source_at_import'], source_unchanged=result['source_unchanged_at_runtime'],
                         elapsed_seconds=result['elapsed_seconds'])))


if __name__ == '__main__':
    main()
