#!/usr/bin/env python3
"""Native bare-Gaussian order-g^2 carrier gap; not a compact quantum mass."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
import scipy
from scipy.linalg import block_diag

from angular_photon_spectrum import brillouin_quadrature
from bulk_cutoff_matching import dimensionless_bare
from bulk_frame_compton import (
    primitive_frame_controls, raw_h, raw_paths, reconstruct_frame,
    source_hashes as frame_hashes,
)
from original_charged_principal import original_frame, source_hashes as principal_hashes
from original_uniform_charge_gap import exterior_hamiltonian
from rotation_band import RotationBand
from unrestricted_field_minimum import BARE

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-continuing-backward-20260930'
PLAN_REVISION = 'b9c426960085aecd28d51c57e27e9cc42092d05a6bf09762578f27840a6eb11d'
TEMPLATE_REVISION = '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'


def source_hashes():
    names = ('original_quantum_mass_diagnostic.py', 'original_quantum_mass_diagnostic.md',
             'angular_photon_spectrum.py', 'original_uniform_charge_gap.py',
             'bulk_cutoff_matching.py')
    return {**frame_hashes(), **principal_hashes(), **{
        name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}}


def accept(actual, expected, label, tolerance=2e-8):
    """One consumer assertion also exercised by the deliberately incomplete laws."""
    error = float(np.linalg.norm(np.asarray(actual)-np.asarray(expected)))
    scale = max(1., float(np.linalg.norm(expected)))
    if not np.isfinite(error) or error > tolerance*scale:
        raise AssertionError(f'{label}: relative-scale residual {error/scale}')
    return error/scale


def native_current(paths, midpoint_angles):
    """Link-angle first jet from the actual native Phi hopping, in batches."""
    result = np.zeros((len(midpoint_angles), 3, 4, 4), dtype=complex)
    for displacement, coupling in paths:
        axis = int(np.flatnonzero(displacement)[0])
        phase = np.exp(1j*(midpoint_angles@displacement))
        result[:, axis] += 1j*(phase[:, None, None]*coupling
                             -phase.conj()[:, None, None]*coupling.conj().T)
    return result


def native_h(onsite, paths, angles):
    result = np.broadcast_to(onsite, (len(angles), 4, 4)).copy()
    for displacement, coupling in paths:
        phase = np.exp(1j*(angles@displacement))
        result += phase[:, None, None]*coupling+phase.conj()[:, None, None]*coupling.conj().T
    return result


def native_contact(paths, angles):
    """Three diagonal link-angle second jets, with no fitted seagull."""
    result = np.zeros((len(angles), 3, 4, 4), dtype=complex)
    for displacement, coupling in paths:
        axis = int(np.flatnonzero(displacement)[0])
        phase = np.exp(1j*(angles@displacement))
        result[:, axis] -= phase[:, None, None]*coupling+phase.conj()[:, None, None]*coupling.conj().T
    return result


def rest_basis(band, unitary):
    return unitary@np.column_stack([band.spinor(np.zeros(3), j) for j in (0, 1)])


def projected_transition(current, projector, transverse, rest):
    applied = np.einsum('njab,bs->njas', current, rest)
    return np.einsum('njas,nab,nkbt,njk->nst', applied.conj(), projector,
                     applied, transverse, optimize=True)


def scalarize(matrix, label, *, coefficient_scale):
    value = float(np.trace(matrix).real/2)
    accept(coefficient_scale*matrix, coefficient_scale*value*np.eye(2), label)
    return value


def matrix_leg_controls(bare, band, frame, unitary):
    """Physical-A API, scalar density and both actual transverse matrix legs."""
    onsite, paths = raw_paths(bare, frame)
    del onsite
    rest = rest_basis(band, unitary)
    controls = []
    for q in (np.array([.37, -.53, .71]), np.array([1.13, .41, -.29])):
        qhat = 2*np.sin(q/2)
        direction = qhat/np.linalg.norm(qhat)
        seed = np.eye(3)[int(np.argmin(abs(direction)))]
        eps1 = np.cross(direction, seed); eps1 /= np.linalg.norm(eps1)
        eps2 = np.cross(direction, eps1)
        p = q/band.a
        rows = []
        for eps in (eps1, eps2):
            direct = np.einsum('j,jab->ab', eps, native_current(paths, q[None]/2)[0])
            blocked = np.einsum('j,jab->ab', eps, native_current(paths, -q[None]/2)[0])
            api = unitary@np.einsum('j,jab->ab', eps, band.current(p, np.zeros(3))/band.a)@unitary.conj().T
            api_blocked = unitary@np.einsum('j,jab->ab', eps, band.current(np.zeros(3), -p)/band.a)@unitary.conj().T
            second = np.einsum('j,jab->ab', eps**2, native_contact(paths, np.zeros((1, 3)))[0])
            api_second = unitary@band.contact(np.zeros(3), np.zeros(3), np.r_[0., eps],
                np.r_[0., eps], p, p)@unitary.conj().T/band.a**2
            rows.append(dict(current=accept(direct, api, 'native photon leg'),
                blocked_current=accept(blocked, api_blocked, 'native blocked photon leg'),
                contact=accept(second, api_second, 'native second jet'),
                transverse=float(abs(eps@qhat)), polarization=eps.tolist()))
        density_plus = unitary@band.projector(p, -1)@unitary.conj().T
        density_minus = unitary@band.projector(-p, -1)@unitary.conj().T
        density_gap = rest.conj().T@(np.eye(4)-density_plus-density_minus)@rest
        scalar_api = unitary@(-band.vertex(p, np.zeros(3), np.array([1., 0., 0., 0.])))@unitary.conj().T
        accept(scalar_api, np.eye(4), 'unit scalar density leg')
        density_error = accept(density_gap, band.mass(p)/band.energy(p)*np.eye(2), 'Gauss Wick scalar leg')
        controls.append(dict(q_angles=q.tolist(), photon_polarizations=rows,
                             scalar_density_error=density_error))
    return controls


def axial_controls(bare, band, frame, unitary):
    """Discriminate actual axial conservation from paired Hamiltonian energies."""
    alpha = np.array([unitary@value@unitary.conj().T for value in band.alpha])
    beta = unitary@band.beta@unitary.conj().T
    chi = -1j*alpha[0]@alpha[1]@alpha[2]
    pairing = 1j*beta@chi
    accept(chi, chi.conj().T, 'Hermitian axial generator')
    accept(chi@chi, np.eye(4), 'unit axial generator')
    for value in alpha:
        accept(chi@value-value@chi, np.zeros((4, 4)), 'axial kinetic commutator')
    accept(chi@beta+beta@chi, np.zeros((4, 4)), 'axial scalar mass breaking')
    q = np.array([.61, -.37, .83])
    h = raw_h(bare, frame, q/band.a)
    mass = band.mass(q/band.a)
    commutator = chi@h-h@chi
    accept(commutator, mass*(chi@beta-beta@chi), 'native axial Ward breaking')
    pairing_error = accept(pairing@h+h@pairing, np.zeros((4, 4)),
                           'paired spectrum despite axial breaking', 2e-7)
    wilson_breaking = float(np.linalg.norm((mass-band.m)*(chi@beta-beta@chi)))
    if wilson_breaking <= 0:
        raise AssertionError('Native Wilson axial breaking disappeared')
    return dict(q_angles=q.tolist(), axial_commutator_norm=float(np.linalg.norm(commutator)),
        Wilson_only_axial_breaking_norm=wilson_breaking, spectral_pairing_anticommutator_error=pairing_error,
        conclusion='paired h energies do not conserve chi; native Wilson term breaks axial symmetry')


def bz_diagnostic(bare, band, frame, unitary, order) -> dict[str, Any]:
    """Independent projector/current evaluation on normalized thermodynamic BZ."""
    points, weights = brillouin_quadrature(order, 'split_gauss')
    if weights is None:
        raise AssertionError('Explicit normalized split-Gauss weights required')
    accept(float(weights.sum()), 1., 'normalized BZ measure', 1e-12)
    onsite, paths = raw_paths(bare, frame)
    rest = rest_basis(band, unitary)
    contact_jets = native_contact(paths, np.zeros((1, 3)))[0]
    rest_contact = np.einsum('as,jab,bt->jst', rest.conj(), contact_jets, rest)
    matrices = {name: np.zeros((2, 2), complex) for name in ('contact', 'new_positive', 'blocked_negative', 'Coulomb')}
    closed = {name: 0. for name in ('contact', 'new_positive', 'blocked_negative', 'Coulomb')}
    uv = {name: 0. for name in ('contact', 'virtual', 'Coulomb', 'reference_self')}
    reference = 0.
    minimum_virtual = float('inf')
    z, r, c = band.a*band.m, band.r, band.v
    for start in range(0, len(points), 1024):
        q, w = points[start:start+1024], weights[start:start+1024]
        qhat = 2*np.sin(q/2); Q2 = np.sum(qhat*qhat, axis=1); Q = np.sqrt(Q2)
        if np.any(Q == 0):
            raise AssertionError('The harmonic q=0 rotor is not a Gaussian oscillator')
        transverse = np.eye(3)[None]-qhat[:, :, None]*qhat[:, None, :]/Q2[:, None, None]
        hp, hm = native_h(onsite, paths, q), native_h(onsite, paths, -q)
        Ep = np.sqrt(np.einsum('nab,nba->n', hp, hp).real/4)
        Em = np.sqrt(np.einsum('nab,nba->n', hm, hm).real/4)
        pp = (np.eye(4)[None]+hp/Ep[:, None, None])/2
        pm = (np.eye(4)[None]-hm/Em[:, None, None])/2
        current_new = native_current(paths, q/2)
        # J(0,-q) acts from the negative sea into e; its adjoint acts on e.
        current_blocked = native_current(paths, -q/2).conj().swapaxes(-1, -2)
        new = projected_transition(current_new, pp, transverse, rest)
        blocked = projected_transition(current_blocked, pm, transverse, rest)
        omega = c*Q/band.a
        leg_without_g2 = 1/(2*Q)
        values = dict(
            contact=.5*leg_without_g2[:, None, None]*np.einsum('njj,jst->nst', transverse, rest_contact),
            new_positive=-leg_without_g2[:, None, None]*new/(Ep-band.m+omega)[:, None, None],
            blocked_negative=leg_without_g2[:, None, None]*blocked/(Em+band.m+omega)[:, None, None],
            Coulomb=c/(2*band.a*Q2[:, None, None])*np.einsum('as,nab,bt->nst', rest.conj(),
                np.eye(4)[None]-(np.eye(4)[None]-hp/Ep[:, None, None])/2-pm, rest))
        for name, value in values.items():
            matrices[name] += np.einsum('n,nst->st', w, value)
        M = z+r*np.sum(1-np.cos(q), axis=1)
        E = np.sqrt(M*M+c*c*np.sum(np.sin(q)**2, axis=1)); Omega = c*Q
        T = np.sum(np.cos(q/2)**2*np.diagonal(transverse, axis1=1, axis2=2), axis=1)
        finite = dict(contact=r/(2*band.a*Q),
            new_positive=-c*c*T*(1-M/E)/(4*band.a*Q*(E-z+Omega)),
            blocked_negative=c*c*T*(1+M/E)/(4*band.a*Q*(E+z+Omega)),
            Coulomb=c*M/(2*band.a*E*Q2))
        virtual_stable = c*c*T*((M/E)*(E+Omega)-z)/(2*band.a*Q*((E+Omega)**2-z*z))
        accept(finite['new_positive']+finite['blocked_negative'], virtual_stable,
               'finite-mass algebraic virtual combination', 2e-7)
        minimum_virtual = min(minimum_virtual, float(virtual_stable.min()))
        for name, value in finite.items():
            closed[name] += float(w@value)
        M0 = r*np.sum(1-np.cos(q), axis=1)
        E0 = np.sqrt(M0*M0+c*c*np.sum(np.sin(q)**2, axis=1))
        uv_values = dict(contact=r/(2*Q), virtual=c*c*M0*T/(2*Q*E0*(E0+Omega)),
                         Coulomb=c*M0/(2*E0*Q2), reference_self=c/(2*Q2))
        for name, value in uv_values.items():
            uv[name] += float(w@value)
        reference += float(w@(c/(2*band.a*Q2)))
    scalar = {name: scalarize(value, f'cubic rest {name}', coefficient_scale=band.a)
              for name, value in matrices.items()}
    errors = {name: accept(scalar[name], closed[name], f'projector versus derived {name}', 2e-6)
              for name in scalar}
    total = sum(scalar.values())
    coefficient = uv['contact']+uv['virtual']+uv['Coulomb']
    if min(coefficient, minimum_virtual, total) <= 0:
        raise AssertionError('Positive native Wilson Gaussian coefficient failed')
    return dict(order_per_split_interval=order, quadrature='split_gauss', points=len(points),
        normalized_measure=float(weights.sum()), field_graph_is_not_BZ_quadrature=True,
        coefficient_per_g2=scalar, virtual_per_g2=scalar['new_positive']+scalar['blocked_negative'],
        carrier_gap_shift_per_g2=total, static_opposite_reference_self_per_g2=reference,
        compensated_reference_gap_shift_per_g2=total+reference,
        a_times_carrier_shift_per_g2=band.a*total,
        UV_frozen_frame_C=coefficient, UV_terms=uv,
        UV_compensated_reference_C=coefficient+uv['reference_self'],
        finite_mass_closed_formula_errors=errors, minimum_finite_virtual_integrand=minimum_virtual,
        scope='thermodynamic bare-Gaussian BZ using this frozen N-site field frame; not a new field graph')


def full_car_benchmark(bare, band, frame, unitary, g2):
    """Entire 256-state CAR algebra, exact one-photon sector resolvents."""
    zero, q = np.zeros(3), np.array([np.pi, 0., 0.])/band.a
    _, paths = raw_paths(bare, frame)
    h = np.asarray(block_diag(raw_h(bare, frame, zero), raw_h(bare, frame, q)), dtype=np.complex128)
    j = native_current(paths, np.array([[np.pi/2, 0., 0.]]))[0, 1]
    current = np.block([[np.zeros((4, 4)), j.conj().T], [j, np.zeros((4, 4))]])
    second = np.asarray(block_diag(native_contact(paths, zero[None])[0, 1],
                        native_contact(paths, (band.a*q)[None])[0, 1]), dtype=np.complex128)
    density = np.block([[np.zeros((4, 4)), np.eye(4)], [np.eye(4), np.zeros((4, 4))]])
    sectors = [exterior_hamiltonian(h, n) for n in range(9)]
    dimensions = [len(states) for _, states in sectors]
    if sum(dimensions) != 256 or dimensions[4:6] != [70, 56]:
        raise AssertionError('All eight-mode CAR states must be retained')
    all_current = np.asarray(block_diag(*(exterior_hamiltonian(current, n)[0] for n in range(9))), dtype=np.complex128)
    all_contact = np.asarray(block_diag(*(exterior_hamiltonian(second, n)[0] for n in range(9))), dtype=np.complex128)
    all_density = np.asarray(block_diag(*(exterior_hamiltonian(density, n)[0] for n in range(9))), dtype=np.complex128)
    offsets = np.r_[0, np.cumsum(dimensions)]
    Q, V = 2., 2
    omega = 2*band.v/band.a
    # Compute exact order-g² coefficients; apply g² only at the reporting boundary.
    leg_coefficient = 1/(2*Q*V)
    ue_coefficient = band.v/(2*band.a)
    coulomb_coefficient = ue_coefficient/(V*Q**2)
    reference_self = coulomb_coefficient
    results = {}
    for n, degeneracy in ((4, 1), (5, 2)):
        H, _ = sectors[n]
        energies, vectors = np.linalg.eigh(H)
        ground_energy = float(energies[0])
        gap = energies-ground_energy
        ground_tolerance = min(1e-8/band.a, band.m/4)
        ground_count = int(np.count_nonzero(gap < ground_tolerance))
        if ground_count != degeneracy:
            raise ArithmeticError(f'Native CAR ground cluster unresolved at a={band.a}: '
                                  f'expected {degeneracy}, measured {ground_count}, '
                                  f'tolerance {ground_tolerance}, rest gap {2*band.m}')
        ground = vectors[:, :degeneracy]
        section = slice(offsets[n], offsets[n+1])
        J, C = all_current[section, section], all_contact[section, section]
        # n=1 is the exact intermediate of a linear vertex on photon vacuum.
        intermediate = J@ground
        virtual = -leg_coefficient*intermediate.conj().T@np.linalg.solve(
            H-ground_energy*np.eye(len(H))+omega*np.eye(len(H)), intermediate)
        contact = .5*leg_coefficient*ground.conj().T@C@ground
        R = all_density[:, section]@ground
        coulomb = coulomb_coefficient*R.conj().T@R
        density_mean = ground.conj().T@all_density[section, section]@ground
        # A static reference -1 at position zero supplies (rho-I)^2.
        reference_cross = -2*coulomb_coefficient*density_mean
        results[n] = dict(energy=ground_energy, virtual=virtual, contact=contact,
                          Coulomb=coulomb, reference_cross=reference_cross)
    gaps = {name: results[5][name]-results[4][name][0, 0]*np.eye(2)
            for name in ('virtual', 'contact', 'Coulomb')}
    exact = sum(gaps.values())
    # Independent occupied/empty formula in the one-body eigenbasis.
    eig, U = np.linalg.eigh(h)
    vertex = U.conj().T@current@U
    occupied = np.flatnonzero(eig < 0)
    positive = np.flatnonzero(eig > 0)
    electrons = positive[np.isclose(eig[positive], band.m, rtol=1e-7, atol=1e-9)]
    if len(occupied) != 4 or len(electrons) != 2:
        raise ArithmeticError(f'Native negative sea/rest doublet unresolved at a={band.a}')
    formula_rows = []
    for electron in electrons:
        empty = [k for k in positive if k != electron]
        new = -leg_coefficient*sum(abs(vertex[jj, electron])**2/(eig[jj]-eig[electron]+omega) for jj in empty)
        blocked = leg_coefficient*sum(abs(vertex[electron, ii])**2/(eig[electron]-eig[ii]+omega) for ii in occupied)
        contact = .5*leg_coefficient*float(np.vdot(U[:, electron], second@U[:, electron]).real)
        occupation = U[:, occupied]@U[:, occupied].conj().T
        # Wick delta Tr[n rho (I-n) rho], with the added rank-one projector.
        added = np.outer(U[:, electron], U[:, electron].conj())
        neutral_wick = np.trace(occupation@density@(np.eye(8)-occupation)@density)
        charged_wick = np.trace((occupation+added)@density@(np.eye(8)-occupation-added)@density)
        coulomb = coulomb_coefficient*float((charged_wick-neutral_wick).real)
        formula_rows.append(dict(new_positive=float(new), blocked_negative=float(blocked),
                                 contact=contact, Coulomb=coulomb))
    formula = {key: sum(row[key] for row in formula_rows)/2 for key in formula_rows[0]}
    expected = sum(formula.values())*np.eye(2)
    error = accept(band.a*exact, band.a*expected, 'complete CAR carrier gap')
    component_errors = dict(virtual=accept(band.a*gaps['virtual'],
        band.a*(formula['new_positive']+formula['blocked_negative'])*np.eye(2),
        'whole-sector virtual gap'),
        contact=accept(band.a*gaps['contact'], band.a*formula['contact']*np.eye(2),
                       'whole-sector contact gap'),
        Coulomb=accept(band.a*gaps['Coulomb'], band.a*formula['Coulomb']*np.eye(2),
                       'whole-sector Gauss density gap'))
    accept(band.a*formula['Coulomb'], band.a*coulomb_coefficient, 'two-momentum Gauss Wick delta')
    reference_cross = results[5]['reference_cross']
    accept(band.a*reference_cross, np.zeros((2, 2)), 'uniform carrier static-reference cross')
    accept(band.a*(exact+reference_self*np.eye(2)+reference_cross),
           band.a*(expected+reference_self*np.eye(2)), 'declared opposite static reference')
    adverse = {}
    for name, removed in (('missing_blocked_negative', formula['blocked_negative']),
                          ('missing_contact', formula['contact']),
                          ('missing_Coulomb_transition', formula['Coulomb'])):
        try:
            accept(band.a*(sum(formula.values())-removed)*np.eye(2), band.a*exact, 'complete CAR carrier gap')
        except AssertionError as failure:
            adverse[name] = dict(rejected_by_same_acceptance=True, error=str(failure), removed=float(g2*removed))
        else:
            raise AssertionError(f'{name} did not fail the same full-CAR acceptance')
    return dict(modes=8, entire_CAR_states=256, conserved_sector_dimensions=dimensions,
        neutral_particles=4, neutral_dimension=70, charged_particles=5, charged_dimension=56,
        momenta_angles=[[0., 0., 0.], [float(np.pi), 0., 0.]], transverse_polarization=[0., 1., 0.],
        V=V, Qhat=Q, oscillator_frequency=omega, vacuum_link_leg_squared=g2*leg_coefficient,
        electric_ue=g2*ue_coefficient, electric_UE1=2*g2*ue_coefficient,
        exact_gap_order_g2=g2*scalarize(exact, 'full CAR scalar gap', coefficient_scale=band.a),
        exact_components={name: g2*scalarize(value, name, coefficient_scale=band.a)
                          for name, value in gaps.items()},
        occupied_empty_components={name: g2*value for name, value in formula.items()},
        neutral_virtual_shift=g2*float(results[4]['virtual'][0, 0].real),
        charged_virtual_shift=g2*scalarize(results[5]['virtual'], 'charged virtual doublet',
                                         coefficient_scale=band.a),
        opposite_static_reference_self=g2*reference_self,
        reference_cross_expectation=g2*scalarize(reference_cross, 'reference cross',
                                              coefficient_scale=band.a),
        acceptance_units='a times order-g2 energy coefficient; reported shifts include physical g2',
        complete_gap_error=error, component_errors=component_errors, adverse=adverse,
        scope='two-momentum algebra benchmark, not a 3D field graph; n=1 exact perturbative intermediate, no oscillator cutoff')


def run(spacings=(1., .001, .0001), orders=(6, 8), field_size=5, g2=.09):
    if field_size < 2 or g2 <= 0 or not orders or min(orders) < 2 or min(spacings) <= 0:
        raise ValueError('Positive spacings/coupling and nontrivial graph/quadrature required')
    before = source_hashes()
    rows = []
    for a in spacings:
        bare = RotationBand(**{**BARE, 'a': float(a)}, e=1.)
        if bare.improved:
            raise AssertionError('Only the unchanged unimproved native source is owned')
        selection, frame = original_frame(bare, field_size, np.zeros(3), True)
        band, unitary, reconstruction = reconstruct_frame(bare, frame)
        unit_band = RotationBand(a=1., m=a*band.m, r=band.r, v=band.v,
                                 e=band.e, improved=band.improved)
        primitive = primitive_frame_controls(dimensionless_bare(a), unit_band, frame, unitary)
        legs = matrix_leg_controls(bare, band, frame, unitary)
        axial = axial_controls(bare, band, frame, unitary)
        integrations = [bz_diagnostic(bare, band, frame, unitary, int(order)) for order in orders]
        car = full_car_benchmark(bare, band, frame, unitary, g2)
        ue, K = band.v*g2/(2*a), band.v/(g2*a)
        rows.append(dict(a=a, primitive_bare=dict(a=bare.a, m=bare.m, r=bare.r, v=bare.v,
            e=bare.e, improved=bare.improved), field_graph_N=field_size, field_graph_V=field_size**3,
            centre_toron=[0, 0, 0], original_full10_Gram=selection['G'].tolist(),
            original_Gram_gradient=float(selection['gradient_residual']), reconstruction=reconstruction,
            native_primitive_controls=primitive, matrix_photon_scalar_leg_controls=legs,
            axial_vs_spectral_pairing=axial,
            photon=dict(g2=g2, ue=ue, UE1=2*ue, e1=1., K=K,
                units='UE1=2 ue/e1; omega=c_e |qhat|/a; link vacuum leg2=g2/(2 |qhat|)',
                q0='excluded harmonic compact rotor; no Gaussian oscillator assignment'),
            BZ=integrations, full_CAR=car,
            order_g2_carrier_shift=g2*integrations[-1]['carrier_gap_shift_per_g2'],
            order_g2_static_reference_self=g2*integrations[-1]['static_opposite_reference_self_per_g2'],
            order_g2_shift_over_tree_mass=g2*integrations[-1]['carrier_gap_shift_per_g2']/band.m))
    after = source_hashes()
    if before != after:
        raise AssertionError('Source changed during native diagnostic')
    return dict(plan_id=PLAN, plan_revision=PLAN_REVISION, template_revision=TEMPLATE_REVISION,
        source_sha256=before, source_bindings_unchanged_during_run=True,
        runtime=dict(executable=sys.executable, python=platform.python_version(), numpy=np.__version__,
            scipy=scipy.__version__, BLAS_threads={key: os.environ.get(key) for key in
            ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS')}),
        inputs=dict(spacings=list(spacings), orders=list(orders), field_graph_N=field_size,
                    g2=g2, primitive_bare=BARE, integration='normalized split_gauss thermodynamic BZ'),
        rows=rows, scope='formal bare Gaussian order-g2 momentum-state carrier-vacuum gap with declared opposite static reference; no compact phase/full interacting pole',
        conclusion='native Wilson order-g2/a coefficient is strictly positive analytically; fixed positive bare mass is not perturbatively UV protected',
        remaining=['nonlinear compact photon phase and Gauss ground', 'renormalized interacting carrier pole and localization',
                   'uniform field/volume BO control', 'actual axial-protection or counterterm construction without losing required gap'],
        all_checks=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a', nargs='+', type=float, default=[1., .001, .0001])
    parser.add_argument('--orders', nargs='+', type=int, default=[6, 8])
    parser.add_argument('--field-size', type=int, default=5)
    parser.add_argument('--g2', type=float, default=.09)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(tuple(args.a), tuple(args.orders), args.field_size, args.g2)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text+'\n')
    print(text)


if __name__ == '__main__':
    main()
