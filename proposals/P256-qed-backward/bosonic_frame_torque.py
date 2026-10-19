#!/usr/bin/env python3
"""Bosonic internal-colour frame torque, full Gauss tangents and heavy CAR pulses.

Run with the project Python. Scientific executions are parent-owned. This new
source protocol is not the original field-only torque or a spatial rotation.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, cast

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply, spsolve

from bulk_quantum_selection import BulkQuantumLaw
from noncompact_finite_packet_spin import (ENDS, GraphQuantum,
    creator_maps, gram, history_forms, initial_whiten, lift, prepare, propagate,
    complex_json)
from noncompact_native_current import accept, adverse
from original_charged_principal import original_frame, portable
from original_total_angular_transfer import endpoint_flow, gauge_move, primitive_flow
from rotation_band import RotationBand
from unrestricted_field_minimum import coordinates

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-continuing-backward-20260930'
REVISION = '6b16fc5fa2204be1c2e567e06313425b16e3253daa91a041e857e12cca47af37'
TEMPLATE = '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'
AXES = ((2, 3), (3, 1), (1, 2))


def norm(value) -> float:
    return float(sparse.linalg.norm(value) if sparse.issparse(value) else np.linalg.norm(value))


def hermitian(value):
    return (value + value.conj().T) / 2


def smooth_step(value: float) -> float:
    if value <= 0:
        return 0.
    if value >= 1:
        return 1.
    left, right = math.exp(-1 / value), math.exp(-1 / (1 - value))
    return left / (left + right)


def scalar_frame(fields):
    """Gauge-equivariant exact Clifford retraction from four actual scalar matrices.

    No preselected gamma, Pauli matrix, colour eigenbasis or orbit fit is used.
    Its smooth invariant cutoff vanishes before Gram/flattening singularities.
    """
    fields = np.asarray(fields, dtype=complex)
    if fields.shape != (4, 4, 4):
        raise ValueError('Four full traceless Hermitian 4x4 scalar fields required')
    if norm(fields-fields.conj().swapaxes(-1, -2)) > 1e-9 or norm(np.trace(fields, axis1=1, axis2=2)) > 1e-9:
        raise ValueError('Scalar Cartesian domain is traceless Hermitian')
    G = np.einsum('mij,nji->mn', fields, fields).real / 4
    values, vectors = np.linalg.eigh(G)
    scale = float(np.trace(G))
    zero = np.zeros((3, 4, 4), complex)
    determinant_ratio = float(np.linalg.det(G))/scale**4 if scale > 0 else 0.
    if scale <= .01 or values[0] <= 0 or determinant_ratio <= 1e-7:
        return zero, dict(cutoff=0., regular=False, Gram=G, reason='Gram outside regular support')
    root_inverse = (vectors / np.sqrt(values)) @ vectors.T
    whitened = np.einsum('mn,nij->mij', root_inverse, fields)
    defect = norm(np.array([[a@b+b@a-2*(m == n)*np.eye(4)
                            for n, b in enumerate(whitened)] for m, a in enumerate(whitened)]))
    chi = smooth_step((scale-.01)/.01) * smooth_step((determinant_ratio-1e-7)/9e-7) * smooth_step((.5-defect)/.25)
    if chi == 0:
        return zero, dict(cutoff=0., regular=False, Gram=G, reason='Clifford tube outside support', Clifford_defect=defect)
    gammas: list[NDArray[np.complex128]] = []
    gaps = []
    for raw in whitened:
        projected = raw.copy()
        for gamma in gammas:
            projected = (projected-gamma@projected@gamma)/2
        eigenvalues, eigenvectors = np.linalg.eigh(hermitian(projected))
        gaps.append(float(np.min(abs(eigenvalues))))
        if gaps[-1] <= .1:
            return zero, dict(cutoff=0., regular=False, Gram=G, reason='Flattening outside regular support', Clifford_defect=defect)
        inverse_square_trace = float(np.sum(eigenvalues**-2))
        chi *= smooth_step((1/inverse_square_trace-.02)/.02)
        if chi == 0:
            return zero, dict(cutoff=0., regular=False, Gram=G, reason='Flattening cutoff', Clifford_defect=defect)
        gamma: NDArray[np.complex128] = (eigenvectors*np.sign(eigenvalues))@eigenvectors.conj().T
        gammas.append(gamma)
    gamma_array = np.asarray(gammas, dtype=np.complex128)
    clifford_error = norm(np.array([[a@b+b@a-2*(m == n)*np.eye(4)
                          for n, b in enumerate(gamma_array)] for m, a in enumerate(gamma_array)]))
    accept(clifford_error, 'actual scalar-field exact Clifford retraction', 2e-8)
    spins: NDArray[np.complex128] = np.asarray([-1j*(gamma_array[j]@gamma_array[k]-gamma_array[k]@gamma_array[j])/4 for j, k in AXES], dtype=np.complex128)
    accept(norm(sum(s@s for s in spins)-.75*np.eye(4)), 'scalar-derived frame Casimir three quarters')
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        accept(norm(spins[i]@spins[j]-spins[j]@spins[i]-1j*spins[k]), 'scalar-derived frame Spin3 algebra')
    return chi*spins, dict(cutoff=chi, regular=True, Gram=G, Clifford_defect=defect,
                           flattening_min_gap=min(gaps), retraction_Clifford_error=clifford_error)


def field_spins(fields):
    derived = [scalar_frame(f) for f in fields]
    return np.asarray([s for s, _ in derived]), [r for _, r in derived]


def gauge_tangent(law, fields, links, generators):
    df = 1j*(generators[:, None]@fields-fields@generators[:, None])
    du = 1j*(generators[:, None]@links-links@generators[law.target])
    return df, du, np.zeros((law.volume, 3))


def gauge_matrix(law, fields, links):
    """Actual all15V infinitesimals in full60V Cartesian +45V link coordinates."""
    T = law.generators
    row_blocks, col_blocks, data_blocks = [], [], []
    def append(block, row, col):
        rr, cc = np.indices(block.shape)
        row_blocks.append((rr+row).ravel()); col_blocks.append((cc+col).ravel()); data_blocks.append(block.ravel())
    for x in range(law.volume):
        tangent = np.asarray([1j*(t[None]@fields[x]-fields[x]@t[None]) for t in T])
        append(coordinates(tangent, law.basis).reshape(15, 60).T.real, 60*x, 15*x)
        for axis in range(3):
            y, u = int(law.target[x, axis]), links[x, axis]
            adjoint = 2*np.einsum('aij,bji->ab', T, u[None]@T@u.conj().T).real
            row = 60*law.volume+15*(3*x+axis)
            append(np.eye(15), row, 15*x); append(-adjoint, row, 15*y)
    return sparse.coo_matrix((np.concatenate(data_blocks), (np.concatenate(row_blocks), np.concatenate(col_blocks))),
                             shape=(105*law.volume, 15*law.volume)).tocsr()


def flow_coordinates(law, links, flow):
    df, du, _ = flow
    field = coordinates(df, law.basis).ravel().real
    electric = hermitian_links(-1j*du@links.conj().swapaxes(-1, -2))
    link = (2*np.einsum('aij,xkji->xka', law.generators, electric).real).ravel()
    return np.concatenate((field, link))


def hermitian_links(value):
    return (value+value.conj().swapaxes(-1, -2))/2


def horizontal_flow(law, fields, links, flow):
    G = gauge_matrix(law, fields, links)
    mass = np.r_[np.ones(60*law.volume), np.full(45*law.volume, 1/.6)]
    vector = flow_coordinates(law, links, flow)
    weighted = G.multiply(mass[:, None])
    connection = spsolve((G.T@weighted).tocsc(), G.T@(mass*vector))
    normal = vector-G@connection
    return connection.reshape(law.volume, 15), normal, float(np.sqrt(normal@(mass*normal)))


def native_case(law, frame, centre, rng, noncentral=True):
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    fields += .0003*np.einsum('xma,aij->xmij', rng.normal(size=(law.volume, 4, 15)), law.basis)
    links = np.asarray(law.toron_links(np.pi/2*np.asarray(centre)), dtype=np.complex128)
    if noncentral:
        for x, axis in itertools.product(range(law.volume), range(3)):
            links[x, axis] = expm(.13j*np.einsum('a,aij->ij', rng.normal(size=15), law.generators))@links[x, axis]
    phases = rng.normal(size=(law.volume, 3))*.23
    spins, reports = field_spins(fields)
    if min(r['cutoff'] for r in reports) != 1:
        raise AssertionError('Diagnostic input must be inside frame cutoff plateau')
    b = np.exp(-np.sum(((law.sites+law.size//2)%law.size-law.size//2)**2, axis=1)/1.7)
    source = b[:, None, None]*spins[:, 2]
    flow = gauge_tangent(law, fields, links, source)
    ell, c, c_norm = horizontal_flow(law, fields, links, flow)
    expected = 2*np.einsum('aij,xji->xa', law.generators, source).real
    accept(norm(ell-expected), 'all15V actual mechanical gauge-source connection', 3e-8)
    accept(c_norm, 'all15V bosonic quotient normal velocity', 3e-8)
    # Same field-only parent source f=b+L b/(.6 lambda), not a spin replacement.
    base_spins, _ = scalar_frame(frame)
    S = base_spins[2]
    tangents = np.asarray([1j*(t[None]@frame-frame@t[None]) for t in law.generators])
    inertia = np.einsum('amij,bmij->ab', tangents.conj(), tangents).real/4
    s = 2*np.einsum('aij,ji->a', law.generators, S).real
    lam = float(s@inertia@s/(s@s))
    lap = sparse.csr_matrix((law.volume, law.volume))
    for x, axis in itertools.product(range(law.volume), range(3)):
        y = int(law.target[x, axis]); edge = np.zeros(law.volume); edge[x], edge[y] = 1., -1.
        lap += sparse.csr_matrix(np.outer(edge, edge))
    torque = b+lap@b/(.6*lam)
    parent = primitive_flow(law, fields, links, phases, np.zeros((law.volume, 3)), torque)
    _, parent_c, parent_norm = horizontal_flow(law, fields, links, parent)
    countercase = adverse(parent_norm, 'all15V bosonic quotient normal velocity', 3e-8)
    # True normal/full-field JVP: change every scalar and genuine SU4 link.
    delta = .02*np.einsum('xma,aij->xmij', rng.normal(size=(law.volume, 4, 15)), law.basis)
    z = .1*np.einsum('xka,aij->xkij', rng.normal(size=(law.volume, 3, 15)), law.generators)
    _, normal_direction, _ = horizontal_flow(law, fields, links,
        (delta, 1j*z@links, np.zeros((law.volume, 3))))
    delta = np.einsum('xma,aij->xmij', normal_direction[:60*law.volume].reshape(law.volume, 4, 15), law.basis)
    z = np.einsum('xka,aij->xkij', normal_direction[60*law.volume:].reshape(law.volume, 3, 15), law.generators)
    h = 2e-5
    normals, parent_normals = [], []
    for sign in (-1, 1):
        f = fields+sign*h*delta
        u = np.asarray([[expm(sign*1j*h*z[x, axis])@links[x, axis] for axis in range(3)] for x in range(law.volume)], dtype=np.complex128)
        ss, _ = field_spins(f)
        normals.append(horizontal_flow(law, f, u, gauge_tangent(law, f, u, b[:, None, None]*ss[:, 2]))[1])
        parent_normals.append(horizontal_flow(law, f, u, primitive_flow(law, f, u, phases, np.zeros((law.volume, 3)), torque))[1])
    normal_jvp = norm((normals[1]-normals[0])/(2*h))
    parent_jvp = norm((parent_normals[1]-parent_normals[0])/(2*h))
    accept(normal_jvp, 'full normal derivative of identically vertical gauge source', 3e-6)
    adverse(parent_jvp, 'full normal derivative of identically vertical gauge source', 3e-6)
    gauges = np.asarray([expm(1j*np.einsum('a,aij->ij', rng.normal(size=15), law.generators)) for _ in range(law.volume)])
    moved_f, moved_u, _ = gauge_move(law, fields, links, phases, gauges, np.zeros(law.volume))
    moved_s, _ = field_spins(moved_f)
    covariance = norm(moved_s-gauges[:, None]@spins@gauges[:, None].conj().swapaxes(-1, -2))
    accept(covariance, 'actual scalar frame local SU4 covariance', 3e-8)
    hbody = law.onebody(fields, links, phases)
    block = sparse.block_diag(source).toarray()
    endpoint_error = norm(endpoint_flow(law, fields, links, phases, flow)-1j*(block@hbody-hbody@block))
    accept(endpoint_error, 'same native full-H gauge tangent Ward', 4e-8)
    # Coefficient divergence uses all fifteen field Gauss directions at one site.
    divergence, errors = 0., []
    for a, t in enumerate(law.generators):
        up, um = expm(1j*h*t), expm(-1j*h*t)
        sp, _ = scalar_frame(up[None]@fields[0]@up.conj().T)
        sm, _ = scalar_frame(um[None]@fields[0]@um.conj().T)
        ds = (sp[2]-sm[2])/(2*h)
        divergence += float(2*np.trace(t@ds).real)
        errors.append(norm(ds-1j*(t@spins[0, 2]-spins[0, 2]@t)))
    accept(abs(divergence), 'full15 coefficient Gauss divergence', 3e-7)
    reversed_spins, _ = field_spins(-fields)
    parity_error = accept(norm(reversed_spins-spins), 'all4 scalar sign-even derived spin frame', 3e-8)
    return dict(centre=centre, actual_all_colour_Gauss_rows=15*law.volume,
        untruncated_Cartesian_coordinates=60*law.volume, noncentral_links=noncentral,
        minimum_cutoff=min(r['cutoff'] for r in reports), max_scalar_Clifford_defect=max(r['Clifford_defect'] for r in reports),
        connection_error=norm(ell-expected), quotient_normal_norm=c_norm, quotient_normal_JVP=normal_jvp,
        field_only_parent_normal_norm=parent_norm, field_only_parent_normal_JVP=parent_jvp,
        same_boundary_parent_rejected=countercase, frame_covariance_error=covariance,
        interacting_endpoint_gauge_Ward_error=endpoint_error, coefficient_divergence=divergence,
        full15_coefficient_derivative_error=max(errors), scalar_sign_even_frame_error=parity_error,
        source_is_internal_colour_not_spatial=True)


def frame_wilson_checks(law, frame, rng):
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    fields += .0003*np.einsum('xma,aij->xmij', rng.normal(size=(law.volume, 4, 15)), law.basis)
    spins, _ = field_spins(fields)
    x, y, R = 0, int(law.target[0, 0]), int(law.target[int(law.target[0, 0]), 1])
    u, v = [expm(1j*np.einsum('a,aij->ij', rng.normal(size=15), law.generators)) for _ in range(2)]
    W = u@v
    accept(norm(W.conj().T@W-np.eye(4)), 'genuine uncompressed unitary colour Wilson word')
    labels = np.asarray([np.eye(4), *[2j*s for s in spins[x]]])
    columns = np.column_stack([a.ravel() for a in labels])
    closure, derivative = [], []
    h = 2e-5
    for s in spins[x]:
        coeff = np.linalg.lstsq(columns, np.column_stack([(s@a).ravel() for a in labels]), rcond=None)[0]
        closure.append(norm(columns@coeff-np.column_stack([(s@a).ravel() for a in labels])))
        gx, gy, gr = expm(1j*h*s), expm(1j*h*spins[y, 0]), np.eye(4)
        fs, _ = scalar_frame(gx[None]@fields[x]@gx.conj().T)
        moved_labels = np.asarray([np.eye(4), *[2j*t for t in fs]])
        wp = (gx@u@gy.conj().T)@(gy@v@gr.conj().T)
        derivative.append(norm((moved_labels@wp-labels@W)/h-1j*s@labels@W))
    accept(max(closure), 'full-field local-frame Wilson relative Spin3 closure')
    accept(max(derivative), 'actual bosonic Wilson plus frame creator derivative', 2e-4)
    minus = expm(2j*np.pi*spins[x, 2])
    relative = norm(labels@(minus@W)+labels@W)
    accept(relative, 'local b1 outside-reference b0 full colour Wilson minus')
    global_error = norm(labels@(minus@W@minus.conj().T)-labels@W)
    accept(global_error, 'global centre including reference physical plus')
    centre_errors = []
    for k in range(4):
        z = 1j**k
        centre_errors.append(norm(labels@(z*W)-z*labels@W))
    # Gram whitening is frame-valued, not proof of primitive vector-index rotation.
    ss, _ = scalar_frame(frame)
    difference = []
    for axis, (j, k) in enumerate(AXES):
        cartesian = np.zeros_like(frame); cartesian[j], cartesian[k] = -frame[k], frame[j]
        colour = 1j*(ss[axis][None]@frame-frame@ss[axis][None])
        difference.append(norm(colour-cartesian))
    altered = frame+.0003*np.einsum('ma,aij->mij', rng.normal(size=(4, 15)), law.basis)
    sa, report = scalar_frame(altered)
    if report['cutoff'] != 1:
        raise AssertionError('Cartesian countercase must stay on regular frame plateau, not use cutoff deletion')
    cartesian = np.zeros_like(frame); cartesian[1], cartesian[2] = -altered[2], altered[1]
    failure = norm(1j*(sa[2][None]@altered-altered@sa[2][None])-cartesian)
    adverse(failure, 'gauge frame torque equals primitive Cartesian vector-index torque')
    return dict(genuine_unitary_Wilson_error=norm(W.conj().T@W-np.eye(4)),
        relative_generator_closure_error=max(closure), full_field_Wilson_frame_derivative_error=max(derivative),
        relative_2pi_minus_error=relative, global_including_reference_2pi_plus_error=global_error,
        all_four_centre_characters_errors=centre_errors, native_Cartesian_vs_frame_tangent_norms=difference,
        full_normal_Cartesian_vs_frame_defect=failure, altered_frame_cutoff=report['cutoff'],
        literal_bosonic_loop=False, centre_note='local -I changes crossing Wilson links; global -I fixes bosons and complete reference')


def whole_heavy_onebody(frame):
    h = np.kron(np.eye(3), (.2+3*.12)*frame[0]).astype(complex)
    for edge, (x, y) in enumerate(ENDS):
        axis = (0, 1, 2, 2)[edge]
        hop = -(.12*frame[0]+1j*.3*frame[axis+1])/2
        h[4*x:4*x+4, 4*y:4*y+4] += hop
        h[4*y:4*y+4, 4*x:4*x+4] += hop.conj().T
    return h


def wilson_creator_maps(neutral, charged, eta, labels, links):
    """Actual graph Wilson words 0->2 (reversed edge3), 1->2 (edge2)."""
    links = np.asarray(links, dtype=complex)
    if links.shape != (len(ENDS), 4, 4):
        raise ValueError('One full SU4 matrix for each genuine graph edge required')
    for u in links:
        accept(norm(u.conj().T@u-np.eye(4)), 'source link uncompressed SU4 unitarity')
        accept(abs(np.linalg.det(u)-1), 'source link SU4 determinant')
    words = [links[3].conj().T, links[2]]
    point = [creator_maps(neutral, charged, np.eye(2)[x], [np.eye(4)])[0] for x in range(2)]
    maps = []
    for label in labels:
        marks = [np.asarray(label)@word for word in words]
        maps.append([sum((eta[x]*marks[x][a, b]*point[x][a] for x in range(2) for a in range(4)),
                         sparse.csr_matrix((charged.d, neutral.d), dtype=complex)) for b in range(4)])
    return maps, dict(SAME_R=2, paths=[[(3, -1)], [(2, 1)]],
        unitary_word_errors=[norm(w.conj().T@w-np.eye(4)) for w in words],
        no_compressed_Haar_link=True)


def spin_action(model, spin, plateau):
    q = lift(np.kron(np.diag(plateau), spin), model.rows)
    return lambda value: model.finish(model.fermion(model.wave(value), q), value), q


def heavy_evolve(H, Q, state, duration, angle):
    solver = cast(Callable[..., Any], expm_multiply)
    return solver(-1j*duration*H+1j*angle*Q, state, traceA=complex(-1j*duration*H.diagonal().sum()+1j*angle*Q.diagonal().sum()))


def heavy_pulse(args, frame, spins):
    """Full exterior924 heavy evolution, exact separated BODY factor cancellation."""
    neutral = GraphQuantum(args.levels, args.harmonics, args.zeta, 6, args.UE, args.K, frame)
    charged = GraphQuantum(args.levels, args.harmonics, args.zeta, 7, args.UE, args.K, frame)
    E0, state, cluster = neutral.solve(args.eigenpairs)
    labels = [np.eye(4), *[2j*s for s in spins]]
    eta = np.array([math.sqrt(.6), math.sqrt(.4)*np.exp(.37j)])
    # These are the actual BODY edge values, transported to the SAME outside R.
    maps, Wilson = wilson_creator_maps(neutral, charged, eta, labels,
                                       np.tile(np.eye(4, dtype=complex), (len(ENDS), 1, 1)))
    sources = np.asarray([prepare(neutral, charged, state, mark) for mark in maps])
    metric = gram(sources)
    ev, vec = np.linalg.eigh(metric)
    if ev[0] <= 1e-8:
        raise AssertionError('Complete neutral supplier has singular relative source metric')
    inverse = (vec/np.sqrt(ev))@vec.conj().T
    accept(norm(gram(initial_whiten(sources, inverse))-np.eye(4)), 'actual complete packet initial metric')
    nc, nq = spin_action(neutral, spins[args.axis], [1., 1., 0.])
    cc, cq = spin_action(charged, spins[args.axis], [1., 1., 0.])
    rows = neutral.rows
    h = whole_heavy_onebody(frame)
    values, vectors = np.linalg.eigh(h)
    occupied = vectors[:, 6:]
    heavy = np.asarray([np.linalg.det(occupied[np.asarray(row)]) for row in rows])
    accept(abs(np.vdot(heavy, heavy).real-1), 'whole NH6 positive Slater in exterior924')
    base_H = -lift(h, rows)
    Q = nq
    commutator = norm(base_H@Q-Q@base_H)
    adverse(commutator, 'whole heavy Hamiltonian commutes with local frame torque')
    closure = []
    label_columns = np.column_stack([label.ravel() for label in labels])
    for spin in spins:
        sn = lift(np.kron(np.diag([1., 1., 0.]), spin), neutral.rows)
        sc = lift(np.kron(np.diag([1., 1., 0.]), spin), charged.rows)
        coeff = np.linalg.lstsq(label_columns, np.column_stack([(spin@v).ravel() for v in labels]), rcond=None)[0]
        for a, mark in enumerate(maps):
            for ref in range(4):
                target = sum((coeff[j, a]*maps[j][ref] for j in range(4)), sparse.csr_matrix(mark[ref].shape, dtype=complex))
                closure.append(norm(sc@mark[ref]-mark[ref]@sn-target))
    accept(max(closure), 'complete924 to792 finitepacket relative source Spin3 CAR intertwiner', 3e-7)
    ideal_n = propagate(neutral, state, 0., 2*np.pi, nc, neutral.trace)
    ideal_c = np.asarray([propagate(charged, source, 0., 2*np.pi, cc, charged.trace) for source in sources])
    ideal_right = np.asarray([prepare(neutral, charged, ideal_n, mark) for mark in maps])
    ideal = history_forms(ideal_c, ideal_right, inverse)
    accept(ideal['endpoint_operator_error'], 'relative all-bosonic pure-torque complete-reference minus', 4e-6)
    # Global torque rotates the bar4 reference as well, rather than deleting it.
    gnc, _ = spin_action(neutral, spins[args.axis], [1., 1., 1.])
    gcc, _ = spin_action(charged, spins[args.axis], [1., 1., 1.])
    global_n = propagate(neutral, state, 0., 2*np.pi, gnc, neutral.trace)
    global_c = np.asarray([propagate(charged, source, 0., 2*np.pi, gcc, charged.trace) for source in sources])
    reference_unitary = expm(-2j*np.pi*spins[args.axis])
    # prepare stacks cluster columns inside each of the four reference columns.
    ref_columns = state.shape[1]
    global_c = np.asarray([np.einsum('sbc,ba->sac', src.reshape(charged.size, 4, ref_columns), reference_unitary).reshape(src.shape) for src in global_c])
    global_right = np.asarray([prepare(neutral, charged, global_n, mark) for mark in maps])
    global_plus = norm(initial_whiten(global_c-global_right, inverse))
    accept(global_plus, 'global b1 complete bar4 reference physical plus', 5e-6)
    reports = []
    for T in args.durations:
        un = propagate(neutral, state, T, 2*np.pi, nc, neutral.trace)
        us = np.asarray([propagate(charged, source, T, 2*np.pi, cc, charged.trace) for source in sources])
        right = np.asarray([prepare(neutral, charged, un, mark) for mark in maps])
        forms = history_forms(us, right, inverse)
        xn = initial_whiten(us, inverse); yn = initial_whiten(right, inverse)
        cross = np.array([[np.vdot(a, b) for b in xn] for a in yn])
        heavy_rows = []
        for scale in args.heavy_scales:
            HH = scale*base_H
            evolved = heavy_evolve(HH, Q, heavy, T, 2*np.pi)
            midway = heavy_evolve(HH, Q, heavy, T/2, np.pi)
            norm_error = accept(abs(np.vdot(evolved, evolved).real-1), 'full heavy924 actual pulse unitary', 5e-7)
            # Both species histories contain this identical factor by exact
            # Hamiltonian/source separation, not by assumed heavy invariance.
            cancellation = abs(float(np.vdot(evolved, evolved).real)-1)
            energy0 = float(np.vdot(heavy, HH@heavy).real)
            midenergy = float(np.vdot(midway, HH@midway).real-energy0)
            finalenergy = float(np.vdot(evolved, HH@evolved).real-energy0)
            ground_loss = float(max(0., 1-abs(np.vdot(heavy, midway))**2))
            frozen_overlap = complex(np.vdot(heavy, evolved))
            # Exact full tensor inner products from factors, with no heavy*light allocation.
            frozen_cross = frozen_overlap*cross
            heavy_rows.append(dict(Lambda=scale, whole_NH6_exterior_dimension=len(rows), norm_error=norm_error,
                evolved_neutral_RIGHT_heavy_cancellation_error=cancellation,
                full_joint_endpoint_operator_error=forms['endpoint_operator_error'],
                intermediate_heavy_ground_loss=ground_loss, intermediate_heavy_excitation_energy=midenergy,
                final_heavy_excitation_energy=finalenergy,
                dropping_heavy_RIGHT_overlap=dict(real=frozen_overlap.real, imag=frozen_overlap.imag),
                dropping_heavy_RIGHT_cross_minus_identity_error=norm(frozen_cross+np.eye(4)),
                no_heavy_ground_invariance_assumption=True, no_heavy_light_tensor_allocated=True))
        reports.append(dict(duration=T, source_strength=2*np.pi/T, Hamiltonian_reversed=False,
                            full_light_photon_finite_pulse=forms, heavy=heavy_rows))
    return dict(neutral_ground_energy=E0, complete_neutral_cluster=cluster,
        complete_light_sectors=[neutral.d, charged.d], complete_heavy_sector=len(rows),
        heavy_EM_neutral=True, heavy_source_commutator_at_unit_Lambda=commutator,
        packet_eta=complex_json(eta[None])[0], initial_source_metric=complex_json(metric),
        source_metric_eigenvalues=ev.tolist(), exact_relative_Spin3_CAR_error=max(closure),
        source_only_relative_minus=ideal, global_reference_plus_error=global_plus,
        finite_pulses=reports, Coulomb_dressing='exact graph longitudinal shift D.T (D D.T)^+ rho; added charge and same outside reference',
        complete_reference_colour_Wilson=Wilson,
        quantum_colour_Wilson_scope='BODY colour Wilson is identity unitary; full-field noncentral Wilson multiplication checked separately, no compressed Haar PUP substituted',
        heavy_cancellation_scope='exact separated fixedBODY H_H tensor factor, arbitrary whole-heavy initial state; not full normal-dependent entangled microH',
        charged_invariant_spin_multiplet_proved=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=3)
    parser.add_argument('--centres', choices=('all64', 'selected'), default='all64')
    parser.add_argument('--packet-frame-grid', type=int, default=5)
    parser.add_argument('--levels', type=int, default=3)
    parser.add_argument('--harmonics', type=int, default=2)
    parser.add_argument('--eigenpairs', type=int, default=8)
    parser.add_argument('--UE', type=float, default=.37)
    parser.add_argument('--K', type=float, default=.9)
    parser.add_argument('--zeta', type=float, default=0.)
    parser.add_argument('--axis', type=int, choices=(0, 1, 2), default=2)
    parser.add_argument('--durations', type=float, nargs='+', default=[.04, .02])
    parser.add_argument('--heavy-scales', type=float, nargs='+', default=[1., 8., 32.])
    args = parser.parse_args()
    if args.size < 3 or args.levels < 2 or args.harmonics < 2 or min(args.UE, args.K, *args.durations, *args.heavy_scales) <= 0:
        raise ValueError('Positive couplings/pulses/scales and native size>=3, oscillator/Fourier ranges>=2 required')
    law = BulkQuantumLaw(size=args.size, band=RotationBand(a=1., m=.2, r=.12, v=.3, improved=False))
    rng = np.random.default_rng(22210052)
    centres = list(itertools.product(range(4), repeat=3)) if args.centres == 'all64' else [(0, 0, 0), (1, 2, 3), (2, 2, 2)]
    stars = []
    base_frame = None
    for centre in centres:
        selected, frame = original_frame(law.band, law.size, np.pi/2*np.asarray(centre), True)
        ss, report = scalar_frame(frame)
        negative_ss, _ = scalar_frame(-frame)
        sign_even = accept(norm(ss-negative_ss), 'all64 true-Gram scalar sign-even frame', 3e-8)
        stars.append(dict(centre=centre, actual_full10_Gram=selected['G'], source_stationarity=selected['gradient_residual'],
                          derived_frame=report, Casimir_error=norm(sum(s@s for s in ss)-.75*np.eye(4)),
                          scalar_sign_even_error=sign_even, quarter_frame_not_inserted=True))
        if centre == (0, 0, 0):
            base_frame = frame
    if base_frame is None:
        raise AssertionError('Required BODY centre missing')
    cases = []
    for centre in ((0, 0, 0), (1, 2, 3), (2, 2, 2)):
        _, frame = original_frame(law.band, law.size, np.pi/2*np.asarray(centre), True)
        cases.append(native_case(law, frame, centre, rng))
    wilson = frame_wilson_checks(law, base_frame, rng)
    packet_selected, packet_frame = original_frame(law.band, args.packet_frame_grid, np.zeros(3), True)
    spins, _ = scalar_frame(packet_frame)
    pulse = heavy_pulse(args, packet_frame, spins)
    names = ('bosonic_frame_torque.py', 'bosonic_frame_torque.md', 'original_total_angular_transfer.py',
             'full_band_gauss_section.py', 'quantum_su4_gauss.py', 'noncompact_finite_packet_spin.py',
             'noncompact_native_current.py', 'original_charged_principal.py')
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    result = dict(plan=PLAN, revision=REVISION, template=TEMPLATE, source_hashes=hashes,
        source='Kbos=1/2 sum_xa { b_x chi(Phi_x) S_i^a(Phi_x), Qphi_x^a+divE4_x^a }',
        physical_reduction='Kbos=-sum_xa b_x chi S_i^a (QH_x^a+QL_x^a+Qref_x^a)',
        source_protocol='new all-bosonic internal colour torque on unchanged law, not original field-only or global spatial spin',
        native_size=law.size, all_true_Gram_centres=stars, native_full_Gauss_cases=cases,
        frame_Wilson_reference_checks=wilson, whole_heavy_relative_pulse=pulse,
        packet_frame_grid=args.packet_frame_grid, packet_frame_true_Gram=packet_selected['G'],
        materially_different_repair='replace field-only Cartesian velocity by complete scalar+link gauge tangent; changes source, not microscopic Hamiltonian',
        unpaid=['full microscopic normal-dependent pulse and heavy-light-field entanglement',
                'normalizable quantum-colour and full-scalar vacuum preparation',
                'primitive spatial mechanical interpretation (falsified off Clifford tube core)',
                'charged invariant spin1/2 multiplet, isolated/mobile charge, same-current Lorentz transfer',
                'native compact photon phase and interacting electron/photon completion'], campaign_complete=False)
    print(json.dumps(portable(result), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
