#!/usr/bin/env python3
"""Actual modified-heavy Riesz line and complete-reference internal-frame echo.

Finite three-site normal-section arithmetic, NOT a quantum all60V normal
simulation. The companion proof supplies the full-field conditional BO theorem.
Scientific execution belongs to the integration owner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, cast

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

from bosonic_frame_torque import AXES, field_spins, norm, scalar_frame
from noncompact_finite_packet_spin import (
    ENDS, GraphQuantum, complex_json, creator_maps, gram, initial_whiten,
    lift, prepare, propagate,
)
from noncompact_native_current import accept, adverse
from original_charged_principal import original_frame, portable
from rotation_band import RotationBand
from sea_selection import matrix_unit_basis

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-continuing-backward-20260930'
REVISION = '0004af9d7c06f05f2fb7b9969804d0634014cc82d83934ddf7b563356e800a8e'
TEMPLATE = '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'
WINDOW = np.array([1., 1., 0.])


def endpoint_blocks(fields, links):
    """Unimproved native endpoint average on all four diagnostic graph edges."""
    onsite = sparse.block_diag([(.2+3*.12)*f[0] for f in fields]).toarray()
    hops = []
    for edge, (x, y) in enumerate(ENDS):
        axis = (0, 1, 2, 2)[edge]
        fx = -(.12*fields[x, 0]+1j*.3*fields[x, axis+1])/2
        fy = -(.12*fields[y, 0]+1j*.3*fields[y, axis+1])/2
        hop = np.zeros((12, 12), complex)
        hop[4*x:4*x+4, 4*y:4*y+4] = (fx@links[edge]+links[edge]@fy)/2
        hops.append(hop)
    heavy = onsite.copy()
    for hop in hops:
        heavy += hop+hop.conj().T
    return onsite, hops, heavy


def quantum_section(args, fields, links, number):
    """Retain GraphQuantum's exact phases, Coulomb and full number sector."""
    model = GraphQuantum(args.levels, args.harmonics, args.zeta, number,
                         args.UE, args.K, fields[0])
    onsite, hops, _ = endpoint_blocks(fields, links)
    model.onsite = lift(onsite, model.rows)
    model.hops = [lift(hop, model.rows) for hop in hops]
    model.trace = float(model.d*np.sum(model.phenergy)
                        + model.phdim*(np.sum(model.coulomb)
                                       + model.onsite.diagonal().sum().real))
    return model


def marked_maps(neutral, charged, eta, spins, links):
    """Local labels followed by the genuine oriented Wilson word to R=2."""
    points = [creator_maps(neutral, charged, np.eye(2)[x], [np.eye(4)])[0]
              for x in range(2)]
    words = [links[3].conj().T, links[2]]
    local = [[np.eye(4), *[2j*s for s in spins[x]]] for x in range(2)]
    maps = []
    for a in range(4):
        marks = [local[x][a]@words[x] for x in range(2)]
        maps.append([sum((eta[x]*marks[x][i, b]*points[x][i]
                          for x in range(2) for i in range(4)),
                         sparse.csr_matrix((charged.d, neutral.d), dtype=complex))
                     for b in range(4)])
    return maps, local, words


def matrix_action(model, onebody):
    lifted = lift(onebody, model.rows)
    return lambda value: model.finish(model.fermion(model.wave(value), lifted), value), lifted


def slater(occupied, rows):
    """Exact vector in the ENTIRE exterior sector, not per-site occupations."""
    return np.asarray([np.linalg.det(occupied[np.asarray(row)]) for row in rows])


def riesz_line(h, q, scale, gamma, rows):
    # H_H + gamma K_H = -Lambda dGamma(h + gamma q/Lambda), K_H=-Q.
    fast = h+gamma*q/scale
    values, vectors = np.linalg.eigh(fast)
    occupied = vectors[:, 6:]
    projector = occupied@occupied.conj().T
    omega = slater(occupied, rows)
    energy = -scale*float(values[6:].sum())
    gap = scale*float(values[6]-values[5])
    if gap <= 1e-10:
        raise AssertionError('Actual fixed-NH6 occupied Riesz cluster is not separated')
    return fast, projector, occupied, omega, energy, gap


def heavy_slater_step(h, q, scale, duration, angle, occupied, offset):
    # Exact whole-CAR evolution of a Slater vector, including noncommuting h,q.
    evolved = expm(1j*(duration*scale*h+angle*q))@occupied
    return evolved, complex(np.exp(1j*duration*offset))


def initial_metric(sources):
    metric = gram(sources)
    values, vectors = np.linalg.eigh(metric)
    if values[0] <= 1e-9:
        raise AssertionError('Actual complete-reference initial G is singular')
    inverse = (vectors/np.sqrt(values))@vectors.conj().T
    accept(norm(gram(initial_whiten(sources, inverse))-np.eye(4)),
           'exact initial measured G normalization', 8e-7)
    return metric, values, inverse


def factor_forms(first, right, inverse, heavy_first, heavy_right) -> dict[str, Any]:
    """Full tensor history forms, evaluated without allocating its tensor product."""
    x, y = initial_whiten(first, inverse), initial_whiten(right, inverse)
    hx, hy = float(np.vdot(heavy_first, heavy_first).real), float(np.vdot(heavy_right, heavy_right).real)
    cross = np.array([[np.vdot(a, b) for b in x] for a in y])*np.vdot(heavy_right, heavy_first)
    gx, gy = hx*gram(x), hy*gram(y)
    error_matrix = gx+gy+cross+cross.conj().T
    error_matrix = (error_matrix+error_matrix.conj().T)/2
    error = math.sqrt(max(0., float(np.linalg.eigvalsh(error_matrix)[-1])))
    return dict(endpoint_operator_error=error, charged_Gram=complex_json(gx),
                RIGHT_Gram=complex_json(gy), RIGHT_eigenvalues=np.linalg.eigvalsh(gy).tolist(),
                raw_cross=complex_json(cross), cross_minus_identity_error=norm(cross+np.eye(4)))


def tensor_cancellation_error(first, right, inverse, heavy_first, heavy_right):
    """Same actual full-history Gram/cross accept boundary for all RIGHT choices."""
    x, y = initial_whiten(first, inverse), initial_whiten(right, inverse)
    cross = np.array([[np.vdot(a, b) for b in x] for a in y])
    hx = float(np.vdot(heavy_first, heavy_first).real)
    hy = float(np.vdot(heavy_right, heavy_right).real)
    overlap = complex(np.vdot(heavy_right, heavy_first))
    return max(norm((hx-1)*gram(x)), norm((hy-1)*gram(y)), norm((overlap-1)*cross))


def endpoint_accept(forms, tolerance, label):
    return accept(forms['endpoint_operator_error'], label, tolerance)


def section_geometry(fields, links, spins, basis, axis):
    """Off-Clifford gauge-tangent Ward and coefficient divergence positive controls."""
    q = sparse.block_diag([WINDOW[x]*spins[x, axis] for x in range(3)]).toarray()
    eps = 2e-5
    gx = [expm(1j*eps*WINDOW[x]*spins[x, axis]) for x in range(3)]
    gm = [g.conj().T for g in gx]
    fp = np.asarray([gx[x][None]@fields[x]@gm[x] for x in range(3)])
    fm = np.asarray([gm[x][None]@fields[x]@gx[x] for x in range(3)])
    up = np.asarray([gx[x]@links[e]@gm[y] for e, (x, y) in enumerate(ENDS)])
    um = np.asarray([gm[x]@links[e]@gx[y] for e, (x, y) in enumerate(ENDS)])
    h = endpoint_blocks(fields, links)[2]
    derivative = (endpoint_blocks(fp, up)[2]-endpoint_blocks(fm, um)[2])/(2*eps)
    ward = accept(norm(derivative-1j*(q@h-h@q)), 'full endpoint gauge-tangent Ward', 2e-7)
    divergence = 0.
    for t in basis/np.sqrt(8):
        u = expm(1j*eps*t)
        sp, _ = scalar_frame(u[None]@fields[0]@u.conj().T)
        sm, _ = scalar_frame(u.conj().T[None]@fields[0]@u)
        divergence += float(2*np.trace(t@(sp[axis]-sm[axis])/(2*eps)).real)
    accept(abs(divergence), 'all15 coefficient divergence', 4e-7)
    j, k = AXES[axis]
    cartesian = np.zeros_like(fields[0]); cartesian[j], cartesian[k] = -fields[0, k], fields[0, j]
    mechanical_defect = norm(1j*(spins[0, axis][None]@fields[0]-fields[0]@spins[0, axis][None])-cartesian)
    return q, dict(endpoint_Ward_error=ward, coefficient_divergence=divergence,
                   Cartesian_mechanical_defect=mechanical_defect,
                   quotient_normal_velocity='identically zero: actual source is all15 gauge tangent')


def run_section(args, frame, noise, rng):
    basis = np.asarray(matrix_unit_basis())
    fields = np.tile(frame, (3, 1, 1, 1))
    fields += noise*np.einsum('xma,aij->xmij', rng.normal(size=(3, 4, 15)), basis)
    links = np.tile(np.eye(4, dtype=complex), (4, 1, 1))
    # A nonuniform true colour gauge moves fields AND all crossing links.
    gauge = np.asarray([expm(.19j*np.einsum('a,aij->ij', rng.normal(size=15), basis)) for _ in range(3)])
    fields = gauge[:, None]@fields@gauge[:, None].conj().swapaxes(-1, -2)
    links = np.asarray([gauge[x]@links[e]@gauge[y].conj().T for e, (x, y) in enumerate(ENDS)])
    spins, reports = field_spins(fields)
    if min(r['cutoff'] for r in reports) != 1:
        raise AssertionError('Normal-section input left the actual torque plateau')
    q, geometry = section_geometry(fields, links, spins, basis, args.axis)
    if noise > 0:
        geometry['mechanical_adverse'] = adverse(geometry['Cartesian_mechanical_defect'],
            'internal frame drive equals Cartesian mechanical rotation', 2e-8)
    neutral = quantum_section(args, fields, links, 6)
    charged = quantum_section(args, fields, links, 7)
    E0, state, cluster = neutral.solve(args.eigenpairs)
    eta = np.array([math.sqrt(.6), math.sqrt(.4)*np.exp(.37j)])
    maps, local, words = marked_maps(neutral, charged, eta, spins, links)
    sources = np.asarray([prepare(neutral, charged, state, mark) for mark in maps])
    metric, eigenvalues, inverse = initial_metric(sources)
    nc, nq = matrix_action(neutral, q)
    cc, cq = matrix_action(charged, q)
    label_columns = np.column_stack([v.ravel() for v in local[0]])
    coeff = np.linalg.lstsq(label_columns, np.column_stack([(spins[0, args.axis]@v).ravel() for v in local[0]]), rcond=None)[0]
    # CAR commutator follows left label multiplication, not V_b S_i.
    closure = []
    for a, mark in enumerate(maps):
        for ref in range(4):
            target = sum((coeff[b, a]*maps[b][ref] for b in range(4)), sparse.csr_matrix(mark[ref].shape, dtype=complex))
            closure.append(norm(cq@mark[ref]-mark[ref]@nq-target))
    accept(max(closure), 'actual nonuniform-frame complete-reference LEFT CAR intertwiner', 3e-7)
    alpha = np.array([1+.4j, -.3+.7j, .2-.9j, -.6-.1j])
    combined = np.einsum('a,aij->ij', alpha, sources)
    accept(abs(np.vdot(combined, combined)-alpha.conj()@metric@alpha),
           'arbitrary complex labels use exact initial G', 2e-7)
    ideal_n = state.copy(); ideal_c = sources.copy()
    for angle in args.angles:
        ideal_n = propagate(neutral, ideal_n, 0., angle, nc, neutral.trace)
        ideal_c = np.asarray([propagate(charged, s, 0., angle, cc, charged.trace) for s in ideal_c])
    ideal_right = np.asarray([prepare(neutral, charged, ideal_n, mark) for mark in maps])
    _, _, h = endpoint_blocks(fields, links)
    rows = neutral.rows
    hvalues, hvectors = np.linalg.eigh(h)
    occupied = hvectors[:, 6:]
    heavy0 = slater(occupied, rows)
    accept(abs(np.vdot(heavy0, heavy0).real-1), 'actual whole924 initial heavy Slater norm', 2e-8)
    base = -lift(h, rows)
    Q = lift(q, rows)
    noncommuting = norm(base@Q-Q@base)
    adverse(noncommuting, 'actual heavy Hamiltonian commutes with localized Spin source', 2e-8)
    source_only = factor_forms(ideal_c, ideal_right, inverse, heavy0, heavy0)
    endpoint_accept(source_only, 5e-6, 'same full-history relative minus boundary')
    # Wrong source window cuts the actual two-site packet, rather than a dummy row.
    halo_q = sparse.block_diag([spins[0, args.axis], np.zeros((4, 4)), np.zeros((4, 4))]).toarray()
    hn, _ = matrix_action(neutral, halo_q); hc, _ = matrix_action(charged, halo_q)
    halo_n, halo_c = state.copy(), sources.copy()
    for angle in args.angles:
        halo_n = propagate(neutral, halo_n, 0., angle, hn, neutral.trace)
        halo_c = np.asarray([propagate(charged, s, 0., angle, hc, charged.trace) for s in halo_c])
    halo_right = np.asarray([prepare(neutral, charged, halo_n, mark) for mark in maps])
    halo_forms = factor_forms(halo_c, halo_right, inverse, heavy0, heavy0)
    halo_rejection = adverse(halo_forms['endpoint_operator_error'], 'same full-history relative minus boundary', 5e-6)
    pulse_reports = []
    for scale in args.scales:
        duration = scale**(-args.power)
        un, us = state.copy(), sources.copy()
        ho = occupied.copy(); hp = 1.+0j
        line_reports = []
        for angle in args.angles:
            gamma = angle/duration
            fast, projector, modified_occupied, omega, energy, gap = riesz_line(h, q, scale, gamma, rows)
            # The actual shifted scalar offset, not an asymptotic energy expansion.
            H = scale*base-gamma*Q
            exact_residual = norm(H@omega-energy*omega)
            accept(exact_residual/max(1., abs(gamma)), 'actual modified full-CAR Riesz eigenline', 3e-8)
            frozen_residual = norm(H@heavy0-np.vdot(heavy0, H@heavy0)*heavy0)
            normalized_frozen = frozen_residual/max(1., abs(gamma))
            rejection = (adverse(normalized_frozen, 'actual modified full-CAR Riesz eigenline', 3e-8)
                         if gamma != 0 else 'zero-strength leg: unmodified line is the actual line')
            point_projector = occupied@occupied.conj().T
            alignment = complex(np.linalg.det(occupied.conj().T@modified_occupied))
            line_reports.append(dict(angle=angle, gamma=gamma, exact_offset=energy,
                offset_from_unpulsed=energy+scale*float(hvalues[6:].sum()),
                actual_manybody_gap=gap, projector_change=norm(projector-point_projector),
                Slater_line_distance=math.sqrt(max(0., 2-2*abs(alignment))),
                full_CAR_residual=exact_residual, frozen_line_residual=frozen_residual,
                frozen_line_rejected_at_actual_boundary=rejection,
                heavy_source_mean=float(np.trace(point_projector@q).real)))
            un = propagate(neutral, un, duration, angle, nc, neutral.trace)
            us = np.asarray([propagate(charged, s, duration, angle, cc, charged.trace) for s in us])
            ho, phase = heavy_slater_step(h, q, scale, duration, angle, ho, energy)
            hp *= phase
        heavy_actual = hp*slater(ho, rows)
        accept(abs(np.vdot(heavy_actual, heavy_actual).real-1), 'actual full-heavy echo unitary', 5e-7)
        right = np.asarray([prepare(neutral, charged, un, mark) for mark in maps])
        forms = factor_forms(us, right, inverse, heavy_actual, heavy_actual)
        light_forms = factor_forms(us, right, inverse, heavy0, heavy0)
        accept(abs(forms['endpoint_operator_error']-light_forms['endpoint_operator_error']),
               'full evolved RIGHT heavy cancels without ground invariance', 5e-7)
        pulse_reports.append(dict(Lambda=scale, leg_duration=duration, power=args.power,
            modified_fast_blocks=line_reports, full_reference_history=forms,
            heavy_final_initial_overlap=complex_json(np.array([[np.vdot(heavy0, heavy_actual)]])),
            initial_heavy_not_assumed_invariant=True, photon_Coulomb_H_never_reversed=True))
    # Adverse RIGHT omission uses a legal arbitrary full-heavy state, not a
    # selected low-energy line whose offset-free error must itself vanish.
    # This proves the arbitrary-heavy cancellation boundary at finite strength.
    excited = hvectors[:, [0, 7, 8, 9, 10, 11]]
    arbitrary = (heavy0+1j*slater(excited, rows))/math.sqrt(2)
    probe_scale = args.adverse_scale
    solver = cast(Callable[..., Any], expm_multiply)
    arbitrary_final = arbitrary.copy()
    probe_n, probe_c = state.copy(), sources.copy()
    for angle in args.angles:
        generator = -1j*args.adverse_duration*probe_scale*base+1j*angle*Q
        arbitrary_final = solver(generator, arbitrary_final, traceA=complex(generator.diagonal().sum()))
        probe_n = propagate(neutral, probe_n, args.adverse_duration, angle, nc, neutral.trace)
        probe_c = np.asarray([propagate(charged, s, args.adverse_duration, angle, cc, charged.trace) for s in probe_c])
    probe_right = np.asarray([prepare(neutral, charged, probe_n, mark) for mark in maps])
    good = factor_forms(probe_c, probe_right, inverse, arbitrary_final, arbitrary_final)
    good_error = tensor_cancellation_error(probe_c, probe_right, inverse, arbitrary_final, arbitrary_final)
    accept(good_error, 'same actual full-H history Gram and cross cancellation', 5e-7)
    frozen_error = tensor_cancellation_error(probe_c, probe_right, inverse, arbitrary_final, arbitrary)
    frozen_rejection = adverse(frozen_error, 'same actual full-H history Gram and cross cancellation', 5e-7)
    # A projected-away RIGHT sets its norm/cross to zero and fails this very
    # same assertion; no final metric inversion can conceal it.
    omitted_error = tensor_cancellation_error(probe_c, probe_right, inverse, arbitrary_final, np.zeros_like(arbitrary_final))
    omitted_rejection = adverse(omitted_error, 'same actual full-H history Gram and cross cancellation', 5e-7)
    return dict(normal_section_noise=noise, torque_plateau=1., geometry=geometry,
        actual_Clifford_defects=[r.get('Clifford_defect', 0.) for r in reports],
        complete_heavy_dimension=len(rows), complete_light_dimensions=[neutral.d, charged.d],
        quantum_photon_dimensions=[args.levels, 2*args.harmonics+1],
        neutral_energy=E0, neutral_cluster=cluster, initial_G=complex_json(metric),
        initial_G_eigenvalues=eigenvalues.tolist(), complex_label_alpha=complex_json(alpha[None])[0],
        complex_packet_eta=complex_json(eta[None])[0], source_only=source_only,
        LEFT_intertwiner_error=max(closure), Wilson_words_to_same_R=[[(3, -1)], [(2, 1)]],
        Wilson_unitarity_errors=[norm(w.conj().T@w-np.eye(4)) for w in words],
        heavy_noncommutator=noncommuting, pulses=pulse_reports,
        adverse=dict(cut_packet_halo=halo_rejection, frozen_heavy_RIGHT=frozen_rejection,
                     projected_away_RIGHT=omitted_rejection,
                     allowed_recovery=good, recovery_Gram_cross_error=good_error,
                     frozen_RIGHT_Gram_cross_error=frozen_error, omitted_RIGHT_Gram_cross_error=omitted_error,
                     arbitrary_heavy_state_scope='legal whole NH6, not low-energy BO'),
        sectors=dict(NH=6, NL_neutral=6, NL_charged=7, outside_bar4_components=4,
                     static_EM_charge_at_R=-1, total_periodic_Gauss_charge=0),
        no_final_renormalization=True,
        finite_section_scope='fixed classical field/link section; coupled quantum photons/light plus full heavy CAR, no normal kinetic or quantum colour vacuum')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--levels', type=int, default=3)
    parser.add_argument('--harmonics', type=int, default=2)
    parser.add_argument('--eigenpairs', type=int, default=8)
    parser.add_argument('--zeta', type=float, default=0.)
    parser.add_argument('--UE', type=float, default=.37)
    parser.add_argument('--K', type=float, default=.9)
    parser.add_argument('--axis', type=int, choices=(0, 1, 2), default=2)
    parser.add_argument('--frame-grid', type=int, default=5)
    parser.add_argument('--noises', type=float, nargs='+', default=[0., .0003])
    parser.add_argument('--scales', type=float, nargs='+', default=[64., 512.])
    parser.add_argument('--power', type=float, default=1/8)
    parser.add_argument('--angles', type=float, nargs=2, default=[math.pi, math.pi])
    parser.add_argument('--adverse-scale', type=float, default=8.)
    parser.add_argument('--adverse-duration', type=float, default=.07)
    args = parser.parse_args()
    if args.levels < 2 or args.harmonics < 2 or args.frame_grid < 5 or args.eigenpairs < 2:
        raise ValueError('Oscillator/Fourier cutoffs>=2, native frame grid>=5 and eigenpairs>=2 required')
    if not 0 < args.power < 1/4 or min(args.UE, args.K, *args.scales, args.adverse_scale, args.adverse_duration) <= 0:
        raise ValueError('Positive couplings/scales/duration and legal 0<p<1/4 required')
    if min(args.noises) < 0 or abs(sum(args.angles)-2*math.pi) > 1e-12:
        raise ValueError('Nonnegative normal noises and two pulse angles summing to2pi required')
    selected, frame = original_frame(RotationBand(a=1., m=.2, r=.12, v=.3, improved=False),
                                     args.frame_grid, np.zeros(3), True)
    rng = np.random.default_rng(22210055)
    cases = [run_section(args, frame, noise, rng) for noise in args.noises]
    files = ('noncompact_frame_pulse_transfer.py', 'noncompact_frame_pulse_transfer.md',
             'bosonic_frame_torque.py', 'bosonic_frame_torque.md',
             'noncompact_finite_packet_spin.py', 'noncompact_native_current.py',
             'original_slow_kinetic_bo.md', 'noncompact_marked_metric.md')
    result: dict[str, Any] = dict(plan=PLAN, revision=REVISION, template=TEMPLATE,
        source_hashes={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
        native_frame_grid=args.frame_grid, native_true_Gram=selected['G'], cases=cases,
        source='all-bosonic K=-g(Phi)(QH+QL+Qref) on exact complete Gauss quotient',
        fast_block_sign='H_H=-Lambda dGamma(h), so h_gamma=h+gamma g/Lambda',
        theorem_scope='fixed a,N in existence-only original BO globality range; noncompact photons remain infinite in proof',
        physical_interpretation='internal colour frame, NOT local Cartesian/mechanical Spin3',
        numerical_normal_vacuum_simulated=False, quantum_CAR_is_assumed=True,
        thermodynamic_or_photon_phase_proved=False, campaign_complete=False)
    print(json.dumps(portable(result), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
