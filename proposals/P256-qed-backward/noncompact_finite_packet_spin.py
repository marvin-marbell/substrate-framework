#!/usr/bin/env python3
"""Unfiltered finite-width CAR source, quantum photons and mechanical echo.

Parent-runnable diagnostic of the declared noncompact law. No native cubic
admission, full-field normal ground, or total charged-spin theorem is asserted.
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
from scipy import sparse
from scipy.sparse.linalg import eigsh, expm_multiply

from noncompact_native_current import HermitianAction, accept, adverse, displacement
from original_charged_principal import original_frame
from original_physical_rotation_lift import recover_gamma
from rotation_band import RotationBand
from sea_selection import matrix_unit_basis

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-continuing-backward-20260930'
REVISION = 'a406dca7a4cc642318250c028a763daa96fa66a5999da5dae0c96e53910c5c91'
TEMPLATE = '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'
D = np.array([[1., 1., 0., -1.], [-1., -1., 1., 0.], [0., 0., -1., 1.]])
C = np.array([1., -1., 0., 0.])
F = C / math.sqrt(2)
HARMONIC = np.array([1., 1., 2., 2.]) / math.sqrt(10)
ENDS = ((0, 1), (0, 1), (1, 2), (2, 0))
NU = (1, 1, 2, 2)
REFERENCE = 2
SITE_WEIGHTS = np.array([0., -1., -3.])


def lift(onebody, rows):
    """Complete canonical exterior power; no local occupation projection."""
    lookup = {row: i for i, row in enumerate(rows)}
    rr, cc, vv = [], [], []
    entries = [np.flatnonzero(abs(onebody[:, old]) > 0) for old in range(12)]
    for column, row in enumerate(rows):
        for at, old in enumerate(row):
            rest = row[:at] + row[at + 1:]
            for new in entries[old]:
                if new in rest:
                    continue
                target = tuple(sorted((*rest, int(new))))
                rr.append(lookup[target]); cc.append(column)
                vv.append((-1) ** (at + sum(k < new for k in rest)) * onebody[new, old])
    return sparse.coo_matrix((vv, (rr, cc)), shape=(len(rows), len(rows))).tocsr()


class BatchAction(HermitianAction):
    def _matmat(self, x):
        return self.apply(x)


class GraphQuantum:
    """Whole NL=6/7 twelve-mode sectors and exact compressed infinite phases."""
    def __init__(self, n, M, zeta, number, UE, K, frame):
        self.n, self.M, self.zeta = n, M, zeta
        self.number, self.UE, self.K = number, UE, K
        self.rows = tuple(itertools.combinations(range(12), number))
        self.d = len(self.rows)
        self.ell = np.arange(-M, M + 1)
        self.l = len(self.ell)
        self.phdim, self.size = n * self.l, n * self.l * self.d
        self.omega = 2 * math.sqrt(UE * K)
        self.zscale = math.sqrt(UE / self.omega)
        self.phenergy = (self.omega * (np.arange(n)[:, None] + .5)
                         + UE * (self.ell[None, :] + zeta) ** 2 / 10).ravel()
        self.density = np.array([[sum(4*x <= k < 4*x+4 for k in row)
                                  for x in range(3)] for row in self.rows], float)
        self.sector = ((np.tile(self.ell, n)[:, None]-self.density@SITE_WEIGHTS) % 5).astype(int)
        rho = self.density - 2
        if number == 7:
            rho[:, REFERENCE] -= 1
        self.rho = rho
        self.green = np.linalg.pinv(D @ D.T, hermitian=True)
        self.coulomb = UE * np.einsum('xi,ij,xj->x', rho, self.green, rho)
        self.onsite = lift(np.kron(np.eye(3), (.2 + 3*.12) * frame[0]), self.rows)
        self.hops, self.phases = [], []
        for e, (x, y) in enumerate(ENDS):
            hop = np.zeros((12, 12), complex)
            axis = (0, 1, 2, 2)[e]
            hop[4*x:4*x+4, 4*y:4*y+4] = -(.12*frame[0] + 1j*.3*frame[axis+1])/2
            self.hops.append(lift(hop, self.rows))
            shift = sparse.diags(np.ones(self.l - NU[e]), -NU[e], shape=(self.l, self.l), dtype=complex)
            self.phases.append(sparse.kron(sparse.csr_matrix(displacement(n, F[e]*self.zscale)), shift, format='csr'))
        ladder = sparse.diags(np.sqrt(np.arange(1, n)), 1, shape=(n, n))
        self.z = sparse.kron(self.zscale*(ladder + ladder.T), sparse.eye(self.l), format='csr')
        self.py = np.tile((self.ell + zeta)/math.sqrt(10), n)
        self.operator = BatchAction(self.size, self.action)
        self.trace = float(self.d*np.sum(self.phenergy)
                           + self.phdim*(np.sum(self.coulomb) + self.onsite.diagonal().sum().real))

    def wave(self, value):
        return np.asarray(value).reshape(self.phdim, self.d, -1)

    def finish(self, wave, original):
        return wave.reshape(np.asarray(original).shape)

    def fermion(self, wave, operator):
        k = wave.shape[2]
        return (operator @ wave.transpose(1, 0, 2).reshape(self.d, -1)).reshape(self.d, self.phdim, k).transpose(1, 0, 2)

    def photon(self, wave, operator):
        return (operator @ wave.reshape(self.phdim, -1)).reshape(wave.shape)

    def action(self, value):
        wave = self.wave(value)
        out = np.asarray((self.phenergy[:, None, None] + self.coulomb[None, :, None])*wave, dtype=complex)
        out += self.fermion(wave, self.onsite)
        for phase, hop in zip(self.phases, self.hops):
            out += self.photon(self.fermion(wave, hop), phase)
            out += self.photon(self.fermion(wave, hop.conj().T), phase.conj().T)
        return self.finish(out, value)

    def current(self, value, weights):
        wave = self.wave(value); out = np.zeros_like(wave, dtype=complex)
        for w, phase, hop in zip(weights, self.phases, self.hops):
            out += 1j*w*(self.photon(self.fermion(wave, hop), phase)
                         - self.photon(self.fermion(wave, hop.conj().T), phase.conj().T))
        return self.finish(out, value)

    def contact(self, value, first, second):
        wave = self.wave(value); out = np.zeros_like(wave, dtype=complex)
        for w, u, phase, hop in zip(first, second, self.phases, self.hops):
            out -= w*u*(self.photon(self.fermion(wave, hop), phase)
                        + self.photon(self.fermion(wave, hop.conj().T), phase.conj().T))
        return self.finish(out, value)

    def sector_phase(self, value):
        return self.finish(self.wave(value)*np.exp(2j*np.pi*self.sector/5)[:, :, None], value)

    def sector_weights(self, value):
        probability = np.sum(abs(self.wave(value))**2, axis=2)
        total = float(probability.sum())
        return [float(probability[self.sector == t].sum()/total) for t in range(5)]

    def solve(self, count):
        solver = cast(Callable[..., Any], eigsh)
        seed = np.random.default_rng(22210051 + self.number).normal(size=self.size)
        values, vectors = solver(self.operator, k=count, which='SA', tol=3e-11, v0=seed)
        order = np.argsort(values); values, vectors = values[order], vectors[:, order]
        chosen = np.flatnonzero(values-values[0] < 2e-7)
        if len(chosen) == count:
            raise AssertionError('Complete neutral cluster reaches requested spectrum edge; increase --eigenpairs')
        vectors = np.linalg.qr(vectors[:, chosen])[0]
        errors = [np.linalg.norm(self.action(vectors[:, j])-values[0]*vectors[:, j]) for j in range(len(chosen))]
        accept(max(errors), 'actual complete neutral quantum cluster residual', 5e-7)
        return float(values[0]), vectors/math.sqrt(len(chosen)), dict(multiplicity=len(chosen), next_gap=float(values[len(chosen)]-values[0]), residuals=[float(x) for x in errors])

    def enlarged_residual(self, value, energy, frame):
        target = GraphQuantum(self.n+4, self.M+2, self.zeta, self.number, self.UE, self.K, frame)
        wave = self.wave(value)
        full = np.zeros((target.n, target.l, self.d, wave.shape[2]), complex)
        full[:self.n, target.M-self.M:target.M+self.M+1] = wave.reshape(self.n, self.l, self.d, -1)
        vector = full.reshape(target.size, -1)
        visible = float(np.linalg.norm(target.action(vector)-energy*vector))
        tails = 0.
        for phase, hop in zip(target.phases, target.hops):
            for p, t in ((phase, hop), (phase.conj().T, hop.conj().T)):
                w = target.fermion(target.wave(vector), t)
                pw = target.photon(w, p)
                lost = float(np.vdot(w, w).real-np.vdot(pw, pw).real)
                tails += math.sqrt(max(0., lost)+8e-15*float(np.vdot(w, w).real))
        return dict(enlarged_residual=visible, remaining_exact_phase_tail_upper=tails,
                    full_infinite_comparison_residual_upper=visible+tails)


def mechanical_geometry(frame, selected):
    """Derive ell from actual Cartesian vector tangents and all45 graph Gauss rows."""
    gamma = recover_gamma(np.asarray(selected['G']), frame)
    basis = matrix_unit_basis(); generators = basis/math.sqrt(8)
    tangent = np.array([1j*(g[None]@frame-frame@g[None]) for g in generators])
    inertia = np.einsum('amij,bmij->ab', tangent.conj(), tangent).real/4
    graph_mass = np.kron(np.eye(3), inertia)+(1/.6)*np.kron(D@D.T, np.eye(15))
    plateau = np.array([1., 1., 0.])
    spins, reports = [], []
    for axis, (j, k) in enumerate(((2, 3), (3, 1), (1, 2))):
        change = np.zeros_like(frame); change[j], change[k] = -frame[k], frame[j]
        rhs = np.einsum('amij,mij->a', tangent.conj(), change).real/4
        spin = -1j*(gamma[j]@gamma[k]-gamma[k]@gamma[j])/4
        s = 2*np.einsum('aij,ji->a', generators, spin).real
        lam = float(s@inertia@s/(s@s))
        accept(np.linalg.norm(inertia@s-lam*s), 'actual isometric full15 field inertia channel', 3e-7)
        accept(np.linalg.norm(rhs-inertia@s), 'physical Cartesian angular Gauss supplier', 3e-7)
        torque = plateau+(D@D.T)@plateau/(.6*lam)
        coefficients = np.linalg.solve(graph_mass, (torque[:, None]*rhs).ravel()).reshape(3, 15)
        ell = np.einsum('xa,aij->xij', coefficients, generators)
        error = accept(np.linalg.norm(ell-plateau[:, None, None]*spin), 'full45 graph screened mechanical spin plateau', 3e-7)
        spins.append(spin)
        reports.append(dict(axis=axis, all_colour_Gauss_rows=45, mass_min=float(np.linalg.eigvalsh(graph_mass)[0]),
                            torque=torque.tolist(), actual_connection_error=error, SAME_R_connection=float(np.linalg.norm(ell[REFERENCE]))))
    spins = np.asarray(spins)
    casimir = accept(np.linalg.norm(sum(np.matmul(spins[i], spins[i]) for i in range(3))-.75*np.eye(4)), 'derived relative mechanical Casimir three quarters')
    labels = [np.eye(4, dtype=complex), *[2j*s for s in spins]]
    left = []
    label_columns = np.column_stack([v.ravel() for v in labels])
    for spin in spins:
        coefficients = np.linalg.lstsq(label_columns, np.column_stack([(spin@v).ravel() for v in labels]), rcond=None)[0]
        accept(np.linalg.norm(label_columns@coefficients-np.column_stack([(spin@v).ravel() for v in labels])), 'actual left Spin3 quaternion closure')
        left.append(coefficients)
    accept(np.linalg.norm(sum(s@s for s in left)-.75*np.eye(4)), 'closed source relative spin Casimir three quarters')
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        accept(np.linalg.norm(left[i]@left[j]-left[j]@left[i]-1j*left[k]), 'actual relative Spin3 generator algebra')
    return spins, labels, dict(all_axes=reports, Casimir_error=casimir,
        left_closure=True, source_representation='two complex relative spin1/2 doublets; not total interacting charged spin')


def creator_maps(neutral, charged, eta, labels):
    lookup = {row: i for i, row in enumerate(charged.rows)}
    creators = []
    for mode in range(8):
        rr, cc, vv = [], [], []
        for column, row in enumerate(neutral.rows):
            if mode not in row:
                rr.append(lookup[tuple(sorted((*row, mode)))]); cc.append(column)
                vv.append((-1)**sum(k < mode for k in row))
        creators.append(sparse.coo_matrix((vv, (rr, cc)), shape=(charged.d, neutral.d)).tocsr())
    maps = []
    for label in labels:
        maps.append([sum((eta[x]*label[a, b]*creators[4*x+a] for x in range(2) for a in range(4)),
                         sparse.csr_matrix((charged.d, neutral.d), dtype=complex)) for b in range(4)])
    return maps


def prepare(neutral, charged, state, maps):
    wave = neutral.wave(state); columns = wave.shape[2]
    return np.concatenate([(t @ wave.transpose(1, 0, 2).reshape(neutral.d, -1)).reshape(charged.d, neutral.phdim, columns).transpose(1, 0, 2).reshape(charged.size, columns)
                           for t in maps], axis=1)


def gram(sources):
    return np.array([[np.vdot(a, b) for b in sources] for a in sources])


def initial_whiten(sources, metric_root_inverse):
    return np.einsum('ab,aij->bij', metric_root_inverse, np.asarray(sources))




def propagate(operator, value, Htime=0., angle=0., source=None, trace=0.):
    if Htime == 0 and angle == 0:
        return value.copy()
    def action(x):
        if Htime == 0 and source is not None:
            return 1j*angle*source(x)
        out = -1j*Htime*operator.action(x)
        if source is not None:
            out += 1j*angle*source(x)
        return out
    # i times a Hermitian action: the adjoint must have the opposite sign.
    size = operator.size
    from scipy.sparse.linalg import LinearOperator
    factory = cast(Callable[..., Any], LinearOperator)
    generator = factory((size, size), matvec=action, matmat=action,
                        rmatvec=lambda x: -action(x), rmatmat=lambda x: -action(x), dtype=np.complex128)
    solver = cast(Callable[..., Any], expm_multiply)
    return solver(generator, value, traceA=-1j*Htime*trace)


class Controller:
    """BODY image of the original photon + scalar orbital + Cartesian source."""
    def __init__(self, model, spin, gamma, halo=1.):
        # Same velocity enters both orbital and graph curl contraction.
        self.model = model; self.gamma = gamma*(.28-(-.11))
        q = np.kron(np.diag([1., halo, 0.]), spin)
        self.spin = lift(q, model.rows)
        self.orbitals = []
        # Original scalar edge vbar/(2ia), two parallel edges, entire halo in b=1.
        for e, velocity in ((0, .28), (1, -.11)):
            x, y = ENDS[e]; o = np.zeros((12, 12), complex)
            o[4*x:4*x+4, 4*y:4*y+4] = velocity*np.eye(4)/(2j)
            self.orbitals.append((model.phases[e], lift(o, model.rows)))

    def photon(self, value):
        wave = self.model.wave(value)
        return self.model.finish(-self.gamma*self.model.photon(wave*self.model.py[:, None, None], self.model.z), value)

    def orbital(self, value):
        wave = self.model.wave(value); out = np.zeros_like(wave, dtype=complex)
        for phase, o in self.orbitals:
            out += self.model.photon(self.model.fermion(wave, o), phase)
            out += self.model.photon(self.model.fermion(wave, o.conj().T), phase.conj().T)
        return self.model.finish(out, value)

    def intrinsic(self, value):
        return self.model.finish(self.model.fermion(self.model.wave(value), self.spin), value)

    def first(self, value):
        return self.photon(value)+self.orbital(value)+self.intrinsic(value)

    def second(self, value):
        return -self.photon(value)-self.orbital(value)


def echo(model, controller, initial, duration, variant='good', halo_controller=None):
    first = controller.first if halo_controller is None else halo_controller.first
    halfway = propagate(model, initial, duration, 2*np.pi, first, model.trace)
    if variant == 'missing_inverse_orbital':
        second = lambda x: -controller.photon(x)
    elif variant == 'missing_inverse_photon':
        second = lambda x: -controller.orbital(x)
    else:
        second = controller.second
    return propagate(model, halfway, duration, 2*np.pi, second, model.trace), halfway


def history_forms(first, right, inverse):
    x, y = initial_whiten(first, inverse), initial_whiten(right, inverse)
    first_norm_error = accept(np.linalg.norm(gram(x)-np.eye(4)), 'same initially normalized full-reference history norm', 5e-7)
    cross = np.array([[np.vdot(a, b) for b in x] for a in y])
    defect = x+y
    return dict(endpoint_operator_error=float(math.sqrt(max(0., np.linalg.eigvalsh(gram(defect))[-1]))),
                initially_normalized_history_norm_error=first_norm_error,
                after_RIGHT_metric_eigenvalues=np.linalg.eigvalsh(gram(y)).tolist(),
                after_RIGHT_metric=complex_json(gram(y)),
                raw_normalized_cross=complex_json(cross),
                cross_minus_identity_error=float(np.linalg.norm(cross+np.eye(4))))


def complex_json(value):
    return [[dict(real=float(z.real), imag=float(z.imag)) for z in row] for row in value]


def fibre(args, zeta, frame, labels, spins, state, cluster) -> dict[str, Any]:
    neutral = GraphQuantum(args.levels, args.harmonics, zeta, 6, args.UE, args.K, frame)
    charged = GraphQuantum(args.levels, args.harmonics, zeta, 7, args.UE, args.K, frame)
    E0 = float(np.vdot(state, neutral.action(state)).real)
    eta = np.array([math.sqrt(.6), math.sqrt(.4)*np.exp(.37j)])
    maps = creator_maps(neutral, charged, eta, labels)
    sources = np.asarray([prepare(neutral, charged, state, mark) for mark in maps])
    metric = gram(sources); metric_values, metric_vectors = np.linalg.eigh(metric)
    if metric_values[0] <= 1e-8:
        raise AssertionError('Actual unfiltered ground supplier has singular complete packet metric; no fake point norm or final normalization allowed')
    inverse = (metric_vectors/np.sqrt(metric_values))@metric_vectors.conj().T
    normalized = initial_whiten(sources, inverse)
    accept(np.linalg.norm(gram(normalized)-np.eye(4)), 'complete initial coherent packet metric')
    # Exact CAR complete-reference operator, INCLUDING offsite neutral coherence.
    m = np.kron(eta[:, None], np.eye(4)).reshape(8, 4)
    m = np.vstack((m, np.zeros((4, 4))))
    car_metric = 4*sparse.eye(neutral.d)-lift(m@m.conj().T, neutral.rows)
    actual_metric = sum((t.conj().T@t for t in maps[0]), sparse.csr_matrix((neutral.d, neutral.d), dtype=complex))
    car_error = accept(sparse.linalg.norm(actual_metric-car_metric), 'complete coherent creator CAR metric')
    point_error = float(abs(metric[0, 0]-2))
    hsrc = np.asarray([charged.action(source) for source in sources])
    energy_matrix = np.array([[np.vdot(a, b)-E0*metric[i, j] for j, b in enumerate(hsrc)] for i, a in enumerate(sources)])
    accept(np.linalg.norm(energy_matrix-energy_matrix.conj().T), 'complete complex packet energy Hermitian form', 5e-7)
    neutral_h_sources = np.asarray([prepare(neutral, charged, neutral.action(state), mark) for mark in maps])
    full_commutator = hsrc-neutral_h_sources
    weighted_neutral = neutral_h_sources-E0*sources
    def coulomb(model, value):
        return model.finish(model.wave(value)*model.coulomb[None, :, None], value)
    electric_commutator = np.asarray([coulomb(charged, s) for s in sources])-np.asarray([
        prepare(neutral, charged, coulomb(neutral, state), mark) for mark in maps])
    def form(values):
        return np.array([[np.vdot(a, b) for b in values] for a in sources])
    electric_form = form(electric_commutator)
    light_form = form(full_commutator-electric_commutator)
    weighted_form = form(weighted_neutral)
    transfer_error = accept(np.linalg.norm(energy_matrix-electric_form-light_form-weighted_form),
                            'same fullH packet electric light weighted neutral transfer', 5e-7)
    alpha = np.array([1., .3j, -.2+.4j, .6]); alpha /= np.linalg.norm(alpha)
    packet = np.einsum('a,aij->ij', alpha, sources)
    initial_norm = float(np.vdot(packet, packet).real)
    energy = float((np.vdot(alpha, energy_matrix@alpha)/np.vdot(alpha, metric@alpha)).real)
    direct = float(sum(np.vdot(packet[:, c], charged.action(packet[:, c])).real for c in range(packet.shape[1]))/initial_norm-E0)
    accept(abs(energy-direct), 'same fullH arbitrary complex packet Rayleigh quotient', 5e-7)
    for model, value in ((neutral, state), (charged, packet)):
        accept(np.linalg.norm(model.action(model.sector_phase(value))-model.sector_phase(model.action(value))),
               'same fullH actual mod5 source-addressed sector conservation', 5e-7)
    point_sector_errors = []
    for x in (0, 1):
        point_mark = creator_maps(neutral, charged, np.eye(2)[x], [labels[0]])[0]
        point = prepare(neutral, charged, state, point_mark)
        neutral_right = prepare(neutral, charged, neutral.sector_phase(state), point_mark)
        point_sector_errors.append(accept(np.linalg.norm(charged.sector_phase(point)
            -np.exp(-2j*np.pi*SITE_WEIGHTS[x]/5)*neutral_right),
            'same actual point creator mod5 sector-addressing intertwiner', 5e-7))
    initial_sectors = charged.sector_weights(packet)
    controllers = [Controller(neutral, spins[args.axis], args.photon_shear), Controller(charged, spins[args.axis], args.photon_shear)]
    ncontrol, ccontrol = controllers
    commutator = ccontrol.intrinsic(ccontrol.first(packet))-ccontrol.first(ccontrol.intrinsic(packet))
    accept(np.linalg.norm(commutator), 'same global quantum orbital photon spin commutator', 5e-7)
    # Relative mechanical image acts on the entire preparation map, not only a 2pi sign.
    # State-independent left action is checked by the actual matrices for all3 axes.
    closure_errors = []
    label_columns = np.column_stack([v.ravel() for v in labels])
    for spin in spins:
        coeff = np.linalg.lstsq(label_columns, np.column_stack([(spin@v).ravel() for v in labels]), rcond=None)[0]
        sn = lift(np.kron(np.diag([1., 1., 0.]), spin), neutral.rows)
        sc = lift(np.kron(np.diag([1., 1., 0.]), spin), charged.rows)
        for b in range(4):
            for ref in range(4):
                target = sum((coeff[a, b]*maps[a][ref] for a in range(4)), sparse.csr_matrix(maps[b][ref].shape, dtype=complex))
                closure_errors.append(accept(sparse.linalg.norm(sc@maps[b][ref]-maps[b][ref]@sn-target), 'whole12mode relative Spin3 source intertwiner', 5e-7))
    ideal_neutral, neutral_mid = echo(neutral, ncontrol, state, 0.)
    ideal_pairs = [echo(charged, ccontrol, s, 0.) for s in sources]
    ideal_sources = np.asarray([pair[0] for pair in ideal_pairs])
    midway_packet = np.einsum('a,aij->ij', alpha, np.asarray([pair[1] for pair in ideal_pairs]))
    midpoint_density = np.sum(abs(charged.wave(midway_packet))**2, axis=(0, 2))/initial_norm
    ideal_right = np.asarray([prepare(neutral, charged, ideal_neutral, mark) for mark in maps])
    ideal = history_forms(ideal_sources, ideal_right, inverse)
    accept(ideal['endpoint_operator_error'], 'same fullquantum finite packet minus endpoint', 4e-6)
    # Dropping the actual neutral RIGHT is judged at the SAME endpoint.
    wrong_right = adverse(history_forms(ideal_sources, sources, inverse)['endpoint_operator_error'],
                          'same fullquantum finite packet minus endpoint', 4e-6)
    adversaries = {'missing_RIGHT': wrong_right}
    for variant in ('missing_inverse_orbital', 'missing_inverse_photon', 'offplateau_halo'):
        badn = Controller(neutral, spins[args.axis], args.photon_shear, .4) if variant == 'offplateau_halo' else None
        badc = Controller(charged, spins[args.axis], args.photon_shear, .4) if variant == 'offplateau_halo' else None
        un, _ = echo(neutral, ncontrol, state, 0., variant, badn)
        us = np.asarray([echo(charged, ccontrol, s, 0., variant, badc)[0] for s in sources])
        right = np.asarray([prepare(neutral, charged, un, mark) for mark in maps])
        adversaries[variant] = adverse(history_forms(us, right, inverse)['endpoint_operator_error'], 'same fullquantum finite packet minus endpoint', 4e-6)
    # Microscopic current AND contact independently follow from site density commutators.
    weights = D[0]
    def density(v):
        return charged.finish(charged.wave(v)*charged.density[None, :, 0, None], v)
    current = charged.current(packet, weights)
    independent = 1j*(density(charged.action(packet))-charged.action(density(packet)))
    current_error = accept(np.linalg.norm(current-independent), 'same fullH packet microscopic current continuity', 5e-7)
    contact = charged.contact(packet, weights, weights)
    independent_contact = 1j*(density(current)-charged.current(density(packet), weights))
    contact_error = accept(np.linalg.norm(contact-independent_contact), 'same fullH packet current contact double Ward', 5e-7)
    adversaries['missing_Hamiltonian_transport'] = adverse(np.linalg.norm(current), 'same fullH packet microscopic current continuity', 5e-7)
    adversaries['missing_contact'] = adverse(np.linalg.norm(independent_contact), 'same fullH packet current contact double Ward', 5e-7)
    # Quantum harmonic translation/current retained, no frozen harmonic coordinate.
    def py(v):
        return charged.finish(charged.wave(v)*charged.py[:, None, None], v)
    harmonic_current = charged.current(packet, HARMONIC)
    accept(np.linalg.norm(harmonic_current-1j*(py(charged.action(packet))-charged.action(py(packet)))), 'same fullH retained free harmonic current', 5e-7)
    propagation = []
    for t in (0., .2, .6):
        evolved = propagate(charged, packet, t, 0., None, charged.trace)
        accept(abs(np.vdot(evolved, evolved).real-initial_norm), 'same fullH unfiltered complex packet fixed initial norm', 5e-7)
        rho = np.sum(abs(charged.wave(evolved))**2, axis=(0, 2))/initial_norm
        energy_now = float(np.vdot(evolved, charged.action(evolved)).real/initial_norm-E0)
        accept(abs(energy_now-energy), 'same fullH complex packet conserved Rayleigh excess', 5e-7)
        accept(np.linalg.norm(np.asarray(charged.sector_weights(evolved))-initial_sectors),
               'same fullH finite packet retains BOTH source-addressed mod5 sectors', 5e-7)
        propagation.append(dict(time=t, norm=float(np.vdot(evolved, evolved).real), charge_density=(rho@charged.rho).tolist(),
            current0=float(np.vdot(evolved, charged.current(evolved, weights)).real/initial_norm),
            current_all_sites=[float(np.vdot(evolved, charged.current(evolved, D[x])).real/initial_norm) for x in range(3)],
            source_addressed_sector_weights=charged.sector_weights(evolved), Rayleigh_excess=energy_now))
    finite_pulses = []
    for T in args.durations:
        un, _ = echo(neutral, ncontrol, state, T)
        us = np.asarray([echo(charged, ccontrol, s, T)[0] for s in sources])
        right = np.asarray([prepare(neutral, charged, un, mark) for mark in maps])
        forms = history_forms(us, right, inverse)
        first_drift = full_commutator
        finite_pulses.append(dict(duration_each_leg=T, strength=2*np.pi/T, **forms,
                                 initial_fullH_relative_residual=float(np.linalg.norm(initial_whiten(first_drift, inverse))),
                                 direct_fullH_vs_ideal_source_error=float(np.linalg.norm(initial_whiten(us-ideal_sources, inverse))),
                                 full_neutral_vs_ideal_source_error=float(np.linalg.norm(un-ideal_neutral)),
                                 Hamiltonian_reversed=False, final_metric_renormalized=False))
    complex_mark = [sum((alpha[a]*maps[a][b] for a in range(4)), sparse.csr_matrix(maps[0][b].shape, dtype=complex)) for b in range(4)]
    orbital_mid = float(np.linalg.norm(ccontrol.orbital(packet)
        -prepare(neutral, charged, ncontrol.orbital(state), complex_mark)))
    photon_mid = float(np.linalg.norm(ccontrol.photon(packet)))
    accept(abs(sum(propagation[0]['charge_density'])), 'same complete charged reference Gauss total', 5e-7)
    if min(orbital_mid, photon_mid) <= 1e-9:
        raise AssertionError('Chosen actual quantum packet does not exercise both orbital motion and photon transport')
    return dict(zeta=zeta, neutral_dimension=neutral.size, charged_dimension=charged.size, complete_reference_dimension=4*charged.size,
        whole_light_sectors=[neutral.d, charged.d], E0=E0, actual_neutral_cluster=cluster,
        exact_mod5_sectors=dict(definition='t=(ell-sum_x w_x n_x) mod5', site_weights=SITE_WEIGHTS.tolist(),
            neutral_weights=neutral.sector_weights(state), complex_packet_weights=initial_sectors,
            point_source_intertwiner_errors=point_sector_errors, charged_global_ground_joined=False),
        full_infinite_neutral_residual=neutral.enlarged_residual(state, E0, frame),
        eta=complex_json(eta[None])[0], metric=complex_json(metric), metric_eigenvalues=metric_values.tolist(),
        exact_coherent_CAR_metric_error=car_error, point_norm2_discrepancy=point_error,
        norm_supplier='actual full neutral quantum spectral-cluster density and exact complete-source initial Gram; no occupation filter',
        complex_energy_form=complex_json(energy_matrix), complex_packet_Rayleigh_excess=energy,
        complex_packet_initial_norm=initial_norm,
        electric_commutator_form=complex_json(electric_form), light_commutator_form=complex_json(light_form),
        retained_weighted_neutral_residual_form=complex_json(weighted_form), full_Rayleigh_transfer_error=transfer_error,
        relative_spin_intertwiner_max_error=max(closure_errors), source_only_echo=ideal,
        nonzero_finite_width_orbital_action=orbital_mid, quantum_photon_transport_action=photon_mid,
        source_midpoint_charge_density=(midpoint_density@charged.rho).tolist(),
        source_midpoint_full_neutral_RIGHT_motion=float(np.linalg.norm(neutral_mid-state)),
        current_continuity_error=current_error, current_contact_error=contact_error,
        actual_fullH_packet_motion=propagation, actual_finite_strength_fullH_pulses=finite_pulses,
        adversaries=adversaries, neutral_RIGHT_dropped=False, spin_total_charged_state_certified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--levels', type=int, default=3)
    parser.add_argument('--harmonics', type=int, default=2)
    parser.add_argument('--UE', type=float, default=.37)
    parser.add_argument('--K', type=float, default=.9)
    parser.add_argument('--axis', type=int, choices=(0, 1, 2), default=2)
    parser.add_argument('--photon-shear', type=float, default=.23)
    parser.add_argument('--durations', nargs='+', type=float, default=[.04, .02])
    parser.add_argument('--zeta', type=float, default=0.)
    parser.add_argument('--bloch-width', type=float, default=.2)
    parser.add_argument('--bloch-nodes', type=int, default=3)
    parser.add_argument('--eigenpairs', type=int, default=6)
    args = parser.parse_args()
    if args.levels < 2 or args.harmonics < 2 or min(args.UE, args.K, args.bloch_width) <= 0 or any(t <= 0 for t in args.durations) or args.photon_shear == 0:
        raise ValueError('Positive energies, genuine oscillator/Fourier ranges, Bloch width, pulse durations and nonzero graph curl contraction required')
    if args.bloch_nodes < 1 or abs(args.zeta)+args.bloch_width >= .5:
        raise ValueError('Entire normalizable Bloch packet must fit inside one fibre convention interval')
    band = RotationBand(a=1., m=.2, r=.12, v=.3, improved=False)
    selected, frame = original_frame(band, 5, np.zeros(3), True)
    spins, labels, geometry = mechanical_geometry(frame, selected)
    nodes, weights = np.polynomial.legendre.leggauss(args.bloch_nodes)
    anchor = GraphQuantum(args.levels, args.harmonics, args.zeta, 6, args.UE, args.K, frame)
    anchor_energy, anchor_state, anchor_cluster = anchor.solve(args.eigenpairs)
    results = [fibre(args, args.zeta+args.bloch_width*x, frame, labels, spins, anchor_state, anchor_cluster) for x in nodes]
    # These are quadrature weights of a real L2 direct-integral supplier, NOT native well weights.
    weights = weights/2
    def matrix(report):
        return np.array([[z['real']+1j*z['imag'] for z in row] for row in report])
    aggregate_pulses = []
    for j, T in enumerate(args.durations):
        reports = [row['actual_finite_strength_fullH_pulses'][j] for row in results]
        cross = sum(w*matrix(row['raw_normalized_cross']) for w, row in zip(weights, reports))
        right_metric = sum(w*matrix(row['after_RIGHT_metric']) for w, row in zip(weights, reports))
        aggregate_pulses.append(dict(duration_each_leg=T, raw_cross_quadrature=complex_json(cross),
            RIGHT_metric_quadrature=complex_json(right_metric), RIGHT_metric_eigenvalues=np.linalg.eigvalsh(right_metric).tolist(),
            endpoint_quadrature_operator_upper=math.sqrt(sum(float(w)*row['endpoint_operator_error']**2 for w, row in zip(weights, reports))),
            final_metric_renormalized=False, nonlinear_integral_error_not_certified=True))
    packet_observations = []
    for j, t in enumerate((0., .2, .6)):
        reports = [row['actual_fullH_packet_motion'][j] for row in results]
        packet_observations.append(dict(time=t,
            charge_density_quadrature=np.average(np.asarray([row['charge_density'] for row in reports]), axis=0, weights=weights).tolist(),
            current_all_sites_quadrature=np.average(np.asarray([row['current_all_sites'] for row in reports]), axis=0, weights=weights).tolist()))
    metric_constancy = accept(max(np.linalg.norm(matrix(row['metric'])-matrix(results[0]['metric'])) for row in results),
                             'same physical Bloch packet initial complete-source metric', 5e-7)
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
              ('noncompact_finite_packet_spin.py', 'noncompact_finite_packet_spin.md', 'noncompact_native_current.py',
               'original_finite_packet_echo.py', 'original_finite_packet_echo.md',
               'original_intrinsic_angular_transfer.py', 'original_intrinsic_angular_transfer.md',
               'noncompact_neutral_supplier.md')}
    print(json.dumps(dict(plan=PLAN, revision=REVISION, template=TEMPLATE, source_hashes=hashes,
        law='different noncompact photon root, exact Peierls phases, reduced exact longitudinal Coulomb sectors',
        graph=dict(D=D.tolist(), C=C.tolist(), restored=F.tolist(), retained_harmonic=HARMONIC.tolist(), Fourier_shifts=list(NU), SAME_R=REFERENCE),
        controller_graph_curl_contraction=(args.photon_shear*(.28-(-.11))*HARMONIC/math.sqrt(2)).tolist(),
        native_frame_Gram=np.asarray(selected['G']).tolist(), frame_scope='native N5 frame supplier only, diagnostic three-site graph NOT cubic N0',
        mechanical_geometry=geometry, Bloch_packet=dict(centre=args.zeta, width=args.bloch_width,
            amplitude='g(theta)=(2 width)^(-1/2) on centre+-width; full-R normalizable direct integral',
            quadrature_nodes=args.bloch_nodes, quadrature_weights=weights.tolist(), finite_quadrature_not_exact_integral=True,
            neutral_periodic_coefficients='SAME actual coupled anchor ground-cluster purification in every fibre; off-centre fibres are trials, not grounds',
            anchor_ground_energy=anchor_energy, anchor_ground_cluster=anchor_cluster,
            exact_neutral_Bloch_kinetic_excess=args.UE*args.bloch_width**2/30),
        complete_Bloch_packet_consumer=dict(initial_metric_constancy_error=metric_constancy,
            complex_source_Rayleigh_excess_quadrature=sum(float(w)*row['complex_packet_Rayleigh_excess'] for w, row in zip(weights, results)),
            actual_fullH_motion_quadrature=packet_observations, actual_fullH_pulses_quadrature=aggregate_pulses),
        results=results, full_field_normals_paid=False, native_ground_or_total_spin_paid=False,
        thermodynamic_or_Lorentz_phase_proved=False, campaign_complete=False), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
