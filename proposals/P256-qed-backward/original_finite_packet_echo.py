#!/usr/bin/env python3
"""Finite-width guarded-plateau echo; parent-runnable source/algebra consumer.

No native ground, native photon coupling or finite-H error is numerically supplied.
The active orbital block is an exact invariant block of the complete original
orbital operator, not a Hamiltonian/occupation truncation.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.linalg import eigh, expm
from scipy.special import erfc
from scipy.stats import qmc

from bulk_quantum_selection import BulkQuantumLaw
from original_total_angular_transfer import primitive_flow, scalar_orbital
from original_uniform_charge_gap import car_action, exterior_hamiltonian, sector_basis
from rotation_band import RotationBand
from sea_selection import SELECTED
from unrestricted_field_minimum import BARE, coordinates

ROOT = Path(__file__).resolve().parent
ARCHIVE = Path('/home/administrator/marvin-private/research/issue222/world-model/sources')
PLAN = 'issue-222-continuing-backward-20260930'
REVISION = 'a406dca7a4cc642318250c028a763daa96fa66a5999da5dae0c96e53910c5c91'
TEMPLATE = '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'


def accept(error, label, tolerance=3e-7):
    error = float(error)
    if not math.isfinite(error) or error > tolerance:
        raise AssertionError(f'{label}: {error} > {tolerance}')
    return error


def reject(error, label, tolerance=3e-7):
    try:
        accept(error, label, tolerance)
    except AssertionError as failure:
        return str(failure)
    raise AssertionError('Adversary passed SAME assertion: '+label)


def bind(path, names, heading=None):
    """Load actual complete supplier definitions; never run a supplier main."""
    text = path.read_text()
    section = text if heading is None else text.split(heading, 1)[1]
    body = section.split('```python\n', 1)[1].split('\n```', 1)[0]
    tree = ast.parse(body)
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError('Supplier interface changed: '+str(path))
    space = dict(globals(), coo_matrix=sparse.coo_matrix)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), space)
    return space, hashlib.sha256(text.encode()).hexdigest()


def profiles(law):
    centre = np.full(3, law.size//2)
    cells = law.sites-centre
    radius = np.max(abs(cells), axis=1)
    # Complete support cube1; its one-edge halo cube2 is STRICTLY inside cube3.
    g = .25*(radius <= 1)
    b = (radius <= 3).astype(float).reshape((law.size,)*3)
    R = law.index[(0, 0, 0)]
    if b.reshape(-1)[R] != 0:
        raise ValueError('SAME reference must lie outside the spin plateau')
    # A genuinely new compact two-site packet, not the original Gaussian tails.
    packet = [law.index[tuple(centre+np.array((-2, 1, 1)))],
              law.index[tuple(centre+np.array((-1, 1, 1)))]]
    eta = np.zeros(law.volume, complex)
    eta[packet] = np.array([math.sqrt(.6), math.sqrt(.4)*np.exp(.37j)])
    return cells*law.band.a, g, b, R, packet, eta


def edges(law, links, velocity):
    tail = np.repeat(np.arange(law.volume), 3)
    head = law.target.reshape(-1)
    coefficient = ((velocity+velocity[law.target, np.arange(3)]).reshape(-1)
                   /(4j*law.band.a))
    # velocity[law.target, arange(3)] is the axial component at each head.
    nonzero = np.flatnonzero(abs(coefficient) > 0)
    active = np.unique(np.r_[tail[nonzero], head[nonzero]])
    return tail, head, coefficient, nonzero, active


def edge_commutator(law, links, phases, velocity, ell):
    tail, head, coefficient, nz, unused = edges(law, links, velocity)
    blocks = links.reshape(-1, 4, 4)[nz]*np.exp(1j*phases.reshape(-1)[nz])[:, None, None]
    blocks *= coefficient[nz, None, None]
    defects = blocks@ell[head[nz]]-ell[tail[nz]]@blocks
    return math.sqrt(2)*float(np.linalg.norm(defects))


def scalar_active(law, links, velocity):
    tail, head, coefficient, nz, active = edges(law, links, velocity)
    lookup = np.full(law.volume, -1)
    lookup[active] = np.arange(len(active))
    scalars = np.trace(links.reshape(-1, 4, 4)[nz], axis1=1, axis2=2)/4
    central_error = np.linalg.norm(links.reshape(-1, 4, 4)[nz]-scalars[:, None, None]*np.eye(4))
    accept(central_error, 'SAME central original orbital links')
    rows, columns = lookup[tail[nz]], lookup[head[nz]]

    def orbital(phases):
        entries = coefficient[nz]*scalars*np.exp(1j*phases[nz])
        return sparse.coo_matrix((np.r_[entries, entries.conj()],
            (np.r_[rows, columns], np.r_[columns, rows])), shape=(len(active), len(active))).tocsr()
    return active, orbital


def ordered_echo(law, links, A, velocity, photon, spin, packet, eta, F, W, quartet):
    active, orbital = scalar_active(law, links, velocity)
    d, E = len(active), 3*law.volume
    initial = np.r_[A.reshape(-1).astype(complex), np.eye(d).reshape(-1)]

    def leg(sign):
        def rhs(t, value):
            phases = value[:E].real
            transport = value[E:].reshape(d, d)
            return np.r_[-sign*photon(t, phases), (sign*1j*(orbital(phases)@transport)).reshape(-1)]
        return rhs
    first = solve_ivp(leg(1), (0, 2*np.pi), initial, rtol=2e-10, atol=2e-11, dense_output=True)
    if not first.success:
        raise RuntimeError(first.message)
    second = solve_ivp(leg(-1), (0, 2*np.pi), first.y[:, -1], rtol=2e-10, atol=2e-11)
    if not second.success:
        raise RuntimeError(second.message)
    end = second.y[:, -1]
    orbital_error = accept(np.linalg.norm(end[E:].reshape(d, d)-np.eye(d)),
                          'SAME ENTIRE orbital propagator reversal', 2e-6)
    photon_error = accept(np.linalg.norm(end[:E].real-A.reshape(-1)),
                         'SAME full compact photon echo endpoint', 2e-6)
    M = np.zeros((d, 4, 16), complex)
    lookup = {site: index for index, site in enumerate(active)}
    # Some axes isolate a packet row geometrically; retain these rows separately.
    isolated = []
    for x in packet:
        block = np.concatenate([eta[x]*F[x]*v@W[x] for v in quartet], axis=1)
        if x in lookup:
            M[lookup[x]] = block
        else:
            isolated.append(block)
    row_motion = float(np.linalg.norm(orbital(A.reshape(-1))@M.reshape(d, -1)))
    if row_motion <= 1e-8:
        raise AssertionError('Finite packet must have a nonzero original orbital row on every axis')
    final_scalar = end[E:].reshape(d, d)
    full = np.einsum('xy,yac->xac', final_scalar, M)
    rotation = expm(2j*np.pi*spin)
    final = np.einsum('ab,xbc->xac', rotation, full)
    residual = np.linalg.norm(final+M)**2
    residual += sum(np.linalg.norm(rotation@block+block)**2 for block in isolated)
    denominator = float(np.linalg.norm(M)**2+sum(np.linalg.norm(z)**2 for z in isolated))
    if denominator <= 0:
        raise ArithmeticError('Quadrature supplied a zero packet; no substitute initial norm is allowed')
    endpoint = accept(math.sqrt(residual/denominator), 'SAME finite packet intrinsic minus endpoint', 2e-6)
    halfway = first.sol(np.pi)[E:].reshape(d, d)
    mask = np.array([site in packet for site in active], float)
    intermediate = np.linalg.norm(halfway@np.diag(mask)-np.diag(mask)@halfway)
    missing = reject(np.linalg.norm(first.y[E:, -1].reshape(d, d)-np.eye(d)),
                     'SAME ENTIRE orbital propagator reversal', 2e-6)
    return dict(active_scalar_rows=d, complete_original_rows=4*law.volume,
                full_colour_orbital_dimension=4*d, photon_endpoint=photon_error,
                entire_orbital_endpoint=orbital_error, finite_packet_minus=endpoint,
                intermediate_packet_subspace_motion=float(intermediate),
                original_finite_packet_orbital_row_norm=row_motion,
                wrong_missing_countertransport=missing,
                algebraic_exterior_functor_retains_every_occupation=True)


def normal_controller_jets(law, frame, links, photon_lib, b, f, velocity, axis, normal):
    """Full 105V raw / 15V Gauss projection, BOTH actual source JVPs.

    This is an all-coordinate JVP comparison, not a native normal ground solve
    or a numerical spectral bound. No normal mode is selected or discarded.
    """
    V = law.volume
    Df, I, rhs, spins = normal['geom'](law, frame)
    ec, ell = normal['solve_ell'](law, I, rhs[axis], f)
    tail = np.repeat(np.arange(V), 3)
    head = law.target.reshape(-1)
    incidence = photon_lib['incidence'](law)
    c = 1/.6
    val, vec = eigh(I)
    k = 2*np.pi*np.arange(law.size)/law.size
    lp = 2*((1-np.cos(k))[:, None, None]+(1-np.cos(k))[None, :, None]
            +(1-np.cos(k))[None, None, :])

    def inverse(rhs_all):
        transform = np.fft.fftn(rhs_all.reshape((law.size,)*3+(15,)), axes=(0, 1, 2))@vec
        return (np.fft.ifftn(transform/(val+c*lp[..., None]), axes=(0, 1, 2)).real@vec.T).reshape(V, 15)

    def unpack(y):
        return (np.einsum('xma,aij->xmij', y[:60*V].reshape(V, 4, 15), law.basis),
                np.einsum('ea,aij->eij', y[60*V:].reshape(3*V, 15), law.generators))

    def project(y):
        fields = y[:60*V].reshape(V, 60)
        h = y[60*V:].reshape(3*V, 15)
        charge = inverse(fields@Df+c*(incidence@h))
        return np.r_[(fields-charge@Df.T).reshape(-1), (h-incidence.T@charge).reshape(-1)]

    def rotate(fields):
        result = np.zeros_like(fields)
        i, j = ((2, 3), (3, 1), (1, 2))[axis]
        result[:, i], result[:, j] = -fields[:, j], fields[:, i]
        return result

    C, anchors = photon_lib['photon_matrices'](law)
    # Linearized compact colour clovers use ALL the same anchored loops.
    def jet(y, v, torque, connection):
        df, h = unpack(y)
        loops = (C@y[60*V:].reshape(3*V, 15))
        colour_coeff = np.zeros((V, 3, 3, 15))
        for loop, (x, j, i, orientation) in enumerate(anchors):
            colour_coeff[x, j, i] += orientation*loops[loop]/(4*law.band.a**2)
            colour_coeff[x, i, j] -= orientation*loops[loop]/(4*law.band.a**2)
        result_f = torque[:, None, None, None]*rotate(df)
        for j in range(3):
            forward = law.target[:, j]
            positions = law.sites.copy(); positions[:, j] = (positions[:, j]-1)%law.size
            backward = np.array([law.index[tuple(site)] for site in positions])
            hf, hb = h[3*np.arange(V)+j], h[3*backward+j]
            difference = (df[forward]-df[backward]+1j*((hf+hb)[:, None]@frame-frame@(hf+hb)[:, None]))/(2*law.band.a)
            result_f -= v[:, j, None, None, None]*difference
        result_h = np.zeros((V, 3, 15))
        for i in range(3):
            ysite = law.target[:, i]
            for j in range(3):
                average = (v[:, j]+v[ysite, j])/2
                result_h[:, i] -= law.band.a*average[:, None]*(colour_coeff[:, j, i]+colour_coeff[ysite, j, i])/2
        result_f -= 1j*(connection[:, None]@df-df@connection[:, None])
        sums = connection[tail]+connection[head]
        correction = -.5j*(sums@h-h@sums)
        result_h += (2*np.einsum('aij,eji->ea', law.generators, correction).real).reshape(V, 3, 15)
        return project(np.r_[coordinates(result_f, law.basis).real.reshape(-1), result_h.reshape(-1)])

    y = project(np.random.default_rng(22210071+axis).normal(size=105*V))
    y /= np.linalg.norm(y)
    fields = np.tile(frame, (V, 1, 1, 1))
    df, h = unpack(y)
    step = 2e-5
    reports = []
    for leg, v, torque, connection in ((1, velocity, f.reshape(-1), ell),
                                       (2, -velocity, np.zeros(V), np.zeros_like(ell))):
        def curve(s):
            changed = np.array([expm(1j*s*z) for z in h]).reshape(V, 3, 4, 4)@links
            Yf, Yu, unused = primitive_flow(law, fields+s*df, changed, np.zeros((V, 3)), v, np.zeros(V))
            Yf += torque[:, None, None, None]*rotate(fields+s*df)
            Yf -= 1j*(connection[:, None]@(fields+s*df)-(fields+s*df)@connection[:, None])
            tangent = (-1j*Yu@changed.conj().swapaxes(-1, -2)).reshape(3*V, 4, 4)
            moved = changed.reshape(3*V, 4, 4)
            tangent -= connection[tail]-moved@connection[head]@moved.conj().swapaxes(-1, -2)
            # RNC differential of log(exp(i h)): retain the BCH half contact.
            tangent -= .5j*s*(h@tangent-tangent@h)
            return project(np.r_[coordinates(Yf, law.basis).real.reshape(-1),
                                 2*np.einsum('aij,eji->ea', law.generators, tangent).real.reshape(-1)])
        numerical = (curve(step)-curve(-step))/(2*step)
        expected = jet(y, v, torque, connection)
        error = accept(np.linalg.norm(numerical-expected), 'SAME FULL normal controller RNC jet', 2e-6)
        affine_norm = float(np.linalg.norm(curve(0.)))
        if leg == 2:
            accept(affine_norm, 'SAME torqueless BODY affine controller')
        omitted_B = reject(np.linalg.norm(numerical), 'SAME FULL normal controller RNC jet', 2e-6)
        reports.append(dict(leg=leg, raw_coordinates=105*V, Gauss_rows=15*V,
                            physical_normal_axes=90*V, source_jet_error=error,
                            full_normal_jet_norm=float(np.linalg.norm(expected)),
                            actual_BODY_affine_norm=affine_norm, wrong_omit_B=omitted_B,
                            BODY_affine_c_zero=(leg == 2), B_zero_assumed=False))
    return reports


def all64_geometry(law, family, normal, f, velocity_list, links, A, b, packet):
    reports = []
    for row in family['all64']:
        frame = normal['frameG'](np.asarray(row['G']))
        unused, I, rhs, spins = normal['geom'](law, frame)
        for axis in range(3):
            unused, ell = normal['solve_ell'](law, I, rhs[axis], f)
            true_links = law.toron_links(np.pi/2*np.asarray(row['centre']))
            target = b.reshape(-1, 1, 1)*spins[axis]
            defect = ell-target
            # Global one-body operator norm of this block-diagonal connection.
            delta = float(np.max(np.max(abs(np.linalg.eigvalsh(defect)), axis=1)))
            ref = float(np.max(abs(np.linalg.eigvalsh(ell[0]))))
            comm = edge_commutator(law, true_links, A, velocity_list[axis], ell)
            # Individual charged/neutral unitary differences add; no Z is
            # supplied, so this is NOT a normalized native marked residual.
            bound = min(4., 2*np.pi*((4*law.volume+1)*delta+ref))
            reports.append(dict(centre=row['centre'], axis=axis,
                true10_Gram=row['G'], original_scalar_offset=row['original_classical_scalar'],
                global_connection_operator_defect=delta, SAME_reference_norm=ref,
                global_orbital_connection_commutator=comm,
                whole_CAR_unitary_comparison_upper=bound,
                packet_sites=packet, physical_Gauss_rows=15*law.volume))
    return reports


def whole_occupation(packet_o, packet_h, spin, eta, F, W):
    """Entire two-site 8-mode algebra, all sectors; explicitly not native ground."""
    rows, charged = sector_basis(8, 4), sector_basis(8, 5)
    lookup = {state: i for i, state in enumerate(charged)}
    creators = []
    for mode in range(8):
        matrix = np.zeros((len(charged), len(rows)), complex)
        for j, state in enumerate(rows):
            action = car_action(state, mode, True)
            if action is not None:
                matrix[lookup[action[0]], j] = action[1]
        creators.append(matrix)

    def creator(M):
        return sum((np.kron(creators[i], np.eye(4)[:, b, None])*M[i, b]
                    for i in range(8) for b in range(4)), np.zeros((4*len(charged), len(rows)), complex))
    PB = np.diag([float((s & 15).bit_count() == 2 and (s >> 4).bit_count() == 2) for s in rows])
    quartet = [np.eye(4), *[2j*s for s in spin]]
    marks = [np.vstack([eta[x]*F[x]*v@W[x] for x in range(2)]) for v in quartet]
    maps = [creator(M)@PB for M in marks]
    # Complete PB midpoint density, not an assigned native ground probability.
    rho = PB/np.trace(PB)
    gram = np.array([[np.trace(rho@a.conj().T@z) for z in maps] for a in maps])
    Z = float(np.sum(abs(eta*F)**2))
    norm = accept(np.linalg.norm(gram-2*Z*np.eye(4)), 'SAME finiteB raw 2Z quartet Gram')
    J0, unused = exterior_hamiltonian(packet_o, 4)
    Jc, unused = exterior_hamiltonian(packet_o, 5)
    H0, unused = exterior_hamiltonian(packet_h, 4)
    Hc, unused = exterior_hamiltonian(packet_h, 5)
    contact_reports = []
    for generator, charge, name in ((J0, Jc, 'orbital/current'), (H0, Hc, 'HL')):
        B = maps[0]; CM = creator(marks[0])
        full = np.kron(charge, np.eye(4))@B-B@generator
        onebody = packet_o if name == 'orbital/current' else packet_h
        row = creator(onebody@marks[0])@PB
        contact = CM@(generator@PB-PB@generator)
        good = accept(np.linalg.norm(full-row-contact), 'SAME FULL finiteB '+name+' projector contact')
        bad = reject(np.linalg.norm(full-row), 'SAME FULL finiteB '+name+' projector contact')
        contact_reports.append(dict(kind=name, residual=good, omitted_contact=bad,
                                    contact_norm=float(np.linalg.norm(contact))))
    q = np.kron(np.eye(2), spin[2])
    Q0, unused = exterior_hamiltonian(q, 4)
    Qc, unused = exterior_hamiltonian(q, 5)
    first0 = expm(1j*(J0+Q0)*2*np.pi)
    firstc = expm(1j*(Jc+Qc)*2*np.pi)
    U0 = expm(-1j*J0*2*np.pi)@first0
    Uc = np.kron(expm(-1j*Jc*2*np.pi)@firstc, np.eye(4))
    endpoint = accept(np.linalg.norm(Uc@maps[0]+maps[0]@U0), 'SAME finiteB full neutral RIGHT endpoint')
    adverse_endpoints = {}
    badq = q.copy(); badq[:4, :4] *= .4
    generator = next(z for z in SELECTED if np.linalg.norm(z@spin[2]-spin[2]@z) > .1)
    bado = packet_o.copy()
    bado[:4, 4:] = expm(.4j*generator)@bado[:4, 4:]
    bado[4:, :4] = bado[:4, 4:].conj().T
    for name, altered_o, altered_q in (('offplateau_halo', packet_o, badq),
                                       ('noncentral_link', bado, q)):
        a0, unused = exterior_hamiltonian(altered_o, 4)
        ac, unused = exterior_hamiltonian(altered_o, 5)
        b0, unused = exterior_hamiltonian(altered_q, 4)
        bc, unused = exterior_hamiltonian(altered_q, 5)
        bad0 = expm(-2j*np.pi*a0)@expm(2j*np.pi*(a0+b0))
        badc = np.kron(expm(-2j*np.pi*ac)@expm(2j*np.pi*(ac+bc)), np.eye(4))
        adverse_endpoints[name] = reject(np.linalg.norm(badc@maps[0]+maps[0]@bad0),
                                        'SAME finiteB full neutral RIGHT endpoint')
    Pmid = expm(1.3j*J0)@PB@expm(-1.3j*J0)
    restored = accept(np.linalg.norm(U0@PB@U0.conj().T-PB), 'SAME endpoint finiteB projector cyclicity')
    # At an intermediate angle RIGHT is genuinely nonidentity even when endpoint
    # plateau parity becomes scalar in this two-site neutral comparison.
    angle = .63
    R0 = expm(1j*(J0+Q0)*angle)
    Rc = np.kron(expm(1j*(Jc+Qc)*angle), np.eye(4))
    row_rotation = expm(1j*(packet_o+q)*angle)
    transported = creator(row_rotation@marks[0])@(R0@PB@R0.conj().T)
    right = accept(np.linalg.norm(Rc@maps[0]-transported@R0), 'SAME entire propagated neutral RIGHT')
    frozen = reject(np.linalg.norm(Rc@maps[0]-transported), 'SAME entire propagated neutral RIGHT')
    raw_cross = np.array([[np.trace(rho@(a@U0).conj().T@(Uc@z)) for z in maps] for a in maps])
    cross = accept(np.linalg.norm(raw_cross+gram), 'SAME finiteB raw minus cross amplitude')
    # Full global Fock space is the direct sum of every occupation sector.
    sectors = []
    for number in range(9):
        lift, states = exterior_hamiltonian(packet_o, number)
        rotation, unused = exterior_hamiltonian(q, number)
        sectors.append(dict(number=number, dimension=len(states),
            orbital_connection_commutator=accept(np.linalg.norm(lift@rotation-rotation@lift),
                'SAME every-occupation orbital spin commutator')))
    return dict(neutral_dimension=len(rows), charged_dimension=len(charged), opposite_reference=4,
        complete_Fock_dimension=sum(row['dimension'] for row in sectors), all_occupations=sectors,
        raw_2Z=2*Z, raw_Gram_error=norm, raw_cross_error=cross,
        full_endpoint=endpoint, projector_endpoint=restored,
        intermediate_projector_motion=float(np.linalg.norm(Pmid-PB)),
        full_neutral_RIGHT=right, wrong_frozen_RIGHT=frozen, contacts=contact_reports,
        adverse_same_finiteB_endpoints=adverse_endpoints,
        comparison_only=True, native_ground_probability_assigned=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=9)
    parser.add_argument('--spacing', type=float, default=.6)
    parser.add_argument('--heat', type=float, required=True)
    parser.add_argument('--haar-power', type=int, default=7)
    parser.add_argument('--native-input', type=Path, default=Path('/tmp/issue222-one-factor-source-native-input.json'))
    parser.add_argument('--normal', type=Path, default=ARCHIVE/'issue222-fivepoint-normal-mechanical-spin-construction.md')
    parser.add_argument('--regional', type=Path, default=ARCHIVE/'issue222-native-regional-mechanical-transport-construction.md')
    parser.add_argument('--echo', type=Path, default=ARCHIVE/'issue222-spin-echo-parent-observation.md')
    args = parser.parse_args()
    if args.size < 9 or args.size % 2 != 1 or args.spacing != .6 or not 0 < args.heat <= 2:
        raise ValueError('Guard geometry needs odd N>=9; inherited input is explicitly a=.6; auxiliary 0<t<=2')
    if not 1 <= args.haar_power <= 16:
        raise ValueError('Haar quadrature power must be in [1,16]')
    normal, normal_hash = bind(args.normal, {'frameG', 'geom', 'lap', 'solve_ell'})
    regional, regional_hash = bind(args.regional, {'incidence', 'paths', 'heat_link', 'full_heat_vertex'})
    photon, echo_hash = bind(args.echo, {'photon_matrices', 'photon_source'}, '## Complete executed source')
    photon.update(incidence=regional['incidence'])
    data = json.loads(args.native_input.read_text())
    for name, digest in data['source_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Actual native input owner changed: '+name)
    family = next(row for row in data['families'] if row['N'] == args.size)
    if len(family['all64']) != 64:
        raise ValueError('All64 true10 Grams required, no isotropic fit')
    law = BulkQuantumLaw(args.size, band=RotationBand(**{**BARE, 'a':args.spacing}))
    base = next(row for row in family['all64'] if row['centre'] == [0, 0, 0])
    frame = normal['frameG'](np.array(base['G']))
    unused, inertia, rhs, spin = normal['geom'](law, frame)
    positions, g, b, R, packet, eta = profiles(law)
    s = 2*np.einsum('aij,ji->a', law.generators, spin[0]).real
    lam = float(s@inertia@s/(s@s))
    f = b+normal['lap'](b)/(.6*lam)
    # Exact comparison is isometric BODY, separately checked rather than fitted.
    for axis in range(3):
        unused, ell = normal['solve_ell'](law, inertia, rhs[axis], f)
        accept(np.linalg.norm(ell-b.reshape(-1, 1, 1)*spin[axis]), 'SAME actual zero-class isometric connection', 2e-6)
    links = law.toron_links(np.pi/2*np.array([1, 2, 3]))
    A = np.random.default_rng(22210070).uniform(-.47, .47, (law.volume, 3))
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    C, anchors = photon['photon_matrices'](law)
    D = regional['incidence'](law)
    haar = 2*np.pi*qmc.Sobol(law.volume-1, scramble=True, seed=22210072).random_base2(args.haar_power)
    haar = np.column_stack((np.zeros(len(haar)), haar))
    F, firstjet, secondjet, heat_info = regional['full_heat_vertex'](law, D, A, R, haar, args.heat)
    independent = 2*np.pi*qmc.Sobol(law.volume-1, scramble=True, seed=22210073).random_base2(args.haar_power)
    independent = np.column_stack((np.zeros(len(independent)), independent))
    F2, unused, unused, independent_info = regional['full_heat_vertex'](law, D, A, R, independent, args.heat)
    heat_info.update(independent_scramble_packet_difference=float(np.linalg.norm(F[packet]-F2[packet])),
                     independent_scramble=independent_info, native_ground_expectation=False)
    direction = np.random.default_rng(22210074).normal(size=3*law.volume)
    direction /= np.linalg.norm(direction)
    step = 1e-4
    plus = regional['full_heat_vertex'](law, D, A+step*direction.reshape(law.volume, 3), R, haar, args.heat)[0]
    minus = regional['full_heat_vertex'](law, D, A-step*direction.reshape(law.volume, 3), R, haar, args.heat)[0]
    heat_info['first_quotient_jet_error'] = accept(np.linalg.norm((plus-minus)/(2*step)-firstjet@direction),
                                                'SAME full Iq/I0 photon quotient first jet', 2e-6)
    angle = A.reshape(1, -1)+np.asarray(D.T@haar.T).T
    heat, derivative, unused, unused = regional['heat_link'](angle, args.heat)
    logweight = np.log(heat).sum(axis=1)
    weights = np.exp(logweight-logweight.max()); weights /= weights.sum()
    denominator_score = weights@(derivative/heat)
    missing_I0 = firstjet+F[:, None]*denominator_score[None]
    heat_info['wrong_missing_I0'] = reject(np.linalg.norm(missing_I0-firstjet),
                                         'SAME full Iq/I0 photon quotient first jet', 2e-6)
    W, unused = regional['paths'](law, links, R)
    quartet = [np.eye(4), *[2j*z for z in spin]]
    velocities = [g[:, None]*np.cross(np.eye(3)[axis], positions) for axis in range(3)]
    source_reports, normal_reports = [], []
    for axis, velocity in enumerate(velocities):
        ell = b.reshape(-1, 1, 1)*spin[axis]
        tail, head, unused, nz, active = edges(law, links, velocity)
        if np.any(b.reshape(-1)[tail[nz]] != 1) or np.any(b.reshape(-1)[head[nz]] != 1):
            raise AssertionError('Nonzero original edge escaped plateau including its halo')
        global_error = accept(edge_commutator(law, links, A, velocity, ell), 'SAME GLOBAL orbital connection commutator')
        # Wrong plateau cuts through a real nonzero edge: no point-row shortcut.
        badell = ell.copy()
        halo_site = next(int(site) for site in np.r_[tail[nz], head[nz]] if g[site] == 0)
        badell[halo_site] *= .4
        halo = reject(edge_commutator(law, links, A, velocity, badell), 'SAME GLOBAL orbital connection commutator')
        badlinks = links.copy()
        edge = next(int(edge) for edge in nz if tail[edge] == halo_site or head[edge] == halo_site)
        x, j = divmod(edge, 3)
        generator = next(z for z in law.generators if np.linalg.norm(z@spin[axis]-spin[axis]@z) > .1)
        badlinks[x, j] = expm(.4j*generator)@badlinks[x, j]
        noncentral = reject(edge_commutator(law, badlinks, A, velocity, ell), 'SAME GLOBAL orbital connection commutator')
        flow, lip = photon['photon_source'](law, velocity, C, anchors)
        owner = primitive_flow(law, fields, links, A, velocity, f.reshape(-1))[2].reshape(-1)
        owner_error = accept(np.linalg.norm(flow(0, A.reshape(-1))-owner), 'SAME original full compact clover photon source')
        report = ordered_echo(law, links, A, velocity, flow, spin[axis], packet, eta, F, W, quartet)
        report.update(axis=axis, global_commutator=global_error, wrong_offplateau_halo=halo,
                      wrong_noncentral_link=noncentral, actual_photon_owner=owner_error,
                      full_photon_Lipschitz_upper=lip,
                      photon_first_jet_norm=float(np.linalg.norm(firstjet[packet])),
                      photon_second_jet_norm=float(np.linalg.norm(secondjet[packet])))
        source_reports.append(report)
        normal_reports.extend(normal_controller_jets(law, frame, links, photon, b, f, velocity, axis, normal))
    # SAME two packet sites, heat values, paths and amplitudes supply the separate
    # full 8-mode bond algebra; spectators/global Fock remain analytic suppliers.
    o = scalar_orbital(law, links, A, velocities[2])
    h = law.onebody(fields, links, A)
    x, y = packet
    modes = np.r_[np.arange(4*x, 4*x+4), np.arange(4*y, 4*y+4)]
    if np.linalg.norm(o[4*x:4*x+4, 4*y:4*y+4]) == 0:
        raise AssertionError('The finiteB comparison needs its real original nonzero orbital edge')
    operator = whole_occupation(o[np.ix_(modes, modes)], h[np.ix_(modes, modes)],
                                spin, eta[packet], F[packet], W[packet])
    geometry = all64_geometry(law, family, normal, f, velocities, links, A, b, packet)
    hashes = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
              ('original_finite_packet_echo.py', 'original_finite_packet_echo.md',
               'original_total_angular_transfer.py', 'original_uniform_charge_gap.py')}
    result = dict(plan=PLAN, revision=REVISION, template=TEMPLATE, source_hashes=hashes,
        suppliers=dict(normal=normal_hash, regional=regional_hash, executed_echo90F7=echo_hash),
        geometry=dict(N=args.size, spacing=args.spacing, all_links=3*law.volume,
            all_anchored_clovers=len(anchors), all15V_Gauss=15*law.volume,
            all90V_normals=90*law.volume, plateau_cells=3, velocity_support_cells=1,
            packet_sites=packet, SAME_R=R, auxiliary_heat=args.heat),
        source=source_reports, both_controller_full_normal_jets=normal_reports,
        whole_occupation_comparison=operator, all64_all3_actual_geometry=geometry,
        original_full_heat_quadrature=heat_info,
        true_weights='p_theta Z_theta times complete residual matrix BEFORE division; not assigned here',
        native_N0_certified=False, native_ground_computed=False, finite_H_error_paid=False,
        native_UE_K_assigned=False, mobile_spin_proved=False, campaign_complete=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
