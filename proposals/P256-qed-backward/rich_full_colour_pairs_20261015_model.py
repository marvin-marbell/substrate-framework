"""Exact source/current-invariant mirror representation of the frozen CAR24 law.

No original-mirror pair cut: Hadamard mirror eigenmodes turn every H(A), J(A)
and K(A) into operators conserving rotated-mirror flavour/spin populations.
The reference branch fills all 12 negative-X eigenmodes; the physical original
positive-mirror uud insertion has only its upper-X creation component surviving.
Disconnected rotated-occupation sectors are not a global-vacuum theorem.
"""
from itertools import product
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from rich_colour_sea_20261014_model import (
    mode, string, charge_conjugate, masks_for, one_body, gauge_sector,
    scalar_field, source_matrix, generators, loop_field,
)


def project(sector, operator):
    return (sector['P'].conj().T @ operator @ sector['P']).toarray()


def rotated_conjugation(source, target, loop):
    lookup = {mask: index for index, mask in enumerate(target['masks'])}
    rows, values = [], []
    for mask in source['masks']:
        other, sign = charge_conjugate(mask)
        rows.append(lookup[other])
        # O^T (i tau_y) O = -i tau_y for the real Hadamard O.
        values.append(sign * (-1) ** mask.bit_count())
    matter = sparse.csr_matrix((values, (rows, range(len(rows)))), shape=(len(target['masks']), len(source['masks'])))
    swap = np.eye(19); swap[1:10, 1:10] = 0; swap[10:, 10:] = 0
    swap[1:10, 10:] = np.eye(9); swap[10:, 1:10] = np.eye(9)
    raw = sparse.kron(matter, swap, format='csr')
    return project_between(target, raw, source), raw


def project_between(target, operator, source):
    return (target['P'].conj().T @ operator @ source['P']).toarray()


def rotated_operators(sector, loop):
    masks = sector['masks']; n = len(masks); identity = sparse.eye(n, format='csr')
    routes = []; commutators = []
    for flavour in range(2):
        signed = one_body(masks, [(mode(flavour, m, s, c), mode(flavour, m, s, c), 1 - 2*m)
                                 for m, s, c in product(range(2), range(2), range(3))])
        forward = sparse.csr_matrix((19*n, 19*n), dtype=complex)
        for a, b in product(range(3), repeat=2):
            body = one_body(masks, [(mode(flavour, m, s, a), mode(flavour, m, s, b), 1 - 2*m)
                                    for m, s in product(range(2), range(2))])
            forward += sparse.kron(body, loop['U'][a, b], format='csr')
        direct = sparse.kron(signed, sparse.eye(19), format='csr')
        commutators.append(max(float(sparse.linalg.norm(g @ route - route @ g)) for g in sector['G'] for route in [direct, forward]))
        routes.append((project(sector, direct), project(sector, forward)))
    mz = np.array([sum((.5 if (i % 6)//3 == 0 else -.5) for i in range(24) if mask & (1 << i)) for mask in masks])
    lower_masks = sorted({step[0] for mask in masks for f, m, c in product(range(2), range(2), range(3))
                          if (step := string(mask, [(mode(f,m,1,c),True),(mode(f,m,0,c),False)])) is not None})
    lookup = {mask:i for i,mask in enumerate(lower_masks)}; rr,cc,vv = [],[],[]
    for j,mask in enumerate(masks):
        for f,m,c in product(range(2),range(2),range(3)):
            step = string(mask, [(mode(f,m,1,c),True),(mode(f,m,0,c),False)])
            if step is not None: rr.append(lookup[step[0]]);cc.append(j);vv.append(step[1])
    lower = sparse.csr_matrix((vv,(rr,cc)),shape=(len(lower_masks),n))
    spin = project(sector,sparse.kron(lower.conj().T @ lower+sparse.diags(mz*(mz-1)),sparse.eye(19)))
    matter_casimir = sum(q @ q for q in sector['Q'])
    result = dict(routes=routes,S2=spin,octet=project(sector,sparse.kron(matter_casimir/3,sparse.eye(19))),route_gauss_commutators=commutators)
    result.update({key:project(sector,sparse.kron(identity,loop[key])) for key in ['C','B','boundary']})
    return result


def law(operators, scalar, parameters, angle=0.):
    n = operators['S2'].shape[0]; nf = len(scalar['grid'])
    h = np.kron(np.eye(n),scalar['H']) + np.kron(parameters['electric']*operators['C']/2-parameters['plaquette']*operators['B'],np.eye(nf))
    j = np.zeros_like(h,dtype=complex); k = j.copy()
    for charge,(direct,forward) in zip((2/3,-1/3),operators['routes']):
        phase = np.exp(1j*charge*angle); coupling = parameters['scalar_coupling']*parameters['loop_route']/2
        route = phase*forward+phase.conjugate()*forward.conj().T
        h = h + np.kron(parameters['scalar_coupling']*direct+coupling*route,scalar['phi'])
        j -= np.kron(1j*charge*coupling*(phase*forward-phase.conjugate()*forward.conj().T),scalar['phi'])
        k += np.kron(-charge**2*coupling*route,scalar['phi'])
    return h,j,k


def build(parameters):
    loop = loop_field(); scalar = scalar_field(parameters)
    neutral = gauge_sector(masks_for((0,0),0,0),loop)
    baryon = gauge_sector(masks_for((2,1),1,0),loop)
    anti_masks = sorted(charge_conjugate(mask)[0] for mask in baryon['masks'])
    anti = gauge_sector(anti_masks,loop)
    sectors = dict(neutral=neutral,baryon=baryon,antibaryon=anti)
    ops = {name:rotated_operators(sector,loop) for name,sector in sectors.items()}
    nf = len(scalar['grid']); cn,cnraw = rotated_conjugation(neutral,neutral,loop); cb,cbraw = rotated_conjugation(baryon,anti,loop)
    cn = np.kron(cn,np.eye(nf));cb = np.kron(cb,np.eye(nf))
    # Original c_positive^dagger=(a_positive^dagger+a_negative^dagger)/sqrt2.
    # Every a_negative mode is occupied in this exact reference branch.
    sources = {str(spin):np.kron(source_matrix(neutral,baryon,spin)/np.sqrt(8),np.eye(nf)) for spin in [.5,1.5]}
    return dict(parameters=parameters,loop=loop,scalar=scalar,sectors=sectors,operators=ops,
                laws={name:law(op,scalar,parameters) for name,op in ops.items()},Cneutral=cn,Cbaryon=cb,
                sources=sources,anti_sources={spin:cb @ source @ cn.conj().T for spin,source in sources.items()})


def spin_spectrum(h, spin):
    values, vectors = eigh(spin)
    labels = np.rint((np.sqrt(np.maximum(0,1+4*values))-1)).astype(int)
    energies, states, spins = [], [], []
    for label in np.unique(labels):
        p = vectors[:,labels == label]; e,v = eigh(p.conj().T @ h @ p)
        energies.extend(e);states.append(p @ v);spins.extend([label/2]*len(e))
    order = np.argsort(energies)
    return np.asarray(energies)[order],np.column_stack(states)[:,order],np.asarray(spins)[order]


def retarded(energies,vectors,current,state,frequencies,eta,contact):
    energy = float(np.vdot(state,(vectors*energies) @ (vectors.conj().T @ state)).real)
    gaps = energies-energy;weights = abs(vectors.conj().T @ current @ state)**2
    first = np.array([sum(weights/(w-gaps+1j*eta)) for w in frequencies])
    second = np.array([-sum(weights/(w+gaps+1j*eta)) for w in frequencies])
    seagull = float(np.vdot(state,contact @ state).real)
    return dict(energy=energy,gaps=gaps,weights=weights,first=first,second=second,retarded=first+second,contact=seagull,kernel=first+second+seagull)


def rotate_to_original(mask):
    """Exact CAR wedge Hadamard transform of one rotated determinant."""
    states = {0:1.}
    for index in reversed([i for i in range(24) if mask & (1 << i)]):
        f,rem = divmod(index,12);m,rem = divmod(rem,6);s,c = divmod(rem,3)
        following = {}
        for current,value in states.items():
            for original_mirror in range(2):
                coefficient = (-1 if m == original_mirror == 1 else 1)/np.sqrt(2)
                step = string(current,[(mode(f,original_mirror,s,c),True)])
                if step is not None: following[step[0]] = following.get(step[0],0)+value*coefficient*step[1]
        states = following
    return states
