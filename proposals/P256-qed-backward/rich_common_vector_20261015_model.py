"""Finite periodic zero-Bloch-fibre Peierls vector on frozen common CAR32 root.
No frozen producer is run. Helper models rebuild every quantum state from inputs.
"""
from __future__ import annotations
from itertools import combinations, product
import numpy as np
from scipy import sparse
from rich_common_colour_lepton_20261015_model import (
    mode, string, one_body, gauge_sector, operators, law, build, conjugate_sector,
    conjugation, project)


def signatures(mask):
    return tuple(sum(bool(mask & (1 << mode(f, m, s, c))) for c in members)
        for f, m, s, members in product(range(2), range(2), range(2), [range(3), [3]]))


def complete_masks(signature):
    groups = [list(combinations([mode(f, m, s, c) for c in members], n))
        for (f, m, s, members), n in zip(product(range(2), range(2), range(2), [range(3), [3]]), signature)]
    return sorted(sum(1 << index for group in choices for index in group) for choices in product(*groups))


def gamma_matrices():
    # Hadamard on the original mirror: X->Z, Y->-Y. Actual ordered Clifford input.
    pauli = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]], complex), np.diag([1., -1.])]
    return [np.kron(-pauli[1], spin) for spin in pauli]


def vector_terms(axis):
    gamma = gamma_matrices()[axis]
    terms = []
    for f, c, m, s, n, t in product(range(2), range(4), range(2), range(2), range(2), range(2)):
        q = ([2/3, 2/3, 2/3, 0.] if f == 0 else [-1/3, -1/3, -1/3, -1.])[c]
        value = q * gamma[2*m+s, 2*n+t]
        if value != 0:
            terms.append((mode(f, m, s, c), mode(f, n, t, c), value))
    return terms


def reachable_signatures(base):
    reached = set()
    for mask in base['masks']:
        for axis in range(3):
            for a, b, _ in vector_terms(axis):
                step = string(mask, [(a, True), (b, False)])
                if step is not None:
                    reached.add(signatures(step[0]))
    return sorted(reached)


def current_block(base, target, axis, velocity, nf):
    raw = sparse.kron(one_body(base['masks'], vector_terms(axis), target['masks']), sparse.eye(19), format='csr')
    matter = (target['P'].conj().T @ raw @ base['P']).toarray()
    return -velocity * np.kron(matter, np.eye(nf))


def closure(sector):
    present = set(sector['masks']); missing = set()
    for mask in present:
        for f, m, s, a, b in product(range(2), range(2), range(2), range(4), range(4)):
            if (a == 3) != (b == 3):
                continue
            step = string(mask, [(mode(f,m,s,a),True),(mode(f,m,s,b),False)])
            if step is not None and step[0] not in present:
                missing.add(step[0])
    return len(missing)

def spin_reverse_sector(sector):
    """Actual CAR implementation of pi frame rotation about y; no spin filter."""
    targets, signs = [], []
    for mask in sector['masks']:
        occupied = [i for i in range(32) if mask & (1 << i)]
        mapped = [i ^ 4 for i in occupied]
        inversions = sum(a > b for j, a in enumerate(mapped) for b in mapped[j+1:])
        signs.append((-1) ** (inversions + sum((i % 8) // 4 == 0 for i in occupied)))
        targets.append(sum(1 << i for i in mapped))
    masks = sorted(targets); lookup = {mask:i for i,mask in enumerate(masks)}
    rotation = sparse.csr_matrix((signs, ([lookup[m] for m in targets], range(len(targets)))),
        shape=(len(masks), len(targets)))
    return dict(masks=masks, P=sparse.kron(rotation,sparse.eye(19),format='csr')@sector['P'])


def path_derivation(p):
    gamma = gamma_matrices(); mass = np.kron(np.diag([1., -1.]), np.eye(2))
    length = p['cycle_length']; angles = np.asarray(p['derivative_angles']); checks = []
    # Actual site shift on each finite periodic cycle; exact normalized constant fibre.
    shift = np.roll(np.eye(length), 1, axis=0); fibre = np.ones(length) / np.sqrt(length)
    for axis, g in enumerate(gamma):
        hopping = g / (2j)
        for q in [2/3, -1/3, -1.]:
            for angle in angles:
                phase = np.exp(1j*q*angle)
                site = np.kron(phase*shift, hopping) + np.kron(phase.conjugate()*shift.T, hopping.conj().T)
                inclusion = np.kron(fibre[:,None], np.eye(4))
                reduced = inclusion.conj().T @ site @ inclusion
                checks.append(float(np.linalg.norm(reduced - np.sin(q*angle)*g)))
    sy = np.array([[0,-1j],[1j,0]],complex); iy = 1j*sy
    time_reversal = np.kron(np.diag([1.,-1.]),iy)
    particle_hole = np.kron(iy,iy)
    symmetry = dict(P_vector=[float(np.linalg.norm(mass@g@mass+g)) for g in gamma],
        T_vector=[float(np.linalg.norm(time_reversal@g.conj()@time_reversal.conj().T+g)) for g in gamma],
        C_vector=[float(np.linalg.norm(-particle_hole@g.T@particle_hole.conj().T+g)) for g in gamma],
        T_square=float(np.linalg.norm(time_reversal@time_reversal.conj()+np.eye(4))))
    return dict(cycle_length=length, zero_fibre_path_defect=max(checks),
        mass_vector_anticommutators=[float(np.linalg.norm(mass@g+g@mass)) for g in gamma],
        chosen_PCT_defects=symmetry,
        vector_commutators=[[float(np.linalg.norm(a@b-b@a)) for b in gamma] for a in gamma],
        dispersion='Single exact zero-Bloch fibre of three periodic nearest-neighbour cycles. T_i=Gamma_i/(2i), forward exp(iqA_i) and adjoint reverse give v sin(qA_i) Gamma_i. Opposite paths cancel at A=0; their derivatives add. Frame-dependent Clifford hopping is an explicit input, not an emergence derivation. Remaining Bloch fibres are not populated or claimed solved.',
        derivative='Hvector(0)=0; J_i=-v q Gamma_i, Kvector_ij(0)=0. The scalar mass grade Z anticommutes every Gamma_i; different currents do not commute.')
