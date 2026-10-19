"""Exact symmetric63 auxiliary <=2-PH native N4 loop PHP, not a ground/pole certificate.

All local four-spin masks and all internal one/two/AP arcs are retained. Omitted
PH, Fourier and external-site sectors are priced separately, never deleted law.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
from itertools import combinations, combinations_with_replacement, permutations
import json
from math import factorial, sqrt
from pathlib import Path
import sys
import time
import zipfile
from typing import Callable, cast

import induced_phi_nonuniform_field_car as native_source
import joined_phi_u1_stationary_photon as photon
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
PREFIX = 'joined_phi_u1_stationary_collective_pairs'
PLAN = dict(id='issue-222-induced-photon-delta13-continuation-20261008',
            captured_revision='10b3c5e3e75de73f67e3e6a20f2a91c75ed63c9e17762fd848f15b8697372ed2',
            assignment_revision='444bae46f65f84b4ee667368135eaf25229de62d786588bc980fc8511e28ce49',
            template='general', template_revision='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
            work_item='induced-photon', requirements=['verification', 'outcome'])
CAPTURED = {**native_source.CAPTURED, Path(photon.__file__).name: Path(photon.__file__).read_bytes(),
            Path(__file__).name: Path(__file__).read_bytes()}
REF = photon.reference()
CAR = photon.car
DENSITY = photon.density


def ph(mask: int) -> int:
    return (mask & ~REF).bit_count()


def local_masks() -> list[int]:
    occupied = [i for i in range(16) if (REF >> i) & 1]
    empty = [i for i in range(16) if not (REF >> i) & 1]
    result = [REF]
    for grade in (1, 2):
        for holes in combinations(occupied, grade):
            for particles in combinations(empty, grade):
                result.append(REF ^ sum(1 << i for i in holes) ^ sum(1 << i for i in particles))
    if [sum(ph(m) == g for m in result) for g in (0, 1, 2)] != [1, 64, 784]:
        raise ArithmeticError('Full16-mode local CAR mask count failed')
    return result


def auxiliary_move(state: tuple[int, ...], b: int, a: int, copies: int) -> tuple[tuple[int, ...], float]:
    counts = Counter(state)
    counts[REF] = copies - len(state)
    if a == b:
        return state, float(counts[b])
    factor = sqrt(counts[b] * (counts[a] + 1))
    result = list(state)
    if b != REF:
        result.remove(b)
    if a != REF:
        result.append(a)
    return tuple(sorted(result)), factor


def matter_basis(masks: list[int]) -> tuple[list[tuple[int, tuple[int, ...]]], np.ndarray, np.ndarray]:
    one = [m for m in masks if ph(m) == 1]
    two = [m for m in masks if ph(m) == 2]
    states = [(REF, ())] + [(m, ()) for m in one] + [(REF, (m,)) for m in one]
    states += [(m, ()) for m in two] + [(REF, (m,)) for m in two]
    states += [(a, (b,)) for a in one for b in one]
    states += [(REF, (a, b)) for a, b in combinations_with_replacement(sorted(one), 2)]
    grades = np.array([ph(a) + sum(ph(b) for b in aux) for a, aux in states], np.int8)
    sectors = np.array([0] + [1]*64 + [2]*64 + [3]*784 + [4]*784 + [5]*4096 + [6]*2080, np.int8)
    if len(states) != 7873 or len(set(states)) != 7873:
        raise ArithmeticError('Complete <=2 collective basis count failed')
    return states, grades, sectors


def car_symmetry_controls(masks: list[int]) -> dict:
    # Product-copy isometry is independently expanded. Even local bilinears
    # commute across copies; each copy always has eight fermions, hence copy
    # permutations carry no odd-fermion sign. No Slater/Wick factorization.
    one = [m for m in masks if ph(m) == 1]
    two = [m for m in masks if ph(m) == 2]
    states = [()] + [(m,) for m in one + two] + list(combinations_with_replacement(sorted(one), 2))
    # Every offsite local mode pair, in both directions, tests all repeated
    # labels and same-copy2PH states, not merely vacuum sqrt(F).
    operators = [(c, a) for c in range(16) for a in range(16) if c//4 != a//4]
    rows = []
    for copies in (2, 3):
        maximum = 0.; wrong_maximum = 0.; reached = 0; repeated_reached = 0
        for state in states:
            words = set(permutations((REF,)*(copies-len(state)) + state))
            normalizer = sqrt(len(words))
            for create, annihilate in operators:
                actual = {}
                for word in words:
                    for j, b in enumerate(word):
                        transition = CAR(b, create, annihilate)
                        if transition is None:
                            continue
                        a, sign = transition
                        changed = word[:j] + (a,) + word[j+1:]
                        actual[changed] = actual.get(changed, 0.) + sign/normalizer
                predicted = {}
                for b in set(state) | {REF}:
                    if b == REF and len(state) == copies:
                        continue
                    transition = CAR(b, create, annihilate)
                    if transition is None:
                        continue
                    a, sign = transition
                    new_state, factor = auxiliary_move(state, b, a, copies)
                    new_words = set(permutations((REF,)*(copies-len(new_state)) + new_state))
                    for word in new_words:
                        predicted[word] = predicted.get(word, 0.) + sign*factor/sqrt(len(new_words))
                    reached += 1
                    if ((len(state) == 2 and state[0] == state[1]) or
                            (len(new_state) == 2 and new_state[0] == new_state[1])):
                        repeated_reached += 1
                        # WRONG independent identical-label creation would omit
                        # the n_a+1 factor. It must disagree on real CAR words.
                        wrong = sqrt(Counter(state)[b] if b != REF else copies-len(state))
                        wrong_maximum = max(wrong_maximum, abs(factor-wrong))
                keys = actual.keys() | predicted.keys()
                defect = max((abs(actual.get(k, 0.)-predicted.get(k, 0.)) for k in keys), default=0.)
                maximum = max(maximum, defect)
        if maximum > 2e-14 or wrong_maximum < .1 or repeated_reached == 0:
            raise ArithmeticError('F2/F3 explicit CAR symmetry/repeated-label wrong assertion failed')
        rows.append(dict(copies=copies, symmetric_states=len(states), operators=len(operators),
                         local_transitions_reached=reached, repeated_label_transitions_reached=repeated_reached,
                         product_isometry_maximum_defect=maximum, wrong_missing_occupancy_factor_defect=wrong_maximum))
    return dict(derivation='Normalized orbit sum |n>=sqrt(prod n! / F!) sum distinct copy words. Sum_f |a><b|_f has coefficient sqrt(n_b(n_a+1)) for a!=b and n_b for a=b, times local16-mode CAR sign. Vacuum n0=63-number excited copies, not an effective flavor. Local2PH and two copies1PH are distinct orthogonal states.',
                exact_combinatorial_identity=True, floating_explicit_F2_F3_controls=rows, no_Wick=True)


def geometry(data: dict) -> dict:
    native = data['native']
    positions = np.array([(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)])
    vertices = [int(np.flatnonzero(np.all(native['sites'] == x, axis=1))[0]) for x in positions]
    index = {v: j for j, v in enumerate(vertices)}
    species = []; all_paths = []
    for profile in data['profiles'][1:]:
        onsite, rotation = np.linalg.eigh(float(profile['onsite'])*native['frame'][0])
        paths = native_source.native_paths(profile, native)
        internal = [p for p in paths if p['source'] in index and p['target'] in index]
        if len(internal) != 6:
            raise ArithmeticError('All six1/2-link internal native arcs required')
        hops = []
        for p in internal:
            h = p['seam']*np.einsum('m,mij->ij', p['coefficients'], native['frame'])
            hops.append(rotation.conj().T @ h @ rotation)
        species.append(dict(profile=profile, onsite=onsite, rotation=rotation, hops=hops))
        all_paths.append(paths)
    paths = [p for p in all_paths[0] if p['source'] in index and p['target'] in index]
    B = np.zeros((192, len(paths)), int); incidence = np.zeros((4, len(paths)), int)
    for e, p in enumerate(paths):
        B[list(p['edges']), e] = 1
        incidence[index[p['source']], e] = 1; incidence[index[p['target']], e] = -1
    embedding = np.zeros((64, 4), int); embedding[vertices, np.arange(4)] = 1
    if np.max(np.abs(native['D'] @ B - embedding @ incidence)) > 1e-12:
        raise ArithmeticError('Actual primitive path Dinc=rho assertion failed')
    tree = []; visited = {0}
    while len(visited) < 4:
        for e, p in enumerate(paths):
            x, y = index[p['source']], index[p['target']]
            if (x in visited) != (y in visited):
                tree.append(e); visited.update((x, y)); break
        else:
            raise ArithmeticError('Internal graph is disconnected')
    chords = [e for e in range(6) if e not in tree]
    T = np.zeros((6, 4), int)
    T[tree, :3] = np.rint(np.linalg.inv(incidence[:3, tree])).astype(int)
    cycles = np.zeros((6, 3), int)
    for j, e in enumerate(chords):
        cycles[e, j] = 1; cycles[:, j] -= T @ incidence[:, e]
    if not np.array_equal(incidence @ cycles, np.zeros((4, 3), int)):
        raise ArithmeticError('Integer primitive graph cycle assertion failed')
    modes = B.T @ native['Q']
    windings = np.array([(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0),
                         (0, -1, 0), (0, 0, 1), (0, 0, -1)], int)
    fields = []
    for qi in ((0, 0, 0), (1, 0, 0), (2, 0, 0)):
        q = 2*np.pi*np.array(qi)/4
        f = np.array([np.exp(1j*native['sites'][e//3] @ q)/8 if e % 3 == 1 else 0j for e in range(192)])
        fields.append(B.T @ f)
    paths_metadata = [dict(source=int(p['source']), target=int(p['target']), axis=int(p['axis']),
                           edges=list(map(int, p['edges'])), seam=int(p['seam']),
                           displacement=list(map(int, p['displacement'])),
                           coefficients=[str(c) for c in p['coefficients']]) for p in all_paths[0]]
    outside_norm = 0.
    for item, native_paths in zip(species, all_paths, strict=True):
        for p in native_paths:
            if p['source'] in index and p['target'] in index:
                continue
            hop = np.einsum('m,mij->ij', p['coefficients'], native['frame'])
            # ||c†_x h c_y + hc|| <=2 sum singular values(h), per copy;
            # characters are unitary, so this is uniform over all incoming Pi.
            outside_norm += int(item['profile']['load'])*2*float(np.linalg.svd(hop, compute_uv=False).sum())
    return dict(vertices=vertices, index=index, species=species, paths=paths, all_paths=paths_metadata,
                B=B, incidence=incidence, embedding=embedding, tree=tree, chords=chords, T=T,
                cycles=cycles, modes=modes, windings=windings, fields=np.array(fields),
                outside_norm_upper_float=outside_norm)


def local_transitions(item: dict, graph: dict, masks: list[int]) -> list[dict[int, list[tuple[int, complex]]]]:
    result = []
    for e, p in enumerate(graph['paths']):
        x, y = graph['index'][p['source']], graph['index'][p['target']]
        h = item['hops'][e]
        for direction in (1, -1):
            op = []
            for i in range(4):
                for j in range(4):
                    coefficient = complex(h[i, j])
                    if abs(coefficient) < 1e-13:
                        continue
                    c, a = 4*x+i, 4*y+j
                    if direction == -1:
                        c, a = a, c; coefficient = coefficient.conjugate()
                    op.append((c, a, coefficient))
            lookup = {}
            for b in masks:
                moves = {}
                for c, a, coefficient in op:
                    transition = CAR(b, c, a)
                    if transition is not None:
                        new_mask, sign = transition
                        moves[new_mask] = moves.get(new_mask, 0j) + sign*coefficient
                lookup[b] = [(a, v) for a, v in moves.items() if abs(v) > 1e-13]
            result.append(lookup)
    return result


def assemble(data: dict, graph: dict, masks: list[int]) -> dict:
    states, grades, sectors = matter_basis(masks); n = len(states); nw = len(graph['windings'])
    lookup = {s: j for j, s in enumerate(states)}
    dlocal = {m: DENSITY(m) for m in masks}
    rho = np.array([dlocal[a] + sum((dlocal[b] for b in aux), np.zeros(4, int)) for a, aux in states])
    # Include all initial filled copies and all spectator sites BEFORE HC.
    total_initial = np.full((n, 64), 128, int)
    for j, (a, aux) in enumerate(states):
        for x, v in enumerate(graph['vertices']):
            active_count = ((a >> (4*x)) & 15).bit_count()
            auxiliary_count = sum(((b >> (4*x)) & 15).bit_count() for b in aux)
            total_initial[j, v] = active_count + auxiliary_count + 2*(63-len(aux))
    charge64 = total_initial - 128
    if not np.array_equal(charge64, rho @ graph['embedding'].T):
        raise ArithmeticError('Full64 initial Ref4 density assertion failed')
    L = data['native']['Lplus']
    hc = 64*np.einsum('ni,ij,nj->n', charge64, L, charge64)
    wrong_hc = np.array([64*(dlocal[a] @ L[np.ix_(graph['vertices'], graph['vertices'])] @ dlocal[a] +
                                  sum(dlocal[b] @ L[np.ix_(graph['vertices'], graph['vertices'])] @ dlocal[b] for b in aux))
                         for a, aux in states])
    onsite = []
    for item in graph['species']:
        onsite.append({m: sum(item['onsite'][s]*(((m >> (4*x+s)) & 1)-int(s < 2))
                             for x in range(4) for s in range(4)) for m in masks})
    on = np.array([onsite[0][a]+sum(onsite[1][b] for b in aux) for a, aux in states])
    lift = rho @ graph['T'].T
    if not np.array_equal(lift @ graph['incidence'].T, rho):
        raise ArithmeticError('Aggregate-charge integer tree lift assertion failed')
    k = lift[:, None, :] + (graph['windings'] @ graph['cycles'].T)[None, :, :]
    p = k @ graph['modes']; he = 64*np.sum(p*p, axis=2)
    diagonal = (on[:, None]+hc[:, None]+he).reshape(-1)
    local = [local_transitions(item, graph, masks) for item in graph['species']]
    rows = []; columns = []; values = []; primitive = []
    b_rows = []; b_columns = []; b_values = []; boundary = {}; boundary_states = []; boundary_kind = []
    primitive_counts = np.zeros((6, 2, 2), int)
    winding_lookup = {tuple(w): j for j, w in enumerate(graph['windings'])}
    for col, (active, aux) in enumerate(states):
        for e in range(6):
            for direction in (1, -1):
                op = 2*e + int(direction == -1)
                movements = [(0, (a, aux), coefficient) for a, coefficient in local[0][op][active]]
                counts = Counter(aux); counts[REF] = 63-len(aux)
                for b in counts:
                    if counts[b] == 0:
                        continue
                    for a, coefficient in local[1][op][b]:
                        changed, factor = auxiliary_move(aux, b, a, 63)
                        movements.append((1, (active, changed), coefficient*factor))
                for species, new_state, coefficient in movements:
                    dest = lookup.get(new_state)
                    newrho = DENSITY(new_state[0]) + sum((DENSITY(b) for b in new_state[1]), np.zeros(4, int))
                    newlift = graph['T'] @ newrho
                    delta = lift[col].copy(); delta[e] += direction; delta -= newlift
                    dw = delta[graph['chords']]
                    if not np.array_equal(delta, graph['cycles'] @ dw):
                        raise ArithmeticError('Incoming path lift differs by noninteger cycle')
                    for wi, winding in enumerate(graph['windings']):
                        neww = tuple(winding+dw); wj = winding_lookup.get(neww)
                        cindex = col*nw+wi
                        if dest is not None and wj is not None:
                            rows.append(dest*nw+wj); columns.append(cindex); values.append(coefficient)
                            primitive.append((e, direction)); primitive_counts[e, int(direction == -1), species] += 1
                        else:
                            key = (new_state, neww)
                            bi = boundary.get(key)
                            if bi is None:
                                bi = len(boundary); boundary[key] = bi
                                boundary_states.append((new_state, neww))
                                boundary_kind.append(int(dest is None))
                            b_rows.append(bi); b_columns.append(cindex); b_values.append(coefficient)
    shape = (n*nw, n*nw)
    rows = np.array(rows); columns = np.array(columns); values = np.array(values); primitive = np.array(primitive)
    hop = coo_matrix((values, (rows, columns)), shape=shape).tocsr()
    H = hop + diags(diagonal)
    B = coo_matrix((b_values, (b_rows, b_columns)), shape=(len(boundary), n*nw)).tocsr()
    defect = H-H.conj().T
    hermitian = float(np.max(abs(defect.data), initial=0.))
    if hermitian > 2e-10:
        raise ArithmeticError('Collective PHP Hermiticity failed')
    currents = []; contacts = []
    for field in graph['fields']:
        vertex = primitive[:, 1]*field[primitive[:, 0]]
        currents.append(coo_matrix((1j*vertex*values/64, (rows, columns)), shape=shape).tocsr())
        contacts.append(coo_matrix((-abs(vertex)**2*values/64**2, (rows, columns)), shape=shape).tocsr())
    wilson = []
    for chord in range(3):
        r = []; c = []
        for wi, w in enumerate(graph['windings']):
            shifted = w.copy(); shifted[chord] += 1
            wj = winding_lookup.get(tuple(shifted))
            if wj is not None:
                c.extend((np.arange(n)*nw+wi).tolist()); r.extend((np.arange(n)*nw+wj).tolist())
        wilson.append(coo_matrix((np.ones(len(r)), (r, c)), shape=shape).tocsr())
    return dict(H=H, B=B, states=states, grades=grades, sectors=sectors, rho64=charge64,
                hc=hc, hc_wrong=wrong_hc, cross=hc-wrong_hc, onsite=on, k=k, p=p, he=he,
                currents=currents, contacts=contacts, wilson=wilson, boundary_states=boundary_states,
                boundary_kind=np.array(boundary_kind), hermitian=hermitian,
                primitive_counts=primitive_counts, hop=hop)


def covariance_controls(matrix: dict, graph: dict) -> dict:
    states = matrix['states']; one = [m for m in local_masks() if ph(m) == 1]
    pair = next((a, b) for a in one for b in one if np.any(DENSITY(a)) and np.array_equal(DENSITY(a), -DENSITY(b)))
    j = states.index((pair[0], (pair[1],)))
    same_aux = states.index((REF, tuple(sorted(pair))))
    correct = matrix['hc'][j]; wrong = matrix['hc_wrong'][j]
    if abs(correct) > 1e-12 or wrong < 1 or np.linalg.norm(matrix['p'][j, 0]) > 1e-12:
        raise ArithmeticError('Opposite active/aux pair density/Pi cancellation assertion failed')
    if abs(matrix['hc'][same_aux]) > 1e-12 or matrix['hc_wrong'][same_aux] < 1:
        raise ArithmeticError('Opposite distinct auxiliary pair cancellation assertion failed')
    repeating = next(i for i, (_, aux) in enumerate(states) if len(aux) == 2 and aux[0] == aux[1] and np.any(DENSITY(aux[0])))
    if abs(matrix['hc'][repeating]-2*matrix['hc_wrong'][repeating]) > 1e-10:
        raise ArithmeticError('Repeated same-label density-square covariance assertion failed')
    # Wrong independent electric path selfprices fail against the actual
    # incoming aggregate lift + primitive character, including cancellation.
    e = graph['chords'][0]; w = graph['windings'][1] @ graph['cycles'].T
    incoming = w @ graph['modes']; step = graph['modes'][e]
    electric_cross = 64*(np.dot(incoming+step, incoming+step)-np.dot(incoming, incoming)-np.dot(step, step))
    if abs(electric_cross) < 1e-7:
        raise ArithmeticError('Incoming Pi cross-term control was blind')
    return dict(opposite_active_aux_basis_index=j, opposite_aux_aux_basis_index=same_aux,
                exact_aggregate_rho_zero=True, actual_HC=correct, wrong_removed_cross_HC=wrong,
                repeated_aux_basis_index=repeating, repeated_actual_HC=float(matrix['hc'][repeating]),
                repeated_wrong_independent_HC=float(matrix['hc_wrong'][repeating]),
                incoming_electric_cross=electric_cross, assertions_reached=True)


def lanczos_diagnostic(H, energy: float, vector: np.ndarray, source: np.ndarray, steps: int = 32) -> dict:
    source = source-vector*np.vdot(vector, source); weight = float(np.vdot(source, source).real)
    if weight < 1e-25:
        return dict(weight=weight, steps=0, status='No connected source weight')
    q = source/sqrt(weight); previous = np.zeros_like(q); beta = 0.; alphas = []; betas = []; stored = []
    tail = 0.
    for _ in range(steps):
        stored.append(q)
        r = H@q-energy*q-beta*previous
        alpha = float(np.vdot(q, r).real); r -= alpha*q
        for v in stored:
            r -= v*np.vdot(v, r)
        tail = float(np.linalg.norm(r)); alphas.append(alpha)
        if tail < 1e-12:
            break
        betas.append(tail); previous, q = q, r/tail; beta = tail
    t = np.diag(alphas)
    if len(alphas) > 1:
        b = np.array(betas[:len(alphas)-1]); t += np.diag(b, 1)+np.diag(b, -1)
    nodes, rotation = np.linalg.eigh(t); weights = weight*abs(rotation[0])**2
    return dict(weight=weight, steps=len(alphas), terminal_Lanczos_beta=tail,
                returned_nodes=nodes.tolist(), returned_weights=weights.tolist(),
                imaginary_frequency_quadrature=[dict(nu=nu, real=float(np.sum(weights*nodes/(nodes**2+nu**2))))
                                                for nu in (1., 10., 100.)],
                status='Finite Krylov diagnostic with terminal residual reported. No spectral-tail/lower bound: nodes are NOT poles, static susceptibility not certified.')


def trial(H, B, initial: np.ndarray, indices: np.ndarray, matrix: dict) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    h = H[indices][:, indices]; b = B[:, indices]
    vals, vecs = cast(Callable[..., tuple[np.ndarray, np.ndarray]], eigsh)(h, k=3, which='SA', tol=2e-10, maxiter=12000, v0=initial)
    order = np.argsort(vals); vals = vals[order]; vecs = vecs[:, order]
    v = vecs[:, 0]; energy = float(np.vdot(v, h@v).real)
    residual = float(np.linalg.norm(h@v-energy*v))
    full = np.zeros(H.shape[0], complex); full[indices] = v
    enlargement_residual = float(np.linalg.norm(H@full-energy*full))
    escaped = B@full
    fourier = float(np.linalg.norm(escaped[matrix['boundary_kind'] == 0]))
    higher_ph = float(np.linalg.norm(escaped[matrix['boundary_kind'] == 1]))
    return dict(dimension=len(indices), Ritz_energy_shift=energy,
                returned_Ritz_values=vals.tolist(), returned_Ritz_spacings=(vals[1:]-vals[0]).tolist(),
                PHP_Ritz_residual=residual, residual_in_common_le2_PHP=enlargement_residual,
                exact_internal_omitted_Fourier_residual=fourier, exact_internal_omitted_PH_residual=higher_ph,
                internal_native_cluster_residual=sqrt(enlargement_residual**2+fourier**2+higher_ph**2),
                status='Sparse returned Ritz states/spacings only, not certified compressed lowest spectrum/ground/gap.'), full, vals, vecs


def run(output: Path) -> dict:
    started = time.monotonic(); masks = local_masks()
    controls = car_symmetry_controls(masks)
    data = native_source.load_mean(g_override=[Fraction('0.28917434559291342'), *[Fraction('0.10475219290309079')]*3],
                                   mean_descriptor={'law':'Actual captured N4 root display; no root rerun, no source/state adoption'})
    native = data['native']
    if native['Q'].shape != (192, 129) or native['Lplus'].shape != (64, 64):
        raise ArithmeticError('Actual full129 electric/full64 Coulomb law required')
    graph = geometry(data); matrix = assemble(data, graph, masks); controls['covariance'] = covariance_controls(matrix, graph)
    H = matrix['H']; B = matrix['B']; nw = len(graph['windings'])
    lower_indices = np.flatnonzero(np.repeat(matrix['grades'] <= 1, nw)); upper_indices = np.arange(H.shape[0])
    initial = np.zeros(len(lower_indices)); initial[0] = 1
    low, lowvector, lowvals, lowvecs = trial(H, B, initial, lower_indices, matrix)
    high, highvector, highvals, highvecs = trial(H, B, lowvector, upper_indices, matrix)
    improvement = low['Ritz_energy_shift']-high['Ritz_energy_shift']
    pairweight = float(np.sum(abs(highvector[np.repeat(matrix['grades'] == 2, nw)])**2))
    if improvement < 1e-5 or pairweight < 1e-8:
        raise ArithmeticError('Simultaneous-pair contribution assertion failed')
    results = []
    for label, report, vector in [('le1', low, lowvector), ('le2', high, highvector)]:
        prob = np.sum(abs(vector.reshape(-1, nw))**2, axis=1)
        report.update(PH_sector_weights={name:float(prob[matrix['sectors'] == j].sum()) for j, name in enumerate(
            ['reference','active1','auxiliary1','active2','auxiliary_samecopy2','active1_aux1','auxiliary_distinctcopy1pairs'])},
                      HC_expectation=float(prob@matrix['hc']), HC_total_cross_expectation=float(prob@matrix['cross']),
                      wrong_independent_density_HC_expectation=float(prob@matrix['hc_wrong']),
                      HE_expectation=float(np.sum(abs(vector.reshape(-1,nw))**2*matrix['he'])),
                      rho_mean64=(prob@matrix['rho64']).tolist(),
                      full_body_residual_upper_float=report['internal_native_cluster_residual']+graph['outside_norm_upper_float'])
        diagnostics = []
        for qi, J, C in zip(((0,0,0),(1,0,0),(2,0,0)), matrix['currents'], matrix['contacts'], strict=True):
            # Same trial law, but each projected observable follows its own
            # <=1 or <=2 projector. Retained two channels and mixed contact.
            if label == 'le1':
                index = lower_indices
                jj = J[index][:,index]; cc=C[index][:,index]; hh=H[index][:,index]; vv=vector[index]
            else:
                jj=J;cc=C;hh=H;vv=vector
            plus=lanczos_diagnostic(hh,report['Ritz_energy_shift'],vv,jj@vv)
            minus=lanczos_diagnostic(hh,report['Ritz_energy_shift'],vv,jj.conj().T@vv)
            diagnostics.append(dict(spatial_q_integer=qi, spatial_q=[2*np.pi*x/4 for x in qi],
                                    physical_current_plus=plus, physical_current_minus=minus,
                                    physical_contact_real=float(np.vdot(vv,cc@vv).real)))
        report['physical_current_diagnostics']=diagnostics
        report['Wilson_diagnostics']=[]
        for j, W in enumerate(matrix['wilson']):
            if label == 'le1':
                ww=W[lower_indices][:,lower_indices]; hh=H[lower_indices][:,lower_indices];vv=vector[lower_indices]
            else:
                ww=W;hh=H;vv=vector
            report['Wilson_diagnostics'].append(dict(cycle=j, expectation_real=float(np.vdot(vv,ww@vv).real),
                plus=lanczos_diagnostic(hh,report['Ritz_energy_shift'],vv,ww@vv),
                minus=lanczos_diagnostic(hh,report['Ritz_energy_shift'],vv,ww.conj().T@vv)))
        results.append(report)
    encoded_states=np.array([(a, *(list(aux)+[REF]*(2-len(aux)))) for a,aux in matrix['states']],np.int32)
    boundary_masks=np.array([(state[0],*(list(state[1])+[REF]*(3-len(state[1])))) for state,w in matrix['boundary_states']],np.int32)
    arrays=dict(H_data=H.data,H_indices=H.indices,H_indptr=H.indptr,H_shape=np.array(H.shape),
                omitted_internal_data=B.data,omitted_internal_indices=B.indices,omitted_internal_indptr=B.indptr,
                omitted_internal_shape=np.array(B.shape),omitted_internal_masks=boundary_masks,
                omitted_internal_windings=np.array([w for state,w in matrix['boundary_states']]),
                omitted_internal_kind=matrix['boundary_kind'],matter_masks=encoded_states,
                matter_PH=matrix['grades'],matter_sector=matrix['sectors'],windings=graph['windings'],
                rho64=matrix['rho64'],HC=matrix['hc'],HC_wrong_no_cross=matrix['hc_wrong'],HE=matrix['he'],
                onsite_shift=matrix['onsite'],integer_graph_flux=matrix['k'],full129_momentum=matrix['p'],
                Q=native['Q'],Lplus=native['Lplus'],D=native['D'],primitive_B=graph['B'],
                graph_incidence=graph['incidence'],tree_lift=graph['T'],integer_cycles=graph['cycles'],
                vertices=np.array(graph['vertices']),tree=np.array(graph['tree']),chords=np.array(graph['chords']),
                native_path_characters=graph['modes'],external_probe_path_fields=graph['fields'],
                primitive_transition_counts=matrix['primitive_counts'],le1_indices=lower_indices,
                le1_trial=lowvector,le2_trial=highvector,le1_returned_values=lowvals,le2_returned_values=highvals,
                le1_returned_vectors=lowvecs,le2_returned_vectors=highvecs)
    for kind, objects in [('physical_current',matrix['currents']),('physical_contact',matrix['contacts']),('Wilson',matrix['wilson'])]:
        for j,obj in enumerate(objects):
            for suffix in ('data','indices','indptr'):
                arrays[f'{kind}_{j}_{suffix}']=getattr(obj,suffix)
    npz=output.with_suffix('.npz');cast(Callable[..., None], np.savez_compressed)(npz,**arrays)
    inputs={Path(data['descriptor'][key]['path']).name:(ROOT/Path(data['descriptor'][key]['path']).name).read_bytes()
            for key in ('root_receipt','full_receipt')}
    sources={name:dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),unchanged=(ROOT/name).read_bytes()==raw)
             for name,raw in CAPTURED.items()}
    if not all(row['unchanged'] for row in sources.values()):
        raise RuntimeError('Native/source captured epoch changed')
    charged_reference=sum(int(item['profile']['load'])*64*float(item['onsite'][:2].sum()) for item in graph['species'])
    return dict(plan=PLAN,law=dict(N=4,V=64,a='1/64',HE=64,HC=64,diffusion=128,Ref=4,
                charged_loads=[1,63],neutral_load=1,charged_physical_scale=1,full_EM_dimensions=129,
                full_density_sites=64,Phi_colour_stiffness='Lambda^(5/4) NOT charged scale',
                fixed_mean=data['descriptor'],bare_magnetic_term=False),
                derivation=dict(reference='All64 sites/all64 charged copies fill negativeGamma0 rank2. Ref4 subtracts128/site. Neutral heavy and fullPhi/colour stay spectator in this fixed-body trial, not actual full ground.',
                Hilbert_space='Configuration beta0 Floquet fiber, NOT ordinary real-cover L2 vacuum. Actual native integer path Fourier characters p=Q^T B k. No spatial-q/configuration-beta identification.',
                occupation=controls['derivation'],Coulomb='HC=64 rho_total^T full64Lplus rho_total, with rho_total=active+sum63aux-initialRef4 including all filled spectators BEFORE squaring. Cross terms retained; no Wick.',
                electric='HE=64||sum incoming native path characters||^2 in all129 coordinates. Tree T depends aggregate rho; k=T rho+Cw. Every applied path shifts k by its exact directed unit column; integer cycle difference checked.',
                physical_units='Literal exp(iTheta)=exp(i a Aphysical), a1/64: Jphysical=aJTheta, contactphysical=a^2contactTheta. Probe f_e=exp(iq.site)/8 on y links. Both Jq/Jq† channels, no bounded fullPhi physical-current claim.',
                auxiliary_local_dimension=849,all_matter_states_le2=7873,Fourier='Seven L1<=1 windings in all3 internal graph cycles; other windings explicitly in omitted_internal matrix.',
                matrix_order='row=matter_index*7+winding_index. Auxiliary mask REF padding is vacuum; up to2 excited copies retained,63 minus count remain reference. Sector labels0..6 listed in trial weights.',
                exactness='Combinatorial CAR/permutation and integer-lift identities exact; concrete arrays/eigsh/Lanczos use floating arithmetic at captured root display, not interval spectrum certificate.'),
                controls=controls,basis=dict(dimension=H.shape[0],matrix_nnz=H.nnz,omitted_internal_rows=B.shape[0],
                omitted_internal_nnz=B.nnz,hermiticity_defect=matrix['hermitian'],vertices=graph['vertices'],
                internal_path_indices=[j for j,p in enumerate(graph['all_paths']) if p['source'] in graph['index'] and p['target'] in graph['index']],
                native_paths_all=graph['all_paths'],all_native_paths_per_species=len(graph['all_paths']),
                omitted_external_paths_per_species=len(graph['all_paths'])-6),
                omitted_sectors=dict(internal='All reached totalPH>2 and Fourier states outside L1<=1 are orthogonal external rows; amplitudes/cancellations summed before residual squaring. Serialized B=(1-P)Hinternal P.',
                outside_vertices='Every other one/two/AP native path is still in the full law. Their action leaves this cluster PHP. Uniform triangle upper price2 sum_s load_s sum_outside nuclear_norm(h_path); reported full-body residual upper is finite floating evaluation, deliberately broad, not a certified spectrum tail.',
                outside_native_operator_norm_upper_float=graph['outside_norm_upper_float'],
                fullPhi_colour_neutral='FullPhi/Haar and neutral dynamic sectors not trial-characterized. No claim this finite-body trial supplies whole primitive ground.'),
                trials=results,ground_directed_change=dict(energy_improvement=improvement,pair_probability=pairweight,
                total_charged_reference_energy=charged_reference,
                le1_charged_trial_energy=charged_reference+low['Ritz_energy_shift'],
                le2_charged_trial_energy=charged_reference+high['Ritz_energy_shift'],
                decision='The simultaneous-pair sector lowers the SAME-law trial and has nonzero probability. Inspect explicit internal leakage and external broad residual price before enlargement; no sparse-returned spacing is promoted to physical gap/pole.'),
                parent_arrays=dict(path=npz.name,sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),
                format='NPZ CSR arrays H and B; basis/rho/129momentum/full64Lplus/path/lift/trials; physical J/C and3 Wilson CSR share H shape. No pickle.'),
                sources=sources,inputs={name:dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)) for name,raw in inputs.items()},
                runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),
                conclusion=dict(ground=False,gap_certificate=False,photon_pole=False,full_Phi=False,
                status='Finite new collective-pair ground-directed trial completed; full stationary/Phi ground and same-ground photon dispersion/pole remain OPEN.'),
                elapsed_seconds=time.monotonic()-started)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        result=run(args.output)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    archive=args.output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
        for name,raw in CAPTURED.items():bundle.writestr(name,raw)
        for name in result['inputs']:bundle.write(ROOT/name,name)
        bundle.write(args.output,args.output.name);bundle.write(args.output.with_suffix('.npz'),args.output.with_suffix('.npz').name)
    print(json.dumps(dict(receipt=str(args.output),archive=str(archive),arrays=result['parent_arrays'],
        trials=[dict(dimension=r['dimension'],energy=r['Ritz_energy_shift'],residual=r['PHP_Ritz_residual']) for r in result['trials']],
        simultaneous_pair=result['ground_directed_change'],elapsed_seconds=result['elapsed_seconds']),indent=2))


if __name__=='__main__':main()
