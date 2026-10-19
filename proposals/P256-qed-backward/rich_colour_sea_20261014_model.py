"""One NEW CAR24 sea, scalar mirror route and actual PW19 colour loop.
The pair cut is variational, not the full fixed-number or all-number vacuum.
"""
from __future__ import annotations
from itertools import combinations, product
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
from rich_colour_composite_20261014_model import generators, loop_field, epsilon


def mode(flavour, mirror, spin, colour):
    return 12 * flavour + 6 * mirror + 3 * spin + colour


def apply(mask, index, create):
    occupied = bool(mask & (1 << index))
    if occupied == create:
        return None
    return mask ^ (1 << index), (-1) ** ((mask & ((1 << index) - 1)).bit_count())


def string(mask, operations):
    amplitude = 1
    for index, create in reversed(operations):
        step = apply(mask, index, create)
        if step is None:
            return None
        mask, sign = step
        amplitude *= sign
    return mask, amplitude


def charge_conjugate(mask):
    """C c_i^dagger C^-1 = (i tau_y x i sigma_y)_ij c_j."""
    target = (1 << 24) - 1
    amplitude = 1
    for index in reversed([i for i in range(24) if mask & (1 << i)]):
        f, remainder = divmod(index, 12)
        mirror, remainder = divmod(remainder, 6)
        spin, colour = divmod(remainder, 3)
        other = mode(f, 1 - mirror, 1 - spin, colour)
        target, sign = apply(target, other, False)
        amplitude *= sign * (1 if mirror == spin else -1)
    return target, amplitude


def ordered_frame():
    """Hypothesis: primitive quaternion frame and TWO ordered mirror paths."""
    f = [np.array([[0, 1], [1, 0]], complex),
         np.array([[0, -1j], [1j, 0]], complex), np.diag([1., -1.])]
    exchange = np.array([[0, 1], [1, 0]], complex)
    oriented = np.array([[0, -1j], [1j, 0]], complex)
    scalar = np.kron(exchange, np.eye(2))
    vectors = [np.kron(oriented, x) for x in f]
    spin = [np.kron(np.eye(2), x / 2) for x in f]
    cliff = [scalar] + vectors
    defect = max(np.linalg.norm(a @ b + b @ a - (2 * np.eye(4) if i == j else 0))
                 for i, a in enumerate(cliff) for j, b in enumerate(cliff))
    equations = np.vstack([np.kron(np.eye(4), a) - np.kron(a.T, np.eye(4)) for a in cliff])
    irreducible_commutant = 16 - int(np.linalg.matrix_rank(equations))
    intertwiners = [np.kron(np.eye(4), np.eye(3)[:, a:a+1] @ np.eye(3)[b:b+1, :])
                    for a, b in product(range(3), repeat=2)]
    intertwiner_defect = max(np.linalg.norm(t @ np.kron(a, np.eye(3)) - np.kron(a, np.eye(3)) @ t)
                            for t in intertwiners for a in cliff)
    return dict(scalar=scalar, vectors=vectors, spin=spin,
                clifford_defect=float(defect), multiplicity=3, module_dimension=12,
                irreducible_commutant_dimension=irreducible_commutant,
                repeated_commutant_dimension=9 * irreducible_commutant, intertwiner_defect=float(intertwiner_defect),
                derivation='Ordered mirror exchange/oriented paths tensor primitive quaternion frame; no free Dirac kinetic operator. Multiplicity3 and gauging SU3 are inputs.')


def masks_for(delta, twice_m, pair_cut):
    negative = [[mode(f, 1, s, c) for s, c in product(range(2), range(3))] for f in range(2)]
    positive = [[mode(f, 0, s, c) for s, c in product(range(2), range(3))] for f in range(2)]
    sea = sum(1 << i for indices in negative for i in indices)
    out = []
    for hu in range(pair_cut + 1):
        for hd in range(pair_cut + 1 - hu):
            pu, pd = hu + delta[0], hd + delta[1]
            if not (0 <= pu <= 6 and 0 <= pd <= 6):
                continue
            for holes_u, holes_d, particles_u, particles_d in product(
                    combinations(negative[0], hu), combinations(negative[1], hd),
                    combinations(positive[0], pu), combinations(positive[1], pd)):
                mask = sea
                for i in holes_u + holes_d + particles_u + particles_d:
                    mask ^= 1 << i
                mz = sum(1 if (i % 6) // 3 == 0 else -1 for i in range(24) if mask & (1 << i))
                if mz == twice_m:
                    out.append(mask)
    return sorted(out)


def one_body(masks, terms):
    lookup = {mask: i for i, mask in enumerate(masks)}
    rows, cols, values = [], [], []
    for column, mask in enumerate(masks):
        for a, b, value in terms:
            step = string(mask, [(a, True), (b, False)])
            if step is not None and step[0] in lookup:
                rows.append(lookup[step[0]]); cols.append(column); values.append(value * step[1])
    return sparse.csr_matrix((values, (rows, cols)), shape=(len(masks), len(masks)), dtype=complex)


def conjugation(source_masks, target_masks, loop):
    lookup = {mask: i for i, mask in enumerate(target_masks)}
    rows, values = [], []
    for mask in source_masks:
        target, value = charge_conjugate(mask)
        rows.append(lookup[target]); values.append(value)
    matter = sparse.csr_matrix((values, (rows, range(len(rows)))), shape=(len(target_masks), len(source_masks)))
    swap = np.eye(19)
    swap[1:10, 1:10] = 0; swap[10:, 10:] = 0
    swap[1:10, 10:] = np.eye(9); swap[10:, 1:10] = np.eye(9)
    return sparse.kron(matter, swap, format='csr')


def gauge_sector(masks, loop):
    n = len(masks); identity = sparse.eye(n, format='csr'); ts = generators()
    matter = []
    for t in ts:
        terms = [(mode(f, m, s, a), mode(f, m, s, b), t[a, b])
                 for f, m, s, a, b in product(range(2), range(2), range(2), range(3), range(3)) if t[a, b] != 0]
        matter.append(one_body(masks, terms))
    gauss = [sparse.kron(q, sparse.eye(19), format='csr') + sparse.kron(identity, loop['L'][a] + loop['R'][a], format='csr') for a, q in enumerate(matter)]
    allowed = np.flatnonzero((abs(gauss[6].diagonal()) < 1e-9) & (abs(gauss[7].diagonal()) < 1e-9))
    casimir = sum(g[:, allowed].conj().T @ g[:, allowed] for g in gauss).tocsr()
    graph = casimir.copy(); graph.data = np.where(abs(graph.data) > 1e-10, 1., 0.); graph.eliminate_zeros()
    blocks, labels = connected_components(graph, directed=False)
    rows, cols, values = [], [], []; count = 0; minimum_nonzero = np.inf
    for block in range(blocks):
        indices = np.flatnonzero(labels == block)
        eigenvalues, vectors = eigh(casimir[indices][:, indices].toarray())
        nonzero = eigenvalues[eigenvalues > 1e-8]
        if len(nonzero): minimum_nonzero = min(minimum_nonzero, float(nonzero[0]))
        kernel = vectors[:, eigenvalues < 1e-8]
        for j in range(kernel.shape[1]):
            rows.extend(allowed[indices]); cols.extend([count] * len(indices)); values.extend(kernel[:, j]); count += 1
    projector = sparse.csr_matrix((values, (rows, cols)), shape=(19 * n, count))
    gauss_error = max(float(sparse.linalg.norm(g @ projector)) for g in gauss)
    return dict(masks=masks, P=projector, Q=matter, G=gauss,
                gauge_error=gauss_error, cartan_dimension=len(allowed), raw_dimension=19 * n,
                dimension=count, minimum_nonzero_gauss=minimum_nonzero)


def project(sector, matrix):
    return (sector['P'].conj().T @ matrix @ sector['P']).toarray()


def operators(sector, loop):
    masks = sector['masks']; n = len(masks); identity = sparse.eye(n, format='csr')
    transfers = []; gauge_route_errors = []
    for flavour in range(2):
        direct = one_body(masks, [(mode(flavour, 0, s, c), mode(flavour, 1, s, c), 1.) for s, c in product(range(2), range(3))])
        forward = sparse.csr_matrix((19 * n, 19 * n), dtype=complex)
        backward = forward.copy()
        for a, b in product(range(3), repeat=2):
            body = one_body(masks, [(mode(flavour, 0, s, a), mode(flavour, 1, s, b), 1.) for s in range(2)])
            forward += sparse.kron(body, loop['U'][a, b], format='csr')
            backward += sparse.kron(body, loop['U'][b, a].conj().T, format='csr')
        direct_raw = sparse.kron(direct, sparse.eye(19), format='csr')
        gauge_route_errors.append(max(float(sparse.linalg.norm(g @ route - route @ g))
                                      for g in sector['G'] for route in [direct_raw, forward, backward]))
        transfers.append((project(sector, direct_raw), project(sector, forward), project(sector, backward)))
    occupation = np.array([sum(bool(mask & (1 << mode(f, 0, s, c))) for f, s, c in product(range(2), range(2), range(3))) for mask in masks], float)
    positive = project(sector, sparse.kron(sparse.diags(occupation), sparse.eye(19)))
    mz = np.array([sum((.5 if (i % 6) // 3 == 0 else -.5) for i in range(24) if mask & (1 << i)) for mask in masks])
    # S+S- uses the intermediate M-1 masks without selecting any spin j.
    lower_masks = sorted({step[0] for mask in masks for f, m, c in product(range(2), range(2), range(3))
                          if (step := string(mask, [(mode(f, m, 1, c), True), (mode(f, m, 0, c), False)])) is not None})
    lower_lookup = {mask: i for i, mask in enumerate(lower_masks)}
    rr, cc, vv = [], [], []
    for j, mask in enumerate(masks):
        for f, m, c in product(range(2), range(2), range(3)):
            step = string(mask, [(mode(f, m, 1, c), True), (mode(f, m, 0, c), False)])
            if step is not None: rr.append(lower_lookup[step[0]]); cc.append(j); vv.append(step[1])
    lower = sparse.csr_matrix((vv, (rr, cc)), shape=(len(lower_masks), n))
    s2 = project(sector, sparse.kron(lower.conj().T @ lower + sparse.diags(mz * (mz - 1)), sparse.eye(19)))
    matter_casimir = sum(q @ q for q in sector['Q'])
    octet = project(sector, sparse.kron(matter_casimir / 3, sparse.eye(19)))
    loopops = {key: project(sector, sparse.kron(identity, loop[key])) for key in ['C', 'B', 'boundary']}
    return dict(transfers=transfers, gauge_route_errors=gauge_route_errors, positive=positive, S2=s2, octet=octet, **loopops)


def scalar_field(parameters):
    count = parameters['scalar_points']; phi = np.linspace(-parameters['scalar_extent'], parameters['scalar_extent'], count)
    dx = phi[1] - phi[0]
    kinetic = (2 * np.eye(count) - np.eye(count, k=1) - np.eye(count, k=-1)) / (2 * parameters['scalar_metric'] * dx ** 2)
    potential = parameters['scalar_quartic'] * phi ** 4 - parameters['scalar_quadratic'] * phi ** 2
    momentum = -1j * (np.eye(count, k=1) - np.eye(count, k=-1)) / (2 * dx)
    return dict(phi=np.diag(phi), H=kinetic + np.diag(potential), P=momentum,
                boundary=np.diag([1.] + [0.] * (count - 2) + [1.]), grid=phi)


def law(op, scalar, parameters, angle=0.):
    count = op['S2'].shape[0]; field_count = scalar['H'].shape[0]
    h = np.kron(np.eye(count), scalar['H']) + np.kron(parameters['electric'] * op['C'] / 2 - parameters['plaquette'] * op['B'], np.eye(field_count))
    current = np.zeros_like(h, dtype=complex); contact = current.copy()
    for charge, (direct, forward, backward) in zip((2/3, -1/3), op['transfers']):
        phase = np.exp(1j * charge * angle)
        coupling = parameters['scalar_coupling'] * parameters['loop_route'] / 2
        route = phase * forward + phase.conjugate() * backward
        t = parameters['scalar_coupling'] * direct + coupling * route
        dt = 1j * charge * coupling * (phase * forward - phase.conjugate() * backward)
        ddt = -charge ** 2 * coupling * route
        h = h + np.kron(t + t.conj().T, scalar['phi'])
        current -= np.kron(dt + dt.conj().T, scalar['phi'])
        contact += np.kron(ddt + ddt.conj().T, scalar['phi'])
    return h, current, contact


def source_matrix(neutral, charged, spin):
    target = {mask: i for i, mask in enumerate(charged['masks'])}
    rows, cols, values = [], [], []
    spin_amplitudes = {(0, 0, 1): np.sqrt(2/3), (0, 1, 0): -np.sqrt(1/6), (1, 0, 0): -np.sqrt(1/6)} if spin == .5 else {(0, 0, 1): 1/np.sqrt(3), (0, 1, 0): 1/np.sqrt(3), (1, 0, 0): 1/np.sqrt(3)}
    for j, mask in enumerate(neutral['masks']):
        for colours in product(range(3), repeat=3):
            e = epsilon(*colours)
            if not e: continue
            for spins, amplitude in spin_amplitudes.items():
                ops = [(mode(f, 0, s, c), True) for f, s, c in zip((0, 0, 1), spins, colours)]
                step = string(mask, ops)
                if step is not None and step[0] in target:
                    rows.append(target[step[0]]); cols.append(j); values.append(e * amplitude * step[1] / np.sqrt(6))
    raw = sparse.csr_matrix((values, (rows, cols)), shape=(len(charged['masks']), len(neutral['masks'])))
    return (charged['P'].conj().T @ sparse.kron(raw, sparse.eye(19)) @ neutral['P']).toarray()


def build(parameters):
    loop = loop_field(); scalar = scalar_field(parameters)
    neutral = gauge_sector(masks_for((0, 0), 0, parameters['pair_cut']), loop)
    baryon = gauge_sector(masks_for((2, 1), 1, parameters['pair_cut']), loop)
    anti_masks = sorted(charge_conjugate(mask)[0] for mask in baryon['masks'])
    anti_raw = conjugation(baryon['masks'], anti_masks, loop)
    antibaryon = gauge_sector(anti_masks, loop)
    cn_raw = conjugation(neutral['masks'], neutral['masks'], loop)
    cn = (neutral['P'].conj().T @ cn_raw @ neutral['P']).toarray()
    sectors = dict(neutral=neutral, baryon=baryon, antibaryon=antibaryon)
    ops = {name: operators(sector, loop) for name, sector in sectors.items()}
    cab = (antibaryon['P'].conj().T @ anti_raw @ baryon['P']).toarray()
    nfield = scalar['H'].shape[0]
    cfull = np.kron(cn, np.eye(nfield)); cbfull = np.kron(cab, np.eye(nfield))
    hn, jn, kn = law(ops['neutral'], scalar, parameters)
    hn = (hn + cfull @ hn @ cfull.conj().T) / 2
    jn = (jn - cfull @ jn @ cfull.conj().T) / 2
    kn = (kn + cfull @ kn @ cfull.conj().T) / 2
    hb, jb, kb = law(ops['baryon'], scalar, parameters)
    ha, ja, ka = law(ops['antibaryon'], scalar, parameters)
    raw_c_defects = dict(H=float(np.linalg.norm(ha @ cbfull - cbfull @ hb)),
                         J=float(np.linalg.norm(ja @ cbfull + cbfull @ jb)),
                         K=float(np.linalg.norm(ka @ cbfull - cbfull @ kb)))
    # Explicit C completion of the one raw path law on BOTH number sectors.
    hb = (hb + cbfull.conj().T @ ha @ cbfull) / 2
    jb = (jb - cbfull.conj().T @ ja @ cbfull) / 2
    kb = (kb + cbfull.conj().T @ ka @ cbfull) / 2
    ha = cbfull @ hb @ cbfull.conj().T
    ja = -cbfull @ jb @ cbfull.conj().T
    ka = cbfull @ kb @ cbfull.conj().T
    laws = dict(neutral=(hn, jn, kn), baryon=(hb, jb, kb), antibaryon=(ha, ja, ka))
    sources = {str(spin): np.kron(source_matrix(neutral, baryon, spin), np.eye(nfield)) for spin in (.5, 1.5)}
    anti_sources = {spin: cbfull @ source @ cfull.conj().T for spin, source in sources.items()}
    return dict(parameters=parameters, frame=ordered_frame(), loop=loop, scalar=scalar, sectors=sectors,
                operators=ops, laws=laws, Cneutral=cfull, Cbaryon=cbfull, Cbaryon_raw=anti_raw,
                raw_C_defects=raw_c_defects, sources=sources, anti_sources=anti_sources)
