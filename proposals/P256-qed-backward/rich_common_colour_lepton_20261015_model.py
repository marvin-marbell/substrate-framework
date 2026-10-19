"""NEW CAR32 common quaternion/mirror root, quantum radial field and PW19 loop.
Exact Hadamard-rotated mirror-population closure, NOT an original-mirror pair cut.
Only pure Haar-loop, epsilon and SU3-generator helpers are reused.
"""
from __future__ import annotations
from itertools import combinations, product
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
from rich_colour_composite_20261014_model import generators, loop_field, epsilon


def mode(flavour, mirror, spin, member):
    return 16 * flavour + 8 * mirror + 4 * spin + member


def apply(mask, index, create):
    if bool(mask & (1 << index)) == create:
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
    """Unitary Fock C: particle/hole, i tau_y mirror, i sigma_y frame spin."""
    target = (1 << 32) - 1
    amplitude = 1
    for index in reversed([i for i in range(32) if mask & (1 << i)]):
        f, rest = divmod(index, 16)
        m, rest = divmod(rest, 8)
        s, c = divmod(rest, 4)
        target, sign = apply(target, mode(f, 1 - m, 1 - s, c), False)
        amplitude *= sign * (1 if m == s else -1)
    return target, amplitude


def common_root():
    pauli = [np.array([[0, 1], [1, 0]], complex),
             np.array([[0, -1j], [1j, 0]], complex), np.diag([1., -1.])]
    cliff = [np.kron(pauli[0], np.eye(2))] + [np.kron(pauli[1], f) for f in pauli]
    phi = [np.kron(c, np.eye(4)) for c in cliff]
    z = np.kron(np.eye(4), np.diag([1., 1., 1., -3.]))
    intertwiners = [np.kron(np.eye(4), np.eye(4)[:, a:a+1] @ np.eye(4)[b:b+1])
                    for a, b in product(range(4), repeat=2)]
    split = [t for t in intertwiners if np.linalg.norm(t @ z - z @ t) < 1e-12]
    charges = [2/3, 2/3, 2/3, 0., -1/3, -1/3, -1/3, -1.]
    monomials = [np.eye(4, dtype=complex)]
    for length in range(1, 5):
        for subset in combinations(range(4), length):
            a = np.eye(4, dtype=complex)
            for index in subset:
                a = a @ cliff[index]
            monomials.append(a)
    return dict(cliff=cliff, Phi=phi, Z=z, charge_pattern=charges,
        clifford_defect=float(max(np.linalg.norm(a @ b + b @ a - (2 * np.eye(16) if i == j else 0))
            for i, a in enumerate(phi) for j, b in enumerate(phi))),
        algebra_monomial_rank=int(np.linalg.matrix_rank(np.column_stack([x.ravel() for x in monomials]))),
        full_commutant_dimension=len(intertwiners), split_commutant_dimension=len(split),
        splitting_commutator=float(max(np.linalg.norm(z @ a - a @ z) for a in phi)),
        inputs='Ordered mirror/quaternion C4, FOUR repeats, fixed Z rank3+1, flavour doublet, CAR, selected Q=T3+(B-L)/2 and SU3 gauging are primitive choices. Phi_a=phi*C_a tensor I4 has ONE quantized shared radial amplitude. No angular/order/split-field fluctuations or spontaneous rank/charge selection are proved.',
        stabilizer='Chosen connected SU16 stabilizer has SU3 plus U1_B-L. Extra U1 treatment and choosing/gauging EM are unpaid inputs, not consequences of rank count.')


def masks_for(delta, twice_m):
    """Complete occupied rotated-negative sea plus specified rotated-positive insertions."""
    members = [range(3), range(3), [3], [3]]
    flavours = [0, 1, 0, 1]
    sea = sum(1 << mode(f, 1, s, c) for f, s, c in product(range(2), range(2), range(4)))
    pos = [[mode(f, 0, s, c) for s, c in product(range(2), cs)] for f, cs in zip(flavours, members)]
    out = []
    for choice in product(*[list(combinations(ps, n)) for ps, n in zip(pos, delta)]):
        mask = sea
        for indices in choice:
            for index in indices:
                mask |= 1 << index
        mz = sum(1 if (i % 8) // 4 == 0 else -1 for i in range(32) if mask & (1 << i))
        if mz == twice_m:
            out.append(mask)
    return sorted(out)


def one_body(source_masks, terms, target_masks=None):
    target_masks = source_masks if target_masks is None else target_masks
    lookup = {mask: i for i, mask in enumerate(target_masks)}
    rows, cols, values = [], [], []
    for column, mask in enumerate(source_masks):
        for a, b, value in terms:
            step = string(mask, [(a, True), (b, False)])
            if step is not None and step[0] in lookup:
                rows.append(lookup[step[0]]); cols.append(column); values.append(value * step[1])
    return sparse.csr_matrix((values, (rows, cols)), shape=(len(target_masks), len(source_masks)), dtype=complex)


def conjugation(source_masks, target_masks):
    lookup = {mask: i for i, mask in enumerate(target_masks)}
    rows, values = [], []
    for mask in source_masks:
        target, value = charge_conjugate(mask)
        rows.append(lookup[target]); values.append(value)
    matter = sparse.csr_matrix((values, (rows, range(len(rows)))), shape=(len(target_masks), len(source_masks)))
    swap = np.zeros((19, 19)); swap[0, 0] = 1
    swap[1:10, 10:] = np.eye(9); swap[10:, 1:10] = np.eye(9)
    return sparse.kron(matter, swap, format='csr')


def gauss_operators(masks, loop):
    n = len(masks)
    qs = [one_body(masks, [(mode(f, m, s, a), mode(f, m, s, b), t[a, b])
          for f, m, s, a, b in product(range(2), range(2), range(2), range(3), range(3)) if t[a, b] != 0]) for t in generators()]
    gs = [sparse.kron(q, sparse.eye(19), format='csr') + sparse.kron(sparse.eye(n), loop['L'][a] + loop['R'][a], format='csr') for a, q in enumerate(qs)]
    return qs, gs


def gauge_sector(masks, loop):
    qs, gs = gauss_operators(masks, loop)
    allowed = np.flatnonzero((abs(gs[6].diagonal()) < 1e-9) & (abs(gs[7].diagonal()) < 1e-9))
    casimir = sum(g[:, allowed].conj().T @ g[:, allowed] for g in gs).tocsr()
    graph = casimir.copy(); graph.data = np.where(abs(graph.data) > 1e-10, 1., 0.); graph.eliminate_zeros()
    blocks, labels = connected_components(graph, directed=False)
    rows, cols, values = [], [], []; count = 0
    for block in range(blocks):
        indices = np.flatnonzero(labels == block)
        eigenvalues, vectors = eigh(casimir[indices][:, indices].toarray())
        kernel = vectors[:, eigenvalues < 1e-8]
        for j in range(kernel.shape[1]):
            rows.extend(allowed[indices]); cols.extend([count] * len(indices)); values.extend(kernel[:, j]); count += 1
    projector = sparse.csr_matrix((values, (rows, cols)), shape=(19 * len(masks), count))
    return dict(masks=masks, P=projector, Q=qs, G=gs, dimension=count, raw_dimension=19 * len(masks),
                gauge_error=max(float(sparse.linalg.norm(g @ projector)) for g in gs))


def conjugate_sector(sector, loop):
    masks = sorted(charge_conjugate(mask)[0] for mask in sector['masks'])
    c = conjugation(sector['masks'], masks)
    p = c @ sector['P']
    qs, gs = gauss_operators(masks, loop)
    return dict(masks=masks, P=p, Q=qs, G=gs, dimension=p.shape[1], raw_dimension=p.shape[0],
                gauge_error=max(float(sparse.linalg.norm(g @ p)) for g in gs))


def project(sector, matrix):
    return (sector['P'].conj().T @ matrix @ sector['P']).toarray()


def routes(source_masks, loop, target_masks=None):
    """Half of each rotated density route; law adds its adjoint exactly once."""
    target_masks = source_masks if target_masks is None else target_masks
    rows = 19 * len(target_masks); cols = 19 * len(source_masks)
    out = []
    for f in range(2):
        for colourless in [False, True]:
            members = [3] if colourless else range(3)
            density = [(mode(f, m, s, c), mode(f, m, s, c), (.5 if m == 0 else -.5))
                       for m, s, c in product(range(2), range(2), members)]
            direct_body = one_body(source_masks, density, target_masks)
            direct = sparse.kron(direct_body, sparse.eye(19), format='csr')
            forward = sparse.csr_matrix((rows, cols), dtype=complex); backward = forward.copy()
            if colourless:
                # Chosen gauge-singlet Wilson-trace polarizability, not a lepton colour charge.
                forward = sparse.kron(direct_body, loop['W'] / 3, format='csr')
                backward = sparse.kron(direct_body, loop['W'].conj().T / 3, format='csr')
            else:
                for a, b in product(range(3), repeat=2):
                    body = one_body(source_masks, [(mode(f, m, s, a), mode(f, m, s, b), (.5 if m == 0 else -.5))
                                    for m, s in product(range(2), range(2))], target_masks)
                    forward += sparse.kron(body, loop['U'][a, b], format='csr')
                    backward += sparse.kron(body, loop['U'][b, a].conj().T, format='csr')
            charge = ([0., -1.] if colourless else [2/3, -1/3])[f]
            out.append((charge, direct, forward, backward))
    return out


def operators(sector, loop):
    masks = sector['masks']; n = len(masks)
    rs = routes(masks, loop)
    transfers = [(q, project(sector, d), project(sector, f), project(sector, b)) for q, d, f, b in rs]
    route_errors = [max(float(sparse.linalg.norm(g @ route - route @ g)) for g in sector['G'] for route in [d, f, b]) for _, d, f, b in rs]
    positive = np.array([sum(bool(mask & (1 << mode(f, 0, s, c))) for f, s, c in product(range(2), range(2), range(4))) for mask in masks], float)
    mz = np.array([sum(.5 if (i % 8) // 4 == 0 else -.5 for i in range(32) if mask & (1 << i)) for mask in masks])
    numbers = np.array([mask.bit_count() for mask in masks], float)
    charges = np.array([sum(([2/3, 2/3, 2/3, 0.] if i // 16 == 0 else [-1/3, -1/3, -1/3, -1.])[i % 4]
                            for i in range(32) if mask & (1 << i)) for mask in masks])
    lower_masks = sorted({step[0] for mask in masks for f, m, c in product(range(2), range(2), range(4))
        if (step := string(mask, [(mode(f, m, 1, c), True), (mode(f, m, 0, c), False)])) is not None})
    lower = one_body(masks, [(mode(f, m, 1, c), mode(f, m, 0, c), 1.) for f, m, c in product(range(2), range(2), range(4))], lower_masks)
    s2 = project(sector, sparse.kron(lower.conj().T @ lower + sparse.diags(mz * (mz - 1)), sparse.eye(19)))
    diagonal = lambda x: project(sector, sparse.kron(sparse.diags(x), sparse.eye(19)))
    colour_casimir = sum(q @ q for q in sector['Q'])
    loopops = {key: project(sector, sparse.kron(sparse.eye(n), loop[key])) for key in ['C', 'B', 'boundary']}
    return dict(transfers=transfers, route_errors=route_errors, positive=diagonal(positive), S2=s2,
        N=diagonal(numbers), Q=diagonal(charges), matter_octet=project(sector, sparse.kron(colour_casimir / 3, sparse.eye(19))), **loopops)


def scalar_field(p):
    count = p['scalar_points']; grid = np.linspace(-p['scalar_extent'], p['scalar_extent'], count); dx = grid[1] - grid[0]
    kinetic = (2 * np.eye(count) - np.eye(count, k=1) - np.eye(count, k=-1)) / (2 * p['scalar_metric'] * dx**2)
    potential = p['scalar_quartic'] * grid**4 - p['scalar_quadratic'] * grid**2
    return dict(H=kinetic + np.diag(potential), phi=np.diag(grid), P=-1j * (np.eye(count, k=1) - np.eye(count, k=-1)) / (2 * dx),
        grid=grid, kinetic=kinetic, potential=np.diag(potential), boundary=np.diag([1.] + [0.] * (count - 2) + [1.]))


def law(op, scalar, p, angle=0.):
    nf = len(scalar['grid']); n = op['S2'].shape[0]
    h = np.kron(np.eye(n), scalar['H']) + np.kron(p['electric'] * op['C'] / 2 - p['plaquette'] * op['B'], np.eye(nf))
    j = np.zeros_like(h, dtype=complex); k = j.copy()
    for q, direct, forward, backward in op['transfers']:
        phase = np.exp(1j * q * angle); alpha = p['scalar_coupling'] * p['loop_route'] / 2
        route = phase * forward + phase.conjugate() * backward
        t = p['scalar_coupling'] * direct + alpha * route
        dt = 1j * q * alpha * (phase * forward - phase.conjugate() * backward)
        ddt = -q**2 * alpha * route
        h += np.kron(t + t.conj().T, scalar['phi'])
        j -= np.kron(dt + dt.conj().T, scalar['phi'])
        k += np.kron(ddt + ddt.conj().T, scalar['phi'])
    return h, j, k


def source_matrix(neutral, target, kind, spin=.5):
    lookup = {mask: i for i, mask in enumerate(target['masks'])}; rows, cols, values = [], [], []
    amplitudes = {(0, 0, 1): np.sqrt(2/3), (0, 1, 0): -np.sqrt(1/6), (1, 0, 0): -np.sqrt(1/6)} if spin == .5 else {(0, 0, 1): 1/np.sqrt(3), (0, 1, 0): 1/np.sqrt(3), (1, 0, 0): 1/np.sqrt(3)}
    for j, mask in enumerate(neutral['masks']):
        terms = []
        if kind == 'lepton':
            terms = [([(mode(1, 0, 0, 3), True)], 1.)]
        else:
            for colours in product(range(3), repeat=3):
                if epsilon(*colours):
                    for spins, amplitude in amplitudes.items():
                        terms.append(([(mode(f, 0, s, c), True) for f, s, c in zip((0, 0, 1), spins, colours)], epsilon(*colours) * amplitude / np.sqrt(6)))
        for operations, value in terms:
            step = string(mask, operations)
            if step is not None and step[0] in lookup:
                rows.append(lookup[step[0]]); cols.append(j); values.append(value * step[1] * (2 ** (-.5) if kind == 'lepton' else 2 ** (-1.5)))
    raw = sparse.csr_matrix((values, (rows, cols)), shape=(len(target['masks']), len(neutral['masks'])))
    return (target['P'].conj().T @ sparse.kron(raw, sparse.eye(19)) @ neutral['P']).toarray()


def occupation_closure(sector):
    """Enumerate every matter target of H(A), J(A), K(A), at arbitrary A."""
    present = set(sector['masks']); missing = set()
    for mask in sector['masks']:
        for f, m, s, a, b in product(range(2), range(2), range(2), range(4), range(4)):
            if (a == 3) != (b == 3):
                continue
            step = string(mask, [(mode(f, m, s, a), True), (mode(f, m, s, b), False)])
            if step is not None and step[0] not in present:
                missing.add(step[0])
    return dict(missing_targets=len(missing), rotated_mirror_offdiagonal_coefficient=0.,
        argument='For every A, W(A) is Hermitian. Original coupling tau_x tensor [g I+g r W(A)/2] becomes tau_z density under Hadamard. H,J,K preserve each rotated-mirror/flavour/spin/colour-vs-lepton population. Every active colour target is enumerated, all PW19 loop and DVR field states retained. Original-positive creation on filled rotated-negative sea equals 2^-1/2 times rotated-positive creation, exactly. This is exact source-connected closure, not full all-population/all-N vacuum minimization.')


def build(p):
    loop = loop_field(); scalar = scalar_field(p)
    sectors = {'neutral': gauge_sector(masks_for((0, 0, 0, 0), 0), loop),
               'baryon': gauge_sector(masks_for((2, 1, 0, 0), 1), loop),
               'lepton': gauge_sector(masks_for((0, 0, 0, 1), 1), loop)}
    for name, anti in [('baryon', 'antibaryon'), ('lepton', 'antilepton')]:
        sectors[anti] = conjugate_sector(sectors[name], loop)
    cn = (sectors['neutral']['P'].conj().T @ conjugation(sectors['neutral']['masks'], sectors['neutral']['masks']) @ sectors['neutral']['P']).toarray()
    cs = {'neutral': np.kron(cn, np.eye(len(scalar['grid'])))}
    for name, anti in [('baryon', 'antibaryon'), ('lepton', 'antilepton')]:
        c = (sectors[anti]['P'].conj().T @ conjugation(sectors[name]['masks'], sectors[anti]['masks']) @ sectors[name]['P']).toarray()
        cs[name] = np.kron(c, np.eye(len(scalar['grid'])))
    ops = {name: operators(sector, loop) for name, sector in sectors.items()}
    sources = {'baryon': {str(s):np.kron(source_matrix(sectors['neutral'], sectors['baryon'], 'baryon', s), np.eye(len(scalar['grid']))) for s in [.5, 1.5]},
               'lepton': {'0.5':np.kron(source_matrix(sectors['neutral'], sectors['lepton'], 'lepton'), np.eye(len(scalar['grid'])))}}
    for name, anti in [('baryon', 'antibaryon'), ('lepton', 'antilepton')]:
        sources[anti] = {s:cs[name] @ source @ cs['neutral'].conj().T for s, source in sources[name].items()}
    return dict(parameters=p, loop=loop, scalar=scalar, root=common_root(), sectors=sectors, operators=ops, C=cs, sources=sources)
