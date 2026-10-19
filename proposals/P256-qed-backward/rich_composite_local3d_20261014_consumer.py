"""Eight-site 3D cube, mobile odd CAR carriers and eight dynamic local neutral cells.

The neutral cells are literal n=0,1 oscillator compressions, not one global mode.
Pure spin and finite-Krylov helpers are reused from the frozen previous candidate.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from rich_composite_20261014_consumer import spin_basis, spin_pair, spectral_measure, retarded


COORDS = np.array(list(itertools.product((0, 1), repeat=3)), dtype=float)
EDGES = [(v, v ^ (1 << a)) for v in range(8) for a in range(3) if v < (v ^ (1 << a))]


def field_permutation(mask, shift):
    return sum(((mask >> v) & 1) << (v ^ shift) for v in range(8))


def field_operators(p):
    annihilators = []
    for v in range(8):
        columns = [m for m in range(256) if (m >> v) & 1]
        rows = [m ^ (1 << v) for m in columns]
        annihilators.append(sparse.coo_matrix((np.ones(128), (rows, columns)), shape=(256, 256)).tocsr())
    occupations = np.array([[((m >> v) & 1) for v in range(8)] for m in range(256)])
    h = sparse.diags(p['omega'] * occupations.sum(axis=1), format='csr')
    for v, w in EDGES:
        hop = annihilators[v].T @ annihilators[w]
        h -= p['field_hopping'] * (hop + hop.T)
    return h, annihilators, occupations


def build(p, flavors=(0, 1, 2)):
    """Same law in each constituent sector; full position x spin x local-field basis."""
    positions = np.array(list(itertools.permutations(range(8), len(flavors))), dtype=int)
    lookup = {tuple(x): k for k, x in enumerate(positions)}
    spins = spin_basis(len(flavors)); ns = len(spins)
    charges = np.array(p['charges'])[list(flavors)]
    pair_ops = {(i, j): spin_pair(spins, i, j) for i in range(len(flavors)) for j in range(i + 1, len(flavors))}
    load = np.zeros((len(positions), 8)); coulomb = np.zeros(len(positions))
    rows, cols, data = [], [], []
    pair_distances = {}
    for i, j in pair_ops:
        pair_distances[i, j] = np.linalg.norm(COORDS[positions[:, i]] - COORDS[positions[:, j]], axis=1)
    for k, x in enumerate(positions):
        for (i, j), op in pair_ops.items():
            d = pair_distances[i, j][k]
            adjacent = int(x[i] ^ x[j]).bit_count() == 1
            if adjacent:
                load[k, x[i]] += .5; load[k, x[j]] += .5
            coulomb[k] += p['alpha'] * charges[i] * charges[j] / np.sqrt(d*d + p['softening']**2)
            exchange = p['spin_J'] * adjacent * op
            a, b = np.nonzero(exchange)
            rows.extend(k*ns+a); cols.extend(k*ns+b); data.extend(exchange[a, b])
        for i in range(len(flavors)):
            for axis in range(3):
                y = x.copy(); y[i] ^= 1 << axis
                dest = lookup.get(tuple(y))
                if dest is not None:
                    for s in range(ns):
                        rows.append(dest*ns+s); cols.append(k*ns+s); data.append(-p['hopping'])
    matter = sparse.coo_matrix((data, (rows, cols)), shape=(len(positions)*ns,)*2).tocsr()
    matter += sparse.diags(np.repeat(coulomb, ns))
    field, annihilators, occupations = field_operators(p)
    h = sparse.kron(matter, sparse.eye(256), format='csr')
    h += sparse.kron(sparse.eye(matter.shape[0]), field, format='csr')
    for v, b in enumerate(annihilators):
        h -= p['binding_g'] * sparse.kron(sparse.diags(np.repeat(load[:, v], ns)), b+b.T, format='csr')
    meta = dict(positions=positions, lookup=lookup, spins=spins, ns=ns, q=charges,
                pair_ops=pair_ops, distances=pair_distances, load=load, b=annihilators,
                occupations=occupations, flavors=list(flavors))
    return h, meta


def sector_embedding(meta, parity=(0, 0, 0)):
    """Exact simultaneous cube XOR/reflection sectors, including local field motion.

    The cube has three spatial directions but only two sites per direction.
    These discrete graph symmetries are not continuum translational invariance.
    """
    positions = meta['positions']; ns = meta['ns']; lookup = meta['lookup']
    relative = [x for x in positions if x[0] == 0]
    rows, cols, data = [], [], []
    for r, x in enumerate(relative):
        for s in range(ns):
            for mask in range(256):
                column = (r*ns+s)*256+mask
                for shift in range(8):
                    dest = lookup[tuple(x ^ shift)]
                    field = field_permutation(mask, shift)
                    sign = (-1)**sum(parity[a]*((shift >> a) & 1) for a in range(3))
                    rows.append((dest*ns+s)*256+field); cols.append(column); data.append(sign/np.sqrt(8))
    return sparse.coo_matrix((data, (rows, cols)), shape=(len(positions)*ns*256, len(relative)*ns*256)).tocsr()


def ground(h, number=1):
    values, vectors = eigsh(h, k=number, which='SA', tol=1e-10,
                            v0=np.linspace(.13, 1.17, h.shape[0]))
    order = np.argsort(values)
    return values[order], vectors[:, order]


def sector_ground(h, meta, parity=(0, 0, 0), number=1):
    u = sector_embedding(meta, parity)
    reduced = (u.T @ h @ u).tocsr()
    values, vectors = ground(reduced, number)
    return values, u @ vectors, u, reduced


def density(qvector, meta):
    factors = np.exp(1j*(COORDS[meta['positions']] @ np.asarray(qvector))) @ meta['q']
    return np.repeat(factors, meta['ns']*256)


def current_contact(p, qvector, polarization, meta):
    """Peierls -dH/dA(-q) and d2H/dA(q)dA(-q) on embedded straight edges."""
    qvector = np.asarray(qvector); polarization = np.asarray(polarization)
    rows, cols, jd, cd = [], [], [], []
    for k, x in enumerate(meta['positions']):
        for i, charge in enumerate(meta['q']):
            for axis in range(3):
                y = x.copy(); y[i] ^= 1 << axis
                dest = meta['lookup'].get(tuple(y))
                if dest is None:
                    continue
                start = COORDS[x[i]]; end = COORDS[y[i]]; delta = end-start
                integral = float(polarization @ delta) * np.exp(1j*qvector @ ((start+end)/2)) * np.sinc((qvector @ delta)/(2*np.pi))
                for s in range(meta['ns']):
                    rows.append(dest*meta['ns']+s); cols.append(k*meta['ns']+s)
                    jd.append(1j*p['hopping']*charge*integral)
                    cd.append(p['hopping']*charge**2*abs(integral)**2)
    shape = (len(meta['positions'])*meta['ns'],)*2
    j = sparse.coo_matrix((jd, (rows, cols)), shape=shape).tocsr()
    c = sparse.coo_matrix((cd, (rows, cols)), shape=shape).tocsr()
    return sparse.kron(j, sparse.eye(256), format='csr'), sparse.kron(c, sparse.eye(256), format='csr')


def fragmentation(p, energy):
    rows = []
    for pair in itertools.combinations(range(3), 2):
        hp, mp = build(p, pair)
        ep = float(sector_ground(hp, mp)[0][0])
        singleton = -3*p['hopping']
        rows.append(dict(pair=list(pair), pair_energy=ep, singleton_energy=singleton,
                         sum_energy=ep+singleton, binding_margin=ep+singleton-energy))
    return rows


def run(p, output):
    h, m = build(p)
    values, block, u0, h0 = sector_ground(h, m, number=3)
    energy = float(values[0]); psi = block[:, 0]
    amp = psi.reshape(len(m['positions']), m['ns'], 256)
    probability = np.sum(abs(amp)**2, axis=(1, 2))
    pairs = []
    s2 = np.eye(m['ns'])*2.25
    for (i, j), op in m['pair_ops'].items():
        corr = np.einsum('psb,st,ptb->p', amp.conj(), op, amp).real
        pairs.append(dict(pair=[i, j], mean_distance=float(probability @ m['distances'][i, j]),
                          spin_dot=float(corr.sum()), nearest_neighbor_probability=float(probability @ (m['distances'][i, j] == 1))))
        s2 += 2*op
    actual_s2 = float(np.einsum('psb,st,ptb->', amp.conj(), s2, amp).real)
    field_rdm = amp.reshape(-1, 256).conj().T @ amp.reshape(-1, 256)
    eig = np.linalg.eigvalsh(field_rdm); positive = eig[eig > 1e-14]
    field_rows = []
    for v, b in enumerate(m['b']):
        # Every position has its own conditional deformation load; no mean-load replacement.
        displaced = np.einsum('psb,bc,psc->p', amp.conj(), (b+b.T).toarray(), amp).real
        occupied = float(np.diag(field_rdm).real @ m['occupations'][:, v])
        field_rows.append(dict(site=v, occupation=occupied, local_load_displacement=float(m['load'][:, v] @ displaced),
                               mean_load=float(probability @ m['load'][:, v]), displacement=float(displaced.sum())))
    coordinates = COORDS[m['positions']]
    relative = coordinates-coordinates.mean(axis=1, keepdims=True)
    signed_radius = float(probability @ np.sum(np.sum(relative**2, axis=2)*m['q'], axis=1))
    form = []
    for qv in ([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [2, 0, 0]):
        f = probability @ (np.exp(1j*(relative @ np.asarray(qv))) @ m['q'])
        form.append(dict(qvector=qv, real=float(f.real), imag=float(f.imag)))
    breakup = fragmentation(p, energy)
    adverse_p = dict(p, binding_g=0.0)
    ah, am = build(adverse_p)
    ae = float(sector_ground(ah, am)[0][0])
    adverse = dict(energy=ae, breakup=fragmentation(adverse_p, ae), three_singles_margin=-9*p['hopping']-ae)
    responses = []
    saved = dict(ground=psi, energies=values, positions=m['positions'], charges=m['q'])
    for axis in range(3):
        # axis here is Cartesian; bit order is reversed in COORDS.
        bit = 2-axis; parity = [0, 0, 0]; parity[bit] = 1
        ev, vb, uq, hq = sector_ground(h, m, tuple(parity), number=4)
        multiplicity = int(np.sum(abs(ev-ev[0]) < 1e-7))
        qv = np.zeros(3); qv[axis] = np.pi
        rho = density(qv, m); source = rho*psi
        pol = np.eye(3)[axis]
        j, c = current_contact(p, qv, pol, m)
        js = j @ psi
        continuity = np.linalg.norm(h @ source-energy*source-np.pi*js)
        dm = spectral_measure(hq, energy, uq.T @ source, p['krylov_steps'])
        jm = spectral_measure(hq, energy, uq.T @ js, p['krylov_steps'])
        density_vertex = vb[:, :multiplicity].conj().T @ source
        current_vertex = vb[:, :multiplicity].conj().T @ js
        recoil = float(ev[0]-energy)
        transverse = []
        for transverse_axis in range(3):
            if transverse_axis == axis:
                continue
            tj, tc = current_contact(p, qv, np.eye(3)[transverse_axis], m)
            tm = spectral_measure(h, energy, tj @ psi, p['krylov_steps'])
            transverse.append(dict(polarization_axis=transverse_axis, current=tm,
                                   contact=float(np.vdot(psi, tc @ psi).real),
                                   retarded=retarded(tm, p['frequencies'], p['eta'])))
        responses.append(dict(axis=axis, qvector=qv.tolist(), odd_sector_energies=ev.tolist(),
                              ground_block_multiplicity=multiplicity, recoil=recoil,
                              charge_elastic_ground_block_norm=float(np.linalg.norm(density_vertex)),
                              current_elastic_ground_block_norm=float(np.linalg.norm(current_vertex)),
                              ward_vertex_error=float(np.linalg.norm(recoil*density_vertex-np.pi*current_vertex)),
                              full_continuity_source_error=float(continuity), density=dm, current=jm,
                              contact=float(np.vdot(psi, c @ psi).real), transverse=transverse,
                              retarded_density=retarded(dm, p['frequencies'], p['eta']),
                              retarded_current=retarded(jm, p['frequencies'], p['eta'])))
        saved['odd_block_'+str(axis)] = vb
    result = dict(candidate='COMPOSITE_LOCAL_CUBE_NEUTRAL_FIELD_V1', inputs=p,
                  dimension=h.shape[0], exact_even_sector_dimension=h0.shape[0], energy=energy,
                  even_sector_energies=values.tolist(), S2=actual_s2, Sz=.5, charge=float(m['q'].sum()),
                  breakup=breakup, three_singles_margin=-9*p['hopping']-energy,
                  adverse_no_shared_field=adverse, pair_correlations=pairs, field=field_rows,
                  field_matter_entropy=float(-np.sum(positive*np.log(positive))),
                  neutral_total_occupation=float(np.diag(field_rdm).real @ m['occupations'].sum(axis=1)),
                  local_occupation_one_probability='Each site occupation is literal n=0,1 finite quantum-cell law, not a converged harmonic approximation.',
                  signed_charge_radius_squared=signed_radius, intrinsic_embedded_form=form,
                  response=responses, ground_eigen_residual=float(np.linalg.norm(h @ psi-energy*psi)))
    Path(output+'.json').write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(output+'.npz', **saved)
    print(json.dumps(dict(energy=energy, S2=actual_s2, binding=[r['binding_margin'] for r in breakup],
                          adverse_binding=[r['binding_margin'] for r in adverse['breakup']],
                          recoil=[r['recoil'] for r in responses], dimension=h.shape[0])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.request).read_text()), args.output)
