"""Mobile charged flavors and local oriented axial singlet-triplet neutral cells.

This is a finite planar graph law, not a continuum rotational or QCD claim.
Primitive spin/CAR and the carrier-assisted spin-orbit hopping are explicit inputs.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

COORDS = np.array([[-.5, -.5, 0], [.5, -.5, 0], [.5, .5, 0], [-.5, .5, 0]])
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]
PAULI = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]], complex), np.diag([1, -1]).astype(complex)]


def tensor_local(operator, site, count, size):
    factors = [sparse.eye(size, format='csr') for _ in range(count)]
    factors[site] = sparse.csr_matrix(operator)
    result = factors[0]
    for factor in factors[1:]:
        result = sparse.kron(result, factor, format='csr')
    return result


def carrier_operators(removed=False):
    if removed:
        zero = sparse.csr_matrix((1, 1), dtype=complex)
        return zero, [[zero]*3 for _ in EDGES], [[zero]*3 for _ in EDGES], [zero]*3
    energy = sparse.csr_matrix((256, 256), dtype=complex)
    qs, ps = [], []
    angular = [sparse.csr_matrix((256, 256), dtype=complex) for _ in range(3)]
    for edge in range(4):
        energy += tensor_local(np.diag([0, 1, 1, 1]), edge, 4, 4)
        qrow, prow = [], []
        for a in range(3):
            q = np.zeros((4, 4), complex); q[a+1, 0] = q[0, a+1] = 1
            momentum = np.zeros((4, 4), complex); momentum[a+1, 0] = 1j; momentum[0, a+1] = -1j
            qrow.append(tensor_local(q, edge, 4, 4)); prow.append(tensor_local(momentum, edge, 4, 4))
            ell = np.zeros((4, 4), complex)
            b, c = (a+1) % 3, (a+2) % 3
            ell[b+1, c+1] = -1j; ell[c+1, b+1] = 1j
            angular[a] += tensor_local(ell, edge, 4, 4)
        qs.append(qrow); ps.append(prow)
    return energy, qs, ps, angular


def build(p, flavors=(0, 1, 2), loading='difference', removed=False):
    """Full positions x all spin states x all local carrier internal states."""
    count = len(flavors); ns = 2**count
    positions = np.array(list(itertools.permutations(range(4), count)), int)
    lookup = {tuple(x): i for i, x in enumerate(positions)}
    en, qs, ps, angular = carrier_operators(removed)
    nf = en.shape[0]; im = sparse.eye(len(positions)*ns, format='csr'); iff = sparse.eye(nf, format='csr')
    spins = [[tensor_local(PAULI[a]/2, i, count, 2) for a in range(3)] for i in range(count)]
    h = p['omega']*sparse.kron(im, en, format='csr')
    coulomb = np.zeros(len(positions)); charges = np.array(p['charges'])[list(flavors)]
    loads = [[sparse.csr_matrix((len(positions)*ns,)*2, dtype=complex) for _ in range(3)] for _ in EDGES]
    hops = []
    for k, x in enumerate(positions):
        for i, j in itertools.combinations(range(count), 2):
            distance = np.linalg.norm(COORDS[x[i]]-COORDS[x[j]])
            coulomb[k] += p['alpha']*charges[i]*charges[j]/distance
        for edge, (v, w) in enumerate(EDGES):
            occupied = [i for i in range(count) if x[i] in (v, w)]
            if len(occupied) == 2:
                i = next(i for i in occupied if x[i] == v); j = next(i for i in occupied if x[i] == w)
                projection = sparse.coo_matrix(([1.], ([k], [k])), shape=(len(positions),)*2).tocsr()
                for a in range(3):
                    sign = -1 if loading == 'difference' else 1
                    loads[edge][a] += sparse.kron(projection, spins[i][a]+sign*spins[j][a], format='csr')
                # Explicit axial material pre-twist; zero in the isotropic candidate.
                h -= p.get('axial_bias', 0.)*sparse.kron(sparse.kron(projection, sparse.eye(ns), format='csr'), qs[edge][2], format='csr')
            for i in occupied:
                destination = w if x[i] == v else v
                y = x.copy(); y[i] = destination
                dest = lookup.get(tuple(y))
                if dest is None:
                    continue
                orient = 1 if x[i] == v else -1
                position_hop = sparse.coo_matrix(([1.], ([dest], [k])), shape=(len(positions),)*2).tocsr()
                term = -p['hopping']*sparse.kron(sparse.kron(position_hop, sparse.eye(ns), format='csr'), iff, format='csr')
                for a in range(3):
                    term += 1j*orient*p['spin_orbit']*sparse.kron(sparse.kron(position_hop, spins[i][a], format='csr'), qs[edge][a], format='csr')
                h += term
                hops.append((term, charges[i], COORDS[x[i]], COORDS[destination]))
    h += sparse.kron(sparse.diags(np.repeat(coulomb, ns)), iff, format='csr')
    for edge in range(4):
        for a in range(3):
            h -= p['loading_g']*sparse.kron(loads[edge][a], ps[edge][a], format='csr')
    spin_generators = [sparse.kron(sparse.kron(sparse.eye(len(positions)), sum(s[a] for s in spins), format='csr'), iff, format='csr') for a in range(3)]
    carrier_generators = [sparse.kron(im, angular[a], format='csr') for a in range(3)]
    total = [spin_generators[a]+carrier_generators[a] for a in range(3)]
    return h.tocsr(), dict(positions=positions, charges=charges, ns=ns, nf=nf, hops=hops,
                           S=spin_generators, L=carrier_generators, J=total, carrier_energy=sparse.kron(im, en, format='csr'),
                           isotropic=p.get('axial_bias', 0.) == 0.)


def ground(h, number=6):
    rng = np.random.default_rng(1947)
    initial = rng.normal(size=h.shape[0])+1j*rng.normal(size=h.shape[0])
    ev, vv = eigsh(h, k=min(number, h.shape[0]-2), which='SA', tol=2e-10, v0=initial)
    order = np.argsort(ev)
    return ev[order], vv[:, order]


def expectation(psi, operator):
    return float(np.vdot(psi, operator @ psi).real)


def quadratic_generator(psi, operators):
    return float(sum(np.vdot(op @ psi, op @ psi).real for op in operators))


def time_reverse(state, m):
    spin_flip = np.array([[0., 1.], [-1., 0.]])
    operation = sparse.csr_matrix([[1.]])
    for _ in range(int(np.log2(m['ns']))):
        operation = sparse.kron(operation, spin_flip, format='csr')
    shaped = state.conj().reshape(len(m['positions']), m['ns'], m['nf'])
    return np.einsum('ab,pbf->paf', operation.toarray(), shaped).reshape(-1)


def polarized_ground(ev, vv, m):
    # Single-vector Lanczos need not return every exact degenerate partner.
    # Restore the actual symmetry orbit rather than trusting column multiplicity.
    block = vv[:, abs(ev-ev[0]) < 1e-9]
    for _ in range(5):
        additions = [block, np.stack([time_reverse(v, m) for v in block.T], axis=1)]
        generators = m['J'] if m['isotropic'] else [m['J'][2]]
        additions.extend(op @ block for op in generators)
        trial = np.concatenate(additions, axis=1)
        vectors, singular, _ = np.linalg.svd(trial, full_matrices=False)
        restored = vectors[:, singular > 1e-8]
        if restored.shape[1] == block.shape[1]:
            block = restored
            break
        block = restored
    jz = block.conj().T @ (m['J'][2] @ block)
    z, rotation = np.linalg.eigh(jz)
    m['ground_basis'] = block @ rotation
    m['ground_magnetic_numbers'] = z
    plus = block @ rotation[:, -1]
    minus = time_reverse(plus, m)
    return plus, minus, z


def current_contact(m, q, polarization, magnetic=False):
    """Derivatives of the full hopping, including explicit carrier-assisted spin-orbit."""
    size = m['J'][0].shape[0]
    current = sparse.csr_matrix((size, size), dtype=complex)
    contact = sparse.csr_matrix((size, size), dtype=complex)
    q = np.asarray(q); polarization = np.asarray(polarization)
    for term, charge, start, end in m['hops']:
        delta = end-start
        if magnetic:
            line = .5*np.cross(start, end)[2]
        else:
            line = (polarization @ delta)*np.exp(1j*q @ ((start+end)/2))*np.sinc((q @ delta)/(2*np.pi))
        current += -1j*charge*line*term
        contact += -charge**2*abs(line)**2*term
    return current.tocsr(), contact.tocsr()


def spectrum(h, energy, source, steps):
    weight = float(np.vdot(source, source).real)
    if weight < 1e-24:
        return dict(total_weight=weight, poles=[], weights=[])
    vector = source/np.sqrt(weight); old = np.zeros_like(vector)
    diagonal, off = [], []; beta = 0.
    basis = []
    for _ in range(steps):
        basis.append(vector.copy())
        work = h @ vector-energy*vector-beta*old
        alpha = float(np.vdot(vector, work).real); diagonal.append(alpha)
        work -= alpha*vector
        # Full reorthogonalization prevents duplicate poles masquerading as weight.
        matrix = np.stack(basis, axis=1)
        work -= matrix @ (matrix.conj().T @ work)
        next_beta = float(np.linalg.norm(work))
        if next_beta < 1e-11:
            break
        off.append(next_beta); old, vector = vector, work/next_beta; beta = next_beta
    tridiagonal = np.diag(diagonal)
    if len(diagonal) > 1:
        tridiagonal += np.diag(off[:len(diagonal)-1], 1)+np.diag(off[:len(diagonal)-1], -1)
    poles, vectors = np.linalg.eigh(tridiagonal)
    return dict(total_weight=weight, poles=poles.tolist(), weights=(weight*abs(vectors[0])**2).tolist(), krylov_dimension=len(diagonal))


def fragment_threshold(p, energy, loading='difference', removed=False):
    singleton_energies = {}
    for i in range(3):
        h, _ = build(p, (i,), loading, removed)
        singleton_energies[i] = float(ground(h, 2)[0][0])
    fragments = []
    for pair in itertools.combinations(range(3), 2):
        h, _ = build(p, pair, loading, removed)
        ep = float(ground(h, 2)[0][0]); other = next(i for i in range(3) if i not in pair)
        threshold = ep+singleton_energies[other]
        fragments.append(dict(pair=list(pair), pair_energy=ep, singleton_energy=singleton_energies[other], threshold=threshold, binding_margin=threshold-energy))
    return dict(pair_plus_single=fragments, three_dressed_singles=sum(singleton_energies.values()), three_singles_margin=sum(singleton_energies.values())-energy)


def high_angular_sector(h, m):
    """Comparison sector Jz=3/2, without restricting the delivered ground search."""
    local = np.zeros((4, 4), complex)
    local[0, 0] = 1; local[3, 1] = 1
    local[1, 2] = 1/np.sqrt(2); local[2, 2] = 1j/np.sqrt(2)
    local[1, 3] = 1/np.sqrt(2); local[2, 3] = -1j/np.sqrt(2)
    field_rotation = sparse.csr_matrix([[1.]])
    magnetic_numbers = np.array([0.])
    for _ in EDGES:
        field_rotation = sparse.kron(field_rotation, local, format='csr')
        magnetic_numbers = (magnetic_numbers[:, None]+np.array([0, 0, 1, -1])[None, :]).reshape(-1)
    count = int(np.log2(m['ns']))
    spin_numbers = np.array([sum(.5 if s == 0 else -.5 for s in bits) for bits in itertools.product((0, 1), repeat=count)])
    numbers = np.tile((spin_numbers[:, None]+magnetic_numbers[None, :]).reshape(-1), len(m['positions']))
    rotation = sparse.kron(sparse.eye(len(m['positions'])*m['ns']), field_rotation, format='csr')
    embedding = rotation[:, abs(numbers-1.5) < 1e-12]
    reduced = (embedding.conj().T @ h @ embedding).tocsr()
    ev, vv = ground(reduced, 2); state = embedding @ vv[:, 0]
    return dict(energy=float(ev[0]), J2=quadratic_generator(state, m['J']), dimension=reduced.shape[0])


def polarized_ensemble_response(m, operator):
    """Trace over orbital degeneracy before attributing a current to spin."""
    numbers = m['ground_magnetic_numbers']; basis = m['ground_basis']
    entries = []
    for target in (numbers.max(), numbers.min()):
        block = basis[:, abs(numbers-target) < 1e-7]
        projected = block.conj().T @ (operator @ block)
        entries.append(dict(Jz=float(target), orbital_multiplicity=block.shape[1],
                            mean=float(np.trace(projected).real/block.shape[1]),
                            moment_eigenvalues=np.linalg.eigvalsh(projected).tolist()))
    return dict(plus=entries[0], minus=entries[1], polarization_contrast=entries[0]['mean']-entries[1]['mean'],
                unpolarized_mean=float(np.trace(basis.conj().T @ (operator @ basis)).real/basis.shape[1]))


def summary(p, loading='difference', removed=False, fragments=True):
    h, m = build(p, loading=loading, removed=removed)
    ev, vv = ground(h, 8); psi, minus, z = polarized_ground(ev, vv, m)
    result = dict(energy=float(ev[0]), low_energies=ev.tolist(), dimension=h.shape[0], ground_Jz_eigenvalues=z.tolist(),
                  S2=quadratic_generator(psi, m['S']), carrier_L2=quadratic_generator(psi, m['L']), J2=quadratic_generator(psi, m['J']),
                  Sz=expectation(psi, m['S'][2]), carrier_Lz=expectation(psi, m['L'][2]), Jz=expectation(psi, m['J'][2]),
                  carrier_excitation_number=expectation(psi, m['carrier_energy']), eigen_residual=float(np.linalg.norm(h @ psi-ev[0]*psi)),
                  J_generator_source_commutators=[float(np.linalg.norm(h @ (op @ psi)-op @ (h @ psi))) for op in m['J']])
    result['restored_ground_block_max_residual'] = float(np.linalg.norm(h @ m['ground_basis']-ev[0]*m['ground_basis'], axis=0).max())
    if fragments:
        result['breakup'] = fragment_threshold(p, ev[0], loading, removed)
    return result, (h, m, psi, minus)


def run(p, output):
    # First fork is executed, not retrospectively erased; sum loading is the FM competitor.
    failed, _ = summary(dict(p, spin_orbit=0.), loading='sum')
    selected, (h, m, psi, minus) = summary(p)
    selected['lowest_Jz_three_halves'] = high_angular_sector(h, m)
    selected['high_angular_gap'] = selected['lowest_Jz_three_halves']['energy']-selected['energy']
    no_spin_loading, _ = summary(dict(p, loading_g=0.))
    no_carrier, _ = summary(dict(p, loading_g=0., spin_orbit=0.), removed=True)
    no_spin_dependent, _ = summary(dict(p, loading_g=0., spin_orbit=0.))
    no_orbit, (hh, mm, pp, pm) = summary(dict(p, spin_orbit=0.), fragments=False)
    magnetic, magnetic_contact = current_contact(m, [0, 0, 0], [0, 0, 0], magnetic=True)
    wrong_magnetic, _ = current_contact(mm, [0, 0, 0], [0, 0, 0], magnetic=True)
    response = []
    for q in ([0., 0., 0.], [1.1, .7, 0.]):
        operators, contacts = zip(*(current_contact(m, q, pol) for pol in np.eye(3)))
        sources = np.stack([op @ psi for op in operators], axis=1)
        tensor = sources.conj().T @ sources
        diagonal = [spectrum(h, selected['energy'], sources[:, a], p['krylov_steps']) for a in range(3)]
        response.append(dict(q=q, total_current_gram_real=tensor.real.tolist(), total_current_gram_imag=tensor.imag.tolist(),
                             contact_diagonal=[expectation(psi, c) for c in contacts], spectral_diagonal=diagonal,
                             polarized_mean_current_plus=[[float((np.vdot(psi, op @ psi)).real), float((np.vdot(psi, op @ psi)).imag)] for op in operators],
                             polarized_mean_current_minus=[[float((np.vdot(minus, op @ minus)).real), float((np.vdot(minus, op @ minus)).imag)] for op in operators]))
    q = np.array([1.1, .7, 0.]); charge = np.exp(1j*(COORDS[m['positions']] @ q)) @ m['charges']
    rho = np.repeat(charge, m['ns']*m['nf']); jr, _ = current_contact(m, q, q)
    continuity = np.linalg.norm(h @ (rho*psi)-selected['energy']*rho*psi-jr @ psi)
    # With phase exp(+iq r), [H,rho]=q.J for J=-dH/dA as defined here.
    amp = psi.reshape(len(m['positions'])*m['ns'], m['nf'])
    rdm = amp.conj().T @ amp; weights = np.linalg.eigvalsh(rdm); positive = weights[weights > 1e-13]
    bonds = []
    for edge, (v, w) in enumerate(EDGES):
        bond_j = sparse.csr_matrix(h.shape, dtype=complex)
        for term, charge_i, start, end in m['hops']:
            if (np.array_equal(start, COORDS[v]) and np.array_equal(end, COORDS[w])) or (np.array_equal(start, COORDS[w]) and np.array_equal(end, COORDS[v])):
                orientation = 1 if np.array_equal(start, COORDS[v]) else -1
                bond_j += -1j*charge_i*orientation*term
        bonds.append(dict(edge=[v, w], polarized_current_plus=expectation(psi, bond_j), polarized_current_minus=expectation(minus, bond_j)))
    result = dict(candidate='ORIENTED_AXIAL_NEUTRAL_BOND_COMPOSITE_V1', inputs=p, selected=selected,
                  first_sum_loading_fork=failed, no_spin_loading=no_spin_loading, no_carrier=no_carrier, no_spin_orbit=no_orbit,
                  magnetic_moment_plus=expectation(psi, magnetic), magnetic_moment_minus=expectation(minus, magnetic),
                  magnetic_contact=expectation(psi, magnetic_contact), no_spin_orbit_magnetic_moment=expectation(pp, wrong_magnetic),
                  magnetic_spectrum=spectrum(h, selected['energy'], magnetic @ psi, p['krylov_steps']),
                  physical_bond_currents=bonds, response=response, full_density_continuity_source_error=float(continuity),
                  carrier_matter_entropy=float(-sum(positive*np.log(positive))), charge=float(sum(m['charges'])))
    result['no_spin_dependent'] = no_spin_dependent
    result['magnetic_polarization_ensemble'] = polarized_ensemble_response(m, magnetic)
    result['no_spin_orbit_magnetic_polarization_ensemble'] = polarized_ensemble_response(mm, wrong_magnetic)
    result['isotropic_magnetic_generator_source_commutators'] = [
            float(np.linalg.norm(magnetic @ (op @ psi)-op @ (magnetic @ psi))) for op in m['J']]
    result['isotropic_spin_response_warning'] = 'The current commutes with internal J. Opposite time-reversed current states alone do not prove spin-dependent magnetism; orbital degeneracy can carry that contrast.'
    # Material alternative: occupied bonds can have a time-even axial pre-twist.
    # This permits spin-current locking but explicitly sacrifices internal SU(2).
    biased_parameters = dict(p, axial_bias=.35)
    biased, (bh, bm, bp, bminus) = summary(biased_parameters)
    bj, bc = current_contact(bm, [0, 0, 0], [0, 0, 0], magnetic=True)
    biased_no_orbit, (boh, bom, bop, _) = summary(dict(biased_parameters, spin_orbit=0.), fragments=False)
    boj, _ = current_contact(bom, [0, 0, 0], [0, 0, 0], magnetic=True)
    alternative_responses = []
    for q in ([0., 0., 0.], [1.1, .7, 0.]):
        entries = []
        for pol in np.eye(3):
            j, c = current_contact(bm, q, pol)
            entries.append(dict(polarization=pol.tolist(), contact=expectation(bp, c),
                                spectrum=spectrum(bh, biased['energy'], j @ bp, p['krylov_steps'])))
        alternative_responses.append(dict(q=q, Cartesian=entries))
    result['axial_material_alternative'] = dict(inputs=biased_parameters, state=biased,
            magnetic_moment_plus=expectation(bp, bj), magnetic_moment_minus=expectation(bminus, bj),
            magnetic_contact=expectation(bp, bc), magnetic_spectrum=spectrum(bh, biased['energy'], bj @ bp, p['krylov_steps']),
            no_spin_orbit_state=biased_no_orbit, no_spin_orbit_magnetic_moment=expectation(bop, boj),
            response=alternative_responses, scope='Axial pre-twist is a material input. Jz conserved; J2 not a conserved spin-half invariant.')
    result['axial_material_alternative']['magnetic_polarization_ensemble'] = polarized_ensemble_response(bm, bj)
    result['axial_material_alternative']['no_spin_orbit_magnetic_polarization_ensemble'] = polarized_ensemble_response(bom, boj)
    Path(output+'.json').write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(output+'.npz', ground_plus=psi, ground_minus=minus, positions=m['positions'], charges=m['charges'],
                        axial_ground_plus=bp, axial_ground_minus=bminus)
    print(json.dumps(dict(selected=selected, first_sum_loading=failed, no_spin_loading=no_spin_loading, no_carrier=no_carrier,
                          magnetic_moment_plus=result['magnetic_moment_plus'], magnetic_moment_minus=result['magnetic_moment_minus'],
                          continuity_error=float(continuity))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--request', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); run(json.loads(Path(args.request).read_text()), args.output)
