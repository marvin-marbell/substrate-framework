"""Mobile odd-fermion elastic breather: a separate finite candidate, not a proton.

H = -t sum allowed directed hops + omega b^dag b - g K(x)(b+b^dag)
    + J sum f(dij) Si.Sj + alpha sum qi qj/sqrt(dij^2+a0^2).
K=sum f(dij), f(d)=exp(-d^2/(2 R^2)); coincident positions excluded.
No archived scientific entrypoints or helper code are imported.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import eigsh


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spin_basis(count):
    # Full spin sector for dimers, Sz=+1/2 for trimer, +1/2 for singleton.
    if count == 3:
        return [(0, 1, 1), (1, 0, 1), (1, 1, 0)]
    if count == 2:
        return [(0, 0), (0, 1), (1, 0), (1, 1)]
    return [(1,)]


def spin_pair(spins, i, j):
    out = np.zeros((len(spins), len(spins)))
    lookup = {s: k for k, s in enumerate(spins)}
    for k, s in enumerate(spins):
        out[k, k] = 0.25 if s[i] == s[j] else -0.25
        if s[i] != s[j]:
            flipped = list(s); flipped[i], flipped[j] = flipped[j], flipped[i]
            out[lookup[tuple(flipped)], k] = 0.5
    return out


def build(p, flavors=(0, 1, 2)):
    """Return actual H, ordered hard-core positions, spin block and metadata.

    Primitive CAR ordering is flavor-major. Within fixed one-per-flavor
    sectors the hopping has no extra sign; three-body source remains odd.
    """
    n, levels = p['N'], p['phonon_levels']
    count = len(flavors)
    positions = np.array(list(itertools.permutations(range(n), count)), dtype=int)
    lookup = {tuple(x): k for k, x in enumerate(positions)}
    spins = spin_basis(count); ns = len(spins)
    pair_ops = {(i, j): spin_pair(spins, i, j) for i in range(count) for j in range(i+1, count)}
    q = np.array(p['charges'])[list(flavors)]
    rows, cols, data = [], [], []
    load = np.zeros(len(positions)); distances = []
    coulomb = np.zeros(len(positions))
    for k, x in enumerate(positions):
        dp = {}
        for i, j in pair_ops:
            d = abs(int(x[i])-int(x[j])); d = min(d, n-d)
            f = np.exp(-0.5*(d/p['range'])**2)
            dp[i, j] = d; load[k] += f
            coulomb[k] += p['alpha']*q[i]*q[j]/np.sqrt(d*d+p['coulomb_softening']**2)
            op = p['spin_J']*f*pair_ops[i, j]
            for a, b in zip(*np.nonzero(op)):
                rows.append(k*ns+a); cols.append(k*ns+b); data.append(op[a, b])
        distances.append(dp)
        for i in range(count):
            for step in (-1, 1):
                y = x.copy(); y[i] = (y[i]+step) % n
                dest = lookup.get(tuple(y))
                if dest is not None:
                    for s in range(ns):
                        rows.append(dest*ns+s); cols.append(k*ns+s); data.append(-p['hopping'])
    matter = sparse.coo_matrix((data, (rows, cols)), shape=(len(positions)*ns,)*2).tocsr()
    matter += sparse.diags(np.repeat(coulomb, ns))
    b = sparse.diags(np.sqrt(np.arange(1, levels)), 1, shape=(levels, levels))
    phonon = sparse.diags(p['omega']*np.arange(levels))
    h = sparse.kron(matter, sparse.eye(levels), format='csr')
    h += sparse.kron(sparse.eye(matter.shape[0]), phonon, format='csr')
    h -= p['binding_g']*sparse.kron(sparse.diags(np.repeat(load, ns)), b+b.T, format='csr')
    return h, positions, spins, {'flavors': flavors, 'q': q, 'load': load, 'distances': distances,
                                'pair_ops': pair_ops, 'levels': levels, 'ns': ns, 'lookup': lookup}


def ground(h):
    values, vectors = eigsh(h, k=1, which='SA', tol=1e-10,
                            v0=np.linspace(0.3, 1.1, h.shape[0]))
    return float(values[0]), vectors[:, 0]


def density(qmomentum, positions, meta):
    """rho(q)=sum qi exp(+i q xi); returned diagonal vector, charges in e units."""
    return np.repeat(np.exp(1j*qmomentum*positions)@meta['q'], meta['ns']*meta['levels'])


def current_contact(p, momentum, positions, spins, meta):
    """Physical Peierls current and (+q,-q) second derivative contact.

    H[A] forward hop = -t exp(+i qi integral_edge A) |x+1><x|.
    J(q)=-dH/dA(-q) with rho(+q) convention obeys [H,rho(q)]=q J(q).
    Edge averaging is sinc(q/2), not a guessed continuum current.
    Contact is d2H/dA(q)dA(-q), energy/electric-source squared units.
    """
    ns, levels, n = len(spins), meta['levels'], p['N']
    rows, cols, jvals, cvals = [], [], [], []
    sinc = np.sinc(momentum/(2*np.pi))
    for k, x in enumerate(positions):
        for i, charge in enumerate(meta['q']):
            y = x.copy(); y[i] = (y[i]+1) % n
            dest = meta['lookup'].get(tuple(y))
            if dest is None:
                continue
            phase = np.exp(1j*momentum*(x[i]+0.5))
            for s in range(ns):
                a, b = dest*ns+s, k*ns+s
                rows.extend([a, b]); cols.extend([b, a])
                jvals.extend([1j*p['hopping']*charge*sinc*phase,
                              -1j*p['hopping']*charge*sinc*phase])
                cvals.extend([p['hopping']*charge**2*sinc**2]*2)
    shape = (len(positions)*ns,)*2
    j = sparse.coo_matrix((jvals, (rows, cols)), shape=shape).tocsr()
    c = sparse.coo_matrix((cvals, (rows, cols)), shape=shape).tocsr()
    return sparse.kron(j, sparse.eye(levels), format='csr'), sparse.kron(c, sparse.eye(levels), format='csr')


def spectral_measure(h, energy, source, steps):
    """Lanczos finite Krylov spectral measure, not all exact eigenpoles.

    Full reorthogonalization; weights retain the actual source norm.
    """
    norm2 = float(np.vdot(source, source).real)
    if norm2 < 1e-25:
        return {'source_norm2': norm2, 'gaps': [], 'weights': [], 'steps': 0}
    v = source/np.sqrt(norm2); vectors = []; diagonal = []; off = []
    previous = np.zeros_like(v); beta = 0.0
    for _ in range(steps):
        vectors.append(v)
        w = h@v-energy*v-beta*previous
        a = float(np.vdot(v, w).real); diagonal.append(a); w -= a*v
        for old in vectors:
            w -= np.vdot(old, w)*old
        bn = float(np.linalg.norm(w))
        if bn < 1e-12:
            break
        if len(diagonal) < steps:
            off.append(bn)
        previous, v, beta = v, w/bn, bn
    tridiagonal = np.diag(diagonal)+np.diag(off, 1)+np.diag(off, -1)
    gaps, rotation = np.linalg.eigh(tridiagonal)
    return {'source_norm2': norm2, 'gaps': gaps.tolist(),
            'weights': (norm2*rotation[0]**2).tolist(), 'steps': len(diagonal)}


def retarded(measure, frequencies, eta):
    gaps = np.array(measure['gaps']); weights = np.array(measure['weights'])
    out = []
    for omega in frequencies:
        z = omega+1j*eta
        value = np.sum(weights*(1/(z-gaps)-1/(z+gaps)))
        out.append({'omega': omega, 'real': float(value.real), 'imag': float(value.imag)})
    return out


def intrinsic(p, positions, probability, charges):
    """Anchored local-chart intrinsic density, not <rho(q)> of a plane wave."""
    n = p['N']; offsets = (positions-positions[:, :1]+n/2) % n-n/2
    relative = offsets-offsets.mean(axis=1, keepdims=True)
    span = offsets.max(axis=1)-offsets.min(axis=1)
    rows = []
    for q in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0):
        parts = probability@(np.exp(1j*q*relative)*charges)
        rows.append({'q': q, 'F_real': float(parts.sum().real), 'F_imag': float(parts.sum().imag),
                     'constituent_real': parts.real.tolist(), 'constituent_imag': parts.imag.tolist()})
    return {'chart': 'unwrap each flavor relative to flavor0, subtract equal-mass center; local bound-regime observable',
            'form_factors': rows, 'signed_charge_radius_squared': float(probability@(relative**2@charges)),
            'rms_matter_radius': float(np.sqrt(probability@np.mean(relative**2, axis=1))),
            'chart_noncompact_probability_span_ge_half_ring': float(probability@(span>=n/2))}


def run(p, output):
    h, positions, spins, meta = build(p)
    energy, psi = ground(h)
    levels, ns = meta['levels'], meta['ns']
    amplitudes = psi.reshape(len(positions), ns, levels)
    prob = np.sum(abs(amplitudes)**2, axis=(1, 2))
    osc_rdm = np.einsum('xsn,xsm->nm', amplitudes, amplitudes.conj())
    eigen = np.linalg.eigvalsh(osc_rdm); positive = eigen[eigen>1e-14]
    phonon_number = float(np.diag(osc_rdm).real@np.arange(levels))
    pair_rows = []
    for (i, j), op in meta['pair_ops'].items():
        correlation = float(np.einsum('xsn,st,xtn->', amplitudes.conj(), op, amplitudes).real)
        distance = np.array([d[i, j] for d in meta['distances']])
        pair_rows.append({'flavors': [i, j], 'spin_dot': correlation,
                          'mean_distance': float(prob@distance),
                          'distance_distribution': {str(d): float(prob@(distance==d)) for d in range(1, p['N']//2+1)}})
    s2small = 2.25*np.eye(ns)+2*sum(meta['pair_ops'].values())
    s2 = float(np.einsum('xsn,st,xtn->', amplitudes.conj(), s2small, amplitudes).real)
    spin_z = np.array(spins)-0.5
    spin_per_flavor = np.einsum('xsn,si,xsn->i', amplitudes.conj(), spin_z, amplitudes).real
    # mu_z = sum qi Si_z/m, Dirac g=2 stipulated for constituent magnetic probes.
    magnetic_moment = float(meta['q']@spin_per_flavor)
    thresholds = []
    for pair in itertools.combinations(range(3), 2):
        hp, _, _, _ = build(p, pair); ep, _ = ground(hp)
        remaining = tuple(i for i in range(3) if i not in pair)
        hs, _, _, _ = build(p, remaining); es, _ = ground(hs)
        thresholds.append({'pair': list(pair), 'pair_energy': ep, 'single_energy': es,
                           'fragment_energy': ep+es, 'binding_below_fragment': ep+es-energy})
    disconnected = dict(p); disconnected['binding_g'] = 0.0
    hf, _, _, _ = build(disconnected); ef, _ = ground(hf)
    response = []
    for mode in (1, 2):
        q = 2*np.pi*mode/p['N']; rho = density(q, positions, meta)
        j, contact = current_contact(p, q, positions, spins, meta)
        source = rho*psi; source -= psi*np.vdot(psi, source)
        jsource = j@psi; jsource -= psi*np.vdot(psi, jsource)
        measure = spectral_measure(h, energy, source, p['lanczos_steps'])
        jm = spectral_measure(h, energy, jsource, p['lanczos_steps'])
        ward = h@(rho*psi)-rho*(h@psi)-q*(j@psi)
        response.append({'mode': mode, 'q': q, 'density_spectral_measure': measure,
                         'current_spectral_measure': jm,
                         'chi_rho_retarded': retarded(measure, p['frequencies'], p['eta']),
                         'chi_J_retarded': retarded(jm, p['frequencies'], p['eta']),
                         'contact_expectation': float(np.vdot(psi, contact@psi).real),
                         'ground_lab_density_expectation_abs': float(abs(np.vdot(psi, rho*psi))),
                         'continuity_source_residual': float(np.linalg.norm(ward)),
                         'energy_weighted_density_mass': float(np.vdot(source, h@source-energy*source).real)})
    result = {'candidate': p['candidate'], 'parameters': p, 'dimension': h.shape[0],
              'ground_energy': energy, 'ground_residual_norm': float(np.linalg.norm(h@psi-energy*psi)),
              'bound_against_all_pair_single_channels_in_selected_finite_domain': all(t['binding_below_fragment']>0 for t in thresholds),
              'thresholds': thresholds, 'no_breather_same_other_law_energy': ef,
              'spin_squared': s2, 'spin_z_per_flavor': spin_per_flavor.tolist(),
              'total_spin_z': float(spin_per_flavor.sum()), 'total_charge': float(meta['q'].sum()),
              'magnetic_moment_z_stipulated_constituent_g2_m1': magnetic_moment,
              'composite_CAR_parity': -1, 'two_disjoint_trimer_exchange_sign': (-1)**9,
              'pair_correlations': pair_rows, 'intrinsic_charge_structure': intrinsic(p, positions, prob, meta['q']),
              'mean_pair_load': float(prob@meta['load']), 'phonon_occupation': phonon_number,
              'top_phonon_level_probability': float(osc_rdm[-1, -1].real),
              'matter_breather_entanglement_entropy': float(-positive@np.log(positive)),
              'response': response, 'state_archive': output.with_suffix('.npz').name,
              'limitations': ['1D finite ring and one neutral collective coordinate, not 3D local substrate field',
                              'binding gaps are finite-domain fragment energies, not infinite-volume scattering thresholds',
                              'distinct flavors carry input fractional charges, no SU3 color Gauss or QCD identification',
                              'hard-core occupancy is a stipulated infinite-repulsion regime',
                              'no radiative photon Hilbert space, sea pairs or relativistic composite boost construction',
                              'Lanczos response is a finite Krylov approximation; no full pole/resolvent theorem',
                              'magnetic constituent g factors are inputs, not derived proton magnetic moment']}
    np.savez_compressed(output.with_suffix('.npz'), psi=psi, positions=positions, spins=np.array(spins),
                        phonon_density=osc_rdm, position_probability=prob, energy=energy)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'energy': energy, 'binding': [x['binding_below_fragment'] for x in thresholds],
                      'S2': s2, 'Q': float(meta['q'].sum()), 'dimension': h.shape[0]}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    run(request, args.output)
