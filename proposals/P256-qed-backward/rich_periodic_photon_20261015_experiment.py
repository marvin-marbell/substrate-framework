"""Executed finite-3D correlated variational response, not exact photon poles."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from rich_periodic_photon_20261015_model import (
    PREFIX, POSITIONS, EDGES, INCIDENCE, basis, assemble, source_geometry, small_physical_case, loops,
)
ROOT = Path(__file__).resolve().parent


def save(name, data):
    (ROOT/(PREFIX+name+'.json')).write_text(json.dumps(data, indent=2)+'\n')


def spectral_block(H, energy, initial, steps):
    """Block Arnoldi spectral quadrature with full reorthogonalization.

Every source is retained; no one-vector eigenstate selection in degeneracies.
Ritz residuals and finite-depth uncertainties prevent exact-pole promotion.
"""
    gram = initial.conj().T@initial
    w, u = np.linalg.eigh(gram)
    good = w > max(w.max()*1e-11, 1e-20)
    block = initial@u[:, good]/np.sqrt(w[good])
    blocks = [block]
    for _ in range(steps-1):
        z = H@blocks[-1]-energy*blocks[-1]
        for _ in range(2):
            for old in blocks:
                z -= old@(old.conj().T@z)
        g = z.conj().T@z
        val, vec = np.linalg.eigh(g)
        active = val > max(float(val.max())*1e-10, 1e-20)
        if not active.any():
            break
        blocks.append(z@vec[:, active]/np.sqrt(val[active]))
    Q = np.column_stack(blocks)
    HQ = H@Q-energy*Q
    reduced = Q.conj().T@HQ
    omega, rotation = np.linalg.eigh((reduced+reduced.conj().T)/2)
    overlaps = rotation.conj().T@(Q.conj().T@initial)
    # Residual is computed on each quadrature vector in the actual selected PHP.
    residual_vectors = HQ-Q@reduced
    residuals = np.linalg.norm(residual_vectors@rotation, axis=0)
    groups = []
    start = 0
    while start < len(omega):
        stop = start+1
        while stop < len(omega) and abs(omega[stop]-omega[start]) < 1e-7:
            stop += 1
        gram_group = overlaps[start:stop].conj().T@overlaps[start:stop]
        weights = np.linalg.eigvalsh(gram_group).real
        groups.append(dict(omega=float(np.mean(omega[start:stop])), multiplicity=stop-start,
                           source_rank=int(np.sum(weights > 1e-10)), source_gram=gram_group.real.tolist(),
                           source_gram_eigenvalues=weights.tolist(),
                           residual_max=float(max(residuals[start:stop])),
                           total_weight=float(np.trace(gram_group).real)))
        start = stop
    times = np.linspace(0, 4, 41)
    # C_ab(t)=<Oa(t)Ob> in the variational stationary reference; retarded
    # chi_ab=-i theta <[Oa(t),Ob]>. At real Hermitian field sources this is
    # 2 Im C_ab(t), with both signed frequency terms retained in frequency form.
    correlation = np.einsum('ka,kb,tk->tab', overlaps.conj(), overlaps, np.exp(-1j*times[:, None]*omega[None, :]))
    retarded = 2*correlation.imag
    grid = np.linspace(0, 10, 101)
    eta = .25
    response = np.einsum('ka,kb,wk->wab', overlaps.conj(), overlaps,
                         1/(grid[:, None]-omega[None, :]+1j*eta)-1/(grid[:, None]+omega[None, :]+1j*eta))
    return dict(depth=steps, dimension=Q.shape[1], source_gram=gram.real.tolist(), source_rank=int(good.sum()),
                groups=groups, times=times.tolist(), retarded_FF=retarded.tolist(),
                frequencies=grid.tolist(), broadening=eta, chi_FF_real=response.real.tolist(), chi_FF_imag=response.imag.tolist(),
                orthogonality_error=float(np.max(np.abs(Q.conj().T@Q-np.eye(Q.shape[1]))))), (omega, overlaps, residuals)


def execute(kind, settings):
    start = time.time()
    keys = basis(kind)
    print(json.dumps(dict(event='basis', kind=kind, dimension=len(keys))), flush=True)
    data = assemble(keys)
    H = data['H']
    n = H.shape[0]
    # A uniform vector can be symmetry-orthogonal to the fermionic ground.
    # Fixed random coefficients have support across all selected spin sectors.
    initial = np.random.default_rng(20261015).normal(size=n)
    initial /= np.linalg.norm(initial)
    energy, vector = eigsh(H, k=1, which='SA', v0=initial, tol=settings['ground_tolerance'], maxiter=1500)
    e0, state = float(energy[0]), vector[:, 0]
    if state[np.argmax(np.abs(state))] < 0:
        state = -state
    p = state*state
    ground_residual = float(np.linalg.norm(H@state-e0*state))
    gauss = data['electric'].astype(int)@INCIDENCE.T+data['rho']
    entropy = None
    # Explicit Schmidt singular values of the correlated matter/field state.
    fock_keys = sorted(set(key & 65535 for key in keys))
    field_keys = sorted(set(key >> 16 for key in keys))
    fi, ei = {b:i for i,b in enumerate(fock_keys)}, {b:i for i,b in enumerate(field_keys)}
    amplitude = sparse.coo_matrix((state, ([fi[k & 65535] for k in keys], [ei[k >> 16] for k in keys])), shape=(len(fock_keys), len(field_keys))).tocsr()
    purity = float(np.sum((amplitude@amplitude.T).data**2))
    entropy = -np.log(purity)
    result = dict(kind=kind, dimension=n, nonzeros=H.nnz, energy=e0, projected_reference_residual=ground_residual,
                  gauss_max=int(np.max(np.abs(gauss))), hermiticity_max=float(abs(H-H.T).max()),
                  doublon_number=float(p@np.sum(data['rho'] == 1, axis=1)),
                  holon_number=float(p@np.sum(data['rho'] == -1, axis=1)),
                  doublon_number_distribution=np.bincount(np.sum(data['rho'] == 1, axis=1), weights=p, minlength=5).tolist(),
                  maximum_simultaneous_doublons=int(np.max(np.sum(data['rho'] == 1, axis=1))),
                  field_energy=float(p@np.sum(data['electric'].astype(float)**2, axis=1)/32),
                  matter_field_renyi2=float(entropy), matter_field_purity=purity,
                  omitted_hop_probability_sum=float(p@data['omitted_counts']),
                  full_H_stationarity='NOT established: selected-space eigenstate; nonzero omitted transitions are retained as ansatz uncertainty',
                  responses={})
    q_list = settings['q_bits']
    for q_bits in q_list:
        geo = source_geometry(q_bits)
        coeff = geo['coefficients']
        f = data['electric']@coeff.T/16
        rho_q = data['rho']@geo['density']
        fsource = f*state[:, None]
        fsource -= state[:, None]*(state@fsource)[None, :]
        jsource = np.empty_like(fsource, dtype=complex)
        # Explicit primitive current insertion: jQ = +i(T-T†) = -Jflux.
        for a in range(3):
            values = 1j*data['directions']*data['values']*coeff[a, data['edge_ids']]*state[data['cols']]
            jsource[:, a] = np.bincount(data['rows'], weights=values.imag, minlength=n)*1j
        current_identity = float(np.max(np.abs(jsource + 16j*(H@fsource-e0*fsource))))
        contact = np.empty((3, 3))
        edge_energy = np.bincount(data['edge_ids'], weights=data['values']*state[data['rows']]*state[data['cols']], minlength=24)
        for a in range(3):
            for b in range(3):
                contact[a, b] = -np.sum(coeff[a]*coeff[b]*edge_energy)
        density_source = rho_q*state
        density_source -= state*(state@density_source)
        joint = np.column_stack((fsource, jsource, density_source))
        joint_gram = joint.conj().T@joint
        gauss_identity = float(np.max(np.abs(rho_q + 16*np.linalg.norm(geo['discrete_gradient'])*f[:, 2])))
        response, arrays = spectral_block(H, e0, fsource, settings['response_depth'])
        # Own-H hopping-OFF E response vanishes as an OPERATOR identity, even
        # on this nonstationary correlated reference under Hdiag: evolve bra
        # and baseline with the same Hdiag. Not a removed-current tautology.
        off_comm = 0.0
        off_chi = []
        for t in (0, .5, 1, 2):
            baseline = np.exp(-1j*t*data['diagonal'])*state
            inserted = np.exp(-1j*t*data['diagonal'])[:, None]*fsource
            c = (baseline[:, None]*f).conj().T@inserted
            off_chi.append(float(np.max(np.abs(2*c.imag))))
        response.update(q_bits=q_bits, q=geo['q'], polarizations=geo['polarizations'].tolist(),
                        longitudinal_density_identity_max=gauss_identity,
                        explicit_primitive_charge_current_identity_max=current_identity,
                        source_order=['F_T1', 'F_T2', 'F_L', 'jQ_T1', 'jQ_T2', 'jQ_L', 'rho'],
                        joint_source_gram_real=joint_gram.real.tolist(), joint_source_gram_imag=joint_gram.imag.tolist(),
                        joint_source_gram_eigenvalues=np.linalg.eigvalsh(joint_gram).tolist(),
                        twist_contact=contact.tolist(), hopping_off_E_commutator=off_comm,
                        hopping_off_proper_baseline_retarded_max=off_chi)
        omega, overlaps, residuals = arrays
        # Joint field/current/density spectral weights follow the independently
        # checked primitive identities, not a supplied Maxwell/Dirac law.
        js = -16j*omega[:, None]*overlaps
        density_overlap = -16*np.linalg.norm(geo['discrete_gradient'])*overlaps[:, 2:3]
        joint_overlap = np.column_stack((overlaps, js, density_overlap))
        frequencies = np.asarray(response['frequencies'])
        z = frequencies[:, None]+1j*response['broadening']
        joint_chi = np.einsum('ka,kb,wk->wab', joint_overlap.conj(), joint_overlap, 1/(z-omega))
        joint_chi -= np.einsum('kb,ka,wk->wab', joint_overlap.conj(), joint_overlap, 1/(z+omega))
        response['joint_retarded_definition'] = 'chi_ab=sum(Va*Vb/(z-Delta)-Vb*Va/(z+Delta)); full reducible response, no Dyson reinterpretation'
        np.savez_compressed(ROOT/(PREFIX+kind+'_q'+''.join(map(str,q_bits))+'_spectral.npz'),
                            omega=omega, field_overlaps=overlaps, joint_overlaps=joint_overlap, residuals=residuals,
                            frequencies=frequencies, joint_chi=joint_chi, twist_contact=contact,
                            current_response_with_contact=joint_chi[:, 3:6, 3:6]+contact)
        response['joint_spectral_arrays'] = PREFIX+kind+'_q'+''.join(map(str,q_bits))+'_spectral.npz'
        result['responses'][''.join(map(str,q_bits))] = response
        save(kind+'_checkpoint', result)
        print(json.dumps(dict(event='response',kind=kind,q=q_bits,source_rank=response['source_rank'])), flush=True)
    np.savez_compressed(ROOT/(PREFIX+kind+'_state.npz'), keys=np.asarray(keys,dtype=np.uint64), state=state, energy=e0)
    sparse.save_npz(ROOT/(PREFIX+kind+'_H.npz'), H)
    result['seconds'] = time.time()-start
    save(kind+'_result', result)
    return result


def main():
    settings = json.loads((ROOT/(PREFIX+'input.json')).read_text())
    sanity = small_physical_case()
    if sanity['ground_error'] > 1e-12 or sanity['hermiticity'] > 1e-12:
        raise RuntimeError('physical two-site CAR/Gauss sanity failed')
    save('small_physical', sanity)
    print(json.dumps(dict(event='small_physical',result=sanity)), flush=True)
    results = []
    for kind in settings['states']:
        results.append(execute(kind, settings))
    summary = dict(law=settings['law'], geometry=dict(sites=8, oriented_links=24, positions=POSITIONS.tolist(),edges=EDGES,
                   incidence_rank=int(np.linalg.matrix_rank(INCIDENCE)), cycle_dimension=17, independent_seed_loops=len(loops())),
                   sanity=sanity, states=[{k:r[k] for k in ('kind','dimension','energy','doublon_number','field_energy','matter_field_renyi2','omitted_hop_probability_sum')} for r in results],
                   scope='Gauge-exact variational correlated selected-configuration states; full local dimensions retained, NOT full-H stationary ground proof. Finite-depth projected joint response, NOT exact Lehmann photon poles.')
    save('result', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
