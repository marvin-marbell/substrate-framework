"""NEW finite quantum occupied-reference/source/scalar/current calculation."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import expm_multiply
from rich_scalar_core_20261014_model import build, sector, full_spectrum, source_block


def expectation(op, vectors):
    return float(np.trace(vectors.conj().T @ (op @ vectors)).real / vectors.shape[1])


def quantum_state(model, number):
    indices, values, vectors = full_spectrum(model, number)
    degenerate = values-values[0] < 1e-9
    return indices, values, vectors, vectors[:,degenerate]


def distribution(vectors, nphi):
    return np.mean(np.sum(abs(vectors.reshape(-1,nphi,vectors.shape[1]))**2,axis=0),axis=1)


def state_data(model, number, solved):
    indices, values, _, ground = solved
    phi = model['phi']; prob = distribution(ground,model['nphi'])
    t = model['T'][indices][:,indices]
    q = model['Q'][indices][:,indices]
    s2 = sparse.kron(model['matter']['S2'], sparse.eye(model['nphi']),format='csr')[indices][:,indices]
    def restricted(op):
        return op[indices][:,indices]
    identity = sparse.eye(model['nphi'],format='csr')
    energies = {
        'scalar_kinetic': expectation(restricted(sparse.kron(np.eye(16),model['scalar_kinetic'],format='csr')),ground),
        'positive_elastic': expectation(restricted(sparse.kron(np.eye(16),model['scalar_potential'],format='csr')),ground),
        'holonomy_kinetic': expectation(restricted(sparse.kron(model['matter']['kinetic'],identity,format='csr')),ground),
        'positive_frame_Casimir': model['parameters']['frame_cost']*expectation(s2,ground),
        'scalar_transfer': model['coupling']*expectation((q if model['frozen'] is None else model['frozen']*sparse.eye(len(indices)))@t,ground)}
    maximum = int(np.argmax(prob))
    return {'N': number, 'charge_relative_N2': 2-number, 'dimension': len(indices),
            'energy': float(values[0]), 'ground_multiplicity': ground.shape[1],
            'lowest_energies': values[:12].tolist(),
            'mean_phi': float(prob@phi), 'mean_phi2': float(prob@(phi**2)),
            'mean_phi4': float(prob@(phi**4)), 'phi_peak': float(phi[maximum]),
            'central_to_peak_density': float(prob[len(phi)//2]/prob[maximum]),
            'boundary_two_point_probability': float(prob[:2].sum()+prob[-2:].sum()),
            'mean_transfer': expectation(t,ground), 'phi_transfer': expectation(q@t,ground),
            'frame_S2': expectation(s2,ground), 'probability': prob.tolist(),
            'energy_components': energies,
            'ground_residual': float(np.linalg.norm(model['H'][indices][:,indices]@ground-ground*values[0]))}


def calculate(p, label, coupling=None, frozen=None, detailed=True):
    model=build(p,coupling=coupling,frozen=frozen)
    solved={n:quantum_state(model,n) for n in (1,2,3)}
    states={str(n):state_data(model,n,solved[n]) for n in (1,2,3)}
    in_idx, neutral_values, neutral_vectors, reference = solved[2]
    e0=float(neutral_values[0]); ng=reference.shape[1]
    # N0/N4 are the actual empty/filled scalar Hamiltonian, not energy repinned.
    scalar_values=eigh(model['Hscalar'].toarray(),eigvals_only=True,check_finite=False)
    vacuum_energy=float(scalar_values[0])
    result={'label':label, 'parameters':p, 'coupling':model['coupling'], 'frozen_phi':frozen,
            'states':states, 'global_sector_energies':[vacuum_energy, states['1']['energy'],e0,states['3']['energy'],vacuum_energy],
            'addition_gap':states['3']['energy']-e0, 'removal_gap':states['1']['energy']-e0,
            'neutral_first_strict_excitation':float(neutral_values[np.flatnonzero(neutral_values-e0>1e-9)[0]]-e0),
            'phi_grid':model['phi'].tolist(), 'units':'hbar=1; lattice spacing=1; all energies in declared input hopping units; phi is dimensionless',
            'reference':'full N2 ground-space maximally mixed projector; no phi-sign selection, no selected unoccupied one-particle vacuum',
            'no_spatial_core_claim':True}
    payload={label+'_reference':reference,label+'_phi':model['phi']}
    source_spectra={}
    gram=np.zeros((4,4),complex)
    zlist=np.array([-.6+.17j, .2+.17j,1.1+.17j,2.4+.17j,4.2+.17j])
    green=np.zeros((len(zlist),4,4),complex)
    for number,addition,name in [(3,True,'addition'),(1,False,'removal')]:
        indices, energies, vectors, ground=solved[number]
        sources=[source_block(model,a,in_idx,indices,addition)@reference for a in range(4)]
        amplitudes=np.stack([vectors.conj().T@v for v in sources],axis=2)
        weights=np.sum(abs(amplitudes)**2,axis=1)/ng
        deltas=energies-e0
        # Summed CAR is tested as an operator identity, not a normalized column.
        for a in range(4):
            for b in range(4):
                if addition:
                    gram[a,b]+=np.trace(sources[a].conj().T@sources[b])/ng
                else:
                    gram[a,b]+=np.trace(sources[b].conj().T@sources[a])/ng
        for zi,z in enumerate(zlist):
            for a in range(4):
                for b in range(4):
                    if addition:
                        residues=np.sum(amplitudes[:,:,a].conj()*amplitudes[:,:,b],axis=1)/ng
                        green[zi,a,b]+=np.sum(residues/(z-deltas))
                    else:
                        residues=np.sum(amplitudes[:,:,b].conj()*amplitudes[:,:,a],axis=1)/ng
                        green[zi,a,b]+=np.sum(residues/(z+deltas))
        source_spectra[name]={'energy':deltas.tolist(),'weights_by_source':weights.tolist(),
            'total_by_source':np.sum(weights,axis=0).tolist(),
            'lowest_source_supported_energy':float(np.min(deltas[weights.sum(axis=1)>1e-10])),
            'lowest_eigenspace_weight':float(weights[energies-energies[0]<1e-9].sum()),
            'immediate_mean_phi2_by_source':[float(distribution(v,model['nphi'])@(model['phi']**2)/(np.linalg.norm(v)**2/ng)) for v in sources]}
        payload[label+'_'+name+'_energies']=deltas
        payload[label+'_'+name+'_amplitudes']=amplitudes
        if addition:
            coeff=np.array([1.,1j,.4,-.2j]); coeff/=np.linalg.norm(coeff)
            injected=sum(coeff[a]*sources[a][:,0] for a in range(4))
            injected/=np.linalg.norm(injected)
            h=model['H'][indices][:,indices]
            j=model['current'][0][indices][:,indices]
            pulsed=expm_multiply(-1j*p['current_pulse']*j,injected)
            times=np.linspace(0,p['time_end'],p['time_points'])
            histories={}; full_prob={}
            for preparation,initial in [('source_only',injected),('source_then_current',pulsed)]:
                traj=expm_multiply(-1j*h,initial,start=0,stop=p['time_end'],num=p['time_points'],endpoint=True)
                probs=np.sum(abs(traj.reshape(len(times),-1,model['nphi']))**2,axis=1)
                currents=np.real(np.einsum('ti,ti->t',traj.conj(),(j@traj.T).T))
                histories[preparation]={'current_x':currents.tolist(),'phi2':(probs@(model['phi']**2)).tolist(),
                    'norm_error':float(np.max(abs(np.sum(probs,axis=1)-1)))}
                full_prob[preparation]=probs
                payload[label+'_'+preparation+'_phi_probability']=probs
            result['source_current_event']={'preparation':'normalized coherent N3 source on first neutral ground eigenvector; neutral unique or else explicit first-vector event only',
                'source_coefficients_real':coeff.real.tolist(),'source_coefficients_imag':coeff.imag.tolist(),
                'pulse':'exp(-i pulse Jx), Jx=dH/dkx actual uniform Peierls link derivative; not spin',
                'times':times.tolist(),'histories':histories,
                'last_field_probability_l1_difference':float(np.sum(abs(full_prob['source_only'][-1]-full_prob['source_then_current'][-1])))}
            payload[label+'_injected']=injected
    result['source_spectra']=source_spectra
    result['all_virtual_G']={'frequencies_real':zlist.real.tolist(),'frequencies_imag':zlist.imag.tolist(),'real':green.real.tolist(),'imag':green.imag.tolist(),
                            'CAR_gram_error':float(np.linalg.norm(gram-np.eye(4))), 'dimension':'4x4 source matrix; full N1/N3 scalar-Fock spectra, no upper cutoff'}
    # Actual response current in the neutral reference and contact, not Noether S.
    currents=[j[in_idx][:,in_idx] for j in model['current']]
    result['neutral_current']=[expectation(j,reference) for j in currents]
    result['neutral_current_connected_variance']=[expectation(j@j,reference)-expectation(j,reference)**2 for j in currents]
    result['current_contact']=[expectation(c[in_idx][:,in_idx],reference) for c in model['contact']]
    if detailed:
        frame=model['matter']['frame']
        result['source_geometry']={'raw_dimension':224,'physical_dimension':16,
            'gauss_residual':float(max(np.linalg.norm(g@frame['P']) for g in frame['G'])),
            'raw_source_leakage':float(max(np.linalg.norm((np.eye(224)-frame['P']@frame['P'].conj().T)@d@frame['P']) for d in frame['raw_source'])),
            'physical_source_CAR':float(max(np.linalg.norm(a.conj().T@b+b@a.conj().T-(np.eye(16) if i==j else 0)) for i,a in enumerate(frame['source']) for j,b in enumerate(frame['source']))),
            'scalar_rotation_commutator':float(max(np.linalg.norm(model['matter']['transfer']@s-s@model['matter']['transfer']) for s in frame['S'])),
            'odd_S2_halfspin_error':float(max(np.linalg.norm(model['matter']['S2'][np.ix_([x for x in range(16) if x.bit_count()==n],[x for x in range(16) if x.bit_count()==n])]-.75*np.eye(4)) for n in (1,3)))}
        step=2e-5; kp=np.array(p['k']); km=kp.copy(); kp[0]+=step; km[0]-=step
        hp=build(p,k=kp)['H']; hm=build(p,k=km)['H']
        result['link_derivative_errors']={'current':float(np.linalg.norm(((hp-hm)/(2*step)-model['current'][0]).data)),
            'contact':float(np.linalg.norm(((hp+hm-2*model['H'])/step**2-model['contact'][0]).data))}
    return result,payload


def main():
    p=json.loads(Path(sys.argv[1]).read_text()); output=Path(sys.argv[2])
    coupled,arrays=calculate(p,'dynamic')
    frozen_value=float(np.sqrt(coupled['states']['2']['mean_phi2']))
    decoupled,aa=calculate(p,'decoupled',coupling=0,detailed=False); arrays.update(aa)
    frozen,aa=calculate(p,'frozen_rms',frozen=frozen_value,detailed=False); arrays.update(aa)
    # Resolution is a new quantum regulator consumer, not a classical minimum.
    refined_p={**p,'scalar_points':p['refined_points']}
    refined,aa=calculate(refined_p,'refined_dynamic',detailed=False); arrays.update(aa)
    result={'kind':'DYNAMIC_SCALAR_TRANSFER_OCCUPIED_FOCK_MASS_GENERATION_CANDIDATE_NOT_SELF_BOUND_ELECTRON',
            'dynamic':coupled,'decoupled':decoupled,'frozen_rms':frozen,'refined_dynamic':refined,
            'comparisons':{'dynamic_minus_decoupled_addition_gap':coupled['addition_gap']-decoupled['addition_gap'],
                'dynamic_minus_frozen_addition_gap':coupled['addition_gap']-frozen['addition_gap'],
                'charged_minus_neutral_phi2':coupled['states']['3']['mean_phi2']-coupled['states']['2']['mean_phi2'],
                'resolution_addition_gap_difference':refined['addition_gap']-coupled['addition_gap'],
                'resolution_neutral_phi2_difference':refined['states']['2']['mean_phi2']-coupled['states']['2']['mean_phi2']}}
    output.write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(output.with_suffix('.npz'),**arrays)
    print(json.dumps({'kind':result['kind'],'gaps':{k:result[k]['addition_gap'] for k in ('dynamic','decoupled','frozen_rms','refined_dynamic')},'comparisons':result['comparisons'],'phi2':{k:coupled['states'][k]['mean_phi2'] for k in ('1','2','3')},'mean_phi':coupled['states']['2']['mean_phi'],'neutral_quantum_gap':coupled['neutral_first_strict_excitation'],'multiplicity':coupled['states']['2']['ground_multiplicity']}))


if __name__=='__main__':
    main()
