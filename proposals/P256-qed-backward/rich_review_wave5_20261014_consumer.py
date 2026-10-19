"""Independent finite-law consumers; never invokes an author event/producer CLI."""
from __future__ import annotations
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.linalg import eigh, expm
from scipy.sparse.linalg import expm_multiply
from rich_frame_gauss_20261014_model import frame_gauss, carrier, spatial_consumer, angular
from rich_quantum_collision_20261014_model import Sector, Parameters, maximum_entry

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_review_wave5_20261014'


def encode(x):
    if isinstance(x, np.ndarray):
        return encode(x.tolist())
    if isinstance(x, complex):
        return [x.real, x.imag]
    if isinstance(x, np.generic):
        return encode(x.item())
    if isinstance(x, dict):
        return {k: encode(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [encode(v) for v in x]
    return x


def norm(a):
    return float(np.linalg.norm(a))


def frame_review(request):
    p = json.loads((ROOT/'rich_frame_gauss_20261014_input.json').read_text())
    f = frame_gauss(); projector = f['P']; n = np.diag(f['N'])
    sources = f['source']; size = len(n)
    gauss = max(norm(g@projector) for g in f['G'])
    j2 = sum(s@s for s in f['S'])
    car = max(norm(a.conj().T@b+b@a.conj().T-(np.eye(size) if i == j else 0)) for i,a in enumerate(sources) for j,b in enumerate(sources))
    bare = np.column_stack([(projector.conj().T@a@projector)[:,0] for a in f['bare']])
    # The centre-even SO3 coordinate domain contains only integer j rows.
    integer = np.array([j.is_integer() for j,_,_ in f['basis']]).repeat(4)
    odd_ix = np.flatnonzero(abs(f['N']-1)<1e-8)
    centre_even_odd = norm(projector[integer][:,odd_ix])
    excluded = {}
    for j in [1.5, 2.]:
        right = [np.kron(np.eye(int(2*j+1)),a) for a in angular(j)]
        colour = angular(.5)
        generators = [np.kron(a,np.eye(2))+np.kron(np.eye(len(a)),b) for a,b in zip(right,colour)]
        excluded[str(j)] = float(eigh(sum(g@g for g in generators),eigvals_only=True)[0])
    sectors = {}
    for occupation in [0,1,2]:
        spectra = []
        for l in range(p['lmax']+1):
            c = carrier(p,l,occupation)
            values, vectors = eigh(c['H'])
            total2 = sum(a@a for a in c['total'])
            spectra.append({'l':l,'K_copies':2*l+1,'dimension_one_K':len(values),'ground_energy':values[0], 'ground_total_J2':np.vdot(vectors[:,0],total2@vectors[:,0]).real,'rotation_commutator':max(norm(c['H']@a-a@c['H']) for a in c['total'])})
        sectors[str(occupation)] = spectra
    neutral = carrier(p,0,0); en, un = eigh(neutral['H']); neutral_state = un[:,0]
    odd = carrier(p,0,1); eo, uo = eigh(odd['H'])
    # Actual source intertwines number and frame; field is unchanged at insertion.
    v = np.column_stack([np.kron(a[np.ix_(odd['sector'],neutral['sector'])],np.eye(odd['nb']))@neutral_state for a in sources])
    z = complex(*request['frame_z'])
    good_g = v.conj().T@np.linalg.solve(z*np.eye(len(eo))-odd['H']+en[0]*np.eye(len(eo)),v)
    wrong_g = v.conj().T@np.linalg.solve(z*np.eye(len(eo))-odd['H']+eo[0]*np.eye(len(eo)),v)
    wavevector = np.array(request['frame_wavevector'])
    spatial = spatial_consumer(p,wavevector)
    vals, vecs = eigh(spatial['H']); g = vecs[:,0]
    density = spatial['rho']; current = spatial['current']; h = spatial['H']
    # rho(k)=int exp(-ikx)rho(x): [H,rho]+k.j=0.
    ward = h@density-density@h+sum(k*a for k,a in zip(wavevector,current))
    legs = np.column_stack([a@g for a in current]); gram = legs.conj().T@legs
    wrong_spin_current = [np.kron(np.kron(np.eye(len(spatial['spatial']['basis'])),a[np.ix_(odd_ix,odd_ix)]),np.eye(odd['nb'])) for a in f['S']]
    badward = h@density-density@h+sum(k*a for k,a in zip(wavevector,wrong_spin_current))
    off = carrier(dict(p,common_inertia=1e12),0,1); eoff,uoff=eigh(off['H'])
    mu = [uo[:,:2].conj().T@a@uo[:,:2] for a in odd['mu']]
    muoff = [uoff[:,:2].conj().T@a@uoff[:,:2] for a in off['mu']]
    result = {'Gauss_residual':gauss,'CAR_residual':car,'physical_J2_eigenvalues':eigh(j2,eigvals_only=True),'centre_even_domain_odd_norm':centre_even_odd,'bare_source_Gram':bare.conj().T@bare,'excluded_higher_frame_G2_min':excluded,'all_number_orbital_sectors':sectors,'source_Gram':v.conj().T@v,'source_G':good_g,'wrong_odd_reference_G_distance':norm(good_g-wrong_g),'spatial_dimension':len(h),'wavevector':wavevector,'continuity_absolute':norm(ward),'continuity_relative':norm(ward)/max(norm(h@density-density@h),1e-30),'wrong_Noether_as_current_continuity':norm(badward),'current_leg_Gram':gram,'current_leg_Gram_eigenvalues':eigh(gram,eigvals_only=True),'current_longitudinal_ground_leg':norm(legs@wavevector),'magnetic_projected_norm':norm(np.array(mu)),'same_halfspin_cross_metric_off_magnetic_norm':norm(np.array(muoff)),'odd_source_satellite_total_weight':np.sum(np.abs(uo[:,2:].conj().T@v)**2),'field_feedback_ground_Q':[np.vdot(uo[:,0],np.kron(np.eye(2),q)@uo[:,0]).real for q in odd['field']['Q']]}
    np.savez_compressed(ROOT/(PREFIX+'_frame_states.npz'),source_vectors=v,source_G=good_g,spatial_H=h,spatial_state=g,current_legs=legs,current_Gram=gram)
    return result


def expectation(waves, operator):
    return np.einsum('tai,tai->ti',waves.conj(),np.stack([operator@w for w in waves])).real


def collision_review(request):
    author = json.loads((ROOT/'rich_quantum_collision_20261014_input.json').read_text())
    p = Parameters(**author['parameters']); s = Sector(p)
    e,g,_,_=s.polarized_ground()
    incoming = np.column_stack([a@g-g*np.vdot(g,a@g) for a in s.F[:2]])
    outgoing = np.column_stack([a@g-g*np.vdot(g,a@g) for a in s.F[2:]])
    vals,vecs = eigh(outgoing.conj().T@outgoing)
    out = outgoing@vecs@np.diag(1/np.sqrt(vals))
    # Independent asymmetric finite preparation, not the author eta/time grid.
    q = s.F[1]+request['collision_electric_weight']*s.loop_electric[0]
    pulse = expm_multiply(-1j*request['collision_eta']*q,g,traceA=-1j*request['collision_eta']*q.diagonal().sum())
    raw = incoming@np.array([1.,1j*.55]); packet = raw/np.linalg.norm(raw)
    seeds=np.column_stack([g,pulse,packet]); times=np.array(request['collision_times'])
    detector=(np.sum(s.electric[:,[10,11]]**2,axis=1)>0).astype(float)
    up=.5+s.spin[2].diagonal().real
    weights=np.column_stack([detector*up,detector*(1-up),(1-detector)*up,(1-detector)*(1-up)])
    cases={}; stored={'seeds':seeds,'incoming':incoming,'outgoing':out,'basis':np.array(s.basis,dtype=np.int64)}
    for label,model in [('full',s),('full_cut',Sector(replace(p,cut=True))),('axial',Sector(p,vector=False)),('current_off',Sector(replace(p,active_hop=0.,supplier_hop=0.)))]:
        if model.basis!=s.basis:
            raise RuntimeError('Control basis differs')
        # Direct full-H expm at selected times; no author event/projection helper.
        waves=np.stack([expm_multiply(-1j*t*model.H,seeds,traceA=-1j*t*model.H.diagonal().sum())*np.exp(1j*t*e) for t in times])
        probabilities=np.abs(waves)**2
        joint=np.einsum('tai,ao->tio',probabilities,weights)
        spin=expectation(waves,s.spin[2])
        amps=np.einsum('ar,tai->tri',out.conj(),waves)
        projected=np.einsum('ar,tri->tai',out,amps)
        outprob=np.sum(abs(amps)**2,axis=1)
        projectedspin=expectation(projected,s.spin[2])
        povm=np.stack([outprob/2+projectedspin,outprob/2-projectedspin,1-outprob],axis=2)
        wrong_label=np.abs(amps[:,0,:])**2
        energy=expectation(waves,model.H)
        far=joint[:,:,:2].sum(axis=2)
        delta=far-far[:,[0]]
        cases[label]={'joint_PVM':joint,'out_then_Sz_POVM':povm,'fixed_Sz':spin,'strict_far_change_from_own_baseline':delta,'max_unitary_far_change':np.max(abs(delta[:,1])),'postselected_initial_far_change':delta[0,2],'max_Sz_change_from_own_initial':np.max(abs(spin-spin[[0]]),axis=0),'norm_error':np.max(abs(probabilities.sum(axis=1)-1)),'joint_sum_error':np.max(abs(joint.sum(axis=2)-1)),'POVM_sum_error':np.max(abs(povm.sum(axis=2)-1)),'minimum_POVM_probability':povm.min(),'energy_drift':np.max(abs(energy-energy[[0]])),'source_label_not_fixed_Sz_probability_error':np.max(abs(wrong_label-povm[:,:,0])),'constraints':model.constraints(),'Sz_commutator':maximum_entry(model.H@s.spin[2]-s.spin[2]@model.H),'full_xyz_spin_commutators':[maximum_entry(model.H@a-a@model.H) for a in s.spin],'current_norms':[maximum_entry(a) for a in model.total_currents]}
        stored[label+'_final']=waves[-1]
    np.savez_compressed(ROOT/(PREFIX+'_collision_states.npz'),**stored)
    return {'dimension':s.dimension,'ground_energy':e,'ground_residual':norm(s.H@g-e*g),'new_times':times,'new_eta':request['collision_eta'],'incoming_Gram':incoming.conj().T@incoming,'outgoing_Gram':outgoing.conj().T@outgoing,'source_projector_Sz_noncommutation':norm(s.spin[2]@out-out@(out.conj().T@(s.spin[2]@out))),'cases':cases}


def main():
    request=json.loads((ROOT/(PREFIX+'_input.json')).read_text())
    result={'frame':frame_review(request),'collision':collision_review(request)}
    (ROOT/(PREFIX+'_result.json')).write_text(json.dumps(encode(result),indent=2))
    print(json.dumps(encode({'frame':{k:result['frame'][k] for k in ['Gauss_residual','CAR_residual','continuity_relative','wrong_Noether_as_current_continuity','wrong_odd_reference_G_distance','magnetic_projected_norm','same_halfspin_cross_metric_off_magnetic_norm']},'collision':{k:{a:v[a] for a in ['max_unitary_far_change','max_Sz_change_from_own_initial','source_label_not_fixed_Sz_probability_error']} for k,v in result['collision']['cases'].items()}})))


if __name__=='__main__':
    main()
