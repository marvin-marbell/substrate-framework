"""NEW exact two-current response: complete one-current-connected population blocks.
No finite-A time evolution is performed in the linear-response closure.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import eigh, expm
from scipy import sparse
from rich_common_vector_20261015_model import (
    build, law, operators, gauge_sector, complete_masks, reachable_signatures,
    current_block, closure, path_derivation, conjugate_sector, conjugation,
    one_body, vector_terms, spin_reverse_sector, signatures as occupation_signature)


def norm(x):
    return float(np.linalg.norm(x))


def mean(state, operator):
    return float(np.vdot(state, operator @ state).real)


def cj(value):
    value = np.asarray(value)
    return {'real':value.real.tolist(), 'imag':value.imag.tolist()}


def spectrum(h, spin):
    ss, vectors = eigh(spin); labels = np.rint(np.sqrt(np.maximum(0, 1+4*ss))-1).astype(int)
    energies, states, spins = [], [], []
    for label in np.unique(labels):
        projector = vectors[:, labels == label]; e, v = eigh(projector.conj().T @ h @ projector)
        energies.extend(e); states.append(projector @ v); spins.extend([label/2]*len(e))
    order = np.argsort(energies)
    return np.array(energies)[order], np.column_stack(states)[:,order], np.array(spins)[order]


def matrix_kernel(gaps, amplitudes, frequencies, eta):
    residues = np.einsum('ni,nj->nij', amplitudes.conj(), amplitudes)
    chi = np.array([np.sum(residues/(w-gaps[:,None,None]+1j*eta)
        - residues.transpose(0,2,1)/(w+gaps[:,None,None]+1j*eta),axis=0) for w in frequencies])
    return chi, residues


def evolve(state, e, v, times):
    return v @ (np.exp(-1j*np.outer(e,times))*(v.conj().T@state)[:,None])


def main():
    inp,out = map(Path,sys.argv[1:3]); p=json.loads(inp.read_text()); model=build(p)
    nf=len(model['scalar']['grid']); frequencies=np.array(p['frequencies']); times=np.array(p['times'])
    stored={}; bases={}; sources={}; responses={}; event_results={}; audits={}; pair_cache={}
    for name,sector in model['sectors'].items():
        op=model['operators'][name]; h,j,k=law(op,model['scalar'],p); spin=np.kron(op['S2'],np.eye(nf))
        e,v,spins=spectrum(h,spin); bases[name]=(h,j,k,e,v,spins)
        for key,value in dict(H=h,Jscalar=j,Kscalar=k,e=e,v=v,spin=spin,masks=np.array(sector['masks'],np.uint64)).items(): stored[name+'_base_'+key]=value
    e0=bases['neutral'][3][0]; reference=bases['neutral'][4][:,0]; receivers={'neutral':reference}
    for name,byspin in model['sources'].items():
        h,j,k,e,v,spins=bases[name]; sources[name]={}
        for label,source in byspin.items():
            state=source@reference; amplitudes=v.conj().T@state; weights=abs(amplitudes)**2; bright=np.flatnonzero(weights>1e-11)
            gap=e[bright[0]]-e0; pole=abs(e-e[bright[0]])<1e-8; receiver=v[:,pole]@amplitudes[pole]; receiver/=norm(receiver)
            sources[name][label]=dict(raw_strength=norm(state)**2,lowest_bright_gap=float(gap),raw_pole_residue=float(weights[pole].sum()),
                all_gaps=(e-e0).tolist(),all_raw_weights=weights.tolist(),source_spin_residual=norm(np.kron(model['operators'][name]['S2'],np.eye(nf))@state-float(label)*(float(label)+1)*state),
                N=mean(state,np.kron(model['operators'][name]['N'],np.eye(nf)))/norm(state)**2,Q=mean(state,np.kron(model['operators'][name]['Q'],np.eye(nf)))/norm(state)**2)
            stored[name+'_source_'+label]=source; stored[name+'_raw_source_'+label]=state
            if label=='0.5': receivers[name]=receiver
    for name,state in receivers.items():
        sector=model['sectors'][name]; op=model['operators'][name]; h,j,k,e,v,spins=bases[name]; estate=mean(state,h)
        amplitude=np.zeros((len(e),3),complex); amplitude[:,2]=v.conj().T@j@state
        chi,residues=matrix_kernel(e-estate,amplitude,frequencies,p['eta']); scalar_only=chi.copy()
        contact=np.zeros((3,3)); contact[2,2]=mean(state,k); chi+=contact
        direct=amplitude.conj().T@amplitude; block_results=[]; vectors=[]; c_audits=[]; eigen_error=0.; leak_error=0.; gauge_error=0.
        isanti=name.startswith('anti'); original=name[4:] if isanti else name
        if isanti: blocks=pair_cache[original]
        else: blocks=[]
        signatures=reachable_signatures(sector) if not isanti else [x['signature'] for x in blocks]
        reversed_base=spin_reverse_sector(sector)
        reversed_chi=scalar_only.copy(); spin_reverse_errors=[]
        for number,signature in enumerate(signatures):
            if isanti:
                positive=blocks[number]; target=conjugate_sector(positive['sector'],model['loop']); top=operators(target,model['loop'])
                th,_,_=law(top,model['scalar'],p); te,tv=positive['e'],positive['v']
            else:
                target=gauge_sector(complete_masks(signature),model['loop']); top=operators(target,model['loop']); th,_,_=law(top,model['scalar'],p)
                te,tv=eigh(th); blocks.append(dict(signature=signature,sector=target,e=te,v=tv))
            if target['dimension']==0: continue
            js=[current_block(sector,target,axis,p['vector_velocity'],nf) for axis in range(3)]
            amps=np.column_stack([tv.conj().T@operator@state for operator in js]); gaps=te-estate
            addition,weights=matrix_kernel(gaps,amps,frequencies,p['eta']); chi+=addition; direct+=amps.conj().T@amps
            eigen_error=max(eigen_error,norm(th@tv-tv*te[None,:])); gauge_error=max(gauge_error,target['gauge_error'])
            for axis in range(3):
                raw=sparse.kron(one_body(sector['masks'],vector_terms(axis),target['masks']),sparse.eye(19),format='csr')
                mapped=raw@sector['P']; leak_error=max(leak_error,float(sparse.linalg.norm(mapped-target['P']@(target['P'].conj().T@mapped))))
            targetanti=conjugate_sector(target,model['loop']); ah,_,_=law(operators(targetanti,model['loop']),model['scalar'],p)
            basemasks=sorted(__import__('rich_common_colour_lepton_20261015_model').charge_conjugate(mask)[0] for mask in sector['masks'])
            cbase=(conjugation(sector['masks'],basemasks)@sector['P'])
            anti_base=dict(masks=basemasks,P=cbase)
            anti_js=[current_block(anti_base,targetanti,axis,p['vector_velocity'],nf) for axis in range(3)]
            c_audits.append(dict(H=norm(ah-th),J=[norm(aj+jj) for aj,jj in zip(anti_js,js)]))
            reversed_target=spin_reverse_sector(target)
            reversed_js=[current_block(reversed_base,reversed_target,axis,p['vector_velocity'],nf) for axis in range(3)]
            spin_reverse_errors.append(max(norm(rj-sign*jj) for rj,jj,sign in zip(reversed_js,js,[-1,1,-1])))
            reversed_amplitudes=np.column_stack([tv.conj().T@operator@state for operator in reversed_js])
            reversed_addition,_=matrix_kernel(gaps,reversed_amplitudes,frequencies,p['eta'])
            reversed_chi+=reversed_addition
            block_results.append(dict(index=number,signature=list(occupation_signature(target['masks'][0])),raw_dimension=target['raw_dimension'],gauge_dimension=target['dimension'],quantum_dimension=len(te),
                H0_missing_targets=closure(target),gauss_error=target['gauge_error'],all_gaps=gaps.tolist(),all_residue_matrices=cj(weights),strength=weights[:,range(3),range(3)].real.sum(axis=0).tolist()))
            prefix=name+'_pair'+str(number)
            for key,value in dict(H=th,e=te,v=tv,J=np.array(js),masks=np.array(target['masks'],np.uint64),Pdata=target['P'].data,Pindices=target['P'].indices,Pindptr=target['P'].indptr,Pshape=target['P'].shape).items(): stored[prefix+'_'+key]=value
            zstate=js[2]@state; vectors.append((zstate,te,tv,top))
        if not isanti: pair_cache[name]=blocks
        responses[name]=dict(frequencies=frequencies.tolist(),eta=p['eta'],contact=contact.tolist(),retarded_with_contact=cj(chi),
            scalar_only_Y_removed=cj(scalar_only+contact),new_vector_difference=cj(chi-scalar_only-contact),integrated_residue_matrix=cj(direct),
            actual_spin_reversed_retarded_with_contact=cj(reversed_chi+contact),spin_reverse_operator_residual=max(spin_reverse_errors),
            scalar_base_virtual_states=len(e),one_current_blocks=block_results,total_virtual_states=len(e)+sum(b['quantum_dimension'] for b in block_results),
            interpretation='Exact two-current linear response around stationary same-law source pole, both retarded orders, all one-current-connected H0 population sectors and scalar base retained. Residues/conditional source strengths NOT pair-creation probabilities. No nonlinear finite-A closure or finite pulse claim.')
        # A current-conditioned quantum receiver, not a claimed physical finite pulse.
        total_strength=sum(norm(zstate)**2 for zstate,_,_,_ in vectors)
        feedback={key:np.zeros(len(times)) for key in ['phi','phi2','C','B','scalar_energy']}
        for zstate,te,tv,top in vectors:
            states=evolve(zstate,te,tv,times)/np.sqrt(total_strength) if total_strength>0 else evolve(zstate,te,tv,times)
            obs={'phi':np.kron(np.eye(top['S2'].shape[0]),model['scalar']['phi']),
                'phi2':np.kron(np.eye(top['S2'].shape[0]),model['scalar']['phi']@model['scalar']['phi']),
                'scalar_energy':np.kron(np.eye(top['S2'].shape[0]),model['scalar']['H']),
                'C':np.kron(top['C'],np.eye(nf)),'B':np.kron(top['B'],np.eye(nf))}
            for key,operator in obs.items(): feedback[key]+=[mean(states[:,i],operator) for i in range(len(times))]
        baseobs={key:np.kron(op[key],np.eye(nf)) for key in ['C','B']}
        baseobs.update(phi=np.kron(np.eye(sector['dimension']),model['scalar']['phi']),phi2=np.kron(np.eye(sector['dimension']),model['scalar']['phi']@model['scalar']['phi']),scalar_energy=np.kron(np.eye(sector['dimension']),model['scalar']['H']))
        baseline={key:mean(state,operator) for key,operator in baseobs.items()}
        fieldkick=np.kron(np.eye(sector['dimension']),expm(-1j*p['field_kick']*model['scalar']['P']))@state
        fieldstates=evolve(fieldkick,e,v,times)
        field={key:[mean(fieldstates[:,i],operator)-baseline[key] for i in range(len(times))] for key,operator in baseobs.items()}
        event_results[name]=dict(times=times.tolist(),current_conditioned_strength=total_strength,receiver='Normalized vector-current J_z|stationary pole>, all its complete H0 population blocks propagated coherently; number remains fixed. This is a conditional quantum receiver, NOT finite EM pulse yield.',
            current_conditioned_observables={key:value.tolist() for key,value in feedback.items()},baseline=baseline,
            current_conditioned_excess={key:(value-baseline[key]).tolist() for key,value in feedback.items()},
            field_only_preparation='I_matter,loop tensor exp(-i field_kick Pphi) on SAME source pole, untouched reference baseline, full H0 propagation.',field_only_excess=field)
        audits[name]=dict(base_H0_missing_targets=closure(sector),one_current_signature_count=len(signatures),current_Gauss_projection_residual=leak_error,max_Gauss_residual=gauge_error,max_pair_eigen_residual=eigen_error,C_related_pair_audits=c_audits,
            vector_spin_response_xy_imag=chi[:,0,1].imag.tolist(),spin_conjugate_xy_statement='Reversing source frame M changes the antisymmetric xy response; frame is internal supplied SU2, not resolved spatial Noether angular momentum.')
        stored[name+'_receiver']=state
        print(json.dumps(dict(sector=name,blocks=len(block_results),virtuals=responses[name]['total_virtual_states'],vector_strength=total_strength,closure=max([b['H0_missing_targets'] for b in block_results]+[0]))),flush=True)
    output=dict(kind='NEW_COMMON_VECTOR_FINITE_PERIODIC_CANDIDATE_NOT_QED_OR_ADMISSION',parameters=p,path_derivation=path_derivation(p),sources=sources,responses=responses,quantum_feedback=event_results,audits=audits,
        boundaries='Frozen common3+1 CAR32 reference, physical selected Q, quantum scalar and PW19 loop preserved. Explicit Clifford spin-dependent periodic hopping/gauging/topology inputs. New vector paths enter H(A) and its actual derivative; H0 unchanged. Exact response is not all-number vacuum, finite-A propagation, spatial continuity/COM, microscopic spin derivation, photon or Nature closure. No cross-G/D, additive mass or spin projection of a moving state.')
    out.write_text(json.dumps(output,indent=2)+'\n'); np.savez_compressed(out.with_suffix('.npz'),**stored)


if __name__=='__main__':
    main()
