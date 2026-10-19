"""NEW physical unfiltered-source receivers and independent exact-law audit."""
from itertools import product
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
import rich_full_colour_pairs_20261015_model as full
import rich_common_colour_lepton_20261015_model as common
from rich_review_wave12_20261015_physics import (
    length,mean,cj,mode,coordinate,apply_string,source_witness,spin_spectrum,
    spectral_response,physical_events,
)

ROOT=Path(__file__).resolve().parent


def diagonal(sector,values):
    return (sector['P'].conj().T @ sparse.kron(sparse.diags(values),sparse.eye(19)) @ sector['P']).toarray()


def occupation_audit(sector,members):
    present=set(sector['masks']);missing=set();wrong=set()
    for mask in present:
        for f,m,s,a,b in product(range(2),range(2),range(2),range(members),range(members)):
            if members==4 and (a==3)!=(b==3):
                continue
            step=apply_string(mask,[(mode(f,m,s,a,members),True),(mode(f,m,s,b,members),False)])
            if step is not None and step[0] not in present:
                missing.add(step[0])
        for f,m,s,c in product(range(2),range(2),range(2),range(members)):
            step=apply_string(mask,[(mode(f,1-m,s,c,members),True),(mode(f,m,s,c,members),False)])
            if step is not None and step[0] not in present:
                wrong.add(step[0])
    return dict(actual_density_missing_targets=len(missing),noncommuting_mirror_control_missing_targets=len(wrong),meaning='All actual colour/lepton density targets close. A mirror-changing kinetic law would NOT inherit this closure; it is not implemented here.')


def coefficient_audit(loop,p,members):
    u=np.block([[loop['U'][a,b] for b in range(3)] for a in range(3)])
    ut=np.block([[loop['U'][b,a] for b in range(3)] for a in range(3)])
    entries=[(2/3,u),(-1/3,u)]
    if members==4:
        entries.extend([(0.,loop['W']/3),(-1.,loop['W']/3)])
    out=[];tauX=np.array([[0,1],[1,0]]);tauZ=np.diag([1.,-1.]);rotation=np.array([[1,1],[1,-1.]])/np.sqrt(2)
    for q,kernel in entries:
        for angle in [.37,-.91]:
            phase=np.exp(1j*q*angle);w=phase*kernel+phase.conjugate()*kernel.conj().T
            terms=[p['scalar_coupling']*(np.eye(len(kernel))+p['loop_route']*w/2),-p['scalar_coupling']*p['loop_route']/2*(1j*q*(phase*kernel-phase.conjugate()*kernel.conj().T)),-p['scalar_coupling']*p['loop_route']/2*q*q*w]
            o=np.kron(rotation,np.eye(len(kernel)));records={}
            for name,t in zip(['H','J','K'],terms):
                original=np.kron(tauX,t);rotated=o.conj().T @ original @ o
                records[name]=dict(Hadamard_error=length(rotated-np.kron(tauZ,t)),original_mirror_integral_commutator=length(original @ np.kron(tauX,np.eye(len(kernel)))-np.kron(tauX,np.eye(len(kernel))) @ original))
            out.append(dict(q=q,angle=angle,Hermitian_kernel_error=length(w-w.conj().T),coefficients=records))
    return dict(preprojection=out,U_compression_norm=float(np.linalg.norm(u,2)),colour_transposed_U_compression_norm=float(np.linalg.norm(ut,2)),D_min=float(eigh(np.eye(57)+p['loop_route']*(u+u.conj().T)/2,eigvals_only=True)[0]),hole_D_min=float(eigh(np.eye(57)+p['loop_route']*(ut+ut.conj().T)/2,eigvals_only=True)[0]))


def full_neutral_bounds(model,e0):
    p=model['parameters'];field=model['scalar'];loop=model['loop'];nf=len(field['grid'])
    base=np.kron(np.eye(19),field['H'])+np.kron(p['electric']*loop['C']/2-p['plaquette']*loop['B'],np.eye(nf));bounds={}
    for count in range(1,12):
        lower=base.copy()
        for a,phi in enumerate(field['grid']):
            defects=2*(count if phi>=0 else 12-count)
            potential=-p['scalar_coupling']*abs(phi)*(12*np.eye(19)+4*p['loop_route']*loop['B']-defects*(1-p['loop_route'])*np.eye(19))
            indices=np.arange(19)*nf+a;lower[np.ix_(indices,indices)]+=potential
        bounds[str(count)]=float(eigh(lower,eigvals_only=True)[0])
    return dict(lower_bounds=bounds,lowest=min(bounds.values()),selected_extremal=e0,full_fixedN12_extremal_certificate=min(bounds.values())>e0,conditions='Positive g,0<=r<1; BOTH particle U and colour-transposed hole U are compressions of pointwise unitaries. Retains full field kinetic and whole PW19 loop, lowers over every other neutral occupation. No charged-block/all-N certificate.')


def independent_C(source,target,members,nf):
    lookup={mask:i for i,mask in enumerate(target['masks'])};rows=[];values=[]
    for mask in source['masks']:
        occupied=[i for i in range(8*members) if mask & (1 << i)];other=(1 << (8*members))-1;amplitude=1
        for index in reversed(occupied):
            f,m,s,c=coordinate(index,members)
            other,sign=apply_string(other,[(mode(f,1-m,1-s,c,members),False)])
            amplitude*=sign*(1 if m==s else -1)
        rows.append(lookup[other]);values.append(amplitude*((-1)**len(occupied) if members==3 else 1))
    matter=sparse.csr_matrix((values,(rows,range(len(rows)))),shape=(len(target['masks']),len(source['masks'])))
    swap=np.zeros((19,19));swap[0,0]=1;swap[1:10,10:]=np.eye(9);swap[10:,1:10]=np.eye(9)
    raw=sparse.kron(matter,swap,format='csr')
    return np.kron((target['P'].conj().T @ raw @ source['P']).toarray(),np.eye(nf))


def run(kind):
    prefix='rich_review_wave12_20261015_'+kind;case=json.loads((ROOT/(prefix+'_input.json')).read_text());p=case['model_parameters'];members=3 if kind=='full' else 4;api=full if members==3 else common;model=api.build(p);scalar=model['scalar'];nf=len(scalar['grid']);arrays={};sectors={};laws={};spectra={};ops={}
    qref=sum((2/3 if coordinate(i,members)[0]==0 else -1/3) if coordinate(i,members)[3]<3 else (0 if coordinate(i,members)[0]==0 else -1) for i in range(8*members) if model['sectors']['neutral']['masks'][0] & (1 << i))
    for name,sector in model['sectors'].items():
        op=dict(model['operators'][name]);h,j,k=api.law(op,scalar,p);laws[name]=h,j,k
        numbers=[mask.bit_count() for mask in sector['masks']]
        charges=[sum((2/3 if coordinate(i,members)[0]==0 else -1/3) if coordinate(i,members)[3]<3 else (0 if coordinate(i,members)[0]==0 else -1) for i in range(8*members) if mask & (1 << i))-qref for mask in sector['masks']]
        op.update(N=diagonal(sector,numbers),Q=diagonal(sector,charges));ops[name]=op
        spin=np.kron(op['S2'],np.eye(nf));e,v,s=spin_spectrum(h,spin);spectra[name]=e,v,s
        sectors[name]=dict(actual_N=sorted(set(numbers)),actual_Q=sorted(set(round(q,12) for q in charges)),dimension=len(e),gauge_dimension=sector['dimension'],Gauss_residual=max(length(g @ sector['P']) for g in sector['G']),closure=occupation_audit(sector,members),ground_energy=float(e[0]),eigen_residual=length(h @ v-v*e),spin_H_commutator=length(h @ spin-spin @ h),spin_J_commutator=length(j @ spin-spin @ j),all_energies=e.tolist(),all_frame_spins=s.tolist(),ground_loop_boundary=mean(v[:,0],np.kron(op['boundary'],np.eye(nf))),ground_phi_boundary=mean(v[:,0],np.kron(np.eye(sector['dimension']),scalar['boundary'])))
        arrays.update({name+'_'+label:value for label,value in dict(H=h,J=j,K=k,energies=e,vectors=v,frame_spins=s,Q=op['Q'],N=op['N']).items()})
    e0,v0,_=spectra['neutral'];reference=v0[:,0];arrays['reference']=reference
    sources=model['sources'] if members==4 else {'baryon':model['sources'],'antibaryon':model['anti_sources']}
    source_results={};receivers={'neutral':reference};initials={'neutral':reference};literal={}
    for name,matrices in sources.items():
        e,v,spin=spectra[name];source_results[name]={};bare={}
        for label,source in matrices.items():
            state=source @ reference;bare[label]=state;weights=abs(v.conj().T @ state)**2;bright=np.flatnonzero(weights>1e-12);pole=abs(e-e[bright[0]])<1e-8;projected=v[:,pole] @ (v[:,pole].conj().T @ state);strength=length(state)**2
            source_results[name][label]=dict(raw_strength=strength,all_raw_residues=weights.tolist(),sum_rule=float(weights.sum()),lowest_bright_gap=float(e[bright[0]]-e0[0]),pole_residue=float(weights[pole].sum()),conditional_pole_fraction=float(weights[pole].sum()/strength),source_frameS2_residual=length(np.kron(ops[name]['S2'],np.eye(nf)) @ state-float(label)*(float(label)+1)*state),actual_Q=mean(state,np.kron(ops[name]['Q'],np.eye(nf)))/strength,actual_N=mean(state,np.kron(ops[name]['N'],np.eye(nf)))/strength)
            arrays[name+'_literal_source_'+label]=source;arrays[name+'_raw_source_state_'+label]=state
            if label=='0.5':
                receivers[name]=projected/length(projected)
            if name in ['baryon','lepton']:
                witness,proof_arrays=source_witness(model,members,name,float(label),model['sectors'][name],source);literal[name+'_'+label]=witness;arrays.update({name+'_'+label+'_CAR_'+key:value for key,value in proof_arrays.items()})
        incoming=bare['0.5']+(1j*case['threehalf_amplitude']*bare['1.5'] if '1.5' in bare else 0);initials[name]=incoming/length(incoming)
    cn=model['Cneutral'] if members==3 else model['C']['neutral'];C={'neutral':dict(reference_overlap=float(abs(np.vdot(reference,cn @ reference))**2),H=length(laws['neutral'][0] @ cn-cn @ laws['neutral'][0]),J=length(laws['neutral'][1] @ cn+cn @ laws['neutral'][1]))}
    C['neutral']['independent_signed_CAR_error']=length(cn-independent_C(model['sectors']['neutral'],model['sectors']['neutral'],members,nf))
    for name,anti in [('baryon','antibaryon')]+([('lepton','antilepton')] if members==4 else []):
        c=model['Cbaryon'] if members==3 else model['C'][name];hp,jp,kp=api.law(ops[name],scalar,p,case['pulse_angle']);hm,jm,km=api.law(ops[anti],scalar,p,-case['pulse_angle']);nmax=8*members
        C[name]=dict(H_finite_A=length(hm @ c-c @ hp),J_finite_A=length(jm @ c+c @ jp),K_finite_A=length(km @ c-c @ kp),Q_flip=length(np.kron(ops[anti]['Q'],np.eye(nf)) @ c+c @ np.kron(ops[name]['Q'],np.eye(nf))),N_complement=length(np.kron(ops[anti]['N'],np.eye(nf)) @ c+c @ np.kron(ops[name]['N'],np.eye(nf))-nmax*c),unfiltered_source_C_fidelity=float(abs(np.vdot(initials[anti],c @ initials[name]))**2))
        C[name]['independent_signed_CAR_error']=length(c-independent_C(model['sectors'][name],model['sectors'][anti],members,nf))
        C[name]['independent_anti_source_errors']={label:length(sources[anti][label]-independent_C(model['sectors'][name],model['sectors'][anti],members,nf) @ source @ cn.conj().T) for label,source in sources[name].items()}
        arrays['C_'+name]=c
    responses={};events={};off_energies={}
    for name,initial in initials.items():
        h,j,k=laws[name];e,v,_=spectra[name];op=ops[name];receiver=receivers[name]
        responses[name]=spectral_response(h,j,k,e,v,receiver,case['frequencies'],case['eta'])
        delta=2e-4;hp,_,_=api.law(op,scalar,p,delta);hm,_,_=api.law(op,scalar,p,-delta)
        responses[name]['current_derivative_error']=length((hp-hm)/(2*delta)+j);responses[name]['contact_derivative_error']=length((hp-2*h+hm)/delta**2-k);responses[name]['wrong_Noether_current_error']=length((hp-hm)/(2*delta)+np.kron(op['S2'],np.eye(nf)))
        direction=-1 if name.startswith('anti') else 1
        def law_at(angle,off=False):
            return api.law(op,scalar,dict(p,scalar_coupling=0.) if off else p,angle)
        localcase=dict(case,pulse_angle=direction*case['pulse_angle']);event,event_arrays=physical_events(h,j,k,e,v,initial,op,scalar,law_at,localcase);events[name]=event;arrays.update({name+'_'+key:value for key,value in event_arrays.items()})
        off_energies[name]=float(eigh(law_at(0.,True)[0],eigvals_only=True)[0])
    result=dict(kind='NEW_WAVE12_PHYSICAL_CONSUMER_NOT_AUTHOR_PRODUCER_REPLAY',model_parameters=p,case=case,sectors=sectors,sources=source_results,literal_original_CAR=literal,coefficient_preprojection=coefficient_audit(model['loop'],p,members),C=C,current_responses=responses,unfiltered_source_events=events,scalar_couplingoff_energies=off_energies,scalar_off_gaps={name:value-off_energies['neutral'] for name,value in off_energies.items() if name!='neutral'},bound=full_neutral_bounds(model,float(e0[0])) if members==3 else 'No whole disconnected-population common-vacuum certificate claimed.',phi_grid=scalar['grid'].tolist(),meaning='Same chosen occupied reference, shared quantum scalar/PW19 and physical Q-current. Static rotated populations are NOT Dirac interband pair creation. Original phi-gap/cut1 claims not inherited. EM pulse is external Wilson probe, scalar kick is actual FIELD-only, strength/conditional fraction and complete PVM separate. No spatial photon/COM/full NoEther/Nature admission.')
    opposite,_,_=api.law(ops['neutral'],scalar,dict(p,scalar_coupling=-p['scalar_coupling']))
    reflection=np.kron(np.eye(model['sectors']['neutral']['dimension']),np.eye(nf)[::-1])
    result['opposite_occupied_branch']=dict(parity_similarity_error=length(opposite-reflection @ laws['neutral'][0] @ reflection),energy=float(eigh(opposite,eigvals_only=True)[0]),meaning='Exact density reversal of filled rotated-positive branch of SAME law, calculated through sign reversal; not a changed fitted coupling or tunnelling between conserved sectors.')
    result['C_convention']='FullPair includes (-1)^N from Hadamard transformation of prior original C. New CommonRoot defines C in rotated modes without that factor: in original modes it differs by conserved fermion parity, an explicit primitive C convention. No channel-dependent phase fitting; all charged channels are odd and retain C charge/current/source identities.'
    (ROOT/(prefix+'_result.json')).write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(ROOT/(prefix+'_states.npz'),**arrays)
    print(json.dumps(dict(dimensions={name:value['dimension'] for name,value in sectors.items()},sources={name:{spin:{key:value for key,value in row.items() if key!='all_raw_residues'} for spin,row in rows.items()} for name,rows in source_results.items()},literal_errors={name:value['original_source_error'] for name,value in literal.items()},C=C,bound=result['bound']),indent=2))
