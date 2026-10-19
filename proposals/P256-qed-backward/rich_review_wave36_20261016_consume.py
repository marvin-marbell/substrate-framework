"""Selected actual quantum Phi-source EOM and rich endpoint K3 consumer.
Imports only declared pure primitives; never invokes a producer CLI.
"""
from __future__ import annotations
import gc
import json
from pathlib import Path
import resource
import time
import numpy as np
from scipy import sparse
from scipy.linalg import eigh, solve_sylvester
import rich_composite_geometry_20261015_model as model
from rich_composite_geometry_20261015_science import load
from rich_composite_geometry_20261015_runtime import primitive_for
from rich_composite_geometry_20261015_cloud_probe import neutral_state, inner, add
from rich_composite_geometry_20261015_source_eom import source_operator, bare_state, commutator_terms, apply_terms
from rich_composite_geometry_20261015_cloud_transport import scalar_inner

ROOT=Path(__file__).resolve().parent
P='rich_review_wave36_20261016_'
OWNER='rich_composite_geometry_20261015_'


def combine(states, coefficients):
    result={}
    for state,c in zip(states,coefficients):
        for key,field in state.items():add(result,key,c,field)
    return result


def phi_action(state, matrix):
    return {key:field@matrix.T for key,field in state.items()}


def norm(state):return float(inner(state,state).real)


def projected_weight(gram, overlaps):
    values,vectors=eigh(gram)
    threshold=max(float(values[-1]),1.)*1e-11
    keep=values>threshold
    weight=float(np.sum(abs(vectors[:,keep].conj().T@overlaps)**2/values[keep]).real)
    return weight,dict(rank=int(keep.sum()),dimension=len(values),eigenvalues=values.tolist(),threshold=threshold)


def review_prior():
    rows={}
    for label in ('charged','neutral_reference'):
        data=np.load(ROOT/(OWNER+'correlated_cloud_residual1_'+label+'_residual_data.npz'))
        result=json.loads((ROOT/(OWNER+'correlated_cloud_residual1_'+label+'_result.json')).read_text())
        total=float(data['component_Gram'].sum().real)
        rows[label]=dict(recomputed_raw_residual_norm2=total,reported=result['full_primitive_residual_norm2'],difference=total-result['full_primitive_residual_norm2'],RMS=float(np.sqrt(total/result['raw_state_norm2'])))
    data=np.load(ROOT/(OWNER+'correlated_source_eom1_B_data.npz'))
    g=json.loads((ROOT/(OWNER+'correlated_source_eom1_B_result.json')).read_text())['raw_source_strength']
    d=float(data['EOM_branch_Gram'].sum().real);k=data['source_EOM_row'].sum()
    rows['bare_EOM']=dict(raw_G=g,norm2=d,mean_frequency_real=float((k/g).real),source_orthogonal_RMS=float(np.sqrt((d-abs(k)**2/g)/g)))
    rows['cancelled_catalogues']={name:json.loads((ROOT/(OWNER+name+'_exit.json')).read_text()) for name in ('tetra_poles1','correlated_onehop_poles1')}
    return rows


def endpoint_consumer():
    cp=load(ROOT/(OWNER+'correlated_cloud_transport1_transport_CP.pkl'))
    p=cp['p'];scalar,rotor=model.field_and_rotor(p);phi=np.diag(scalar['grid'])
    axis=np.asarray(model.geometry(p)['axes'])[0]
    rotations=[sum((axis[a]*sparse.csr_matrix(rotor['coordinate'][i][a]).toarray() for a in range(3)),np.zeros((10,10),complex)) for i in range(3)]
    loop=model.loop_field();trace=np.asarray(loop['W'])
    labels=list(cp['field_CP_by_degree'][3]);qpack=cp['quark_packets'];fields=cp['field_CP_by_degree'][3]
    qgram=np.array([[scalar_inner(qpack[a],qpack[b]) for b in labels] for a in labels])
    fgram=np.array([[np.vdot(fields[a],fields[b]) for b in labels] for a in labels])
    full=float(np.sum(qgram*fgram).real)
    targets=cp['target_quark_packets'];cloud=cp['target_white_field'];endpoints=[];names=[]
    for name,q in targets.items():
        f=cloud[name]
        for family,field in (('B',f),('PhiB',f@phi.T),('TrUB',np.einsum('pq,nqrf->nprf',trace,f)),('Phi3B',f@np.linalg.matrix_power(phi,3).T)):
            names.append(name+':'+family);endpoints.append([(q,field)])
    half_names=['half_triplet:-1','half_triplet:1']
    sigma=(np.array([[0,1],[1,0]]),np.array([[0,1j],[-1j,0]]),np.diag([-1,1]))
    for row,name in enumerate(half_names):
        terms=[]
        for column,source_name in enumerate(half_names):
            for i in range(3):
                if sigma[i][column,row]!=0:terms.append((targets[source_name],sigma[i][column,row]*np.einsum('rs,npsf->nprf',rotations[i],cloud[source_name])))
        names.append(name+':sigmaRdB');endpoints.append(terms)
    gram=np.array([[sum(scalar_inner(aq,bq)*np.vdot(af,bf) for aq,af in a for bq,bf in b) for b in endpoints] for a in endpoints])
    overlaps=np.array([sum(scalar_inner(q,qpack[label])*np.vdot(f,fields[label]) for q,f in endpoint for label in labels) for endpoint in endpoints])
    results={}
    for title,families in [('bare',('B',)),('rich_Phi_R_U',('B','PhiB','TrUB','sigmaRdB')),('material_Phi3_extension',('B','PhiB','TrUB','sigmaRdB','Phi3B'))]:
        keep=[i for i,name in enumerate(names) if name.split(':')[-1] in families]
        captured,receipt=projected_weight(gram[np.ix_(keep,keep)],overlaps[keep])
        results[title]=dict(raw_captured=captured,raw_rest=full-captured,captured_fraction_of_FULL_K3=captured/full,metric=receipt)
    core=sum(float(np.linalg.norm(sum(scalar_inner(q,qpack[label])*fields[label] for label in labels))**2) for q in targets.values())
    np.savez(ROOT/(P+'endpoint_final_raw.npz'),endpoint_Gram=gram,endpoint_K3_overlaps=overlaps,quark_Gram=qgram,field_Gram=fgram,names=np.array(names))
    return dict(full_K3_norm2=full,unchanged_source_quark_core_ALL_fields=core,changed_quark_core_raw_weight=full-core,projections=results,loop_endpoint='W=TrU, actual complex Haar Galerkin Wilson trace; not ReTrU',spin_endpoint='Creation-spin coefficient sigma_(mu,m), ascending m basis, predeclared geometry.axes[0]',scope='Frozen actual K3 raw image consumed once for NEW endpoint projections. Counts/core are diagnostics, never particle admission. Phi3 is material endpoint ansatz extension, not a new Hamiltonian or finite-time mobility.')


def source_consumer(config):
    data=load(ROOT/config['neutral_basis_snapshot']);primitive=primitive_for(data);p=data['p']
    coupling=load(ROOT/config['neutral_coupling_snapshot']);eigen=np.load(ROOT/config['neutral_states_snapshot'])
    omega=neutral_state(data,coupling,eigen['states'][:,0]);scalar,rotor=model.field_and_rotor(p)
    del data,coupling,eigen;gc.collect()
    phi=np.diag(scalar['grid']);hphi=np.asarray(scalar['H']);comm=hphi@phi-phi@hphi
    operator=source_operator(config['source_site'],config['source_kind'],config['source_twice_m'],False)
    bare=bare_state(primitive,omega,operator);dressed=phi_action(bare,phi)
    branches=apply_terms(primitive,omega,commutator_terms(primitive,operator),scalar,rotor)
    keys=list(branches);states=[branches[key] for key in keys];base=combine(states,np.ones(len(states)))
    derivative=phi_action(bare,comm);wrong=phi_action(base,phi);correct=combine([wrong,derivative],[1,1])
    sources=[bare,dressed];images=[base,correct]
    gram=np.array([[inner(a,b) for b in sources] for a in sources]);k=np.array([[inner(a,b) for b in images] for a in sources]);gdot=1j*(k-k.conj().T)
    wrong_k=np.array([[inner(a,b) for b in [base,wrong]] for a in sources])
    # Independent operator-product boundary: source applied after scalar action,
    # rather than reusing the commutator matrix supplied to the right-hand path.
    first=phi_action(dressed,hphi);second=phi_action(phi_action(bare,hphi),phi)
    explicit=combine([first,second],[1,-1]);derivative_error=norm(combine([explicit,derivative],[1,-1]))
    wrong_residual=norm(combine([explicit,{}],[1,-1]))
    if derivative_error>1e-20:raise AssertionError('Correct derivative failed actual field product consumer')
    if wrong_residual<1e-4:raise AssertionError('Dropped Hphi derivative did not reach discriminating field consumer')
    values,vectors=eigh(gram);sqrtg=(vectors*np.sqrt(values))@vectors.conj().T;inverse=(vectors*(1/np.sqrt(values)))@vectors.conj().T
    dot_s=solve_sylvester(sqrtg,sqrtg,gdot)
    generator=inverse@k@inverse+1j*dot_s@inverse
    # Same primitive chord Peierls derivative, not source-charge times hopping.
    charges=np.array([2/3 if flavour==0 else -1/3 for edge,flavour in keys]);chord=np.array([edge==3 for edge,flavour in keys])
    current_coeff=-1j*charges*chord;contact_coeff=-charges**2*chord
    current=combine(states,current_coeff);contact=combine(states,contact_coeff)
    currents=[current,phi_action(current,phi)];contacts=[contact,phi_action(contact,phi)]
    theta_step=1e-4;delta=charges*chord
    plus=combine(states,np.exp(1j*delta*theta_step));minus=combine(states,np.exp(-1j*delta*theta_step))
    fd_current=combine([plus,minus],[-1/(2*theta_step),1/(2*theta_step)])
    fd_contact=combine([plus,base,minus],[1/theta_step**2,-2/theta_step**2,1/theta_step**2])
    current_error=norm(combine([phi_action(fd_current,phi),currents[1]],[1,-1]))
    contact_error=norm(combine([phi_action(fd_contact,phi),contacts[1]],[1,-1]))
    raw_eom_gram=np.array([[inner(a,b) for b in images] for a in images])
    current_gram=np.array([[inner(a,b) for b in currents] for a in currents]);contact_gram=np.array([[inner(a,b) for b in contacts] for a in contacts])
    np.savez(ROOT/(P+'source_raw.npz'),G=gram,K=k,Gdot=gdot,wrong_dropped_derivative_K=wrong_k,EOM_Gram=raw_eom_gram,Hphi_Phi_commutator=comm,sqrtG=sqrtg,dot_sqrtG=dot_s,moving_orthonormal_generator=generator,current_Gram=current_gram,contact_Gram=contact_gram)
    result=dict(reference_norm2=norm(omega),source_names=['B','PhiB'],raw_source_G=gram.real.tolist(),raw_EOM_norms2=np.diag(raw_eom_gram).real.tolist(),Phi_derivative_norm2=norm(derivative),wrong_dropped_derivative_error_norm2=wrong_residual,correct_product_derivative_error_norm2=derivative_error,Gdot_norm=float(np.linalg.norm(gdot)),dropped_derivative_Gdot_error=float(np.linalg.norm(1j*(wrong_k-wrong_k.conj().T)-gdot)),moving_connection_norm=float(np.linalg.norm(dot_s@inverse)),moving_generator_nonhermiticity=float(np.linalg.norm(generator-generator.conj().T)),fixed_G_Hermitization_metric_defect=float(np.linalg.norm(gdot)),current_norms2=np.diag(current_gram).real.tolist(),contact_norms2=np.diag(contact_gram).real.tolist(),dressed_current_finite_difference_error_norm2=current_error,dressed_contact_finite_difference_error_norm2=contact_error,derivative_scope='Full Phi7 [Hphi,Phi] at operator-product boundary. Hrot/SU3electric/plaquette and every scalar/vector hopping coefficient commute EXACTLY with selected Phi, so their [H,D] terms are zero algebraically, NOT field freezing. All Phi/R/PW/sea fields remain in Phi[H,B] and changed local CAR support. No R/U EOM asserted from this Phi result.',metric_scope='Gdot=i(K-Kdag) on evolving hot sea; no K Hermitization. Orthonormal generator includes actual Sylvester sqrtG derivative connection.',source_scope='Literal local cubic source on ALL inputs, no count mask. Source strength is raw CAR metric, not normalized probability. Phi endpoint is quantum coordinate, not c-number RMS.',spin_scope='B is halfspin under chosen finite J=L_left+S_CAR rotation; Phi scalar preserves it. This is chosen CAR-spin/rotor action, not a derivation of primitive spin from field-only physical Noether origin.')
    return result


def main():
    started=time.time();config=json.loads((ROOT/(OWNER+'correlated_source_eom1_input.json')).read_text())
    result=dict(kind='INDEPENDENT_ACTUAL_QUANTUM_PHI_SOURCE_EOM_CURRENT_AND_RICH_K3_ENDPOINTS',prior_raw_review=review_prior(),endpoint=endpoint_consumer(),source=source_consumer(config))
    result['resource']=dict(elapsed_seconds=time.time()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    result['limits']='Candidate operator expansion only; no two-level truncation, Dirac proof, full-law stationary vacuum, baryon admission, finite-time mobility/binding/lifetime, mass or RMS subtraction. R/U endpoints retain projected field products but their own complete EOM is not computed here.'
    (ROOT/(P+'result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
