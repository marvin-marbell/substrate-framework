"""Finite same-span field-only unitary versus matched EM-off no-signalling control.
SAME entangled source state in both laws here; explicitly NOT a factorized-reference
four-point control. Raw source norms retained, no particle-position/spin receiver.
"""
from pathlib import Path
from typing import Any
import hashlib,json,resource,sys,time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator,expm_multiply
from rich_review_wave20_20261015_engine import OwningH
from rich_spatial_translation_20261015_model import shell_fields
ROOT=Path(__file__).resolve().parent


def main()->None:
    started=time.monotonic();request:Any=json.loads(Path(sys.argv[1]).read_text());limit=request['memory_limit_MiB']*1024**2;resource.setrlimit(resource.RLIMIT_AS,(limit,limit))
    for name,expected in request['required_input_sha256'].items():
        with (ROOT/name).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=expected:raise ValueError('Exact owning input mismatch: '+name)
    law=json.loads((ROOT/request['law_input']).read_text());primary=json.loads((ROOT/request['primary_result']).read_text());snap=np.load(ROOT/request['primary_states']);cached=np.load(ROOT/request['reconstruction']);site=primary['exact_transverse_fiber']['selected_source_site'];field=shell_fields(law);off=shell_fields(dict(law,charge=0.));np_=field['nphi'];ne=field['shell_dimension'];nb=field['dimension'];U=field['q_drive']
    # The exact offset is read from the pure owning law builder, never inferred
    # from a spectrum/reference energy. It is only an overall phase here.
    from rich_spatial_translation_20261015_model import build
    physical=build(law);offset=physical['offset'];del physical
    arrays={};reports={}
    for name,number,key,index in [('positive',162,'coupled_positive_sources',site),('negative',162,'coupled_negative_sources',site)]:
        body=cached[name+'_projected_components'][index];state=snap[key][:,0];raw=float(np.vdot(state,state).real)
        def act(k: Any,v: Any)->Any:return (k@v.reshape(number,nb).T).T.ravel()
        kicked=act(U,state)
        def reductions(v: Any)->Any:
            cube=v.reshape(number,np_,ne);matter=cube.reshape(number,-1);matter_phi=cube.reshape(number*np_,ne);em=cube.reshape(-1,ne);return dict(body=matter@matter.conj().T,body_phi=matter_phi@matter_phi.conj().T,EM=em.T@em.conj())
        r0=reductions(state);rk=reductions(kicked);initial={part:float(np.linalg.norm(rk[part]-r0[part])) for part in r0};current_terms=field['derivatives'][0]
        def current(v: Any)->Any:
            a=v.reshape(number,nb);result=np.zeros_like(a)
            for f,k in zip(body,current_terms):
                if k.nnz:result-=(k@(f@a).T).T
            return result.ravel()
        data=dict(raw_source_norm2=raw,raw_kicked_source_norm2=float(np.vdot(kicked,kicked).real),field_unitary_defect=float(np.linalg.norm((U.conj().T@U).toarray()-np.eye(nb))),field_only_t0_raw_reduced_density_difference_Frobenius=initial,instantaneous_coupled_original_current_difference=float((np.vdot(kicked,current(kicked))-np.vdot(state,current(state))).real),instantaneous_cut_current_commutator_action_norm=float(np.linalg.norm(current(act(field['coordinates'][0],state))-act(field['coordinates'][0],current(state)))),arms={})
        for arm,fields in [('coupled',field),('EM_off_same_entangled_source',off)]:
            h=OwningH(body,fields,offset);e=float(np.vdot(state,h@state).real/raw);shifted=LinearOperator(h.shape,matvec=lambda v:h@v-e*v,matmat=lambda v:h@v-e*v,rmatvec=lambda v:h.H@v-e*v,rmatmat=lambda v:h.H@v-e*v,dtype=complex);rows=[];kinetic=sparse.kron(sparse.eye(np_),fields['em_kinetic'],format='csr')
            for t in request['times']:
                value=expm_multiply((-1j*t)*shifted,np.column_stack([state,kicked]),traceA=-1j*t*(h.tr-e*h.shape[0]));base,driven=value.T;rb=reductions(base);rd=reductions(driven);rows.append(dict(time=t,raw_reduced_density_difference_Frobenius={part:float(np.linalg.norm(rd[part]-rb[part])) for part in rb},norm2_baseline=float(np.vdot(base,base).real),norm2_driven=float(np.vdot(driven,driven).real),raw_EM_kinetic_difference=float((np.vdot(driven,act(kinetic,driven))-np.vdot(base,act(kinetic,base))).real)))
                arrays[name+'_'+arm+'_t'+str(t)+'_baseline']=base;arrays[name+'_'+arm+'_t'+str(t)+'_driven']=driven
            data['arms'][arm]=rows
        reports[name]=data
    result=dict(kind='NEW_SAME_FULL_SPAN_FIELD_ONLY_UNITARY_INTERNAL_FEEDBACK_WITH_MATCHED_NO_SIGNALLING_CONTROL',request=request,field_kick=dict(operator='exp(i kappa Q_xcos_PROJECTED)',kappa=law['kick_strength'],meaning='Existing prescribed law kick, NOT fitted. Unit operator on matter/Phi. Finite-cut Q phase may change original current immediately because projected coordinate/Peierls operators fail continuum commutation; that is NOT instantaneous matter reaction.'),reports=reports,meaning='All raw source states keep norm2~.5. Reduced body/body-Phi and EM matrices have that raw trace; Frobenius differences are NOT probabilities/TV or isolated-particle receivers. Same entangled source used for coupled and complete EM-off matched local-unitary diagnostic; the latter is NOT mislabeled as factorized-reference four-point control. Product EM-off evolution cannot transmit a field-only unitary to body-Phi despite initial entanglement. Nonzero later coupled matter response establishes actual finite-H internal feedback, not continuum causality/photon/scattering/NoEther or a cure for field/charged-H cuts.',elapsed_seconds=time.monotonic()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    output=Path(sys.argv[2]);output.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(output.with_suffix('.npz'),**arrays);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
