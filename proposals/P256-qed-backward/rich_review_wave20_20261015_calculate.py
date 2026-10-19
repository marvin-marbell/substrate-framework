"""NEW independent complete same-H connected response, not old consumer replay."""
from typing import Any
from pathlib import Path
from itertools import product
import hashlib,json,resource,sys,time
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from rich_review_wave20_20261015_engine import Engine,OwningH,field_act,norm2,request_dag,run_surface,fourier
ROOT=Path(__file__).resolve().parent


def digest(path: Path)->str:
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def terms_act(engine: Any,name: str,terms: Any,state: Any)->Any:
    one=state.ndim==1;a=state.reshape(engine.n[name],engine.nb,-1);out=np.zeros_like(a,complex)
    for f,k in zip(engine.model['sectors'][name]['matrices'],terms):
        if k.nnz:
            left=(f@a.reshape(engine.n[name],-1)).reshape(a.shape);out+=(k@left.transpose(1,0,2).reshape(engine.nb,-1)).reshape(engine.nb,engine.n[name],-1).transpose(1,0,2)
    return out.reshape(state.shape) if not one else out.ravel()


def shifted_field(engine: Any,shift: Any)->Any:
    field=engine.own_field;terms=list(field['coupling']);index=2
    for axis in range(3):
        for level in (range(3) if axis<2 else [0]):
            profile=np.zeros(4)
            if axis<2:profile[2*axis:2*axis+2]=engine.law['charge']*np.array([np.cos(2*np.pi*level/3),np.sin(2*np.pi*level/3)])
            phase=float(profile@shift);c,s=terms[index:index+2];terms[index]=np.cos(phase)*c-np.sin(phase)*s;terms[index+1]=np.sin(phase)*c+np.cos(phase)*s;index+=2
    return dict(field,coupling=terms)


def derivative_evidence(engine: Any,saved: Any,arrays: Any)->Any:
    field=engine.own_field;direction=np.array([.2,-.31,.17,.29]);eps=3e-4;report={};vertex_saved=np.load(ROOT/saved['vertices_arrays'])
    for name,key in [('neutral','coupled_reference'),('positive','coupled_positive_sources'),('negative','coupled_negative_sources')]:
        vectors=saved['states'][key];vectors=vectors[:,None] if vectors.ndim==1 else vectors;currents=np.array([-vectors.conj().T@terms_act(engine,name,terms,vectors) for terms in field['derivatives']]);contacts=np.array([[vectors.conj().T@terms_act(engine,name,terms,vectors) for terms in row] for row in field['contacts']]);arrays[name+'_current']=currents;arrays[name+'_contact']=contacts;state=vectors[:,0];args=(engine.model['sectors'][name]['matrices'],);offset=engine.model['raw']['offset'];hp=OwningH(*args,shifted_field(engine,eps*direction),offset);hm=OwningH(*args,shifted_field(engine,-eps*direction),offset);h0=engine.H['coupled'][name];j=-sum(direction[a]*terms_act(engine,name,field['derivatives'][a],state) for a in range(4));k=sum(direction[a]*direction[b]*terms_act(engine,name,field['contacts'][a][b],state) for a,b in product(range(4),repeat=2));data=dict(raw_source_norm2=np.sum(abs(vectors)**2,axis=0).tolist(),first_physical_source_FD_relative_error=float(np.linalg.norm((hp@state-hm@state)/(2*eps)+j)/np.linalg.norm(j)),second_physical_contact_FD_relative_error=float(np.linalg.norm((hp@state+hm@state-2*(h0@state))/eps**2-k)/np.linalg.norm(k)),own_vs_author_full_current_array_error=float(np.linalg.norm(currents-vertex_saved[name+'_current_source_matrix'])),own_vs_author_full_contact_array_error=float(np.linalg.norm(contacts-vertex_saved[name+'_contact_source_matrix'])),contact_Frobenius=float(np.linalg.norm(contacts)))
        if name!='neutral':
            allowed=[];forbidden=[];wrong=[]
            for pol,eta in product(range(2),[-1,1]):
                matrix=(currents[2*pol]+1j*eta*currents[2*pol+1])/np.sqrt(2)
                for outmode,inmode in product(range(12),repeat=2):
                    kout,kin=outmode//4,inmode//4;correct=(kin+(eta if name=='positive' else -eta))%3;(allowed if kout==correct else forbidden).append(matrix[outmode,inmode])
                    if name=='negative' and kout!=(kin+eta)%3:wrong.append(matrix[outmode,inmode])
            data.update(allowed_original_current_Frobenius=float(np.linalg.norm(allowed)),forbidden_original_current_Frobenius=float(np.linalg.norm(forbidden)),wrong_positive_momentum_rule_on_removed_mode_violation_Frobenius=float(np.linalg.norm(wrong)))
        data['actual_complete_EM_off_source_derivative_action_norms']=[float(np.linalg.norm(terms_act(engine,name,terms,state))) for terms in engine.fields['EM_off_factorized']['derivatives']];report[name]=data
    return dict(sectors=report,meaning='Mode-weighted J=-dH/dQmode and contact d2H/dQa dQb from literal before-cut Peierls profiles. Shift is physical classical profile offset BEFORE projection, not exponentiating compressed Q. Jacobian q[cos(2pi level/3),sin(2pi level/3)] with lattice link length1; not unweighted SI current or Pcm. Full-H correlator already contains contact: no additional seagull.')


def control_ground(engine: Any)->Any:
    field=engine.fields['EM_off_factorized'];np_=field['nphi'];ne=field['shell_dimension'];em=field['em_kinetic'];bare=field['boson'].toarray().reshape(np_,ne,np_,ne);phi=np.einsum('peqe->pq',bare)/ne-np.trace(em)*np.eye(np_)/ne;f=engine.model['sectors']['neutral']['matrices'];constant=f[0]+sum(f[2::2]);h=np.kron(constant,np.eye(np_))+np.kron(f[1],np.diag(engine.law['coupling']*field['phi']))+np.kron(np.eye(27),phi)-engine.model['raw']['offset']*np.eye(27*np_);ev,uv=eigh(h);ee,ue=eigh(em);state=np.kron(uv[:,0],ue[:,0]);error=np.linalg.norm(field['boson'].toarray()-np.kron(phi,np.eye(ne))-np.kron(np.eye(np_),em));return state,dict(energy=float(ev[0]+ee[0]),bare_factorized_reconstruction_error=float(error),matter_phi_dimension=27*np_,EM_dimension=ne)


def eom(engine: Any,name: str,state: Any)->Any:
    field=engine.own_field;states=field['shell_states'];lookup={tuple(s):i for i,s in enumerate(states)};records=[];hs=engine.H['coupled'][name]@state
    for axis,q in enumerate(field['coordinates']):
        a=np.zeros((len(states),len(states)),complex)
        for column,s in enumerate(states):
            if s[axis]:target=list(s);target[axis]-=1;a[lookup[tuple(target)],column]=np.sqrt(s[axis])
        p=sparse.kron(sparse.eye(field['nphi']),-1j*(a-a.conj().T)/(np.sqrt(2)*engine.law['shell_width']),format='csr');qs=field_act(engine,name,q,state);ps=field_act(engine,name,p,state);j=-terms_act(engine,name,field['derivatives'][axis],state);dq=1j*(engine.H['coupled'][name]@qs-field_act(engine,name,q,hs))-ps/(engine.law['volume']/2);dp=1j*(engine.H['coupled'][name]@ps-field_act(engine,name,p,hs))-j;records.append(dict(Qdot_minus_Pi_over_I_norm=float(np.linalg.norm(dq)),Pidot_minus_original_J_norm=float(np.linalg.norm(dp))))
    return dict(raw_state_norm2=norm2(state),profiles=records,meaning='Finite-cut EOM witness only; no finite CCR/current repair or bound on connected tensor.')


def tensor_report(tensor: Any,orders: Any)->Any:
    allowed=[];forbidden=[];groups={name:[] for name in ['source_first_last_N55','source_first_last_N53','other_final_source_return','intermediate_source_return']};per_order=[]
    for oi,order in enumerate(orders):
        source_slots=sorted([order.index(0),order.index(3)]);end=source_slots[-1];branch='N55' if order.index(3)<order.index(0) else 'N53';group=('source_first_last_'+branch if source_slots==[0,3] else 'other_final_source_return' if end==3 else 'intermediate_source_return');pieces=[]
        for outsign,insign in product(range(2),repeat=2):
            expected=([-1,1][insign]-[-1,1][outsign])%3;piece=tensor[:,oi,expected,:,:,:,:,outsign,insign].ravel();allowed.append(piece);pieces.append(piece);groups[group].append(piece)
            for other in range(3):
                if other!=expected:forbidden.append(tensor[:,oi,other,:,:,:,:,outsign,insign].ravel())
        per_order.append(dict(order_index=oi,chronological_order=list(order),source_branch=branch,return_class=group,allowed_connected_Frobenius=float(np.linalg.norm(np.concatenate(pieces)))))
    def magnitude(values: Any)->float:return float(np.linalg.norm(np.concatenate(values))) if values else 0.
    return dict(allowed_connected_Frobenius=magnitude(allowed),forbidden_connected_Frobenius=magnitude(forbidden),source_return_class_allowed_norms={name:magnitude(values) for name,values in groups.items()},per_order=per_order,equal_time_dressing_norm=float(np.linalg.norm(tensor[0])),last_minus_equal_time_dynamical_variation_norm=float(np.linalg.norm(tensor[-1]-tensor[0])),each_spacing_connected_norm=[float(np.linalg.norm(value)) for value in tensor])


def main()->None:
    started=time.monotonic();request:Any=json.loads(Path(sys.argv[1]).read_text());output=Path(sys.argv[2]);limit=request['memory_limit_MiB']*1024**2;resource.setrlimit(resource.RLIMIT_AS,(limit,limit))
    for name,expected in request['required_input_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Frozen source/state/input bytes changed: '+name)
    config=json.loads((ROOT/request['primary_input']).read_text());law=json.loads((ROOT/config['law_input']).read_text());orth=json.loads((ROOT/config['orthogonal_input']).read_text());config.update({k:orth[k] for k in ['local_singular_relative_cut','tensor_singular_relative_cut']});engine=Engine(config,law,np.load(ROOT/config['reconstruction']));snap=np.load(ROOT/request['primary_states']);saved:Any=dict(request,states=snap);author=np.load(ROOT/request['primary_arrays']);orders,requests,pairs=request_dag();spacings=request['time_spacings'];times=sorted({round(t*m,12) for t in spacings for m in range(4)});arrays={};report={};bindings={};rng=np.random.default_rng(20201015)
    for name,sec in engine.model['sectors'].items():
        probe=rng.normal(size=sec['h'].shape[0])+1j*rng.normal(size=sec['h'].shape[0]);probe/=np.linalg.norm(probe);a=engine.H['coupled'][name]@probe;b=sec['h']@probe;bindings[name]=dict(dimension=len(probe),independent_H_vs_frozen_install_relative_action_error=float(np.linalg.norm(a-b)/np.linalg.norm(b)),actual_adjoint_bilinear_error=float(abs(np.vdot(probe,a)-np.vdot(engine.H['coupled'][name].H@probe,probe))))
    derivatives=derivative_evidence(engine,saved,arrays);off,offfactor=control_ground(engine);references={'coupled':snap['coupled_reference'],'EM_off_factorized':off};reference_bindings={'EM_off_saved_vs_new_factorized_overlap':float(abs(np.vdot(off,snap['EM_off_factorized_reference']))),'factorization':offfactor}
    for arm,omega in references.items():
        reference=engine.set_reference(arm,omega);singles={name:[] for name in engine.n};first={}
        for op in range(28):
            name,value=engine.act('neutral',op,omega);first[op]=(name,len(singles[name]));singles[name].append(value)
        singles={name:np.column_stack(value) for name,value in singles.items()};state_binding={name:float(np.linalg.norm(value-snap[arm+{'neutral':'_field_sources','positive':'_positive_sources','negative':'_negative_sources'}[name]]*(np.vdot(snap[arm+'_reference'],omega)))) for name,value in singles.items()};gram_plus=singles['positive'].conj().T@singles['positive'];gram_minus=singles['negative'].conj().T@singles['negative'];car=gram_plus+gram_minus.T-np.eye(12);wrongjump=gram_plus-gram_minus.T-np.eye(12);arrays[arm+'_reference']=omega
        for name,value in singles.items():arrays[arm+'_'+name+'_initial_sources']=value
        reference['source_snapshot_phase_aligned_errors']=state_binding;reference['ground_CAR_jump_error']=float(np.linalg.norm(car));reference['wrong_bosonic_removal_sign_jump_error']=float(np.linalg.norm(wrongjump));reference['raw_initial_source_norms']={name:np.sum(abs(value)**2,axis=0).tolist() for name,value in singles.items()};fieldweights=[dict(profile=axis,full_beforecut_norm2=engine.full_action_norm2('neutral',axis,omega),retained_norm2=norm2(singles['neutral'][:,axis]),omitted_norm2=engine.full_action_norm2('neutral',axis,omega)-norm2(singles['neutral'][:,axis])) for axis in range(4)];two={}
        for t in times:two[t]={name:value.conj().T@engine.propagate(name,value,t) for name,value in singles.items()}
        for name in engine.n:arrays[arm+'_own_two_'+name]=np.array([two[t][name] for t in times])
        fulls=[];discs=[];connecteds=[];losses=[]
        for tau in spacings:
            full,disc,audits,catalog=run_surface(engine,tau,orders,requests,pairs,singles,first,two);ff=fourier(full);dd=fourier(disc);cc=ff-dd;fulls.append(ff);discs.append(dd);connecteds.append(cc)
            for direction,groups in audits.items():
                for name,records in groups.items():
                    arrays[f'{arm}_tau{tau}_{direction}_{name}_weights']=records
                    if len(records):
                        row=int(np.argmax(records[:,3]));losses.append(dict(spacing=tau,direction=direction,sector=name,operator_pair=catalog[direction][name][row],beforeNorm2=float(records[row,0]),fullPhysicalActionNorm2=float(records[row,1]),retainedNorm2=float(records[row,2]),omittedDifference=float(records[row,3])))
            arrays[arm+'_full']=np.array(fulls);arrays[arm+'_disconnected']=np.array(discs);arrays[arm+'_connected']=np.array(connecteds);np.savez_compressed(output.with_name(output.name.replace('_result.json','_checkpoint.npz')),**arrays);output.with_name(output.name.replace('_result.json','_checkpoint.json')).write_text(json.dumps(dict(arm=arm,completed_spacings=spacings[:len(fulls)],elapsed_seconds=time.monotonic()-started))+'\n');print(json.dumps(dict(arm=arm,spacing=tau,connected_norm=float(np.linalg.norm(cc)),elapsed_seconds=time.monotonic()-started)),flush=True)
        tensor=np.array(connecteds);data=tensor_report(tensor,orders);data.update(reference=reference,raw_pair_maxima=losses,raw_field_action_weights=fieldweights,raw_DAG_catalog=catalog,own_vs_author_shared_time_tensor_errors={str(tau):dict(full_error=float(np.linalg.norm(fulls[i]-author[arm+'_four_full_centered'][config['time_spacings'].index(tau)])),disconnected_error=float(np.linalg.norm(discs[i]-author[arm+'_four_disconnected'][config['time_spacings'].index(tau)])),connected_error=float(np.linalg.norm(connecteds[i]-author[arm+'_four_connected'][config['time_spacings'].index(tau)]))) for i,tau in enumerate(spacings) if tau in config['time_spacings']},wrong_normalized_G_but_raw_four_point_control_norm=float(np.linalg.norm(np.array(fulls)-2*np.array(discs))),source_return_interpretation='A source returning only at final vacuum bra has no additional neutral-source projection error: <Omega|P0=<Omega|. Intermediate neutral return before remaining field insertion/evolution can depend on omitted neutral channels. Maximum ket pair[C0,A0] is intermediate for orders[3,0,1,2]/[3,0,2,1], not source-first/source-last[3,1,2,0]/[3,2,1,0]. Charged H/field cuts remain separate, no action deficit is a global tensor bound.')
        if arm=='coupled':data['actual_EOM_defects']={name:eom(engine,name,omega if name=='neutral' else singles[name][:,0]) for name in engine.n}
        report[arm]=data
    result=dict(kind='NEW_INDEPENDENT_FULL_SPAN_SAME_LAW_CONNECTED_CHARGED_TWO_FIELD_PHYSICAL_CONSUMER',request=request,bindings=bindings,basis_bindings=engine.model['basis_bindings'],independent_physical_mixed_TT_source_check=engine.projection_check,reference_bindings=reference_bindings,chronological_orders=[list(order) for order in orders],time_spacings=spacings,two_point_times=times,derivative_current_contact=derivatives,reports=report,wrong_reused_entangled_reference_equal_time_connected_norm=report['coupled']['equal_time_dressing_norm'],meaning='Full physical27/162 GLOBAL TT spans times scalar5/field15 retained. Both actual55/53 sectors, all24 orders, all3kz/4canonical components/2pols/2q signs, own full-H G/D subtraction, no raw normalization/filter. Factorized EM-off goes through SAME generic full-H propagation and request-DAG consumer. Canonical spinor tensor is not isolated-spin PVM; T cumulant is not retarded scattering/on-shell/photon amplitude; no Pcm=J or cross-law current-tail import.',limits=['Source-coordinate maps~1e-10 distinct from whole-H precision; initial CAR closure not all-pair closure.','Return loss can be substantial only on stated reference/action. Final-exclusive overlap versus intermediate neutral projection distinguished; not rigorous connected-tensor error bound.','Finite-shell Q/P/source EOM defects, inherited H omission.723842, severe3cells/shell2/oneq/scalar5/frozenframe remain.','No low16 residue replacement, restmass/photon/amputation, spinflip/helicity/proton/Nature/COM admission or second contact insertion.'],elapsed_seconds=time.monotonic()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    output.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(output.with_suffix('.npz'),**arrays);print(json.dumps(dict(result=output.name,summary={arm:{key:value for key,value in data.items() if key in ['allowed_connected_Frobenius','forbidden_connected_Frobenius','source_return_class_allowed_norms','last_minus_equal_time_dynamical_variation_norm']} for arm,data in report.items()},elapsed_seconds=result['elapsed_seconds'],peak_RSS_KiB=result['peak_RSS_KiB'])),flush=True)


if __name__=='__main__':main()
