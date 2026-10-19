"""Own fresh CAR32 physical states on a hash-bound operator-only compiled47 law."""
from pathlib import Path
import hashlib
import json
import os
import pickle
import sys
import numpy as np
from scipy.linalg import expm
from tenpy.networks.mps import MPS
import rich_local_frame_spatial_carrier_20261016_model as law
import rich_spatial_frame_photon_20261016_fields as fields
from rich_review_wave49_20261016_integrated_storage import save_state,restore,stage_binding,load_stage,save_stage,source_operator

ROOT=Path(__file__).resolve().parent;P='rich_review_wave49_20261016_'
ADVANCE_LEDGERS=[]


def load_compiled(path,expected):
    path=Path(path)
    with path.open('rb') as stream:
        actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=expected:raise ValueError(('compiled operator hash mismatch',actual,expected))
        stream.seek(0);model=pickle.load(stream)
    if any(isinstance(v,MPS) for v in model.__dict__.values()):raise ValueError('operator artifact contains a state')
    model._review_base_sha=actual
    model._review_base_RF_names=set(model.ops[model.rfsite].values())
    model._review_capture_receipt=os.environ.get('REVIEW49_INPUT_RECEIPT')
    return model,dict(path=str(path),sha256=actual,model_signature=model.signature,term_count=len(model.terms),parameters=model.parameters)


def norm2(psi):
    if np.iscomplexobj(psi.norm) or psi.norm<0:raise ValueError('MPS norm metadata must be real nonnegative; phases belong in tensors')
    return float(psi.overlap(psi).real)


def advance(m,psi,H=None,dt=.02,steps=1,chi=16,label='',initial_time=0.):
    """Restore raw factors/phases and persist each completed OWN physical step."""
    out=psi.copy();operator=m.H if H is None else H;ledgers=[];removed=0.
    binding=stage_binding(m,operator,psi,dt,steps,chi,initial_time)
    prior=load_stage(m,label,binding)
    if prior is not None:
        out,checkpoint=prior;ledgers=checkpoint['ledgers'];removed=checkpoint['removed']
    for _ in range(len(ledgers),steps):
        ledger=operator.evolve(out,dt,chi);ledgers.append(ledger)
        removed+=ledger['log_removed_raw_norm_factor']
        if not ledger['centering_phase_restored']:raise ValueError('missing physical centering phase')
        norm2(out)
        save_stage(m,label,binding,out,removed,ledgers)
    # evolve() normalizes every intermediate stage. Its first factor includes
    # any unnormalized source/current-image input amplitude: restore it ONCE.
    out.norm*=np.exp(removed)
    record=dict(label=label,dt=dt,steps=steps,chi_cap=chi,log_restored_raw_norm_factor=removed,final_raw_norm2=norm2(out),steps_ledger=ledgers)
    index=len(ADVANCE_LEDGERS);ADVANCE_LEDGERS.append(record)
    stages=[s for v in ledgers for s in v['stages']]
    info=dict(dt=dt,steps=steps,chi_cap=chi,stage_count=len(stages),svd_eps_sum=float(sum(v['svd_eps'] for v in stages)),svd_eps_max=float(max((v['svd_eps'] for v in stages),default=0)),log_restored_raw_norm_factor=removed,restored_centering_phase_exponent=float(sum(v['restored_scalar_phase_exponent'] for v in ledgers)),raw_norm2=norm2(out),ledger_index=index,normalization_and_phase_restored=True,method='Exact operator SUM; approximate symmetric group/RK2 dynamics plus MPS compression, not exponential or convergence proof')
    return out,info


def recipe(m,name):
    key=next((k for k,v in m.ops[m.rfsite].items() if v==name),None)
    return m.rfrecipes.get(key)


def choose_terms(m,whole=False):
    result=[]
    for term in m.terms:
        rf=next((name for name,i in term[1] if i==m.rfsite),None);r=recipe(m,rf)
        if r is not None and (whole or r[0]=='phase'):continue
        result.append(term)
    return result


def source_for_terms(m,terms,mode=0,label='convection_J0'):
    return source_operator(m,mode,terms=terms,label=label)


def normalized(psi):
    out=psi.copy();out.norm/=np.sqrt(norm2(out));return out


def measure(m,psi,J):
    """Every observable is a normalized mean; raw physical norm is separate."""
    raw=norm2(psi);state=normalized(psi);out=m.measure(state)
    out['norm2']=raw;out['observable_normalization']='Every displayed mean divides by the physical state norm²; raw norm and tensor phases retained separately.'
    out['local_LEFT_Casimir']=[float(law.physical_element(state,m.observable(site,m.g['casimir'].toarray()),state).real) for site in m.gsites]
    out['physical_lab_current']=[float(law.physical_element(state,j,state).real) for j in J]
    out['field_shell_weight']=float(law.physical_element(state,m.observable(m.rfsite,np.kron(np.eye(m.nR),np.diag([sum(v)==m.parameters['field_cap'] for v in m.basis]))),state).real)
    basis=[tuple(v) for v in m.basis]
    out['physical_field_Pi']=[float(law.physical_element(state,m.observable(m.rfsite,np.kron(np.eye(m.nR),fields.primitive(basis,basis,a,'p').toarray())),state).real) for a in (0,4)]
    out['total_n_fraction']=(np.array(out['n_cells'])/sum(out['n_cells'])).tolist()
    return out




def product_element(m,terms,bra,ket):
    bv=[bra.get_B(i,form='B').to_ndarray().reshape(m.sites[i].dim) for i in range(bra.L)]
    kv=[ket.get_B(i,form='B').to_ndarray().reshape(m.sites[i].dim) for i in range(ket.L)]
    cache={};total=0j
    for coefficient,factors in terms:
        indices=dict((i,name) for name,i in factors);value=complex(coefficient)
        for i in range(ket.L):
            name=indices.get(i,'Id');key=(i,name)
            if key not in cache:cache[key]=np.vdot(bv[i],m.sites[i].get_op(name).to_ndarray()@kv[i])
            value*=cache[key]
            if abs(value)<1e-20:break
        total+=value
    return total*np.conj(bra.norm)*ket.norm


def cache_probe(m):
    assert len(m.terms)==3529 and len(m.fsites)==32 and m.ng==55 and m.nR==35 and m.nf==9
    assert [t for group in m.H.groups for t in group]==m.terms
    old=np.load(ROOT/(P+'local_primitives_raw.npz'));D=old['source_multipliers']
    derror=max(float(np.linalg.norm(a-D[mu,s])) for (mu,s),a in m.D.items())
    seed=m.seed();seed_direct=product_element(m,m.terms,seed,seed);seed_mpo=law.physical_element(seed,m.H,seed)
    half=[i for i,(j,x,y) in enumerate(m.g['labels']) if j==.5 and x==-.5]
    gi=max(half,key=lambda i:sum(abs(a[i,0])**2 for (mu,s),a in m.D.items() if s==0))
    def charged(cell):
        labels=[]
        for v in range(4):
            f=m.ions[8*v:8*v+8].copy()
            if v==cell:f[0]=1
            labels+=['1',str(gi if v==cell else 0)]+[str(x) for x in f]
        labels+=['0']
        return MPS.from_product_state(m.sites,labels,dtype=complex,unit_cell_width=1)
    ket=charged(0);bra=charged(1)
    hop_direct=product_element(m,m.terms,bra,ket);hop_mpo=law.physical_element(bra,m.H,ket)
    off_direct=product_element(m,choose_terms(m),bra,ket)
    assert derror<1e-10 and abs(seed_direct-seed_mpo)<1e-9 and abs(hop_direct-hop_mpo)<1e-9
    assert abs(hop_direct)>1e-8 and abs(off_direct)<1e-12
    scaled=seed.copy();scaled.norm*=2
    identity=law.PhysicalOperatorSum(m,[m.term(3.7,{m.rsites[0]:np.eye(3)})])
    phase_state,phase_ledger=advance(m,scaled,identity,dt=.02,label='actual_state_centering_phase_control')
    amplitude=scaled.overlap(phase_state)/norm2(scaled);expected=np.exp(-1j*3.7*.02)
    phase_error=float(abs(amplitude-expected));wrong_unrestored=float(abs(amplitude-1))
    assert phase_error<1e-9 and wrong_unrestored>.01 and abs(norm2(phase_state)-4)<1e-9
    source0,s0=m.source(seed,0,0,True,chi=16);q0=dict(m.last_source_quality)
    phase_seed=scaled.copy();phase_seed.set_B(0,np.exp(.37j)*phase_seed.get_B(0,form='B'),form='B')
    source1,s1=m.source(phase_seed,0,0,True,chi=16);q1=dict(m.last_source_quality)
    source_phase_error=float(abs(source0.overlap(source1)-np.exp(.37j)))
    assert abs(s1-4*s0)<1e-9 and source_phase_error<1e-9
    return dict(exact_cached_D_vs_own55g_audit=derror,seed_H_sum_error=float(abs(seed_direct-seed_mpo)),sewn_charge_PLUS_LEFT_transfer_matrix_element=[float(hop_mpo.real),float(hop_mpo.imag)],independent_direct_product_error=float(abs(hop_direct-hop_mpo)),hopping_OFF_transfer=float(abs(off_direct)),full_graph_audit=m.graph_audit,centering_phase_control=dict(actual_amplitude=[float(amplitude.real),float(amplitude.imag)],analytic_error=phase_error,wrong_unrestored_phase_error=wrong_unrestored,raw_input_output_norm2=[norm2(scaled),norm2(phase_state)],ledger=phase_ledger),source_norm_phase_control=dict(raw_strength_scale_error=float(abs(s1-4*s0)),source_phase_error=source_phase_error,source0_quality=q0,scaled_source_quality=q1),meaning='Own product basis states carry charge and physical local LEFT at distinct cells; all3529 terms retained, independent direct factor matrix elements match lazy bounded-MPO contractions. Actual full-site states exercise scalar centering phases and source weight/phase scaling; no producer evolved state.')


def run():
    m,input_record=load_compiled(sys.argv[1],sys.argv[2])
    probe_path=ROOT/(P+'integrated_operator_probe.json')
    captured=json.loads(probe_path.read_text()) if probe_path.exists() else None
    if captured is not None:
        if captured['compiled_operator_input']['sha256']!=input_record['sha256']:raise ValueError('old primitive input mismatch')
        probe=captured['probe']
    else:
        probe=cache_probe(m)
        probe_path.write_text(json.dumps(dict(compiled_operator_input=input_record,probe=probe),indent=2)+'\n')
    neutral,ne=advance(m,m.seed(),dt=.004,steps=1,label='neutral_preparation')
    J=[source_operator(m,a) for a in (0,4)];contact=source_operator(m,0,4)
    nohopterms=choose_terms(m);wholeterms=choose_terms(m,whole=True)
    nohopH=m.mpo(nohopterms);wholeH=m.mpo(wholeterms)
    convection=source_for_terms(m,nohopterms);wholeJ=source_for_terms(m,wholeterms,label='field_decoupled_J0')
    neutral_coarse,nc=advance(m,neutral,dt=.02,label='neutral_coarse',initial_time=.004)
    neutral_fine,nf=advance(m,neutral,dt=.01,steps=2,label='neutral_fine',initial_time=.004)
    neutral0=measure(m,neutral,J);nco=measure(m,neutral_coarse,J);nfi=measure(m,neutral_fine,J)
    records={};arrays={};save_state(arrays,'neutral',neutral);save_state(arrays,'neutral_coarse',neutral_coarse);save_state(arrays,'neutral_fine',neutral_fine)
    assertions={}
    for create in (True,False):
        branch='addition' if create else 'removal';strengths=[];qualities=[]
        for component in range(8):
            _,strength=m.source(neutral,0,component,create,chi=16);strengths.append(strength);qualities.append(dict(m.last_source_quality))
        generator=-m.geo['spin'][2].T if create else m.geo['spin'][2]
        eigenvalues,eigenvectors=np.linalg.eigh(generator);possible=[i for i,v in enumerate(eigenvalues) if v>.49]
        mu=max(possible,key=lambda i:np.dot(abs(eigenvectors[:,i])**2,strengths));polarization=eigenvectors[:,mu]
        state,raw_strength=m.source(neutral,0,polarization,create,chi=16);quality=dict(m.last_source_quality)
        before=measure(m,state,J);unitary=np.kron(np.eye(m.nR),expm(.19j*m.Q[0]+.13j*m.Q[4]))
        state.apply_local_op(m.rfsite,m.register(m.rfsite,unitary),unitary=True,renormalize=False)
        state=normalized(state);initial=measure(m,state,J)
        coarse,ce=advance(m,state,dt=.02,label=branch+'_coarse',initial_time=.004);fine,fe=advance(m,state,dt=.01,steps=2,label=branch+'_fine',initial_time=.004)
        nohop,he=advance(m,state,nohopH,dt=.01,steps=2,label=branch+'_nohop',initial_time=.004);whole,we=advance(m,state,wholeH,dt=.01,steps=2,label=branch+'_whole_OFF',initial_time=.004)
        co=measure(m,coarse,J);fi=measure(m,fine,J);wo=measure(m,whole,J)
        charge=-(np.array(fi['n_cells'])-nfi['n_cells']);spin=np.array(fi['local_LEFT_Casimir'])-nfi['local_LEFT_Casimir']
        charge0=-(np.array(initial['n_cells'])-neutral0['n_cells']);spin0=np.array(initial['local_LEFT_Casimir'])-neutral0['local_LEFT_Casimir']
        dc=charge-charge0;ds=spin-spin0;dv=(np.array(fi['local_LEFT'])-nfi['local_LEFT'])-(np.array(initial['local_LEFT'])-neutral0['local_LEFT'])
        differences={key:(np.array(fi[key])-co[key]).tolist() for key in ('n_cells','local_LEFT','local_r2','field_Q','physical_field_Pi','physical_lab_current')}
        glegs=[]
        for label,nstate,cstate in (('coarse',neutral_coarse,coarse),('fine',neutral_fine,fine)):
            observer,_=m.source(nstate,0,polarization,create,chi=16);observer_quality=dict(m.last_source_quality)
            amplitude=np.sqrt(quality['retained_source_norm2']*observer_quality['retained_source_norm2'])*observer.overlap(cstate)
            glegs.append(dict(discretization=label,source_weighted_twoleg=[float(amplitude.real),float(amplitude.imag)],observer_source_quality=observer_quality))
        image=state.copy();image_error=m.H.apply(image,{'trunc_params':{'chi_max':16,'svd_min':1e-12}})
        hnorm=norm2(image);piece_square_sum=float(sum(v['uncompressed_norm2'] for v in image_error.pieces))
        response=[]
        for index,j in enumerate(J):
            v=state.copy();err=j.apply(v,{'compression_method':'zip_up','trunc_params':{'chi_max':16,'svd_min':1e-12}})
            v,re=advance(m,v,dt=.002,label=branch+'_current_image_'+str(index),initial_time=.004);response.append((v,re,float(err.eps)))
        observed,oe=advance(m,state,dt=.002,label=branch+'_current_observer',initial_time=.004)
        c01=law.physical_element(observed,J[1],response[0][0]);c10=law.physical_element(observed,J[0],response[1][0])
        current=float(law.physical_element(normalized(state),J[0],normalized(state)).real)
        conv=float(law.physical_element(normalized(state),convection,normalized(state)).real);zero=float(law.physical_element(normalized(state),wholeJ,normalized(state)).real)
        overlap=abs(fine.overlap(whole))/np.sqrt(norm2(fine)*norm2(whole))
        assertions[branch]=dict(actual_initial_Noether_polarization=abs(before['local_LEFT'][0][2])>.1,local_excess_charge_moves=bool(np.linalg.norm(dc[1:])>1e-10),local_Noether_vector_moves=bool(np.linalg.norm(dv[1:])>1e-10),whole_charge_motion_OFF_source=abs(zero)<1e-10)
        records[branch]=dict(source_strengths_all8=strengths,source_quality_all8=qualities,source_strength=raw_strength,source_quality=quality,source_polarization=[[float(c.real),float(c.imag)] for c in polarization],initial=initial,final=fi,coarse_final=co,whole_OFF=wo,hopping_OFF=measure(m,nohop,J),evolved_neutral=nfi,physical_excess_charge=charge.tolist(),physical_excess_local_LEFT=spin.tolist(),excess_charge_change=dc.tolist(),excess_LEFT_change=ds.tolist(),physical_Noether_vector_density_change=dv.tolist(),wrong_global_spin_density_error=float(np.linalg.norm(dv)),physical_source_current=current,hopping_OFF_convection_current=conv,whole_charge_motion_OFF_current=zero,final_whole_OFF_overlap=float(overlap),kick_work=initial['energy']-before['energy'],same_H_energy_drift=fi['energy']-initial['energy'],mixed_contact=float(law.physical_element(normalized(state),contact,normalized(state)).real),evolution=dict(full=fe,coarse=ce,nohop=he,whole=we,neutral=nf),dt_halfdt=dict(time=.02,coarse_dt=.02,fine_dt=.01,normalized_observable_differences=differences,raw_phase_restored_overlap=[float(fine.overlap(coarse).real),float(fine.overlap(coarse).imag)],source_weighted_twoleg=glegs),coherent_compressed_H_image_norm2=hnorm,piece_squared_norm_sum_NOT_Hnorm=piece_square_sum,H_image_compression_eps_sum=float(image_error.eps),H_image_error_meaning=image_error.meaning,physical_current_time_orders=[[float(c01.real),float(c01.imag)],[float(c10.real),float(c10.imag)]],response_evolution=[response[0][1],response[1][1],oe],response_image_svd_eps=[response[0][2],response[1][2]],physics_assertions=assertions[branch])
        for label,psi in dict(initial=state,final=fine,coarse=coarse,nohop=nohop,whole=whole,neutral_final=neutral_fine,H_image=image).items():save_state(arrays,branch+'_'+label,psi)
        np.savez_compressed(ROOT/(P+'integrated_consumer_raw.npz'),**arrays)
        print(branch,json.dumps(records[branch]),flush=True)
    arrays.update(positions=m.positions,modes=m.modes,g_labels=np.array(m.g['labels']),material_labels=np.array(m.material['labels']),radial_nodes=m.rnodes,field_basis=m.basis,ions=m.ions)
    np.savez_compressed(ROOT/(P+'integrated_consumer_raw.npz'),**arrays)
    result=dict(kind='OWN_FRESH_FULLCAR32_INTEGRATED47_COMPILED_OPERATOR_CONSUMER',compiled_operator_input=input_record,operator_primitive_action_probe=probe,model_signature=m.signature,parameters=m.parameters,neutral_evolution=ne,neutral_dt_control=dict(coarse=nc,fine=nf),records=records,physics_assertions=assertions,evolution_ledger_file=P+'integrated_consumer_evolution_ledger.json',limits='No giant constructor or producer evolved state replay. Full frozen quantum law/all3529terms retained; own fresh MPS sources. Exact operator representation is separate from approximate splitting/RK2/compression dynamics, whose actual observables and source-weighted phase-sensitive twoleg amplitudes have dt/halfdt data. All global scalar phases are in physical tensors; real positive norm metadata and removed positive factors restored. Coherent H-image norm includes cross terms AFTER disclosed image compression; sum of piece norm² is not H². No full-H exterior, ground/convergence/localization/photon/mass closure.')
    (ROOT/(P+'integrated_consumer_evolution_ledger.json')).write_text(json.dumps(ADVANCE_LEDGERS,indent=2)+'\n')
    (ROOT/(P+'integrated_consumer_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    assert all(v for branch in assertions.values() for v in branch.values()),assertions

if __name__=='__main__':run()
