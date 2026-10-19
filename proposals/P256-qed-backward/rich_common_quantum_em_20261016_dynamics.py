"""Fresh physical neutral-pair source and reciprocal rich matter/field responses.
Only new neutral and neutral PAIR sectors: no charged-sector subtraction.
"""
from pathlib import Path
import gc,json,sys,time,resource
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from rich_common_quantum_em_20261016_operator import Model,ChargeOperator


def observable(model,kind,control=False,theta=0.,species='white',edge=2):
    return model.observer(kind,theta=theta,species=species,edge=edge,source_species='white',source_edge=2,decoupled=control)


def prepare_pair(neutral,joint,reference,source_prefix,out):
    source=sparse.load_npz(source_prefix+'_source_orth.npz');gram=sparse.load_npz(source_prefix+'_source_full_CAR_gram_orth.npz');x=reference.reshape(neutral.n,len(neutral.grid),len(neutral.flux));flat=x.reshape(neutral.n,-1);shift=np.asarray(neutral.geometry['Wilson_transverse_shift']);where={tuple(m):i for i,m in enumerate(joint.flux)};inputs=[];outputs=[]
    for i,m in enumerate(neutral.flux):
        j=where.get(tuple(m+shift))
        if j is not None:inputs.append(i);outputs.append(j)
    full_raw=float(np.vdot(flat,gram@flat).real);raw_after_flux=float(np.vdot(x[:,:,inputs].reshape(neutral.n,-1),gram@x[:,:,inputs].reshape(neutral.n,-1)).real);unshifted=source@flat;body_before_flux=float(np.vdot(unshifted,unshifted).real);pair=np.zeros((joint.n,len(joint.grid),len(joint.flux)),complex);pair[:,:,outputs]=(source@x[:,:,inputs].reshape(neutral.n,-1)).reshape(joint.n,len(joint.grid),len(inputs));retained=float(np.vdot(pair,pair).real)
    if retained<=0:raise ValueError('Actual canonical physical pair source has zero retained strength')
    pair=pair.reshape(-1)/np.sqrt(retained);coo=source.tocoo();delta=np.array([1,0,-1,0]);wards=[float(np.linalg.norm(coo.data*(joint.q[coo.row,s]-neutral.q[coo.col,s]-delta[s])))for s in range(4)];integer_delta=3*delta;edge_integer_flow=integer_delta@neutral.lp@neutral.d+shift@neutral.gti@neutral.t.T;endpoint=edge_integer_flow@neutral.d.T-integer_delta
    np.savez_compressed(out+'_pair_preparation.npz',state=pair,neutral_reference=reference,Wilson_shift=shift,integer_E_edge_shift=edge_integer_flow)
    receipt={'source':'P_body P_flux [Bdag0 Wilson(path2->0->1->2->0) Ldag2] on FRESH new-H neutral reference; L FIRST inside complete CAR word','no_intermediate_Gauss_or_body_projection_after_L_or_B':True,'why_implementation_factorizes':'Body projector commutes with Wilson field shift; canonical full CAR source assembled before target projection. Gauss chi phase is included by final physical-basis lift, not by projecting either charged factor.','raw_full_CAR_source_strength_NOT_probability':full_raw,'raw_source_strength_after_flux_cut':raw_after_flux,'target_body_strength_before_flux_cut':body_before_flux,'retained_body_AND_flux_strength':retained,'raw_CAR_body_lost_strength':full_raw-body_before_flux,'raw_CAR_flux_lost_strength':full_raw-raw_after_flux,'target_body_flux_lost_strength':body_before_flux-retained,'retained_fraction_of_raw_source_NOT_particle_probability':retained/full_raw,'Wilson_transverse_shift':shift.tolist(),'Wilson_INTEGER_edge_shift_projected_to_retained_configuration_manifold':edge_integer_flow.tolist(),'Wilson_PHYSICAL_edge_flux_shift':(edge_integer_flow/3).tolist(),'Wilson_endpoint_integer_Gauss_residual':float(np.linalg.norm(endpoint)),'source_physical_site_charge_commutator_residuals':wards,'neutral_total_occupation_values':np.unique(neutral.counts.sum(axis=1)).tolist(),'pair_total_occupation_values':np.unique(joint.counts.sum(axis=1)).tolist(),'neutral_total_physical_Q_values':np.unique(neutral.q.sum(axis=1)).tolist(),'pair_total_physical_Q_values':np.unique(joint.q.sum(axis=1)).tolist(),'neutral_Noether_J':0,'pair_Noether_J_M':[1,1],'spin_half_components_are_primitive_source_inputs_not_separately_solved_particles':True,'normalization':'Single conditional normalization by retained raw source strength; NO expectation clamping or subsequent state renormalization.','no_old_solved_state_coefficients_used':True};Path(out+'_pair_source_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');return pair,receipt


def sample(model,h,state,labels,time_value,control,theta,current,contact,save_contact):
    hs=h@state;currents={species:np.einsum('ij,ij->j',state.conj(),operator@state).real for species,operator in current.items()};contacts={species:np.einsum('ij,ij->j',state.conj(),operator@state).real for species,operator in contact.items()}if save_contact else {};result={}
    for column,label in enumerate(labels):
        psi=state[:,column];event={'time':time_value,**model.readout(psi),'own_H_preparation_or_evolution_energy':float(np.vdot(psi,hs[:,column]).real),'physical_current':{species:float(value[column])for species,value in currents.items()},'physical_Hsecond_contact':{species:float(value[column])for species,value in contacts.items()}}
        x=psi.reshape(model.n,len(model.grid),len(model.flux));hx=hs[:,column].reshape(x.shape);event['actual_transverse_generator_force']=[float(2*np.vdot(x*model.flux[None,None,:,a],hx).imag)for a in range(2)];result[label]=event
    return result


def trajectory(model,initial,labels,config,out,control=False,theta=0.):
    h=model.hamiltonian(control,theta);current={species:observable(model,'current',control,theta,species,edge)for species,edge in(('quark',0),('white',2))};contact={species:observable(model,'contact',control,theta,species,edge)for species,edge in(('quark',0),('white',2))};state=np.column_stack(initial);times=config['times'];events={label:[]for label in labels};previous=0.;trace=h.trace();started=time.time()
    for index,t in enumerate(times):
        if t>previous:state=expm_multiply((-1j*(t-previous))*h,state,traceA=(-1j*(t-previous))*trace)
        rows=sample(model,h,state,labels,t,control,theta,current,contact,index in(0,len(times)-1))
        for label,event in rows.items():events[label].append(event)
        if index in(0,len(times)-1):np.savez_compressed(out+'_t'+str(index)+'_states.npz',states=state,labels=np.asarray(labels),time=t)
        Path(out+'_checkpoint.json').write_text(json.dumps({'events':events,'completed_time':t,'control_transverse_links_identity':control,'external_white_phase':theta,'seconds':time.time()-started},indent=2)+'\n');print(json.dumps({'stage':'actual_NEW_quantum_charge_link_time_evolution','out':out,'time':t,'columns':labels,'control_transverse_links_identity':control,'external_white_phase':theta,'seconds':time.time()-started,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}),flush=True);previous=t
    return events


def difference(event,baseline):
    result={}
    for key in('Q_site','quark_Q_site','white_Q_site','compact_quark_count_site_NOT_tagged_particle_PVM','flux_mean','flux_second','loop_sine','loop_cosine','actual_transverse_generator_force'):
        result[key]=(np.asarray(event[key])-np.asarray(baseline[key])).tolist()
    result['physical_current']={s:event['physical_current'][s]-baseline['physical_current'][s]for s in('quark','white')};result['Phi']=event['Phi']-baseline['Phi'];result['Phi2']=event['Phi2']-baseline['Phi2'];return result


def contrast(pair,neutral):
    result={key:(np.asarray(pair[key])-np.asarray(neutral[key])).tolist()for key in pair if key not in('time','physical_current','Phi','Phi2')};result['physical_current']={s:pair['physical_current'][s]-neutral['physical_current'][s]for s in('quark','white')};result['Phi']=pair['Phi']-neutral['Phi'];result['Phi2']=pair['Phi2']-neutral['Phi2'];return result


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=config['out'];started=time.time();neutral=Model(config['neutral_matter_prefix']);joint=Model(config['joint_matter_prefix'])
    with np.load(config['reference_prefix']+'_states.npz')as z:reference=z['states'][:,0].copy();own_reference_energy=float(z['energies'][0])
    pair,source_receipt=prepare_pair(neutral,joint,reference,config['source_prefix'],out);result={'kind':'SAME_NEW_H_CLOSED_GAUSS_NEUTRAL_PAIR_AND_RECIPROCAL_QUANTUM_FIELD_RESPONSE','own_new_neutral_reference_energy':own_reference_energy,'source':source_receipt,'models':{'neutral':neutral.receipt,'pair':joint.receipt},'predeclared_config':config,'sectors':{}}
    for sector,model,psi in(('neutral',neutral,reference),('pair',joint,pair)):
        x=psi.reshape(model.n,len(model.grid),len(model.flux));kicks=[(x*np.exp(1j*config['beta']*model.flux[None,None,:,a])).reshape(-1)for a in range(2)];sector_result={}
        for control in(False,True):
            name='full_live_quantum_links'if not control else'TRANSVERSE_link_identity_active_matter_source_control';h=model.hamiltonian(control);hp=model.hamiltonian(control,config['theta']);activation=float(np.linalg.norm(hp@psi-h@psi));block=trajectory(model,[psi,*kicks],['baseline','field_kick_y','field_kick_z'],config,out+'_'+sector+'_'+name,control);pulse=trajectory(model,[psi],['white_pulse'],config,out+'_'+sector+'_'+name+'_white',control,config['theta']);block.update(pulse);block['external_white_source_activation_state_norm']=activation;block['same_age_responses']={label:[{'time':row['time'],**difference(row,base)}for row,base in zip(block[label],block['baseline'])]for label in('field_kick_y','field_kick_z','white_pulse')};sector_result[name]=block
            Path(out+'_partial_result.json').write_text(json.dumps({'completed_sectors':result['sectors'],'current_sector':sector,'current_results':sector_result,'source':source_receipt},indent=2)+'\n');gc.collect()
        result['sectors'][sector]=sector_result
    result['added_neutral_pair_source_contrasts']={}
    for control in('full_live_quantum_links','TRANSVERSE_link_identity_active_matter_source_control'):
        p=result['sectors']['pair'][control];n=result['sectors']['neutral'][control];result['added_neutral_pair_source_contrasts'][control]={'definition':'[pair kicked - pair same-age baseline] - [neutral kicked - neutral same-age baseline], each own conditional normalized preparation in SAME direct-sum new H. Not four charged sectors or a scattering S matrix.','baseline_pair_minus_neutral':[{'time':pp['time'],**difference(pp,nn)}for pp,nn in zip(p['baseline'],n['baseline'])],'reciprocal_responses':{label:[{'time':pp['time'],**contrast(pp,nn)}for pp,nn in zip(p['same_age_responses'][label],n['same_age_responses'][label])]for label in('field_kick_y','field_kick_z','white_pulse')}}
    result['interpretation']='Actual Gauss-physical source and rich sea/field dynamics in an explicitly fundamental SOURCE-SELECTED finite-matrix matter law. Source/body/flux compression diagnosed, not hidden. Both signed field generators and all Phi7/CAR/PW/R states in declared spaces propagate. No old states, pointcharge compound, charged-sector subtraction, sourceOFF-as-mediator-control, expectation clamping, Dyson photon, binding/mass/lifetime or labEM claim.';result['resources']={'elapsed_seconds':time.time()-started,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'stage':'NEW_same_law_quantum_field_pair_consumer_complete',**result['resources']}),flush=True)


if __name__=='__main__':main()
