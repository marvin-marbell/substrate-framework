"""Fresh N33/N31 and ONE-Fock N36, with active-source mechanism controls."""
from pathlib import Path
import gc
import json
import resource
import time
import numpy as np
from scipy import sparse
from rich_composite_geometry_20261015_science import load
from rich_common_species_review_20261016_math import native_space,prepare,generator,pair,pair_operator,evolve,observe,continuity,count_distribution,total_variation
from rich_common_species_review_20261016_mechanism import original_source_audit,field_and_CAR_algebra,cross_action
ROOT=Path(__file__).resolve().parent
P='rich_common_species_review_20261016_'
O='rich_common_native_species_20261016_'
NAMES=('Gram','rotation','electric','plaquette','quark_vector','white_vector','quark_scalar','white_scalar','loopB','loopC')
FIELDS=('Phi','Phi2','Pphi','Phi_energy','rotation','loopB','loopC','electric')


def save_result(value):
    (ROOT/(P+'result.json')).write_text(json.dumps(value,indent=2)+'\n')


def finite_derivatives(space,state,species,edge,theta):
    eps=1e-4;plus=pair_operator(space,pair(space,species,edge,0,theta+eps));minus=pair_operator(space,pair(space,species,edge,0,theta-eps));middle=pair_operator(space,pair(space,species,edge,0,theta));actualj=pair_operator(space,pair(space,species,edge,1,theta));actualc=pair_operator(space,pair(space,species,edge,2,theta));phasefree=pair_operator(space,pair(space,species,edge,1,0.))
    current_error=float(np.linalg.norm(-(plus@state-minus@state)/(2*eps)-actualj@state));contact_error=float(np.linalg.norm((plus@state+minus@state-2*(middle@state))/(eps*eps)-actualc@state));wrong=float(np.linalg.norm((actualj@state)-(phasefree@state)))
    if current_error>2e-6 or contact_error>2e-5:raise AssertionError('Actual finite-theta current/contact disagrees with own pulsed-H derivative')
    return dict(theta=theta,current_action_difference=current_error,contact_action_difference=contact_error,wrong_dropped_finite_phase_current_action_error=wrong,source_current_initial=float(np.vdot(state,actualj@state).real),source_contact_initial=float(np.vdot(state,actualc@state).real))


def carrier(meta,operators,number,config):
    space=native_space(meta,operators,number);descriptor=dict(sector=number,site=config['carrier_source_site'],twice_J=1,family='bare');state,strength=prepare(space,descriptor);h=generator(space);off=generator(space,'ALLhopOFF');times=meta['p']['times'];initial=count_distribution(state,space['data'])
    trajectories={};raw=dict(prepared_y=state,times=np.array(times),sector_global_indices=space['keep'])
    for name,op,hops in [('full',h,True),('ALLhopOFF',off,False)]:
        states=evolve(op,state,times);trajectories[name]=[dict(t=float(t),count_TV=total_variation(count_distribution(v,space['data']),initial),**observe(space,v,op,hops_on=hops)) for t,v in zip(times,states)];raw[name+'_y']=np.array(states)
    if max(row['count_TV'] for row in trajectories['ALLhopOFF'])>2e-10:raise AssertionError('Carrier ALL scalar+vector hops OFF failed actual count consumer')
    edge=config['white_probe_edge'];theta=config['theta'];duration=config['pulse_duration'];delta=pair(space,'white',edge,0,theta);pulse=generator(space,delta=delta);after=evolve(pulse,state,[duration])[0];age=evolve(h,state,[duration])[0]
    events={}
    for name,seed in [('pulse',after),('same_age_no_pulse',age)]:
        states=evolve(h,seed,times);events[name]=[dict(t_after_pulse=float(t),absolute_age=float(t+duration),**observe(space,v,h)) for t,v in zip(times,states)];raw[name+'_y']=np.array(states)
    result=dict(N=number,body_dimension=space['n'],full_Phi_dimension=space['n']*space['nf'],source=strength,metric=space['metric'],Ward=continuity(space),trajectories=trajectories,events=events,finite_theta_derivatives=finite_derivatives(space,state,'white',edge,theta),scope='Independent canonical-source preparation and full native evolution, not imported prepared/eigen/evolved arrays. N33 and N31 are different number-sector source orders, not an intersector current or asserted charge-conjugate spectrum.')
    np.savez(ROOT/(P+f'N{number}_raw.npz'),**raw);print(json.dumps(dict(stage='fresh_common_carrier',N=number,source=strength,final=trajectories['full'][-1])),flush=True)
    return result


def baseline(space,state,config,mode):
    h=generator(space,mode);times=list(config['during_pulse_times'])+[config['pulse_duration']+t for t in config['after_pulse_times']];states=evolve(h,state,times);initial=count_distribution(state,space['data']);hops=mode!='ALLhopOFF'
    events=[dict(t_absolute=float(t),count_TV=total_variation(count_distribution(v,space['data']),initial),**observe(space,v,h,hops_on=hops)) for t,v in zip(times,states)]
    return times,states,events


def pulse_trajectory(space,state,config,mode,species):
    hops=mode!='ALLhopOFF';edge=config[species+'_probe_edge'];theta=config['theta'];during=config['during_pulse_times'];after=config['after_pulse_times'];duration=config['pulse_duration'];delta=pair(space,species,edge,0,theta,hops);full=generator(space,mode);pulsed=generator(space,mode,delta);early=evolve(pulsed,state,during);late=evolve(full,early[-1],after);states=early+late;times=list(during)+[duration+t for t in after]
    events=[]
    for i,(t,v) in enumerate(zip(times,states)):
        angle=theta if i<len(during) else 0.;op=pulsed if i<len(during) else full;events.append(dict(t_absolute=float(t),**observe(space,v,op,species,edge,angle,hops)))
    delta_action=pair_operator(space,delta)@state
    return states,events,dict(actual_source_DeltaH_action_norm2=float(np.vdot(delta_action,delta_action).real),source_theta=theta,source_edge=edge)


def response(events,base,source,config):
    receiver='quark' if source=='white' else 'white';rows=[]
    for event,reference in zip(events,base):
        if event['t_absolute']!=reference['t_absolute']:raise AssertionError('Response baseline not same absolute age')
        current=np.array(event['current'][receiver])-reference['current'][receiver];charge=np.array(event['Q'+receiver+'_site'])-reference['Q'+receiver+'_site'];fields=np.array([event[key]-reference[key] for key in FIELDS])
        rows.append(dict(t_absolute=event['t_absolute'],delta_receiver_current_all_edges=current.tolist(),delta_pinned_receiver_current=float(current[config[receiver+'_probe_edge']]),delta_receiver_charge_all_sites=charge.tolist(),delta_shared_fields=fields.tolist()))
    result=dict(source=source,receiver=receiver,rows=rows,max_abs_receiver_current=float(np.max(abs(np.array([row['delta_receiver_current_all_edges'] for row in rows])))),max_abs_receiver_charge=float(np.max(abs(np.array([row['delta_receiver_charge_all_sites'] for row in rows])))),max_abs_shared_fields=np.max(abs(np.array([row['delta_shared_fields'] for row in rows])),axis=0).tolist(),field_order=list(FIELDS))
    if max(abs(value) for value in rows[0]['delta_receiver_current_all_edges']+rows[0]['delta_receiver_charge_all_sites'])>2e-12:raise AssertionError('Other-species state response changed instantaneously at pulse onset')
    return result


def joint(meta,operators,config):
    space=native_space(meta,operators,36);descriptor=dict(sector=36,B_site=config['joint_initial_pair'][0],L_site=config['joint_initial_pair'][1],twice_J=2,family='bare');state,strength=prepare(space,descriptor)
    result=dict(N=36,total_J=1,total_M=1,body_dimension=space['n'],full_Phi_dimension=space['n']*space['nf'],source=strength,metric=space['metric'],Ward=continuity(space),field_CAR_algebra=field_and_CAR_algebra(meta),initial_cross_actions={},models={})
    for source in ('white','quark'):result['initial_cross_actions'][source]=cross_action(space,state,config,source)
    result['finite_theta_derivatives']={source:finite_derivatives(space,state,source,config[source+'_probe_edge'],config['theta']) for source in ('white','quark')}
    for mode in ('full','coordinate_only','ALLhopOFF'):
        times,base_states,base_events=baseline(space,state,config,mode);item=dict(baseline=base_events,pulses={},response={});raw=dict(prepared_y=state,times=np.array(times),baseline_y=np.array(base_states),sector_global_indices=space['keep'])
        for source in ('white','quark'):
            states,events,source_receipt=pulse_trajectory(space,state,config,mode,source);item['pulses'][source]=dict(events=events,**source_receipt);item['response'][source]=response(events,base_events,source,config);raw[source+'_pulse_y']=np.array(states)
        if mode=='ALLhopOFF':
            if max(row['count_TV'] for row in base_events)>2e-10:raise AssertionError('Joint actual ALLhopOFF material counts moved')
            if max(item['response'][source]['max_abs_receiver_current'] for source in ('white','quark'))>2e-10:raise AssertionError('Controlled zero hopping produced a current response')
        if mode=='coordinate_only':
            item['definition']='Remove Hphi kinetic, Hrotation, HEcolour-electric, RETAIN scalar potential, plaquette, ALL quark+white scalar/vector hops and unchanged active theta source/current/contact. This is NOT frozen classical fields or all mediators OFF: finite projected R/PW/body coordinate algebra remains quantum and noncommuting.'
            item['Phi2_baseline_conservation_error']=max(abs(row['Phi2']-base_events[0]['Phi2']) for row in base_events)
            item['active_source_action_norms2']={source:item['pulses'][source]['actual_source_DeltaH_action_norm2'] for source in ('white','quark')}
        np.savez(ROOT/(P+'joint_'+mode+'_raw.npz'),**raw);result['models'][mode]=item
        (ROOT/(P+'joint_partial_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(stage='fresh_joint_response',model=mode,responses={source:dict(current=item['response'][source]['max_abs_receiver_current'],charge=item['response'][source]['max_abs_receiver_charge']) for source in ('white','quark')})),flush=True);gc.collect()
    candidate=max(result['models']['coordinate_only']['response'][source]['max_abs_receiver_current'] for source in ('white','quark'))
    if candidate>1e-8:
        try:
            assert candidate<2e-10,'Hypothesis that field-kinetic backreaction is necessary for native cross-response rejected: active-source coordinate-only quantum model responds'
        except AssertionError as error:result['kinetic_necessity_hypothesis_assertion']=str(error)
    else:result['kinetic_necessity_hypothesis_assertion']='Coordinate-only response below chosen discrimination threshold; report raw result, do not infer all mechanisms absent.'
    result['whole_sea_charge_totals']=dict(quark=sum(result['models']['full']['baseline'][0]['Qquark_site']),white=sum(result['models']['full']['baseline'][0]['Qwhite_site']),total=sum(result['models']['full']['baseline'][0]['Qsite']))
    return result


def main():
    started=time.time();carrier_meta=load(ROOT/(O+'carrier1_body_basis.pkl'));joint_meta=load(ROOT/(O+'joint1_body_basis.pkl'));source_audit=original_source_audit(carrier_meta,joint_meta)
    carrier_ops={name:sparse.load_npz(ROOT/(O+'carrier1_op_'+name+'.npz')) for name in NAMES};joint_ops={name:sparse.load_npz(ROOT/(O+'joint1_op_'+name+'.npz')) for name in NAMES}
    result=dict(kind='FRESH_COMMON_SPECIES_NATIVE_COUPLING_AND_MECHANISM_REVIEW_NOT_ADDED_PAIR_VERTEX',source_audit=source_audit,carriers={},joint={})
    config=json.loads((ROOT/(O+'carrier1_input.json')).read_text())
    for number in (33,31):result['carriers'][str(number)]=carrier(carrier_meta,carrier_ops,number,config);save_result(result);gc.collect()
    result['joint']=joint(joint_meta,joint_ops,json.loads((ROOT/(O+'joint1_input.json')).read_text()));result['resource']=dict(elapsed_seconds=time.time()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    result['limits']='Whole quark/white sea N36 cross-response is not yet a source-connected ADDED B/L pair vertex. Matched N32/N35/N33/N36 normalizations/current responses in a common controlled representation and connected canonical-source cumulant are not supplied. Old N35 and new N33/N36 sector spans differ; do not invent normalized inclusion-exclusion. No ground gate, mass, binding/proton/lifetime, Maxwell/Coulomb/lightcone, freeCOM, or primitive field-only spin origin. Positive finite noncommuting geometry coupling is legitimate within this native quantum law, not automatically a defect or untruncated-law interaction proof.'
    save_result(result);print(json.dumps(dict(kind=result['kind'],resource=result['resource'])),flush=True)


if __name__=='__main__':main()
