"""Actual ONE-Fock N36 Bdag Ldag Omega, bidirectional shared-field response.
No tensor of separately evolved particles; no stationary-source mass assumption.
"""
from pathlib import Path
import json,sys,gc
import numpy as np
from scipy import sparse
from rich_composite_geometry_20261015_dressing_science import spectrum,evolve,tensor_frobenius
from rich_common_native_species_20261016_runtime import execute
from rich_common_native_species_20261016_analysis import native,prepare,observe,currents,ward,make_generator,source_pair,source_profile,expectation_pair,count_probabilities
import rich_composite_geometry_20261015_model as model


def finite_ward(space,species,edge,theta,hops_on):
    d0,dphi=source_pair(space,species,edge,0,theta,hops_on);c0=space['c0']if hops_on else space['ops']['rotation']+space['ops']['electric']+space['ops']['plaquette'];cphi=space['cphi']if hops_on else space['cphi']*0;c0=c0+d0;cphi=cphi+dphi;result={}
    for measured in ('quark','white'):
        div0=[sparse.csr_matrix(c0.shape,dtype=complex)for _ in range(4)];divphi=[sparse.csr_matrix(cphi.shape,dtype=complex)for _ in range(4)]
        for bond,(start,end)in enumerate(model.EDGES):
            j0,jphi=source_pair(space,measured,bond,1,theta if measured==species and bond==edge else 0.,hops_on);div0[start]+=j0;div0[end]-=j0;divphi[start]+=jphi;divphi[end]-=jphi
        result[measured]=[]
        for site in range(4):
            q=sparse.diags(space['data']['Q'+measured][:,site]);result[measured].append(tensor_frobenius(1j*(c0@q-q@c0)+div0[site],1j*(cphi@q-q@cphi)+divphi[site],space['scalar']['grid']))
    return result


def field_values(event):return np.array([event[key]for key in ('Phi','Phi2','Phi_momentum','Phi_energy','rotation','loopB','loopC','electric')])


def event(state,t,space,species,edge,theta,hops_on):
    result={'t_absolute':float(t),**observe(state,space)};result['current']=currents(state,space,species,edge,theta,hops_on);first,second=source_pair(space,species,edge,2,theta,hops_on);result['source_contact']=expectation_pair(state,first,second,space);result['source_theta']=theta;return result


def trajectory(space,prepared,config,species,hops_on,out):
    theta=config['theta'];duration=config['pulse_duration'];edge=config[species+'_probe_edge'];during=config['during_pulse_times'];after=config['after_pulse_times'];delta0,deltaphi=source_pair(space,species,edge,0,theta,hops_on);full=make_generator(space,hops_on=hops_on);pulse=make_generator(space,delta0,deltaphi,hops_on);early=evolve(pulse,prepared,during);late=evolve(full,early[-1],after);times=list(during)+[duration+t for t in after];states=early+late
    events=[event(state,t,space,species,edge,theta if index<len(during)else 0.,hops_on)for index,(t,state)in enumerate(zip(times,states))];np.savez(out,**{f'state_{index:02d}':state for index,state in enumerate(states)},times=np.asarray(times));return events,{'source_deltaH_body_norm':float(np.linalg.norm(delta0.data)+np.linalg.norm(deltaphi.data)),'finite_source_Ward':finite_ward(space,species,edge,theta,hops_on),'source_order':'Pulse ONLY '+species+' microscopic hopping with exp(i q theta), then zero source. State continuous at0; atpulseend sourceoperator is leftlimit. Receivercurrent operator unchanged by othersector source; no instantaneous cross-sector transfer claim.'}


def baseline(space,prepared,config,hops_on,out):
    times=list(config['during_pulse_times'])+[config['pulse_duration']+t for t in config['after_pulse_times']];states=evolve(make_generator(space,hops_on=hops_on),prepared,times);initial=count_probabilities(prepared,space['gram'],space['data'],space['nf']);events=[]
    for t,state in zip(times,states):
        measured=event(state,t,space,'white',config['white_probe_edge'],0.,hops_on);counts=count_probabilities(state,space['gram'],space['data'],space['nf']);measured['material_count_PVM_TV']=sum(abs(counts.get(key,0)-initial.get(key,0))for key in counts.keys()|initial.keys())/2;events.append(measured)
    np.savez(out,**{f'state_{index:02d}':state for index,state in enumerate(states)},times=np.asarray(times));return events


def connected(events,reference,source,config):
    receiver='quark'if source=='white'else'white';receiver_edge=config[receiver+'_probe_edge'];rows=[]
    for measured,base in zip(events,reference):
        if measured['t_absolute']!=base['t_absolute']:raise ValueError('Causal baseline is NOT same age')
        delta_current=np.asarray(measured['current'][receiver])-np.asarray(base['current'][receiver]);delta_charge=np.asarray(measured['Q'+receiver+'_site_mean'])-np.asarray(base['Q'+receiver+'_site_mean']);delta_field=field_values(measured)-field_values(base)
        rows.append({'t_absolute':measured['t_absolute'],'receiver':receiver,'delta_receiver_current_all_edges':delta_current.tolist(),'delta_pinned_receiver_current':float(delta_current[receiver_edge]),'delta_receiver_charge_all_sites':delta_charge.tolist(),'delta_shared_fields':delta_field.tolist()})
    return {'rows':rows,'max_abs_receiver_current':float(np.max(np.abs([row['delta_receiver_current_all_edges']for row in rows]))),'max_abs_pinned_receiver_current':float(max(abs(row['delta_pinned_receiver_current'])for row in rows)),'max_abs_receiver_charge':float(np.max(np.abs([row['delta_receiver_charge_all_sites']for row in rows]))),'max_abs_shared_fields':np.max(np.abs([row['delta_shared_fields']for row in rows]),axis=0).tolist(),'field_order':['Phi','Phi2','Pphi','Hphi','Hrot','loopB','loopC','electric'],'instantaneous_cross_current':rows[0]['delta_receiver_current_all_edges'],'instantaneous_cross_charge':rows[0]['delta_receiver_charge_all_sites'],'meaning':'Response of ACTUAL jointly evolved N36state minus independently evolved SAME-AGE baseline. Retarded in time,NOT relativisticlightcone/Maxwell/Coulomb proof. Sharedfield AND finite source/body projection effects not falsely separated; no tensoringparticle answers.'}


def analyze(operators,labels,probes,scalar,p,reference,config,out):
    space=native(operators,labels,36,scalar);energy,vectors,residual=spectrum(space['c0'],space['cphi'],space['gram'],space['inverse'],scalar);baryon,lepton=config['joint_initial_pair'];seed=next(probe for probe in probes if probe['descriptor']['family']=='bare'and probe['descriptor']['B_site']==baryon and probe['descriptor']['L_site']==lepton);prepared,strength=prepare(seed,space)
    result={'CAR_N':36,'Q_total':0,'Noether_J':1,'Noether_M':1,'body_dimension':len(space['keep']),'full_independent_Phi_dimension':len(space['keep'])*space['nf'],'metric':space['metric'],'energies':energy.tolist(),'offset_to_HOT_reference':float(energy[0]-reference),'generalized_residuals':residual,'ground':observe(vectors[:,0],space),'canonical_joint_source_strength':seed['raw_strength'],'retained_joint_preparation_strength':strength,'sources':[source_profile(probe,space,energy,vectors,reference)for probe in probes],'Ward_zero_source_after_actual_G':ward(space),'protocol':config,'source_order':'Bdag0 Ldag2 Omega with L FIRST,epsilon/sqrt6 and whitecoefficient1,fullunmasked actualoccupiedsource. Bothhalfspinupcomponents give totalJ1. ONEFockCAR64 and ONEPhi/R/PW; NO product of separately evolved N35/N33 answers.','charge_semantics':'Qquark includes whole positive quarksea(background+4,afterB+5),Qwhite includes whole negativewhitesea(background-4,afterL-5); totalQ0. Connected DIFFERENCES compare sameinitialjointstate and sameage,not taggedparticles. QuarkCsite and netQsite DIFFER.'}
    np.savez(out+'_native_states.npz',energies=energy,vectors=vectors,prepared=prepared,keep=space['keep']);full_baseline=baseline(space,prepared,config,True,out+'_baseline_states.npz');result['full_baseline']=full_baseline;result['pulses']={};result['connected']={}
    for source in ('white','quark'):
        events,receipt=trajectory(space,prepared,config,source,True,out+'_'+source+'_pulse_states.npz');result['pulses'][source]={'events':events,**receipt};result['connected'][source]=connected(events,full_baseline,source,config);Path(out+'_science_checkpoint.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'stage':'ACTUAL_COMMON_JOINT_CROSS_RESPONSE','source':source,'current_response':result['connected'][source]['max_abs_receiver_current'],'charge_response':result['connected'][source]['max_abs_receiver_charge'],'initial_response':result['connected'][source]['instantaneous_cross_current']}),flush=True);gc.collect()
    off_baseline=baseline(space,prepared,config,False,out+'_ALLhopsOFF_baseline_states.npz');control={'definition':'SAMEjointnativeH/initialstate,REMOVEALLquarkANDwhite scalarANDvectorhops,retainHphi/Hrot/HE/plaquette. Every source/current/contact derived from actualcontrolledH,not suppliedzero/no-op fallback. MATERIALcountPVM fixed,not LABdensity if chargedRrotates.','baseline':off_baseline,'pulses':{},'connected':{}}
    # Own controlled-state evolution reaches the physical count-continuity assertion.
    control['actual_count_PVM_TV']=[entry['material_count_PVM_TV']for entry in off_baseline]
    for source in ('white','quark'):
        events,receipt=trajectory(space,prepared,config,source,False,out+'_ALLhopsOFF_'+source+'_states.npz');control['pulses'][source]={'events':events,**receipt};control['connected'][source]=connected(events,off_baseline,source,config)
    result['ALL_scalar_AND_vector_hops_OFF_control']=control;result['mechanism_scope']='Only actual commonPhi/R/PW/CAR interactions and declaredsource/body projection present. Scalarhops sharePhi,Hphi momentum generates feedback; vectorhops shareR with actualHrot and finiteRcommutators; quarkcolourlink responds indirectly,whitehasnoSU3charge. Observedcrossfieldchanges trace these commoncoordinates but doNOT uniquelyattributechannels or prove Maxwell/Coulomb. No neutralground permission theorem,stationarysourcepole,binding or lifetime inference.';return result


def main():
    config=json.loads(Path(sys.argv[1]).read_text())
    if config['mode']!='joint':raise ValueError('Joint mode required')
    execute(config,sys.argv[2],analyze)


if __name__=='__main__':main()
