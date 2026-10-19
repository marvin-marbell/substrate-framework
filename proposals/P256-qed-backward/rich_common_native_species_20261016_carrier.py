"""Actual N33 electron-like and complementary N31 removal on SAME occupied law."""
from pathlib import Path
import json,sys
import numpy as np
from rich_composite_geometry_20261015_dressing_science import spectrum,evolve,metric_operator
from rich_common_native_species_20261016_runtime import execute
from rich_common_native_species_20261016_analysis import native,prepare,observe,currents,ward,make_generator,source_pair,source_profile,count_probabilities,expectation_pair


def analyze(operators,labels,probes,scalar,p,reference,config,out):
    result={'sectors':{},'source_order':'N33 cdag_downwhite_spinUP and N31 c_downwhite_spinUP atsite0,canonicalcoefficient1. Removal is ORIGINAL complementary order,not asserted Cimage/proton or a separately chosen Hilbert copy. Source0 fixed before outcomes; FULLunmasked source,NOcompactselection.'};raw={}
    for number in (33,31):
        space=native(operators,labels,number,scalar);energy,vectors,residual=spectrum(space['c0'],space['cphi'],space['gram'],space['inverse'],scalar);local_probes=[probe for probe in probes if probe['descriptor']['sector']==number]
        seed=next(probe for probe in local_probes if probe['descriptor']['family']=='bare'and probe['descriptor']['site']==config['carrier_source_site']);prepared,strength=prepare(seed,space);raw[number]=seed['raw_strength'];initial=count_probabilities(prepared,space['gram'],space['data'],space['nf']);full=make_generator(space);off=make_generator(space,hops_on=False);events={}
        for name,generator in (('full',full),('ALL_scalar_AND_vector_hops_OFF',off)):
            entries=[]
            for t,state in zip(p['times'],evolve(generator,prepared,p['times'])):
                counts=count_probabilities(state,space['gram'],space['data'],space['nf']);tv=sum(abs(counts.get(key,0)-initial.get(key,0))for key in counts.keys()|initial.keys())/2;entries.append({'t':t,**observe(state,space),'current':currents(state,space,hops_on=name=='full'),'count_PVM_TV':tv})
            events[name]=entries
        theta=config['theta'];duration=config['pulse_duration'];delta0,deltaphi=source_pair(space,'white',config['white_probe_edge'],0,theta);pulse=make_generator(space,delta0,deltaphi);pulsed=evolve(pulse,prepared,[duration])[0];baseline=evolve(full,prepared,[duration])[0]
        pulse_events=[{'t_after_pulse':t,**observe(state,space),'current':currents(state,space)}for t,state in zip(p['times'],evolve(full,pulsed,p['times']))];baseline_events=[{'t_after_same_age':t,**observe(state,space),'current':currents(state,space)}for t,state in zip(p['times'],evolve(full,baseline,p['times']))]
        sources=[source_profile(probe,space,energy,vectors,reference)for probe in local_probes]
        sector={'CAR_N':number,'Q_total':-1 if number==33 else 1,'Noether_J':.5,'Noether_M':.5 if number==33 else -.5,'body_dimension':len(space['keep']),'full_Phi_dimension':len(space['keep'])*space['nf'],'metric':space['metric'],'energies':energy.tolist(),'offset_to_HOT_reference':float(energy[0]-reference),'generalized_residuals':residual,'ground':observe(vectors[:,0],space),'sources':sources,'unmasked_preparation_strength':strength,'Ward_after_actual_G':ward(space),'events':events,'white_material_pulse':pulse_events,'same_age_no_pulse':baseline_events,
            'source_semantics':'Raworiginalcreation/removal strengths and source residues areNOTprobabilities; no stationarity/physicalpole/mass claim. OwnN31 calculation,not assumed C relation. FullPhi7/nativecoordinate dynamics,NOTlowpole propagation. Primitive spin/charges INPUT; finiteNoether Lleft+SCAR,not continuumtranslation/COM.'}
        sector['owning_white_source_contact']={'theta0_initial':expectation_pair(prepared,*source_pair(space,'white',config['white_probe_edge'],2),space),'theta_pulse_initial':expectation_pair(prepared,*source_pair(space,'white',config['white_probe_edge'],2,theta),space),'theta_pulse_end':expectation_pair(pulsed,*source_pair(space,'white',config['white_probe_edge'],2,theta),space),'law':'Actual H(theta) second derivative INCLUDINGexp(i theta q); not a dropped contact or a phase-free finite-theta current.'}
        result['sectors'][str(number)]=sector;np.savez(out+f'_N{number}_states.npz',energies=energy,vectors=vectors,prepared=prepared,keep=space['keep']);Path(out+'_science_checkpoint.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'stage':'ACTUAL_COMMON_CARRIER_STATES_AND_MOTION','N':number,'energy':float(energy[0]),'source_strength':strength,'final':events['full'][-1]}),flush=True)
    result['original_CAR_complement']={'addition_raw':raw[33],'removal_raw':raw[31],'sum':raw[33]+raw[31],'expected_occupied_Omega_norm':1.,'meaning':'SAME originalorbital {c,c†}=1 on owningOmega; rawstrengthsum is NOT sum of separately renormalized probabilities. ActualOmega norm recorded at outerresult; no C theorem assumed.'};return result


def main():
    config=json.loads(Path(sys.argv[1]).read_text())
    if config['mode']!='carrier':raise ValueError('Carrier mode required')
    execute(config,sys.argv[2],analyze)


if __name__=='__main__':main()
