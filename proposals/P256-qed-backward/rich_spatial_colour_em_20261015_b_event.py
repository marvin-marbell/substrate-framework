"""OWN remaining B/anti-B field reaction, exact ALL-M factors of the SAME H.
Pure operator optimization is reused; NO reviewer reference/state or new cut enters.
The earlier complete-coordinate driver and optional range expansion are interrupted,
not retrospectively advertised as clean natural executions.
"""
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from rich_review_wave19_20261015_all_m import AllM
from rich_spatial_colour_em_20261015_snapshot import load,source,charge_receiver,matrix
from rich_spatial_colour_em_20261015_model import assemble,angular,observable,MatterMap
from rich_spatial_colour_em_20261015_measure import drive,joint_receiver
from rich_spatial_colour_em_20261015_react import mean,scalar_measure,measurement_operators,co_moving_continuity


def main():
    inp,out=map(Path,sys.argv[1:3]);request=json.loads(inp.read_text());root=inp.parent;started=time.time()
    state_json=root/request['state_json'];state_npz=root/request['state_npz'];metadata,data,model,laws=load(state_json,state_npz)
    base=model['base'];field=model['field'];p=model['parameters'];reference=data['neutral_reference'];nf=len(base['scalar']['grid']);nr=base['rotor']['dimension'];ne=field['dimension']
    hashes={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in [state_json,state_npz]};results={};stored={};times=np.asarray(request['times'],float)
    def checkpoint(stage):
        payload=dict(kind='OWN_ACTUAL_SAME_H_ALL_M_COMPOSITE_FIELD_REACTION_NOT_PHOTONS',stage=stage,request=request,input_sha256=hashes,reference_energy=metadata['reference_energy'],results=results,elapsed_seconds=time.time()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Same original native charge/phi7/R10/fieldN1 H, coefficients and own newly solved reference. AllM is ONLY a complete operator coordinate factorization; every retained M is covered and full-H action is measured. No reviewer state, Frame kinetic, smaller regulator, source-pole filter or physical-current patch. Raw-source normalization is preparation, never reaction probability.')
        out.write_text(json.dumps(payload,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),**stored)
    for case in request['cases']:
        name=case['name'];key=case['key'];label=name+'_'+key;h=laws[name];op=base['operators'][name]
        model['parameters']=dict(p,field_kick=request['field_kick']*case['drive_sign'],drive_profile=request['drive_profile'])
        insertion=source(data,model,name,key);raw=insertion@reference;strength=float(np.linalg.norm(raw)**2);initial=raw/np.sqrt(strength)
        metric=matrix(data,name+'_source_'+key+'_full_metric');full=mean(reference,observable(model,'neutral',matter=metric));energy=mean(initial,h)
        kicked,drive_evidence=drive(model,name,initial);drive_evidence['applied_field_kick']=model['parameters']['field_kick']
        moving=AllM(model,name,h);currents=[assemble(model,name,derivative=(mu,)) for mu in range(6)];operators=measurement_operators(model,name,currents);jops=[angular(model,name,axis) for axis in range(3)]
        labels,effects=charge_receiver(data,model,name);routes=[assemble(model,name,route_derivative=edge) for edge in range(2)]
        result=dict(source=dict(full_raw_strength=full,retained_raw_strength=strength,signed_source_complement=full-strength,raw_preparation_cost=energy-metadata['reference_energy']),drive=drive_evidence,all_M_representation=moving.evidence,charge_labels=labels,times=[],baseline=[],reaction=[],joint_baseline=[],joint_reaction=[],co_moving_initial=co_moving_continuity(h,operators['rho0'],routes,initial))
        results[label]=result;stored[label+'_initial']=initial;stored[label+'_kicked']=kicked;base_states=[];pulse_states=[];baseline=initial.copy();pulse=kicked.copy();previous=0.
        for t in times:
            baseline=moving.evolve(baseline,float(t-previous),energy);pulse=moving.evolve(pulse,float(t-previous),energy);previous=t;base_states.append(baseline);pulse_states.append(pulse)
            values=[]
            for state in [baseline,pulse]:
                row={key:mean(state,value) for key,value in operators.items()};row.update(scalar_measure(state,model,name));row['source_removal_strength']=float(np.linalg.norm(insertion.H@state)**2);row['whole_H_energy']=mean(state,h);row['J2']=float(sum(np.linalg.norm(j@state)**2 for j in jops))
                for axis in range(3):row['J_total'+str(axis)]=row['frame_S'+str(axis)]+row['L_body'+str(axis)]+row['F_field'+str(axis)]
                values.append(row)
            result['times'].append(float(t));result['baseline'].append(values[0]);result['reaction'].append(values[1]);result['joint_baseline'].append(joint_receiver(baseline,effects,base['scalar'],base['rotor'],field).tolist());result['joint_reaction'].append(joint_receiver(pulse,effects,base['scalar'],base['rotor'],field).tolist())
            stored[label+'_baseline']=np.asarray(base_states);stored[label+'_reaction']=np.asarray(pulse_states);checkpoint('event_'+label+'_t'+str(float(t)))
        result['excess']={key:[a[key]-b[key] for a,b in zip(result['reaction'],result['baseline'])] for key in result['baseline'][0]}
        result['joint_TV']=[float(np.sum(abs(np.asarray(a)-np.asarray(b)))/2) for a,b in zip(result['joint_reaction'],result['joint_baseline'])]
        result['joint_normalization_error']=float(max(abs(np.sum(x)-1) for x in result['joint_reaction']+result['joint_baseline']))
        result['norm_error']=float(max(abs(np.linalg.norm(x)-1) for x in base_states+pulse_states))
        result['baseline_H_energy_drift']=float(max(abs(row['whole_H_energy']-result['baseline'][0]['whole_H_energy']) for row in result['baseline']))
        result['reaction_H_energy_drift']=float(max(abs(row['whole_H_energy']-result['reaction'][0]['whole_H_energy']) for row in result['reaction']))
        checkpoint('event_complete_'+label)
        frozen=assemble(model,name,coupled=False);frozen_moving=AllM(model,name,frozen);fbase=frozen_moving.evolve(initial,float(times[-1]),energy);fpulse=frozen_moving.evolve(kicked,float(times[-1]),energy)
        material={key:mean(fpulse,value)-mean(fbase,value) for key,value in operators.items() if key in ['Q','rho0','rho1','C','B','Coulomb','frame_S2']};a=scalar_measure(fpulse,model,name);b=scalar_measure(fbase,model,name);material.update({key:a[key]-b[key] for key in a});material['source_removal_strength']=float(np.linalg.norm(insertion.H@fpulse)**2-np.linalg.norm(insertion.H@fbase)**2)
        result['fully_transverse_off_control']=dict(all_M_representation=frozen_moving.evidence,time=float(times[-1]),material_excess=material,zero_control_profile_probe_derivative=[0.]*6,meaning='Remove A/probe dependence in BOTH hopping and covariant rotor. Coulomb/body/bare rotor/Pi² remain. Same initial state and field-only kick, actual paired time evolution; no reoptimized reference. Zero derivative defines this decoupled CONTROL, not a gauge-faithful claim that moving charged matter has no density current.')
        stored[label+'_coupling_off_baseline']=fbase;stored[label+'_coupling_off_reaction']=fpulse
        if name.startswith('anti'):
            positive=name[4:];partner=positive+'_'+key;c=MatterMap(matrix(data,positive+'_C_matter'),nf,nr,ne,field['C']);overlap=np.vdot(initial,c@stored[partner+'_initial']);pos_current=[assemble(model,positive,derivative=(mu,)) for mu in range(6)];indices=[next(i for i,q in enumerate(results[partner]['charge_labels']) if abs(q+anti_q)<1e-12) for anti_q in labels];expected=np.asarray(results[partner]['joint_reaction'])[:,indices,:,::-1]
            result['actual_C_event']=dict(source_phase_real=float(overlap.real),source_phase_imag=float(overlap.imag),initial_C_error=float(np.linalg.norm(overlap*initial-c@stored[partner+'_initial'])),baseline_C_errors=[float(np.linalg.norm(overlap*a-c@b)) for a,b in zip(base_states,stored[partner+'_baseline'])],reaction_C_errors=[float(np.linalg.norm(overlap*a-c@b)) for a,b in zip(pulse_states,stored[partner+'_reaction'])],physical_current_C_odd_errors=[float(np.linalg.norm(overlap*(a@pulse_states[-1])+c@(b@stored[partner+'_reaction'][-1]))) for a,b in zip(currents,pos_current)],receiver_C_TV=[float(np.sum(abs(a-b))/2) for a,b in zip(result['joint_reaction'],expected)],meaning='Independent CAR-removal preparation and actual anti-H propagation with opposite C-conjugate Q impulse. Measured source phase; EM charge and field-sign labels reverse, phi stays even. No prepared positive state is substituted for the anti initial state.')
        checkpoint('complete_'+label)
        print(json.dumps(dict(case=label,max_material={key:max(abs(x) for x in result['excess'][key]) for key in ['rho0','phi2','C','B','source_removal_strength']},all_M=moving.evidence,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)),flush=True)
    checkpoint('complete')


if __name__=='__main__':main()
