"""New observable on saved states: actual direct charged-hop field-cut weights.
No trajectory replay, m-cap scan, operator replacement or clamped-loop inference.
"""
from pathlib import Path
import json,sys,time,resource,gc
import numpy as np
from rich_common_quantum_em_20261016_operator import Model


def legs(model,state):
    x=np.asarray(state).reshape(model.n,len(model.grid),len(model.flux));flat=x.reshape(model.n,-1);where={tuple(m):i for i,m in enumerate(model.flux)};result=[]
    for edge,category,name in((0,0,'up_quark_n2'),(0,2,'down_quark_nminus1'),(2,3,'charged_white_nminus3')):
        part=next(value for value in model.parts if value['edge']==edge and value['category']==category)
        for direction in(1,-1):
            c0=part['c0']if direction==1 else part['c0'].conj().T;cphi=part['cphi']if direction==1 else part['cphi'].conj().T;y=(c0@flat).reshape(x.shape);y+=(cphi@flat).reshape(x.shape)*model.grid[None,:,None];shift=direction*np.asarray(part['shift']);valid=np.asarray([i for i,m in enumerate(model.flux)if tuple(m+shift)in where],int);before=float(np.vdot(y,y).real);kept=float(np.vdot(y[:,:,valid],y[:,:,valid]).real)
            result.append({'root':name,'edge':edge,'direction':direction,'integer_charge':part['integer_charge'],'physical_charge':part['physical_charge'],'direct_integer_field_shift':shift.tolist(),'structural_shift_rank_in_85fluxes':len(valid),'actual_body_hop_output_norm2_before_field_cut':before,'actual_direct_field_leg_output_norm2_retained':kept,'actual_field_projection_lost_norm2_UNCLAMPED':before-kept,'retained_fraction_of_virtual_leg':kept/before if before else None,'units':'energy² on owning normalized native state; raw strengths NOT particle probabilities','scope':'Final Gram/Noether/body-selected hop, scalar+vector interference retained. Before-field norm embeds its output shift on unrestricted integer flux only to MEASURE the existing cut; no state evolution or physical cut is changed.'})
    return result


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=config['out'];started=time.time();models={};records=[]
    for entry in config['states']:
        prefix=entry['matter_prefix']
        if prefix not in models:models[prefix]=Model(prefix)
        model=models[prefix]
        with np.load(entry['file'])as data:
            source=data[entry['array']];state=source if source.ndim==1 else source[:,entry['column']];record={'state_label':entry['label'],'state_file':entry['file'],'array':entry['array'],'column':entry.get('column'),'matter_prefix':prefix,'state_norm':float(np.vdot(state,state).real),'direct_harmonics':legs(model,state)};records.append(record)
        print(json.dumps({'stage':'actual_selected_state_direct_charge_harmonics','state':entry['label'],'charged_legs':len(record['direct_harmonics'])}),flush=True);gc.collect()
    direct={}
    if config.get('dynamics_result'):
        data=json.loads(Path(config['dynamics_result']).read_text())
        for sector in('neutral','pair'):
            events=data['sectors'][sector]['full_live_quantum_links'];base=events['baseline'][0];pulse=events['white_pulse'][0];j=pulse['physical_current']['white']-base['physical_current']['white'];force=np.asarray(pulse['actual_transverse_generator_force'])-np.asarray(base['actual_transverse_generator_force']);profile=np.asarray(models[config['states'][0]['matter_prefix']].profile[2]);expected=3*profile*j;direct[sector]={'same_initial_state':True,'actual_full_H_white_source_activation_norm':events['external_white_source_activation_state_norm'],'actual_delta_white_current_t0':j,'actual_delta_integer_field_force_t0':force.tolist(),'direct_white_current_expected_force':expected.tolist(),'residual':float(np.linalg.norm(force-expected)),'meaning':'At t0 state is identical and external phase changes ONLY white bond2; this instantaneous force difference is a DIRECT charged-white quantum-field leg, not later indirect quark/Phi response. Later total field response may contain indirect channels.'}
    result={'kind':'SAVED_STATE_DIRECT_QUANTUM_CHARGE_HARMONIC_SUPPORT_NOT_CAP_SCAN','selected_states':records,'direct_white_initial_force_from_completed_data':direct,'primitive_leg_choices_INPUT':[{'edge':0,'root':'up_quark_n2'},{'edge':0,'root':'down_quark_nminus1'},{'edge':2,'root':'charged_white_nminus3'}],'no_trajectory_or_finished_case_rerun':True,'no_input_state_operator_field_cut_or_constant_changed':True,'not_before_body_projection':'Body/CAR/Phi/R/PW cut remains explicit; this measures direct FIELD harmonic loss, not full primitive/all-pair closure. A nonzero clamped Wilson loop alone was NOT used as quantum-support proof.','resources':{'seconds':time.time()-started,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}};Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'stage':'selected_state_direct_harmonic_support_complete',**result['resources']}),flush=True)


if __name__=='__main__':main()
