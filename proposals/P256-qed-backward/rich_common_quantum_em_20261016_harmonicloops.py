"""Actual loop-rich quantum-state harmonic/ordered-circuit field-cut weights.
Analytic field-shift fusion retains EVERY intermediate hard-wall projector.
No new state/cap, no clamped Wilson norm promoted to quantum support.
"""
from pathlib import Path
import json,sys,time,resource
import numpy as np
from rich_common_quantum_em_20261016_operator import Model
from rich_common_quantum_em_20261016_harmonics import legs


def circuits(model,state):
    original=np.asarray(state).reshape(model.n,len(model.grid),len(model.flux));where={tuple(m):i for i,m in enumerate(model.flux)};result=[];paths={'triangle01_12_20':[(0,1),(1,1),(4,-1)],'square01_12_23_30':[(0,1),(1,1),(2,1),(3,-1)]}
    for category,root in((2,'down_quark_nminus1'),(3,'charged_white_nminus3')):
        for name,path in paths.items():
            x=original.copy();shift=np.zeros(2,int);valid=np.ones(len(model.flux),bool);prefixes=[]
            for edge,direction in path:
                part=next(value for value in model.parts if value['edge']==edge and value['category']==category);c0=part['c0']if direction==1 else part['c0'].conj().T;cphi=part['cphi']if direction==1 else part['cphi'].conj().T;flat=x.reshape(model.n,-1);x=(c0@flat).reshape(original.shape)+(cphi@flat).reshape(original.shape)*model.grid[None,:,None];shift+=direction*np.asarray(part['shift']);prefixes.append(shift.tolist());valid&=np.asarray([tuple(m+shift)in where for m in model.flux])
            final_valid=np.asarray([tuple(m+shift)in where for m in model.flux]);before=float(np.vdot(x,x).real);net=float(np.vdot(x[:,:,final_valid],x[:,:,final_valid]).real);actual=float(np.vdot(x[:,:,valid],x[:,:,valid]).real);result.append({'root':root,'ordered_NONBACKTRACKING_path':name,'ordered_edges_and_directions':path,'integer_field_prefix_shifts':prefixes,'net_integer_field_circulation':shift.tolist(),'source_fluxes_valid_for_every_intermediate_projection':int(valid.sum()),'source_fluxes_valid_for_net_shift_only':int(final_valid.sum()),'actual_body_word_output_norm2_before_field_projection':before,'net_shift_only_projected_output_norm2_NOT_actual_word':net,'ACTUAL_every_intermediate_field_projector_output_norm2':actual,'actual_retained_fraction':actual/before if before else None,'additional_early_projection_loss_UNCLAMPED':net-actual,'units':'energy^(2*path_length), raw operator strength NOT a particle probability','meaning':'Body/Phi operators are m-independent tensor factors. Compute their full ordered product on source m, then retain source indices whose EVERY cumulative integer shift stays in the declared85fluxes. This equals the actual projected quantum operator word, NOT direct Pexp(i netshift.a)P shortcut.'})
    return result


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=config['out'];started=time.time();model=Model(config['matter_prefix'])
    with np.load(config['state_file'])as data:state=data['states'][:,0].copy()
    primitive=legs(model,state);print(json.dumps({'stage':'loop_rich_actual_quantum_direct_primitive_harmonics','legs':len(primitive)}),flush=True);loops=circuits(model,state);result={'kind':'OWN_LOOP_RICH_QUANTUM_REFERENCE_ACTUAL_CHARGED_FIELD_SUPPORT','state_file':config['state_file'],'state_scope':'Fresh loop-rich neutral OWN quantum reference, NOT clamped matter state or first-law B/L state','state_norm':float(np.vdot(state,state).real),'direct_primitive_charged_harmonics':primitive,'actual_quantum_nonbacktracking_circuits':loops,'no_state_operator_field_cut_or_physics_constant_changed':True,'not_a_cut_scan_or_trajectory_rerun':True,'limitations':'Final fundamental body/Phi/R/PW cuts retained. Nonzero clamped circuit alone NOT proof quantum support; actual every-intermediate field-cut weights shown. Nonzero quantum loop support is NOT magnetic stiffness, photon spin1/dispersion/binding or complete pair closure.','resources':{'seconds':time.time()-started,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}};Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'stage':'actual_quantum_harmonic_and_loop_support_complete',**result['resources']}),flush=True)


if __name__=='__main__':main()
