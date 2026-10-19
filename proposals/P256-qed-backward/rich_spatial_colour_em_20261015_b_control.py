"""Proper B transverse-off causal comparison at an already-positive endpoint.
FULL t1 endpoints are frozen author data; OFF evolves from the SAME t0 raw
source and SAME field-only preparation. Nothing switches OFF only after t1.
"""
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from rich_review_wave19_20261015_all_m import AllM
from rich_spatial_colour_em_20261015_snapshot import load,source,charge_receiver
from rich_spatial_colour_em_20261015_model import assemble,observable
from rich_spatial_colour_em_20261015_measure import joint_receiver
from rich_spatial_colour_em_20261015_react import mean,scalar_measure


def main():
    inp,out=map(Path,sys.argv[1:3]);request=json.loads(inp.read_text());root=inp.parent;started=time.time()
    paths={key:root/request[key] for key in ['state_json','state_npz','event_json','event_npz']};hashes={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()}
    metadata,data,model,laws=load(paths['state_json'],paths['state_npz']);event=json.loads(paths['event_json'].read_text());saved=np.load(paths['event_npz']);base=model['base'];field=model['field']
    expected={paths[key].name:hashes[paths[key].name] for key in ['state_json','state_npz']}
    if event['input_sha256']!=expected:raise ValueError('Positive endpoint is not from this exact own coupled H/reference checkpoint')
    name=request['name'];key=request['source_key'];label=name+'_'+key;positive=event['results'][label];target=float(request['time']);index=positive['times'].index(target)
    initial=saved[label+'_initial'];kicked=saved[label+'_kicked'];positive_baseline=saved[label+'_baseline'][index];positive_reaction=saved[label+'_reaction'][index]
    insertion=source(data,model,name,key);op=base['operators'][name];operators={key:observable(model,name,matter=op[key]) for key in ['Q','rho0','rho1','C','B','N','PW_boundary']}
    def material(state):
        row={key:mean(state,value) for key,value in operators.items()};row.update(scalar_measure(state,model,name));row['source_removal_strength']=float(np.linalg.norm(insertion.H@state)**2)
        return row
    frozen=assemble(model,name,coupled=False);moving=AllM(model,name,frozen);phase_energy=positive['baseline'][0]['whole_H_energy'];stored={};result={}
    def checkpoint(stage):
        payload=dict(kind='OWN_PROPER_B_TRANSVERSE_OFF_CAUSAL_CONTROL_FROM_SAME_T0',stage=stage,request=request,input_sha256=hashes,source_event_stage=event['stage'],source_event_natural_completion=False,source=positive['source'],time=target,all_M_control=moving.evidence,results=result,elapsed_seconds=time.time()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,protocol='FULL endpoints are the frozen own positive t1 data. OFF baseline and OFF pulse both start at original t0, with the SAME raw source and SAME field-only Q impulse; evolve under the proper off H from0 to1. No off-at-t1 switch, reoptimized reference, positive replay or changed cut/coupling.',numerics=dict(method='Standard SciPy adaptive expm_multiply through complete ALL-M factors; no fixed-K truncation',independent_temporal_error_estimate=None,meaning='No separately stored Schrodinger-residual/refinement estimate. H-action/norm/energy consistency are not such a certificate. Its absence is not evidence of an incorrect trajectory or a new research-permission gate. No sibling photon failure is rerun.'))
        out.write_text(json.dumps(payload,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),**stored)
    checkpoint('constructed_proper_off_H')
    off_baseline=moving.evolve(initial,target,phase_energy);stored['off_baseline']=off_baseline;checkpoint('off_baseline_from_t0_complete')
    off_reaction=moving.evolve(kicked,target,phase_energy);stored['off_reaction']=off_reaction
    full_base=positive['baseline'][index];full_react=positive['reaction'][index];off_base=material(off_baseline);off_react=material(off_reaction)
    result['full_material_excess']={key:full_react[key]-full_base[key] for key in off_base};result['off_material_excess']={key:off_react[key]-off_base[key] for key in off_base}
    result['full_physical_profile_current_excess']=[full_react['current'+str(mu)]-full_base['current'+str(mu)] for mu in range(6)]
    result['zero_off_profile_probe_derivative']=[0.]*6
    labels,effects=charge_receiver(data,model,name);full_joint_base=np.asarray(positive['joint_baseline'][index]);full_joint_react=np.asarray(positive['joint_reaction'][index]);off_joint_base=joint_receiver(off_baseline,effects,base['scalar'],base['rotor'],field);off_joint_react=joint_receiver(off_reaction,effects,base['scalar'],base['rotor'],field)
    result['receiver']=dict(charge_labels=labels,full_joint_baseline=full_joint_base.tolist(),full_joint_reaction=full_joint_react.tolist(),off_joint_baseline=off_joint_base.tolist(),off_joint_reaction=off_joint_react.tolist(),full_inclusive_TV=float(np.sum(abs(full_joint_react-full_joint_base))/2),off_inclusive_TV=float(np.sum(abs(off_joint_react-off_joint_base))/2),full_charge_phi_marginal_TV=float(np.sum(abs(full_joint_react.sum(axis=2)-full_joint_base.sum(axis=2)))/2),off_charge_phi_marginal_TV=float(np.sum(abs(off_joint_react.sum(axis=2)-off_joint_base.sum(axis=2)))/2),max_normalization_error=float(max(abs(np.sum(x)-1) for x in [full_joint_base,full_joint_react,off_joint_base,off_joint_react])),meaning='Inclusive underlying commuting charge/phi/ONE continuous-field-sign effects, no reaction-norm or postselection. An OFF inclusive TV can remain nonzero solely because the field changes; matter marginal and actual matter observables test backreaction.')
    result['norm_errors']=dict(full_baseline=float(abs(np.linalg.norm(positive_baseline)-1)),full_reaction=float(abs(np.linalg.norm(positive_reaction)-1)),off_baseline=float(abs(np.linalg.norm(off_baseline)-1)),off_reaction=float(abs(np.linalg.norm(off_reaction)-1)))
    result['off_H_definition']='Remove A and its probes in BOTH Peierls hopping AND covariant rotor. Keep Coulomb/body/bare-rotor/Pi^2. Exact zero probe derivative defines this decoupled CONTROL; it is not a gauge-faithful claim that charged material has no density current.'
    result['scope']='Actual positive composite source/current/scalar/colour feedback versus its matched causal control. No late t2/t4 or anti-B trajectory is implied; C source/operator evidence and independent19 lepton/anti data stay separately attributed. No photon/gauge/COM/cross-section/full-H-range/ordering admission.'
    checkpoint('complete')
    print(json.dumps(dict(time=target,full_material_excess=result['full_material_excess'],off_material_excess=result['off_material_excess'],receiver={key:value for key,value in result['receiver'].items() if key.endswith('TV')},peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)),flush=True)


if __name__=='__main__':main()
