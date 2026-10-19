"""Bounded SAME-span physical orthogonal correction and matched spin experiment."""
from typing import Any
import hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import eigsh,expm_multiply
from rich_spatial_orthogonal_20261015_model import (build,install_operator,operator_quality,source_matrices,
    neutral_occupation,rotation_structure,spin_receiver)
from rich_spatial_translation_20261015_model import packet,source_state
from rich_spatial_helicity_20261015_model import (field_structure,field_drive,rotation_evidence,
    angular_class_probabilities,physical_receivers)
from rich_spatial_helicity_20261015_calculate import light_observation,two_time_response
ROOT=Path(__file__).resolve().parent


def digest(p: Any)->str:return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main()->None:
    started=time.time();inp=Path(sys.argv[1]);config:Any=json.loads(inp.read_text());output=Path(sys.argv[2]);law:Any=json.loads((ROOT/config['law_input']).read_text())
    model:Any=build(config,law);quality={name:operator_quality(model,name) for name in ['neutral','positive','negative']}
    cache=np.load(ROOT/config['cache_states']);old:Any=json.loads((ROOT/config['previous_experiment']).read_text())
    neutral=model['neutral'];positive=model['positive'];negative=model['negative'];nb=model['field']['dimension']
    mapped=(neutral['old_to_new'][0]@cache['neutral_ground'].reshape(27,nb)).ravel()
    energy,vector=eigsh(neutral['h'],k=1,which='SA',tol=law['eigensolver_tolerance'],maxiter=5000,v0=mapped)
    neutral['ground']=vector[:,0];neutral['energy']=float(energy[0]);overlap=np.vdot(neutral['ground'],mapped)
    cached_difference=float(np.linalg.norm(neutral['ground']-mapped*overlap.conjugate()/abs(overlap)))
    f=packet(model,law);sp=source_matrices(model,1,f);sm=source_matrices(model,-1,f)
    plus=source_state(model,1,sp);minus=source_state(model,-1,sm);pn=float(np.vdot(plus,plus).real);mn=float(np.vdot(minus,minus).real)
    a0=neutral['ground'].reshape(27,nb);occ=neutral_occupation(model,f);rawminus=float(np.vdot(a0,occ@a0).real)
    initial=plus/np.sqrt(pn);mapped_source=np.concatenate([(m@cache['charged_initial'].reshape(9,162,nb)[b]).ravel() for b,m in enumerate(positive['old_to_new'])])
    phase=np.vdot(initial,mapped_source);source_difference=float(np.linalg.norm(initial-mapped_source*phase.conjugate()/abs(phase)))
    car=sum(s.conj().T@s for s in sp)+sum(s.conj().T@s for s in sm)
    structure=field_structure(model,law);initials={'baseline':initial};drive_info={}
    for sigma,name in [(1,'circular_plus'),(-1,'circular_minus')]:initials[name],drive_info[name]=field_drive(model,structure,initial,sigma,config['drive_Q_amplitude'],config['drive_P_amplitude'])
    source=dict(raw_creation_norm2=1-rawminus,raw_removal_norm2=rawminus,retained_creation_norm2=pn,retained_removal_norm2=mn,
        omitted_creation_weight=1-rawminus-pn,omitted_removal_weight=rawminus-mn,packet_CAR_operator_defect=float(np.linalg.norm(car-np.eye(27))))
    maps=dict(mapped_old_neutral_physical_norm=float(np.linalg.norm(mapped)),new_neutral_vs_mapped_old_distance=cached_difference,
        mapped_old_charged_source_physical_norm=float(np.linalg.norm(mapped_source)),new_canonical_source_vs_mapped_old_distance=source_difference,
        meaning='Old-to-new map Rphysical Xold measured, NOT polar-unitarized. Fresh owning-H neutral state and canonical sources computed, not old source renormalized to preserve a signal.')
    reconstruction_npz=output.with_name(output.name.replace('_result.json','_reconstruction.npz'));reconstruction_json=reconstruction_npz.with_suffix('.json')
    arrays=dict(neutral_ground=neutral['ground'],charged_initial=initial,negative_source=minus,
        field_terms=np.array([k.toarray() for k in model['field']['coupling']]),field_boson=model['field']['boson'].toarray())
    for name in ['neutral','positive','negative']:
        arrays[name+'_projected_components']=np.array(model[name]['matrices'])
        for b,t in enumerate(model[name]['trains']):arrays[f'{name}_physical_CP_R_{b}']=t['R'];arrays[f'{name}_old_to_new_{b}']=model[name]['old_to_new'][b]
    np.savez_compressed(reconstruction_npz,**arrays)
    reconstruction_json.write_text(json.dumps(dict(kind='OWNING_H_AND_NEW_PHYSICAL_STATE_SNAPSHOT_BEFORE_TRAJECTORIES_HEAVY_RECEIVERS',
        source_sha256={p.name:digest(p) for p in [Path(__file__),ROOT/'rich_spatial_orthogonal_20261015_model.py',ROOT/'rich_spatial_orthogonal_20261015_tensor.py']},
        orthogonality=model['orthogonality'],span_evidence=model['span_evidence'],operator_quality=quality,source=source,cache_basis_maps=maps,
        neutral_energy=neutral['energy'],neutral_eigenresidual=float(np.linalg.norm(neutral['h']@neutral['ground']-neutral['energy']*neutral['ground']))),indent=2)+'\n')
    print('Physical H/state reconstruction snapshot complete',quality,flush=True)
    times=np.linspace(0,config['duration'],config['time_points']);trajectories={};light={}
    for frozen,label in [(False,'coupled'),(True,'frozen')]:
        install_operator(positive,model,frozen)
        for name,state in initials.items():
            key=label+'_'+name;trajectories[key]=expm_multiply(-1j*positive['h'],state,start=0,stop=times[-1],num=len(times),traceA=-1j*positive['trace'])
            light[key]=[light_observation(model,structure,s) for s in trajectories[key]]
            print('Orthogonal physical arm',key,'complete',flush=True)
    install_operator(positive,model,False)
    checkpoint_npz=output.with_name(output.name.replace('_result.json','_checkpoint.npz'));checkpoint_json=checkpoint_npz.with_suffix('.json')
    np.savez_compressed(checkpoint_npz,neutral_ground=neutral['ground'],**trajectories)
    checkpoint_json.write_text(json.dumps(dict(kind='USEFUL_MATCHED_STATES_BEFORE_SPIN_POVM_CROSS_RESPONSE',times=times.tolist(),observations=light,
        operator_quality=quality,source=source,basis_maps=maps,input_sha256=digest(inp),reconstructed_H_snapshot=reconstruction_npz.name),indent=2)+'\n')
    rotations=rotation_structure(model,structure)
    rotation_checks={name:rotation_evidence(model,rotations,name,state) for name,state in [('neutral',neutral['ground']),('positive',initial),('negative',minus/np.sqrt(mn))]}
    receiver=spin_receiver(model,config['spin_fourier_points']);neutral_receiver=physical_receivers(model,structure,receiver,neutral['ground'],True)
    arms={}
    for key,states in trajectories.items():arms[key]=[dict(**o,physical_receiver=physical_receivers(model,structure,receiver,state),total_discrete_angular_class=angular_class_probabilities(model,rotations,'positive',state)) for o,state in zip(light[key],states)]
    comparisons={};old_comparison={}
    for label in ['coupled','frozen']:
        comparisons[label]={};base=arms[label+'_baseline']
        for name in ['circular_plus','circular_minus']:
            pulse=arms[label+'_'+name];l1=[float(np.sum(abs(np.array(p['physical_receiver']['spin_distribution'])-b['physical_receiver']['spin_distribution']))) for p,b in zip(pulse,base)]
            comparisons[label][name]=dict(spin_mean_difference=[p['physical_receiver']['spin_mean']-b['physical_receiver']['spin_mean'] for p,b in zip(pulse,base)],
                spin_distribution_L1_difference=l1,spin_distribution_total_variation=[v/2 for v in l1],
                total_momentum_L2_drift=[float(np.linalg.norm(np.array(p['momentum']['total'])-pulse[0]['momentum']['total'])) for p in pulse],
                total_discrete_class_L2_drift=[float(np.linalg.norm(np.array(p['total_discrete_angular_class']['probabilities'])-pulse[0]['total_discrete_angular_class']['probabilities'])) for p in pulse])
            old_arms=old['arms'][label+'_'+name]
            old_comparison[label+'_'+name]=dict(new_minus_old_full_spin_mean=[p['physical_receiver']['spin_mean']-o['physical_receiver']['spin_mean'] for p,o in zip(pulse,old_arms)],
                new_vs_old_spin_distribution_L1=[float(np.sum(abs(np.array(p['physical_receiver']['spin_distribution'])-o['physical_receiver']['spin_distribution']))) for p,o in zip(pulse,old_arms)])
    responses={}
    for name in ['baseline','circular_plus']:
        responses[name]=[]
        for anchor in config['response_anchors']:
            index=int(np.argmin(abs(times-anchor)));responses[name].append(two_time_response(model,trajectories['coupled_'+name][index],float(times[index]),config))
    model_info={}
    for site,local in enumerate(model['locals']):
        model_info[str(site)]={str(n):dict(rank=data['frame'].shape[1],discarded_singular_norm=data['discarded_norm'],reconstruction_norm=data['reconstruction_norm'],isometry_defect=data['isometry_defect']) for n,data in local.items()}
    result=dict(kind='SAME_PHYSICAL_SPAN_ORTHOGONAL_OWNING_H_POLARIZATION_SPIN_RESPONSE_NOT_POSTHOC_H_REPAIR',input=config,law_input=law,
        cause_probe='rich_spatial_orthogonal_20261015_probe1_result.json: small raw assembly skew amplified by legacy Gram sandwich, not a large conjugation bug. Reported old whole-H failure accepted without reconfirmation.',
        physical_basis=dict(construction='Actual local-Fock column SVD isometries then CP-to-TT sequential physical SVD, charged block last with even6-electron block reorder signs. All27/162 true state directions retained, no Gram eigensandwich used for H/current.',
            local_isometries=model_info,whole_physical_orthogonality=model['orthogonality'],span_and_basis_maps=model['span_evidence'],cache_maps=maps,
            projection='Owning local operators projected by direct physical TT transfer. Sources, translation, rotation and ambient spin POVM via physical TT-bra/acted-CP contractions and a SINGLE physical state-coordinate right inverse, never Cdag Hraw C.'),
        operator_quality=quality,actual_adjoint='rmatvec/rmatmat explicitly use Fdag,Kdag,Bdag; not assumed equal to matvec. No H+Hdag averaging or polar-unitary replacements.',
        neutral_energy=neutral['energy'],neutral_H_state_residual=float(np.linalg.norm(neutral['h']@neutral['ground']-neutral['energy']*neutral['ground'])),source=source,
        drives=drive_info,rotation=rotation_checks,physical_spin_POVM=dict(normalization_operator_defects=receiver['normalization_defects'],maximum_effect_hermiticity_defects=receiver['maximum_effect_hermiticity_defects'],meaning=receiver['meaning']),
        neutral_receiver=neutral_receiver,times=times.tolist(),arms=arms,comparisons=comparisons,new_vs_old_same_protocol=old_comparison,two_time_cross_response=responses,
        metrics='Spin-distribution difference is L1; total variation is HALF L1. Mean-Sz difference is neither. Momentum/density Euclidean norms are L2, never called TV.',
        snapshots=dict(reconstruction=reconstruction_npz.name,checkpoint=checkpoint_npz.name),
        limitations=['Faithful finite numerical Hermitian projection of SAME span is not all-Fock closure: omitted action.723842 and severe shell2/oneq/three-cell regulator remain.',
            'Full-sea canonical spin receiver is not isolated electron spin; Lpol is fixed+z polarization angular momentum, NOT automatic particle helicity. Both old pulses lowered Sz; do not fit a spin flip.',
            'Primitive frame/CAR/charge, supplied scalar potential, frozen Phi orientation, absent nonuniform longitudinal interaction and no continuum/Lorentz/bulk/strong/proton/Nature claims.',
            'Nonstationary finite two-time/windowed response, not on-shell Compton or stationary spectral convergence. Q/P preparation has legitimate initial coordinate-shift current plus finite-cut remainder as previously calibrated.',
            'Remaining measured roundoff in one-sided CP receiver/source coordinate maps recorded separately from direct-H/physical-basis quality; no clipping or normalization repair.'],
        elapsed_seconds=time.time()-started,peak_resident_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    np.savez_compressed(output.with_suffix('.npz'),neutral_ground=neutral['ground'],charged_initial=initial,**trajectories)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(result=str(output),quality=quality,comparisons=comparisons,cache_maps=maps,elapsed=result['elapsed_seconds'],peak_resident_KiB=result['peak_resident_KiB'])),flush=True)


if __name__=='__main__':main()
