"""Actual finite circular-polarization drive, full-sea spin POVM and two-time response."""
from typing import Any
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from scipy.linalg import block_diag
from scipy.sparse.linalg import expm_multiply
from rich_spatial_helicity_20261015_model import (build,field_structure,field_drive,rotation_structure,
    rotation_evidence,angular_class_probabilities,spin_receiver,physical_receivers,apply_terms,
    install_operator,packet,source_matrices,source_state,momentum_probabilities)
from rich_spatial_translation_20261015_calculate import field_action,complex_array
from rich_spatial_collective_20261015_calculate import neutral_one_body
ROOT=Path(__file__).resolve().parent


def digest(path: Any)->str:return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def real(x: Any)->float:return float(np.real(x))


def operator_vectors(model: Any,state: Any)->Any:
    sec=model['positive']
    return [-apply_terms(sec,d,state) for d in model['field']['derivatives']]+[field_action(model,sec,q,state) for q in model['field']['coordinates']]


def light_observation(model: Any,structure: Any,state: Any)->Any:
    sec=model['positive'];a=state.reshape(sum(sec['sizes']),model['field']['nphi'],model['field']['shell_dimension'])
    rho=a.reshape(sum(sec['sizes']),-1).conj().T@a.reshape(sum(sec['sizes']),-1)
    def emmean(op: Any)->float:return real(np.vdot(a,np.einsum('ij,mpj->mpi',op,a)))
    return dict(norm=real(np.vdot(state,state)),momentum=momentum_probabilities(model,sec,state),
        field_Q_means=[emmean(q) for q in structure['q']],field_P_means=[emmean(p) for p in structure['p']],
        field_polarization_angular_mean=emmean(structure['lpol']),field_purity=real(np.trace(rho@rho)),
        phi_mean=real(np.vdot(state,field_action(model,sec,model['field']['phi_operator'],state))),
        currents=[-real(np.vdot(state,apply_terms(sec,d,state))) for d in model['field']['derivatives']],
        contacts=[[real(np.vdot(state,apply_terms(sec,d,state))) for d in row] for row in model['field']['contacts']])


def two_time_response(model: Any,anchor_state: Any,anchor: float,config: Any)->Any:
    sec=model['positive'];lags=np.linspace(0,config['response_max_lag'],config['response_points'])
    start=np.column_stack([anchor_state]+operator_vectors(model,anchor_state))
    propagated=expm_multiply(-1j*sec['h'],start,start=0,stop=lags[-1],num=len(lags),traceA=-1j*sec['trace'])
    ordered=[];commutator=[]
    for states in propagated:
        left=np.column_stack(operator_vectors(model,states[:,0]));correlation=left.conj().T@states[:,1:]
        ordered.append(correlation);commutator.append(-1j*(correlation-correlation.conj()))
    ordered=np.array(ordered);commutator=np.array(commutator)
    channels=[];w=[]
    for circular in [-1,1]:
        for direction in [-1,1]:
            w.append(np.array([1,1j*direction,1j*circular,-circular*direction])/2)
            channels.append(dict(circular_profile=circular,travelling_profile=direction))
    basis=block_diag(np.array(w),np.array(w));circular_ordered=np.einsum('ai,tij,bj->tab',basis.conj(),ordered,basis)
    circular_commutator=np.einsum('ai,tij,bj->tab',basis.conj(),commutator,basis)
    frequency=[]
    for omega in config['response_frequencies']:
        frequency.append(np.trapezoid(np.exp(1j*omega*lags)[:,None,None]*circular_commutator,x=lags,axis=0))
    contacts=[[real(np.vdot(anchor_state,apply_terms(sec,d,anchor_state))) for d in row] for row in model['field']['contacts']]
    return dict(anchor_time=anchor,lags=lags.tolist(),real_profile_order=['Jxc','Jxs','Jyc','Jys','Qxc','Qxs','Qyc','Qys'],
        ordered_full_complex_correlation=complex_array(ordered),retarded_commutator_right_limit=complex_array(commutator),
        circular_profile_order=channels+channels,circular_ordered_complex_correlation=complex_array(circular_ordered),
        circular_retarded_cross_matrix=complex_array(circular_commutator),finite_window_frequencies=config['response_frequencies'],
        finite_window_circular_response=complex_array(np.array(frequency)),instantaneous_contact_matrix=contacts,
        meaning='Nonstationary actual charged packet: <Oi(tprime+tau) Oj(tprime)> and -i commutator, theta(0)=right limit. Full8x8 current/field/mixed matrices. Circular basis uses receiver adjoint W* C W^T. Finite0..lag window Fourier transform is NOT stationary Lehmann poles, identified photon frequency, particle helicity or Compton cross section.')


def main()->None:
    started=time.time();inputpath=Path(sys.argv[1]);config:Any=json.loads(inputpath.read_text());output=Path(sys.argv[2])
    lawpath=ROOT/config['law_input'];cachepath=ROOT/config['cache_states'];resultpath=ROOT/config['cache_result']
    law:Any=json.loads(lawpath.read_text());cached_result:Any=json.loads(resultpath.read_text());model:Any=build(law);cache=np.load(cachepath)
    neutral=model['neutral'];positive=model['positive'];negative=model['negative'];neutral['ground']=cache['neutral_ground']
    initial=cache['charged_initial'];neutral_energy=cached_result['energies']['neutral'];neutral['energy']=neutral_energy
    hcheck=float(np.linalg.norm(neutral['h']@neutral['ground']-neutral_energy*neutral['ground']))
    f=packet(model,law);plus=source_state(model,1,source_matrices(model,1,f));minus=source_state(model,-1,source_matrices(model,-1,f))
    pn=real(np.vdot(plus,plus));mn=real(np.vdot(minus,minus));fresh=plus/np.sqrt(pn);overlap=np.vdot(fresh,initial)
    source_check=float(np.linalg.norm(initial-fresh*overlap/abs(overlap)))
    a0=neutral['ground'].reshape(neutral['sizes'][0],-1);op=neutral_one_body(model,[np.outer(x,x.conj()) for x in f]);rawminus=real(np.vdot(a0,op@a0))
    structure=field_structure(model,law);initials={'baseline':initial};drive_info={}
    for chirality,name in [(1,'circular_plus'),(-1,'circular_minus')]:
        initials[name],drive_info[name]=field_drive(model,structure,initial,chirality,config['drive_Q_amplitude'],config['drive_P_amplitude'])
    times=np.linspace(0,config['duration'],config['time_points']);trajectories={};checkpoint_observations={}
    for frozen,label in [(False,'coupled'),(True,'frozen')]:
        install_operator(positive,model,frozen)
        for name,state in initials.items():
            key=label+'_'+name
            trajectories[key]=expm_multiply(-1j*positive['h'],state,start=0,stop=times[-1],num=len(times),traceA=-1j*positive['trace'])
            checkpoint_observations[key]=[light_observation(model,structure,s) for s in trajectories[key]]
            print('Snapshot arm',key,'complete',flush=True)
    install_operator(positive,model,False)
    checkpoint_npz=output.with_name(output.name.replace('_result.json','_checkpoint.npz'))
    checkpoint_json=output.with_name(output.name.replace('_result.json','_checkpoint.json'))
    bindings=dict(input_sha256=digest(inputpath),law_input_sha256=digest(lawpath),cache_states_sha256=digest(cachepath),cache_result_sha256=digest(resultpath),
        current_model_sha256=digest(ROOT/'rich_spatial_helicity_20261015_model.py'),current_calculate_sha256=digest(Path(__file__)))
    np.savez_compressed(checkpoint_npz,neutral_ground=neutral['ground'],**{k:v for k,v in trajectories.items()})
    checkpoint_json.write_text(json.dumps(dict(kind='USEFUL_ACTUAL_STATES_SAVED_BEFORE_HEAVY_RECEIVERS_AND_RESPONSE',bindings=bindings,
        actual_cached_neutral_H_state_residual=hcheck,charged_source_reconstruction_residual=source_check,times=times.tolist(),drives=drive_info,
        observations=checkpoint_observations,peak_resident_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),indent=2)+'\n')
    print('Bound state checkpoint complete; constructing physical rotations/spin POVM',flush=True)
    rotations=rotation_structure(model,structure)
    rotation_checks={name:rotation_evidence(model,rotations,name,state) for name,state in [('neutral',neutral['ground']),('positive',initial),('negative',minus/np.sqrt(mn))]}
    receiver=spin_receiver(model,config['spin_fourier_points'])
    neutral_receiver=physical_receivers(model,structure,receiver,neutral['ground'],True)
    arms={}
    for key,states in trajectories.items():
        observations=[]
        for light,state in zip(checkpoint_observations[key],states):
            observations.append(dict(**light,physical_receiver=physical_receivers(model,structure,receiver,state),
                total_discrete_angular_class=angular_class_probabilities(model,rotations,'positive',state)))
        arms[key]=observations
    print('Full occupied-sea spin/field angular receivers complete',flush=True)
    responses={}
    for arm in ['baseline','circular_plus']:
        responses[arm]=[]
        for anchor in config['response_anchors']:
            index=int(np.argmin(abs(times-anchor)))
            responses[arm].append(two_time_response(model,trajectories['coupled_'+arm][index],float(times[index]),config))
    comparisons={}
    for label in ['coupled','frozen']:
        base=arms[label+'_baseline'];comparisons[label]={}
        for name in ['circular_plus','circular_minus']:
            pulse=arms[label+'_'+name]
            comparisons[label][name]=dict(
                spin_mean_difference=[p['physical_receiver']['spin_mean']-b['physical_receiver']['spin_mean'] for p,b in zip(pulse,base)],
                spin_distribution_difference_l1=[float(np.sum(abs(np.array(p['physical_receiver']['spin_distribution'])-b['physical_receiver']['spin_distribution']))) for p,b in zip(pulse,base)],
                polarization_angular_difference=[p['field_polarization_angular_mean']-b['field_polarization_angular_mean'] for p,b in zip(pulse,base)],
                total_momentum_drift=[float(np.linalg.norm(np.array(p['momentum']['total'])-pulse[0]['momentum']['total'])) for p in pulse],
                total_discrete_angular_drift=[float(np.linalg.norm(np.array(p['total_discrete_angular_class']['probabilities'])-pulse[0]['total_discrete_angular_class']['probabilities'])) for p in pulse])
    result=dict(kind='ACTUAL_CIRCULAR_POLARIZATION_DRIVE_FULL_SEA_SPIN_POVM_RECOIL_NOT_IDENTIFIED_PHOTON_HELICITY',input=config,law_input=law,bindings=bindings,
        literal_H_cache_checks=dict(neutral_eigenstate_action_residual=hcheck,cached_charged_source_reconstruction=source_check,
            meaning='Rebuilt explicit pure-helper actual3 H, no old producer replay; loaded source/state checked on literal owning H. No cached spectra relabelled.'),
        source=dict(raw_creation_norm2=1-rawminus,raw_removal_norm2=rawminus,raw_CAR_sum=1.,retained_creation_norm2=pn,retained_removal_norm2=mn,
            omitted_creation_weight=1-rawminus-pn,omitted_removal_weight=rawminus-mn),
        field_structure=dict(polarization_translation_commutator=structure['commutator_norm'],angular_integer_eigenvector_residual=structure['angular_integer_variance'],
            momentum_integer_eigenvector_residual=structure['momentum_integer_variance'],
            channels=[dict(angular_z=c['angular_z'],field_momentum=c['momentum'],rank=c['vectors'].shape[1]) for c in structure['channels']],
            interpretation='Lpol is fixed+z POLARIZATION ANGULAR MOMENTUM/Stokes rotation, not automatic single-particle helicity. Opposite propagation changes helicity interpretation; mixed directions/matter dressing forbid sign(totalP)*Lpol identification.'),
        rotation=rotation_checks,physical_spin_POVM=dict(normalization_operator_defects=receiver['normalization_defects'],
            maximum_effect_hermiticity_defects=receiver['maximum_effect_hermiticity_defects'],semantics=receiver['meaning']),
        neutral_physical_receiver=neutral_receiver,drives=drive_info,times=times.tolist(),arms=arms,comparisons=comparisons,two_time_cross_response=responses,
        snapshot=dict(states=checkpoint_npz.name,metadata=checkpoint_json.name,stage='Before heavy physical spin receiver and two-time response'),
        limitations=['Literal actual3 H/regulator/source reused; inherited whole-operator Gram/translation defects~9e-6, shell2/oneq/three cells and original missing-H residual remain. No precision/rank/gap polishing or all-Fock closure.',
            'Canonical total Sz receiver acts on entire occupied sea; not an isolated added-particle spin measurement. Its physical joint ambient PVM compresses to a POVM if Gaussian span not spin invariant. All signed numerical artifacts retained without clipping.',
            'Cubic physical space+primitive frame rotation gives finite C4/C8 parity and angular classes modulo4, NOT continuous SO3/Noether spin theorem. No polar-unitary replacement/artificial H symmetrization.',
            'Q/P drives are true finite field-only unitaries, not full-space phase identities or pure one-photon preparations. No auxiliary basis frequency used as physical photon energy.',
            'Fixed-axis Lpol and circular profile channels are NOT automatically photon helicity. Full joint angular/momentum distributions retained.',
            'Nonstationary packet response is finite two-time/windowed frequency response of same retained H, not stationary Lehmann scattering poles or an on-shell Compton cross section.',
            'Primitive Clifford/frame/CAR/charge/supplied scalar potential/zero nonuniform longitudinal coupling and missing continuum/Lorentz/bulk/strong/proton/Nature closure remain.'],
        elapsed_seconds=time.time()-started,peak_resident_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    arrays=dict(neutral_ground=neutral['ground'],**{k:v for k,v in trajectories.items()})
    np.savez_compressed(output.with_suffix('.npz'),**arrays);output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(result=str(output),cache_checks=result['literal_H_cache_checks'],rotation=rotation_checks,comparisons=comparisons,
        peak_resident_KiB=result['peak_resident_KiB'],elapsed=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
