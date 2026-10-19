"""New natural collective-Gaussian physical consumer, immutable law unchanged."""
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from scipy.sparse.linalg import expm_multiply
from rich_spatial_collective_20261015_model import (build,solve,packet,source_matrices,source_state,
    receiver_matrices,fermionic_leakage,install_operator,multiply_except,second_quantized,exterior_basis)
from rich_spatial_quantum_sea_20261015_calculate import lanczos,retarded


def real(x):return float(np.real(x))


def matter_action(sec,coefficients,state):
    result=np.zeros_like(state)
    for b,n in enumerate(sec['sizes']):
        start,stop=sec['offsets'][b:b+2];a=state[start:stop].reshape(n,-1);out=np.zeros_like(a)
        for f,c in zip(sec['matrices'][b],coefficients):out+=(f@a)*c
        result[start:stop]=out.ravel()
    return result


def neutral_one_body(model,matrices):
    raw=np.zeros_like(model['grams'][0]);basis=exterior_basis(6)
    for b,block in enumerate(model['blocks']):
        op=second_quantized(matrices[b],basis[1],basis[2]);v=block['v0']
        raw+=(v.conj().T@(op@v))*multiply_except(model['grams'],[b])
    x=model['neutral']['whiteners'][0]
    return x.conj().T@raw@x


def observe(model,state,receivers,vacuum):
    sec=model['positive'];nf=sum(sec['sizes']);a=state.reshape(nf,-1)
    probability=np.sum(abs(a)**2,axis=0);coordinates=model['field']['coordinates']
    density=np.array([real(np.vdot(a,n@a)) for n in receivers]);rho=a.conj().T@a
    currents=[];contacts=[]
    for d,dd in zip(model['field']['derivative'],model['field']['contact']):
        currents.append(-real(np.vdot(state,matter_action(sec,np.vstack([np.zeros(len(probability)),d]),state))))
        contacts.append(real(np.vdot(state,matter_action(sec,np.vstack([np.zeros(len(probability)),dd]),state))))
    return dict(norm=real(np.vdot(state,state)),total_particles=float(np.sum(density)),density=density.tolist(),
                charge_excess=(density-vacuum).tolist(),field_probability=probability.tolist(),
                phi_mean=float(probability@coordinates[:,0]),phi_variance=float(probability@(coordinates[:,0]**2)-(probability@coordinates[:,0])**2),
                field_means=(probability@coordinates[:,1:]).tolist(),field_purity=real(np.trace(rho@rho)),currents=currents,contacts=contacts)


def trajectory(model,initial,kicked,receivers,vacuum,config):
    sec=model['positive'];times=np.linspace(0,config['duration'],config['time_points'])
    states=[]
    for state in [initial,kicked]:
        states.append(expm_multiply(-1j*sec['h'],state,start=0,stop=times[-1],num=len(times),traceA=-1j*sec['trace']))
    baseline=[observe(model,s,receivers,vacuum) for s in states[0]]
    pulse=[observe(model,s,receivers,vacuum) for s in states[1]]
    return dict(times=times.tolist(),baseline=baseline,kicked=pulse,
        paired_density_difference_l2=[float(np.linalg.norm(np.array(a['density'])-b['density'])) for a,b in zip(pulse,baseline)],
        paired_field_distribution_difference_l1=[float(np.sum(abs(np.array(a['field_probability'])-b['field_probability']))) for a,b in zip(pulse,baseline)]),states


def main():
    start=time.time();config=json.loads(Path(sys.argv[1]).read_text());output=Path(sys.argv[2]);model=build(config)
    print('Collective basis',model['gaussian_count'],'Gram dimensions',model['neutral']['sizes'],model['positive']['sizes'],flush=True)
    for sec in [model['neutral'],model['positive'],model['negative']]:solve(sec,config['eigensolver_tolerance'])
    print('Energies',[model[k]['energy'] for k in ['neutral','positive','negative']],flush=True)
    residual=fermionic_leakage(model);print('Exact H2 residual',residual['absolute_H_projection_residual'],flush=True)
    source_packet=packet(model,config);plus=source_state(model,1,source_matrices(model,1,source_packet));minus=source_state(model,-1,source_matrices(model,-1,source_packet))
    neutral=model['neutral'];a0=neutral['ground'].reshape(neutral['sizes'][0],-1)
    occupation=neutral_one_body(model,[np.outer(f,f.conj()) for f in source_packet])
    rawminus=real(np.vdot(a0,occupation@a0));rawplus=1-rawminus;pn=real(np.vdot(plus,plus));mn=real(np.vdot(minus,minus))
    initial=plus/np.sqrt(pn);phase=np.exp(1j*config['kick_strength']*model['field']['coordinates'][:,1]);kicked=(initial.reshape(sum(model['positive']['sizes']),-1)*phase).ravel()
    vacuum=[]
    for site,(x,y,z) in enumerate(model['geo']['cells']):
        local=np.zeros((9,12,12),complex);local[:,4*z:4*z+4,4*z:4*z+4]=np.eye(4)/9
        op=neutral_one_body(model,local);vacuum.append(real(np.vdot(a0,op@a0)))
    vacuum=np.array(vacuum);receivers=receiver_matrices(model)
    motion,states=trajectory(model,initial,kicked,receivers,vacuum,config)
    delta=np.array(motion['baseline'][-1]['density'])-motion['baseline'][0]['density']
    print('Computing field-frozen control and same-H spectra',flush=True)
    sec=model['positive'];frozen_coeff=np.repeat(model['coefficients'][:,model['center'],None],len(model['field']['coordinates']),axis=1)
    install_operator(sec,model,frozen_coeff);frozen,_=trajectory(model,initial,kicked,receivers,vacuum,config);install_operator(sec,model)
    plus_spectrum=lanczos(sec['h'],plus,neutral['energy'],config['lanczos_steps'])
    minus_spectrum=lanczos(model['negative']['h'],minus,neutral['energy'],config['lanczos_steps'])
    spectra=[]
    for i,d in enumerate(model['field']['derivative']):
        vector=-matter_action(neutral,np.vstack([np.zeros(len(phase)),d]),neutral['ground']);vector-=neutral['ground']*np.vdot(neutral['ground'],vector)
        j=lanczos(neutral['h'],vector,neutral['energy'],config['lanczos_steps'])
        vector=(a0*model['field']['coordinates'][:,i+1]).ravel();vector-=neutral['ground']*np.vdot(neutral['ground'],vector)
        q=lanczos(neutral['h'],vector,neutral['energy'],config['lanczos_steps'])
        spectra.append(dict(current=j,field=q,current_retarded=retarded(j,config['frequencies'],config['eta']),field_retarded=retarded(q,config['frequencies'],config['eta'])))
    field_probability=np.sum(abs(a0)**2,axis=0);phi=model['field']['coordinates'][:,0]
    result=dict(kind='COLLECTIVE_ALL_PAIR_ORDERS_GAUSSIAN_VARIATIONAL_CANDIDATE_NOT_ALL_FOCK_CLOSURE',input=config,
        same_law='Pure geometry/fields imports from immutable rich_spatial_quantum_sea_20261015_model; ALL actual2 physical parameters and field grid unchanged.',
        blocks=dict(momentum_xy=model['momenta'],modes_per_block=12,occupied_per_block=6,total_modes=108,total_occupied=54,
                    density='Full cross-block density retained; no loss of x/y charge-packet coherence.'),
        collective=dict(gaussian_count=model['gaussian_count'],names=model['generator_names'],generator_norms=model['generator_norms'],
            construction='Global fixed diabatic Slaters exp(+/-step K_M/J) on actual 54-electron sea; each contains all excitation orders. K=Q h P-P h Q, normalized collectively. Same Gaussians at every field node.',
            overlap_method='Exact determinants/exterior algebra in 12-mode blocks, multiplied across nine blocks. No overlap inverses, no 0*inverse singularity. Gram whitening explicitly truncates only disclosed tiny Gram eigenvalues.'),
        gram={name:model[name]['gram_info'] for name in ['neutral','positive','negative']},
        dimensions={name:dict(fermion=sum(model[name]['sizes']),quantum=model[name]['h'].shape[0]) for name in ['neutral','positive','negative']},
        energies={name:model[name]['energy'] for name in ['neutral','positive','negative']},
        gaps=[model['positive']['energy']-neutral['energy'],model['negative']['energy']-neutral['energy']],
        eigensolver_residuals={name:model[name]['eigensolver_residual'] for name in ['neutral','positive','negative']},
        original_H_projection=residual,
        old_same_law_comparison=dict(old_prefix='rich_spatial_quantum_sea_20261015_actual2',old_absolute_H_residual=.8981188467808872,
            old_gaps=[.8140370269740185,.8140115802971764],new_residual_to_old_ratio=residual['absolute_H_projection_residual']/.8981188467808872,
            meaning='Material collective basis fork, no physical parameter adjustment. Neither old nor new variational gap is a rigorous bound.'),
        source=dict(family='ALL canonical creation/removal directions on EVERY neutral Gaussian, represented by complete occupied/complement six-dimensional local frames in each block; not selected energy branch sources.',
            raw_creation_norm2=rawplus,raw_removal_norm2=rawminus,raw_CAR_sum=rawplus+rawminus,retained_creation_norm2=pn,retained_removal_norm2=mn,
            omitted_creation_weight=rawplus-pn,omitted_removal_weight=rawminus-mn,
            normalization='Only initial charged source normalized. Subsequent diagonal field-only unitary has NO projection or renormalization.'),
        neutral=dict(phi_mean=float(field_probability@phi),phi_variance=float(field_probability@(phi**2)-(field_probability@phi)**2),
            field_purity=real(np.trace((a0.conj().T@a0)@(a0.conj().T@a0))),density=vacuum.tolist(),total_particles=float(sum(vacuum))),
        motion_and_reaction=motion,frozen_field_coupling=frozen,
        spatial_motion=dict(density_change_l2=float(np.linalg.norm(delta)),axis_marginal_change_l2=[float(np.linalg.norm(np.bincount(model['geo']['cells'][:,axis],weights=delta,minlength=3))) for axis in range(3)],
            t0_body_difference=motion['paired_density_difference_l2'][0],field_kick_norm_error=abs(float(np.linalg.norm(kicked))-1)),
        source_G=dict(addition=plus_spectrum,removal=minus_spectrum,retarded=retarded(plus_spectrum,config['frequencies'],config['eta'],True,minus_spectrum)),
        current_and_field_spectra=spectra,
        limitations=['Fixed collective Gaussian variational span is not complete full-CAR dynamics; exact H2 residual measures missing H action at the chosen field grid.',
            'Same severe field regulator, supplied scalar ordered potential, frozen matrix orientation/Clifford ansatz, primitive charge/CAR and zero nonuniform longitudinal interaction as old law.',
            'One nonzero common q with both polarizations/sin-cos partners, not localized field packet or bulk photon.',
            'Internal frame inputs do not prove physical Noether spin or Lorentz symmetry. No strong/composite/proton/QCD or Nature closure.',
            'Finite Lanczos quadratures include both signs and raw norms but are not claimed spectrally converged; finite Gram truncation explicitly reported.',
            'Frozen control uses same original-law current operators only as labelled diagnostics; actual frozen-H body has no field-dependent hopping.'],
        elapsed_seconds=time.time()-start,peak_resident_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    np.savez_compressed(output.with_suffix('.npz'),neutral_ground=neutral['ground'],charged_initial=initial,charged_kicked=kicked,
        baseline_states=states[0],kicked_states=states[1],coordinates=model['field']['coordinates'],vacuum_density=vacuum,
        neutral_whitener=neutral['whiteners'][0])
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(result=str(output),gaps=result['gaps'],residual=residual,source=result['source'],motion=result['spatial_motion'],elapsed=result['elapsed_seconds'],peak_resident_KiB=result['peak_resident_KiB'])),flush=True)


if __name__=='__main__':main()
