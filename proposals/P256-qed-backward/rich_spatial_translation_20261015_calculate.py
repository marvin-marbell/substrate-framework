"""NEW finite-lattice matter/field momentum exchange on one projected law."""
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import expm_multiply
from rich_spatial_translation_20261015_model import (build,solve,packet,source_matrices,source_state,
    receiver_matrices,dense_body_operator,neutral_one_body,apply_terms,install_operator,translate,momentum_probabilities,symmetry_data)
from rich_spatial_quantum_sea_20261015_calculate import lanczos,retarded


def real(x):return float(np.real(x))


def field_action(model,sec,operator,state):
    return (operator@state.reshape(sum(sec['sizes']),model['field']['dimension']).T).T.ravel()


def total_project(model,sec,state,r):
    return sum(np.exp(2j*np.pi*r*l/3)*translate(model,sec,state,l,l) for l in range(3))/3


def complex_array(array):return np.stack([np.real(array),np.imag(array)],axis=-1).tolist()


def spectral_matrix(model,vectors,config):
    sec=model['neutral'];h=sec['h'];energy=sec['energy'];z=np.array(config['frequencies'])+1j*config['eta'];responses=[]
    total_positive=np.zeros((len(z),4,4),complex);total_negative=np.zeros_like(total_positive)
    for momentum in range(3):
        v=np.column_stack([total_project(model,sec,x,momentum) for x in vectors]);positive=np.zeros_like(total_positive);negative=np.zeros_like(total_negative);columns=[]
        for j in range(4):
            norm=float(np.linalg.norm(v[:,j]));basis=[];alpha=[];beta=[]
            if norm<1e-12:
                columns.append(dict(norm2=norm**2,energies=[],residues=[]));continue
            q=v[:,j]/norm;previous=np.zeros_like(q);oldbeta=0.
            for iteration in range(config['lanczos_steps']):
                basis.append(q);r=h@q-energy*q-oldbeta*previous;a=real(np.vdot(q,r));r-=a*q
                for old in basis:r-=old*np.vdot(old,r)
                newbeta=float(np.linalg.norm(r));alpha.append(a)
                if newbeta<1e-11 or iteration==config['lanczos_steps']-1:break
                beta.append(newbeta);previous,q=q,r/newbeta;oldbeta=newbeta
            e,u=eigh_tridiagonal(np.array(alpha),np.array(beta));cross=v.conj().T@np.column_stack(basis)
            residues=(cross@u)*(norm*u[0])[None,:]
            for i in range(4):
                positive[:,i,j]=np.sum(residues[i][None,:]/(z[:,None]-e[None,:]),axis=1)
                negative[:,i,j]=np.sum(residues[i][None,:]/(z[:,None]+e[None,:]),axis=1)
            columns.append(dict(norm2=norm**2,energies=e.tolist(),residues_i_by_energy=complex_array(residues),last_beta=newbeta))
        response=positive-np.transpose(negative,(0,2,1));total_positive+=positive;total_negative+=negative
        responses.append(dict(total_C3_momentum=momentum,raw_Gram=complex_array(v.conj().T@v),positive_branch_columns=columns,retarded_matrix=complex_array(response)))
    return dict(order=['xcos','xsin','ycos','ysin'],momentum_resolved=responses,
        total_retarded_matrix=complex_array(total_positive-np.transpose(total_negative,(0,2,1))),
        raw_total_Gram=complex_array(np.column_stack(vectors).conj().T@np.column_stack(vectors)),
        meaning='Same FULL coupled variational H, signed commutator branches and cross polarizations; finite reorthogonalized Krylov quadratures, not independent Dyson self-energy or convergence proof.')


def observe(model,state,receivers,spin,vacuum):
    sec=model['positive'];a=state.reshape(sum(sec['sizes']),model['field']['dimension']);density=np.array([real(np.vdot(a,n@a)) for n in receivers])
    phi=real(np.vdot(state,field_action(model,sec,model['field']['phi_operator'],state)))
    phi2=real(np.vdot(state,field_action(model,sec,model['field']['phi2_operator'],state)))
    currents=[-real(np.vdot(state,apply_terms(sec,d,state))) for d in model['field']['derivatives']]
    contacts=[[real(np.vdot(state,apply_terms(sec,d,state))) for d in row] for row in model['field']['contacts']]
    rho=a.conj().T@a
    return dict(norm=real(np.vdot(state,state)),density=density.tolist(),charge_excess=(density-vacuum).tolist(),total_particles=float(np.sum(density)),
        primitive_frame_spin_z=real(np.vdot(a,spin@a)),phi_mean=phi,phi_variance=phi2-phi*phi,
        field_coordinate_means=[real(np.vdot(state,field_action(model,sec,q,state))) for q in model['field']['coordinates']],
        field_purity=real(np.trace(rho@rho)),currents=currents,contact_matrix=contacts,
        momentum=momentum_probabilities(model,sec,state))


def trajectory(model,initial,kicked,receivers,spin,vacuum,config):
    sec=model['positive'];times=np.linspace(0,config['duration'],config['time_points']);states=[]
    for state in [initial,kicked]:states.append(expm_multiply(-1j*sec['h'],state,start=0,stop=times[-1],num=len(times),traceA=-1j*sec['trace']))
    observations=[[observe(model,s,receivers,spin,vacuum) for s in trajectory] for trajectory in states]
    baseline,pulse=observations
    return dict(times=times.tolist(),baseline=baseline,kicked=pulse,
        paired_density_difference_l2=[float(np.linalg.norm(np.array(a['density'])-b['density'])) for a,b in zip(pulse,baseline)],
        total_momentum_drift_baseline=[float(np.linalg.norm(np.array(a['momentum']['total'])-baseline[0]['momentum']['total'])) for a in baseline],
        total_momentum_drift_kicked=[float(np.linalg.norm(np.array(a['momentum']['total'])-pulse[0]['momentum']['total'])) for a in pulse]),states


def main():
    started=time.time();config=json.loads(Path(sys.argv[1]).read_text());output=Path(sys.argv[2]);model=build(config)
    print('Translation orbit/shell dimensions',model['gaussian_count'],model['field']['shell_dimension'],{name:model[name]['sizes'] for name in ['neutral','positive','negative']},flush=True)
    for name in ['neutral','positive','negative']:solve(model[name],config['eigensolver_tolerance'])
    print('Same-law projected energies',{name:model[name]['energy'] for name in ['neutral','positive','negative']},flush=True)
    neutral=model['neutral'];sec=model['positive'];nb=model['field']['dimension'];a0=neutral['ground'].reshape(neutral['sizes'][0],nb)
    f=packet(model,config);plus=source_state(model,1,source_matrices(model,1,f));minus=source_state(model,-1,source_matrices(model,-1,f))
    number=neutral_one_body(model,[np.outer(x,x.conj()) for x in f]);rawminus=real(np.vdot(a0,number@a0));pn=real(np.vdot(plus,plus));mn=real(np.vdot(minus,minus))
    initial=plus/np.sqrt(pn);kicked=field_action(model,sec,model['field']['q_drive'],initial)
    vacuum=[]
    for x,y,z in model['geo']['cells']:
        local=np.zeros((9,12,12),complex);local[:,4*z:4*z+4,4*z:4*z+4]=np.eye(4)/9
        number=neutral_one_body(model,local);vacuum.append(real(np.vdot(a0,number@a0)))
    vacuum=np.array(vacuum);receivers=receiver_matrices(model)
    local_spin=np.zeros((9,9,12,12),complex)
    for b in range(9):local_spin[b,b]=np.kron(np.eye(3),np.kron(np.eye(2),np.diag([.5,-.5])))
    spin=dense_body_operator(model,local_spin)
    symmetry={name:symmetry_data(model,model[name],model[name]['ground']) for name in ['neutral','positive','negative']}
    symmetry['charged_source']=symmetry_data(model,sec,initial)
    coupled,states=trajectory(model,initial,kicked,receivers,spin,vacuum,config)
    print('Coupled recoil complete',coupled['paired_density_difference_l2'][-1],flush=True)
    install_operator(sec,model,True);frozen,_=trajectory(model,initial,kicked,receivers,spin,vacuum,config);install_operator(sec,model)
    plus_spectrum=lanczos(sec['h'],plus,neutral['energy'],config['lanczos_steps']);minus_spectrum=lanczos(model['negative']['h'],minus,neutral['energy'],config['lanczos_steps'])
    currents=[];fields=[]
    for d,q in zip(model['field']['derivatives'],model['field']['coordinates']):
        v=-apply_terms(neutral,d,neutral['ground']);v-=neutral['ground']*np.vdot(neutral['ground'],v);currents.append(v)
        v=field_action(model,neutral,q,neutral['ground']);v-=neutral['ground']*np.vdot(neutral['ground'],v);fields.append(v)
    response_current=spectral_matrix(model,currents,config);response_field=spectral_matrix(model,fields,config)
    delta=np.array(coupled['baseline'][-1]['density'])-coupled['baseline'][0]['density']
    final=coupled['kicked'][-1]['momentum'];start=coupled['kicked'][0]['momentum']
    result=dict(kind='JOINT_TRANSLATION_PRESERVING_FINITE_PROJECTION_WITH_ACTUAL_MATTER_FIELD_RECOIL',input=config,
        physical_law='Same untruncated spatial/collective actual2 law and physical parameters. Complete-shell transverse projection and translation-orbit Gaussian span are a DIFFERENT finite regulator, not literally old finite H.',
        field_basis=dict(cutoff=config['shell_cutoff'],width=config['shell_width'],states=model['field']['shell_states'],dimension=model['field']['shell_dimension'],
            meaning='Complete four-coordinate total oscillator shells are BASIS ONLY. Physical bare energy is exact before-cut P^2/2I, including offdiagonal n<->n+2 and highest-shell raising contribution; NO harmonic Q^2 magnetic restoring potential.',
            peierls='Exact projected exp(i c.Q) displacement matrix elements of untruncated law; not exp(i P c.Q P).',
            drive='Exactly unitary finite field-only drive exp(i kappa P Q_xcos P), interpreted as projected Q-only drive Hamiltonian. NOT equal to P exp(i kappa Q_xcos) P. No projection loss or conditional normalization. Compressed current/Q commutation need not hold at this severe cutoff.'),
        matter_orbit=dict(raw_gaussians=model['gaussian_count'],names=model['orbit_names'],closure='All three actual physical z translations of all eleven fixed global Gaussian Slaters. Exact phase-equivalent duplicates retained until actual Gram null-space removal; not discarded by energy labels.',
            transverse_translation='Original x/y Fourier blocks remain exact. Neutral total transverse translations are1; N55 block(kx,ky) has eigenvalues exp(-ikx),exp(-iky), N53 conjugate. Offdiagonal charged-block density retained for x/y motion.'),
        gram={name:model[name]['gram_info'] for name in ['neutral','positive','negative']},
        dimensions={name:dict(fermion=sum(model[name]['sizes']),quantum=model[name]['h'].shape[0]) for name in ['neutral','positive','negative']},
        energies={name:model[name]['energy'] for name in ['neutral','positive','negative']},gaps=[sec['energy']-neutral['energy'],model['negative']['energy']-neutral['energy']],
        eigensolver_residuals={name:model[name]['eigensolver_residual'] for name in ['neutral','positive','negative']},
        symmetry=symmetry,neutral_momentum=momentum_probabilities(model,neutral,neutral['ground']),
        source=dict(raw_creation_norm2=1-rawminus,raw_removal_norm2=rawminus,raw_CAR_sum=1.,retained_creation_norm2=pn,retained_removal_norm2=mn,
            omitted_creation_weight=1-rawminus-pn,omitted_removal_weight=rawminus-mn,
            meaning='All canonical creation/removal directions on ALL translated neutral Gaussians. Actual N54/55/53 occupancy; source Gram-cut loss explicit.'),
        coupled=coupled,frozen_interaction=frozen,
        recoil=dict(kicked_matter_momentum_change_l2=float(np.linalg.norm(np.array(final['matter'])-start['matter'])),
            kicked_field_momentum_change_l2=float(np.linalg.norm(np.array(final['field'])-start['field'])),
            kicked_joint_distribution_change_l2=float(np.linalg.norm(np.array(final['joint_matter_field'])-start['joint_matter_field'])),
            total_momentum_distribution_change_l2=float(np.linalg.norm(np.array(final['total'])-start['total'])),
            interpretation='Matter and field exchange physical lattice momentum modulo2pi; their commuting C3 joint PVM resolves actual recoil, not oscillator occupation or source-overlap labels.'),
        spatial_motion=dict(density_change_l2=float(np.linalg.norm(delta)),axis_marginal_changes=[float(np.linalg.norm(np.bincount(model['geo']['cells'][:,a],weights=delta,minlength=3))) for a in range(3)],
            t0_body_difference=coupled['paired_density_difference_l2'][0],field_drive_norm_error=abs(float(np.linalg.norm(kicked))-1)),
        source_G=dict(addition=plus_spectrum,removal=minus_spectrum,retarded=retarded(plus_spectrum,config['frequencies'],config['eta'],True,minus_spectrum)),
        full_current_response=response_current,full_transverse_field_response=response_field,
        limitations=['New shell cutoff is a severe finite projection, not convergence or old finite-H equality. No imposed photon dispersion or fitted restoring coefficient.',
            'Original collective residual.723842 was large. Translation completion and exact symmetry/source closure do not establish accurate all-Fock time evolution or rigorous gap bounds; no new neutral residual-polish claim.',
            'Projected Q-only drive is a true field unitary with untouched initial matter density/spin, not a full-space phase-kick identity at finite shell cutoff. Current/contact t0 changes are disclosed, not suppressed.',
            'Complete shell rotation gives exact momentum PVMs; compressed Q coordinates can fail to commute and are not assigned a false simultaneous sharp-coordinate PVM.',
            'Primitive Clifford/frame spin/statistics/charge, supplied scalar potential, zero nonuniform longitudinal coupling and finite nonlocalized q remain. No continuous Noether/Lorentz/freeCOM/bulk-photon/strong/proton/Nature claim.',
            'Full coupled response is not an independent bare polarization/self-energy. Finite Krylov quadratures and both signs/raw matrix weights retained without spectral convergence claim.'],
        elapsed_seconds=time.time()-started,peak_resident_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    arrays=dict(neutral_ground=neutral['ground'],charged_initial=initial,charged_kicked=kicked,baseline_states=states[0],kicked_states=states[1],
        shell_states=np.array(model['field']['shell_states']),phi_nodes=model['field']['phi'],vacuum_density=vacuum,field_translation=model['field']['em_translation'])
    for name in ['neutral','positive','negative']:
        for b,(x,t) in enumerate(zip(model[name]['whiteners'],model[name]['matter_translation'])):
            arrays[f'{name}_whitener_{b}']=x;arrays[f'{name}_translation_{b}']=t
    np.savez_compressed(output.with_suffix('.npz'),**arrays);output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(result=str(output),symmetry=symmetry,recoil=result['recoil'],motion=result['spatial_motion'],peak_resident_KiB=result['peak_resident_KiB'],elapsed=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
