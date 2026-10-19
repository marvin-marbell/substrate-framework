"""Actual coordinate-fork total energy/current/recoil/source and same-state controls."""
import json
from pathlib import Path
import platform
import resource
import sys
import time
import numpy as np
import scipy

ROOT=Path(__file__).resolve().parent
PREFIX='rich_photon_collective_20261015_'

def group_residues(stem,data,mode_groups,repair_stem):
    from scipy import sparse
    with np.load(ROOT/(stem+'_primitive.npz')) as archive:
        coordinates=archive['coordinates'];factors=archive['factors'];profiles=archive['profiles']
    ground=np.load(ROOT/(stem+'_ground.npz'))['state'];n=data['geometry']['matter'];nf=data['geometry']['field'];nb=5*nf;gx=ground.reshape(n,nb);matrices=[sparse.load_npz(ROOT/(stem+f'_Hcomponent{k}.npz')) for k in range(len(profiles))]
    with np.load(ROOT/(stem+'complete_vectors.npz')) as archive:energies=archive['energies']-data['energy'];vectors=archive['vectors']
    qsource=np.empty((len(ground),8),complex);jsource=np.zeros_like(qsource);flatf=factors.reshape(len(profiles),nb);contact=np.zeros((8,8))
    for axis in range(8):
        q=(ground.reshape(n,5,nf)*coordinates[axis]).ravel();q-=np.vdot(ground,q)*ground;qsource[:,axis]=q
    for k in range(2,len(profiles),2):
        if not np.any(profiles[k]):continue
        cosine=matrices[k]@gx;sine=matrices[k+1]@gx;force=(cosine*flatf[k+1]-sine*flatf[k]).ravel()
        for axis in np.flatnonzero(profiles[k]):jsource[:,axis]+=profiles[k,axis]*force
        coupling_energy=float(np.vdot(gx,cosine*flatf[k]+sine*flatf[k+1]).real);contact-=coupling_energy*np.outer(profiles[k],profiles[k])
    for axis in range(8):jsource[:,axis]-=np.vdot(ground,jsource[:,axis])*ground
    transform=np.zeros((8,8),complex)
    for axis in range(0,8,2):transform[axis:axis+2,axis:axis+2]=np.array([[1.,1.],[1j,-1j]])/np.sqrt(2)
    amplitudes=dict(coordinate=vectors.conj().T@qsource,current=vectors.conj().T@jsource);gram=dict(coordinate=qsource.conj().T@qsource,current=jsource.conj().T@jsource);residues=[]
    def matrix(value):return dict(real=value.real.tolist(),imag=value.imag.tolist())
    captured={label:np.zeros((8,8),complex) for label in amplitudes}
    for group in mode_groups:
        indices=np.array(group['vector_indices'],dtype=int);item=dict(energy=group['energy'],multiplicity=len(indices),complete_below_upper_edge=group['complete_below_upper_edge'],max_residual=group['max_residual'],completeness_meaning=group['completeness_meaning'])
        for label,amplitude in amplitudes.items():
            residue=amplitude[indices].conj().T@amplitude[indices];captured[label]+=residue;item[label+'_real_axes_residue']=matrix(residue);item[label+'_signed_q_residue']=matrix(transform.conj().T@residue@transform)
        cross=amplitudes['coordinate'][indices].conj().T@amplitudes['current'][indices];item['coordinate_current_cross_residue']=matrix(cross);item['coordinate_current_signed_q_cross_residue']=matrix(transform.conj().T@cross@transform)
        force_defect=amplitudes['current'][indices]+(data['geometry']['volume']/2)*energies[indices,None]**2*amplitudes['coordinate'][indices]
        item['mode_J_plus_Iomega2Q_axis_norm']=np.sqrt(np.sum(abs(force_defect)**2,axis=0)).tolist()
        item['mode_force_meaning']='For canonical Qdot=P/I,Pdot=literalJ and an exact vacuum, J_n=-I omega_n^2 Q_n. This measured lowgroup mismatch exposes finiteh/domain/CI force defects; it is NOT patched or called trajectoryerror/currentrestoration proof.'
        residues.append(item)
    initial_pulse=np.load(ROOT/(stem+'_preparation.npz'))['pulse'];initial_amplitude=vectors.conj().T@initial_pulse;pulse_groups=[]
    for sample,t in enumerate(data['numerical_request']['times']):
        state=initial_pulse if sample==0 else np.load(ROOT/(repair_stem+f'_fine_state{sample}.npz'))['state'];amplitude=vectors.conj().T@state;expected=initial_amplitude*np.exp(-1j*(energies+data['energy'])*t);measurements=[]
        for group in mode_groups:
            indices=np.array(group['vector_indices'],dtype=int);measurements.append(dict(energy=group['energy'],multiplicity=len(indices),raw_unconditioned_pulse_weight=float(np.sum(abs(amplitude[indices])**2)),raw_excess_energy_in_group=float(np.dot(energies[indices],abs(amplitude[indices])**2)),complex_mode_phase_prediction_defect=float(np.linalg.norm(amplitude[indices]-expected[indices])),complete=group['complete_below_upper_edge']))
        pulse_groups.append(dict(time=t,groups=measurements,raw_total_captured_state_weight=float(np.sum(abs(amplitude)**2)),raw_uncaptured_state_weight=float(np.vdot(state,state).real-np.sum(abs(amplitude)**2)),meaning='Actual unconditioned prepared/evolved state contracted on own H eigengroup projectors. NO projected/filtered state was prepared or propagated; weights not renormalized to discard vacuum or missing tails.'))
    return dict(groups=residues,unconditioned_pulse_group_measurements=pulse_groups,full_real_contact_matrix=contact.tolist(),signed_q_receiver_contact_matrix=matrix(transform.conj().T@contact@transform),full_real_axes_source_Grams={label:matrix(value) for label,value in gram.items()},raw_uncaptured_real_axes_Grams={label:matrix(gram[label]-captured[label]) for label in gram},signed_q_order=['band1_x_qplus','band1_x_qminus','band1_y_qplus','band1_y_qminus','band2_x_qplus','band2_x_qminus','band2_y_qplus','band2_y_qminus'],convention='Sources s+=(Qc+iQs)Omega/sqrt2,s-=adjoint-source counterpart. W_ab=<s_a|P_group|s_b>; signed-q residues retain offdiagonal mixing from finite coordinate-domain C5 loss. Contact=<partial_a partial_b H> in energy/Q^2; Qsource Gram inQ^2,current Gram in(energy/Q)^2. Full retained source tails remain raw,notnormalizedaway.')

def run():
    start=time.monotonic();mode=sys.argv[1];stem=PREFIX+mode;data=json.loads((ROOT/(stem+'_data.json')).read_text());base=PREFIX+data['base_mode'];vacuum=data['ground'];records=data['records'];coupled=records['fine'];offpulse=records['same_state_off_pulse'];offbase=records['same_state_off_baseline'];keys=('onsite','Phi_matter','z_bond','transverse_Peierls','global_Phi','electric','reference','total');profiles=[];metrics=[]
    completion=json.loads((ROOT/(base+'grouprecover_data.json')).read_text());mode_groups=completion['groups']
    source_stem=PREFIX+data['source_evolution_mode']
    if not all(item['passed'] for item in completion['source_closure']):raise RuntimeError('Missing-mode completion has not closed physical source weights')
    baseline_matter=np.array(vacuum['matter_C5_PVM']);baseline_density=np.array(vacuum['density']);baseline_current=np.array(vacuum['transverse_current'])
    for point,a,b in zip(coupled,offpulse,offbase):
        delta={key:[point['profile'][z][key]-vacuum['profile'][z][key] for z in range(5)] for key in keys};offdelta={key:[a['profile'][z][key]-b['profile'][z][key] for z in range(5)] for key in keys};total=np.array(delta['total']);fourier=np.fft.fft(total);matter=np.array(point['matter_C5_PVM'])-baseline_matter;offmatter=np.array(a['matter_C5_PVM'])-np.array(b['matter_C5_PVM']);meanE=np.array(point['meanE'])-np.array(vacuum['meanE']);offE=np.array(a['meanE'])-np.array(b['meanE']);work=point['energy']-vacuum['energy']
        profiles.append(dict(time=point['time'],signed_total_excitation_energy=delta,same_state_complete_EMoff_pulse_minus_baseline=offdelta,total_preparation_work=work,profile_sum_minus_work=float(total.sum()-work),energy_peak_layer=int(np.argmax(total)),spatial_harmonics_real=fourier.real.tolist(),spatial_harmonics_imag=fourier.imag.tolist(),global_Phi_uniform_share=delta['global_Phi'],meaning='SIGNED totalH excess,notpositivephotonprobability. All matter/Peierls/Phi/electric terms; globalPhi uniformshare is bookkeeping,notlocaltransport.'))
        metrics.append(dict(time=point['time'],transverse_current_response_L2=float(np.linalg.norm(np.array(point['transverse_current'])-baseline_current)),z_number_current_response_L2=float(np.linalg.norm(np.array(point['z_number_current'])-np.array(vacuum['z_number_current']))),matter_density_response_L2=float(np.linalg.norm(np.array(point['density'])-baseline_density)),off_matter_density_pulse_response_L2=float(np.linalg.norm(np.array(a['density'])-np.array(b['density']))),matter_C5_recoil_L2=float(np.linalg.norm(matter)),off_matter_C5_pulse_response_L2=float(np.linalg.norm(offmatter)),field_angular_change=point['field_angular_mean']-vacuum['field_angular_mean'],coupled_minus_same_state_off_meanE_L2=float(np.linalg.norm(meanE-offE)),quantum_Phi_response=point['Phi_mean']-vacuum['Phi_mean'],sea_virtual_occupation_response=point['positive_reference_occupation']-vacuum['positive_reference_occupation'],Q_boundary_weight=point['Q_boundary_weight'],energy_drift=point['energy']-coupled[0]['energy']))
    initial=profiles[0]['signed_total_excitation_energy'];initial_components={key:float(sum(initial[key])) for key in keys};eiggroups=[]
    for group in mode_groups:
        eiggroups.append(dict(energy=group['energy'],multiplicity=group['multiplicity'],complete_below_upper_edge=group['complete_below_upper_edge'],coordinate_axis_weights=group['coordinate_axis_weights'],current_axis_weights=group['current_axis_weights'],max_residual=group['max_residual']))
    spectra=[];frequency=np.linspace(0,2.5,251);eta=.025
    for source in data['spectra']:
        item=dict(axis=source['axis'],band=source['axis']//4+1,polarization='x' if source['axis']%4<2 else 'y',quadrature='cos' if source['axis']%2==0 else 'sin',contact=source['contact'])
        for label in ('coordinate','literal_current'):
            group_label='current' if label=='literal_current' else 'coordinate'
            record=source[label];energies=np.array(record['energies']);weights=np.array(record['weights']);norm=record['raw_norm'];centroid=float(np.dot(weights,energies)/norm);width=float(np.sqrt(max(0,np.dot(weights,(energies-centroid)**2)/norm)));captured=sum(group[group_label+'_axis_weights'][source['axis']] for group in mode_groups);dominant=max(mode_groups,key=lambda group:group[group_label+'_axis_weights'][source['axis']]);response=np.array([np.dot(weights,1/(w+1j*eta-energies)-1/(w+1j*eta+energies)) for w in frequency]);item[label]=dict(raw_norm=norm,full_Lanczos_weight_sum=float(weights.sum()),centroid=centroid,width=width,own_low_groups_raw_capture=captured,own_low_groups_raw_uncaptured=norm-captured,dominant_actual_group_energy=dominant['energy'],dominant_actual_group_weight=dominant[group_label+'_axis_weights'][source['axis']],dominant_actual_group_fraction=dominant[group_label+'_axis_weights'][source['axis']]/norm,dominant_actual_group_complete=dominant['complete_below_upper_edge'],retarded_real=response.real.tolist(),retarded_imag=response.imag.tolist())
        spectra.append(item)
    with np.load(ROOT/(base+'_source_grams.npz')) as grams:
        plus=grams['addition'];minus=grams['removal'];sources=dict(raw_addition_diagonal=plus.diagonal().real.tolist(),raw_removal_diagonal=minus.diagonal().real.tolist(),CAR_sum_defect=float(np.linalg.norm(plus+minus.T-np.eye(20))),removal_hermiticity=float(np.linalg.norm(minus-minus.conj().T)),minimum_removal_eigenvalue=float(np.linalg.eigvalsh(minus).min()),minimum_addition_eigenvalue=float(np.linalg.eigvalsh(plus).min()))
    accepted_steps=[receipt for receipt in data['lanczos_steps']['fine'] if 'endpoint_residual' in receipt]
    result=dict(kind='COMPLETED_CONTROLLED_SHORT_COORDINATE_FORK_TOTAL_ENERGY_CURRENT_RECOIL_AND_OWN_GROUPS',mode=mode,base_mode=data['base_mode'],geometry=data['geometry'],energy=data['energy'],ground=data['ground'],profiles=profiles,metrics=metrics,initial_energy_component_totals=initial_components,pure_electric_preparation_defect=initial_components['total']-initial_components['electric'],exact_groups=eiggroups,source_spectra=spectra,retarded_frequency_grid=frequency.tolist(),retarded_eta=eta,source_grams=sources,bare_exact_groups=data['bare_groups'],field_EOM=data['field_EOM'],temporal_resolution_comparisons=data['comparisons'],phase_interpretation='SHORT interval and unchanged nearzero initialQ; no fitted frequency,groupvelocity or photonnumber is inferred.',max_total_profile_sum_defect=max(abs(p['profile_sum_defect']) for arm in records.values() for p in arm),max_excess_sum_defect=max(abs(p['profile_sum_minus_work']) for p in profiles),max_endpoint_residual=max(r['endpoint_residual'] for r in accepted_steps),integrated_defect_sum=sum(r['local_defect_indicator'] for r in accepted_steps),electric_profile_operator_sum_defect=data['electric_profile_operator_sum_defect'],eigensolver_residual=data['eigensolver_residual'],limits=data['limits'],wall_seconds=time.monotonic()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__))
    result['own_complete_group_residues']=group_residues(base,data,mode_groups,source_stem)
    result['group_completion_source_closure']=completion['source_closure']
    result['original_partial13_flags_not_authoritative']=True
    result['wall_seconds']=time.monotonic()-start
    result['peak_RSS_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (ROOT/(stem+'energy_data.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(mode=mode,wall_seconds=result['wall_seconds'],pure_electric_preparation_defect=result['pure_electric_preparation_defect'],max_profile_sum_defect=result['max_total_profile_sum_defect'])),flush=True)

if __name__=='__main__':run()
