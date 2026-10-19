"""Natural physical consumer of NEW exact packet states: tails, flow, mode phase and omitted action."""
from __future__ import annotations
import hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.linalg import eigh,expm
from rich_mobile_packet_20261015_model import install,apply_field,apply_terms,field as owning_field,geometry
from rich_mobile_packet_20261015_calculate import field_measure
ROOT=Path(__file__).resolve().parent


def digest(p: Path) -> str:return hashlib.sha256(p.read_bytes()).hexdigest()
def n2(x: np.ndarray) -> float:return float(np.vdot(x,x).real)


def omitted_current(masks: list,h: np.ndarray) -> tuple:
    """Exact outside-basis physical dGamma(current) action, streamed over boson columns."""
    retained=set(masks);lookup={};rows=[];cols=[];values=[]
    targets=[[(int(i),h[i,j]) for i in np.flatnonzero(abs(h[:,j])>1e-13) if i!=j] for j in range(180)]
    for k,m in enumerate(masks):
        for j in range(180):
            if not m&(1<<j):continue
            first=m^(1<<j);sgn=-1 if (m&((1<<j)-1)).bit_count()%2 else 1
            for i,value in targets[j]:
                if first&(1<<i):continue
                dest=first|(1<<i)
                if dest in retained:continue
                row=lookup.setdefault(dest,len(lookup));parity=-1 if (first&((1<<i)-1)).bit_count()%2 else 1
                rows.append(row);cols.append(k);values.append(value*sgn*parity)
    return sparse.csr_matrix((values,(rows,cols)),shape=(len(lookup),len(masks))),len(lookup)


def scalar_profile(x: list) -> dict:
    p=np.array(x,float);z=np.arange(5);moment=np.sum(p*np.exp(2j*np.pi*z/5))
    return dict(values=p.tolist(),sum=float(p.sum()),L1=float(abs(p).sum()),minimum=float(p.min()),maximum=float(p.max()),
        circular_first_Fourier_real=float(moment.real),circular_first_Fourier_imag=float(moment.imag),circular_phase=float(np.angle(moment)),
        peak_index=int(p.argmax()),outside_peak_signed_sum=float(p.sum()-p.max()),negative_weight=float(-p[p<0].sum()),
        meaning='Signed excess charge/energy is NOT a probability; phase is a spatial Fourier descriptor, not a particle position.')


def main() -> None:
    started=time.time();prefix=ROOT/sys.argv[1];path=lambda suffix:Path(str(prefix)+suffix)
    result=json.loads(path('_result.json').read_text());receipt=json.loads(path('_exit.json').read_text())
    supplier=ROOT/result.get('artifact_supplier_prefix',prefix.name);sourcepath=lambda suffix:Path(str(supplier)+suffix)
    statepath=lambda label,index:ROOT/result.get('state_paths',{}).get(f'{label}:{index}',prefix.name+f'_{label}_state{index}.npz')
    pr=np.load(sourcepath('_primitive.npz'))
    inputs=[path('_result.json'),path('_exit.json'),path('_manifest.json'),sourcepath('_primitive.npz'),sourcepath('_initial_states.npz'),sourcepath('_family.json')]
    f=dict(terms=pr['field_terms'],off=pr['field_off'],boson=pr['field_boson'])
    # Literal owning before-cut physical Peierls current; rederive derivative coefficient from immutable profile geometry.
    bands=2;derivatives=np.zeros((8,len(f['terms']),225,225),complex)
    k=2
    for axis in range(3):
        for level in (range(5) if axis<2 else [0]):
            profile=np.zeros(8)
            if axis<2:
                for band in range(bands):profile[4*band+2*axis:4*band+2*axis+2]=result['input']['charge']*np.array([np.cos(2*np.pi*(band+1)*level/5),np.sin(2*np.pi*(band+1)*level/5)])
            for a in range(8):derivatives[a,k]=-profile[a]*f['terms'][k+1];derivatives[a,k+1]=profile[a]*f['terms'][k]
            k+=2
    initial=np.load(sourcepath('_initial_states.npz'));EOM={};omissions={};sources={};source_returns={};quantum_phi={}
    for name in ['neutral','positive']:
        maskpath=sourcepath('_'+name+'_masks.json');inputs.append(maskpath);masks=[int(m) for m in json.loads(maskpath.read_text())['masks']]
        matrices=[]
        for k in range(len(f['terms'])):
            p=sourcepath(f'_{name}_Hcomponent{k}.npz');inputs.append(p);matrices.append(sparse.load_npz(p))
        sec=dict(masks=masks,matrices=matrices);offset=float(pr['reference_energies'][:90].sum());h=install(sec,f,offset)
        state=initial['neutral' if name=='neutral' else 'charged'];hs=h@state;checks=[]
        for axis in [0,4]:
            qstate=apply_field(state,pr['field_q'][axis]);pstate=apply_field(state,pr['field_p'][axis]);current=-apply_terms(sec,derivatives[axis],state)
            dq=1j*(h@qstate-apply_field(hs,pr['field_q'][axis]))-pstate/(result['input']['volume']/2)
            dp=1j*(h@pstate-apply_field(hs,pr['field_p'][axis]))-current
            checks.append(dict(axis=axis,Qdot_minus_P_over_I_norm=float(np.linalg.norm(dq)),Pdot_minus_literal_current_norm=float(np.linalg.norm(dp)),
                literal_current_action_retained_norm2=n2(current)))
        EOM[name]=checks
        outside,count=omitted_current(masks,pr['onebody_components'][-1]);actions=[]
        anchors=[('initial',state)]
        if name=='positive':
            # Evaluate the owning OFF-law zero-profile-charge derivative on actual
            # saved states; coupled-current insertions are separately labeled.
            off_derivatives=derivatives*(0.0/result['input']['charge']);physical_off=[]
            for label in ['charged_same_state_EMoff_no_incoming','charged_same_state_EMoff_incoming']:
                for index in [0,8,16]:
                    p=statepath(label,index);inputs.append(p);psi=np.load(p)['state']
                    physical_off.append(dict(arm=label,index=index,owning_complete_EMoff_profile_current=[float(np.vdot(psi,-apply_terms(sec,d,psi)).real) for d in off_derivatives]))
            EOM['physical_EMoff_current']=physical_off
            EOM['primary_off_current_key_semantics']='Continuation reports owning zero-charge OFF current and retains the coupled-current insertion separately as counterfactual. Dynamics and physical z-current are actual owning-law quantities.'
        labels=['charged_no_incoming','charged_incoming'] if name=='positive' else ['neutral_incoming']
        for label in labels:
            p=statepath(label,16);inputs.append(p);anchors.append((label+'_t8',np.load(p)['state']))
        if name=='positive':
            indices=np.flatnonzero(np.all(pr['momenta'][:,:2]==result['input']['transverse_source_momentum'],axis=1));sourceops=[]
            for j in range(20):
                p=sourcepath(f'_add_source{j}.npz');inputs.append(p);sourceops.append(sparse.load_npz(p))
            for label,psi in anchors:
                x=psi.reshape(-1,225);population=np.sum(abs(x)**2,axis=1);weights=[]
                for j,op in zip(indices,sourceops):
                    full=float(sum(population[r] for r,m in enumerate(masks) if m&(1<<int(j))));retained=n2(op.conj().T@x)
                    weights.append(dict(orbital=int(j),full_physical_removal_norm2=full,retained_neutral_return_norm2=retained,omitted_difference=full-retained))
                source_returns[label]=weights
        for label,psi in anchors:
            phi=result['input']['phi_center']+2*result['input']['phi_width']*(np.arange(5)-2)/5
            probabilities=np.sum(abs(psi.reshape(-1,5,45))**2,axis=2);phiweight=probabilities.sum(axis=0)
            positive_count=np.array([sum(bool(m&(1<<j)) for j in range(90,180)) for m in masks]);mean_phi=float(phiweight@phi);mean_count=float(positive_count@probabilities.sum(axis=1))
            quantum_phi[name+':'+label]=dict(phi_grid=phi.tolist(),actual_phi_distribution=phiweight.tolist(),mean_phi=mean_phi,variance_phi=float(phiweight@(phi**2)-mean_phi**2),
                positive_reference_band_occupancy=mean_count,phi_positive_band_covariance=float(np.einsum('i,ip,p->',positive_count,probabilities,phi)-mean_phi*mean_count),
                meaning='Actual coupled quantum scalar/filled-sea state, not an instantaneous frozen scalar or isolated-valence sea.')
            x=psi.reshape(-1,225);missing=0.
            for start in range(0,225,8):missing+=n2(outside@x[:,start:start+8])
            retained=n2(matrices[-1]@x)
            actions.append(dict(label=label,retained_physical_total_z_number_current_action_norm2=retained,omitted_norm2=missing,full_action_norm2=retained+missing,
                omitted_fraction=missing/(retained+missing) if retained+missing else None))
        omissions[name]=dict(outside_masks=count,actions=actions,meaning='Literal physical total z-current before matter projection; exact omitted neighbor action streamed over225bosoncolumns. NOT an error bound on the evolution, not all transverse current omissions.')
        del h,matrices,outside
    analyses={};times=np.array(result['times']);vacfield=np.array(result['vacuum_field']['electric_energy_density_before_cut'])
    for label,records in result['arms'].items():
        extra=[];field=[];modephase={}
        for r in records:
            extra.append(scalar_profile(r['signed_excess_number_z']));energy=np.array(r['field']['electric_energy_density_before_cut']);e=np.array(r['field']['electric_profile']);intensity=np.sum(e*e,axis=1)
            field.append(dict(full_positive_energy_fraction=(energy/energy.sum()).tolist(),signed_energy_above_vacuum=scalar_profile((energy-vacfield).tolist()),
                actual_mean_electric_intensity=intensity.tolist(),mean_electric_intensity_fraction=(intensity/intensity.sum()).tolist() if intensity.sum()>1e-28 else None,
                peak_energy_index=int(energy.argmax()),electric_integral_minus_own_kinetic=r['field']['electric_energy_integral']-r['field']['electric_kinetic']))
        for band in range(2):
            for pol in range(2):
                idx=4*band+2*pol;series=np.array([complex(r['field']['Q'][idx],r['field']['Q'][idx+1])/np.sqrt(2) for r in records]);phase=np.unwrap(np.angle(series))
                modephase[f'band{band+1}_pol{pol}']=dict(complex_mean_coordinate=[[float(z.real),float(z.imag)] for z in series],amplitudes=abs(series).tolist(),unwrapped_phase=phase.tolist(),
                    observed_end_to_end_phase_rate=float((phase[-1]-phase[0])/(times[-1]-times[0])),
                    meaning='Actual evolving mean profile Fourier phase. May contain mixed/dephasing modes; not an imposed dispersion or identified sharp photon frequency.')
        density=np.array([r['number_density_z'] for r in records]);total=np.array([r['momentum']['total'] for r in records]);energies=np.array([r['energy'] for r in records])
        analyses[label]=dict(signed_charge_profiles=extra,field_localization=field,field_mode_phase=modephase,
            actual_density_change_from_t0_L2=np.linalg.norm(density-density[0],axis=1).tolist(),total_momentum_PVM_drift_L2=np.linalg.norm(total-total[0],axis=1).tolist(),
            energy_drift=(energies-energies[0]).tolist(),max_continuity_defect=max(r['physical_continuity_L2'] for r in records),
            max_Lanczos_endpoint_residual=max((x['endpoint_Schrodinger_residual'] for x in result['numerical_steps'][label] if 'endpoint_Schrodinger_residual' in x),default=None))
    # Actual differential charge reaction beyond neutral-sea polarization.
    charged_delta=np.array([r['number_density_z'] for r in result['arms']['charged_incoming']])-np.array([r['number_density_z'] for r in result['arms']['charged_no_incoming']])
    neutral_delta=np.array([r['number_density_z'] for r in result['arms']['neutral_incoming']])-np.array(result['vacuum_density_z'])
    differential=dict(charged_density_change=charged_delta.tolist(),neutral_sea_density_change=neutral_delta.tolist(),charged_minus_neutral_density_reaction=(charged_delta-neutral_delta).tolist(),
        charged_minus_neutral_density_reaction_L2=np.linalg.norm(charged_delta-neutral_delta,axis=1).tolist())
    field_deltas={}
    for rootname in ['charged','charged_same_state_EMoff']:
        field_deltas[rootname]=np.array([r['field']['electric_profile'] for r in result['arms'][rootname+'_incoming']])-np.array([r['field']['electric_profile'] for r in result['arms'][rootname+'_no_incoming']])
    differential['mean_electric_coupled_minus_same_state_bare']= (field_deltas['charged']-field_deltas['charged_same_state_EMoff']).tolist()
    differential['mean_electric_coupled_minus_same_state_bare_L2']=np.linalg.norm((field_deltas['charged']-field_deltas['charged_same_state_EMoff']).reshape(len(times),-1),axis=1).tolist()
    # Additional bare-electric packet itself: finite projected P^2 can mimic an oscillator.
    fullfield=owning_field(result['input'],geometry(result['input']))
    if np.linalg.norm(fullfield['terms']-pr['field_terms'])>1e-11:raise ValueError('Bare-control owning field snapshot mismatch')
    eb,ub=eigh(fullfield['kinetic']);barevac=ub[:,0];barepulse=initial['field_drive']@barevac;bare_records=[];bare_states=[];bare_scales=[]
    for axis,q in enumerate(fullfield['q']):
        source=q@barevac;weight=n2(source);action=(fullfield['kinetic']-eb[0]*np.eye(45))@source
        bare_scales.append(dict(axis=axis,raw_retained_weight=weight,centroid=float(np.vdot(source,action).real/weight),second_moment=n2(action)/weight))
    for t in times:
        psi=expm(-1j*t*fullfield['kinetic'])@barepulse;embedded=np.zeros(225,complex);embedded[:45]=psi
        bare_records.append(field_measure(fullfield,embedded));bare_states.append(psi)
    bare_statefile=ROOT/'rich_mobile_packet_20261015_bare_electric_states.npz'
    np.savez_compressed(bare_statefile,times=times,states=np.array(bare_states),bare_ground=barevac,eigenvalues=eb)
    bare_control=dict(kind='ACTUAL_FIELD_ONLY_BARE_PROJECTED_P2_PACKET_CONTROL_NOT_PHOTON',ground_energy=float(eb[0]),eigenvalues=eb.tolist(),coordinate_source_scales=bare_scales,records=bare_records,
        output_state_file=bare_statefile.name,output_state_sha256=digest(bare_statefile),
        meaning='Same shell45/inertia22.5/field-only pulse, own bare-electric reference and literal projected P^2 evolution. Its finite-box gap or moving mean/energy profile is NOT induced photon physics. Separate same-entangled-state OFF trajectories control actual matter reaction.')
    sources={p.name:digest(p) for p in inputs};output=ROOT/'rich_mobile_packet_20261015_observed.json'
    data=dict(kind='ACTUAL_NEW_PACKET_STATES_PHYSICAL_CURRENT_TAILS_AND_CUTOFF_CONSUMER',primary_natural_returncode=receipt['returncode'],
        inputs_sha256=sources,EOM=EOM,physical_current_omissions=omissions,raw_reverse_source_weights=source_returns,quantum_phi_feedback=quantum_phi,differential_response=differential,bare_electric_control=bare_control,arms=analyses,elapsed_seconds=time.time()-started,
        peak_resident_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    output.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(dict(result=output.name,elapsed=data['elapsed_seconds'],peak_resident_KiB=data['peak_resident_KiB'],physical_current_omissions=omissions,EOM=EOM)),flush=True)


if __name__=='__main__':main()
