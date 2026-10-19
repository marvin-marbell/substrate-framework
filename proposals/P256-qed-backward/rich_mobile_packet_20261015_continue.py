"""Complete the interrupted NEW physical experiment from exact owned snapshots.
No frozen producer replay; completed charged baseline is consumed, not repeated.
"""
from __future__ import annotations
import json,resource,sys,time,gc
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from rich_mobile_packet_20261015_model import geometry,field,install,apply_terms
from rich_mobile_packet_20261015_calculate import physical_receivers,observe,step,norm2,savejson
ROOT=Path(__file__).resolve().parent


def main() -> None:
    started=time.time();prefix=ROOT/sys.argv[1];out=lambda s:Path(str(prefix)+s);inp=json.loads(out('_input.json').read_text());supplier=ROOT/inp['supplier_prefix'];old=lambda s:Path(str(supplier)+s)
    c=json.loads(old('_input.json').read_text());checkpoint=json.loads(old('_checkpoint.json').read_text());initial=json.loads(old('_initial.json').read_text());states=np.load(old('_initial_states.npz'));pr=np.load(old('_primitive.npz'))
    geo=geometry(c);f=field(c,geo)
    equality=dict(components=float(np.linalg.norm(geo['transformed']-pr['onebody_components'])),field_terms=float(np.linalg.norm(f['terms']-pr['field_terms'])),boson=float(np.linalg.norm(f['boson']-pr['field_boson'])))
    if max(equality.values())>1e-11:raise ValueError(('Owning snapshot changed',equality))
    sectors={}
    for name in ['neutral','positive']:
        sec=dict(name=name,masks=[int(m) for m in json.loads(old('_'+name+'_masks.json').read_text())['masks']],matrices=[sparse.load_npz(old(f'_{name}_Hcomponent{k}.npz')) for k in range(len(f['terms']))])
        physical_receivers(geo,sec,c);sectors[name]=sec
    neutral=sectors['neutral'];positive=sectors['positive'];vac=states['neutral'];vacdensity=np.array([float(np.vdot(vac.reshape(-1,225),o@vac.reshape(-1,225)).real) for o in neutral['density']])
    times=np.linspace(0,inp['duration'],inp['time_points']);arms=dict(checkpoint['completed_arms']);diagnostics={name:[] for name in arms};state_paths={};files=[]
    for j in [0,8,16]:state_paths[f'charged_no_incoming:{j}']=old(f'_charged_no_incoming_state{j}.npz').name
    def save_anchor(label: str,j: int,state: np.ndarray) -> None:
        if j in [0,8,16]:
            p=out(f'_{label}_state{j}.npz');np.savez_compressed(p,state=state,time=times[j]);files.append(p.name);state_paths[f'{label}:{j}']=p.name
    # Recompute only unfinished incoming trajectory from its last available full exact snapshot(t0).
    for label,sec,start in [('charged_incoming',positive,states['charged_field']),('neutral_incoming',neutral,states['neutral_field'])]:
        h=install(sec,f,geo['offset']);state=start.copy();records=[];steps=[]
        for j,t in enumerate(times):
            if j:state,d=step(h,state,float(times[j]-times[j-1]),inp['krylov_dimension']);steps.append(d)
            records.append(observe(geo,f,sec,h,state,vacdensity,c));save_anchor(label,j,state)
            savejson(out('_checkpoint.json'),dict(completed_arms=arms,current_arm=label,times=times[:j+1].tolist(),current_records=records,step_diagnostics=steps,state_paths=state_paths))
            print('Continued actual physical trajectory',label,float(t),flush=True)
        arms[label]=records;diagnostics[label]=steps;del state,h;gc.collect()
    # Exact complete EM-off Kronecker sum, with all quantum sea/Phi retained.
    n=len(positive['masks']);mats=positive['matrices'];constant=mats[0].copy()
    for k in range(2,len(mats),2):constant+=mats[k]
    hmp=sparse.kron(constant,sparse.eye(5),format='csr')+sparse.kron(mats[1],sparse.diags(c['coupling']*f['phi']),format='csr')+sparse.kron(sparse.eye(n),sparse.csr_matrix(f['phi_h']),format='csr')-geo['offset']*sparse.eye(n*5,format='csr')
    matterphi=states['charged'].reshape(n*5,45);trace=hmp.diagonal().sum()
    off_amplitudes=expm_multiply(-1j*hmp,matterphi,start=0,stop=times[-1],num=len(times),traceA=-1j*trace)
    hoff=install(positive,f,geo['offset'],True);off_records={name:[] for name in ['charged_same_state_EMoff_no_incoming','charged_same_state_EMoff_incoming']};drive=states['field_drive']
    off_derivatives=f['derivatives']*(0.0/c['charge'])
    # Different field-only initial states; same actual propagated matterPhi amplitudes.
    for j,t in enumerate(times):
        freefield=expm(-1j*t*f['kinetic']);base=off_amplitudes[j]@freefield.T;pulse=off_amplitudes[j]@drive.T@freefield.T
        for label,x in [('charged_same_state_EMoff_no_incoming',base),('charged_same_state_EMoff_incoming',pulse)]:
            state=x.ravel();record=observe(geo,f,positive,hoff,state,vacdensity,c)
            record['counterfactual_coupled_profile_current']=record.pop('transverse_profile_current')
            record['transverse_profile_current']=[float(np.vdot(state,-apply_terms(positive,d,state)).real) for d in off_derivatives]
            record['profile_current_semantics']='Own EMoff derivative: profile charge0. Coupled-current insertion separately retained as counterfactual diagnostic.'
            off_records[label].append(record);save_anchor(label,j,state)
        print('Actual same-state factorized EMoff trajectories',float(t),flush=True)
    arms.update(off_records);diagnostics.update({name:[dict(method='Exact sparse matterPhi expm_multiply batch plus exact45field exponential; same actual amplitudes, no response hardcode.')] for name in off_records})
    comparisons={}
    for rootname in ['charged','charged_same_state_EMoff']:
        a0=arms[rootname+'_no_incoming'];a1=arms[rootname+'_incoming'];comparisons[rootname]=dict(density_L2=[float(np.linalg.norm(np.array(p['number_density_z'])-b['number_density_z'])) for p,b in zip(a1,a0)],
            current_L2=[float(np.linalg.norm(np.array(p['number_flux_z'])-b['number_flux_z'])) for p,b in zip(a1,a0)],whole_sea_Sz_difference=[p['whole_sea_canonical_Sz']-b['whole_sea_canonical_Sz'] for p,b in zip(a1,a0)],
            matter_momentum_L2=[float(np.linalg.norm(np.array(p['momentum']['matter'])-b['momentum']['matter'])) for p,b in zip(a1,a0)])
    structure={name:dict(matter_dimension=len(sec['masks']),joint_dimension=len(sec['masks'])*225) for name,sec in sectors.items()}
    result=dict(kind='NATURALLY_COMPLETED_SAME_H_TWO_BAND_PACKET_EXPERIMENT_AFTER_TIMEOUT',input=c,continuation_input=inp,structure=structure,initial=initial,
        artifact_supplier_prefix=supplier.name,state_paths=state_paths,snapshot_equality=equality,coupled_scales=json.loads(old('_scales.json').read_text())['scales'],
        times=times.tolist(),vacuum_density_z=vacdensity.tolist(),vacuum_field=observe(geo,f,neutral,install(neutral,f,geo['offset']),vac,vacdensity,c)['field'],
        arms=arms,comparisons=comparisons,numerical_steps=diagnostics,artifact_files=files,
        limitations=['First capture tool3600s timeout is retained, not a natural exit. Completed baseline consumed unchanged; its individual Lanczos endpoint records were not in completed-arm checkpoint and remain unavailable.',
            'Selected nonlinear two-pair CI, actual C4/C nonclosure and physical current/source omissions remain; no Gaussian-span equivalence or asymptotic photon/spinflip claim.',
            'Source z-localized with transverse plane-wave fiber; random-sea n/N and signed excess not isolated-particle position.',
            'Physical current is literal before-cut current, not patched to projected continuity; full fieldcut defects retained.'],
        elapsed_seconds=time.time()-started,peak_resident_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    savejson(out('_result.json'),result);print(json.dumps(dict(result=out('_result.json').name,elapsed_seconds=result['elapsed_seconds'],peak_resident_KiB=result['peak_resident_KiB'],comparisons=comparisons)),flush=True)


if __name__=='__main__':main()
