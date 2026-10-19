"""Continue ONLY event step from own fresh six-profile-response checkpoint.
The upstream actual1 exception was a LinearOperator/CSR interface error, not science.
"""
import hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator,expm_multiply
from rich_spatial_colour_charge_20261015_snapshot import load as load_body
from rich_spatial_colour_em_20261015_join import from_body,assemble,native_C,native_charge_receiver
from rich_spatial_colour_em_20261015_model import observable
from rich_spatial_colour_em_20261015_measure import field_action,joint_receiver
from rich_spatial_colour_em_20261015_density import density,spatial_current,mechanical_operator,no_hop_law
from rich_review_wave19_20261015_em import measurements,norm,mean,cj


def propagate(h,state,t,energy):
    if not t:return state.copy()
    shifted=LinearOperator(h.shape,matvec=lambda x:h@x-energy*x,rmatvec=lambda x:h.H@x-energy*x,matmat=lambda x:h@x-energy*x,dtype=complex)
    return expm_multiply(-1j*t*shifted,state,traceA=-1j*t*(h.trace()-energy*h.shape[0]))


def main():
    inp,out=map(Path,sys.argv[1:3]);request=json.loads(inp.read_text());upstream=inp.parent/request['upstream_json'];statesfile=inp.parent/request['upstream_npz'];metadata=json.loads(upstream.read_text());case=metadata['case'];p=metadata['parameters'];start=time.monotonic();bodyfile=inp.parent/p['native_body_cache']
    if metadata['stage']!='six_profile_full_response':raise ValueError('Require successful completed fresh source/response checkpoint')
    if hashlib.sha256(bodyfile.read_bytes()).hexdigest()!=p['native_body_cache_sha256']:raise ValueError('Pinned BODY mismatch')
    body=load_body(bodyfile);model=from_body(body,p);base=model['base'];field=model['field'];rotor=base['rotor'];nr=rotor['dimension'];data=np.load(statesfile);event={};saved={};positive=None
    result=dict(kind='WAVE19_SAME_NEW_PRIMARY_H_EVENT_CONTINUATION',upstream_stage=metadata['stage'],upstream_json_sha256=hashlib.sha256(upstream.read_bytes()).hexdigest(),upstream_npz_sha256=hashlib.sha256(statesfile.read_bytes()).hexdigest(),parameters=p,case=case,events=event,meaning='Only the failed event propagation step is resumed. Fresh reference/source/full6 response were successfully constructed by own actual1 before its natural1 API exception. No external/old/base/Frame state or law enters this continuation.')
    def checkpoint(stage):
        result.update(stage=stage,resource=dict(seconds=time.monotonic()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));out.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),**saved)
    for name in ['lepton','antilepton']:
        initial=data[name+'_raw_normalized_source'];hs=assemble(model,name);off=assemble(model,name,coupled=False);hopoff=no_hop_law(model,name,hs);curr=[assemble(model,name,derivative=(mu,)) for mu in range(6)];nm=base['operators'][name]['Q'].shape[0];nf=len(base['scalar']['grid']);sign=-1 if name.startswith('anti') else 1;unitary=expm(-1j*sign*case['kick']*(field['Pi'][2]+.37*field['Pi'][4]));kicked=field_action(initial,nm,nf,nr,unitary);labels,effects=native_charge_receiver(model,name);records=[];xs=[];bs=[];receivers=[];phase=mean(initial,hs);times=case['times'];before=measurements(model,name,initial);after=measurements(model,name,kicked)
        for t in times:
            b=propagate(hs,initial,t,phase);x=propagate(hs,kicked,t,phase);bm=measurements(model,name,b);xm=measurements(model,name,x);joint=joint_receiver(x,effects,base['scalar'],rotor,field);records.append(dict(time=t,norm=norm(x)**2,excess={key:xm[key]-bm[key] for key in bm},current_excess=[mean(x,a)-mean(b,a) for a in curr],joint_probability_sum=float(joint.sum()),joint=joint.tolist()));xs.append(x);bs.append(b);receivers.append(joint)
        fbase=propagate(off,initial,times[-1],phase);fpulse=propagate(off,kicked,times[-1],phase);fm=measurements(model,name,fbase);fx=measurements(model,name,fpulse);hb=propagate(hopoff,initial,times[-1],phase);hx=propagate(hopoff,kicked,times[-1],phase);rho=observable(model,name,matter=base['operators'][name]['rho0']);routes=[assemble(model,name,route_derivative=edge) for edge in range(2)];graph=1j*(hs@(rho@initial)-rho@(hs@initial))+sum((a@initial for a in routes),np.zeros_like(initial));k=np.array(case['wavevector']);rhok=density(model,name,k);local=[spatial_current(model,name,k,a) for a in range(3)];wrong=[spatial_current(model,name,k,a,include_mechanical=False) for a in range(3)];change=1j*(hs@(rhok@initial)-rhok@(hs@initial));ward=change+1j*sum((q*(a@initial) for q,a in zip(k,local)),np.zeros_like(initial));bad=change+1j*sum((q*(a@initial) for q,a in zip(k,wrong)),np.zeros_like(initial))
        momentum=observable(model,name,rq=np.kron(np.eye(nr),field['Pi'][2]));electric=observable(model,name,rq=np.kron(np.eye(nr),field['H_E']));force=1j*(hs@(momentum@initial)-momentum@(hs@initial));bare=1j*(electric@(momentum@initial)-momentum@(electric@initial));charge_before=np.array([mean(initial,observable(model,name,matter=a)) for a in effects]);charge_hb=np.array([mean(hb,observable(model,name,matter=a)) for a in effects]);charge_hx=np.array([mean(hx,observable(model,name,matter=a)) for a in effects])
        event[name]=dict(protocol='Unfiltered nonstationary own source; same-state untouched baseline, field-only Pi impulse. Field Q/current can change immediately, not instantaneous matter reaction.',initial_energy_variance_norm=norm(hs@initial-phase*initial),initial_matter_changes={key:after[key]-before[key] for key in ['rho0','rho1','N','Q','C','B','phi2']},initial_current_changes=[mean(kicked,a)-mean(initial,a) for a in curr],records=records,charge_labels=labels,frozen_all_transverse_EM_off_excess={key:fx[key]-fm[key] for key in fm},frozen_probe_current_image_norms=[norm(assemble(model,name,derivative=(mu,),coupled=False)@initial) for mu in range(6)],all_hops_off=dict(material_charge_motion_norm=norm(1j*(hopoff@(rho@initial)-rho@(hopoff@initial))),charge_baseline_TV_from_initial=float(np.sum(abs(charge_hb-charge_before))/2),charge_pulse_TV_from_initial=float(np.sum(abs(charge_hx-charge_before))/2),mechanical_current_image_norms=[norm(mechanical_operator(model,name,derivative=(mu,))@initial) for mu in range(6)],laboratory_density_excess=cj(np.vdot(hx,rhok@hx)-np.vdot(hb,rhok@hb))),material_graph_Ward=norm(graph),spatial_Fourier_Ward=norm(ward),wrong_no_mechanical_Fourier_Ward=norm(bad),physical_current_vs_Pi_force_defect=norm(force-curr[2]@initial),bare_projected_Pi_squared_cut_force=norm(bare),receiver='Native charge PVM x scalar sign x ONE compressed continuous field-sign effect; not a sharp joint6Q PVM or source overlap.')
        saved[name+'_initial']=initial;saved[name+'_kicked']=kicked;saved[name+'_event']=np.array(xs);saved[name+'_baseline']=np.array(bs)
        if name=='lepton':positive=(initial,kicked,xs,receivers,labels)
        else:
            c=native_C(model,'lepton');pi,pk,px,pr,pl=positive;alignment=np.vdot(initial,c@pi);alignment/=abs(alignment);order=[next(i for i,q in enumerate(pl) if abs(q+anti)<1e-10) for anti in labels];expected=np.array(pr)[:,order,:,::-1]
            event[name]['actual_C']=dict(initial_phase_aligned_defect=norm(alignment*initial-c@pi),kick_defect=norm(alignment*kicked-c@pk),all_M_event_defects=[norm(alignment*x-c@y) for x,y in zip(xs,px)],inclusive_reversed_charge_field_TV=[float(np.sum(abs(a-b))/2) for a,b in zip(receivers,expected)])
        checkpoint('event_'+name);print(json.dumps({'finished':name,'final_current':records[-1]['current_excess']}),flush=True)
    checkpoint('complete');print(json.dumps(result['resource']))


if __name__=='__main__':main()
