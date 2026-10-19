"""Fresh SAME-H neutral/N3/N5 coupled IP physics, true Weyl kicks and dt control."""
import gc
import json
from pathlib import Path
import sys
from typing import Any
import numpy as np
from rich_spatial_frame_photon_20261016_run import evolve as lanczos
import rich_exact_electric_picture_20261016_model as m
ROOT=Path(__file__).resolve().parent
P='rich_exact_electric_picture_20261016_'


def propagate(model,state,t0,t1,impulse,dt,hopping=True,coupled=True):
    n=max(1,int(np.ceil((t1-t0)/dt)));h=(t1-t0)/n;v=np.asarray(state,dtype=np.complex128).copy();errors=[]
    for k in range(n):
        absolute=t0+(k+.5)*h
        v,e=lanczos(lambda x:model.apply(x,absolute,impulse,hopping,coupled),v,h,10);errors.append(e)
    return v,dict(method='time-ordered exponential midpoint Magnus2, Hermitian Lanczos10',dt=h,absolute_interval=[t0,t1],steps=n,max_terminal_residual=max(float(e.get('terminal_residual',0.)) for e in errors),norm_defect=float(np.linalg.norm(v)-np.linalg.norm(state)))


def moments(model,state,time,impulse,channels=(0,2,4,6),coupled=True,hopping=True):
    x=state.reshape(model.shape);hm=model.matter;prob=np.sum(abs(x)**2,axis=(1,2,3,4));rprob=np.sum(abs(x)**2,axis=(0,1,3,4))
    Pi=np.array([np.vdot(state,model.field(state,a,time,impulse)).real for a in range(8)])
    a=np.array([np.vdot(state,model.field(state,k,time,impulse,'q')).real for k in range(8)])
    spin=[];spin2=0.
    for op in hm.g['L']:
        z=np.einsum('ij,fjrma->firma',op.toarray(),x,optimize=True);spin.append(float(np.vdot(x,z).real));spin2+=float(np.linalg.norm(z)**2)
    source={str(k):float(-np.vdot(state,model.vertex(state,k,time,impulse,hopping=hopping,coupled=coupled)).real) for k in channels} if channels else {}
    receiver=m.frozen.kernel.profiles(np.array([[0.,0.,0.],[.8,-.4,1.1]]),model.fixed.modes)
    hI=model.apply(state,time,impulse,hopping,coupled);h0=model.free_energy(state,impulse)
    return dict(norm=float(np.linalg.norm(state)**2),physical_energy=float(np.vdot(state,hI+h0).real),HI_energy_NOT_conserved=float(np.vdot(state,hI).real),H0_energy=float(np.vdot(state,h0).real),Pi=Pi.tolist(),a=a.tolist(),receiver_Pi=np.einsum('sla,l->sa',receiver,Pi).tolist(),density=(prob@hm.sector.n).tolist(),r=float(rprob@hm.rnodes),LEFT=spin,LEFT2=spin2,physical_current=source,initial_coefficient_top_weight=float(sum(np.linalg.norm(x[...,i])**2 for i,n in enumerate(model.basis) if sum(n)==model.parameters['field_total_cut'])))


def temporal_difference(end,base,initial_kick,initial):
    return {key:(np.asarray(end[key])-np.asarray(base[key])-np.asarray(initial_kick[key])+np.asarray(initial[key])).tolist() for key in ['Pi','a','density','LEFT','r','LEFT2','receiver_Pi']}


def main():
    cut=int(sys.argv[1]) if len(sys.argv)>1 else 1;label=f'field{cut}';p=dict(m.DEFAULTS,field_total_cut=cut);T=.04;dt=.02
    result: dict[str,Any]=dict(kind='ACTUAL_SAME_H_EXACT_UNBOUNDED_ELECTRIC_INTERACTION_PICTURE',parameters=p,T=T,dt=dt,basis='U0(t) Wimpulse |initial Hermite label>; H0 exact on full L2, no cut free propagator',initial_ray='Fresh frozen48 projected t0 Wilson-source ray at same cut; exactly same UNKICKED physical initial ray. Pulse is full W on that ray, unlike prior exp(i eps PaP); improvement attribution distinguishes kick from free evolution.',control='FIELD-DECOUPLED/HOPPING-DISABLED retains R kinetic and rotating x(R), not charge-position freeze',cases={})
    reference=m.Model(4,p);neutral=reference.preparation();prepared={4:neutral}
    for number,name,component,create in [(4,'neutral',None,None),(5,'addition',0,True),(3,'removal',1,False)]:
        model=reference if number==4 else m.Model(number,p);raw: dict[str,Any]={}
        if number==4:v=neutral;weights=None
        else:
            packets=[model.source(reference,neutral,k,create) for k in range(8)]
            weights=[float(np.linalg.norm(z)**2) for z in packets];v=packets[component]/np.sqrt(weights[component]);del packets
        raw['input']=v;eps0=model.zeros;initial=moments(model,v,0.,eps0)
        base,eb=propagate(model,v,0.,T,eps0,dt);raw['baseline']=base
        baseline=moments(model,base,T,eps0)
        off,eo=propagate(model,v,0.,T,eps0,dt,False,False);raw['field_decoupled_baseline']=off
        offbase=moments(model,off,T,eps0,channels=(),coupled=False,hopping=False)
        protocols={};errors=[eb,eo]
        for inc,out in [(0,6),(2,4)]:
            eps=model.impulse(inc);end,e=propagate(model,v,0.,T,eps,dt);errors.append(e)
            start=moments(model,v,0.,eps,channels=(inc,out));finish=moments(model,end,T,eps,channels=(inc,out))
            # OFF coefficient equation is exactly independent of protocol W; physical states differ.
            offstart=moments(model,v,0.,eps,channels=(),coupled=False,hopping=False)
            offend=moments(model,off,T,eps,channels=(),coupled=False,hopping=False)
            full=temporal_difference(finish,baseline,start,initial);offdelta=temporal_difference(offend,offbase,offstart,initial)
            # Initial density/LEFT never change under the full field-only W.
            expectedwork=.23*initial['Pi'][inc]/p['cavity_inertia']+.23**2/(2*p['cavity_inertia'])
            cross=model.overlap(base,end,eps0,eps)
            protocols[str(inc)]=dict(out=out,initial=start,end=finish,full_temporal=full,decoupled_temporal=offdelta,full_minus_decoupled={key:(np.asarray(full[key])-np.asarray(offdelta[key])).tolist() for key in full},exact_physical_kick_work=start['physical_energy']-initial['physical_energy'],expected_Weyl_work=expectedwork,physical_kick_work_error=abs(start['physical_energy']-initial['physical_energy']-expectedwork),physical_cross_protocol_overlap=[cross.real.item(),cross.imag.item()],WRONG_coefficient_overlap=[np.vdot(base,end).real.item(),np.vdot(base,end).imag.item()],initial_matter_density_difference=float(np.max(abs(np.asarray(start['density'])-initial['density']))))
            raw[f'kick{inc}_end']=end
        case: dict[str,Any]=dict(number=number,dimension=model.dimension,source_weights=weights,source_weight_sum=None if weights is None else sum(weights),initial=initial,baseline=baseline,field_decoupled_baseline=offbase,protocols=protocols,integration=errors)
        case['t0_same_physical_H_action_error']=float(np.linalg.norm(model.physical_energy(v,0.,eps0)-model.fixed.apply(v)))
        case['hopping_disabled_retained_convection_vertex_norm']=float(np.linalg.norm(model.vertex(v,0,0.,eps0,hopping=False,coupled=True)))
        case['field_decoupled_own_source_vertex_norm']=float(np.linalg.norm(model.vertex(v,0,0.,eps0,hopping=False,coupled=False)))
        if number==5 and cut==1:
            finebase,efb=propagate(model,v,0.,T,eps0,dt/2);finepulse,efp=propagate(model,v,0.,T,model.impulse(0),dt/2)
            mb=moments(model,finebase,T,eps0,channels=(0,6));mp=moments(model,finepulse,T,model.impulse(0),channels=(0,6))
            fine=temporal_difference(mp,mb,protocols['0']['initial'],initial)
            coarse=protocols['0']['full_temporal']
            case['physical_dt_comparison']=dict(coarse_dt=dt,fine_dt=dt/2,Pi_out6_change=float(fine['Pi'][6]-coarse['Pi'][6]),current_out6_change=float((mp['physical_current']['6']-mb['physical_current']['6'])-(protocols['0']['end']['physical_current']['6']-baseline['physical_current']['6'])),density_max_change=float(np.max(abs(np.asarray(fine['density'])-coarse['density']))),physical_energy_drift_coarse=baseline['physical_energy']-initial['physical_energy'],physical_energy_drift_fine=mb['physical_energy']-initial['physical_energy'],integration=[efb,efp])
            raw.update(dt_fine_baseline=finebase,dt_fine_kick0=finepulse)
        if number!=4 and cut==1:
            records={}
            for inc,out in [(0,6),(2,4)]:
                Ji=-model.vertex(v,inc,0.,eps0);Ei=model.field(v,inc,0.,eps0)
                jleg,ej=propagate(model,Ji,0.,T,eps0,dt);eleg,ee=propagate(model,Ei,0.,T,eps0,dt)
                Jo=-model.vertex(base,out,T,eps0);Eo=model.field(base,out,T,eps0)
                Cj=np.vdot(Jo,jleg);Ce=np.vdot(Eo,eleg)
                spin=np.einsum('ij,fjrma->firma',model.matter.g['L'][2].toarray(),base.reshape(model.shape),optimize=True).reshape(-1)
                Cs=np.vdot(spin,jleg);K=np.vdot(v,model.vertex(v,inc,0.,eps0,second=out))
                records[str(inc)]=dict(out=out,current_Wightman=[Cj.real.item(),Cj.imag.item()],current_reverse=[Cj.real.item(),-Cj.imag.item()],current_retarded=2*Cj.imag.item(),electric_Wightman=[Ce.real.item(),Ce.imag.item()],electric_reverse=[Ce.real.item(),-Ce.imag.item()],electric_retarded=2*Ce.imag.item(),spin_current=[Cs.real.item(),Cs.imag.item()],physical_contact=[K.real.item(),K.imag.item()],absolute_times=[0.,T],integration=[ej,ee])
                raw[f'J{inc}_leg']=jleg;raw[f'E{inc}_leg']=eleg
            case['two_leg']=records
        result['cases'][name]=case
        np.savez_compressed(ROOT/(P+label+'_'+name+'_raw.npz'),**raw)
        (ROOT/(P+label+'_result.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(name,json.dumps(dict(dimension=model.dimension,Pi06=protocols['0']['full_minus_decoupled']['Pi'][6],Pi24=protocols['2']['full_minus_decoupled']['Pi'][4],OFF06=protocols['0']['decoupled_temporal']['Pi'][6],physical_energy_drift=baseline['physical_energy']-initial['physical_energy'])),flush=True)
        if number!=4:del model
        del raw;gc.collect()
    result['charged_minus_neutral']={name:{axis:{key:(np.asarray(result['cases'][name]['protocols'][axis]['full_minus_decoupled'][key])-np.asarray(result['cases']['neutral']['protocols'][axis]['full_minus_decoupled'][key])).tolist() for key in result['cases'][name]['protocols'][axis]['full_minus_decoupled']} for axis in ['0','2']} for name in ['addition','removal']}
    (ROOT/(P+label+'_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(label,'complete',flush=True)


if __name__=='__main__':main()
