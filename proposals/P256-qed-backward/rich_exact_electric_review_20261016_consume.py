"""Fresh bounded OLD48 CAR8/global-frame coupled electric case, not Common/47."""
from pathlib import Path
import gc
import json
import resource
import time
from typing import Any
import numpy as np
from scipy.linalg import expm
import rich_exact_electric_picture_20261016_model as picture
import rich_exact_electric_picture_20261016_fields as electric
import rich_spatial_frame_photon_20261016_fields as primitive
from rich_exact_electric_review_20261016_tools import neutral_state,source,arnoldi,propagate,observe,temporal,field_overlap_matrix
ROOT=Path(__file__).resolve().parent
P='rich_exact_electric_review_20261016_'


def save(result):
    (ROOT/(P+'result.json')).write_text(json.dumps(result,indent=2)+'\n')


def ladder_check(model,state):
    target=primitive.basis(2);lookup={label:i for i,label in enumerate(target)};e=np.zeros((len(target),model.nfield),complex);q=np.zeros((8,len(target),model.nfield),complex);p=q.copy()
    for column,label in enumerate(model.basis):
        e[lookup[label],column]=1
        for axis in range(8):
            upper=list(label);upper[axis]+=1;row=lookup[tuple(upper)];value=np.sqrt((label[axis]+1)/2);q[axis,row,column]+=value;p[axis,row,column]+=1j*value
            if label[axis]:
                lower=list(label);lower[axis]-=1;row=lookup[tuple(lower)];value=np.sqrt(label[axis]/2);q[axis,row,column]+=value;p[axis,row,column]-=1j*value
    expected=sum(op.conj().T@op for op in p)/(2*model.parameters['cavity_inertia']);field=(np.asarray(state).reshape(model.shape)@expected.T).reshape(-1)
    primitive_error=float(np.linalg.norm(q-model.images['q'])+np.linalg.norm(p-model.images['p']))
    if primitive_error>1e-12:raise AssertionError('Independent field ladder signs/coefficients differ from actual primitives')
    error=float(np.linalg.norm(field-model.free_energy(state)));timepoint=.04;tau=timepoint/model.parameters['cavity_inertia'];b=q+tau*p;full=b[0].conj().T@b[6];cut=(model.fixed.q[0]+tau*model.fixed.pi[0])@(model.fixed.q[6]+tau*model.fixed.pi[6]);wrong=float(np.linalg.norm(full-cut))
    if error>1e-10 or wrong<1e-3:raise AssertionError('Unbounded electric-before-cut product distinction failed')
    return dict(independent_field_ladder_error=primitive_error,independent_full_H0_action_error=error,full_product_minus_square_of_cut_b06=wrong,scope='Own eight-mode raising/lowering images into INITIALcut+1. H0 and b_l b_m are full products before projection, not cutPi/cuta squares.')


def matched_old_rays(model,state,base,offbase,initial,protocol):
    eps=model.impulse(0);zero=model.zeros;T=.04;legacy=(state.reshape(model.shape)@expm(1j*.23*model.fixed.q[0]).T).reshape(-1);p6=lambda v:float(np.vdot(v,model.field(v,6,T,zero)).real)
    initial_shift=p6(legacy)-p6(state)
    exactlegacy,record=propagate(model,legacy,zero,T,.01);offlegacy,offrecord=arnoldi(lambda v:model.apply(v,0.,zero,False,False),legacy,T,18)
    exactfull=(p6(exactlegacy)-p6(base))-initial_shift;exactoff=(p6(offlegacy)-p6(offbase))-initial_shift
    fixed=model.fixed;oldbase,oldb=arnoldi(fixed.apply,state,T,24);oldpulse,oldp=arnoldi(fixed.apply,legacy,T,24);oldoffbase,oldob=arnoldi(lambda v:fixed.apply(v,False,False),state,T,24);oldoffpulse,oldop=arnoldi(lambda v:fixed.apply(v,False,False),legacy,T,24)
    oldfull=(p6(oldpulse)-p6(oldbase))-initial_shift;oldoff=(p6(oldoffpulse)-p6(oldoffbase))-initial_shift
    fullW=protocol['full_minus_decoupled']['Pi'][6]
    overlap=field_overlap_matrix(model.basis,0,.23);physical=np.vdot(legacy,(state.reshape(model.shape)@overlap.T).reshape(-1))
    if abs(initial_shift)<1e-5:raise AssertionError('Wrong finitePaP-as-fullW case did not reach cross-mode detector')
    try:assert abs(initial_shift)<1e-12,'Finite exp(i eps P a P) is NOT full Weyl preparation: other-mode Pi changes initially'
    except AssertionError as error:rejected=str(error)
    np.savez_compressed(ROOT/(P+'matched_old_rays_raw.npz'),legacy_finite_PaP_initial=legacy,exactU0_same_oldray_end=exactlegacy,exactU0_same_oldray_OFF=offlegacy,old_fixedH_base=oldbase,old_fixedH_oldray_end=oldpulse,old_fixedH_OFF_base=oldoffbase,old_fixedH_OFF_oldray=oldoffpulse)
    return dict(initial_Pi6_finite_PaP_shift=initial_shift,initial_Pi6_fullW_shift=protocol['initial']['Pi'][6]-initial['Pi'][6],old_fixedH_same_oldray_temporal_full_minus_OFF=oldfull-oldoff,old_fixedH_OFF_temporal_echo=oldoff,exactU0_same_oldray_temporal_full_minus_OFF=exactfull-exactoff,exactU0_same_oldray_OFF_temporal_echo=exactoff,exactU0_DIFFERENT_fullWray_temporal_full_minus_OFF=fullW,initial_oldray_vs_fullW_physical_squared_overlap=float(abs(physical)**2),wrong_prep_assertion=rejected,integration=[record,offrecord,oldb,oldp,oldob,oldop],interpretation='Three FRESH cases: old fixed projectedH on finitePaP ray; exactU0 interaction-picture family on SAME finitePaP ray; DIFFERENT fullW ray. Do not credit the preparation change to free-solver improvement. Both coupled approximations remain finite Galerkin families.')


def case(model,state,name):
    zero=model.zeros;T=.04;dt=.01;initial=observe(model,state,0.,zero);base,record=propagate(model,state,zero,T,dt);baseline=observe(model,base,T,zero)
    off,offrecord=arnoldi(lambda v:model.apply(v,0.,zero,False,False),state,T,18);offbaseline=observe(model,off,T,zero,False,False,False);protocols={};raw=dict(input=state,baseline=base,field_decoupled_baseline=off);checks=[]
    for inc,out in ((0,6),(2,4)):
        eps=model.impulse(inc);end,evolve_record=propagate(model,state,eps,T,dt);start=observe(model,state,0.,eps);finish=observe(model,end,T,eps);offstart=observe(model,state,0.,eps,False,False,False);offend=observe(model,off,T,eps,False,False,False)
        full=temporal(finish,baseline,start,initial);offdelta=temporal(offend,offbaseline,offstart,initial);difference={key:(np.asarray(full[key])-offdelta[key]).tolist() for key in full}
        work=.23*initial['Pi'][inc]/model.parameters['cavity_inertia']+.23**2/(2*model.parameters['cavity_inertia']);work_error=abs(start['physical_H_energy']-initial['physical_H_energy']-work)
        cross_error=max(abs(start['Pi'][axis]-initial['Pi'][axis]) for axis in range(8) if axis!=inc);density_error=float(np.max(abs(np.asarray(start['density'])-initial['density'])))
        overlap=field_overlap_matrix(model.basis,inc,.23);frozen_overlap=electric.protocol_overlap(model.basis,model.basis,zero,eps);quadrature_error=float(np.linalg.norm(overlap-frozen_overlap));physical=np.vdot(base,(end.reshape(model.shape)@overlap.T).reshape(-1));coefficient=np.vdot(base,end)
        off_source=float(np.linalg.norm(model.vertex(state,out,T,eps,hopping=False,coupled=False)));off_H_protocol_difference=float(np.linalg.norm(model.apply(state,T,eps,False,False)-model.apply(state,T,zero,False,False)))
        if max(work_error,cross_error,density_error,quadrature_error,off_source,off_H_protocol_difference)>2e-10:raise AssertionError('Physical fullW/preparation/controlled-H comparison failed')
        if abs(offdelta['Pi'][out])>2e-10:raise AssertionError('Exact unbounded free electric OFF control developed cross-Pi echo')
        protocols[str(inc)]=dict(out=out,initial=start,end=finish,full_temporal=full,decoupled_temporal=offdelta,full_minus_decoupled=difference,kick_work_error=work_error,initial_cross_mode_Pi_error=cross_error,initial_matter_density_error=density_error,independent_Weyl_overlap_matrix_error=quadrature_error,physical_cross_protocol_overlap=[float(physical.real),float(physical.imag)],WRONG_coefficient_overlap=[float(coefficient.real),float(coefficient.imag)],physical_vs_coefficient_overlap_difference=float(abs(physical-coefficient)),OFF_own_current_vertex_norm=off_source,OFF_H_protocol_difference=off_H_protocol_difference,integration=evolve_record)
        raw['fullW'+str(inc)+'_end']=end
    result:dict[str,Any]=dict(number=model.matter.sector.number,dimension=model.dimension,shape=list(model.shape),initial=initial,baseline=baseline,field_decoupled_baseline=offbaseline,protocols=protocols,integration=[record,offrecord],ladder_before_cut=ladder_check(model,state),t0_same_microscopic_H_action_error=float(np.linalg.norm(model.physical_energy(state,0.,zero)-model.fixed.apply(state))))
    if name=='addition':result['matched_old_preparation']=matched_old_rays(model,state,base,off,initial,protocols['0'])
    np.savez_compressed(ROOT/(P+name+'_case_raw.npz'),**raw);print(json.dumps(dict(stage='OWN_GENUINE_COUPLED_CASE',name=name,Pi06=protocols['0']['full_minus_decoupled']['Pi'][6],Pi24=protocols['2']['full_minus_decoupled']['Pi'][4],physical_energy_drift=baseline['physical_H_energy']-initial['physical_H_energy'])),flush=True)
    return result,base


def main():
    started=time.time();p=dict(picture.DEFAULTS,field_total_cut=1);reference=picture.Model(4,p);neutral=neutral_state(reference)
    modes=reference.fixed.modes;transverse=float(np.max(abs(np.sum(modes[:,:3]*modes[:,3:6],axis=1))))
    if transverse>1e-12:raise AssertionError('Selected profiles are not actually transverse')
    result:dict[str,Any]=dict(kind='INDEPENDENT_EXACT_ELECTRIC_COUPLED_CAR8_GLOBAL_FRAME_CASE',parameters=p,T=.04,dt=.01,modes=modes.tolist(),transversality_error=transverse,scope='OWN fresh N4 neutral and N5 addition, field-initial-label cut1/IR3 If2, both0->6 and2->4 channels. N3/removal and cut2 COUPLED dynamics are NOT independently rerun. This is distinct from Common CAR64/native/47 and all old states.',cases={})
    result['cases']['neutral'],neutral_end=case(reference,neutral,'neutral');save(result)
    model=picture.Model(5,p);raw=source(model,reference,neutral);weight=float(np.linalg.norm(raw)**2)
    if weight<=0:raise AssertionError('Pinned addition source has no strength')
    own_vs_frozen=float(np.linalg.norm(raw-model.source(reference,neutral,0,True)));state=raw/np.sqrt(weight);result['source']=dict(component=0,create=True,raw_weight=weight,own_literal_CAR_Wilson_vs_pure_source_error=own_vs_frozen)
    # Same neutral ray, different charged source-field projection: no cut2 H build.
    b2=primitive.basis(2);raw2=source(model,reference,neutral,target_basis=b2);weight2=float(np.linalg.norm(raw2)**2);state2=raw2/np.sqrt(weight2);ix=[b2.index(label) for label in model.basis];overlap=np.vdot(state.reshape(model.shape),state2.reshape(model.nf,model.ng,model.nr,model.nm,len(b2))[...,ix]);result['initial_cut_ray_comparison']=dict(neutral_squared_overlap=1.,charged_squared_overlap=float(abs(overlap)**2),selected_source_weight_cut1=weight,selected_source_weight_cut2=weight2,scope='Own Wilson/CAR source into cut2 labels only, SAME neutral physical ray and angular/radial inputs. No cut2 coupled evolution, no field-convergence conclusion.');del raw2,state2;gc.collect()
    result['cases']['addition'],charged_end=case(model,state,'addition');save(result)
    late=source(model,reference,neutral_end,time=.04);late_pure=model.source(reference,neutral_end,0,True,.04);wrong=source(model,reference,neutral_end,time=0.);result['absolute_source']=dict(correct_own_vs_pure_error=float(np.linalg.norm(late-late_pure)),wrong_time0_source_action_error=float(np.linalg.norm(late-wrong)))
    # Actual absolute-time source derivative, via impulse->b shift at tau!=0.
    T=.04;tau=T/p['cavity_inertia'];step=.02;plus=model.impulse(6,step);minus=-plus;center=model.apply(charged_end,T);hp=model.apply(charged_end,T,plus);hm=model.apply(charged_end,T,minus);j=model.vertex(charged_end,6,T);contact=model.vertex(charged_end,6,T,second=6)
    result['absolute_current_contact']=dict(source_derivative_action_error=float(np.linalg.norm((hp-hm)/(2*step*tau)-j)),contact_derivative_action_error=float(np.linalg.norm((hp+hm-2*center)/(step*tau)**2-contact)),wrong_time0_current_action_error=float(np.linalg.norm(j-model.vertex(charged_end,6,0.))))
    pi=model.field(charged_end,6,T);hi=model.apply(charged_end,T);pidot=1j*(model.apply(pi,T)-model.field(hi,6,T));result['interacting_canonical_debt']=dict(Pi_dot_minus_actual_source_norm=float(np.linalg.norm(pidot+j)),actual_source_norm=float(np.linalg.norm(j)),meaning='Finite HI compression need not close physical interacting canonical algebra. Exact freeH0 is not an interacting closure proof.')
    result['charged_minus_neutral']={axis:{key:(np.asarray(result['cases']['addition']['protocols'][axis]['full_minus_decoupled'][key])-result['cases']['neutral']['protocols'][axis]['full_minus_decoupled'][key]).tolist() for key in ('Pi','a','density','r','LEFT')} for axis in ('0','2')}
    result['resource']=dict(elapsed_seconds=time.time()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save(result);print(json.dumps(dict(kind=result['kind'],charged_excess_Pi06=result['charged_minus_neutral']['0']['Pi'][6],charged_excess_Pi24=result['charged_minus_neutral']['2']['Pi'][4],resource=result['resource'])),flush=True)


if __name__=='__main__':main()
