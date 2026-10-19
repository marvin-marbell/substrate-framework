"""Fresh material two-pair-kernel evolution; preserve first closure as separate data."""
import json
from pathlib import Path
import numpy as np
import rich_bulk_photon_20261016_state as base
import rich_flux_phase_photon_20261016_state as flux
import rich_pi_flux_dynamics_20261016_state as s
import rich_pi_flux_dynamics_20261016_pair_enriched_state as pair
ROOT=Path(__file__).resolve().parent
P='rich_pi_flux_dynamics_20261016_pair_enriched_'


def save(name,data):
    (ROOT/(P+name+'.json')).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def quad(states,op):return np.einsum('trc,rs,tsc->tc',states.conj(),op,states)


def analyze(label,V,HV,JV,EN,Es,occ,times,geo):
    pr=s.fit_projection(V,HV,1e-7);states=s.evolve(pr,times)
    W=pr['W'];norm=pr['norm'];h=pr['h'];C=np.einsum('ar,trc->tac',W,states)
    def op(images):return W.conj().T@(V.conj().T@images/len(V)/norm)@W
    field=[];current=[];projected=[];full=[];layer=[]
    values=Es@geo[5].T/16
    for p in range(12):
        e=op(values[:,p,None]*V);j=op(JV[:,p])
        field.append(quad(states,e));current.append(quad(states,j))
        projected.append(quad(states,1j*(h@e-e@h)));full.append(quad(states,-j/16))
    for x in range(4):
        en=op(EN[:,2*x]+EN[:,2*x+1]);en=(en+en.conj().T)/2
        layer.append(quad(states,en))
    field=np.array(field).transpose(1,2,0);current=np.array(current).transpose(1,2,0)
    layer=np.array(layer).transpose(1,2,0);projected=np.array(projected).transpose(1,2,0);full=np.array(full).transpose(1,2,0)
    D=(occ==2).astype(float);matter=[]
    for val in [D.sum(axis=1),(Es*Es).sum(axis=1)/32,*[(occ-1)@d for d in geo[4]],*[D@d for d in geo[4]]]:matter.append(quad(states,op(val[:,None]*V)))
    matter=np.array(matter).transpose(1,2,0)
    residual=[];ess=[];metric=[];pair_weight=[]
    modes=np.column_stack([values[:,8:10],values[:,8:10].conj()])
    for it in range(len(times)):
        psi=V@C[it];outside=HV@C[it]-V@(W@(h@states[it]))
        prob=abs(psi)**2;residual.append((abs(outside)**2).sum(axis=0)/prob.sum(axis=0))
        ess.append(prob.sum(axis=0)**2/(prob*prob).sum(axis=0))
        center=modes-prob[:,0]@modes/prob[:,0].sum()
        metric.append((center.conj().T*prob[:,0])@center/prob[:,0].sum())
        # Norm of coherent second component is NOT a probability (families overlap).
        component=V[:,38:]@C[it,38:]
        pair_weight.append((abs(component)**2).sum(axis=0)/prob.sum(axis=0))
    residual=np.array(residual);metric=np.array(metric);pair_weight=np.array(pair_weight)
    energy=quad(states,h).real
    df=field[:,1:]-field[:,0,None];dj=current[:,1:]-current[:,0,None]
    frequency=2*np.pi*np.fft.fftfreq(len(times),times[1]-times[0])
    spectrumE=np.fft.fft(np.concatenate([df[:,:,8:],df[:,:,8:].conj()],axis=2),axis=0)*(times[1]-times[0])
    spectrumJ=np.fft.fft(np.concatenate([dj[:,:,8:],dj[:,:,8:].conj()],axis=2),axis=0)*(times[1]-times[0])
    np.savez_compressed(ROOT/(P+label+'_dynamics.npz'),times=times,states=states,coefficients=C,field=field,current=current,field_response=df,current_response=dj,layer_total_energy=layer,matter=matter,moving_metric=metric,full_current_Ehrenfest=full,projected_Ehrenfest=projected,projection_residual2=residual,importance_ESS=ess,coherent_pair_component_relative_norm=pair_weight,projected_energy=energy,two_sided_angular_frequency=frequency,two_sided_field_spectrum=spectrumE,two_sided_current_spectrum=spectrumJ,**pr)
    peak=lambda a:float(np.max(abs(a)))
    raw=pr['raw_orthogonal_H']
    represented=V@(W@pr['initial']);initial_error=np.sum(abs(V[:,[0,36,37]]-represented)**2,axis=0)/np.sum(abs(V[:,[0,36,37]])**2,axis=0)
    result=dict(samples=len(V),rank=pr['rank'],initial_representation_relative_squared=initial_error.tolist(),wholeH_sampling_antihermitian_relative=float(np.linalg.norm(raw-raw.conj().T)/np.linalg.norm(raw)),initial_energy=energy[0].tolist(),pulse_total_work=(energy[0,1:]-energy[0,0]).tolist(),energy_drift_max=peak(energy-energy[0]),norm_drift_max=peak((abs(states)**2).sum(axis=1)-1),projection_residual_RMS_t0=np.sqrt(residual[0]).tolist(),projection_residual_RMS_max=np.sqrt(residual.max(axis=0)).tolist(),field_response_peak=np.max(abs(df),axis=0).tolist(),current_response_peak=np.max(abs(dj),axis=0).tolist(),initial_field_response=df[0].real.tolist(),initial_current_response=dj[0].real.tolist(),final_field_response=df[-1].real.tolist(),final_current_response=dj[-1].real.tolist(),initial_owned_layer_excitation_energy=(layer[0,1:]-layer[0,0,None]).real.tolist(),final_owned_layer_excitation_energy=(layer[-1,1:]-layer[-1,0,None]).real.tolist(),initial_matter_response=(matter[0,1:]-matter[0,0,None]).real.tolist(),final_matter_response=(matter[-1,1:]-matter[-1,0,None]).real.tolist(),background_field_change=peak(field[:,0]-field[0,0]),background_matter_change=peak(matter[:,0]-matter[0,0]),moving_metric_change=peak(metric-metric[0]),full_vs_projected_Ehrenfest_max=peak(full-projected),importance_ESS_min=np.min(ess,axis=0).tolist(),coherent_pair_component_relative_norm_final=pair_weight[-1].tolist(),layer_energy_sum_error=peak(layer.sum(axis=2).real-energy))
    save(label+'_result',result)
    return result


def main():
    cfg=json.loads((ROOT/'rich_pi_flux_dynamics_20261016_input.json').read_text());geo=s.geometry();times=np.linspace(0,1.5,151)
    F,G,codes,meta=flux.kernel(True,'pair',.8);Fpair,unusedG,unusedcodes,pairmeta=flux.kernel(True,'pair',1.2)
    np.savez_compressed(ROOT/(P+'state.npz'),F=F,Fpair=Fpair,G=G,codes=codes,loops=geo[0],coeff=geo[3],density=geo[4],currents=geo[5])
    data=[];results=[]
    for k,seed in enumerate((9101671,9101687)):
        u,d,E,acc,prop=pair.sample(seed,768,18000,160,F,Fpair,G)
        V,HV,JV,EN,w=pair.measure(u,d,E,F,Fpair,G,codes,*geo,.35)
        occ=np.array([np.bincount(np.r_[uu,dd],minlength=64) for uu,dd in zip(u,d)])
        gauss=max(int(abs(base.INCIDENCE@ee+nn-1).max()) for ee,nn in zip(E,occ))
        np.savez_compressed(ROOT/(P+'chain'+str(k)+'_raw.npz'),up=u,down=d,E=E,occupations=occ,V=V,HV=HV,JV=JV,local_energy=EN,guide_weight=w,accepted=acc,proposed=prop)
        out=analyze('chain'+str(k),V,HV,JV,EN,E,occ,times,geo);out.update(seed=seed,gauss_max=gauss)
        results.append(out);data.append((V,HV,JV,EN,E,occ))
        print(json.dumps(dict(chain=k,rank=out['rank'],work=out['pulse_total_work'],residual=out['projection_residual_RMS_t0'])),flush=True)
    combined=[np.concatenate([d[i] for d in data]) for i in range(6)]
    joined=analyze('joined',*combined,times,geo)
    uncertainty={}
    a=np.load(ROOT/(P+'chain0_dynamics.npz'));b=np.load(ROOT/(P+'chain1_dynamics.npz'))
    for key in ('field_response','current_response','layer_total_energy','matter','projection_residual2'):
        error=abs(a[key]-b[key])/2
        uncertainty[key+'_max_two_chain_SE']=float(error.max())
        np.savez_compressed(ROOT/(P+key+'_two_chain_uncertainty.npz'),chain0=a[key],chain1=b[key],two_chain_SE=error)
    save('result',dict(chains=results,joined=joined,uncertainty=uncertainty,pair_state_parameter=1.2,natural_scientific_completion=True,meaning='Material coherent full64 pair-kernel extension,76 complex state coefficients, same H and pure field pulses. Finite sampled projection, not exact full-H wave. Original38 attempt preserved.'))
    print(json.dumps(dict(natural_scientific_completion=True,rank=joined['rank'],uncertainty=uncertainty)),flush=True)


if __name__=='__main__':main()
