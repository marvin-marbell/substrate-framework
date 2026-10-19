"""Actual finite-time two-channel projected-H evolution, including moving background."""
import json
from pathlib import Path
import time
import numpy as np
import rich_bulk_photon_20261016_state as base
import rich_flux_phase_photon_20261016_state as flux
import rich_pi_flux_dynamics_20261016_state as s
ROOT=Path(__file__).resolve().parent
P='rich_pi_flux_dynamics_20261016_'


def save(name,data):
    (ROOT/(P+name+'.json')).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def quadratic(states,matrix):
    return np.einsum('trc,rs,tsc->tc',states.conj(),matrix,states)


def matrix(V,OV,norm,W):
    raw=V.conj().T@OV/len(V)/norm
    projected=W.conj().T@raw@W
    return (projected+projected.conj().T)/2,projected


def diagnostics(us,ds,Es,F,G,codes,geo,epsilon):
    loops,kl,quad,coeff,density,currents=geo
    u=us[0].copy();d=ds[0].copy();E=Es[0].copy()
    v,h,j,en,w,f,st=s.evaluate(u,d,E,F,G,codes,*geo,epsilon)
    # Exact finite-state commutator, using full H on E*psi via legal neighbors.
    # Independent frozen operator consumer supplies the EOM on unpulsed Omega.
    vv,hh,cc,ww,ss,hb=flux.evaluate(u.copy(),d.copy(),E.copy(),F,G,codes,.45,.15,.08,.2,1e-12,coeff[:2],density[0])
    diagonal=E@coeff[:2].T/16
    comm=hb[:2]-diagonal*hh
    right=1j*vv[3:5]/16
    rng=np.random.default_rng(8231);nodalF=rng.normal(size=(64,64));nodalF[0]=0
    nodalG=rng.normal(size=(64,64))
    neel=np.nonzero(base.POSITIONS.sum(axis=1)%2==0)[0];other=np.setdiff1d(np.arange(64),neel)
    nv,nh,nj,nen,nw,nf,ns=s.evaluate(neel,other,np.zeros(192,np.int64),nodalF,nodalG,codes,*geo,epsilon)
    wrong=hb[:2]+53.259941*diagonal*vv[0]
    return dict(commutator_relative=float(np.linalg.norm(comm-right)/max(np.linalg.norm(right),1e-200)),wrong_HminusE_source_discrepancy=float(np.linalg.norm(wrong-comm)/max(np.linalg.norm(comm),1e-200)),wrong_source_energy_reference=-53.259941,node_psi_squared=float(abs(nv[0])**2),node_Hpsi_squared=float(abs(nh[0])**2),node_guide=float(nw),node_H2_over_guide=float(abs(nh[0])**2/nw))


def analyze(label,V,HV,JV,EN,WV,Es,occupations,geo,times):
    loops,kl,quad,coeff,density,currents=geo
    pr=s.fit_projection(V,HV,1e-7);states=s.evolve(pr,times)
    W=pr['W'];norm=pr['norm'];h=pr['h'];raw=pr['raw_orthogonal_H']
    C=np.einsum('ar,trc->tac',W,states)
    # All physical observations evaluated against full state coefficients, not inferred from dispersion.
    matrices={};field=[];current=[];eom=[];projected_derivative=[];j_asym=[]
    field_values=Es@currents.T/16
    for p in range(12):
        # Complex Fourier O is not Hermitian: preserve q and reconstruct -q by adjoint.
        Eop=W.conj().T@(V.conj().T@(field_values[:,p,None]*V)/len(V)/norm)@W
        Jop=W.conj().T@(V.conj().T@JV[:,p,:]/len(V)/norm)@W
        field.append(quadratic(states,Eop));current.append(quadratic(states,Jop))
        projected_derivative.append(quadratic(states,1j*(h@Eop-Eop@h)))
        eom.append(quadratic(states,-Jop/16))
        j_asym.append(float(np.linalg.norm(Jop-Jop.conj().T)))
        matrices['E'+str(p)]=Eop;matrices['J'+str(p)]=Jop
    field=np.asarray(field).transpose(1,2,0);current=np.asarray(current).transpose(1,2,0)
    eom=np.asarray(eom).transpose(1,2,0);projected_derivative=np.asarray(projected_derivative).transpose(1,2,0)
    layer=[]
    for x in range(4):
        em,rawem=matrix(V,EN[:,2*x,:]+EN[:,2*x+1,:],norm,W)
        layer.append(quadratic(states,em));matrices['energy'+str(x)]=em
    layer=np.array(layer).transpose(1,2,0)
    matter=[]
    D=(occupations==2).astype(float)
    for values in [D.sum(axis=1),Es.astype(float).__pow__(2).sum(axis=1)/32,*[(occupations-1)@dd for dd in density],*[D@dd for dd in density]]:
        op=W.conj().T@(V.conj().T@(values[:,None]*V)/len(V)/norm)@W
        matter.append(quadratic(states,op))
    matter=np.array(matter).transpose(1,2,0)
    wilson=[]
    for p in range(16):
        op,rawop=matrix(V,WV[:,p,:],norm,W)
        wilson.append(quadratic(states,op))
    wilson=np.array(wilson).transpose(1,2,0)
    np.savez_compressed(ROOT/(P+label+'_wilson_dynamics.npz'),times=times,wilson=wilson,loop_vectors=loops)
    # Moving Gram of four +/-q1 transverse observables in the evolving unpulsed state.
    modes=np.column_stack([field_values[:,8:10],field_values[:,8:10].conj()])
    metric=[]
    for it in range(len(times)):
        psi=V@C[it,:,0];prob=abs(psi)**2;z=prob.sum()
        mean=prob@modes/z;center=modes-mean
        metric.append((center.conj().T*prob)@center/z)
    metric=np.array(metric)
    residual=[];full_h2=[];importance_ess=[];projected_energy=quadratic(states,h).real
    for it in range(len(times)):
        psi=V@C[it];full=HV@C[it]
        projected=V@(W@(h@states[it]))
        residual.append(np.sum(abs(full-projected)**2,axis=0)/np.sum(abs(psi)**2,axis=0))
        full_h2.append(np.sum(abs(full)**2,axis=0)/np.sum(abs(psi)**2,axis=0))
        weights=abs(psi)**2;importance_ess.append(weights.sum(axis=0)**2/(weights**2).sum(axis=0))
    residual=np.array(residual);full_h2=np.array(full_h2)
    response_field=field[:,1:]-field[:,0,None,:]
    response_current=current[:,1:]-current[:,0,None,:]
    # Both temporal frequency signs; finite T/window peaks are not stationary poles.
    two_sided_field=np.concatenate([response_field[:,:,8:],response_field[:,:,8:].conj()],axis=2)
    two_sided_current=np.concatenate([response_current[:,:,8:],response_current[:,:,8:].conj()],axis=2)
    frequency=2*np.pi*np.fft.fftfreq(len(times),times[1]-times[0])
    spectrum_field=np.fft.fft(two_sided_field,axis=0)*(times[1]-times[0])
    spectrum_current=np.fft.fft(two_sided_current,axis=0)*(times[1]-times[0])
    exact_columns=V[:,[0,36,37]]
    represented=V@(W@pr['initial'])
    initial_error=np.sum(abs(exact_columns-represented)**2,axis=0)/np.sum(abs(exact_columns)**2,axis=0)
    np.savez_compressed(ROOT/(P+label+'_dynamics.npz'),times=times,states=states,coefficients=C,field=field,current=current,field_response=response_field,current_response=response_current,layer_total_energy=layer,matter=matter,moving_metric=metric,full_current_Ehrenfest=eom,projected_Ehrenfest=projected_derivative,projection_residual2=residual,full_H2=full_h2,importance_ESS=importance_ess,projected_energy=projected_energy,two_sided_angular_frequency=frequency,two_sided_field_spectrum=spectrum_field,two_sided_current_spectrum=spectrum_current,**matrices,**pr)
    def peak(a):return float(np.max(abs(a)))
    summary=dict(label=label,samples=len(V),rank=pr['rank'],gram_eigenvalues=pr['eigenvalues'].tolist(),wholeH_sampling_antihermitian_relative=float(np.linalg.norm(raw-raw.conj().T)/max(np.linalg.norm(raw),1e-200)),initial_representation_relative_squared=initial_error.tolist(),initial_energy=projected_energy[0].tolist(),pulse_total_work=(projected_energy[0,1:]-projected_energy[0,0]).tolist(),energy_drift_max=peak(projected_energy-projected_energy[0]),norm_drift_max=peak(np.sum(abs(states)**2,axis=1)-1),projection_residual_RMS_t0=np.sqrt(residual[0]).tolist(),projection_residual_RMS_max=np.sqrt(residual.max(axis=0)).tolist(),full_H_RMS_t0=np.sqrt(full_h2[0]).tolist(),field_response_peak_by_pulse_and_receiver=np.max(abs(response_field),axis=0).tolist(),current_response_peak_by_pulse_and_receiver=np.max(abs(response_current),axis=0).tolist(),initial_field_response=response_field[0].real.tolist(),initial_current_response=response_current[0].real.tolist(),final_field_response=response_field[-1].real.tolist(),final_current_response=response_current[-1].real.tolist(),initial_owned_layer_excitation_energy=(layer[0,1:]-layer[0,0,None]).real.tolist(),final_owned_layer_excitation_energy=(layer[-1,1:]-layer[-1,0,None]).real.tolist(),initial_matter_response=(matter[0,1:]-matter[0,0,None]).real.tolist(),final_matter_response=(matter[-1,1:]-matter[-1,0,None]).real.tolist(),background_field_change=peak(field[:,0]-field[0,0]),background_matter_change=peak(matter[:,0]-matter[0,0]),moving_metric_change=peak(metric-metric[0]),full_vs_projected_Ehrenfest_max=peak(eom-projected_derivative),importance_ESS_min=np.min(importance_ess,axis=0).tolist(),layer_energy_sum_error=peak(layer.sum(axis=2).real-projected_energy),meaning='Sampled Hermitian WHOLE-H Galerkin state evolution; background evolves. Full H action and currents outside family retained only as measured projection residual/Ehrenfest defect, not dynamically exact.')
    save(label+'_result',summary)
    return summary


def main():
    start=time.time();cfg=json.loads((ROOT/(P+'input.json')).read_text())
    F,G,codes,metadata=flux.kernel(True,'pair',.8)
    geo=s.geometry();all_data=[];results=[];all_occ=[];all_E=[]
    np.savez_compressed(ROOT/(P+'state.npz'),F=F,G=G,codes=codes,loops=geo[0],Kloops=geo[1],loop_quadratic=geo[2],coeff=geo[3],density=geo[4],currents=geo[5],positions=base.POSITIONS,arcs=base.ARCS)
    times=np.linspace(0,cfg['time_final'],cfg['time_points'])
    for chain,seed in enumerate(cfg['seeds']):
        u,d,E,accepted,proposed=base.sample(seed,cfg['n'],cfg['burn'],cfg['thin'],F,G,.45,.15,.08,.2,1e-12)
        V,HV,JV,EN,fs,w,stable=s.measure(u,d,E,F,G,codes,*geo,cfg['pulse_epsilon'])
        WV=s.wilson_images(u,d,E,F,G,codes,*geo,cfg['pulse_epsilon'])
        occupations=np.array([np.bincount(np.r_[uu,dd],minlength=64) for uu,dd in zip(u,d)])
        gauss=max(int(abs(base.INCIDENCE@ee+nn-1).max()) for ee,nn in zip(E,occupations))
        diag=diagnostics(u,d,E,F,G,codes,geo,cfg['pulse_epsilon'])
        np.savez_compressed(ROOT/(P+'chain'+str(chain)+'_raw.npz'),up=u,down=d,E=E,V=V,HV=HV,JV=JV,local_energy=EN,WV=WV,features=fs,guide_weight=w,stable_inverse=stable,occupations=occupations,accepted=accepted,proposed=proposed)
        result=analyze('chain'+str(chain),V,HV,JV,EN,WV,E,occupations,geo,times)
        result.update(seed=seed,gauss_max=gauss,accepted=accepted.tolist(),proposed=proposed.tolist(),diagnostics=diag)
        results.append(result);all_data.append((V,HV,JV,EN,WV));all_occ.append(occupations);all_E.append(E)
        print(json.dumps(dict(chain=chain,rank=result['rank'],work=result['pulse_total_work'],residual=result['projection_residual_RMS_t0'],gauss=gauss)),flush=True)
    combined=[np.concatenate([data[k] for data in all_data]) for k in range(5)]
    joined=analyze('joined',*combined,np.concatenate(all_E),np.concatenate(all_occ),geo,times)
    chains=[np.load(ROOT/(P+'chain'+str(k)+'_dynamics.npz')) for k in range(2)]
    se={}
    for name in ['field_response','current_response','layer_total_energy','matter','projection_residual2']:
        stacked=np.stack([c[name] for c in chains])
        err=abs(stacked[0]-stacked[1])/2
        se[name+'_max_two_chain_SE']=float(err.max())
        np.savez_compressed(ROOT/(P+name+'_two_chain_uncertainty.npz'),chain_values=stacked,two_chain_SE=err)
    save('result',dict(chains=results,joined=joined,uncertainty=se,state_metadata=metadata,elapsed_seconds=time.time()-start,natural_scientific_completion=True))
    print(json.dumps(dict(natural_scientific_completion=True,elapsed_seconds=time.time()-start,joined_rank=joined['rank'],uncertainty=se)),flush=True)


if __name__=='__main__':main()
