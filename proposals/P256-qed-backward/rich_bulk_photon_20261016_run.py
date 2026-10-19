"""Fresh V64 Monte Carlo and exact all-hop source-image variational spectra."""
import json
from pathlib import Path
import sys
import time
import numpy as np
from numba import njit
import rich_bulk_photon_20261016_state as s
ROOT=Path(__file__).resolve().parent
P='rich_bulk_photon_20261016_'


def save(name,value):
    (ROOT/(P+name+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


@njit(cache=True)
def measure(us,ds,Es,F,G,g,a,b,epsilon,floor,coeff,density):
    n=len(us);raw=np.zeros((n,8));vectors=np.empty((n,6),np.complex128)
    images=np.empty((n,3),np.complex128);contacts=np.empty((n,2,2),np.complex128)
    hp=np.empty(n,np.complex128)
    for k in range(n):
        u=us[k].copy();d=ds[k].copy();E=Es[k].copy()
        v,h,c,w,stable,hb=s.evaluate(u,d,E,F,G,g,a,b,epsilon,floor,coeff,density)
        root=np.sqrt(w);vectors[k]=v/root;images[k]=hb/root;contacts[k]=c/w;hp[k]=h/root
        raw[k,0]=abs(v[0])**2/w;raw[k,1]=(v[0].conjugate()*h).real/w;raw[k,2]=abs(h)**2/w
        occ=np.zeros(64,np.int64)
        for i in range(32):occ[u[i]]+=1;occ[d[i]]+=1
        D=0
        for i in range(64):D+=int(occ[i]==2)
        raw[k,3]=D*raw[k,0];raw[k,4]=np.sum(E*E)*raw[k,0]
        raw[k,5]=abs(h)**2/w if not stable else 0.
        raw[k,6]=1. if not stable else 0.;raw[k,7]=D
    return raw,vectors,images,contacts,hp


def estimate(raw,blocks=30):
    norm=raw[:,0].mean();value=raw.mean(axis=0)/norm
    energy=value[1];variance=value[2]-energy**2
    chunks=np.array_split(raw,blocks)
    blockmeans=np.array([x[:,1].sum()/x[:,0].sum() for x in chunks])
    blockvars=np.array([x[:,2].sum()/x[:,0].sum()-(x[:,1].sum()/x[:,0].sum())**2 for x in chunks])
    w=raw[:,0]
    return dict(energy=float(energy),energy_per_site=float(energy/64),energy_blockSE=float(blockmeans.std(ddof=1)/np.sqrt(blocks)),full_H_variance=float(variance),variance_blockSE=float(blockvars.std(ddof=1)/np.sqrt(blocks)),residual=float(np.sqrt(max(0,variance))),doublons=float(value[3]),field_E2=float(value[4]),importance_ESS=float(w.sum()**2/(w@w)),samples=len(raw),unstable_inverse_samples=int(raw[:,6].sum()),unstable_H2_contribution=float(value[5]),doublon_observed_range=[int(raw[:,7].min()),int(raw[:,7].max())],block_means=blockmeans.tolist(),uncertainty='Exploratory consecutive-block SE; importance ESS is NOT autocorrelation ESS; no mixing/convergence proof')


def spectral(v,hb,hp,contacts,q,label,energy,threshold,broadening,frequency):
    # B includes both genuinely distinct +/- Fourier sectors and fermionic density.
    B=np.column_stack([v[:,1:3],v[:,5],v[:,1:3].conj(),v[:,5].conj()])
    HB=np.column_stack([hb,hb.conj()]);psi=v[:,0];norm=float(np.mean(abs(psi)**2))
    mean=np.mean(psi.conj()[:,None]*B,axis=0)/norm
    B=B-psi[:,None]*mean;HB=HB-hp[:,None]*mean
    G=B.conj().T@B/len(B)/norm
    Kraw=B.conj().T@HB/len(B)/norm-energy*G
    defect=float(np.linalg.norm(Kraw-Kraw.conj().T)/max(np.linalg.norm(Kraw),1e-300))
    K=(Kraw+Kraw.conj().T)/2
    eg,ug=np.linalg.eigh(G);keep=eg>threshold*eg.max()
    whitening=ug[:,keep]/np.sqrt(eg[keep])[None,:]
    omega,U=np.linalg.eigh(whitening.conj().T@K@whitening);C=whitening@U
    mode=B@C;Hmode=HB@C
    residuals=np.mean(abs(Hmode-mode*(omega+energy)[None,:])**2,axis=0)/norm
    # Current at -q is -conj J(q), unlike the electric/density source.
    J=np.column_stack([v[:,3:5],-v[:,3:5].conj()])
    jm=np.mean(psi.conj()[:,None]*J,axis=0)/norm;J=J-psi[:,None]*jm
    gramJ=J.conj().T@J/len(J)/norm
    overlap=B.conj().T@J/len(J)/norm;A=C.conj().T@overlap
    projected=A.conj().T@A;leakage=gramJ-projected
    fieldoverlap=B.conj().T@v[:,1:3]/len(B)/norm;fieldA=C.conj().T@fieldoverlap
    Wplus=np.array([np.outer(row[2:4].conj(),row[2:4]) for row in A])
    Wminus=np.array([np.outer(row[:2].conj(),row[:2]) for row in A])
    chi=np.empty((len(frequency),2,2),complex)
    for i,w in enumerate(frequency):
        z=w+1j*broadening
        chi[i]=np.sum(Wplus/(omega-z)[:,None,None]+Wminus.transpose(0,2,1)/(omega+z)[:,None,None],axis=0)
    contact=contacts.mean(axis=0)/norm
    twist=contact[None,:,:]-chi
    # Equal-time q/-q mixing is retained, never identified as scalar weights.
    np.savez_compressed(ROOT/(P+label+'_response.npz'),q=q,G=G,Kraw=Kraw,K=K,gram_eigenvalues=eg,whitening=whitening,C=C,omega=omega,mode_residual_squared=residuals,currentGram=gramJ,currentOverlap=overlap,currentAmplitudes=A,currentLeakage=leakage,fieldAmplitudes=fieldA,Wplus=Wplus,Wminus=Wminus,contact=contact,frequency=frequency,chi=chi,twist=twist,mean=mean,current_mean=jm)
    return dict(q=np.asarray(q).tolist(),basis='F1+,F2+,rho+,F1-,F2-,rho-; exact fullH images',gram_rank=int(keep.sum()),gram_eigenvalues=eg.tolist(),shifted_variational_energies=omega.tolist(),fullH_mode_residual_squared=residuals.tolist(),shiftedH_antihermitian_relative=defect,field_weights=abs(fieldA).tolist() if False else (abs(fieldA)**2).tolist(),current_norms=gramJ.diagonal().real.tolist(),current_outside_basis_norm_squared=leakage.diagonal().real.tolist(),current_outside_basis_fraction=(leakage.diagonal().real/gramJ.diagonal().real).tolist(),contact_real=contact.real.tolist(),contact_imag=contact.imag.tolist(),mixed_weight_plus_minus_difference=float(np.linalg.norm(Wplus-Wminus)),negative_shifted_modes=int(np.sum(omega<0)),interpretation='Finite nonstationary variational source-space spectral diagnostic; residual and current leakage retained; not stationary Lehmann poles or a measured photon')


def main():
    input_name=sys.argv[1] if len(sys.argv)>1 else P+'input.json'
    cfg=json.loads((ROOT/input_name).read_text());sc=cfg['state'];guide=cfg['guide'];mc=cfg['sampling'];sp=cfg['spectral']
    run_label=cfg.get('run_label','')
    frequency=np.linspace(0,sp['frequency_max'],sp['frequency_n']);results={};start=time.time()
    for family in sc['families']:
        F,G,metadata=s.kernel(sc['delta'],family)
        np.savez_compressed(ROOT/(P+family+'_state.npz'),F=F,G=G,positions=s.POSITIONS,arcs=s.ARCS,incidence=s.INCIDENCE,loops=s.LOOPS)
        coeff,density,pol=s.sources(cfg['momenta'][0])
        u,d,E,acc,prop=s.sample(mc['seeds'][0]+9,64,2000,40,F,G,sc['g'],sc['a'],sc['b'],guide['epsilon'],guide['floor'])
        raw,v,hb,c,hp=measure(u,d,E,F,G,sc['g'],sc['a'],sc['b'],guide['epsilon'],guide['floor'],coeff,density)
        quality=estimate(raw,8)
        checkpoint=dict(family=family,metadata=metadata,actual_V64=True,quality=quality,accepted=acc.tolist(),proposed=prop.tolist(),maxGauss=max(int(np.max(abs(s.INCIDENCE@ee+np.bincount(np.r_[uu,dd],minlength=64)-1))) for uu,dd,ee in zip(u,d,E)),elapsed=time.time()-start)
        save(family+'_checkpoint',checkpoint);np.savez_compressed(ROOT/(P+family+'_checkpoint_raw.npz'),up=u,down=d,E=E,raw=raw,vectors=v,Hbasis=hb,Hpsi=hp,contact=c)
        print(json.dumps(dict(checkpoint=checkpoint)),flush=True)
        allu=[];alld=[];allE=[];chains=[]
        for seed in mc['seeds']:
            u,d,E,acc,prop=s.sample(seed,mc['n'],mc['burn'],mc['thin'],F,G,sc['g'],sc['a'],sc['b'],guide['epsilon'],guide['floor'])
            raw,v,hb,c,hp=measure(u,d,E,F,G,sc['g'],sc['a'],sc['b'],guide['epsilon'],guide['floor'],coeff,density)
            quality=estimate(raw,mc['blocks']);quality.update(seed=seed,accepted=acc.tolist(),proposed=prop.tolist())
            chains.append(quality);allu.append(u);alld.append(d);allE.append(E)
            print(json.dumps(dict(family=family,chain=quality)),flush=True)
        u=np.concatenate(allu);d=np.concatenate(alld);E=np.concatenate(allE)
        maxGauss=max(int(np.max(abs(s.INCIDENCE@ee+np.bincount(np.r_[uu,dd],minlength=64)-1))) for uu,dd,ee in zip(u,d,E))
        responses=[]
        for iq,q in enumerate(cfg['momenta']):
            coeff,density,pol=s.sources(q)
            raw,v,hb,c,hp=measure(u,d,E,F,G,sc['g'],sc['a'],sc['b'],guide['epsilon'],guide['floor'],coeff,density)
            quality=estimate(raw,mc['blocks']);label=family+'_q'+str(iq)
            np.savez_compressed(ROOT/(P+label+'_raw.npz'),up=u,down=d,E=E,raw=raw,vectors=v,Hbasis=hb,Hpsi=hp,contact=c,coeff=coeff,density=density,polarizations=pol)
            responses.append(spectral(v,hb,hp,c,q,label,quality['energy'],sp['rank_relative_threshold'],sp['broadening'],frequency))
        results[family]=dict(metadata=metadata,quality=quality,chains=chains,maxGauss=maxGauss,responses=responses)
        save(family+'_result',results[family]);save(run_label+'result',dict(results=results,elapsed_seconds=time.time()-start))
        print(json.dumps(dict(family=family,quality=quality,responses=responses)),flush=True)
    selected=min(results,key=lambda k:results[k]['quality']['energy'])
    save(run_label+'selected_state',dict(family=selected,state=sc,state_file=P+selected+'_state.npz',selection='Lower measured energy within this run on same H, no stationarity/ground claim',quality=results[selected]['quality']))
    print(json.dumps(dict(selected=selected,natural_science_completion=True,elapsed=time.time()-start)),flush=True)


if __name__=='__main__':main()
