"""Consecutive-block bootstrap of owned raw physical measurements, never a new state."""
import json
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parent
P='rich_bulk_photon_20261016_'


def main():
    rng=np.random.default_rng(350901);results={}
    for family in ('singlet','af_dimer'):
        out=[]
        for iq in range(2):
            raw=np.load(ROOT/(P+family+'_q'+str(iq)+'_raw.npz'))
            v=raw['vectors'];hb=raw['Hbasis'];hp=raw['Hpsi'];contact=raw['contact'];n=len(v)
            block=np.array_split(np.arange(n),30);gaps=[];leaks=[];cnorm=[];ct=[];fnorm=[]
            for repeat in range(120):
                ids=np.concatenate([block[k] for k in rng.integers(0,30,30)])
                vv=v[ids];psi=vv[:,0];norm=np.mean(abs(psi)**2);energy=(np.mean(psi.conj()*hp[ids])/norm).real
                B=np.column_stack([vv[:,1:3],vv[:,5],vv[:,1:3].conj(),vv[:,5].conj()]);HB=np.column_stack([hb[ids],hb[ids].conj()])
                mean=np.mean(psi.conj()[:,None]*B,axis=0)/norm;B-=psi[:,None]*mean;HB-=hp[ids,None]*mean
                G=B.conj().T@B/len(ids)/norm;K=B.conj().T@HB/len(ids)/norm-energy*G;K=(K+K.conj().T)/2
                eg,ug=np.linalg.eigh(G);W=ug/np.sqrt(eg)[None,:];gap,U=np.linalg.eigh(W.conj().T@K@W);C=W@U
                J=vv[:,3:5].copy();jm=np.mean(psi.conj()[:,None]*J,axis=0)/norm;J-=psi[:,None]*jm
                JG=J.conj().T@J/len(ids)/norm;A=C.conj().T@(B.conj().T@J/len(ids)/norm)
                leaks.append((1-np.real(np.diag(A.conj().T@A))/np.real(np.diag(JG))));gaps.append(gap)
                cnorm.append(np.real(np.diag(JG)));ct.append(np.real(np.diag(contact[ids].mean(axis=0)/norm)))
                fnorm.append(np.mean(abs(vv[:,1:3])**2,axis=0)/norm)
            samples:dict[str,Any]=dict(sorted_shifted_Ritz=np.array(gaps),current_leakage=np.array(leaks),current_norm=np.array(cnorm),contact=np.array(ct),field_norm=np.array(fnorm))
            summary={key:dict(bootstrap_std=value.std(axis=0,ddof=1).tolist(),percentile_2p5=np.percentile(value,2.5,axis=0).tolist(),percentile_97p5=np.percentile(value,97.5,axis=0).tolist()) for key,value in samples.items()}
            np.savez_compressed(ROOT/(P+family+'_q'+str(iq)+'_bootstrap.npz'),**samples)
            out.append(summary)
        results[family]=out
    result=dict(seed=350901,resamples=120,consecutive_blocks=30,interpretation='Exploratory block bootstrap, sorted Ritz levels can exchange identity; no autocorrelation/mixing or stationary-pole proof; bootstrap reuses owned measured arrays only, never wavefunction evolution/input',results=results)
    (ROOT/(P+'response_uncertainty.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({f:[q['sorted_shifted_Ritz']['bootstrap_std'] for q in v] for f,v in results.items()}),flush=True)


if __name__=='__main__':main()
