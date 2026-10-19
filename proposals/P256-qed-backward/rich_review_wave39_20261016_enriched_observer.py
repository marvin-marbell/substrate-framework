"""New76-family EOM, evolving metric and full-vs-bare energy from OWN raw amplitudes."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
P='rich_review_wave39_20261016_'

def main():
    with np.load(ROOT/(P+'enriched_raw.npz')) as z:V=z['V'];HV=z['HV']
    N=len(V);normal=np.mean(abs(V[:,0])**2);G=V.conj().T@V/N/normal;hraw=V.conj().T@HV/N/normal
    eg,U=np.linalg.eigh(G);keep=eg>eg.max()*1e-8;W=U[:,keep]/np.sqrt(eg[keep]);raw=W.conj().T@hraw@W;H=(raw+raw.conj().T)/2
    # Feature multipliers same for both pair kernels; recover from either nonzero amplitude.
    first=abs(V[:,0])>abs(V[:,38]);offset=np.where(first,0,38);denom=V[np.arange(N),offset]
    def f(k):
        num=V[np.arange(N),offset+k];out=np.zeros(N,complex);np.divide(num,denom,out=out,where=abs(denom)>1e-100);return out
    field=np.column_stack([f(k)+1j*f(k+1) for k in (4,6,12,14)])/16
    diagonal=(32*f(1)+2*f(2)).real
    eig,Q=np.linalg.eigh(H);initial=W.conj().T@G[:,[0,36,37]];initial/=np.linalg.norm(initial,axis=0)
    records=[];metrics=[];fields=[];eoms=[]
    for t in (0.,.04,.08):
        c=Q@(np.exp(-1j*t*eig)[:,None]*(Q.conj().T@initial));psi=V@W@c;hp=HV@W@c;ph=V@W@H@c;norm=np.sum(abs(psi)**2,axis=0)
        metric=np.einsum('ns,nk,nl,ns->skl',psi.conj(),field.conj(),field,psi)/norm[:,None,None];metrics.append(metric)
        obs=np.einsum('ns,nk,ns->sk',psi.conj(),field,psi)/norm[:,None];fields.append(obs)
        eom=lambda h:1j*(np.einsum('ns,nk,ns->sk',h.conj(),field,psi)-np.einsum('ns,nk,ns->sk',psi.conj(),field,h))/norm[:,None]
        eoms.append(eom(hp));bare=np.sum(abs(psi)**2*diagonal[:,None],axis=0)/norm;energy=np.real(np.sum(psi.conj()*hp,axis=0)/norm)
        records.append(dict(time=t,full_energy=energy.tolist(),bare_onsite_electric=bare.tolist(),hopping=(energy-bare).tolist(),physical_full_vs_projected_EOM=np.linalg.norm(eom(hp)-eom(ph),axis=1).tolist()))
    fields=np.array(fields);metrics=np.array(metrics)
    result=dict(rank=int(keep.sum()),records=records,moving_physical_metric_change=float(np.linalg.norm(metrics[-1]-metrics[0])),pulse_minus_evolving_background_change=float(np.linalg.norm((fields[-1,1:]-fields[-1,0])-(fields[0,1:]-fields[0,0]))),both_pair_amplitudes_zero=int(np.sum(abs(denom)<=1e-100)),scope='Weak full-H EOM versus projected trajectory, not exact dynamics. Physical field=E/16, qminus conjugate. Derived only from OWN fresh76 raw coefficient/H-image arrays; no producer output.')
    np.savez_compressed(ROOT/(P+'enriched_observer_raw.npz'),Eplus=fields,Eminus=fields.conj(),metric=metrics,weak_fullEOM=np.array(eoms))
    (ROOT/(P+'enriched_observer_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    assert result['moving_physical_metric_change']>1e-10 and result['pulse_minus_evolving_background_change']>1e-8
if __name__=='__main__':main()
