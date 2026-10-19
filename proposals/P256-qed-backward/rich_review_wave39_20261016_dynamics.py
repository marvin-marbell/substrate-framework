"""Fresh variational-family science, independent Galerkin/evolution/observable consumer."""
import json
from pathlib import Path
import numpy as np
from numba import njit
import rich_bulk_photon_20261016_state as b
import rich_flux_phase_photon_20261016_state as s
import rich_pi_flux_dynamics_20261016_state as wave
ROOT=Path(__file__).resolve().parent
P='rich_review_wave39_20261016_'

@njit(cache=True)
def independent_image(u,d,E,F,codes,loops,kl,quad,coeff,density,epsilon):
    n=np.zeros(64,np.int64)
    for k in range(32):n[u[k]]+=1;n[d[k]]+=1
    psi=np.linalg.det(b.submatrix(F,u,d))*b.envelope(u,d,E,.45,.15,.08)*s.phase(E,codes)
    f=wave.features(n,E,codes,loops,kl,quad,coeff,density,epsilon)
    out=b.diagonal(u,d,E)*psi*f
    for e in range(192):
        aa,bb,mu=b.ARCS[e]
        for direction in (-1,1):
            if abs(E[e]+direction)>1:continue
            source,target=(aa,bb) if direction==1 else (bb,aa)
            for spin in range(2):
                occ=u if spin==0 else d;rows=np.where(occ==source)[0]
                if not len(rows) or np.any(occ==target):continue
                row=rows[0];occ[row]=target;E[e]+=direction;n[source]-=1;n[target]+=1
                nf=wave.features(n,E,codes,loops,kl,quad,coeff,density,epsilon)
                amp=-np.linalg.det(b.submatrix(F,u,d))*b.envelope(u,d,E,.45,.15,.08)*s.phase(E,codes)
                out+=amp*nf
                occ[row]=source;E[e]-=direction;n[source]+=1;n[target]-=1
    return out

def main():
    F,G,codes,_=s.kernel(True,'pair',.8);geo=wave.geometry();epsilon=.35
    chunks=[];samples=[]
    for seed in (391041,391079):
        us,ds,Es,acc,prop=b.sample(seed,256,16000,160,F,G,.45,.15,.08,.2,1e-12)
        chunks.append(wave.measure(us,ds,Es,F,G,codes,*geo,epsilon));samples.append((us,ds,Es))
    V,HV,JV,EN,fs,weights,stable=[np.concatenate([chunk[k] for chunk in chunks]) for k in range(7)]
    N=len(V);normal=np.mean(abs(V[:,0])**2)
    gram=V.conj().T@V/N/normal;Hraw=V.conj().T@HV/N/normal
    eg,U=np.linalg.eigh(gram);keep=eg>eg.max()*1e-8;W=U[:,keep]/np.sqrt(eg[keep])
    Horth=W.conj().T@Hraw@W;H=(Horth+Horth.conj().T)/2
    eig,Q=np.linalg.eigh(H);initial=W.conj().T@gram[:,[0,36,37]];initial/=np.linalg.norm(initial,axis=0)
    times=[0.,.015,.04,.08];records=[];allfields=[];allmetric=[]
    # Explicit real/imag quadratures of 2T at each of two q; -q is conjugate.
    field=fs[:,[4,6,12,14]]+1j*fs[:,[5,7,13,15]]
    for t in times:
        c=Q@(np.exp(-1j*t*eig)[:,None]*(Q.conj().T@initial));coef=W@c
        psi=V@coef;hpsi=HV@coef
        norms=np.sum(abs(psi)**2,axis=0)
        obs=np.einsum('ns,nk,ns->sk',psi.conj(),field,psi)/norms[:,None]
        bare=np.einsum('ns,nks->sk',psi.conj(),np.einsum('nka,as->nks',EN,coef))/norms[:,None]
        cur=np.einsum('ns,nks->sk',psi.conj(),np.einsum('nka,as->nks',JV[:,8:],coef))/norms[:,None]
        eom=1j*(np.einsum('ns,nk,ns->sk',hpsi.conj(),field,psi)-np.einsum('ns,nk,ns->sk',psi.conj(),field,hpsi))/norms[:,None]
        # exact full-H EOM requires current/I, while projection residual can remain large.
        projected_h=V@W@H@c
        projected_eom=1j*(np.einsum('ns,nk,ns->sk',projected_h.conj(),field,psi)-np.einsum('ns,nk,ns->sk',psi.conj(),field,projected_h))/norms[:,None]
        residual=np.linalg.norm(hpsi-projected_h,axis=0)/np.sqrt(norms)
        metrics=np.einsum('ns,nk,nl,ns->skl',psi.conj(),field.conj(),field,psi)/norms[:,None,None]
        allfields.append(obs);allmetric.append(metrics)
        records.append(dict(time=t,projected_energy=np.real(np.einsum('is,ij,js->s',c.conj(),H,c)).tolist(),full_sample_energy_real=np.real(np.sum(psi.conj()*hpsi,axis=0)/norms).tolist(),full_H_residual=residual.tolist(),full_EOM_minus_current_over_I=np.linalg.norm(eom-cur/16,axis=1).tolist(),projected_vs_full_EOM=np.linalg.norm(projected_eom-eom,axis=1).tolist(),bare_onsite_electric_crest=np.real(bare[:,1::2].sum(axis=1)).tolist(),hopping=np.real(bare[:,::2].sum(axis=1)).tolist()))
    own=[];wrong=[]
    us,ds,Es=samples[0]
    for k in (7,91):
        image=independent_image(us[k].copy(),ds[k].copy(),Es[k].copy(),F,codes,*geo[:5],epsilon)/np.sqrt(weights[k])
        own.append(float(np.linalg.norm(image-HV[k])/np.linalg.norm(image)))
        frozen=HV[k,0]*fs[k]
        wrong.append(float(np.linalg.norm(image-frozen)/np.linalg.norm(image)))
    # Current outside E+/E- span is diagnostic UV weight, not an admission gate.
    A=V[:,0,None]*np.column_stack([field,field.conj()]);target=JV[:,8:,0]
    outside=np.linalg.norm(target-A@np.linalg.lstsq(A,target,rcond=1e-8)[0],axis=0)**2/np.linalg.norm(target,axis=0)**2
    # Valid own-H control: remove all hops; exact diagonal H leaves every E moment fixed.
    diagonal=np.array([b.diagonal(u,d,e) for uu,dd,ee in samples for u,d,e in zip(uu,dd,ee)])
    pulse=V[:,36];before=np.sum(abs(pulse[:,None])**2*field,axis=0)/np.sum(abs(pulse)**2)
    afterpulse=pulse*np.exp(-1j*.08*diagonal);after=np.sum(abs(afterpulse[:,None])**2*field,axis=0)/np.sum(abs(afterpulse)**2)
    fields=np.array(allfields);metrics=np.array(allmetric)
    result=dict(samples=N,seeds=[391041,391079],rank=int(keep.sum()),family_size=38,epsilon=epsilon,times=times,independent_H_image_relative=own,wrong_frozen_neighbor_feature_relative=wrong,raw_H_nonhermitian_relative=float(np.linalg.norm(Horth-Horth.conj().T)/np.linalg.norm(Horth)),records=records,field_quadrature_change=float(np.linalg.norm(fields[-1]-fields[0])),pulse_minus_evolving_background=float(np.linalg.norm((fields[-1,1:]-fields[-1,0])-(fields[0,1:]-fields[0,0]))),moving_metric_change=float(np.linalg.norm(metrics[-1]-metrics[0])),current_UV_outside_Espan=outside.tolist(),diagonal_own_H_field_change=float(np.linalg.norm(after-before)),chain_energy=[float(np.vdot(ch[0][:,0],ch[1][:,0]).real/np.vdot(ch[0][:,0],ch[0][:,0]).real) for ch in chunks],scope='Fresh declared pi state, sampled whole-H 38-dimensional Galerkin variational evolution including evolving background; not exact dynamics, solved vacuum or Nature photon pole. Field +/-q conjugacy explicit, both complex quadratures saved. Finite L4/qmin pi/2, no low-q/continuum claim.')
    np.savez_compressed(ROOT/(P+'dynamics_raw.npz'),V=V,HV=HV,JV=JV,EN=EN,features=fs,gram=gram,Hraw=Hraw,W=W,H=H,fields=fields,metrics=metrics,gram_eigenvalues=eg)
    (ROOT/(P+'dynamics_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    assert max(own)<1e-8 and min(wrong)>1e-3
    assert result['field_quadrature_change']>1e-6 and result['pulse_minus_evolving_background']>1e-6
    assert result['diagonal_own_H_field_change']<1e-12
if __name__=='__main__':main()
