"""Independent direct-determinant H/loop consumer; shared state and sampler declared inputs."""
import json
from pathlib import Path
import numpy as np
from numba import njit
import rich_bulk_photon_20261016_state as b
import rich_flux_phase_photon_20261016_state as s
ROOT=Path(__file__).resolve().parent
P='rich_review_wave39_20261016_'

@njit(cache=True)
def direct(u,d,E,F,codes,coeff):
    env=b.envelope(u,d,E,.45,.15,.08)
    psi=np.linalg.det(b.submatrix(F,u,d))*env*s.phase(E,codes)
    h=b.diagonal(u,d,E)*psi
    comm=np.zeros(2,np.complex128)
    for e in range(192):
        aa,bb,mu=b.ARCS[e]
        for direction in (-1,1):
            if abs(E[e]+direction)>1: continue
            source,target=(aa,bb) if direction==1 else (bb,aa)
            for spin in range(2):
                occ=u if spin==0 else d
                rows=np.where(occ==source)[0]
                if len(rows)==0 or np.any(occ==target): continue
                row=rows[0];occ[row]=target;E[e]+=direction
                hv=-np.linalg.det(b.submatrix(F,u,d))*b.envelope(u,d,E,.45,.15,.08)*s.phase(E,codes)
                occ[row]=source;E[e]-=direction
                h+=hv
                for p in range(2):comm[p]+=direction*coeff[p,e]*hv/16.
    return psi,h,comm

@njit(cache=True)
def observe(us,ds,Es,F,G,codes,coeff,density):
    out=np.zeros((len(us),10))
    for k in range(len(us)):
        u=us[k].copy();d=ds[k].copy();E=Es[k].copy()
        env=b.envelope(u,d,E,.45,.15,.08)
        det=np.linalg.det(b.submatrix(F,u,d));gd=np.linalg.det(b.submatrix(G,u,d))
        w=env**2*(det**2+.04*gd**2+1e-24)
        psi,h,comm=direct(u,d,E,F,codes,coeff)
        out[k,0]=psi**2/w;out[k,1]=psi*h/w;out[k,2]=h*h/w
        for loop in range(len(b.LOOPS)):
            for direction in (-1,1):
                ne=E+direction*b.LOOPS[loop]
                if np.any(np.abs(ne)>1):continue
                amplitude=np.linalg.det(b.submatrix(F,u,d))*b.envelope(u,d,ne,.45,.15,.08)*s.phase(ne,codes)
                out[k,3 if loop<192 else 4]+=psi*amplitude/w/(384. if loop<192 else 96.)
        v,hp,c,weight,stable,hb=s.evaluate(u,d,E,F,G,codes,.45,.15,.08,.2,1e-12,coeff,density)
        expected=hb[:2]-v[1:3]*hp/psi if abs(psi)>1e-100 else comm
        out[k,5]=np.max(np.abs(comm-expected))/max(1e-100,np.max(np.abs(comm)))
        out[k,6]=np.max(np.abs(comm+expected))/max(1e-100,np.max(np.abs(comm)))
        out[k,7]=abs(h-hp)/max(1.,abs(h))
        ps0,h0,cm0=direct(u,d,E,F,np.zeros(192,np.int64),coeff)
        out[k,8]=ps0*h0/w
    return out

def main():
    coeff,density,pol=b.sources([np.pi/2,0,0]);result={};raws={}
    for pi in (False,True):
        F,G,codes,metadata=s.kernel(pi,'pair',.8)
        rows=[];chains=[]
        for seed in (390161,390179):
            u,d,E,acc,prop=b.sample(seed,256,14000,160,F,G,.45,.15,.08,.2,1e-12)
            raw=observe(u,d,E,F,G,codes,coeff,density);rows.append(raw)
            raw[:,9]=[np.max(np.abs(b.INCIDENCE@ee-(np.bincount(uu,minlength=64)+np.bincount(dd,minlength=64)-1))) for uu,dd,ee in zip(u,d,E)]
            chains.append(dict(seed=seed,accepted=acc.tolist(),proposed=prop.tolist(),energy=float(raw[:,1].sum()/raw[:,0].sum())))
        raw=np.concatenate(rows);norm=raw[:,0].sum();means=raw.sum(axis=0)/norm
        blocks=np.array([r[:,1:5].sum(axis=0)/r[:,0].sum() for r in np.array_split(raw,16)])
        label='pi' if pi else 'zero';raws[label]=raw
        result[label]=dict(samples=len(raw),energy=float(means[1]),variance=float(means[2]-means[1]**2),plaquette=float(means[3]),winding=float(means[4]),blockSE=(blocks.std(axis=0,ddof=1)/4).tolist(),phase_removed_energy=float(means[8]),EOM_relative_max=float(raw[:,5].max()),wrong_EOM_sign_relative_max=float(raw[:,6].max()),direct_vs_producer_H_max=float(raw[:,7].max()),Gauss_max=float(raw[:,9].max()),chains=chains)
    # Exact node is sampled independently from a legal state, then F is changed explicitly.
    F,G,codes,_=s.kernel(True,'pair',.8)
    u,d,E,_,_=b.sample(390199,1,100,1,F,G,.45,.15,.08,.2,1e-12)
    Fn=F.copy();Fn[u[0,0],:]=0.
    psi,h,comm=direct(u[0].copy(),d[0].copy(),E[0].copy(),Fn,codes,coeff)
    result['node_adverse']=dict(state_amplitude=float(psi),H_amplitude=float(h),ratio_division_would_drop_nonzero=bool(psi==0 and abs(h)>0))
    np.savez_compressed(ROOT/(P+'pi_raw.npz'),**raws)
    (ROOT/(P+'pi_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    assert result['pi']['energy']<result['zero']['energy']
    assert result['pi']['plaquette']<0<result['zero']['plaquette']
    assert result['pi']['winding']>0 and result['zero']['winding']>0
    assert result['pi']['phase_removed_energy']>result['pi']['energy']+20
    assert result['pi']['EOM_relative_max']<1e-9 and result['pi']['wrong_EOM_sign_relative_max']>1e-3
    assert result['node_adverse']['ratio_division_would_drop_nonzero']
if __name__=='__main__':main()
