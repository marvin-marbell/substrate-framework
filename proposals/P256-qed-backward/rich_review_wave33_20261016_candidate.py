"""Independent actual H-image pair consumer on own positive-guide configurations."""
from pathlib import Path
import hashlib
import json
import numpy as np
from numba import njit
from rich_review_wave33_20261016_amplitude_input import amplitude, pair_table
from rich_review_wave33_20261016_projected import move
from rich_periodic_photon_20261015_model import source_geometry
ROOT=Path(__file__).resolve().parent
P='rich_review_wave33_20261016_'


@njit
def evaluate(bs,Es,base_table,pair,coeff):
    raw=np.zeros((len(bs),16))
    for k in range(len(bs)):
        bits=bs[k];E=Es[k].copy();E2=0.;D=0;rho2=0.
        for e in range(24):E2+=E[e]*E[e]
        for v in range(8):
            u=(bits>>(2*v))&1;d=(bits>>(2*v+1))&1
            D+=u*d;rho2+=(u+d-1)**2
        base=amplitude(bits,E,base_table,.15,0.,0.,-10.)
        guide=base**2+.04*np.exp(-.9*D-.3*E2)
        psi=amplitude(bits,E,pair,.15,.08,.10,-10.)
        hpsi=(2.*rho2+E2/32.)*psi
        J=np.zeros(2,np.complex128);F=np.zeros(2);contact=np.zeros(2)
        for p in range(2):
            for e in range(24):F[p]+=coeff[p,e]*E[e]*psi/16.
        for e in range(24):
            for d in (-1,1):
                if abs(E[e]+d)>1:continue
                for s in range(2):
                    target,h=move(bits,e,d,s)
                    if h==0:continue
                    E[e]+=d;other=amplitude(target,E,pair,.15,.08,.10,-10.);E[e]-=d
                    hpsi+=h*other
                    for p in range(2):
                        J[p]-=1j*h*d*coeff[p,e]*other
                        contact[p]-=h*coeff[p,e]**2*other*psi
        raw[k,0]=psi**2/guide
        raw[k,1]=psi*hpsi/guide
        raw[k,2]=hpsi**2/guide
        raw[k,3]=1. if psi==0. else 0.
        raw[k,4]=hpsi**2/guide if psi==0. else 0.
        raw[k,5]=E2*psi**2/guide;raw[k,6]=D*psi**2/guide
        totalE=0.
        for e in range(24):totalE+=E[e]
        raw[k,7]=totalE**2*psi**2/guide
        for p in range(2):
            raw[k,8+p]=F[p]**2/guide
            raw[k,10+p]=abs(J[p])**2/guide
            raw[k,12+p]=(F[p]*J[p]).imag/guide
            raw[k,14+p]=contact[p]/guide
    return raw


def main():
    with np.load(ROOT/(P+'projected_raw.npz')) as inp:
        bs=inp['bits'];Es=inp['E'];base_raw=inp['raw']
    raw=evaluate(bs,Es,pair_table(.45,0.),pair_table(.45,.08),source_geometry([1,1,0])['coefficients'][:2])
    reports=[]
    for i in range(2):
        selected=raw[i*3000:(i+1)*3000]
        norm=selected[:,0].mean();m=selected.mean(axis=0)/norm
        blocks=selected.reshape(20,150,16).mean(axis=1)
        eblocks=blocks[:,1]/blocks[:,0]
        reports.append({'seed':[33017,33053][i],'samples':len(selected),'energy':m[1],'H2':m[2],'full_H_variance':m[2]-m[1]**2,
            'energy_block_SE':float(np.std(eblocks,ddof=1)/np.sqrt(20)),
            'importance_ESS_not_autocorrelation_corrected':float(selected[:,0].sum()**2/np.sum(selected[:,0]**2)),
            'electric2':m[5],'double_occupancy':m[6],'sumE_squared':m[7],
            'field_norm2':m[8:10].tolist(),'current_norm2':m[10:12].tolist(),'field_current_imag':m[12:14].tolist(),'contact':m[14:16].tolist(),
            'base_energy_same_samples':float(base_raw[i*3000:(i+1)*3000,1].sum()/base_raw[i*3000:(i+1)*3000,0].sum())})
    result={'kind':'OWN_REWEIGHTED_ACTUAL_PAIR_HIMAGE_FIELD_CURRENT_CONSUMER','state':{'g':.45,'eta':.08,'a':.15,'b':.08,'tau':.10,'shift':-10.},
        'input':'own independent positive-guide MH configurations from projected_raw.npz, declared amplitude snapshot; not producer samples or evolved results',
        'limitations':'bounded correlated importance samples; block SE and ESS diagnostic only, not convergence or independent ground solve; H2 includes all sampled configurations including original Slater nodes',
        'chains':reports,'source_SHA256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in [Path(__file__).name,P+'projected.py',P+'projected_raw.npz',P+'amplitude_input.py']}}
    np.savez_compressed(ROOT/(P+'candidate_raw.npz'),raw=raw)
    (ROOT/(P+'candidate_findings.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    assert np.isfinite(raw).all()
    assert all(c['importance_ESS_not_autocorrelation_corrected']>100 for c in reports)


if __name__=='__main__':main()
