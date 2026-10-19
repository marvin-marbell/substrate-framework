"""Own MH chain/physical operators; imported declared amplitude is shared input only."""
from pathlib import Path
import hashlib
import json
import numpy as np
from numba import njit
from rich_review_wave33_20261016_amplitude_input import amplitude, pair_table, physical_amplitude, vectors
from rich_periodic_photon_20261015_model import geometry, loops, source_geometry
ROOT=Path(__file__).resolve().parent
P='rich_review_wave33_20261016_'
POS,edges,B=geometry()
ARCS=np.asarray(edges,np.int64)
CYCLES=np.asarray(loops(),np.int64)


@njit
def move(bits,e,d,s):
    src,dst=ARCS[e,0],ARCS[e,1]
    if d==-1:src,dst=dst,src
    p,q=2*src+s,2*dst+s
    if not ((bits>>p)&1) or ((bits>>q)&1):return bits,0.
    sign=1.
    for orbital in range(min(p,q)+1,max(p,q)):
        if (bits>>orbital)&1:sign=-sign
    return bits^(1<<p)^(1<<q),-sign


@njit
def measure(bits,E,table,coeff):
    psi=amplitude(bits,E,table,.15,0.,0.,-10.)
    D=0
    for v in range(8):D+=((bits>>(2*v))&1)*((bits>>(2*v+1))&1)
    E2=0.
    for e in range(24):E2+=E[e]*E[e]
    guide=psi**2+.2**2*np.exp(-.9*D-.3*E2)
    diag=E2/32.
    for v in range(8):
        rho=((bits>>(2*v))&1)+((bits>>(2*v+1))&1)-1
        diag+=2.*rho*rho
    hpsi=diag*psi
    current=np.zeros(2,np.complex128)
    contact=np.zeros(2)
    for e in range(24):
        for d in (-1,1):
            if abs(E[e]+d)>1:continue
            for s in range(2):
                target,h=move(bits,e,d,s)
                if h==0:continue
                E[e]+=d
                other=amplitude(target,E,table,.15,0.,0.,-10.)
                E[e]-=d
                hpsi+=h*other
                for p in range(2):
                    current[p]+=-1j*h*d*coeff[p,e]*other
                    contact[p]+=-h*coeff[p,e]**2*other*psi
    field=np.zeros(2)
    for p in range(2):
        for e in range(24):field[p]+=coeff[p,e]*E[e]*psi/16.
    out=np.empty(16)
    out[0]=psi**2/guide
    out[1]=psi*hpsi/guide
    out[2]=hpsi*hpsi/guide
    out[3]=1.0 if psi==0. else 0.0
    out[4]=hpsi*hpsi/guide if psi==0. else 0.0
    out[5]=E2*psi**2/guide
    out[6]=D*psi**2/guide
    out[7]=sum(E)**2*psi**2/guide
    for p in range(2):
        out[8+p]=field[p]**2/guide
        out[10+p]=abs(current[p])**2/guide
        out[12+p]=(field[p]*current[p]).imag/guide
        out[14+p]=contact[p]/guide
    return guide,out,current


@njit
def chain(seed,n,table,coeff):
    np.random.seed(seed)
    bits=0
    for v in range(8):bits+=1<<(2*v+(v%2))
    E=np.zeros(24,np.int64)
    weight=measure(bits,E,table,coeff)[0]
    bs=np.empty(n,np.int64);Es=np.empty((n,24),np.int64)
    raw=np.empty((n,16));accepted=np.zeros(3,np.int64)
    burn=6000;thin=15
    for k in range(burn+n*thin):
        newbits=bits;newE=E.copy();kind=np.random.randint(3)
        if kind==0:
            e=np.random.randint(24);d=2*np.random.randint(2)-1;s=np.random.randint(2)
            if abs(E[e]+d)<=1:
                target,h=move(bits,e,d,s)
                if h!=0:newbits=target;newE[e]+=d
        elif kind==1:
            flow=CYCLES[np.random.randint(len(CYCLES))]*(2*np.random.randint(2)-1)
            trial=E+flow
            if np.max(np.abs(trial))<=1:newE=trial
        else:
            u=np.random.randint(8);v=np.random.randint(8)
            nu=(bits>>(2*u))&3;nv=(bits>>(2*v))&3
            if (nu==1 and nv==2) or (nu==2 and nv==1):newbits=bits^(3<<(2*u))^(3<<(2*v))
        nw=measure(newbits,newE,table,coeff)[0]
        if np.random.random()<min(1.,nw/weight):
            if newbits!=bits or np.any(newE!=E):accepted[kind]+=1
            bits=newbits;E=newE;weight=nw
        if k>=burn and (k-burn)%thin==0:
            i=(k-burn)//thin;bs[i]=bits;Es[i]=E;raw[i]=measure(bits,E,table,coeff)[1]
    return bs,Es,raw,accepted


def main():
    table=pair_table(.45,0.)
    coeff=source_geometry([1,1,0])['coefficients'][:2]
    allraw=[];allbs=[];allEs=[];reports=[]
    for seed in [33017,33053]:
        bs,Es,raw,accepted=chain(seed,3000,table,coeff)
        allraw.append(raw);allbs.append(bs);allEs.append(Es)
        norm=raw[:,0].mean();mean=raw.mean(axis=0)/norm
        blocks=raw.reshape(20,-1,16).mean(axis=1)
        ratios=blocks[:,1:]/blocks[:,0,None]
        rho=np.array([[((int(b)>>(2*v))&1)+((int(b)>>(2*v+1))&1)-1 for v in range(8)] for b in bs])
        reports.append({'seed':seed,'samples':len(bs),'mean_energy':mean[1],'full_H2':mean[2],'variance':mean[2]-mean[1]**2,'energy_block_SE':float(np.std(ratios[:,0],ddof=1)/np.sqrt(20)),
            'guide_norm_mean':norm,'node_sample_fraction':raw[:,3].mean(),'node_H2_contribution':mean[4],
            'electric2':mean[5],'double_occupancy':mean[6],'winding_sumE2':mean[7],
            'field_norm2':mean[8:10].tolist(),'current_norm2':mean[10:12].tolist(),'field_current_imag':mean[12:14].tolist(),'contact':mean[14:16].tolist(),
            'Gauss_max':int(np.abs(Es@B.T+rho).max()),'accepted':accepted.tolist()})
    raw=np.concatenate(allraw);bs=np.concatenate(allbs);Es=np.concatenate(allEs)
    # Independently reconstruct CAR and local H at selected raw states.
    comparisons=[]
    for i in np.linspace(0,len(bs)-1,16,dtype=int):
        _,_,own=measure(int(bs[i]),Es[i].copy(),table,coeff)
        declared=vectors(int(bs[i]),Es[i].copy(),table,.15,0.,0.,-10.,coeff)
        comparisons.append({'canonical_plus_declared_error':float(np.max(abs(own+declared[3:]))),'canonical_minus_declared_error':float(np.max(abs(own-declared[3:])))})
    valid_i=int(np.argmax(abs(table[bs])))
    e=Es[valid_i].copy();bits=int(bs[valid_i]);good=physical_amplitude(bits,e,table,.15,0.);e[0]+=1
    bad=physical_amplitude(bits,e,table,.15,0.)
    result={'kind':'INDEPENDENT_SHARED_AMPLITUDE_MH_NOT_INDEPENDENT_GROUND_SOLVE','state':{'g':.45,'eta':0.,'a':.15,'b':0.,'tau':0.,'shift':-10.},
        'sampling':'own symmetric primitive-hop/cycle/spinswap MH, strictly positive epsilon.2 guide, own CAR signs and operator sums; two seeds; block errors exploratory, not convergence proof',
        'chains':reports,'current_comparisons':comparisons,'physical_wrapper_valid':good,'physical_wrapper_invalid':bad,
        'source_SHA256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in [Path(__file__).name,P+'amplitude_input.py','rich_periodic_photon_20261015_model.py']}}
    np.savez_compressed(ROOT/(P+'projected_raw.npz'),bits=bs,E=Es,raw=raw)
    (ROOT/(P+'projected_findings.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    assert all(r['Gauss_max']==0 for r in reports)
    assert all(c['canonical_minus_declared_error']<1e-10 for c in comparisons)
    assert max(c['canonical_plus_declared_error'] for c in comparisons)>.01
    assert bad==0.


if __name__=='__main__':main()
