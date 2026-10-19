"""Exact 5^5 net-flow contraction; 24 physical arcs remain distinct in kernels.
This finite cube-specific contraction is for base pair/Slater states, not an MPS.
"""
import hashlib
from itertools import product,combinations
import json
from pathlib import Path
import time
import numpy as np
from numba import njit
import rich_gauss_projected_photon_20261016_state as s
ROOT=Path(__file__).resolve().parent
P='rich_gauss_projected_photon_20261016_'


@njit(cache=True)
def contract(rhos,free,tree,chords,Btinv,Bc,K,Q,L):
    count=len(rhos);Z=np.zeros(count);ZE=np.zeros(count);ZL=np.zeros((count,12,2,2));counts=np.zeros(count,np.int64)
    for r in range(count):
        for k in range(len(free)):
            delta=np.zeros(12,np.int64)
            rhs=-rhos[r,1:].copy()
            for c in range(5):
                delta[chords[c]]=free[k,c]
                for v in range(7):rhs[v]-=Bc[v,c]*free[k,c]
            valid=True
            for j in range(7):
                value=0
                for v in range(7):value+=Btinv[j,v]*rhs[v]
                delta[tree[j]]=value
                if abs(value)>2:valid=False
            if not valid:continue
            counts[r]+=1
            weight=1.0
            for e in range(12):weight*=K[delta[e]+2]
            Z[r]+=weight
            for e in range(12):
                d=delta[e]+2
                ZE[r]+=weight*Q[d]/K[d]/32.0
                for arc in range(2):
                    for direction in range(2):ZL[r,e,arc,direction]+=weight*L[arc,direction,d]/K[d]
    return Z,ZE,ZL,counts


def exact(g=.45,eta=0.,a=.15,b=0.):
    table=s.pair_table(g,eta)
    matter=[];rholist=[]
    for up in combinations(range(8),4):
        for down in combinations(range(8),4):
            bits=sum(1<<(2*v) for v in up)+sum(1<<(2*v+1) for v in down)
            matter.append(bits);rholist.append(tuple(s.charge(bits)))
    rhoseq=sorted(set(rholist))
    rhos=np.asarray(rhoseq,dtype=np.int64);lookup={rho:i for i,rho in enumerate(rhoseq)}
    N=np.zeros(len(rhos));B=np.zeros((len(rhos),12,2,2));sector=np.zeros((5,len(rhos)))
    arcmap={int(arc):(e,k) for e,pair in enumerate(s.PAIRS) for k,arc in enumerate(pair)}
    for bits,rho in zip(matter,rholist):
        r=lookup[rho];A=table[bits];N[r]+=A*A
        D=sum(((bits>>(2*v))&3)==3 for v in range(8));sector[D,r]+=A*A
        for arc in range(24):
            e,k=arcmap[arc]
            for d in (-1,1):
                for spin in range(2):
                    target,h=s.hop(bits,arc,d,spin)
                    B[r,e,k,(d+1)//2]+=A*table[target]*h
    merged=[s.EDGES[int(pair[0])][:2] for pair in s.PAIRS]
    incidence=np.zeros((8,12),dtype=np.int64)
    for e,(v,targetvertex) in enumerate(merged):incidence[v,e]=1;incidence[targetvertex,e]=-1
    tree=[]
    for v in range(1,8):
        parent=v^(v&-v)
        tree.append(next(e for e,(x,y) in enumerate(merged) if x==parent and y==v))
    chords=[e for e in range(12) if e not in tree]
    Btinv=np.rint(np.linalg.inv(incidence[1:,tree])).astype(np.int64)
    assert np.array_equal(Btinv@incidence[1:,tree],np.eye(7,dtype=int))
    free=np.asarray(list(product(range(-2,3),repeat=5)),dtype=np.int64)
    K=np.zeros(5);Q=np.zeros(5);L=np.zeros((2,2,5))
    def w(x,y):return np.exp(-a*(x*x+y*y)-b*(x+y)**2)
    for x,y in product(range(-1,2),repeat=2):
        delta=x-y;weight=w(x,y)**2;K[delta+2]+=weight;Q[delta+2]+=(x*x+y*y)*weight
        for k in range(2):
            for di,d in enumerate((-1,1)):
                nx=x+d if k==0 else x;ny=y+d if k==1 else y
                if abs(nx)<=1 and abs(ny)<=1:L[k,di,delta+2]+=w(x,y)*w(nx,ny)
    Z,ZE,ZL,counts=contract(rhos,free,np.array(tree),np.array(chords),Btinv,incidence[1:,chords],K,Q,L)
    norm=float(N@Z);kinetic=float(np.sum(B*ZL)/norm);electric=float(N@ZE/norm)
    chargeenergy=float((N*2*np.sum(rhos*rhos,axis=1))@Z/norm)
    result=dict(parameters=dict(g=g,eta=eta,a=a,b=b,tau=0.),norm=norm,energy=kinetic+electric+chargeenergy,
                kinetic=kinetic,electric=electric,chargeenergy=chargeenergy,energy_per_physical_arc=(kinetic+electric+chargeenergy)/24,
                pair_sector_weights=(sector@Z/norm).tolist(),matter_configurations=len(matter),nonzero_matter_amplitudes=int(np.count_nonzero(table)),
                charge_patterns=len(rhos),free_net_flows=len(free),admissible_netflows=int(counts.sum()),
                tree_edges=tree,chord_edges=chords,physical_arcs=24,merged_contraction_edges=12,
                method='Exact sum of all 4900 matter configs and3125 free netflow candidates per rho. Each K(delta) sums all9 physical twoarc states; treeGauss exact integer solve. Each hopping kernel shifts ONE physical arc and retains full CAR sign.',
                limitation='Exact norm/energy only for undressed state; not full H2. Parent physics remains finite8sites Ecap1.')
    np.savez_compressed(ROOT/(P+('slater' if eta==0 else 'pair')+'_exact_arrays.npz'),rhos=rhos,matter_norm=N,matter_hop=B,Z=Z,ZE=ZE,ZL=ZL,K=K,Q=Q,L=L,counts=counts,table=table)
    return result


def main():
    started=time.time();results=[]
    for args in ((.45,0.,.15,0.),(.45,.08,.15,.08)):
        result=exact(*args);results.append(result);print(json.dumps(result),flush=True)
    value=dict(results=results,elapsed_seconds=time.time()-started,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/(P+'exact_result.json')).write_text(json.dumps(value,indent=2)+'\n')


if __name__=='__main__':main()
