"""Independent exact finite-ring CAR control and 3D Fourier geometry, not a vacuum solver."""
from itertools import product, combinations
from pathlib import Path
import json
import numpy as np
from typing import Any
from numpy.typing import NDArray

ROOT=Path(__file__).resolve().parent
P='rich_review_wave35_20261016_'


def geometry(L) -> tuple[NDArray[np.int64],NDArray[np.int64],NDArray[np.int64]]:
    pos=np.array(list(product(range(L), repeat=3)),dtype=np.int64)
    lookup={tuple(r):i for i,r in enumerate(pos)}
    edges=[]
    for a,r in enumerate(pos):
        for mu in range(3):
            target=r.copy();target[mu]=(target[mu]+1)%L
            edges.append((a,lookup[tuple(target)],mu))
    arcs=np.asarray(edges,dtype=np.int64);B=np.zeros((len(pos),len(arcs)),dtype=np.int64)
    for e,(a,b,_) in enumerate(edges):B[a,e]=1;B[b,e]=-1
    return pos,arcs,B


def sources(L,q):
    pos,arcs,B=geometry(L);q=np.asarray(q)
    d=2*np.sin(q/2)
    _,_,vh=np.linalg.svd(d[None,:],full_matrices=True)
    pol=vh[1:]
    mid=pos[arcs[:,0]].astype(float)
    # Physical oriented midpoint uses +1/2 BEFORE wrapping, never average endpoints.
    mid[np.arange(len(arcs)),arcs[:,2]]+=.5
    c=np.exp(1j*(mid@q))[None,:]*pol[:,arcs[:,2]]/np.sqrt(len(pos))
    return c,pol,B


def car(bits,p,q,nojw=False):
    p,q=int(p),int(q)
    if not ((bits>>p)&1) or ((bits>>q)&1):return bits,0
    after=bits^(1<<p)
    parity=(bits&((1<<p)-1)).bit_count()+(after&((1<<q)-1)).bit_count()
    return after|(1<<q),-1 if nojw else -(-1)**parity


def ring(L):
    states=[]
    for up in combinations(range(L),L//2):
        for down in combinations(range(L),L//2):
            bits=sum(1<<(2*v) for v in up)+sum(1<<(2*v+1) for v in down)
            rho=np.array([((bits>>(2*v))&1)+((bits>>(2*v+1))&1)-1 for v in range(L)])
            for E in product((-1,0,1),repeat=L):
                if np.all(np.asarray(E)-np.roll(E,1)+rho==0):states.append((bits,E))
    index={x:i for i,x in enumerate(states)};n=len(states)
    H=np.zeros((n,n));bad=np.zeros_like(H);Q=np.array([E for _,E in states])
    for col,(bits,E) in enumerate(states):
        rho=np.array([((bits>>(2*v))&1)+((bits>>(2*v+1))&1)-1 for v in range(L)])
        H[col,col]=np.sum(np.array(E)**2)/32+2*np.sum(rho*rho)
        bad[col,col]=H[col,col]
        for e in range(L):
            for direction in (-1,1):
                if abs(E[e]+direction)>1:continue
                a,b=(e,(e+1)%L) if direction==1 else ((e+1)%L,e)
                for spin in (0,1):
                    target,h=car(bits,2*a+spin,2*b+spin)
                    if not h:continue
                    field=list(E);field[e]+=direction;row=index[(target,tuple(field))]
                    H[row,col]+=h;bad[row,col]-=1
    val,vec=np.linalg.eigh(H);psi=vec[:,0]
    q=2*np.pi/L;c=np.exp(1j*q*(np.arange(L)+.5))/np.sqrt(L)
    Eop=np.diag(Q@c);F=Eop/16
    J=-1j*(H@Eop-Eop@H)
    correct=np.linalg.norm(H@F-F@H-1j*J/16)
    wrong=np.linalg.norm(H@F-F@H+1j*J/16)
    O=[F,J];M=np.stack([vec.conj().T@(o@psi) for o in O],axis=1)
    # Actual O(q) is non-Hermitian: O(-q)=O(q)^dagger. Separate mixed residues.
    minus=np.stack([vec.conj().T@(o.conj().T@psi) for o in O],axis=1)
    Wp=np.einsum('ni,nj->nij',minus.conj(),minus)
    Wm=np.einsum('ni,nj->nij',M,M.conj())
    z=.37+.19j;gaps=val-val[0]
    chi=np.sum(Wp[1:]/(gaps[1:,None,None]-z)+Wm[1:]/(gaps[1:,None,None]+z),axis=0)
    shortcut=np.sum(2*gaps[1:,None,None]*Wp[1:]/(gaps[1:,None,None]**2-z*z),axis=0)
    # Gauss B E+rho=0 and dot E=-J imply dot rho=B J in this convention.
    rho_diag=np.array([[((b>>(2*v))&1)+((b>>(2*v+1))&1)-1 for v in range(L)] for b,_ in states])
    arcJ=[-1j*(H@np.diag(Q[:,e])-np.diag(Q[:,e])@H) for e in range(L)]
    continuity=max(np.linalg.norm(1j*(H@np.diag(rho_diag[:,v])-np.diag(rho_diag[:,v])@H)-(arcJ[v]-arcJ[(v-1)%L])) for v in range(L))
    arrays: dict[str,Any]
    arrays=dict(H=H,H_noJW=bad,E=Q,bits=np.array([b for b,_ in states]),eigenvalues=val,ground=psi,F=F,J=J,Wplus=Wp,Wminus=Wm,chi=chi,shortcut=shortcut)
    np.savez_compressed(ROOT/(P+f'ringL{L}_raw.npz'),**arrays)
    return dict(L=L,physical_dimension=n,energy=float(val[0]),clamped_band_velocities=(2*np.sin(2*np.pi*np.arange(L)/L)).tolist(),quantum_current_frobenius=float(np.linalg.norm(J)),current_ground_norm2=float(np.linalg.norm(J@psi)**2),commutator_correct=float(correct),commutator_wrong_sign=float(wrong),noJW_H_difference=float(np.linalg.norm(H-bad)),continuity_residual=float(continuity),mixed_scalar_shortcut_error=float(np.linalg.norm(chi-shortcut)),q_conjugation_residual=float(np.linalg.norm(Eop.conj().T-np.diag(Q@c.conj()))))


def main():
    reports=[]
    for L in (2,4):
        pos,arcs,B=geometry(L);wrap=sum(pos[b,mu]==0 for a,b,mu in arcs)
        report: dict[str,Any]
        report=dict(L=L,sites=len(pos),arcs=len(arcs),distinct_oriented_arcs=len(set(map(tuple,arcs))),wrap_arcs=int(wrap),incidence_rank=int(np.linalg.matrix_rank(B)),cycle_rank=len(arcs)-int(np.linalg.matrix_rank(B)))
        fourier=[]
        for q in ((2*np.pi/L,0,0),(2*np.pi/L,2*np.pi/L,0)):
            c,pol,_=sources(L,q);cm,_,_=sources(L,-np.array(q))
            fourier.append(dict(q=q,transverse_divergence=float(np.max(abs(B@c.T))),q_minus_conjugation=float(np.max(abs(cm-c.conj()))),source_gram=(c@c.conj().T).real.tolist()))
        report['fourier']=fourier;reports.append(report)
    result=dict(kind='INDEPENDENT_GEOMETRY_AND_EXACT_RING_CONTROL_NOT_V64_VACUUM',geometry=reports,rings=[ring(2),ring(4)])
    (ROOT/(P+'geometry_findings.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
