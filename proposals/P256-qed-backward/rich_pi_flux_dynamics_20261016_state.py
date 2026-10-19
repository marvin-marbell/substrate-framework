"""Pure pi-state Hilbert-family and untruncated one-hop Hamiltonian actions.
The finite family is a variational state projection, never an operator-EOM closure.
"""
import numpy as np
from numba import njit
import rich_bulk_photon_20261016_state as base
import rich_flux_phase_photon_20261016_state as flux


def geometry():
    """Two transverse cycles per x layer; actual four-link qutrit shifts."""
    ids=[]
    for x in range(4):
        site=int(np.nonzero(np.all(base.POSITIONS==[x,0,0],axis=1))[0][0])
        ids.extend([3*site,3*site+1])  # loop ordering (xy,xz,yz) at each site
    loops=base.LOOPS[np.array(ids)]
    a,b=.15,.08
    K= a*np.eye(192)
    for e in range(192):
        v=np.zeros(192);v[e]=1;v[base.NEXT[e]]-=1
        K+=b*np.outer(v,v)
    kl=loops@K
    quad=np.einsum('ij,ij->i',loops,kl)
    q=np.array([[np.pi/2,0,0],[np.pi/2,np.pi/2,0]])
    coeff=np.concatenate([base.sources(qq)[0] for qq in q])
    density=np.array([base.sources(qq)[1] for qq in q])
    receivers=np.zeros((8,192),complex)
    for e,(aa,bb,mu) in enumerate(base.ARCS):
        if mu in (1,2):receivers[2*base.POSITIONS[aa,0]+mu-1,e]=1/16
    currents=np.vstack([receivers,coeff])
    return loops,kl,quad,coeff,density,currents


@njit(cache=True)
def wilson_ratios(E,codes,loops,kl,quad):
    r=np.zeros((len(loops),2))
    for k in range(len(loops)):
        dot=0.;par=0
        for e in range(192):dot+=E[e]*kl[k,e];par+=codes[e]*loops[k,e]
        sign=1. if par%2==0 else -1.
        for j in range(2):
            direction=2*j-1;legal=True
            for e in range(192):
                if abs(E[e]+direction*loops[k,e])>1:legal=False;break
            if legal:r[k,j]=sign*np.exp(-2*direction*dot-quad[k])
    return r


@njit(cache=True)
def features(n,E,codes,loops,kl,quad,coeff,density,epsilon):
    f=np.zeros(38,np.complex128);f[0]=1
    doublon=0.;ee=0.;near=0.
    for site in range(64):doublon+=(1. if n[site]==2 else 0.)
    for e in range(192):
        aa,bb,mu=base.ARCS[e];ee+=E[e]**2
        near+=(n[aa]-1)*(n[bb]-1)
    f[1]=doublon/8;f[2]=ee/64;f[3]=near/32
    k=4
    for iq in range(2):
        for p in range(2):
            val=0.j
            for e in range(192):val+=coeff[2*iq+p,e]*E[e]
            f[k]=val.real;f[k+1]=val.imag;k+=2
        rh=0.j;dd=0.j
        for site in range(64):
            rh+=density[iq,site]*(n[site]-1)
            dd+=density[iq,site]*(1. if n[site]==2 else 0.)
        f[k]=rh.real;f[k+1]=rh.imag;f[k+2]=dd.real;f[k+3]=dd.imag;k+=4
    r=wilson_ratios(E,codes,loops,kl,quad)
    for j in range(8):
        f[20+2*j]=(r[j,0]+r[j,1])/2
        f[21+2*j]=(r[j,0]-r[j,1])/(2j)
    for p in range(2):
        generator=0.
        for e in range(192):generator+=loops[p,e]*E[e]/2
        f[36+p]=np.exp(1j*epsilon*generator)
    return f


@njit(cache=True)
def evaluate(up,down,E,F,G,codes,loops,kl,quad,coeff,density,currents,epsilon):
    det,inv,M,stable=base.detdata(up,down,F)
    gd=np.linalg.det(base.submatrix(G,up,down))
    env=base.envelope(up,down,E,.45,.15,.08)
    w=env*env*(det*det+.04*gd*gd+1e-24)
    ph=flux.phase(E,codes);psi=det*env*ph
    n=np.zeros(64,np.int64)
    for i in range(32):n[up[i]]+=1;n[down[i]]+=1
    f=features(n,E,codes,loops,kl,quad,coeff,density,epsilon)
    v=psi*f;hv=base.diagonal(up,down,E)*v
    jv=np.zeros((len(currents),38),np.complex128)
    hop=np.zeros((8,38),np.complex128)
    for e in range(192):
        aa,bb,mu=base.ARCS[e]
        for direction in (-1,1):
            if abs(E[e]+direction)>1:continue
            source,target=(aa,bb) if direction==1 else (bb,aa)
            for spin in range(2):
                occ=up if spin==0 else down;row=-1;occupied=False
                for i in range(32):
                    if occ[i]==source:row=i
                    if occ[i]==target:occupied=True
                if row<0 or occupied:continue
                ndet=base.replacement(up,down,F,det,inv,M,stable,spin,row,target)
                old=occ[row];occ[row]=target;E[e]+=direction
                n[source]-=1;n[target]+=1
                nf=features(n,E,codes,loops,kl,quad,coeff,density,epsilon)
                amp=-ndet*base.envelope(up,down,E,.45,.15,.08)*ph*(1. if codes[e]%2==0 else -1.)
                n[source]+=1;n[target]-=1;occ[row]=old;E[e]-=direction
                image=amp*nf;hv+=image
                for p in range(len(currents)):
                    jv[p]+=-1j*direction*currents[p,e]*image
                # Partition each hopping arc equally between its endpoint x layers.
                xa=base.POSITIONS[aa,0];xb=base.POSITIONS[bb,0]
                hop[2*xa]+=.5*image;hop[2*xb]+=.5*image
    # Local energy partition: onsite at sites, E² arc energy half at endpoints.
    local=np.zeros(4)
    for site in range(64):local[base.POSITIONS[site,0]]+=2*(n[site]-1)**2
    for e in range(192):
        aa,bb,mu=base.ARCS[e]
        local[base.POSITIONS[aa,0]]+=E[e]**2/64
        local[base.POSITIONS[bb,0]]+=E[e]**2/64
    for x in range(4):hop[2*x+1]=local[x]*v
    return v,hv,jv,hop,w,f,stable


@njit(cache=True)
def measure(us,ds,Es,F,G,codes,loops,kl,quad,coeff,density,currents,epsilon):
    N=len(us);V=np.empty((N,38),np.complex128);HV=V.copy()
    JV=np.empty((N,12,38),np.complex128);energy=np.empty((N,8,38),np.complex128)
    fs=V.copy();weights=np.zeros(N);stable=np.zeros(N)
    for k in range(N):
        v,h,j,en,w,f,st=evaluate(us[k].copy(),ds[k].copy(),Es[k].copy(),F,G,codes,loops,kl,quad,coeff,density,currents,epsilon)
        root=np.sqrt(w);V[k]=v/root;HV[k]=h/root;JV[k]=j/root;energy[k]=en/root
        fs[k]=f;weights[k]=w;stable[k]=st
    return V,HV,JV,energy,fs,weights,stable


@njit(cache=True)
def wilson_images(us,ds,Es,F,G,codes,loops,kl,quad,coeff,density,currents,epsilon):
    out=np.zeros((len(us),16,38),np.complex128)
    for k in range(len(us)):
        u=us[k];d=ds[k];E=Es[k];n=np.zeros(64,np.int64)
        for i in range(32):n[u[i]]+=1;n[d[i]]+=1
        det=np.linalg.det(base.submatrix(F,u,d));gd=np.linalg.det(base.submatrix(G,u,d))
        env=base.envelope(u,d,E,.45,.15,.08)
        w=env*env*(det*det+.04*gd*gd+1e-24)
        psi=det*env*flux.phase(E,codes)/np.sqrt(w)
        ratios=wilson_ratios(E,codes,loops,kl,quad)
        for j in range(8):
            for side in range(2):
                if ratios[j,side]==0:continue
                ne=E+(2*side-1)*loops[j]
                f=features(n,ne,codes,loops,kl,quad,coeff,density,epsilon)
                val=psi*ratios[j,side]*f
                out[k,2*j]+=.5*val
                out[k,2*j+1]+=(-1j if side==0 else 1j)*.5*val
    return out


def fit_projection(V,HV,relative_cut=1e-8):
    """Sampled Galerkin projection of full H; retain pre-symmetrization error."""
    norm=np.mean(abs(V[:,0])**2)
    gram=V.conj().T@V/len(V)/norm
    hraw=V.conj().T@HV/len(V)/norm
    eig,u=np.linalg.eigh(gram);keep=eig>relative_cut*eig.max()
    W=u[:,keep]/np.sqrt(eig[keep])
    raw=W.conj().T@hraw@W
    h=(raw+raw.conj().T)/2
    initial=W.conj().T@gram[:,[0,36,37]]
    initial/=np.linalg.norm(initial,axis=0)
    return dict(norm=norm,G=gram,Hraw=hraw,W=W,h=h,initial=initial,eigenvalues=eig,raw_orthogonal_H=raw,rank=int(keep.sum()),relative_cut=relative_cut)


def evolve(projection,times):
    eig,u=np.linalg.eigh(projection['h'])
    initial=projection['initial']
    return np.array([u@(np.exp(-1j*t*eig)[:,None]*(u.conj().T@initial)) for t in times])
