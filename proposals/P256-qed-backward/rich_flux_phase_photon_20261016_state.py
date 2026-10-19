"""Intrinsic electric-basis phase and flux-adapted states, NOT a changed H.
Pure imports: frozen bulk geometry, envelopes, determinant ratios and proposal kernel.
"""
import numpy as np
from numba import njit
import rich_bulk_photon_20261016_state as base

POSITIONS=base.POSITIONS
ARCS=base.ARCS
INCIDENCE=base.INCIDENCE
LOOPS=base.LOOPS


def flux_codes(pi):
    return np.array([0 if mu==0 else int(POSITIONS[a,0]) if mu==1 else int(POSITIONS[a,0]+POSITIONS[a,1]) for a,b,mu in ARCS],dtype=np.int64)*int(pi)


def kernel(pi,order,delta=.8):
    codes=flux_codes(pi)
    h=np.zeros((64,64))
    for e,(a,b,mu) in enumerate(ARCS):
        sign=1. if codes[e]%2==0 else -1.
        h[a,b]-=sign;h[b,a]-=sign
    xi,orbitals=np.linalg.eigh(h)
    f=delta/(xi+np.sqrt(xi*xi+delta*delta))
    F=(orbitals*f)@orbitals.T
    if order=='af_dimer':
        stagger=(-1.)**POSITIONS.sum(axis=1)
        _,u=np.linalg.eigh(h+np.diag(stagger))
        Pu=u[:,:32]@u[:,:32].T
        translation=np.zeros((64,64))
        for e,(a,b,mu) in enumerate(ARCS):
            if mu==0:translation[b,a]=(-1.)**codes[e]
        F=(Pu+.08*(np.eye(64)-Pu))@translation
    F/=np.linalg.norm(F,axis=1).mean()
    rng=np.random.default_rng(91077)
    perturb=rng.normal(size=(64,64));perturb=(perturb+perturb.T)/2
    G=F+.08*perturb/8
    neel=np.nonzero(POSITIONS.sum(axis=1)%2==0)[0];other=np.setdiff1d(np.arange(64),neel)
    reference=max(abs(np.linalg.det(F[np.ix_(neel,other)])),abs(np.linalg.det(G[np.ix_(neel,other)])),1e-100)
    scale=reference**(-1/32)
    return F*scale,G*scale,codes,dict(pi=bool(pi),order=order,orbital_eigenvalues=xi.tolist(),pair_eigenvalues=f.tolist(),scale=float(scale),state_only_parameters=dict(delta=delta,AF_stagger=1.,minority_projector=.08))


@njit(cache=True)
def phase(E,codes):
    total=0
    for e in range(192):total+=E[e]*codes[e]
    return 1. if total%2==0 else -1.


@njit(cache=True)
def evaluate(up,down,E,F,G,codes,g,a,b,epsilon,floor,coeff,density,external=0.):
    det,inv,M,stable=base.detdata(up,down,F)
    gd=np.linalg.det(base.submatrix(G,up,down))
    env=base.envelope(up,down,E,g,a,b)
    weight=env*env*(det*det+epsilon*epsilon*gd*gd+floor*floor)
    ph=phase(E,codes);psi=det*env*ph
    v=np.zeros(6,np.complex128);v[0]=psi
    n=np.zeros(64,np.int64)
    for i in range(32):n[up[i]]+=1;n[down[i]]+=1
    for p in range(2):v[1+p]=np.dot(coeff[p],E.astype(np.complex128))*psi/16.
    for i in range(64):v[5]+=density[i]*(n[i]-1)*psi
    hpsi=base.diagonal(up,down,E)*psi+0.j
    density_value=np.dot(density,(n-1).astype(np.complex128))
    hdensity_correction=0.j;contact=np.zeros((2,2),np.complex128)
    # external is solely a finite-difference diagnostic along real(coeff[0]);
    # every science measurement passes exactly zero. No A0 is put in H.
    for e in range(192):
        aa,bb,mu=ARCS[e]
        for direction in (-1,1):
            if abs(E[e]+direction)>1:continue
            source,target=(aa,bb) if direction==1 else (bb,aa)
            for spin in range(2):
                occ=up if spin==0 else down;row=-1;occupied=False
                for j in range(32):
                    if occ[j]==source:row=j
                    if occ[j]==target:occupied=True
                if row<0 or occupied:continue
                ndet=base.replacement(up,down,F,det,inv,M,stable,spin,row,target)
                old=occ[row];occ[row]=target;E[e]+=direction
                neighbor_phase=ph*(1. if codes[e]%2==0 else -1.)
                hv=-ndet*base.envelope(up,down,E,g,a,b)*neighbor_phase*np.exp(-1j*external*direction*coeff[0,e].real)
                occ[row]=old;E[e]-=direction
                hpsi+=hv;hdensity_correction+=(density[target]-density[source])*hv
                for p in range(2):
                    v[3+p]-=1j*direction*coeff[p,e]*hv
                    for q in range(2):contact[p,q]-=coeff[p,e].conjugate()*coeff[q,e]*hv*psi
    hb=np.empty(3,np.complex128)
    for p in range(2):hb[p]=np.dot(coeff[p],E.astype(np.complex128))*hpsi/16.+1j*v[3+p]/16.
    hb[2]=density_value*hpsi+hdensity_correction
    return v,hpsi,contact,weight,stable,hb


@njit(cache=True)
def measure(us,ds,Es,F,G,codes,g,a,b,epsilon,floor,coeff,density):
    n=len(us);raw=np.zeros((n,8));vectors=np.empty((n,6),np.complex128)
    images=np.empty((n,3),np.complex128);contacts=np.empty((n,2,2),np.complex128);hp=np.empty(n,np.complex128)
    for k in range(n):
        u=us[k].copy();d=ds[k].copy();E=Es[k].copy()
        v,h,c,w,stable,hb=evaluate(u,d,E,F,G,codes,g,a,b,epsilon,floor,coeff,density)
        root=np.sqrt(w);vectors[k]=v/root;images[k]=hb/root;contacts[k]=c/w;hp[k]=h/root
        D=0
        for site in range(64):
            if np.any(u==site) and np.any(d==site):D+=1
        norm=(v[0].conjugate()*v[0]).real/w
        raw[k,0]=norm;raw[k,1]=(v[0].conjugate()*h).real/w;raw[k,2]=abs(h)**2/w
        raw[k,3]=norm*D;raw[k,4]=norm*np.sum(E*E);raw[k,5]=0. if stable else abs(h)**2/w
        raw[k,6]=0. if stable else 1.;raw[k,7]=D
    return raw,vectors,images,contacts,hp


def physical_amplitude(occupations,E,F,codes,g=.45,a=.15,b=.08):
    occupations=np.asarray(occupations,dtype=np.int64);E=np.asarray(E,dtype=np.int64)
    if occupations.shape!=(2,64) or E.shape!=(192,):return 0.
    if np.any((occupations!=0)&(occupations!=1)) or np.any(abs(E)>1):return 0.
    if np.any(occupations.sum(axis=1)!=32) or np.any(INCIDENCE@E+occupations.sum(axis=0)-1):return 0.
    up=np.nonzero(occupations[0])[0];down=np.nonzero(occupations[1])[0]
    return float(base.parity(up,down)*np.linalg.det(F[np.ix_(up,down)])*base.envelope(up,down,E,g,a,b)*phase(E,codes))
