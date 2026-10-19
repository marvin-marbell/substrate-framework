"""Material coherent two-pair-kernel variational extension, same full Hamiltonian."""
import numpy as np
from numba import njit
import rich_bulk_photon_20261016_state as base
import rich_pi_flux_dynamics_20261016_state as s


@njit(cache=True)
def guide(u,d,field,F,Fpair,G):
    env=base.envelope(u,d,field,.45,.15,.08)
    det=np.linalg.det(base.submatrix(F,u,d));pair=np.linalg.det(base.submatrix(Fpair,u,d));gd=np.linalg.det(base.submatrix(G,u,d))
    return env*env*(det*det+pair*pair+.08*gd*gd+2e-24)


@njit(cache=True)
def sample(seed,n,burn,thin,F,Fpair,G):
    np.random.seed(seed)
    up=np.empty(32,np.int64);down=np.empty(32,np.int64);iu=0;idd=0
    for i in range(64):
        if np.sum(base.POSITIONS[i])%2==0:up[iu]=i;iu+=1
        else:down[idd]=i;idd+=1
    E=np.zeros(192,np.int64);accepted=np.zeros(3,np.int64);proposed=np.zeros(3,np.int64)
    w=guide(up,down,E,F,Fpair,G)
    us=np.empty((n,32),np.int64);ds=us.copy();Es=np.empty((n,192),np.int64)
    for step in range(burn+n*thin):
        nu=up.copy();nd=down.copy();ne=E.copy();kinddraw=np.random.randint(10)
        if kinddraw<6:
            kind=0;e=np.random.randint(192);direction=2*np.random.randint(2)-1;spin=np.random.randint(2)
            aa,bb,mu=base.ARCS[e];source,target=(aa,bb) if direction==1 else (bb,aa)
            occ=nu if spin==0 else nd;row=-1;occupied=False
            for j in range(32):
                if occ[j]==source:row=j
                if occ[j]==target:occupied=True
            if row>=0 and not occupied and abs(E[e]+direction)<=1:occ[row]=target;ne[e]+=direction
        elif kinddraw<9:
            kind=1;loop=base.LOOPS[np.random.randint(len(base.LOOPS))];direction=2*np.random.randint(2)-1
            ne+=direction*loop
            if np.any(np.abs(ne)>1):ne=E.copy()
        else:
            kind=2;i=np.random.randint(32);j=np.random.randint(32);aa=nu[i];bb=nd[j];valid=True
            for m in range(32):
                if nu[m]==bb or nd[m]==aa:valid=False
            if valid:nu[i]=bb;nd[j]=aa
        proposed[kind]+=1;nw=guide(nu,nd,ne,F,Fpair,G)
        if np.random.random()*w<nw:
            if np.any(nu!=up) or np.any(nd!=down) or np.any(ne!=E):accepted[kind]+=1
            up=nu;down=nd;E=ne;w=nw
        if step>=burn and (step-burn)%thin==0:
            j=(step-burn)//thin;us[j]=up;ds[j]=down;Es[j]=E
    return us,ds,Es,accepted,proposed


@njit(cache=True)
def measure(us,ds,Es,F,Fpair,G,codes,loops,kl,quad,coeff,density,currents,epsilon):
    N=len(us);V=np.empty((N,76),np.complex128);HV=V.copy()
    JV=np.empty((N,12,76),np.complex128);EN=np.empty((N,8,76),np.complex128)
    weights=np.zeros(N)
    for k in range(N):
        v,h,j,en,w,f,st=s.evaluate(us[k].copy(),ds[k].copy(),Es[k].copy(),F,G,codes,loops,kl,quad,coeff,density,currents,epsilon)
        vp,hp,jp,enp,wp,fp,stp=s.evaluate(us[k].copy(),ds[k].copy(),Es[k].copy(),Fpair,G,codes,loops,kl,quad,coeff,density,currents,epsilon)
        root=np.sqrt(w+wp);weights[k]=w+wp
        V[k,:38]=v/root;V[k,38:]=vp/root;HV[k,:38]=h/root;HV[k,38:]=hp/root
        JV[k,:,:38]=j/root;JV[k,:,38:]=jp/root;EN[k,:,:38]=en/root;EN[k,:,38:]=enp/root
    return V,HV,JV,EN,weights
