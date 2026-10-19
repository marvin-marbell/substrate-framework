"""Actual L4 CAR/link3 Gauss-projected determinant with linked field correlations.
No Hamiltonian truncation, pair-number cutoff, integer charge encoding or H-image.
"""
import numpy as np
from numba import njit

L=4
POSITIONS=np.array([(x,y,z) for x in range(L) for y in range(L) for z in range(L)],dtype=np.int64)
ARCS=np.array([(i, next(j for j,p in enumerate(POSITIONS) if np.array_equal(p,(r+np.eye(3,dtype=int)[mu])%L)),mu) for i,r in enumerate(POSITIONS) for mu in range(3)],dtype=np.int64)
INCIDENCE=np.zeros((64,192),dtype=np.int64)
for e,(a,b,mu) in enumerate(ARCS):INCIDENCE[a,e]=1;INCIDENCE[b,e]=-1
NEXT=np.array([3*b+mu for a,b,mu in ARCS])
loops=[]
for i,r in enumerate(POSITIONS):
    for mu in range(3):
        for nu in range(mu+1,3):
            c=np.zeros(192,dtype=np.int64);a=3*i+mu;b=3*i+nu
            c[a]=1;c[3*ARCS[a,1]+nu]=1;c[3*ARCS[b,1]+mu]=-1;c[b]=-1;loops.append(c)
for mu in range(3):
    for i,r in enumerate(POSITIONS):
        if r[mu]!=0:continue
        c=np.zeros(192,dtype=np.int64);v=i
        for _ in range(4):e=3*v+mu;c[e]=1;v=ARCS[e,1]
        loops.append(c)
LOOPS=np.array(loops)


def kernel(delta=0.8,family='singlet'):
    momenta=2*np.pi*POSITIONS/4
    xi=-2*np.cos(momenta).sum(axis=1)
    xi[np.abs(xi)<1e-12]=0 # exact half-filled shell: all zero modes have f=1
    f=delta/(xi+np.sqrt(xi*xi+delta*delta))
    phi=np.exp(1j*POSITIONS@momenta.T)/8
    F=((phi*f)@phi.conj().T).real
    if family=='antiferromagnetic':
        h=np.zeros((64,64))
        for a,b,mu in ARCS:h[a,b]-=1;h[b,a]-=1
        stagger=(-1.)**POSITIONS.sum(axis=1)
        _,u=np.linalg.eigh(h+np.diag(stagger))
        _,d=np.linalg.eigh(h-np.diag(stagger))
        Pu=u[:,:32]@u[:,:32].T;Pd=d[:,:32]@d[:,:32].T
        Qu=np.eye(64)-Pu;Qd=np.eye(64)-Pd
        F=Pu@F@Pd+0.08*Qu@F@Qd
    elif family=='af_dimer':
        h=np.zeros((64,64))
        for a,b,mu in ARCS:h[a,b]-=1;h[b,a]-=1
        stagger=(-1.)**POSITIONS.sum(axis=1)
        _,u=np.linalg.eigh(h+np.diag(stagger))
        Pu=u[:,:32]@u[:,:32].T
        translation=np.zeros((64,64))
        for i in range(64):translation[ARCS[3*i,1],i]=1
        F=(Pu+0.08*(np.eye(64)-Pu))@translation
    F/=np.linalg.norm(F,axis=1).mean()
    rng=np.random.default_rng(91077)
    perturb=rng.normal(size=(64,64));perturb=(perturb+perturb.T)/2
    G=F+0.08*perturb/8
    neel=np.nonzero(POSITIONS.sum(axis=1)%2==0)[0];other=np.setdiff1d(np.arange(64),neel)
    reference=max(abs(np.linalg.det(F[np.ix_(neel,other)])),abs(np.linalg.det(G[np.ix_(neel,other)])),1e-100)
    scale=reference**(-1/32)
    return F*scale,G*scale,dict(shell_zero_count=int(np.sum(xi==0)),xi=xi.tolist(),f=f.tolist(),det_reference=reference,family=family)


@njit(cache=True)
def parity(up,down):
    total=0
    for i in range(32):
        for j in range(i+1,32):
            total+=int(up[i]>up[j])+int(down[i]>down[j])
        for j in range(32):total+=int(down[j]<up[i])
    return -1. if total%2 else 1.


@njit(cache=True)
def envelope(up,down,E,g,a,b):
    n=np.zeros(64,np.int64)
    for i in range(32):n[up[i]]+=1;n[down[i]]+=1
    D=0
    for i in range(64):D+=int(n[i]==2)
    s=0.
    for e in range(192):s+=a*E[e]**2+b*(E[e]-E[NEXT[e]])**2
    return np.exp(-g*D-s)


@njit(cache=True)
def submatrix(F,up,down):
    M=np.empty((32,32))
    for i in range(32):
        for j in range(32):M[i,j]=F[up[i],down[j]]
    return M


@njit(cache=True)
def detdata(up,down,F):
    M=submatrix(F,up,down)
    determinant=np.linalg.det(M)
    # An ill-conditioned inverse is never used for amplitude or nodal H2.
    singular=np.linalg.svd(M)[1]
    stable=singular[-1]>1e-11*singular[0]
    inverse=np.zeros((32,32))
    if stable:inverse=np.linalg.inv(M)
    return determinant,inverse,M,stable


@njit(cache=True)
def replacement(up,down,F,det,inv,M,stable,spin,row,target):
    if stable:
        r=0.
        if spin==0:
            for j in range(32):r+=F[target,down[j]]*inv[j,row]
        else:
            for i in range(32):r+=inv[row,i]*F[up[i],target]
        return det*r
    changed=M.copy()
    if spin==0:
        for j in range(32):changed[row,j]=F[target,down[j]]
    else:
        for i in range(32):changed[i,row]=F[up[i],target]
    return np.linalg.det(changed)


@njit(cache=True)
def diagonal(up,down,E):
    n=np.zeros(64,np.int64)
    for i in range(32):n[up[i]]+=1;n[down[i]]+=1
    return 2*np.sum((n-1)**2)+np.sum(E*E)/32.


@njit(cache=True)
def evaluate(up,down,E,F,G,g,a,b,epsilon,floor,coeff,density):
    """Absolute coefficient vectors; no division by state determinant at its nodes.
    v=[psi,F_1 psi,F_2 psi,J_1 psi,J_2 psi,rho_q psi].
    Internally labelled particle CAR parity cancels on H edges: Hxy*psi(y)=-rawdet(y)*envelope(y).
    """
    det,inv,M,stable=detdata(up,down,F)
    gd=np.linalg.det(submatrix(G,up,down))
    env=envelope(up,down,E,g,a,b)
    weight=env*env*(det*det+epsilon*epsilon*gd*gd+floor*floor)
    psi=det*env
    v=np.zeros(6,np.complex128);v[0]=psi
    n=np.zeros(64,np.int64)
    for i in range(32):n[up[i]]+=1;n[down[i]]+=1
    for p in range(2):v[1+p]=np.dot(coeff[p],E.astype(np.complex128))*psi/16.
    for i in range(64):v[5]+=density[i]*(n[i]-1)*psi
    hpsi=diagonal(up,down,E)*psi
    density_value=np.dot(density,(n-1).astype(np.complex128))
    hdensity_correction=0.j
    contact=np.zeros((2,2),np.complex128)
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
                ndet=replacement(up,down,F,det,inv,M,stable,spin,row,target)
                old=occ[row];occ[row]=target;E[e]+=direction
                hv=-ndet*envelope(up,down,E,g,a,b)
                occ[row]=old;E[e]-=direction
                hpsi+=hv
                hdensity_correction+=(density[target]-density[source])*hv
                for p in range(2):
                    v[3+p]-=1j*direction*coeff[p,e]*hv
                    for q in range(2):contact[p,q]-=coeff[p,e].conjugate()*coeff[q,e]*hv*psi
    hb=np.empty(3,np.complex128)
    for p in range(2):
        field=np.dot(coeff[p],E.astype(np.complex128))/16.
        hb[p]=field*hpsi+1j*v[3+p]/16.
    hb[2]=density_value*hpsi+hdensity_correction
    return v,hpsi,contact,weight,stable,hb


def physical_amplitude(occupations,E,F,g=.45,a=.15,b=.08):
    occupations=np.asarray(occupations,dtype=np.int64);E=np.asarray(E,dtype=np.int64)
    if occupations.shape!=(2,64) or E.shape!=(192,):return 0.
    if np.any((occupations!=0)&(occupations!=1)) or np.any(occupations.sum(axis=1)!=32) or np.any(abs(E)>1):return 0.
    if np.any(INCIDENCE@E+occupations.sum(axis=0)-1):return 0.
    up=np.nonzero(occupations[0])[0];down=np.nonzero(occupations[1])[0]
    return float(parity(up,down)*np.linalg.det(F[np.ix_(up,down)])*envelope(up,down,E,g,a,b))


@njit(cache=True)
def sample(seed,n,burn,thin,F,G,g,a,b,epsilon,floor):
    np.random.seed(seed)
    up=np.empty(32,np.int64);down=np.empty(32,np.int64);iu=0;idd=0
    for i in range(64):
        if np.sum(POSITIONS[i])%2==0:up[iu]=i;iu+=1
        else:down[idd]=i;idd+=1
    E=np.zeros(192,np.int64)
    accepted=np.zeros(3,np.int64);proposed=np.zeros(3,np.int64)
    def weight(u,d,field):
        env=envelope(u,d,field,g,a,b)
        det=np.linalg.det(submatrix(F,u,d));gd=np.linalg.det(submatrix(G,u,d))
        return env*env*(det*det+epsilon*epsilon*gd*gd+floor*floor)
    w=weight(up,down,E)
    us=np.empty((n,32),np.int64);ds=us.copy();Es=np.empty((n,192),np.int64)
    for step in range(burn+n*thin):
        nu=up.copy();nd=down.copy();ne=E.copy();k=np.random.randint(10)
        if k<6:
            kind=0;e=np.random.randint(192);direction=2*np.random.randint(2)-1;spin=np.random.randint(2)
            aa,bb,mu=ARCS[e];source,target=(aa,bb) if direction==1 else (bb,aa)
            occ=nu if spin==0 else nd;row=-1;occupied=False
            for j in range(32):
                if occ[j]==source:row=j
                if occ[j]==target:occupied=True
            if row>=0 and not occupied and abs(E[e]+direction)<=1:occ[row]=target;ne[e]+=direction
        elif k<9:
            kind=1;loop=LOOPS[np.random.randint(len(LOOPS))];direction=2*np.random.randint(2)-1
            ne+=direction*loop
            if np.any(np.abs(ne)>1):ne=E.copy()
        else:
            kind=2;i=np.random.randint(32);j=np.random.randint(32)
            aa=nu[i];bb=nd[j];valid=True
            for m in range(32):
                if nu[m]==bb or nd[m]==aa:valid=False
            if valid:nu[i]=bb;nd[j]=aa
        proposed[kind]+=1;nw=weight(nu,nd,ne)
        if np.random.random()*w<nw:
            if np.any(nu!=up) or np.any(nd!=down) or np.any(ne!=E):accepted[kind]+=1
            up=nu;down=nd;E=ne;w=nw
        if step>=burn and (step-burn)%thin==0:
            j=(step-burn)//thin;us[j]=up;ds[j]=down;Es[j]=E
    return us,ds,Es,accepted,proposed


def sources(q):
    q=np.asarray(q,float);k=2*np.sin(q/2)
    # SVD real transverse basis of actual midpoint-incidence symbol.
    _,_,vh=np.linalg.svd(k.reshape(1,3));polarizations=vh[1:]
    coeff=np.empty((2,192),complex)
    for p in range(2):
        for e,(aa,bb,mu) in enumerate(ARCS):
            midpoint=POSITIONS[aa].astype(float);midpoint[mu]+=.5
            coeff[p,e]=polarizations[p,mu]*np.exp(1j*q@midpoint)/8
    density=np.exp(1j*POSITIONS@q)/8
    return coeff,density,polarizations
