"""Full compact rotor positive PH path measure; no electric Hilbert cutoff.
All returned correlators are imaginary-time, finite-beta and finite-Trotter.
"""
import numpy as np
from scipy.linalg import eigh


def geometry(L=4):
    pos=np.array([(x,y,z) for x in range(L) for y in range(L) for z in range(L)])
    arcs=np.array([(i,int(np.flatnonzero(np.all(pos==(r+np.eye(3,dtype=int)[mu])%L,axis=1))[0]),mu) for i,r in enumerate(pos) for mu in range(3)])
    D=np.zeros((len(pos),len(arcs)))
    for e,(a,b,_) in enumerate(arcs):D[a,e]=1;D[b,e]=-1
    return pos,arcs,D


def heat(delta,dt,I=16.):
    """Poisson heat kernel and exact derivatives, wrapped Gaussian n=-2..2.
    For |wrapped delta|<=pi omitted images start at 5pi; report tail separately.
    Normalization is relative to Haar dtheta/(2pi).
    """
    x=(delta+np.pi)%(2*np.pi)-np.pi
    y=x[...,None]+2*np.pi*np.arange(-2,3)
    variance=dt/I
    logterms=-y*y/(2*variance)
    maximum=logterms.max(axis=-1)
    terms=np.exp(logterms-maximum[...,None]);prob=terms/terms.sum(axis=-1)[...,None]
    logK=maximum+np.log(terms.sum(axis=-1))+.5*np.log(2*np.pi/variance)
    score=np.sum(prob*(-y/variance),axis=-1)
    second=np.sum(prob*(y*y/variance**2-1/variance),axis=-1)
    return logK,score,second


def hopping(theta,arcs,V):
    h=np.zeros((V,V),complex)
    for e,(a,b,_) in enumerate(arcs):
        value=-np.exp(1j*theta[e]);h[a,b]+=value;h[b,a]+=value.conjugate()
    return h


def exp_hop(theta,arcs,V,dt):
    w,v=eigh(hopping(theta,arcs,V),check_finite=False)
    return (v*np.exp(-dt*w))@v.conj().T


def product(matrices):
    out=np.eye(len(matrices[0]),dtype=complex)
    for b in matrices:out=b@out
    return out


def path_data(theta,alpha,s,arcs,D,beta,I=16.,U=4.):
    M,V=s.shape;dt=beta/M;lam=np.arccosh(np.exp(dt*U/2))
    T=np.array([exp_hop(t,arcs,V,dt) for t in theta])
    B=T*np.exp(lam*s)[:,None,:]
    P=product(B);Z=np.exp(-1j*alpha)
    sign,logdet=np.linalg.slogdet(np.eye(V)+Z[:,None]*P)
    delta=np.roll(theta,-1,axis=0)-theta
    delta[-1]+=D.T@alpha
    logK,score,second=heat(delta,dt,I)
    scalar=-M*V*dt*U/2-lam*s.sum()-M*V*np.log(2)
    return dict(logweight=float(logK.sum()+2*logdet+scalar),T=T,B=B,P=P,twist=Z,
                score=score,second=second,detphase=sign,lam=lam,delta=delta)


def sources(pos,arcs):
    qs=np.array([[np.pi/2,0,0],[-np.pi/2,0,0]])
    coeff=[]
    for q in qs:
        for pol in (1,2):
            coeff.append(np.array([np.exp(1j*np.dot(q,pos[a]+.5*np.eye(3)[mu]))/np.sqrt(len(pos)) if mu==pol else 0j for a,b,mu in arcs]))
    return qs,np.array(coeff)


def current_matrix(theta,arcs,coeff,V,contact=False):
    out=np.zeros((V,V),complex)
    for e,(a,b,_) in enumerate(arcs):
        z=np.exp(1j*theta[e])
        if contact:out[a,b]+=coeff[e]*z;out[b,a]+=coeff[e]*z.conjugate()
        else:out[a,b]+=1j*coeff[e]*z;out[b,a]-=1j*coeff[e]*z.conjugate()
    return out


def observe(theta,alpha,s,data,pos,arcs,D,beta):
    """Boundary-inserted operator response, including all fermion virtual states.
    Up and hole Wick contractions are summed before compact-path averaging.
    Electric insertions lie on kinetic bonds; no coincident-time shortcut.
    """
    M,V=s.shape;B=data['B'];twist=np.diag(data['twist']);P=data['P']
    # Boundary twist sits after the final slice. n is <c†_j c_i>.
    C=twist@P;n0=np.linalg.solve(np.eye(V)+C,C)
    qs,coeff=sources(pos,arcs);nc=len(coeff)
    EE=np.zeros((nc,M),complex);JJ=EE.copy();EJ=EE.copy()
    oneE=np.zeros((nc,M),complex);oneJ=oneE.copy();contacts=np.zeros((nc,M),complex)
    rho=np.zeros((M,V),complex);doublon=np.zeros(M,complex)
    field=-1j*data['score']@coeff.T
    # E_q E_-q at separated kinetic bonds, plus exact coincident K''/K.
    for p in range(nc):
        partner=p+2 if p<2 else p-2
        oneE[p]=field[:,p]
        for lag in range(M):
            EE[p,lag]=np.mean(field[:,p]*np.roll(field[:,partner],lag))
        correction=-data['second']+data['score']**2
        EE[p,0]+=np.mean(correction@(coeff[p]*coeff[partner]))
    A0=np.array([current_matrix(theta[0],arcs,c,V) for c in coeff])
    means0=np.array([2*np.real(np.trace(a@n0)) for a in A0])
    # Complex Fourier modes: sum conjugate species before Fourier combination.
    def species_mean(a,ah,n):return np.trace(a@n)+np.trace(ah@n.conjugate())
    Ah0=np.array([current_matrix(-theta[0],arcs,-c,V) for c in coeff])
    means0=np.array([species_mean(A0[p],Ah0[p],n0) for p in range(nc)])
    Q=np.eye(V,dtype=complex)
    maxcond=0.
    for lag in range(M):
        if lag:
            Q=B[lag-1]@Q
        invQ=np.linalg.inv(Q);n=Q@n0@invQ
        maxcond=max(maxcond,float(np.linalg.cond(Q)))
        diag=np.diag(n);rho[lag]=diag-diag.conjugate()
        # Original nd=1-nh; species factorization conditional on path.
        doublon[lag]=np.sum(diag*(1-diag.conjugate()))
        A=np.array([current_matrix(theta[lag],arcs,c,V) for c in coeff])
        Ah=np.array([current_matrix(-theta[lag],arcs,-c,V) for c in coeff])
        for p in range(nc):
            partner=p+2 if p<2 else p-2
            mean=species_mean(A[p],Ah[p],n);oneJ[p,lag]=mean
            conn=np.trace(A[p]@Q@(np.eye(V)-n0)@A0[partner]@n0@invQ)
            connh=np.trace(Ah[p]@Q.conjugate()@(np.eye(V)-n0.conjugate())@Ah0[partner]@n0.conjugate()@invQ.conjugate())
            JJ[p,lag]=mean*means0[partner]+conn+connh
            EJ[p,lag]=field[lag,p]*means0[partner]
            # Fourier contact uses f_q f_-q, not a squared complex mode.
            k=current_matrix(theta[lag],arcs,coeff[p]*coeff[partner],V,True)
            kh=current_matrix(-theta[lag],arcs,coeff[p]*coeff[partner],V,True)
            contacts[p,lag]=species_mean(k,kh,n)
    flux=[]
    for i in range(V):
        for mu,nu in ((0,1),(0,2),(1,2)):
            a=3*i+mu;b=3*i+nu
            flux.append(np.cos(theta[:,a]+theta[:,3*arcs[a,1]+nu]-theta[:,3*arcs[b,1]+mu]-theta[:,b]))
    return dict(EE=EE,JJ=JJ,EJ=EJ,E=oneE,J=oneJ,contact=contacts,
                rho=rho,doublon=doublon,plaquette=np.array(flux).mean(axis=0),
                max_propagator_condition=np.array(maxcond),q=qs,
                temporal_flux=np.sum(data['delta'],axis=0))


def hs_slice(s,data,l,rng):
    """Exact local discrete-HS conditional sweep, rank-one determinant updates."""
    B=data['B'];V=s.shape[1]
    before=product(B[:l]) if l else np.eye(V,dtype=complex)
    after=product(B[l:])
    C=before@np.diag(data['twist'])@after
    G=np.linalg.inv(np.eye(V)+C);changed=s.copy();accepted=0
    for v in rng.permutation(V):
        old=changed[l,v];delta=np.expm1(-2*data['lam']*old)
        ratio=1+delta*(1-G[v,v])
        logratio=2*np.log(abs(ratio))+2*data['lam']*old
        if np.log(rng.random())<logratio:
            G-=delta/ratio*np.outer((np.eye(V)-G)[:,v],G[v,:])
            changed[l,v]=-old;accepted+=1
    return changed,accepted


