"""Same-law compact-rotor interval-source response, including finite-M contact.
No Maxwell term. No point-current replacement for transfer derivatives.
"""
import math
import numpy as np
from scipy.linalg import eigh, qr
from scipy.special import exprel
import rich_unitary_rotor_photon_20261016_thermal as rotor


def stable_product(matrices):
    """Pivoted U D T product; D stores scales, T stores normalized triangles."""
    V=len(matrices[0]);Q=np.eye(V,dtype=complex);d=np.ones(V);T=Q.copy()
    for B in matrices:
        Q,R,p=qr((B@Q)*d,mode='economic',pivoting=True,check_finite=False)
        d=np.abs(np.diag(R));d=np.maximum(d,np.finfo(float).tiny)
        T=(R/d[:,None])@T[p]
    return Q,d,T


def stable_logdet(matrices,twist):
    Q,d,T=stable_product(matrices);Q=twist[:,None]*Q
    big=np.maximum(d,1.);small=np.minimum(d,1.)
    A=Q.conj().T/big[:,None]+small[:,None]*T
    phase,logabs=np.linalg.slogdet(A)
    qphase,_=np.linalg.slogdet(Q)
    return phase*qphase,float(logabs+np.log(big).sum())


def stable_green(matrices):
    Q,d,T=stable_product(matrices)
    big=np.maximum(d,1.);small=np.minimum(d,1.)
    right=Q.conj().T/big[:,None]
    return np.linalg.solve(right+small[:,None]*T,right)


def stable_hs_slice(s,data,l,rng):
    B=data['B'];V=s.shape[1]
    matrices=[*B[l:],np.diag(data['twist']),*B[:l]]
    G=stable_green(matrices);changed=s.copy();accepted=0;eye=np.eye(V)
    for v in rng.permutation(V):
        old=changed[l,v];delta=np.expm1(-2*data['lam']*old)
        ratio=1+delta*(1-G[v,v]);logratio=2*np.log(abs(ratio))+2*data['lam']*old
        if np.log(rng.random())<logratio:
            G-=delta/ratio*np.outer((eye-G)[:,v],G[v,:])
            changed[l,v]=-old;accepted+=1
    return changed,accepted


def path_data(theta,alpha,s,arcs,D,beta,I=16.,U=4.,stabilized=False):
    if not stabilized:return rotor.path_data(theta,alpha,s,arcs,D,beta,I,U)
    M,V=s.shape;dt=beta/M;lam=np.arccosh(np.exp(dt*U/2))
    T=np.array([rotor.exp_hop(t,arcs,V,dt) for t in theta])
    B=T*np.exp(lam*s)[:,None,:];twist=np.exp(-1j*alpha)
    sign,logdet=stable_logdet(B,twist)
    delta=np.roll(theta,-1,axis=0)-theta;delta[-1]+=D.T@alpha
    logK,score,second=rotor.heat(delta,dt,I)
    scalar=-M*V*dt*U/2-lam*s.sum()-M*V*np.log(2)
    return dict(logweight=float(logK.sum()+2*logdet+scalar),T=T,B=B,twist=twist,
                score=score,second=second,detphase=sign,lam=lam,delta=delta,
                stabilized=True,logabsdet=logdet)


def exp_divided(x):
    """First and symmetric second divided differences, degeneracies included."""
    first=np.exp(x[None,:])*exprel(x[:,None]-x[None,:])
    a=x[:,None,None];b=x[None,:,None];c=x[None,None,:]
    lo=np.minimum(np.minimum(a,b),c);hi=np.maximum(np.maximum(a,b),c)
    mid=a+b+c-lo-hi;span=hi-lo;safe=np.where(span>1e-3,span,1.)
    second=(np.exp(mid)*exprel(hi-mid)-np.exp(lo)*exprel(mid-lo))/safe
    # Complete homogeneous-polynomial expansion around the mean. At coincident
    # eigenvalues this is exp(x)/2, not a dropped offdiagonal vertex.
    mask=span<=1e-3;mean=(lo[mask]+mid[mask]+hi[mask])/3
    u=lo[mask]-mean;v=mid[mask]-mean;w=hi[mask]-mean
    e1=u+v+w;e2=u*v+u*w+v*w;e3=u*v*w
    hm2=np.zeros_like(mean);hm1=hm2.copy();h=np.ones_like(mean);series=h/2
    for k in range(1,9):
        hn=e1*h-e2*hm1+e3*hm2
        series+=hn/math.factorial(k+2);hm2,hm1,h=hm1,h,hn
    second[mask]=np.exp(mean)*series
    return first,second


def spatial_modes(pos,arcs,qvalues=(np.pi/2,np.pi)):
    result=[];labels=[]
    for q in qvalues:
        for pol in (1,2):
            result.append(np.array([np.exp(1j*q*pos[a,0])/np.sqrt(len(pos)) if mu==pol else 0j for a,b,mu in arcs]))
            labels.append((float(q),pol))
    return np.array(result),np.array(labels)


def interval_vertices(theta,arcs,V,dt,coeff,need_same=False,spectral=None):
    """D exp(-dt h)[h_f], D²exp[h_f,h_g]+Dexp[h_fg].
    Returned g is f conjugate. Mixed h_fg IS the source/contact insertion.
    """
    if spectral is None:
        w,v=eigh(rotor.hopping(theta,arcs,V),check_finite=False)
        first,second=exp_divided(-dt*w)
    else:v,first,second=spectral
    hf=-rotor.current_matrix(theta,arcs,coeff,V)
    hg=-rotor.current_matrix(theta,arcs,coeff.conjugate(),V)
    hfg=rotor.current_matrix(theta,arcs,coeff*coeff.conjugate(),V,True)
    X=v.conj().T@(-dt*hf)@v;Y=v.conj().T@(-dt*hg)@v
    Z=v.conj().T@(-dt*hfg)@v
    mixed=np.einsum('ikj,ik,kj->ij',second,X,Y,optimize=True)
    mixed+=np.einsum('ikj,ik,kj->ij',second,Y,X,optimize=True)
    mixed+=first*Z
    A=v@(first*X)@v.conj().T;AB=v@mixed@v.conj().T
    same=None
    if need_same:
        hff=rotor.current_matrix(theta,arcs,coeff*coeff,V,True)
        same=v@(2*np.einsum('ikj,ik,kj->ij',second,X,X,optimize=True)+first*(v.conj().T@(-dt*hff)@v))@v.conj().T
    return A,A.conj().T,AB,same,v@(first*Z)@v.conj().T


def determinant_jet(B,A,C,AC,twist):
    """Analytic full determinant derivatives, retaining connected feedback."""
    V=len(B[0]);P=np.eye(V,dtype=complex);Pa=np.zeros_like(P);Pb=Pa.copy();Pab=Pa.copy()
    for b,a,c,ac in zip(B,A,C,AC):
        Pab,Pa,Pb,P=b@Pab+a@Pb+c@Pa+ac@P,b@Pa+a@P,b@Pb+c@P,b@P
    W=np.eye(V)+twist[:,None]*P
    a=np.linalg.solve(W,twist[:,None]*Pa);b=np.linalg.solve(W,twist[:,None]*Pb)
    ab=np.linalg.solve(W,twist[:,None]*Pab)
    La=np.trace(a);Lb=np.trace(b);Lab=np.trace(ab)-np.trace(a@b)
    Fa=La+Lb.conjugate();Fb=Lb+La.conjugate()
    return float((2*Lab.real+Fa*Fb).real),complex(Fa),float(2*Lab.real),float(np.linalg.cond(W))


def stable_direction(B,dB,ddB,twist,log0,step):
    """Numerical derivative of the stabilized transfer determinant.
    The local Taylor jet contains exact Frechet first/second vertices. Step-halving
    is reported separately; this numerical evaluation is not asserted exact.
    """
    lp=2*stable_logdet(B+step*dB+.5*step**2*ddB,twist)[1]
    lm=2*stable_logdet(B-step*dB+.5*step**2*ddB,twist)[1]
    first=(lp-lm)/(2*step);second=(lp+lm-2*log0)/step**2
    return second+first**2,first,second


def interval_response(theta,alpha,s,data,pos,arcs,D,beta,qvalues=(np.pi/2,np.pi),I=16.,
                      stabilized=False,derivative_step=.008,compare_step=False):
    """Return C_E(n)=sum_lag exp(iwn lag) <E(lag)E(0)>.
    For n!=0, exact identity C_E(n)=-<W''/W>_n/[4sin²(pi n/M)].
    C_E(0) is independently inserted on kinetic bonds, never inferred from J.
    Spatial q± and frequency ± are conjugate partners. Source time norm is 1/sqrt(M).
    'source_hessian' is FULL partition response, NOT an irreducible matter kernel.
    """
    M,V=s.shape;dt=beta/M;coeff,labels=spatial_modes(pos,arcs,qvalues);nc=len(coeff)
    chi=np.zeros((nc,M));logchi=chi.copy();one=np.zeros((nc,M),complex)
    direct=np.zeros_like(chi);EE=np.zeros_like(chi);step_error=np.zeros_like(chi)
    contact=np.zeros(nc)
    maxcond=0.;scalar=np.exp(data['lam']*s)[:,None,:]
    spectra=[]
    for th in theta:
        w,v=eigh(rotor.hopping(th,arcs,V),check_finite=False)
        first,second=exp_divided(-dt*w);spectra.append((v,first,second))
    for p,f in enumerate(coeff):
        # Direct kinetic-bond insertion at every temporal lag. Compact heat-kernel
        # derivatives sum winding images, and data.delta includes Gauss holonomy.
        ef=-1j*data['score']@f;eg=-1j*data['score']@f.conjugate()
        corr=(-data['second']+data['score']**2)@(f*f.conjugate())
        ct=np.array([np.mean(ef*np.roll(eg,lag)) for lag in range(M)])
        ct[0]+=corr.mean();direct[p]=(np.fft.ifft(ct)*M).real;EE[p,0]=direct[p,0]
        vertices=[interval_vertices(th,arcs,V,dt,f,stabilized,spec) for th,spec in zip(theta,spectra)]
        A=np.array([z[0] for z in vertices])*scalar
        C=np.array([z[1] for z in vertices])*scalar
        AC=np.array([z[2] for z in vertices])*scalar
        contact_B=np.array([z[4] for z in vertices])*scalar/M
        zero=np.zeros_like(data['B'])
        if stabilized:
            AA=np.array([z[3] for z in vertices])*scalar
            CC=np.array([z[3].conj().T for z in vertices])*scalar
            log0=2*stable_logdet(data['B'],data['twist'])[1]
            contact[p]=stable_direction(data['B'],zero,contact_B,data['twist'],log0,derivative_step)[0]
        else:
            contact[p]=determinant_jet(data['B'],zero,zero,contact_B,data['twist'])[0]
        for n in range(M):
            a=np.exp(2j*np.pi*n*np.arange(M)/M)/np.sqrt(M);b=a.conjugate()
            if stabilized:
                da=(a[:,None,None]*A+b[:,None,None]*C)/2
                db=(a[:,None,None]*A-b[:,None,None]*C)/(2j)
                aa=(a[:,None,None]**2*AA+2*(a*b)[:,None,None]*AC+b[:,None,None]**2*CC)/4
                bb=(-a[:,None,None]**2*AA+2*(a*b)[:,None,None]*AC-b[:,None,None]**2*CC)/4
                ra=stable_direction(data['B'],da,aa,data['twist'],log0,derivative_step)
                rb=stable_direction(data['B'],db,bb,data['twist'],log0,derivative_step)
                value=ra[0]+rb[0];mean=ra[1]+1j*rb[1];lc=ra[2]+rb[2]
                if compare_step:
                    finea=stable_direction(data['B'],da,aa,data['twist'],log0,derivative_step/2)
                    fineb=stable_direction(data['B'],db,bb,data['twist'],log0,derivative_step/2)
                    step_error[p,n]=finea[0]+fineb[0]-value
            else:
                value,mean,lc,condition=determinant_jet(data['B'],a[:,None,None]*A,b[:,None,None]*C,
                                                       (a*b)[:,None,None]*AC,data['twist'])
                maxcond=max(maxcond,condition)
            chi[p,n]=value;logchi[p,n]=lc;one[p,n]=mean
            if n:
                EE[p,n]=-value/(4*np.sin(np.pi*n/M)**2)
    return dict(source_hessian=chi,logdet_hessian=logchi,source_one=one,
                interval_contact=-contact/dt,interval_current_matsubara=(chi-contact[:,None])/dt**2,
                EE_matsubara=EE,EE_tau=np.fft.fft(EE,axis=1)/M,
                qminus_EE_matsubara=EE[:,(-np.arange(M))%M],
                qminus_EE_tau=np.fft.ifft(EE,axis=1),
                direct_EE_matsubara=direct,direct_EE_tau=np.fft.fft(direct,axis=1)/M,
                derivative_step_difference=step_error,labels=labels,
                max_source_matrix_condition=np.array(maxcond),
                transverse_divergence=np.max(abs(D@coeff.T),axis=0),
                compact_temporal_winding=(np.sum((data['delta']+np.pi)%(2*np.pi)-np.pi,axis=0)-D.T@alpha)/(2*np.pi))
