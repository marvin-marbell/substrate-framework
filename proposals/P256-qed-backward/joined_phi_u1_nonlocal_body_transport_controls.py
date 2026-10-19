"""Independent dense finite-CAR/Hermite SAME-H adverse receiver controls.

Two sites, two active spins, one auxiliary particle and two static Ref holes.
This is a separately declared small algebra/control, not full N4 convergence.
"""
from __future__ import annotations
from itertools import combinations
from typing import Any
import numpy as np
from scipy.linalg import expm


def creation(n: int, count: int, index: int) -> np.ndarray:
    low=list(combinations(range(n),count));high=list(combinations(range(n),count+1));lookup={z:i for i,z in enumerate(high)}
    out=np.zeros((len(high),len(low)),complex)
    for j,ids in enumerate(low):
        if index not in ids:
            target=tuple(sorted((*ids,index)));out[lookup[target],j]=(-1)**sum(k<index for k in ids)
    return out


def second(h: np.ndarray, count: int) -> np.ndarray:
    n=len(h);size=len(list(combinations(range(n),count)))
    if count==0:return np.zeros((size,size),complex)
    cs=[creation(n,count-1,i) for i in range(n)]
    return sum((h[i,j]*cs[i]@cs[j].conj().T for i in range(n) for j in range(n)),np.zeros((size,size),complex))


def dense_control() -> dict[str,Any]:
    rng=np.random.default_rng(222101064)
    w,_=np.linalg.qr(rng.normal(size=(4,2))+1j*rng.normal(size=(4,2)));p=w@w.conj().T;empty=np.eye(4)-p
    psi=np.array([np.linalg.det(w[list(ids)]) for ids in combinations(range(4),2)])
    aux=np.array([np.sqrt(.3),1j*np.sqrt(.7)])
    orbital=np.kron(psi,aux)
    nb,ns=12,16
    e=[]
    for site in (0,1):
        aa=np.concatenate([.5*creation(4,2,2*site+a) for a in range(2)],axis=0)
        e.append(np.kron(aa,np.eye(2)))
    ids=np.diag([1.,1.,0.,0.]);nab=[second(ids,2),second(np.eye(4)-ids,2)]
    nas=[second(ids,3),second(np.eye(4)-ids,3)]
    naux=[np.diag([1.,0.]),np.diag([0.,1.])]
    rhob=[np.kron(nab[i],np.eye(2))+np.kron(np.eye(6),naux[i])-1.5*np.eye(nb) for i in range(2)]
    rhos=[np.kron(np.eye(2),np.kron(nas[i],np.eye(2))+np.kron(np.eye(4),naux[i]))-(1.5+(i==0))*np.eye(ns) for i in range(2)]
    g=1.7;hc_b=g/4*(rhob[0]-rhob[1])@(rhob[0]-rhob[1]);hc_s=g/4*(rhos[0]-rhos[1])@(rhos[0]-rhos[1])
    rhos_wrong=[rhos[0]+np.eye(ns),rhos[1]]
    hc_wrong=g/4*(rhos_wrong[0]-rhos_wrong[1])@(rhos_wrong[0]-rhos_wrong[1])
    h0=np.diag([.7,-.4,-.6,.2]).astype(complex)
    b1=np.array([[.4+.2j,-.3j],[.17,.23-.1j]])
    b2=np.array([[.12j,.09],[-.08+.03j,.11]])
    forward1=np.zeros((4,4),complex);forward1[:2,2:]=b1
    forward2=np.zeros((4,4),complex);forward2[:2,2:]=b2
    ah=np.array([[.13,.18+.07j],[.18-.07j,-.13]])
    # Literal full shared-Theta multipliers, not averaged one-body propagation.
    cut=24;s=2.;q=.4
    a=np.diag(np.sqrt(np.arange(1,cut)),1);theta=np.sqrt(s)*(a+a.T);pi=1j*(a.T-a)/(2*np.sqrt(s));kin=g*pi@pi
    phases={k:expm(1j*k*theta) for k in (1,2)}
    eye=np.eye(cut)
    def matter(count: int, source: bool, second_link: bool=True) -> np.ndarray:
        base=np.kron(second(h0,count),np.eye(2))+np.kron(np.eye(len(list(combinations(range(4),count)))),ah)
        if source:base=np.kron(np.eye(2),base)
        result=np.kron(eye,base)
        for k,block in ((1,forward1),(2,forward2)):
            if k==2 and not second_link:continue
            v=np.kron(second(block,count),np.eye(2))
            if source:v=np.kron(np.eye(2),v)
            result+=np.kron(phases[k],v)+np.kron(phases[k].conj().T,v.conj().T)
        # Auxiliary shares Theta, nontrivial contact; it does NOT factor after averaging.
        auxhop=np.array([[0,.14+.05j],[0,0]])
        av=np.kron(np.eye(len(list(combinations(range(4),count)))),auxhop)
        if source:av=np.kron(np.eye(2),av)
        return result+np.kron(phases[1],av)+np.kron(phases[1].conj().T,av.conj().T)
    mb= matter(2,False);ms=matter(3,True)
    def hamiltonians(variant: str) -> tuple[np.ndarray,np.ndarray]:
        kb=0*kin if variant=='no_Pi' else kin
        cb=0*hc_b if variant=='no_HC' else hc_b
        cs=0*hc_s if variant=='no_HC' else hc_wrong if variant=='no_Ref' else hc_s
        bb= matter(2,False,False) if variant=='no_second_link' else mb
        ss= matter(3,True,False) if variant=='no_second_link' else ms
        return np.kron(kb,np.eye(nb))+np.kron(eye,cb)+bb,np.kron(kb,np.eye(ns))+np.kron(eye,cs)+ss
    hb,hs=hamiltonians('full');ex=np.kron(expm(1j*q*theta),e[1]);e0=np.kron(eye,e[0])
    vec=np.kron(np.eye(cut)[:,0],orbital);source=e0@vec
    kx=hs@ex-ex@hb;lx=hs@kx-kx@hb
    c0=np.vdot(ex@vec,source);direct=np.vdot(kx@vec,source)
    # Independent initial Gaussian/Wick ROW formula, including the full active+
    # auxiliary density and static hole sector in HS. The small spin has2 labels.
    bb=.25*np.trace(empty[2:,:2]);phase=np.exp(-s*q*q/2)
    density=np.array([np.trace(p[:2,:2]).real+.3-1.5,np.trace(p[2:,2:]).real+.7-1.5])
    dl=np.array([-1.,1.])@np.array([[1.,-1.],[-1.,1.]])/4
    cov=np.array([np.trace(p[2:,2*i:2*i+2]@empty[2*i:2*i+2,:2]) for i in range(2)])
    hcm=g*phase*((2*dl@density+1)*bb-.5*(dl@cov))
    mm=.25*phase*np.trace(h0[2:,2:]@empty[2:,:2])
    wrong=.25*phase*np.trace(empty[2:,:2]@h0[:2,:2])
    for freq,block in ((1,forward1),(2,forward2)):
        full=block+block.conj().T
        # Row1 is reverse orientation -freq. Column0 is likewise reverse,
        # but spin ordering and offdiagonal empties differ.
        mm+=.25*np.exp(-s*(-freq-q)**2/2)*np.trace(full[2:,:2]@empty[:2,:2])
        wrong+=.25*np.exp(-s*(-freq-q)**2/2)*np.trace(empty[2:,2:]@full[2:,:2])
    defect=abs(direct-(mm+hcm))
    if defect>2e-8:raise ArithmeticError(f'Dense CAR/Hermite versus ROW Wick derivative defect {defect}')
    if abs(mm-wrong)<1e-3:raise ArithmeticError('Independent wrong-column control blind')
    t=.03
    ub=expm(-1j*t*hb)@vec;us=expm(-1j*t*hs)@source
    ct=np.vdot(ex@ub,us);linear=c0-1j*t*direct
    cap=.5*t*t*np.linalg.norm(lx,2)*np.linalg.norm(vec)*np.linalg.norm(source)
    if abs(ct-linear)>cap*(1+1e-10):raise ArithmeticError('Same-H finite remainder violated')
    variants={}
    for variant in ('no_Pi','no_HC','no_Ref','no_second_link'):
        vb,vs=hamiltonians(variant)
        value=np.vdot(ex@(expm(-1j*t*vb)@vec),expm(-1j*t*vs)@source)
        gap=abs(ct-value)
        if gap<1e-7:raise ArithmeticError(f'{variant} finite-time discriminator blind')
        variants[variant]=dict(kernel=complex(value),finite_time_contrast=float(gap),
                               initial_moment=complex(np.vdot((vs@ex-ex@vb)@vec,source)))
    # Explicit anchor/dressing boundary: neutral endpoint difference invariant,
    # a single unpaired anchor phase is not. This is not spectral evidence.
    lam=np.array([.31,-.22]);alpha=np.array([.27,-.49]);delta=np.array([-1.,1.])
    neutral=float(delta@(lam-lam[1]));neutral2=float(delta@(lam-lam[0]))
    correct=np.exp(1j*(delta@(lam-alpha)-delta@lam+delta@alpha))
    bare=np.exp(1j*(delta@alpha));wronganchor=np.exp(1j*lam[0])-np.exp(1j*lam[1])
    if abs(correct-1)>1e-12 or abs(neutral-neutral2)>1e-12 or abs(bare-1)<.1 or abs(wronganchor)<.1:
        raise ArithmeticError('Longitudinal/anchor adverse boundary blind')
    return dict(law='Small SAME-H two-site spin2 CAR, active2/3, one auxiliary, static Ref2/1, shared realTheta Hermitecut24; no N4 or untruncated convergence claim',
                cutoff=cut,time=t,raw_source_norm_squared=float(np.vdot(source,source).real),
                C0=complex(c0),m1_dense=complex(direct),m1_Wick_Gaussian=complex(mm+hcm),
                initial_ROW_Wick_defect=float(defect),initial_HC=complex(hcm),initial_matter=complex(mm),
                rejected_column_matter=complex(wrong),row_column_contrast=float(abs(mm-wrong)),
                finite_kernel=complex(ct),finite_linear=complex(linear),observed_remainder=float(abs(ct-linear)),
                finite_matrix_remainder_upper=float(cap),Kx_matrix_norm=float(np.linalg.norm(kx,2)),
                Lx_matrix_norm=float(np.linalg.norm(lx,2)),adverse_controls=variants,
                neutral_anchor_invariance_defect=abs(neutral-neutral2),long_dressing_covariance_defect=float(abs(correct-1)),
                omitted_long_dressing_gauge_defect=float(abs(bare-1)),wrong_single_endpoint_anchor_defect=float(abs(wronganchor)))
