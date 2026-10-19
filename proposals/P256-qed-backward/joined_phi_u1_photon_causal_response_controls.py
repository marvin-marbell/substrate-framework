"""Native-block finite CAR/Hermite photon controls; no cutoff convergence claim."""
from __future__ import annotations
import math
from typing import Any
import numpy as np
from scipy.linalg import eigh
from scipy.special import eval_hermitenorm, roots_hermitenorm
import joined_phi_u1_body_word as body
import joined_phi_u1_finite_current_kernel_controls as car


def build(cutoff: int, source: bool, species: list[Any], native: dict[str, Any]) -> dict[str, Any]:
    # Native x=0,1,2 along axis0, projected onto the first two literal colours.
    # This is a finite control geometry, NOT a truncation of all129 evolution.
    sites = [int(np.flatnonzero(np.all(native['sites']==[x,0,0],axis=1))[0]) for x in range(3)]
    terms=[]; ons=[]
    for sp in species:
        row=[]
        for r,(x,y) in enumerate(zip(sp.x,sp.y)):
            if x in sites and y in sites and int(sp.lengths[r]) in (1,2):
                path=body.joined.old.native_paths(sp.profile,native)[r]
                if path['axis']!=0: continue
                a=np.zeros((6,6),complex)
                ix,iy=sites.index(int(x)),sites.index(int(y))
                a[2*ix:2*ix+2,2*iy:2*iy+2]=sp.hops[r,:2,:2]
                q=float(sp.lengths[r])
                row.extend(((q,a),(-q,a.conj().T)))
        terms.append(row)
        ons.append(np.kron(np.eye(3),sp.onsite[:2,:2]))
    na,nb=(2 if source else 1),1
    aa,bb=car.sector(6,na),car.sector(6,nb); da,db=len(aa),len(bb); fd=da*db
    lp=native['Lplus'][np.ix_(sites,sites)]
    densities=[]
    for a in aa:
        for b in bb:
            rho=np.array([sum(i//2==x for i in a)+sum(i//2==x for i in b)-2/3 for x in range(3)])
            if source:rho[0]-=1
            densities.append(64*rho@lp@rho)
    hc=np.diag(densities)
    z,w=roots_hermitenorm(max(120,4*cutoff)); w=w/math.sqrt(2*math.pi)
    basis=np.stack([eval_hermitenorm(n,z)/math.sqrt(math.factorial(n)) for n in range(cutoff)])
    p2=np.diag((2*np.arange(cutoff)+1)/8.)
    theta=np.zeros((cutoff,cutoff)); pi=np.zeros_like(theta,dtype=complex)
    for n in range(cutoff-1):
        theta[n,n+1]=theta[n+1,n]=math.sqrt(2*(n+1))
        pi[n,n+1]=-1j*math.sqrt(n+1)/(2*math.sqrt(2));pi[n+1,n]=pi[n,n+1].conjugate()
    for n in range(cutoff-2):p2[n,n+2]=p2[n+2,n]=-math.sqrt((n+1)*(n+2))/8
    kinetic=np.kron(64*p2,np.eye(fd)); h=kinetic.astype(complex)+np.kron(np.eye(cutoff),hc)
    j=np.zeros_like(h);c=np.zeros_like(h); single=[]
    for f,count in enumerate((na,nb)):
        def lifted(a: np.ndarray) -> np.ndarray:
            return np.kron(car.lift(a,count),np.eye(db)) if f==0 else np.kron(np.eye(da),car.lift(a,count))
        h+=np.kron(np.eye(cutoff),lifted(ons[f])); small=ons[f].copy()
        for q,a in terms[f]:
            em=(basis*(w*np.exp(1j*q*math.sqrt(2)*z)))@basis.T
            term=np.kron(em,lifted(a));h+=term;j+=-1j*q*term;c+=-q*q*term;small+=a
        _,v=eigh(small);single.append(v[:,0])
    osc=np.eye(cutoff)[:,0]
    if source:
        vectors=[np.kron(osc,np.kron(body.create_small(alpha,1,6)@single[0],single[1]))*.5 for alpha in range(2)]
    else:vectors=[np.kron(osc,np.kron(single[0],single[1]))]
    return dict(H=h,J=j,C=c,Theta=np.kron(theta,np.eye(fd)),Pi=np.kron(pi,np.eye(fd)),kinetic=kinetic,psi=np.stack(vectors,axis=1),cutoff=cutoff,native_sites=sites,signed_paths=[len(x) for x in terms])


def observe(model: dict[str, Any], half: bool=False) -> dict[str, Any]:
    h=model['H']-(.5*model['kinetic'] if half else 0);x=model['Theta'];p=model['Pi'];j=model['J'];psi=model['psi'];c=model['C']
    e,v=eigh(h); xp=v.conj().T@psi; xt=v.conj().T@(x@psi)
    mass=float(np.linalg.norm(psi)**2); contact=float(np.trace(psi.conj().T@c@psi).real)
    a=h@x-x@h; b=h@p-p@h
    dotx=float(np.linalg.norm((1j*a-128*p)@psi));dotp=float(np.linalg.norm((1j*b-j)@psi))
    coefficient=float((np.trace(psi.conj().T@((h@(h@a-a@h)-(h@a-a@h)@h)@x-x@(h@(h@a-a@h)-(h@a-a@h)@h))@psi)).real)/6
    # Direct finite impulse derivative, evaluated through its mixed bilinear.
    rows=[]
    for t in (2.**-12,2.**-13):
        phase=np.exp(-1j*t*e); u=v@(phase[:,None]*xp); ut=v@(phase[:,None]*xt)
        d=float(-2*np.trace(u.conj().T@x@ut).imag)
        m=float(-2*np.trace(u.conj().T@j@ut).imag)
        pres=float(-2*np.trace(u.conj().T@p@ut).imag)
        free=128*t*mass; cubic=-8192/3*contact*t**3
        rows.append(dict(t=t,D=d,Pi_response=pres,mixed_M=m,free=free,cubic=cubic,residual=abs(d-free-cubic),wrong_contact_omitted=abs(d-free),wrong_contact_sign=abs(d-free+cubic)))
    return dict(mass=mass,contact=contact,cubic_coefficient=coefficient,canonical_cubic=-8192/3*contact,dotTheta_on_initial=dotx,dotPi_on_initial=dotp,rows=rows)


def run(species: list[Any],native: dict[str,Any]) -> dict[str,Any]:
    rows={}
    for source in (False,True):
        label='raw_source' if source else 'background';obs=[]
        for cutoff in (12,18):
            model=build(cutoff,source,species,native);good=observe(model);bad=observe(model,True) if cutoff==12 else None
            # Galerkin compression does not preserve canonical commutators.
            # Retain correct-law cutoff defects; compare the halfHE consumer.
            if bad is not None and bad['dotTheta_on_initial']<10*good['dotTheta_on_initial']+1:raise ArithmeticError('HalfHE control blind')
            for r in good['rows']:
                if r['wrong_contact_omitted']<4*r['residual'] or r['wrong_contact_sign']<8*r['residual']:raise ArithmeticError('Finite contact/sign control blind')
            obs.append(dict(cutoff=cutoff,native_sites=model['native_sites'],signed_paths=model['signed_paths'],correct=good,halfHE=bad))
        rows[label]=obs
    return dict(states=rows,scope='actual native N4 block coefficients/AP/one-two paths in declared finite three-site two-colour active+oneaux CAR/Hermite consumer; two Ref channels ONLY in control. Actual all129 source below has fourRef and63aux. Cutoffs12/18 observations are NOT convergence or measured full129 evolution.')
