"""Outward all129 nonstationary retarded ROW signal, unchanged body clock.

Finite initial CAR contractions only; no evolved Wick or stationary swap.
The scalar trace tests the whole kernel and has exact initial value zero.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
import zipfile

import mpmath as mp
import numpy as np
import sympy as sy
from threadpoolctl import threadpool_limits
import joined_phi_u1_body_word as body

ROOT = Path(__file__).resolve().parent
iv: Any = mp.iv
iv.dps = 60
sys.set_int_max_str_digits(0)
PIN = {
    body.ARCHIVE: body.ARCHIVE_SHA,
    'joined_phi_u1_body_word.py': '5ac906cb786f0e6283f3abd402548611f799d06bea77cb73fabec96505c3c7c4',
    'joined_phi_u1_actual_compact_source_action_certificate_verified_receipt_20261010.json': '2a1bbb46852707c82e477f28b5497e303b5ca66453db76d0655379f35cfb9a97',
}
CLOCK = F(1, 2**39)


def exact(x: Any) -> Any:
    q = x if isinstance(x, F) else F(x)
    return iv.mpf(q.numerator)/iv.mpf(q.denominator)


def end(x: Any, upper: bool) -> F:
    sign, mantissa, exponent, _ = x._mpi_[int(upper)]
    return F((-1 if sign else 1)*mantissa)*F(2)**exponent


def interval(x: Any) -> dict[str, str]:
    return {'lower': str(end(x, False)), 'upper': str(end(x, True))}


def up(x: float) -> float:
    if not math.isfinite(x) or x < 0:
        raise ArithmeticError('Invalid ball radius')
    return float(np.nextafter(x, math.inf))


def mag(x: np.ndarray) -> float:
    return up(float(np.max(np.abs(np.real(x))+np.abs(np.imag(x)))))


class Ball:
    """Complex entrywise l1-error ball; uniform radius, IEEE binary64 centre.

    A contraction of n complex products costs <= gamma_(16n) A B.
    For all n here, gamma_(16n) <= n*2^-48. Radius expressions use <32
    positive operations; multiplying by 1+2^-45 and nextafter bounds their
    evaluation rounding. This is a derived operation-count price, not a
    percentage allowance on any physical quantity. Max-l1 avoids sqrt.
    """
    def __init__(self, value: Any, radius: float = 0.):
        self.v = np.asarray(value, dtype=complex)
        self.r = radius

    def __getitem__(self, key: Any) -> Ball:
        return Ball(self.v[key], self.r)

    def __add__(self, other: Any) -> Ball:
        b = asball(other)
        return Ball(self.v+b.v, up((self.r+b.r+2**-48*(mag(self.v)+mag(b.v)))*(1+2**-45)))

    __radd__ = __add__

    def __neg__(self) -> Ball:
        return Ball(-self.v, self.r)

    def __sub__(self, other: Any) -> Ball:
        return self+-asball(other)

    def __rsub__(self, other: Any) -> Ball:
        return asball(other)+-self

    def __mul__(self, other: Any) -> Ball:
        b = asball(other)
        amax, bmax = mag(self.v), mag(b.v)
        r = amax*b.r+bmax*self.r+self.r*b.r+2**-48*amax*bmax
        return Ball(self.v*b.v, up(r*(1+2**-45)))

    __rmul__ = __mul__

    def __matmul__(self, other: Ball) -> Ball:
        return contract('...ij,...jk->...ik', self, other)

    def conj(self) -> Ball:
        return Ball(self.v.conj(), self.r)

    def transpose(self, *axes: int) -> Ball:
        return Ball(self.v.transpose(*axes), self.r)

    def reshape(self, *shape: int) -> Ball:
        return Ball(self.v.reshape(*shape), self.r)

    def reciprocal_real(self) -> Ball:
        x = float(np.real(self.v))
        if abs(float(np.imag(self.v))) > self.r or x <= self.r:
            raise ArithmeticError('Positive scalar denominator not proved')
        return fromiv(1/self.real_interval())

    def real_interval(self) -> Any:
        x = F.from_float(float(np.real(self.v)))
        return exact(x)+iv.mpf([-self.r, self.r])


def asball(x: Any) -> Ball:
    return x if isinstance(x, Ball) else Ball(x)


def fromiv(x: Any) -> Ball:
    lo, hi = end(x, False), end(x, True)
    centre = float((lo+hi)/2)
    error = max(abs(F.from_float(centre)-lo), abs(F.from_float(centre)-hi))
    return Ball(centre, up(float(error)))


def contract(spec: str, *args: Ball) -> Ball:
    """Conservative error for every einsum product, including sparse zeros."""
    left, right = spec.split('->')
    shapes: dict[str, int] = {}
    for labels, a in zip(left.split(','), args):
        if '...' in labels:
            labels = labels.replace('...', '')
            shape = a.v.shape[-len(labels):]
        else:
            shape = a.v.shape
        for c, size in zip(labels, shape):
            shapes[c] = size
    n = math.prod(size for c, size in shapes.items() if c not in right)
    if 16*n*2**-53 > .001:
        raise ArithmeticError('Contraction exceeds gamma simplification')
    total = exact(1)
    centre_bound = exact(1)
    for a in args:
        m = mag(a.v)
        total *= exact(F.from_float(m))+exact(F.from_float(a.r))
        centre_bound *= exact(F.from_float(m))
    err = n*(total-centre_bound)+exact(F(n,2**48))*centre_bound*len(args)
    radius = up(float(end(err, True)))
    return Ball(np.einsum(spec, *[a.v for a in args], optimize=False), radius)


def sumall(a: Ball) -> Ball:
    n = a.v.size
    return Ball(a.v.sum(), up((n*a.r+n*2**-48*mag(a.v))*(1+2**-45)))


def geometry(native: dict[str, Any], paths: list[dict[str, Any]]) -> dict[str, Any]:
    d = np.asarray(native['D'], dtype=np.int64)
    if not np.array_equal(d, native['D']):
        raise ArithmeticError('Nonintegral authoritative incidence')
    lap = sy.Matrix(d@d.T)
    c = sy.ones(64)/64
    lp = (lap+c).inv()-c
    if lap*lp != sy.eye(64)-c or lp != lp.T:
        raise ArithmeticError('Exact all63 density pseudoinverse failed')
    den = int(sy.ilcm(*[v.q for v in lp]))
    lint = np.array([[int(v*den) for v in row] for row in lp.tolist()], dtype=np.int64)
    k0 = np.zeros((len(paths),192), np.int64)
    for i,p in enumerate(paths):
        for e in p['edges']:
            k0[i,e] += 1
    k = np.r_[k0,-k0]
    x = np.array([p['source'] for p in paths]+[p['target'] for p in paths])
    y = np.array([p['target'] for p in paths]+[p['source'] for p in paths])
    delta = np.eye(64,dtype=np.int64)[x]-np.eye(64,dtype=np.int64)[y]
    if not np.array_equal(d@k.T,delta.T):
        raise ArithmeticError('Actual path ROW Ward orientation failed')
    dens = delta@lint@delta.T
    gram = den*(k@k.T)-dens
    # P=I-D^T L+ D: exact rank192-63=129, including all three torons.
    return dict(den=den, lint=lint, k=k, x=x, y=y, delta=delta,
                density=dens, gram=gram, lp=lp)


def rational_array(a: np.ndarray, den: int) -> Ball:
    values = a/den
    error = max((abs(F.from_float(float(v))-F(int(n),den)) for n,v in zip(a.flat,values.flat)), default=F(0))
    return Ball(values,up(float(error)))


def characters(nums: np.ndarray, den: int, variance: int = 2) -> Ball:
    entries = {int(n): fromiv(iv.exp(-exact(F(variance*int(n),2*den)))) for n in np.unique(nums)}
    return Ball(np.array([entries[int(n)].v.item() for n in nums.flat]).reshape(nums.shape),
                max(e.r for e in entries.values()))


def exact_species(profile: dict[str, Any], native: dict[str, Any], saved: dict[str, Any],
                  paths: list[dict[str, Any]]) -> tuple[Ball, Ball, Ball]:
    gamma = np.asarray(body.joined.old.full.supplier.primitive.SELECTED)
    if not np.all(np.isin(np.real(gamma), [-1,0,1])) or not np.all(np.isin(np.imag(gamma),[-1,0,1])):
        raise ArithmeticError('Clifford matrix entries not Gaussian integers')
    for m in range(4):
        for n in range(4):
            if not np.array_equal(gamma[m]@gamma[n]+gamma[n]@gamma[m], (2 if m==n else 0)*np.eye(4)):
                raise ArithmeticError('Exact Clifford law failed')
    g = [exact(F(saved['centre_exact_rational'][0]))]+[exact(F(saved['centre_exact_rational'][1]))]*3
    frame = [fromiv(iv.sqrt(v))*Ball(gamma[i]) for i,v in enumerate(g)]
    lookup = {(p['axis'],p['length']):p for p in profile['paths']}
    forward = []
    for p in paths:
        s = lookup[p['axis'],p['length']]
        forward.append((frame[0]*fromiv(exact(F(s['c0'])))+frame[p['axis']+1]*fromiv(exact(F(s['ci'])))*1j)*p['seam'])
    b = Ball(np.array([v.v for v in forward]),max(v.r for v in forward))
    b = Ball(np.concatenate((b.v,b.v.conj().transpose(0,2,1))),b.r)
    onsite = frame[0]*fromiv(exact(F(profile['onsite'])))
    # AP Bloch P-=(1-h/E)/2, all64 momenta; Fourier uses ROW x-y.
    projectors = []
    momenta = list(product(range(4),repeat=3))
    for ks in momenta:
        coeff = [exact(F(profile['onsite'])),exact(0),exact(0),exact(0)]
        for s in profile['paths']:
            angle = (2*ks[s['axis']]+1)*iv.pi*s['length']/4
            coeff[0] += 2*exact(F(s['c0']))*iv.cos(angle)
            coeff[s['axis']+1] -= 2*exact(F(s['ci']))*iv.sin(angle)
        energy = iv.sqrt(sum((g[i]*coeff[i]**2 for i in range(4)),exact(0)))
        if end(energy,False)<=0:
            raise ArithmeticError('All64 Bloch gap not proved')
        h = sum((frame[i]*fromiv(coeff[i]/energy) for i in range(4)),Ball(np.zeros((4,4))))
        projectors.append((Ball(np.eye(4))-h)*.5)
    sites = native['sites']
    pb = np.empty((64,64,4,4),complex)
    radius = 0.
    cache: dict[tuple[int,...],Ball] = {}
    for x in range(64):
        for y in range(64):
            displacement=tuple(int(v) for v in sites[x]-sites[y])
            if displacement not in cache:
                v = Ball(np.zeros((4,4)))
                for ks,p in zip(momenta,projectors):
                    angle = iv.pi*sum((2*ks[a]+1)*displacement[a] for a in range(3))/4
                    phase = fromiv(iv.cos(angle))+1j*fromiv(iv.sin(angle))
                    v += phase*p*F(1,64)
                cache[displacement]=v
            pb[x,y]=cache[displacement].v
            radius=max(radius,cache[displacement].r)
    p=Ball(pb.transpose(0,2,1,3).reshape(256,256),radius)
    return b,onsite,p


def pair(b: Ball, p: Ball, geo: dict[str, Any]) -> tuple[Ball,Ball]:
    x,y=geo['x'],geo['y']
    pb=p.reshape(64,4,64,4).transpose(0,2,1,3)
    mean=contract('rij,rji->r',b,pb[y,x])
    cov=np.empty((len(x),len(x)),complex)
    radius=0.
    for r in range(len(x)):
        subtract=contract('ij,sjk,skl,sli->s',b[r],pb[y[r],x],b,pb[y,x[r]])
        direct=contract('ij,sjk,ski->s',b[r],b,pb[y,x[r]])
        row=direct*Ball(y[r]==x)-subtract
        cov[r]=row.v
        radius=max(radius,row.r)
    return mean,Ball(cov,radius)


def outer(a: Ball,b: Ball) -> Ball:
    return a.reshape(-1,1)*b.reshape(1,-1)


def conditional(p: Ball, aux: Ball, bs: list[Ball], geo: dict[str, Any],
                ref: bool, load: int=63) -> tuple[Ball,Ball,Ball,Ball]:
    ma,ca=pair(bs[0],p,geo)
    mb,cb=pair(bs[1],aux,geo)
    mean=ma+load*mb
    second=outer(mean,mean)+ca+load*cb
    x,y=geo['x'],geo['y']
    lp=rational_array(geo['lint'],geo['den'])
    delta=Ball(geo['delta'])
    density=contract('xixi->x',p.reshape(64,4,64,4))+load*contract('xixi->x',aux.reshape(64,4,64,4))-2*(1+load)
    if ref:
        density=density-Ball(np.eye(64)[0])
    vr=2*contract('ri,ij->rj',delta,lp)
    diag=rational_array(np.diag(geo['density']),geo['den'])
    fmean=contract('ri,i->r',vr,density)+diag
    hc=2*rational_array(geo['density'],geo['den'])*second
    extra=np.zeros_like(hc.v)
    er=0.
    for b,pp,multiplicity in ((bs[0],p,1),(bs[1],aux,load)):
        pb=pp.reshape(64,4,64,4).transpose(0,2,1,3)
        for r in range(len(x)):
            vv=Ball(np.repeat(vr[r].v,4),vr.r)
            weighted=vv.reshape(-1,1)*pp
            t=weighted-contract('ij,jk->ik',pp,weighted)
            tb=t.reshape(64,4,64,4).transpose(0,2,1,3)
            for s in np.flatnonzero((y[r]==x)|(y==x[r])):
                mm=Ball(0);cc=Ball(0)
                if y[r]==x[s]:
                    c=contract('ij,jk->ik',b[r],b[s])
                    mm+=contract('ij,ji->',c,pb[y[s],x[r]])
                    cc+=contract('ij,ji->',c,tb[y[s],x[r]])
                if y[s]==x[r]:
                    c=contract('ij,jk->ik',b[s],b[r])
                    mm-=contract('ij,ji->',c,pb[y[r],x[s]])
                    cc-=contract('ij,ji->',c,tb[y[r],x[s]])
                value=multiplicity*(mm*fmean[r]+cc)
                extra[r,s]+=value.v
                er=max(er,value.r*(1+load)+2**-40*mag(np.asarray(extra[r,s])))
    hc=64*(hc+Ball(extra,up(er)))
    return mean,second,hc,outer(ma,ma)+load*outer(mb,mb)


def matter_trace(b: Ball,onsite: Ball,p: Ball,geo: dict[str,Any]) -> Ball:
    x=np.r_[geo['x'],np.arange(64)];y=np.r_[geo['y'],np.arange(64)]
    blocks=Ball(np.concatenate((b.v,np.broadcast_to(onsite.v,(64,4,4)))),max(b.r,onsite.r))
    k=np.r_[geo['k'],np.zeros((64,192),np.int64)]
    delta=np.eye(64,dtype=np.int64)[x]-np.eye(64,dtype=np.int64)[y]
    gram=geo['den']*(k@k.T)-delta@geo['lint']@delta.T
    pb=p.reshape(64,4,64,4).transpose(0,2,1,3)
    outgoing=[np.flatnonzero(x==site) for site in range(64)]
    total=Ball(0)
    charcache: dict[int,Ball]={}
    for a in range(len(x)):
        for bb in outgoing[y[a]]:
            ab=contract('ij,jk->ik',blocks[a],blocks[bb])
            for c in outgoing[y[bb]]:
                num=int(gram[a,a]+gram[bb,bb]+gram[c,c]+2*(gram[a,bb]+gram[a,c]+gram[bb,c]))
                if num not in charcache:
                    charcache[num]=fromiv(iv.exp(-exact(F(num,geo['den']))))
                coef=int(-gram[bb,c]+2*gram[a,c]-gram[bb,a])
                if coef:
                    trace=contract('ij,jk,ki->',ab,blocks[c],pb[y[c],x[a]])
                    total+=trace*charcache[num]*fromiv(exact(F(coef,geo['den'])))
    return total


def moments(mean: Ball,second: Ball,hc: Ball,matter: Ball,geo: dict[str,Any],
            own_disconnected: Ball,independent: bool=False,pi: bool=True) -> dict[str,Any]:
    den=geo['den'];gi=geo['gram'];g=rational_array(gi,den)
    chars=characters(np.diag(gi)[:,None]+np.diag(gi)[None,:]+2*gi,den)
    if independent:
        one=characters(np.diag(gi),den)
        cross=outer(mean,mean)-own_disconnected
        adjusted=(outer(one,one)-chars)*cross
    else:
        adjusted=Ball(np.zeros_like(second.v))
    jj=-sumall(g*(chars*second+adjusted))
    electric=-64*sumall(g*g*(chars*(second+second.transpose(1,0))+2*adjusted)) if pi else Ball(0)
    coul=-sumall(g*(chars*hc+128*rational_array(geo['density'],den)*adjusted))
    f=electric+coul+matter
    return dict(Pi=electric.real_interval(),HC=coul.real_interval(),matter=matter.real_interval(),
                F=f.real_interval(),JJ=jj.real_interval())


def ref_trace_contrast(effective: Ball,aux: Ball,bs: list[Ball],geo: dict[str,Any]) -> Any:
    """Independent exact omitted-Ref contrast, before interval subtraction.

    HCwithout-HCphysical=64 [A_r,A_s] (2delta_r L e0).
    This linear CAR expression retains the actual raw four-channel effective
    projector, all auxiliary copies, and the same shared Gaussian character.
    It avoids the artificial loss from subtracting two broad HC intervals.
    """
    x,y=geo['x'],geo['y'];den=geo['den'];gi=geo['gram']
    g=rational_array(gi,den)
    chars=characters(np.diag(gi)[:,None]+np.diag(gi)[None,:]+2*gi,den)
    vr=rational_array(2*geo['delta']@geo['lint'][:,0],den)
    total=Ball(0)
    for b,pp in ((bs[0],effective),(bs[1],F(63,2)*aux)):
        pb=pp.reshape(64,4,64,4).transpose(0,2,1,3)
        for r in range(len(x)):
            for s in np.flatnonzero((y[r]==x)|(y==x[r])):
                c=Ball(0)
                if y[r]==x[s]:
                    c+=contract('ij,jk,ki->',b[r],b[s],pb[y[s],x[r]])
                if y[s]==x[r]:
                    c-=contract('ij,jk,ki->',b[s],b[r],pb[y[r],x[s]])
                total+=64*g[r,s]*chars[r,s]*c*vr[r]
    return total.real_interval()


def remainder(profiles: list[dict[str,Any]],geo: dict[str,Any],mass: F,source: bool,jj: Any) -> dict[str,Any]:
    """New directed evaluation of local-shift/conserved-energy inequalities."""
    lp=geo['lp'];den=geo['den'];gi=geo['gram'];di=geo['density']
    tr=exact(F(str(sy.trace(lp))));tr2=exact(F(str(sy.trace(lp*lp))));diag=exact(F(str(lp[0,0])))
    count=8192+int(source);n=iv.sqrt(exact(mass));a=exact(int(source))
    fourth=3*a*a+(6*128*tr+12*128*exact(F(1,2)))*a+3*((128*tr)**2+2*128**2*tr2)+24*iv.sqrt(15*a)*count*diag**exact(F(3,2))+78*count*diag**2
    hcmean=64*(128*tr+a);hcnorm=64*iv.sqrt(fourth/3)
    ops=[];bvals=[];hmax=[]
    for profile in profiles:
        look={(s['axis'],s['length']):abs(F(s['c0']))+abs(F(s['ci'])) for s in profile['paths']}
        paths=body.joined.old.native_paths(profile,geo['native'])
        op=[look[p['axis'],p['length']] for p in paths]*2
        ops.append(op);bvals.append([4*v for v in op])
        hmax.append(abs(F(profile['onsite']))+2*sum(look.values()))
    m=exact((128+int(source))*hmax[0]+63*128*hmax[1])
    e1=exact(mass)*(1032+hcmean+2*m)
    e2=n*(8*iv.sqrt(exact(129*131))+hcnorm+2*m)
    grad=iv.sqrt(e1/64);root=iv.sqrt(e1)
    def v1(d: Any) -> Any:
        return 2*iv.sqrt(64*d)*root+64*d*n
    allb=[bvals[0][r]+63*bvals[1][r] for r in range(len(gi))]
    eg=[sum((abs(F(int(v),den))*allb[s] for s,v in enumerate(row)),F(0)) for row in gi]
    dg=[sum((abs(F(int(v),den))*allb[s] for s,v in enumerate(row)),F(0)) for row in di]
    x,y=geo['x'],geo['y'];rx=np.r_[x,np.arange(64)];ry=np.r_[y,np.arange(64)]
    k=np.r_[geo['k'],np.zeros((64,192),np.int64)]
    delta=np.eye(64,dtype=np.int64)[rx]-np.eye(64,dtype=np.int64)[ry]
    lint=geo['lint']
    totals={key:exact(0) for key in ('EE','EC','CC','EM','CM','MM')}
    for profile,op,b,h,load in zip(profiles,ops,bvals,hmax,(1,63)):
        oo=op+[abs(F(profile['onsite']))]*64
        for r in range(len(gi)):
            q2=exact(F(int(gi[r,r]),den));q=iv.sqrt(q2);d=exact(F(int(di[r,r]),den));br=exact(b[r])
            vals=dict(EE=br*(64**2*q2**2*n+4*64**2*q**3*grad+4*64*q2*e2),
                EC=2*64*br*(q2*v1(d)+2*q*(iv.sqrt(d)*e2+64*d*grad)),
                CC=br*(4*64*d*e2+4*(64*d)**exact(F(3,2))*root+(64*d)**2*n),
                EM=2*64*br*exact(eg[r])*n,CM=2*64*br*exact(dg[r])*n,
                MM=4*exact(h)**2*br*n)
            for e in np.flatnonzero((ry==x[r])|(y[r]==rx)):
                for xx,yy,present in ((rx[e],y[r],ry[e]==x[r]),(x[r],ry[e],y[r]==rx[e])):
                    if not present:
                        continue
                    dd=np.eye(64,dtype=np.int64)[xx]-np.eye(64,dtype=np.int64)[yy]
                    ds=exact(F(int(dd@lint@dd),den))
                    def norm2(ke: np.ndarray,de: np.ndarray) -> Any:
                        return exact(F(int(den*(ke@ke)-de@lint@de),den))
                    qs=norm2(k[e]+k[r],delta[e]+delta[r]);qd=iv.sqrt(norm2(k[e]+2*k[r],delta[e]+2*delta[r]))
                    c=exact(4*oo[e]*op[r])
                    vals['EM']+=c*(64*(q2+qs)*n+2*64*qd*grad)
                    vals['CM']+=c*(v1(d)+v1(ds))
            for name,val in vals.items():
                totals[name]+=load*q*val
    bstar=sum(totals.values(),exact(0));b0=iv.sqrt(iv.mpf([0,1])*exact(end(jj,True)))
    coefficient=bstar*b0
    return dict(parts={k:interval(v) for k,v in totals.items()},Bstar=interval(bstar),
                HC_mean=interval(hcmean),HC_graph_norm=interval(hcnorm),energy_first=interval(e1),energy_graph_norm=interval(e2),
                current_inserted_norm=interval(b0),coefficient=interval(coefficient),upper=end(coefficient,True)*CLOCK**2)


PROOF = {
 'current_definition':'H=64Pi^2+64rho Lplus rho+sum_active_and_63aux dGamma(h); J_mu=-partial_mu H; C_mu_nu=partial_mu partial_nu H. R_mu_nu(t,0)=-i raw<[exp(iHt)J_mu exp(-iHt),J_nu]>. ROW evolves; source is nonstationary. F_mu_nu=<[[H,J_mu],J_nu]>. No stationarity or channel interchange.',
 'trace_signal':'TrR(0)=sum_mu -i<[J_mu,J_mu]>=0 EXACT. For arbitrary R0 offdiagonal, |TrR(t)|>=t|TrF|-sqrt129 ||Taylor remainder||HS. Thus ||R(t)||HS>=t|TrF|/sqrt129-remainder; no claim whole initial commutator vanishes.',
 'initial_contractions':'P-=(I-h/E)/2 for all64 AP momenta in exact Clifford basis. Gaussian characters exp(-2 kPk/2), P=I-D^T Lplus D exact rational rank129. Four orthogonal Ref3-hole Slaters weighted ||(I-P)e_alpha||^2/4, raw trace1/2; auxiliary connected63 and disconnected63^2 retained at the SAME sharedTheta. Initial finite CAR identities only.',
 'Pi':'Gaussian densityvariance2 and HE64: integration by parts yields FPi=-64 sum_a <{partial_a J_mu,partial_a J_nu}>; evaluated before averaging with full sharedTheta.',
 'HC':'[rho,A_r]=delta_r A_r; [[64rhoLrho,A_r],A_s]=64[A_r,A_s](2delta_r Lrho+delta_r Ldelta_r)+128(delta_r Ldelta_s)A_r A_s. Density covariance Tr(C(I-P)diag(v)P), full64 sites/63 density modes. Refhole -delta0 remains.',
 'remainder':'ad_H^2 J graph bound is derived per actual charge-shift path. E=T+HC positive commuting; conserved H and bounded matter imply <E>t<=<E>0+2m mass, ||E Upsi||<=||Tpsi||+||HCpsi||+2m norm. Local V_r=128delta Lrho+64deltaLdelta, ||V_r phi||<=2sqrt(64d)sqrt(E1)+64d norm; ||V_r^2 phi||<=256d E2+4(64d)^(3/2)sqrt(E1)+(64d)^2norm. EE/EC/CC and EM/CM/MM triangle terms enumerate all signed paths and connected local chains; all-flavour products retain shared fields. Whole-vector triangle bounds Bstar and ||Rsecond||HS<=2Bstar b0 give t^2 Bstar b0. No future Wick/global Hcop.',
 'initial_density_upper':'Site Slater density covariance<=2I, sum64<=128I. Tr<=4096. Source meanrho=u-delta0 gives aLa<=1. Gaussian z with cov L yields Ez(z.rho)^4=3E(rhoLrho)^2. Slater cumulants |k4|<=26TrD^4P, |k3|<=6Tr|D|^3P by Schatten Holder; evaluated fourth-moment polynomial retains source mean, all8192/8193 particles, all Ltrace/L2/diagonal. Tmean1032 and Tnorm8sqrt(129*131) exact.',
 'arithmetic':'mpmath directed interval60decimal scalar roots/trig/exp and positive graph prices. Complex binary64 ball contractions use derived gamma_(16n)<=n2^-48, input rational conversion error exact, uniform entrywise l1 radii; no FLOAT1percent first-moment premise. Proof conditional on IEEE binary64 round-to-nearest basic operations and NumPy einsum finite-product arithmetic; no eigenbasis Q/eigensolver projector enters proof.',
 'scalar_law':'Microscopic onsite,c0,ci are exact rational encodings of original binary scalar_paths outputs, not their shortened decimal displays. Saved centre_exact_rational is the immutable body fixedmean law, not a newly adopted root. OriginalSOURCE/current427 unchanged.',
 'conditions':'Conditional pinned current exact mean/Clifford/physical branch; direct rational fixedmean and static Bloch seas are not interacting Coulomb ground. No measured full129 evolution, original source admission, ground/pole/dispersion/commoncone promotion.'}


def run(output: Path) -> dict[str,Any]:
    start=time.monotonic();own=Path(__file__).read_bytes()
    captured={**body.CAPTURED,Path(__file__).name:own}
    for name,sha in PIN.items():
        raw=(ROOT/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=sha:
            raise ArithmeticError('Pinned source/input changed '+name)
        captured[name]=raw
    saved=json.loads(captured[body.ARCHIVE])['input_descriptor']['exact_NEW_allq_BO_mean']
    # Fixedmean loader only, never solve_mean/load_quantum_mean or old producers.
    centres=[F(x) for x in saved['centre_exact_rational']]
    data=body.joined.old.load_mean(g_override=[centres[0],centres[1],centres[1],centres[1]])
    profiles=data['profiles'][1:];native=data['native']
    paths=body.joined.old.native_paths(profiles[0],native)
    geo=geometry(native,paths);geo['native']=native
    for desc in ('root_receipt','full_receipt'):
        p=Path(data['descriptor'][desc]['path']);captured[p.name]=p.read_bytes()
    print('exact129 geometry derived',flush=True)
    species=[exact_species(p,native,saved,paths) for p in profiles]
    bs=[s[0] for s in species];p=species[0][2];aux=species[1][2]
    bg=conditional(p,aux,bs,geo,False)
    mb=matter_trace(bs[0],species[0][1],p,geo)+63*matter_trace(bs[1],species[1][1],aux,geo)
    background=moments(bg[0],bg[1],bg[2],mb,geo,bg[3])
    print('outward background trace derived',flush=True)
    empty=Ball(np.eye(256))-p
    accum=[Ball(np.zeros_like(v.v)) for v in bg]
    effective=Ball(np.zeros((256,256)))
    weights=[]
    for alpha in range(4):
        v=empty[:,alpha]
        norm=contract('i,i->',v.conj(),v)
        weight=norm*.25
        pp=p+outer(v,v.conj())*norm.reciprocal_real()
        mm=conditional(pp,aux,bs,geo,True)
        for i in range(4):
            accum[i]+=weight*mm[i]
        effective+=weight*pp
        weights.append(interval(weight.real_interval()))
        print('outward raw Ref channel',alpha,flush=True)
    # Exact total source mass: sumalpha ||(1-P)e0alpha||²/4=(4-TrP00)/4=1/2
    # each Bloch P has trace2, hence TrP00=2, independent of eigenbasis.
    ms=matter_trace(bs[0],species[0][1],effective,geo)+F(63,2)*matter_trace(bs[1],species[1][1],aux,geo)
    raw=moments(accum[0],accum[1],accum[2],ms,geo,accum[3])
    rows={}
    for state,mass,source,mom in (('background',F(1),False,background),('raw_source',F(1,2),True,raw)):
        f=mom['F'];lo,hi=end(f,False),end(f,True)
        if lo<=0<=hi:
            raise ArithmeticError('Trace signal interval includes zero')
        abslo=min(abs(lo),abs(hi))
        signal=exact(CLOCK*abslo)/iv.sqrt(exact(129))
        rem=remainder(profiles,geo,mass,source,mom['JJ'])
        lower=end(signal,False)-rem['upper']
        phi=F('1.9697e-134') if not source else F('9.8486e-135')
        if lower<=phi:
            raise ArithmeticError('Whole129 consumer assertion fails '+state)
        rows[state]=dict(raw_norm_squared=str(mass),TrF=interval(f),parts={k:interval(mom[k]) for k in ('Pi','HC','matter')},
            JJ_trace=interval(mom['JJ']),signal_lower=str(end(signal,False)),remainder_upper=str(rem['upper']),
            body_HS_lower=str(lower),Phi_transfer_error_upper=str(phi),post_Phi_HS_lower=str(lower-phi),
            remainder_derivation={k:v for k,v in rem.items() if k!='upper'})
    # Independent actual physical variants reach the same moment consumer.
    wrongaux=conditional(p,aux,bs,geo,False,load=1)
    wrongmatter=matter_trace(bs[0],species[0][1],p,geo)+matter_trace(bs[1],species[1][1],aux,geo)
    ref_defect=ref_trace_contrast(effective,aux,bs,geo)
    controls={
        'Pi_omitted':(background,moments(bg[0],bg[1],bg[2],mb,geo,bg[3],pi=False)),
        'auxiliary_load_1':(background,moments(wrongaux[0],wrongaux[1],wrongaux[2],wrongmatter,geo,wrongaux[3])),
        'independent_Theta_copies':(background,moments(bg[0],bg[1],bg[2],mb,geo,bg[3],independent=True)),
        'Ref_density_omitted':(raw,{'F':raw['F']-ref_defect})}
    adverse={}
    for name,(good,bad) in controls.items():
        defect=ref_defect if name=='Ref_density_omitted' else good['F']-bad['F']
        lo,hi=end(defect,False),end(defect,True)
        if lo<=0<=hi:
            raise ArithmeticError('Same consumer physical control blind '+name)
        adverse[name]=dict(TrF=interval(bad['F']),difference=interval(defect),
                          reaches='same actual129 physical TrF assertion; independently derived physical difference interval excludes zero, with common unchanged terms cancelled algebraically')
    capture=output.with_name('joined_phi_u1_current_signal_interval_certificate_input_bytes.zip')
    with zipfile.ZipFile(capture,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,rawbytes in sorted(captured.items()):
            archive.writestr(name,rawbytes)
    for name,rawbytes in captured.items():
        if (ROOT/name).read_bytes()!=rawbytes:
            raise ArithmeticError('Source/input changed during consumer '+name)
    result=dict(status='conditional outward whole129 retarded ROW signal at unchanged2^-39; no floating1percent premise',
        governing_plan='issue-222-induced-photon-delta13-continuation-20261008',
        assignment_revision='712b1723921b52c5cc7b0fced4b6428a6c5b37ad6278d074bae0895dc40a1416',
        work_item='induced-photon/verification',clock=str(CLOCK),coordinates=129,charged_flavours=64,auxiliary_load=63,
        current_definition=PROOF['current_definition'],proofconditions=PROOF,all129_states=rows,
        source_four_channel_weights=weights,source_total_raw_mass='1/2',diagonal_initial_trace='0 exact; offdiagonal not asserted zero',
        exact_geometry=dict(rank=129,torons=3,all_density_modes=63,denominator=geo['den'],L_trace=str(sy.trace(geo['lp'])),L2_trace=str(sy.trace(geo['lp']**2))),
        adverse_physical_controls=adverse,input_hashes={name:hashlib.sha256(rawbytes).hexdigest() for name,rawbytes in captured.items()},
        captured_input_bytes=dict(path=capture.name,sha256=hashlib.sha256(capture.read_bytes()).hexdigest()),
        runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,mpmath=mp.__version__,interval_decimal_precision=iv.dps),
        elapsed_seconds=time.monotonic()-start)
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        receipt=run(args.output)
    print(json.dumps({'status':receipt['status'],'elapsed_seconds':receipt['elapsed_seconds']}),flush=True)
