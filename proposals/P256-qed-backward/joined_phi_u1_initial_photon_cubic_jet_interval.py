"""Outward INITIAL cubic mixed jet on the immutable fixed-mean BODY law.

No numeric coefficient producer or primary consumer is rerun. Exact scalar
root/trig/exp supplier ar.exact_species is the fixed-mean BODY antecedent
(ar.PROOF.scalar_law), not a new root. Full129 output is an entrywise l1 Ball
in a certified exact orthonormal physical frame anchored to literal Qobs.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, cast
import numpy as np
from threadpoolctl import threadpool_limits
import joined_phi_u1_current_signal_interval_certificate as ar
import joined_phi_u1_initial_car_word_algebra as car
import joined_phi_u1_initial_photon_cubic_jet as conditional
import joined_phi_u1_body_word as body

ROOT=Path(__file__).resolve().parent
PREFIX='joined_phi_u1_initial_photon_cubic_jet_interval'
PROOF={
 'law':'Saved centre_exact_rational is the immutable declared BODY fixed-mean law (ar.PROOF.scalar_law), not a newly adopted root. Exact all64 AP Bloch supplier returns Ball projectors/hops/onsite. All63 density modes, sharedTheta and Ref4to3 retained. No future Wick.',
 'algebra':{key:value for key,value in conditional.PROOF.items() if key!='error'},
 'arithmetic':'All products/contractions use ar.Ball/ar.contract: derived complex binary64 gamma_(16n)<=n*2^-48, interval60decimal scalar prices. Native integer winds and rational Gauss Gram/7680 precede Gaussian characters. Each summation is explicitly priced, not an empirical percentage.',
 'aggregation':'For n signed sparse additions z_l*k_ai*k_bj, integer native k has |k|<=1. Uniform entry radius increment n*r_z + n*2^-48*(max_l1(previous centre)+n*max_l1(z)); derived from gamma16n. Positive radius evaluation uses outward interval arithmetic. No input-dependent cancellation shrinks this price.',
 'frame':'P_T=I-D^T Lplus D is exact rational. B=P_T Qobs, G=B^T B. Outward eta>=||G-I||2, epsilon>=||B-Qobs||2. eta<1 proves Q*=B G^-1/2 is exact orthonormal full129 in ran(P_T). Entrywise ||Q*-Qobs|| bound epsilon+sqrt(1+eta)*((1-eta)^(-1/2)-1) encloses literal-frame correction, without trusting numeric eigenvectors as exact.',
 'scope':'Initial C0,M3,D5=128M3/120,Pi4=M3/24 outward at actual fixed-mean BODY law. Nonstationary antisymmetric channel not symmetrized. No time evolution, finite-time kernel, physical transfer, pole/ground/source adoption. Parent owns tails and admission.'}


def trace(a:ar.Ball,b:ar.Ball)->ar.Ball:
    return ar.contract('...ij,...ji->...',a,b)


def lifted(value:np.ndarray,radius:float)->ar.Ball:
    return ar.Ball(value,radius)


def diag(a:ar.Ball)->ar.Ball:
    return lifted(np.diag(a.v),a.r)


def repeat_sites(a:ar.Ball,spin:int)->ar.Ball:
    return lifted(np.repeat(a.v,spin,axis=-1),a.r)


def sparse_add(out:ar.Ball,k:np.ndarray,aa:np.ndarray,bb:np.ndarray,z:ar.Ball)->ar.Ball:
    """Literal sparse signed accumulation with a proved positive roundoff price."""
    # Native one/two paths have at most2 nonzero distinct edges, all +/-1.
    slots=np.full((len(k),2),-1,int)
    for r,row in enumerate(k):
        edges=np.flatnonzero(row)
        if len(edges)>2:raise ArithmeticError('Non-native derivative wind')
        slots[r,:len(edges)]=edges
    n=0;previous=ar.mag(out.v);zm=ar.mag(z.v)
    for slot_a in range(2):
        za=slots[aa,slot_a]
        for slot_b in range(2):
            zb=slots[bb,slot_b]
            good=(za>=0)&(zb>=0)
            if not np.any(good):continue
            signs=k[aa[good],za[good]]*k[bb[good],zb[good]]
            if not np.all(abs(signs)==1):raise ArithmeticError('Non-native sparse derivative coefficient')
            values=z.v[good]*signs
            np.add.at(out.v,(za[good],zb[good]),values)
            n+=int(np.count_nonzero(good))
    if 16*n*2**-53>.001:raise ArithmeticError('Sparse aggregation exceeds gamma simplification')
    radius=ar.exact(F.from_float(out.r))+n*ar.exact(F.from_float(z.r))+ar.exact(F(n,2**48))*(ar.exact(F.from_float(previous))+n*ar.exact(F.from_float(zm)))
    return lifted(out.v,ar.up(float(ar.end(radius,True))))


class IntervalEngine:
    def __init__(self,geo:dict[str,Any],bs:list[ar.Ball],ons:list[ar.Ball],nsite:int=64,spin:int=4,copies:int=63):
        self.k=geo['k'];self.x=geo['x'];self.y=geo['y'];self.delta=geo['delta'];self.gi=geo['gram'];self.den=geo['den']
        self.h=len(self.x);self.n=nsite;self.spin=spin;self.dim=nsite*spin;self.copies=copies
        self.edge=self.k.shape[1];self.b=bs;self.on=ons
        self.vi=self.delta@geo['lint'];self.v=ar.rational_array(self.vi,self.den)
        self.gram=ar.rational_array(self.gi,self.den);self.q2=diag(self.gram)
        self.overlap=self.k@self.k.T
        self.onechar=ar.characters(np.diag(self.gi),self.den)
        self.twochar=ar.characters(np.diag(self.gi)[:,None]+np.diag(self.gi)[None,:]+2*self.gi,self.den)
        self.pair_records=[]
        for a in range(self.h):
            for b in np.flatnonzero((self.y[a]==self.x)|(self.y==self.x[a])):
                if self.y[b]==self.x[a]:self.pair_records.append((a,int(b),int(self.x[b]),int(self.y[a]),1,0))
                if self.y[a]==self.x[b]:self.pair_records.append((a,int(b),int(self.x[a]),int(self.y[b]),-1,1))
        self.records=np.array(self.pair_records,int)
        xx=np.r_[self.x,np.arange(self.n)];yy=np.r_[self.y,np.arange(self.n)]
        outgoing=[np.flatnonzero(xx==s) for s in range(self.n)]
        self.chains=np.array([(a,int(b),int(c)) for a in range(len(xx)) for b in outgoing[yy[a]] for c in outgoing[yy[b]]],int)
        self.tx=xx;self.ty=yy
        self.tk=np.r_[self.k,np.zeros((self.n,self.edge),int)]
        self.tgram=np.zeros((self.h+self.n,)*2,np.int64);self.tgram[:self.h,:self.h]=self.gi
        a,b,c=self.chains.T
        self.triplechar=ar.characters(self.tgram[a,a]+self.tgram[b,b]+self.tgram[c,c]+2*(self.tgram[a,b]+self.tgram[a,c]+self.tgram[b,c]),self.den)

    def blocks(self,p:ar.Ball)->ar.Ball:
        return p.reshape(self.n,self.spin,self.n,self.spin).transpose(0,2,1,3)

    def density(self,p:ar.Ball)->tuple[ar.Ball,ar.Ball]:
        d=diag(p)
        covariance=lifted(np.diag(d.v),d.r)-p*p.conj()
        return d,covariance

    def linear(self,block:ar.Ball,xx:np.ndarray,yy:np.ndarray,p:ar.Ball,v:ar.Ball,square:bool=False)->tuple[ar.Ball,ar.Ball,ar.Ball|None]:
        pb=self.blocks(p);pr=lifted(p.v.reshape(self.n,self.spin,self.dim)[yy],p.r)
        pc=lifted(p.v.reshape(self.dim,self.n,self.spin).transpose(1,0,2)[xx],p.r)
        vv=repeat_sites(v,self.spin)
        vp=vv.reshape(len(xx),self.dim,1)*pc
        mean=trace(block,pb[yy,xx])
        vy=lifted(v.v[np.arange(len(yy)),yy],v.r)
        cov=trace(block,vy.reshape(len(yy),1,1)*pb[yy,xx]-ar.contract('nij,njk->nik',pr,vp))
        if not square:return mean,cov,None
        identity=np.eye(self.dim).reshape(self.n,self.spin,self.dim)[yy]
        qrow=ar.Ball(identity)-pr
        pvp=ar.contract('ij,njk->nik',p,vp)
        third=trace(block,ar.contract('nij,njk->nik',qrow*vv.reshape(len(xx),1,self.dim),vp-2*pvp))
        return mean,cov,third

    def pair(self,b:ar.Ball,p:ar.Ball)->tuple[ar.Ball,ar.Ball]:
        pb=self.blocks(p);mean=ar.contract('rij,rji->r',b,pb[self.y,self.x])
        centre=np.empty((self.h,self.h),complex);radius=0.
        for r in range(self.h):
            subtract=ar.contract('ij,sjk,skl,sli->s',b[r],pb[self.y[r],self.x],b,pb[self.y,self.x[r]])
            direct=ar.contract('ij,sjk,ski->s',b[r],b,pb[self.y,self.x[r]])
            value=ar.Ball(self.y[r]==self.x)*direct-subtract
            centre[r]=value.v;radius=max(radius,value.r)
        return mean,lifted(centre,radius)

    def state(self,p:ar.Ball,aux:ar.Ball,hole:bool)->dict[str,ar.Ball]:
        ps=[p,aux];loads=[1,self.copies];ds=[self.density(pp) for pp in ps]
        rho=ds[0][0]+self.copies*ds[1][0]-ar.fromiv(ar.exact(F(2*(1+self.copies),self.spin)))
        if hole:rho=rho-ar.fromiv(ar.exact(F(1,self.spin)))*ar.Ball(np.r_[np.ones(self.spin),np.zeros(self.dim-self.spin)])
        vr=repeat_sites(self.v,self.spin)
        rm=ar.contract('ri,i->r',vr,rho)
        variance=ar.contract('ri,ij,rj->r',vr,ds[0][1],vr)+self.copies*ar.contract('ri,ij,rj->r',vr,ds[1][1],vr)
        ma,ca=self.pair(self.b[0],p);mb,cb=self.pair(self.b[1],aux)
        mean=ma+self.copies*mb
        second=ar.outer(mean,mean)+ca+self.copies*cb
        components={key:ar.Ball(np.zeros((self.edge,self.edge))) for key in ('electric_square','electric_matter_connected','electric_matter_collective')}
        native=np.arange(self.h)
        for role,(pp,load) in enumerate(zip(ps,loads)):
            for start in range(0,self.h,64):
                ids=native[start:start+64];xx,yy=self.x[ids],self.y[ids]
                m,c,t=self.linear(self.b[role][ids],xx,yy,pp,self.v[ids],square=True)
                fm=128*(rm[ids]-self.q2[ids]*.5)+64*ar.Ball(np.diag(self.overlap)[ids])
                value=self.onechar[ids]*load*(m*(fm*fm+128**2*(variance[ids]+self.q2[ids]*.125))+256*c*fm+128**2*cast(ar.Ball,t))
                components['electric_square']=sparse_add(components['electric_square'],self.k,ids,ids,-3*value)
            for start in range(0,len(self.records),128):
                a,b,xx,yy,sign,reverse=self.records[start:start+128].T
                ab=ar.contract('nij,njk->nik',self.b[role][b],self.b[role][a])
                ba=ar.contract('nij,njk->nik',self.b[role][a],self.b[role][b])
                block=lifted(np.where(reverse[:,None,None]==0,ab.v,ba.v),max(ab.r,ba.r))
                mu,cv,_=self.linear(block,xx,yy,pp,self.v[a])
                _,cvb,_=self.linear(block,xx,yy,pp,self.v[b])
                fa=128*(rm[a]-(self.q2[a]+self.gram[a,b])*.5)+64*ar.Ball(np.diag(self.overlap)[a])
                qt2=self.q2[a]+self.q2[b]+2*self.gram[a,b]
                ft=128*(rm[a]+rm[b]-qt2*.5)+64*ar.Ball(self.overlap[a,a]+self.overlap[b,b]+2*self.overlap[a,b])
                va=load*ar.Ball(sign)*self.twochar[a,b]*(mu*fa+128*cv)
                vt=load*ar.Ball(sign)*self.twochar[a,b]*(mu*ft+128*(cv+cvb))
                key='electric_matter_connected';out=components[key]
                for aa,bb,z in ((a,a,-3*(va+vt)),(a,b,-2*vt-va),(b,a,2*vt)):out=sparse_add(out,self.k,aa,bb,z)
                components[key]=out
            # Onsite zero-wind terms in both electric placements.
            for start in range(0,self.h,64):
                ids=native[start:start+64];xx,yy=self.x[ids],self.y[ids]
                block=ar.contract('nij,njk->nik',self.on[role][xx],self.b[role][ids])-ar.contract('nij,njk->nik',self.b[role][ids],self.on[role][yy])
                mu,cv,_=self.linear(block,xx,yy,pp,self.v[ids])
                fa=128*(rm[ids]-self.q2[ids]*.5)+64*ar.Ball(np.diag(self.overlap)[ids])
                z=-6*load*self.onechar[ids]*(mu*fa+128*cv)
                components['electric_matter_connected']=sparse_add(components['electric_matter_connected'],self.k,ids,ids,z)
        weighted=self.twochar*ar.Ball(self.overlap)*second
        # All collective species have already been contracted; no second63 comm factor.
        row=ar.contract('rs->r',weighted)
        components['electric_matter_collective']=sparse_add(components['electric_matter_collective'],self.k,native,native,384*row)
        aa,bb=np.nonzero(self.overlap)
        components['electric_matter_collective']=sparse_add(components['electric_matter_collective'],self.k,aa,bb,128*weighted[aa,bb])
        return components

    def matter(self,effective:list[ar.Ball])->ar.Ball:
        out=ar.Ball(np.zeros((self.edge,self.edge)));a,b,c=self.chains.T
        for role,p in enumerate(effective):
            block=lifted(np.r_[self.b[role].v,self.on[role].v],max(self.b[role].r,self.on[role].r))
            pb=self.blocks(p)
            z=ar.contract('nij,njk,nkl,nli->n',block[a],block[b],block[c],pb[self.ty[c],self.tx[a]])*self.triplechar
            for r,s,coefficient in ((c,c,-3),(b,b,6),(a,a,-3),(c,b,-2),(b,c,3),(b,a,3),(a,b,-2),(c,a,-1),(a,c,-1)):
                out=sparse_add(out,self.tk,r,s,coefficient*z)
        return out

    def contact(self,effective:list[ar.Ball])->ar.Ball:
        out=ar.Ball(np.zeros((self.edge,self.edge)));ids=np.arange(self.h)
        for block,p in zip(self.b,effective):
            z=-self.onechar*ar.contract('nij,nji->n',block,self.blocks(p)[self.y,self.x])
            out=sparse_add(out,self.k,ids,ids,z)
        return out

    def evaluate(self,states:list[tuple[ar.Ball,ar.Ball,bool]],aux:ar.Ball)->dict[str,ar.Ball]:
        terms={key:ar.Ball(np.zeros((self.edge,self.edge))) for key in ('electric_square','electric_matter_connected','electric_matter_collective')}
        effective=ar.Ball(np.zeros_like(aux.v));mass=ar.Ball(0)
        for weight,p,hole in states:
            for key,value in self.state(p,aux,hole).items():terms[key]=terms[key]+weight*value
            effective=effective+weight*p;mass=mass+weight
        eps=[effective,mass*self.copies*aux]
        terms['matter']=self.matter(eps)
        terms['M3']=128*sum(terms.values(),ar.Ball(np.zeros((self.edge,self.edge))))
        terms['C0']=self.contact(eps)
        return terms


def physical_frame(native:dict[str,Any],geo:dict[str,Any])->tuple[ar.Ball,dict[str,Any]]:
    d=np.asarray(native['D'],np.int64)
    if 64**2*int(np.max(abs(geo['lint'])))*int(np.max(abs(d)))**2>=2**63:raise ArithmeticError('Exact projector integer arithmetic could overflow')
    numerator=geo['den']*np.eye(192,dtype=np.int64)-d.T@geo['lint']@d
    pt=ar.rational_array(numerator,geo['den'])
    observed=ar.Ball(native['Q'])
    b=pt@observed;gram=b.transpose()@b
    difference=gram-ar.Ball(np.eye(129));projection=b-observed
    eta=129*(ar.exact(F.from_float(ar.mag(difference.v)))+ar.exact(F.from_float(difference.r)))
    epsilon=ar.iv.sqrt(ar.exact(192*129))*(ar.exact(F.from_float(ar.mag(projection.v)))+ar.exact(F.from_float(projection.r)))
    if ar.end(eta,True)>=1:raise ArithmeticError('Exact physical frame polar correction not certified')
    error=epsilon+ar.iv.sqrt(1+eta)*(1/ar.iv.sqrt(1-eta)-1)
    radius=ar.up(float(ar.end(error,True)))
    return ar.Ball(observed.v,radius),{'eta_upper_exact':str(ar.end(eta,True)),'projection_defect_operator_upper_exact':str(ar.end(epsilon,True)),'Qstar_entry_radius':radius,'exact_rank':129,'definition':'Qstar=P_T Qobs*(Qobs^T P_T Qobs)^(-1/2), not unproved literal eigenvectors'}


def interval_boundary_controls()->dict[str,Any]:
    observed=conditional.finite_boundary_controls()
    k=np.array([[1,0],[0,1],[-1,0],[0,-1]],int)
    delta=np.array([[1,-1],[1,-1],[-1,1],[-1,1]],int)
    signs=np.array([1,-1,-1,1])
    geo={'k':k,'delta':delta,'x':np.array([0,0,1,1]),'y':np.array([1,1,0,0]),'gram':4*np.outer(signs,signs),'lint':np.array([[1,-1],[-1,1]]),'den':8}
    blocks=[ar.Ball(np.array([.23+.11j,-.17+.09j,.23-.11j,-.17-.09j])[:,None,None]),ar.Ball(np.array([.13+.07j,.19-.05j,.13-.07j,.19+.05j])[:,None,None])]
    onsite=[ar.Ball(np.array([.31,-.22])[:,None,None]),ar.Ball(np.array([-.12,.18])[:,None,None])]
    u=ar.fromiv(ar.iv.sqrt(ar.exact(F(7,10))));v=ar.fromiv(ar.iv.sqrt(ar.exact(F(3,10))))
    orbital=ar.Ball(np.array([u.v.item(),1j*v.v.item()]),max(u.r,v.r))
    p=ar.outer(orbital,orbital.conj());aux=ar.Ball(np.diag([1.,0.]))
    engine=IntervalEngine(geo,blocks,onsite,nsite=2,spin=1,copies=1)
    direction=ar.fromiv(1/ar.iv.sqrt(ar.exact(2)))*ar.Ball(np.array([1,-1]))
    matrix=engine.evaluate([(ar.Ball(1),p,False)],aux)['M3']
    scalar=ar.contract('i,ij,j->',direction,matrix,direction)
    interval=scalar.real_interval()
    direct=F.from_float(observed['finite_CAR_Hermite'][-1]['canonical_M3'])
    if not ar.end(interval,False)<=direct<=ar.end(interval,True):raise ArithmeticError('Outward tiny coefficient misses independent CAR/Hermite point')
    for name in ('wrong_HE32_defects','missing_HC_defect','wrong_independent_auxiliary_Theta_defect'):
        defect=observed[name][-1] if isinstance(observed[name],list) else observed[name]
        if 2*scalar.r>=defect:raise ArithmeticError('Outward tiny boundary fails adverse discrimination '+name)
    raw=engine.evaluate([(ar.Ball(1),ar.Ball(np.eye(2)),True)],aux)['M3']
    rawscalar=ar.contract('i,ij,j->',direction,raw,direction)
    rawdirect=F.from_float(observed['HC_Ref_raw_direct'])
    if not ar.end(rawscalar.real_interval(),False)<=rawdirect<=ar.end(rawscalar.real_interval(),True):raise ArithmeticError('Outward Ref hole boundary misses direct point')
    if 2*rawscalar.r>=observed['missing_Ref_defect']:raise ArithmeticError('Outward missing Ref discrimination blind')
    observed['outward_small_initial_M3']=ar.interval(interval)
    observed['outward_small_initial_Ref_raw_M3']=ar.interval(rawscalar.real_interval())
    return observed

def run(output:Path)->dict[str,Any]:
    started=time.monotonic();controls=interval_boundary_controls()
    inputs={**body.CAPTURED}
    raw=(ROOT/body.ARCHIVE).read_bytes();inputs[body.ARCHIVE]=raw
    if hashlib.sha256(raw).hexdigest()!=body.ARCHIVE_SHA:raise ArithmeticError('Pinned fixed-mean archive changed')
    saved=json.loads(raw)['input_descriptor']['exact_NEW_allq_BO_mean'];centres=[F(x) for x in saved['centre_exact_rational']]
    data=body.joined.old.load_mean(g_override=[centres[0],centres[1],centres[1],centres[1]])
    native=data['native'];profiles=data['profiles'][1:];paths=body.joined.old.native_paths(profiles[0],native)
    geo=ar.geometry(native,paths)
    species=[ar.exact_species(p,native,saved,paths) for p in profiles]
    bs=[s[0] for s in species];ons=[lifted(np.broadcast_to(s[1].v,(64,4,4)),s[1].r) for s in species];p,aux=[s[2] for s in species]
    engine=IntervalEngine(geo,bs,ons)
    frame,frame_receipt=physical_frame(native,geo)
    empty=ar.Ball(np.eye(256))-p;rawstates=[];weights=[]
    for alpha in range(4):
        v=empty[:,alpha];norm=ar.contract('i,i->',v.conj(),v)
        weight=norm*.25;pp=p+ar.outer(v,v.conj())*norm.reciprocal_real()
        rawstates.append((weight,pp,True));weights.append(ar.interval(weight.real_interval()))
    # Ordered <A V V> via connected block cumulants; dense generic Ball
    # control price failed and is preserved, not accepted as a false interval.
    a=int(np.flatnonzero(geo['x']==0)[0]);ids=np.array([a])
    vv=repeat_sites(engine.v[a],4)
    def ordered_three(pp:ar.Ball)->ar.Ball:
        mu,cov,third=engine.linear(bs[0][ids],geo['x'][ids],geo['y'][ids],pp,engine.v[ids],square=True)
        density,covariance=engine.density(pp)
        mv=ar.contract('i,i->',vv,density)
        variance=ar.contract('i,ij,j->',vv,covariance,vv)
        return (mu*(mv*mv+variance)+2*cov*mv+cast(ar.Ball,third))[0]
    actual=sum((w*ordered_three(pp) for w,pp,_ in rawstates),ar.Ball(0))
    total=sum((w for w,_,_ in rawstates),ar.Ball(0))
    averaged=sum((w*pp for w,pp,_ in rawstates),ar.Ball(np.zeros((256,256))))*total.reciprocal_real()
    wrong=total*ordered_three(averaged);delta=actual-wrong
    if ar.end(delta.real_interval(),False)<=0<=ar.end(delta.real_interval(),True):raise ArithmeticError('Wrong averaged raw Slater not excluded outward')
    controls['outward_wrong_averaged_raw_threefactor_difference']=ar.interval(delta.real_interval())
    arrays={};results={}
    torons=np.zeros((192,3))
    for axis in range(3):torons[axis::3,axis]=.125
    if not np.array_equal(np.asarray(native['D'],int)@torons,np.zeros((64,3))) or not np.array_equal(torons.T@torons,np.eye(3)):raise ArithmeticError('Exact uniform physical toron vectors fail')
    tau=ar.Ball(torons)
    frame_receipt['exact_uniform_torons']='tau_axis[axis::3]=1/8; exact Dtau=0 and tau^Ttau=I. Whole polar coordinate columns not labeled pure torons; explicit192 probes retained.'
    for label,states in (('background',[(ar.Ball(1),p,False)]),('raw_source',rawstates)):
        components=engine.evaluate(states,aux)
        for key in ('C0','M3'):
            value=tau.transpose()@components[key]@tau
            arrays[label+'_'+key+'_exact_toron_centre']=value.v
            arrays[label+'_'+key+'_exact_toron_radius']=np.full((3,3),value.r)
        arrays[label+'_edge_M3_centre']=components['M3'].v
        arrays[label+'_edge_M3_radius']=np.full((192,192),components['M3'].r)
        projected={key:frame.transpose()@value@frame for key,value in components.items()}
        projected['D5']=projected['M3']*ar.fromiv(ar.exact(F(128,120)))
        projected['Pi4']=projected['M3']*ar.fromiv(ar.exact(F(1,24)))
        for key,value in projected.items():
            arrays[label+'_'+key+'_centre']=value.v
            arrays[label+'_'+key+'_radius']=np.full((129,129),value.r)
            if not np.all(np.isfinite(value.v)) or not math.isfinite(value.r):raise ArithmeticError('Nonfinite outward coefficient')
        results[label]={key:{'entry_l1_radius':value.r,'centre_frobenius':float(np.linalg.norm(value.v)),'centre_max_l1':ar.mag(value.v)} for key,value in projected.items()}
        # Explicit repair price for old literal-Q C0 congruence; free massI is canonical, not a photon correction.
        cedge=components['C0'];deltaq=ar.exact(F.from_float(frame.r))
        op=192*(ar.exact(F.from_float(ar.mag(cedge.v)))+ar.exact(F.from_float(cedge.r)))
        frame_error=ar.iv.sqrt(ar.exact(129))*deltaq*(2+3*deltaq)*op
        results[label]['prior_literal_Q_C0_frame_HS_correction_upper_exact']=str(ar.end(frame_error,True))
        results[label]['frame_correction_rule']='||Qstar-Qobs||op<=deltaq; ||Cstar-Cobs||HS<=sqrt129*deltaq*(2+3deltaq)*||Cedge||op. Must add separately to old C0 margin; not assumed included in old entry radius.'
        print(json.dumps({'state':label,'M3_radius':projected['M3'].r}),flush=True)
    arrays['physical_Qobs']=native['Q'];arrays['physical_Qstar_entry_radius']=np.array(frame.r);arrays['exact_L_integer']=geo['lint'];arrays['exact_gram_integer']=geo['gram'];arrays['integer_winds']=geo['k']
    arrays['exact_uniform_torons']=torons
    npz=output.with_suffix('.npz');np.savez_compressed(npz,**arrays)
    for desc in ('root_receipt','full_receipt'):
        path=Path(data['descriptor'][desc]['path']);inputs[path.name]=path.read_bytes()
    for module in (ar,car,conditional):
        path=Path(cast(str,module.__file__));inputs[path.name]=path.read_bytes()
    source=Path(__file__).read_bytes();inputs[Path(__file__).name]=source
    result={'status':'ALL129_INITIAL_CUBIC_JET_OUTWARD_FIXEDMEAN_BODY','proof':PROOF,'controls':controls,'all129_states':results,'raw4_weights':weights,'physical_frame':frame_receipt,'Gauss_denominator':geo['den'],'ordered_sparse_triple_chains_per_role':len(engine.chains),'source_sha256':hashlib.sha256(source).hexdigest(),'input_sha256':{name:hashlib.sha256(content).hexdigest() for name,content in inputs.items()},'npz_sha256':hashlib.sha256(npz.read_bytes()).hexdigest(),'runtime':{'executable':sys.executable,'python':sys.version},'elapsed_seconds':time.monotonic()-started,'measured_time_evolution':False,'future_remainder':'parent-owned','admission':'parent review required; no automatic finite kernel/pole/source adoption'}
    if any((ROOT/name).read_bytes()!=content for name,content in inputs.items()):raise ArithmeticError('Input/source bytes changed during interval calculation')
    import zipfile
    with zipfile.ZipFile(output.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
        for name,content in inputs.items():archive.writestr(name,content)
    output.write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/(PREFIX+'_receipt.json'));args=parser.parse_args()
    with threadpool_limits(limits=1):result=run(args.output)
    print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds']}),flush=True)
