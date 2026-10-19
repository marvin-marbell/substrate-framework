"""Actual signed ROW paired-source initial jets; no evolved Slater assumption.

The polynomial is right-normal on the BACKGROUND Ref4 state. A creation carries
its actual Ref hole: d=e_x-e_0, not e_x. Commutator winds never become a bare
Coulomb self coefficient. Parent composes physical endpoint/initial errors.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
import zipfile
import numpy as np
from threadpoolctl import threadpool_limits
import joined_phi_u1_body_word as body
import joined_phi_u1_current_signal_interval_certificate as ar
import joined_phi_u1_initial_car_word_algebra as car
import joined_phi_u1_nonlocal_signal_interval_certificate as first

ROOT=Path(__file__).resolve().parent
PREFIX='joined_phi_u1_nonlocal_paired_higher'
CLOCK=F(1,65536)
REVISION='d394739ef3bfd685860107612d31df7ba391b51ee7a687e9a287512206368c2a'
PROOF={
 'paired':'DnE=HS D(n-1)E-D(n-1)E HB. Cn=(-i)^n <DnE psi,E0psi>. Compute its conjugate through <E0dag DnE> on original background; never <E0dag US E0> or a particle-hole oscillator.',
 'normal':'For A=exp(iqW Theta) creation_x B times ordered even CAR factors, net density displacement d=ex-e0+sum bilinear displacements. [HE+HC,A]=A(128(qW Pi+d Lrho)+64(qW²+dLd)). Every term preserves this Ref4to3 identity. [s,O_r]=qW.qr+dLdelta_r; full native case equals integer W.kr. Onsite/current and both signed one/two-link AP/Phi blocks retained.',
 'matter':'[dGamma(h),creation B]=creation(hB). Ordered even factors each use the literal connected same-copy CAR commutator; auxiliary summed factors retain63 only in initial collective contractions. A[M,P(s)] adds A O_r (P(s)-P(s+comm(s,O_r))). Nonempty subset expansion is exact, not a scalar K0 approximation.',
 'initial':'E0dag creation_x B gives 1/4(delta_x0 TrB - dGamma(|x>B<0|)). Remaining operator factors preserve order. Polynomial rho factors lie to the RIGHT and act on background Ref4: rho=sum64 n-128. Lplus directions have exact zero sum, so that constant vanishes. Ordered initial CAR determinants use occupied/complement permutation signs; auxiliary partitions use falling factorial63. All-real Gaussian right-normal mean=-qW/2, covariance1/8, character exp(-qW²). No future Wick.',
 'arithmetic':'Exact saved mean Bloch supplier, interval60 scalar trig/root/exp and rational physical Gram. ar.Ball prices every binary64 contraction/sum through derived gamma bounds; no percentage guard. Sparse cycles reduce to four-colour endpoint blocks; density-only cycles remain full256 and are cached without dropping correlations.',
 'parity':'Find exact Gaussian-integer J with gamma_mu J=J conjugate(gamma_mu), J Jdag=I. Antiunitary Theta-reflection times Gamma(J)K leaves full H64/Pi/HC, Ref and Ex invariant. Occupied rank128 mean seas and filled Ref4 are invariant rays; real even Gaussian invariant. Thus actual BODY C(-t)=conjugate(C(t)): real even, imaginary odd. This says nothing about the physical endpoint kernel. C0 offsite zero follows full Clifford trace, not a numerical cutoff.',
 'scope':'Initial same-H all12 C3 and optional C5 plus local full-H future bounds. Proposed Taylor enclosures are not measured evolution. Physical Ex/E0 and both-time initial error are separate parent obligations; no clock/source/pole/cone adoption.'}

@dataclass
class Factor:
    role:int
    x:int
    y:int
    b:Any
    density:int=-1

@dataclass
class Term:
    coeff:F
    x:int
    b:Any
    factors:tuple[Factor,...]
    wind:tuple[int,...]
    poly:tuple[int,...]
    degree:int

class Engine:
    def __init__(self,geo:dict[str,Any],bs:list[Any],ons:list[Any],ps:list[Any],he:int=64,hc:int=64,ref:bool=True,omit_second:bool=False):
        self.geo=geo;self.bs=bs;self.ons=ons;self.ps=ps;self.he=he;self.hc=hc;self.ref=ref
        self.k=geo['k'];self.x=geo['x'];self.y=geo['y'];self.den=geo['den'];self.lint=geo['lint'];self.dmat=geo['D']
        self.h=len(self.x);self.wind_ids={};self.winds=[];self.deltas=[];self.v=[];self.moment_cache={};self.path_cache={};self.cycle_cache={};self.character_cache={}
        self.active=[r for r in range(self.h) if not omit_second or int(self.k[r]@self.k[r])==1]
        self.connectors=[[p,ar.Ball(np.eye(256))-p] for p in ps]
        self.counts={}

    def register(self,w:tuple[int,...],extra:bool=False)->int:
        key=(w,extra)
        if key not in self.wind_ids:
            i=len(self.winds);self.wind_ids[key]=i
            a=np.asarray(w,dtype=np.int64);d=self.dmat@a
            if extra:d=d+np.eye(64,dtype=np.int64)[0]
            self.winds.append(a);self.deltas.append(d)
            v=ar.rational_array(d@self.lint,self.den)
            self.v.append(ar.Ball(np.repeat(v.v,4),v.r))
        return self.wind_ids[key]

    def physical_gram(self,a:np.ndarray,b:np.ndarray)->F:
        da=self.dmat@a;db=self.dmat@b
        return F(int(a@b))-F(int(da@self.lint@db),self.den)

    def displacement_gram(self,i:int,j:int)->F:
        return F(int(self.deltas[i]@self.lint@self.deltas[j]),self.den)

    def shifted(self,w:tuple[int,...],r:int)->tuple[int,...]:
        return tuple(int(v) for v in np.asarray(w,dtype=np.int64)+self.k[r])

    def expand(self,terms:list[Term])->list[Term]:
        out=[]
        for t in terms:
            wi=self.register(t.wind,not self.ref);w=self.winds[wi]
            # HE and HC have separate constants; wrong-law controls do not hide either.
            const=self.he*self.physical_gram(w,w)+self.hc*self.displacement_gram(wi,wi)
            if const:out.append(Term(t.coeff*const,t.x,t.b,t.factors,t.wind,t.poly,t.degree))
            # With unequal HE/HC, s here carries normalized separate weights in expectation/shift.
            out.append(Term(t.coeff*128,t.x,t.b,t.factors,t.wind,t.poly+(wi,),t.degree))
            for r in self.active:
                if self.y[r]==t.x:
                    out.append(Term(t.coeff,int(self.x[r]),self.bs[0][r]@t.b,t.factors,self.shifted(t.wind,r),t.poly,t.degree+1))
            out.append(Term(t.coeff,t.x,self.ons[0]@t.b,t.factors,t.wind,t.poly,t.degree))
            for pos,f in enumerate(t.factors):
                for r in self.active:
                    options=[]
                    if self.y[r]==f.x:options.append((int(self.x[r]),f.y,self.bs[f.role][r]@f.b,1))
                    if f.y==self.x[r]:options.append((f.x,int(self.y[r]),f.b@self.bs[f.role][r],-1))
                    for x,y,b,sign in options:
                        ff=t.factors[:pos]+(Factor(f.role,x,y,b),)+t.factors[pos+1:]
                        out.append(Term(t.coeff*sign,t.x,t.b,ff,self.shifted(t.wind,r),t.poly,t.degree+1))
                b=self.ons[f.role]@f.b-f.b@self.ons[f.role]
                out.append(Term(t.coeff,t.x,t.b,t.factors[:pos]+(Factor(f.role,f.x,f.y,b),)+t.factors[pos+1:],t.wind,t.poly,t.degree))
            if t.poly:
                for r in self.active:
                    shifts=[F(self.he,64)*self.physical_gram(self.winds[i],self.k[r])+F(self.hc,64)*F(int(self.deltas[i]@self.lint@self.geo['delta'][r]),self.den) for i in t.poly]
                    for size in range(1,len(t.poly)+1):
                        for selected in itertools.combinations(range(len(t.poly)),size):
                            c=-math.prod(shifts[i] for i in selected)
                            if not c:continue
                            poly=tuple(i for pos,i in enumerate(t.poly) if pos not in selected)
                            for role in range(2):
                                f=Factor(role,int(self.x[r]),int(self.y[r]),self.bs[role][r])
                                out.append(Term(t.coeff*c,t.x,t.b,t.factors+(f,),self.shifted(t.wind,r),poly,t.degree+1))
        return out

    def connector_path(self,role:int,types:tuple[int,...],ds:tuple[int,...])->Any:
        key=(role,types,ds)
        if key not in self.path_cache:
            p=self.connectors[role][types[0]]
            for i,d in enumerate(ds):
                v=self.v[d]
                p=(p* v.reshape(1,256))@self.connectors[role][types[i+1]]
            self.path_cache[key]=p
        return self.path_cache[key]

    def cycle(self,role:int,fs:list[Factor],ids:list[int])->Any:
        # A sparse factor rotates the cycle, preserving its original connector orientation.
        sparse=[j for j,i in enumerate(ids) if fs[i].density<0]
        if sparse:
            s=sparse[0];ids=ids[s:]+ids[:s];start=fs[ids[0]]
            product=start.b;current=start
            pos=0
            while pos<len(ids):
                nxt=(pos+1)%len(ids);dens=[];types=[];previous=ids[pos]
                while fs[ids[nxt]].density>=0:
                    target=ids[nxt];types.append(int(previous<target));dens.append(fs[target].density);previous=target;nxt=(nxt+1)%len(ids)
                target=ids[nxt];types.append(int(previous<target))
                path=self.connector_path(role,tuple(types),tuple(dens))
                block=path[4*current.y:4*current.y+4,4*fs[target].x:4*fs[target].x+4]
                product=product@block
                if nxt==0:break
                product=product@fs[target].b;current=fs[target];pos=nxt
            return ar.contract('ii->',product)
        key=(role,tuple(fs[i].density for i in ids),tuple(int(ids[j]<ids[(j+1)%len(ids)]) for j in range(len(ids))))
        if key not in self.cycle_cache:
            ds=key[1];types=key[2]
            if len(ds)==1:
                # Each original sea has exactly2/site; neutral Lplus direction has sum0.
                value=ar.Ball(0)
            else:
                p=self.connector_path(role,types,ds[1:])
                value=ar.contract('i,ii->',self.v[ds[0]],p)
            self.cycle_cache[key]=value
        return self.cycle_cache[key]

    def ordered(self,role:int,fs:list[Factor])->Any:
        if not fs:return ar.Ball(1)
        total=ar.Ball(0);n=len(fs)
        for perm in itertools.permutations(range(n)):
            seen=set();value=ar.Ball(1);cycles=0
            for start in range(n):
                if start in seen:continue
                cycles+=1;ids=[];j=start
                while j not in seen:seen.add(j);ids.append(j);j=perm[j]
                value=value*self.cycle(role,fs,ids)
            sign=(-1)**(n-cycles+sum(i<j for i,j in enumerate(perm)))
            total=total+sign*value
        return total

    def roles(self,fs:list[Factor])->Any:
        a=[f for f in fs if f.role==0];b=[f for f in fs if f.role==1]
        aa=self.ordered(0,a);bb=ar.Ball(0)
        for partition in car.set_partitions(tuple(range(len(b)))):
            value=ar.Ball(math.prod(63-k for k in range(len(partition))))
            for block in partition:value=value*self.ordered(1,[b[i] for i in sorted(block)])
            bb=bb+value
        return aa*bb

    def transition(self,t:Term,density:tuple[int,...])->Any:
        base=list(t.factors)
        result=ar.Ball(0)
        for roles in itertools.product(range(2),repeat=len(density)):
            fs=base+[Factor(role,0,0,None,d) for role,d in zip(roles,density)]
            value=-self.roles([Factor(0,t.x,0,t.b)]+fs)
            if t.x==0:value=value+ar.contract('ii->',t.b)*self.roles(fs)
            result=result+value*.25
        return result

    def evaluate(self,terms:list[Term])->Any:
        total=ar.Ball(0)
        for t in terms:
            w=np.asarray(t.wind,dtype=np.int64)
            q2=self.physical_gram(w,w)
            if q2 not in self.character_cache:self.character_cache[q2]=ar.fromiv(ar.iv.exp(-ar.exact(q2)))
            character=self.character_cache[q2]
            value=ar.Ball(0);n=len(t.poly)
            for size in range(n+1):
                for gaussian in itertools.combinations(range(n),size):
                    gs=[t.poly[i] for i in gaussian];ds=tuple(t.poly[i] for i in range(n) if i not in gaussian)
                    means=[ar.fromiv(ar.exact(-F(self.he,64)*self.physical_gram(self.winds[i],w)/2)) for i in gs]
                    cov=[[ar.fromiv(ar.exact(F(self.he,64)**2*self.physical_gram(self.winds[i],self.winds[j])/8)) for j in gs] for i in gs]
                    gm=car.gaussian_linear_moment(means,cov)
                    value=value+gm*self.transition(t,ds)*ar.fromiv(ar.exact(F(self.hc,64)**len(ds)))
            total=total+ar.fromiv(ar.exact(t.coeff))*character*value
        return total.conj()

def enclosed(b:Any)->dict[str,Any]:
    z=complex(b.v.item());r=F.from_float(b.r)
    return {'real':{'lower':str(F.from_float(z.real)-r),'upper':str(F.from_float(z.real)+r)},'imaginary':{'lower':str(F.from_float(z.imag)-r),'upper':str(F.from_float(z.imag)+r)},'centre':{'real':z.real,'imaginary':z.imag},'entry_l1_radius_upper_exact':str(r)}

def parity_witness()->dict[str,Any]:
    gamma=np.asarray(body.joined.old.full.supplier.primitive.SELECTED)
    for size in range(5):
        for indices in itertools.combinations(range(4),size):
            j:Any=np.eye(4,dtype=complex)
            for i in indices:j=j@gamma[i]
            if all(np.array_equal(g@j,j@g.conj()) for g in gamma) and np.array_equal(j@j.conj().T,np.eye(4)):
                return {'gamma_product_indices':indices,'exact_intertwining':True,'J_real':j.real.tolist(),'J_imaginary':j.imag.tolist(),'kernel_BODY_parity':'real-even/imaginary-odd','physical_parity_claimed':False}
    raise ArithmeticError('No exact Clifford antiunitary witness')

def run(output:Path,order:int=3,controls:bool=False)->dict[str,Any]:
    started=time.monotonic();sys.set_int_max_str_digits(0)
    captured={**body.CAPTURED,Path(__file__).name:Path(__file__).read_bytes()}
    data,species,inputs,_=body.full_inputs();captured.update(inputs)
    saved=json.loads(captured[body.ARCHIVE])['input_descriptor']['exact_NEW_allq_BO_mean']
    native=data['native'];paths=body.joined.old.native_paths(species[0].profile,native)
    geo=ar.geometry(native,paths);geo['D']=native['D'].astype(np.int64)
    exact=[ar.exact_species(s.profile,native,saved,paths) for s in species]
    bs=[v[0] for v in exact];ons=[v[1] for v in exact];ps=[v[2] for v in exact]
    rows=[];parity=parity_witness()
    indices=np.flatnonzero(geo['y']==0)
    if len(indices)!=12:raise ArithmeticError('Actual all12 signed ROW paths missing')
    for rr in indices:
        r=int(rr);engine=Engine(geo,bs,ons,ps)
        terms=[Term(F(1),int(geo['x'][r]),ar.Ball(np.eye(4)),(),tuple(map(int,geo['k'][r])),(),1)]
        coefficients={};counts={}
        for n in range(order+1):
            if n%2==1:
                coefficient=engine.evaluate(terms);coefficients[str(n)]=enclosed(coefficient);counts[str(n)]=len(terms)
            if n<order:terms=engine.expand(terms)
        rows.append({'path':r,'ROW':int(geo['x'][r]),'COLUMN':0,'length':int(geo['k'][r]@geo['k'][r]),'C0_exact':'0','m_derivatives':coefficients,'ordered_terms':counts,'cache_paths':len(engine.path_cache),'C_derivative_convention':'(-i)^n*m_n','raw_E0_norm_squared_exact':'1/2','controls':{}})
        print(json.dumps({'path':r,'m3':coefficients.get('3'),'elapsed':time.monotonic()-started}),flush=True)
        if controls and r==int(indices[0]):
            for label,kwargs in [('no_HC',{'hc':0}),('half_HE',{'he':32}),('omit_Ref',{'ref':False}),('omit_second',{'omit_second':True})]:
                e=Engine(geo,bs,ons,ps,**kwargs);ts=[Term(F(1),int(geo['x'][r]),ar.Ball(np.eye(4)),(),tuple(map(int,geo['k'][r])),(),1)]
                for _ in range(3):ts=e.expand(ts)
                v=e.evaluate(ts);base=coefficients['3'];contrast=enclosed(v)
                blo=F(base['real']['lower']);bhi=F(base['real']['upper']);vlo=F(contrast['real']['lower']);vhi=F(contrast['real']['upper'])
                rows[-1]['controls'][label]={'m3':contrast,'discriminates_at_guarded_real_m3':vhi<blo or bhi<vlo,'initial_m1_blind':{'no_HC':True,'half_HE':True,'omit_Ref':None,'omit_second':False}[label],'Ref_control_scope':'omit_Ref removes only source Refhole; predecessor simultaneous staticRef omission is a distinct scalar-blind control'}
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file and Path(file).parent==ROOT and Path(file).suffix=='.py':captured.setdefault(Path(file).name,Path(file).read_bytes())
    if any((ROOT/name).read_bytes()!=raw for name,raw in captured.items()):raise ArithmeticError('Execution input bytes changed')
    archive=output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,raw in sorted(captured.items()):z.writestr(name,raw)
        z.writestr('actual_geometry.json',json.dumps({k:geo[k].tolist() if isinstance(geo[k],np.ndarray) else geo[k] for k in ('k','x','y','delta','lint','den','D')},default=str))
    result={'status':'ACTUAL12_INITIAL_ORDERED_PAIRED_HIGHER_JETS','proof':PROOF,'parity':parity,'receiver_rows':rows,'clock_exact':str(CLOCK),'clock_adopted':False,'physical_composition_claimed':False,'raw_source_mass_exact':'1/2','order':order,'source_sha256':hashlib.sha256(captured[Path(__file__).name]).hexdigest(),'input_sha256':{name:hashlib.sha256(raw).hexdigest() for name,raw in captured.items()},'immutable_inputs':{'path':archive.name,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},'governing_plan':first.PLAN,'captured_revision':REVISION,'template_revision':'89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe','runtime':{'python':sys.version,'executable':sys.executable,'numpy':np.__version__},'elapsed_seconds':time.monotonic()-started}
    output.write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--order',type=int,choices=(3,5),default=3);parser.add_argument('--controls',action='store_true');args=parser.parse_args()
    with threadpool_limits(limits=1):receipt=run(args.output,args.order,args.controls)
    print(json.dumps({'status':receipt['status'],'elapsed_seconds':receipt['elapsed_seconds']}),flush=True)
