"""Derived outward initial n2/n4 guard on the completed conditional source epoch.

Exact fixed-rational mean Dirac seas; no root/primary solve, no future Wick.
"""
from __future__ import annotations
from collections import defaultdict
from fractions import Fraction as F
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any
import zipfile
import numpy as np
from threadpoolctl import threadpool_limits
import joined_phi_u1_initial_density_higher_jets as jets
import joined_phi_u1_current_signal_interval_certificate as ar

ROOT=Path(__file__).resolve().parent
PREFIX='joined_phi_u1_initial_density_higher_jets_interval'


def mm(a: ar.Ball,b: ar.Ball)->ar.Ball:
    """Same entrywise l1 gamma16n proof as ar.contract, positive-only radius.

    Four input-radius/roundoff products, <32 positive radius operations; the
    factor1+2^-45 and nextafter price their evaluation, not a fitted tolerance.
    """
    n=a.v.shape[-1]
    if 16*n*2**-53>.001:raise ArithmeticError('Matrix contraction gamma precondition')
    aa,bb=ar.mag(a.v),ar.mag(b.v)
    radius=n*(aa*b.r+bb*a.r+a.r*b.r+F(2,2**48)*aa*bb)
    return ar.Ball(a.v@b.v,ar.up(float(radius)*(1+2**-45)))


def trace(a: ar.Ball)->ar.Ball:
    n=a.v.shape[0]
    return ar.Ball(np.trace(a.v),ar.up(n*(a.r+2**-48*ar.mag(a.v))*(1+2**-45)))


class OutwardEngine:
    """Independent typed Ball symbol/evaluator; not a float-engine subtype."""
    def __init__(self,data,geo,geom,exact_species):
        self.k=np.asarray(geo['k'],np.int64);self.geo=geo
        self.den=int(geom['denominator']);self.lint=np.asarray(geom['L_integer'],np.int64)
        self.d=np.asarray(data['native']['D'],np.int64);self.term_cap=2000000
        self.winds: list[dict[str,Any]]=[];self.wind_ids={};self.wind_add={};self.wind_gram={}
        self.factors: list[tuple[int,int,int,ar.Ball]]=[]
        self.factor_ids={};self.factor_comm={};self.factor_onsite={};self.diagonal_ids={}
        self.native=[[self.factor(role,int(geo['x'][j]),int(geo['y'][j]),b[j]) for j in range(len(self.k))] for role,(b,_,_) in enumerate(exact_species)]
        self.native_wind=[self.wind(tuple((int(e),int(z)) for e,z in enumerate(row) if z)) for row in self.k]
        self.zero=self.wind(());self.word_counts=[]
        self.species=[SimpleNamespace(onsite=onsite) for _,onsite,_ in exact_species]
        p=exact_species[0][2];size=p.v.shape[0];q=ar.Ball(np.eye(size))-p
        self.incident=[np.flatnonzero((geo['x']==x)|(geo['y']==x)).tolist() for x in range(size//4)]
        self.states: list[tuple[ar.Ball,ar.Ball,bool]]=[(ar.Ball(-.5),p,False)]
        self.channel_enclosures=[]
        for alpha in range(4):
            norm=q[alpha,alpha];v=q[:,alpha:alpha+1]
            added=mm(v,v.conj().transpose(1,0))*norm.reciprocal_real()
            self.states.append((norm*F(1,4),p+added,True))
            self.channel_enclosures.append(dict(alpha=alpha,norm=ar.interval(norm.real_interval()),weight=ar.interval((norm*F(1,4)).real_interval())))
        self.aux=exact_species[1][2]
        self.complements=[ar.Ball(np.eye(size))-pp for _,pp,_ in self.states]+[ar.Ball(np.eye(size))-self.aux]
        self.gaussian_cache={};self.character_cache={};self.car_cache={};self.cycle_cache={}
        self.partitions={n:list(jets.wick.set_partitions(tuple(range(n)))) for n in range(7)}
        self.permutations={n:list(jets.itertools.permutations(range(n))) for n in range(7)}

    def wind(self,key):
        if key in self.wind_ids:return self.wind_ids[key]
        delta=sum((self.d[:,e]*z for e,z in key),np.zeros(self.d.shape[0],np.int64));vn=delta@self.lint
        wind2=sum(z*z for _,z in key);wid=len(self.winds);self.wind_ids[key]=wid
        self.winds.append(dict(key=key,w2=wind2,delta=delta,vn=vn,v=vn/self.den,q2=F(wind2)-F(int(delta@vn),self.den)))
        return wid

    def add_wind(self,a,b):
        key=(a,b) if a<=b else (b,a)
        if key not in self.wind_add:
            values=dict(self.winds[a]['key'])
            for e,z in self.winds[b]['key']:values[e]=values.get(e,0)+z
            self.wind_add[key]=self.wind(tuple((e,z) for e,z in sorted(values.items()) if z))
        return self.wind_add[key]

    def gram(self,a,b):
        key=(a,b) if a<=b else (b,a)
        if key not in self.wind_gram:
            wa,wb=self.winds[a],self.winds[b];values=dict(wa['key'])
            dot=sum(values.get(e,0)*z for e,z in wb['key'])
            self.wind_gram[key]=F(dot)-F(int(wa['delta']@wb['vn']),self.den)
        return self.wind_gram[key]

    def overlaps(self,z):
        wind=self.winds[z]
        if 'overlaps' not in wind:
            wind['overlaps']=sum((self.k[:,e]*v for e,v in wind['key']),np.zeros(len(self.k),np.int64))
        return wind['overlaps']

    def factor(self,role,row,col,matrix):
        matrix=ar.asball(matrix)
        key=(role,row,col,matrix.v.tobytes(),matrix.r)
        if key not in self.factor_ids:
            self.factor_ids[key]=len(self.factors);self.factors.append((role,row,col,matrix))
        return self.factor_ids[key]

    def diagonal(self,role,w):
        key=(role,w)
        if key not in self.diagonal_ids:
            exact=[ar.fromiv(ar.exact(F(int(z),self.den))) for z in self.winds[w]['vn']]
            values=np.repeat(np.array([complex(z.v) for z in exact]),4)
            self.diagonal_ids[key]=self.factor(role,-1,w,ar.Ball(values,max(z.r for z in exact)))
        return self.diagonal_ids[key]

    def commute(self,f,j):
        key=(f,j)
        if key not in self.factor_comm:
            role,row,col,a=self.factors[f];_,r,c,b=self.factors[self.native[role][j]]
            out=[]
            if c==row:out.append((1,self.factor(role,r,col,mm(b,a))))
            if col==r:out.append((-1,self.factor(role,row,c,mm(a,b))))
            self.factor_comm[key]=tuple(out)
        return self.factor_comm[key]

    def derivative(self,terms,order):
        out=defaultdict(complex)
        def put(w,fs,ss,c):
            if c:out[w,fs,tuple(sorted(ss))]+=c
            if len(out)>self.term_cap:raise jets.SymbolCostExceeded(f'Ball order{order}: exceeds{self.term_cap}; no partial coefficient accepted')
        for (w,fs,ss),coef in terms.items():
            put(w,fs,ss,coef*64*self.winds[w]['w2'])
            if w!=self.zero:put(w,fs,ss+(w,),coef*128)
            for position,f in enumerate(fs):
                role,row,col,a=self.factors[f]
                if f not in self.factor_onsite:
                    k=self.species[role].onsite
                    self.factor_onsite[f]=self.factor(role,row,col,mm(k,a)-mm(a,k))
                put(w,fs[:position]+(self.factor_onsite[f],)+fs[position+1:],ss,coef)
                for j in sorted(set(self.incident[row]+self.incident[col])):
                    ww=self.add_wind(w,self.native_wind[j])
                    for sign,ff in self.commute(f,j):
                        put(ww,fs[:position]+(ff,)+fs[position+1:],ss,coef*sign)
            if ss:
                linked=sorted(set(j for z in ss for j in np.flatnonzero(self.overlaps(z))))
                for j in linked:
                    shifts=[int(self.overlaps(z)[j]) for z in ss];ww=self.add_wind(w,self.native_wind[j])
                    for mask in range(1,1<<len(ss)):
                        multiplier=math.prod(shifts[p] for p in range(len(ss)) if mask>>p&1)
                        if not multiplier:continue
                        remain=tuple(z for p,z in enumerate(ss) if not mask>>p&1)
                        for role in (0,1):put(ww,fs+(self.native[role][j],),remain,-coef*multiplier)
        result={k:v for k,v in out.items() if v!=0}
        self.word_counts.append(dict(order=order,grouped_terms=len(result),unmerged_nonzero_keys=len(out),winds=len(self.winds),bilinear_blocks=len(self.factors)))
        return result

    def initial(self,x):
        return {(self.native_wind[j],(self.native[0][j],),()):complex(int(self.geo['y'][j]==x)-int(self.geo['x'][j]==x)) for j in self.incident[x]}

    def cycle(self,p,identity,fs,connections):
        key=(identity,fs,connections)
        if key in self.cycle_cache:return self.cycle_cache[key]
        sparse=next((i for i,f in enumerate(fs) if self.factors[f][1]>=0),None)
        if sparse is not None:
            fs=fs[sparse:]+fs[:sparse];connections=connections[sparse:]+connections[:sparse]
        _,row,col,a=self.factors[fs[0]];complement=self.complements[identity];size=p.v.shape[0]
        if row>=0:
            start=np.arange(4*row,4*row+4);indices=np.arange(4*col,4*col+4);product=a
            for position in range(1,len(fs)):
                c=p if connections[position-1] else complement
                _,r,t,b=self.factors[fs[position]]
                if r>=0:
                    target=np.arange(4*r,4*r+4)
                    product=mm(mm(product,c[np.ix_(indices,target)]),b);indices=np.arange(4*t,4*t+4)
                else:
                    product=mm(product,c[indices,:])*b.reshape(1,size);indices=np.arange(size)
            c=p if connections[-1] else complement
            answer=trace(mm(product,c[np.ix_(indices,start)]))
        else:
            c=p if connections[0] else complement;product=a.reshape(size,1)*c
            for position in range(1,len(fs)):
                b=self.factors[fs[position]][3];c=p if connections[position] else complement
                product=mm(product*b.reshape(1,size),c)
            answer=trace(product)
        if len(self.cycle_cache)>=200000:self.cycle_cache.clear()
        self.cycle_cache[key]=answer
        return answer

    def ordered(self,p,identity,fs):
        key=(identity,fs)
        if key in self.car_cache:return self.car_cache[key]
        total=ar.Ball(0);n=len(fs)
        for perm in self.permutations[n]:
            seen=set();value=ar.Ball(1);cycles=0
            for start in range(n):
                if start in seen:continue
                cycles+=1;idx=start;cycle=[];connections=[]
                while True:
                    seen.add(idx);target=perm[idx];cycle.append(fs[idx]);connections.append(idx>=target)
                    if target==start:break
                    idx=target
                value*=self.cycle(p,identity,tuple(cycle),tuple(connections))
            total+=(-1)**(n-cycles+sum(i<j for i,j in enumerate(perm)))*value
        if len(self.car_cache)>=200000:self.car_cache.clear()
        self.car_cache[key]=total
        return total

    def role_moment(self,state,fs,copies=63):
        active=tuple(f for f in fs if self.factors[f][0]==0)
        auxiliary=tuple(f for f in fs if self.factors[f][0]==1);total=ar.Ball(0)
        for partition in self.partitions[len(auxiliary)]:
            value=ar.Ball(math.prod(copies-k for k in range(len(partition))))
            for block in partition:value*=self.ordered(self.aux,5,tuple(auxiliary[i] for i in sorted(block)))
            total+=value
        return self.ordered(self.states[state][1],state,active)*total

    def expectation(self,w,fs,ss,state,copies=63,Ref_hole=True,field_scale=F(1)):
        raw=self.states[state][2] and Ref_hole
        total=ar.Ball(0)
        for assignment in jets.itertools.product(range(3),repeat=len(ss)):
            gaussian=tuple(p for p,role in enumerate(assignment) if role==0)
            factors=fs+tuple(self.diagonal(role-1,z) for role,z in zip(assignment,ss) if role)
            key=(w,ss,raw,gaussian,field_scale)
            if key not in self.gaussian_cache:
                means=[-field_scale*self.gram(ss[p],w)/2-(F(int(self.winds[ss[p]]['vn'][0]),self.den) if raw else 0) for p in gaussian]
                covariance=[[self.gram(ss[p],ss[q])/(8*field_scale) for q in gaussian] for p in gaussian]
                self.gaussian_cache[key]=ar.fromiv(ar.exact(jets.wick.gaussian_linear_moment(means,covariance)))
            total+=self.gaussian_cache[key]*self.role_moment(state,factors,copies)
        key=(w,field_scale)
        if key not in self.character_cache:self.character_cache[key]=ar.fromiv(ar.iv.exp(-ar.exact(field_scale*self.winds[w]['q2'])))
        return self.character_cache[key]*total

    def evaluate(self,terms,order,**options):
        total=ar.Ball(0)
        for (w,fs,ss),coef in terms.items():
            # Generator coefficients are literal exact binary integers here.
            if coef.imag or abs(coef.real)>2**53 or not coef.real.is_integer():raise ArithmeticError('Nonexact generator coefficient')
            value=ar.Ball(0)
            for state,(weight,_,_) in enumerate(self.states):value+=self.expectation(w,fs,ss,state,**options)*weight
            total+=value*coef*(1j**order)
        return total


def run()->dict[str,Any]:
    started=time.monotonic();ar.iv.dps=75
    oldpath=ROOT/'joined_phi_u1_initial_density_higher_jets_actual_receipt_20261010.json'
    oldraw=oldpath.read_bytes();old=json.loads(oldraw)
    if old['source_sha256']!=jets.digest((ROOT/'joined_phi_u1_initial_density_higher_jets.py').read_bytes()):raise ArithmeticError('Completed conditional producer changed')
    data,sp,inputs,_=jets.body.full_inputs();geo=jets.current.geometry(data,sp[0]);geom=jets.prior.prices.exact_geometry(data['native'],geo)
    captures={**jets.body.CAPTURED,**inputs,oldpath.name:oldraw,Path(__file__).name:Path(__file__).read_bytes()}
    saved=json.loads(captures[jets.body.ARCHIVE])['input_descriptor']['exact_NEW_allq_BO_mean']
    paths=jets.body.joined.old.native_paths(sp[0].profile,data['native'])
    exact_species=[ar.exact_species(s.profile,data['native'],saved,paths) for s in sp[:2]]
    coefficient_cache={};rows=[];counts=[]
    for oldrow in old['receiver_rows']:
        x=oldrow['receiver']
        if x not in coefficient_cache:
            engine=OutwardEngine(data,geo,geom,exact_species);terms=engine.initial(x);coefficients={}
            for order in range(2,5):
                terms=engine.derivative(terms,order)
                if order%2==0:
                    value=engine.evaluate(terms,order)
                    interval=ar.interval(value.real_interval());conditional=F.from_float(oldrow['initial_coefficients'][str(order)]['real'])
                    # Inclusion is informative, not required: the Ball recomputes
                    # the actual fixedmean law independently of eigensea centres.
                    coefficients[order]=dict(enclosure=interval,centre_real=float(np.real(value.v)),entry_l1_radius=value.r,
                                             conditional_centre_inside=F(interval['lower'])<=conditional<=F(interval['upper']))
            coefficient_cache[x]=coefficients;counts.append(dict(receiver=x,counts=engine.word_counts,channel_enclosures=engine.channel_enclosures))
            print(json.dumps(dict(receiver=x,coefficients=coefficients,elapsed_seconds=time.monotonic()-started)),flush=True)
        coefficients=coefficient_cache[x]
        lower=sum(jets.CLOCK**n*F(c['enclosure']['lower'])/math.factorial(n) for n,c in coefficients.items())
        upper=sum(jets.CLOCK**n*F(c['enclosure']['upper'])/math.factorial(n) for n,c in coefficients.items())
        tail=F(oldrow['full_H_remainder_upper_exact']);physical=F(oldrow['both_endpoints_physical_error_exact'])
        rows.append(dict(path=oldrow['path'],receiver=x,length=oldrow['length'],initial_coefficients=coefficients,
                         coordinate_change_enclosure_exact=dict(lower=str(lower-tail),upper=str(upper+tail)),
                         physical_change_enclosure_exact=dict(lower=str(lower-tail-physical),upper=str(upper+tail+physical)),
                         sixth_full_H_tail_exact=str(tail),both_endpoints_physical_error_exact=str(physical),
                         physical_signed_lower_margin_exact=str(lower-tail-physical),physical_signed_positive=lower>tail+physical))
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file and Path(file).parent==ROOT and Path(file).suffix=='.py':captures.setdefault(Path(file).name,Path(file).read_bytes())
    for name,raw in captures.items():
        if (ROOT/name).read_bytes()!=raw:raise ArithmeticError('Outward source/input changed '+name)
    archive=ROOT/(PREFIX+'_input_bytes_20261010.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in sorted(captures.items()):z.writestr(name,raw)
    result: dict[str,Any]=dict(status='INITIAL_DENSITY_N4_OUTWARD_PAID_SIXTH_SIGNS_SURVIVE' if all(r['physical_signed_positive'] for r in rows) else 'INITIAL_DENSITY_N4_OUTWARD_FAILED_REGIME',
                clock_exact=str(jets.CLOCK),clock_adopted=False,plan_revision_captured=jets.REVISION,governor_review_revision='37461b2a09eb359a6d740e95a1d4af6f427ac5f623d3e4ecc79404236dbd53d2',
                fixedmean_law='original immutable centre_exact_rational, not a root solve/new root/precision adoption',arithmetic_proof=ar.PROOF['arithmetic'],
                fast_matrix_radius_proof=mm.__doc__,initial_only_Wick=True,full_H_tail_source='completed conditional receipt exact rational root-priced native hierarchy',
                raw_mass_exact='1/2',actual_four_channels=True,all63_auxiliaries=True,all129=True,all768_signed_native=True,
                all12=rows,grouped_counts=counts,n6_cost_failure_preserved=[r['symbolic_cost_failure'] for r in old['receiver_rows']],
                source_sha256=jets.digest(Path(__file__).read_bytes()),conditional_source_sha256=old['source_sha256'],conditional_receipt_sha256=jets.digest(oldraw),
                input_archive=dict(path=archive.name,sha256=jets.digest(archive.read_bytes()),files={n:jets.digest(v) for n,v in captures.items()}),
                runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started,
                whole_kernel_pole_or_source_admission=False)
    output=ROOT/(PREFIX+'_actual_receipt_20261010.json');output.write_text(json.dumps(result,indent=2)+'\n')
    capsule=dict(numeric_exit=0,natural_exit=True,source_stable=True,source_sha256=result['source_sha256'],elapsed_seconds=result['elapsed_seconds'],
                 receipt=dict(path=output.name,sha256=jets.digest(output.read_bytes())),input_archive=result['input_archive'])
    (ROOT/(PREFIX+'_natural_exit_capsule_20261010.json')).write_text(json.dumps(capsule,indent=2)+'\n')
    return result


if __name__=='__main__':
    with threadpool_limits(limits=1):result=run()
    print(json.dumps(dict(status=result['status'],actual12=len(result['all12']),elapsed_seconds=result['elapsed_seconds'])),flush=True)
