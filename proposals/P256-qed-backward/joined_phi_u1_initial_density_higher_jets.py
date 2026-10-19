"""Initial connected CAR/Weyl density jets on the unchanged joined N4 law.

The sparse symbol algebra is exact algebraically; initial numerical contractions
are explicitly conditional floating arithmetic. No future Slater is asserted.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import sys
import time
import zipfile
from typing import Any
import numpy as np
from threadpoolctl import threadpool_limits
import joined_phi_u1_body_word as body
import joined_phi_u1_current_moments as current
import joined_phi_u1_parity_density_closure as prior
import joined_phi_u1_electric_hop_transport as hop
import joined_phi_u1_initial_car_word_algebra as wick
import joined_phi_u1_local_electric_moments as moments

ROOT=Path(__file__).resolve().parent
CLOCK=F(1,65536)
REVISION='c718a919be1c0b08f69f377d4bd1b2d68a34356eb482255d492aecbe2ea8f36b'
PROOF={
 'symbol':'A term is exp(i Q^T w.Theta) times an ORDERED product of active/collective-auxiliary bilinears times an ordered monomial of commuting s_v=Q^T v.Pi+(Lplus Dv).rho. All bilinear blocks have definite charge displacement and their total displacement is Dw. On the common Schwartz/finite-CAR core [HE+HC,term]=64 term (2s_w+|w|^2). This includes HE64 and HC64 exactly, not a half-Coulomb or a Pi-only replacement.',
 'connected':'Matter commutation uses the Leibniz CAR commutator on the ordered factors: same species dGamma([A,B]); distinct charged roles commute. For identical auxiliary sums the same-copy commutator is a single sum with NO factor63. A native hop j shifts each s_v by integer v.inc_j. The polynomial part is minus word B_j times (p(s+v.inc_j)-p(s)); only overlaps of physical native edges can contribute. Onsite commutes s and contributes [K,A] to every ordered block. This visits connected endpoint/edge overlaps, not 768^order free choices.',
 'initial':'ONLY at t0 expand s into independent Gaussian Pi and density bilinears of the active plus63 auxiliary Slaters. The Gaussian means after right-normal exp(iq.Theta) are -q_v.q/2; covariance q_v.q_z/8; overlap exp(-q^2). Native/composite Grams are exact rationals. Background Ref4 cancels fixed-mean baseline; raw Ref3 gives constant -v[0]. Actual four raw weights and channel projectors, not their averaged Slater, are retained. Ordered CAR permutations and unlabeled-copy partitions use the parent algebra with falling factorial(63,k).',
 'tail':'Time parity of the full BODY and actual fourRef density matrix is inherited from the independently exercised supplier. For an even density, after actual n4 the full-H sixth hierarchy remainder is t^6 sup|n6(u)|/720. If actual n6 is obtained, the eighth remainder is t^8 sup|n8(u)|/40320. Triangle word degrees (m,d) use E transitions128m and128m^2, onsite2mK, connected CAR mDcar, and polynomial finite-difference binomial(d,l)L_l. Positive evolved native s moments price every polynomial; no evolved Wick or stationarity.',
 'physical':'Physical n_x is not its frame coordinate: norm4 and rank4 give eta_n<=8 b1 packet_radius. Both0/t density endpoints cost2eta_n. Actual E0 defect eta0=b1 packet_radius gives two endpoint prices8(2eta0+eta0^2). Evolution intertwining costs8t epsilon. These costs are added, not absorbed in an initial coefficient.',
 'status':'Initial coefficients and roundoff observations are conditional, not outward Ball enclosures. Exact rational positive remainder prices do not change that status. A symbolic-work guard reports a failed construction regime without removing carriers, laws or source terms; no truncated coefficient is substituted.'}


def digest(raw: bytes)->str:
    return hashlib.sha256(raw).hexdigest()


class SymbolCostExceeded(RuntimeError):
    pass


class Engine:
    """Grouped sparse gauge-complete symbols; no free-hop enumeration."""
    def __init__(self,data: dict[str,Any],species: list[Any],geo: dict[str,Any],geom: dict[str,Any],term_cap: int):
        self.data=data;self.species=species;self.geo=geo;self.term_cap=term_cap
        self.k=np.asarray(geo['k'],np.int64);self.den=int(geom['denominator']);self.lint=np.asarray(geom['L_integer'],np.int64)
        self.d=np.asarray(data['native']['D'],np.int64)
        self.winds=[];self.wind_ids={};self.wind_add={};self.wind_gram={}
        self.factors=[];self.factor_ids={};self.factor_comm={};self.factor_onsite={};self.diagonal_ids={}
        self.native=[]
        for role,s in enumerate(species[:2]):
            self.native.append([self.factor(role,int(geo['x'][j]),int(geo['y'][j]),b) for j,b in enumerate(current.blocks(s))])
        self.native_wind=[self.wind(tuple((int(e),int(z)) for e,z in enumerate(row) if z)) for row in self.k]
        self.zero=self.wind(())
        self.incident=[]
        for x in range(64):self.incident.append(np.flatnonzero((geo['x']==x)|(geo['y']==x)).tolist())
        self.word_counts=[]
        self.states=[(-.5,species[0].p,False)]+[(c['weight'],c['p'],True) for c in hop.initial_channels(species[0])]
        self.aux=species[1].p
        self.complements=[np.eye(len(p))-p for _,p,_ in self.states]+[np.eye(len(self.aux))-self.aux]
        self.car_cache={};self.cycle_cache={}
        self.partitions={n:list(wick.set_partitions(tuple(range(n)))) for n in range(7)}
        self.permutations={n:list(itertools.permutations(range(n))) for n in range(7)}

    def wind(self,key: tuple)->int:
        if key in self.wind_ids:return self.wind_ids[key]
        delta=sum((self.d[:,e]*z for e,z in key),np.zeros(self.d.shape[0],np.int64));vn=delta@self.lint
        wind2=sum(z*z for _,z in key)
        wid=len(self.winds);self.wind_ids[key]=wid
        self.winds.append(dict(key=key,w2=wind2,delta=delta,vn=vn,v=vn/self.den,
                               q2=F(wind2)-F(int(delta@vn),self.den)))
        return wid

    def add_wind(self,a: int,b: int)->int:
        key=(a,b) if a<=b else (b,a)
        if key not in self.wind_add:
            values=dict(self.winds[a]['key'])
            for e,z in self.winds[b]['key']:values[e]=values.get(e,0)+z
            self.wind_add[key]=self.wind(tuple((e,z) for e,z in sorted(values.items()) if z))
        return self.wind_add[key]

    def gram(self,a: int,b: int)->F:
        key=(a,b) if a<=b else (b,a)
        if key not in self.wind_gram:
            wa,wb=self.winds[a],self.winds[b]
            values=dict(wa['key'])
            dot=sum(values.get(e,0)*z for e,z in wb['key'])
            self.wind_gram[key]=F(dot)-F(int(wa['delta']@wb['vn']),self.den)
        return self.wind_gram[key]

    def overlaps(self,z: int)->np.ndarray:
        wind=self.winds[z]
        if 'overlaps' not in wind:
            wind['overlaps']=sum((self.k[:,e]*v for e,v in wind['key']),np.zeros(len(self.k),np.int64))
        return wind['overlaps']

    def factor(self,role: int,row: int,col: int,matrix: np.ndarray)->int:
        matrix=np.asarray(matrix,np.complex128)
        key=(role,row,col,matrix.tobytes())
        if key not in self.factor_ids:
            self.factor_ids[key]=len(self.factors);self.factors.append((role,row,col,matrix))
        return self.factor_ids[key]

    def diagonal(self,role: int,w: int)->int:
        key=(role,w)
        if key not in self.diagonal_ids:
            self.diagonal_ids[key]=self.factor(role,-1,w,np.repeat(self.winds[w]['v'],4))
        return self.diagonal_ids[key]

    def commute(self,f: int,j: int)->tuple:
        key=(f,j)
        if key not in self.factor_comm:
            role,row,col,a=self.factors[f];_,r,c,b=self.factors[self.native[role][j]]
            out=[]
            if c==row:out.append((1,self.factor(role,r,col,b@a)))
            if col==r:out.append((-1,self.factor(role,row,c,a@b)))
            self.factor_comm[key]=tuple(out)
        return self.factor_comm[key]

    def derivative(self,terms: dict,order: int)->dict:
        out=defaultdict(complex)
        def put(w: int,fs: tuple,ss: tuple,c: complex):
            if c:out[w,fs,tuple(sorted(ss))]+=c
            if len(out)>self.term_cap:raise SymbolCostExceeded(f'order{order}: grouped symbol count exceeds{self.term_cap}; no partial jet accepted')
        for (w,fs,ss),coef in terms.items():
            wind=self.winds[w]
            put(w,fs,ss,coef*64*wind['w2'])
            if w!=self.zero:put(w,fs,ss+(w,),coef*128)
            for position,f in enumerate(fs):
                role,row,col,a=self.factors[f]
                if f not in self.factor_onsite:
                    b=self.species[role].onsite@a-a@self.species[role].onsite
                    self.factor_onsite[f]=self.factor(role,row,col,b) if np.any(b) else None
                ff=self.factor_onsite[f]
                if ff is not None:put(w,fs[:position]+(ff,)+fs[position+1:],ss,coef)
                for j in sorted(set(self.incident[row]+self.incident[col])):
                    ww=self.add_wind(w,self.native_wind[j])
                    for sign,ff in self.commute(f,j):
                        put(ww,fs[:position]+(ff,)+fs[position+1:],ss,coef*sign)
            if ss:
                linked=sorted(set(j for z in ss for j in np.flatnonzero(self.overlaps(z))))
                for j in linked:
                    shifts=[int(self.overlaps(z)[j]) for z in ss]
                    ww=self.add_wind(w,self.native_wind[j])
                    for mask in range(1,1<<len(ss)):
                        multiplier=math.prod(shifts[p] for p in range(len(ss)) if mask>>p&1)
                        if not multiplier:continue
                        remain=tuple(z for p,z in enumerate(ss) if not mask>>p&1)
                        for role in (0,1):put(ww,fs+(self.native[role][j],),remain,-coef*multiplier)
        result={k:v for k,v in out.items() if v!=0}
        self.word_counts.append(dict(order=order,grouped_terms=len(result),unmerged_nonzero_keys=len(out),winds=len(self.winds),bilinear_blocks=len(self.factors)))
        return result

    def initial(self,x: int)->dict:
        return {(self.native_wind[j],(self.native[0][j],),()):complex(int(self.geo['y'][j]==x)-int(self.geo['x'][j]==x)) for j in self.incident[x]}

    def cycle(self,p: np.ndarray,identity: int,fs: tuple,connections: tuple)->complex:
        key=(identity,fs,connections)
        if key in self.cycle_cache:return self.cycle_cache[key]
        # Cyclically rotate to a rank<=4 hopping block whenever possible.
        sparse=next((i for i,f in enumerate(fs) if self.factors[f][1]>=0),None)
        if sparse is not None:
            fs=fs[sparse:]+fs[:sparse];connections=connections[sparse:]+connections[:sparse]
        _,row,col,a=self.factors[fs[0]]
        complement=self.complements[identity]
        if row>=0:
            start=np.arange(4*row,4*row+4);indices=np.arange(4*col,4*col+4);product=a
            for position in range(1,len(fs)):
                c=p if connections[position-1] else complement
                _,r,t,b=self.factors[fs[position]]
                if r>=0:
                    target=np.arange(4*r,4*r+4)
                    product=product@c[np.ix_(indices,target)]@b;indices=np.arange(4*t,4*t+4)
                else:
                    product=(product@c[indices,:])*b[None,:];indices=np.arange(len(p))
            c=p if connections[-1] else complement
            answer=complex(np.trace(product@c[np.ix_(indices,start)]))
        else:
            c=p if connections[0] else complement;product=a[:,None]*c
            for position in range(1,len(fs)):
                b=self.factors[fs[position]][3];c=p if connections[position] else complement
                product=(product*b[None,:])@c
            answer=complex(np.trace(product))
        if len(self.cycle_cache)>=200000:self.cycle_cache.clear()
        self.cycle_cache[key]=answer
        return answer

    def ordered(self,p: np.ndarray,identity: int,fs: tuple)->complex:
        key=(identity,fs)
        if key in self.car_cache:return self.car_cache[key]
        total=0j;n=len(fs)
        for perm in self.permutations[n]:
            seen=set();value=1+0j;cycles=0
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

    def role_moment(self,state: int,fs: tuple,copies: int=63)->complex:
        active=tuple(f for f in fs if self.factors[f][0]==0)
        auxiliary=tuple(f for f in fs if self.factors[f][0]==1)
        total=0j
        for partition in self.partitions[len(auxiliary)]:
            value=math.prod(copies-k for k in range(len(partition)))
            for block in partition:value*=self.ordered(self.aux,5,tuple(auxiliary[i] for i in sorted(block)))
            total+=value
        return self.ordered(self.states[state][1],state,active)*total

    def expectation(self,w: int,fs: tuple,ss: tuple,state: int,copies: int=63,Ref_hole: bool=True,field_scale: F=F(1))->complex:
        raw=self.states[state][2] and Ref_hole
        means=[-field_scale*self.gram(z,w)/2-(F(int(self.winds[z]['vn'][0]),self.den) if raw else 0) for z in ss]
        total=0j
        for assignment in itertools.product(range(3),repeat=len(ss)):
            gaussian=[p for p,role in enumerate(assignment) if role==0]
            factors=fs+tuple(self.diagonal(role-1,z) for role,z in zip(assignment,ss) if role)
            mean=[means[p] for p in gaussian]
            covariance=[[self.gram(ss[p],ss[q])/(8*field_scale) for q in gaussian] for p in gaussian]
            total+=float(wick.gaussian_linear_moment(mean,covariance))*self.role_moment(state,factors,copies)
        return math.exp(-float(field_scale*self.winds[w]['q2']))*total

    def evaluate(self,terms: dict,order: int,copies: int=63,Ref_hole: bool=True,field_scale: F=F(1),single_channel: bool=False)->dict[str,Any]:
        pieces=[];absolute=0.
        phase=1j**order
        for (w,fs,ss),coef in terms.items():
            chosen=[(0,-.5),(1,.5)] if single_channel else [(state,weight) for state,(weight,_,_) in enumerate(self.states)]
            value=coef*sum(weight*self.expectation(w,fs,ss,state,copies,Ref_hole,field_scale) for state,weight in chosen)*phase
            pieces.append(value);absolute+=abs(value)
        value=complex(math.fsum(z.real for z in pieces),math.fsum(z.imag for z in pieces))
        return dict(real=value.real,imaginary=value.imag,absolute_term_sum=absolute,
                    arithmetic='conditional IEEE complex128; exact rational geometry/Gaussian polynomial, no outward contraction enclosure',
                    grouped_terms=len(terms))


def higher_budget(B: F,dcar: F,onsite: F,force: list[F],horizon: list[F],order: int)->dict[str,Any]:
    terms={(1,0):B}
    for _ in range(order-1):
        out=defaultdict(F)
        for (m,d),c in terms.items():
            out[m,d+1]+=c*128*m;out[m,d]+=c*(128*m*m+2*m*onsite);out[m+1,d]+=c*m*dcar
            for l in range(1,d+1):out[m+1,d-l]+=c*math.comb(d,l)*force[l]
        terms=dict(out)
    derivative=sum((c*horizon[d] for (m,d),c in terms.items()),F(0))
    error=CLOCK**order*derivative/math.factorial(order)
    return dict(order=order,derivative_upper_exact=str(derivative),Taylor_remainder_upper_exact=str(error),positive_terms={f'{m},{d}':str(c) for (m,d),c in sorted(terms.items())})

def connected_control()->dict[str,Any]:
    """Independent literal finite-Fock commutators, NOT source-text tests."""
    from types import SimpleNamespace
    w=np.vstack((np.eye(4),np.eye(4)))/math.sqrt(2)
    p=w@w.T;q=(np.eye(8)-p)[:,:4]
    matrix=np.diag([.125,.25,.375,.5]).astype(complex)
    s=SimpleNamespace(p=p,w=w,empty_source=q,hops=np.asarray([matrix]),onsite=np.diag([.03,-.02,.04,-.01]))
    aux=SimpleNamespace(p=p,w=w,empty_source=q,hops=np.asarray([np.zeros((4,4),complex)]),onsite=np.zeros((4,4),complex))
    geo=dict(k=np.array([[1],[-1]]),x=np.array([0,1]),y=np.array([1,0]))
    engine=Engine(dict(native=dict(D=np.array([[1],[-1]]))),[s,aux],geo,dict(denominator=4,L_integer=np.array([[1,-1],[-1,1]])),100000)
    number=wick.finite_car(np.diag([1.]*4+[0.]*4))
    difference=wick.finite_car(np.diag([1.]*4+[-1.]*4))
    hmat=np.block([[s.onsite,matrix],[matrix,s.onsite]])
    matter=wick.finite_car(hmat)
    initial=engine.initial(0);symbols=initial;observations=[]
    def slater(occupied: np.ndarray)->np.ndarray:
        vector=np.zeros(256,complex)
        for sites in itertools.combinations(range(8),occupied.shape[1]):
            vector[sum(1<<i for i in sites)]=np.linalg.det(occupied[list(sites)])
        return vector
    bg=slater(w)
    source_vectors=[slater(np.column_stack((w,q[:,a]/np.linalg.norm(q[:,a])))) for a in range(4)]
    h0=16*difference@difference+matter
    shifted=difference-np.eye(256)
    hraw=16*shifted@shifted+matter
    comm0=number.copy();commraw=number.copy()
    for order in range(1,5):
        comm0=h0@comm0-comm0@h0;commraw=hraw@commraw-commraw@hraw
        if order>1:symbols=engine.derivative(symbols,order)
        if order%2:continue
        direct=(1j**order)*(-.5*np.vdot(bg,comm0@bg)+sum(engine.states[a+1][0]*np.vdot(v,commraw@v) for a,v in enumerate(source_vectors)))
        computed=engine.evaluate(symbols,order,copies=0)
        defect=abs(direct-computed['real']-1j*computed['imaginary'])
        relative=defect/max(1.,abs(direct))
        if relative>3e-10:raise ArithmeticError(f'Connected full-CAR order{order} control disagrees: {defect}, relative{relative}')
        observations.append(dict(order=order,direct_real=float(direct.real),direct_imaginary=float(direct.imag),computed=computed,defect=defect,relative_defect=relative))
    return dict(scope='independent two-site four-colour finiteCAR/HC64/onsite, auxiliary load0 control ONLY; actual consumer unchanged64 charged',observations=observations)



def independent_control()->dict[str,Any]:
    # Exact small CAR, nontrivial Coulomb, collective copies and Gaussian Weyl.
    p=np.array([[.5,.5],[.5,.5]],complex)
    a=np.array([[1.,2j],[-2j,-1.]])
    b=np.array([[2.,1.],[1.,-2.]])
    psi=np.array([0.,1.,1.,0.])/math.sqrt(2)
    rows=[]
    for n in (2,4,6):
        fs=([a,b]*3)[:n];vector=psi.astype(complex)
        for f in reversed(fs):vector=wick.finite_car(f)@vector
        observed=complex(np.vdot(psi,vector));value=wick.ordered_car_moment(p,fs)
        defect=abs(observed-value)
        if defect>2e-10:raise ArithmeticError('Independent higher ordered CAR control')
        rows.append(dict(order=n,defect=defect,real=observed.real,imaginary=observed.imag))
    exact=wick.gaussian_linear_moment([F(-1,4)]*6,[[F(1,32)]*6 for _ in range(6)])
    # Independent direct Hermite/normal recursion E(X^n)=mu E(X^(n-1))+(n-1)var E(X^(n-2)).
    mu=F(-1,4);var=F(1,32);g=[F(1),mu]
    for n in range(2,7):g.append(mu*g[-1]+(n-1)*var*g[-2])
    if exact!=g[6]:raise ArithmeticError('Independent Weyl sixth contraction')
    wrong=wick.ordered_car_moment(p,[a,b])- (np.trace(p@a)*np.trace(p@b)+np.trace(p@a@p@b))
    if abs(wrong)<1:raise ArithmeticError('Wick-plus adverse control blind')
    copy=wick.collective_car_moment(p,[b]*4,63)
    wrongcopy=63**4*wick.ordered_car_moment(p,[b]*4)
    if abs(copy-wrongcopy)<1:raise ArithmeticError('63 copies adverse control blind')
    return dict(ordered_CAR=rows,Gaussian_sixth_exact=str(exact),wrong_Wick_plus_defect=float(abs(wrong)),wrong_auxiliary_copy_weight_defect=float(abs(copy-wrongcopy)),independent_second=prior.initial_coefficient_oracle())


def run(output: Path,term_cap: int=1000000,maximum_order: int=6)->dict[str,Any]:
    start=time.monotonic();control=independent_control();control['connected_full_CAR']=connected_control()
    data,sp,inputs,_=body.full_inputs();geo=current.geometry(data,sp[0]);geom=prior.prices.exact_geometry(data['native'],geo)
    captured={**body.CAPTURED,**inputs};channels=hop.initial_channels(sp[0])
    control['parity']=prior.parity_controls(data,sp,captured)
    control['Gauss_and_physical_density']=hop.controls(data['native'],geo,geom)
    engine=Engine(data,sp,geo,geom,term_cap)
    supplier=moments.MomentBounds(data,order=7)
    requests=[supplier.request_word([j]) for j in range(len(geo['k']))]
    force=[max(F(r['force_Lp_upper'][p]) for r in requests) for p in range(8)]
    # The raw bounds originate per normalized channel, then common norm factor
    # is algebraically removed from this linear hierarchy; no changed mass.
    horizon=[max(F(r['states']['raw_source']['horizon_a_upper'][p])/F(r['states']['raw_source']['norm_upper']) for r in requests) for p in range(8)]
    b=supplier.b[0]
    shared=(geo['y'][:,None]==geo['x'][None,:])|(geo['x'][:,None]==geo['y'][None,:])
    dcar=max(2*sum((bb[j] for j in np.flatnonzero(shared[i])),F(0)) for bb in supplier.b for i in range(len(b)))
    onsite=max(abs(F(s.profile['onsite'])) for s in sp)
    source_name='joined_phi_u1_actual_compact_source_action_certificate_verified_receipt_20261010.json'
    transfer_name='joined_phi_u1_nonlocal_body_transport_verified_source_transfer_receipt_20261010.json'
    for name in (source_name,transfer_name):captured[name]=(ROOT/name).read_bytes()
    source=json.loads(captured[source_name]);transfer=json.loads(captured[transfer_name])
    epsilon=F(source['action_input']['epsilon']['upper']);radius=F(source['source_weights']['prepared_radius_L2']['upper']);b1=F(transfer['fundamental_active_W1_upper_exact'])
    eta0=b1*radius;etan=8*eta0;physical=8*CLOCK*epsilon+2*etan+8*(2*eta0+eta0*eta0)
    results={};rows=[]
    for i in np.flatnonzero(geo['y']==0):
        x=int(geo['x'][i])
        if x not in results:
            receiver_start=time.monotonic();terms=engine.initial(x);coefficients={};cost_failure=None
            for order in range(2,maximum_order+1):
                try:terms=engine.derivative(terms,order)
                except SymbolCostExceeded as e:cost_failure=str(e);break
                if order%2==0:
                    coefficients[order]=engine.evaluate(terms,order)
                    if order==2:
                        known=prior.second_coefficients(x,sp,geo,channels,data['native']['Lplus'])['complete_second']
                        defect=abs(coefficients[order]['real']-known)
                        if defect>2e-7:raise ArithmeticError(f'Initial grouped second disagrees with independent native second at{x}: {defect}')
                        coefficients[order]['independent_second_defect']=defect
                        variants={name:engine.evaluate(terms,2,**parameters)['real'] for name,parameters in {
                            'wrong_Ref4_raw':dict(Ref_hole=False),'wrong_auxiliary_load0':dict(copies=0),
                            'wrong_field_density_variance4':dict(field_scale=F(2)),
                            'wrong_single_raw_channel':dict(single_channel=True)}.items()}
                        components=prior.second_coefficients(x,sp,geo,channels,data['native']['Lplus'])
                        variants['wrong_HC32']=known-components['HC_second']/2
                        coefficients[order]['actual_wrong_controls']={name:dict(real=v,defect=abs(v-known)) for name,v in variants.items()}
            results[x]=dict(coefficients=coefficients,symbolic_cost_failure=cost_failure,elapsed_seconds=time.monotonic()-receiver_start)
            print(json.dumps(dict(receiver=x,orders=list(coefficients),cost_failure=cost_failure,elapsed_seconds=results[x]['elapsed_seconds'])),flush=True)
        result=results[x];coeff=result['coefficients'];B=sum((b[j] for j in engine.incident[x]),F(0))
        tails={n:higher_budget(B,dcar,onsite,force,horizon,n) for n in (4,6,8)}
        achieved=max(coeff);tail_order=achieved+2
        estimate=sum(float(CLOCK**n)*c['real']/math.factorial(n) for n,c in coeff.items())
        error=F(tails[tail_order]['Taylor_remainder_upper_exact'])
        rows.append(dict(path=int(i),receiver=x,length=int(geo['k'][i]@geo['k'][i]),initial_coefficients=coeff,conditional_density_change=estimate,
                         remainder_order=tail_order,full_H_remainder_upper_exact=str(error),all_tail_prices=tails,
                         physical_density_eta_n_exact=str(etan),actual_E0_eta0_exact=str(eta0),both_endpoints_physical_error_exact=str(physical),
                         conditional_signed_margin=abs(estimate)-float(error+physical),conditional_sign_survives=abs(estimate)>float(error+physical),
                         symbolic_cost_failure=result['symbolic_cost_failure'],nohop_density_change_exact='0'))
    if len(rows)!=12:raise ArithmeticError('Full actual twelve scope')
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file and Path(file).parent==ROOT and Path(file).suffix=='.py':captured.setdefault(Path(file).name,Path(file).read_bytes())
    for name,raw in captured.items():
        if name=='parity_integer_clifford_geometry.json':continue
        if (ROOT/name).read_bytes()!=raw:raise ArithmeticError('Source/input mutation '+name)
    archive=output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in sorted(captured.items()):z.writestr(name,raw)
    numerical=output.with_suffix('.npz')
    np.savez_compressed(numerical,signed_winds=geo['k'],signed_q=geo['q'],signed_delta=geo['delta'],active_P=sp[0].p,auxiliary_P=sp[1].p,
                        raw_four_projectors=np.asarray([c['p'] for c in channels]),raw_four_weights=np.asarray([c['weight'] for c in channels]))
    result=dict(status='INITIAL_DENSITY_HIGHER_JETS_CONDITIONAL_SIGNS_SURVIVE' if all(r['conditional_sign_survives'] for r in rows) else 'INITIAL_DENSITY_HIGHER_JETS_FAILED_REGIME',
                plan='issue-222-induced-photon-delta13-continuation-20261008',plan_revision=REVISION,clock_exact=str(CLOCK),clock_adopted=False,
                proof=PROOF,receiver_rows=rows,controls=control,symbolic_aggregation_counts=engine.word_counts,
                term_cap=term_cap,maximum_requested_order=maximum_order,raw_mass_exact='1/2',all63_auxiliary=True,all129_field=True,
                all768_signed_native=True,HC_HE_coefficient_exact='64',initial_arithmetic='conditional, NOT outward',finite_time_kernel_admitted=False,
                source_sha256=digest(Path(__file__).read_bytes()),inputs=dict(path=archive.name,sha256=digest(archive.read_bytes()),files={n:digest(v) for n,v in captured.items()}),
                numerical=dict(path=numerical.name,sha256=digest(numerical.read_bytes())),runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,platform=platform.platform()),
                elapsed_seconds=time.monotonic()-start)
    output.write_text(json.dumps(result,indent=2,default=hop.encode)+'\n');return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--term-cap',type=int,default=1000000)
    parser.add_argument('--maximum-order',type=int,choices=(4,6),default=6)
    args=parser.parse_args()
    with threadpool_limits(limits=1):receipt=run(args.output,args.term_cap,args.maximum_order)
    print(json.dumps(dict(status=receipt['status'],actual12=len(receipt['receiver_rows']),elapsed_seconds=receipt['elapsed_seconds'])),flush=True)
