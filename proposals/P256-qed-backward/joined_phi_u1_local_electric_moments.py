"""Gauss-complete local electric moments on the unchanged full N4 BODY.

Derived inequalities, not interacting Wick or measured evolution.  The NEW
2^-16 horizon is a probe, not adoption of a clock/source/quantum law.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
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
import joined_phi_u1_current_signal_interval_certificate as ar
import joined_phi_u1_electric_locality_response as locality
import joined_phi_u1_electric_hop_transport as hop

ROOT = Path(__file__).resolve().parent
PREFIX = 'joined_phi_u1_local_electric_moments'
CLOCK = F(1, 2**16)
REVISION = '9a2fd8299a741b12cbc9effe2c6a264ce87e2deef1bf73787109d7604849ce49'
PINS = {
 'joined_phi_u1_current_signal_interval_certificate.py': '386c684f42aa3715dbf370d075cb1b4f5a1b0a645c2784fd9ace2c97ee491810',
 'joined_phi_u1_electric_locality_response.py': '1274a84f49a0b5d873ca01ba9b30ae2ce6fa4157126b55c4a75f5a5e1abce045',
 'joined_phi_u1_electric_hop_transport.py': '205301d6416f274ace25a0f012f702858c5e66580d9211d36b5b0719f53b1e0b',
 'induced_phi_current_n4_root_static_slow_certificate_actual_receipt.json': '79f49c752b4c2cc716d0291f1c38d448093a94f231412900be2c0e730c67dfee',
}
PROOF = {
 'tilt': 'For orthonormal occupied W and real t, B=Wdag exp(tV)W>0; Pt=exp(tV/2)W B^-1 Wdag exp(tV/2) is an orthogonal projector. Lprime=Tr(Pt V). Ptprime=(V Pt+Pt V)/2-Pt V Pt, hence Lsecond=Tr(Pt V²)-Tr(Pt V Pt V)=|| (1-Pt)V Pt ||HS². In the Pt/(1-Pt) block decomposition Tr(V²)=||A||HS²+||C||HS²+2||B||HS². Therefore 0<=Lsecond<=Tr(V²)/2 for ALL REAL t, with NO assumption [Pt,V]=0. For V=diag(v) tensor I4 this is2sum(v²). Integrating Hessian twice gives log E exp(t(X-EX))<=t²sigma²/2 for positive and negative real t.',
 'product': 'Only at t=0: independent occupied active and63 auxiliary Slaters plus independent Gaussian Pi. Full64site V is used. Sigma²=||q||²/8+128sum(v²). Ref background4 and source3 are deterministic and shift only the mean. Uniform rank128 Bloch seas have local total number2; sum(v)=0. An added normalized empty orbital contributes udag V u, and Ref hole subtracts v0. Thus |mu_source|<=2vmax; background mu=0. Four unnormalized source channels weight ||(1-P)e0alpha||²/4; sum weights=.5 exactly by uniform local density2. No individual weight equality is assumed.',
 'tail': 'Chernoff optimization gives Pr(|X-EX|>=z)<=2exp(-z²/(2sigma²)). Layer-cake integration yields E|X-EX|^(2k)<=2(2sigma²)^k k! for k>=1; k1 improves to sigma² from the Hessian at0. For odd absolute powers use Cauchy-Schwarz between adjacent even moments. Minkowski gives ||s^k psi||/||psi|| <=(central_even_bound^(1/(2k))+|mu|)^k. No unitvariance substitution.',
 'cumulants': 'Centered cumulants are derivatives of logdet at0, not derivatives of a future Slater state. The formal determinant series gives kappa_r=sum_(m=1..r) (-1)^(m+1) r!/m sum_(n1+...+nm=r,ni>=1) Tr(P V^n1 ... P V^nm)/(n1!...nm!). Alternatively moment-cumulant partitions give |kappa_r|<=sum_pi (|pi|-1)! product_B E|X-EX|^|B|. The API implements this latter positive bound through a recurrence enumerating block-size counts, with centered first moment exactly0. Gaussian cumulants vanish above2. These are absolute cumulant prices, not the invalid inference coefficientwise bound from a real mgf inequality.',
 'hierarchy': 'On a common Schwartz/finiteCAR core, s=q.Pi+v.rho commutes HE64+HC64 and all onsite terms. A charged signed hop Or obeys [s,Or]=(w.inc_r)Or by q.qr+delta Lplus deltar=w.inc_r. Hence [s^k,Or]=Or sum_(j<k) binom(k,j)(w.inc_r)^(k-j)s^j. Bounded hopping preserves the graph domains by this finite commutator inequality. Duhamel and Gronwall comparison give a_k(t)<=a_k(0)+int sum_(j<k)binom(k,j)L_(k-j)a_j. N lower-triangular nilpotent: a(t)<=sum_(m=0..k)t^m N^m a(0)/m!. It applies to full-H vectors, not future Wick. Lp uses literal signed paths with active load1 and all63 aux load63; neutral charge0 and onsite commute and contribute0, NOT omitted nonzero force.',
 'insertion': 'For a bounded ordered hop word B=O_i1...O_im, s B=B(s+sum w.inc_ir). Thus ||s^k B psi||<=product b_ir sum_(j<=k)binom(k,j)|sum overlap|^(k-j)a_j(0). Apply the SAME hierarchy to U(t)Bpsi with this initial vector, not to a guessed norm of B U(t)psi. At a future insertion B U(t)psi the shifted triangle uses a_j(t). J/C directional vectors are sums of these words with respective exact coefficient weights |q_r.f| or |q_r.f q_r.g|. For their vector norms sum the paid word bounds. Electric derivatives 64(2s+inc²) are finite polynomials and priced by polynomial_norm. Neither an insertion norm nor a source-weighted moment is left unpaid.',
 'arithmetic': 'Geometry/incidence/Lplus and all hierarchy polynomial coefficients are exact rationals. Root-box square roots, moment roots and norms are mpmath directed intervals; their UPPER rational endpoints are propagated as exact positive bounds. Root exact-law conditionality is inherited, not re-proved. Floating full_inputs seas/channel weights and noncommuting tilt controls are observations only; not used to admit a future Slater or strengthen a bound.',
}

def upper(x: Any) -> F:
    return ar.end(x, True)

def root_upper(x: F, degree: int = 2) -> F:
    if x < 0 or degree < 1:
        raise ArithmeticError('Invalid positive moment root')
    return upper(ar.exact(x)**ar.exact(F(1,degree)))

def central_moment(sigma2: F, n: int) -> F:
    if n == 0: return F(1)
    if n == 1: return root_upper(sigma2)
    if n == 2: return sigma2
    if n % 2: return root_upper(central_moment(sigma2,n-1)*central_moment(sigma2,n+1))
    k=n//2
    return 2*(2*sigma2)**k*math.factorial(k)

def cumulant_price(sigma2: F, n: int) -> F:
    if n == 1: return F(0)
    if n == 2: return sigma2
    # Block counts c_l with sum l*c_l=n; centered singleton blocks vanish.
    total=F(0)
    def visit(length: int, left: int, blocks: int, denom: int, product: F) -> None:
        nonlocal total
        if length > n:
            if left == 0 and blocks:
                total += F(math.factorial(n)*math.factorial(blocks-1),denom)*product
            return
        for count in range(left//length+1):
            visit(length+1,left-length*count,blocks+count,
                  denom*math.factorial(count)*math.factorial(length)**count,
                  product*central_moment(sigma2,length)**count)
    visit(2,n,0,1,F(1))
    return total

def evolve(initial: list[F], force: list[F], horizon: F) -> list[F]:
    if horizon < 0: raise ArithmeticError('Negative comparison horizon')
    order=len(initial)-1
    total=list(initial); term=list(initial)
    for m in range(1,order+1):
        term=[sum((F(math.comb(k,j))*force[k-j]*term[j] for j in range(k)),F(0))*horizon/m for k in range(order+1)]
        total=[a+b for a,b in zip(total,term)]
    return total

class MomentBounds:
    """Reusable arbitrary integer192-wind API, including linked sums/differences.

    Signed native indices0..383 are forward source-row/target-column;
    indices384..767 are their adjoints. No winding length restriction.
    """
    def __init__(self, data: dict[str,Any]|None=None, order: int=4):
        if order < 4: raise ArithmeticError('At least fourth moment hierarchy required')
        self.order=order
        ar.iv.dps=75
        if data is None:
            data,self.species,self.inputs,_=body.full_inputs()
        else:
            self.species=[];self.inputs={}
        self.data=data
        self.paths=body.joined.old.native_paths(data['profiles'][1],data['native'])
        self.geo=ar.geometry(data['native'],self.paths)
        self.k=self.geo['k']; self.den=self.geo['den']
        self.lint=self.geo['lint']; self.d=np.asarray(data['native']['D'],np.int64)
        self.saved=data['descriptor']['exact_saved_Phi_mean'] if 'exact_saved_Phi_mean' in data['descriptor'] else json.loads((ROOT/body.ARCHIVE).read_bytes())['input_descriptor']['exact_NEW_allq_BO_mean']
        g0=F(self.saved['parameters']['g0']['upper']);gs=F(self.saved['parameters']['gs']['upper'])
        self.b=[]
        for profile in data['profiles'][1:]:
            lookup={(p['axis'],p['length']):p for p in profile['paths']}
            forward=[]
            for path in self.paths:
                p=lookup[path['axis'],path['length']]
                forward.append(upper(4*(ar.exact(abs(F(p['c0'])))*ar.iv.sqrt(ar.exact(g0))+ar.exact(abs(F(p['ci'])))*ar.iv.sqrt(ar.exact(gs)))))
            self.b.append(forward+forward)
        self.groups={}
        for r in range(len(self.k)):
            strength=self.b[0][r]+63*self.b[1][r]
            self.groups.setdefault(strength,[]).append(r)
        self.cache: dict[tuple[Any,...],dict[str,Any]]={}

    def wind(self, indices: list[int]) -> np.ndarray:
        if any(not 0<=i<len(self.k) for i in indices): raise ArithmeticError('Invalid signed path index')
        return np.sum(self.k[indices],axis=0,dtype=np.int64) if indices else np.zeros(192,np.int64)

    def request_word(self, indices: list[int], horizon: F=CLOCK) -> dict[str,Any]:
        return self.request_wind(self.wind(indices),horizon)

    def request_wind(self, w: Any, horizon: F=CLOCK) -> dict[str,Any]:
        values=list(w)
        if len(values)!=192 or any(F(v).denominator!=1 for v in values):
            raise ArithmeticError('Wind must be192 exact integers')
        # Fast bounded integer path; arbitrary larger winds use Python integers.
        dtype=object if max((abs(int(v)) for v in values),default=0)>10**6 else np.int64
        w=np.asarray([int(v) for v in values],dtype=dtype)
        delta=self.d@w
        support=[(y,int(v)) for y,v in enumerate(delta) if v]
        vn=[sum(int(self.lint[x,y])*value for y,value in support) for x in range(64)]
        v2=F(sum(v*v for v in vn),self.den**2)
        long=F(sum(int(delta[x])*vn[x] for x in range(64)),self.den)
        wind2=sum(int(x)**2 for x in w)
        q2=F(wind2)-long
        if q2<0 or sum(vn)!=0: raise ArithmeticError('Gauss geometry positivity/zero mean failed')
        vmax=F(max(map(abs,vn)),self.den)
        overlap=self.k@w
        force=[F(0)]*(self.order+1)
        for b,indices in self.groups.items():
            nums=[abs(int(overlap[r])) for r in indices]
            for p in range(1,self.order+1): force[p]+=b*sum(n**p for n in nums)
        signature=(q2,v2,vmax,*force,horizon)
        if signature not in self.cache:
            sigma2=q2/8+128*v2
            states={}
            for label,mass,mu in [('background',F(1),F(0)),('raw_source',F(1,2),2*vmax)]:
                norm=root_upper(mass)
                initial=[norm]+[norm*(root_upper(central_moment(sigma2,2*k),2*k)+mu)**k for k in range(1,self.order+1)]
                initial[1]=norm*root_upper(sigma2+mu*mu)
                states[label]={'mass_exact':str(mass),'norm_upper':str(norm),'mean_abs_upper':str(mu),
                    'initial_a_upper':[str(x) for x in initial],
                    'horizon_a_upper':[str(x) for x in evolve(initial,force,horizon)]}
            self.cache[signature]={'price_class_index':len(self.cache),'horizon_exact':str(horizon),'order':self.order,
                'q_squared_exact':str(q2),'density_v_squared_exact':str(v2),'vmax_exact':str(vmax),
                'centered_sigma_squared_upper_exact':str(sigma2),
                'force_Lp_upper':[str(x) for x in force],
                'centered_absolute_moment_upper':[str(central_moment(sigma2,n)) for n in range(2*self.order+1)],
                'centered_cumulant_abs_upper':[str(cumulant_price(sigma2,n)) for n in range(1,self.order+1)],
                'states':states}
        return {**self.cache[signature],'wind_sparse':[[i,int(v)] for i,v in enumerate(w) if v],
                'delta_sparse':[[i,int(v)] for i,v in enumerate(delta) if v],
                'v_numerator':vn,'v_denominator':self.den,
                'overlaps_signed_native':[int(x) for x in overlap]}

    def inserted(self, w: Any, indices: list[int], species: list[int]|None=None,
                 horizon: F=CLOCK) -> dict[str,Any]:
        """Paid U(t)Bpsi AND B U(t)psi moments; ordered words may repeat paths."""
        request=self.request_wind(w,horizon)
        species=[0]*len(indices) if species is None else species
        if len(indices)!=len(species) or any(s not in (0,1) for s in species):
            raise ArithmeticError('Insertion species must be active0/auxiliary1 for each factor')
        coefficient=F(1); shift=0
        for i,s in zip(indices,species):
            if not 0<=i<len(self.k): raise ArithmeticError('Invalid inserted signed path')
            coefficient*=self.b[s][i]
            shift+=request['overlaps_signed_native'][i]
        def shifted(a: list[F]) -> list[F]:
            return [coefficient*sum((F(math.comb(k,j))*abs(shift)**(k-j)*a[j] for j in range(k+1)),F(0)) for k in range(self.order+1)]
        states={}
        force=[F(v) for v in request['force_Lp_upper']]
        for label,state in request['states'].items():
            initial=shifted([F(x) for x in state['initial_a_upper']])
            future=shifted([F(x) for x in state['horizon_a_upper']])
            states[label]={'initial_Bpsi_a_upper':[str(x) for x in initial],
                'U_Bpsi_horizon_a_upper':[str(x) for x in evolve(initial,force,horizon)],
                'B_Upsi_horizon_a_upper':[str(x) for x in future]}
        return {'word_indices':indices,'species':species,'word_norm_upper':str(coefficient),
                'electric_shift_exact':shift,'states':states,'horizon_exact':str(horizon)}

    @staticmethod
    def polynomial_norm(coefficients: list[Any], moments: list[Any]) -> F:
        if len(coefficients)>len(moments): raise ArithmeticError('Insufficient paid moment order')
        return sum((abs(F(c))*F(a) for c,a in zip(coefficients,moments)),F(0))

    def directional_insertion(self, w: Any, weights: list[Any], species: int=0,
                              horizon: F=CLOCK) -> dict[str,Any]:
        """Weights are caller's exact J/C direction coefficients, not norm guesses."""
        if len(weights)!=len(self.k): raise ArithmeticError('All signed paths required')
        sums={label:{key:[F(0)]*(self.order+1) for key in ('initial_Bpsi_a_upper','U_Bpsi_horizon_a_upper','B_Upsi_horizon_a_upper')} for label in ('background','raw_source')}
        for i,weight in enumerate(weights):
            weight=abs(F(weight))
            if not weight: continue
            item=self.inserted(w,[i],[species],horizon)
            for label,fields in sums.items():
                for key,values in fields.items():
                    fields[key]=[a+weight*F(b) for a,b in zip(values,item['states'][label][key])]
        return {label:{key:[str(a) for a in values] for key,values in fields.items()} for label,fields in sums.items()}

    def current_contact_insertions(self, w: Any, horizon: F=CLOCK) -> dict[str,Any]:
        """Paid all129 vector/tensor HS insertion moments; all64 charged copies."""
        request=self.request_wind(w,horizon)
        force=list(map(F,request['force_Lp_upper']))
        result={}
        for name in ('J','C'):
            totals={label:{key:[F(0)]*(self.order+1) for key in ('initial_Bpsi_a_upper','U_Bpsi_horizon_a_upper','B_Upsi_horizon_a_upper')} for label in ('background','raw_source')}
            for r in range(len(self.k)):
                q2=F(int(self.geo['gram'][r,r]),self.den)
                weight=root_upper(q2) if name=='J' else q2
                coefficient=weight*(self.b[0][r]+63*self.b[1][r])
                shift=abs(request['overlaps_signed_native'][r])
                for label,state in request['states'].items():
                    def shifted(field: str) -> list[F]:
                        a=list(map(F,state[field]))
                        return [coefficient*sum((F(math.comb(k,j))*shift**(k-j)*a[j] for j in range(k+1)),F(0)) for k in range(self.order+1)]
                    initial=shifted('initial_a_upper')
                    fields={'initial_Bpsi_a_upper':initial,'U_Bpsi_horizon_a_upper':evolve(initial,force,horizon),'B_Upsi_horizon_a_upper':shifted('horizon_a_upper')}
                    for key,values in fields.items():
                        totals[label][key]=[a+b for a,b in zip(totals[label][key],values)]
            result[name]={label:{key:[str(a) for a in values] for key,values in fields.items()} for label,fields in totals.items()}
        return {'horizon_exact':str(horizon),'norm_convention':'J:directsum129 vector norm; C:129x129 HilbertSchmidt tensor norm; sum load1+63 paid triangle','states_by_insertion':result}

def tilted_control() -> dict[str,Any]:
    # Deliberately noncommuting rank2 Slater projector and diagonal V.
    a=np.array([[1.,2.],[2.,-1.],[1.,1.],[-2.,1.]])
    w=np.linalg.qr(a)[0];v=np.diag([-.7,.2,1.1,-.3]);cap=float(np.trace(v@v)/2)
    rows=[]
    for t in [-20.,-2.,0.,2.,20.]:
        e=np.diag(np.exp(t*np.diag(v)/2));b=w.T@(e@e)@w
        pt=e@w@np.linalg.solve(b,w.T@e)
        variance=float(np.trace(pt@v@v)-np.trace(pt@v@pt@v))
        block=float(np.linalg.norm((np.eye(4)-pt)@v@pt)**2)
        if abs(variance-block)>1e-8 or variance>cap+1e-8 or variance < -1e-8:
            raise ArithmeticError('Noncommuting tilted Hessian control failed')
        rows.append(dict(t=t,variance=variance,block_variance=block,commutator_norm=float(np.linalg.norm(pt@v-v@pt))))
    if rows[2]['commutator_norm']<.1: raise ArithmeticError('Commuting control is blind')
    return {'conditional_binary64_noncommuting_control':rows,'all_real_theorem':'PROOF.tilt, not a finite t extrapolation','variance_cap':cap}

def run(output: Path, order: int=4) -> dict[str,Any]:
    started=time.monotonic(); captured={}
    for name,pin in PINS.items():
        raw=(ROOT/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=pin: raise ArithmeticError('Pinned supplier changed '+name)
        captured[name]=raw
    api=MomentBounds(order=order); captured.update(api.inputs);captured.update(body.CAPTURED)
    geo=api.geo;geo['native']=api.data['native']
    controls=locality.controls(geo,api.paths)
    controls['tilted_projector']=tilted_control()
    if (len(api.k),set(np.sum(api.k*api.k,axis=1)),[p['load'] for p in api.data['profiles']])!=(768,{1,2},[1,1,63]):
        raise ArithmeticError('Full native topology/loads missing')
    # Repeated edge and toron winds exercise composite API, not only native edges.
    catalogue=[];allbounds={};native_ids=[]
    def take(w: np.ndarray) -> int:
        r=api.request_wind(w)
        # Geometry and force values determine prices; wind reconstruction kept separately.
        price={key:val for key,val in r.items() if key not in ('wind_sparse','delta_sparse','v_numerator','v_denominator','overlaps_signed_native')}
        key=r['price_class_index']
        if key not in allbounds: allbounds[key]=len(catalogue);catalogue.append(price)
        return allbounds[key]
    for w in api.k: native_ids.append(take(w))
    x,y=geo['x'],geo['y'];dots=api.k@api.k.T
    connected=(x[:,None]==x[None,:])|(x[:,None]==y[None,:])|(y[:,None]==x[None,:])|(y[:,None]==y[None,:])|(dots!=0)
    selected=sorted(set(int(site) for site in x[y==0]))
    pairs=np.argwhere(np.triu(connected))
    words=[]
    for i,j in pairs:
        i=int(i);j=int(j)
        for sign in (-1,1):
            words.append([i,j,sign,take(api.k[i]+sign*api.k[j])])
    torons=[]
    for axis in range(3):
        wind=np.zeros(192,np.int64);wind[axis::3]=1
        r=api.request_wind(wind)
        if F(r['q_squared_exact'])!=64 or F(r['density_v_squared_exact'])!=0:
            raise ArithmeticError('Missing toron completion')
        torons.append(r)
    sample=api.request_word([0]);sigma=F(sample['centered_sigma_squared_upper_exact'])
    missing_aux=F(sample['q_squared_exact'])/8+2*F(sample['density_v_squared_exact'])
    if sigma<=missing_aux or central_moment(F(1),4)!=16:
        raise ArithmeticError('Aux/unitvariance control blind')
    # Density variance2 owns Pi variance1/8; wrong variance1 gives1/4.
    controls['missing_aux_sigma2_exact']=str(missing_aux)
    controls['full_sigma2_exact']=str(sigma)
    controls['wrong_density_unitvariance_Pi_sigma2_exact']=str(F(sample['q_squared_exact'])/4+128*F(sample['density_v_squared_exact']))
    controls['central_fourth_unit_sigma_bound_exact']='16'
    controls['variance_quarter_counterexample']={'V_exact':[[-1,0],[0,1]],'P_exact':[['1/2','1/2'],['1/2','1/2']],'actual_variance_exact':'1','correct_traceV2_half':'1','wrong_traceV2_quarter':'1/2'}
    controls['Gaussian_Pi_moments_exact']={str(k):str(F(math.prod(range(1,2*k,2)),8**k)) for k in range(1,order+1)}
    controls['wrong_unit_Pi_fourth_exact']='3'
    controls['native_counts']={'signed':768,'one':int(np.count_nonzero(np.sum(api.k*api.k,axis=1)==1)),'two':int(np.count_nonzero(np.sum(api.k*api.k,axis=1)==2)),'AP_negative_forward':sum(p['seam']==-1 for p in api.paths)}
    controls['species_force_accounting']={'active':1,'auxiliary':63,'neutral_heavy_charge':0,'neutral_force_exact':0,'onsite_force_exact':0,'HE':64,'HC':64}
    channels=hop.initial_channels(api.species[0])
    channel_receipt=[{'alpha':c['alpha'],'weight_conditional_binary64':c['weight'],'rank':c['w'].shape[1],'Ref_occupancy':3} for c in channels]
    # Source and future insertion observations are NOT used to tighten exact prices.
    examples=[]
    for word in ([0],[0,1],[0,384],[0,0,0,0]):
        examples.append(api.inserted(api.wind([0,1]),list(word)))
    weights=[F(1) if r<3 else F(0) for r in range(768)]
    inserted=api.directional_insertion(api.k[0],weights)
    current_contact=api.current_contact_insertions(api.k[0])
    maxima={}
    for label in ('background','raw_source'):
        maxima[label]={field:[str(max(F(c['states'][label][field][k]) for c in catalogue)) for k in range(order+1)] for field in ('initial_a_upper','horizon_a_upper')}
    native_maxima={label:{field:[str(max(F(catalogue[i]['states'][label][field][k]) for i in native_ids)) for k in range(order+1)] for field in ('initial_a_upper','horizon_a_upper')} for label in ('background','raw_source')}
    for module in tuple(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file and Path(file).parent==ROOT and Path(file).suffix=='.py':captured.setdefault(Path(file).name,Path(file).read_bytes())
    if not all((ROOT/name).read_bytes()==raw for name,raw in captured.items()): raise ArithmeticError('Input/source mutation')
    archive=output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in sorted(captured.items()):z.writestr(name,raw)
    result={'status':'DERIVED_LOCAL_ELECTRIC_FULL_H_MOMENT_UPPER_BOUNDS','governing_plan':body.PLAN,'governing_revision':REVISION,'template_revision':'89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
        'clock_exact':str(CLOCK),'proof':PROOF,'operator_controls':controls,'four_source_channels':channel_receipt,'raw_source_mass_exact':'1/2','background_Ref':4,'source_Ref':3,
        'price_catalogue':catalogue,'native_signed_path_price_ids':native_ids,'linked_words_columns':['first_signed_path','second_signed_path','second_sign','price_id'],'linked_words':words,
        'selected_density_sites':selected,'toron_winds':torons,'maximum_all_catalogue_moments':maxima,'paid_insertion_examples':examples,'paid_directional_insertion_example':inserted,
        'maximum_native_one_two_moments':native_maxima,
        'paid_all129_current_contact_insertions':current_contact,
        'interface':{'class':'MomentBounds','request_wind':'integer192 wind; arbitrary sums/differences; order>=4','request_word':'signed native path indices in literal forward384/adjoint384 order','inserted':'both U(t)Bpsi and B U(t)psi, ordered paths and species0active/1aux','directional_insertion':'exact caller weights per all768 signed paths, absolute sum of fully paid bounds','polynomial_norm':'exact coefficient list and paid a_k list; reject insufficient order'},
        'arithmetic_scope':'Exact rational geometry/hierarchy; outward root endpoints; conditional pinned root theorem; floating Slater weights/tilt controls observational only. No scientific sign/pole/cone/source admission.',
        'input_bytes':{'path':archive.name,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'stable':True},'source_input_sha256':{name:hashlib.sha256(raw).hexdigest() for name,raw in captured.items()},
        'runtime':{'executable':sys.executable,'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'mpmath':ar.mp.__version__,'sympy':ar.sy.__version__,'interpreter_sha256':hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()},'elapsed_seconds':time.monotonic()-started}
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/(PREFIX+'_receipt.json'));parser.add_argument('--order',type=int,default=4);args=parser.parse_args()
    with threadpool_limits(limits=1): receipt=run(args.output,args.order)
    print(json.dumps({'status':receipt['status'],'signed_paths':len(receipt['native_signed_path_price_ids']),'linked_words':len(receipt['linked_words']),'price_classes':len(receipt['price_catalogue']),'elapsed_seconds':receipt['elapsed_seconds']}),flush=True)
