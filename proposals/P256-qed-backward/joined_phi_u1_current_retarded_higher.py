"""Whole129 initial retarded-current trace construction and rooted full-H prices.

This consumer never calls M3 an RJJ coefficient. Failed complete expansions are
retained, and no incomplete seed sum is emitted as an initial coefficient.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import zipfile
from typing import Any
import numpy as np
from threadpoolctl import threadpool_limits
import joined_phi_u1_body_word as body
import joined_phi_u1_current_moments as current
import joined_phi_u1_current_signal_interval_certificate as ar
import joined_phi_u1_initial_density_higher_jets as symbols
import joined_phi_u1_local_electric_moments as local

ROOT=Path(__file__).resolve().parent
PREFIX='joined_phi_u1_current_retarded_higher'
CLOCK=F(1,65536)
PROOF={
 'kernel':'RJJ(t)=+i raw<[J(t),J]>, J=-partial H. Thus R1=-F (old certificate uses -i), R3=raw<[adH^3 J,J]>, NOT mixed M3. Both complete initial preparations have antiunitary parity: J is odd, H and preparation are even, so RJJ is real odd, RJJ(0)=0. No naked stationarity.',
 'trace':'Tr129=sum_rs (q_r.q_s) signed ordered native CAR/Weyl contraction. q_r.q_s=k_r P_T k_s EXACT rational; all129 incl3torons. |Tr129 R|/sqrt129<=||R||HS, a full-space HS witness, not 129 singular-value lower.',
 'symbol':'A native e^(ikTheta) CAR word times right-normal product of s_z=q_z.Pi+delta_z.Lrho. adH adds128 s_w+64|k_w|^2, onsite/CAR connected substitutions, and negative finite differences. Final [A,J] has opposite connected substitutions and positive finite differences. The two current factors -i contribute -q_r.q_s. Ref3,1+63 copies, sameTheta and ordered products remain in every expectation.',
 'tail':'A5=i^5 adH^5 J is Hermitian. Whole HS commutator norm <=2 ||A5 Upsi||directsum129 ||Jpsi||directsum129. Integral Taylor remainder at odd order3 is <=B5*b0*t^5/60, b0^2=outward initial TrJJ. Future A5 price uses positive native moments and rooted per-hop spin trace norms, never future Wick or global operator J for b0.',
 'root':'For Clifford hopping block aGamma0+i bGamma_axis (Hermitian anticommuting unitary Gammas), trace norm<=4sqrt(a^2+b^2) (Cauchy-Schwarz over4 singular values). a=c0 sqrt(g0), b=ci sqrt(gs); exact saved rational BODY mean. Same-role CAR substitution Dcar<=2 max_native sum_incident b_j. No unrooted g0/g_s substitution.',
 'physical':'Parent owns compact-Phi finite-time left/right transfer. Physical owninitial R0 is NOT setzero by BODY parity; separate2[eta*n*jbody*n+jbody*n*eta*n+(eta*n)^2] required. Source/clock/pole/commoncone not adopted.',
 'arithmetic':'Positive prices directed mpmath rational endpoints. Grouped coefficient centres are conditional complex128, never outward coefficients; no arbitrary guard. If any complete-symbol cap fails, no R3 value is emitted. Initial trace supplier stays outward and is not rerun.',
}

def sha(raw:bytes)->str:
    return hashlib.sha256(raw).hexdigest()


class TraceEngine(symbols.Engine):
    """All129 trace closes native current pairs before initial expectation."""
    def initial_current(self,role:int,seed:int)->dict:
        return {(self.native_wind[seed],(self.native[role][seed],),()):1+0j}

    def close_current(self,terms:dict,seed:int)->dict:
        out=defaultdict(complex)
        seedwind=self.native_wind[seed]
        def put(w:int,fs:tuple,ss:tuple,c:complex):
            if c:out[w,fs,tuple(sorted(ss))]+=c
            if len(out)>self.term_cap:
                raise symbols.SymbolCostExceeded(f'final full129 trace seed{seed}: grouped count exceeds{self.term_cap}; no partial R3 accepted')
        for (w,fs,ss),coefficient in terms.items():
            for position,f in enumerate(fs):
                role,row,col,_=self.factors[f]
                for j in sorted(set(self.incident[row]+self.incident[col])):
                    dot=float(self.gram(seedwind,self.native_wind[j]))
                    if not dot:continue
                    ww=self.add_wind(w,self.native_wind[j])
                    for sign,ff in self.commute(f,j):
                        # [word,B]=-sum word-with-[B,f]; -i*-i=-1.
                        put(ww,fs[:position]+(ff,)+fs[position+1:],ss,coefficient*dot*sign)
            if ss:
                linked=sorted(set(j for z in ss for j in np.flatnonzero(self.overlaps(z))))
                for j in linked:
                    dot=float(self.gram(seedwind,self.native_wind[j]))
                    if not dot:continue
                    shifts=[int(self.overlaps(z)[j]) for z in ss]
                    ww=self.add_wind(w,self.native_wind[j])
                    for mask in range(1,1<<len(ss)):
                        multiplier=math.prod(shifts[p] for p in range(len(ss)) if mask>>p&1)
                        if not multiplier:continue
                        remain=tuple(z for p,z in enumerate(ss) if not mask>>p&1)
                        for role in (0,1):
                            put(ww,fs+(self.native[role][j],),remain,-coefficient*dot*multiplier)
        return {k:v for k,v in out.items() if v}

    def trace_expectation(self,terms:dict)->dict[str,Any]:
        pieces=[[] for _ in self.states]
        absolute=[0. for _ in self.states]
        for (w,fs,ss),coefficient in terms.items():
            for state in range(len(self.states)):
                z=coefficient*self.expectation(w,fs,ss,state)
                pieces[state].append(z);absolute[state]+=abs(z)
        centres=[complex(math.fsum(z.real for z in row),math.fsum(z.imag for z in row)) for row in pieces]
        raw=sum(self.states[k][0]*centres[k] for k in range(1,len(centres)))
        return {'background':{'real':centres[0].real,'imaginary':centres[0].imag,'absolute_term_sum':absolute[0]},'raw_source':{'real':raw.real,'imaginary':raw.imag,'absolute_term_sum':sum(self.states[k][0]*absolute[k] for k in range(1,len(absolute)))}}


def rooted_prices(data:dict[str,Any],geo:dict[str,Any])->dict[str,Any]:
    archive=json.loads((ROOT/body.ARCHIVE).read_text())
    saved=archive['input_descriptor']['exact_NEW_allq_BO_mean']
    centres=list(map(F,saved['centre_exact_rational']))
    # These are the actual immutable BODY mean, not a newly solved root.
    bs=[]
    for profile in data['profiles'][1:]:
        lookup={(p['axis'],p['length']):p for p in profile['paths']}
        forward=[]
        paths=body.joined.old.native_paths(profile,data['native'])
        for path in paths:
            p=lookup[path['axis'],path['length']]
            square=F(p['c0'])**2*centres[0]+F(p['ci'])**2*centres[1]
            forward.append(local.root_upper(square)*4)
        bs.append(forward+forward)
    incident=(geo['y'][:,None]==geo['x'][None,:])|(geo['x'][:,None]==geo['y'][None,:])
    dcar=max(2*sum((b[j] for j in np.flatnonzero(incident[r])),F(0)) for b in bs for r in range(len(b)))
    exact_geo=ar.geometry(data['native'],body.joined.old.native_paths(data['profiles'][1],data['native']))
    q2=[F(int(exact_geo['gram'][r,r]),int(exact_geo['den'])) for r in range(len(bs[0]))]
    q=[local.root_upper(x) for x in q2]
    aj=sum(((bs[0][r]+63*bs[1][r])*q[r] for r in range(len(q))),F(0))
    onsite=max(abs(F(s['onsite']))*local.root_upper(centres[0]) for s in data['profiles'][1:])
    return dict(Dcar_upper_exact=str(dcar),J_directsum_operator_upper_exact=str(aj),onsite_upper_exact=str(onsite),native_spin_trace_envelope_exact=[[str(x) for x in b] for b in bs],fixedmean_exact=[str(x) for x in centres])


def budget(prices:dict[str,Any],higher:dict[str,Any],signal:dict[str,Any])->dict[str,Any]:
    dcar=F(prices['Dcar_upper_exact']);onsite=F(prices['onsite_upper_exact']);aj=F(prices['J_directsum_operator_upper_exact'])
    force=[max(F(row['force_Lp_upper_exact'][p]) for row in higher['native_price_classes']) for p in range(9)]
    terms={(1,0):F(1)}
    for _ in range(5):
        out=defaultdict(F)
        for (m,d),c in terms.items():
            out[m,d+1]+=c*128*m
            out[m,d]+=c*(128*m*m+2*m*onsite)
            out[m+1,d]+=c*m*dcar
            for j in range(1,d+1):out[m+1,d-j]+=c*math.comb(d,j)*force[j]
        terms=dict(out)
    states={}
    for label in ('background','raw_source'):
        aa=list(map(F,higher['maximum_native_horizon_a_upper_exact'][label]))
        b5=aj*sum((c*aa[d] for (m,d),c in terms.items()),F(0))
        jj=F(signal['all129_states'][label]['JJ_trace']['upper']);b0=local.root_upper(jj)
        remainder=b5*b0*CLOCK**5/60
        f=signal['all129_states'][label]['TrF'];fmin=min(abs(F(f['lower'])),abs(F(f['upper'])))
        linear=fmin*CLOCK/local.root_upper(F(129))
        # This threshold changes the decision once COMPLETE outward TrR3 exists.
        available_trace=fmin*CLOCK-remainder*local.root_upper(F(129))
        threshold=max(F(0),available_trace)*6/CLOCK**3
        states[label]=dict(raw_mass_exact=signal['all129_states'][label]['raw_norm_squared'],initial_J_vector_norm_upper_exact=str(b0),A5_future_vector_norm_upper_exact=str(b5),HS_t5_remainder_upper_exact=str(remainder),linear_trace_HS_lower_exact=str(linear),maximum_adverse_TrR3_abs_for_positive_HS_lower_exact=str(threshold),remainder_to_linear_display=float(remainder/linear),R3_present=False,finite_time_HS_lower_admitted=False)
    return dict(states=states,positive_fifth_terms_exact={f'{m},{d}':str(c) for (m,d),c in sorted(terms.items())})


def boundary_control()->dict[str,Any]:
    """Independent two-field finite-Fock R3 with Pi/HC/Ref/path order."""
    from types import SimpleNamespace
    from scipy.linalg import expm
    import itertools
    import joined_phi_u1_initial_car_word_algebra as car
    w=np.vstack((np.eye(4),np.eye(4)))/math.sqrt(2)
    p=w@w.T;empty=(np.eye(8)-p)[:,:4]
    blocks=np.array([np.diag([.125,.25,.375,.5]),np.diag([.03,.04,.05,.06])],complex)
    onsite=np.diag([.03,-.02,.04,-.01])
    species=SimpleNamespace(p=p,w=w,empty_source=empty,hops=blocks,onsite=onsite)
    aux=SimpleNamespace(p=p,w=w,empty_source=empty,hops=np.zeros_like(blocks),onsite=np.zeros_like(onsite))
    k=np.array([[1,0,1],[0,1,-1],[-1,0,-1],[0,-1,1]])
    d=np.array([[1,-1,0],[-1,1,0]])
    geo=dict(k=k,x=np.array([0,1,1,0]),y=np.array([1,0,0,1]))
    if not np.array_equal(k@d.T,np.eye(2)[geo['x']]-np.eye(2)[geo['y']]):
        raise ArithmeticError('Boundary native wind and CAR charge shift disagree')
    engine=TraceEngine(dict(native=dict(D=d)),[species,aux],geo,dict(denominator=8,L_integer=np.array([[1,-1],[-1,1]])),200000)
    lp=np.array([[1,-1],[-1,1]])/8
    projection=np.eye(3)-d.T@lp@d
    if not np.array_equal(projection@projection,projection):
        raise ArithmeticError('Boundary physical projector is not idempotent')
    q=np.column_stack((np.array([1,1,0])/math.sqrt(2),np.array([0,0,1])))
    qhop=k@q;hopf=[]
    for j in range(4):
        a=np.zeros((8,8),complex)
        a[4*geo['x'][j]:4*geo['x'][j]+4,4*geo['y'][j]:4*geo['y'][j]+4]=blocks[j%2]
        hopf.append(car.finite_car(a))
    on=car.finite_car(np.kron(np.eye(2),onsite))
    difference=np.real(np.diag(car.finite_car(np.diag([1.]*4+[-1.]*4))))
    def slater(columns:np.ndarray)->np.ndarray:
        out=np.zeros(256,complex)
        for indices in itertools.combinations(range(8),columns.shape[1]):
            out[sum(1<<j for j in indices)]=np.linalg.det(columns[list(indices)])
        return out
    fermions=[slater(w)]+[slater(np.column_stack((w,empty[:,a]/np.linalg.norm(empty[:,a])))) for a in range(4)]
    analytic=[0j]*5
    adverse={label:[0j]*5 for label in ('wrong_Ref4','wrong_field_density_variance4')}
    variants:list[tuple[str,dict[str,Any]]]=[('wrong_Ref4',dict(Ref_hole=False)),('wrong_field_density_variance4',dict(field_scale=F(2)))]
    for role in (0,1):
        for seed in range(4):
            terms=engine.initial_current(role,seed)
            for order in range(1,4):terms=engine.derivative(terms,order)
            closed=engine.close_current(terms,seed)
            for state in range(5):
                analytic[state]+=sum(c*engine.expectation(ww,fs,ss,state,copies=0) for (ww,fs,ss),c in closed.items())
                for label,kw in variants:
                    adverse[label][state]+=sum(c*engine.expectation(ww,fs,ss,state,copies=0,**kw) for (ww,fs,ss),c in closed.items())
    raw=sum(engine.states[a+1][0]*analytic[a+1] for a in range(4))
    rows=[]
    for levels in (12,20,28):
        annih=np.diag(np.sqrt(np.arange(1,levels)),1)
        theta=math.sqrt(2)*(annih+annih.T)
        pi=(annih-annih.T)/(2j*math.sqrt(2));electric=pi@pi
        weyl=[[expm(1j*qhop[j,a]*theta) for a in range(2)] for j in range(4)]
        def axis(a:np.ndarray,op:np.ndarray,n:int)->np.ndarray:
            return np.moveaxis(np.tensordot(op,a,axes=(1,n)),0,n)
        def matter(a:np.ndarray,j:int)->np.ndarray:
            return axis(axis(a,weyl[j][0],0),weyl[j][1],1)@hopf[j].T
        for label,chosen in (('background',[0]),('raw_source',list(range(1,5)))):
            values=[]
            for state in chosen:
                initial=np.zeros((levels,levels,256),complex);initial[0,0]=fermions[state]
                coulomb=8*(difference-(1 if state else 0))**2
                def h(a:np.ndarray)->np.ndarray:
                    return 64*(axis(a,electric,0)+axis(a,electric,1))+a*coulomb+a@on.T+sum((matter(a,j) for j in range(4)),np.zeros_like(a))
                def jop(a:np.ndarray,mu:int)->np.ndarray:
                    return sum((-1j*qhop[j,mu]*matter(a,j) for j in range(4)),np.zeros_like(a))
                hp=[initial]
                for _ in range(3):hp.append(h(hp[-1]))
                total=0j
                for mu in range(2):
                    jj=jop(initial,mu);hjj=[jj]
                    for _ in range(3):hjj.append(h(hjj[-1]))
                    for a in range(4):
                        total+=(-1)**a*math.comb(3,a)*np.vdot(hp[3-a],jop(hjj[a],mu))
                        total-=(-1)**a*math.comb(3,a)*np.vdot(hjj[3-a],jop(hp[a],mu))
                values.append(total)
            value=values[0] if label=='background' else sum(engine.states[a+1][0]*values[a] for a in range(4))
            expected=analytic[0] if label=='background' else raw
            rows.append(dict(levels=levels,state=label,direct_real=value.real,direct_imaginary=value.imag,analytic_real=expected.real,analytic_imaginary=expected.imag,relative_defect=float(abs(value-expected)/max(1.,abs(expected)))))
    if max(row['relative_defect'] for row in rows[-2:])>2e-6:
        raise ArithmeticError(f'Independent finite-Fock R3 does not converge: {rows[-2:]}')
    contrasts={}
    for label,values in adverse.items():
        wrong=sum(engine.states[a+1][0]*values[a+1] for a in range(4))
        contrast=float(abs(wrong-raw))
        if contrast<=1e-6:raise ArithmeticError(f'Adverse raw R3 control is blind: {label}')
        contrasts[label]=dict(real=wrong.real,defect=contrast)
    return dict(scope='two-site4colour/two physical fields/PiHC64/Ref3 raw4/onsite/two signed paths; finite boundary not BODY129',observations=rows,wrong_i3_convention_defect=float(2*abs(analytic[0])),adverse_raw_R3=contrasts,auxiliary_control='zero auxiliary boundary is projection-blind; not credited as a63-copy discriminator',wrong_M3_relabel='No M3 input enters this RJJ contraction')


def full_trace_attempt(engine:TraceEngine,seed_cap:int,global_terms:bool)->dict[str,Any]:
    started=time.monotonic();completed=[];seeds=len(engine.native_wind)
    sums={label:[] for label in ('background','raw_source')}
    failure=None
    grouped={}
    try:
        if global_terms:
            for role in (0,1):
                for seed in range(seeds):
                    for (w,fs,ss),z in engine.initial_current(role,seed).items():
                        grouped[(seed,w,fs,ss)]=z
            # Independent seed tag is essential; joining it away would change
            # q_r.q_s, silently turning the whole trace into a scalar current.
            for order in range(1,4):
                out={}
                byseed=defaultdict(dict)
                for (seed,w,fs,ss),z in grouped.items():byseed[seed][w,fs,ss]=z
                for seed,part in byseed.items():
                    advanced=engine.derivative(part,order)
                    for (w,fs,ss),z in advanced.items():out[seed,w,fs,ss]=z
                    if len(out)>engine.term_cap:
                        raise symbols.SymbolCostExceeded(f'global adH{order}: seed-tagged complete symbol count{len(out)} exceeds{engine.term_cap}')
                grouped=out
            byseed=defaultdict(dict)
            for (seed,w,fs,ss),z in grouped.items():byseed[seed][w,fs,ss]=z
            closed_total={}
            for seed,part in byseed.items():
                for key,z in engine.close_current(part,seed).items():
                    closed_total[key]=closed_total.get(key,0j)+z
                    if len(closed_total)>engine.term_cap:
                        raise symbols.SymbolCostExceeded(f'global full129 trace: merged final count exceeds{engine.term_cap}; no partial R3 accepted')
            value=engine.trace_expectation(closed_total)
            return dict(representation='global_seed_tagged',scope_seeds_exact=2*seeds,completed_seeds=2*seeds,completed_seed_costs=[],closed_count=len(closed_total),symbol_cost_failure=None,elapsed_seconds=time.monotonic()-started,complete=True,TrR3_conditional_centres={label:row['real'] for label,row in value.items()})
        for role in (0,1):
            for seed in range(seeds):
                if seed_cap and len(completed)>=seed_cap:
                    raise symbols.SymbolCostExceeded(f'declared seed work cap{seed_cap}: whole1536 not complete; no partial coefficient accepted')
                terms=engine.initial_current(role,seed)
                counts=[]
                for order in range(1,4):
                    terms=engine.derivative(terms,order);counts.append(len(terms))
                closed=engine.close_current(terms,seed)
                value=engine.trace_expectation(closed)
                for label in sums:sums[label].append(value[label]['real'])
                completed.append(dict(role=role,seed=seed,adH_counts=counts,closed_count=len(closed)))
                print(json.dumps({'representation':'seed_streamed_full129trace','completed':len(completed),'seed':seed,'role':role,'closed':len(closed)}),flush=True)
    except symbols.SymbolCostExceeded as exc:
        failure=str(exc)
    result:dict[str,Any]=dict(representation='global_seed_tagged' if global_terms else 'seed_streamed_full129_trace',scope_seeds_exact=2*seeds,completed_seeds=len(completed),completed_seed_costs=completed,symbol_cost_failure=failure,elapsed_seconds=time.monotonic()-started,complete=failure is None and len(completed)==2*seeds)
    if result['complete']:
        result['TrR3_conditional_centres']={label:math.fsum(values) for label,values in sums.items()}
    return result


def run(output:Path,term_cap:int,seed_cap:int)->dict[str,Any]:
    started=time.monotonic()
    controls=boundary_control()
    data,sp,inputs,_=body.full_inputs();geo=current.geometry(data,sp[0]);geom=symbols.prior.prices.exact_geometry(data['native'],geo)
    captured={**body.CAPTURED,**inputs}
    dependencies=['joined_phi_u1_initial_density_higher_jets.py','joined_phi_u1_initial_car_word_algebra.py','joined_phi_u1_local_electric_moments.py','joined_phi_u1_current_signal_interval_certificate.py','joined_phi_u1_current_signal_interval_certificate_receipt.json','joined_phi_u1_native_high_order_moment_composition_actual_receipt_20261010.json',body.ARCHIVE]
    for name in dependencies:captured[name]=(ROOT/name).read_bytes()
    if sha(captured[body.ARCHIVE])!=body.ARCHIVE_SHA:raise ArithmeticError('Fixed-mean archive changed')
    signal=json.loads(captured['joined_phi_u1_current_signal_interval_certificate_receipt.json'])
    higher=json.loads(captured['joined_phi_u1_native_high_order_moment_composition_actual_receipt_20261010.json'])
    if signal['coordinates']!=129 or higher['native_signed_path_count']!=768 or F(higher['clock_exact'])!=CLOCK:raise ArithmeticError('Fullspace/native/clock prerequisite mismatch')
    prices=rooted_prices(data,geo);tails=budget(prices,higher,signal)
    graph_name='joined_phi_u1_full_current_graph_transfer_verified_source_receipt_20261010.json'
    captured[graph_name]=(ROOT/graph_name).read_bytes()
    graph=json.loads(captured[graph_name])['global_constants']
    eta=F(graph['endpoint_norm']['upper']);jbody=F(graph['body_J_operator']['upper'])
    owninitial={label:str(F(signal['all129_states'][label]['raw_norm_squared'])*(4*eta*jbody+2*eta*eta)) for label in ('background','raw_source')}
    # Retain both exact cost regimes: vector tensor expansion, then trace-first
    # seed streaming. Neither guard truncates an emitted coefficient.
    global_engine=TraceEngine(data,sp,geo,geom,term_cap)
    global_attempt=full_trace_attempt(global_engine,seed_cap,True)
    del global_engine
    engine=TraceEngine(data,sp,geo,geom,term_cap)
    streamed=full_trace_attempt(engine,seed_cap,False)
    status='COMPLETE_CONDITIONAL_INITIAL_R3_NOT_OUTWARD' if streamed['complete'] else 'RJJ_R3_COMPLETE_SCOPE_CONSTRUCTIONS_FAILED_NO_FINITE_TIME_ADMISSION'
    own=Path(__file__).read_bytes();captured[Path(__file__).name]=own
    for name,raw in captured.items():
        if (ROOT/name).is_file() and (ROOT/name).read_bytes()!=raw:raise ArithmeticError(f'Input changed during execution: {name}')
    archive=output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
        for name,raw in captured.items():bundle.writestr(name,raw)
    result=dict(status=status,governing_plan='issue-222-induced-photon-delta13-continuation-20261008',assignment_revision='d394739ef3bfd685860107612d31df7ba391b51ee7a687e9a287512206368c2a',template_revision='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',clock_exact=str(CLOCK),clock_adopted=False,coordinates=129,proof=PROOF,boundary_controls=controls,rooted_prices=prices,future_current_budget=tails,physical_owninitial_R0_HS_error_upper_exact=owninitial,attempts=[global_attempt,streamed],source_sha256=sha(own),input_sha256={name:sha(raw) for name,raw in captured.items()},input_archive_sha256=sha(archive.read_bytes()),runtime={'python':sys.version,'executable':sys.executable},elapsed_seconds=time.monotonic()-started,source_or_pole_adopted=False)
    output.write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/(PREFIX+'_receipt_20261010.json'))
    parser.add_argument('--term-cap',type=int,default=200000)
    parser.add_argument('--seed-cap',type=int,default=0)
    args=parser.parse_args()
    if args.term_cap<1 or args.seed_cap<0:parser.error('Positive term cap and nonnegative work cap required')
    with threadpool_limits(limits=1):result=run(args.output,args.term_cap,args.seed_cap)
    print(json.dumps({'status':result['status'],'attempts':result['attempts'],'elapsed_seconds':result['elapsed_seconds']}),flush=True)
