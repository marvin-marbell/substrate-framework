"""NEW whole129 RJJ trace enclosure from exact Clifford nuclear prices.

The complete initial cubic trace is enclosed by a local current-commutator
price, not measured as a signed coefficient. Failed symbolic epochs remain.
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
import joined_phi_u1_current_retarded_higher as owner
import joined_phi_u1_current_signal_interval_certificate as ar
import joined_phi_u1_local_electric_moments as local

ROOT=Path(__file__).resolve().parent
PREFIX='joined_phi_u1_current_retarded_higher_trace_budget'
T=F(1,65536)
PROOF={
 'exact_native_nuclear_norm':'B=i a Gamma0+b Gamma_axis, a=c0 sqrt(g0), b=ci sqrt(gs), exact anticommuting Hermitian Clifford generators. BdagB=(a^2+b^2)I+2ab i Gamma0 Gamma_axis (sign irrelevant), eigenvalues(a+b)^2,(a-b)^2 each multiplicity2. Thus ||B||trace=2|a+b|+2|a-b|=4max(|a|,|b|). dGamma(B) norm<=||B||trace on arbitrary Fock sector. Actual roots, not unrooted metric coefficients or an observed eigensolver. Same onsite root retained.',
 'connected_operator_norm':'For generated same-role CAR matrix A, ||B A||trace and ||A B||trace<=||B||op||A||trace; exact Clifford ||B||op=|a|+|b|. Thus Dcar=2(max_all64 incoming||B||op+max_all64 outgoing||B||op), max over roles, covers ALL64x64 generated endpoints including diagonal/long chains. Native-endpoint-only union prices were undercoverage and are retained solely as failed historical premises.',
 'local_trace':'For each complete adH3 native right-normal monomial e^(iwTheta) F1...Fm prod(s_z), close the FINAL current BEFORE a vector triangle. All129 trace q_seed.q_j bounded by ||q_seed|| qmax. Same-role CAR substitutions cost mDcar, and polynomial finite differences cost sum_(j>=1) binom(d,j)L_j*a_(d-j). Therefore |TrR3|<=n*AJ*qmax sum c_(m,d)[mDcar*a_d+sum binom(d,j)L_j*a_(d-j)]. Every native factor/63aux/Ref/sharedTheta/Pi/HC/path channel remains; no stationary equality or photon M3 substitution.',
 'initial_moments':'The exact native moment supplier initial_a upper includes original BODY mass and Ref3 bias. Outer Cauchy n=a0 preserves original raw mass1/2; no source normalization. Positive full-H horizon moments separately price A5; initial Wick is not used in the future.',
 'kernel':'BODY antiunitary odd RJJ gives R0=R2=R4=0. For requested +i convention, TrR1=-TrF and TrR3=<[adH^3J,J]>. Complete rigorous initial TrR3 interval[-B3,+B3] is an enclosure, NOT a measured signed tensor. |TrR(t)|/sqrt129 lower >=t|minTrF|/sqrt129-B3*t^3/(6sqrt129)-A5*b0*t^5/60. b0=sqrt outward whole129 initialTrJJ. No field/current is narrowed.',
 'Jacobi':'Exact identity [adH3J,J]=adH2[adHJ,J]-[adH2J,adHJ]. Finite full-matrix control rejects omitted second term and illicit stationary zero of adH2 expectation. Actual native full129 Gram precontraction F build is attempted with declared exact work guard; cost failure is never a coefficient.',
 'physical':'Body kernel only. Parent compact-Phi left/right transfer and physical own-initial R0 price stay separate. The admitted body polynomial/HS enclosure is not measured evolution or source/clock/ground/pole/commoncone adoption.',
 'arithmetic':'All positive composition coefficients are exact fractions plus mpmath outward upper roots. Native shape/Clifford identities exact. Closure conditional on the pinned native right-normal/Holder moment theorem and IEEE directed interval suppliers; no percentage guards.'}


def sha(raw:bytes)->str:
    return hashlib.sha256(raw).hexdigest()


def hierarchy(order:int,dcar:F,onsite:F,force:list[F])->dict[tuple[int,int],F]:
    terms={(1,0):F(1)}
    for _ in range(order):
        out=defaultdict(F)
        for (m,d),c in terms.items():
            out[m,d+1]+=c*128*m;out[m,d]+=c*(128*m*m+2*m*onsite);out[m+1,d]+=c*m*dcar
            for j in range(1,d+1):out[m+1,d-j]+=c*math.comb(d,j)*force[j]
        terms=dict(out)
    return terms


def jacobi_control()->dict[str,Any]:
    h=np.array([[1.,.2,.3],[.2,-.7,.4],[.3,.4,2.]])
    currents=[np.array([[0,1j,.3j],[-1j,0,.2j],[-.3j,-.2j,0]]),np.array([[0,.4j,.7j],[-.4j,0,-.1j],[-.7j,.1j,0]])]
    psi=np.array([1.,2.,-1.])/math.sqrt(6)
    def ad(a:np.ndarray)->np.ndarray:return h@a-a@h
    direct=np.zeros_like(currents[0]);right=np.zeros_like(direct);wrong=np.zeros_like(direct);stationary=np.zeros_like(direct)
    for j in currents:
        a1=ad(j);a2=ad(a1);a3=ad(a2);f=a1@j-j@a1
        second=a2@a1-a1@a2
        direct+=a3@j-j@a3;right+=ad(ad(f))-second;wrong+=ad(ad(f));stationary-=second
    defect=float(np.max(np.abs(direct-right)))
    omission=float(abs(np.vdot(psi,(direct-wrong)@psi)));stationarity=float(abs(np.vdot(psi,(direct-stationary)@psi)))
    if defect>1e-12 or min(omission,stationarity)<1e-6:raise ArithmeticError('Jacobi adverse control fails')
    return dict(matrix_identity_defect=defect,omitted_cross_commutator_defect=omission,wrong_stationary_adH2_zero_defect=stationarity,scope='independent finite full-matrix identity control, not BODY evolution')


def jacobi_native_attempt(data:dict[str,Any],sp:list[Any],geo:dict[str,Any],geom:dict[str,Any],cap:int)->dict[str,Any]:
    engine=owner.TraceEngine(data,sp,geo,geom,cap)
    started=time.monotonic();merged={};seeds=0;failure=None;counts=[];cross_started=False
    try:
        for role in (0,1):
            for seed in range(len(engine.native_wind)):
                a1=engine.derivative(engine.initial_current(role,seed),1)
                closed=engine.close_current(a1,seed)
                for key,z in closed.items():
                    merged[key]=merged.get(key,0j)+z
                    if len(merged)>cap:raise owner.symbols.SymbolCostExceeded(f'Jacobi full129 Gram precontract F: keys{len(merged)}>cap{cap}, completed native seeds{seeds}; no R3 admitted')
                merged={k:z for k,z in merged.items() if z};seeds+=1
        counts.append(len(merged))
        for order in (2,3):
            merged=engine.derivative(merged,order);counts.append(len(merged))
        # -[A2_current,A1_current] contributes +q_r.q_s because both
        # native currents have prefactor -i. Expand both ordered products,
        # shifting only the right-normal s polynomials across the other word.
        cross_started=True
        first={};second={}
        for role in (0,1):
            for seed in range(len(engine.native_wind)):
                first[role,seed]=engine.derivative(engine.initial_current(role,seed),1)
                second[role,seed]=engine.derivative(first[role,seed],2)
        def overlap(a:int,b:int)->int:
            entries=dict(engine.winds[a]['key'])
            return sum(entries.get(e,0)*z for e,z in engine.winds[b]['key'])
        for (role,r),aa in second.items():
            for (other,s),bb in first.items():
                dot=float(engine.gram(engine.native_wind[r],engine.native_wind[s]))
                if not dot:continue
                for (wa,fa,sa),ca in aa.items():
                    for (wb,fb,sb),cb in bb.items():
                        ww=engine.add_wind(wa,wb)
                        for sign,fs,shifted,unshifted,wind in ((1,fa+fb,sa,sb,wb),(-1,fb+fa,sb,sa,wa)):
                            shifts=[overlap(z,wind) for z in shifted]
                            for mask in range(1<<len(shifted)):
                                multiplier=math.prod(shifts[p] for p in range(len(shifted)) if mask>>p&1)
                                if not multiplier:continue
                                remain=tuple(z for p,z in enumerate(shifted) if not mask>>p&1)
                                key=ww,fs,tuple(sorted(remain+unshifted))
                                merged[key]=merged.get(key,0j)+sign*dot*ca*cb*multiplier
                                if len(merged)>cap:raise owner.symbols.SymbolCostExceeded(f'Jacobi full129 cross term exceeds cap{cap}; no partial R3 accepted')
        merged={key:z for key,z in merged.items() if z}
        value=engine.trace_expectation(merged)
        return dict(representation='Jacobi_native_full129_Gram_precontract',scope_seeds=1536,completed_F_seeds=seeds,first_term_counts=counts,symbol_cost_failure=None,complete_R3=True,elapsed_seconds=time.monotonic()-started,TrR3_conditional_centres=value)
    except owner.symbols.SymbolCostExceeded as exc:
        failure=str(exc)
    # The cross term is necessary. A completed first term alone is not R3.
    return dict(representation='Jacobi_native_full129_Gram_precontract',scope_seeds=1536,completed_F_seeds=seeds,first_term_counts=counts,symbol_cost_failure=failure,complete_R3=False,elapsed_seconds=time.monotonic()-started,cross_term_not_reached_due_to_cost=not cross_started,first_term_not_a_coefficient=True)


def run(output:Path,jacobi_cap:int,reuse_jacobi:Path|None=None)->dict[str,Any]:
    started=time.monotonic();source=Path(__file__).read_bytes()
    names=['joined_phi_u1_current_retarded_higher_final_receipt_20261010.json','joined_phi_u1_current_signal_interval_certificate_receipt.json','joined_phi_u1_native_high_order_moment_composition_actual_receipt_20261010.json',owner.body.ARCHIVE]
    captured={name:(ROOT/name).read_bytes() for name in names}
    inherited=json.loads(captured[names[0]]);signal=json.loads(captured[names[1]]);higher=json.loads(captured[names[2]])
    if inherited['source_sha256']!=sha((ROOT/'joined_phi_u1_current_retarded_higher.py').read_bytes()):raise ArithmeticError('Inherited final source changed')
    saved=json.loads(captured[owner.body.ARCHIVE])['input_descriptor']['exact_NEW_allq_BO_mean'];centres=list(map(F,saved['centre_exact_rational']))
    data=owner.body.joined.old.load_mean(g_override=[centres[0],centres[1],centres[1],centres[1]])
    paths=owner.body.joined.old.native_paths(data['profiles'][1],data['native']);geo=ar.geometry(data['native'],paths)
    gamma=np.asarray(owner.body.joined.old.full.supplier.primitive.SELECTED)
    for a in range(4):
        for b in range(4):
            if not np.array_equal(gamma[a]@gamma[b]+gamma[b]@gamma[a],(2 if a==b else 0)*np.eye(4)):raise ArithmeticError('Clifford exact norm premise fails')
    clifford_controls=[]
    for axis in range(1,4):
        for aa,bb in ((.375,.125),(.25,.25)):
            block=1j*aa*gamma[0]+bb*gamma[axis]
            singular=np.linalg.svd(block,compute_uv=False)
            exact_trace=4*max(abs(aa),abs(bb));exact_op=abs(aa)+abs(bb)
            trace_defect=abs(float(sum(singular))-exact_trace);op_defect=abs(float(max(singular))-exact_op)
            wrong_normal=4*math.sqrt(abs(aa*aa-bb*bb))
            if max(trace_defect,op_defect)>1e-12 or abs(wrong_normal-exact_trace)<1e-3:raise ArithmeticError('Exact Clifford nuclear/operator controls fail')
            clifford_controls.append(dict(axis=axis,a=aa,b=bb,trace_defect=trace_defect,operator_defect=op_defect,wrong_normal_eigenvalue_price_defect=abs(wrong_normal-exact_trace)))
    bs=[];operators=[]
    for profile in data['profiles'][1:]:
        lookup={(p['axis'],p['length']):p for p in profile['paths']};forward=[];op=[]
        for path in paths:
            p=lookup[path['axis'],path['length']]
            aa=abs(F(p['c0']))*local.root_upper(centres[0]);bb=abs(F(p['ci']))*local.root_upper(centres[1])
            forward.append(4*max(aa,bb));op.append(aa+bb)
        bs.append(forward+forward);operators.append(op+op)
    dcar=max(2*(max(sum((b[j] for j in np.flatnonzero(geo['y']==x)),F(0)) for x in range(64))+max(sum((b[j] for j in np.flatnonzero(geo['x']==y)),F(0)) for y in range(64))) for b in operators)
    aj=sum(((bs[0][r]+63*bs[1][r])*local.root_upper(F(int(geo['gram'][r,r]),geo['den'])) for r in range(len(bs[0]))),F(0))
    onsite=F(inherited['rooted_prices']['onsite_upper_exact']);qmax=F(127,100)
    forces=[max(F(row['force_Lp_upper_exact'][p]) for row in higher['native_price_classes']) for p in range(9)]
    old3=hierarchy(3,F(inherited['rooted_prices']['Dcar_upper_exact']),onsite,forces)
    c3=hierarchy(3,dcar,onsite,forces);c5=hierarchy(5,dcar,onsite,forces)
    sqrt129=ar.end(ar.iv.sqrt(ar.exact(F(129))),False);states={}
    for label in ('background','raw_source'):
        initial=[max(F(row['states'][label]['initial_a_upper_exact'][p]) for row in higher['native_price_classes']) for p in range(9)]
        horizon=list(map(F,higher['maximum_native_horizon_a_upper_exact'][label]))
        b0=F(inherited['future_current_budget']['states'][label]['initial_J_vector_norm_upper_exact'])
        linear=F(inherited['future_current_budget']['states'][label]['linear_trace_HS_lower_exact'])
        oldA3=F(inherited['rooted_prices']['J_directsum_operator_upper_exact'])*sum((c*horizon[d] for (m,d),c in old3.items()),F(0))
        oldtail=oldA3*b0*T**3/3
        b3=aj*qmax*initial[0]*sum((c*(m*dcar*initial[d]+sum((F(math.comb(d,j))*forces[j]*initial[d-j] for j in range(1,d+1)),F(0))) for (m,d),c in c3.items()),F(0))
        a5=aj*sum((c*horizon[d] for (m,d),c in c5.items()),F(0));fifth=a5*b0*T**5/60
        cubic=b3*T**3/(6*sqrt129)
        lower=linear-cubic-fifth
        states[label]=dict(raw_mass_exact=signal['all129_states'][label]['raw_norm_squared'],original_rooted_third_future_HS_remainder_upper_exact=str(oldtail),original_third_to_linear_display=float(oldtail/linear),complete_initial_TrR3_enclosure_exact={'lower':str(-b3),'upper':str(b3)},initial_R3_is_signed_measured_coefficient=False,local_cubic_HS_error_upper_exact=str(cubic),full_H_fifth_HS_remainder_upper_exact=str(fifth),linear_HS_lower_exact=str(linear),BODY_whole129_HS_lower_exact=str(lower),BODY_whole129_HS_lower_display=float(lower),cubic_plus_fifth_to_linear_display=float((cubic+fifth)/linear),resolved_BODY=lower>0,physical_own_initial_R0_HS_error_upper_exact=inherited['physical_owninitial_R0_HS_error_upper_exact'][label])
    # Actual Jacobi scope probe retains failures while the independent complete
    # local trace enclosure supplies the kernel; never call F alone an R3.
    data2,sp,inputs,_=owner.body.full_inputs();ng=owner.current.geometry(data2,sp[0]);geom=owner.symbols.prior.prices.exact_geometry(data2['native'],ng)
    captured.update(inputs);captured.update(owner.body.CAPTURED)
    if reuse_jacobi is None:
        jacobi=jacobi_native_attempt(data2,sp,ng,geom,jacobi_cap)
    else:
        import ast
        import inspect
        raw=reuse_jacobi.read_bytes();cached=json.loads(raw)
        cached_archive=reuse_jacobi.with_suffix('.zip');archive_bytes=cached_archive.read_bytes()
        if sha(archive_bytes)!=cached['input_archive_sha256']:raise ArithmeticError('Reused Jacobi archive changed')
        with zipfile.ZipFile(cached_archive) as bundle:oldcode=bundle.read(Path(__file__).name).decode()
        tree=ast.parse(oldcode);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='jacobi_native_attempt')
        lines=oldcode.splitlines(keepends=True)
        oldfn=''.join(lines[node.lineno-1:node.end_lineno])
        if oldfn!=inspect.getsource(jacobi_native_attempt):raise ArithmeticError('Reused Jacobi program is not exact')
        for name,content in captured.items():
            if name in cached['input_sha256'] and sha(content)!=cached['input_sha256'][name]:raise ArithmeticError(f'Reused Jacobi input changed:{name}')
        jacobi={**cached['Jacobi_native_attempt'],'reused_from':reuse_jacobi.name,'reused_receipt_sha256':sha(raw),'reuse_scope':'same exact function and inputs, completed cost failure; not a new numerical R3'}
        captured[reuse_jacobi.name]=raw
    for name in ('joined_phi_u1_current_retarded_higher.py','joined_phi_u1_initial_density_higher_jets.py','joined_phi_u1_initial_car_word_algebra.py','joined_phi_u1_local_electric_moments.py','joined_phi_u1_current_signal_interval_certificate.py'):captured[name]=(ROOT/name).read_bytes()
    captured[Path(__file__).name]=source
    if Path(__file__).read_bytes()!=source:raise ArithmeticError('Actual source changed during run')
    for name,raw in captured.items():
        if (ROOT/name).is_file() and (ROOT/name).read_bytes()!=raw:raise ArithmeticError(f'Input changed:{name}')
    archive=output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
        for name,raw in captured.items():bundle.writestr(name,raw)
    result:dict[str,Any]
    result=dict(status='WHOLE129_BODY_CURRENT_HS_LOWER_BY_COMPLETE_LOCAL_CUBIC_ENCLOSURE' if all(s['resolved_BODY'] for s in states.values()) else 'EXACT_LOCAL_CUBIC_ENCLOSURE_NOT_USEFUL_AT_DECLARED_CLOCK',governing_plan=inherited['governing_plan'],assignment_revision=inherited['assignment_revision'],template_revision=inherited['template_revision'],clock_exact=str(T),clock_adopted=False,proof=PROOF,exact_nuclear_Dcar_upper_exact=str(dcar),exact_nuclear_J_directsum_operator_upper_exact=str(aj),states=states,third_positive_hierarchy_exact={f'{m},{d}':str(c) for (m,d),c in c3.items()},fifth_positive_hierarchy_exact={f'{m},{d}':str(c) for (m,d),c in c5.items()},Jacobi_control=jacobi_control(),Jacobi_native_attempt=jacobi,source_sha256=sha(source),input_sha256={name:sha(raw) for name,raw in captured.items()},input_archive_sha256=sha(archive.read_bytes()),elapsed_seconds=time.monotonic()-started,runtime_executable=sys.executable,physical_transfer_admitted=False,measured_evolution=False,exact_signed_R3_debt_retained=True)
    result['exact_Clifford_norm_controls']=clifford_controls
    result['Dcar_endpoint_scope']='all64x64, generated diagonal and long CAR endpoints; incoming/outgoing operator norms, nuclear norms only for initial carrier'
    result['historical_third_price_scope']='native-endpoint price has undercoverage; observed failure remains a failure even before adding missed branches; not a valid future upper for generated chains'
    result['superseded_nuclear_native_endpoint_epoch']='1839de1c / b535bd72: useful margins withdrawn because Dcar missed generated endpoints; complete histories retained'
    output.write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/(PREFIX+'_receipt_20261010.json'));parser.add_argument('--jacobi-cap',type=int,default=200000);parser.add_argument('--reuse-jacobi',type=Path);args=parser.parse_args()
    with threadpool_limits(limits=1):result=run(args.output,args.jacobi_cap,args.reuse_jacobi)
    print(json.dumps({'status':result['status'],'states':result['states'],'Jacobi_native_attempt':result['Jacobi_native_attempt']}),flush=True)
