"""Paid source-sensitive full-H paired Wilson tails, never co-local K0.

m counts literal native winds, including the receiver wind BEFORE any same-copy
CAR collapse. All s_W factors are commuting native linear-statistic sums.
Minkowski/Holder costs m^d; force finite differences cost m^l. Thus no native
pair overlap bound is improperly asserted for a composite wind.
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
from typing import Any
import zipfile
import numpy as np
from threadpoolctl import threadpool_limits
import joined_phi_u1_body_word as body
import joined_phi_u1_local_electric_moments as local
import joined_phi_u1_nonlocal_paired_higher as paired
ROOT=Path(__file__).resolve().parent
PROOF={
 'domain':'Same finite charged sectors and Schwartz Gaussian core: HS,Hbg selfadjoint, DnE maps graph domains. Cn(t)=(-i)^n<DnE UB(t)psi,US(t)E0psi>, hence |Cn(t)|<=||DnE UB(t)psi|| sqrt(.5). Unbounded Pi/HC never use operatornorm; source norm is conserved raw .5.',
 'count':'Every right-normal term has one creation, at most m-1 even native hop factors, and d commuting linear statistics s_W. All winds are sums of at most m literal native winds. Onsite contributes no new wind. Electric shift costs128m to d+1,64|W|²<=128m² to d; connected CAR costs mDcar to m+1; onsite costs2m onmax to same m. Force finite difference of l statistics costs binom(d,l)m^l L_l to m+1,d-l. The m label is not reset after same-copy chain collapse.',
 'holder':'All s_W commute and are sums of at most m native s. Spectral Holder and triangle give ||prod(d s_W)psi_t|| <= m^d max_native||s_native^d psi_t||. Positive full-H nilpotent hierarchy supplies the latter. Norms are BACKGROUND because DnE acts UBpsi; US E0psi appears only through its conserved raw norm. No future interacting Slater or co-local remainder substitution.',
 'root':'Dcar=4max_role,site(sum b of every native signed hop incident at site) bounds2union(x,y) for ALL64x64 endpoint pairs, including diagonal and CAR-generated long chains. b=4(|c0|sqrt(g0upper)+|ci|sqrt(gsupper)); connected dGamma chains are bounded by this valid deliberately loose CAR supplier. Bx incoming creation sum uses the same root-priced b. onmax=max absolute onsite times sqrt(g0upper). No observed onebody operator norm sets a price. First native-pair-only fifth price5.3245e-7 is retained failed-domain history, not admitted.',
 'parity':'Derived actual BODY antiunitary implies imaginary C even derivatives at0 vanish; Taylor imaginary through3 uses uniform fifth derivative remainder t5/120. Optional through5 uses uniform seventh t7/5040. Initial C0 remains actual0; physical endpoint/initial errors are explicitly not composed here.',
 'limits':'All inequalities are positive rational bounds. A negative margin is a failed price regime, not a physical no-go. Higher coefficients must be the actual ordered paired-source guarded producer at the same frozen law.'}

def recurrence(order:int,dcar:F,onsite:F,force:list[F])->dict[tuple[int,int],F]:
    terms={(1,0):F(1)}
    for _ in range(order):
        out:dict[tuple[int,int],F]=defaultdict(F)
        for (m,d),c in terms.items():
            out[m,d+1]+=c*128*m
            out[m,d]+=c*(128*m*m+2*m*onsite)
            out[m+1,d]+=c*m*dcar
            for l in range(1,d+1):out[m+1,d-l]+=c*math.comb(d,l)*m**l*force[l]
        terms=dict(out)
    return terms

def run(output:Path,coefficient:Path|None=None)->dict[str,Any]:
    started=time.monotonic();sys.set_int_max_str_digits(0)
    captured={**body.CAPTURED,Path(__file__).name:Path(__file__).read_bytes()}
    predecessor_name='joined_phi_u1_nonlocal_signal_interval_certificate_typed_verified_receipt_20261010.json'
    predecessor_raw=(ROOT/predecessor_name).read_bytes()
    if hashlib.sha256(predecessor_raw).hexdigest()!='6faff3a4576d6de88617ce08c21321cf8f68761ce63355f85d188e2087ff266b':
        raise ArithmeticError('Frozen typed actual12 predecessor receipt changed')
    predecessor=json.loads(predecessor_raw);captured[predecessor_name]=predecessor_raw
    predecessor_source=ROOT/'joined_phi_u1_nonlocal_signal_interval_certificate.py'
    if hashlib.sha256(predecessor_source.read_bytes()).hexdigest()!='9e0dd896cd748a1709072b08f7e16bb81f8710c60985a0daa330753843640d4a':
        raise ArithmeticError('Frozen actual12 typed receiver source changed')
    data,species,inputs,_=body.full_inputs();captured.update(inputs)
    supplier=local.MomentBounds(data,order=7);geo=supplier.geo
    requests=[supplier.request_word([r],paired.CLOCK) for r in range(len(supplier.k))]
    forces=[max(F(req['force_Lp_upper'][d]) for req in requests) for d in range(8)]
    moments=[max(F(req['states']['background']['horizon_a_upper'][d]) for req in requests) for d in range(8)]
    # Connected CAR chains may join ANY pair, including diagonal/long endpoints.
    # Four times the largest full signed incident sum bounds 2*union(x,y)
    # for every64x64 endpoint pair without assuming a native stencil closure.
    dcar=4*max(sum((b[s] for s in range(len(b)) if geo['x'][s]==x or geo['y'][s]==x),F(0)) for b in supplier.b for x in range(64))
    root=F(supplier.saved['parameters']['g0']['upper'])
    onsite=max(local.upper(local.ar.exact(abs(F(s.profile['onsite'])))*local.ar.iv.sqrt(local.ar.exact(root))) for s in species)
    budgets={}
    for order in (2,3,5,7):
        terms=recurrence(order,dcar,onsite,forces)
        norm=sum((c*m**d*moments[d] for (m,d),c in terms.items()),F(0))
        derivative=norm*local.root_upper(F(1,2))
        remainder=derivative*paired.CLOCK**order/math.factorial(order)
        budgets[str(order)]={'DnE_background_norm_upper_exact':str(norm),'Cn_uniform_upper_exact':str(derivative),'Taylor_remainder_upper_exact':str(remainder),'remainder_display':float(remainder),'terms_exact':{f'{m},{d}':str(c) for (m,d),c in sorted(terms.items())}}
    rows=[]
    for rr in np.flatnonzero(geo['y']==0):
        r=int(rr);x=int(geo['x'][r]);bx=sum((supplier.b[0][s] for s in range(len(supplier.k)) if geo['y'][s]==x),F(0))
        rows.append({'path':r,'ROW':x,'COLUMN':0,'length':int(supplier.k[r]@supplier.k[r]),'Bx_incoming_upper_exact':str(bx),'initial_source_norm_squared_exact':'1/2'})
    execution_binding:dict[str,Any]|None=None
    if coefficient:
        raw=coefficient.read_bytes();captured[coefficient.name]=raw;coeff=json.loads(raw)
        if coeff['clock_exact']!=str(paired.CLOCK) or coeff['raw_source_mass_exact']!='1/2' or len(coeff['receiver_rows'])!=12:raise ArithmeticError('Wrong paired coefficient epoch')
        # Preserve actual execution99d7 epoch. Later annotation/control-label
        # corrections do not retroactively turn these numbers into a new run.
        if hashlib.sha256(raw).hexdigest()!='ae8f44fe6623c0c4825c7bc8577a30c64d3bf2c855d711bed040935755041449' or coeff['source_sha256']!='99d7c4f660225b4ddcadb42d23ca7b9d3a3dc671f8c31862a61d541a0812507e':
            raise ArithmeticError('Unreviewed actual paired coefficient execution epoch')
        archive_name=coeff['immutable_inputs']['path']
        archive_raw=(ROOT/archive_name).read_bytes();captured[archive_name]=archive_raw
        if hashlib.sha256(archive_raw).hexdigest()!=coeff['immutable_inputs']['sha256']:
            raise ArithmeticError('Actual paired coefficient input capsule changed')
        with zipfile.ZipFile(ROOT/archive_name) as execution:
            executed=execution.read('joined_phi_u1_nonlocal_paired_higher.py')
        if hashlib.sha256(executed).hexdigest()!=coeff['source_sha256']:
            raise ArithmeticError('Actual coefficient source bytes disagree with execution receipt')
        capsule_name='joined_phi_u1_nonlocal_paired_higher_cubic_first_natural_exit_20261010.json'
        capsule_raw=(ROOT/capsule_name).read_bytes();captured[capsule_name]=capsule_raw
        capsule=json.loads(capsule_raw)
        if capsule['returncode']!=0 or not capsule['natural_numeric_exit'] or not capsule['source_unchanged'] or capsule['receipt']['sha256']!=hashlib.sha256(raw).hexdigest() or capsule['source_sha256']!=coeff['source_sha256']:
            raise ArithmeticError('Actual coefficient natural numeric exit binding invalid')
        execution_binding={
            'source_sha256':coeff['source_sha256'],'receipt_sha256':hashlib.sha256(raw).hexdigest(),
            'capsule':capsule_name,'natural_numeric_exit':True,'elapsed_seconds':capsule['elapsed_seconds'],
            'current_annotation_edition_is_not_a_new_coefficient_execution':True,
            'historical_metadata_correction':'Execution99d7 initial_m1_blind=true on omit_Ref was unsupported: that control removes ONLY source Refhole. It is not predecessor simultaneous staticRef omission. m3 intervals/discrimination remain actual; no claim about that control m1 is used.'}
        for row in rows:
            source=next(s for s in coeff['receiver_rows'] if s['path']==row['path'])
            m1=source['m_derivatives']['1']['real'];m3=source['m_derivatives']['3']['real']
            old=next(s for s in predecessor['receiver_rows'] if s['path']==row['path'])
            if not (F(m1['lower'])<=F(old['m1']['real']['lower'])<=F(old['m1']['real']['upper'])<=F(m1['upper'])):
                raise ArithmeticError('Actual ROW first coefficient fails frozen typed antecedent containment')
            for degree in ('1','3'):
                imag=source['m_derivatives'][degree]['imaginary']
                if not F(imag['lower'])<=0<=F(imag['upper']):
                    raise ArithmeticError('Actual odd coefficient contradicts proved BODY parity')
            row['typed_predecessor_m1_enclosed']=True
            row['initial_Pi_HC_scalar_projection_exact']=dict(Pi=old['Pi_exact'],HC=old['HC_exact'])
            t=paired.CLOCK;tail=F(budgets['5']['Taylor_remainder_upper_exact'])
            lo=-t*F(m1['upper'])+t**3*F(m3['lower'])/6-tail
            hi=-t*F(m1['lower'])+t**3*F(m3['upper'])/6+tail
            lower=min(abs(lo),abs(hi)) if lo*hi>0 else F(0)
            realcap=F(budgets['2']['Taylor_remainder_upper_exact'])
            row['real_BODY_interval']={'lower':str(-realcap),'upper':str(realcap)}
            row['real_BODY_quadratic_upper_exact']=str(realcap)
            row.update({'C0_BODY_exact':'0','imaginary_BODY_interval':{'lower':str(lo),'upper':str(hi)},'signed_BODY_absolute_lower_exact':str(lower),'signed_BODY':lower>0,'imaginary_BODY_display':[float(lo),float(hi)],'physical_endpoint_and_both_time_initial_errors_composed':False})
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file and Path(file).parent==ROOT and Path(file).suffix=='.py':captured.setdefault(Path(file).name,Path(file).read_bytes())
    if any((ROOT/name).read_bytes()!=raw for name,raw in captured.items()):raise ArithmeticError('Bound source/input changed during composition')
    archive=output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,raw in sorted(captured.items()):z.writestr(name,raw)
    result={'status':'PAIRED_SOURCE_LOCAL_FUTURE_TAIL_COMPOSED' if not coefficient else ('ACTUAL12_LONGER_CLOCK_BODY_SIGNED' if all(r['signed_BODY'] for r in rows) else 'ACTUAL12_LONGER_CLOCK_PAIRED_PRICE_REGIME_FAILED'),'proof':PROOF,'clock_exact':str(paired.CLOCK),'clock_adopted':False,'budgets':budgets,'Dcar_ROOT_upper_exact':str(dcar),'onsite_ROOT_upper_exact':str(onsite),'maximum_native_background_moments_exact':list(map(str,moments)),'maximum_native_force_Lp_exact':list(map(str,forces)),'receiver_rows':rows,'source_sha256':hashlib.sha256(captured[Path(__file__).name]).hexdigest(),'input_sha256':{name:hashlib.sha256(raw).hexdigest() for name,raw in captured.items()},'immutable_inputs':{'path':archive.name,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},'physical_composition_claimed':False,'governing_plan':paired.first.PLAN,'captured_revision':paired.REVISION,'elapsed_seconds':time.monotonic()-started,'runtime':{'executable':sys.executable,'python':sys.version}}
    result['actual_source_initial']={
        'raw_mass_exact':predecessor['source_total_norm_squared_exact'],
        'four_Ref_channel_norm_squared_enclosures':predecessor['four_source_norm_squared_enclosures'],
        'origin':'Frozen typed actual12 receiver root-box intervals contain immutable saved-centre BODY source',
        'Ref_background':'4','Ref_source':'3','density_species':'active1+aux63',
        'sourceE0':'(1/2) sum_alpha cdag_0alpha R0alpha; raw and unnormalized',
        'both_physical_endpoints_and_both_time_initial_errors':'Parent-owned; not omitted or assumed zero',
        'initial_scalar_projection_blind_controls':'ROW/COLUMN, initialPi/HC and simultaneous staticRef omission are blind in predecessor; not counted as higher discriminants'}
    if execution_binding is not None:
        result['coefficient_execution']=execution_binding
    output.write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--coefficient',type=Path);a=p.parse_args()
    with threadpool_limits(limits=1):r=run(a.output,a.coefficient)
    print(json.dumps({'status':r['status'],'remainders':{k:v['remainder_display'] for k,v in r['budgets'].items()},'elapsed_seconds':r['elapsed_seconds']}),flush=True)
