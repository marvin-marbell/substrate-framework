"""Actual12 longer-clock electric-resummed charge-transport attempt.

The receipt is an honest conditional floating contraction with rigorous algebraic
norm-price formulas, not an interval certificate or full-field evolution.
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
import zipfile
from typing import Any
import numpy as np
from scipy.linalg import expm
from threadpoolctl import threadpool_limits
import joined_phi_u1_body_word as body
import joined_phi_u1_current_moments as current
import joined_phi_u1_finite_current_kernel as prices

ROOT = Path(__file__).resolve().parent
CLOCK = F(1, 2**16)
PLAN = 'issue-222-induced-photon-delta13-continuation-20261008'
REVISION = '887df7470f23d7cfcba29846a4854f2c17841ff42d90d77c9fe2f840683bbb48'
PROOF = {
 'electric': 'Pi O=O(Pi+q), rho O=O(rho+delta); HE+HC gives O*64(2s+inc^2). Gauss q.q+deltaLdelta=inc.inc. Thus O_i(t)=O_i exp(i64t(2s_i+inc_i^2)).',
 'cross': 'Move exp(i128t s_i) across O_j using [s_i,O_j]=inc_i.inc_j O_j. Commuting underlying even CAR words acquire ratio exp(i128(t-s)inc_i.inc_j). Physical disjoint paths commute; Coulomb omission and halfHE destroy this cancellation.',
 'onsite': 'K=sum c_xdag onsite c_x preserves each total site occupation and commutes HE,HC. Include K exactly; hopping spin block becomes exp(i onsite t) A exp(-i onsite t). Density commutes K, so its no-hop contrast is EXACTLY ZERO even with initial Pauli coherence.',
 'first': 'Delta n first = i integral sum_i (1[col=x]-1[row=x]) (<O_i(s)>rawSrc - .5<O_i(s)>bg) ds. All63 auxiliary determinant factors multiply each active channel; Ref3 adds phase exp(-if0).',
 'character': 'For Slater W, z=diag(exp(if)), D=det(Wdag z W), R=z W(Wdag z W)^-1 Wdag; <dGamma(A)Gamma(z)>=D Tr(A R). Gaussian Weyl factor is exp(-q²-qPiShift²/16-i q.qPiShift/2), with qPiShift=128s q, density variance2. Background exp(-i128 sum f) and raw Refhole exp(-if0) are retained.',
 'second': 'Calculated second word at zero electric/onsite times, with exact initial Fourier Gaussian exp(-|q_i+q_j|²), and full CAR commutator matrices. This is a rigorously priced alternative to exact two-time characters: all63 factors equal1 at zero time, their density variance enters the electric-word replacement price. No future Wick is used.',
 'replacement': 'On a unit initial Slater channel, ||s_i psi|| <= sqrt(q_i²/8 + delta_i L Kdensity L delta_i + (delta_i L meanrho)²). Across a preceding hopping word s_i shifts by inc_i.inc_j. Electric product-vs-zero norm <=64t b_i b_j[2(r_i+r_j+|inc_i.inc_j|)+inc_i²+inc_j²], capped by2 b_i b_j. Rotation replacement <=2t||onsite||(b_i b_j), for each factor; thus add4t||onsite|| b_i b_j. Both orderings are paid.',
 'local_remainder': 'For word factor i, sum_j commutator prices /b_i <= D=max_i[2 sum_sameSpecies_sharedEndpoint b_j +128t sum_allSpecies |inc_i.inc_j| b_j]. The sine price permits this without a global M² bound. First density commutator Bx=sum_incidentActive b_i. Length n word obeys next commutator <=n D. Ordered simplex gives Bx t (Dt)^(n-1)/n. Tail n>=3 <=Bx t (Dt)^2/[3(1-Dt)] if Dt<1. Source mass .5 plus half-background mass .5 gives total multiplier1. Extensive unbounded Pi never receives an operator norm.',
 'quadrature': 'Composite midpoint first-word quadrature; Lipschitz derivative bound Lx=sum_incident b_i[64(2r_i+inc_i²)+2||onsite||], source plus half-background total mass1. Integral error <=Lx t²/(4 panels).',
 'scope': 'Floating Slater/root-centre contractions and floating price evaluation are conditional numerical arithmetic, not outward interval admission. Exact formulas and rational stencil envelopes separately supplied. Any nonpositive margin is failure of this construction, not global impossibility.'}


def encode(value: Any) -> Any:
    if isinstance(value,F):return str(value)
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,(np.ndarray,complex)):return body.packed(value)
    raise TypeError('Unsupported scientific capsule value '+type(value).__name__)


def density_stats(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    blocks = p.reshape(64,4,64,4).transpose(0,2,1,3)
    mean = np.trace(blocks[np.arange(64),np.arange(64)],axis1=1,axis2=2).real
    cov = np.diag(mean)-np.einsum('xyij,yxji->xy',blocks,blocks).real
    return mean, cov


def initial_channels(active: Any) -> list[dict[str, Any]]:
    channels = []
    for alpha in range(4):
        v = active.empty_source[:,alpha]
        norm = float(np.vdot(v,v).real)
        w = np.column_stack((active.w,v/math.sqrt(norm)))
        channels.append(dict(alpha=alpha,weight=norm/4,w=w,p=w@w.conj().T))
    body.require(abs(sum(c['weight'] for c in channels)-.5))
    return channels


def weighted_character(w: np.ndarray, f: np.ndarray, a: np.ndarray, x: int, y: int) -> tuple[complex, complex]:
    z = np.repeat(np.exp(1j*f),4)
    zw = z[:,None]*w
    overlap = w.conj().T@zw
    det = complex(np.linalg.det(overlap))
    # Only the literal four-colour block needed by the consumer is formed.
    r = zw[4*y:4*y+4]@np.linalg.solve(overlap,w[4*x:4*x+4].conj().T)
    return det*complex(np.trace(a@r)), det




def second_word(x: int, geo: dict[str, Any], blocks: np.ndarray, excess: np.ndarray) -> tuple[float, float]:
    total=0j; absolute=0.
    for i in np.flatnonzero((geo['x']==x)|(geo['y']==x)):
        weight=int(geo['y'][i]==x)-int(geo['x'][i]==x)
        for j in np.flatnonzero((geo['y']==geo['x'][i])|(geo['x']==geo['y'][i])):
            value=0j
            if geo['y'][j]==geo['x'][i]:
                row,col=int(geo['x'][j]),int(geo['y'][i])
                value+=np.trace(blocks[j]@blocks[i]@excess[4*col:4*col+4,4*row:4*row+4])
            if geo['y'][i]==geo['x'][j]:
                row,col=int(geo['x'][i]),int(geo['y'][j])
                value-=np.trace(blocks[i]@blocks[j]@excess[4*col:4*col+4,4*row:4*row+4])
            damping=np.exp(-float(np.sum((geo['q'][i]+geo['q'][j])**2)))
            total+=weight*damping*value;absolute+=abs(damping*value)
    body.require(abs(total.imag),1e-7)
    return float(total.real),absolute


def controls(native: dict[str, Any], geo: dict[str, Any], geom: dict[str, Any]) -> dict[str, Any]:
    k=geo['k'].astype(np.int64);d=geo['delta'].astype(np.int64);q=geo['q'];lp=geom['L_integer']/geom['denominator']
    cross=q@q.T+d@lp@d.T; exact=k@k.T
    body.require(float(np.max(abs(cross-exact))))
    pairs=np.argwhere((exact==0)&(np.abs(d@lp@d.T)>.01))
    chosen=None
    for i,j in pairs:
        if len(set((int(geo['x'][i]),int(geo['y'][i]),int(geo['x'][j]),int(geo['y'][j]))))==4:
            chosen=(int(i),int(j));break
    if chosen is None:raise ArithmeticError('No physical disjoint-path adverse witness')
    i,j=chosen; transverse=float(q[i]@q[j]);longitudinal=float(d[i]@lp@d[j])
    if abs(transverse)<.01 or abs(.5*transverse+longitudinal)<.005:raise ArithmeticError('Cross controls blind')
    phase_controls={}
    pi=np.linspace(-.2,.3,129);rho=np.arange(64)%3-1
    for label,gE,gC in [('full',64.,64.),('omittedHC',64.,0.),('halfHE',32.,64.)]:
        def energy(p: np.ndarray,r: np.ndarray) -> float:
            return float(gE*(p@p)+gC*(r@lp@r))
        ti,tj=.017,.006
        # Legal independent-species occupied-col/vacant-row CAR transitions
        # have the same even-CAR sign in both orderings, hence cancel here.
        phase_ij=ti*(energy(pi+q[j]+q[i],rho+d[j]+d[i])-energy(pi+q[j],rho+d[j]))+tj*(energy(pi+q[j],rho+d[j])-energy(pi,rho))
        phase_ji=tj*(energy(pi+q[i]+q[j],rho+d[i]+d[j])-energy(pi+q[i],rho+d[i]))+ti*(energy(pi+q[i],rho+d[i])-energy(pi,rho))
        observed=complex(np.exp(1j*(phase_ij-phase_ji)))
        predicted=complex(np.exp(2j*(ti-tj)*(gE*transverse+gC*longitudinal)))
        body.require(abs(observed-predicted),1e-8)
        phase_controls[label]=dict(commutator_modulus=abs(observed-1),Weyl_energy_word_ratio_real=observed.real,Weyl_energy_word_ratio_imaginary=observed.imag,ratio_formula_defect=abs(observed-predicted))
    if phase_controls['omittedHC']['commutator_modulus']<.01 or phase_controls['halfHE']['commutator_modulus']<.005:raise ArithmeticError('Electric word adverse controls blind')
    # Legal CAR boundary: active vacancy, occupied Ref at0, density rho=0.
    # E0 creates active and removes Ref. Omitting Ref changes rho by e0.
    e=np.eye(64)[0];omitted=64*float(e@lp@e)
    if omitted<=0:raise ArithmeticError('Missing Ref boundary control blind')
    return dict(gauss_completion_defect=float(np.max(abs(cross-exact))),disjoint_pair=[i,j],full_cross=float(cross[i,j]),omitted_longitudinal_cross=transverse,halfHE_cross=.5*transverse+longitudinal,physical_incidence_dot=int(exact[i,j]),electric_Weyl_number_transition_controls=phase_controls,neutral_E0_fullRef_energy_increment_exact='0',omittedRef_legal_vacancy_energy_increment=omitted,Ref_boundary='active vacancy and occupied Ref0; after creation/removal full total site charge unchanged; omittedRef increments active density',literal_paths=len(k),one_two_lengths=sorted(set(int(v) for v in np.sum(k*k,axis=1))))


def determinant_oracle() -> dict[str, float]:
    """Independent fixed two-particle CAR action, not a future Wick state."""
    from itertools import combinations
    rng=np.random.default_rng(22216129)
    w,_=np.linalg.qr(rng.normal(size=(4,2))+1j*rng.normal(size=(4,2)))
    pairs=list(combinations(range(4),2));psi=np.array([np.linalg.det(w[list(pair)]) for pair in pairs])
    f=np.array([.17,-.31,.08,.12]);a=np.zeros((4,4),complex);a[0,3]=.7+.2j
    lifted=np.zeros((6,6),complex)
    for col,pair in enumerate(pairs):
        for source in pair:
            if source!=3:continue
            rest=list(pair);sign=(-1)**rest.index(source);rest.remove(source)
            if 0 in rest:continue
            rest.append(0);rest.sort()
            lifted[pairs.index((rest[0],rest[1])),col]=sign*a[0,3]
    z=np.diag(np.exp(1j*f));overlap=w.conj().T@z@w
    formula=np.linalg.det(overlap)*np.trace(a@z@w@np.linalg.solve(overlap,w.conj().T))
    direct=np.vdot(psi,lifted@(np.exp(1j*np.array([sum(f[list(pair)]) for pair in pairs]))*psi))
    defect=float(abs(formula-direct));body.require(defect,1e-12)
    return dict(independent_six_state_CAR_character_defect=defect,direct_absolute=float(abs(direct)),density_phase_nonzero=float(np.max(abs(f))))


def run(output: Path, panels: int) -> dict[str, Any]:
    start=time.monotonic();t=float(CLOCK)
    data,sp,inputs,_=body.full_inputs();native=data['native'];geo=current.geometry(data,sp[0]);geom=prices.exact_geometry(native,geo)
    channels=initial_channels(sp[0]);ctrl=controls(native,geo,geom)
    ctrl['initial_density_character_oracle']=determinant_oracle()
    paths=body.joined.old.native_paths(sp[0].profile,native)
    ctrl['literal_AP_seam_negative_count']=sum(p['seam']==-1 for p in paths)
    if ctrl['literal_AP_seam_negative_count']==0:raise ArithmeticError('Missing actual AP paths')
    envelope=np.array([4*float(abs(F(p['c0']))+abs(F(p['ci']))) for p in sp[0].profile['paths']])
    lookup={(p['axis'],p['length']):envelope[i] for i,p in enumerate(sp[0].profile['paths'])}
    forward=np.array([lookup[(p['axis'],p['length'])] for p in paths]);b=np.r_[forward,forward]
    shared=(geo['y'][:,None]==geo['x'][None,:])|(geo['x'][:,None]==geo['y'][None,:])
    dots=np.abs(geo['k']@geo['k'].T)
    # Auxiliary has the same literal paths and coefficients; charges all1.
    D_naive=float(np.max(2*shared@b+2*64*(dots>0)@b))+8*float(abs(F(sp[0].profile['onsite'])))
    D=float(np.max(2*shared@b+128*t*64*dots@b))
    r=t*D
    if r>=1:raise ArithmeticError('Local resummed series has no summable tail at specified clock')
    meanaux,covaux=density_stats(sp[1].p)
    states: list[dict[str,Any]]=[dict(weight=.5,p=sp[0].p,ref=False)]+[dict(weight=c['weight'],p=c['p'],ref=True) for c in channels]
    rms=np.zeros(len(b))
    v=geo['delta']@native['Lplus']
    for state in states:
        mean,cov=density_stats(state['p']);mean=mean+63*meanaux-128
        if state['ref']:mean[0]-=1
        squared=np.sum(geo['q']**2,axis=1)/8+np.einsum('iv,vw,iw->i',v,cov+63*covaux,v)+(v@mean)**2
        rms=np.maximum(rms,np.sqrt(np.maximum(squared,0)))
    measured_rms=rms.copy()
    # Safe projector inequality Var(dGamma(V))<=Tr(V²)/2: four spin
    # components and all64 charged species give128 sum_v v². No /4 error.
    rms=np.sqrt(np.sum(geo['q']**2,axis=1)/8+128*np.sum(v*v,axis=1)+4*np.max(abs(v),axis=1)**2)
    if np.max(measured_rms-rms)>1e-8:raise ArithmeticError('Initial covariance exceeds safe full-density price')
    indices=np.flatnonzero(geo['y']==0)
    receivers=[int(geo['x'][i]) for i in indices]
    needed=np.flatnonzero(np.isin(geo['x'],receivers)|np.isin(geo['y'],receivers))
    times=(np.arange(panels)+.5)*t/panels
    first,character_info=first_words_selected(times,data,sp,geo,channels,needed)
    excess=sum((c['weight']*c['p'] for c in channels),np.zeros_like(sp[0].p))-.5*sp[0].p
    blocks=current.blocks(sp[0]);rows=[];onsite=float(np.linalg.norm(sp[0].onsite,2));kk=np.sum(geo['k']**2,axis=1)
    for i in indices:
        x=int(geo['x'][i]);incident=np.flatnonzero((geo['x']==x)|(geo['y']==x))
        weight=(geo['y'][needed]==x).astype(int)-(geo['x'][needed]==x).astype(int)
        first_value=complex(1j*t*np.mean(first@weight));body.require(abs(first_value.imag),1e-7)
        second,absolute=second_word(x,geo,blocks,excess);second_value=-t*t*second/2
        B=float(np.sum(b[incident]));tail=B*t*r*r/(3*(1-r))
        naive_r=t*D_naive
        first_error=float(np.sum(b[incident]*(64*(2*rms[incident]+kk[incident])+2*onsite)))*t*t/(4*panels)
        replacement=0.
        for h in incident:
            for j in range(len(b)):
                # Same-species CAR part plus electric cross part; different
                # species only cross part and exactly zero at time0.
                electric=64*t*(2*(rms[h]+rms[j]+dots[h,j])+kk[h]+kk[j])+4*t*onsite
                if shared[h,j]:replacement+=t*t*b[h]*b[j]*min(2,electric)
                if dots[h,j]:replacement+=t*t*128*t*64*dots[h,j]*b[h]*b[j]
        eta_numeric=1e-9
        controlled_error=tail+first_error+replacement+eta_numeric
        signal=first_value.real+second_value;margin=abs(signal)-controlled_error
        rows.append(dict(path=int(i),row=x,col=0,length=int(kk[i]),clock_exact=str(CLOCK),initial_excess_density=float(np.trace(excess[4*x:4*x+4,4*x:4*x+4]).real),first_electric_resummed_density_change=first_value.real,second_word_density_change=second_value,second_word_absolute_contraction_sum=absolute,no_hop_density_change_exact='0',source_minus_half_background_charge_signal=signal,local_remainder=tail,first_character_quadrature_price=first_error,second_word_replacement_price=replacement,floating_roundoff_reserve_conditional=eta_numeric,controlled_body_error=controlled_error,body_signed_margin=margin,body_controlled=margin>0,Bx=B,naive_support_only_ratio=naive_r,resummed_local_ratio=r))
    numerical=output.with_suffix('.npz')
    np.savez_compressed(numerical,midpoint_times=times,selected_signed_paths=needed,first_source_minus_half_background_words=first,rawRef_weights=np.array([c['weight'] for c in channels]),active_initial_projector=sp[0].p,auxiliary_initial_projector=sp[1].p,source_initial_projectors=np.array([c['p'] for c in channels]),signed_k=geo['k'],signed_q=geo['q'],signed_delta=geo['delta'],initial_excess_projector=excess,electric_phase_rms_upper_conditional=rms)
    if len(rows)!=12:raise ArithmeticError('Not literal actual12')
    # Density norm4 is fixed CAR, but Kato W need not preserve local density.
    # ||Wdag Px W-Px||trace<=8||W-I||; dGamma norm<=one-body trace norm.
    # Local Haar U preserves Px; only fundamental active Kato W is paid.
    source_name='joined_phi_u1_actual_compact_source_action_certificate_verified_receipt_20261010.json'
    transfer_name='joined_phi_u1_nonlocal_body_transport_verified_source_transfer_receipt_20261010.json'
    source_raw=(ROOT/source_name).read_bytes();transfer_raw=(ROOT/transfer_name).read_bytes();source=json.loads(source_raw);transfer=json.loads(transfer_raw)
    epsilon=F(source['action_input']['epsilon']['upper']);radius=F(source['source_weights']['prepared_radius_L2']['upper']);b1=F(transfer['fundamental_active_W1_upper_exact']);eta0=b1*radius
    eta_density=8*b1*radius
    root_half=F(1) # conservative rational upper sqrt(.5)
    fixed_error=8*CLOCK*epsilon # density norm4, state norm<=1
    physical_source_error=4*(2*root_half*eta0+eta0*eta0)
    for row in rows:
        row.update(density_endpoint_T_defect_upper_exact=str(eta_density),density_endpoint_derivation='rank4 Px; one-body trace norm(Wdag Px W-Px)<=8||W-I||; dGamma norm<=trace norm; local Haar commutes Px, Kato W does not',fixed_Tsrc_density_evolution_transfer_exact=str(fixed_error),actual_E0_vs_fixed_Tsrc_source_density_price_exact=str(physical_source_error),physical_margin_conditional=row['body_signed_margin']-float(fixed_error+eta_density+physical_source_error),physical_controlled=row['body_signed_margin']>float(fixed_error+eta_density+physical_source_error),receiver_Ex_endpoint_eta_exact=str((b1+F(row['length']))*radius),source_E0_endpoint_eta_exact=str(eta0))
    captured={**body.CAPTURED,**inputs,source_name:source_raw,transfer_name:transfer_raw,Path(__file__).name:Path(__file__).read_bytes()}
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file and Path(file).parent==ROOT and Path(file).suffix=='.py':captured.setdefault(Path(file).name,Path(file).read_bytes())
    for name,raw in captured.items():
        if (ROOT/name).read_bytes()!=raw:raise ArithmeticError('Source/input changed '+name)
    archive=output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in sorted(captured.items()):z.writestr(name,raw)
        profile_bytes=[{key:profile[key] for key in ('role','load','scale','onsite','paths','raw_paths') if key in profile} for profile in data['profiles']]
        z.writestr('actual_geometry.json',json.dumps(dict(D=native['D'],Q=native['Q'],frame=native['frame'],L_integer=geom['L_integer'],denominator=geom['denominator'],signed_k=geo['k'],row=geo['x'],col=geo['y'],profiles=profile_bytes,paths=paths),default=encode))
    result=dict(status='LONGER_CLOCK_LOCAL_ELECTRIC_TRANSPORT_ATTEMPT_CONTROLLED' if all(row['physical_controlled'] for row in rows) else 'LONGER_CLOCK_LOCAL_ELECTRIC_TRANSPORT_ATTEMPT_NOT_CONTROLLED',proof=PROOF,receiver_rows=rows,controls=ctrl,fourRef_raw_weights=[c['weight'] for c in channels],raw_mass_exact='1/2',all63_auxiliary_factors=True,character=character_info,clock_exact=str(CLOCK),local_prices=dict(naive_D=D_naive,resummed_D=D,naive_ratio=t*D_naive,resummed_ratio=r,material_repair='exact electric disjoint-path cancellation, sine-weighted auxiliary commutators and exact onsite spin rotation sharpen the coarse support-only series'),physical_scope='Fixed Tsrc and actual E0 source paid separately. Density norm4 fixed-CAR; actual local density Kato endpoint priced8b1radius, not exact. Ex endpoint distinct, not TExTdag equality. No claim of controlled offsite receiver kernel at this clock.',arithmetic_scope='initial determinant contractions and bound evaluation are conditional float64, not outward interval certificates; numeric reserve1e-9 is explicitly conditional',domains='HE self-adjoint unbounded Pi on Gaussian/CAR analytic core; exact phases extended unitarily; bounded fixed-CAR densities and hopping words; T uniform body map only',governing_plan=PLAN,governing_revision=REVISION,template_revision='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',parent_admission='PENDING',exclusions=['future interacting Wick','global M² remainder','pole/period/mass','clock adoption','quantum ground','full129 measured evolution','probability normalization'],input_sha256={n:hashlib.sha256(raw).hexdigest() for n,raw in captured.items()},input_bytes=dict(path=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,panels=panels),elapsed_seconds=time.monotonic()-start)
    result['numerical_capsule']=dict(path=numerical.name,sha256=hashlib.sha256(numerical.read_bytes()).hexdigest())
    output.write_text(json.dumps(result,indent=2,default=encode)+'\n');return result


def first_words_selected(times: np.ndarray,data: dict[str,Any],sp: list[Any],geo: dict[str,Any],channels: list[dict[str,Any]],needed: np.ndarray) -> tuple[np.ndarray,dict[str,Any]]:
    # The indexing wrapper keeps every literal path block, not a source model.
    selected={k:v[needed] if isinstance(v,np.ndarray) and len(v)==len(geo['x']) else v for k,v in geo.items()}
    allblocks=current.blocks(sp[0])
    return contract_selected(times,data,sp,selected,channels,allblocks[needed])


def contract_selected(times: np.ndarray,data: dict[str,Any],sp: list[Any],geo: dict[str,Any],channels: list[dict[str,Any]],blocks: np.ndarray) -> tuple[np.ndarray,dict[str,Any]]:
    result=np.zeros((len(times),len(geo['x'])),complex);lp=data['native']['Lplus'];minimum=1.;departure=0.
    for ti,t in enumerate(times):
        rot=expm(1j*sp[0].onsite*t)
        for i in range(len(geo['x'])):
            f=128*t*(geo['delta'][i]@lp);z=np.repeat(np.exp(1j*f),4)
            auxdet=complex(np.linalg.det(sp[1].w.conj().T@(z[:,None]*sp[1].w)));departure=max(departure,abs(auxdet**63-1))
            common=auxdet**63*np.exp(-1j*128*np.sum(f));a=rot@blocks[i]@rot.conj().T
            q2=float(geo['q'][i]@geo['q'][i]);kk=float(geo['k'][i]@geo['k'][i]);shift=128*t
            gaussian=np.exp(-q2-shift*shift*q2/16-1j*shift*q2/2+1j*64*t*kk)
            bg,det=weighted_character(sp[0].w,f,a,int(geo['x'][i]),int(geo['y'][i]));minimum=min(minimum,abs(det),abs(auxdet));src=0j
            for c in channels:
                word,det=weighted_character(c['w'],f,a,int(geo['x'][i]),int(geo['y'][i]));minimum=min(minimum,abs(det));src+=c['weight']*word*np.exp(-1j*f[0])
            result[ti,i]=gaussian*common*(src-.5*bg)
    return result,dict(minimum_character_determinant_modulus=minimum,maximum_all63_auxiliary_factor_departure_from1=departure)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--panels',type=int,default=4);args=parser.parse_args()
    if args.panels<1:raise ValueError('Positive midpoint panel count required')
    if args.output.exists() or args.output.with_suffix('.zip').exists():raise FileExistsError('Immutable output exists')
    with threadpool_limits(limits=1):receipt=run(args.output,args.panels)
    print(json.dumps(dict(status=receipt['status'],actual12=len(receipt['receiver_rows']),elapsed_seconds=receipt['elapsed_seconds'],minimum_physical_margin=min(r['physical_margin_conditional'] for r in receipt['receiver_rows']))),flush=True)
