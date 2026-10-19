"""Joined nonstationary photon coordinate/Pi response, fixed currentN4/compactPhi.

CLI is a scientific consumer, never an all129 evolution measurement. Proof is
embedded in the immutable receipt; identities, outward bounds, and finite
observations have distinct labels. No pole or stationary vacuum is inferred.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile
from typing import Any
import numpy as np
from threadpoolctl import threadpool_limits
import joined_phi_u1_body_word as body
import joined_phi_u1_current_moments as moments
import joined_phi_u1_current_signal_interval_certificate as arithmetic
import joined_phi_u1_photon_causal_response_controls as finite

ROOT=Path(__file__).resolve().parent
PREFIX='joined_phi_u1_photon_causal_response'
CLOCK=F(1,2**39)
PIN={
 'joined_phi_u1_actual_compact_source_action_certificate_verified_receipt_20261010.json':'2a1bbb46852707c82e477f28b5497e303b5ca66453db76d0655379f35cfb9a97',
 'induced_phi_compact_source_global_domain_control_corrected_receipt_20261010.json':'f6a6b6d79496e4e24d0a5df4762a9098b0db24debe16c2f82c32572696592d2f',
 'joined_phi_u1_full_current_graph_transfer_verified_source_receipt_20261010.json':'2cc91cbe6b8b64f6d270ff04a9b946b22bd3a9a52ff9627e4bf15325f0c0adf4',
 'joined_phi_u1_current_signal_interval_certificate.py':'386c684f42aa3715dbf370d075cb1b4f5a1b0a645c2784fd9ace2c97ee491810',
}
iv=arithmetic.iv
ex=arithmetic.exact
sys.set_int_max_str_digits(0)

PROOF={
 'model':'Fixed real R129, finite charged sectors active128/129 +63aux128, Ref4/3; neutral charge0 separates exactly in BODY. H_b=64Pi^2+64rho_all Lplus rho+M(Theta). Initial Gaussian DENSITY variance2, fixed exact-root meanPhi seas. Purification is direct sum of four raw Ref3 channels, mass1/2; background mass1. No evolved Slater/Wick premise.',
 'signs':'On Schwartz core [Theta_i,Pi_j]=i delta_ij, J=-d_i M, C=d_i d_j M. i[H,Theta]=128Pi and i[H,Pi]=J. For impulse e^(i eta Theta_j), derivative <e^(-i eta Theta_j)Theta_i(t)e^(i eta Theta_j)>=i<[Theta_i(t),Theta_j]>. Thus Dprime(0)=128 mass I, not negative. Mmix=i<[J(t),Theta]>, Mmix(0)=0 and Mmixprime(0)=-128<C>. D=128t mass I+128 integral_0^t(t-s)Mmix(s)ds. Cubic coefficient=-8192<C>/3. Pi response=Dprime/128=mass I+integral_0^t Mmix(s)ds, quadratic correction=-64<C>t^2.',
 'sharpened_mixed_identity':'Backward Theta(-t)=Theta-128tPi+128 integral_0^t(t-u)J(-u)du. Since i[J,Pi]=C, EXACT Mmix(t)=-128t<C(t)>+128 integral_0^t(t-u)i<[J(t),J(t-u)]>du. This is a new unequal-time identity, not reuse of current-current F remainder. It requires only the original state and finite-time canonical dynamics, not stationarity.',
 'source_weighted_mixed_remainder':'Let n=||psi||, a=sum_oriented_paths load*4b*|q|, c=sum load*4b*|q|^2. CAR path triangle gives ||J U_s psi||_HS<=a n; this is a finite-sector state-weighted coefficient price, NEVER a bound for physical fullPhi J, Theta, or Pi. Pi(s)=Pi+integral J implies P_s<=P0+t*a*n. For a hop A_r, [HC,A_r]=A_r V_r, V_r=128delta_r Lrho+64d_r; fixed charged sector |rho_x|<=129 (background128), including the Ref hole. v_r=128 sum_x|delta_r L_x|*129+64d_r prices V_r. Local [M,A_r] is a one-body commutator with nuclear price2hmax*4b, not extensive M squared. Therefore K_C=sum load*|q|^2*4b*(64|q|^2*n+128|q|P_s+v_r*n+2hmax*n) bounds ||ad_H C U_s psi||_HS. |<C(s)>-<C(0)>|_HS<=n*K_C*s. Cauchy-Schwarz bounds the unequal-time current commutator by2a^2*n^2. Hence ||Mmix(t)+128t<C0>||_HS<=128(n*K_C+a^2*n^2)t^2. Independent D error <=(128^2/12)(n*K_C+a^2*n^2)t^4 and Pi error <=(128/3)(n*K_C+a^2*n^2)t^3. Bounds hold uniformly0<=s<=unchanged2^-39. All terms are directed interval evaluations of rational inputs/exact Q-Gram. No current-current remainder imported.',
 'cheap_route_comparison':'The generic double-ad_H current Cauchy-Schwarz mixed route costs (128/12)*Bstar*sqrt(258mass)*t^4. Existing finite-current Bstar prices are not certified here and used only as a labelled comparison observation. Actual execution found this cheap estimate already nonvacuous at about14.5percent of contact, correcting a pre-execution vacuity expectation. The independently certified backwards-coordinate representation materially sharpens it about80x to .1803percent, without changing state, parameters, source, or clock.',
 'domains':'Body HC finite constant CAR matrix and M smooth bounded Fourier multipliers => selfadjoint on D(sumPi^2); Gaussian, Theta_jGaussian, Pi_jGaussian and finite Fourier insertions are Schwartz. Commutator identities first hold on that core. Cutoff-coordinate approximations and conserved/finite-time graph estimates extend their weak expectations; no bounded Theta/Pi assertion. Physical corrected global Friedrichs D(H_F^2) source-domain receipt supplies Tpsi and TTheta_jpsi because W and all compactPhi scalar factors are Theta-independent. It supplies local current/contact insertion classes; no bare electromagnetic B^2 term and no global R_B bound.',
 'physical_transfer':'Actual isometry T is Theta-independent with H_F T=T(H_b+Ephi)+R, ||R|| from body<=epsilon. For Delta_s=U_F(s)T-exp(-iEphi s)TU_b(s), ||Delta_s v||<=s epsilon||v|| and ||H_F Delta_s v||<=|Ephi|s epsilon||v||+s epsilon||H_b v||+2epsilon||v||. Use ACTUAL global graph coercivity (Lambda/128)||X^2 xi||^2<=||xi||||H_Fxi||+C||xi||^2 AFTER propagation, C=434313856Lambda. Thus current-on-error HS<=kappa[d^2(dh+Cd^2)/(Lambda/128)]^(1/4). For all initial Theta insertions bTheta=sqrt(258mass), ||H_bTheta psi||_HS <=sqrt(2*64^2*129*131*133/64)*n+(HCsector+mbody)*bTheta. Compare z=<UF Tpsi,J_F UF TTheta psi> to body z using current-on-Delta_psi, body vector price jbody*n, and ACTUAL endpoint ||(J_FT-TJ_b)v||_HS<=eta||v||. Error Mmix<=2[(currentDelta+eta*n)(bTheta+dTheta)+jbody*n*dTheta]. This explicitly pays new Theta insertions; no current-current transfer copied. Uniform mixed transfer integrated gives errorD<=64t^2 errorM and errorPi<=t errorM. Physical Theta/Pi intertwiners and their canonical Volterra law are exact. Endpoint eta includes full5760 Phi covariance, orderedSU4 and real W_Kato dependence, not T J Tdag replacement.',
 'mapping':'Pphys=I-D^T Lplus D is EXACT rank129:126 coexact plus3torons, no longitudinal kernel. Q is the owner real orthonormal basis with QQ^T=Pphys. Physical192 link response=Pphys*(128t mass)-8192t^3*(Pphys Cedge Pphys)/3 with same HS remainder; coordinate matrix=Q^T response Q. Uniform toron vectors have 1/8 on the64 links of each axis and zero elsewhere, Dtau=0. a=1/64 and literal exp(iTheta)=exp(i a Aphys) imply Theta=a Aphys, D_A=D_Theta/a^2, J_A=a J_Theta, C_A=a^2 C_Theta. Gauge parameters longitudinalD^Tlambda are removed, not confused with physical torons.',
 'signed_contact_certificate':'In physical192 exact projector space, choose the proof-only scalar gamma=200mass (not a model parameter). Let r=||C-gamma Pphys||HS, computed outward from the whole entrywise ball and exact rational projector. C restricted to physical129 is >=(gamma-r)I. The CLI requires gamma-r>0 and pays D remainder/physical transfer against the minimum directional cubic signal too. Hence signed cubic corrections are negative in EVERY physical unit direction at the specified nonstationary clock; this proves no global stationarity or pole. Numerical eigenvalues remain observations only.',
 'limits':'Exact identities and outward inequalities are conditional on pinned exact mean/Clifford/branch/source-domain/action certificates. Finite CAR/Hermite controls are numerical observations, not convergence or all129 evolution. All129 capsule is signed coupled coordinate/Pi/mixed polynomial plus paid error ball, NOT measured evolution. Negative curvature if present remains at initial NONSTATIONARY scope; no instability-to-global-vacuum extrapolation. Stationarity, interacting ground, poles, massless particle, dispersion/commoncone, Compton, originalSOURCE and CURRENT427 adoption remain OPEN.'}


def bounds(x: Any) -> dict[str,str]:
    lo,hi=arithmetic.end(x,False),arithmetic.end(x,True)
    return dict(lower=str(lo),upper=str(hi),lower_decimal=str(float(lo)) if abs(lo)<F(10)**300 else 'large exact rational',upper_decimal=str(float(hi)) if abs(hi)<F(10)**300 else 'large exact rational')


def contact(mean: Any,geo: dict[str,Any]) -> Any:
    k=arithmetic.Ball(geo['k']);den=geo['den']
    chars=arithmetic.characters(np.diag(geo['gram']),den)
    edge=-(k.transpose(1,0)@((chars*mean).reshape(-1,1)*k))
    d=geo['native']['D'].astype(np.int64)
    projector=arithmetic.rational_array(den*np.eye(192,dtype=np.int64)-d.T@geo['lint']@d,den)
    return projector@edge@projector


def norm_lower(ball: Any) -> Any:
    squares=sum((F.from_float(float(z.real))**2+F.from_float(float(z.imag))**2 for z in ball.v.flat),F(0))
    return iv.sqrt(ex(squares))-ex(F.from_float(ball.r))*192


def signed_contact_lower(c: Any,geo: dict[str,Any],mass: F) -> Any:
    d=geo['native']['D'].astype(np.int64);den=geo['den']
    projector=arithmetic.rational_array(den*np.eye(192,dtype=np.int64)-d.T@geo['lint']@d,den)
    residual=c-projector*(200*mass)
    squares=sum((F.from_float(float(z.real))**2+F.from_float(float(z.imag))**2 for z in residual.v.flat),F(0))
    return ex(200*mass)-iv.sqrt(ex(squares))-192*ex(F.from_float(residual.r))


def mixed_prices(profiles: list[Any],geo: dict[str,Any],mass: F,source: bool) -> dict[str,Any]:
    n=iv.sqrt(ex(mass));a=ex(0);cop=ex(0);rows=[]
    for profile,load in zip(profiles,(1,63)):
        lookup={(s['axis'],s['length']):abs(F(s['c0']))+abs(F(s['ci'])) for s in profile['paths']}
        hmax=abs(F(profile['onsite']))+2*sum(lookup.values())
        paths=body.joined.old.native_paths(profile,geo['native'])
        for r,path in enumerate(paths+paths):
            idx=r%len(paths);b=4*lookup[path['axis'],path['length']]
            q2=ex(F(int(geo['gram'][r,r]),geo['den']));q=iv.sqrt(q2)
            dd=geo['delta'][r]@geo['lint'];d=F(int(geo['density'][r,r]),geo['den'])
            vp=128*(128+int(source))*sum((abs(F(int(x),geo['den'])) for x in dd),F(0))+64*d
            a+=load*ex(b)*q;cop+=load*ex(b)*q2;rows.append((load,b,q2,q,vp,hmax))
    momentum=iv.sqrt(ex(F(129,8)*mass))+ex(CLOCK)*a*n
    kc=ex(0)
    for load,b,q2,q,vp,hmax in rows:
        kc+=load*q2*ex(b)*(64*q2*n+128*q*momentum+ex(vp)*n+2*ex(hmax)*n)
    coeff=n*kc+a*a*ex(mass)
    return dict(path_current_price=bounds(a),path_contact_price=bounds(cop),momentum_at_clock=bounds(momentum),contact_commutator_state_price=bounds(kc),mixed_quadratic_coefficient=bounds(128*coeff),D_remainder=bounds(ex(F(128**2,12)*CLOCK**4)*coeff),Pi_remainder=bounds(ex(F(128,3)*CLOCK**3)*coeff),M_remainder=bounds(128*ex(CLOCK**2)*coeff))


def physical_prices(graph: dict[str,Any],mass: F,profiles: list[Any]) -> dict[str,Any]:
    get=lambda key:ex(F(graph['global_constants'][key]['upper']))
    eps=ex(F(graph['actual_epsilon']['upper']));t=ex(CLOCK);n=iv.sqrt(ex(mass));bt=iv.sqrt(ex(258*mass))
    mb=(129)* (abs(F(profiles[0]['onsite']))+2*sum((abs(F(x['c0']))+abs(F(x['ci'])) for x in profiles[0]['paths']),F(0)))+8064*(abs(F(profiles[1]['onsite']))+2*sum((abs(F(x['c0']))+abs(F(x['ci'])) for x in profiles[1]['paths']),F(0)))
    cop=F(64)*F(1,2)*64*129**2
    hb=(8*iv.sqrt(ex(129*131))+ex(cop+mb))*n
    hbt=iv.sqrt(ex(F(2*64**2*129*131*133,64)))*n+ex(cop+mb)*bt
    def delta(norm:Any,hbn:Any)->tuple[Any,Any,Any]:
        d=t*eps*norm;h=get('Erefabs')*d+t*eps*hbn+2*eps*norm
        w=get('current_kappa')*(d*d*(d*h+get('additive_C')*d*d)/get('graph_quartic'))**ex(F(1,4))
        return d,h,w
    d,h,w=delta(n,hb);dt,ht,wt=delta(bt,hbt)
    error=2*((w+get('endpoint_norm')*n)*(bt+dt)+get('body_J_operator')*n*dt)
    return dict(Theta_inserted_norm=bounds(bt),Hbpsi=bounds(hb),HbTheta_insertions=bounds(hbt),Delta=bounds(d),HDelta=bounds(h),current_on_Delta=bounds(w),DeltaTheta=bounds(dt),HDeltaTheta=bounds(ht),current_on_DeltaTheta=bounds(wt),mixed_transfer=bounds(error),D_transfer=bounds(64*t*t*error),Pi_transfer=bounds(t*error))


def run(output: Path) -> dict[str,Any]:
    started=time.monotonic();iv.dps=75;captured={}
    for module in tuple(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file and Path(file).parent==ROOT and Path(file).suffix=='.py':
            captured[Path(file).name]=Path(file).read_bytes()
    def take(name:str,pin:str|None=None)->bytes:
        raw=(ROOT/name).read_bytes()
        if pin and hashlib.sha256(raw).hexdigest()!=pin:raise ArithmeticError('Pinned bytes changed '+name)
        captured[name]=raw;return raw
    # The accepted NPZ hash is resolved from its actual immutable owner receipt,
    # rather than transcribing a shortened assignment hash.
    mr=json.loads(take('joined_phi_u1_current_moments_receipt.json'))
    npz=take('joined_phi_u1_current_moments_receipt.npz')
    if hashlib.sha256(npz).hexdigest()!=mr['matrix_capsule']['sha256']:raise ArithmeticError('Current coefficient capsule mismatch')
    for name,pin in PIN.items():
        if name.endswith('.npz'):continue
        take(name,pin)
    data,species,inputs,_=body.full_inputs();captured.update(inputs);captured.update(body.CAPTURED)
    if (moments.G,moments.S,body.S)!=(64.,2.,2.):raise ArithmeticError('Current/body electric or Gaussian ABI changed')
    saved=json.loads(captured[body.ARCHIVE])['input_descriptor']['exact_NEW_allq_BO_mean']
    profiles=data['profiles'][1:];native=data['native'];paths=body.joined.old.native_paths(profiles[0],native)
    geo=arithmetic.geometry(native,paths);geo['native']=native
    exacts=[arithmetic.exact_species(p,native,saved,paths) for p in profiles]
    bs=[v[0] for v in exacts];p=exacts[0][2];aux=exacts[1][2]
    empty=arithmetic.Ball(np.eye(256))-p
    active_raw=p*F(1,2)+F(1,4)*(empty[:,:4]@empty[:,:4].conj().transpose(1,0))
    # Linear CAR first moment: disconnected densities remain in HC and its
    # source/graph prices; HC has identically zero Theta derivative.
    def mean_for(pa:Any,pb:Any)->Any:
        def one(b:Any,pp:Any)->Any:
            pb0=pp.reshape(64,4,64,4).transpose(0,2,1,3)
            return arithmetic.contract('rij,rji->r',b,pb0[geo['y'],geo['x']])
        return one(bs[0],pa)+63*one(bs[1],pb)
    means={'background':mean_for(p,aux),'raw_source':mean_for(active_raw,aux*F(1,2))}
    graph=json.loads(captured['joined_phi_u1_full_current_graph_transfer_verified_source_receipt_20261010.json'])
    old=np.load(ROOT/'joined_phi_u1_current_moments_receipt.npz');arrays={};states={};q=native['Q']
    for label,mass,source in (('background',F(1),False),('raw_source',F(1,2),True)):
        c=contact(means[label],geo);lower=norm_lower(c)
        signed_lower=signed_contact_lower(c,geo,mass)
        if arithmetic.end(lower,False)<=0:raise ArithmeticError('Contact enclosure inconclusive')
        pricing=mixed_prices(profiles,geo,mass,source);physical=physical_prices(graph,mass,profiles)
        signal=ex(F(8192,3)*CLOCK**3)*lower;pi_signal=ex(64*CLOCK**2)*lower
        rem=F(pricing['D_remainder']['upper']);transfer=F(physical['D_transfer']['upper']);margin=arithmetic.end(signal,False)-rem-transfer
        if margin<=0:raise ArithmeticError('Signed all129 contact unresolved '+label)
        cc=q.T@c.v@q; cc=(cc+cc.conj().T)/2
        exact_defect=float(np.linalg.norm(cc-old[label+'_contact']))
        if exact_defect>192*c.r+1e-6:raise ArithmeticError('Native cubic coefficient does not match actual owner')
        arrays[label+'_contact']=cc
        arrays[label+'_D_cubic_coefficient']=-8192/3*cc
        arrays[label+'_D_correction_at_clock']=-8192/3*float(CLOCK)**3*cc
        arrays[label+'_D_at_clock']=128*float(CLOCK)*float(mass)*np.eye(129)+arrays[label+'_D_correction_at_clock']
        arrays[label+'_Pi_correction_at_clock']=-64*float(CLOCK)**2*cc
        arrays[label+'_Pi_at_clock']=float(mass)*np.eye(129)+arrays[label+'_Pi_correction_at_clock']
        arrays[label+'_M_at_clock']=-128*float(CLOCK)*cc
        arrays[label+'_physical192_D_correction']=-8192/3*float(CLOCK)**3*c.v
        torons=[]
        for axis in range(3):
            tau=np.zeros(192)
            # Native link ordering site-major,axis-minor, as native D verifies.
            tau[axis::3]=1/8
            body.require(float(np.linalg.norm(native['D']@tau)))
            val=arithmetic.contract('i,ij,j->',arithmetic.Ball(tau),c,arithmetic.Ball(tau)).real_interval()
            torons.append(bounds(val))
        cheap=float(CLOCK)**4*128/12*float(np.sqrt(258*float(mass)))*json.loads(take('joined_phi_u1_finite_current_kernel_receipt.json'))['all129_states'][label]['Sobolev']['Bstar']
        states[label]=dict(raw_norm_squared=str(mass),contact_HS_lower=bounds(lower),projected192_contact_entry_ball_radius=str(F.from_float(c.r)),contact_trace=bounds(arithmetic.contract('ii->',c).real_interval()),toron_contacts=torons,contact_eigenvalue_observations=np.linalg.eigvalsh(cc).tolist(),owner_coefficient_HS_difference=exact_defect,mixed_body_prices=pricing,physical_transfer=physical,cubic_signal_HS_lower=bounds(signal),Pi_signal_HS_lower=bounds(pi_signal),signed_contact_post_body_and_Phi_HS_margin=str(margin),cheap_mixed_route_observed_remainder=cheap,cheap_to_signal_observed=cheap/float(arithmetic.end(signal,False)),material_remainder_improvement_observed=cheap/float(rem),D_remainder_to_signal=float(rem/arithmetic.end(signal,False)),Pi_post_bound_HS_margin=str(arithmetic.end(pi_signal,False)-F(pricing['Pi_remainder']['upper'])-F(physical['Pi_transfer']['upper'])))
        direction_signal=ex(F(8192,3)*CLOCK**3)*signed_lower
        direction_margin=arithmetic.end(direction_signal,False)-rem-transfer
        if direction_margin<=0:raise ArithmeticError('All-direction signed correction unresolved '+label)
        states[label].update(contact_minimum_directional_outward_lower=bounds(signed_lower),minimum_directional_D_cubic_signal=bounds(direction_signal),minimum_directional_D_signed_margin=str(direction_margin),source_fourRef_weights=old['source_channel_weights'].tolist() if source else None)
    prior=json.loads(take(PREFIX+'_actual_20261010_receipt.json','35ab545dfbb1ea612473c7f8972a1da6ec1efbafef57b13b1f53b5a224b08cdf'))
    failure=json.loads(take(PREFIX+'_actual_20261010_runner_failure.json'))
    if failure['scientific_consumer_exit_code']!=0:raise ArithmeticError('Prior scientific controls incomplete')
    control_file=getattr(finite,'__file__',None)
    if not isinstance(control_file,str):raise ArithmeticError('Finite control source path unavailable')
    control_path=Path(control_file)
    if prior['source_input_sha256'][control_path.name]!=hashlib.sha256(control_path.read_bytes()).hexdigest():
        raise ArithmeticError('Observed finite control source changed')
    control=prior['finite_native_controls']
    control['execution_provenance']=dict(receipt=PREFIX+'_actual_20261010_receipt.json',receipt_sha256=failure['receipt_sha256'],scientific_natural_exit_code=0,prior_consumer_runtime=prior['runtime'],prior_consumer_elapsed_seconds=prior['elapsed_seconds'],runner_failure_retained='Incorrect cheap-route vacuity assertion AFTER successful scientific consumer; fixed in current runner. Finite controls are reused immutable actual observations, NOT rerun or newly measured by this CLI.')
    for module in tuple(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file and Path(file).parent==ROOT and Path(file).suffix=='.py':
            captured.setdefault(Path(file).name,Path(file).read_bytes())
    captured[Path(__file__).name]=Path(__file__).read_bytes()
    npzpath=output.with_suffix('.npz');np.savez_compressed(npzpath,**arrays)
    capture=output.with_suffix('.zip')
    with zipfile.ZipFile(capture,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in sorted(captured.items()):z.writestr(name,raw)
    stable=all((ROOT/name).read_bytes()==raw for name,raw in captured.items())
    if not stable:raise ArithmeticError('Runtime source/input mutation')
    result=dict(status='NATURAL_CONSUMER_READY_OUTWARD_ALL129_SIGNED_NONSTATIONARY_COORDINATE_PI_RESPONSE',governing_plan=body.PLAN,governing_revision='7d1bacd0f61d2373186458d14953515e817c7b863732d858571b4db0736387a5',template_revision='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',work_item='induced-photon/verification',clock_exact=str(CLOCK),proof=PROOF,all129_states=states,finite_native_controls=control,all129_capsule=dict(path=npzpath.name,sha256=hashlib.sha256(npzpath.read_bytes()).hexdigest(),arrays={name:list(v.shape) for name,v in arrays.items()},scope='polynomial and paid HS error balls; tiny corrections stored separately to avoid subtractive rounding; NOT measured evolution'),input_bytes=dict(path=capture.name,sha256=hashlib.sha256(capture.read_bytes()).hexdigest()),source_input_sha256={name:hashlib.sha256(raw).hexdigest() for name,raw in captured.items()},stable_source_input_bytes=stable,runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,interval_digits=iv.dps),elapsed_seconds=time.monotonic()-started)
    output.write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/(PREFIX+'_actual_receipt.json'));args=parser.parse_args()
    with threadpool_limits(limits=1):result=run(args.output)
    print(json.dumps(dict(status=result['status'],elapsed_seconds=result['elapsed_seconds'],states={k:v['D_remainder_to_signal'] for k,v in result['all129_states'].items()})),flush=True)
