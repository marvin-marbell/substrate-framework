"""NEW fixed-body-to-global-field all129 current transfer, conditional root/chart.

Theorem (no globally bounded fullfield RB). On the body Schwartz graph core,
HT=T(Hb+E)+R, T is an isometry and ||R||<=epsilon uniformly in ALL Theta
and fixed-N fibre vectors. Closedness of H and graph density extend T:D(Hb)
->D(H). Duhamel yields d=||Delta v||<=t epsilon||v|| and
h=||H Delta v||<=|E|d+t epsilon||Hb v||+2epsilon||v||, from EXACTLY
HDelta=E Delta+Delta Hb+UF R-exp(-iEt)R UB. No bounded-current hypothesis.

Bare Gram gives c0=Lambda/64. Matter Schur gives M>=-B X. Young absorbs
HALF the quartic: H+C>=c X^4, c=c0/2. Factor B<=Lambda B0;
sup(B0 x-x^4/128)^3=27*128*B0^4/256. Rational cuberoot upper prices
C=Lambda C0, not 1+YoungCube. Cauchy twice gives
||X xi||^4<=||xi||^2(<xi,Hxi>+C||xi||^2)/c.
Thus ||J xi||_l2<=kappa*(d^2*(d*h+C*d^2)/c)^1/4, valid globally
AFTER propagation; no compact tube hypothesis is applied to UF errors.

Native charged paths give kappa=sum_signed_paths 4*load*b_r*|q_r|;
Q orthonormal implies |q|<=length. Nuclear block bound4 retains both
Phi endpoints, all63 copies and AP seams. Heavy charge0 never enters J.
Endpoint A=Jfull T-T Jbody is uniform in Theta. Field Lipschitz and the
actual REAL radial full-Fock W1 give ||A||<=eta=2 sqrt(TrC)(1+9p)
*2*(3*kappa+2*jbody*W1). Gaussian degree3 Holder and proven normalizer2
supply this price. Haar conjugation is unitary and the common Z4 lift N8324
is trivial. Full support X<=14+3R is used ONLY for T and R, never UF R.

Endpoint graph identity: HA=E A+A Hb+(Kfull T-T Kbody)+Jfull R-R Jbody,
K=[H,J]. Bosonic potentials commute J. Neutral CAR commutes charged J.
The native Cartesian/SU4 differential operator bounds Kfull T using the
initial kinetic form energy, all129 Theta Gaussian derivatives, HC and
charged M. Energy-form bound avoids any future Wick/Slater assertion.

For retarded C=<J UF Tpsi,UF J Tpsi>, body Cb=<Jb UBpsi,UB Jbpsi>,
a=Jfull Delta psi+A UBpsi, b=UF Apsi+Delta Jbpsi. Then Frobenius
||Rfull-Rbody||<=2[||a|| b0+jbody*n||b||+||a||||b||]. Both current
insertions and all129x129 entries are included, not one response entry.
Raw source n^2=.5, background n^2=1. Body Hb here is charged HE+HC+M;
neutral root sea is assigned to E, not counted twice. Full static reference
minus unit charged rootsea is bounded by bare mean quartic, neutral rootsea,
and chi/2 Gmax TrA. No old spectral gap/amputation law is imported.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import zipfile
import numpy as np
from threadpoolctl import threadpool_limits
import joined_phi_u1_body_word as body
import joined_phi_u1_current_moments as moments
import induced_phi_nonuniform_gauss_quantum as owner

ROOT=Path(__file__).resolve().parent
PREFIX='joined_phi_u1_full_current_graph_transfer'
PARENT='joined_phi_u1_actual_compact_source_action_certificate_receipt.json'
sys.set_int_max_str_digits(0)
PLAN='issue-222-induced-photon-delta13-continuation-20261008'
REV='14e500c5d51931331025515a442791f07cc0af03c15e7c178384b7c133767175'
TEMPLATE='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'

def upper(v: dict[str,Any]) -> F:
    for k in ('upper','exact_upper','exact'):
        if k in v:return F(v[k])
    raise ValueError('No authoritative rational upper')

def root_bound(mean:dict[str,Any])->F:
    gsum=F(mean['parameters']['g0']['upper'])+3*F(mean['parameters']['gs']['upper'])
    if not 0<gsum<F(2,3):raise ArithmeticError('Missing actual root Phi bound')
    return gsum

def kinetic_norm_squared(variance:F)->F:
    return F(64)**2*129*131/(16*variance**2)

def bind_variance(variance:float)->None:
    if variance!=2:raise ArithmeticError('Actual Theta density variance2 required')

def radius_price(gb:dict[str,Any])->Any:
    return 2*owner.iv.sqrt(owner.exact(upper(gb['full_trace_covariance'])))*(1+9*owner.exact(upper(gb['corrector_L2'])))

def cuberoot_upper(v: F) -> F:
    # Exact integer bisection; this is a derived price, not floating cuberoot.
    lo,hi=0,1
    while F(hi)**3<v:hi*=2
    while hi-lo>1:
        m=(lo+hi)//2
        if F(m)**3>=v:hi=m
        else:lo=m
    return F(hi)

def bind_parent(p: dict[str,Any]) -> F:
    split=p['reference_split']
    for k in ('charged_root_sea_constant_removed','W_Theta_independent','source_maps_fixed_body_Hilbert_not_fullfield_RB'):
        if split[k] is not True:raise ArithmeticError('Missing actual transfer antecedent: '+k)
    e=upper(p['action_input']['epsilon'])
    if e<0:raise ArithmeticError('Negative residual price')
    return e

def run(parent: Path|None, output: Path) -> dict[str,Any]:
    start=time.monotonic(); owner.iv.dps=110
    iv,ex,cap=owner.iv,owner.exact,owner.bounds
    captured:dict[str,bytes]={}
    sources={}
    for module in tuple(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file and Path(file).parent==ROOT and Path(file).suffix=='.py':
            sources[Path(file).name]=Path(file).read_bytes()
    sources[Path(__file__).name]=Path(__file__).read_bytes()
    def take(name:str,pin:str|None=None)->dict[str,Any]:
        raw=(ROOT/name).read_bytes()
        if pin and hashlib.sha256(raw).hexdigest()!=pin:raise ArithmeticError('Pinned bytes changed: '+name)
        captured[name]=raw;return json.loads(raw)
    kernel=take('joined_phi_u1_finite_current_kernel_receipt.json','0fff4a884cd6c822851142b04c21f0e2c3d53829a6f92ea752a44ad9edcd18ea')
    gc=take('induced_phi_gaussian_action_certificate_receipt.json')
    wc=take('induced_phi_global_occupied_kato_derivative_control_receipt.json')
    domain=take('induced_phi_compact_source_global_domain_control_corrected_receipt_20261010.json','f6a6b6d79496e4e24d0a5df4762a9098b0db24debe16c2f82c32572696592d2f')
    arch=take(body.ARCHIVE,body.ARCHIVE_SHA)
    desc=arch['input_descriptor']; mean=desc['exact_NEW_allq_BO_mean']
    gb=gc['outward_derived_certificate']['bounds']
    if upper(gb['relative_full_form_error'])>=F(1,4):raise ArithmeticError('Full covariance comparison fails')
    gsum=root_bound(mean)
    if gc['exact_original_wave_ABI']['exact_mean']!=mean:
        raise ArithmeticError('Full covariance and current body use different roots')
    parameters={'N':4,'V':64,'a':'1/64','UEphys':1,'u':64,'Lambda':'2^8192','chi':'2^-4096','normal':5760,'local_angles':960,'total_particles':8324}
    if any(desc['parameters'][k]!=v for k,v in parameters.items()):
        raise ArithmeticError('Actual source parameter ABI changed')
    data,species,inputs,_=body.full_inputs();captured.update(inputs)
    bind_variance(body.S);bind_variance(moments.S)
    Q=data['native']['Q'];body.require(float(np.linalg.norm(Q.T@Q-np.eye(129),2)))
    profiles=data['profiles']
    Ks=[abs(F(p['onsite']))+2*sum((abs(F(r['c0']))+abs(F(r['ci'])) for r in p['paths']),F(0)) for p in profiles]
    if [p['load'] for p in profiles]!=[1,1,63]:raise ArithmeticError('Charged multiplicities changed')
    lam,chi,R=F(2)**8192,F(2)**-4096,F(2)**-20
    c0,c=lam/64,lam/128
    B=128*lam*Ks[0]+129*Ks[1]+8064*Ks[2]
    B0=128*Ks[0]+129*Ks[1]+8064*Ks[2] # Lambda>=1.
    C0=cuberoot_upper(27*128*B0**4/256);C=lam*C0
    assert C0**3>=27*128*B0**4/256
    Xroot=F(14);Xs=Xroot+3*R
    mbody=129*Ks[1]+8064*Ks[2]
    Gmax=F(108)**2 # local full cometric bound; reference only.
    E=4096*lam*gsum**2+128*lam*Ks[0]*Xroot+chi*Gmax*upper(gb['full_trace_A'])/2
    # Actual native paths, BOTH orientations, all63 auxiliary copies.
    jk=[F(0)]*4; physical=[];defects=[]
    for sp in species:
        paths=body.joined.old.native_paths(sp.profile,data['native'])
        geo=moments.geometry(data,sp)
        defects.append(geo['ward_defect'])
        for path,q in zip(paths,sp.path_modes,strict=True):
            length=int(path['length'])
            row=next(v for v in sp.profile['paths'] if v['axis']==path['axis'] and v['length']==length)
            b=abs(F(row['c0']))+abs(F(row['ci']))
            if abs(sum(abs(complex(v)) for v in path['coefficients'])-float(b))>1e-9:
                raise ArithmeticError('Native coefficients disagree with exact profile price')
            if np.linalg.norm(q)>length+1e-9:raise ArithmeticError('Q path contraction fails')
            for k in range(4):jk[k]+=2*4*sp.profile['load']*b*length**(k+1)
        physical.append(dict(role=sp.profile['role'],forward_paths=len(paths),signed_paths=2*len(paths),load=sp.profile['load'],ward_defect=geo['ward_defect'],Q_path_norms=np.linalg.norm(sp.path_modes,axis=1).tolist()))
    gauge_controls=[body.joined.old.gauge_ward(p,data['native'],data['geom'],(1,1,0)) for p in profiles]
    generators=data['geom']['M']/np.sqrt(8)
    gram=np.einsum('aij,bji->ab',generators,generators)
    casimir=np.einsum('aij,ajk->ik',generators,generators)
    body.require(float(np.max(np.abs(gram-np.eye(15)/2))))
    body.require(float(np.max(np.abs(casimir-np.eye(4)*15/8))))
    kappa,j1,j2,j3=map(ex,jk); jbody=kappa*ex(Xroot)
    radius=radius_price(gb)
    W1=upper(wc['conditional_cases'][0]['Fock_directional_first_upper'])
    eta=radius*(3*kappa+2*jbody*ex(W1))
    # Normalizer2 is already in radius, not multiplied again.
    # rho_x is bounded by129 including Ref3 hole, all64 flavours.
    # Native periodic N4 density Laplacian has minimum positive eigenvalue2.
    # HC=64 rho Lplus rho <=64*(1/2)*64*129² (no extra half).
    if np.linalg.eigvalsh(data['native']['Lplus'])[-1]>.5+1e-9:
        raise ArithmeticError('Native density inverse violates N4 price')
    Cop=ex(F(64)*F(1,2)*64*129**2); tt=ex(F(2)**-39)
    eps=None
    if parent is not None:
        raw=parent.read_bytes();captured[parent.name]=raw
        pr=json.loads(raw);eps=ex(bind_parent(pr))
        if pr['domain_receipt_sha256']!=hashlib.sha256(captured['induced_phi_compact_source_global_domain_control_corrected_receipt_20261010.json']).hexdigest():
            raise ArithmeticError('Source action and graph transfer use different domain epochs')
    # Independent proof mode honestly remains pending; no small epsilon substituted.
    result:dict[str,Any]=dict(status='PENDING_ACTUAL_PARENT_NORM_BINDING' if eps is None else 'ACTUAL_PARENT_BOUND_CONDITIONAL_FULL129_GRAPH_TRANSFER',governing_plan=PLAN,governing_revision=REV,template_revision=TEMPLATE,
        proof=__doc__,arithmetic='110-digit outward mpmath intervals from exact rational inputs; body1percentF/floating allowance NOT interval-certified',
        global_constants={k:cap(v) for k,v in dict(bare_quartic=ex(c0),graph_quartic=ex(c),matter_B=ex(B),Young_C0=ex(C0),additive_C=ex(C),Erefabs=ex(E),current_kappa=kappa,body_J_operator=jbody,endpoint_norm=eta,prepared_full_radius=radius,real_W1=ex(W1)).items()},
        actual_native_paths=physical,charged_particles_source=8193,charged_particles_background=8192,total_N=8324,neutral_charge=0,body_reference_charged_root_constant_removed=True,
        body_first_moment_scope='Existing all129 conditional first-moment signal minus body remainder, with declared 1percentF and floating upward allowance; raw source norm squared .5 and background1 unchanged',states={})
    if eps is not None:
        def delta(n:Any,hb:Any)->tuple[Any,Any,Any]:
            d=tt*eps*n;h=ex(E)*d+tt*eps*hb+2*eps*n
            weighted=kappa*(d**2*(d*h+ex(C)*d**2)/ex(c))**ex(F(1,4))
            return d,h,weighted
        for state,entry in kernel['all129_states'].items():
            mass=F(str(entry['raw_norm_squared']));n=iv.sqrt(ex(mass))
            # Gaussian Pi: ||HEpsi||=64 sqrt(129*131)/(4s), s=2;
            # ||gradpsi||=sqrt(129/(4s)); source Ref channels independent Theta.
            kinetic=iv.sqrt(ex(kinetic_norm_squared(F(2))))*n
            grad=iv.sqrt(ex(F(129,8)))*n
            hb=kinetic+(Cop+ex(mbody))*n
            # All129 contact derivatives price [Hb,J] on initial Gaussian.
            kb=ex(64)*(j2*n+2*j1*grad)*ex(Xroot)+2*(Cop+ex(mbody)*ex(Xroot))*jbody*n
            hbj=jbody*hb+kb
            d,h,wdelta=delta(n,hb)
            dj,hj,wj=delta(jbody*n,hbj)
            # Global native kinetic form KfullT: positivity plus matter support
            # bounds initial kinetic, no tube after time or Wick propagation.
            kineticform=ex(E)*n**2+n*hb+eps*n**2+ex(B*Xs)*n**2
            Pfull=iv.sqrt(2*kineticform/ex(chi))
            # 3840 CartPhi derivatives linear in Phi; 2880 SU4 generators,
            # Tr(TaTb)=delta/2 and Casimir15/8. Native paths length<=2.
            jgrad=8*iv.sqrt(ex(3840))*kappa+iv.sqrt(ex(F(3,10)*2880))*j1*ex(Xs)
            jlap=ex(F(3,10)*F(15,8))*j2*ex(Xs)
            kf=ex(chi)/2*(jlap*n+2*jgrad*Pfull)+ex(64)*ex(Xs)*(j2*n+2*j1*grad)+2*(Cop+ex(mbody*Xs))*kappa*ex(Xs)*n
            endpointgraph=ex(E)*eta*n+eta*hb+kf+kb+(kappa*ex(Xs)+jbody)*eps*n
            endpoint_d=eta*n
            propagated_endpoint_current=kappa*(endpoint_d**2*(endpoint_d*endpointgraph+ex(C)*endpoint_d**2)/ex(c))**ex(F(1,4))
            left=wdelta+eta*n
            right=eta*n+dj
            b0=ex(F.from_float(entry['current_inserted_state_norm']))*ex(F(1000001,1000000))
            transfer=2*(left*b0+jbody*n*right+left*right)
            margin=ex(F.from_float(entry['clock']['signal_minus_remainder']))
            accepted=owner.endpoint(transfer,1)<owner.endpoint(margin,0)
            # Physical wrong-law control reaches the consumer, not source echo.
            wrong_lambda_transfer=transfer*ex(lam)**2
            if owner.endpoint(wrong_lambda_transfer,0)<=owner.endpoint(margin,1):
                raise ArithmeticError('Wrong heavy scale on charged current control is blind')
            result['states'][state]=dict(raw_norm_squared=str(mass),prices={k:cap(v) for k,v in dict(Hbpsi=hb,HbJpsi=hbj,Kbody=kb,KfullT=kf,Delta_norm=d,HDelta_norm=h,current_Delta=wdelta,Delta_Jbody_norm=dj,HDelta_Jbody_norm=hj,current_Delta_Jbody=wj,endpoint_graph_norm=endpointgraph,propagated_endpoint_current=propagated_endpoint_current,current_left_error=left,current_right_error=right,physical_full129_retarded_transfer_error=transfer,wrong_Lambda_chargedscale_transfer=wrong_lambda_transfer,existing_body_signal_margin=margin).items()},below_existing_body_signal_margin=accepted,clock_exact=str(F(2)**-39))
            if not accepted:raise ArithmeticError('Actual useful graph transfer fails: '+state)
        result['actual_epsilon']=cap(eps)
    result['controls']=controls(jk,lam,radius,eta,gb,mean)
    result['actual_SU4_Noether_and_generator_controls']=dict(profiles=gauge_controls,
        all15_generator_Gram_defect=float(np.max(np.abs(gram-np.eye(15)/2))),
        Casimir15over8_defect=float(np.max(np.abs(casimir-np.eye(4)*15/8))),
        generator_count=len(generators),Phi_real_dimension=3840,colour_real_dimension=2880)
    result['input_sha256']={k:hashlib.sha256(v).hexdigest() for k,v in captured.items()}
    result['source_sha256']={k:hashlib.sha256(v).hexdigest() for k,v in sources.items()}
    result['conditional_limits']=['Captured N4 root/static positive sea/chart/gaps/Schwartz domain remain conditional.','Uniform actual parent map residual is not global fullfield RB.','No measured full129 time evolution, ground, pole, common cone, original SOURCE/CURRENT427 admission.','No electromagnetic magnetic term; physical body HE/HC only.']
    result['elapsed_seconds']=time.monotonic()-start
    output.write_text(json.dumps(result,indent=2)+'\n')
    archive=output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in captured.items():z.writestr('inputs/'+name,raw)
        for name,raw in sources.items():z.writestr('sources/'+name,raw)
    result['stable_input_source_bytes']=all((ROOT/k).read_bytes()==v for k,v in {**captured,**sources}.items())
    if not result['stable_input_source_bytes']:raise ArithmeticError('Runtime bytes changed during consumer')
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result

def controls(jk:list[F],lam:F,radius:Any,eta:Any,gb:dict[str,Any],mean:dict[str,Any])->dict[str,Any]:
    # Exact H=-1/2D²+x4 on normalized shifted Gaussian density N(L,1/2).
    # xi=eps*fL. J=x: ||Jxi||²=eps²(L²+1/2), <xi,Hxi>=
    # eps²(L4+3L²+1), so quartic graph bounds actual unbounded insertion.
    e,L=F(2)**-40,F(2)**80
    norm2=e**2;j2=e**2*(L**2+F(1,2));energy=e**2*(L**4+3*L**2+1)
    graphfourth=norm2*energy
    assert j2>norm2*1000 and j2**2<=graphfourth
    # Unit variance changes the consumer's Hb and HbJ/contact derivative prices.
    HE2=kinetic_norm_squared(F(2))
    HE1=kinetic_norm_squared(F(1))
    assert HE1==4*HE2
    try:bind_variance(1)
    except ArithmeticError:pass
    else:raise ArithmeticError('Wrong unit variance source check is blind')
    wrong_kappa=jk[0]*lam
    assert wrong_kappa!=jk[0] and wrong_kappa>jk[0]*F(2)**8000
    fullC=upper(gb['full_trace_covariance'])
    # Cov omission zeroes a nonzero endpoint price, not incidental forwarding.
    if owner.endpoint(eta,0)<=0 or fullC<=0:raise ArithmeticError('Endpoint covariance adverse control blind')
    wrong_gb={**gb,'full_trace_covariance':{'upper':'0'}}
    if owner.endpoint(radius_price(wrong_gb),1)!=0:
        raise ArithmeticError('Cov omission does not change actual endpoint consumer')
    missing=dict(mean);missing.pop('parameters')
    try:root_bound(missing)
    except KeyError:root_rejected=True
    else:raise ArithmeticError('Root omission did not reject physical bound input')
    return dict(shifted_Gaussian=dict(exact_H='-.5D²+x^4',epsilon=str(e),L=str(L),norm_squared=str(norm2),current_norm_squared=str(j2),energy_form=str(energy),graph_current_fourth_upper=str(graphfourth),norm_only_current_bound_falsified=True,graph_coercivity_controls=True),wrong_Lambda_charged_current_kappa_ratio_exact=str(lam),wrong_unitvariance_HE_norm_squared_ratio='4',wrong_covariance_omission_endpoint_becomes_zero=True,missing_actual_root_rejected=root_rejected,actual_coefficient_path_and_Ward_controls='Native fullinput literal assembly and Q projection checked at runtime; all charged128/129 and63aux, native ordered one/two/AP, all129 contact derivatives retained')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--parent',type=Path);parser.add_argument('--output',type=Path,default=ROOT/(PREFIX+'_domain_corrected_receipt_20261010.json'));args=parser.parse_args()
    with threadpool_limits(limits=1):result=run(args.parent,args.output)
    print(json.dumps(dict(status=result['status'],output=str(args.output),sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),states={s:v['prices']['physical_full129_retarded_transfer_error']['upper_decimal'] for s,v in result['states'].items()},stable=result['stable_input_source_bytes'],elapsed=result['elapsed_seconds']),indent=2))
