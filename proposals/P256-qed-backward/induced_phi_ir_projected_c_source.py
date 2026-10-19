#!/usr/bin/env python3
"""NEW pure preferred222 projected-heavy source, with actual finite-sea control.

The native N6 complete sea is an exactly factorized CAR state, not the macro
wave or photon vacuum. Existing prepared densities and receipts are unchanged.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any
import numpy as np
from scipy.linalg import block_diag, expm, null_space
from induced_phi_backreaction import PhiConfig, SELECTED, models, scalar_paths
from induced_phi_ir_preferred_source import slater, paired_charge, CAR_Hamiltonian
from induced_phi_ir_c_full_action import BASIS, T, B, orbit, field_C, SOURCE_BYTES as ACTION_BYTES
from induced_phi_ir_c_mixture_functional import scalar_exp, scalar_log, cumulant_law
from induced_phi_ir_well_transfer import rational, endpoint

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**ACTION_BYTES,**{name:(ROOT/name).read_bytes() for name in (
    'induced_phi_ir_c_mixture_functional.py','induced_phi_ir_well_transfer.py',Path(__file__).name)}}
SOURCE_HASHES={name:hashlib.sha256(raw).hexdigest() for name,raw in SOURCE_BYTES.items()}


def primitive_block(model: Any, phi: np.ndarray, links: np.ndarray, theta: np.ndarray) -> np.ndarray:
    onsite,paths=scalar_paths(model); h=float(onsite)*phi[0]
    for displacement,scalars in paths:
        direction=int(np.flatnonzero(displacement)[0]);length=int(displacement[direction])
        transport=np.linalg.matrix_power(links[direction],length)
        F=np.einsum('m,mij->ij',scalars,phi)
        hopping=(F@transport+transport@F)/2
        phase=np.exp(1j*theta@displacement)
        h+=phase*hopping+phase.conjugate()*hopping.conj().T
    return h


def occupied_state(h: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    values,vectors=np.linalg.eigh(h)
    if values[3]>=0 or values[4]<=0:raise ArithmeticError('Native half-filled occupied cluster lost its gap')
    return slater(vectors[:,:4]),values


def project_section(ray: np.ndarray,base: np.ndarray) -> tuple[np.ndarray,float]:
    overlap=np.vdot(ray,base)
    if abs(overlap)<=.5:raise ArithmeticError('Actual projected Slater section has no positive denominator')
    return ray*overlap/abs(overlap),float(abs(overlap)**2)


def native_complete_sea(cfg: PhiConfig,mean: dict[str,Any]) -> dict[str,Any]:
    enclosure=mean['centre_enclosure']
    g0=(rational(enclosure['g0_lower'])+rational(enclosure['g0_upper']))/2
    gs=(rational(enclosure['gs_lower'])+rational(enclosure['gs_upper']))/2
    frame=np.sqrt(np.array([float(g0),float(gs),float(gs),float(gs)]))[:,None,None]*SELECTED
    normal=null_space(orbit(frame).T)
    if normal.shape!=(60,45):raise ArithmeticError('Actual Cart60 orbit does not have45 normal coordinates')
    local=np.einsum('aij,bji->ab',BASIS,B@BASIS.conj()@B.conj().T).real/4
    R=block_diag(-local,local,local,local); RN=normal.T@R@normal
    normal_error=float(np.linalg.norm(R@normal-normal@RN))
    orth_error=float(np.linalg.norm(RN.T@RN-np.eye(45)))
    q=.004*np.sin(.31*np.arange(90)+.4)/np.sqrt(90)
    cq=np.concatenate((RN@q[:45],(-local@q[45:].reshape(3,15).T).T.ravel()))
    delta=np.einsum('ma,aij->mij',(normal@q[:45]).reshape(4,15),BASIS)
    phi=frame+delta; cphi=field_C(phi)
    links=np.array([expm(1j*np.einsum('a,aij->ij',t,T)) for t in q[45:].reshape(3,15)])
    clinks=B@links.conj()@B.conj().T
    constructed_cphi=frame+np.einsum('ma,aij->mij',(normal@cq[:45]).reshape(4,15),BASIS)
    constructed_clinks=np.array([expm(1j*np.einsum('a,aij->ij',t,T)) for t in cq[45:].reshape(3,15)])
    pullback_error=max(float(np.linalg.norm(cphi-constructed_cphi)),float(np.linalg.norm(clinks-constructed_clinks)))
    C,implementation=paired_charge(); target=np.argmax(abs(C),axis=0);signs=C[target,np.arange(256)]
    def charge(v: np.ndarray) -> np.ndarray:
        out=np.empty_like(v);out[target]=signs*v;return out
    size=6; seen=set();pairs=[]
    for index in itertools.product(range(size),repeat=3):
        if index in seen:continue
        mate=tuple((-i-1)%size for i in index)
        seen.update((index,mate)); pairs.append((index,mate))
    if len(pairs)!=108 or len(seen)!=216:raise ArithmeticError('Anti-periodic native sea omitted momentum modes')
    rows=[];identity=np.eye(4);gap=np.inf;max_cov=0.;min_overlap=1.;max_projector=0.;max_phase=0.
    wrong_phase=100*float((cq-q)@(cq-q)/np.linalg.norm(cq-q))
    for _,model,role in models(cfg):
        heavy=role=='neutral heavy -Lambda';weight=-1 if heavy else 1
        base_phase=1.+0j; project_cov=0.;project_overlap=1.;trace_sum=0.;native_pair_trace_max=0.;wrong_cov=0.
        for ordinal,(index,mate) in enumerate(pairs):
            theta=2*np.pi*(np.array(index)+.5)/size;other=2*np.pi*(np.array(mate)+.5)/size
            h0=weight*np.asarray(block_diag(primitive_block(model,frame,np.array([identity]*3),theta),primitive_block(model,frame,np.array([identity]*3),other)))
            hq=weight*np.asarray(block_diag(primitive_block(model,phi,links,theta),primitive_block(model,phi,links,other)))
            hc=weight*np.asarray(block_diag(primitive_block(model,cphi,clinks,theta),primitive_block(model,cphi,clinks,other)))
            base,_=occupied_state(h0);ray,values=occupied_state(hq);cray,cvalues=occupied_state(hc)
            state,overlap=project_section(ray,base);cstate,coverlap=project_section(cray,base)
            cbase=charge(base);phase=np.vdot(base,cbase);phase/=abs(phase)
            base_phase*=phase;max_phase=max(max_phase,float(np.linalg.norm(cbase-phase*base)))
            error=float(np.linalg.norm(charge(state)-phase*cstate));project_cov=max(project_cov,error)
            project_overlap*=min(overlap,coverlap);min_overlap=min(min_overlap,overlap,coverlap)
            gap=min(gap,float(values[4]-values[3]),float(cvalues[4]-cvalues[3]))
            actual_trace=float(np.trace(hq).real);trace_sum+=actual_trace;native_pair_trace_max=max(native_pair_trace_max,abs(actual_trace))
            positive_overlap=np.vdot(base,state)
            max_projector=max(max_projector,abs(float(positive_overlap.imag)),abs(float(positive_overlap.real)-np.sqrt(overlap)))
            wrong_cov=max(wrong_cov,float(np.linalg.norm(charge(state)-phase*cstate*np.exp(1j*wrong_phase))))
            if ordinal==0 and heavy:
                H=CAR_Hamiltonian(hq);HC=CAR_Hamiltonian(hc)
                # The complete real-space hopping matrix has trace0 exactly.
                # A selected momentum pair need not: retain its CAR constant.
                car_error=float(np.linalg.norm(C@H@C.conj().T-HC-np.trace(hq)*np.eye(256)))
                if car_error>1e-9:raise ArithmeticError('Native literal CAR transformation dropped its trace constant')
        max_cov=max(max_cov,project_cov)
        rows.append(dict(role=role,occupied=2*size**3,species_count=1 if heavy else (63 if role=='auxiliary filled charged' else 1),
            physical_multiplier='2^8192' if heavy else '1',C_base_phase=[base_phase.real,base_phase.imag],
            max_projected_section_C_error=project_cov,complete_sea_projected_overlap_squared=project_overlap,
            complete_sea_onebody_trace=trace_sum,max_partial_pair_trace=native_pair_trace_max,wrong_field_phase_C_error=wrong_cov))
    if max(max_cov,max_phase,max_projector,normal_error,orth_error,pullback_error)>1e-10 or min_overlap<.9 or min(row['wrong_field_phase_C_error'] for row in rows)<.01:
        raise ArithmeticError('Actual projected complete-sea source control failed')
    return dict(size=size,sites=216,momentum_pairs=108,all_allowed_momenta=216,Fock_dimension_per_pair=256,
        actual_pair_charge=implementation,field_coordinates=90,normal_coordinates=45,normal_C_error=normal_error,
        normal_C_orthogonality_error=orth_error,field_and_link_pullback_error=pullback_error,
        minimal_pair_projection_overlap_squared=min_overlap,minimum_occupied_cluster_gap=gap,
        max_C_base_ray_error=max_phase,max_projected_section_C_error=max_cov,max_positive_base_overlap_error=max_projector,
        literal_partial_pair_CAR_trace_constant_error=car_error,profiles=rows,
        scope='Actual complete N6 anti-periodic sea, factorized into108 actual p,-p CAR Slaters. Uniform generic Cart45+noncommuting SU4 link45 physical fields. No Fock-dimensional truncation of this finite uniform-field sea; tensor-product observables are evaluated without allocating2^864 components. Diagnostic macrocentre midpoint ONLY; source uses exact implicit macro222 root. Native displacement.004 is a control, not eta2^-2048.',
        trace_accounting='Full real-space trace0 is exact because onsitePhi0 is traceless and both path lengths1/2 are<N. Selected momentum-pair traces are NONZERO; literal CAR transform includes Tr(hpair) I. Summing all216 modes cancels these constants; no physical energy subtraction or counterterm.')


def complex_value(value: dict[str,float]) -> complex:
    return complex(value['real'],value['imag'])


def pair_regrouping(mixture: dict[str,Any]) -> dict[str,Any]:
    components=mixture['components'];k=np.array([[complex_value(v) for v in row['bare_cumulants']] for row in components])
    if len(components)!=128:raise ArithmeticError('Mixture lacks actual128 components')
    paired=[];direct=[]
    for j in range(0,128,2):
        original,image=components[j:j+2]
        if original['C_image'] or not image['C_image'] or original['twist']!=image['twist']:raise ArithmeticError('C pair ownership mismatch')
        a=k[j];paired.append([0,a[1]+a[0]**2,0,a[3]+4*a[0]*a[2]-2*a[0]**4])
        z=(scalar_exp(np.array([complex_value(v) for v in original['bare_log_coefficients']]))+scalar_exp(np.array([complex_value(v) for v in image['bare_log_coefficients']])))/2
        direct.append(scalar_log(z)[1:]*np.array([1,2,6,24]))
    paired=np.asarray(paired);direct=np.asarray(direct);law=cumulant_law(paired);old=cumulant_law(k)
    error=float(np.max(abs(law['total']-old['total'])/(1+abs(old['total']))))
    pair_error=float(np.max(abs(paired-direct)/(1+abs(direct))))
    if max(error,pair_error)>1e-9:raise ArithmeticError('Pair regrouping lost persistent-current cumulants')
    return dict(same_density=True,components=64,identity='Wpair=even(Wt)+logcosh(odd(Wt)); k2pair=k2t+k1t^2; k4pair=k4t+4k1t*k3t-2k1t^4; finalk4=E[k4pair]+3Var(k2pair). Complex bilinear covariance, no conjugation.',
        paired_cumulants=paired,total=law['total'],direct128_relative_error=error,direct_pair_Z_before_log_relative_error=pair_error,
        retained_mean_4k1k3=complex(np.mean(4*k[::2,0]*k[::2,2])),retained_mean_minus2k1fourth=complex(np.mean(-2*k[::2,0]**4)),
        limitation='Regrouping is the SAME frozenrho128. Nativebarefermion coefficients are not fullmacroquantumEM derivatives.')


def reprice_projected_section(well: dict[str,Any]) -> dict[str,Any]:
    p={name:rational(value) for name,value in well['prices'].items()};D1=p['Berry_D1_upper'];D2=p['Berry_D2_upper']
    displacement=2*D1*p['eta']
    if displacement>=Fraction(1,4) or D1**2>D2:raise ArithmeticError('Projected section denominator/derivative guard failed')
    newD1=4*D1;newD2=4*D2+52*D1**2
    if newD2>64*D2:raise ArithmeticError('Actual projected section repricing exceeded conservative multiplier')
    names=('heavy_Berry_Gauss_affine_action_upper','physical_state_norm_distance_upper','density_trace_norm_distance_upper',
        'difference_graph_norm_upper','weighted_Phi_action_error','all_flavour_current_action_error','all_flavour_contact_action_error',
        'all_flavour_current_expectation_error','all_flavour_contact_expectation_error')
    return dict(eta=endpoint(p['eta']),projector_displacement_upper=endpoint(displacement),
        projected_overlap_squared_lower=endpoint((1-displacement)**2,False),projected_D1_upper=endpoint(newD1),projected_D2_upper=endpoint(newD2),
        derivative_guard_D1_squared_le_D2=True,budget_multiplier=64,prices={name:endpoint(64*p[name]) for name in names},
        proof='For rank1 P=|Omega><Omega|, ||dP||<=2D1, ||d2P||<=2D2+2D1^2. v=P Omega0, ||v||>=1/2. Real normalized-vector derivative has norm<=1/r and Hessian norm<=3/r^2. Thus D1new<=4D1, D2new<=4D2+52D1^2<=64D2. All fibre-action summands nonnegative and linear in D1/D2, so fibre<=64old; unchanged charged/boundary/scalar terms imply norm<=64old. Both graph caps<=64old. Heinz exponents3/4+1/4 imply weightedPhi<=64old; literal current/contact action and expectation follow. These are NEW conservative prices, not oldreceipt edits.',
        unchanged='Original quantumlaw/parameters, eta/clock4096, exactimplicit means, exactspectralPhi shapes, slow/fast counts and scalarminmax/Agmon proof unchanged; native rounding images remain separately priced.')


def run(paths: dict[str,str]) -> dict[str,Any]:
    start=time.monotonic();inputs={}
    for role,path in paths.items():
        raw=Path(path).read_bytes();inputs[role]=dict(path=path,sha256=hashlib.sha256(raw).hexdigest(),capsule=json.loads(raw))
    mean=inputs['mean']['capsule'];well=inputs['well']['capsule'];preferred=inputs['preferred']['capsule'];action=inputs['action']['capsule'];mixture=inputs['mixture']['capsule']
    if action['input_descriptor']['inputs']['preferred_source']['sha256']!=inputs['preferred']['sha256']:raise ArithmeticError('Full action has a different preferredsource epoch')
    for role in ('mean','well'):
        if preferred['input_descriptor']['input_receipts'][role]['sha256']!=inputs[role]['sha256']:raise ArithmeticError('Preferredsource/mean/well receipt mismatch')
    for role in ('action','mixture'):
        cap=inputs[role]['capsule']
        if not all(cap['source_unchanged_since_import'].values()):raise ArithmeticError('Frozen scientific source changed during its own execution')
        for name,entry in cap['source_at_import'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=entry['sha256']:raise ArithmeticError('Joined final source drift '+name)
    cfg=PhiConfig(Lambda=1.,**mean['microscopic_config'])
    native=native_complete_sea(cfg,mean);prices=reprice_projected_section(well);regrouped=pair_regrouping(mixture)
    descriptor=dict(input_receipts={role:{key:value[key] for key in ('path','sha256')} for role,value in inputs.items()},
        branch='NEWpurepreferred222 projected-heavy C-equivariant source',N=262144,a='1/64',Lambda='2^8192',eta='2^-2048',clock=4096,
        governing_plan='issue-222-induced-photon-delta13-continuation-20261008',plan_revision='d21e32f5b786dc66274299d2b39a81fd9675c4a4572ab82f19586755b6d5da40',
        template_revision='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',work_item='joined-explanatory-campaign')
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
        exact_source='At EXACT implicit cubic222 C_t, choose normalized heavy filled upper-h Slater Omega0. For every actual chart q in the eta support let P_F(q) be the rank1 manybody heavy-ground spectral projector of the original -Lambda h(Phi,U). Omega_proj(q)=P_F(q)Omega0/sqrt(<Omega0,P_F(q)Omega0>). Equivalently normalize the wedge of the actual positive one-body spectral projector applied to the centre occupied columns; positive overlap fixes its phase. Tensor exactchargedcentre negativeSlaters(active1+aux63), actualRef4, normalizedevenrealEM H2 packet and unchanged exactslow/fast spectralGaussian radialcut. Extend with actual jointGauss w^-1/2 and only globalZ4. This is a NEW pure preparation, not rho64/rho128/Cpaired222 or finiteSCREEN.',
        exact_C_proof='Original fullaction gives C_F P_F(q) C_Fdag=P_F(Cq). At selfinverse222 the unique filled gapped free Slater Omega0 is a C_F eigenray, C_F Omega0=e^{i beta} Omega0. Positive real denominator is C invariant, hence C_F Omega_proj(q)=e^{i beta}Omega_proj(Cq) with CONSTANTphase. Spectralslow/fast Gaussian precisions commute with their C pullbacks because actual Hamiltonian Hessian and orbit R0 at the exactfixedmean do; spectralfunctionalcalculus includes degeneracies. Radialcut, normalizedevenEM packet, fixedbareQref1 filledRef4NET0 and fulljointJac/Haar W preserve C. Globalcentre action trivial sinceNtotalmod4=0. Therefore the NEW completepure density is C invariant. No arbitrarySlater phase gauge assumed.',
        exact_functional='For this FIXED normalizedpure rho_proj, Z[A]=Tr(U_full[A]rho_proj U_full[0]dag), Z[0]=1, W[A]=logZ[A] on its local analytic branch. Fullaction intertwining and fixedrhoC give Z[A]=Z[-A]; oddfunctionalderivatives, when defined, vanish. There are NO classical betweencomponent cumulants for thispurebranch, but intrinsicchi4 is NOT zero or evaluated. External electron+Refhole source is different and excluded.',
        actual_quantum_native_consumer=native,new_projected_section_transfer=prices,same_frozen128_pair_regrouping=regrouped,
        mathematical_review_required='Actualalias/two-cut preferred222 selection, localwell/minmax/Agmon transfer, originalfullaction common domain/Jac, rank1projector derivatives and new64x transfer proof. Noautomaticadmission from nativecontrols.',
        limitations='Preparedfinite localwell source only; no globalground, bulkstationarity, noncompactEM/photonvacuum, commoncone, dressedCompton or QED ontology proof. Barefermion mixturejets retained as their own N6 branch. No490/738 Gamma3 transfer or UVmasscountertermfork.',
        source_at_import={name:dict(sha256=SOURCE_HASHES[name],bytes_utf8=raw.decode()) for name,raw in SOURCE_BYTES.items()},
        source_unchanged_since_import={name:(ROOT/name).read_bytes()==raw for name,raw in SOURCE_BYTES.items()},
        runtime=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,platform=platform.platform(),argv=sys.argv),elapsed_seconds=time.monotonic()-start)


def serialize(value: Any) -> Any:
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,complex):return dict(real=value.real,imag=value.imag)
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for role in ('mean','well','preferred','action','mixture'):parser.add_argument('--'+role,required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    result=run({role:getattr(args,role) for role in ('mean','well','preferred','action','mixture')})
    Path(args.output).write_text(json.dumps(result,indent=2,default=serialize)+'\n')
    print(json.dumps({name:result[name] for name in ('input_sha256','actual_quantum_native_consumer','new_projected_section_transfer','same_frozen128_pair_regrouping','elapsed_seconds')},indent=2,default=serialize))
