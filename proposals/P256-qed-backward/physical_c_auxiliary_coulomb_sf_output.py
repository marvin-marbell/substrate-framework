#!/usr/bin/env python3
"""Actual SF C SF output, retaining the full negative-transfer complement.

C=exp(-u rho_total Lplus rho_total), u=.02 for auxiliarycharge1 in the
6a declared coordinates.  Intermediate R_C state change and final SF-damped
output are distinct.  No NF Fock matrix, adopted sea truncation, or fake floor.
The signed static-background extension uses the captured integer macroQ,
not the effective canonical charge e0.  Its screened Gram and output bounds
are conditional pure-Abelian auxiliary-sea results, not a RAWB/Khalf source
or physical pole admission.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from decimal import Decimal,localcontext,ROUND_HALF_EVEN
import os
from pathlib import Path
import platform
import sys
import time
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import mpmath as mp
import numpy as np
from scipy.linalg import eigh
from scipy.special import roots_hermitenorm
import physical_c_auxiliary_coulomb_character as coul
ROOT=Path(__file__).resolve().parent;SOURCE_BYTES=Path(__file__).read_bytes()
COUL_BYTES=coul.SOURCE_BYTES


def complement_data(s: dict) -> dict:
    n=len(s['e'])//2;negative=s['r'][:,:n];W=s['W']
    return dict(W=W,negative=negative,Tplus=W.conj().T@s['T']@W,Tminus=np.exp(s['e'][:n]))


def complement_log(data: dict,eta: np.ndarray,zeta: np.ndarray) -> np.ndarray:
    W,negative=data['W'],data['negative']
    qe=np.repeat(np.exp(1j*eta),4,axis=1);qz=np.repeat(np.exp(1j*zeta),4,axis=1)
    Ae=W.conj().T[None]@(qe[:,:,None]*W[None]);Az=W.conj().T[None]@(qz[:,:,None]*W[None])
    Be=W.conj().T[None]@(qe[:,:,None]*negative[None]);Cz=negative.conj().T[None]@(qz[:,:,None]*W[None])
    M0=Ae@data['Tplus'][None]@Az
    retained=(Be*data['Tminus'][None,None,:])@Cz
    correction=np.linalg.solve(M0,retained)
    return coul.log_oneplus(correction)


def exact_double_charge(nf: int,order: int,b: dict,s: dict,background: np.ndarray|None=None) -> complex:
    data=complement_data(s);v=len(b['targets'])
    index=np.array(list(itertools.product(range(order),repeat=4)),int)
    eta=np.column_stack((2*np.pi*index[:,:2]/order,np.zeros(len(index))))
    zeta=np.column_stack((2*np.pi*index[:,2:]/order,np.zeros(len(index))))
    values=np.empty(len(index),complex)
    for first in range(0,len(index),512):
        e,z=eta[first:first+512],zeta[first:first+512]
        log=coul.graph_character_log(s,s,e)+coul.graph_character_log(s,s,z)+complement_log(data,e,z)
        values[first:first+512]=np.exp(nf*log)
    coefficients=np.fft.fftn(values.reshape((order,)*4))/(order**4)
    charge=np.fft.fftfreq(order)*order;grid=np.array(list(itertools.product(charge,repeat=4)))
    rho=np.column_stack((grid[:,:2],-np.sum(grid[:,:2],axis=1)))
    sigma=np.column_stack((grid[:,2:],-np.sum(grid[:,2:],axis=1)))
    if background is not None:rho+=background;sigma+=background
    energy=np.einsum('ni,ij,nj->n',rho,b['Lplus'],rho)+np.einsum('ni,ij,nj->n',sigma,b['Lplus'],sigma)
    return complex(np.sum(coefficients.ravel()*np.exp(-.02*energy)))


def importance_output(nf: int,order: int,b: dict,s: dict,background: np.ndarray|None=None) -> dict:
    rank=len(b['gauge_eigenvalues']);prior=np.diag(.04/b['gauge_eigenvalues']);cov=nf*s['cov']
    proposal=np.linalg.inv(np.linalg.inv(prior)+b['gauge_frame'].T@cov@b['gauge_frame'])
    e,r=eigh(proposal);nodes,weights=roots_hermitenorm(order);weights/=math.sqrt(2*math.pi)
    index=np.array(list(itertools.product(range(order),repeat=2*rank)),int)
    eta=(nodes[index[:,:rank]]@(r*np.sqrt(e)).T)@b['gauge_frame'].T
    zeta=(nodes[index[:,rank:]]@(r*np.sqrt(e)).T)@b['gauge_frame'].T
    weight=np.prod(weights[index],axis=1);data=complement_data(s);denominator=0j;increment=0j;absolute_weight=0.
    for first in range(0,len(index),512):
        a,z=eta[first:first+512],zeta[first:first+512]
        log=nf*(coul.graph_character_log(s,s,a)+coul.graph_character_log(s,s,z))
        log+=.5*np.einsum('ni,ij,nj->n',a,cov,a)+.5*np.einsum('ni,ij,nj->n',z,cov,z)
        factor=weight[first:first+512]*np.exp(log)
        if background is not None:factor*=np.exp(1j*((a+z)@background))
        correction=nf*complement_log(data,a,z)
        denominator+=np.sum(factor);increment+=np.sum(factor*np.expm1(correction))
        absolute_weight+=float(np.sum(abs(factor)))
    relative=increment/denominator
    return dict(order=order,relative_norm2_increment=[float(relative.real),float(relative.imag)],
        conditional_fullK_output_outside_bare_Slater_fraction=float((relative/(1+relative)).real),
        total_positive_norm_ratio=1+float(relative.real),
        signed_doubleHS_cancellation_price=absolute_weight/abs(denominator),
        evaluation='Full retained Tminus term via log1p/eigenvalues and expm1; no subtraction of two unit-sized norms.')


def direct_retained_check(b: dict,s: dict) -> dict:
    eta=np.array([[.15,-.22,.37]]);zeta=np.array([[-.11,.21,.08]])
    data=complement_data(s);Qe=np.diag(np.repeat(np.exp(1j*eta[0]),4));Qz=np.diag(np.repeat(np.exp(1j*zeta[0]),4))
    M=s['W'].conj().T@Qe@s['T']@Qz@s['W'];sign,log=np.linalg.slogdet(M);psign,plog=np.linalg.slogdet(data['Tplus'])
    direct=np.exp(-2j*np.sum(eta+zeta))*sign/psign*np.exp(log-plog)
    factored=np.exp(coul.graph_character_log(s,s,eta)[0]+coul.graph_character_log(s,s,zeta)[0]+complement_log(data,eta,zeta)[0])
    return dict(direct_centered_whole_transfer_character=[float(direct.real),float(direct.imag)],
        factored_character=[float(factored.real),float(factored.imag)],equality_error=abs(direct-factored),
        whole_scalar_pref=s['pref'],whole_loglambda=s['loglambda'],
        retained_complement_log=[float(complement_log(data,eta,zeta)[0].real),float(complement_log(data,eta,zeta)[0].imag)])


def macro_output_bound() -> dict:
    with mp.workdps(160):
        k=mp.mpf('1e-250');nf=mp.mpf('6.25e1148');v=mp.mpf(2)**72
        bmin=1/(2*k)-3;gap=bmin-1/bmin;r=2*(3/bmin)/gap
        amin=(1-r*r)/(1+r*r);relative=4*r*r/(bmin*bmin*amin*amin)
        delta=2*v*nf*relative/(1-relative)
        charge_delta=mp.mpf(coul.macro_Riccati_character_bound()['uniform_macro_log_character_error'])
        phase_factor=mp.exp(4*charge_delta)/(mp.cos(charge_delta)**2)
        # At this scale expm1 is essential; 1+delta would round to1.
        outside=mp.expm1(delta)*phase_factor
        return dict(assumptions='PureAbelian/Phi0/SU4I heavy same-C sea at a fixed field. Exact positive Coul HS measure, auxiliarycharge1,u=.02. Not the full field/current/source or activeQ join.',
            transfer_complement_ratio='lambda_max(Tminus)/lambda_min(Tplus)<=1/Bmin² by minmax; Tplus>=Bmin, low eigenvalues<=1/Bmin.',
            charge_block_invertibility='sigma_min(W†Q W)>=(1-r²)/(1+r²), r bound on occupied graphR. Exact graph factorization, not an arbitrary Gram/fullrank assumption.',
            complement_bound='||E||<=4r²/[Bmin²*amin²], E=(Aeta Tplus Azeta)^-1 B_eta Tminus C_zeta. This RETAINS rather than deletes complement.',
            NF_log_retained_factor_bound=mp.nstr(delta,70),
            positive_HS_relative_output_norm_increment_bound=mp.nstr(outside,70),
            phase_control='Actual χ^NF differs positive leading charge character by uniform charge_delta. Therefore integral absolute-weight/positiveGram ratio<=exp(2charge_delta)/cos(charge_delta); doubleHS factor exp(4charge_delta)/cos².',
            conclusion='Final SF C SF output may be close to the bare sea even when intermediate Coul chargefilter output is far outside it. Small fullK complement does NOT make prepGram/source normalization free.',
            scalar_contract='Every SF carries exp(pref/2). Normalized denominator is FULL lambdaVac=exp(pref+sum positive h), raisedNF; scalar cancellation only after this exact normalization.',
            debts=['Mixed field/activeQ/Coul-source coupling and actual realGauss/centre consumer.','ORIGINALfixedbody prep/output Gram, fieldheat/outside/Phi/noncentralSU4 and physical pole.'])


def scaled_integer_charge(beta: str) -> dict:
    with localcontext() as context:
        context.prec=110;context.rounding=ROUND_HALF_EVEN
        pi=Decimal('3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651')
        target=(4*pi*Decimal(beta)/Decimal(coul.alpha.Parameters().inverse_alpha)).sqrt()
        q=int(target.to_integral_value())
        return dict(beta=beta,Q=str(q),target=str(target),e0=str(Decimal(q)/Decimal(beta).sqrt()),
            rounding='Decimal110 ROUND_HALF_EVEN, same captured Alpha56 charge_contract; no nonzero-Q floor.')


def exact_single_background(nf: int,b: dict,s: dict,background: np.ndarray) -> complex:
    order=4*nf+1;index=np.array(list(itertools.product(range(order),repeat=2)),int)
    eta=np.column_stack((2*np.pi*index/order,np.zeros(len(index))))
    values=np.empty(len(index),complex)
    for first in range(0,len(index),512):values[first:first+512]=np.exp(nf*coul.graph_character_log(s,s,eta[first:first+512]))
    coefficients=np.fft.fftn(values.reshape(order,order))/(order**2)
    charges=np.fft.fftfreq(order)*order;grid=np.array(list(itertools.product(charges,repeat=2)))
    rho=np.column_stack((grid,-np.sum(grid,axis=1)))+background
    energy=np.einsum('ni,ij,nj->n',rho,b['Lplus'],rho)
    return complex(np.sum(coefficients.ravel()*np.exp(-.02*energy)))


def signed_HS_single(nf: int,order: int,b: dict,s: dict,background: np.ndarray) -> dict:
    prior=np.diag(.04/b['gauge_eigenvalues']);cov=nf*s['cov']
    proposal=np.linalg.inv(np.linalg.inv(prior)+b['gauge_frame'].T@cov@b['gauge_frame'])
    e,r=eigh(proposal);nodes,hw=roots_hermitenorm(order);hw/=math.sqrt(2*math.pi)
    index=np.array(list(itertools.product(range(order),repeat=len(e))),int)
    eta=(nodes[index]@(r*np.sqrt(e)).T)@b['gauge_frame'].T
    log=nf*coul.graph_character_log(s,s,eta)+.5*np.einsum('ni,ij,nj->n',eta,cov,eta)
    weights=np.prod(hw[index],axis=1)*np.exp(log)
    baseline=np.sum(weights);signed=np.sum(weights*np.exp(1j*(eta@background)))
    ratio=signed/baseline;factor=float(np.sqrt(np.linalg.det(proposal)/np.linalg.det(prior)))
    return dict(order=order,g0=float((factor*baseline).real),g_background=float((factor*signed).real),
        normalized_ratio=[float(ratio.real),float(ratio.imag)],
        signed_cancellation_price=float(np.sum(abs(weights))/abs(signed)),
        actual_covariance_Gaussian_prediction=float(np.exp(-.5*background@(b['gauge_frame']@proposal@b['gauge_frame'].T)@background)))


def signed_background_controls(b: dict) -> dict:
    v=np.array([1.,-1.,0.]);toy_charge=coul.alpha.charge_contract(coul.alpha.Parameters())['bare_e']
    finite=[];large=[];screening=[]
    for nf in (1,4):
        s=coul.sea(.08,b,np.zeros_like(b['a']));background=toy_charge*v
        g0=coul.charge_fourier(s,s,nf,b,.02)['expectation'];g=exact_single_background(nf,b,s,background)
        output=exact_double_charge(nf,4*nf+1,b,s,background)
        finite.append(dict(NF=nf,fixed_Qreal_toy=toy_charge,g_ratio=[float((g/g0).real),float((g/g0).imag)],
            exact_fullK_output_norm2=float(output.real),fullK_output_outside_fraction=1-abs(g)**2/output.real,
            HS_controls=[signed_HS_single(nf,order,b,s,background) for order in (16,32)],
            fullK_importance_controls=[importance_output(nf,order,b,s,background) for order in (8,12)]))
    for nf in (10000,1000000):
        k=.08;beta_decimal=16*Decimal(nf)*Decimal(str(k))**4;contract=scaled_integer_charge(str(beta_decimal))
        q=int(contract['Q']);background=q*v;s=coul.sea(k,b,np.zeros_like(b['a']))
        large.append(dict(NF=nf,k=k,charge_contract=contract,background=background.tolist(),
            HS_controls=[signed_HS_single(nf,order,b,s,background) for order in (16,32,48)],
            fullK_importance_controls=[importance_output(nf,order,b,s,background) for order in (8,12,16)]))
    for index,k in enumerate((.08,.04,.02,.01)):
        nf=1000000*16**index;beta_decimal=16*Decimal(nf)*Decimal(str(k))**4
        contract=scaled_integer_charge(str(beta_decimal));q=int(contract['Q']);background=q*v
        s=coul.sea(k,b,np.zeros_like(b['a']));beta=float(beta_decimal);resistance=float(v@b['Lplus']@v)
        prediction=math.exp(-.02*q*q*resistance/(1+.04*beta))
        control=signed_HS_single(nf,32,b,s,background)
        screening.append(dict(k=k,NF=nf,charge_contract=contract,
            actual_normalized_ratio=control['normalized_ratio'],signed_cancellation_price=control['signed_cancellation_price'],
            leading_graph_Gaussian_prediction=prediction,
            actual_covariance_Gaussian_prediction=control['actual_covariance_Gaussian_prediction'],
            inferred_effective_Coul_coefficient=-math.log(control['normalized_ratio'][0])/(q*q*resistance),
            leading_effective_Coul_coefficient=.02/(1+.04*beta)))
    return dict(fixed_Qreal_toy=finite,scaled_integer_Q_cases=large,fixed_beta_k_controls=screening,
        coordinate_scope='Fixed bare_e=.3028 controls are native REALcover toys, not macroQ. Scaled integer Q(beta) follows captured Decimal charge_contract; background lives in physicaltheta charge units. Primitive macroQ and canonical e0=Q/sqrtbeta are distinct.',
        identity='g_u(b)=E_eta chi_aux(eta)^NF exp(i eta*b)=sum_rho p(rho)exp[-u(rho+b)Lplus(rho+b)], including centered phase perflavour. REAL HS integral, no noninteger compact representation claim.')


def macro_signed_screening() -> dict:
    captured=coul.alpha.charge_contract(coul.alpha.Parameters())
    with mp.workdps(210):
        beta=mp.mpf(captured['beta']);q=mp.mpf(captured['Q']);e0=q/mp.sqrt(beta);u=mp.mpf('.02')
        V=mp.mpf(2)**72;L=mp.mpf(2)**24;diameter=3*L/2;dimension=V-1
        resistance=(V-1)/(3*V);effective=u/(1+2*u*beta)
        gaussian=mp.exp(-effective*q*q*resistance)
        radius=beta**(-mp.mpf(1)/3);quartic=2*V*beta*radius**4
        contour_constant=mp.mpf(2)**20;t=mp.pi**2/(2*u);z=contour_constant*mp.exp(-t)
        alias_log=6*V*z**6/((1-z)*(1-mp.exp(-6*t)))
        alpha_energy=12*V*radius**2
        alias_relative=mp.exp(alpha_energy/(2*u))*mp.expm1(alias_log)
        h=1/mp.sqrt(12*V*(beta+1/(2*u)))
        tail_log=dimension*mp.log(mp.pi/h)+mp.mpf('.5')+alias_log-2*beta*radius**2/(mp.pi**2*diameter)
        gaussian_tail_log=mp.log(2*dimension)-radius**2*(beta+1/(2*u))/(2*diameter)
        tail_log=max(tail_log,gaussian_tail_log)
        tail_budget=quartic*mp.mpf('1e-20')
        if tail_log>=mp.log(tail_budget):raise ArithmeticError('Macro saddle tail does not meet the stated envelope')
        charge_error=mp.mpf(coul.macro_Riccati_character_bound()['uniform_macro_log_character_error'])
        delta=mp.expm1(quartic+alias_relative+charge_error)+2*tail_budget
        ratio_error=2*delta/(1-delta);lower=gaussian-ratio_error
        if lower<=0:raise ArithmeticError('Signed macro normalizer lower bound failed')
        complement_error=mp.mpf(macro_output_bound()['NF_log_retained_factor_bound'])
        output_bound=mp.expm1(complement_error)*mp.exp(4*charge_error)/(mp.cos(charge_error)**2*lower**2)
        return dict(captured_Alpha56_charge_contract=captured,Q=captured['Q'],e0=mp.nstr(e0,90),
            effective_Coul_coefficient=mp.nstr(effective,80),
            effective_active_Q_squared_coefficient=mp.nstr(effective*q*q,80),
            macro_nearest_pair_Lplus_resistance=mp.nstr(resistance,80),
            Gaussian_signed_Gram_ratio=mp.nstr(gaussian,80),normalized_signed_ratio_absolute_error_bound=mp.nstr(ratio_error,70),
            actual_signed_ratio_lower_bound=mp.nstr(lower,70),
            signed_fullSF_output_complement_bound=mp.nstr(output_bound,70),
            scaled_HS_phase='eta=xi/sqrtbeta, exp(i eta*Qv)=exp(i e0 xi*v), e0=Q/sqrtbeta. It is mild, not exp(-bareuQ²) postselection.',
            integer_alias_contract='Q is the exact captured integer; all wrapped HS sheets eta+2pi*m give phase1 forQv. FixedQreal toy is NOT retagged as macrocompactQ.',
            saddle_proof='Root-fixed angles alpha0=0. Within sup|alpha|<=a=beta^-1/3, beta sum[cosDelta-1]+beta alphaLalpha/2 is bounded by2V beta a^4. The exact HS origin prior exp(-alphaLalpha/(4u)) is kept, giving Gaussian precision(beta+1/(2u))L. Source Fourier phase has modulus1, so the same absolute integral error applies to signed numerator. Divide only after the explicit positive Gaussian ratio lower bound minus the priced error.',
            uniform_inner_quartic_log_error=mp.nstr(quartic,70),
            wrapped_prior_alias_proof='Integer height n with root0 has energy>=sum oriented level-surface areas, minimum6 on macro3D torus. Connected dual-plaquette surfaces count<=6V*C^A: degree12 DFS gives<=12^(2A), orientations overcounted by C=2^20. Thus logZ(t)<=6V(Cexp(-t))^6/[(1-Cexp(-t))(1-exp(-6t))]. Completing square bounds all nonzero HS sheets at t=pi²/(2u). Their relative inner weight<=exp(alphaLalpha/(2u))*expm1(logZ). No sheet is deleted.',
            wrapped_HS_alias_relative_error_bound=mp.nstr(alias_relative,70),
            outer_tail_proof='Outside root-angle sup<=a, geodesic path to root length<=diameter gives sumdist(Delta,2piZ)^2>=a²/diameter; 1-cos>=2dist²/pi². Wrapped Gaussian weight<=its value0 by its positive Fourier coefficients. Box |alpha|<=h gives baseline lower volume(2h)^d exp(-1/2); hence periodic tail logratio<=d log(pi/h)+1/2+logZ-2beta a²/(pi² diameter). Separately, the unwrapped Gaussian anchored-coordinate variance is resistance(site,root)/(beta+1/(2u))<=diameter/(beta+1/(2u)); its union-Chernoff tail has logbound log(2d)-a²(beta+1/(2u))/(2diameter). The larger of both logbounds is used.',
            normalized_outer_tail_log_upper=mp.nstr(tail_log,70),normalized_outer_tail_conservative_bound=mp.nstr(tail_budget,70),
            tail_contract='The computed logarithmic tail bound is strictly below log(quartic_error*1e-20). Tail is bounded, NOT replaced by0 or a fake floor.',
            nearest_resistance_proof='Foster identity sum_edges resistance=V-1 and cubic torus edge symmetry with3V edges gives vLplusv=(V-1)/(3V) for nearest-site dipole.',
            conditional_scope='PureAbelian Phi0/SU4I prepared same-C auxiliary sea, actual collectiveCoul coefficientu=.02 and capturedmacroQ. Relative static Coul Gram screening and fixed-field SFoutput, not RAWB/Khalf photon source, complete realGauss/centre, or electronmass1.')


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--capsule',required=True);args=parser.parse_args();start=time.time()
    b=coul.graph_data((3,1,1));s=coul.sea(.08,b,np.zeros_like(b['a']));finite=[]
    for nf in (1,4):
        g=coul.charge_fourier(s,s,nf,b,.02)['expectation'];q=exact_double_charge(nf,4*nf+1,b,s)
        finite.append(dict(NF=nf,exact_output_norm2=[float(q.real),float(q.imag)],vacuum_channel_amplitude=[float(g.real),float(g.imag)],
            dropped_Tminus_norm2=abs(g)**2,conditional_fullK_output_outside_bare_Slater_fraction=1-abs(g)**2/q.real,
            observed_relative_increment=q.real/abs(g)**2-1,
            importance_controls=[importance_output(nf,order,b,s) for order in (8,12)]))
    large=[dict(NF=nf,beta=16*nf*.08**4,UA=.02*16*nf*.08**4,controls=[importance_output(nf,order,b,s) for order in (8,12,16)]) for nf in (10000,1000000)]
    comparator=[]
    for order in (24,48):
        half=coul.hs_integral(s,s,1000000,b,.01,order,importance=True)
        full=coul.hs_integral(s,s,1000000,b,.02,order,importance=True)
        square=coul.hs_integral(s,s,1000000,b,.04,order,importance=True)
        comparator.append(dict(order=order,Rhalf_output_probability=half.real,Rhalf_output_norm2=full.real,
            Rhalf_conditional_outside=1-abs(half)**2/full.real,C_output_probability=full.real,C_output_norm2=square.real,
            C_conditional_outside=1-abs(full)**2/square.real,
            meaning='Same ringNF1m,k.08: Rhalf=exp[-u rhoLrho/2] is CHART factor; C=Rhalf² is MIDDLE fulltransfer. Neither intermediate output equals final SF C SF output.'))
    report=dict(direct_retained_character=direct_retained_check(b,s),exact_finiteNF=finite,large_finiteNF=large,macro_bound=macro_output_bound(),
        signed_background=signed_background_controls(b),macro_signed_screening=macro_signed_screening(),
        same_graph_intermediate_vs_final=comparator,
        kernel='K=SF C SF, C=exp[-u rho_total Lplus rho_total]. K/lambdaVac=Snormalized C Snormalized; output norm²=<Omega|C(TF/lambdaVac)C|Omega>. DoubleHS oneflavour determinant raisedNF, no NF allocation.',
        scope='Exact finiteNF fixed-field SF/Coul output and uniform pureAbelian macro complement bound. Not complete realGauss/source/mixedfield admission.')
    execution=dict(source_sha256=hashlib.sha256(SOURCE_BYTES).hexdigest(),source_bytes_at_IMPORT=SOURCE_BYTES.decode(),
        Coul_source_sha256=hashlib.sha256(COUL_BYTES).hexdigest(),Coul_source_bytes_at_IMPORT=COUL_BYTES.decode(),
        loops_source_bytes_at_IMPORT=coul.loops.SOURCE_BYTES.decode(),core_sha256=hashlib.sha256(coul.loops.CORE_BYTES).hexdigest(),
        core_source_bytes_at_IMPORT=coul.loops.CORE_BYTES.decode(),frozen_dependencies=coul.loops.IMPORT_EPOCH,
        root_charge_source_sha256=hashlib.sha256(coul.ROOT_CHARGE_BYTES).hexdigest(),
        root_charge_source_bytes_at_IMPORT=coul.ROOT_CHARGE_BYTES.decode(),
        argv=sys.argv,input_parameters=dict(shape=[3,1,1],k=.08,u=.02,finiteNF=[1,4],largeNF=[10000,1000000],GHorders=[8,12,16],
            signed_GHorders=[16,32,48],fixed_beta_screening_k=[.08,.04,.02,.01],
            fixed_beta_screening_NF=[1000000,16000000,256000000,4096000000],macroV='2^72',macroL='2^24',
            macroNF='6.25e1148',macro_k='1e-250',macro_beta='1e150',auxiliary_charge=1,
            macro_active_Q='exact captured Alpha56 integer charge_contract',fixed_Qreal_toy='captured bare_e, explicitly nonmacro'),
        python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,mpmath=mp.__version__,
        thread_environment={name:os.environ.get(name) for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},elapsed_seconds=time.time()-start)
    Path(args.capsule).write_text(json.dumps(dict(report=report,execution=execution),indent=2)+'\n')
    print(json.dumps(dict(capsule=args.capsule,report=report,elapsed_seconds=execution['elapsed_seconds']),indent=2),flush=True)


if __name__=='__main__':main()
