#!/usr/bin/env python3
"""Actual C occupied-frame collective charge/Coulomb character, no NF array.

Auxiliaries have charge1.  Declared6a UA=beta*.02 gives longitudinal
coefficient UA/beta=.02, not the active Q² coefficient.  Slater sea is an
explicit prepared input, not deletion of other Fock occupations.  A separate
fullFock character control retains every grade.  Exact finite-N charge Fourier
sums and Gaussian HS quadratures price preparation and non-Slater output.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import mpmath as mp
import numpy as np
from scipy.linalg import eigh
from scipy.special import roots_hermitenorm,ndtri
from scipy.stats import qmc
import physical_c_bare_electric_loops as loops
core=loops.core;alpha=loops.alpha
ROOT=Path(__file__).resolve().parent;SOURCE_BYTES=Path(__file__).read_bytes()
ROOT_CHARGE_BYTES=(ROOT/'covariant_root_transfer_source_6911f028.py').read_bytes()


def graph_data(shape: tuple[int,int,int]) -> dict[str,Any]:
    b=loops.background(shape);v=len(b['targets']);incidence=np.zeros((v,3*v))
    for x in range(v):
        for j in range(3):
            incidence[x,3*x+j]-=1;incidence[b['targets'][x,j],3*x+j]+=1
    L=incidence@incidence.T;e,r=eigh(L);keep=e>1e-10
    b.update(incidence=incidence,L=L,Lplus=(r[:,keep]/e[keep])@r[:,keep].T,
        gauge_frame=r[:,keep],gauge_eigenvalues=e[keep])
    return b


def sea(k: float,b: dict[str,Any],theta: np.ndarray) -> dict[str,Any]:
    law=loops.make_law(k,b);p=law.potential(theta,b['phi'],b['links']);e,r=eigh(p['h']);n=len(e)//2
    W=r[:,n:];beta_frame=law.change.conj().T@W;bottom=beta_frame[n:]
    R=beta_frame[:n]@np.linalg.inv(bottom)
    q,u=eigh(np.eye(n)+R.conj().T@R)
    canonical=law.change@np.vstack((R,np.eye(n)))@((u/q**.5)@u.conj().T)
    P=canonical@canonical.conj().T;v=len(b['targets']);cov=np.zeros((v,v))
    for x in range(v):
        for y in range(v):
            if x!=y:cov[x,y]=-float(np.linalg.norm(P[4*x:4*x+4,4*y:4*y+4])**2)
    # Exact pure-projector identity, not cancellation of two O(1) traces.
    np.fill_diagonal(cov,-cov.sum(axis=1))
    return dict(R=R,W=canonical,P=P,cov=cov,
        mean=np.array([np.trace(P[4*x:4*x+4,4*x:4*x+4]).real-2 for x in range(v)]),
        pref=p['pref'],loglambda=p['pref']+float(np.sum(e[n:])),e=e,r=r,
        T=(r*np.exp(e))@r.conj().T,
        bottom_minimum_singular_value=float(np.linalg.svd(bottom,compute_uv=False)[-1]),
        projector_idempotence=float(np.linalg.norm(P@P-P)))


def log_oneplus(matrix: np.ndarray) -> np.ndarray:
    return np.sum(np.log1p(np.linalg.eigvals(matrix)),axis=-1)


def graph_character_log(left: dict,right: dict,eta: np.ndarray) -> np.ndarray:
    """Centered Pe exactly; det(D_eta) cancels e^-2i sum eta algebraically."""
    d=np.repeat(np.exp(1j*eta),2,axis=1)
    mixed=left['R'].conj().T[None]@(d[:,:,None]*right['R'][None])
    mixed/=d[:,:,None]
    normalizer=(log_oneplus(left['R'].conj().T@left['R'])+log_oneplus(right['R'].conj().T@right['R']))/2
    return log_oneplus(mixed)-normalizer


def direct_character(left: dict,right: dict,eta: np.ndarray) -> complex:
    Q=np.diag(np.repeat(np.exp(1j*eta),4))
    return np.exp(-2j*float(np.sum(eta)))*np.linalg.det(left['W'].conj().T@Q@right['W'])


def charge_fourier(left: dict,right: dict,nf: int,b: dict,u: float,extra: int=0) -> dict:
    """Exact degree-bounded charge sum on disclosed three-site ring."""
    order=4*nf+1+extra;angles=2*np.pi*np.arange(order)/order
    index=np.array(list(itertools.product(range(order),repeat=2)),int)
    eta=np.column_stack((angles[index[:,0]],angles[index[:,1]],np.zeros(len(index))))
    values=np.empty(len(eta),complex)
    for first in range(0,len(eta),512):
        values[first:first+512]=np.exp(nf*graph_character_log(left,right,eta[first:first+512]))
    coefficients=np.fft.fftn(values.reshape(order,order))/(order**2)
    charge=np.fft.fftfreq(order)*order;x,y=np.meshgrid(charge,charge,indexing='ij')
    rho=np.column_stack((x.ravel(),y.ravel(),(-x-y).ravel()))
    energy=np.einsum('ni,ij,nj->n',rho,b['Lplus'],rho)
    weights=coefficients.ravel()
    return dict(expectation=complex(np.sum(weights*np.exp(-u*energy))),
        half_expectation=complex(np.sum(weights*np.exp(-u*energy/2))),
        total_charge_coefficient=complex(np.sum(weights)),order=order,
        maximum_coefficient_imaginary=float(np.max(abs(weights.imag))),minimum_coefficient_real=float(np.min(weights.real)))


def hs_integral(left: dict,right: dict,nf: int,b: dict,u: float,order: int,importance: bool=False) -> complex:
    nodes,weights=roots_hermitenorm(order);weights=weights/math.sqrt(2*math.pi)
    points=np.array(list(itertools.product(range(order),repeat=len(b['gauge_eigenvalues']))),int)
    z=nodes[points];w=np.prod(weights[points],axis=1)
    prior=np.diag(2*u/b['gauge_eigenvalues']);proposal=prior;factor=1.;covariance=nf*left['cov']
    if importance:
        precision=np.linalg.inv(prior)+b['gauge_frame'].T@covariance@b['gauge_frame']
        proposal=np.linalg.inv(precision);factor=float(np.sqrt(np.linalg.det(proposal)/np.linalg.det(prior)))
    q,r=eigh(proposal);eta=z@(r*np.sqrt(q)).T@b['gauge_frame'].T
    values=np.empty(len(eta),complex)
    for first in range(0,len(eta),512):
        piece=eta[first:first+512];log=nf*graph_character_log(left,right,piece)
        if importance:log+=.5*np.einsum('ni,ij,nj->n',piece,covariance,piece)
        values[first:first+512]=np.exp(log)
    return complex(factor*np.sum(w*values))


def finite_preparation() -> dict:
    b=graph_data((3,1,1));v=len(b['targets']);profile=np.zeros((v,3));profile[:,0]=1/math.sqrt(v)
    rows=[]
    for nf in (1,4,16):
        k=.08;beta=16*nf*k**4;step=.003;u=.02
        states=[sea(k,b,a*profile/math.sqrt(beta)) for a in (-step,0,step)]
        gram=np.empty((3,3),complex);bare=np.empty_like(gram);diagonal=[]
        for j in range(3):
            for l in range(j,3):
                result=charge_fourier(states[j],states[l],nf,b,u)
                gram[j,l]=result['expectation'];gram[l,j]=result['expectation'].conjugate()
                bare[j,l]=np.exp(nf*graph_character_log(states[j],states[l],np.zeros((1,v)))[0]);bare[l,j]=bare[j,l].conjugate()
                if j==l:
                    g=result['expectation'];half=result['half_expectation']
                    diagonal.append(dict(index=j,probability=g.real,imaginary=g.imag,
                        loss=1-g.real,uncompressed_filter_change_norm2=1-2*half.real+g.real,
                        filter_output_outside_original_Slater_state_norm2=g.real-abs(half)**2,
                        conditional_outside_original_Slater_fraction=1-abs(half)**2/g.real,
                        whole_scalar_pref=states[j]['pref'],whole_loglambda=states[j]['loglambda'],
                        minimum_real_fourier_coefficient=result['minimum_coefficient_real']))
        probabilities=np.real(np.diag(gram));normalized=gram/np.sqrt(np.outer(probabilities,probabilities))
        check=charge_fourier(states[1],states[1],nf,b,u,extra=2)
        quadratures=[dict(order=order,real=hs_integral(states[1],states[1],nf,b,u,order).real,
            exact_Fourier_error=abs(hs_integral(states[1],states[1],nf,b,u,order)-gram[1,1])) for order in (12,24,48)]
        wrong_units=charge_fourier(states[1],states[1],nf,b,beta*u)
        rows.append(dict(k=k,NF=nf,beta=beta,UA=beta*u,auxiliary_Coul_coefficient=beta*u/beta,
            canonical_A_nodes=[-step,0,step],physical_theta_nodes=[-step/math.sqrt(beta),0,step/math.sqrt(beta)],
            bare_Gram=pack_complex(bare),Coul_filtered_Gram=pack_complex(gram),normalized_filtered_Gram=pack_complex(normalized),
            unfiltered_transverse_metric=-math.log(abs(bare[0,2]))/(2*step)**2,
            normalized_filtered_transverse_metric=-math.log(abs(normalized[0,2]))/(2*step)**2,
            incorrect_UA_as_Coul_coefficient=dict(coefficient=beta*u,middle_probability=wrong_units['expectation'].real,
                falsifier='Using UA itself instead of UA/beta changes the actual chargefilter; correct declared auxiliary coefficient is .02.'),
            preparation=diagonal,exact_Fourier_order_control=abs(check['expectation']-gram[1,1]),HS_quadrature=quadratures))
    return dict(shape=[3,1,1],profile=profile.tolist(),L=b['L'].tolist(),Lplus=b['Lplus'].tolist(),rows=rows,
        geometry_scope='Exact finite-NF control on periodic3site ring. Profile unitL2, divergence-free harmonic transverse, NOT a contractiblecurl/macroscopic spatial plaquette proof.')


def pack_complex(matrix: np.ndarray) -> dict:
    return dict(real=np.real(matrix).tolist(),imaginary=np.imag(matrix).tolist())


def covariance_controls() -> list[dict]:
    rows=[]
    for shape in ((4,3,1),(4,4,4),(2,2,1)):
        b=graph_data(shape)
        for k in (.08,.04,.02,.01):
            s=sea(k,b,np.zeros_like(b['a']));ratio=s['cov']/(16*k**4)
            rows.append(dict(shape=list(shape),k=k,NF_Covrho_over_beta=ratio.tolist(),graphL=b['L'].tolist(),
                trace_ratio=float(np.trace(ratio)/np.trace(b['L'])),operator_norm_remainder=float(np.linalg.norm(ratio-b['L'],2)),
                maximum_mean_charge=float(np.max(abs(s['mean']))),projector_idempotence=s['projector_idempotence'],
                bottom_minimum_singular_value=s['bottom_minimum_singular_value'],
                interpretation='Short even length2 cancels F at theta0; NOT macro Laplacian' if shape==(2,2,1) else 'Actual positiveC transfer occupiedframe, not nearestDirac projector'))
    return rows


def fullFock_operator_control() -> dict:
    b=graph_data((3,1,1));s=sea(.08,b,np.zeros_like(b['a']));v=3;u=.02;rows=[]
    baseline=s['pref']+float(np.sum(np.logaddexp(0,s['e'])))
    for nf in (1,2):
        order=4*nf+1;index=np.array(list(itertools.product(range(order),repeat=v)),int)
        eta=2*np.pi*index/order;values=[]
        sign0,log0=np.linalg.slogdet(np.eye(4*v)+s['T'])
        for a in eta:
            Q=np.diag(np.repeat(np.exp(1j*a),4));sign,log=np.linalg.slogdet(np.eye(4*v)+Q@s['T'])
            centered=np.exp(-2j*np.sum(a))*sign/sign0*np.exp(log-log0)
            values.append(centered**nf)
        coefficients=np.fft.fftn(np.array(values).reshape((order,)*v))/(order**v)
        charge=np.fft.fftfreq(order)*order;rho=np.array(list(itertools.product(charge,repeat=v)))
        energy=np.einsum('ni,ij,nj->n',rho,b['Lplus'],rho)
        probability=np.sum(coefficients.ravel()*np.exp(-u*energy))
        hs=[]
        for horder in (12,24):
            nodes,hw=roots_hermitenorm(horder);hw/=math.sqrt(2*math.pi)
            grid=np.array(list(itertools.product(range(horder),repeat=2)),int)
            phases=(nodes[grid]*np.sqrt(2*u/b['gauge_eigenvalues']))@b['gauge_frame'].T
            terms=[]
            for a in phases:
                Q=np.diag(np.repeat(np.exp(1j*a),4));sign,log=np.linalg.slogdet(np.eye(4*v)+Q@s['T'])
                terms.append((np.exp(-2j*np.sum(a))*sign/sign0*np.exp(log-log0))**nf)
            value=np.sum(np.prod(hw[grid],axis=1)*np.array(terms))
            hs.append(dict(order=horder,value=[float(value.real),float(value.imag)],exact_Fourier_error=abs(value-probability)))
        rows.append(dict(NF=nf,order=order,normalized_Coul_fullFock_trace=[float(probability.real),float(probability.imag)],
            unfiltered_whole_scalar_logtrace=nf*baseline,occupation0_probability_one_species=float(np.exp(s['pref']-baseline)),
            global_charge_not_projected=True,Gaussian_HS_control=hs))
    return dict(rows=rows,identity='For every occupation charge vector rho, E_eta exp(i eta*rho)=exp(-u rho Lplus rho), eta covariance2u Lplus. Thus HS operator identity holds in ALL occupation sectors, with centered e^-2i sumeta perflavour; occupation0 explicitly retained.',
        distinction='This Nt1 fullFock trace is an operator identity control. Its thermal covariance is NOT the prepared positiveTF Slater covariance used for the sea bundle; no sea truncation of the primitive is made.')


def quadrature_repair() -> list[dict]:
    b=graph_data((3,1,1));s=sea(.08,b,np.zeros_like(b['a']));rows=[]
    for nf in (10000,1000000):
        beta=16*nf*.08**4;plain=[];repaired=[]
        for order in (12,24,48):
            plain.append(dict(order=order,value=hs_integral(s,s,nf,b,.02,order).real))
            repaired.append(dict(order=order,value=hs_integral(s,s,nf,b,.02,order,True).real))
        rows.append(dict(NF=nf,beta=beta,UA=.02*beta,plain_HS=plain,actual_covariance_importance_HS=repaired,
            construction='Proposal precision=priorHSprecision+NF actual Slater Covrho; exact Gaussian changeofmeasure, no physical kernel change, clipping or floor. Finite quadrature convergence is observed, not a macro alltail assertion.'))
    return rows


def macro_Riccati_character_bound() -> dict:
    with mp.workdps(160):
        k=mp.mpf('1e-250');nf=mp.mpf('6.25e1148');v=mp.mpf(2)**72
        b=1/(2*k);bmin=b-3;gap=bmin-1/bmin;E=3/bmin
        rbound=2*E/gap;r0=3/b**2
        eps=9/(b*bmin)*(1/bmin+1/b)+(10*rbound+3*rbound**2)/bmin**2
        delta=2*v*nf*(4*r0*eps+2*eps**2+rbound**4/(1-rbound**2))
        covariance_density=(16*r0*(eps+2*rbound**2+rbound**3)+8*(eps+2*rbound**2+rbound**3)**2)/(16*k**4)
        return dict(assumptions='Pure Abelian Phi0/SU4I, all macro spatial extents>=3, auxiliary charge1, actual C=positiveT4 operator identity already captured. No noncentral colour or interacting Phi bound.',
            exact_charge_graph_identity='chi_jk(eta)=det(I+Deta^-1 Rj†Deta Rk)/sqrt(det(I+Rj†Rj)det(I+Rk†Rk)). It includes centered e^-2i sumeta algebraically, not omitted background.',
            occupied_graph_bound='Bmin=b-3,b=1/(2k), ||F||<=3. Top-subspace Sylvester gives ||X||<=||E||/gap with gap=Bmin-1/Bmin, ||E||<=3/Bmin. If this ratio<=1/2, ||R||<=2||E||/gap. No arbitrary fullrank/Gram assumption.',
            expansion='R0=-b^-2 F. Riccati gives ||R-R0||<=9/(b Bmin)*(1/Bmin+1/b)+(10*rbound+3*rbound²)/Bmin².',
            characteristic_bound='Uniform all eta and arbitrary mixed Abelian spatial phases: |NF logchi_jk-beta sumedges[cos(eta_x-eta_y+delta_theta_xy)-1]| <=2V NF[4*r0*eps+2eps²+rbound^4/(1-rbound²)]. beta=16NF k4; genuine macro has no length2/selfedge degeneracy.',
            uniform_macro_log_character_error=mp.nstr(delta,70),
            NF_Covrho_over_beta_leading='graphL, derived from R0 spin blocks: each oriented neighbor ||R0xy||F²=8k4, paired edges16k4. No nearestDirac assumption.',
            covariance_quadratic_remainder_per_site_for_unit_sup_charge_direction=mp.nstr(covariance_density,70),
            positive_HS_relative_Gram_bound='Leading chi_jk=exp(beta sum cosminus1) is nonnegative. With positive normalized HS Gaussian, multiplicative character remainder gives relative Gram error<=exp(delta)-1 despite tiny preparation norm. This does NOT normalize an arbitrary signed/global projected source.',
            exact_leading_charge_kernel='Glead_jk=E_eta exp[beta sumedges(cos(eta_x-eta_y+delta_theta_xy)-1)], eta covariance2u Lplus, u=.02. Entire periodic HS integral retained; Gaussian saddle alone is not the exact finite-beta expression.',
            coexact_small_phase_prediction='For canonical deltaA=sqrtbeta*deltatheta with incidence*deltaA=0, local Gaussian saddle gives normalized Gjk≈exp(-||deltaA||²/2). Gradient metric is1/[2(1+2u beta)]. This is a conditional asymptotic prediction; aliases, volumeuniform saddle and actual compressed core consumer need separate control.',
            prep_cost_prediction='Gaussian charge approximation gives Gram≈(1+2u beta)^(-(V-1)/2); discrete charge/periodic HS aliases require their theta-function factor, so this expression is NOT asserted as exact macro preparation cost.',
            uncompressed_warning='Coulomb chargefilter produces a non-Slater output; its norm and complement cannot be inferred from bare sea overlap.',
            scope='Actual finiteNF Coul/HS character and preparation controls plus uniform Riccati charge-character bound. Not complete realGauss/source/current or physicalmass1 admission.')


def coexact_HS_kernel() -> dict:
    b=graph_data((4,3,1));profile=b['a']/np.linalg.norm(b['a']);u=.02;k=.04;step=.03;rows=[]
    for nf in (10000,1000000):
        beta=16*nf*k**4;states=[sea(k,b,a*profile/math.sqrt(beta)) for a in (-step,0,step)]
        covariance=nf*states[1]['cov'];results=[]
        for bits in (10,12,14):
            normal=ndtri(qmc.Sobol(len(b['gauge_eigenvalues']),scramble=True,rng=np.random.default_rng(222940)).random_base2(bits))
            def integral_matrix(coefficient: float) -> np.ndarray:
                prior=np.diag(2*coefficient/b['gauge_eigenvalues'])
                proposal=np.linalg.inv(np.linalg.inv(prior)+b['gauge_frame'].T@covariance@b['gauge_frame'])
                e,r=eigh(proposal);eta=normal@(r*np.sqrt(e)).T@b['gauge_frame'].T
                factor=float(np.sqrt(np.linalg.det(proposal)/np.linalg.det(prior)))
                correction=.5*np.einsum('ni,ij,nj->n',eta,covariance,eta);gram=np.empty((3,3),complex)
                for j in range(3):
                    for l in range(j,3):
                        total=0j
                        for first in range(0,len(eta),512):
                            value=nf*graph_character_log(states[j],states[l],eta[first:first+512])+correction[first:first+512]
                            total+=np.sum(np.exp(value))
                        gram[j,l]=factor*total/len(eta);gram[l,j]=gram[j,l].conjugate()
                return gram
            gram=integral_matrix(u);half=integral_matrix(u/2)
            probabilities=np.real(np.diag(gram));normalized=gram/np.sqrt(np.outer(probabilities,probabilities))
            bare=np.exp(nf*graph_character_log(states[0],states[2],np.zeros((1,len(profile))))[0])
            results.append(dict(Sobol_power=bits,points=2**bits,Coul_Gram=pack_complex(gram),normalized_Gram=pack_complex(normalized),
                middle_preparation_probability=float(probabilities[1]),middle_half_Coul_expectation=float(half[1,1].real),
                middle_conditional_output_outside_bare_Slater_fraction=1-abs(half[1,1])**2/probabilities[1],
                unfiltered_coexact_metric=-math.log(abs(bare))/(2*step)**2,
                normalized_filtered_coexact_metric=-math.log(abs(normalized[0,2]))/(2*step)**2))
        rows.append(dict(NF=nf,k=k,beta=beta,UA=.02*beta,auxiliary_Coul_coefficient=.02,canonical_A_nodes=[-step,0,step],controls=results))
    return dict(shape=[4,3,1],unitL2_coexact_profile=profile.tolist(),
        incidence_profile_norm=float(np.linalg.norm(b['incidence']@profile.ravel())),rows=rows,
        interpretation='Actual mixed canonical-A/Coul HS character kernel on coexact curl, using same-C positiveTF frames and charge1. Importance Gaussian is exact changeofmeasure; Sobol integration is finite precision/convergence evidence, not a certified macro quadrature or fullGauss/source normalization.')


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--capsule',required=True);args=parser.parse_args();start=time.time()
    finite=finite_preparation();covariance=covariance_controls();full=fullFock_operator_control();repair=quadrature_repair();coexact=coexact_HS_kernel()
    b=graph_data((3,1,1));s=sea(.08,b,np.zeros_like(b['a']));eta=np.array([.17,-.23,.31])
    direct=direct_character(s,s,eta);stable=np.exp(graph_character_log(s,s,eta[None])[0])
    coordinate=dict(auxiliary_charge=1,active_Q_is_separate=True,declared_family=core.full_family_definition(),
        Utheta='.02',UA='beta*.02',auxiliary_longitudinal_coefficient='UA/beta=.02',
        forbidden_substitution='activeQ²*.02 is not auxiliary Coulomb coefficient',
        root_charge_source_sha256=hashlib.sha256(ROOT_CHARGE_BYTES).hexdigest(),root_charge_source_bytes_at_IMPORT=ROOT_CHARGE_BYTES.decode())
    report=dict(coordinate_contract=coordinate,finite_preparation=finite,actual_covariance_controls=covariance,
        fullFock_HS_operator_control=full,quadrature_material_repair=repair,actual_coexact_Coul_kernel=coexact,macro_Riccati_bound=macro_Riccati_character_bound(),
        centered_Pe_direct_vs_graph_character_error=abs(direct-stable),
        omitted_background_character=[float((direct*np.exp(2j*np.sum(eta))).real),float((direct*np.exp(2j*np.sum(eta))).imag)],
        debts=['Actual coexact mixed-field HS kernel on full macrograph beyond the uniform charge-character bound.','Complete realGauss/global/coherent centre and ORIGINALfixedbody preparation/filter normalization.','Phi/noncentralSU4, actual field/output/source/current/pole join; no physicalmass1 claim.'])
    execution=dict(source_sha256=hashlib.sha256(SOURCE_BYTES).hexdigest(),source_bytes_at_IMPORT=SOURCE_BYTES.decode(),
        loops_source_sha256=hashlib.sha256(loops.SOURCE_BYTES).hexdigest(),loops_source_bytes_at_IMPORT=loops.SOURCE_BYTES.decode(),
        core_sha256=hashlib.sha256(loops.CORE_BYTES).hexdigest(),core_source_bytes_at_IMPORT=loops.CORE_BYTES.decode(),frozen_dependencies=loops.IMPORT_EPOCH,
        argv=sys.argv,python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,mpmath=mp.__version__,
        input_parameters=dict(auxiliary_charge=1,Utheta=.02,finite_NF=[1,4,16],finite_k=.08,
            covariance_k=[.08,.04,.02,.01],coexact_NF=[10000,1000000],coexact_k=.04,
            coexact_canonical_A_nodes=[-.03,0,.03],Sobol_rng_seed=222940,Sobol_powers=[10,12,14]),
        thread_environment={name:os.environ.get(name) for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},elapsed_seconds=time.time()-start)
    Path(args.capsule).write_text(json.dumps(dict(report=report,execution=execution),indent=2)+'\n')
    print(json.dumps(dict(capsule=args.capsule,elapsed_seconds=execution['elapsed_seconds'],Pe_error=abs(direct-stable),
        preparation=[dict(NF=r['NF'],diagonal=r['preparation'],bare_metric=r['unfiltered_transverse_metric'],filtered_metric=r['normalized_filtered_transverse_metric'],HS=r['HS_quadrature']) for r in finite['rows']],
        covariance=[{key:r[key] for key in ('shape','k','trace_ratio','operator_norm_remainder')} for r in covariance],repair=repair,coexact=coexact,
        macro_bound=report['macro_Riccati_bound']),indent=2),flush=True)


if __name__=='__main__':main()
