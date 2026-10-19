#!/usr/bin/env python3
"""True-Weyl all-coexact/free-harmonic graphs of the prepared MinSet state.

No Lambda-growing photon energy window is used for physical moments. The
ordinary Dirichlet law/current acts on its actual regular full-field tube.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from fractions import Fraction as F
from pathlib import Path
from typing import Any, cast
import sys
if __name__=='__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key]='1'
import numpy as np
from numpy.polynomial.hermite import hermgauss
from improved_phi_band import C,NSTAR,UE
from noncompact_quantitative_normal_budget import ceil_log2
from prepared_neutral_local_well import exact_capsule
from prepared_neutral_coherent_bank import IDENTITY


def rational(value:Any) -> F:
    if isinstance(value,dict):return F(int(value['exact_numerator_hex'],16),int(value['exact_denominator_hex'],16))
    return F(value)


def stirling(order:int) -> list[int]:
    row=[1]
    for n in range(order):
        new=[0]*(len(row)+1)
        for k,c in enumerate(row):new[k]+=k*c;new[k+1]+=c
        row=new
    return row


def touchard(order:int,x:F) -> F:
    return sum((F(c)*x**k for k,c in enumerate(stirling(order))),F())


def positive_sqrt(value:F) -> F:
    if value==0:return F()
    p=(ceil_log2(value)+1)//2
    return F(2)**p


def falling(value:int,order:int) -> int:
    return math.prod(value-j for j in range(order))


def creator_polynomial(order:int) -> list[int]:
    # a_g (N)_p a_g† <= (N+1)(N+1)_p. Exact falling-basis conversion.
    values=[(n+1)*falling(n+1,order) for n in range(order+2)]
    out=[]
    for q in range(order+2):
        delta=sum((-1)**(q-j)*math.comb(q,j)*values[j] for j in range(q+1))
        if delta%math.factorial(q):raise AssertionError('Nonintegral creator polynomial')
        out.append(delta//math.factorial(q))
    if min(out)<0:raise AssertionError('Creator majorant loses positivity')
    return out


def coherent_harmonic_initial_graphs() -> dict[str,Any]:
    """Initial ONE-filter graphs, not a driven Noether estimate.

    At every actual Clifford mean h anticommutes with chirality, while each
    original quaternion mark commutes with it. Thus its negative-sea trace is
    half the Hilbert-Schmidt trace for ANY complex profile and harmonic theta.
    The fixed R4 is a subprojector of the positive cluster. This gives the
    anti bank a pointwise Gram between the base Gram and twice that Gram.
    Applying this also to d_j/N times the profile, |d_j/N|<=1, bounds the
    harmonic derivative without replacing the signed tree or raw quartet.
    Direct fn quasimodes on theta*tau and ptheta*tau, including their actual
    harmonic/light commutators, pay the weighted graph remainder. An
    unbounded theta or ptheta is never applied to a norm-only error.
    """
    return dict(theta_coordinate='theta=N mean(A); ptheta=-i d/dtheta',
        physical_harmonic_H='UE/N ptheta^2',
        profile_scope='ANY original complex eta; SOURCE19 and prepared SOFT',
        chirality_identity='Tr(M^dagger Pnegative M)=||M||HS^2/2',
        fixed_R4_identity='0<=R4<=Ppositive; anti Gram in [base,2 base]',
        RAW_gaussian_theta_moment_squared=F(3,2),
        normalized_SOURCE_shadow_theta_norm_squared_upper=F(3),
        normalized_SOURCE_shadow_ptheta_bound='sqrt(2)*(sqrt(3/2)+sqrt(3))<5',
        actual_ONE_fn_theta_norm_upper=F(2),
        actual_ONE_fn_ptheta_norm_upper=F(8),
        graph_transfer='direct weighted fn/fq quasimodes on theta*tau and ptheta*tau; SOURCE theta derivative retained; inverse-neutral32 and SOURCE relative1/4 already bound',
        no_driven_graph_claim=True)


def spectator_ledger(sector:dict[str,Any],time:F=F(6)) -> dict[str,Any]:
    data=sector['ledger'];CL=rational(data['CL_upper']);common=data['actual_common_coefficient_log2_upper']
    if data['identity']!=IDENTITY:
        raise ValueError('Actual same-fixed-Gauss coherent-bank ONE-global-filter successor required')
    neutral_inverse=rational(data['neutral_inverse_norm_upper'])
    V=NSTAR**3
    # k in wrapped integer cube, omega(k)>=4c|k|/N. Shell count<=24j²+2.
    # The diagonal transverse projector is<=1 (not an extra two/mode count).
    link_covariance=4*UE/C
    path_covariance=16*UE/C # actual improved paths have length<=2
    d=positive_sqrt(path_covariance);v=F(4,NSTAR);K=CL
    # All3 free harmonic momenta, actual sigma_theta=1 Gaussian on R³.
    b=[F(1)]
    for r in range(1,7):b.append(positive_sqrt(F(math.prod(range(1,2*r+2,2)),2**r)))
    epsilon=UE/(V*2**40);suppression_bits=2*(ceil_log2(1/epsilon)+4096)
    error=F(1,2**(suppression_bits//2))
    # Fourier fn at s/L. Every positive Touchard term has L^-j, j>=1.
    # For p+r<=6, sum_j S(p+r,j)K^j <=6!*(1+K)^6.
    coefficient_log=2048+ceil_log2(F(math.factorial(6))*(1+K)**6*sum(b))+8
    selector=(8*(coefficient_log+suppression_bits//2+8)+2)//3
    if selector>=2**315:raise AssertionError('Actual neutral window fails Weyl/harmonic graph selector')
    initial={}
    for p in range(7):
        for r in range(7-p):
            initial[p,r]=(F(1) if (p,r)==(0,0) else 2*(b[r]+error) if p==0 else neutral_inverse*d**p*error)
    def evolve(initial_values,p,r):
        return sum((F(math.comb(p,i)*math.comb(r,j))*initial_values[i,j]
                    *d**(p-i)*v**(r-j)*touchard(p+r-i-j,K*abs(time))
                    for i in range(p+1) for j in range(r+1)),F())
    evolved={(p,r):evolve(initial,p,r) for p,r in initial}
    CB2=F(2)**(2*data['full_source_jets_raw_log2_upper'][0]);Z2=F(1,16);ratio=CB2/Z2
    neutral_factorial={p:initial[p,0]**2 for p in range(1,7)}
    creator_factorial={}
    for p in range(1,6):
        coefficients=creator_polynomial(p)
        # Constant baseline uses ||B Xi||²/Z_creator²<=1, NOT CB²/Z².
        creator_factorial[p]=F(coefficients[0])+ratio*sum((F(c)*neutral_factorial[q] for q,c in enumerate(coefficients) if q),F())
    # The physical creation-state mixed graph is obtained BEFORE any norm
    # transfer: Leibniz on B plus the exact positive number polynomial.
    creator_mixed={}
    for p in range(5):
        for r in range(5-p):
            bound=F()
            for j in range(r+1):
                poly=creator_polynomial(p)
                weighted=sum((F(c)*initial[q,j]**2 for q,c in enumerate(poly)),F())
                bound+=math.comb(r,j)*4**(r-j)*positive_sqrt(weighted)
            creator_mixed[p,r]=positive_sqrt(ratio)*bound
    # Source19 weighted Gaussian trace (separately bound by Guard) lets the
    # source theta and p_theta graphs keep their actual O1 scale: fn/fq
    # intertwining is applied ALSO on theta*tau, p_theta*tau and dtheta B.
    # Unbounded a† is controlled by true-Weyl current pullthrough, notnorm1.
    alpha=UE/NSTAR;Ktheta=CL*v
    gamma_theta_initial=F(2);gamma_Pi_initial=F(8)
    gamma_theta_time=gamma_theta_initial+2*alpha*(gamma_Pi_initial*abs(time)+Ktheta*time*time/2)
    return dict(identity='PNFF-MinSet-true-Weyl-allmode-and-free-harmonic-graphs-v1',
        actual_prepared_neutral_identity=data['identity'],
        actual_neutral_norm_lower=data['neutral_preparation_norm_lower'],
        physical_SOURCE_relative_norm_lower=data['source_relative_to_actual_Xi_norm_lower'],
        unnormalized_SOURCE_fn_norm_lower=data['source_total_unfiltered_norm_lower'],
        initial_photon_cloud_squared_extra_bits=data['neutral_fn_cloud_extra_squared_log2_bits'],
        fixed_reference_charge='Nref-4 BOTH channels; total2V+4 before/after SOURCE',
        coherent_bank_probabilities=data['actual_prepared_bank_probability_interval'],
        legal_energy_origin=data['legal_Hamiltonian_energy_origin'],
        initial_actual_EM_energy_upper=sector['initial_energy']['EM_initial_energy_upper'],
        initial_actual_H_R_energy_upper=sector['initial_energy']['absolute_initial_energy_upper'],
        ordinary_true_OUT_weighted_graph=sector['ordinary_weighted_OUT'],
        actual_neutral_window_common_log2=common,physical_time=time,
        state='ONE neutral gamma fn tau, sigma_theta=1, raw SOURCE/globaltruecreator normalization; allcoexact/allharmonic/Root retained',
        canonical_law='UE Pi²+(K/2)curl², c²=2UEK; free Hharm=(UE/N)p_theta²; no oscillator confinement',
        single_link_allmode_displacement_squared_upper=link_covariance,path_allmode_displacement_squared_upper=path_covariance,
        allmode_IR_proof='omega>=4c|r|/N; maxshell j count<=24j²+2; (1/V)sum1/omega<4/c; transverse projector diagonal<=1; zero mode excluded and harmonic retained separately',
        current_Weyl_norm_upper=K,displacement_norm_upper=d,harmonic_word_gradient_upper=v,
        actual_commutator='[a_m,H_D]=omega_m a_m+sum_e d_e,m R_e W_e; ||D_a^q D_p^h Hlight||<=CL*d^q*v^h onregularH_D tube',
        photon_harmonic_joint_recurrence='Jpr_prime<=CL sum_(q,h)!=0 bin(p,q)bin(r,h)d^q v^h J_(p-q,r-h); Jpr=||a_tensor_p ptheta_tensor_r Upsi||',
        true_vacuum_trial_solution='Jpr<=d^p sum_j bin(r,j)b_j v^(r-j) Touchard_(p+r-j)(CL|t|); no freevac invariance for fullH',
        Fourier_Cf_j_upper=2**2048,Fourier_orders=6,filter_moment_selector_log2_Lambda=selector,
        actual_window_lower='L>=Lambda^(3/8) from selectedphysicalL=4*2^1787662Lambda^(3/8), not inverse upperbound inference',
        unchanged_selected_log2_Lambda_lower=2**315,selected_tower_pays_actual_spectator_graphs=True,
        initial_graph_suppression_bits=suppression_bits,initial_neutral_factorial_moments=neutral_factorial,
        initial_joint_graph_upper={f'{p},{r}':x for (p,r),x in initial.items()},
        ordinary_time_joint_graph_upper={f'{p},{r}':x for (p,r),x in evolved.items()},
        true_creator_polynomial={p:creator_polynomial(p) for p in range(1,6)},
        actual_initial_creator_factorial_moments=creator_factorial,
        creator_number_baseline_NOT_multiplied_by_CB=True,
        actual_initial_creator_mixed_graph_upper={f'{p},{r}':x for (p,r),x in creator_mixed.items()},
        harmonic_initial_Gamma_theta_norm_upper=gamma_theta_initial,harmonic_initial_Gamma_Pi_norm_upper=gamma_Pi_initial,
        harmonic_ordinary_Gamma_theta_norm_upper=gamma_theta_time,harmonic_ordinary_Gamma_Pi_norm_upper=gamma_Pi_initial+Ktheta*abs(time),
        harmonic_force_upper=Ktheta,harmonic_coordinate_drift=2*alpha*(gamma_Pi_initial*abs(time)+Ktheta*time*time/2),
        source_macro_theta_phase='exp(-i theta.(Birth-Ref)/N) kept EXACT; mean O1 dipole shift, notcalledsmall or erased; onlyactualmatched common phases maycancel',
        coarse_Lambda_growing_N_window_NOT_used=True,modecount_NOT_inserted=True,
        domain_scope='ORDINARY exact regular H_D; wholeH probability transfer uses alreadybound SOURCE/creator/union errors, not unbounded N timesnormerror',
        coherent_harmonic_initial_graphs=coherent_harmonic_initial_graphs(),
        not_paid=['driven physical Noether photon growth','Born mean/current calibration and leading probability','physical SO3 closure','fundamental quantum law/parameter debt'])


def native_Weyl_graphs() -> dict[str,Any]:
    x,w=hermgauss(100);measure=w/math.sqrt(math.pi);kappa=.41
    phase=np.exp(1j*kappa*x);rows=[]
    for n in (0,1,7):
        coefficients=np.zeros(n+1);coefficients[n]=1/math.sqrt(2**n*math.factorial(n))
        polynomial=np.polynomial.hermite.hermval(x,coefficients)
        derivative=np.polynomial.hermite.hermval(x,np.polynomial.hermite.hermder(coefficients))
        # a=(x+d/dx)/sqrt2 on polynomial*exp(-x²/2).
        aW=phase*(derivative+1j*kappa*polynomial)/math.sqrt(2)
        Wa=phase*derivative/math.sqrt(2)
        observed=float(np.sum(measure*abs(aW-Wa)**2));expected=kappa*kappa/2
        error=float(np.sqrt(np.sum(measure*abs((aW-Wa)-1j*kappa*phase*polynomial/math.sqrt(2))**2)))
        if abs(observed-expected)>1e-12 or error>1e-10:raise AssertionError('Actual untruncated Weyl current identity failed')
        rows.append(dict(photon_number=n,commutator_squared_norm=observed,operator_defect=error,creator_norm=math.sqrt(n+1)))
    cov=[]
    for N in (3,5,8):
        total=0.
        for q in np.ndindex((N,N,N)):
            if q==(0,0,0):continue
            gradient=np.exp(2j*np.pi*np.array(q)/N)-1;nu=float(np.vdot(gradient,gradient).real)
            transverse=1-abs(gradient[2])**2/nu
            total+=transverse/math.sqrt(nu)/N**3
        if total>4:raise AssertionError('Actual allmode infrared displacement bound failed')
        cov.append(dict(size=N,coexact_modes=2*(N**3-1),single_link_covariance_in_UE_over_c=total,upper=4))
    return dict(actual_Hermite_Weyl_rows=rows,actual_allmode_covariance=cov,
        scope='untruncated oscillator CCR/Weyl and exact finiteperiodic allmodecovariance; notsharedfieldnativeT6 or fullphysicalNoether')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--global-filter-stdout',type=Path,required=True);a=p.parse_args()
    sector=json.loads(a.global_filter_stdout.read_bytes())
    root=Path(__file__).resolve().parent
    hashes={Path(cast(str,m.__file__)).name:hashlib.sha256(Path(cast(str,m.__file__)).read_bytes()).hexdigest()
        for m in tuple(sys.modules.values()) if getattr(m,'__file__',None)
        and Path(cast(str,m.__file__)).parent==root and Path(cast(str,m.__file__)).suffix=='.py'}
    result:dict[str,Any]=dict(inputs={str(a.global_filter_stdout):hashlib.sha256(a.global_filter_stdout.read_bytes()).hexdigest()},source_sha256=hashes,
                executed_global_filter_source_sha256=sector['loaded_source_sha256'],
                legal=spectator_ledger(sector),native=native_Weyl_graphs())
    result['source_unchanged']=all(hashlib.sha256((root/name).read_bytes()).hexdigest()==h for name,h in hashes.items())
    print(json.dumps(exact_capsule(result),indent=2,sort_keys=True))

if __name__=='__main__':main()
