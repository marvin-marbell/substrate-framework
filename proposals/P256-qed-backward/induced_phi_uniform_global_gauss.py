"""NEW global SU4 Gauss reconstruction of uniform Phi35 plus finite CAR.

Scalar/orbit restriction only: not all90 Phi/colour, local SU4, Theta/Pi,
original canonical-Y SOURCE or a true microscopic vacuum admission.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
import time
from typing import Any
ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={Path(__file__).name:Path(__file__).read_bytes()}
sys.set_int_max_str_digits(0)
import induced_phi_uniform_quantum_backreaction as uniform
SOURCE_BYTES.update(uniform.SOURCE_BYTES)
import numpy as np
UNIFORM_SOURCE_SHA='0c3d65dce2d22181073cc9bdd373a7ba7c99977f895ffab1a097e7ae070c9da7'
UNIFORM_RECEIPT_SHA='ee8f8ebd6b20db73b7a5412ca828bc9d2687fc28b1c7bfb5fb3cd27eb5843166'


def ratio(value:F)->dict[str,str]:
    uniform.iv.dps=80
    x=uniform.exact(value)
    return dict(numerator=str(value.numerator),denominator=str(value.denominator),decimal=uniform.bounds(x)['upper_decimal'])


def global_gauss_prices(g0:F,gs:F,V:int,n:int,sigma2:F)->dict[str,Any]:
    """Exact rational operator/packet bounds; no numerical eigenvalue gate."""
    if not g0>gs/2>0 or V<=0 or n<=0 or n%4 or sigma2<=0:
        raise ValueError('Actual positive native point and centre-invariant fixed CAR sector required')
    lower=V*gs/2*(1-gs/(4*g0)); cf=F(15,8)
    a2=V*g0/8; b2=F(1,64); c=F(1,8)
    C0=15*c/(8*lower)+285*a2/(16*lower**2)
    C2=285*b2/(16*lower**2)
    rot=n*n*cf/(2*lower)
    geom=8*(C0+2*C2*sigma2)
    return dict(orbit_metric_ALL_phi_lower=lower,fundamental_Casimir=cf,rotational_operator_norm_divided_chi_upper=rot,geometric_potential_absolute_pointwise_divided_G_C0=C0,geometric_potential_absolute_pointwise_divided_G_C2=C2,geometric_packet_action_norm_divided_chi_upper=geom,total_reference_packet_residual_divided_chi_upper=rot+geom,source_operator_norm_upper=F(2))


def analytic_metric_spectrum(g0:float,gs:float,V:int,phi:float)->list[float]:
    b=phi/np.sqrt(8*V); rows=[]
    for flags in __import__('itertools').product((0,1),repeat=4):
        if not any(flags):continue
        f0,f1,f2,f3=flags; anti=(f0+f1+f2)%2
        if f0 and anti:
            if f1 or f2:continue
            center=g0+b*b+gs*(1+f3); radius=np.sqrt(gs*gs+4*g0*b*b)
            rows.extend([V/2*(center-radius),V/2*(center+radius)])
        else:rows.append(V/2*(g0*f0+gs*(f1+f2+f3)+b*b*anti))
    if len(rows)!=15:raise ArithmeticError('All15 SU4 orbit sectors required')
    return sorted(rows)


def literal_orbit(native:dict[str,Any],g0:F,gs:F)->dict[str,Any]:
    primitive=uniform.supplier.static.primitive; gamma=np.asarray(primitive.SELECTED)
    actual=native['frame']/np.sqrt([native['g0']]+[native['gs']]*3)[:,None,None]
    if np.max(abs(actual-gamma))>1e-12:raise ArithmeticError('Native exact Clifford orientation changed')
    generators=primitive.clifford_cartesian_basis()/np.sqrt(8)
    T=1j*gamma[0]@gamma[1]@gamma[2]/np.sqrt(8); V=native['V']
    tangent=np.zeros((4,4,4),complex);tangent[0]=T/np.sqrt(V)
    casimir=sum(t@t for t in generators)
    rows=[];maximum=0.;radial=V/4*np.real(np.einsum('aij,aji->',tangent,tangent))
    phi_star=np.sqrt(8*V*(float(g0)-float(gs)**2/(4*float(g0))))
    for phi in (0.,1.,-1.,phi_star,-phi_star,100.):
        frame=np.sqrt([float(g0)]+[float(gs)]*3)[:,None,None]*gamma+phi*tangent
        comm=1j*(generators[:,None]@frame[None]-frame[None]@generators[:,None])
        slope=1j*(generators[:,None]@tangent[None]-tangent[None]@generators[:,None])
        H=V/4*np.real(np.einsum('amij,bmji->ab',comm,comm))
        Hp=V/4*np.real(np.einsum('amij,bmji->ab',slope,comm)+np.einsum('amij,bmji->ab',comm,slope))
        Hpp=V/2*np.real(np.einsum('amij,bmji->ab',slope,slope))
        eigen=np.linalg.eigvalsh(H);expected=analytic_metric_spectrum(float(g0),float(gs),V,phi)
        error=float(np.max(abs(eigen-expected)));maximum=max(maximum,error)
        R=np.linalg.solve(H,np.eye(15));jp=.5*np.trace(R@Hp);jpp=.5*np.trace(R@Hpp-R@Hp@R@Hp)
        cross=V/4*np.real(np.einsum('amij,mji->a',comm,tangent))
        rows.append(dict(phi=phi,ALL15_orbit_metric_eigenvalues=eigen.tolist(),analytic_spectrum_error=error,radial_angular_cross=float(np.max(abs(cross))),log_J_prime=float(jp),log_J_second=float(jpp),geometric_potential_divided_chi=float(2*jpp+jp*jp),orbit_Hprime_norm=float(np.linalg.norm(Hp,2)),orbit_Hsecond_norm=float(np.linalg.norm(Hpp,2))))
    if maximum>5e-10 or abs(radial-.125)>1e-14:raise ArithmeticError('Literal global orbit/primitive metric join failed')
    return dict(actual_gamma_native_error=float(np.max(abs(actual-gamma))),fundamental_Casimir_identity_error=float(np.max(abs(casimir-(15/8)*np.eye(4)))),radial_primitive_metric=float(radial),ALL_phi_literal_observations=rows,global_lower_attainment_phi=float(phi_star))


def annihilation(mode:int)->np.ndarray:
    result=np.zeros((256,256),complex)
    for state in range(256):
        if state&(1<<mode):result[state^(1<<mode),state]=(-1)**((state&((1<<mode)-1)).bit_count())
    return result


def literal_source()->dict[str,Any]:
    a=[annihilation(i) for i in range(8)]; E=np.zeros((256,256),complex)
    for i in range(4):E+=.5*a[i].conj().T@a[i+4]
    generators=np.asarray(uniform.supplier.static.primitive.clifford_cartesian_basis())/np.sqrt(8)
    comm=[]
    for T in generators:
        Q=np.zeros_like(E)
        for i in range(4):
            for j in range(4):Q+=T[i,j]*(a[i].conj().T@a[j]+a[i+4].conj().T@a[j+4])
        comm.append(float(np.max(abs(Q@E-E@Q))))
    nr=np.zeros_like(E);ne=np.zeros_like(E)
    for i in range(4):nr+=a[i+4].conj().T@a[i+4];ne+=a[i].conj().T@a[i]
    errors=dict(ALL15_SU4_source_commutator=max(comm),electromagnetic_co_local_source_commutator=float(np.max(abs((nr+ne)@E-E@(nr+ne)))),reference_hole_shift=float(np.max(abs(nr@E-E@nr+E))),active_addition_shift=float(np.max(abs(ne@E-E@ne-E))))
    if max(errors.values())>1e-13:raise ArithmeticError('Actual CAR compensated singlet source failed')
    return dict(physical_eight_mode_CAR_dimension=256,source='E0=1/2 sum4 c_active0alpha^dag Ralpha; actual Ref4->3, not a second neutral-heavy species',actual_source_operator_norm=float(np.linalg.norm(E,2)),source_operator_norm_upper=2,observed_errors=errors)


def run(path:Path)->dict[str,Any]:
    started=time.monotonic();raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=UNIFORM_RECEIPT_SHA or hashlib.sha256(SOURCE_BYTES[Path(uniform.__file__).name]).hexdigest()!=UNIFORM_SOURCE_SHA:raise ArithmeticError('Actual frozen0c3/ee8 uniform point required')
    cap=json.loads(raw);desc=cap['input_descriptor'];point=desc['declared_NEW_POINT_parameters'];g0=F(point['g0_exact_binary64']);gs=F(point['gs_exact_binary64']);mean=desc['actual_root_mean_input_UNCHANGED'];native=uniform.supplier.quantum.build_native(mean,4)
    if g0!=F(native['g0']) or gs!=F(native['gs']):raise ArithmeticError('Wrong oldGram point: native input must match actual0c3 mean')
    variance=cap['actual_uniform_coupled_quantum_backreaction_and_realtime_transfer']['actual_coupled_ground_Phi_variance'];sigma2=(F(variance['lower'])+F(variance['upper']))/2
    count=mean['input_descriptor']['microscopic_config']['auxiliary_count'];n=2*native['V']*(2+count)+4;prices=global_gauss_prices(g0,gs,native['V'],n,sigma2)
    orbit=literal_orbit(native,g0,gs);source=literal_source();chi=F(2)**-4096;times=[]
    for t in (F(0),F(1,4096),F(4096)):
        budget=2*abs(t)*prices['source_operator_norm_upper']**2*chi*prices['total_reference_packet_residual_divided_chi_upper']
        times.append(dict(time=str(t),raw_Heisenberg_source_ADDITIONAL_Noether_and_volume_error_upper=ratio(budget)))
    descriptor=dict(plan=desc['plan'],assigned_scientific_revision=uniform.ASSIGNED_REVISION,observed_current_plan_revision='1c4f98f4e0e84bd598b150f73bb276ac882acd511d6af2de5848b4d15a80417a',uniform_dependency_execution_plan_revision=uniform.BOUND_REVISION,input_uniform=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()),declared_actual_POINT_g0=str(g0),declared_actual_POINT_gs=str(gs),V=native['V'],bulk_filled_species=2+count,static_compensating_Reference4=4,conserved_total_finite_CAR_particles=n,centre_invariant_n_mod4=n%4,sigma2_NEW_POINT=str(sigma2),chi='2^-4096',radial_cometric_G='8chi',law='NEW scalarPhi35 plus globalSU4/Z4 orbit restriction and actual65 bulkCAR species+staticReference4; no local/full90/Theta/Pi/newground admission')
    physical=dict(actual_global_orbit=orbit,actual_compensated_CAR_source=source,exact_rational_prices={k:ratio(v) for k,v in prices.items()},additional_raw_common_phase_source_error_budgets=times,
        Gauss_reconstruction='Psi(U,phi)=U_CAR Psi_body(phi) f_sigma(phi)/sqrt(J(phi)), normalized Haar onSU4/Z4 and dphi withJ=sqrt(detHglob). Totaln=8324 divisible4, so centre acts trivially. (Q_Phi+Q_CAR+Q_Ref)Psi=0 exactly by equivariance. E0 commutes all15 combinedgenerators; Ref4 full determinant andRef3+active129 source preserve centre/charge. No Q_stat substitution: Q_CAR is actual all65 finite sea+Ref angularNoether charge.',
        exact_global_metric_proof='Hglob=V/4 sum_a Tr[T_A,Phi_a]^dag[T_B,Phi_a]. Phi0=sqrt(g0)Gamma0+phi iGamma012/sqrt(8V), Phi_i=sqrt(gs)Gamma_i. Pauli flaglabels all15 nonzero4-bit vectors. Crossonlyf0=1 andf1=f2, two2x2pairs f3=0/1: eigen V/2(g0+b²+gs(1+f3) +/-sqrt(gs²+4g0b²)). Other11 eigen V/2(g0f0+gs sumfi+b²(f0+f1+f2 mod2)). ALLphi min>=V gs/2(1-gs/(4g0)), attained by lowerpair atb²=g0-gs²/(4g0). No Gaussian-tail singularity or numerical eigenvalue-derived gate.',
        operator_and_geometry_proof='Finite CAR fixedn: sum_A Q_A²<=n² CF, CF15/8 exact fundamental identity. Hrot=chi/2 Q^T Hglob^-1 Q has bounded norm<=chi n² CF/(2L) globally. Radialmetric1/8 andradial-orbitcross0 forALLphi because[Phi0,T35]=0. Flat measure conjugation f/sqrtJ yields Vgeom=G/4 jsecond+G/8 jprime², j=logJ. Hprime norm<=sqrt(Vg0/8)+|phi|/8; Hsecond norm<=1/8; Hinv<=1/L gives |Vgeom|<=G(C0+C2 phi²). Gaussian norm action<=G(C0+2C2sigma²), using sqrt(Ephi4)=sqrt3 sigma²<2sigma². Includes ALLtails and fermions, no angular charge expectation replaced by zero.',
        budget_scope='ADDITIONAL rigorous Duhamel residual budget against SAME common harmonic/Kato reference f_sigma stationary and phi-independent sourceE(0), ||E||<=2. Bothvacuum andsource legs priced by2|t| ||E||² eta. Actual E(phi) endpoints, scalar nonlinear/Kato residual and fieldpropagation must be independently included by consumer; numbers are NOT a claimed completed fullunitary/source result.')
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),actual_global_Gauss_Noether_scalar_source_prices=physical,source_at_import={name:dict(path=str(ROOT/name),sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},input_unchanged=path.read_bytes()==raw,runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--uniform',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();result=run(args.uniform)
    if not all(result['source_unchanged_since_import'].values()) or not result['input_unchanged']:raise ArithmeticError('Actual scientific source/input changed duringexecution')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(input_sha256=result['input_sha256'],prices=result['actual_global_Gauss_Noether_scalar_source_prices']['exact_rational_prices'],times=result['actual_global_Gauss_Noether_scalar_source_prices']['additional_raw_common_phase_source_error_budgets'],elapsed_seconds=result['elapsed_seconds']),indent=2))
