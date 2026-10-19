"""ALL-angle NEW Jordan 2+1+1 exclusion by scalar fourth-moment pressure.

Only the globally traceless flat neutral STATIC problem is certified. This is
not an adopted microscopic law, fullPhi quantum ground, or photon statement.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import time
from typing import Any
import zipfile

import numpy as np
import sympy as sp
from threadpoolctl import threadpool_limits

import induced_phi_nonuniform_field_car as field
import joined_phi_u1_matrix_jordan_flat_bounds as flat
import joined_phi_u1_stationary_flat_holonomy as old
import joined_phi_u1_stationary_phi_sea_envelope as sea

ROOT = Path(__file__).resolve().parent
PREFIX = 'joined_phi_u1_jordan_211_'
I = old.Interval
CAPS: dict[str, Any] = {
    'partition': [2, 1, 1], 'cone_u0_squared': '49/50',
    'cone_support_absolute_upper': '52', 'outward_bits': old.BITS,
    'occupation': 'GLOBAL top128 of256; half absolute trace ONLY an upper pressure bound',
    'scalar_directions': 256, 'angle_samples_control_only': 8,
    'seed': 2222111010, 'optimizer': 'NONE', 'root_solver': 'NONE',
}
PROOF = {
    'problem': 'NEW independent Jordan law q=256 sum_mu,nu Tr({Phi_mu,Phi_nu}^2); E/Lambda=q-S128. Arbitrary Hermitian2 t_mu I+v_mu.sigma and scalar u,w, with ONLY 2t+u+w=0. ALL compatible continuous characters, including coincidences/rank drops, are covered. Original SOURCE/CURRENT427 unaltered.',
    'global_occupation': 'Sum ALL256 bands is64 A (2t0+u0+w0)=0. Therefore GLOBAL S128<=Tr|H|/2. No separate block tracelessness, separate block particle counts, or two particles per momentum are imposed. Relaxing the trace and determinant restrictions after this global inequality only lowers the energy bound.',
    'failed_first_route': 'Covariance-only Jensen/Cauchy yields (e3+e1)/2=-1435.7401657359928; its gap to central is real and cannot be resolved by optimizer absence. A genuinely different scalar FOURTH-moment defect is used below.',
    'scalar_moments': 'For any scalar direction a of Euclidean norm1, Y=x.a=A a0+Z. The64 momentum grid at ANY continuous shift is centrally symmetric under simultaneous k_j->k_j+2, so EZ=EZ^3=0. Four-point orthogonality gives VarZ=2 sum_j(c0_j^2 a0^2+ci_j^2 a_j^2). Thus Var(Y^2)=4 A^2 a0^2 VarZ+Var(Z^2)>=8 A^2(sum c0_j^2) a0^4. No fixed angle or determinant sheet is used.',
    'scalar_low_cone': 'If a0^2<=r, EY^2<=C0 r+Csp_max(1-r), since C0>=Csp_max. Hence E|Y|<=sqrt(C0 r+Csp_max(1-r)).',
    'scalar_high_cone': 'If a0^2>=r, |Y|<=M=A+2sum|c0_j|+2 sqrt(sum ci_j^2) sqrt(1-r)<=52. On[0,M^2], sqrt(z) has second derivative<=-1/(4M^3); continuity includes z=0. Taylor about m=EY^2, then averaging, gives E|Y|<=sqrt(m)-Var(Y^2)/(8M^3)<=sqrt(C0)-A^2(sum c0_j^2)r^2/M^3. The quartic-cross/fourth-moment term is essential. No positive definiteness is assumed.',
    'scalar_radial': 'Let L=max(low-cone bound,high-cone bound), outward rational upper. Scalar q=1024||a||^4 and half-trace pressure<=32 L||a||. Its analytic infimum is -(3/4)(256 L^4)^(1/3), outward enclosed. Both scalar blocks are retained independently; their global trace relation is not silently set to zero.',
    'two_block': 'For H_mu,nu=v_mu.v_nu PSD rank<=3 and s=t.t, q2=2048 Q, Q=||H||F^2+s^2+4tHt+2sTrH exactly. Put m=Tr(CH)+tCt and D3=sum top3 eigenvalues(C)^2. Rank Cauchy gives (TrCH)^2<=D3||H||F^2; TrCH<=Cmax TrH, tCt<=Cmax s, Cmax^2<=D3 imply m^2<=D3 Q. Half-trace pressure<=64sqrt(m), so block energy>=e3/2, e3=-3(256 D3)^(1/3). This retains every scalar and every Pauli cross term and rank-deficient block.',
    'sector_lower': 'Add the two-block lower and TWO scalar lowers AFTER the global occupation inequality: E211/Lambda>=e3/2-(3/2)(256 L^4)^(1/3). Determinant torus for2+1+1 is connected; arbitrary first two character angles and theta3=2pi n-2theta1-theta2 cover every allowed character. The proof in fact covers independent angles without determinant restriction.',
    'boundary': 'Exclusion is ONLY flat neutral128 static NEW-law sector2+1+1. Other proper sectors, off-flat/fullPhi/SU4 kinetic/Gauss, all64 charged canonical species with HC/Pi, stationary physical same-ground photon pole/dispersion, commoncone/originalSOURCE/Compton/campaign remain OPEN. No covariance or root is adopted.',
}


def covariance(profile: dict[str, Any]) -> list[F]:
    return [profile['onsite']**2+2*sum(row['c0']**2 for row in profile['paths']),
            *[2*next(row['ci'] for row in profile['paths'] if row['axis']==j)**2 for j in range(3)]]


def certificate(profile: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    c = covariance(profile)
    r, support = F(CAPS['cone_u0_squared']), F(CAPS['cone_support_absolute_upper'])
    a = profile['onsite']
    c0sq = sum(row['c0']**2 for row in profile['paths'])
    cisq = sum(row['ci']**2 for row in profile['paths'])
    if not (a>0 and c[0]>=max(c[1:])):
        raise ArithmeticError('Scalar cone ordering prerequisite failed')
    support_bound = I.of(a+2*sum(abs(row['c0']) for row in profile['paths']))+2*I.of(cisq*(1-r)).sqrt()
    if support_bound.hi > I.of(support).lo:
        raise ArithmeticError('Declared scalar support cap is not outward proved')
    low = I.of(c[0]*r+max(c[1:])*(1-r)).sqrt()
    defect = a*a*c0sq*r*r/support**3
    high = I.of(c[0]).sqrt()-I.of(defect)
    pressure = F(max(low.hi, high.hi), old.SCALE)
    d3 = sum(z*z for z in sorted(c, reverse=True)[:3])
    block2 = -F(3,2)*I.of(256*d3).cube_root()
    scalar = -F(3,4)*I.of(256*pressure**4).cube_root()
    lower = block2+2*scalar
    previous = next(row for row in incoming['exact_ALLangle_sector_lower_bounds']['partitions'] if row['partition']==[2,1,1])
    central = next(row for row in incoming['exact_ALLangle_sector_lower_bounds']['partitions'] if row['partition']==[4])
    central_upper = F(central['certified_trial_upper']['upper'])
    separation = lower-I.of(central_upper)
    if separation.lo<=0:
        raise ArithmeticError('New ALL-angle certificate did not close sector gap')
    return dict(exact_C=[str(z) for z in c],r=str(r),support_cap=str(support),support_proof=support_bound.public(),
                low_cone_pressure=low.public(),high_cone_pressure=high.public(),fourth_moment_pressure_defect=str(defect),
                rational_ALLangle_scalar_pressure_upper=str(pressure),two_block_lower=block2.public(),each_scalar_lower=scalar.public(),
                ALLangle_sector_lower=lower.public(),central_trial_upper=str(central_upper),
                strict_separation_above_central_trial=separation.public(),remaining_gap_to_exclusion='0',
                improvement_over_previous=(lower-I.of(F(previous['ALL_continuous_angles_lower']['lower']))).public(),
                previous_unpaid_gap=(I.of(central_upper)-I.of(F(previous['ALL_continuous_angles_lower']['lower']))).public(),
                prior_sector_trial_upper=previous['certified_trial_upper'],sector_bracket={'lower':lower.public()['lower'],'upper':previous['certified_trial_upper']['upper']},
                certified_excluded_as_global_flat_minimum=True)


def exact_quartic_controls() -> dict[str, Any]:
    sigma = [sp.Matrix([[0,1],[1,0]]),sp.Matrix([[0,-sp.I],[sp.I,0]]),sp.diag(1,-1)]
    t = sp.Matrix([F(1,7),F(-2,9),F(3,11),F(1,13)])
    v = sp.Matrix([[F(1,5),F(2,7),F(-1,9)],[F(1,11),F(1,3),F(2,13)],[F(-1,4),F(1,8),F(1,6)],[F(1,10),F(-1,12),F(1,14)]])
    u = sp.Matrix([F(2,5),F(-1,6),F(1,7),F(-2,11)])
    w = -2*t-u
    matrices = [sp.diag(t[mu]*sp.eye(2)+sum((v[mu,j]*sigma[j] for j in range(3)),sp.zeros(2)),u[mu],w[mu]) for mu in range(4)]
    q = sum(256*sp.trace((p*z+z*p)**2) for p in matrices for z in matrices)
    h = v*v.T; s = (t.T*t)[0]
    cross = 2048*(4*(t.T*h*t)[0]+2*s*sp.trace(h))
    formula = 2048*(sp.trace(h*h)+s*s)+cross+1024*((u.T*u)[0]**2+(w.T*w)[0]**2)
    if q!=formula or cross<=0 or any(sp.trace(m)!=0 for m in matrices):
        raise ArithmeticError('Exact traceful Pauli identity or omission control failed')
    rank1 = [sp.diag(F(1,3),F(1,3),F(-1,3),F(-1,3)),sp.zeros(4),sp.zeros(4),sp.zeros(4)]
    exact_q = sum(256*sp.trace((p*z+z*p)**2) for p in rank1 for z in rank1)
    exact_gram = sum(4096*(sp.trace(p*z)/4)**2 for p in rank1 for z in rank1)
    if exact_q!=exact_gram:
        raise ArithmeticError('Admissible211 rank1 Jordan saturation failed')
    return dict(exact_traceful_quartic=str(q),exact_Pauli_formula=str(formula),wrong_omitted_cross_quartic=str(q-cross),
                exact_cross_omission_deficit=str(cross),rank1_saturation_quartic=str(exact_q),rank1_Gram_quartic=str(exact_gram),
                field_real=np.asarray([np.real(np.asarray(m,complex)) for m in matrices]).tolist(),field_imag=np.asarray([np.imag(np.asarray(m,complex)) for m in matrices]).tolist())


def controls(profile: dict[str, Any], native: dict[str, Any], cert: dict[str, Any], algebra: dict[str, Any], profiles: list[dict[str, Any]]) -> dict[str, Any]:
    rng = np.random.default_rng(CAPS['seed'])
    pressure = float(F(cert['rational_ALLangle_scalar_pressure_upper']))
    dirs = rng.normal(size=(CAPS['scalar_directions'],4));dirs/=np.linalg.norm(dirs,axis=1)[:,None]
    r = float(F(CAPS['cone_u0_squared']))
    special = np.asarray([[1,0,0,0],[0,1,0,0],[np.sqrt(r),np.sqrt(1-r),0,0],[np.sqrt(r+1e-5),0,np.sqrt(1-r-1e-5),0]])
    dirs = np.concatenate([dirs,special])
    c = np.asarray([float(z) for z in covariance(profile)])
    a = float(profile['onsite']);c0sq = sum(float(row['c0'])**2 for row in profile['paths'])
    max_pressure = 0.;max_cov_error = 0.;min_variance_slack = float('inf');wrong_even_moment_errors = []
    for angle in rng.uniform(-np.pi,np.pi,(CAPS['angle_samples_control_only'],3)):
        _,x = old.coefficients(profile,angle)
        max_cov_error=max(max_cov_error,float(np.max(abs(x.T@x/64-np.diag(c)))))
        y=x@dirs.T;z=y-a*dirs[:,0]
        variances=np.mean(y**4,axis=0)-np.mean(y*y,axis=0)**2
        lower_variances=8*a*a*c0sq*dirs[:,0]**4
        min_variance_slack=min(min_variance_slack,float(np.min(variances-lower_variances)))
        max_pressure=max(max_pressure,float(np.max(np.mean(abs(y),axis=0))))
        wrong_even_moment_errors.append(float(np.max(abs(np.mean(y*y,axis=0)-(a*a*dirs[:,0]**2+c[1:]@(dirs[:,1:]**2).T)))))
    if max_pressure>pressure+1e-10 or max_cov_error>1e-9 or min_variance_slack<-1e-7 or min(wrong_even_moment_errors)<1:
        raise ArithmeticError('Scalar angle/moment/adverse control failed')
    phi=np.asarray(algebra['field_real'])+1j*np.asarray(algebra['field_imag'])
    angles=flat.angle_values((2,1,1),np.asarray([.37,-.29,.61,-.43,.19,.53]),(1,-1,2))
    native_rows=[flat.native_control(p,native,phi,(2,1,1),angles) for p in profiles]
    native_errors=max(abs(value) for row in native_rows for value in row['errors'].values())
    if native_errors>1e-7:
        raise ArithmeticError('Independent literal native211 consumer failed')
    adverse=np.zeros((4,4,4),complex);adverse[0]=np.diag([3,-1,-1,-1])/5
    filling=flat.native_control(profile,native,adverse,(2,1,1),angles)
    raw=filling['native_oldlaw_control']
    deficits=[abs(raw['global_largest128']-raw[k]) for k in ('wrong_fixed_block_half','wrong_top2_per_momentum')]
    if min(deficits)<1:
        raise ArithmeticError('211 global occupation adverse control failed')
    return dict(controls_are_floating_not_proof=True,scalar_directions=len(dirs),sampled_ALLangle_covariance_error=max_cov_error,
                scalar_observed_pressure_max=max_pressure,certified_pressure_upper=pressure,variance_inequality_min_slack=min_variance_slack,
                wrong_scalar_cos_moment_omission_min_error=min(wrong_even_moment_errors),all_three_native_roles=native_rows,
                native_consumer_max_error=native_errors,global_occupation_adverse=filling,wrong_occupation_deficits=deficits)


def run(output: Path) -> dict[str, Any]:
    started=time.monotonic()
    caps_path=output.with_name(PREFIX+'caps_20261010.json')
    caps_path.write_text(json.dumps(CAPS,indent=2)+'\n')
    snapshots=dict(field.CAPTURED)
    for module in list(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and Path(path).suffix=='.py' and Path(path).parent==ROOT:
            snapshots.setdefault(Path(path).name,Path(path).read_bytes())
    incoming_name='joined_phi_u1_matrix_jordan_flat_actual_20261010_receipt.json'
    packet_name='joined_phi_u1_stationary_jordan_parent_packet_20261010.json'
    for name in [*sea.INPUTS,incoming_name,packet_name,caps_path.name]:
        snapshots[name]=(ROOT/name).read_bytes()
    for name,pin in sea.INPUTS.items():
        if hashlib.sha256(snapshots[name]).hexdigest()!=pin:
            raise ArithmeticError('Immutable microscopic input mismatch: '+name)
    archive=output.with_name(output.stem+'_input_bytes.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,raw in sorted(snapshots.items()):z.writestr(name,raw)
    before={name:hashlib.sha256(raw).hexdigest() for name,raw in snapshots.items()}
    incoming=json.loads(snapshots[incoming_name])
    data=field.load_mean(g_override=[1,1,1,1],mean_descriptor={'scope':'211 fresh native frame, NO mean/root solver or saved covariance adoption'})
    profiles,native=data['profiles'],data['native']
    cert=certificate(profiles[0],incoming)
    algebra=exact_quartic_controls()
    observed=controls(profiles[0],native,cert,algebra,profiles)
    after={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in snapshots}
    if before!=after:raise ArithmeticError('Source/input changed during scientific epoch')
    result=dict(kind='RIGOROUS_ALLANGLE_NEW_JORDAN_211_STATIC_EXCLUSION',proof=PROOF,caps=CAPS,
                exact_outward_certificate=cert,independent_exact_algebra_controls=algebra,independent_native_controls=observed,
                source_input_sha256=before,archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                source_inputs_unchanged=True,elapsed_seconds=time.monotonic()-started,
                sector_decision='2+1+1 excluded as flat neutral STATIC global minimizer of NEW Jordan law',
                fullPhi_ground='OPEN',same_ground_physical_photon='OPEN',originalSOURCE_CURRENT427_adoption=False)
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):result=run(args.output)
    print(json.dumps({'receipt':str(args.output),'sector_decision':result['sector_decision'],
                      'lower':result['exact_outward_certificate']['ALLangle_sector_lower'],
                      'separation':result['exact_outward_certificate']['strict_separation_above_central_trial'],
                      'elapsed_seconds':result['elapsed_seconds']},indent=2))
