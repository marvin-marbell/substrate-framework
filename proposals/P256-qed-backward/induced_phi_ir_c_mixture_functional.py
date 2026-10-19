#!/usr/bin/env python3
"""Fixed-density C-paired mixture before log; actual native N6 Slater control.

The native jets below are bare fermion jets, never full quantum source jets.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
from scipy.linalg import expm
from induced_phi_backreaction import PhiConfig, models, scalar_paths, IMPORT_BYTES as BASE_BYTES
from induced_phi_ir_charge_symmetry import (B, D, mean_frames, local_charge_unitary,
                                           quadratic, ref_gauss_control)

ROOT = Path(__file__).resolve().parent
SOURCE_BYTES = {**BASE_BYTES, **{name: (ROOT/name).read_bytes() for name in
    ('induced_phi_centre_holonomy.py', 'induced_phi_ir_charge_symmetry.py', Path(__file__).name)}}
SOURCE_AT_IMPORT = {name: dict(sha256=hashlib.sha256(raw).hexdigest(), bytes_utf8=len(raw))
                    for name, raw in SOURCE_BYTES.items()}
ORDER = 4
DIRECTION = np.array([1., .37, -.23])


def product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    result = np.zeros_like(a)
    for n in range(ORDER+1):
        for j in range(n+1):
            result[n] += a[j]@b[n-j]
    return result


def exponential_jet(h: np.ndarray, clock: float, terms: int = 24) -> tuple[np.ndarray, dict[str, Any]]:
    """Noncommutative polynomial exponential, scaling/Taylor/squaring.

    No diagonalization of H(A), commuting-derivative assumption, or finite differences.
    The degree4 quotient algebra preserves every word contributing to these jets.
    """
    radius = .01
    cap = clock*sum(float(np.max(np.linalg.norm(h[j], axis=(-2,-1))))*radius**j
                    for j in range(ORDER+1))
    scale = max(0, math.ceil(math.log2(max(cap/.25, 1.))))
    x = -1j*clock*h/2**scale
    identity = np.zeros_like(x); identity[0] = np.eye(4)
    term = identity.copy(); result = identity.copy()
    for k in range(1, terms+1):
        term = product(term, x)/k
        result += term
    for _ in range(scale):
        result = product(result, result)
    m = cap/2**scale
    tail = math.exp(m)*m**(terms+1)/math.factorial(terms+1)
    # Telescoping product bound for repeated squaring on the Cauchy circle.
    error = 2**scale*tail*math.exp(cap+2**scale*tail)
    return result, dict(cauchy_radius=radius, polynomial_exponent_circle_norm_cap=cap,
                        squarings=scale, exponential_truncation_coefficient4_cap=error/radius**4,
                        scope='Exact-arithmetic Taylor tail only; IEEE rounding independently contrasted, not enclosed')


def path_hamiltonian(model: Any, frame: np.ndarray, momenta: np.ndarray,
                     amplitude: complex = 0.) -> np.ndarray:
    onsite, paths = scalar_paths(model)
    h = np.broadcast_to(onsite*frame[0], (len(momenta),4,4)).astype(complex).copy()
    for displacement, scalars in paths:
        hopping = np.einsum('m,mij->ij', scalars, frame)
        angle = momenta@displacement+amplitude*(DIRECTION@displacement)
        h += np.exp(1j*angle)[:,None,None]*hopping
        h += np.exp(-1j*angle)[:,None,None]*hopping.conj().T
    return h


def path_jet(model: Any, frame: np.ndarray, momenta: np.ndarray) -> np.ndarray:
    onsite, paths = scalar_paths(model)
    result = np.zeros((5,len(momenta),4,4), complex)
    result[0] = onsite*frame[0]
    for displacement, scalars in paths:
        hopping = np.einsum('m,mij->ij', scalars, frame)
        phase = np.exp(1j*(momenta@displacement))[:,None,None]
        slope = DIRECTION@displacement
        for n in range(5):
            result[n] += phase*hopping*(1j*slope)**n/math.factorial(n)
            result[n] += phase.conj()*hopping.conj().T*(-1j*slope)**n/math.factorial(n)
    return result


def trace_log_jet(matrix: np.ndarray) -> np.ndarray:
    # Normalize by the computed constant matrix; in exact arithmetic it is I.
    matrix = matrix@np.linalg.inv(matrix[0])
    x = matrix.copy(); x[0] = 0
    power = x.copy(); result = np.zeros(5, complex)
    for n in range(1,5):
        result += (-1)**(n+1)*np.sum(np.trace(power, axis1=-2, axis2=-1), axis=1)/n
        power = product(power, x)
    return result


def scalar_exp(logarithm: np.ndarray) -> np.ndarray:
    out = np.zeros(5, complex); out[0] = 1
    for n in range(1,5):
        out[n] = sum(j*logarithm[j]*out[n-j] for j in range(1,n+1))/n
    return out


def scalar_log(series: np.ndarray) -> np.ndarray:
    normalized = series/series[0]
    out = np.zeros(5, complex)
    for n in range(1,5):
        out[n] = normalized[n]-sum(j*out[j]*normalized[n-j] for j in range(1,n))/n
    return out


def cumulant_law(k: np.ndarray) -> dict[str, Any]:
    # Algebraic cumulants: complex bilinear covariance, NOT |k1|^2.
    mean = np.mean(k, axis=0); x = k[:,0]-mean[0]; y = k[:,1]-mean[1]
    parts = [dict(mean_k1=mean[0]),
             dict(mean_k2=mean[1], variance_k1=np.mean(x*x)),
             dict(mean_k3=mean[2], three_cov_k1_k2=3*np.mean(x*y), cum3_k1=np.mean(x**3)),
             dict(mean_k4=mean[3], four_cov_k1_k3=4*np.mean(x*(k[:,2]-mean[2])),
                  three_variance_k2=3*np.mean(y*y), six_cum_k1_k1_k2=6*np.mean(x*x*y),
                  cum4_k1=np.mean(x**4)-3*np.mean(x*x)**2)]
    return dict(terms=parts, total=np.array([sum(row.values()) for row in parts]),
                formula='k2=E[k2]+Var(k1); k3=E[k3]+3Cov(k1,k2)+Cum3(k1); k4=E[k4]+4Cov(k1,k3)+3Var(k2)+6Cum(k1,k1,k2)+Cum4(k1)',
                convention='k_n=d^n W/dA^n at0, complex connected action derivatives; no conjugation in Cov/Cum')


def prepare_component(cfg: PhiConfig, twist: tuple[int,...], frame: np.ndarray,
                      paired: bool, clock: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grid = np.asarray(list(itertools.product(range(6), repeat=3)), float)
    momenta = 2*np.pi*(grid+np.asarray(twist)/4)/6
    if paired:
        momenta = -momenta
        frame = np.einsum('mn,nij->mij', D, B@frame.conj()@B.conj().T)
    logarithm = np.zeros(5, complex); records = []; errors = []
    for load, model, role in models(replace(cfg, Lambda=1.)):
        if role.startswith('neutral'):
            continue  # Heavy has q=0, hence its relative EM functional is identically1.
        h = path_jet(model, frame, momenta)
        values, vectors = np.linalg.eigh(h[0]); occupied = vectors[:,:,:2]
        projector = occupied@occupied.conj().transpose(0,2,1)
        unitary, tail = exponential_jet(h, clock)
        baseline = (vectors*np.exp(1j*clock*values)[:,None,:])@vectors.conj().transpose(0,2,1)
        projected = occupied.conj().transpose(0,2,1)@(unitary@baseline)@occupied
        profile_log = trace_log_jet(projected)
        other, _ = exponential_jet(h, clock, 18)
        second_log = trace_log_jet(occupied.conj().transpose(0,2,1)@(other@baseline)@occupied)
        errors.append(float(np.max(abs(profile_log-second_log)/(1+abs(profile_log)))))
        logarithm += load*profile_log
        records.append(dict(load=int(load), model=model, role=role, frame=frame, momenta=momenta,
                            occupied=occupied, baseline=baseline, h=h, profile_log=profile_log))
        current = -1j*clock*np.sum(np.trace(projector@h[1],axis1=-2,axis2=-1))
        assert abs(current-profile_log[1]) < 1e-8*(1+abs(current))
        assert float(np.max(abs(projector@projector-projector))) < 1e-12
        assert float(np.max(abs(h[0]@projector-projector@h[0]))) < 1e-10
        assert float(np.max(values[:,:2])) < 0 < float(np.min(values[:,2:]))
        records[-1]['proof'] = dict(role=role, load=int(load), occupied_modes=432,
            negative_band_max=float(np.max(values[:,:2])), positive_band_min=float(np.min(values[:,2:])),
            projector_idempotence_error=float(np.max(abs(projector@projector-projector))),
            noncommuting_H0_H1_max=float(np.max(abs(h[0]@h[1]-h[1]@h[0]))),
            initial_current_derivative=current, tail=tail,
            primitive_paths=[dict(displacement=d.tolist(), coefficients=s.tolist()) for d,s in scalar_paths(model)[1]])
    row = dict(twist=list(twist), C_image=paired, weight=dict(numerator='1',denominator='128'),
               bare_log_coefficients=logarithm, bare_cumulants=logarithm[1:]*np.array([1,2,6,24]),
               profiles=[record['proof'] for record in records], independent_18_vs_24_relative_error=max(errors))
    return row, records


def finite_functional(records: list[dict[str, Any]], clock: float, amplitude: complex) -> complex:
    total = 0j
    for record in records:
        h = path_hamiltonian(record['model'],record['frame'],record['momenta'],amplitude)
        # Clifford-frame exact finite curve; independent of the jet recurrence.
        r2 = np.trace(h@h,axis1=-2,axis2=-1)/4
        assert float(np.max(abs(h@h-r2[:,None,None]*np.eye(4))/(1+abs(r2[:,None,None])))) < 1e-12
        radius = np.sqrt(r2)
        evolution = np.cos(clock*radius)[:,None,None]*np.eye(4)-1j*(np.sin(clock*radius)/radius)[:,None,None]*h
        overlap = record['occupied'].conj().transpose(0,2,1)@(evolution@record['baseline'])@record['occupied']
        total += record['load']*np.sum(np.log(np.linalg.det(overlap)))
    return complex(np.exp(total))


def native_car_control(records: list[dict[str, Any]], clock: float, amplitude: float) -> dict[str, Any]:
    C, a, local = local_charge_unitary(); maximum = 0.; rows = []
    for record in records:
        index = 17; occupied = record['occupied'][index]; state = np.zeros(16,complex)
        for i,j in itertools.combinations(range(4),2):
            state[(1<<i)+(1<<j)] = np.linalg.det(occupied[[i,j],:])
        h0 = record['h'][0,index]
        hA = path_hamiltonian(record['model'],record['frame'],record['momenta'][index:index+1],amplitude)[0]
        actual = np.vdot(state,expm(1j*clock*quadratic(a,h0))@expm(-1j*clock*quadratic(a,hA))@state)
        one = np.linalg.det(occupied.conj().T@expm(1j*clock*h0)@expm(-1j*clock*hA)@occupied)
        error = abs(actual-one); maximum = max(maximum,error)
        rows.append(dict(role=record['role'], fock16_overlap=actual, determinant_overlap=one, error=error))
    assert maximum < 1e-11
    return dict(local_C=local, Ref=ref_gauss_control(C,a), fock16_rows=rows, maximum_error=maximum,
                scope='Actual CAR 16-dimensional momentum block controls; whole N6 Slater is product over all216 blocks and64 charged flavours')


def serialize(value: Any) -> Any:
    if isinstance(value,np.ndarray): return serialize(value.tolist())
    if isinstance(value,dict): return {str(k):serialize(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [serialize(v) for v in value]
    if isinstance(value,(complex,np.complexfloating)): return dict(real=float(value.real),imag=float(value.imag))
    if isinstance(value,np.generic): return value.item()
    return value


def run(holonomy_path: str, clock: float) -> dict[str, Any]:
    start = time.monotonic(); raw = Path(holonomy_path).read_bytes(); holo = json.loads(raw)
    assert holo['input_descriptor']['size'] == 6
    assert len(holo['classes']) == 10
    assert holo['source_at_import']['induced_phi_centre_holonomy.py'] == SOURCE_AT_IMPORT['induced_phi_centre_holonomy.py']['sha256']
    cfg = PhiConfig(**holo['input_descriptor']['config']); frames = mean_frames(holo)
    assert len(frames) == 64
    descriptor = dict(native_size=6, sites=216, charged_active=1, charged_auxiliary=63,
        neutral_heavy_EM_charge=0, Ref_bare_Qref=1, Ref_occupation=4, Ref_background=-4,
        Ref_net_charge=0, mobile_background_per_site=-128, clock=clock, hbar=1,
        external_A='dimensionless uniform real link angle A*(1,.37,-.23), primitive two-link phase sums retained',
        frozen_receipt_sha256=hashlib.sha256(raw).hexdigest(), config=asdict(cfg),
        roots='All10 G_decimal nativeN6 classes from actual old holonomy receipt, reconstructed all64 by its reflection/permutation action; not macro ROOT or midpoint mean',
        twist='All64 fundamental central SU4 seams; Bloch shifts t/4. C image uses -p, -twist, D B Phi* Bdag and negative one-body generator/time orientation.',
        source='NEW rho128=1/128 sum_t(rho_t+C rho_t Cdag); fixed normalized density, not retag of oldrho64',
        backend='NumPy complex128 noncommutative polynomial exp/scaling-squaring+Slater determinant; independent exact Clifford finite curves and SciPy Fock16 expm',
        plan=dict(id='issue-222-induced-photon-delta13-continuation-20261008', delegated_revision='d13d7d28ebfd7cb8386e7442897c596481aeab95cd55def1a8b3a60c1e89f4dd', template_revision='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe', work_item='joined-explanatory-campaign'))
    components = []; prepared = []; pair_errors = []; car = None
    for twist, frame in frames.items():
        original, original_records = prepare_component(cfg,twist,frame,False,clock)
        image, image_records = prepare_component(cfg,twist,frame,True,clock)
        pair_errors.append(float(np.max(abs(image['bare_cumulants']-original['bare_cumulants']*np.array([-1,1,-1,1]))/(1+abs(original['bare_cumulants'])))))
        components.extend((original,image)); prepared.extend((original_records,image_records))
        if twist == (0,0,1): car = native_car_control(original_records+image_records,clock,.003)
    logs = np.array([c['bare_log_coefficients'] for c in components])
    z_series = np.mean(np.array([scalar_exp(v) for v in logs]),axis=0)
    mixed_log = scalar_log(z_series); mixed_k = mixed_log[1:]*np.array([1,2,6,24])
    law = cumulant_law(np.array([c['bare_cumulants'] for c in components]))
    law_error = float(np.max(abs(mixed_k-law['total'])/(1+abs(mixed_k))))
    contrast = mixed_k-np.mean(np.array([c['bare_cumulants'] for c in components]),axis=0)
    finite = []; maximum_curve_error = 0.
    for amplitude in (.001,-.001,.003,-.003,.006,-.006):
        z = np.array([finite_functional(r,clock,amplitude) for r in prepared])
        actual = np.log(np.mean(z)); wrong = np.mean(np.log(z))
        finite.append(dict(A=amplitude, mix_sumZ_then_log=actual, wrong_avglog=wrong,
                           difference=actual-wrong,
                           direct_pair_Z_even_error=float(np.max(abs(z[0::2]-np.array([finite_functional(r,clock,-amplitude) for r in prepared[1::2]]))))))
    # Independent Cauchy coefficient extraction of finite determinant curves.
    radius = .01; nodes = 64; angles = 2*np.pi*np.arange(nodes)/nodes
    curves = np.array([np.mean([finite_functional(r,clock,radius*np.exp(1j*angle)) for r in prepared]) for angle in angles])
    curve_logs = np.log(curves)
    extracted = np.array([np.mean(curve_logs*np.exp(-1j*n*angles))/radius**n*math.factorial(n) for n in range(1,5)])
    maximum_curve_error = float(np.max(abs(extracted-mixed_k)/(1+abs(mixed_k))))
    zeros = np.array([finite_functional(r,clock,0.) for r in prepared])
    quarter = next(c for c in components if c['twist']==[0,0,1] and not c['C_image'])
    assert abs(quarter['bare_cumulants'][0]) > 1e-4
    assert max(pair_errors) < 1e-7 and law_error < 1e-10
    assert abs(mixed_k[0]) < 1e-7 and abs(mixed_k[2]) < 1e-5*(1+abs(mixed_k[3]))
    assert abs(contrast[1]) > 1e-3 and abs(contrast[3]) > 1e-3
    assert maximum_curve_error < 1e-4
    assert float(np.max(abs(zeros-1))) < 1e-8
    # Scalar-path norm price: unitaries are contractions for real A. This is
    # an all-clock Duhamel budget, not a vanishing individual full-source current.
    caps = []
    for load,model,role in models(replace(cfg,Lambda=1.)):
        if role.startswith('neutral'): continue
        _,paths = scalar_paths(model)
        cap = 2*sum(float(np.sum(abs(s)))*abs(float(DIRECTION@d)) for d,s in paths)
        caps.append(dict(role=role,flavours=int(load), path_current_per_Phi_norm_cap=cap))
    macro_volume = 262144**3
    macro_budget = sum(v['flavours']*v['path_current_per_Phi_norm_cap'] for v in caps)*4*macro_volume*4096
    stable = {name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()}
    assert all(stable.values())
    return serialize(dict(branch='NEW fixed normalized128 C-paired density functional; barefermion N6 controls only',
        input_descriptor=descriptor, input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
        input_receipt_sha256=hashlib.sha256(raw).hexdigest(), components=components,
        normalized_weights=dict(components=128,numerator_sum=128,common_denominator=128,exact_sum=1,
            original64_density_weights=dict(numerator='1',denominator='64',components=64,exact_sum=1)),
        native_functional_definition='Z_s[A]=Tr(U_s[A] rho_s U_s[0]dag)=product_{flavour,p}det(Vocc† exp(+it h0) exp(-it hA) Vocc), normalized atA0; rho_s fixed fullnegative Slater; U_s=exp(-it dGamma(h_s))',
        native_C_pair_definition='h_C(p_C,A)=-B h(p,-A)* Bdag, P_C=B(1-P)*Bdag. Manybody C is EVEN realmonomial with C²=I. Complementary Slater determinant plus zero trace ofh gives Z_C(A)=Z_s(-A); minus one-body generator is t->-t, NOT discarded source chi1.',
        mixture=dict(Z='sum exact(1/128)*Z_s[A]', W='log Z128[A] on analytic branchW0=0',
                     cumulants=mixed_k, cumulant_law=law, sumZ_log_vs_cumulant_law_relative_error=law_error,
                     max_direct_C_pair_jet_relative_error=max(pair_errors),
                     paired_odd_residual=[mixed_k[0],mixed_k[2]],
                     quarter_individual_chi1=quarter['bare_cumulants'][0],
                     wrong_average_log_cumulants=np.mean(logs[:,1:],axis=0)*np.array([1,2,6,24]),
                     mix_minus_average_log=contrast),
        native_CAR=car, finite_A_controls=finite, independent_Cauchy=dict(radius=radius,nodes=nodes,
            actual_finite_curve_cumulants=extracted, jet_relative_error=maximum_curve_error,
            caveat='Independent numerical analytic-circle extraction; no rigorous IEEE/Cauchy alias enclosure or full-source jet claim'),
        normalization_zero_max_error=float(np.max(abs(zeros-1))),
        full_macro_source=dict(definition='Z_s_full[A]=Tr(U_s_full[A] rho_s U_s_full[0]dag); Z128_full=sum1/128 Z_s_full; W128_full=logZ128_full',
            density='FIXED normalized rho128=(rho64+C rho64 Cdag)/2, valid because actual C² density identity; Ref fixed onsite filled4 SU4singlet with bareQref1 andbackground-4 remains unchanged',
            evenness='CONDITIONAL on full-action C covariance and parent/peer domain+stat/Jac review. IfC U_full[A] Cdag=U_full[-A], fixedrho128 C invariance gives Z_full[A]=Z_full[-A]. No electron+Refhole invariance asserted.',
            individual_chi1='RETAINED. Native initial persistent current is not fullquantum finiteclock chi1=0. C removes only paired totalodd jets when fullaction identity holds.',
            graph_current_budget=dict(path_caps=caps, macro_N=262144, clock=4096,
                bound_multiplier_on_sup_time_source_mean_Phi_operator_norm=macro_budget,
                meaning='|chi1_s_full| <=4V T sum_flavour pathcap * sup_t <||Phi||_max>. Duhamel/CAR graph bound, not evaluated unbounded scalar moment. Each bilinear norm<=4||h||; physical heavy q0 and fixedRef do not contribute. No zero claim.'),
            jets_computed=False, native_chi4_is_full_source_chi4=False,
            no_inheritance='No490/738 Gamma3, ghostkernel, photonfit, macro EM vacuum or commoncone claim'),
        source_at_import=SOURCE_AT_IMPORT, source_unchanged_since_import=stable,
        source_snapshot_at_import=dict(path=Path(__file__).name,sha256=SOURCE_AT_IMPORT[Path(__file__).name]['sha256'],text=SOURCE_BYTES[Path(__file__).name].decode()),
        runtime=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,platform=platform.platform()),
        elapsed_seconds=time.monotonic()-start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--holonomy',required=True)
    parser.add_argument('--clock',type=float,default=.01)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    result = run(args.holonomy,args.clock)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('input_sha256','mixture','normalization_zero_max_error','independent_Cauchy','source_at_import','elapsed_seconds')},indent=2))
