#!/usr/bin/env python3
"""Positive-proposal relative Gauss denominator, retaining actual canonical law.

Gaussian importance sampling is a proposal, NEVER the interacting state. The
sampled numerator still uses ALL canonical coefficients, Pauli components,
auxiliary powers, normal-Phi heat and the literal coherent integer source.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import marshal
import os
from pathlib import Path
import sys
import time
from typing import Any,cast
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parent
OWN_BYTES=Path(__file__).read_bytes()
LOADED_MAIN_CODE=sys._getframe().f_code
MAIN_CODE_MATCH=LOADED_MAIN_CODE==compile(OWN_BYTES,LOADED_MAIN_CODE.co_filename,'exec',dont_inherit=True,optimize=sys.flags.optimize)
if not MAIN_CODE_MATCH:raise ArithmeticError('Loaded conditioned-denominator code differs from captured bytes')
import numpy as np
from scipy.linalg import eigh
from scipy.special import ndtri
from scipy.special import logsumexp
from scipy.stats import qmc
import covariant_relative_ring_boundary as ring


def auxiliary_character(gamma: np.ndarray,sites: int,kappa: float,points: np.ndarray) -> np.ndarray:
    """ONE exact fixed-N two-clock coefficient normalized at zero characters.

    The underlying JOINT occupation weights are |Gamma_N(T)_(sigma,tau)|².
    Thus this is a characteristic function of an actual positive bounded
    two-clock occupation law, not an assertion that each character is positive.
    """
    h,p,_=ring.ring_half(gamma,sites,kappa,0.,0.)
    number=2*sites
    first=np.zeros((len(points),sites));second=np.zeros_like(first)
    first[:,1:]=points[:,:sites-1];second[:,1:]=points[:,sites-1:]
    g=np.repeat(np.exp(1j*first),4,axis=1);r=np.repeat(np.exp(1j*second),4,axis=1)
    C=h[None,:,:]@(g[:,:,None]*(h@h)[None,:,:]*r[:,None,:])@h[None,:,:]
    polynomial,scale,_=ring.canonical_polynomial(C,number)
    coefficient=polynomial[:,number]*scale**number
    zero,zero_scale,_=ring.canonical_polynomial((h@h@h@h)[None,:,:],number)
    normalizer=zero[0,number]*zero_scale[0]**number
    return coefficient/normalizer*np.exp(-2j*(first.sum(axis=1)+second.sum(axis=1)))


def canonical_covariance(gamma: np.ndarray,sites: int,kappa: float,step: float) -> tuple[np.ndarray,dict[str,Any]]:
    dimension=2*(sites-1);identity=np.eye(dimension)
    points=[np.zeros(dimension)];locations={}
    for i in range(dimension):
        for sign in (-1,1):locations[i,sign]=len(points);points.append(sign*step*identity[i])
    pairs={}
    for i in range(dimension):
        for j in range(i):
            for si in (-1,1):
                for sj in (-1,1):pairs[i,j,si,sj]=len(points);points.append(step*(si*identity[i]+sj*identity[j]))
    values=auxiliary_character(gamma,sites,kappa,np.array(points))
    logarithms=np.log(values);covariance=np.zeros((dimension,dimension))
    means=np.zeros(dimension)
    for i in range(dimension):
        positive=logarithms[locations[i,1]];negative=logarithms[locations[i,-1]]
        covariance[i,i]=-float(np.real(positive+negative-2*logarithms[0]))/step**2
        means[i]=float(np.imag(positive-negative))/(2*step)
        for j in range(i):
            value=sum(si*sj*logarithms[pairs[i,j,si,sj]] for si in (-1,1) for sj in (-1,1))
            covariance[i,j]=covariance[j,i]=-float(np.real(value))/(4*step**2)
    spectrum=eigh(covariance,eigvals_only=True)
    ring.require(bool(spectrum[0]>0),'An actual canonical covariance must be positive; no artificial ridge or fallback is permitted')
    return covariance,dict(finite_difference_step=step,exact_canonical_character_samples=len(points),
        covariance_spectrum=spectrum,character_means=means,
        scope='Finite-difference covariance used ONLY to construct the positive sampling proposal. Neither the Hessian nor a central-limit theorem is assumed exact in the scientific integrand.')


def importance_points(covariance: np.ndarray,copies: int,power: int,seed: int) -> tuple[np.ndarray,np.ndarray,np.ndarray,dict[str,Any]]:
    values,frame=eigh(covariance);dimension=len(values);sites=dimension//2+1
    uniforms=qmc.Sobol(dimension,scramble=True,rng=seed).random_base2(power)
    initial=ndtri(uniforms)
    normals=np.concatenate((initial,-initial))
    points=(normals/np.sqrt(copies*values))@frame.T
    inside=np.all(abs(points)<=math.pi,axis=1)
    quadratic=np.sum(normals*normals,axis=1)
    weights=np.where(inside,np.exp(quadratic/2),0.)
    first=np.zeros((len(points),sites));second=np.zeros_like(first)
    first[:,1:]=points[:,:sites-1];second[:,1:]=points[:,sites-1:]
    gauges=np.stack((np.repeat(np.exp(1j*first),4,axis=1),np.repeat(np.exp(1j*second),4,axis=1)),axis=1)
    background=np.exp(-2j*(first.sum(axis=1)+second.sum(axis=1)))
    logJ=-.5*dimension*math.log(2*math.pi)-.5*float(np.sum(np.log(copies*values)))
    return gauges,background,weights,dict(actual_samples=len(points),rejected_outside_compact_cube=int(np.count_nonzero(~inside)),
        actual_character_coordinate_max=float(np.max(abs(points))),common_Gauss_Jacobian_log=logJ,
        covariance_reference=covariance,
        antithetic_actual_charge_characters=True,
        exact_reweighting='Integral_Haar F = J_ref E_standardNormal[1_cube exp(|z|²/2) F(P_ref^-1/2 z)]. ONE COMMON J_ref is analytically removed from neutral AND source and every holonomy. This is exact change of measure, not dividing an absolute error by a tiny macroZ.',
        Gaussian_proposal_is_not_the_state=True)


def auxiliary_reference_scalar(gamma: np.ndarray,sites: int,kappa: float,theta: float=0.) -> float:
    """Zero-character TWO-clock coefficient, with FOUR actual SF-half scalars."""
    h,p,_=ring.ring_half(gamma,sites,kappa,theta,0.)
    polynomial,scale,_=ring.canonical_polynomial((h@h@h@h)[None,:,:],2*sites)
    W=ring.common_vacuum_scale(gamma,sites,kappa)
    return float(np.real(polynomial[0,2*sites]*scale[0]**(2*sites)))*(p/W)**4


def logarithmic_relative(rows: list[dict[str,Any]],outside_logs: np.ndarray) -> dict[str,Any]:
    neutral:np.ndarray=np.array([complex(row['neutral_positive_completed_trace']) for row in rows],dtype=np.complex128)
    source:np.ndarray=np.array([complex(row['source_positive_completed_trace']) for row in rows],dtype=np.complex128)
    ring.require(bool(np.min(np.real(neutral))>0 and np.min(np.real(source))>0),
        'The completed integral after exact common scalar extraction must remain positive')
    logN=np.log(np.real(neutral))+outside_logs;logS=np.log(np.real(source))+outside_logs
    meanN=float(cast(float,logsumexp(logN,return_sign=False)))-math.log(len(rows))
    meanS=float(cast(float,logsumexp(logS,return_sign=False)))-math.log(len(rows))
    n=np.exp(logN-meanN);s=np.exp(logS-meanS)
    return dict(neutral_relative_density=n,source_relative_density=s,
        neutral_relative_Fourier=np.fft.fft(n)/len(n),source_relative_Fourier=np.fft.fft(s)/len(s),
        actual_positive_source_to_neutral_two_clock_ratio=np.real(source)/np.real(neutral),
        actual_restored_neutral_log_trace=logN,actual_restored_source_log_trace=logS,
        actual_restored_neutral_log_sampled_mean=meanN,actual_restored_source_log_sampled_mean=meanS,
        neutral_relative_common_shift_variation=float(np.ptp(n)),source_relative_common_shift_variation=float(np.ptp(s)),
        maximum_neutral_imaginary_fraction=float(np.max(abs(np.imag(neutral))/np.real(neutral))),
        maximum_source_imaginary_fraction=float(np.max(abs(np.imag(source))/np.real(source))),
        normalization='Exact per-holonomy neutral auxiliary canonical scalar is RESTORED in both log traces BEFORE normalization. logsumexp is numerical arithmetic, never charged-own-max energy subtraction. Tiny positive densities may underflow in their displayed float, but their true positive log traces are retained.',
        Fourier_scope='DFT on the stated finite cyclic grid; these are alias-folded sampled coefficients, not continuum Fourier coefficients or a priced theta-Haar quadrature.')


def replicate_summary(cases: list[dict[str,Any]]) -> dict[str,Any]:
    neutral=np.array([case['relative']['neutral_relative_density'] for case in cases],float)
    source=np.array([case['relative']['source_relative_density'] for case in cases],float)
    neutral_mean=neutral.mean(axis=0);source_mean=source.mean(axis=0)
    return dict(neutral_relative_density_replicate_mean=neutral_mean,source_relative_density_replicate_mean=source_mean,
        neutral_relative_density_replicate_range=np.ptp(neutral,axis=0),source_relative_density_replicate_range=np.ptp(source,axis=0),
        neutral_relative_Fourier_replicate_mean=np.fft.fft(neutral_mean)/len(neutral_mean),
        source_relative_Fourier_replicate_mean=np.fft.fft(source_mean)/len(source_mean),
        empirical_range_is_not_a_rigorous_bound=True)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rings',type=int,nargs='+',default=[6,8,10])
    parser.add_argument('--shifts',type=int,default=8)
    parser.add_argument('--aux-kappa',type=float,default=.04)
    parser.add_argument('--aux-copies',type=int,default=10**8)
    parser.add_argument('--points-power',type=int,default=11)
    parser.add_argument('--seeds',type=int,nargs='+',default=[222,223,224])
    parser.add_argument('--normal-amplitude',type=float,default=.0002)
    parser.add_argument('--packet-width',type=float,default=.8)
    parser.add_argument('--batch',type=int,default=256)
    args=parser.parse_args();start=time.monotonic()
    parameters=ring.supplier.Parameters();gamma,_=ring.supplier.phi_frame(parameters)
    body=max(1/(2*(math.expm1(a*m)+4)) for a in parameters.spacings for m in parameters.masses)
    _,ground,H,half=ring.supplier.normal_heat();heat=half@half
    coordinates=np.linspace(-args.normal_amplitude,args.normal_amplitude,len(ground))
    cases=[]
    for sites in args.rings:
        centre=ring.centre_ref_control(sites,args.aux_copies)
        one,packet=ring.source_packet(gamma,sites,body,args.packet_width)
        _,harmonic=ring.integer_source_paths(sites)
        covariance,derivative=canonical_covariance(gamma,sites,args.aux_kappa,.0025)
        coarse,coarse_derivative=canonical_covariance(gamma,sites,args.aux_kappa,.005)
        covariance_error=float(np.linalg.norm(covariance-coarse,2)/eigh(covariance,eigvals_only=True)[0])
        reference=auxiliary_reference_scalar(gamma,sites,args.aux_kappa)
        replicate_rows=[];shifts=2*math.pi*np.arange(args.shifts)/args.shifts
        theta_references=np.array([auxiliary_reference_scalar(gamma,sites,args.aux_kappa,float(theta)) for theta in shifts])
        outside_logs=args.aux_copies*(np.log(theta_references)-math.log(reference))
        for seed in args.seeds:
            gauges,background,weights,importance=importance_points(covariance,args.aux_copies,args.points_power,seed)
            rows=[ring.complete_pair(gamma,sites,body,args.aux_kappa,args.aux_copies,float(theta),coordinates,heat,
                one,harmonic,gauges,background,args.batch,weights=weights,auxiliary_reference=float(theta_reference))
                for theta,theta_reference in zip(shifts,theta_references)]
            replicate_rows.append(dict(seed=seed,importance=importance,rows=rows,
                relative=logarithmic_relative(rows,outside_logs)))
        cases.append(dict(sites=sites,canonical_number_per_flavour=2*sites,full_components_per_site=4,
            actual_body_kappa=body,actual_auxiliary_kappa=args.aux_kappa,auxiliary_copies_exact=str(args.aux_copies),
            actual_global_centre_control=centre,
            source_packet=packet,normal_coordinates=coordinates,normal_ground=ground,normal_H=H,actual_normal_heat=heat,
            holonomies=shifts,proposal_derivative=derivative,proposal_coarse_derivative=coarse_derivative,
            empirical_relative_proposal_covariance_difference=covariance_error,
            ONE_common_auxiliary_canonical_zero_character_scalar=reference,
            removed_common_log_scalar_auxiliary=args.aux_copies*math.log(reference),
            per_holonomy_true_canonical_zero_character_scalars=theta_references,
            RESTORED_holonomy_log_scalars_relative_to_ONE_common_reference=outside_logs,
            replicates=replicate_rows,relative_summary=replicate_summary(replicate_rows)))
    hashes=dict(ring.supplier.frozen.COMPILED)
    hashes['covariant_canonical_gauss_boundary_kernel']=ring.SUPPLIER_HASH
    hashes['covariant_relative_ring_boundary']=hashlib.sha256(ring.OWN_BYTES).hexdigest()
    hashes[Path(__file__).stem]=hashlib.sha256(OWN_BYTES).hexdigest()
    captured={name:ring.supplier.frozen.SNAPSHOT[name].decode() for name in ring.supplier.frozen.COMPILED}
    captured['covariant_canonical_gauss_boundary_kernel']=ring.supplier.OWN_BYTES.decode()
    captured['covariant_relative_ring_boundary']=ring.OWN_BYTES.decode();captured[Path(__file__).stem]=OWN_BYTES.decode()
    unchanged=all((ROOT/(name+'.py')).read_text()==captured[name] for name in hashes)
    ring.require(unchanged,'Source bytes changed during importance calculation')
    result=dict(plan=ring.PLAN,governing_revision=ring.REVISION,native_epoch='NEW relative strong-auxiliary EXACT-canonical conditioned Gauss denominators',
        parameters=vars(args)|{'output':str(args.output)},cases=cases,compiled_source_hashes=hashes,captured_source_bytes=captured,
        compiled_source_unchanged=unchanged,actual_frozen_body_kappa=body,actual_frozen_body_8k=8*body,
        main_code_matches_captured_source=MAIN_CODE_MATCH,
        loaded_main_code_sha256=hashlib.sha256(marshal.dumps(LOADED_MAIN_CODE)).hexdigest(),
        ring_main_code_matches_captured_source=ring.MAIN_CODE_MATCH,
        runtime=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=sys.modules['scipy'].__version__),
        elapsed_seconds=time.monotonic()-start,macro_conditioned_admission=False,
        exact_positive_law='For a fixed physical Phi pair, cyclic trace of SFw Gchi SFw SFz Gxi SFz is sum_sigma,tau |(SFz SFw)_sigma,tau|² exp(i chi·rho_sigma+i xi·rho_tau). Source replaces this by sum_Ref |SFcharged_z S_Ref SFneutral_w|². This is a JOINT positive occupation law; its characters need not be positive. Each auxiliary copy has its OWN fixed-N polynomial; shared Gauss integration occurs after their integer tensor power.',
        macro_route='Relative lattice-charge Fourier/local-central-limit bounds in the exact positive two-clock occupation law; common Gaussian Jacobian and common canonical scalar cancel identically. For the actual Nt/fullPhi cycle, use the frozen all-occupation auxiliary mixed-SVD/positive occupied-minor kernel and its Schatten_Nt thermal price before attempting a conditional law. No uniform toron stationarity is inherited from this native calculation.',
        limits=['The importance integrand is the REAL full canonical/Gauss/Phi/source coefficient; the Gaussian is only the positive proposal.',
            'Replicate variation and finite-difference Hessian agreement are empirical diagnostics, NOT rigorous intervals or a macro source admission.',
            'Native rings and clock2 do NOT replace macroL=2^24,Nt=2^20, all60/45 Phi, statistical SU4 fibre or the declared125 photon preparation.',
            'Root6911 bodyκ remains .1248905293. AlphaCκ .1256305951/full new completed law is distinct and not imported here.'])
    args.output.write_text(json.dumps(ring.supplier.portable(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),source_sha256=hashes[Path(__file__).stem],elapsed_seconds=result['elapsed_seconds'],
        variations=[dict(sites=c['sites'],neutral=float(np.ptp(c['relative_summary']['neutral_relative_density_replicate_mean'])),
            source=float(np.ptp(c['relative_summary']['source_relative_density_replicate_mean']))) for c in cases],macro_conditioned_admission=False)))
    print('Natural exit: actual relative canonical/Gauss/source coefficients under positive proposal; no macro admission')


if __name__=='__main__':main()
