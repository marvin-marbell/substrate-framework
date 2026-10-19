#!/usr/bin/env python3
"""Exact mean-Clifford Root6911 reference, not an interacting-state admission.

SF halfstep is sqrt(det Bplus) Gamma(sqrt(T)). Its neutral vacuum at N=2V
has log Wsf=sum_p(log B(p)+E(p)); both charged sectors use THIS SAME neutral
normalization. All spatial twists/statistical torons are retained.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.abc
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
OWN_BYTES = Path(__file__).read_bytes()
LOADED_CODE = sys._getframe().f_code
if LOADED_CODE != compile(OWN_BYTES, LOADED_CODE.co_filename, 'exec', dont_inherit=True, optimize=sys.flags.optimize):
    raise ArithmeticError('Loaded derivation differs from captured source')
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
CAPTURED = {Path(__file__).name: OWN_BYTES}
class CapturedLoader(importlib.abc.Loader):
    def __init__(self, name: str, path: Path):
        self.name, self.path = name, path
    def create_module(self, spec: Any) -> Any:
        return None
    def exec_module(self, module: Any) -> None:
        data = self.path.read_bytes()
        CAPTURED[self.path.name] = data
        module.__file__ = str(self.path)
        exec(compile(data, str(self.path), 'exec', dont_inherit=True), module.__dict__)
class CapturedFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        candidate = ROOT / (fullname + '.py')
        if candidate.is_file():
            return importlib.util.spec_from_loader(fullname, CapturedLoader(fullname, candidate))
        return None
_finder = CapturedFinder()
sys.meta_path.insert(0, _finder)
import numpy as np
import scipy
from scipy.linalg import eigh
import mpmath as mp
import covariant_canonical_gauss_boundary_kernel as supplier
sys.meta_path.remove(_finder)
CAPTURED.setdefault('covariant_canonical_gauss_boundary_kernel.py', supplier.OWN_BYTES)
source = supplier.frozen
physical_root = supplier.physical_root
for _stem in source.COMPILED:
    _data = source.SNAPSHOT[_stem]
    if hashlib.sha256(_data).hexdigest() != source.COMPILED[_stem]:
        raise ArithmeticError('Source2a capture hash mismatch')
    if _stem + '.py' in CAPTURED and CAPTURED[_stem + '.py'] != _data:
        raise ArithmeticError('Conflicting loaded captures')
    CAPTURED[_stem + '.py'] = _data
for _name, _data in physical_root.SOURCE_BYTES_AT_IMPORT.items():
    if _name in CAPTURED and CAPTURED[_name] != _data:
        raise ArithmeticError('Conflicting Root captures')
    CAPTURED[_name] = _data
EXPECTED = {'covariant_prepared_gauss_state.py': '2a0ecfb2341fceb7866d5062cf33d1da71b2fafa4b148829d21d14f0650fe307',
            'covariant_root_transfer.py': '6911f0283dc0f84f4cc05202deb67dcd3266d5c64ca1a05e149f1567451e0f9c',
            'covariant_canonical_gauss_boundary_kernel.py': '741fbc1f196061cb7f0e2c479770e992304ac2de90492f002ef4cbd931315d50'}
for _name, _digest in EXPECTED.items():
    if hashlib.sha256(CAPTURED[_name]).hexdigest() != _digest:
        raise ArithmeticError('Frozen source changed: ' + _name)
PLAN = 'issue-222-induced-photon-delta13-continuation-20261008'
REVISION = '1124ce49865f8f7cc5f912ef9569edc52ad3227fb532fd78d5637658c923b990'
SELECTED_NAME = 'covariant_prepared_gauss_state_conditioned_source_physical_Dminus_verified_20261008.json'


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ArithmeticError(message)


def bloch_data(momentum: Any, am: float) -> dict[str, Any]:
    """Real or complex momenta, physical E branch and SF vacuum density."""
    p = np.asarray(momentum)
    B = math.exp(am) + np.sum(1 - np.cos(p), axis=-1)
    s2 = np.sum(np.sin(p)**2, axis=-1)
    dminus, dplus = (B - 1)**2 + s2, (B + 1)**2 + s2
    # Each factor's right-half-plane square root continues the physical branch.
    discriminant_root = np.sqrt(dminus.astype(complex)) * np.sqrt(dplus.astype(complex))
    vacuum = (B**2 + 1 + s2 + discriminant_root) / 2
    g = np.log(vacuum)
    E = g - np.log(B)
    return dict(B=B, s2=s2, dminus=dminus, dplus=dplus, E=E, g=g)


def strip_certificate(am: float) -> dict[str, Any]:
    """Analytic certificate; finite complex samples are discriminators, not proof."""
    require(0 < am <= .03, 'Certificate requires 0<am<=.03')
    delta = math.expm1(am)
    eta = delta / 128
    relative = 8 / 128 + 16 / 128**2
    return dict(am=am, delta=delta, eta=eta, discriminant_factor_relative_perturbation_upper=relative,
        g_absolute_upper=16, Fourier_coefficient_upper='16 exp(-eta |n|_1)',
        proof=[
            'For real x, B0=1+delta+t, t=sum(1-cos x) in [0,6], s0=sin x. Dminus0=(delta+t)^2+||s0||2^2>=delta^2; Dplus0=(2+delta+t)^2+||s0||2^2>4.',
            'For |Im z_j|<=eta, |cos z_j-cos x_j| and |sin z_j-sin x_j|<=eta exp(eta), by integration along the vertical segment. Thus the complex Euclidean perturbation of vplus/minus=(B0+/-1,s0) has norm<=sqrt(12) eta exp(eta)<4eta, since eta<=expm1(.03)/128<.001.',
            '|Dplus/minus(z)-Dplus/minus(x)|<=2||v||2||dv||2+||dv||2^2. Dividing by Dreal and using sqrt(Dreal)>=delta gives <=8eta/delta+16(eta/delta)^2=0.0634765625<1. Both factors remain in the right half-plane. Their separate principal square roots multiply to the positive physical discriminant root; no zero or branch crossing in the entire polytube.',
            'Re B>=exp(am)-3eta exp(eta)>1. For A=B^2+1+s2, real A0>=2; |dB|<3.01eta, ||ds||<1.74eta, B0<7.04, ||s0||<=sqrt(3), imply |dA|<50eta<.05. Hence Re A>1.95. Both D factors have argument magnitude<asin(.064), so Re sqrt(Dminus)sqrt(Dplus)>0. Therefore C=(A+sqrtD)/2 has Re C>.975 and g=Log C is single-valued periodic holomorphic.',
            '|B|<8, |s2|<4; |A|<69, |Dminus|<85, |Dplus|<85 give |C|<77. Re C>.975 yields |log|C||<log77 and |arg C|<pi/2, hence |g|<6<16. Cauchy contour displacement coordinate by coordinate yields |g_hat(n)|<=16 exp(-eta |n|1).'],
        physical_branch='g=Log[(B^2+1+s2+sqrt(Dminus)*sqrt(Dplus))/2]=log B+E; E>=am for real p')


def winding_budget(am: float, L: int) -> dict[str, Any]:
    require(L > 0, 'Positive finite spatial period required')
    certificate = strip_certificate(am)
    with mp.workdps(80):
        eta = mp.expm1(mp.mpf(str(am))) / 128
        r = mp.exp(-eta * L)
        epsilon = 256 * mp.mpf(L)**3 * r / (1-r)**3
        return dict(L=L, V=str(L**3), eta=mp.nstr(eta,30), etaL=mp.nstr(eta*L,30),
            log_vacuum_twist_error_upper=mp.nstr(epsilon,30), log10_vacuum_twist_error_upper=float(mp.log10(epsilon)),
            uniform_over_all_twists=True, all_eight_statistical_torons=True,
            Poisson_proof='Sum_p g((2pi n+theta)/L)=V sum_k g_hat(Lk) exp(i k.theta). The theta-minus-zero difference is <=32V[((1+r)/(1-r))^3-1]=32V(6r+2r^3)/(1-r)^3<=256V r/(1-r)^3. No toron window or selected coset is used.',
            analytic_certificate=certificate)


def reference_budget(a: float, m: float, L: int, Nt: int, beta_power: int = 150) -> dict[str, Any]:
    """Callable common-neutral SF / canonical thermal / connection reference."""
    am = float(mp.mpf(str(a)) * mp.mpf(str(m)))
    winding = winding_budget(am,L)
    with mp.workdps(80):
        ammp = mp.mpf(str(a))*mp.mpf(str(m))
        V = L**3
        eps = mp.mpf(winding['log_vacuum_twist_error_upper'])
        gap = mp.exp(ammp)-mp.exp(-ammp)
        stablegap = -mp.expm1(-ammp)
        Aone = 480/(gap*L)
        connection = (2*V+1)*Aone
        price = 8*connection/mp.sqrt(mp.power(10,beta_power)*L)
        clocks=[]
        for p in (Nt,Nt+1,Nt+2):
            logx = 2*mp.log(2*V)-2*ammp*p
            small = logx<=-mp.log(2)
            clocks.append(dict(power=p, small_regime=bool(small), log10_x=float(logx/mp.log(10)),
                canonical_neutral_excited_trace_log10_upper=float((mp.log(2)+logx)/mp.log(10)) if small else None,
                charged_excited_relative_trace_log10_upper=float((mp.log(2)+logx)/mp.log(10)) if small else None,
                charged_total_trace_common_neutral_log10_upper=float((mp.log(2*V)-ammp*p+mp.log1p(2*mp.exp(logx)))/mp.log(10)) if small else None))
        return dict(a=a,m=m,am=mp.nstr(ammp,30),kappa=mp.nstr(1/(2*(mp.expm1(ammp)+4)),30),
            spatial_size=L,time_slices=Nt,onebody_T_gap=mp.nstr(ammp,30),neutral_pair_T_gap=mp.nstr(2*ammp,30),
            stable_projector_gap_from_one=mp.nstr(stablegap,30),stable_unstable_spectral_gap=mp.nstr(gap,30),
            canonical_neutral_number=str(2*V),canonical_charged_numbers=[str(2*V-1),str(2*V+1)],
            SF_charged_common_neutral_norm_upper=mp.nstr(mp.exp(-ammp/2+eps),30),
            SF_charged_common_neutral_log_norm_upper=mp.nstr(-ammp/2+eps,30),
            SF_charged_uniform_ratio_to_local_neutral_upper=mp.nstr(mp.exp(-ammp/2),30),
            neutral_normalization='ONE Wsf(theta=0), not an own-charged maximum; all theta use Wsf(theta)/Wsf(0)<=exp(epsilon).',
            winding=winding,clock_thermal_prices=clocks,
            canonical_thermal_proof='At N=2V, j holes and j stable particles cost >=2am*j per whole T. Relative trace <=sum_{j=1}^{2V} binom(2V,j)^2 exp(-2am*j*p). With x=(2V)^2 exp(-2am*p)<=1/2, binom(n,j)<=n^j/j!, and sum x^j/(j!)^2<=x/(1-x)<=2x. For N=2V+/-1, binom(n,j)binom(n,j+1)<=n*n^(2j)/(j!(j+1)!), giving total <=n exp(-am*p)(1+2x). Multiplicity n is retained.',
            SF_identity='T has reciprocal eigenvalues exp(+/-E), each twice; sqrt(T) has exp(+/-E/2). sqrt(det Bplus)=product_p B(p). Neutral maximum=product_p B exp(E). Charged maxima add or remove one mode, so each/Wsf(theta)<=exp(-min_p E/2)<=exp(-am/2).',
            gap_proof='cosh E=(B+B^-1+s2/B)/2 >=(exp(am)+exp(-am))/2 since B>=exp(am)>1. Static p=0 gives E=am exactly.',
            connection_onebody_norm_upper=mp.nstr(Aone,30),connection_canonical_norm_upper=mp.nstr(connection,30),
            Gaussian_toron_connection_operator_price_log10=float(mp.log10(price)),
            connection_proof='Physical chiral T=[[B^-1,-F/B],[-Fdag/B,B+Fdag F/B]]. For scalar links, ||F||<=sqrt3<2, |dtheta B|<=1/L, ||dtheta F||<=1/L. Block differentiation gives ||dtheta T||<16/L<=240/L. Stable/unstable Sylvester separation is exp(am)-exp(-am), not .019 uniformly. Kato generator [dP,P] has norm <=480/(L crossgap); dGamma on N<=2V+1 costs N times this. Gaussian three-toron heat price 8 connection/sqrt(beta L) is a conditional reference comparison.',
            scope='Exact frozen-mean-Clifford/freeBloch spectral identities and toron reference bounds ONLY. No interacting Slater, Root interacting gap, RN-Loewner, weighted BKS, separate character positivity, Pe or receiver admission.',
            actual_Grams=None,microscopic_fullpacket_admission_proved=False)


def complex_discriminator(am: float, seed: int) -> dict[str, Any]:
    eta=strip_certificate(am)['eta']; rng=np.random.default_rng(seed)
    real=np.concatenate((rng.uniform(-math.pi,math.pi,(2048,3)),rng.normal(0,math.expm1(am),(1024,3)),
                         np.array(list(itertools.product((0,math.pi),repeat=3)))))
    imaginary=rng.choice((-1,1),real.shape)*eta*.999999
    z=real+1j*imaginary
    data=bloch_data(z,am); baseline=bloch_data(real,am)
    ratios=[float(np.max(np.abs(data[key]-baseline[key])/np.abs(baseline[key]))) for key in ('dminus','dplus')]
    require(max(ratios)<strip_certificate(am)['discriminant_factor_relative_perturbation_upper'], 'Complex roots violate analytic perturbation certificate')
    require(float(np.max(np.abs(data['g'])))<16,'Complex g exceeds analytic bound')
    rootpoints=np.array([[1j*am*factor,0,0] for factor in (.999,1,1.001)])
    roots=bloch_data(rootpoints,am)['dminus']
    require(abs(roots[1])<2e-14,'Adversarial physical complex root at Im p=am not located')
    require(roots[0].real>0 and roots[2].real<0,'Root crossing discriminator failed')
    return dict(sample_count=len(z),seed=seed,imaginary_coordinate_magnitude=eta*.999999,
        factor_relative_perturbation_observed_max=ratios,g_abs_observed_max=float(np.max(np.abs(data['g']))),
        near_root_imaginary_p1=[am*.999,am,am*1.001],near_root_Dminus=[[float(x.real),float(x.imag)] for x in roots],
        nearest_static_axis_root_distance=am,certified_eta_over_root_distance=eta/am,
        scope='Material adverse complex-root samples corroborate, but do not replace, the analytic polytube proof.')


def momentum_grid(L: int, theta: Any) -> np.ndarray:
    n=np.array(list(itertools.product(range(L),repeat=3)))
    return (2*math.pi*n+np.asarray(theta))/L


def physical_periodic_spectrum(gamma: np.ndarray, am: float, L: int, theta: Any) -> dict[str, Any]:
    sites=list(itertools.product(range(L),repeat=3));indices={x:i for i,x in enumerate(sites)}
    links=[]
    for axis in range(3):
        shift=np.zeros((len(sites),len(sites)),complex)
        for x,i in indices.items():
            y=list(x);y[axis]=(y[axis]+1)%L
            shift[i,indices[tuple(y)]]=np.exp(1j*theta[axis]/L)
        links.append(np.kron(shift,np.eye(4)))
    modes=4*len(sites);V=len(sites);zero=np.zeros((modes,modes),complex)
    half,whole_scalar,T=supplier.physical_spatial_onebody(gamma,1/(2*(math.expm1(am)+4)),links,[zero]*4,np.eye(modes))
    ev=eigh(T,eigvals_only=True);hv=eigh(half,eigvals_only=True)
    data=bloch_data(momentum_grid(L,theta),am);E=data['E'].real
    expected=np.sort(np.concatenate((np.repeat(np.exp(-E),2),np.repeat(np.exp(E),2))))
    defect=float(np.max(np.abs(ev-expected)))
    neutral=.5*math.log(whole_scalar)+float(np.sum(np.log(hv[2*V:])))
    analytic=float(np.sum(data['g'].real))
    charged=[.5*math.log(whole_scalar)+float(np.sum(np.log(hv[modes-N:]))) for N in (2*V-1,2*V+1)]
    require(defect<2e-12 and abs(neutral-analytic)<2e-10,'Exact physical Root spectrum / SF vacuum mismatch')
    require(max(charged)-neutral<=-am/2+2e-10,'Charged SF common-neutral bound failed')
    # Independent physical halfstep trace: all hole/particle occupations survive.
    Nt=8192; weights=np.repeat(np.exp(-E*Nt),2);elementary=np.zeros(2*V+1);elementary[0]=1
    for weight in weights:
        elementary[1:]+=weight*elementary[:-1].copy()
    exact_tail=float(np.dot(elementary[1:],elementary[1:]))
    upper=2*(2*V)**2*math.exp(-2*am*Nt)
    require(exact_tail<=upper*(1+1e-8),'Physical canonical thermal bound failed')
    return dict(L=L,V=V,theta=list(theta),Root_spectrum_max_abs_defect=defect,
        reciprocal_pair_max_abs_defect=float(np.max(np.abs(ev*ev[::-1]-1))),
        SF_square_root_max_abs_defect=float(np.max(np.abs(hv**2-ev))),
        minimum_onebody_T_energy_observed=float(np.log(ev[2*V])),
        minimum_neutral_T_pair_energy_observed=float(np.log(ev[2*V]/ev[2*V-1])),
        log_sqrt_detBplus=.5*math.log(whole_scalar),logSF_neutral_exact=neutral,
        logSF_neutral_Bloch=analytic,logSF_vacuum_identity_abs_defect=abs(neutral-analytic),
        charged_log_ratio_to_local_neutral=[x-neutral for x in charged],
        wrong_whole_detB_SF_log_error=.5*math.log(whole_scalar),
        canonical_Nt=Nt,canonical_excited_trace_exact=exact_tail,canonical_excited_trace_upper=upper)


def run() -> dict[str, Any]:
    selected_bytes=(ROOT/SELECTED_NAME).read_bytes();selected=json.loads(selected_bytes)
    parameters=supplier.Parameters(**selected['inputs']['parameters']);gamma,frame=supplier.phi_frame(parameters)
    require(parameters.kappa_power==400 and parameters.beta_power==150 and parameters.spatial_size==2**24 and parameters.time_slices==2**20,'Wrong selected physical epoch')
    require(hashlib.sha256(selected_bytes).hexdigest().startswith('ae1'),'Selected Source capsule must be original ae1 physical_Dminus_verified, not final dd73')
    family=[];native=[];finite=[];complex_rows=[]
    twists=list(itertools.product((0.,math.pi),repeat=3))+[(.37,-1.1,2.4)]
    for index,(a,m) in enumerate(itertools.product(parameters.spacings,parameters.masses)):
        am=float(mp.mpf(str(a))*mp.mpf(str(m)))
        family.append(reference_budget(a,m,parameters.spatial_size,parameters.time_slices,parameters.beta_power))
        complex_rows.append(complex_discriminator(am,20261009+index))
        baseline=None
        for theta in twists:
            row=physical_periodic_spectrum(gamma,am,2,theta)
            if baseline is None: baseline=row['logSF_neutral_exact']
            row.update(a=a,m=m,log_neutral_ratio_to_common_theta0=row['logSF_neutral_exact']-baseline,
                charged_log_ratio_to_common_theta0=[x+row['logSF_neutral_exact']-baseline for x in row['charged_log_ratio_to_local_neutral']])
            native.append(row)
        for L in (4,8,16):
            base=float(np.sum(bloch_data(momentum_grid(L,(0,0,0)),am)['g'].real))
            values=[float(np.sum(bloch_data(momentum_grid(L,t),am)['g'].real))-base for t in twists]
            finite.append(dict(a=a,m=m,L=L,all8_and_generic_twist_log_vacuum_changes=values,
                               analytic_winding_upper=winding_budget(am,L)['log_vacuum_twist_error_upper']))
    require(all(mp.mpf(row['SF_charged_common_neutral_norm_upper'])<1 for row in family),'Macro common-neutral charged contraction failed')
    maximum=max(family,key=lambda x:float(x['kappa']))
    unchanged=all((ROOT/name).read_bytes()==data for name,data in CAPTURED.items())
    require(unchanged,'Captured scientific source changed during execution')
    return dict(plan=PLAN,governing_revision=REVISION,work_item='shared-compton-transfer',
        source_provenance={name:dict(sha256=hashlib.sha256(data).hexdigest(),captured_utf8=data.decode()) for name,data in sorted(CAPTURED.items())},
        selected_source_capsule=dict(path=SELECTED_NAME,sha256=hashlib.sha256(selected_bytes).hexdigest(),
            actual_parameters=selected['inputs']['parameters'],original_active_harmonic_budget=selected['macro']['actual_active_harmonic_canonical_eigenbundle']),
        runtime=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,mpmath=mp.__version__,platform=platform.platform(),argv=sys.argv,loaded_code_matches_captured_source=True,sources_unchanged=unchanged),
        actual_phi_frame=frame,actual_parameter_tuple_count=len(family),family=family,
        maximum_kappa_corner=dict(a=maximum['a'],m=maximum['m'],am=maximum['am'],kappa=maximum['kappa'],
            onebody_T_gap=maximum['onebody_T_gap'],pair_T_gap=maximum['neutral_pair_T_gap'],
            correction='Uniform .01/.02 gaps and .019 separation are invalid across actual all9 tuples; correct global corner is am=.0035/pair=.007.'),
        complex_strip_discriminators=complex_rows,native_exact_physical_Root_periodic_spectra=native,
        finite_periodic_Bloch_twist_sums=finite,
        callable_interface=['bloch_data(momentum,am)','strip_certificate(am)','winding_budget(am,L)','reference_budget(a,m,L,Nt,beta_power=150)'],
        conditions=['Free meanClifford physical Root fiber, zero normal remainders, scalar flat links, actual 4V components.',
                    '0<am<=.03; finite positive L; canonical cold estimate requires x=(2V)^2 exp(-2am p)<=1/2.',
                    'Exact spectral identities are reference facts only. Extension to full interacting Root/source/heat sandwiches requires a separate actual operator comparison.'],
        actual_Grams=None,Root_interacting_gap_proved=False,Pe_admission_proved=False,receiver_admission_proved=False,microscopic_fullpacket_admission_proved=False)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'covariant_physical_mf_vacuum_winding_reusable_import_20261009_capsule.json')
    parser.add_argument('--receipt',type=Path,default=ROOT/'covariant_physical_mf_vacuum_winding_reusable_import_20261009_exit_receipt.json')
    args=parser.parse_args();start=time.monotonic();result=run();result['elapsed_seconds']=time.monotonic()-start
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    digest=hashlib.sha256(args.output.read_bytes()).hexdigest()
    receipt=dict(capsule_path=str(args.output),capsule_sha256=digest,source_sha256=hashlib.sha256(OWN_BYTES).hexdigest(),
        completed_scientific_calculation=True,natural_exit_code=None,natural_exit_observed_by='External launching parent must observe actual process termination; this in-process receipt does not invent an exit code.',
        elapsed_seconds=result['elapsed_seconds'],actual_tuple_count=len(result['family']),physical_Root_rows=len(result['native_exact_physical_Root_periodic_spectra']),
        source_hashes={name:hashlib.sha256(data).hexdigest() for name,data in CAPTURED.items()},actual_Grams=None,microscopic_fullpacket_admission_proved=False)
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(capsule_sha256=digest,actual_tuple_count=len(result['family']),physical_Root_rows=len(result['native_exact_physical_Root_periodic_spectra']),maximum_kappa_corner=result['maximum_kappa_corner'],elapsed_seconds=result['elapsed_seconds'])))


if __name__=='__main__':
    main()
