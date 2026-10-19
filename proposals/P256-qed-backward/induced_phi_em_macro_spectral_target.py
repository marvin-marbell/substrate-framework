"""Actual aligned-source normalized spectral target/global bounds; no pole gate."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
from induced_phi_em_macro_aligned_certified import SOURCE_BYTES as ALIGNED_BYTES
from induced_phi_em_macro_preparation import scalars,build_native,centre
from induced_phi_em_gaussian_variational import matter_hamiltonian
from induced_phi_em_gain_certificate import IV,iv,endpoints,encoded
from induced_phi_ir_well_transfer import rational,sqrt_bounds
ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**ALIGNED_BYTES,Path(__file__).name:Path(__file__).read_bytes()}
REVISION='b9e13b2491321aa804f7e52cf4ba3cb12c0a224717cb07b23e487afc0768463b'


def run(source_path: Path,clock: int) -> dict[str,Any]:
    started=time.monotonic();raw=source_path.read_bytes();source=json.loads(raw);state=source['input_descriptor'];N=int(state['N']);V=int(state['V']);M=int(state['M']);s=rational(state['variance']);a=F(float(state['a']));g=F(float(state['UE']))/a
    if state['kappa']!=0 or rational(state['eta_preparation_ONLY'])!=F(7,20) or s!=F(3,5) or N!=262144 or state['charged_counts']!=[1,63] or clock<0:raise ValueError('No source/preparation/clock substitution')
    inputs={}
    for role in ('mean','refiner'):
        spec=state['inputs'][role];data=(ROOT/spec['path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=spec['sha256']:raise ArithmeticError('Actual input epoch drift')
        inputs[role]=json.loads(data)
    IV.dps=65;mids,errors=centre(inputs['mean'],inputs['refiner']);g0upper=mids[0]+errors[0];gsupper=mids[1]+errors[1];root0=sqrt_bounds(g0upper)[1];roots=sqrt_bounds(gsupper)[1]
    native=build_native(inputs['mean'],6);coeffs=[scalars(p) for p in native['Pprofiles']];perV_lower_shift=F(0)
    for p in coeffs:
        hmax=root0*abs(p['onsite'])+6*sum((root0*abs(p['path'][l][0])+roots*abs(p['path'][l][1]) for l in (1,2)),F(0));perV_lower_shift+=2*p['load']*hmax
    B=V*perV_lower_shift;Eupper=V*rational(source['energy_per_V_upper']);contact=[rational(x) for x in source['reference_literal_Theta_contact_interval']];relative=[g/(2*s)+s*x for x in contact]
    normalized_energy_upper=Eupper+relative[1];shifted_source_upper=Eupper+B;shifted_conditioned_upper=normalized_energy_upper+B
    if min(shifted_source_upper,shifted_conditioned_upper)<=0:raise ArithmeticError('Actual Href positive-shift bound invalid')
    # Exact positive-semigroup spectral Markov bound, source-bound and conserved at ALLtimes.
    spectral_cut=100*shifted_conditioned_upper;tail=shifted_conditioned_upper/spectral_cut
    theta_source_upper=sqrt_bounds(s)[1]+2*g*clock*sqrt_bounds(shifted_source_upper/g)[1]
    theta_conditioned_upper=sqrt_bounds(3*s)[1]+2*g*clock*sqrt_bounds(shifted_conditioned_upper/g)[1]
    retarded_upper=2*sqrt_bounds(s)[1]*theta_source_upper
    shifted=np.zeros(3*native['V']);shifted[1::3]=2*np.pi
    theta=native['Q']@(np.linspace(-.25,.381,native['M']));controls=[]
    for p in native['Pprofiles']:
        h=matter_hamiltonian(native,p,theta);hp=matter_hamiltonian(native,p,theta+shifted)
        controls.append(dict(load=p['load'],literal_primitive_2pi_harmonic_translation_max_error=float(np.max(abs(h-hp)))))
    gauge_error=float(np.linalg.norm(native['D']@shifted));wrong_shift=shifted/2
    wrong=float(np.max(abs(matter_hamiltonian(native,native['Pprofiles'][0],theta+wrong_shift)-matter_hamiltonian(native,native['Pprofiles'][0],theta))))
    if max(row['literal_primitive_2pi_harmonic_translation_max_error'] for row in controls)>1e-10 or gauge_error>1e-10 or wrong<1:raise ArithmeticError('Literal quantum translation discriminator failed')
    log_overlap=endpoints(-IV.pi**2*iv(F(V)/(2*s)))
    desc=dict(source_sha256=hashlib.sha256(raw).hexdigest(),source_input_sha256=source['input_sha256'],N=N,V=V,M=M,a=state['a'],UE=state['UE'],variance=encoded(s),clock=clock,plan=state['plan'],plan_revision=REVISION,condition='chi_f=Theta_f Psi/sqrt(s), f_y(x)=sqrt(2/V)cos(2pi x/N), literalDf0/norm1, sameactualsource',control_N=6)
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),actual_conditioned_norm_squared=encoded(F(1)),actual_conditioned_source_overlap=encoded(F(0)),actual_same_Href_conditioned_relative_energy_interval=[encoded(x) for x in relative],actual_Href_positive_lower_shift=encoded(B),actual_shifted_source_energy_upper=encoded(shifted_source_upper),actual_shifted_conditioned_energy_upper=encoded(shifted_conditioned_upper),conditioned_positive_spectral_cut=encoded(spectral_cut),conditioned_spectral_tail_probability_upper=encoded(tail),alltime_source_norm_squared=encoded(F(1)),alltime_conditioned_norm_squared=encoded(F(1)),clock_source_Theta_norm_upper=encoded(theta_source_upper),clock_conditioned_Theta_norm_upper=encoded(theta_conditioned_upper),clock_retarded_commutator_absolute_upper=encoded(retarded_upper),literal_translation_controls=controls,literal_translation_Gauss_constraint_error=gauge_error,wrong_pi_translation_defect=wrong,actual_macro_adjacent_Gaussian_translation_log_overlap_interval=[encoded(x) for x in log_overlap],
        spectral_target='Href isSAMEoriginalfrozenmean fullquantumCAR/realGauss/Pi/Coulomb/all64 generator, noth_eta orRPA. Positiveunitmass probabilitynu_f(S)=<chi_f,1_S(Href+B)chi_f>; survivalA_f(t)=<chi_f,exp[-i(Href+B)t]chi_f>=integral exp(-i lambda t)nu_f(dlambda), ALLrealt. Thisisnormalizedsame-source CONDITIONED spectralfunctional, notstationaryvacuum Lehmannphoton spectraldensity; Esource expectation notground/centralfrequency. Actualtailnu(lambda>=R)<=<H+B>chi/R, alltimeunitarynorm andenergy conserved.',
        conditioned_proof='GaussianfixedcentreCAR independentTheta: ||Theta_fPsi||²=s and<Theta_f>=0 exactly. MultiplicationbyTheta commutesphysicalGauss/colour, sochi isphysicalandnormalized withoutsea-overlap division. AllnonThetaPhi/colour/neutralheavy/Coulomb energiescancel inconditionalrelativeenergy; electricPi shiftg/(2s), weightedGaussianPeierls shift s<Contact_f>. Href shift intervalexact; correspondingfullPhi shift needsseparateparentcontact-price/domainreview.',
        global_bound_proof='Literalonsite+sixunitaryaxialshifts onebody hmax<=sqrtg0|mass|+6sum_l(sqrtg0|F0|+sqrtgs|Fj|). Actualeachflavourparticlecount2V => matter>=-B; retainfullCpositive/Pi, Href+B>=gPi_Q². Fixedsourceenergy/Cupper givesalltimePi norm bound. Theta_R corecommutator/energy conservation yield||Theta_fU(t)v||<=||Theta_fv||+2g|t|sqrt(<H+B>v/g), exactforALLt, notTaylor/extrapolated4096clock. Cauchy commutator<=2sqrt(s)||ThetaU(t)Psi||. Boundiswide, NO fullpropagatorvalueoruseful4096transfer claimed.',
        actual_full_H_compactness_audit='AtACTUALfiniteN allnoncompactharmonics retained. Translateharmoniccoordinate q_y by2pi sqrtV, i.e.ALLyedgeTheta+2pi. Everyoriginalone/twolinkphase gains2pi*integer, evenwitharbitraryPhi/noncommutingSU4, sofullPeierlsHunchanged; heavyEM0, Phi/colourkinandpotentialsunchanged; Pi²translationinvariant; rhoLrhoThetaindependent; Dshift0 andGaussunchanged. T commutesactualfullH/Friedrichs/core, notjustHref. AcompactEMsmoothcorepacket translatedbywidelyspacedintegerTpowershasdisjointsupport/norm1 andsamefiniteHgraphnorm; henceglobalD(H)graphembedding NOTcompact andfullresolvent NOTcompact. Gaussianpreparedstates translatedinsteadhaveoverlap exp[-pi²V(n-m)²/(2s)]→0. ThisdoesNOTproveabsenceofband/poles/pointeigenvalues orpurelyac spectrum; separateFloquet/weightedThomas analysis required, noisolatedpolegate imposed. Realharmoniccompactification wouldchangefrozenlaw andisnotperformed.',
        limitations='Actualnormalizedconditioned spectralmeasure/globalbounds, notnumericallyevaluatedfullH timekernel/4096propagator. Href consumer distinct fromfullPhi; no stationaryvac/pole/residue/commoncone/Compton/statP-Q promotion. Infinitevolumetightness NOTclaimed: positivecutscalesactualV; graphnoncompactness isactualfiniteN.',source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for role in ('source','output'):parser.add_argument('--'+role,type=Path,required=True)
    parser.add_argument('--clock',type=int,required=True);args=parser.parse_args();report=run(args.source,args.clock);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({key:report[key] for key in ('input_sha256','actual_same_Href_conditioned_relative_energy_interval','conditioned_spectral_tail_probability_upper','clock_retarded_commutator_absolute_upper','literal_translation_controls','wrong_pi_translation_defect','elapsed_seconds')},indent=2))
