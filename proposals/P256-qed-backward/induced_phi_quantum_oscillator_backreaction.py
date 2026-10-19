"""Actual Phi35+all64+neutralheavy coupledPH Gaussian quantum CONTROL.

PrimitiveLambda8192/chi-4096/cometric andnativePauli normalmode retained.
Actualheavy force/poles neverdeclareEMneutral=Phi-neutral. Nofullvacuum/SOURCE
cutover: GaussianlinearPH truncation, not fullnonlinearPhi/SU4/Pi propagation.
"""
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
ROOT=Path(__file__).resolve().parent
OWN_BYTES=Path(__file__).read_bytes()
import induced_phi_static_backreaction_bound as static
volume=static.volume;control=static.control;pauli=static.pauli;quantum=static.quantum;loops=static.loops;primitive=static.primitive
SOURCE_BYTES={**static.SOURCE_BYTES,Path(primitive.__file__).name:Path(primitive.__file__).read_bytes(),Path(__file__).name:OWN_BYTES}
VOLUME_SHA='7ccf67dddd91d89e085ae9b3efd9a98b052a1a4388af280fb18979edba9237e5'
FEEDBACK_SHA='29c0bd2270a294cd4108a46e302e3f2004cf8c18a7f71da5f13f8ec36bc43d93'


def actual_species(mean:dict[str,Any])->tuple[dict[str,Any],list[dict[str,Any]],list[dict[str,Any]]]:
    native=quantum.build_native(mean,4);V=native['V'];rank=2*V;zero=np.zeros(3*V);mode=pauli.magnetic_shape(native);lv,lu=np.linalg.eigh(native['Lplus']);positive=lv>1e-12
    if int(np.sum(positive))!=V-1:raise ArithmeticError('Actualnative63densitycouplingrank changed')
    density=lu[:,positive]*np.sqrt(2*native['UE']/native['a']*lv[positive])[None,:];rho=[np.diag(np.repeat(d,4)).astype(complex) for d in density.T];parts=[];controls=[]
    for p in native['Pprofiles']:
        h=quantum.matter_hamiltonian(native,p,zero);Fp=pauli.operator(native,p,zero,mode);data=control.spectrum(h,[Fp,*rho],rank)
        for om in (0.,1.,10.,100.):
            B=control.bubble(data,om);mixed=float(np.max(abs(B[0,1:])))
            if mixed>1e-9:raise ArithmeticError('NativePhi35 ALL63density dynamicselection failed')
            controls.append(dict(role=p['label'],actual_Euclidean_frequency=om,actual_Phi35_to_ALL63_density_bubble_max=mixed,actual_Phi35_positive_bubble=float(B[0,0].real)))
        parts.append(dict(role=p['label'],load=p['load'],scale='charged',delta=data['delta'],weight=abs(data['amplitudes'][0])**2,actual_unscaled_gap=data['gap']))
    cfg=primitive.PhiConfig(Lambda=1.,**mean['microscopic_config']);load,model,role=primitive.models(cfg)[0]
    if load!=1 or not role.startswith('neutral') or model.improved:raise ArithmeticError('Actualneutralheavy-minus-Lambda law changed')
    onsite,paths=primitive.scalar_paths(model);hp=[]
    for displacement,coefficients in paths:
        support=[p for p in native['Pprofiles'][0]['paths'] if np.array_equal(displacement,p['displacement'])]
        if len(support)!=V:raise ArithmeticError('ActualneutralheavyAPorderedpathsupport changed')
        hp.extend({**p,'scalars':coefficients,'hop':p['seam']*np.einsum('m,mij->ij',coefficients,native['frame'])} for p in support)
    heavy=dict(load=1,model=model,label=role,onsite=onsite*native['frame'][0],paths=hp);h=-quantum.matter_hamiltonian(native,heavy,zero);Fh=-pauli.operator(native,heavy,zero,mode);data=control.spectrum(h,[Fh],rank)
    parts.append(dict(role=role,load=1,scale='heavy',delta=data['delta'],weight=abs(data['amplitudes'][0])**2,actual_unscaled_gap=data['gap']))
    return native,parts,controls


def coupled_quantum(mean:dict[str,Any])->dict[str,Any]:
    loops.iv.dps=800  # ResolvephysicalLambda^(-1/4) additivequantumbackreaction.
    native,parts,controls=actual_species(mean);iv=loops.iv;a=loops.exact(F(1,64));g0=static.interval(mean,'g0');gs=static.interval(mean,'gs');Lambda=loops.exact(F(2)**8192);chi=loops.exact(F(2)**-4096);qhat2=loops.exact(F(2));G=8*chi
    # ACTUAL CartnormalI/8 embedding/cometric, not flatmatrixB convention.
    embedding=static.embedding(native,np.pi/2);kappa=loops.exact(F(2)**10240)*qhat2/40+Lambda*g0/(2*a);omega=iv.sqrt(G*kappa);variance=G/(2*omega)
    heavy,charged,_,_,_,prices=static.path_prices(mean,a,g0,gs);Bbound=Lambda*heavy+charged;epsilon=Bbound/kappa
    if epsilon.b>=1:raise ArithmeticError('ActualPhi/PH Gaussian stiffness not positive')
    leading=iv.mpf(0);rows=[]
    for p in parts:
        scale=Lambda if p['scale']=='heavy' else iv.mpf(1);energy=iv.mpf(0);bubble=iv.mpf(0);weight_sum=iv.mpf(0)
        for d,w in zip(p['delta'].flat,p['weight'].flat,strict=True):
            delta=loops.exact(F.from_float(float(d)));weight=loops.exact(F.from_float(float(w)));weight_sum+=weight;energy+=variance*scale*scale*weight/(scale*delta+omega);bubble+=2*scale*weight/delta
        energy*=p['load'];bubble*=p['load'];leading+=energy
        rows.append(dict(role=p['role'],load=p['load'],physical_scale=p['scale'],actual_unscaled_PH_gap=p['actual_unscaled_gap'],actual_unscaled_Phi35_PH_Frobenius_squared=loops.bounds(weight_sum),actual_numerical_spectral_Phi35_static_bubble=loops.bounds(bubble),actual_numerical_spectral_second_order_quantum_vacuum_lowering_magnitude=loops.bounds(energy),spectral_input='ActualnativeHermitianeigh/bothPHbands. Binary64spectralvaluesareobservednumericalcontrols, NOTunpricedexactmicroscopicintervals.'))
    # For the ACTUAL scalarpositiveGaussian PH Hamiltonian, r(omega)<=eps.
    # -log(1-r) between r and r/(1-eps), so covariance and energy envelopes
    # follow without tryingtofloatLambda or resolveoscillatoryhugefrequency.
    maximum=variance*Bbound/2/(1-epsilon);varhigh=variance/(1-epsilon);relative=epsilon/(1-epsilon)
    return dict(N=4,V=64,actual_Lambda='2^8192',actual_chi='2^-4096',actual_qtheta='pi/2',actual_Cartnormal_Phi_cometric=loops.bounds(G),actual_fullprimitive_alignment_PLUS_quartic_Pauli_stiffness=loops.bounds(kappa),actual_free_Phi_quantum_frequency=loops.bounds(omega),actual_free_Phi_variance=loops.bounds(variance),actual_coupled_Gaussian_Phi_variance_enclosure=dict(lower=variance.a.__str__(),upper=varhigh.b.__str__(),free_bounds=loops.bounds(variance),multiplicative_relative_increase_upper=loops.bounds(relative)),actual_GLOBALcoupling_r_over_1minus_r_upper=loops.bounds(relative),actual_fullGaussian_vacuum_energy_lowering_magnitude_upper=loops.bounds(maximum),actual_observed_spectral_second_order_vacuum_lowering_magnitude=loops.bounds(leading),actual_numerical_scalar_Gaussian_higher_order_relative_remainder_upper=loops.bounds(epsilon/(2*(1-epsilon))),actual_ALLspecies_spectral_force_and_energy_consumers=rows,actual_ALL63density_dynamic_Phi_selection_controls=controls,actual_full90primitive_embedding=embedding,actual_nonperturbative_species_operator_envelopes=prices,quantum_derivation='ActualH_gauss=.5G Pphi²+.5kappa phi²+sum Delta bdag b+phi sum(Fph bdag+Fph* b)+.5rho². ALL64chargedphysicalF/Delta, actualneutralheavy -Lambda h/F givesforceLambda andDeltaLambda. Phi35densitycouplingzerofollowscompleteCliffordtrace: rho=I, hspanGamma0..3, F=Gamma0 iGamma1Gamma2 timesactualspatialscalar, everymixedtraceoddGamma ortrGamma5=0. ALL63dynamicmixedbubbles exercised, so unchangedpositiveCoulombdensitybath decouplesONLYfromlinearPhi35Gaussiansector, NOTfromfullnonlinearh(Phi); prior29c nonzeroPhi-densityquantumfeedback retained. ScalarGaussianresolventD=[omega_E²/G+kappa-BPhi(omega_E)]^-1, canonicalpositive r=G BPhi/(omega_E²+Omega²)<=Bbound/kappa. ExactGaussianlogdetenergynegative; variancefree<=variancecoupled<=variancefree/(1-eps); magnitudeDeltaE<=variancefree Bbound/[2(1-eps)]. NumericalsecondorderDeltaE=-sum variancefree |F|²/(Delta+Omega) actuallyexecuted, exactcanonicalhigherGaussianlogtail boundedbyeps/[2(1-eps)] relativewithinthenumericallyspecifiedspectralmodel. No bareB², fittedmodefrequency, floatingLambda, ignoredheavyforce orzero-valuedunderflow.',scope='NEWscalarPhi35+PH harmonicquantumCONTROL usingactualprimitiveCartalignment/quartic/cometric andALLcharged+heavy spectralF. It isNOTfull90Phi/SU4/Pi/fermionnonlinearGaussvacuum, originalcanonicalY/E/Ref3 SOURCE ororiginalclock4096 transfer. PositivequarticPhi4 andPhi bdag b, actualfullPhi/SU4 chart/Noether/nonlinear terms areNOTpromotedasquantized. Sameparameters/lawbytespreserved; NEWGaussianapproximationexplicit. NumericalleadingenergyisNOTmicroscopicrigorousinterval; operatorenvelopespriceGaussianmodelindependently.')


def run(volume_path:Path,feedback_path:Path)->dict[str,Any]:
    started=time.monotonic();raw=volume_path.read_bytes();other=feedback_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=VOLUME_SHA or hashlib.sha256(other).hexdigest()!=FEEDBACK_SHA:raise ArithmeticError('Frozen7cc/29c actualquantumPhi inputepochs required')
    cap=json.loads(raw);mean=cap['input_descriptor']['actual_root_mean_input'];feedback=json.loads(other)
    if mean!=feedback['input_descriptor']['actual_root_mean_input']:raise ArithmeticError('ActualPhi root epochs disagree')
    desc=dict(plan=cap['input_descriptor']['plan'],plan_revision=pauli.REVISION,scientific_execution_revision=pauli.REVISION,input_receipts=dict(volume=dict(path=str(volume_path),sha256=hashlib.sha256(raw).hexdigest()),Phi_quantumdensity=dict(path=str(feedback_path),sha256=hashlib.sha256(other).hexdigest())),actual_root_mean_input=mean,original_source_abi_UNCHANGED=cap['input_descriptor']['original_source_abi_UNCHANGED'],actual_parameters=dict(Lambda='2^8192',chi='2^-4096',a='1/64',g_Pi_AND_Coul='64',N=4,original_clock_UNCHANGED='4096'),branch='NEWcoupledPhi35+allheavy/allchargedPHGaussianquantumCONTROL; noSOURCE/vacuumpromotion')
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),actual_quantum_Phi_heavy_charged_PH_oscillator_backreaction=coupled_quantum(mean),source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--volume',type=Path,required=True);p.add_argument('--feedback',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run(a.volume,a.feedback)
    if not all(r['source_unchanged_since_import'].values()):raise RuntimeError('Sourcechanged duringactualPhi quantumoscillator execution')
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(input_sha256=r['input_sha256'],physical=r['actual_quantum_Phi_heavy_charged_PH_oscillator_backreaction'],elapsed_seconds=r['elapsed_seconds']),indent=2))
