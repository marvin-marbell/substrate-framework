"""Original-source first moment and actual joined electron/photon quantum beta probes."""
from __future__ import annotations
import argparse
import cmath
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
import induced_real_em_photon_heat_receiver as receiver
quantum=receiver.quantum
SOURCE_BYTES={**receiver.SOURCE_BYTES,Path(__file__).name:OWN_BYTES}
REVISION='1178517cdc4236ce3a4af6a26bed880eb20679df709826463e4db7819ea0d208'


def mean_from_source(heat:dict[str,Any])->dict[str,Any]:
    _,record=quantum.checked(heat['input_descriptor']['input_receipts']['translation'])
    # Actualfrozen provenancechain, not a second native/macromean convention.
    for key in ('response','source','transition','receiver','source_energy','green','response','inserted','force','current','retarded','clock','bridge','mean'):
        _,record=quantum.checked(record['input_descriptor']['input_receipts'][key])
    return record


def initial_source_moment(mean:dict[str,Any],samples:int,seed:int)->dict[str,Any]:
    native=quantum.build_native(mean,2);V,M=native['V'],native['M'];g=64.
    x=int(np.flatnonzero(np.all(native['sites']==(1,1,0),axis=1))[0]);ref=0
    G=native['Lplus'];dipole=G[:,x]-G[:,ref]
    ref_density=np.eye(V)[ref]
    rng=np.random.default_rng(seed);values=[];errors=[];added_traces=[]
    for _ in range(samples):
        theta=native['Q']@(np.sqrt(2)*rng.normal(size=M))
        state=quantum.graph_state(native,native['Pprofiles'][0],theta)
        P=state['P'];Q=np.eye(4*V)-P
        added=Q[:,4*x:4*x+4]@Q[4*x:4*x+4,:]/2
        h=quantum.matter_hamiltonian(native,native['Pprofiles'][0],theta)
        car=float(np.real(np.trace(h@added)))
        blocks=P.reshape(V,4,V,4).transpose(0,2,1,3)
        db=added.reshape(V,4,V,4).transpose(0,2,1,3)
        density=np.real(np.trace(db[np.arange(V),np.arange(V)],axis1=1,axis2=2))
        exchange=np.real(np.einsum('jiab,ijba->ij',blocks,db))
        second=np.diag(density)-2*exchange-np.outer(density,ref_density)-np.outer(ref_density,density)+np.outer(ref_density,ref_density)
        direct=float(g*np.sum(G*second))
        A=np.array([float(np.real(np.trace(blocks[x,j]@blocks[j,x]))) for j in range(V)])
        alternative=float(g*dipole@A)
        errors.append(abs(direct-alternative));added_traces.append(float(np.real(np.trace(added))))
        values.append([car,direct,car+direct])
    if max(errors)>1e-9 or max(abs(value-1) for value in added_traces)>1e-10:
        raise ArithmeticError('Actualsource CAR/Coulomb identity failed')
    return dict(actual_ORIGINAL_Gaussian_source_energy_addition=quantum.summarize(np.asarray(values)),column_order=['active_SOURCE_CAR_addition','primitive_g64_SOURCE_Coulomb_addition','total_SOURCE_addition'],samples=samples,seed=seed,real_EM_modes=M,variance=2,source_added_DeltaP='Q_SOURCE Rx Q_SOURCE/2 atEACHsample; Tr1, NOTQsea',source_added_trace_range=[min(added_traces),max(added_traces)],direct_density_second_moment_vs_chiral_dipole_error=max(errors),aux63_heavy_ref_energy_addition='0 exactly: auxiliaries/heavyunchanged; Refbare1 isELECTRICcharge notmass',actual_SOURCE_electric_Pi_energy_addition_exact='0',electric_proof='E independentrealTheta oncoexactslice sinceQtransverse^Telllongitudinal=0. Gamma5 P Gamma5=1-P =>pointactive density2 and derivative0. Ref4restrictedE†E=(2-rho_x)/4; chiralPH map makes derivativecarrierPi-densitycov0 forEACHrealdirection. SameREALGaussian dOmega cross derivative<rho_x>=0; source normp=.5 pointwise; gPi² formenergyaddition0. DoesNOTmeanPi dynamicsvanish: actualelectricONLYsourceheat569a was.8307 insteadidentity1.',thermal_initial_derivative_identity='r(beta)=KC(beta)/(p KN(beta)), p=.5, SAMEneutraloffset; r(0)=1 and-rprime(0)=<HC>chargedSOURCE-<HN>neutralSOURCE=SOURCEenergyaddition. No stationarystate/lateSlater/massgap assumption; ALLfiniteStranglevels haveSprime(0)=-H soSAMEnativefirstmoment. Finitebeta-logratioNOTpole/mass andrequiresownsampling/Trottererror.' )


def run(heat_path:Path,betas:list[F],samples:int,moment_samples:int,steps:list[int],seed:int)->dict[str,Any]:
    started=time.monotonic();raw=heat_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=receiver.HEAT_SHA:
        raise ArithmeticError('Actualfrozen569a quantumheat ABI required')
    heat=json.loads(raw);mean=mean_from_source(heat)
    moment=initial_source_moment(mean,moment_samples,seed)
    rows={}
    for beta in betas:
        kernels=quantum.actual_quantum_kernel(mean,beta,samples,steps,seed)
        electron={}
        for n,row in kernels['rows'].items():
            stats=row['ALLcomplex_quantum_source_kernels']
            values=np.asarray(stats['real'])+1j*np.asarray(stats['imag'])
            ratio=2*values[1]/values[0]
            logarithmic_slope=-cmath.log(complex(ratio))/float(beta)
            se=np.hypot(np.asarray(stats['sampling_SE_real']),np.asarray(stats['sampling_SE_imag']))
            delta_se=float(2*se[1]/abs(values[0])+2*abs(values[1])*se[0]/abs(values[0])**2)
            electron[n]=dict(actual_p_half_normalized_SOURCE_heat_ratio=dict(real=float(np.real(ratio)),imag=float(np.imag(ratio))),actual_finite_beta_complex_logarithmic_source_slope=dict(real=logarithmic_slope.real,imag=logarithmic_slope.imag),FIRST_ORDER_ONLY_delta_method_ratio_sampling_SE=delta_se,not_confidence_or_finite_denominator_error_certificate=True,not_mass_or_pole=True)
        rows[str(beta)]=dict(actual_all17_real_EM_shared_Coulomb_all64_CAR_kernels=kernels,electron_SOURCE_heat_consumer=electron,photon_SAMEexactGram_physical_receiver={n:receiver.receive(row) for n,row in kernels['rows'].items()})
    descriptor=dict(plan=heat['input_descriptor']['plan'],plan_revision=REVISION,scientific_execution_revision=REVISION,input_receipts=dict(heat=dict(path=str(heat_path),sha256=hashlib.sha256(raw).hexdigest())),producer_input_sha256=heat['input_sha256'],source_abi=heat['input_descriptor']['source_abi'],native_N=2,betas=[str(beta) for beta in betas],samples=samples,moment_samples=moment_samples,steps=steps,seed=seed,a='1/64',UEphys='1',g_electric_AND_Coulomb='64',mass_resolved_flavours=dict(active_count=1,active_mass='.2',auxiliary_count=63,auxiliary_mass='2'),clock4096_UNCHANGED=True,source='UNCHANGEDunfilteredGaussian2/HermY000AP222, nativeMIDPOINTframe, x110/Ref0 rawALL4Refhole E/2, NEVERsourcebetafilter',observable='Joinedactualquantumheatprobes plusDIRECTinitialsourceenergyGaussianmoment, NOTSupplierbetaNormgenerator orphysicalmass/pole')
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),actual_initial_original_source_moment=moment,actual_joined_electron_photon_quantum_beta_rows=rows,scope='FrozenHref ONLY: realall17noncompactPi +sameg64fullsharedrhoLplusrho +all64CAR andRAWcomplexCURRENT/photonpackets. NativeMC/finiteStrang notcontinuumconfidencecertificate. Phi/SU4/heavyquantumevolution absent; no macro/fullPhi thermaladmission orrealTime4096analyticcontinuation, no ground/pole/filter/sourcepromotion. Sharedseed acrossbetas doesNOTsupplyunreportedcrossbeta covariance. PhysicalreceiverSAMEexactGammaA4096I; complexcurrentphotonnotwhitenedwithoutownGram. No broadphysicsretagging.',source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--heat',type=Path,required=True)
    parser.add_argument('--beta',type=F,nargs='+',required=True)
    parser.add_argument('--samples',type=int,required=True)
    parser.add_argument('--moment-samples',type=int,required=True)
    parser.add_argument('--steps',type=int,nargs='+',required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run(args.heat,args.beta,args.samples,args.moment_samples,args.steps,args.seed)
    if not all(result['source_unchanged_since_import'].values()):raise ArithmeticError('Source changed during actual calculation')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(input_sha256=result['input_sha256'],actual_initial_original_source_moment=result['actual_initial_original_source_moment'],actual_electron_beta_rows={b:r['electron_SOURCE_heat_consumer'] for b,r in result['actual_joined_electron_photon_quantum_beta_rows'].items()},elapsed_seconds=result['elapsed_seconds']),indent=2))
