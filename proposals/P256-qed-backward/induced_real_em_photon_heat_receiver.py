"""Physical two-polarization receiver of frozen real-Pi/shared-Coulomb quantum heat."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np

ROOT=Path(__file__).resolve().parent
OWN_BYTES=Path(__file__).read_bytes()
import induced_real_em_coulomb_heat as quantum
SOURCE_BYTES={**quantum.SOURCE_BYTES,Path(__file__).name:OWN_BYTES}
REVISION='1178517cdc4236ce3a4af6a26bed880eb20679df709826463e4db7819ea0d208'
HEAT_SHA='94be77a706af8620f1f514e7074056bc7930a836c88736edb482173edfe0226b'


def matrix(stats:dict[str,Any])->np.ndarray:
    return np.asarray(stats['real'],dtype=float)+1j*np.asarray(stats['imag'],dtype=float)


def encode(value:np.ndarray)->dict[str,Any]:
    return dict(real=np.real(value).tolist(),imag=np.imag(value).tolist())


def receive(row:dict[str,Any])->dict[str,Any]:
    a=1/64
    # Source norm at EVERYfield =p=.5. OriginalunfilteredThetaGaussian s=2.
    # GammaTheta=p*s I=I; GammaA=GammaTheta/a²=4096I EXACT, no samplefit.
    root_inverse_A=a*np.eye(2)
    physical_A_heat=matrix(row['charged_photon_SOURCE_heat_kernel'])/a**2
    transfer=root_inverse_A@physical_A_heat@root_inverse_A
    # SharedrealA0/realBrownian measure has endpoint/path reverseinvolution.
    # K(reverse)=Kdag: exactHermitianpairing, not |det| orPSDclipping.
    paired=(transfer+transfer.conj().T)/2
    values,vectors=np.linalg.eigh(paired)
    U=np.array([[1,1],[1j,-1j]],dtype=complex)/np.sqrt(2)
    stats=row['charged_photon_SOURCE_heat_kernel']
    marginal_sample_L2_SE=np.hypot(np.asarray(stats['sampling_SE_real']),np.asarray(stats['sampling_SE_imag']))
    packets={}
    for i,name in enumerate(('circular_plus','circular_minus')):
        u=U[:,i]
        scalar=complex(np.vdot(u,transfer@u))
        hermitian_scalar=complex(np.vdot(u,paired@u))
        # Triangleinequality for centeredSAMPLEvectors, keepsunknowncrosscov.
        # This is a conservative SAMPLESE bound, NOT confidencecertificate.
        se=float(np.sum(np.outer(np.abs(u),np.abs(u))*marginal_sample_L2_SE))
        packets[name]=dict(normalized_physical_A_source_coefficients=encode(a*u),exact_source_norm_squared='1',actual_raw_complex_thermal_value=dict(real=scalar.real,imag=scalar.imag),actual_reverse_paired_thermal_value=hermitian_scalar.real,conservative_complex_sampling_SE_upper=se,WRONG_omitting_imaginary_z_source_norm_squared='1/2')
    current_photon=matrix(row['charged_complexCURRENT_photon_SOURCE_heat_kernel'])
    return dict(exact_original_source_Gram_physical_A=[['4096','0'],['0','4096']],exact_original_source_Gram_Theta=[['1','0'],['0','1']],actual_physical_A_heat_matrix=encode(physical_A_heat),actual_same_Gram_normalized_thermal_transfer=encode(transfer),actual_reverse_involution_paired_transfer=encode(paired),actual_same_Gram_circular_transfer=encode(U.conj().T@paired@U),paired_estimated_transfer_eigenvalues=values.tolist(),paired_estimated_eigenvectors=encode(vectors),PSD_clipping=False,actual_raw_antiHermitian_sampling_norm=float(np.linalg.norm(transfer-transfer.conj().T,2)),packet_rows=packets,WRONG_creator_only_p_half_as_photon_Gram_transfer=encode(2*transfer),WRONG_Gram_transfer_change_norm=float(np.linalg.norm(transfer,2)),actual_complexCURRENT_physical_JA_times_A_source_heat_matrix=encode(current_photon),current_photon_receiver_normalization='NOTperformed: thispacketGamma isNOTp*sI; actualcomplexCURRENT SOURCEGram requires itsOWN originalGaussianintegral. No sourceGramfit/substitution.',marginal_complex_SAMPLE_SE_matrix=marginal_sample_L2_SE.tolist(),statistical_and_Strang_error_certified=False)


def run(heat_path:Path)->dict[str,Any]:
    started=time.monotonic();raw=heat_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=HEAT_SHA:
        raise ArithmeticError('Frozen569ae5 realPi/Coulomb heat receipt required')
    heat=json.loads(raw);kernels=heat['actual_real_EM_Coulomb_quantum_kernels']
    if (kernels['N'],kernels['M'],kernels['ALLcharged_species'],kernels['variance'],kernels['g_electric_AND_Coulomb'],kernels['Ref_N_to_C'])!=(2,17,64,2.,64.,[4,3]):
        raise ArithmeticError('No mode/law/source/Ref substitution')
    descriptor=dict(plan=heat['input_descriptor']['plan'],plan_revision=REVISION,scientific_execution_revision=REVISION,input_receipts=dict(heat=dict(path=str(heat_path),sha256=hashlib.sha256(raw).hexdigest())),producer_input_sha256=heat['input_sha256'],source_abi=heat['input_descriptor']['source_abi'],native_N=2,beta=kernels['beta'],steps=kernels['Strang_steps'],samples=kernels['samples'],seed=kernels['seed'],a='1/64',UEphys='1',charged_flavours=64,receiver='ORIGINALrawALL4RefholechargedSOURCEphotonA_y/A_z packets; SAMEexactGram onboththermalcomparands; (Ay±iAz)/sqrt2 physicalcoefficientsa, no filter/gate/fit',clock4096_UNCHANGED=True,macro_and_fullPhi_claim=False)
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),actual_physical_photon_thermal_receiver={n:receive(row) for n,row in kernels['rows'].items()},source_Gram_proof='PointwiseALL4Refhole source E†E normp=.5 atEVERYTheta byGamma5HermYspintrace2/Ref4. Entireunfilteredrealcoexact+harmonyGaussian variance2 gives <Theta_iTheta_j>=2deltaij foractualunitorthogonalnativeNyquisty/zmodes. Hence GammaTheta_ij=<Theta_iEpsi,Theta_jEpsi>=p*2deltaij=deltaij EXACT, independentinteractingheat. A=Theta/a givesGammaA=4096I; inverseRoot=aI; T=GammaA^-1/2 KA GammaA^-1/2=KTheta, withSAMEoriginalfullpacketGram and no reference/sampledGram replacement. Circularphysicalreceivercoefficients a(1,±i)/sqrt2 yieldEXACTunitSOURCEpacketnorm; deletingiz losesHALFnorm. Creator-onlyp=.5 isNOTphotonpacketGram.',quantum_operator_proof='AtfixedStrang S=e^-dtT/2 e^-dtHCAR/2 e^-dtC e^-dtHCAR/2 e^-dtT/2, T=g64Pi²,C=g64rhoLplusrho positiveprimitiveoperators, HCARboundedselfadjointfiniteCAR. The sandwichedoperator S isselfadjointpositive andS^n positive. ExactSOURCEthermalGram isHermitianPSD regardlessnonstationarySOURCE. Same-neutralconstantoffset multipliespositiverealnumber, notchargedownphase. RealBrownian/all7realA0 HS expectation restoresoperator. Reverseendpoint/Brownianwords andA0reverse+signflip formmeasurepreservinginvolution withKreverse=Kdag, permitsHermitianaveragingwithoutphasequenching. Estimatedeigenvalues/SE NOTPSD/continuumcertificate; no absdet/positivityclip.',physical_scope='ACTUALconsumer of569ae5 realALL17EMquantum+sharedCoulomb+all64CAR sourceheat, finiteStrang/MC atdeclaredbeta1/4096, physicalAconversionandnormalizedcircularreceiver. NOT retardedGreen, survival/preparationgate, sourcebetaNormgenerator, photonpole/gap/commoncone/scattering, fullPhi/SU4/heavy heat. Current×photonrawphysicalpacketheat consumed withneta0 butNO numericalcurrentGramwhitening. Frozen17f/e363/5f26/priorbytes unchanged; realunitary4096bounds NOT analyticallycontinued.',source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--heat',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run(args.heat)
    if not all(result['source_unchanged_since_import'].values()):raise ArithmeticError('Source changed during actual calculation')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('input_sha256','actual_physical_photon_thermal_receiver','elapsed_seconds')},indent=2))
