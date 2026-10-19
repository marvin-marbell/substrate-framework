"""Original-source charged Green first moment and two-source receiver comparison.

Native evolution is exact full64 charged FREE-CAR component at fixed fields,
explicitly not the full quantum EM/Coulomb propagator. Macro first-moment and
receiver prices retain the original full Phi/EM Hamiltonian and neutral phase.
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
from induced_phi_em_compensated_source_energy import SOURCE_BYTES as ENERGY_BYTES
from induced_phi_em_ygraph_clock_transfer import checked, exact
from induced_phi_em_ygraph_initial_contact import positive_exact
from induced_phi_em_ygraph_evolving_field_clock import su
from induced_phi_em_macro_preparation import build_native
from induced_reference_electric_graph import graph_state

ROOT = Path(__file__).resolve().parent
SOURCE_BYTES = {**ENERGY_BYTES,Path(__file__).name:Path(__file__).read_bytes()}
REVISION = '6ebc34a912158254456c535bd10259fca04201b14f446a778bba68d5ef63ecf2'


def actual_two_sector_car(mean: dict[str, Any]) -> dict[str, Any]:
    native = build_native(mean,2)
    V = native['V']
    positions = [(1,1,0),(1,0,1)]
    endpoints = [int(np.flatnonzero(np.all(native['sites']==p,axis=1))[0]) for p in positions]
    states = [graph_state(native,p,np.zeros(3*V)) for p in native['Pprofiles']]
    active, aux = states
    if (active['load'],aux['load']) != (1,63):
        raise ArithmeticError('Actual active1/aux63 profiles required')
    P, Q, occupied = active['P'],np.eye(4*V)-active['P'],active['occupied']
    spectra = [np.linalg.eigh(s['physical_h']) for s in states]
    row0 = np.array([[np.trace(Q[4*x:4*x+4,4*y:4*y+4])/4 for y in endpoints] for x in endpoints])
    source_variance = sum(s['load']*np.trace(s['P']@s['physical_h']@(np.eye(4*V)-s['P'])@s['physical_h']).real for s in states)
    matter_additions = []
    derivative_covariances = []
    for x in endpoints:
        Rx = np.zeros_like(P)
        Rx[4*x:4*x+4,4*x:4*x+4]=np.eye(4)
        matter_additions.append(float(np.trace(active['physical_h']@Q@Rx@Q).real/2))
        derivative_covariances.append(complex(np.trace(active['physical_h']@P@Rx@Q)))
    rows = []
    for t in (0.,1/128,1/64):
        unitaries = [(vectors*np.exp(-1j*t*values))@vectors.conj().T for values,vectors in spectra]
        U, Uaux = unitaries
        evolved = U@occupied
        Qevolved = U@Q@U.conj().T
        aux_evolved = Uaux@aux['occupied']
        true_aux = np.linalg.det(aux_evolved.conj().T@aux_evolved)**63
        wrong_aux = np.linalg.det(aux['occupied'].conj().T@aux_evolved)**63
        compact = np.array([[np.trace((U@Q)[4*x:4*x+4,4*y:4*y+4])/4 for y in endpoints] for x in endpoints])
        actual, frozen = np.zeros((2,2),complex),np.zeros((2,2),complex)
        for i,x in enumerate(endpoints):
            for j,y in enumerate(endpoints):
                for a in range(4):
                    bra_added = Qevolved[:,4*x+a]
                    ket_added = U@Q[:,4*y+a]
                    true_bra = np.column_stack((evolved,bra_added))
                    ket = np.column_stack((evolved,ket_added))
                    actual[i,j] += np.linalg.det(true_bra.conj().T@ket)*true_aux/4
                    frozen_bra = np.column_stack((occupied,Q[:,4*x+a]))
                    frozen[i,j] += np.linalg.det(frozen_bra.conj().T@ket)*wrong_aux/4
        defect = float(np.linalg.norm(actual-compact,2))
        wrong = float(np.linalg.norm(frozen-actual,2))
        if defect>1e-10 or (t>0 and wrong<1e-3):
            raise ArithmeticError('Actual two-sector CAR Green/frozen-neutral-bra discrimination failed')
        rows.append(dict(t=t, actual_TWO_sector_full64_FREE_CAR_Green=dict(real=actual.real.tolist(),imag=actual.imag.tolist()), exact_onebody_UQ_Green=dict(real=compact.real.tolist(),imag=compact.imag.tolist()), true_TWOsector_identity_error=defect, WRONG_frozen_neutral_bra_full64_CAR_Green=dict(real=frozen.real.tolist(),imag=frozen.imag.tolist()), WRONG_frozen_neutral_bra_operator_error=wrong))
    if np.linalg.norm(row0-.5*np.eye(2))>1e-10 or source_variance<=1e-3 or max(abs(c) for c in derivative_covariances)>1e-10:
        raise ArithmeticError('Actual source Gram/nonstationary/common-phase control failed')
    return dict(N=2,V=V,source_positions=[list(p) for p in positions],Ref_position=[0,0,0],active_particles_N=2*V,active_particles_C=2*V+1,auxiliary_count=63,auxiliary_particles_each=2*V,Ref_particles_N=4,Ref_particles_C=3,actual_TWO_source_INITIAL_CREATOR_Gram=dict(real=row0.real.tolist(),imag=row0.imag.tolist()), actual_full64_free_CAR_source_variance=float(source_variance), actual_diagonal_free_CAR_addition_first_moments=matter_additions, actual_neutral_CAR_energy_density_covariances=[dict(real=c.real,imag=c.imag) for c in derivative_covariances],evolution=rows,scope='EXACTfull32mode active/auxiliary Slaters and63auxdeterminants atTheta0/nativePhi000, bothN/C freeCARunitaries/Refhole. Actual component ofHref only: NOquantumEMPi/Coulomb/compactPhi propagation/pole inferred. Macroboundbelow usesoriginalFULL_H, notthisfreeCAR control.')


def run(energy_path: Path) -> dict[str, Any]:
    started=time.monotonic()
    raw=energy_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!='c62fe6dd74a2200414ae0bef990e479377ff9f9634467e3756296563daee92e2':
        raise ArithmeticError('Actualdcc fullSOURCEenergy epoch required')
    energy=json.loads(raw)
    if energy['source_at_import']['induced_phi_em_compensated_source_energy.py']['sha256']!=hashlib.sha256(SOURCE_BYTES['induced_phi_em_compensated_source_energy.py']).hexdigest():
        raise ArithmeticError('Actual SOURCEenergy source changed')
    _,green=checked(energy['input_descriptor']['input_receipts']['green'])
    _,response=checked(green['input_descriptor']['input_receipts']['response'])
    _,inserted=checked(response['input_descriptor']['input_receipts']['inserted'])
    _,force=checked(inserted['input_descriptor']['input_receipts']['force'])
    _,current=checked(force['input_descriptor']['input_receipts']['current'])
    _,retarded=checked(current['input_descriptor']['input_receipts']['retarded'])
    _,field=checked(retarded['input_descriptor']['input_receipts']['clock'])
    _,bridge=checked(field['input_descriptor']['input_receipts']['bridge'])
    _,mean=checked(bridge['input_descriptor']['input_receipts']['mean'])
    gd=green['input_descriptor']
    if (gd['N'],gd['V'],gd['a'],gd['UEphys'],gd['Lambda'],gd['variance'],gd['clock'])!=(262144,2**54,'1/64','1','2^8192','2','4096') or energy['actual_creation_probability_exact']!='1/2':
        raise ArithmeticError('Original full law/source required')
    covariance_price=exact(energy['original_neutral_carrier_Phi_colour_kinetic_upper'])+exact(bridge['corrected_uniform_all_Theta_all_CAR_charged_Phi_freezing_action'])
    moment_lower=exact(energy['full_same_H_compensated_SOURCE_energy_addition_positive_lower'])-covariance_price
    moment_upper=exact(energy['full_same_H_compensated_SOURCE_energy_addition_upper'])+covariance_price
    source_norm=su(F(1,2))
    insertion=exact(green['original_FULL_vs_reference_initial_CREATOR_error'])
    full_Gram_offdiag=2*source_norm*insertion
    full_Gram_lower=F(1,2)-full_Gram_offdiag
    green_operator_error=2*exact(green['alltime_NONSTATIONARY_original_FULL_charged_Green_comparison_error'])
    if full_Gram_lower<=0 or moment_lower<=0 or green_operator_error>=F(1,1000):
        raise ArithmeticError('Original-source receiver/moment inequality failed')
    desc=dict(plan=gd['plan'],plan_revision=REVISION,scientific_execution_revision=REVISION,input_receipts=dict(source_energy=dict(path=str(energy_path),sha256=hashlib.sha256(raw).hexdigest())),source_abi=gd['source_abi'],N=gd['N'],V=gd['V'],M=gd['M'],a=gd['a'],UEphys=gd['UEphys'],Lambda=gd['Lambda'],variance=gd['variance'],clock=gd['clock'],source_positions=[gd['source_position'],gd['receiver_position']],Ref_position=gd['Ref_position'],original_preparation=gd['neutral_original_preparation'],creator=gd['charged_insertion'],common_energy_convention=gd['common_energy_convention'],consumer='TWO-source rawGij(t)=<UNPsi,E_i† UC E_jPsi>; SAMEexactfullINITIALsourceGram usedonbothsides, notsourcewhitening orreplacementbylowerbound',native_control=dict(N=2,positions=[[1,1,0],[1,0,1]],theta='0',t=[0,'1/128','1/64'],fields='STATICnativePhi000 andTheta0 actualoriginalchargedmatter component ONLY'))
    values=dict(actual_FULL_source_normalized_diagonal_Green_first_imaginary_moment_positive_lower=moment_lower,actual_FULL_source_normalized_diagonal_Green_first_imaginary_moment_upper=moment_upper,original_neutral_energy_creation_Gram_covariance_price=covariance_price,actual_FULL_initial_TWOsource_creation_Gram_offdiagonal_absolute_upper=full_Gram_offdiag,actual_FULL_initial_TWOsource_creation_Gram_minimum_eigenvalue_lower=full_Gram_lower,alltime_FULL_vs_Href_TWOsource_charged_Green_operator_error=green_operator_error,SAME_EXACT_full_INITIAL_TWOsource_Gram_relative_charged_Green_error=green_operator_error/full_Gram_lower)
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),**{k:positive_exact(v) for k,v in values.items()},actual_FULL_CREATOR_Gram_diagonals_exact=['1/2','1/2'],actual_reference_TWOsource_CREATOR_Gram_exact='(1/2) I2',actual_FULL_source_diagonal_Green_at0_exact='1/2',actual_FULL_source_diagonal_Green_derivative_realpart_at0_exact='0',actual_native_two_sector_CAR_receiver=actual_two_sector_car(mean),
        full_NONSTATIONARY_first_moment_proof='ForD(H_N)originalPsi/D(H_C)E_iPsi, Gii(t)=<UNPsi,E_i† UC E_iPsi> hasGii(0)=p1/2 andGii_prime(0)/p=-i[DeltaE_SOURCE+(1/2)<H_N rho_active,i>], becauseEXACTRef4 restrictionE_i†E_i=(2-rho_active,i)/4 independentPhi/U/Theta. NOT frozenneutralbra frequency andnochargedownenergyoffset. InitialcentredALLdensitythirdmoments0fromonsiteGamma5Pgamma5=Q givesCoulombenergy-densitycov0. InitialEMPi²densitycov0fromrealGaussian/nx2/PHparticle-holeonsitedensityidentity usedactualc42; nointeractingEMenergydrop. CommonPhi scalar/neutralheavy covariance0point; inORIGINALpositivePhi/colourform, n_active,i commutesallonsiteSU4connectionq andliteralPhi/colourkinetic. Itsmean0 EACHphysicalPhi/Theta; thusrealcommonPhiGaussian-gradient andrealcross covariance0, remainingcarrier covariance magnitude<=2neutralcarrierPhi-price because||rho_active,i||<=2. Reference000primitivechargedmatter h anticommutesfixedGamma5; sourceGamma5PΓ5=Q makes TrhP RxQ purelyimaginary. Other63flavourmatter/heavycov0givenTheta/Phi sinceactivecentredmean0. OriginalPhi matterfreezingprice boundsremaininghalf-covariance<=b367ALLCARprice. SourceisHOMOGENEOUS undertwistedAP222translations: actualPhi222 constantroot/translation-invariant spectralGaussian+b2, projectedheavyconstant-phase uniformreference, χHermYgraph000/AP222, isotropicallMcoexactGaussGaussian2. Ref4singletNET0/colour0 leavesHN translational; globalactiveNconserved henceeverylocaldensityinitialtime derivative0. Thusimaginarypart<HN rho>=0andGii_prime hasrealpart0. Remainingnormalizedfrequencycorrection <=neutralcarrierPhi-price+initialmatterprice; executedpositivefullSOURCEenergyinterval yieldspositivefirstimagmoment. ThisONLYinitialnonstationaryGreenfirstmoment, notFourierpole/massgap.',
        source_Gram_and_same_receiver_proof='ForSPECIFICHermsourceGamma5 PGamma5=Q withSAMEonsiteGamma5 EACHreferencecell, Trspin Qxy=2delta_xy EXACTEACHTheta (cyclictrace). ALL4Refholes andreferenceW0=I onnonwrapstraightpaths thenGref_INITIAL=(1/2)I2, notgenericSlaterneutrality. FullactualSU4Wilson paths initialerrorsdeltaE_i<=2sqrt(N/4)eta, bothfull/referencepacketnormsqrt(1/2). Exactdiagonals.5; offdiagonalfullGramabsolute <=sqrt(.5)(deltaE_i+deltaE_j), notzeroed/fitted. ActualFULLGram isdefinedby< E_iPsi,E_jPsi >; inequalityonlyprovesminEig>=.5-offdiag. Originalc42 BOTHsector/commonPhi/insertion comparisonsapplyallfouri/j becausepositionsnonwrapequallength. 2x2matrixoperatorerror<=2entryerror. SAMEexactFullGram^-1/2 G FullGram^-1/2 inbothfull/reference forms hasrelativeerror<=matrixerror/minEig. BoundnotreplacingactualGram/sourcecolumns/neutralprobabilities; no sourceownnormwhitening/admissiongate.',
        scope='NEWfulloriginalPhi chargedINITIALGreen derivative/receiver comparison, parentreviewrequired forhomogeneousTWISTEDsource/creation-energy covariance/chiral andcarrierpremises. ACTUALnativeall64FREE-CAR two-sectordeterminants discriminateWRONGfrozenneutralbra; notnumericalfullquantumHreforfullPhiGreenvalues. No stationaryground/chargedgap/electronmass/pole/commoncone/Compton/Qstatclosure. Fundamentalquantumparametersdeclaredinputs; oldphoton/source/receiverprobabilities unchanged.',
        source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-energy',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.source_energy)
    if not all(result['source_unchanged_since_import'].values()):
        raise ArithmeticError('Source changed during actual calculation')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('input_sha256','actual_FULL_source_normalized_diagonal_Green_first_imaginary_moment_positive_lower','actual_native_two_sector_CAR_receiver','elapsed_seconds')},indent=2))
