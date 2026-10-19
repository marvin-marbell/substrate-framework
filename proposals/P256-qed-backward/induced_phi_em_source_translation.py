"""Original neutral SOURCE translation equivariance and exact charged Green moment.

This is full-source symmetry, not merely pointwise mean density homogeneity.
The fixed Ref4 singlet is invisible in H_N, not in charged Ref3 H_C.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
from scipy.linalg import expm
from induced_phi_em_charged_photon_response import SOURCE_BYTES as RESPONSE_BYTES
from induced_phi_em_compensated_charged_green import local_car_matrix, T as GENERATORS
from induced_phi_em_ygraph_clock_transfer import checked,exact
from induced_phi_em_ygraph_initial_contact import positive_exact
from induced_phi_em_macro_preparation import build_native
from induced_reference_electric_graph import graph_state
from induced_phi_backreaction import PhiConfig,models,scalar_paths

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**RESPONSE_BYTES,Path(__file__).name:Path(__file__).read_bytes()}
REVISION='1178517cdc4236ce3a4af6a26bed880eb20679df709826463e4db7819ea0d208'


def translations(native:dict[str,Any],shift:tuple[int,...])->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    N,V=native['N'],native['V'];sites=native['sites']
    index={tuple(site):i for i,site in enumerate(sites)}
    S=np.zeros((V,V));twisted=np.zeros((V,V));L=np.zeros((3*V,3*V))
    for i,site in enumerate(sites):
        dest=(site+shift)%N;j=index[tuple(dest)]
        sign=(-1)**sum((int(site[a])+shift[a])//N for a in range(3))
        S[j,i]=1;twisted[j,i]=sign
        for axis in range(3):
            L[3*j+axis,3*i+axis]=1
    return S,twisted,L,np.kron(twisted,np.eye(4))


def original_heavy(native:dict[str,Any],mean:dict[str,Any])->np.ndarray:
    cfg=PhiConfig(Lambda=1.,**mean['microscopic_config'])
    load,model,_role=models(cfg)[0]
    if load!=1 or model.improved:
        raise ArithmeticError('Actual original neutral-heavy profile required')
    onsite,paths=scalar_paths(model)
    V,N=native['V'],native['N'];frame=native['frame']
    h=np.kron(np.eye(V),onsite*frame[0])
    index={tuple(site):i for i,site in enumerate(native['sites'])}
    for displacement,coefficients in paths:
        hop=np.einsum('m,mij->ij',coefficients,frame)
        axis=int(np.flatnonzero(displacement)[0]);length=int(displacement[axis])
        for i,site in enumerate(native['sites']):
            j=index[tuple((site+displacement)%N)]
            seam=-1 if int(site[axis])+length>=N else 1
            h[4*i:4*i+4,4*j:4*j+4]+=seam*hop
            h[4*j:4*j+4,4*i:4*i+4]+=seam*hop.conj().T
    return h


def covariances(states:list[dict[str,Any]],g:float,native:dict[str,Any])->tuple[np.ndarray,float]:
    V=native['V'];cov=np.zeros((V,V));matter=np.zeros(V,complex)
    for si,state in enumerate(states):
        P,Q=state['P'],np.eye(4*V)-state['P']
        blocks=P.reshape(V,4,V,4).transpose(0,2,1,3)
        local=np.array([np.trace(P[4*x:4*x+4,4*x:4*x+4]) for x in range(V)])
        cov+=state['load']*(np.diag(np.real(local))-np.sum(np.abs(blocks)**2,axis=(2,3)))
        if si==0:
            h=state['physical_h'];vector=P@h@Q
            matter=np.array([np.trace(vector[4*x:4*x+4,4*x:4*x+4]) for x in range(V)])
    return matter,float(g*np.sum(native['Lplus']*cov.T))


def native_full_source_components(mean:dict[str,Any])->dict[str,Any]:
    native=build_native(mean,2);V,M=native['V'],native['M'];sites=native['sites']
    theta=native['Q']@(.13*np.sin(.7*np.arange(M)))
    states=[graph_state(native,p,theta) for p in native['Pprofiles']]
    point_cov,coulomb=covariances(states,64.,native)
    ref=np.eye(16,dtype=complex)[:,15]
    ref_number_error=float(np.linalg.norm((local_car_matrix(np.eye(4))-4*np.eye(16))@ref))
    ref_colour_error=max(float(np.linalg.norm(local_car_matrix(t)@ref)) for t in GENERATORS)
    frames=[expm(1j*.11*sum(np.sin(.4*(x+1)*(a+1))*t for a,t in enumerate(GENERATORS))) for x in range(V)]
    Gauge=np.zeros((4*V,4*V),complex)
    for x,U in enumerate(frames):Gauge[4*x:4*x+4,4*x:4*x+4]=U
    hH=original_heavy(native,mean)
    hv,hb=np.linalg.eigh(hH)
    if np.sum(hv>0)!=2*V:
        raise ArithmeticError('Actual -Lambda heavy upper-band filled2V reference absent')
    WH=hb[:,hv>0]
    Wphysical=Gauge@WH
    heavy_overlap=np.linalg.det(Wphysical.conj().T@WH)
    if abs(heavy_overlap)<.5:
        raise ArithmeticError('Native projected-heavy exactline overlap too small')
    orbit=[];rows=[]
    index={tuple(site):i for i,site in enumerate(sites)}
    for shift in itertools.product((0,1),repeat=3):
        S,t,L,T4=translations(native,shift)
        translated=[graph_state(native,p,L@theta) for p in native['Pprofiles']]
        c,C=covariances(translated,64.,native);orbit.append(c)
        T2=np.kron(t,np.eye(2))
        charged=[dict(load=int(st['load']),physical_H_covariance_error=float(np.linalg.norm(st2['physical_h']-T4@st['physical_h']@T4.conj().T,2)),canonical_Y_P_covariance_error=float(np.linalg.norm(st2['P']-T4@st['P']@T4.conj().T,2)),canonical_Y_occupied_FRAME_covariance_error=float(np.linalg.norm(st2['occupied']-T4@st['occupied']@T2.conj().T,2)),RIGHT_basis_exterior_phase_real=float(np.linalg.det(T2).real),RIGHT_basis_exterior_phase_imag=float(np.linalg.det(T2).imag)) for st,st2 in zip(states,translated)]
        gdiag=np.diag(t@S.T)
        frames_shifted=[frames[index[tuple((site-shift)%2)]] for site in sites]
        Gshift=np.zeros_like(Gauge)
        for x,U in enumerate(frames_shifted):Gshift[4*x:4*x+4,4*x:4*x+4]=U
        Phi=np.array([[U@C0@U.conj().T for C0 in native['frame']] for U in frames])
        Phi_shift=np.array([[U@C0@U.conj().T for C0 in native['frame']] for U in frames_shifted])
        phi_error=float(np.linalg.norm(Phi_shift-Phi[np.argmax(S,axis=1)]))
        link_error=0.
        for x,site in enumerate(sites):
            old=index[tuple((site-shift)%2)]
            for axis in range(3):
                dest=site.copy();dest[axis]=(dest[axis]+1)%2;y=index[tuple(dest)]
                oldsite=sites[old];oldtarget=oldsite.copy();oldtarget[axis]=(oldtarget[axis]+1)%2;oldy=index[tuple(oldtarget)]
                z=-1 if site[axis]==1 else 1;oldz=-1 if oldsite[axis]==1 else 1
                actual_translated=gdiag[x]*gdiag[y]*oldz*frames[old]@frames[oldy].conj().T
                reconstructed=z*frames_shifted[x]@frames_shifted[y].conj().T
                link_error=max(link_error,float(np.linalg.norm(actual_translated-reconstructed,2)))
        WHshift=Gshift@WH
        overlap_shift=np.linalg.det(WHshift.conj().T@WH)
        phase_ref=np.linalg.det(WH.conj().T@T4@WH)
        projected_overlap=(overlap_shift/abs(overlap_shift)).conjugate()*(heavy_overlap/abs(heavy_overlap))*np.linalg.det(WHshift.conj().T@T4@Wphysical)
        heavy_cov=float(np.linalg.norm(T4@Gauge@hH@Gauge.conj().T@T4.conj().T-Gshift@hH@Gshift.conj().T,2))
        # Actual periodic Fourier phases: allPhi45 andcolourcov blocks at q commuteT.
        fourier=np.exp(2j*np.pi*sites@sites.T/2)/np.sqrt(V)
        FourierT=fourier.conj().T@S@fourier
        phase_error=float(np.linalg.norm(FourierT-np.diag(np.diag(FourierT)),2))
        Pzero=np.ones((V,V))/V
        zero_error=float(np.linalg.norm(S@Pzero@S.T-Pzero))
        rotations=native['Q'].T@L@native['Q']
        EM_cov_error=float(np.linalg.norm(rotations@rotations.T-np.eye(M),2))
        row=dict(shift=list(shift),actual_charged_source_frames=charged,EMGaussian_variance2_covariance_error=EM_cov_error,EMGaussian_radius_change=float(abs(np.vdot(theta,theta)-np.vdot(L@theta,L@theta))),Phi_colour_periodic_Fourier_block_intertwining_error=phase_error,Phi_zero_q_slow_ALL45_invariance_error=zero_error,neutral_Coulomb_SOURCE_energy_change=abs(C-coulomb),point_SOURCE_matter_covariance_TRANSLATION_error=float(np.linalg.norm(c-S@point_cov)),physical_Noe_noncommuting_Phi_transport_error=phi_error,physical_Noe_central_restored_link_transport_error=link_error,actual_original_heavy_covariant_physical_H_error=heavy_cov,actual_projected_heavy_norm_squared_change=float(abs(abs(overlap_shift)**2-abs(heavy_overlap)**2)),actual_projected_heavy_exterior_phase_intertwining_error=float(abs(projected_overlap-phase_ref)),actual_heavy_reference_translation_phase=dict(real=float(np.real(phase_ref)),imag=float(np.imag(phase_ref))))
        if max([EM_cov_error,phase_error,zero_error,abs(C-coulomb),link_error,phi_error,heavy_cov,abs(projected_overlap-phase_ref),*sum([[item['physical_H_covariance_error'],item['canonical_Y_P_covariance_error'],item['canonical_Y_occupied_FRAME_covariance_error']] for item in charged],[])])>1e-9:
            raise ArithmeticError('Actual FULLsource translation component failed')
        rows.append(row)
    orbit_average=np.mean(orbit,axis=0)
    counter_native=build_native(mean,3)
    counter_theta=counter_native['Q']@(.13*np.sin(.7*np.arange(counter_native['M'])))
    counter_states=[graph_state(counter_native,p,counter_theta) for p in counter_native['Pprofiles']]
    counter_cov,_=covariances(counter_states,64.,counter_native)
    counter_density_error=max(abs(np.trace(counter_states[0]['P'][4*x:4*x+4,4*x:4*x+4])-2) for x in range(counter_native['V']))
    parity=np.sum(sites,axis=1)%2
    parity_operator=np.diag(np.repeat((-1)**parity,2))
    R=states[0]['R'];Mblock=states[0]['M']
    Rparity_error=float(np.linalg.norm(R@parity_operator-parity_operator@R))
    sameparity_M_nonlocal=max(float(np.linalg.norm(Mblock[2*x:2*x+2,2*y:2*y+2])) for x in range(V) for y in range(V) if x!=y and parity[x]==parity[y])
    onsite_M_nonscalar=max(float(np.linalg.norm(Mblock[2*x:2*x+2,2*x:2*x+2]-np.trace(Mblock[2*x:2*x+2,2*x:2*x+2])/2*np.eye(2))) for x in range(V))
    if np.linalg.norm(orbit_average)>1e-10 or np.linalg.norm(counter_cov)<1e-8 or counter_density_error>1e-10 or ref_number_error>1e-12 or ref_colour_error>1e-12 or max(Rparity_error,sameparity_M_nonlocal,onsite_M_nonscalar)>1e-9:
        raise ArithmeticError('Exact source covariance/noncollapsedRef4 adverse failed')
    counterexample=dict(N=3,V=counter_native['V'],M=counter_native['M'],actual_point_active_density_2_defect=float(counter_density_error),actual_POINT_covariance_norm=float(np.linalg.norm(counter_cov)),actual_POINT_covariance_real=np.real(counter_cov).tolist(),actual_POINT_covariance_imag=np.imag(counter_cov).tolist(),actual_fixed_active_NUMBER_covariance_sum_real=float(np.real(counter_cov.sum())),actual_fixed_active_NUMBER_covariance_sum_imag=float(np.imag(counter_cov.sum())),scope='ActualN3 108globalmobileonebodymodes/active54 source, aux63, samephysicalmasses/g64/meanimage/all55realcoexact coordinates. FrozenTheta MATTERcomponent counterexample to pointmean2=>covariance0, notFULLsourceGaussian covariance ormacrogeometry.')
    return dict(N=2,V=V,full_translation_orbit=8,actual_profiles=[dict(load=int(st['load']),mass=float(st['profile']['model'].m)) for st in states],Ref4_exact_number_minus_background_action=ref_number_error,Ref4_actual_all15_colour_generator_action=ref_colour_error,actual_projected_heavy_overlap_norm_squared=float(abs(heavy_overlap)**2),point_active_density_each_Theta_exact_2=True,actual_N2_point_SOURCE_matter_covariance_norm=float(np.linalg.norm(point_cov)),actual_N2_bipartite_R_parity_error=Rparity_error,actual_N2_sameparity_M_nonlocal= sameparity_M_nonlocal,actual_N2_onsite_M_nonscalar=onsite_M_nonscalar,N2_collapsed_control_explanation='Knearestflipsbipartiteparity,R=f(K²)preservesit; N2twolinkGamma0hopscollapseonsitescalarM. OppositeparityM R hasnodensitydiagonal,andonsiteMspinidentitycommuteslocalR, henceTrspin[M,R]xx=0. DoNOTexpectnonzeropointHN_MATTERrho counterexample atN2; N3noncollapsedactualcontrolbelow.',actual_complete_translation_orbit_matter_covariance_error=float(np.linalg.norm(orbit_average)),actual_noncollapsed_point_density_covariance_counterexample=counterexample,rows=rows,scope='ActualN2 originalmodel full32modeHermYcanonicalframes/all17coexactEM/8twistedtranslations, physicalnoncommutingSU4gaugePhi/linkNoeintertwining, actualoriginalneutralheavyupper16/projectedline exteriorphase, full15Ref4CARsinglet. SeparateN3noncollapsedpointcounterexample. Componentalgebra/control, notmanybodyPhiGaussianintegral or4096simulation. Covarianceproofbelow usesFULLsourcefunctionalEquivariance, notpointmean2.')


def run(response_path:Path)->dict[str,Any]:
    started=time.monotonic();raw=response_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!='fee24268839d0955c698787b79f9832841e65cc19f9a34fb1363fab087eeaacd':
        raise ArithmeticError('Actual e363 charged response/source epoch required')
    response=json.loads(raw)
    _,source=checked(response['input_descriptor']['input_receipts']['source'])
    _,transition=checked(source['input_descriptor']['input_receipts']['transition'])
    _,receiver=checked(transition['input_descriptor']['input_receipts']['receiver'])
    _,energy=checked(receiver['input_descriptor']['input_receipts']['source_energy'])
    _,green=checked(energy['input_descriptor']['input_receipts']['green'])
    _,four=checked(green['input_descriptor']['input_receipts']['response'])
    _,inserted=checked(four['input_descriptor']['input_receipts']['inserted'])
    _,force=checked(inserted['input_descriptor']['input_receipts']['force'])
    _,current=checked(force['input_descriptor']['input_receipts']['current'])
    _,retarded=checked(current['input_descriptor']['input_receipts']['retarded'])
    _,field=checked(retarded['input_descriptor']['input_receipts']['clock'])
    _,bridge=checked(field['input_descriptor']['input_receipts']['bridge'])
    _,mean=checked(bridge['input_descriptor']['input_receipts']['mean'])
    _,original_energy=checked(bridge['input_descriptor']['input_receipts']['energy'])
    _,initial=checked(original_energy['input_descriptor']['input_receipts']['initial'])
    _,domain=checked(initial['input_descriptor']['input_receipts']['domain'])
    _,pure=checked(domain['input_descriptor']['input_receipts']['purePhi'])
    well_raw,well=checked(pure['input_descriptor']['input_receipts']['well'])
    gd=green['input_descriptor'];V=gd['V']
    if (gd['N'],gd['a'],gd['UEphys'],gd['Lambda'],gd['variance'],gd['clock'])!=(262144,'1/64','1','2^8192','2','4096') or green['charged_sector_counts']['active']!=2*V+1 or green['charged_sector_counts']['Ref']!=3 or (130*V+4)%4:
        raise ArithmeticError('No original source/law/Ref/background epoch substitution')
    lower=exact(energy['full_same_H_compensated_SOURCE_energy_addition_positive_lower'])
    upper=exact(energy['full_same_H_compensated_SOURCE_energy_addition_upper'])
    desc=dict(plan=gd['plan'],plan_revision=REVISION,scientific_execution_revision=REVISION,input_receipts=dict(response=dict(path=str(response_path),sha256=hashlib.sha256(raw).hexdigest())),N=gd['N'],V=V,M=gd['M'],a=gd['a'],UEphys=gd['UEphys'],Lambda=gd['Lambda'],variance=gd['variance'],clock=gd['clock'],source_abi=gd['source_abi'],source='UNCHANGEDORIGINALneutralPsiN fullPhi222Gaussian/b2/projectedheavy +canonicalY000all64/Gaussian2/AP222/Ref4; NEWlocalE_U chargedsource normp=.5',neutral_particle_counts=dict(active=2*V,auxiliary_each=2*V,auxiliary_count=63,heavy=2*V,Ref=4,total=130*V+4),fixed_background=dict(uniform=128,reference_extra=4,site=[0,0,0]),original_Phi_source_definition_epoch=dict(sha256=hashlib.sha256(well_raw).hexdigest(),PhiCovariance_and_jointNoe_ONLY=well['physical_state_definition']),original_well_EM_and_negativeBand_source_overrides='OldwellcompactEMproduct/negativeband64mixture NOTours: currentlyselected222, allThetaunfilteredisoGaussian2, canonicalHermY000 state. PhiCov/cut/jointNoe unchanged; projectedheavycurrent0212 suppliedphysicalsection.',exact_normalized_diagonal_green_initial_derivative='-i DeltaE_source, NO residual covariance price',source_covariance_theorem_status='NewFULLsource physicaltranslation and projectedsection theorem forparentreview; nativecomponentcontrols, notsourceGaussianintegral')
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),FULL_source_active_HN_rho_covariance_exact='0',normalized_diagonal_charged_GREEN_initial_frequency_positive_lower=positive_exact(lower),normalized_diagonal_charged_GREEN_initial_frequency_upper=positive_exact(upper),unnormalized_CREATOR_norm_squared_exact='1/2',ALLtime_neutral_evolved_CREATOR_source_norm_squared_exact='1/2',alltime_4096_original_charged_Green_absolute_upper_exact='1/2',actual_native_FULLsource_symmetry_components=native_full_source_components(mean),
        actual_Phi_and_EM_source_translation_proof='ACTUALd2c4wellPhi: atq0 ALL45normalCov=Lambda^-3/4 Hslow(G222)^-1/2/2, q0 translationidentity regardlessinternalHslow; periodicq!=0 Phi45Cov=Lambda^-7/8 I/[2sqrt(lap/5)]. ColouractualR0r blocks covariancePT/PL withfunctionsONLYlap,r, q0 THREEidenticalcolourmodes. TranslationatEACHperiodicq multipliesphaseidentityinternal45/Phi/colour, realcos/sinpairs rotatewithinidenticalcovariance, soFULLPhiGaussian precision/Covcommutestranslations. Radialb2cutnormeta andexactnormalization preserved. ActualCURRENT EM isisotropicGaussianvariance2 onWHOLErealcoexact+harmonykerD, NOToldwellbasiswisecompactproduct. EdgepermutationLintertwinesD,Lplus,PT, orthogonalonkerD, soEMGaussiancommutesL regardlessarbitraryQbasis. No conditionalonepointstatistic substitutedforfullsourcefunctional.',
        actual_projected_heavy_and_Y_frame_translation_proof='EachoriginalactualheavyPhi/SU4physicalHamiltonian isbuiltuniformoriginalRotationBand(.2,.12,.3,improvedFalse)/cfg/pathlaw, sofullfamilyhH(TPhi,TU)=T4 hH(Phi,U)T4dag. Upperfilled2V manybodyPH(Phi,U) spectralprojector (Hactual=-Lambda hH), referenceOmegaH222 alltwistedmomenta/twospins isTRANSLATIONeigenvector withconstantphase. Exactprojectedsection PH(Phi,U)OmegaH_ref/sqrt<OmegaHref,PHOmegaHref> thereforetransformsTwithinverseconstantrefphase; denominatornorminvariant. No choiceofindividualeigenvector/pinnedspin/node reference. ChargedcanonicalY000usesSAMEAP222seams; transformedTheta: K(TTheta)=T2 K(Theta)T2dag, dscalaruniform =>Y/R/sqrtR transformsame. PhysicaloccupiedW(TTheta)=T4 W(Theta)T2dag; exteriorRIGHTdetT2dag constant (AP222 twooccupiedspins givesphase1). Fullall1+63 Slaterdensity thereforeequivariant, evennon-eigenHref. PhysicalPhi222root !=chargedC000root butBOTHconstantframes andtwistedrepresentation, no root-equality asserted.',
        actual_Noe_physical_isometry_and_Ref4_compensation_proof='OriginalphysicalNoe map Phi_x=V_x(C222+N0Yx)Vxdag,Uxy=Vx zxy222 exp(iAxy.T) Vydag; zcentre−I. Translationmovescentral-seamrepresentative, restoredbylocalcentralg_x=±I withdetg_x=1; adg=I soY/A normalcoordinates JUSTpermute, allradiiandFouriercov unchanged. Haarproduct unchanged, w_local/orbitGram andw_joint=const*w_local/4 merelypermuted; REALw_joint^-1/2 equivariantextension ISOMETRYintertwinesphysicalunitarytranslation. No local4^V projection. TotalCAR130V+4 divisible4 preservesdiagonalZ4descent. Ref4=filledwedge4 atFIXED0: dGamma(tA)Ref4=TrtA Ref4=0 ALL15, andnRef−4=0exact soactualBACKGROUND128+4delta0 reducesrho=sum64(n−2) onHNsector; fixedRef0 leavesNOEM/SU4pinning. AnyRefbare1onsiteactsfilleddetbyconstanttrace, soalsoinvisible. Compensatingg0 onRefquartet givesdetg0=1. Hence ORIGINALFULLH_N andphysicaldomaincommuteactualtwistedtranslations, includingprimitivePiQ², rhoLplusrho, CartesianPhi/SU4kinetics/measure andallliteralone/two-path currents. NottruechargedRef3/H_Ctranslation: its−delta0 remains; no chargedhomogeneity assumed.',
        exact_full_covariance_and_green_moment_proof='FULLsourceD(H) anddensityrhoPsi isunitarytranslationinvariant fromALLthreefunctionalpieces+physicalNoe above. Forc_x=<HN Psi,rhoactive_x Psi>, HNcommutesT andTsourceconstantphase/densityinvariance =>cx SAMEforEVERYx. ExactactiveN=2V =>sum_xrhoactive_x Psi=0, henceSUMcx=0 andcx=0 EXACT, notjust <rho>=0. CurrentnativefrozenTheta counterexamplehaspointn2butnonzeroHN_MATTERrho covariance, showingwhyweverifyFULLsource. GreenGxx=<E_x UNPsi,UC E_xPsi> hasG(0)=p=.5, Gprime(0)/p=−i[DeltaE_SOURCE+cx/(4p)]=−iDeltaE_SOURCEexact. ThuspositivefullsameH initialfrequency interval istheEXACT272.4936/14.8593sourceenergy interval, no residual. FullH stillnonstationary, firstmomentNOTmass/pole/spectralgap. HN/sourceT invariancealsoalltimes<active nx>=2; Ref4fixed+unitaryW means||E_x UNPsi||²=.5 ALL4096 while||UC E_yPsi||²=.5 byunitarity =>|Gxy|<=.5. No source/H/receiver/clock/probabilitymodification.',
        scope='Newexactfullsourceequivariance/covariancecorollary submittedparentreview, physicallyboundactualsourceparameter/section/isometrydefinitions andnativeallcomponents. NativeNoe/heavy/CAR/EM/Fouriercontrols notwholequantumPhi integral/native-to-macro promotion. Sourceenergypositiveinitialfrequency NOTmassgap/pole/dispersion/commoncone/ComptonBorn; original2515pricedcovariancecaps frozen, no retagging.',
        source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--response',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.response)
    if not all(result['source_unchanged_since_import'].values()):raise ArithmeticError('Source changed during actual calculation')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('input_sha256','FULL_source_active_HN_rho_covariance_exact','actual_native_FULLsource_symmetry_components','elapsed_seconds')},indent=2))
