"""Original charged E_U source: full-Phi photon response on actual packets.

Corrects the inherited scalar-versus-complex CURRENT Gram price explicitly,
without modifying frozen source/receipt epochs. No stationary/pole assumption.
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
from scipy.linalg import expm
from induced_phi_em_charged_photon_source import SOURCE_BYTES as CHARGED_BYTES
from induced_phi_em_ygraph_clock_transfer import checked, exact
from induced_phi_em_ygraph_initial_contact import positive_exact
from induced_phi_em_ygraph_evolving_field_clock import su
from induced_phi_em_macro_preparation import build_native, scalars
from induced_reference_electric_graph import graph_state
from induced_reference_field_following import path_derivative
from induced_phi_ir_well_transfer import rational

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**CHARGED_BYTES,Path(__file__).name:Path(__file__).read_bytes()}
REVISION='1178517cdc4236ce3a4af6a26bed880eb20679df709826463e4db7819ea0d208'


def unitary(values:np.ndarray, vectors:np.ndarray, clock:float)->np.ndarray:
    return (vectors*np.exp(1j*clock*values))@vectors.conj().T


def native_charged_echo(mean:dict[str,Any])->dict[str,Any]:
    """Actual full32mode/ALL64 FREECAR component, coherent four Refhole source."""
    native=build_native(mean,2)
    V,M=native['V'],native['M']
    x=int(np.flatnonzero(np.all(native['sites']==(1,1,0),axis=1))[0])
    theta=native['Q']@(.13*np.sin(.7*np.arange(M)))
    f=np.cos(np.pi*native['sites'][:,0])/np.sqrt(V)
    directions=np.zeros((2,3*V))
    directions[0,1::3]=f;directions[1,2::3]=f
    if np.linalg.norm(native['D']@directions.T)>1e-12 or np.linalg.norm(directions@directions.T-np.eye(2))>1e-12:
        raise ArithmeticError('Actual unitcoexact y/z polarizations not admitted')
    active,aux=[graph_state(native,p,theta) for p in native['Pprofiles']]
    states=[active,aux]
    currents=[[path_derivative(native,st['profile'],theta,f,1) for f in directions] for st in states]
    hspec=[np.linalg.eigh(st['physical_h']) for st in states]
    P,Q=active['P'],np.eye(4*V)-active['P']
    columns=[]
    for b in range(4):
        u=Q[:,4*x+b]
        norm=float(np.vdot(u,u).real)
        if norm<=0:
            raise ArithmeticError('Actual E source Refhole column vanished')
        W=np.column_stack((active['occupied'],u/np.sqrt(norm)))
        columns.append((norm/2,W,W@W.conj().T))
    if max(np.linalg.norm(W.conj().T@W-np.eye(2*V+1)) for _,W,_ in columns)>1e-10:
        raise ArithmeticError('Actual rawsource charged Slater exterior basis not orthonormal')
    Pa=aux['P'];Qa=np.eye(4*V)-Pa
    Wa=aux['occupied'];p=.5
    epsilons=[1/512,1/1024,1/2048]
    alpha=np.array([1,1j])/np.sqrt(2)
    complex_current_Gram=np.zeros((2,2),complex)
    for weight,_,Pc in columns:
        Qc=np.eye(4*V)-Pc
        means=np.array([np.trace(Pc@j)+63*np.trace(Pa@ja) for j,ja in zip(currents[0],currents[1])])
        for i in range(2):
            for j in range(2):
                complex_current_Gram[i,j]+=p*weight*(np.trace(Pc@currents[0][i].conj().T@Qc@currents[0][j])+63*np.trace(Pa@currents[1][i].conj().T@Qa@currents[1][j])+means[i].conjugate()*means[j])
    complex_norm=float(np.vdot(alpha,complex_current_Gram@alpha).real)
    jplus=[sum(alpha[i]*row[i] for i in range(2)) for row in currents]
    baseline=p*sum(weight*np.linalg.det(W.conj().T@W) for weight,W,_ in columns)*np.linalg.det(Wa.conj().T@Wa)**63
    complex_norm_steps=[1/8192,1/16384,1/32768]
    exterior_norms=[]
    for epsilon in complex_norm_steps:
        norm_sum=0.
        for phase in np.exp(2j*np.pi*np.arange(8)/8):
            z=epsilon*phase
            words=[]
            for vertex in jplus:
                A=expm(z*vertex)
                words.append(A.conj().T@A)
            norm_value=p*sum(weight*np.linalg.det(W.conj().T@words[0]@W) for weight,W,_ in columns)*np.linalg.det(Wa.conj().T@words[1]@Wa)**63
            norm_sum+=float(np.real(norm_value-baseline))/8
        exterior_norms.append(norm_sum/(epsilon*epsilon))
    complex_exterior=(4*exterior_norms[-1]-exterior_norms[-2])/3
    complex_error=abs(complex_exterior-complex_norm)
    missing_z_source_norm_squared=float(abs(alpha[1])**2*complex_current_Gram[1,1].real)
    if complex_error>1e-3 or missing_z_source_norm_squared<=1e-6:
        raise ArithmeticError('Actual BOTHaxes complexSOURCE/exterior adverse failed')
    rows=[]
    source_H_residuals=[float(np.linalg.norm((np.eye(4*V)-st['P'])@st['physical_h']@st['P'])) for st in states]
    for clock in (.013,.125):
        jt=[];jminus=[]
        for si,(values,vectors) in enumerate(hspec):
            U=unitary(values,vectors,clock)
            jt.append([U@j@U.conj().T for j in currents[si]])
            jminus.append([U.conj().T@j@U for j in currents[si]])
        direct=np.zeros((2,2),complex)
        stationary_wrong=np.zeros((2,2),complex)
        echoes=[]
        for i in range(2):
            for j in range(2):
                for weight,_,Pc in columns:
                    direct[i,j]+=-1j*p*weight*(np.trace(Pc@(jt[0][i]@currents[0][j]-currents[0][j]@jt[0][i]))+63*np.trace(Pa@(jt[1][i]@currents[1][j]-currents[1][j]@jt[1][i])))
                    def ordered(af:np.ndarray, ag:np.ndarray, bf:np.ndarray, bg:np.ndarray)->complex:
                        muf=np.trace(Pc@af)+63*np.trace(Pa@bf)
                        mug=np.trace(Pc@ag)+63*np.trace(Pa@bg)
                        return complex(np.trace(Pc@af@(np.eye(4*V)-Pc)@ag)+63*np.trace(Pa@bf@Qa@bg)+muf*mug)
                    forward=ordered(jt[0][i],currents[0][j],jt[1][i],currents[1][j])
                    wrong_reverse=ordered(jminus[0][j],currents[0][i],jminus[1][j],currents[1][i])
                    stationary_wrong[i,j]+=-1j*p*weight*(forward-wrong_reverse)
        for epsilon in epsilons:
            response=np.zeros((2,2),float)
            for i in range(2):
                for j in range(2):
                    sums=[]
                    for sign in (1,-1):
                        words=[]
                        for si in range(2):
                            av,ab=np.linalg.eigh(jt[si][i]);bv,bb=np.linalg.eigh(currents[si][j])
                            A=unitary(av,ab,sign*epsilon);B=unitary(bv,bb,sign*epsilon)
                            words.append(A@B@A.conj().T@B.conj().T)
                        active_echo=sum(weight*np.linalg.det(W.conj().T@words[0]@W) for weight,W,_ in columns)
                        auxiliary_echo=np.linalg.det(Wa.conj().T@words[1]@Wa)**63
                        sums.append(p*active_echo*auxiliary_echo)
                    response[i,j]=-float(np.imag((sums[0]+sums[1])/2))/(epsilon*epsilon)
            echoes.append(response)
        richardson=(4*echoes[-1]-echoes[-2])/3
        error=float(np.linalg.norm(richardson-np.real(direct),2))
        wrong_error=float(np.linalg.norm(stationary_wrong-direct,2))
        if error>1e-3 or wrong_error<1e-4 or np.linalg.norm(np.imag(direct))>1e-9:
            raise ArithmeticError('Actual64 exterior charged retarded echo/control failed')
        rows.append(dict(time=clock,actual_nonstationary_unchanged_E_SOURCE_retarded_CURRENT=np.real(direct).tolist(),all4rank17_ACTIVE_times_rank16_AUX_power63_exterior_echo_retarded=richardson.tolist(),actual_exterior_echo_errors=[float(np.linalg.norm(row-np.real(direct),2)) for row in echoes],actual_exterior_echo_Richardson_error=error,WRONG_stationary_one_sided_current_reverse_error=wrong_error))
    return dict(N=2,V=V,source=[1,1,0],Ref=[0,0,0],coherent_Ref3_hole_blocks=4,active_charged_rank=2*V+1,auxiliary_rank=2*V,auxiliary_load=63,raw_CREATOR_probability=sum(weight for weight,_,_ in columns)*p,source_H_non_eigen_residuals=source_H_residuals,actual_coexact_axes=['y','z'],native_Nyquist_q_pi_control_not_macro_profile=True,actual_adversarial_complex_polarization=dict(real=np.real(alpha).tolist(),imag=np.imag(alpha).tolist()),actual_charged_complex_SOURCE_CURRENT_norm_squared=complex_norm,actual_ALL4rank17_AUX63_exterior_complex_CURRENT_norm_squared=complex_exterior,actual_exterior_complex_CURRENT_norm_error=complex_error,actual_exterior_complex_norm_steps=complex_norm_steps,actual_exterior_complex_norm_errors=[abs(value-complex_norm) for value in exterior_norms],WRONG_dropping_imag_z_SOURCE_current_ray_norm_squared=missing_z_source_norm_squared,group_commutator_steps=epsilons,rows=rows,scope='ACTUAL64mobileFREECARquadraticcomponentoforiginalHref, fixedNUMERICreferencePhi image/all17Theta, Ref3spectators; actualfull32modeunitcoexacty/z andcomplex(y+iz)/sqrt2 CURRENTsource, ALL4weightedrank17exterior determinants timesALL63rank16aux, no2sitecompression/noemptyZ/noselectedspin. N2NyquistprofiledifferentfromMACROnonNyquistsqrt2/V profile, nofield/kernelpromotion. NotPi/Coulomb/Phi/SU4/RefHamiltonian evolution ormacrophotoncurve. Sourcecomparison finiteclock4096unchanged.')


def run(source_path:Path)->dict[str,Any]:
    started=time.monotonic()
    raw=source_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!='339fe6eb5ec2f02c0b3c1035d53a99b76bf4d9cf36cec176ffc0f8b9a0ac756e':
        raise ArithmeticError('Requires actual5f26 source/CAR epoch, notbeta/physicalC')
    source=json.loads(raw)
    _,transition=checked(source['input_descriptor']['input_receipts']['transition'])
    _,receiver=checked(transition['input_descriptor']['input_receipts']['receiver'])
    _,energy=checked(receiver['input_descriptor']['input_receipts']['source_energy'])
    _,green=checked(energy['input_descriptor']['input_receipts']['green'])
    _,response=checked(green['input_descriptor']['input_receipts']['response'])
    _,inserted=checked(response['input_descriptor']['input_receipts']['inserted'])
    _,force=checked(inserted['input_descriptor']['input_receipts']['force'])
    _,current=checked(force['input_descriptor']['input_receipts']['current'])
    _,retarded=checked(current['input_descriptor']['input_receipts']['retarded'])
    _,field=checked(retarded['input_descriptor']['input_receipts']['clock'])
    _,bridge=checked(field['input_descriptor']['input_receipts']['bridge'])
    _,mean=checked(bridge['input_descriptor']['input_receipts']['mean'])
    _,condition=checked(bridge['input_descriptor']['input_receipts']['conditioning'])
    _,norms=checked(source['input_descriptor']['input_receipts']['source_norm'])
    gd=green['input_descriptor'];fp=force['prices']
    if (gd['N'],gd['a'],gd['UEphys'],gd['Lambda'],gd['variance'],gd['clock'])!=(262144,'1/64','1','2^8192','2','4096') or green['actual_creation_probability_exact']!='1/2':
        raise ArithmeticError('Unchanged original source/law/clock required')
    V,M=gd['V'],gd['M'];s,a,g,T,chi,p=F(2),F(1,64),F(64),F(4096),F(2)**-4096,F(1,2)
    sqrt2,n,sqrts=su(F(2)),su(p),su(s)
    Jscalar=exact(bridge['corrected_initial_whole_Theta_current_operator_price'])
    Jcomplex=sqrt2*Jscalar
    Cprice=exact(bridge['corrected_initial_whole_Theta_contact_operator_price'])
    price_ref=exact(source['reference_charged_CURRENT_Gram_vs_half_neutral_operator_price'])
    Gref_upper=p*F(norms['macro_reference_two_current_action_Gram_maximum_eigenvalue_upper'])+price_ref
    Gref_lower=p*F(condition['reference_current_Gram_lower'])-price_ref
    njref=su(Gref_upper)
    complex_current_map_phi_price=n*Jcomplex
    repaired_Gram_price=2*njref*complex_current_map_phi_price+complex_current_map_phi_price**2
    gram_lower=Gref_lower-repaired_Gram_price
    gram_upper=Gref_upper+repaired_Gram_price
    nj=su(gram_upper)
    epsilon=exact(green['charged_sector_recomputed_alltime_original_Phi_reference_wave_error'])
    gap=exact(field['original_reference_vs_cut_low_reference_norm_error'])
    fibre=exact(green['charged_sector_recomputed_alltime_Born_fibre_residual'])
    residual=fibre+2*exact(field['alltime_reference_to_C000_matter_operator_action_price'])+exact(field['alltime_uniform_low_spectral_boundary_residual'])
    high=exact(field['field_original_source_high_spectral_norm_upper']);tail=F(2)**-8192
    if epsilon!=2*high+2*tail+T*residual or gap!=high+tail:
        raise ArithmeticError('Actual charged/originalPhi spectral clock prices disagree')
    epsilon_cut=gap+T*residual
    insertion=exact(green['original_FULL_vs_reference_initial_CREATOR_error'])
    c,B=exact(fp['c']),exact(fp['B'])
    Jref,Jcap=exact(fp['Jref_ALLthetaCAR_operator_cap']),exact(fp['FULL_initial_current_operator_cap'])
    Cany,Csame=exact(fp['FULL_current_arbitrary_EM_direction_contact_cap']),exact(fp['FULL_current_initial_directional_contact_cap'])
    Jbody,Vcap,Jrho=exact(fp['initial_full_current_covariant_body_derivative_cap']),exact(fp['initial_full_potential_operator_cap']),exact(fp['actual_global_current_rho_cap'])
    R1=sqrt2*exact(retarded['recomputed_all64_ray_jets'][0]);pi0=su(1/(4*s))+R1
    neutral_kinetic=exact(inserted['source_prices']['actual_source_positive_kinetic_form_upper'])
    charged_carrier=exact(energy['actual_full_charged_carrier_Phi_colour_kinetic_upper'])
    initial_charged_kinetic=p*(neutral_kinetic+charged_carrier)
    base_energy=p*(exact(force['actual_full_source_force_rows']['base']['conserved_full_H_plus_B_form_upper'])+exact(energy['full_same_H_compensated_SOURCE_energy_addition_upper']))
    current_energy=2*Jcap*Jcap*initial_charged_kinetic+p*(chi*(4+512/exact(retarded['actual222_orbit_min']))*(90*V)*Jbody*Jbody+2*g*M*Cany*Cany)+(Vcap+B)*nj*nj
    kappa=exact(source['actual_reference_CURRENT_onebody_operator_upper'])
    theta_jref=su(exact(inserted['source_prices']['actual_reference_initial_Theta_current_Gram_upper']))+kappa*sqrts
    theta_jfull=theta_jref+sqrts*complex_current_map_phi_price
    delta_J=complex_current_map_phi_price+Jref*insertion
    delta_PiJ=n*Jcomplex*pi0+Jref*insertion*pi0+n*Cprice+exact(bridge['reference_whole_Theta_contact_operator_cap'])*insertion
    delta_ThetaJ=sqrts*delta_J
    freeze_J=sqrt2*exact(field['alltime_reference_to_C000_current_operator_action_price'])
    field_abs=exact(retarded['absolute_reference_field_classical_offset_cap']);Ecut=exact(field['field_low_spectral_cut'])
    root0=su(rational(mean['centre_enclosure']['g0_upper']));roots=su(rational(mean['centre_enclosure']['gs_upper']))
    CARref=F(0)
    for profile in build_native(mean,2)['Pprofiles']:
        row=scalars(profile)
        hrow=abs(row['onsite'])*root0+6*sum(abs(c0)*root0+abs(cj)*roots for c0,cj in row['path'].values())
        CARref+=4*V*row['load']*hrow
    Href_potential=CARref+exact(green['charged_sector_global_Coulomb_operator_upper'])
    specs=dict(charged_base=dict(n=n,nref=n,energy=base_energy,pi=n*pi0,anypi=n*pi0,theta=n*sqrts,fulltheta=n*sqrts,delta_pi=insertion*pi0,delta_theta=insertion*sqrts,delta_wave=insertion),
        initial_charged_FULL_current=dict(n=nj,nref=njref,energy=current_energy,pi=n*(Jref*pi0+Csame),anypi=n*(Jref*pi0+Cany),theta=theta_jref,fulltheta=theta_jfull,delta_pi=delta_PiJ,delta_theta=delta_ThetaJ,delta_wave=delta_J))
    rows={}
    for name,packet in specs.items():
        wave=packet['nref']*epsilon+packet['delta_wave']
        cutwave=packet['nref']*epsilon_cut+packet['delta_wave']
        ref_form=packet['nref']**2*(B+field_abs+Ecut+residual+Href_potential)+g*M*packet['anypi']**2
        delta_form=2*(packet['energy']+ref_form)
        rho_delta=su(su(cutwave*cutwave*delta_form/c))
        action=Jrho*rho_delta+freeze_J*packet['nref']+Jref*gap*packet['nref']
        Pi_error=packet['delta_pi']+packet['pi']*epsilon+T*action+2*T*gap*Jref*packet['nref']+T*T*residual*Jref*packet['nref']/2
        ref_pi=packet['pi']+T*Jref*packet['nref']
        Theta_error=packet['delta_theta']+packet['theta']*epsilon+2*g*(T*Pi_error+2*T*gap*ref_pi+T*T*residual*ref_pi/2)
        ref_theta=packet['theta']+2*g*T*packet['pi']+g*T*T*Jref*packet['nref']
        inserted_error=packet['delta_theta']+packet['theta']*epsilon
        factor=F(2) if name=='charged_base' else F(6)
        tensor=factor*(Theta_error*packet['fulltheta']+ref_theta*inserted_error)
        rows[name]=dict(initial_source_map_norm_upper=packet['n'],reference_initial_source_map_norm_upper=packet['nref'],actual_FULL_initial_H_plus_B_form_upper=packet['energy'],reference_CUTLOW_alltime_H_plus_B_form_upper=ref_form,initial_source_map_difference=packet['delta_wave'],alltime_original_reference_source_map_difference=wave,alltime_FULL_minus_CUTLOW_form_upper=delta_form,alltime_Phi_weighted_packet_difference=rho_delta,alltime_actual_current_action_difference=action,alltime_actual_Pi_packet_difference=Pi_error,alltime_actual_Theta_packet_difference=Theta_error,initial_Theta_inserted_packet_propagator_difference=inserted_error,alltime_actual_canonical_Theta_retarded_tensor_error=tensor)
    physical=dict(EXACT_p_half_normalized_charged_physical_A_retarded_error=rows['charged_base']['alltime_actual_canonical_Theta_retarded_tensor_error']/(p*a*a),physical_A_retarded_on_physical_JA_charged_source_error=rows['initial_charged_FULL_current']['alltime_actual_canonical_Theta_retarded_tensor_error'],EXACT_FULL_initial_charged_CURRENT_Gram_relative_Theta_tensor_error=rows['initial_charged_FULL_current']['alltime_actual_canonical_Theta_retarded_tensor_error']/gram_lower,EXACT_FULL_initial_charged_physical_JA_Gram_relative_A_tensor_error=rows['initial_charged_FULL_current']['alltime_actual_canonical_Theta_retarded_tensor_error']/(a*a*gram_lower))
    if gram_lower<=0 or max(physical.values())>=F(1,1000):
        raise ArithmeticError('Actual charged photon response transfer not useful')
    values=dict(corrected_initial_complex_TWO_CURRENT_CREATOR_map_Phi_price=complex_current_map_phi_price,corrected_FULL_initial_charged_CURRENT_Gram_minimum_eigenvalue_lower=gram_lower,corrected_FULL_initial_charged_CURRENT_Gram_maximum_eigenvalue_upper=gram_upper,corrected_FULL_initial_charged_CURRENT_Gram_physical_JA_lower=a*a*gram_lower,corrected_complex_FULL_vs_half_neutral_CURRENT_Gram_price=price_ref+repaired_Gram_price,charged_INITIAL_positive_kinetic_form_upper=initial_charged_kinetic,charged_INITIAL_CURRENT_H_plus_B_form_upper=current_energy,actual_charged_LOW_reference_generator_residual=residual,actual_charged_Href_only_ALLCAR_plus_Coulomb_operator_cap=Href_potential)
    desc=dict(plan=gd['plan'],plan_revision=REVISION,scientific_execution_revision=REVISION,input_receipts=dict(source=dict(path=str(source_path),sha256=hashlib.sha256(raw).hexdigest())),source_abi=gd['source_abi'],N=gd['N'],V=V,M=M,a=str(a),UEphys='1',Lambda=gd['Lambda'],variance=str(s),clock=str(T),source='UNCHANGEDlocalE_U=.5cdagW_SU4r all4Refhole active+1/Ref3 andoriginalPsi_N; sourceXi_Cnorm²p=.5',reference='SAMEoriginalPhi222exactDiric quantumfield evolution timesHref_C000, SAMEneutralElightfree000 offset; no charged ownphase subtraction',complex_Gram_correction='Frozen5f26 SOURCECAR/contact/referencechargedGram retained; its per-axisJprice fullcomplexGram premise NOTinherited. NEWsqrt2complexmap price and njref actualchargedreferenceGramupper reprice explicitly.',native_control='N2 actual32mode64mobileFREECARcomponentsource echo/unitcoexacty/z/adversarialcomplex(y+iz)/sqrt2; Ref3spectators, noPhi/Pi/Coulomb/SU4/Refevolution')
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),prices={k:positive_exact(v) for k,v in values.items()},actual_charged_response_rows={name:{key:positive_exact(value) for key,value in row.items()} for name,row in rows.items()},actual_charged_physical_responses={k:positive_exact(v) for k,v in physical.items()},actual_native_charged_nonstationary_exterior_echo=native_charged_echo(mean),
        charged_source_kinetic_and_inserted_packet_proof='OriginalsourceCphaseE_U commutesALLcoexactPi andpointTrP_xx=2. Gamma5 P Gamma5=Q, Gamma5 BfGamma5=−Bfdag forEACHrealcoexactderivativeBf=Q DPf P givesTrRx BfdagBf=TrRx BfBfdag, hencepointwiseALLTheta <PiPsi,(n_x−2)PiPsi>=0 (realGaussian cross also0), notjust summedEMenergy. E_U Pi source norm=sqrtp originalPi norm, alsoTheta-weighted. SharedrealOmega compactPhi/halfdensity scalar derivative cross cancels; chargedcarrierkin upper retainsWilson/body/connection and qC. Thus initialpositivechargedKIN<=p(K_NEUTRAL+chargedcarrierK), noPhiZPE/BassignedtoEM. FullH+B source conservedform<=p(E_NEUTRAL_plus+actual272.4936energyadditionupper). Productrule forJfull E_U retains2Jop² chargedKIN + pchi metric D Jbody² +2pgM Cany² +(Vcap+B)*ACTUALchargedCURRENTGramupper. WeightedThetaJref E0 source <=ThetaJneutralSOURCE +kappa sqrt(s) byJ E=E J+[J,E], EopfixedRef4<=1, allfourRefholeKnorm<=kappa. No nJsource onlateJ Usource.',
        charged_full_Phi_packet_transfer_proof='EachREFERENCEinitialchargedpacket E0Psi_N/JrefE0Psi_N/ThetaE0Psi_N sharesORIGINALPhiGaussian/heavyconstant000 factor andchargedCARsector2V+1,Ref3. Recomputedc42qC Bornfibre suppliesRC=fibreC+2matterPrice+boundary, EPSC=2HIGH+2tail+4096RC verifiedexact. GenericALLCARPhi transfer scalesactualchargedPACKETnorm. SOURCEinitialE_U−E0 uniformPhi priceinsertion; Jinitialdifference=sqrtp sqrt2 Jscalar+Jrefop insertion; PiINITIALdifference paysbothJpricePi andCprice plusJref/Cref multipliedWilsondifference, sincePi E_U=E_U Pi. ThetaINITIALdifference=sqrt(s)delta_J. FullminusCUTLOW FORMtriangle andHeinzquarticrho retained, weightedPhiδprice thenmovingPhi_currentfreeze+HIGHgap. StrongPi/ThetaWard allchargedpackets useslastDuhamelRESTART2HIGHgap+tauRC terms, notoperatorcurrentnormSOURCE/gap/spectralground substitution. Originalfieldscalarunitary cancelsONLYafterCOMPLETEchargedsectorpacketcomparisons. HrefpositiveEMbudget usesinitialPi gradients andg64CAR/CoulombcapsONLY, notPhiZPE. ActualsameE_light_NEUTRALphaseinBOTHCfull/ref.',
        complex_tensor_units_and_receiver_proof='CorrectarbitrarycomplexCURRENTmap: sqrt2 multipliesb367PER-axisoperatorprice; normchargedE source sqrtp=.707 meansinitialchargedcurrentmapPhierror=sqrt(2p)Jscalar. SourcecurrentbasisGram EXACTfullcomplexarray kept, eigenloweronlypricebound. ForCURRENTsource2×photonprobe2, F=<Theta_i U_CXi_alpha,U_CTheta_jXi_beta>, reverse=(Fdag)^PTprobe notentrywiseconj. partialtranspose norm<=2 andSchmidt2 givesSIXbrackets. BaseoneSOURCE usesTWObrackets. JA=aJTheta, physicalA=Theta/a, sofourpointret onphysicalJA source no net a factor; relativephysicalJAgram gives/a². BASErawXi_Cprobp=.5 normalizationexplicit/p. UnchangedfullactualINITIALsourceGram usedinBOTHreceivercomparands; neverreplacebybound/neutralpGram orindependentreferencewhitening. No interpolation/pole/groundstationarity admission.',
        scope='ActualMACROsamefullH quantumPhi/EM charged photon/sourceCURRENTpacket finiteclock4096 comparison prices; parentreviewrequirednewpointwisePi density/covariantKIN/productform extension andoldcoercivity/domain/P7/spectralchartpremises. Nativeactualfull64mobileCARsource retardedgroupcommutator/exteriorcofactorcontrol, notfullHref/fullPhi/macrophotoncurve. No dispersion/pole/massgap/commoncone/ComptonBorn claim; no frozenprobability/source/clockgate change. Prior5f26 FULLcomplexinitialGram-pricepremise supersededexplicitly, allfrozenbytes preserved.',
        source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.source)
    if not all(result['source_unchanged_since_import'].values()):
        raise ArithmeticError('Source changed during actual physical calculation')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('input_sha256','actual_charged_physical_responses','actual_native_charged_nonstationary_exterior_echo','elapsed_seconds')},indent=2))
