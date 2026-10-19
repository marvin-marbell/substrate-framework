"""Original compensated source transition graph and all-clock Green bounds.

Keeps the original 4096 clock domain. An additional rigorously derived small
clock interval is a corollary, not a replacement for that domain or a pole.
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
from induced_phi_em_compensated_green_receiver_typed import SOURCE_BYTES as RECEIVER_BYTES
from induced_phi_em_ygraph_clock_transfer import checked, exact
from induced_phi_em_ygraph_initial_contact import positive_exact
from induced_phi_em_ygraph_evolving_field_clock import su, sl
from induced_phi_em_macro_preparation import build_native, scalars
from induced_reference_electric_graph import graph_state
from induced_phi_ir_well_transfer import rational, binary_exponent

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**RECEIVER_BYTES,Path(__file__).name:Path(__file__).read_bytes()}
REVISION='6ebc34a912158254456c535bd10259fca04201b14f446a778bba68d5ef63ecf2'


def actual_density_transition(mean: dict[str, Any]) -> dict[str, Any]:
    native=build_native(mean,2)
    V,M=native['V'],native['M']
    x=int(np.flatnonzero(np.all(native['sites']==(1,1,0),axis=1))[0])
    ref=int(np.flatnonzero(np.all(native['sites']==(0,0,0),axis=1))[0])
    theta=native['Q']@(.13*np.sin(.7*np.arange(M)))
    G=native['Lplus']
    v=G[:,x]-G[:,ref]
    R=float(v[x]-v[ref])
    A=np.diag(np.repeat(v,4))
    Rx=np.zeros((4*V,4*V),complex)
    Rx[4*x:4*x+4,4*x:4*x+4]=np.eye(4)
    variance=graph_variance_upper=0.
    profiles=[]
    active_cov=active_third=0.
    for profile in native['Pprofiles']:
        state=graph_state(native,profile,theta)
        P,Q=state['P'],np.eye(4*V)-state['P']
        value=float(np.trace(P@A@Q@A).real)
        commP=A@P-P@A
        Ahalf=np.diag(np.repeat(v,2))
        commY=Ahalf@state['Y']-state['Y']@Ahalf
        upper=float(np.linalg.norm(commY,'fro')**2)
        variance+=profile['load']*value
        graph_variance_upper+=profile['load']*upper
        if profile['load']==1:
            active_cov=float(np.trace(P@Rx@Q@A).real)
            active_third=float((np.trace(P@Rx@A@A)-np.trace(P@Rx@P@A@A)-2*np.trace(P@A@P@Rx@A)+2*np.trace(P@Rx@P@A@P@A)).real)
        if value<0 or value>upper+1e-10 or abs(float(np.linalg.norm(commP,'fro')**2)-2*value)>1e-10:
            raise ArithmeticError('Actual source density/Frechet graph control failed')
        profiles.append(dict(load=int(profile['load']),actual_density_dipole_variance=value,actual_Hermitian_Y_commutator_squared_upper=upper,actual_P_commutator_squared=float(np.linalg.norm(commP,'fro')**2)))
    g=native['UE']/native['a']
    exact_norm_squared=g*g*(2*variance+R*R/2-R*active_cov-active_third)
    coulomb_addition=g*(R-active_cov)
    alternative_norm_squared=g*g*(2*variance-R*R/2)+g*R*coulomb_addition
    if exact_norm_squared<0 or abs(active_third)>1e-10 or abs(exact_norm_squared-alternative_norm_squared)>1e-8 or variance>graph_variance_upper+1e-10:
        raise ArithmeticError('Actual Coulomb source transition norm control failed')
    return dict(N=2,V=V,source=[1,1,0],Ref=[0,0,0],actual_all64_density_dipole_variance=variance,actual_all64_Hermitian_graph_variance_upper=graph_variance_upper,active_density_dipole_covariance=active_cov,actual_active_centred_density_third_moment=active_third,actual_real_dipole_effective_resistance=R,actual_source_Coulomb_addition=coulomb_addition,actual_full64_source_Coulomb_transition_norm_squared=exact_norm_squared,actual_source_Coulomb_transition_norm=float(np.sqrt(exact_norm_squared)),alternative_norm_identity_error=abs(exact_norm_squared-alternative_norm_squared),profiles=profiles,scope='ACTUALnativeN2 exactconditional64-Slater density moments oforiginalHermYsource. Frechetcontractor validnative; macroprimitivepathFrobeniusconstant isseparatelyN>4 (N2nearestslotcollisions), notpromotedfromnative.')


def run(receiver_path: Path) -> dict[str,Any]:
    started=time.monotonic()
    raw=receiver_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!='b9737da70f151016af5d02e3f9e474161eb1e28758bbb4cf7a1e56f2f88413c6':
        raise ArithmeticError('Actual2515 nonstationaryreceiver epoch required')
    receiver=json.loads(raw)
    if receiver['source_at_import']['induced_phi_em_compensated_green_receiver_typed.py']['sha256']!=hashlib.sha256(SOURCE_BYTES['induced_phi_em_compensated_green_receiver_typed.py']).hexdigest():
        raise ArithmeticError('Receiver source changed')
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
    gd=green['input_descriptor']
    if (gd['N'],gd['V'],gd['a'],gd['UEphys'],gd['Lambda'],gd['clock'])!=(262144,2**54,'1/64','1','2^8192','4096') or green['actual_creation_probability_exact']!='1/2':
        raise ArithmeticError('No clock/law/preparation/reference substitution')
    enc=mean['centre_enclosure']
    root0lo=sl(rational(enc['g0_lower']))
    gsupper=rational(enc['gs_upper'])
    g,chi,V,L=F(gd['UEphys'])/F(gd['a']),F(2)**-4096,gd['V'],gd['N']//4
    factor=F(0)
    for profile in build_native(mean,2)['Pprofiles']:
        row=scalars(profile)
        if row['path'][2][1]!=0:
            raise ArithmeticError('Primitive source K requires actual nearest-only spatial Clifford term')
        dmin=2*row['onsite']*root0lo+g
        factor+=row['load']*4*row['path'][1][1]**2*gsupper/(dmin*dmin)
    G00=exact(green['actual_three_dimensional_Lplus_diagonal_upper'])
    Rupper=4*G00
    variance_upper=factor*Rupper
    coulomb_source_norm=g*su(2*variance_upper+Rupper*Rupper/2)
    matter_source_norm=exact(energy['actual_full_INITIAL_source_CAR_matter_addition_upper'])
    metric=exact(field['actual222_metric_upper'])
    rmin=exact(field['actual222_orbit_min'])
    rmax=exact(field['actual222_orbit_max'])
    radius=min(F(1,1000),rmin/(16*su(rmax)))
    gradient=exact(bridge['corrected_actual_Phi222_carrier_gradient_upper'])
    carrier=exact(energy['original_neutral_carrier_global_covariant_derivative_upper'])
    D=90*V
    colour_source_gradient=4*su(metric)*(gradient+carrier+8*D/radius)
    colour_source_norm=F(3,10)*chi*(2*su(F(30*L))*colour_source_gradient+F(15*L,8)*su(F(1,2)))
    full_source_norm=matter_source_norm+coulomb_source_norm+colour_source_norm
    # Global fixedRef4 neutral block, allTheta/CAR: no source variance reused here.
    lapmin=F(16,gd['N']**2)
    neutral_density_operator=128*su(F(V))
    global_coulomb_norm=g*(2*su(Rupper/lapmin)*neutral_density_operator+Rupper)
    reference_alltime_operator=matter_source_norm+global_coulomb_norm
    source_norm=su(F(1,2))
    short_clock=F(2)**(-binary_exponent(32*source_norm*reference_alltime_operator)-1)
    additional_Green_error=source_norm*reference_alltime_operator*short_clock+exact(green['alltime_NONSTATIONARY_original_FULL_charged_Green_comparison_error'])
    short_real_lower=F(1,2)-additional_Green_error
    if short_clock>=F(gd['clock']) or short_real_lower<=F(1,4) or full_source_norm>=1024:
        raise ArithmeticError('Source transition/derived additionalGreen interval failed')
    desc=dict(plan=gd['plan'],plan_revision=REVISION,scientific_execution_revision=REVISION,input_receipts=dict(receiver=dict(path=str(receiver_path),sha256=hashlib.sha256(raw).hexdigest())),source_abi=gd['source_abi'],N=gd['N'],V=V,M=gd['M'],a=gd['a'],UEphys=gd['UEphys'],Lambda=gd['Lambda'],variance=gd['variance'],clock_UNCHANGED=gd['clock'],source_position=gd['source_position'],receiver_position=gd['receiver_position'],Ref_position=gd['Ref_position'],background='128+4deltaRef0',Ref_bare_charge='1',N_sector='active2V/aux63each2V/heavyneutral2V/Ref4',C_sector='active2V+1/aux63each2V/heavyneutral2V/Ref3',transition='K=H_C E−E H_N; SAMEoriginalfullH/SAMEneutralphase, no sectorownoffset',additional_clock_corollary='Derivedinside original4096domain; nottimerchange/4096replacement',native_control=dict(N=2,source=[1,1,0],Ref=[0,0,0],theta='0.13 sin(0.7 arange(M)) in allMcoexact coordinates'))
    values=dict(actual_ALL64_Hermitian_source_density_Dirichlet_variance_factor=factor,actual_realGauss_source_Ref_resistance_upper=Rupper,actual_INITIAL_all64_source_density_dipole_variance_upper=variance_upper,actual_INITIAL_source_Coulomb_transition_norm_upper=coulomb_source_norm,actual_INITIAL_source_matter_transition_norm_upper=matter_source_norm,actual_original_INITIAL_source_full_compact_colour_gradient_upper=colour_source_gradient,actual_INITIAL_source_compact_colour_transition_norm_upper=colour_source_norm,actual_FULL_original_INITIAL_charged_transition_norm_upper=full_source_norm,ALLtime_reference_fixedRef4_to_Ref3_transition_OPERATOR_norm_upper=reference_alltime_operator,ADDITIONAL_derived_safe_Green_clock=short_clock,ADDITIONAL_original_FULL_diagonal_Green_realpart_lower=short_real_lower)
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),**{k:positive_exact(v) for k,v in values.items()},ALLtime_original_FULL_neutral_CREATOR_packet_norm_squared_exact='1/2',ALLtime_original_FULL_charged_packet_norm_squared_exact='1/2',ALLtime_original_FULL_RAW_charged_Green_absolute_upper_exact='1/2',ALLtime_same_exact_SOURCE_Gram_normalized_diagonal_Green_absolute_upper_exact='1',fixed_Ref4_CREATOR_operator_norm_upper_exact='1',actual_native_density_transition=actual_density_transition(mean),
        ALLtime_neutral_receiver_proof='RestrictedfixedRef4 operatoridentity E_U†E_U=(4-n_active,x)/4 holdsALLPhi/Theta/neutralCAR (ALLcoherent4Refholes), notSlater/dynamicsassumption. NeutralHN Ref4singletcolour0/NET0 cancelsstaticRefbackground at0 exactly, hencehomogeneous undertwistedAP222translations. OriginalSourcehomogeneous (actualPhi222constantroot/spectralGaussian/radialcut/projectedheavyuniformreference/χ000AP222/isotropicallMrealGauss); UNpreserveshomogeneity. GlobalactiveN=2V fixed implies<n_x>_UN=2 ALLclocks, so||E_x UNPsi||²=.5 EXACT. ChargedUCinitialpacketnorm²=.5 byunitarity. Green=<E_xUNPsi,UC E_yPsi>, Cauchy =>ALL|t|<=4096 absolute<=.5 andsameexactdiagGram-normalized<=1. NOTstationarity/spectralpositive-poleclaim. RestrictedRef4 Eop<=1 usefulbutneverextends toselectedspincolumn.',
        SOURCE_density_and_Coulomb_graph_proof='ForVNode scalarreal andHermYsourceP(Y), equivariance P(e^(isV)Y e^(-isV))=e^(isV)P(Y)e^(-isV), exactgraphmetric ||DP(deltaY)||F²=2TrRdeltaYRdeltaY<=2||deltaY||F². ThereforepureSlater Var(dGamma V)=TrPVQV=||[V,P]||F²/2<=||[Vhalf,Y]||F². MacroN>4 nearestKblocks unique orderedslots, spin2eachhopnorm²2cj1²gs, forward/reverse =>||[Vhalf,Y]||F²<=4cj1²gs/d² sum_edges(v_x-v_y)². All1+63conditionaldensitymeans0, so variancesADDlinearly, NOaux63²/classicaldensitymeanterm. v=G(delta_x-deltaRef), Dirichletgradientnorm²R_eff, givingVar_total<=factorR_eff<=factorRupper. Originalsourcechiralcentredthirdmoment0 andRef4E†E=(2-rho_active,x)/4 makeEXACT ||gE(2vrho+R)Psi||²=g²[2Var_total+R²/2−R Cov(rho_active,x,vrho)]. PositiveSOURCEC delta=g(R−Cov), maxprinciplev boundsSOURCEC delta<=gR, henceCov>=0 andnorm²<=g²(2Var+R²/2). Coulomb SOURCEtransition graphfiniteindependentV, notalltimevariance bound or interactingSlaterclaim.',
        FULL_source_transition_action_proof='ORIGINALphysicalunreducedE=(1/2)c†W_SU4(U)r withREALlongitudinalU1phase; reducedcoexactphase1. PrimitiveEMPi_Q kineticcommutesE; Coulombcomm above; allbosonscalarPhi/SU4potential/neutralheavycommuteE. ORIGINALCartesianPhi kineticcommutesE (EdependsphysicalSU4links notCartPhi): thisisnotdroppinggaugefixedbodyfibre; completejointlift retainedinunreducedcompactcolourgradient envelope. Mattercomm onfixedRef4: K_h=(1/2)c†h|x>W r, itsoperatornorm²<=TrRxh²/4<=||h||² (notwholeCARextensivecap), actualb367Phi-freezingadditionpricedinsideinitialsupport. Compactcolourphysicalcoefficient3chi/10 andleftMaurer productrule: Kcolour=−(3chi/10)[2sum_e,a(L_aE)L_aPsi−(15/8)L E Psi]. PlainWilsononeappearance/edge gives sum||L_aE||op²<=30L andLapCasimir15/8 peredge; ALLRef source normsqrt(.5). Actualsourcecompactgradientboundedby4sqrtM[actualb367Gaussian-gradient+GN+8D/radius], includingprojectedheavyD1/orbitqN/jointmetric/realhalf-densitylogFP/Haarjets. Not totalPhi sourceenergyassignedEM; noflatmetric/frozenPhi/evolving-sourceconfinement. Triangle givesstrictoriginalSOURCE K norm finite<1024; body/colourgradientenvelopesrequireparentreview.',
        original_quantum_Green_smallclock_corollary='ReferenceHref000 E0 independentTheta/Phi: E0mapsD(HrefN)toD(HrefC), K0boundedALLTheta/CAR onfixedRef4 neutralblock. Mattertransitionnorm<=rowhcap; Coulombneutralrho localbound128 =>||rho||<=128sqrtV, PSDG gives||v||<=sqrt(Rupper/lapmin), lapmin>=16/N²; Eop<=1. ThusglobalK0<=hcap+g[2sqrt(Rupper/lapmin)128sqrtV+Rupper], no initialsourcevariance substitutedforevolvedstate. Domain/variationofconstants yields|Gref(t)−.5|<=sqrt(.5)K0|t|. ChooseDYADICt0<=1/(32sqrt(.5)K0) fromexecutedconstant; existingc42full-versus-referenceGreenerrorpaiduniformlyALL4096, sooriginalFULL ReGii(t)>=.5−sqrt(.5)K0 t0−transfererror>.25 forALL|t|<=t0. NewactualfullquantumGreen nonzerointerval isCONSEQUENCEinsideold4096domain, notshrinkingcompletion/pole/speedfit. Source-specificINITIAL K norm cannotreplaceglobalK0atlatertime.',
        scope='NEWsource quantumchargedtransitiongraph/alltimeGreenamplitudebound/additionalexactGreeninterval. Parentreviewneeded twistedhomogeneity/HermgraphdensityFrechet/uniqueprimitivepaths andunreducedcompactgradient-jointlift estimates. ACTUALnative64Slater densitymoments/Gauss sourcecommuter control, notquantumfullpropagatorcurve. No ground/stationarity/chargedgap/mass/pole/commoncone/Compton/statQpromotion. FrozenneutralSource/priorprobabilities/current427 andindependentAlpha/Rootlawbranchesunchanged.',
        source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receiver',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.receiver)
    if not all(result['source_unchanged_since_import'].values()):
        raise ArithmeticError('Source changed during actual calculation')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('input_sha256','actual_FULL_original_INITIAL_charged_transition_norm_upper','ADDITIONAL_derived_safe_Green_clock','actual_native_density_transition','elapsed_seconds')},indent=2))
