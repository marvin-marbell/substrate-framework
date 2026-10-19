"""Noncollapsed spin-shape contact and original full-Phi SOURCE/transfer prices."""
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

sys.set_int_max_str_digits(0)
ROOT=Path(__file__).resolve().parent
OWN_BYTES=Path(__file__).read_bytes()
import induced_phi_spin_shape_contact as contact
from induced_phi_em_ygraph_evolving_field_clock import su,sl
from induced_phi_em_ygraph_clock_transfer import exact,positive_exact
quantum=contact.quantum
SOURCE_BYTES={**contact.SOURCE_BYTES,Path(__file__).name:OWN_BYTES}
REVISION='d86132736ac1f3d1592b01da60e6e981c5005500265f0d012e979610896318c7'
CONTACT_SHA='161f61b1cf6d2ac148aad52f6d88f3f59b350fa341e748d86847e52d70b6d4fa'


def coefficient_data(profile:dict[str,Any])->tuple[F,F]:
    _,paths=contact.transfer.packets.force.scalar_paths(profile['model']);square=F(0);absolute=F(0)
    for displacement,coeff in paths:
        if any(z!=0 for z in coeff[1:]) and int(np.sum(abs(displacement)))!=1:
            raise ArithmeticError('Actualdarkforce requiresoriginalone-linkPhi5 paths')
        for z in coeff[1:]:
            re,im=F.from_float(float(z.real)),F.from_float(float(z.imag));square+=re*re+im*im;absolute+=abs(re)+abs(im)
    return square,absolute


def full_colour_force(native:dict[str,Any],profile:dict[str,Any],delta:np.ndarray,theta:np.ndarray,links:np.ndarray)->np.ndarray:
    V=native['V'];H=np.zeros((4*V,4*V),dtype=complex)
    for path in profile['paths']:
        df=np.einsum('a,aij->ij',path['scalars'],delta)
        if not np.any(df):continue # EXACTzeroPhi0 direction, notmissingcoupling.
        transporter=np.eye(4,dtype=complex)
        for edge in path['edges']:transporter=transporter@links[edge]
        phase=path['seam']*np.exp(1j*np.sum(theta[list(path['edges'])]))
        hop=(df@transporter+transporter@df)*phase/(2*np.sqrt(V))
        x,y=path['source'],path['target'];H[4*x:4*x+4,4*y:4*y+4]+=hop;H[4*y:4*y+4,4*x:4*x+4]+=hop.conj().T
    return H


def noncollapsed_control(mean:dict[str,Any],seed:int)->dict[str,Any]:
    native=quantum.build_native(mean,4);V,M=native['V'],native['M']
    geometry=contact.transfer.packets.force.physical_normal_geometry(native);mode=contact.spin_shape(native,geometry['normal'],geometry['shape'])
    profiles=[contact.derivative_profile(p,mode['delta']) for p in native['Pprofiles']]
    squares,absolute=coefficient_data(native['Pprofiles'][0]);target=float(squares/(24*V));rng=np.random.default_rng(seed)
    x=int(np.flatnonzero(np.all(native['sites']==(1,1,0),axis=1))[0]);errors=[];ann=[];values=[];colour_control={}
    for sample in range(4):
        theta=native['Q']@(np.sqrt(2)*rng.normal(size=M))
        for si,(profile,derivative) in enumerate(zip(native['Pprofiles'],profiles)):
            state=quantum.graph_state(native,profile,theta);P=state['P'];Q=np.eye(4*V)-P;dark=quantum.matter_hamiltonian(native,derivative,theta)/np.sqrt(V)
            ann.append(float(np.linalg.norm(state['empty'].conj().T@dark@state['occupied'],2))+abs(complex(np.trace(P@dark))))
            if si==0:
                D=Q[:,4*x:4*x+4]@Q[4*x:4*x+4,:]/2;raw=float(np.real(np.trace(D@dark@dark)))/2;values.append(raw);errors.append(abs(raw-target))
        if sample==0:
            # Genericphysicalnoncommutingcolour field OUTSIDEtinyoriginalcut;
            # operator-control, notareplacement/sourceintegration/probability.
            angles=.01*rng.normal(size=(3*V,15));links=np.array([expm(1j*np.einsum('A,Aij->ij',a,contact.transfer.packets.force.T)) for a in angles]);Lie=float(np.linalg.norm(angles));mu=0.;cov=0.;op_errors=[]
            for profile,derivative in zip(native['Pprofiles'],profiles):
                state=quantum.graph_state(native,profile,theta);U=full_colour_force(native,profile,mode['delta'],theta,links);ref=quantum.matter_hamiltonian(native,derivative,theta)/np.sqrt(V)
                local_mu=float(np.real(np.trace(state['P']@U)));ph=state['empty'].conj().T@U@state['occupied'];mu+=profile['load']*local_mu;cov+=profile['load']*float(np.sum(abs(ph)**2));op_errors.append(float(np.linalg.norm(U-ref,2)))
            single=float(2*absolute/sl(F(24*V)));cap=128*V*single*Lie;norm=float(np.sqrt(mu*mu+cov))
            colour_control=dict(real_colour_Cart_norm=Lie,actual_noncommuting_link_commutator_norm=float(np.linalg.norm(links[0]@links[1]-links[1]@links[0],2)),actual_full_colour_FORCE_neutral_SOURCE_norm=norm,WRONG_exact_dark_at_noncentral_colour='0 (incorrect)',actual_force_operator_difference_max=max(op_errors),uniform_onebody_difference_cap=single*Lie,uniform_ALL64_SOURCE_difference_cap=cap,not_original_source_colour_sample=True)
            if max(op_errors)>single*Lie+1e-9 or norm>cap+1e-9:raise ArithmeticError('Actualfullcolourspinforce rowbound failed')
    if max(errors)>1e-10 or max(ann)>1e-9:raise ArithmeticError('Actualnoncollapsedconstantcontact/sourceannihilator failed')
    return dict(N=4,V=V,M=M,active_N_to_C=[2*V,2*V+1],aux63_N=2*V,heavy_EMneutral_N=2*V,Ref_N_to_C=[4,3],total_CAR=130*V+4,global_Z4_mod4=(130*V+4)%4,source_position=[1,1,0],original_realGaussian_modes=M,exact_noncollapsed_RAWcontact_Gram=str(squares/(24*V)),actual_ALL4contact_Gram_values=values,constant_contact_identity_error=max(errors),neutral_dark_PH_trace_error=max(ann),generic_full_colour_operator_control=colour_control)


def run(contact_path:Path)->dict[str,Any]:
    started=time.monotonic();raw=contact_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=CONTACT_SHA:raise ArithmeticError('Actual6120 spinshape/contact epoch required')
    c=json.loads(raw);_,t=quantum.checked(c['input_descriptor']['input_receipts']['transfer']);_,g=quantum.checked(t['input_descriptor']['input_receipts']['gram']);_,f=quantum.checked(g['input_descriptor']['input_receipts']['force']);_,j=quantum.checked(f['input_descriptor']['input_receipts']['join']);_,heat=quantum.checked(j['input_descriptor']['input_receipts']['heat'])
    mean=contact.transfer.packets.force.joined.mean_from_source(heat)
    _,record=quantum.checked(heat['input_descriptor']['input_receipts']['translation']);green={};field={};field_spec={}
    for key in ('response','source','transition','receiver','source_energy','green','response','inserted','force','current','retarded','clock'):
        spec=record['input_descriptor']['input_receipts'][key];_,record=quantum.checked(spec)
        if key=='green':green=record
        if key=='clock':field=record;field_spec=spec
    gd=green['input_descriptor'];V,N=gd['V'],gd['N'];a=F(gd['a']);p=F(1,2);s=F(2);L=F(N,4)
    if (N,V,gd['Lambda'],gd['variance'],gd['clock'])!=(262144,262144**3,'2^8192','2','4096') or green['charged_sector_counts']['active']!=2*V+1 or green['charged_sector_counts']['Ref']!=3:
        raise ArithmeticError('No actualmacro/fullPhi/source/Ref/clock substitution')
    native=quantum.build_native(mean,2);data=[coefficient_data(profile) for profile in native['Pprofiles']]
    squares=data[0][0];Gamma=squares/(24*V);root=su(Gamma);rootlo=sl(Gamma)
    rows=[2*absolute/sl(F(24*V)) for _,absolute in data]
    nrow=2*V*(rows[0]+63*rows[1]);crow=(2*V+1)*rows[0]+63*2*V*rows[1]
    Lie=exact(field['alltime_original_reference_field_link_Lie_L2_upper']);epsilonC=exact(green['charged_sector_recomputed_alltime_original_Phi_reference_wave_error'])
    leakN=nrow*Lie;gram_price=su(p)*crow*Lie;wave_price=gram_price+rows[0]*su(L)*Lie
    lower=(rootlo-gram_price)**2;upper=(root+gram_price)**2;Gram_error=2*root*gram_price+gram_price**2
    photon_Gamma=s*Gamma/a**2;photon_root=su(photon_Gamma);photon_Gramprice=su(s)*gram_price/a;photon_waveprice=su(s)*wave_price/a
    photon_lower=(sl(photon_Gamma)-photon_Gramprice)**2;photon_upper=(photon_root+photon_Gramprice)**2
    # ConditionalCOROLLARY oforiginalsector-uniformfieldclock, parentreview.
    # AllCARcounts/qC unchanged; Phi-independentreferencepackets; sourcePhi
    # cut/energy/angularbounds needNOTheta derivatives (76ad literalproof).
    scalar_wave=wave_price+root*epsilonC;photon_wave=photon_waveprice+photon_root*epsilonC
    scalar_kernel_error=(root+gram_price)*scalar_wave+root*wave_price
    photon_kernel_error=(photon_root+photon_Gramprice)*photon_wave+photon_root*photon_waveprice
    if rootlo<=gram_price or sl(photon_Gamma)<=photon_Gramprice or Gram_error>=Gamma/2:raise ArithmeticError('ActualfullSOURCEchargedshape positiveGram lost')
    values=dict(exact_REFERENCE_macro_RAWcharged_shape_SOURCE_Gram=Gamma,full_original_Phi_neutral_spinforce_SOURCE_norm_upper=leakN,full_original_Phi_neutral_spinforce_SOURCE_Gram_upper=leakN**2,full_original_Phi_charged_spinforce_SAMEEU_SOURCE_norm_difference_upper=gram_price,full_original_Phi_charged_spinforce_SOURCE_Gram_lower=lower,full_original_Phi_charged_spinforce_SOURCE_Gram_upper=upper,full_original_Phi_charged_spinforce_SOURCE_Gram_difference_upper=Gram_error,full_original_Phi_charged_spinforce_referenceE0_SOURCE_wave_price=wave_price,exact_REFERENCE_macro_physical_A_spin_SOURCE_Gram_eigenvalue=photon_Gamma,full_original_Phi_physical_A_spin_SOURCE_Gram_lower=photon_lower,full_original_Phi_physical_A_spin_SOURCE_Gram_upper=photon_upper,full_original_Phi_physical_A_spin_ALLcomplex_Gram_map_price=photon_Gramprice,full_original_Phi_physical_A_spin_ALLcomplex_initial_wave_price=photon_waveprice,conditional_generic_chargedsector_4096_spin_SOURCE_wave_price=scalar_wave,conditional_generic_chargedsector_4096_physical_A_spin_SOURCE_wave_price=photon_wave,conditional_generic_4096_spin_SOURCE_kernel_absolute_price=scalar_kernel_error,conditional_generic_4096_spin_SOURCE_SAMEFULLGram_relative_price=scalar_kernel_error/lower,conditional_generic_4096_physical_A_spin_SOURCE_kernel_absolute_price=photon_kernel_error,conditional_generic_4096_physical_A_spin_SOURCE_SAMEFULLGram_relative_price=photon_kernel_error/photon_lower)
    descriptor=dict(plan=gd['plan'],plan_revision=REVISION,scientific_execution_revision=REVISION,input_receipts=dict(contact=dict(path=str(contact_path),sha256=hashlib.sha256(raw).hexdigest()),field=field_spec),source_abi=gd['source_abi'],N=N,V=V,M=gd['M'],a=gd['a'],UEphys=gd['UEphys'],Lambda=gd['Lambda'],variance=gd['variance'],clock=gd['clock'],probe='CovariantbodyCartspinshape potentialforceprobe δPhi0=0,δPhi_i=iGamma_jGamma_k/sqrt24; sourceΨ ORIGINALfixed, sameEU all4holes, q0normalization1/sqrtV. ReferenceC000 is35shape; NOTclaimfullcanonicalNoePiPhi force/currentmode sterile',source='UNCHANGEDfullPhi222Gaussian/b2/projectedheavy, canonicalY000/all64/unfilteredGaussian2/AP222, Ref4→3; no filter/gate/meanretag',producer_native_transfer_sha256=CONTACT_SHA)
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),**{key:positive_exact(value) for key,value in values.items()},actual_N4_noncollapsed_constantcontact_and_generic_colour_control=noncollapsed_control(mean,20261013),exact_macro_contact_proof='ForN>=3 one-linkPhi5 hasdistinctx±axis neighbours; trueimprovedtwo-linkPhi5coefficientszero verifiedactualpaths. Cartspinshape f=(sum_i ci Sigma_i forward+HC)/sqrt24V commutesGamma5. SourceGamma5 P Gamma5=Q =>TrRx Q f²=TrRx P f²=TrRx f²/2. Referenceannihilator[P,f]=0 givesrawALL4contactGram=p Tr(QRxQ/2)f²=TrRx f²/8. Actual(f²)xx=2sumpositivepath|ci|² I/(24V), so RAWGamma=sum|ci|²/(24V) EXACT atEVERYunfilteredTheta andANYcreationx, independentmass/gs/Phi spectators. N2double-edgecollisiondidNOTstandinfor macro; actualevenN4/all129-modepointcontrols verifyconstant.',full_Phi_source_price_proof='LiteraloriginalfullCARpath=(fx Transport+Transport fy)/2. Potentialderivative infixedcovariantbodyCartΣprobe removesPhi amplitudesEXACT; onlycompactcolourTransport varies fromcentralreference. ForSigma op1, ||exp(iA)-I||<=||A||op<=CartA_norm; one-linkdarkforce onebodydifference<=rows_alpha ||xi||, rows=2sum|ci|/sqrt24V outward. ALLCARfixedcountsgiveNrow=2V(rowactive+63rowaux),Crow=(2V+1)rowactive+126Vrowaux, validALLTheta/allΦ/allCARstates. Original76adconservedscalarPhi-sourcefastcoercivity suppliesACTUALlinkLieL2 (notarbitrarysource). NeutralreferenceforceSOURCE0 =>fullnorm<=NrowLie. ChargedsourceE_U pointnormsqrtp andALL4RefholeunitaryWilson tracing leavesSAMEaddedD independentWilson: fullF E_U vsreferenceF E_U norm<=sqrtp CrowLie. ReferencecontactnormGamma EXACT independentPhi/Theta/W; trianglegivesfullGamma lower/upper. Forinitialwave E_U→E0, useF_refPsi_N=0: F_ref(EU−E0)Psi=[F_ref,EU−E0]Psi, SINGLEactivecreator cap rowactive timesWilson||Uline−Uref||L2<=sqrtL Lie, notglobalCARcap. FullPhi/Haar/projectedheavy normalization andNoeisometry retainthesourceL2 bounds; notfullneutralexactdarkpromotion.',ALLcomplex_photon_Gram_proof='ReferencecontactnormGamma pointwiseALLTheta, soORIGINALGaussian2 givesGamma_Aspin=(2Gamma/a²)I2 EXACT. Full-minus-refSAMEEU forcepointwisebound Crow|xi| E_U normsqrtp independentTheta; initialPhi andEMGaussianproduct andpconstant => ||sum_i u_i Theta_i(force_delta)E_U Psi||<=sqrt(s)||u|| gram_price forALLCOMPLEXu, notanaxispriceborrowedwithout sqrt2. Dividea forphysicalA. Polar-Gram bounds givefullactualsourceGammaLower/Upper; SAMEfullGamma mustnormalizebothkernelcomparands. No sample/referencewhitening/substitution.',conditional_fieldclock_extension='SUBMITTEDPARENTREVIEW, notachievedfullphysicaltime-propagator result. Referencecontact andAcontactpackets dependONLYTheta/CAR, preserveexactchargedcounts active2V+1/aux63×2V/heavy2V/Ref3 andcommonNEUTRALoffset. SourcePhi/b2/heavysection SAME; original76ad/c42 cutoff-high/energy-angular/Born qC/residual bounds sector-uniform andexplicitlyneedNOTheta derivatives inPhi block. ThusifparentvalidatesextensiontoTHESEarbitrarybody-Phi-independentTheta/CARpackets, U_Cfull Vfull vsU_Phi_ref tensorU_Href_C Vref <=initialwaveprice+||Vref||epsilonC ALL4096. Derivedkernelprice usesFULLsourceΓlower onbothcomparands, retainsnonstationaryPhi-referenceevolution, no ground/survival/phasefactorfit. Nativeactualquantumcontact already6120; thiscapsule doesNOTnumericallyevaluate macroU.',scope='NEWnoncollapsedphysicalcontact/sourceGram andboundedfullPhiSOURCEleak/chargedopening, plusexplicitconditionalgeneric4096wave/kernelcorollaryforparentmathreview. No fullPiPhi/heavy/colourtime numericalsimulation, rootoptimization/electronmass/photonpole/commoncone/Compton ormacrotransferapproval. Notnewsource/filter/admissiongate. Nativegenericcolour.01 control isoffpreparedcut/operatorproof ONLY; realfullSOURCEpricesusepinnedLie moment. Frozenallsource/receipts/probabilities/clocks preserved.',source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contact',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run(args.contact)
    if not all(result['source_unchanged_since_import'].values()):raise ArithmeticError('Source changed during actual calculation')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(input_sha256=result['input_sha256'],price_binary_exponents={key:[value['binary_exponent_lower'],value['binary_exponent_upper']] for key,value in result.items() if isinstance(value,dict) and 'binary_exponent_lower' in value},actual_N4_noncollapsed_constantcontact_and_generic_colour_control=result['actual_N4_noncollapsed_constantcontact_and_generic_colour_control'],elapsed_seconds=result['elapsed_seconds']),indent=2))
