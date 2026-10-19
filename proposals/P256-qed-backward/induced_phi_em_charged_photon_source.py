"""Same original-law coherent Refhole source: charged photon-current Gram.

Full colour trace, original Hermitian-Y preparation and static Ref background
are retained. Initial current/contact and weak Ward data, not a Compton pole.
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
from induced_phi_em_compensated_transition_graph import SOURCE_BYTES as TRANSITION_BYTES
from induced_phi_em_compensated_charged_green import local_car_matrix
from induced_phi_em_ygraph_clock_transfer import checked, exact
from induced_phi_em_ygraph_initial_contact import positive_exact
from induced_phi_em_ygraph_evolving_field_clock import su
from induced_phi_em_macro_preparation import build_native, scalars
from induced_reference_electric_graph import graph_state
from induced_reference_field_following import path_derivative
from induced_phi_ir_well_transfer import rational

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**TRANSITION_BYTES,Path(__file__).name:Path(__file__).read_bytes()}
REVISION='1178517cdc4236ce3a4af6a26bed880eb20679df709826463e4db7819ea0d208'


def matrix_value(value: np.ndarray) -> dict[str,Any]:
    return dict(real=np.real(value).tolist(),imag=np.imag(value).tolist())


def quasifree_marginal(P: np.ndarray) -> np.ndarray:
    """Exact mode marginal of the actual global source, never a local projector."""
    n=len(P)
    values,vectors=np.linalg.eigh(P)
    if values.min()<-1e-12 or values.max()>1+1e-12:
        raise ArithmeticError('Actual source marginal outside CAR interval')
    rho=np.zeros((1<<n,1<<n),complex)
    for occupied in itertools.product((0,1),repeat=n):
        columns=np.flatnonzero(occupied)
        weight=np.prod([values[j] if occupied[j] else 1-values[j] for j in range(n)])
        ray=np.zeros(1<<n,complex)
        for rows in itertools.combinations(range(n),len(columns)):
            mask=sum(1<<int(j) for j in rows)
            ray[mask]=np.linalg.det(vectors[np.ix_(rows,columns)]) if len(columns) else 1
        rho+=weight*np.outer(ray,ray.conj())
    return rho


def reference_hole_blocks(n: int, source_modes: tuple[int,...]) -> list[np.ndarray]:
    """Exact fixedRef4 -> signedRef3 CAR blocks in canonical active-beforeRef order."""
    blocks=[np.zeros((1<<n,1<<n),complex) for _ in source_modes]
    for b,source in enumerate(source_modes):
        for mask in range(1<<n):
            if mask & (1<<source):
                continue
            swaps=mask.bit_count()+b+(mask & ((1<<source)-1)).bit_count()
            blocks[b][mask | (1<<source),mask]=(-1 if swaps & 1 else 1)/2
    return blocks


def actual_car_current(mean: dict[str,Any]) -> dict[str,Any]:
    native=build_native(mean,2)
    V,M=native['V'],native['M']
    x=int(np.flatnonzero(np.all(native['sites']==(1,1,0),axis=1))[0])
    y=int(np.flatnonzero(np.all(native['sites']==(1,1,1),axis=1))[0])
    modes=np.array([4*s+a for s in (x,y) for a in range(4)])
    theta=native['Q']@(.13*np.sin(.7*np.arange(M)))
    states=[graph_state(native,p,theta) for p in native['Pprofiles']]
    active,aux=states
    if (active['load'],aux['load'])!=(1,63):
        raise ArithmeticError('Tagged active1/aux63 required')
    directions=np.eye(3*V)[[3*x+2,3*y+2]]
    hs=[path_derivative(native,active['profile'],theta,f,1) for f in directions]
    contact=path_derivative(native,active['profile'],theta,directions[0],2)
    mask=np.ones(4*V,bool);mask[modes]=False
    if max(np.linalg.norm(h[mask]) for h in [*hs,contact])>1e-10:
        raise ArithmeticError('Literal native edge vertices not closed on actual two-site subsystem')
    P,Q=active['P'],np.eye(4*V)-active['P']
    Rx=np.zeros_like(P);Rx[4*x:4*x+4,4*x:4*x+4]=np.eye(4)
    rho=quasifree_marginal(P[np.ix_(modes,modes)])
    Js=[local_car_matrix(h[np.ix_(modes,modes)]) for h in hs]
    Ed=reference_hole_blocks(8,(0,1,2,3))
    creator_Gram=sum(e.conj().T@e for e in Ed)
    probability=float(np.trace(rho@creator_Gram).real)
    source_number=local_car_matrix(np.diag([1.]*4+[0.]*4))
    density=source_number-2*np.eye(256)
    aux_mu=np.array([np.trace(aux['P']@h) for h in hs])
    aux_cov=np.array([[np.trace(aux['P']@hf.conj().T@(np.eye(4*V)-aux['P'])@hg) for hg in hs] for hf in hs])
    mu=np.array([np.trace(P@h) for h in hs])+63*aux_mu
    K=[j+63*aux_mu[i]*np.eye(256) for i,j in enumerate(Js)]
    K_rho=[k@rho for k in K]
    neutral=np.array([[np.vdot(ki,K_rho[j])+63*aux_cov[i,j] for j in range(2)] for i,ki in enumerate(K)])
    density_K_rho=[density@k for k in K_rho]
    weighted=np.array([[np.vdot(ki,kj) for kj in density_K_rho] for ki in K])
    current_creator=[[k@e for e in Ed] for k in K]
    creator_current=[[e@k for e in Ed] for k in K]
    current_creator_rho=[[value@rho for value in row] for row in current_creator]
    creator_current_rho=[[value@rho for value in row] for row in creator_current]
    wick=np.zeros((2,2),complex)
    charged=np.zeros((2,2),complex)
    EJ=np.zeros((2,2),complex)
    mixture=np.zeros((2,2),complex)
    added_states=[]
    for a in range(4):
        u=Q[:,4*x+a]
        norm=float(np.vdot(u,u).real)
        if norm<=0:
            raise ArithmeticError('An actual source Refhole column has zero norm')
        addition=np.outer(u,u.conj())/norm
        Pa,Qa=P+addition,Q-addition
        ma=np.array([np.trace(Pa@h) for h in hs])+63*aux_mu
        added_states.append((norm,Pa,Qa,ma))
    for i,hf in enumerate(hs):
        for j,hg in enumerate(hs):
            wick[i,j]=mu[i].conjugate()*np.trace(P@Rx@Q@hg)+mu[j]*np.trace(P@hf.conj().T@Q@Rx)+np.trace(P@hf.conj().T@Q@Rx@Q@hg)-np.trace(P@hf.conj().T@Q@hg@P@Rx)
            for b in range(4):
                charged[i,j]+=np.vdot(current_creator[i][b],current_creator_rho[j][b])
                EJ[i,j]+=np.vdot(creator_current[i][b],creator_current_rho[j][b])
            charged[i,j]+=probability*63*aux_cov[i,j]
            EJ[i,j]+=probability*63*aux_cov[i,j]
            for norm,Pa,Qa,ma in added_states:
                second=np.trace(Pa@hf.conj().T@Qa@hg)+63*aux_cov[i,j]+ma[i].conjugate()*ma[j]
                mixture[i,j]+=.5*(norm/2)*second
    reference_contact=float(np.trace(P@contact).real+63*np.trace(aux['P']@contact).real)
    contact_CAR=local_car_matrix(contact[np.ix_(modes,modes)])
    actual_contact=sum(np.vdot(e,contact_CAR@e@rho) for e in Ed).real+probability*63*np.trace(aux['P']@contact).real
    added=Q@Rx@Q/2
    predicted_contact=probability*(reference_contact+np.trace(contact@added).real)
    wick_error=float(np.linalg.norm(weighted-wick,2))
    weighted_error=float(np.linalg.norm(EJ-(.5*neutral-.25*weighted),2))
    mixture_error=float(np.linalg.norm(charged-mixture,2))
    wrong_commuting=float(np.linalg.norm(charged-EJ,2))
    gram_identity=float(np.linalg.norm(creator_Gram-(4*np.eye(256)-source_number)/4,2))
    if abs(probability-.5)>1e-10 or gram_identity>1e-10 or wick_error>1e-8 or weighted_error>1e-8 or mixture_error>1e-8 or wrong_commuting<1e-6 or abs(actual_contact-predicted_contact)>1e-8:
        raise ArithmeticError('Actual full64 charged current/contact CAR boundary control failed')
    return dict(N=2,V=V,source=[1,1,0],other_active_site=[1,1,1],Ref=[0,0,0],actual_probe_edges=[int(3*x+2),int(3*y+2)],active_subsystem_modes=8,ALLactive_marginal_CAR_states=256,actual_Ref3_hole_blocks=4,actual_source_marginal_projector_defect=float(np.linalg.norm(P[np.ix_(modes,modes)]@P[np.ix_(modes,modes)]-P[np.ix_(modes,modes)])),exact_creation_probability=probability,CREATOR_Gram_identity_error=gram_identity,original_full64_neutral_CURRENT_Gram=matrix_value(neutral),actual_full64_charged_SOURCE_CURRENT_Gram=matrix_value(charged),actual_density_weighted_CURRENT_Gram=matrix_value(weighted),exact_Wick_density_weighted_error=wick_error,exact_EJ_density_WEIGHT_identity_error=weighted_error,actual_added_Q_source_Slater_mixture_Gram_error=mixture_error,WRONG_commuting_J_with_CREATOR_Gram_error=wrong_commuting,original_full64_neutral_CONTACT=reference_contact,actual_unnormalized_charged_CONTACT=float(actual_contact),actual_Q_creation_CONTACT_identity_error=float(abs(actual_contact-predicted_contact)),scope='ACTUALfull256two-siteCAR MIXEDmarginal fromglobal32modeHermYsource; ALL4Ref3 blocks/all63auxmean+PH retained. Literalunreducededgevertices (onecoexact/onegradientlinearcombination), notmacro y/z photonmode orfullquantumpropagation.')


def run(transition_path:Path, source_norm_path:Path)->dict[str,Any]:
    started=time.monotonic()
    raw=transition_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!='33ce9d5d919467b7054935f716d5896f6e25d908c0faf81d8804c538f8d223ed':
        raise ArithmeticError('Actual6ec516 compensated transition epoch required')
    transition=json.loads(raw)
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
    _,condition=checked(bridge['input_descriptor']['input_receipts']['conditioning'])
    _,mean=checked(bridge['input_descriptor']['input_receipts']['mean'])
    norm_raw=source_norm_path.read_bytes()
    if hashlib.sha256(norm_raw).hexdigest()!='541db4ec7cf425f8a1307b4e1c9168b18465130b6e312607ab66b568faa05733':
        raise ArithmeticError('Actual SOURCE-only closedloop variance epoch required')
    norms=json.loads(norm_raw)
    gd=green['input_descriptor']
    if (gd['N'],gd['V'],gd['a'],gd['UEphys'],gd['Lambda'],gd['variance'],gd['clock'])!=(262144,2**54,'1/64','1','2^8192','2','4096') or green['actual_creation_probability_exact']!='1/2':
        raise ArithmeticError('No law/source/field/Ref substitution')
    kappa=exact(response['ALLspecies_onebody_current_operator_cap'])
    Lip=F(norms['total_conditional_current_mean_Gaussian_Lipschitz_upper'])
    Sref=su(F(norms['macro_reference_two_current_action_Gram_maximum_eigenvalue_upper']))
    s=F(gd['variance'])
    density_price=8*kappa*(su(s)*Lip+kappa)
    reference_price=density_price/4+2*Sref*kappa+kappa*kappa
    source_norm=Sref+kappa
    Jprice=exact(bridge['corrected_initial_whole_Theta_current_operator_price'])
    full_price=reference_price+2*source_norm*su(F(1,2))*Jprice+Jprice*Jprice/2
    Gneutral_lower=F(condition['reference_current_Gram_lower'])
    charged_lower=Gneutral_lower/2-full_price
    charged_upper=F(norms['macro_reference_two_current_action_Gram_maximum_eigenvalue_upper'])/2+full_price
    contact_cap=F(0)
    root0=su(rational(mean['centre_enclosure']['g0_upper']))
    roots=su(rational(mean['centre_enclosure']['gs_upper']))
    for profile in build_native(mean,2)['Pprofiles']:
        row=scalars(profile)
        contact_cap=max(contact_cap,F(8,gd['V'])*sum(length**2*(abs(c0)*root0+abs(cj)*roots) for length,(c0,cj) in row['path'].items()))
    Cprice=exact(bridge['corrected_initial_whole_Theta_contact_operator_price'])
    contact_interval=bridge['corrected_full_H_initial_contact_interval']
    contact_lower=(exact(contact_interval[0])-contact_cap-Cprice)/2
    contact_upper=(exact(contact_interval[1])+contact_cap+Cprice)/2
    a=F(gd['a']);g=F(gd['UEphys'])/a
    if min(charged_lower,contact_lower)<=0 or full_price>=1:
        raise ArithmeticError('Actual charged photon source Gram/contact bound failed')
    desc=dict(plan=gd['plan'],plan_revision=REVISION,scientific_execution_revision=REVISION,input_execution_revision=transition['input_descriptor']['scientific_execution_revision'],input_receipts=dict(transition=dict(path=str(transition_path),sha256=hashlib.sha256(raw).hexdigest()),source_norm=dict(path=str(source_norm_path),sha256=hashlib.sha256(norm_raw).hexdigest())),source_abi=gd['source_abi'],N=gd['N'],V=gd['V'],M=gd['M'],a=gd['a'],UEphys=gd['UEphys'],Lambda=gd['Lambda'],variance=gd['variance'],clock_UNCHANGED=gd['clock'],source_position=gd['source_position'],Ref_position=gd['Ref_position'],charged_source='NEWlocal fullcolourE=.5 c† W_SU4 r, coherentALL4Refholes/Ref3/realGausslongdress, creationnorm².5. Original neutralHermYgraph000/AP222+Phi222 preparation unchanged.',law='89cd9 induced UEphys1/a=g64 and SAME64 rhoLplusrho, NOTphysicalC6a66 doubleC/.02/Rootkappa400',photon_probes='Allmacroallowednonzero/nonNyquist integerq alongx, realcosqx unitprofilesy/z witharbitrarycomplexpolarizations; physicalTheta=aA,JA=aJTheta',native_control=dict(N=2,source=[1,1,0],active_other=[1,1,1],Ref=[0,0,0],vertices='literaltwoorientedz-edge currents/contact, no macro promotion'))
    values=dict(actual_reference_CURRENT_onebody_operator_upper=kappa,actual_SOURCE_density_weighted_CURRENT_Gram_operator_price=density_price,reference_charged_CURRENT_Gram_vs_half_neutral_operator_price=reference_price,FULL_charged_CURRENT_Gram_vs_half_reference_neutral_operator_price=full_price,FULL_initial_unchanged_COHERENT_Refhole_CURRENT_Gram_minimum_eigenvalue_lower=charged_lower,FULL_initial_unchanged_COHERENT_Refhole_CURRENT_Gram_maximum_eigenvalue_upper=charged_upper,FULL_INITIAL_current_CREATOR_source_action_map_upper=su(charged_upper),FULL_physical_JA_CREATOR_SOURCE_Gram_minimum_eigenvalue_lower=a*a*charged_lower,charged_CURRENT_Gram_relative_to_half_neutral_reference_error=2*full_price/Gneutral_lower,actual_reference_CONTACT_onebody_operator_upper=contact_cap,FULL_INITIAL_unnormalized_charged_CONTACT_lower=contact_lower,FULL_INITIAL_unnormalized_charged_CONTACT_upper=contact_upper,EXACT_p_half_normalized_charged_Theta_weak_third_lower=8*g*g*contact_lower,EXACT_p_half_normalized_charged_Theta_weak_third_upper=8*g*g*contact_upper,EXACT_p_half_normalized_charged_physical_A_weak_third_lower=8*g*g*contact_lower/(a*a),EXACT_p_half_normalized_charged_physical_A_weak_third_upper=8*g*g*contact_upper/(a*a))
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),**{k:positive_exact(v) for k,v in values.items()},exact_charged_CREATOR_norm_squared='1/2',exact_INITIAL_source_raw_Theta_retarded_first_derivative=str(-g),exact_p_half_normalized_charged_Theta_retarded_first_derivative=str(-2*g),exact_p_half_normalized_charged_physical_A_retarded_first_derivative=str(-2*g/(a*a)),actual_native_charged_CAR_current_contact=actual_car_current(mean),
        SOURCE_Wick_and_commutator_proof='AtREFERENCEsource EACHTheta, originalall64pureSlaterP; nointeractingSlater assumption. EopfixedRef4<=1 andE†E=(2-rho_active,x)/4. M_fg=<Jf†rho_active,x Jg>=mu_all_f*TrPactive RxQhg+mu_all_g TrPactive hf†Q Rx+TrPactive hf†Q RxQhg−TrPactive hf†QhgPactive Rx, fullallflavourmeansBEFOREproducts, not63singlemean. RankRx4 givesfirsttwo<=4kappa(|mu_f|+|mu_g|), twoactivecubictraces<=8kappa². UniformcomplexSOURCEmeanPoincare E|mu_f|<=sqrt(s)Lip (actualGaussianmean0/twistedtranslations) yields||M||op<=8kappa(sqrt(s)Lip+kappa). EJ_SOURCEGram=.5Gneutral−.25M EXACT. [Jf,E] rank4creationhasneutralRef4 SOURCE/allblocknorm<=kappa (samefullcolourrank4traceargument); Gram(JE)−Gram(EJ) norm<=2Sref kappa+kappa². HencechargedreferenceG withinprice of.5neutralG; notassumeE/Jcommuting. FULLPhiJdifferenceb367ALLCARprice onchargedpacketnormsqrt(.5); fullcolourReftrace atEACHPhi/Theta cancelscreationW fromANYmultiplicativecurrentJ†J, soJref E_U CURRENTGram=Jref E0 Gram EXACT, noJop×Eerror/supportatlaterclock. Add2currentmapNormsqrt(.5)Jprice+.5Jprice² foractualFULLINITIALchargedGram. LowerboundneverreplacesactualcomplexGram/sourcevectors/unchangedneutralprobabilities.',
        charged_CONTACT_and_weak_Ward_proof='TracingcoherentRefholes givesaddedactiveDeltaP=Q_SOURCE RxQ_SOURCE/2, Tr1 andNOTphysicalnegativebandQsea/source-adaptedZemptyspin0. AnyonebodyCONTACT multiplication expectationafterE unnormalized=.5[neutralCONTACT+Trh_CONTACT DeltaP]; ALLother63unchangedconditionalmeans. Activeonebodycontactrowcap8/V sum_L L²(|c0|sqrtg0+|cj|sqrtgs); actualfullPhi contactfreezingoperatorprice added. NeutralFULLcontactintervalactualb367, no sourceΦ-framezeroing. C_yz=0 EXACTallPhi fromindependentaxialPeierls paths. WeakglobalfullHcore/P7: normp=.5 gives rawD_Theta_prime0=−2gp=−g, third=4g²<XiC,C XiC>. EXACTp-normalizationgives−2g andthird8g²contact_raw; physicalA fieldresponse /a², JAcurrentGram*a². Domainsremainoriginalcompactb2H2/SchwartzYjets/E smoothWilson/ref3 withrecomputedqC; noD(H²)/groundstationarityclaim. LiteralPhi/SU4momentumtermsnotdiscardedfrom[H,J]; universalJacobi/Thetaextension handlesfullH asinneutralP7.',
        scope='NEWchargedPHOTONCURRENT SOURCE Gram/contact/weakretardedinitialcoefficient inSAME89cd9 inducedfullPhi/EMlaw; parentreviewrequired fororderedWick3/rank4bounds/creator-tracePhi cancellation/contactdomain. ACTUALnativeall256CAR+Ref3/all63aux conditionalmixedmarginal sourcecontrol, notmacroGaussianquadrature/wholequantumcurrentpropagator. No ComptonphysicalBorn/massgap/pole/commoncone/PauliQstatclaim. SupplierZemptyspin0/transverseWilson chargedbeta andphysicalCdoubleC remainDISTINCTlaws/sources. No oldsource/prep/receiver/Z/clock/probabilityretag.',
        source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transition',type=Path,required=True)
    parser.add_argument('--source-norm',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.transition,args.source_norm)
    if not all(result['source_unchanged_since_import'].values()):
        raise ArithmeticError('Source changed during actual calculation')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('input_sha256','FULL_initial_unchanged_COHERENT_Refhole_CURRENT_Gram_minimum_eigenvalue_lower','actual_native_charged_CAR_current_contact','elapsed_seconds')},indent=2))
