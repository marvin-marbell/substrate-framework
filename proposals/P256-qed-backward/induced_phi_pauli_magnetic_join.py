"""Physical betaSigma_z/sin(qx) Phi35 probe, raw joined quantum source kernels."""
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
from induced_reference_electric_graph import to_spin_basis
from induced_phi_em_macro_preparation import scalars

ROOT=Path(__file__).resolve().parent
OWN_BYTES=Path(__file__).read_bytes()
import induced_phi_spin_photon_joined_heat as joined
contact=joined.contact
quantum=joined.quantum
SOURCE_BYTES={**joined.SOURCE_BYTES,Path(__file__).name:OWN_BYTES}
REVISION='d86132736ac1f3d1592b01da60e6e981c5005500265f0d012e979610896318c7'
JOIN_SHA='874b20286c2023a0291cbbd9660d2a4256c13258c9172b687751ec62df57febd'


def magnetic_shape(native:dict[str,Any])->dict[str,Any]:
    gamma=native['frame']/np.sqrt([native['g0']]+[native['gs']]*3)[:,None,None]
    delta=np.zeros((4,4,4),dtype=complex);delta[0]=gamma[0]@(1j*gamma[1]@gamma[2])/np.sqrt(8)
    cart=contact.transfer.packets.force.cart_coordinates(delta).reshape(60);geometry=contact.transfer.packets.force.physical_normal_geometry(native)
    shape_error=float(np.linalg.norm(cart-geometry['shape']@(geometry['shape'].T@cart)))
    f=np.sqrt(2/native['V'])*np.sin(2*np.pi*native['sites'][:,0]/4)
    if shape_error>1e-10 or abs(np.linalg.norm(cart)-1)>1e-10 or abs(np.linalg.norm(f)-1)>1e-10:raise ArithmeticError('ActualunitnormalGram-zeroPauli35shape failed')
    return dict(delta=delta[0],mode=f,Cart_norm=float(np.linalg.norm(cart)),true35shape_error=shape_error,spatial_mode_norm=float(np.linalg.norm(f)))


def operator(native:dict[str,Any],profile:dict[str,Any],theta:np.ndarray,mode:dict[str,Any])->np.ndarray:
    V=native['V'];onsite,_=contact.transfer.packets.force.scalar_paths(profile['model']);H=np.zeros((4*V,4*V),dtype=complex)
    for x in range(V):H[4*x:4*x+4,4*x:4*x+4]=onsite*mode['mode'][x]*mode['delta']
    for path in profile['paths']:
        x,y=path['source'],path['target'];coef=path['scalars'][0]*(mode['mode'][x]+mode['mode'][y])/2
        hop=coef*mode['delta']*path['seam']*np.exp(1j*np.sum(theta[list(path['edges'])]))
        H[4*x:4*x+4,4*y:4*y+4]+=hop;H[4*y:4*y+4,4*x:4*x+4]+=hop.conj().T
    return H


def endpoints(native:dict[str,Any],theta:np.ndarray,mode:dict[str,Any])->tuple[list[dict[str,Any]],list[np.ndarray],list[np.ndarray]]:
    states=[quantum.graph_state(native,p,theta) for p in native['Pprofiles']]
    return states,[state['occupied'] for state in states],[operator(native,p,theta,mode) for p in native['Pprofiles']]


def jet(left:np.ndarray,right:np.ndarray,Fbra:np.ndarray,Fket:np.ndarray,word:np.ndarray)->tuple[complex,complex,complex,complex]:
    UR=word@right;UF=word@(Fket@right);LF=Fbra@left
    B=left.conj().T@UR;L=LF.conj().T@UR;R=left.conj().T@UF;S=LF.conj().T@UF
    return quantum.OverlapCofactors(B).jet(L,R,S)


def source_kernel(native:dict[str,Any],left:list[np.ndarray],right:list[np.ndarray],Fbra:list[np.ndarray],Fket:list[np.ndarray],words:list[np.ndarray],x:int)->np.ndarray:
    creator=np.zeros((4*native['V'],4),dtype=complex);creator[4*x:4*x+4]=np.eye(4)
    aux=jet(left[1],right[1],Fbra[1],Fket[1],words[1]);b,bL,bR,_=aux;result=np.zeros((2,2),dtype=complex)
    for hole in range(4):
        L=np.column_stack((left[0],creator[:,hole]));R=np.column_stack((right[0],creator[:,hole]))
        active=jet(L,R,Fbra[0],Fket[0],words[0]);a,aL,aR,_=active
        norm,mixed=quantum.combine(active,aux)
        firstL=aL*b**63+63*a*bL*b**62;firstR=aR*b**63+63*a*bR*b**62
        result+=np.array([[mixed,firstL],[firstR,norm]])/4
    return result


def native_PH_gap(native:dict[str,Any])->dict[str,float]:
    p=native['Pprofiles'][1];scalar=scalars(p);onsite=abs(float(scalar['onsite']));two=abs(complex(scalar['path'][2][0]));margin=onsite-2*two
    spatial=joined.full.coefficient_data(p)[1];k=abs(complex(scalar['path'][1][1]))*np.sqrt(native['gs'])
    state=quantum.graph_state(native,p,np.zeros(3*native['V']));d=state['denominator'];Kcap=2*float(spatial)*np.sqrt(native['gs'])
    if margin<=0:raise ArithmeticError('ActualimprovedWilsonone-z-linkPHmargin notpositive')
    frob=2*k*k*margin*margin;lower=frob/(d*d*(1+(Kcap/d)**2)**2)
    return dict(onsite_scalar=onsite,two_link_Phi0_coefficient_abs=two,one_z_link_margin=margin,source_K_z_coefficient_abs=k,source_Y_denominator=d,source_K_uniform_operator_cap=Kcap,unique_one_z_scalar_spin_anticommutator_Frobenius_squared_lower=frob,uniform_aux_PH_norm_squared_lower=lower,RAWcharged_aux63_PH_Schur_lower=63*lower/2)


def actual_kernel(mean:dict[str,Any],beta:F,samples:int,steps:list[int],seed:int)->dict[str,Any]:
    native=quantum.build_native(mean,4);V,M=native['V'],native['M'];a=float(native['a']);g=float(native['UE']/native['a']);s=2.;mode=magnetic_shape(native);gap=native_PH_gap(native)
    x=int(np.flatnonzero(np.all(native['sites']==(1,1,0),axis=1))[0]);ref=0;f=np.sqrt(2/V)*np.cos(2*np.pi*native['sites'][:,0]/4)
    directions=np.zeros((2,3*V));directions[0,1::3]=f;directions[1,2::3]=f
    maximum=max(steps);bfloat=float(beta);fractions=np.arange(2*maximum+1)/(2*maximum)
    if beta<=0 or samples<4 or min(steps)<=0 or any(maximum%n for n in steps):raise ValueError('Positivebeta/sampling/pairedStrang contract required')
    lv,lb=np.linalg.eigh(native['Lplus']);A0map=lb[:,lv>1e-12]*np.sqrt(lv[lv>1e-12])
    if A0map.shape[1]!=V-1:raise ArithmeticError('ActualGaussCoulombrank changed')
    Efree=sum(p['load']*np.linalg.eigvalsh(quantum.matter_hamiltonian(native,p,np.zeros(3*V)))[:2*V].sum() for p in native['Pprofiles'])
    scalar=(1+g*bfloat/(2*s))**(-M/2);normalizer=scalar*np.exp(bfloat*Efree)
    rng=np.random.default_rng(seed);ledgers={n:[] for n in steps};source=[];PH=[];anticom_errors=[];reverse_errors=[];identities=[np.eye(4*V,dtype=complex)]*2
    for sample in range(samples):
        middle=rng.normal(0,np.sqrt(s),M);difference=rng.normal(0,np.sqrt(2*g*bfloat/(1+g*bfloat/(2*s))),M)
        motion=np.vstack((np.zeros(M),np.cumsum(rng.normal(0,np.sqrt(g*bfloat/maximum),(2*maximum,M)),axis=0)))
        coordinates=middle+(fractions[:,None]-.5)*difference+motion-fractions[:,None]*motion[-1];fields=coordinates@native['Q'].T
        fine_A0=np.sqrt(2*g*bfloat/maximum)*rng.normal(size=(maximum,V-1))@A0map.T
        _,Wk,Fk=endpoints(native,fields[0],mode);_,Wb,Fb=endpoints(native,fields[-1],mode)
        states,Wm,Fm=endpoints(native,native['Q']@middle,mode);tm=directions@(native['Q']@middle)
        source.append(joined.lift(source_kernel(native,Wm,Wm,Fm,Fm,identities,x),tm,tm,a))
        state=states[1];ph=state['empty'].conj().T@Fm[1]@state['occupied'];observed=float(np.sum(abs(ph)**2));PH.append(observed)
        hb=to_spin_basis(Fm[1],state['B']);A=hb[np.ix_(state['pp'],state['pp'])];rv,rw=np.linalg.eigh(state['R']);rootR=(rw*np.sqrt(rv))@rw.conj().T
        predicted=rootR@(state['Y']@A+A@state['Y'])@rootR;prednorm=float(np.sum(abs(predicted)**2));anticom_errors.append(abs(observed-prednorm))
        if observed+1e-8<gap['uniform_aux_PH_norm_squared_lower']:raise ArithmeticError('ActualauxPHbelowproposeduniformone-z-link bound')
        tb=directions@fields[-1];tk=directions@fields[0];alpha_sum=fine_A0.sum(axis=0);phase=np.exp(1j*(128*alpha_sum.sum()+alpha_sum[ref]))
        for n in steps:
            width=maximum//n;dt=bfloat/n;words=[np.eye(4*V,dtype=complex),np.eye(4*V,dtype=complex)];blocks=[]
            for j in range(n):
                alpha=fine_A0[j*width:(j+1)*width].sum(axis=0);D=np.repeat(np.exp(-1j*alpha),4)
                for si,p in enumerate(native['Pprofiles']):
                    H=quantum.half_heat(native,p,fields[width*(2*j+1)],dt);words[si]=((H*D[None,:])@H)@words[si]
                    if sample==0:blocks.append((si,H,alpha))
            K=normalizer*phase*joined.lift(source_kernel(native,Wb,Wk,Fb,Fk,words,x),tb,tk,a);ledgers[n].append(K)
            if sample==0:
                reverse=[np.eye(4*V,dtype=complex),np.eye(4*V,dtype=complex)]
                for si,H,alpha in reversed(blocks):reverse[si]=((H*np.repeat(np.exp(1j*alpha),4)[None,:])@H)@reverse[si]
                KR=normalizer*phase.conjugate()*joined.lift(source_kernel(native,Wk,Wb,Fk,Fb,reverse,x),tk,tb,a);reverse_errors.append(float(np.linalg.norm(KR-K.conj().T,2)))
    if max(anticom_errors)>1e-8 or max(reverse_errors)>1e-6:raise ArithmeticError('ActualPauliPH/reverse sourcecontract failed')
    rows=[]
    for n in steps:
        raw=np.array(ledgers[n]);paired=(raw+raw.conj().transpose(0,2,1))/2
        rows.append(dict(Strang_steps=n,RAW_ALL3_source_quantum_heat=quantum.summarize(raw),actual_reversepaired_RAW_heat=quantum.summarize(paired),charged_Phi_Pauli_to_Ay_RAW_transfer=quantum.summarize(paired[:,0,1]),charged_Phi_Pauli_to_Az_RAW_transfer=quantum.summarize(paired[:,0,2]),source_Gram_NOT_numerically_whitened=True))
    return dict(N=4,V=V,M=M,beta=str(beta),samples=samples,seed=seed,Strang_steps=steps,source_position=[1,1,0],real_EM_modes=M,shared_REAL_A0_modes=V-1,ALLcharged_flavours=64,g_Pi_AND_Coulomb=g,a=a,mode_definition='deltaPhi0(x)=sqrt(2/V)sin(2pi x/N)*Gamma0[iGamma1Gamma2]/sqrt8; trueCart35Pauli betaSigma_z, NOTSigma_z alone. PhotonA_y/zunitcos(qx), Bz=partialxAy hasSAMEsineprofile.',Cart_norm=mode['Cart_norm'],true35shape_error=mode['true35shape_error'],spatial_Phi_mode_norm=mode['spatial_mode_norm'],RAW_SOURCE_Gram_Gaussian_sampling=quantum.summarize(np.array(source)),EXACT_SOURCE_photon_A_Gram=[['4096','0'],['0','4096']],native_uniform_aux_PH_proposed_bound=gap,actual_aux_PH_norm_squared_range=[min(PH),max(PH)],actual_CHIRAL_PH_anticommutator_norm_identity_error=max(anticom_errors),actual_reversed_physical_word_ALL3_SOURCE_error=max(reverse_errors),rows=rows,samepath_Strang_differences=[dict(coarse=m,fine=n,RAW_difference=quantum.summarize(np.array(ledgers[n])-np.array(ledgers[m]))) for m,n in zip(steps[:-1],steps[1:])],uniform_PH_proof='SUBMITTEDparentmathreview. InsourceGamma0spinbasis PauliF=diag(A,−A), A=Sigma_z*C_sin/sqrt8. VemptydagF W=−i sqrtR{Y,A}sqrtR, Y=K/d. Scalarspintraceof{K,A} picksONLYK_z, sinceSigma_zPaulianticommutesK_x/y. sin(qx) constantalongz; uniqueone-z-linkblock fromonsite andtwo-linkzbacktracking hascoefficient2ci sqrtgs f_sin(onsite−c02)/sqrt8. AtN4 extra3linkfold maysubtractatmost|c02|; cross-axispaths andone-linkWilsongiveOTHERdisplacementsortracelessspin, hencefrobenius²>=2|ci|²gs(onsite−2|c02|)² forEVERYTheta. sqrtRlower yieldsPH²>=frob/[d²(1+(Kcap/d)²)²]. All63auxPHsectorsorthogonaltoactive/photoncarrier, RAWchargedp=.5 =>sourceGramSchurPhi>=p63PHlower. ThisisspecificPauli-probeproof, NOTassertionarbitraryforcepacketSPD. Boundisnativefloatcontrol/proposedlemma here; exactMACROoutwardpricesnotyetconstructed.',force_component_scope='ObservableONLY64CHARGEDmatterpotential-gradient, ALL64coherentmeans BEFOREproducts; forcepacketsinclude63auxPH, notjustactivecommutator. Neutralheavy/Phi-boson/Noe-self-forceNOTincludedintheprobe, NEVERdeclaredzero/removedfromfullH.',physics_boundary='OriginalUNFILTEREDsourceE BEFOREheat, Ref3ALL4minors/4, sameNEUTRALfreeCARoffset. ACTUALprimitiveg64Pi/all129REALbridges/shared63A0C/all64CARfiniteStrang quantumheat. FullPhi/heavy/colourtimeNOTsimulated andunitaryPhiwavepricesNOTextendedtoheat. RawsourceGramestimatesNOTusedtocalibratemetricorSPD/pole/speed/commoncone/massgap/Compton. Proposalnotachievedmacrotheorem; rawcomplexphase/signkept/noPSDclip.')


def run(join_path:Path,beta:F,samples:int,steps:list[int],seed:int)->dict[str,Any]:
    started=time.monotonic();raw=join_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=JOIN_SHA:raise ArithmeticError('Frozen13ea/874b observation epoch required')
    cap=json.loads(raw);_,f=quantum.checked(cap['input_descriptor']['input_receipts']['full_source']);_,c=quantum.checked(f['input_descriptor']['input_receipts']['contact']);_,t=quantum.checked(c['input_descriptor']['input_receipts']['transfer']);_,g=quantum.checked(t['input_descriptor']['input_receipts']['gram']);_,force=quantum.checked(g['input_descriptor']['input_receipts']['force']);_,stage=quantum.checked(force['input_descriptor']['input_receipts']['join']);_,heat=quantum.checked(stage['input_descriptor']['input_receipts']['heat']);mean=contact.transfer.packets.force.joined.mean_from_source(heat)
    desc=dict(plan=cap['input_descriptor']['plan'],plan_revision=REVISION,scientific_execution_revision=REVISION,input_receipts=dict(prior_joined_heat=dict(path=str(join_path),sha256=hashlib.sha256(raw).hexdigest())),source_abi=cap['input_descriptor']['source_abi'],native_N=4,beta=str(beta),samples=samples,seed=seed,Strang_steps=steps,source='UNCHANGEDoriginalunfilteredcanonicalY/all64/Gaussian2/AP222/Ref4to3E0; SAMEactualmacroPhiLambda2^8192meanmidpointnativefrozenimage, NOTnewmean/source/filter',probe='PhysicallydefinedTRUE35shapeDiracPauli betaSigma_z/sinqx alignedmagneticBz ofcosqxAy; notfittedkernel/eigenvector/selectedsurvivalgate; PHall64 meanscoherent',prior_observation='Cyclicneutral-darkq0spincomponentphotonjointheatcrossroundoff, preservedunchanged. NewPauliPhi35vertex isdistinctphysicalobservable, notretagthatresult.')
    desc['native_source_counts']=dict(active=[128,129],auxiliary_species=63,auxiliary_each=128,neutralheavy_species=1,neutralheavy=128,Ref=[4,3],total_CAR=8324,Z4_mod4=0)
    desc['actual_law_and_spacing']=dict(a='1/64',UEphys='1',g_Pi_AND_Coulomb='64',active_bare_mass='1/5',auxiliary_bare_mass='2',charged_count=64,neutralheavy_EMcharge='0',neutralheavy_bare_mass='1/5',Lambda='2^8192',chi='2^-4096',field_scope='SameoriginalmacroLAW89 inputs, nativefrozenC000chargedreferenceONLY; Phi/neutralheavy/SU4timeNOTnumericallypropagated, notremovedfromfullH.')
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),actual_Pauli_magnetic_ALL64_Phi_component_photon_quantum_heat=actual_kernel(mean,beta,samples,steps,seed),source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--join',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--beta',type=F,default=F(1,4096));p.add_argument('--samples',type=int,default=32);p.add_argument('--steps',type=int,nargs='+',default=[1,2,4]);p.add_argument('--seed',type=int,default=20261016)
    a=p.parse_args();r=run(a.join,a.beta,a.samples,a.steps,a.seed)
    if not all(r['source_unchanged_since_import'].values()):raise ArithmeticError('Source changed duringscientificcalculation')
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(input_sha256=r['input_sha256'],physical=r['actual_Pauli_magnetic_ALL64_Phi_component_photon_quantum_heat'],elapsed_seconds=r['elapsed_seconds']),indent=2))
