"""Analytic neutral-dark Phi shape SOURCE channel, charged contact and quantum transfer."""
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
import induced_phi_force_quantum_heat as transfer
from induced_reference_electric_graph import from_spin_basis
quantum=transfer.quantum
SOURCE_BYTES={**transfer.SOURCE_BYTES,Path(__file__).name:OWN_BYTES}
REVISION='1178517cdc4236ce3a4af6a26bed880eb20679df709826463e4db7819ea0d208'
TRANSFER_SHA='e55618f0d8d812916893a8bf185f0767b8b39771f4006fe954f505f697c287ae'


def spin_shape(native:dict[str,Any],normal:np.ndarray,shape:np.ndarray)->dict[str,Any]:
    roots=np.sqrt([native['g0']]+[native['gs']]*3);gamma=native['frame']/roots[:,None,None]
    delta=np.zeros((4,4,4),dtype=complex)
    for i,(j,k) in enumerate(((2,3),(3,1),(1,2)),start=1):delta[i]=1j*gamma[j]@gamma[k]/np.sqrt(24)
    cart=transfer.packets.force.cart_coordinates(delta).reshape(60)
    coefficients=normal.T@cart
    return dict(delta=delta,cart=cart,coefficients=coefficients,packet46=np.r_[coefficients,0],Cart_norm=float(np.linalg.norm(cart)),normal_defect=float(np.linalg.norm(cart-normal@coefficients)),shape_defect=float(np.linalg.norm(cart-shape@(shape.T@cart))))


def derivative_profile(profile:dict[str,Any],delta:np.ndarray)->dict[str,Any]:
    onsite,_=transfer.packets.force.scalar_paths(profile['model'])
    return dict(profile,onsite=onsite*delta[0],paths=[dict(path,hop=path['seam']*np.einsum('a,aij->ij',path['scalars'],delta)) for path in profile['paths']])


def endpoint(native:dict[str,Any],profiles:list[dict[str,Any]],theta:np.ndarray,x:int)->tuple[list[np.ndarray],list[float]]:
    V=native['V'];frames=[];controls=[]
    for si,(profile,derivative) in enumerate(zip(native['Pprofiles'],profiles)):
        state=quantum.graph_state(native,profile,theta);F=quantum.matter_hamiltonian(native,derivative,theta)/np.sqrt(V)
        K=state['K']/np.sqrt(24*V*native['gs']);pred=np.zeros((4*V,4*V),dtype=complex)
        pred[np.ix_(state['pp'],state['pp'])]=K;pred[np.ix_(state['nn'],state['nn'])]=K
        physical=from_spin_basis(pred,state['B'])
        P=state['P'];controls.extend([float(np.linalg.norm(F-physical,2)),float(np.linalg.norm(P@F-F@P,2)),abs(complex(np.trace(P@F))),float(np.linalg.norm(state['empty'].conj().T@F@state['occupied'],2))])
        W=state['occupied']
        if si==0:
            f=np.zeros((4*V,4),dtype=complex);f[4*x:4*x+4]=np.eye(4)
            W=np.column_stack((W,F@f))
        frames.append(W)
    return frames,controls


def direct_contact_kernel(Fbra:list[np.ndarray],Fket:list[np.ndarray],words:list[np.ndarray],nocc:int)->complex:
    A=Fbra[0].conj().T@words[0]@Fket[0];B=quantum.OverlapCofactors(Fbra[1].conj().T@words[1]@Fket[1]).determinant
    return sum(quantum.OverlapCofactors(A[np.ix_(np.r_[np.arange(nocc),nocc+b],np.r_[np.arange(nocc),nocc+b])]).determinant for b in range(4))*B**63/4


def run(transfer_path:Path)->dict[str,Any]:
    started=time.monotonic();raw=transfer_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=TRANSFER_SHA:raise ArithmeticError('Actualb235 full45quantumtransfer epoch required')
    receipt=json.loads(raw);_,gram=quantum.checked(receipt['input_descriptor']['input_receipts']['gram']);_,force=quantum.checked(gram['input_descriptor']['input_receipts']['force']);_,join=quantum.checked(force['input_descriptor']['input_receipts']['join']);_,heat=quantum.checked(join['input_descriptor']['input_receipts']['heat'])
    mean=transfer.packets.force.joined.mean_from_source(heat);native=quantum.build_native(mean,2);V,M=native['V'],native['M'];g=64.;s=2.
    normal=np.asarray(force['actual_geometry']['normal_basis']);shape=np.asarray(force['actual_geometry']['shape_basis']);mode=spin_shape(native,normal,shape)
    u=mode['packet46'];profiles=[derivative_profile(profile,mode['delta']) for profile in native['Pprofiles']]
    descriptor0=receipt['input_descriptor'];beta=quantum.F(descriptor0['beta']);samples=descriptor0['samples'];steps=descriptor0['steps'];seed=descriptor0['seed'];maximum=max(steps);bfloat=float(beta)
    x=int(np.flatnonzero(np.all(native['sites']==(1,1,0),axis=1))[0]);ref=0
    lv,lb=np.linalg.eigh(native['Lplus']);positive=lv>1e-12;A0map=lb[:,positive]*np.sqrt(lv[positive])
    if A0map.shape[1]!=V-1:raise ArithmeticError('RealprimitiveGaussA0 rank failed')
    free_energy=sum(profile['load']*np.linalg.eigvalsh(quantum.matter_hamiltonian(native,profile,np.zeros(3*V)))[:2*V].sum() for profile in native['Pprofiles'])
    scalar=(1+g*bfloat/(2*s))**(-M/2)*np.exp(bfloat*free_energy)
    rng=np.random.default_rng(seed);fractions=np.arange(2*maximum+1)/(2*maximum);ledgers={n:[] for n in steps};algebra=[];initial=[]
    for _ in range(samples):
        middle=rng.normal(0,np.sqrt(s),M);difference=rng.normal(0,np.sqrt(2*g*bfloat/(1+g*bfloat/(2*s))),M)
        increments=rng.normal(0,np.sqrt(g*bfloat/maximum),(2*maximum,M));motion=np.vstack((np.zeros(M),np.cumsum(increments,axis=0)));bridge=motion-fractions[:,None]*motion[-1]
        coordinates=middle+(fractions[:,None]-.5)*difference+bridge;fields=coordinates@native['Q'].T
        alpha_fine=np.sqrt(2*g*bfloat/maximum)*rng.normal(size=(maximum,V-1))@A0map.T;summed=alpha_fine.sum(axis=0);phase=np.exp(1j*(128*summed.sum()+summed[ref]))
        Fket,controls=endpoint(native,profiles,fields[0],x);algebra.extend(controls)
        Fbra,controls=endpoint(native,profiles,fields[-1],x);algebra.extend(controls)
        # OriginalsourceGaussian atMIDDLE, notendpointproposal marginal.
        theta=native['Q']@middle;state=quantum.graph_state(native,native['Pprofiles'][0],theta);F=quantum.matter_hamiltonian(native,profiles[0],theta)/np.sqrt(V);Q=np.eye(4*V)-state['P'];D=Q[:,4*x:4*x+4]@Q[4*x:4*x+4,:]/2
        initial.append(float(np.real(np.trace(D@F@F)))/2)
        for n in steps:
            width=maximum//n;dt=bfloat/n;words=[np.eye(4*V,dtype=complex),np.eye(4*V,dtype=complex)]
            for j in range(n):
                theta=fields[width*(2*j+1)];alpha=alpha_fine[j*width:(j+1)*width].sum(axis=0);density=np.repeat(np.exp(-1j*alpha),4)
                for si,profile in enumerate(native['Pprofiles']):
                    H=quantum.half_heat(native,profile,theta,dt);words[si]=((H*density[None,:])@H)@words[si]
            ledgers[n].append(scalar*phase*direct_contact_kernel(Fbra,Fket,words,2*V))
    GN=transfer.packets.force.joined.receiver.matrix(gram['actual_ORIGINAL_neutral_quantum_FORCE45_and_complex_JA_SOURCE_Gram']);GC=transfer.packets.force.joined.receiver.matrix(gram['actual_ORIGINAL_RAWcharged_quantum_SOURCE_Gram'])
    source_N=complex(np.vdot(u,GN@u));source_C=complex(np.vdot(u,GC@u));rows={};comparison_errors=[]
    for n in steps:
        old=receipt['actual_all45_Phi_force_and_complex_current_quantum_EM_Coulomb_transfer'][str(n)]['actual_RAWneutral_and_charged_force45_complexJA_SOURCE_heat'];mat=np.asarray(old['real'])+1j*np.asarray(old['imag'])
        neutral=complex(np.vdot(u,mat[0]@u));charged=complex(np.vdot(u,mat[1]@u));direct=complex(np.mean(ledgers[n]));comparison_errors.append(abs(direct-charged))
        rows[str(n)]=dict(actual_direct_ALL4_contact_source_quantum_heat=quantum.summarize(np.asarray(ledgers[n])),independent_FULL46packet_charged_source_heat=dict(real=charged.real,imag=charged.imag),independent_FULL46packet_neutral_dark_source_heat=dict(real=neutral.real,imag=neutral.imag),direct_vs_full46_contact_identity_error=abs(direct-charged),neutral_dark_packet_exact='0 atANYbeta/S_n/quantumU; FdarkPsi_N=0 pointwise, notlateFdarkUNPsi=0')
    if abs(mode['Cart_norm']-1)>1e-10 or mode['normal_defect']>1e-10 or mode['shape_defect']>1e-10 or max(algebra)>1e-9 or max(comparison_errors)>1e-7:raise ArithmeticError('Actualspinshape/contact/sourceannihilator quantumidentity failed')
    descriptor=dict(plan=descriptor0['plan'],plan_revision=REVISION,scientific_execution_revision=REVISION,input_receipts=dict(transfer=dict(path=str(transfer_path),sha256=hashlib.sha256(raw).hexdigest())),source_abi=descriptor0['source_abi'],native_N=2,beta=str(beta),samples=samples,steps=steps,seed=seed,source='ORIGINALunfilteredGaussian2/HermY000AP222, all64CAR, rawALL4RefholeE/2x110 Ref0; newPHYSICALforceprobe ONLY, no sourcefilter/eigenfit/removal',Phi_shape='dPhi0=0,dPhi_i=iGamma_jGamma_k/sqrt24 cyclici,j,k; primitiveCartunitfull35shape, not10Gram/gauge15',clock4096_UNCHANGED=True)
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),actual_analytic_shape_mode=dict(Cart=mode['cart'].tolist(),normal45_coefficients=mode['coefficients'].tolist(),Cart_norm=mode['Cart_norm'],normal_defect=mode['normal_defect'],shape_defect=mode['shape_defect']),actual_ALLtheta_sample_dark_operator_algebra_max_error=max(algebra),actual_OLDfull46_neutral_source_dark_Gram=dict(real=source_N.real,imag=source_N.imag),actual_OLDfull46_RAWcharged_source_contact_Gram=dict(real=source_C.real,imag=source_C.imag),actual_fresh_ORIGINALsource_RAWcharged_contact_Gram_sampling=quantum.summarize(np.asarray(initial)),actual_same_H_quantum_contact_and_dark_packet_transfer=rows,exact_source_dark_channel_proof='NativeactualGamma5=Gamma0Gamma1Gamma2Gamma3 andB=[plus,Gamma5plus] fixorientation. Sigma_i=+iGamma_jGamma_k cyclic hasBblocksdiag(S_i,S_i), wherephysicalGamma_i hasoffdiag(+iS_i,-iS_i). Sourceh_K offdiag(+iK,-iK), K=sqrt(gs)sum Peierls ci S_i. ThereforeuniformCartunitspinshape dPhi_i=Sigma_i/sqrt24 givesphysicalq0 f_dark=diag(K,K)/sqrt(24Vgs). CanonicalHermY sourceP ismatrixfunctionofY=K/d: [P,f_dark]=0, Wdagf_darkW=K/sqrt(24Vgs), TrK=0. ThusEACHflavour dGamma(f_dark)Psi_flavour=0, ALL64sumannihilatesORIGINALneutralSOURCEpointwiseALLrealTheta. AnyquantumU/beta/Trotter SOURCEheatpacket then0, withoutdroppingphysicalmode orassuminglateinteractingSlater.',charged_opening_and_transfer_proof='Fdark Epsi=[Fdark,E]psi=.5sumALL4 cdag(f_dark f_xb)rRefb psi becauseFdarkpsi0. ThisisactualPHYSICALchargedcontactprobe, notmodifiedbaseSOURCE/filter. NormalizedchargedsourceD=Q_SOURCE Rx Q_SOURCE/2 and[P,f_dark]=0 implyrawcontactGram=p TrD f_dark²,p=.5. Directquantumheat thereforeusesliteralALL4rank17Slaterminorframes[W16,f_dark(theta)f_xb] onbothends, timesaux16det^63/4 andRef3phase/SAMEneutraloffset. Thiscontactreduction isEXACTinitialpacketidentity throughfullyinteractingprimitivePi/Coulomb/all64CAR quantumtransfer, NOTevolvingSlaterclosure. Actualdirectcoupledquantumcalculation agreesindependentfull46mixedjetheat. Modefull35shape/gaugeorthogonal analyticallyfixed, notsampleeigenvectorfit. Chargedopensotherwise SOURCE-darkshapechannel; no claimphysicalPhi mode absent/zeroenergy/massless.',scope='ACTUALneutralSOURCEannihilator/chargedopening+native interactingrealEM/Coulomb quantumtransfer. DoesNOTsayFdark U_NPsi stays0 orfullPhi/shapeHamiltoniansterile. No actualPhi/heavy/SU4time evolution/newmean/pole/commoncone/Compton; finiteMC/Strang, no exactsourceGramwhitening/PSDclip/wholePhianalyticcontinuation. Alloldmodes/laws/sourcebytes andsourceprobabilities/clocks preserved.',source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transfer',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run(args.transfer)
    if not all(result['source_unchanged_since_import'].values()):raise ArithmeticError('Source changed during actual calculation')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('input_sha256','actual_ALLtheta_sample_dark_operator_algebra_max_error','actual_OLDfull46_neutral_source_dark_Gram','actual_OLDfull46_RAWcharged_source_contact_Gram','actual_same_H_quantum_contact_and_dark_packet_transfer','elapsed_seconds')},indent=2))
