"""N4 all-mode primitive quantum transfer of charged-Phi-component/A source packets."""
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

ROOT=Path(__file__).resolve().parent
OWN_BYTES=Path(__file__).read_bytes()
import induced_phi_spin_shape_full_source as full
contact=full.contact
quantum=full.quantum
SOURCE_BYTES={**full.SOURCE_BYTES,Path(__file__).name:OWN_BYTES}
REVISION='d86132736ac1f3d1592b01da60e6e981c5005500265f0d012e979610896318c7'
FULL_SHA='c2e4ef0ea2083693b81c6e3d8ee120d8c53a7cacc12f7a57132617872177ff80'


def endpoint(native:dict[str,Any],theta:np.ndarray,x:int,derivative:dict[str,Any])->tuple[list[np.ndarray],np.ndarray]:
    frames=[quantum.graph_state(native,p,theta)['occupied'] for p in native['Pprofiles']]
    creator=np.zeros((4*native['V'],4),dtype=complex);creator[4*x:4*x+4]=np.eye(4)
    force=quantum.matter_hamiltonian(native,derivative,theta)/np.sqrt(native['V'])
    return frames,np.stack((force@creator,creator))


def border_kernel(left:list[np.ndarray],right:list[np.ndarray],bra:np.ndarray,ket:np.ndarray,words:list[np.ndarray])->np.ndarray:
    """ALL4Refhole sum of rank-(2V+1) Slaters, singular-safe firstcompound."""
    UR=words[0]@right[0];UV=np.einsum('xy,jyb->jxb',words[0],ket)
    base=quantum.OverlapCofactors(left[0].conj().T@UR)
    beta=np.einsum('ixb,xk->ibk',bra.conj(),UR)@base.right
    gamma=np.einsum('kx,jxb->jkb',base.left@left[0].conj().T,UV)
    direct=np.einsum('ixb,jxb->ij',bra.conj(),UV)
    border=base.determinant*direct-base.phase*np.einsum('ibk,k,jkb->ij',beta,base.one,gamma)
    auxiliary=quantum.OverlapCofactors(left[1].conj().T@words[1]@right[1]).determinant
    return border*auxiliary**63/4


def lift(kernel:np.ndarray,bra_theta:np.ndarray,ket_theta:np.ndarray,a:float)->np.ndarray:
    L=np.array([[1,0],[0,bra_theta[0]/a],[0,bra_theta[1]/a]],dtype=complex)
    R=np.array([[1,0],[0,ket_theta[0]/a],[0,ket_theta[1]/a]],dtype=complex)
    return L.conj()@kernel@R.T


def literal_complex_minor_control(left:list[np.ndarray],right:list[np.ndarray],bra:np.ndarray,ket:np.ndarray,words:list[np.ndarray],matrix:np.ndarray)->float:
    u=np.array([.3+.4j,-.7+.2j]);v=np.array([-.2+.1j,.4+.6j]);observed=0j
    l=np.einsum('i,ixb->xb',u,bra);r=np.einsum('i,ixb->xb',v,ket)
    for b in range(4):
        L=np.column_stack((left[0],l[:,b]));R=np.column_stack((right[0],r[:,b]))
        observed+=np.linalg.det(L.conj().T@words[0]@R)/4
    observed*=np.linalg.det(left[1].conj().T@words[1]@right[1])**63
    return float(abs(observed-u.conj()@matrix@v))


def actual_kernel(mean:dict[str,Any],beta:F,samples:int,steps:list[int],seed:int)->dict[str,Any]:
    native=quantum.build_native(mean,4);V,M=native['V'],native['M'];a=float(native['a']);g=1/a;s=2.
    if a!=1/64 or samples<4 or beta<=0 or min(steps)<=0 or any(max(steps)%n for n in steps):raise ValueError('Actuala/positivebeta/sampling/pairedStrang levels required')
    geometry=contact.transfer.packets.force.physical_normal_geometry(native);mode=contact.spin_shape(native,geometry['normal'],geometry['shape'])
    derivative=contact.derivative_profile(native['Pprofiles'][0],mode['delta'])
    x=int(np.flatnonzero(np.all(native['sites']==(1,1,0),axis=1))[0]);ref=0
    f=np.sqrt(2/V)*np.cos(2*np.pi*native['sites'][:,0]/4);directions=np.zeros((2,3*V));directions[0,1::3]=f;directions[1,2::3]=f
    squares,_=full.coefficient_data(native['Pprofiles'][0]);Gamma=np.array([float(squares/(24*V)),s/(2*a*a),s/(2*a*a)])
    root=np.sqrt(Gamma);maximum=max(steps);bfloat=float(beta);fractions=np.arange(2*maximum+1)/(2*maximum)
    lv,lb=np.linalg.eigh(native['Lplus']);A0map=lb[:,lv>1e-12]*np.sqrt(lv[lv>1e-12])
    if A0map.shape[1]!=V-1:raise ArithmeticError('PhysicalCoulomb zeromode/rank changed')
    free_energy=sum(p['load']*np.linalg.eigvalsh(quantum.matter_hamiltonian(native,p,np.zeros(3*V)))[:2*V].sum() for p in native['Pprofiles'])
    free_scalar=(1+g*bfloat/(2*s))**(-M/2);normalizer=free_scalar*np.exp(bfloat*free_energy)
    rng=np.random.default_rng(seed);ledgers={n:[] for n in steps};creator_ledgers={n:[] for n in steps};electric=[];source_errors=[];reverse_errors=[];minor_errors=[]
    identities=[np.eye(4*V,dtype=complex)]*2
    for sample in range(samples):
        middle=rng.normal(0,np.sqrt(s),M);difference=rng.normal(0,np.sqrt(2*g*bfloat/(1+g*bfloat/(2*s))),M)
        motion=np.vstack((np.zeros(M),np.cumsum(rng.normal(0,np.sqrt(g*bfloat/maximum),(2*maximum,M)),axis=0)))
        coordinates=middle+(fractions[:,None]-.5)*difference+motion-fractions[:,None]*motion[-1]
        fields=coordinates@native['Q'].T;fine_A0=np.sqrt(2*g*bfloat/maximum)*rng.normal(size=(maximum,V-1))@A0map.T
        Wk,Vk=endpoint(native,fields[0],x,derivative);Wb,Vb=endpoint(native,fields[-1],x,derivative)
        same=border_kernel(Wk,Wk,Vk,Vk,identities);source_errors.append(float(np.linalg.norm(same-np.diag([Gamma[0],.5]),2)))
        tk=directions@fields[0];tb=directions@fields[-1];alpha_sum=fine_A0.sum(axis=0)
        charged_phase=np.exp(1j*(128*alpha_sum.sum()+alpha_sum[ref]))
        electric.append(free_scalar*lift(border_kernel(Wb,Wk,Vb,Vk,identities),tb,tk,a))
        for n in steps:
            width=maximum//n;dt=bfloat/n;words=[np.eye(4*V,dtype=complex),np.eye(4*V,dtype=complex)];blocks=[]
            for j in range(n):
                alpha=fine_A0[j*width:(j+1)*width].sum(axis=0);D=np.repeat(np.exp(-1j*alpha),4)
                for si,p in enumerate(native['Pprofiles']):
                    H=quantum.half_heat(native,p,fields[width*(2*j+1)],dt)
                    words[si]=((H*D[None,:])@H)@words[si]
                    if sample==0:blocks.append((si,H,alpha))
            core=border_kernel(Wb,Wk,Vb,Vk,words);K=normalizer*charged_phase*lift(core,tb,tk,a)
            ledgers[n].append(K);creator_ledgers[n].append(normalizer*charged_phase*core[1,1])
            if sample==0:
                minor_errors.append(abs(normalizer)*literal_complex_minor_control(Wb,Wk,Vb,Vk,words,core))
                reverse=[np.eye(4*V,dtype=complex),np.eye(4*V,dtype=complex)]
                for si,H,alpha in reversed(blocks):reverse[si]=((H*np.repeat(np.exp(1j*alpha),4)[None,:])@H)@reverse[si]
                KR=normalizer*charged_phase.conjugate()*lift(border_kernel(Wk,Wb,Vk,Vb,reverse),tk,tb,a)
                reverse_errors.append(float(np.linalg.norm(KR-K.conj().T,2)))
    if max(source_errors)>1e-9 or max(reverse_errors)>1e-7 or max(minor_errors)>1e-8:raise ArithmeticError('Actualsource/contact/minor/reverse quantumcontract failed')
    rows=[];circular=np.array([0,1,1j])/np.sqrt(2)
    for n in steps:
        raw=np.array(ledgers[n]);paired=(raw+raw.conj().transpose(0,2,1))/2;normalized=paired/root[None,:,None]/root[None,None,:]
        average=normalized.mean(axis=0);eigen=np.linalg.eigvalsh(average)
        cross=np.einsum('i,sij,j->s',np.array([1,0,0]),normalized,circular)
        rows.append(dict(Strang_steps=n,raw_FULLcomplex_source_heat=quantum.summarize(raw),exact_source_Gram_normalized_reversepaired_transfer=quantum.summarize(normalized),transfer_eigenvalues=eigen.tolist(),charged_spin_component_to_physical_circular_A_transfer=quantum.summarize(cross),raw_creator_SOURCE_heat=quantum.summarize(np.asarray(creator_ledgers[n])),no_PSD_or_phase_clipping=True))
    paired_differences=[dict(fine=n,coarse=m,raw_heat_difference=quantum.summarize(np.array(ledgers[n])-np.array(ledgers[m]))) for m,n in zip(steps[:-1],steps[1:])]
    return dict(N=4,V=V,M=M,source_position=[1,1,0],source_counts=dict(active=[2*V,2*V+1],aux_count=63,aux_each=2*V,heavy=2*V,Ref=[4,3],total=130*V+4,Z4_mod4=(130*V+4)%4),beta=str(beta),samples=samples,seed=seed,Strang_steps=steps,a=a,UEphys=1,g_Pi_AND_Coulomb=g,shared_real_A0_modes=V-1,all_REAL_electric_modes=M,source_axes=['chargedmatter_Phi_potentialforce_spinshape','physical_A_y','physical_A_z'],exact_UNFILTERED_SOURCE_Gram_diagonal=Gamma.tolist(),exact_source_Gram_proof='FrefPsiN=0; FrefEpsi=[Fref,E]psi. RawspinGamma=sum|ci|²/(24V) forNONcollapsedN4, pointwiseALLTheta. Gamma5FrefGamma5=Fref andGamma5PGamma5=Q implyTrRx QFref=TrRx Fref/2=0 sinceonsiteFref0, henceALLPhi-force/physicalA SOURCEcrossGrams0 pointwise. ORIGINALsourceGaussianvariance2 andp=.5 giveGamma_AA=4096I2. EXACToriginalpreheatmetric, notsampleGram/postfilterZC/probabilityretag.',exact_free_EM_Gaussian_scalar=free_scalar,common_NEUTRAL_free_CAR_offset=float(free_energy),source_point_Gram_identity_error=max(source_errors),literal_ALL4_complex_bordered_minor_error=max(minor_errors),physical_reversed_word_ALL3_SOURCE_adjoint_error=max(reverse_errors),electric_ONLY_quantum_source_heat=quantum.summarize(np.array(electric)),rows=rows,samepath_paired_Strang_differences=paired_differences,scope='ACTUALN4finiteStrangquantumPi+sharedCoulomb+ALL64CARsourceheat, notfullPhi/heavy/SU4time orcontinuum/Trotter/integralcertificate. Phiobservable isCHARGEDmatterpotentialforceCOMPONENT; heavyPhi/boson/Noe-self-forceNOTincludedintheprobe, NEVERdeclaredzero/removedfromfullH. SourceNoe/fullPhi waveprices areunitary andNOTextendedtoheat. No pole/commoncone/Compton/massgap/vacuum/admission. Imagphase/signkept; reversedphysicalpaths implementoperatoradjoint; no sampled/filtered metric whitening.')


def run(full_path:Path,beta:F,samples:int,steps:list[int],seed:int)->dict[str,Any]:
    started=time.monotonic();raw=full_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=FULL_SHA:raise ArithmeticError('Actualae43/c2e4fullSOURCEepochrequired')
    cap=json.loads(raw);_,c=quantum.checked(cap['input_descriptor']['input_receipts']['contact']);_,t=quantum.checked(c['input_descriptor']['input_receipts']['transfer']);_,g=quantum.checked(t['input_descriptor']['input_receipts']['gram']);_,force=quantum.checked(g['input_descriptor']['input_receipts']['force']);_,joined=quantum.checked(force['input_descriptor']['input_receipts']['join']);_,heat=quantum.checked(joined['input_descriptor']['input_receipts']['heat']);mean=contact.transfer.packets.force.joined.mean_from_source(heat)
    desc=dict(plan=cap['input_descriptor']['plan'],plan_revision=REVISION,scientific_execution_revision=REVISION,input_receipts=dict(full_source=dict(path=str(full_path),sha256=hashlib.sha256(raw).hexdigest())),source_abi=cap['input_descriptor']['source_abi'],beta=str(beta),samples=samples,seed=seed,Strang_steps=steps,native_N=4,source='OriginalUNFILTEREDcanonicalY/all64/AP222/realGaussian2/ALL4Refholes E0 BEFOREquantumheat. FrozenPhiC000nativeimageofactualmacroinputs, NOTnewmean/fullPhi-simulation. NoEafterneutralfilter/postfilterp=.5 claim.',probe='Chargedmatterpotentialforce COMPONENT plusphysicalA_y/z; NOTtotalPhi canonicalforce',shared_Coulomb='Actual63realA0Gaussfields SHAREDall64detflavours, integerRef3phase, g=64; noindependentflavourdecoupling',mean_scope='SamepinnedmacroactualPhiLambda2^8192 meanenclosuremidpoint nativeimage, notLambda1rootrefit')
    result=dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),actual_N4_joined_Phi_component_photon_quantum_heat=actual_kernel(mean,beta,samples,steps,seed),source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--beta',type=F,default=F(1,4096));parser.add_argument('--samples',type=int,default=32);parser.add_argument('--steps',type=int,nargs='+',default=[1,2,4]);parser.add_argument('--seed',type=int,default=20261015)
    args=parser.parse_args();result=run(args.full_source,args.beta,args.samples,args.steps,args.seed)
    if not all(result['source_unchanged_since_import'].values()):raise ArithmeticError('Source changed duringactualquantumcalculation')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    data=result['actual_N4_joined_Phi_component_photon_quantum_heat']
    print(json.dumps(dict(input_sha256=result['input_sha256'],N=data['N'],V=data['V'],M=data['M'],source_Gram=data['exact_UNFILTERED_SOURCE_Gram_diagonal'],source_error=data['source_point_Gram_identity_error'],minor_error=data['literal_ALL4_complex_bordered_minor_error'],reverse_error=data['physical_reversed_word_ALL3_SOURCE_adjoint_error'],rows=data['rows'],elapsed_seconds=result['elapsed_seconds']),indent=2))
