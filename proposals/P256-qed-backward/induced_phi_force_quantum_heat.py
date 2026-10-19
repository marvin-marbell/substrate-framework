"""All45 Phi-force/complex-current SOURCE transfer through primitive real quantum Href."""
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
import induced_phi_quantum_force_source_gram as packets
quantum=packets.quantum
SOURCE_BYTES={**packets.SOURCE_BYTES,Path(__file__).name:OWN_BYTES}
REVISION='1178517cdc4236ce3a4af6a26bed880eb20679df709826463e4db7819ea0d208'
GRAM_SHA='e9ea9de905aab23e9f4c5e29ec2e4cf012aa721c7b6e27aa5ba9fbacade77f1d'


def matrix_jet(M:np.ndarray,L:np.ndarray,R:np.ndarray,left_bra:np.ndarray,right_ket:np.ndarray)->tuple[complex,np.ndarray,np.ndarray,np.ndarray]:
    cof=quantum.OverlapCofactors(M)
    X=cof.left@L@cof.right;Y=cof.left@R@cof.right
    dx=np.diagonal(X,axis1=1,axis2=2);dy=np.diagonal(Y,axis1=1,axis2=2)
    bra=cof.left@left_bra;ket=right_ket@cof.right
    # Contract ONLYdiagZ: never allocate 46²*n² mixedoverlap tensor.
    first_mixed=np.einsum('ikp,jpk,k->ij',bra,ket,cof.one,optimize=True)
    second_mixed=dx@cof.two@dy.T-np.einsum('ikl,jlk,kl->ij',X,Y,cof.two,optimize=True)
    return cof.determinant,cof.phase*(dx@cof.one),cof.phase*(dy@cof.one),cof.phase*(first_mixed+second_mixed)


def combine(a:tuple[complex,np.ndarray,np.ndarray,np.ndarray],b:tuple[complex,np.ndarray,np.ndarray,np.ndarray])->tuple[complex,np.ndarray,np.ndarray,np.ndarray]:
    A,AL,AR,AM=a;B,BL,BR,BM=b
    return A*B**63,AL*B**63+63*A*BL*B**62,AR*B**63+63*A*BR*B**62,AM*B**63+63*(np.outer(AL,BR)+np.outer(BL,AR)+A*BM)*B**62+63*62*A*np.outer(BL,BR)*B**61


def prepared_endpoint(native:dict[str,Any],derivatives:list[list[dict[str,Any]]],theta:np.ndarray,x:int,directions:np.ndarray)->tuple[list[np.ndarray],list[np.ndarray]]:
    V=native['V'];frames=[];observables=[]
    for si,(profile,profiles) in enumerate(zip(native['Pprofiles'],derivatives)):
        state=quantum.graph_state(native,profile,theta);W=state['occupied']
        if si==0:
            f=np.zeros((4*V,4),dtype=complex);f[4*x:4*x+4]=np.eye(4);W=np.column_stack((W,f))
        frames.append(W)
        matrices=[quantum.matter_hamiltonian(native,derivative,theta)/np.sqrt(V) for derivative in profiles]
        J=(quantum.path_derivative(native,profile,theta,directions[0])+1j*quantum.path_derivative(native,profile,theta,directions[1]))/(64*np.sqrt(2))
        observables.append(np.stack([*matrices,J]))
    return frames,observables


def kernel_matrices(Fbra:list[np.ndarray],Fket:list[np.ndarray],Obra:list[np.ndarray],Oket:list[np.ndarray],words:list[np.ndarray],nocc:int,controls:bool=False)->tuple[tuple[complex,np.ndarray,np.ndarray,np.ndarray],tuple[complex,np.ndarray,np.ndarray,np.ndarray],list[float]]:
    jets=[];checks=[]
    for si,(Fb,Fk,Oi,Oj,U) in enumerate(zip(Fbra,Fket,Obra,Oket,words)):
        bra=(Oi@Fb).conj().swapaxes(-1,-2);ket=U@(Oj@Fk);UF=U@Fk
        M=Fb.conj().T@UF;L=bra@UF;R=Fb.conj().T@ket
        selections=[np.arange(nocc)]
        if si==0:selections.extend(np.r_[np.arange(nocc),nocc+b] for b in range(4))
        results=[]
        for indices in selections:
            m=M[np.ix_(indices,indices)];l=L[:,indices][:,:,indices];r=R[:,indices][:,:,indices];left=bra[:,indices];right=ket[:,:,indices]
            jet=matrix_jet(m,l,r,left,right);results.append(jet)
            if controls:
                cof=quantum.OverlapCofactors(m)
                for i,j in ((0,0),(0,45),(45,0),(45,45),(10,35)):
                    scalar=cof.jet(l[i],r[j],left[i]@right[j])
                    checks.append(abs(scalar[3]-jet[3][i,j]))
        jets.append(results)
    neutral=combine(jets[0][0],jets[1][0])
    charged_jets=[combine(a,jets[1][0]) for a in jets[0][1:]]
    charged=(sum(a[0] for a in charged_jets)/4,sum((a[1] for a in charged_jets),np.zeros(46,dtype=complex))/4,sum((a[2] for a in charged_jets),np.zeros(46,dtype=complex))/4,sum((a[3] for a in charged_jets),np.zeros((46,46),dtype=complex))/4)
    return neutral,charged,checks


def run(gram_path:Path,beta:F,samples:int,steps:list[int],seed:int)->dict[str,Any]:
    started=time.monotonic();raw=gram_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=GRAM_SHA:raise ArithmeticError('Actualad509 full45quantumforce sourceGram required')
    gram=json.loads(raw);_,force=quantum.checked(gram['input_descriptor']['input_receipts']['force']);_,join=quantum.checked(force['input_descriptor']['input_receipts']['join']);_,heat=quantum.checked(join['input_descriptor']['input_receipts']['heat'])
    mean=packets.force.joined.mean_from_source(heat);native=quantum.build_native(mean,2);V,M=native['V'],native['M'];g=64.;s=2.;bfloat=float(beta)
    maximum=max(steps)
    if samples<4 or beta<=0 or min(steps)<=0 or any(maximum%n for n in steps):raise ValueError('Physicalpositivebeta/samples/dividingStranglevels required')
    normal=np.asarray(force['actual_geometry']['normal_basis']);shape=np.asarray(force['actual_geometry']['shape_basis']);Rshape=normal.T@shape
    derivatives=[packets.derivative_profiles(profile,normal) for profile in native['Pprofiles']]
    x=int(np.flatnonzero(np.all(native['sites']==(1,1,0),axis=1))[0]);ref=0
    f=np.cos(np.pi*native['sites'][:,0])/np.sqrt(V);directions=np.zeros((2,3*V));directions[0,1::3]=f;directions[1,2::3]=f
    lv,lb=np.linalg.eigh(native['Lplus']);positive=lv>1e-12;A0map=lb[:,positive]*np.sqrt(lv[positive])
    if A0map.shape[1]!=V-1:raise ArithmeticError('PrimitiveGaussrealA0 rank failed')
    free_energy=sum(profile['load']*np.linalg.eigvalsh(quantum.matter_hamiltonian(native,profile,np.zeros(3*V)))[:2*V].sum() for profile in native['Pprofiles'])
    free_scalar=(1+g*bfloat/(2*s))**(-M/2);normalizer=free_scalar*np.exp(bfloat*free_energy)
    rng=np.random.default_rng(seed);fractions=np.arange(2*maximum+1)/(2*maximum)
    ledgers={n:[] for n in steps};bases={n:[] for n in steps};reverse_errors=[];wrong_reverse=[];scalar_errors=[]
    for sample in range(samples):
        middle=rng.normal(0,np.sqrt(s),M);difference=rng.normal(0,np.sqrt(2*g*bfloat/(1+g*bfloat/(2*s))),M)
        increments=rng.normal(0,np.sqrt(g*bfloat/maximum),(2*maximum,M));motion=np.vstack((np.zeros(M),np.cumsum(increments,axis=0)))
        bridge=motion-fractions[:,None]*motion[-1];coordinates=middle+(fractions[:,None]-.5)*difference+bridge;fields=coordinates@native['Q'].T
        alpha_fine=np.sqrt(2*g*bfloat/maximum)*rng.normal(size=(maximum,V-1))@A0map.T;summed_alpha=alpha_fine.sum(axis=0)
        phases=np.array([np.exp(1j*128*summed_alpha.sum()),np.exp(1j*(128*summed_alpha.sum()+summed_alpha[ref]))])
        Fket,Oket=prepared_endpoint(native,derivatives,fields[0],x,directions);Fbra,Obra=prepared_endpoint(native,derivatives,fields[-1],x,directions)
        for n in steps:
            width=maximum//n;dt=bfloat/n;words=[np.eye(4*V,dtype=complex),np.eye(4*V,dtype=complex)];blocks=[]
            for j in range(n):
                midpoint=width*(2*j+1);alpha=alpha_fine[j*width:(j+1)*width].sum(axis=0);D=np.repeat(np.exp(-1j*alpha),4)
                for si,profile in enumerate(native['Pprofiles']):
                    H=quantum.half_heat(native,profile,fields[midpoint],dt);words[si]=((H*D[None,:])@H)@words[si]
                    if sample==0:blocks.append((si,H,alpha))
            N,C,errors=kernel_matrices(Fbra,Fket,Obra,Oket,words,2*V,controls=sample==0);scalar_errors.extend(errors)
            matrices=normalizer*np.stack([N[3]*phases[0],C[3]*phases[1]])
            ledgers[n].append(matrices);bases[n].append(normalizer*np.array([N[0],C[0]])*phases)
            if sample==0:
                reverse=[np.eye(4*V,dtype=complex),np.eye(4*V,dtype=complex)]
                for si,H,alpha in reversed(blocks):
                    D=np.repeat(np.exp(1j*alpha),4);reverse[si]=((H*D[None,:])@H)@reverse[si]
                RN,RC,_=kernel_matrices(Fket,Fbra,Oket,Obra,reverse,2*V)
                reversed_matrices=normalizer*np.stack([RN[3]*phases[0].conjugate(),RC[3]*phases[1].conjugate()])
                reverse_errors.append(float(np.linalg.norm((reversed_matrices-matrices.conj().swapaxes(-1,-2)).reshape(92,46),2)))
                wrong_reverse.append(float(np.linalg.norm((reversed_matrices-matrices.conj()).reshape(92,46),2)))
    rows={}
    for n in steps:
        a=np.asarray(ledgers[n]);m=np.mean(a,axis=0);paired=(m+m.conj().swapaxes(-1,-2))/2
        shape_trace=np.einsum('ai,nsab,bi->ns',Rshape,a[:,:,:45,:45],Rshape,optimize=True)
        rows[str(n)]=dict(actual_RAWneutral_and_charged_force45_complexJA_SOURCE_heat=quantum.summarize(a),sector_order=['neutralSOURCE','rawALL4RefholechargedSOURCE_p_half'],actual_reverse_paired_SOURCE_heat=packets.force.joined.receiver.encode(paired),actual_ALL35shape_SOURCE_heat_trace=quantum.summarize(shape_trace),actual_BASE_SOURCE_heat=quantum.summarize(np.asarray(bases[n])),actual_physical_JAplus_SOURCE_heat=quantum.summarize(a[:,:,-1,-1]),SOURCE_Gram_reference=dict(path=str(gram_path),sha256=GRAM_SHA,exact_integral_definition=gram['source_quantum_Gram_proof'],sample_matrix_certified_lower_bound=False),whitening='NOTperformed: SAMEoriginalfull46SOURCEGram isdefinedbyGaussianintegral; MCad509Gamma neitherexactnorlowerbound. DoNOTsubstitute/referencewhiten/clip.' )
    # Independent prior physical quantum-current producer with SAMEpath inputs.
    prior_errors={}
    if (str(beta),samples,steps,seed)==('1/4096',32,[1,2,4],20261009):
        for n in steps:
            old=heat['actual_real_EM_Coulomb_quantum_kernels']['rows'][str(n)]['ALLcomplex_quantum_source_kernels'];v=np.asarray(old['real'])+1j*np.asarray(old['imag']);base=np.mean(bases[n],axis=0);current=np.mean(np.asarray(ledgers[n])[:,:,-1,-1],axis=0)
            prior_errors[str(n)]=dict(BASE_SOURCE_error=float(np.max(np.abs(base-v[:2]))),independent_original_physical_JA_SOURCE_current_error=float(np.max(np.abs(current-v[2:]/64**2))))
    scale=max(float(np.linalg.norm(np.mean(ledgers[n],axis=0).reshape(92,46),2)) for n in steps)
    if max(reverse_errors)>1e-9*scale or max(scalar_errors)>1e-9*scale or not all(np.all(np.isfinite(ledgers[n])) for n in steps):raise ArithmeticError('Physicalall45 quantumtransfer/cofactor/adjoint failed')
    descriptor=dict(plan=gram['input_descriptor']['plan'],plan_revision=REVISION,scientific_execution_revision=REVISION,input_receipts=dict(gram=dict(path=str(gram_path),sha256=hashlib.sha256(raw).hexdigest())),source_abi=gram['input_descriptor']['source_abi'],native_N=2,beta=str(beta),samples=samples,steps=steps,seed=seed,a='1/64',UEphys='1',g_electric_AND_Coulomb='64',real_EM_modes=M,charged_flavours=64,packets='FULL45q0physicalPhi-CARforce +physicalcomplexJAplus ORIGINALSOURCEpackets, Ref4→3/all4coherent, ACTIVE16→17, AUX63×16; no filter/gate/lateSlater. SAMEneutralfreeCARoffset',clock4096_UNCHANGED=True,original_SOURCE_Gram_semantics='ExactFULLGaussian46packetGamma definesbothcomparands; numericalad509estimateNOTwhitening/lowerbound')
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),actual_all45_Phi_force_and_complex_current_quantum_EM_Coulomb_transfer=rows,actual_reversed_physical_path_full46_adjoint_error=max(reverse_errors),WRONG_entrywise_conjugate_without_probe_exchange_error=max(wrong_reverse),actual_vector_cofactor_vs_scalar_selected_complex_brackets_error=max(scalar_errors),actual_independent_frozen569a_current_and_base_comparison=prior_errors,finite_Strang_law=heat['exact_finite_Strang_quantum_kernel'],all45_transfer_proof='PropagateFULL32modeonebody U throughactualCARhalf/SHAREDrealA0Coulomb/REALall17electricBrownian words; sourceendpoints F_W andrawfALL4, each46observable O. ForM=FbdagUFk,L_i=FbdagOi_bra†UFk,R_j=FbdagUOj_ketFk,S_ij=FbdagOi_bra†UOj_ketFk, SVD compoundexcluded1/2 giveFULLmatrixmixedjets. Contractdiag(U_svd† S_ij V_svd) directly fromtransformedbra/ket, no46²*n²allocation, no inverse/detdivision/singularityclip. Active a andauxb63 totalmatrixproductrule aijb63+63(ai bj+bi aj+a bij)b62+63*62a bi bj b61; ALL64crossmeans retained. ChargedALL4rank17literalminorjets summed/4 withSAMEaux andRef3phase, p unchanged. Matrixreverse=adjoint withPROBEexchange, notentrywiseconjugate; actualreversephysicalwords/endpoints/A0sign controlswhole46matrices. ScalarcofactorcontrolsincludecomplexJA/forceorderedcross. These areinteractingfullHref quantum SOURCEheatmatrices, NOTinteractingSlateransatz.',scope='Actualall45/35shape/complexcurrentSOURCEpackets throughnativeprimitivePi+Coulomb+all64CAR finiteStrang, finiteMC. SourceforceΦq0 observables areinsertions, NOTPhi coordinatekinetic/heavy/SU4quantumtime evolution. No fullPhi analyticcontinuation of4096bounds, no continuumconfidence/pole/commoncone/Compton/ground/sourcepromotion. No sampled-sourceGramnormalization, sourceprobabilities/clocks/alloldbytes fixed.',source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gram',type=Path,required=True)
    parser.add_argument('--beta',type=F,required=True)
    parser.add_argument('--samples',type=int,required=True)
    parser.add_argument('--steps',type=int,nargs='+',required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run(args.gram,args.beta,args.samples,args.steps,args.seed)
    if not all(result['source_unchanged_since_import'].values()):raise ArithmeticError('Source changed during actual calculation')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(input_sha256=result['input_sha256'],actual_shape_rows={n:r['actual_ALL35shape_SOURCE_heat_trace'] for n,r in result['actual_all45_Phi_force_and_complex_current_quantum_EM_Coulomb_transfer'].items()},actual_reversed_physical_path_full46_adjoint_error=result['actual_reversed_physical_path_full46_adjoint_error'],WRONG_entrywise_conjugate_without_probe_exchange_error=result['WRONG_entrywise_conjugate_without_probe_exchange_error'],actual_independent_frozen569a_current_and_base_comparison=result['actual_independent_frozen569a_current_and_base_comparison'],elapsed_seconds=result['elapsed_seconds']),indent=2))
