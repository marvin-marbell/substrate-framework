"""Original-source all45 q0 Phi-force quantum noise, joined physical complex current."""
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
import induced_phi_quantum_source_force as force
quantum=force.quantum
SOURCE_BYTES={**force.SOURCE_BYTES,Path(__file__).name:OWN_BYTES}
REVISION='1178517cdc4236ce3a4af6a26bed880eb20679df709826463e4db7819ea0d208'
FORCE_SHA='e0ad44ae121730dcb123250de9e60b7bdeff44538c6778f06fef876b72cea689'


def compound_gram(blocks:np.ndarray)->np.ndarray:
    return np.einsum('ixy,jxy->ij',blocks.conj(),blocks)


def derivative_profiles(profile:dict[str,Any],normal:np.ndarray)->list[dict[str,Any]]:
    onsite,_=force.scalar_paths(profile['model']);directions=[]
    for column in normal.T:
        dframe=np.einsum('aA,Aij->aij',column.reshape(4,15),force.T)
        directions.append(dict(profile,onsite=onsite*dframe[0],paths=[dict(path,hop=path['seam']*np.einsum('a,aij->ij',path['scalars'],dframe)) for path in profile['paths']]))
    return directions


def source_observable_jet(state:dict[str,Any],observables:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    W=state['occupied'];applied=observables@W
    ph=state['empty'].conj().T@applied
    mean=np.einsum('xh,ixh->i',W.conj(),applied)
    return mean,compound_gram(ph),applied


def literal_all4_rank17_gram(state:dict[str,Any],observables:np.ndarray,mu:np.ndarray,total_cov:np.ndarray,active_cov:np.ndarray,x:int)->np.ndarray:
    P,Q,W=state['P'],np.eye(len(state['P']))-state['P'],state['occupied']
    total=np.zeros((46,46),dtype=complex)
    for b in range(4):
        q=Q[:,4*x+b];norm=float(np.real(np.vdot(q,q)))
        if norm<=0:raise ArithmeticError('Physicalpositive sourcehole norm failed')
        Wb=np.column_stack((W,q/np.sqrt(norm)));Pb=Wb@Wb.conj().T
        applied=observables@Wb;ph=(np.eye(len(P))-Pb)@applied
        added_mean=np.einsum('x,ix->i',q.conj(),observables@q)/norm
        mean=mu+added_mean
        total+=(norm/2)*(total_cov-active_cov+compound_gram(ph)+np.outer(mean.conj(),mean))
    return total


def run(force_path:Path,samples:int,seed:int)->dict[str,Any]:
    started=time.monotonic();raw=force_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=FORCE_SHA:raise ArithmeticError('Actualfd07 full45force epoch required')
    receipt=json.loads(raw);_,join=quantum.checked(receipt['input_descriptor']['input_receipts']['join']);_,heat=quantum.checked(join['input_descriptor']['input_receipts']['heat'])
    mean=force.joined.mean_from_source(heat);native=quantum.build_native(mean,2);V,M=native['V'],native['M']
    normal=np.asarray(receipt['actual_geometry']['normal_basis']);shape=np.asarray(receipt['actual_geometry']['shape_basis']);shape_normal=normal.T@shape
    derivatives=[derivative_profiles(profile,normal) for profile in native['Pprofiles']]
    directions=[]
    for axis in (1,2):
        d=np.zeros(3*V);d[axis::3]=np.cos(np.pi*native['sites'][:,0])/np.sqrt(V);directions.append(d)
    x=int(np.flatnonzero(np.all(native['sites']==(1,1,0),axis=1))[0]);rng=np.random.default_rng(seed)
    neutral=[];charged=[];means_N=[];means_C=[];shape_means=[];shape_noise=[];wrong_means=[];mixture_errors=[];force_controls=[];minimum_perfield=[]
    for sample in range(samples):
        theta=native['Q']@(np.sqrt(2)*rng.normal(size=M))
        mu=np.zeros(46,dtype=complex);cov=np.zeros((46,46),dtype=complex);wrong=np.zeros((46,46),dtype=complex)
        states=[];operators=[];active_cov=np.zeros((46,46),dtype=complex)
        for si,(profile,profiles) in enumerate(zip(native['Pprofiles'],derivatives)):
            state=quantum.graph_state(native,profile,theta)
            matrices=[quantum.matter_hamiltonian(native,derivative,theta)/np.sqrt(V) for derivative in profiles]
            Jplus=(quantum.path_derivative(native,profile,theta,directions[0])+1j*quantum.path_derivative(native,profile,theta,directions[1]))/(64*np.sqrt(2))
            obs=np.stack([*matrices,Jplus]);local_mu,local_cov,_=source_observable_jet(state,obs)
            mu+=profile['load']*local_mu;cov+=profile['load']*local_cov
            wrong+=profile['load']*np.outer(local_mu.conj(),local_mu)
            if si==0:active_cov=local_cov
            if sample==0:force_controls.append(float(np.max(np.abs(local_mu[:45]-normal.T@force.force_cart(native,profile,state['P'],theta)/np.sqrt(V)))))
            states.append(state);operators.append(obs)
        GN=cov+np.outer(mu.conj(),mu)
        P=states[0]['P'];Q=np.eye(4*V)-P;B=Q[:,4*x:4*x+4]/np.sqrt(2);obs=operators[0]
        added_applied=obs@B
        occupied_added=states[0]['occupied'].conj().T@added_applied
        added_occupied=B.conj().T@(obs@states[0]['occupied'])
        addition=compound_gram(added_applied)-compound_gram(occupied_added)-compound_gram(added_occupied)
        added_mean=np.einsum('xb,ixb->i',B.conj(),added_applied)
        GC=GN+addition+np.outer(mu.conj(),added_mean)+np.outer(added_mean.conj(),mu)
        muC=mu+added_mean
        if sample==0:mixture_errors.append(float(np.linalg.norm(GC-literal_all4_rank17_gram(states[0],obs,mu,cov,active_cov,x),2)))
        neutral.append(GN);charged.append(GC);means_N.append(mu);means_C.append(muC)
        shape_means.append(shape_normal.T@mu[:45]);shape_noise.append(shape_normal.T@GN[:45,:45]@shape_normal)
        wrong_means.append(GN-(cov+wrong))
        minimum_perfield.append(min(float(np.linalg.eigvalsh((GN+GN.conj().T)/2)[0]),float(np.linalg.eigvalsh((GC+GC.conj().T)/2)[0])))
    N,C=np.asarray(neutral),np.asarray(charged);muN=np.mean(means_N,axis=0);muC=np.mean(means_C,axis=0)
    GN,GC=np.mean(N,axis=0),np.mean(C,axis=0)
    covN=GN-np.outer(muN.conj(),muN);covC=GC-np.outer(muC.conj(),muC)
    mean_shape=np.mean(shape_means,axis=0);noise_shape=np.mean(shape_noise,axis=0)-np.outer(mean_shape.conj(),mean_shape)
    normscale=max(float(np.linalg.norm(GN,2)),float(np.linalg.norm(GC,2)))
    if max(force_controls)>1e-8 or max(mixture_errors)>1e-9*normscale or min(minimum_perfield)<-1e-10*normscale:
        raise ArithmeticError('Actualforce/current quantumGram orRefhole mixture failed')
    descriptor=dict(plan=receipt['input_descriptor']['plan'],plan_revision=REVISION,scientific_execution_revision=REVISION,input_receipts=dict(force=dict(path=str(force_path),sha256=hashlib.sha256(raw).hexdigest())),source_abi=receipt['input_descriptor']['source_abi'],native_N=2,samples=samples,seed=seed,real_EM_modes=M,variance='2',a='1/64',UEphys='1',g_electric_AND_Coulomb='64',source='ORIGINALunfilteredGaussian2/canonicalHermY000AP222, Ref4; coherentALL4rawRefhole E/2 withnormalizedchargedconditionalDeltaP=Q_SOURCE Rx Q_SOURCE/2; no filter/pole',observables='FULL45uniformNoe-normal physicalq0 forceoperators dHCAR/dPhi_normal /sqrtV andphysicalJ_Aplus=a(JTheta_y+iJTheta_z)/sqrt2; allcomplexGram/crossblocks',clock4096_UNCHANGED=True)
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),actual_ORIGINAL_neutral_quantum_FORCE45_and_complex_JA_SOURCE_Gram=quantum.summarize(N),actual_ORIGINAL_normalizedcharged_quantum_FORCE45_and_complex_JA_SOURCE_Gram=quantum.summarize(C),actual_ORIGINAL_RAWcharged_quantum_SOURCE_Gram=quantum.summarize(C/2),actual_neutral_total_quantum_plus_field_covariance=force.joined.receiver.encode(covN),actual_normalizedcharged_total_quantum_plus_field_covariance=force.joined.receiver.encode(covC),actual_global_mean_neutral=force.joined.receiver.encode(muN),actual_global_mean_normalizedcharged=force.joined.receiver.encode(muC),actual_all35shape_quantum_noise_covariance=force.joined.receiver.encode(noise_shape),actual_all35shape_mean_norm=float(np.linalg.norm(mean_shape)),actual_all35shape_noise_trace=float(np.real(np.trace(noise_shape))),actual_all35shape_noise_eigenvalues=np.linalg.eigvalsh((noise_shape+noise_shape.conj().T)/2).tolist(),actual_full46_neutral_Gram_eigenvalues=np.linalg.eigvalsh((GN+GN.conj().T)/2).tolist(),actual_full46_charged_Gram_eigenvalues=np.linalg.eigvalsh((GC+GC.conj().T)/2).tolist(),actual_perfield_Gram_minimum_eigenvalue=min(minimum_perfield),actual_literal_all4_rank17_mixture_Gram_error=max(mixture_errors),actual_FORCE45_means_vs_literal_Cartesian_force_error=max(force_controls),WRONG_independent_flavour_means_instead_ALL64_total_Gram_difference_norm=float(np.linalg.norm(np.mean(wrong_means,axis=0),2)),WRONG_zero35shape_based_on_tiny_mean_loses_noise_trace=float(np.real(np.trace(noise_shape))),PSD_clipping=False,receiver_whitening='NOTperformed: originalfull46GaussianintegralGamma isdefinedphysically butMCmatrixisNOTcertifiedlowerbound/exactGram; no sampled-sourceGramfit substituted',source_quantum_Gram_proof='INITIALconditionalSlaters ONLY. ForALL64 totalobs O_i, mu_i=sumload TrP O_i BEFOREouter; GN_ij=sumload TrP O_i† Q O_j+conj(mu_i)mu_j. Keepcomplexorderedforce-currentcrossblocks. ChargedRef3 normalizedALL4holemixedsource hasDeltaP=B Bdag=Q_SOURCE Rx Q_SOURCE/2,Tr1. Rank1normalordered CAR selfterm cancelsaddedmean² BEFOREmixing: GC-GN=TrDeltaP O_i† O_j-TrDeltaP O_i† P O_j-TrP O_i† DeltaP O_j+conj(mu_i)t_j+conj(t_i)mu_j, t=TrDeltaP O. Compute viaexactfiniteGramsofO B,Wdag O B,Bdag O W; no inverse/phaseclip/lateinteractingSlaterassumption. Firstactualsampleindependentliteral4rank17Slatermixture verifieswhole46complexGram. Sourcefieldexpectation averagesFULLGram includingTOTALconditionalmeanouter, then subtractGLOBALmeanouter onlyforcovariance; preservesclassicalfieldmeanfluctuations andquantumPHnoise. RawchargedsourceGram=.5GC, originalpunchanged.',scope='ACTUALnative originalsourcequantumforce/noise/currentGram, notquantumPhi-time evolution/fullPhiGaussheavyforceintegral/newmean/minimum/pole/gap/commoncone/Compton orprovedsourceGramlowerbound. Full45normalgeometry andALL35shape retained evenwhenshapeMEANzero. Source andallfrozenreceipts/clocks/probabilities unchanged; mean/photontransferpreviouscaps neitherretagged norfit.',source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force',type=Path,required=True)
    parser.add_argument('--samples',type=int,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run(args.force,args.samples,args.seed)
    if not all(result['source_unchanged_since_import'].values()):raise ArithmeticError('Source changed during actual calculation')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('input_sha256','actual_all35shape_mean_norm','actual_all35shape_noise_trace','actual_literal_all4_rank17_mixture_Gram_error','WRONG_independent_flavour_means_instead_ALL64_total_Gram_difference_norm','elapsed_seconds')},indent=2))
