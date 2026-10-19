"""Correlated actual Pauli/Pi/shared-Coulomb quantum beta continuation."""
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
import induced_phi_pauli_source_metric as metric
pauli=metric.pauli
quantum=pauli.quantum
SOURCE_BYTES={**metric.SOURCE_BYTES,Path(__file__).name:OWN_BYTES}
METRIC_SHA='2586ec47ddeb662c414f6dda563f3216d5ed6b7681ba30ebdef1a917837a803e'


def ratio_summary(numerator:np.ndarray,denominator:np.ndarray)->dict[str,Any]:
    mn=np.mean(numerator);md=np.mean(denominator)
    if md==0:raise ArithmeticError('Conditional heat ratio has exactzero sampled denominator')
    ratio=mn/md;residual=(numerator-ratio*denominator)/md
    return dict(mean_real=float(ratio.real),mean_imag=float(ratio.imag),linearized_standard_error_real=float(residual.real.std(ddof=1)/np.sqrt(len(residual))),linearized_standard_error_imag=float(residual.imag.std(ddof=1)/np.sqrt(len(residual))),definition='ratio of raw unfiltered-source quantum heat matrix element to raw creator heat; conditional postheat observable ONLY, NOTsourceGram whitening or p=1/2 replacement.')


def physical(mean:dict[str,Any],betas:list[F],samples:int,steps:list[int],seed:int)->dict[str,Any]:
    native=quantum.build_native(mean,4);V,M=native['V'],native['M'];a=float(native['a']);g=float(native['UE']/native['a']);s=2.;mode=pauli.magnetic_shape(native)
    x=int(np.flatnonzero(np.all(native['sites']==(1,1,0),axis=1))[0]);ref=0;f=np.sqrt(2/V)*np.cos(2*np.pi*native['sites'][:,0]/4)
    directions=np.zeros((2,3*V));directions[0,1::3]=f;directions[1,2::3]=f
    maximum=max(steps);fractions=np.arange(2*maximum+1)/(2*maximum)
    if samples<4 or min(betas)<=0 or min(steps)<=0 or any(maximum%n for n in steps):raise ValueError('Positive paired quantum beta/Strang sampling required')
    lv,lb=np.linalg.eigh(native['Lplus']);A0map=lb[:,lv>1e-12]*np.sqrt(lv[lv>1e-12])
    if A0map.shape[1]!=V-1:raise ArithmeticError('Actual shared Coulomb rank changed')
    Efree=sum(p['load']*np.linalg.eigvalsh(quantum.matter_hamiltonian(native,p,np.zeros(3*V)))[:2*V].sum() for p in native['Pprofiles'])
    rng=np.random.default_rng(seed);ledgers={(b,n):[] for b in betas for n in steps};creator={(b,n):[] for b in betas for n in steps};source=[]
    identities=[np.eye(4*V,dtype=complex),np.eye(4*V,dtype=complex)]
    for sample in range(samples):
        middle=rng.normal(0,np.sqrt(s),M);zdiff=rng.normal(size=M);zmotion=rng.normal(size=(2*maximum,M));zA0=rng.normal(size=(maximum,V-1))
        _,Wm,Fm=pauli.endpoints(native,native['Q']@middle,mode);tm=directions@(native['Q']@middle)
        source.append(pauli.joined.lift(pauli.source_kernel(native,Wm,Wm,Fm,Fm,identities,x),tm,tm,a))
        for beta in betas:
            b=float(beta);difference=np.sqrt(2*g*b/(1+g*b/(2*s)))*zdiff
            motion=np.vstack((np.zeros(M),np.cumsum(np.sqrt(g*b/maximum)*zmotion,axis=0)))
            coordinates=middle+(fractions[:,None]-.5)*difference+motion-fractions[:,None]*motion[-1];fields=coordinates@native['Q'].T
            fine_A0=np.sqrt(2*g*b/maximum)*zA0@A0map.T
            _,Wk,Fk=pauli.endpoints(native,fields[0],mode);_,Wb,Fb=pauli.endpoints(native,fields[-1],mode);tb=directions@fields[-1];tk=directions@fields[0]
            alphas=fine_A0.sum(axis=0);phase=np.exp(1j*(128*alphas.sum()+alphas[ref]));normalizer=(1+g*b/(2*s))**(-M/2)*np.exp(b*Efree)
            for n in steps:
                width=maximum//n;dt=b/n;words=[np.eye(4*V,dtype=complex),np.eye(4*V,dtype=complex)]
                for j in range(n):
                    alpha=fine_A0[j*width:(j+1)*width].sum(axis=0);D=np.repeat(np.exp(-1j*alpha),4)
                    for si,p in enumerate(native['Pprofiles']):
                        H=quantum.half_heat(native,p,fields[width*(2*j+1)],dt);words[si]=((H*D[None,:])@H)@words[si]
                core=normalizer*phase*pauli.source_kernel(native,Wb,Wk,Fb,Fk,words,x);raw=pauli.joined.lift(core,tb,tk,a)
                ledgers[beta,n].append((raw+raw.conj().T)/2);creator[beta,n].append(core[1,1])
    rows=[];arrays={k:np.array(v) for k,v in ledgers.items()};Z={k:np.array(v) for k,v in creator.items()}
    for beta in betas:
        for n in steps:
            K=arrays[beta,n];z=Z[beta,n]
            rows.append(dict(beta=str(beta),Strang_steps=n,actual_reversepaired_RAW_ALL3_quantum_heat=quantum.summarize(K),RAW_creator_quantum_heat=quantum.summarize(z),RAW_Pauli_to_Ay=quantum.summarize(K[:,0,1]),RAW_Pauli_to_Az=quantum.summarize(K[:,0,2]),conditional_Pauli_to_Ay_over_creator_heat=ratio_summary(K[:,0,1],z),conditional_Pauli_to_Az_over_creator_heat=ratio_summary(K[:,0,2],z)))
    differences=[]
    for n in steps:
        for left,right in zip(betas[:-1],betas[1:]):differences.append(dict(beta_left=str(left),beta_right=str(right),Strang_steps=n,definition='RIGHTminusLEFT; actual paired commonGaussian beta samples; not a speed/mass/parameter fit',RAW_Pauli_to_Ay_difference=quantum.summarize(arrays[right,n][:,0,1]-arrays[left,n][:,0,1]),RAW_creator_difference=quantum.summarize(Z[right,n]-Z[left,n])))
    refinement=[]
    for beta in betas:
        for lo,hi in zip(steps[:-1],steps[1:]):refinement.append(dict(beta=str(beta),steps_low=lo,steps_high=hi,RAW_Pauli_to_Ay_high_minus_low=quantum.summarize(arrays[beta,hi][:,0,1]-arrays[beta,lo][:,0,1]),RAW_ALL3_high_minus_low=quantum.summarize(arrays[beta,hi]-arrays[beta,lo])))
    return dict(N=4,V=V,M=M,samples=samples,seed=seed,betas=[str(b) for b in betas],Strang_steps=steps,real_EM_modes=M,shared_REAL_A0_modes=V-1,ALLcharged_flavours=64,g_Pi_AND_Coulomb=g,source_position=[1,1,0],RAW_SOURCE_Gram_sampling_NOT_whitening=quantum.summarize(np.array(source)),EXACT_SOURCE_photon_Gram='4096 I2',source_probability_before_heat='1/2',conditional_after_heat_probability_NOT_half=True,rows=rows,correlated_beta_differences=differences,correlated_Strang_refinement=refinement,scope='Native frozenPhi reference actual Pi+sharedCoulomb+ALL64CAR, primitive Wilson one/twolinks. Phi/SU4/heavy time NOT numerically propagated; not removed from fullH. Pauli observable chargedmatter Phi-gradient component ONLY. No pole/commoncone/vacuum/retarded/QED promotion, no source filter/gate.')


def run(path:Path,betas:list[F],samples:int,steps:list[int],seed:int)->dict[str,Any]:
    started=time.monotonic();raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=METRIC_SHA:raise ArithmeticError('Frozen06db/2586 sourceMetric epoch required')
    cap=json.loads(raw);_,record=quantum.checked(cap['input_descriptor']['input_receipts']['magnetic']);_,record=quantum.checked(record['input_descriptor']['input_receipts']['prior_joined_heat'])
    for key in ('full_source','contact','transfer','gram','force','join','heat'):_,record=quantum.checked(record['input_descriptor']['input_receipts'][key])
    mean=pauli.contact.transfer.packets.force.joined.mean_from_source(record)
    desc=dict(plan=cap['input_descriptor']['plan'],plan_revision=pauli.REVISION,scientific_execution_revision=pauli.REVISION,input_receipts=dict(metric=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())),source_abi=cap['input_descriptor']['source_abi'],source='UNCHANGEDoriginalcanonicalY/Gaussian2/all64/AP222/ALL4Refholes/E_BEFORE_actual_heat. SAMEoriginalmacroPhiLambda2^8192meanmidpointnative; NOTnewmean/filter/probability.',actual_law_and_spacing=dict(a='1/64',UEphys='1',g_Pi_AND_Coulomb='64',active_mass='1/5',aux_mass='2',charged_count=64,neutralheavy_mass='1/5',neutralheavy_EMcharge='0',Lambda='2^8192',chi='2^-4096',original_clock='4096'),native_N=4,betas=[str(b) for b in betas],samples=samples,Strang_steps=steps,seed=seed,common_random_beta_coupling='Same independent standardnormal draws rescaled by exact physical electric and sharedA0 bridge laws for everybeta, preserving EACHoriginal marginal. No state/kernel conditioning gate.')
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),actual_correlated_Pauli_quantum_beta_continuation=physical(mean,betas,samples,steps,seed),source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--metric',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--betas',type=F,nargs='+',required=True);p.add_argument('--samples',type=int,required=True);p.add_argument('--steps',type=int,nargs='+',required=True);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();r=run(a.metric,a.betas,a.samples,a.steps,a.seed)
    if not all(r['source_unchanged_since_import'].values()):raise ArithmeticError('Source changed during physical beta calculation')
    a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(dict(input_sha256=r['input_sha256'],physical=r['actual_correlated_Pauli_quantum_beta_continuation'],elapsed_seconds=r['elapsed_seconds']),indent=2))
