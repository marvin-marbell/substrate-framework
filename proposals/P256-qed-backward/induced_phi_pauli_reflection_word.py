"""Actual Pin-z/AP222 reflected Pauli/EM/shared-Coulomb source word."""
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


def physical(mean:dict[str,Any])->dict[str,Any]:
    native=quantum.build_native(mean,4);V,M=native['V'],native['M'];beta=F(1,2048);b=float(beta);n=8;g=float(native['UE']/native['a']);a=float(native['a']);s=2.;mode=pauli.magnetic_shape(native)
    lookup={tuple(site):i for i,site in enumerate(native['sites'])};perm=np.array([lookup[(int(x),int(y),int(-z)%4)] for x,y,z in native['sites']])
    gamma=native['frame']/np.sqrt([native['g0']]+[native['gs']]*3)[:,None,None];gamma5=gamma[0]@gamma[1]@gamma[2]@gamma[3];pin=1j*gamma[3]@gamma5
    U=np.zeros((4*V,4*V),complex);Re=np.zeros((3*V,3*V));Ps=np.zeros((V,V))
    for x,site in enumerate(native['sites']):
        r=int(perm[x]);Ps[r,x]=1;U[4*r:4*r+4,4*x:4*x+4]=(1 if site[2]==0 else -1)*pin
        for axis in range(3):
            dest=r
            if axis==2:dest=lookup[(int(site[0]),int(site[1]),int(-site[2]-1)%4)]
            Re[3*dest+axis,3*x+axis]=-1 if axis==2 else 1
    f=np.sqrt(2/V)*np.cos(2*np.pi*native['sites'][:,0]/4);directions=np.zeros((2,3*V));directions[0,1::3]=f;directions[1,2::3]=f
    x=int(np.flatnonzero(np.all(native['sites']==(1,1,0),axis=1))[0]);ref=0;sign=np.diag([1.,1.,-1.]);rng=np.random.default_rng(20261019)
    fractions=np.arange(2*n+1)/(2*n);middle=rng.normal(0,np.sqrt(s),M);difference=rng.normal(0,np.sqrt(2*g*b/(1+g*b/(2*s))),M)
    motion=np.vstack((np.zeros(M),np.cumsum(rng.normal(0,np.sqrt(g*b/n),(2*n,M)),axis=0)))
    fields=(middle+(fractions[:,None]-.5)*difference+motion-fractions[:,None]*motion[-1])@native['Q'].T
    lv,lb=np.linalg.eigh(native['Lplus']);A0map=lb[:,lv>1e-12]*np.sqrt(lv[lv>1e-12]);alphas=np.sqrt(2*g*b/n)*rng.normal(size=(n,V-1))@A0map.T
    projector=native['Q']@native['Q'].T;kinematic=dict(real_edge_isometry=float(np.linalg.norm(Re.T@Re-np.eye(3*V))),coexact_projector_equivariance=float(np.linalg.norm(Re@projector@Re.T-projector)),Coulomb_Lplus_equivariance=float(np.linalg.norm(Ps@native['Lplus']@Ps.T-native['Lplus'])),physical_AP222_Pin_unitarity=float(np.linalg.norm(U.conj().T@U-np.eye(4*V))),physical_photon_yz_parity=float(np.linalg.norm(directions@Re-np.diag([1.,-1.])@directions)))
    errors=[];results=[];Efree=sum(p['load']*np.linalg.eigvalsh(quantum.matter_hamiltonian(native,p,np.zeros(3*V)))[:2*V].sum() for p in native['Pprofiles']);normalizer=(1+g*b/(2*s))**(-M/2)*np.exp(b*Efree)
    base_words=[]
    for reflected in (False,True):
        theta=fields@Re.T if reflected else fields;hs=alphas@Ps.T if reflected else alphas
        _,Wk,Fk=pauli.endpoints(native,theta[0],mode);_,Wb,Fb=pauli.endpoints(native,theta[-1],mode);words=[np.eye(4*V,dtype=complex),np.eye(4*V,dtype=complex)]
        for j in range(n):
            D=np.repeat(np.exp(-1j*hs[j]),4)
            for si,p in enumerate(native['Pprofiles']):
                H=quantum.half_heat(native,p,theta[2*j+1],b/n);words[si]=((H*D[None,:])@H)@words[si]
                if reflected:
                    original=quantum.matter_hamiltonian(native,p,fields[2*j+1]);mirrored=quantum.matter_hamiltonian(native,p,theta[2*j+1]);errors.append(float(np.linalg.norm(mirrored-U@original@U.conj().T,2)))
        if not reflected:base_words=words
        else:
            errors.extend(float(np.linalg.norm(w-U@original@U.conj().T,2)) for w,original in zip(words,base_words))
            for p,Factual in zip(native['Pprofiles'],Fk):errors.append(float(np.linalg.norm(Factual-U@pauli.operator(native,p,fields[0],mode)@U.conj().T,2)))
        total=hs.sum(axis=0);phase=np.exp(1j*(128*total.sum()+total[ref]));core=normalizer*phase*pauli.source_kernel(native,Wb,Wk,Fb,Fk,words,x)
        results.append(pauli.joined.lift(core,directions@theta[-1],directions@theta[0],a))
    mismatch=float(np.linalg.norm(results[1]-sign@results[0]@sign,2));paired=(results[0]+results[1])/2
    if max(kinematic.values())>1e-10 or max(errors)>1e-9 or mismatch>1e-6:raise ArithmeticError('Actual Pin/AP/shared-Coulomb/sourceword reflection contract failed')
    def matrix(A:np.ndarray)->dict[str,Any]:return dict(real=np.real(A).tolist(),imag=np.imag(A).tolist())
    return dict(N=4,V=V,M=M,beta=str(beta),Strang_steps=n,seed=20261019,ALLcharged_flavours=64,shared_REAL_A0_modes=V-1,g_Pi_AND_Coulomb=g,source_position=[1,1,0],kinematic_errors=kinematic,actual_physical_H_force_and_quantum_word_equivariance_error=max(errors),actual_ALL4hole_ALL64_joined_raw_kernel_reflection_error=mismatch,actual_original_RAW_3source_word=matrix(results[0]),actual_reflected_RAW_3source_word=matrix(results[1]),actual_antithetic_average_RAW_3source_word=matrix(paired),exact_ensemble_selection_rule='For fixed sourcex110 and Ref0 (both z=0), Pin-z/AP222 reflection preserves actual original Gaussian2/Pi/sharedA0/CAR/source marginals. SourcePhiPauli and Ay even, Az odd. Hence ensemble Pauli/Az and Ay/Az heat entries EXACTzero; Pauli/Ay is parity-even and NOT forbidden. Antithetic averaging is an exact measure-preserving estimator, NOT filtering/gating/zeroing a physical channel.',scope='Native frozenPhi reference ONLY, fullheavy/Phi/SU4 time not propagated. Observable chargedmatterPhi-gradient component, not fullcanonicalPhi force. No magneticbridge/pole/commoncone/vacuum/retarded claim.')


def run(path:Path)->dict[str,Any]:
    started=time.monotonic();raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=METRIC_SHA:raise ArithmeticError('Frozen06db/2586 sourceMetric epoch required')
    cap=json.loads(raw);_,record=quantum.checked(cap['input_descriptor']['input_receipts']['magnetic']);_,record=quantum.checked(record['input_descriptor']['input_receipts']['prior_joined_heat'])
    for key in ('full_source','contact','transfer','gram','force','join','heat'):_,record=quantum.checked(record['input_descriptor']['input_receipts'][key])
    mean=pauli.contact.transfer.packets.force.joined.mean_from_source(record)
    desc=dict(plan=cap['input_descriptor']['plan'],plan_revision=pauli.REVISION,scientific_execution_revision=pauli.REVISION,input_receipts=dict(metric=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())),source_abi=cap['input_descriptor']['source_abi'],native_N=4,beta='1/2048',Strang_steps=8,seed=20261019,source='UNCHANGED originalall64/canonicalY/Gaussian2/AP222/ALL4Refholes/E_BEFOREactualheat; actualone/twolinkWilson H, actualPi+sharedA0HS. Mirror sourcex110 and Ref0 arefixed. AP222 sitephase s(z)=+1 forz0,-1otherwise retained.',a='1/64',UEphys='1',g_Pi_AND_Coulomb='64',Lambda='2^8192',chi='2^-4096',original_clock='4096')
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),actual_physical_Pauli_source_reflection_word=physical(mean),source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--metric',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run(a.metric)
    if not all(r['source_unchanged_since_import'].values()):raise ArithmeticError('Source changed during physical reflection calculation')
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(input_sha256=r['input_sha256'],physical=r['actual_physical_Pauli_source_reflection_word'],elapsed_seconds=r['elapsed_seconds']),indent=2))
