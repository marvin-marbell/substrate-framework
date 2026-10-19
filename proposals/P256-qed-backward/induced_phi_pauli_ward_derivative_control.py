"""Independent primitive quantum-word differential for finite bridge Ward."""
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
import induced_phi_pauli_finite_beta_ward as finite
pauli=finite.pauli
quantum=finite.quantum
SOURCE_BYTES={**finite.SOURCE_BYTES,Path(__file__).name:OWN_BYTES}
FINITE_SOURCE_SHA='4c6e494f82a80afb5514219cbce5a44f817d9b1baf315b187ec303c39c73db83'


def independent_word(native:dict[str,Any],fields:np.ndarray,alphas:np.ndarray,mode:dict[str,Any],x:int,beta:float)->complex:
    _,Wk,Fk=pauli.endpoints(native,fields[0],mode);_,Wb,Fb=pauli.endpoints(native,fields[-1],mode);n=len(alphas);words=[np.eye(4*native['V'],dtype=complex),np.eye(4*native['V'],dtype=complex)]
    for j in range(n):
        D=np.repeat(np.exp(-1j*alphas[j]),4)
        for si,p in enumerate(native['Pprofiles']):
            H=quantum.half_heat(native,p,fields[2*j+1],beta/n);words[si]=((H*D[None,:])@H)@words[si]
    return complex(pauli.source_kernel(native,Wb,Wk,Fb,Fk,words,x)[0,1])


def physical(mean:dict[str,Any])->dict[str,Any]:
    native=quantum.build_native(mean,4);V,M=native['V'],native['M'];beta=F(1,4096);b=float(beta);n=4;g=float(native['UE']/native['a']);s=2.;mode=pauli.magnetic_shape(native);x=int(np.flatnonzero(np.all(native['sites']==(1,1,0),axis=1))[0]);ref=0
    f=np.sqrt(2/V)*np.cos(2*np.pi*native['sites'][:,0]/4);dy=np.zeros(3*V);dy[1::3]=f;fractions=np.arange(2*n+1)/(2*n);rng=np.random.default_rng(20261024)
    middle=rng.normal(0,np.sqrt(s),M);difference=rng.normal(0,np.sqrt(2*g*b/(1+g*b/(2*s))),M);motion=np.vstack((np.zeros(M),np.cumsum(rng.normal(0,np.sqrt(g*b/n),(2*n,M)),axis=0)));fields=(middle+(fractions[:,None]-.5)*difference+motion-fractions[:,None]*motion[-1])@native['Q'].T
    lv,lb=np.linalg.eigh(native['Lplus']);A0map=lb[:,lv>1e-12]*np.sqrt(lv[lv>1e-12]);alphas=np.sqrt(2*g*b/n)*rng.normal(size=(n,V-1))@A0map.T
    bra=[finite.endpoint(native,p,fields[-1],mode,dy) for p in native['Pprofiles']];ket=[finite.endpoint(native,p,fields[0],mode,dy) for p in native['Pprofiles']];words=[np.eye(4*V,dtype=complex),np.eye(4*V,dtype=complex)];dwords=[[np.zeros((4*V,4*V),complex),np.zeros((4*V,4*V),complex)] for _ in native['Pprofiles']]
    for j in range(n):
        D=np.repeat(np.exp(-1j*alphas[j]),4);fraction=fractions[2*j+1]-.5
        for si,p in enumerate(native['Pprofiles']):
            H,DH=finite.half_and_tangent(native,p,fields[2*j+1],dy,b/n);block=(H*D[None,:])@H;dblock=(DH*D[None,:])@H+(H*D[None,:])@DH
            dwords[si]=[dblock@words[si]+block@dwords[si][0],fraction*dblock@words[si]+block@dwords[si][1]];words[si]=block@words[si]
    _,value,derivatives=finite.source_jet(native,bra,ket,words,dwords,x,.5,-.5)
    Efree=sum(p['load']*np.linalg.eigvalsh(quantum.matter_hamiltonian(native,p,np.zeros(3*V)))[:2*V].sum() for p in native['Pprofiles']);total=alphas.sum(axis=0);scale=(1+g*b/(2*s))**(-M/2)*np.exp(b*Efree)*np.exp(1j*(128*total.sum()+total[ref]))
    baseline=scale*independent_word(native,fields,alphas,mode,x,b);baseline_error=abs(baseline-scale*value);rows=[]
    for index,kind in enumerate(('source_midpoint','endpoint_difference')):
        shape=np.ones(2*n+1) if index==0 else fractions-.5
        for epsilon in (1e-4,5e-5):
            shift=epsilon*shape[:,None]*dy[None,:];plus=scale*independent_word(native,fields+shift,alphas,mode,x,b);minus=scale*independent_word(native,fields-shift,alphas,mode,x,b);numerical=(plus-minus)/(2*epsilon);analytic=scale*derivatives[index]
            rows.append(dict(physical_direction=kind,computational_epsilon=epsilon,actual_primitive_recomputed_derivative_real=float(numerical.real),actual_primitive_recomputed_derivative_imag=float(numerical.imag),actual_analytic_ALL64_source_and_word_derivative_real=float(analytic.real),actual_analytic_ALL64_source_and_word_derivative_imag=float(analytic.imag),physical_neutraloffset_absolute_error=float(abs(numerical-analytic))))
    if baseline_error>1e-8 or max(r['physical_neutraloffset_absolute_error'] for r in rows)>1e-8:raise ArithmeticError('Actual independent physical source-word differential failed')
    return dict(N=4,V=V,M=M,beta=str(beta),Strang_steps=n,seed=20261024,ALLcharged_flavours=64,shared_REAL_A0_modes=V-1,g_Pi_AND_Coulomb=g,source_position=[1,1,0],actual_physical_neutraloffset_baseline_cofactor_word_error=float(baseline_error),actual_bridge_differential_controls=rows,meaning='Compare new analytic source-ray/orderedfullH/exprel/cofactor ALL63 derivative with INDEPENDENT frozenprimitive graph_state occupied_section+half_heat+source_kernel recalculation at actual Gaussian bridge shifts. Both computationalsteps shown; epsilon is ONLY derivativecontrol, not a fittedphysicalparameter/sourcecut/filter. Sameactualsource and physicalPi/CAR/sharedCoulomb word, all4Refholes.',scope='NativefrozenPhi/chargedpartialPhi-gradient ONLY, not macropropagator/totalcanonicalPhi force. No finitebeta ensemblebridge/pole/commoncone/vacuum/QED promotion. Alloldsource/probability/clockbytes unchanged.')


def run(ward_path:Path,selection_path:Path)->dict[str,Any]:
    started=time.monotonic()
    if hashlib.sha256(finite.OWN_BYTES).hexdigest()!=FINITE_SOURCE_SHA:raise ArithmeticError('Frozen4c6e finiteWard source-atIMPORT required')
    wraw,sraw=ward_path.read_bytes(),selection_path.read_bytes()
    if hashlib.sha256(wraw).hexdigest()!=finite.WARD_SHA or hashlib.sha256(sraw).hexdigest()!=finite.SELECTION_SHA:raise ArithmeticError('Actual9953/9183 inputepochs required')
    cap=json.loads(wraw);_,record=quantum.checked(cap['input_descriptor']['input_receipts']['moment']);_,record=quantum.checked(record['input_descriptor']['input_receipts']['beta']);_,record=quantum.checked(record['input_descriptor']['input_receipts']['metric']);_,record=quantum.checked(record['input_descriptor']['input_receipts']['magnetic']);_,record=quantum.checked(record['input_descriptor']['input_receipts']['prior_joined_heat'])
    for key in ('full_source','contact','transfer','gram','force','join','heat'):_,record=quantum.checked(record['input_descriptor']['input_receipts'][key])
    mean=pauli.contact.transfer.packets.force.joined.mean_from_source(record)
    desc=dict(plan=cap['input_descriptor']['plan'],plan_revision=pauli.REVISION,scientific_execution_revision=pauli.REVISION,input_receipts=dict(ward=dict(path=str(ward_path),sha256=hashlib.sha256(wraw).hexdigest()),selection=dict(path=str(selection_path),sha256=hashlib.sha256(sraw).hexdigest())),finite_Ward_source_at_IMPORT_sha256=FINITE_SOURCE_SHA,source_abi=cap['input_descriptor']['source_abi'],native_N=4,beta='1/4096',Strang_steps=4,seed=20261024,source='UNCHANGEDoriginalGaussian2/canonicalY/ALL64/AP222/ALL4Refholes/Ebeforeheat; independentactualprimitiveword physicalperturbations ONLY numericaldifferential control.',a='1/64',UEphys='1',g_Pi_AND_Coulomb='64',Lambda='2^8192',chi='2^-4096',original_clock='4096')
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),actual_independent_finite_Ward_physical_differential=physical(mean),source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ward',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run(a.ward,a.selection)
    if not all(r['source_unchanged_since_import'].values()):raise ArithmeticError('Source changed during actual independentphysical differential')
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(input_sha256=r['input_sha256'],physical=r['actual_independent_finite_Ward_physical_differential'],elapsed_seconds=r['elapsed_seconds']),indent=2))
