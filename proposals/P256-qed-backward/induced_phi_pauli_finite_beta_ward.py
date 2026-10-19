"""Finite-beta SAME-source physical Pi/CAR/Coulomb magnetic bridge Ward."""
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
from scipy.special import exprel
from induced_reference_electric_graph import to_spin_basis

ROOT=Path(__file__).resolve().parent
OWN_BYTES=Path(__file__).read_bytes()
import induced_phi_pauli_gaussian_ward as gaussian
import induced_phi_pauli_electric_coulomb_selection as selection
pauli=gaussian.pauli
quantum=pauli.quantum
SOURCE_BYTES={**gaussian.SOURCE_BYTES,**selection.SOURCE_BYTES,Path(__file__).name:OWN_BYTES}
WARD_SHA='9953ff62d8b83d1adeecb00cffb873b4e6f91508fa578c155343b985346540fd'
SELECTION_SHA='918347f1c764fa66c5088fdae20c26a6c6f80bdb05f34e3a866af3d993265298'


def endpoint(native:dict[str,Any],p:dict[str,Any],theta:np.ndarray,mode:dict[str,Any],dy:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,float]:
    state=quantum.graph_state(native,p,theta,section=False);B=state['B'];Y,R=state['Y'],state['R'];pp,nn=state['pp'],state['nn'];rank=len(pp);V=native['V']
    Dh=quantum.path_derivative(native,p,theta,dy);Dhb=to_spin_basis(Dh,B);DY=(-1j*Dhb[np.ix_(pp,nn)])/state['denominator'];DR=-R@(DY@Y+Y@DY)@R
    rv,rw=np.linalg.eigh(R)
    if min(rv)<=0:raise ArithmeticError('Actual canonical source rootR notpositive')
    roots=np.sqrt(rv);S=(rw*roots)@rw.conj().T;DS=rw@((rw.conj().T@DR@rw)/(roots[:,None]+roots[None,:]))@rw.conj().T
    values,base=np.linalg.eigh(p['onsite']);negative=base[:,values<0]
    if negative.shape!=(4,2):raise ArithmeticError('Actual source onsite negativeplane changed')
    C=np.kron(np.eye(V),B[:,2:].conj().T@negative)
    W=np.empty((4*V,rank),complex);DW=np.empty_like(W);W[pp]=-1j*Y@S;W[nn]=S;DW[pp]=-1j*(DY@S+Y@DS);DW[nn]=DS
    W=np.einsum('aA,xAk->xak',B,W.reshape(V,4,rank)).reshape(4*V,rank)@C;DW=np.einsum('aA,xAk->xak',B,DW.reshape(V,4,rank)).reshape(4*V,rank)@C
    overlap=np.einsum('sa,xsk->xak',negative.conj(),W.reshape(V,4,rank)).reshape(rank,rank)
    if np.min(np.linalg.eigvalsh(overlap))<=0:raise ArithmeticError('Original positive-onsite source Slater phase lost')
    Fop,_,DF,_=gaussian.vertices(native,p,theta,mode,dy)
    error=float(np.linalg.norm(DW.conj().T@W+W.conj().T@DW))+abs(complex(np.trace(W.conj().T@DW)))+float(np.linalg.norm(W@W.conj().T-state['P']))+float(np.linalg.norm(overlap-overlap.conj().T))
    return W,DW,Fop,DF,error


def half_and_tangent(native:dict[str,Any],p:dict[str,Any],theta:np.ndarray,dy:np.ndarray,dt:float)->tuple[np.ndarray,np.ndarray]:
    h=quantum.matter_hamiltonian(native,p,theta);Dh=quantum.path_derivative(native,p,theta,dy);values,vectors=np.linalg.eigh(h);tau=dt/2;ev=np.exp(-tau*values);H=(vectors*ev)@vectors.conj().T
    # Entire divided difference exprel(0)=1: no eigen-gap inverse/mask.
    divided=-tau*ev[None,:]*exprel(-tau*(values[:,None]-values[None,:]));DH=vectors@(divided*(vectors.conj().T@Dh@vectors))@vectors.conj().T
    return H,DH


def species_jet(L:np.ndarray,R:np.ndarray,DL:list[np.ndarray],DR:list[np.ndarray],Fop:np.ndarray,DF:list[np.ndarray],U:np.ndarray,DU:list[np.ndarray])->list[tuple[complex,complex,complex,complex]]:
    UR=U@R;LF=Fop@L;B=L.conj().T@UR;Fmat=LF.conj().T@UR;cofactor=quantum.OverlapCofactors(B);jets=[]
    for dL,dR,dF,dU in zip(DL,DR,DF,DU):
        DUR=dU@R+U@dR;dLF=dF@L+Fop@dL;DB=dL.conj().T@UR+L.conj().T@DUR;DFmat=dLF.conj().T@UR+LF.conj().T@DUR
        jets.append(cofactor.jet(Fmat,DB,DFmat))
    return jets


def source_jet(native:dict[str,Any],bra:list[tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,float]],ket:list[tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,float]],words:list[np.ndarray],dwords:list[list[np.ndarray]],x:int,bra_difference:float,ket_difference:float)->tuple[complex,complex,np.ndarray]:
    aux=species_jet(bra[1][0],ket[1][0],[bra[1][1],bra_difference*bra[1][1]],[ket[1][1],ket_difference*ket[1][1]],bra[1][2],[bra[1][3],bra_difference*bra[1][3]],words[1],dwords[1]);creator=np.zeros((4*native['V'],4),complex);creator[4*x:4*x+4]=np.eye(4);norm=0j;first=0j;derivatives=np.zeros(2,complex)
    for hole in range(4):
        L=np.column_stack((bra[0][0],creator[:,hole]));R=np.column_stack((ket[0][0],creator[:,hole]));dL=np.column_stack((bra[0][1],np.zeros(4*native['V'],complex)));dR=np.column_stack((ket[0][1],np.zeros(4*native['V'],complex)))
        active=species_jet(L,R,[dL,bra_difference*dL],[dR,ket_difference*dR],bra[0][2],[bra[0][3],bra_difference*bra[0][3]],words[0],dwords[0])
        a,aF,_,_=active[0];b,bF,_,_=aux[0];norm+=a*b**63/4;first+=(aF*b**63+63*a*bF*b**62)/4
        for j,((a,aF,da,daF),(b,bF,db,dbF)) in enumerate(zip(active,aux)):
            derivatives[j]+=(daF*b**63+63*aF*db*b**62+63*da*bF*b**62+63*a*dbF*b**62+63*62*a*bF*db*b**61)/4
    return norm,first,derivatives


def physical(mean:dict[str,Any],beta:F,samples:int,steps:list[int],seed:int)->dict[str,Any]:
    if beta<=0 or samples<4 or min(steps)<=0 or any(max(steps)%n for n in steps):raise ValueError('Positive physical beta/sampling/pairedStrang contract required')
    native=quantum.build_native(mean,4);V,M=native['V'],native['M'];b=float(beta);g=float(native['UE']/native['a']);a=float(native['a']);s=2.;mode=pauli.magnetic_shape(native);maximum=max(steps)
    x=int(np.flatnonzero(np.all(native['sites']==(1,1,0),axis=1))[0]);ref=0;f=np.sqrt(2/V)*np.cos(2*np.pi*native['sites'][:,0]/4);dy=np.zeros(3*V);dy[1::3]=f
    lv,lb=np.linalg.eigh(native['Lplus']);A0map=lb[:,lv>1e-12]*np.sqrt(lv[lv>1e-12]);Efree=sum(p['load']*np.linalg.eigvalsh(quantum.matter_hamiltonian(native,p,np.zeros(3*V)))[:2*V].sum() for p in native['Pprofiles']);normalizer=(1+g*b/(2*s))**(-M/2)*np.exp(b*Efree)
    delta_variance=2*g*b/(1+g*b/(2*s));fractions=np.arange(2*maximum+1)/(2*maximum);rng=np.random.default_rng(seed);ward={n:[] for n in steps};bare={n:[] for n in steps};creator={n:[] for n in steps};errors=[];reverse_errors=[]
    for sample in range(samples):
        middle=rng.normal(0,np.sqrt(s),M);difference=rng.normal(0,np.sqrt(delta_variance),M);motion=np.vstack((np.zeros(M),np.cumsum(rng.normal(0,np.sqrt(g*b/maximum),(2*maximum,M)),axis=0)))
        fields=(middle+(fractions[:,None]-.5)*difference+motion-fractions[:,None]*motion[-1])@native['Q'].T;alphas=np.sqrt(2*g*b/maximum)*rng.normal(size=(maximum,V-1))@A0map.T
        ket=[endpoint(native,p,fields[0],mode,dy) for p in native['Pprofiles']];bra=[endpoint(native,p,fields[-1],mode,dy) for p in native['Pprofiles']];errors.extend(e[4] for e in [*ket,*bra]);Aket=float(dy@fields[0]/a);Abra=float(dy@fields[-1]/a);total=alphas.sum(axis=0);phase=np.exp(1j*(128*total.sum()+total[ref]))
        for n in steps:
            width=maximum//n;dt=b/n;words=[np.eye(4*V,dtype=complex),np.eye(4*V,dtype=complex)];dwords=[[np.zeros((4*V,4*V),complex),np.zeros((4*V,4*V),complex)] for _ in native['Pprofiles']]
            for j in range(n):
                alpha=alphas[j*width:(j+1)*width].sum(axis=0);D=np.repeat(np.exp(-1j*alpha),4);field_index=width*(2*j+1);fraction=fractions[field_index]-.5
                for si,p in enumerate(native['Pprofiles']):
                    H,DH=half_and_tangent(native,p,fields[field_index],dy,dt);block=(H*D[None,:])@H;dblock=(DH*D[None,:])@H+(H*D[None,:])@DH
                    dwords[si]=[dblock@words[si]+block@dwords[si][0],fraction*dblock@words[si]+block@dwords[si][1]];words[si]=block@words[si]
            norm,C,DC=source_jet(native,bra,ket,words,dwords,x,.5,-.5);rwords=[w.conj().T for w in words];rdwords=[[d.conj().T for d in ds] for ds in dwords];rnorm,RC,DRC=source_jet(native,ket,bra,rwords,rdwords,x,-.5,.5)
            raw=normalizer*(phase*C*Aket+phase.conjugate()*RC*Abra)/2
            improved=normalizer/a*(phase*(s*DC[0]-delta_variance*DC[1]/2)+phase.conjugate()*(s*DRC[0]+delta_variance*DRC[1]/2))/2
            bare[n].append(raw);ward[n].append(improved);creator[n].append(normalizer*phase*norm);reverse_errors.append(float(normalizer*abs(rnorm-norm.conjugate())))
    if max(errors)>1e-8 or max(reverse_errors)>1e-4:raise ArithmeticError('Actual source tangent/reverse quantum norm failed')
    rows=[]
    for n in steps:
        w,o,z=np.array(ward[n]),np.array(bare[n]),np.array(creator[n]);rows.append(dict(Strang_steps=n,actual_reversepaired_SAME_SOURCE_finite_beta_Ward_Phi_to_Ay=quantum.summarize(w),actual_same_samples_reversepaired_bare_Phi_to_Ay=quantum.summarize(o),actual_correlated_Ward_minus_bare=quantum.summarize(w-o),RAW_creator_heat=quantum.summarize(z),conditional_Ward_Phi_to_Ay_over_creator_heat=gaussian.initial.continuation.ratio_summary(w,z)))
    return dict(N=4,V=V,M=M,beta=str(beta),samples=samples,seed=seed,Strang_steps=steps,ALLcharged_flavours=64,shared_REAL_A0_modes=V-1,g_Pi_AND_Coulomb=g,original_source_probability_before_heat='1/2',actual_source_flatRay_tangent_projector_error=max(errors),actual_reverse_word_norm_error=max(reverse_errors),rows=rows,paired_Strang_Ward_high_minus_low=quantum.summarize(np.array(ward[steps[-1]])-np.array(ward[steps[0]])),exact_Gaussian_bridge_Ward='Theta_y,ket=m_y-d_y/2, independentVar(m)=2,Var(d)=2gBeta/(1+gBeta/4). E[Theta_y,ket C]/a=[2E d_mC-(Vard/2)E d_dC]/a; reverse sourceexchange uses PLUSVard/2. OriginalGauss/Pi/A0/source marginals unchanged. ActualsourceW/F endpoints and everyactualfullH halfheat word differentiated. ALL64coherentmeans and crossflavour terms retained BEFOREproducts.',source_ray_and_word_derivative='Rsource root Frechet DS_ij=DR_ij/(sqrtR_i+sqrtR_j); fixed positive-onsite section C preservesexactsourcephase, flatdetBerry and actualP checked. Actualphysicalh exponential Frechet=-tau exp(-tau lambda_j) exprel[-tau(lambda_i-lambda_j)]; entire atdegeneracy, no eigen-gapinverse/masks. Inverse-free compound cofactor jets differentiate braF-source/carword/ket-source and ALL63 productrule.',scope='NativefrozenPhi physicalEM/CARreference, originalE_BEFOREheat. NoPhi/SU4/heavy time propagated, none removedfromfullH. ChargedmatterPhi-gradient component ONLY. No sampleGammawhitening/sourcefilter/speedfit, no pole/commoncone/vacuum/retarded/QEDpromotion. FiniteStrang/MCerrors displayed; not deterministicintegralcertificate.')




def run(ward_path:Path,selection_path:Path,beta:F,samples:int,steps:list[int],seed:int)->dict[str,Any]:
    started=time.monotonic();wraw=ward_path.read_bytes();sraw=selection_path.read_bytes()
    if hashlib.sha256(wraw).hexdigest()!=WARD_SHA or hashlib.sha256(sraw).hexdigest()!=SELECTION_SHA:raise ArithmeticError('Frozen1b4/9953 AND758/9183 physicalepochs required')
    cap=json.loads(wraw);_,record=quantum.checked(cap['input_descriptor']['input_receipts']['moment']);_,record=quantum.checked(record['input_descriptor']['input_receipts']['beta']);_,record=quantum.checked(record['input_descriptor']['input_receipts']['metric']);_,record=quantum.checked(record['input_descriptor']['input_receipts']['magnetic']);_,record=quantum.checked(record['input_descriptor']['input_receipts']['prior_joined_heat'])
    for key in ('full_source','contact','transfer','gram','force','join','heat'):_,record=quantum.checked(record['input_descriptor']['input_receipts'][key])
    mean=pauli.contact.transfer.packets.force.joined.mean_from_source(record)
    desc=dict(plan=cap['input_descriptor']['plan'],plan_revision=pauli.REVISION,scientific_execution_revision=pauli.REVISION,input_receipts=dict(ward=dict(path=str(ward_path),sha256=hashlib.sha256(wraw).hexdigest()),selection=dict(path=str(selection_path),sha256=hashlib.sha256(sraw).hexdigest())),source_abi=cap['input_descriptor']['source_abi'],native_N=4,beta=str(beta),samples=samples,seed=seed,Strang_steps=steps,source='UNCHANGEDoriginalunfilteredE/canonicalY/Gaussian2/ALL64/AP222/ALL4Refholes. ExactGaussianbridge Ward estimator changes NOsource/filter/gate/probability. ActualfullPi+sharedCoulomb+CAR retained.',a='1/64',UEphys='1',g_Pi_AND_Coulomb='64',Lambda='2^8192',chi='2^-4096',original_clock='4096')
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),actual_finite_beta_Pauli_SAME_SOURCE_Ward_transfer=physical(mean,beta,samples,steps,seed),source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ward',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--beta',type=F,required=True);p.add_argument('--samples',type=int,required=True);p.add_argument('--steps',type=int,nargs='+',required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();r=run(a.ward,a.selection,a.beta,a.samples,a.steps,a.seed)
    if not all(r['source_unchanged_since_import'].values()):raise ArithmeticError('Source changed during actual finitebeta Ward calculation')
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(input_sha256=r['input_sha256'],physical=r['actual_finite_beta_Pauli_SAME_SOURCE_Ward_transfer'],elapsed_seconds=r['elapsed_seconds']),indent=2))
