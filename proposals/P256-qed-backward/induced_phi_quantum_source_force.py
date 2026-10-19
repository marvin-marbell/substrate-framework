"""All45 uniform Noe-normal charged-CAR Phi forces on the original EM source."""
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
from scipy.linalg import null_space

ROOT=Path(__file__).resolve().parent
OWN_BYTES=Path(__file__).read_bytes()
import induced_real_em_joined_heat_continuation as joined
from induced_phi_backreaction import scalar_paths
from induced_phi_ir_c_full_action import T
quantum=joined.quantum
SOURCE_BYTES={**joined.SOURCE_BYTES,Path(__file__).name:OWN_BYTES}
REVISION='1178517cdc4236ce3a4af6a26bed880eb20679df709826463e4db7819ea0d208'
JOIN_SHA='0adf562a3c59874cbe77e1be352884c1769faa00a07d368e6b1ac86326b93991'


def exact_bound(value:F)->dict[str,Any]:
    n,d=value.numerator,value.denominator
    e=n.bit_length()-d.bit_length()
    if (n<d*2**e if e>=0 else n*2**(-e)<d):e-=1
    return dict(numerator=str(n),denominator=str(d),binary_interval_exponents=[e,e+1])


def cart_coordinates(frames:np.ndarray)->np.ndarray:
    return 2*np.real(np.einsum('Aij,aji->aA',T,frames))


def physical_normal_geometry(native:dict[str,Any])->dict[str,Any]:
    frame=native['frame']
    orbit=np.column_stack([cart_coordinates(1j*(t[None,:,:]@frame-frame@t[None,:,:])).reshape(60) for t in T])
    Qnormal=null_space(orbit.T)
    if Qnormal.shape!=(60,45):raise ArithmeticError('Actualfull15orbit/45normal rank failed')
    roots=np.sqrt([native['g0']]+[native['gs']]*3);gamma=frame/roots[:,None,None]
    gram_directions=[];labels=[]
    for i in range(4):
        for j in range(i,4):
            Gdir=np.zeros((4,4));Gdir[i,j]=1 if i==j else 1/np.sqrt(2);Gdir[j,i]=Gdir[i,j]
            droot=Gdir/(roots[:,None]+roots[None,:])
            dframe=np.einsum('ab,bij->aij',droot,gamma)
            # RemovesONLY actualverticalgauge15; Gram variationunchanged.
            tangent=Qnormal@(Qnormal.T@cart_coordinates(dframe).reshape(60))
            gram_directions.append(tangent);labels.append([i,j])
    GramJ=np.column_stack(gram_directions)
    shape=null_space((Qnormal.T@GramJ).T)
    Qshape=Qnormal@shape
    if Qshape.shape!=(60,35):raise ArithmeticError('ActualfullTENGram/35shape rank failed')
    return dict(orbit=orbit,normal=Qnormal,gram=GramJ,shape=Qshape,labels=labels,orbit_min_singular_value=float(np.linalg.svd(orbit,compute_uv=False)[-1]),normal_vertical_defect=float(np.linalg.norm(orbit.T@Qnormal)),shape_Gram_defect=float(np.linalg.norm(GramJ.T@Qshape)))


def force_cart(native:dict[str,Any],profile:dict[str,Any],P:np.ndarray,theta:np.ndarray)->np.ndarray:
    V=native['V'];blocks=P.reshape(V,4,V,4).transpose(0,2,1,3)
    onsite,_=scalar_paths(profile['model'])
    matrices=np.zeros((4,4,4),dtype=complex)
    matrices[0]=onsite*np.sum(blocks[np.arange(V),np.arange(V)],axis=0)
    for path in profile['paths']:
        x,y=path['source'],path['target']
        z=path['seam']*np.exp(1j*np.sum(theta[list(path['edges'])]))*path['scalars']
        matrices+=z[:,None,None]*blocks[y,x]+z.conj()[:,None,None]*blocks[x,y]
    return np.real(np.einsum('Aij,aji->aA',T,matrices)).reshape(60)


def literal_force_control(native:dict[str,Any],profile:dict[str,Any],P:np.ndarray,theta:np.ndarray,force:np.ndarray)->float:
    onsite,_=scalar_paths(profile['model']);errors=[]
    for a in range(4):
        for A,t in enumerate(T):
            derivative=dict(profile,onsite=onsite*t if a==0 else np.zeros((4,4),dtype=complex),paths=[dict(path,hop=path['seam']*path['scalars'][a]*t) for path in profile['paths']])
            Hdir=quantum.matter_hamiltonian(native,derivative,theta)
            actual=float(np.real(np.trace(P@Hdir)))
            errors.append(abs(actual-force[15*a+A]))
    return max(errors)


def coefficient_row_sum(profile:dict[str,Any])->F:
    onsite,paths=scalar_paths(profile['model'])
    total=F.from_float(abs(float(onsite)))
    for _,coefficients in paths:
        total+=2*sum((F.from_float(abs(float(z.real)))+F.from_float(abs(float(z.imag))) for z in coefficients),F(0))
    return total


def run(join_path:Path,samples:int,seed:int)->dict[str,Any]:
    started=time.monotonic();raw=join_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=JOIN_SHA:raise ArithmeticError('Actual428c joinedquantum source epoch required')
    join=json.loads(raw);_,heat=quantum.checked(join['input_descriptor']['input_receipts']['heat'])
    mean=joined.mean_from_source(heat);native=quantum.build_native(mean,2);V,M=native['V'],native['M']
    geometry=physical_normal_geometry(native);normal=geometry['normal'];gram=geometry['gram'];shape=geometry['shape']
    zero=np.zeros(3*V)
    vacuum=sum((p['load']*force_cart(native,p,p['P'],zero) for p in native['Pprofiles']),np.zeros(60))/V
    source_values=[];source_minus_vacuum=[];charged_additions=[];literal_errors=[];shape_values=[];gram_values=[]
    rng=np.random.default_rng(seed);x=int(np.flatnonzero(np.all(native['sites']==(1,1,0),axis=1))[0])
    for sample in range(samples):
        theta=native['Q']@(np.sqrt(2)*rng.normal(size=M));total=np.zeros(60);added_force=np.zeros(60)
        for si,profile in enumerate(native['Pprofiles']):
            state=quantum.graph_state(native,profile,theta);P=state['P']
            force=force_cart(native,profile,P,theta);total+=profile['load']*force
            if sample==0:literal_errors.append(literal_force_control(native,profile,P,theta,force))
            if si==0:
                Q=np.eye(4*V)-P;added=Q[:,4*x:4*x+4]@Q[4*x:4*x+4,:]/2
                added_force=force_cart(native,profile,added,theta)
        perV=total/V;delta=added_force/V
        source_values.append(normal.T@perV);source_minus_vacuum.append(normal.T@(perV-vacuum));charged_additions.append(normal.T@delta)
        shape_values.append(shape.T@(perV-vacuum));gram_values.append(gram.T@(perV-vacuum))
    row_sums=[coefficient_row_sum(p) for p in native['Pprofiles']]
    # ||dPhi_a||op<=||Cart_a||/sqrt2<=||Cart_a||; everyPeierls/SU4linkunitary.
    # Nocc=2V perflavour, so ||force/V||<=2 sum load*rowsum for ANYsource.
    cap=2*sum((p['load']*row for p,row in zip(native['Pprofiles'],row_sums)),F(0))
    delta_cap=row_sums[0]/V
    original_scaled_source_vs_vacuum_cap=2*cap*F(1,2**8192)
    if max(literal_errors)>1e-9 or max(np.linalg.norm(value) for value in source_values)>float(cap)+1e-9 or max(np.linalg.norm(value) for value in charged_additions)>float(delta_cap)+1e-9:
        raise ArithmeticError('ActualCartesianforce/normalcoupling bound failed')
    descriptor=dict(plan=join['input_descriptor']['plan'],plan_revision=REVISION,scientific_execution_revision=REVISION,input_receipts=dict(join=dict(path=str(join_path),sha256=hashlib.sha256(raw).hexdigest())),source_abi=join['input_descriptor']['source_abi'],native_N=2,samples=samples,seed=seed,real_EM_modes=M,variance='2',a='1/64',UEphys='1',g_electric_AND_Coulomb='64',Phi_force='CHARGEDCARpotential contribution ONLY: -dHPhi operator onORIGINALunfilteredThetaGaussian/HermY000AP222, fixedsource; all60uniformCart directions, actual15vertical+45normal, fullTENGram+35shape. NOTsourcefamily derivative/minimum/reoptimizedsource',clock4096_UNCHANGED=True,original_mean_heavy_weight='1',original_mean_active_weight='2^-8192',original_mean_aux_weight='63*2^-8192')
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),actual_source_normal45_potential_gradient_per_V=quantum.summarize(np.asarray(source_values)),actual_ORIGINAL_source_minus_negativeband_reference_normal45_gradient_per_V=quantum.summarize(np.asarray(source_minus_vacuum)),actual_RAWall4Refholecharged_minus_neutral_normal45_gradient_per_V=quantum.summarize(np.asarray(charged_additions)),actual_fullTEN_Gram_source_minus_reference_derivative=quantum.summarize(np.asarray(gram_values)),actual_all35shape_source_minus_reference_derivative=quantum.summarize(np.asarray(shape_values)),actual_reference_vacuum_normal45_gradient_per_V=(normal.T@vacuum).tolist(),actual_geometry=dict(orbit_rank=15,normal_rank=45,Gram_rank=10,shape_rank=35,orbit_min_singular_value=geometry['orbit_min_singular_value'],normal_vertical_defect=geometry['normal_vertical_defect'],shape_Gram_defect=geometry['shape_Gram_defect'],normal_basis=normal.tolist(),Gram_horizontal_jacobian=gram.tolist(),Gram_Frobenius_labels=geometry['labels'],shape_basis=shape.tolist(),normalization='Cart15 tA:TrtAtB=deltaAB/2; homogeneousCart60 Euclidean directions; perV physicalpotentialgradient, force=-gradient'),actual_literal120_onebody_direction_control_error=max(literal_errors),uniform_ALLTheta_and_ALLstate_charged_potential_normal_force_per_V_cap=exact_bound(cap),uniform_SOURCE_added_DeltaP_normal_force_per_V_cap=exact_bound(delta_cap),uniform_original_Lambda_scaled_SOURCE_vs_reference_charged_potential_force_cap=exact_bound(original_scaled_source_vs_vacuum_cap),row_coefficient_caps=[exact_bound(value) for value in row_sums],source_pressure_proof='HCAR islinearactualuniformPhi Cartframes atALLunfilteredrealTheta. ForfixedsourceP, derivativeexpectation TrP dH; literalonsite/pathphase formula ALL60 directions. Sourceactive andaux63 combinedBEFOREmean/variance; coherentRef3 source traces toDeltaP=Q_SOURCE Rx Q_SOURCE/2, Tr1. Vertical15 removedONLYthroughactualorbit i[t,Phi], not two-variable encoding. Full45normal containsTENGram and35othernormals; offdiagGram directions horizontalized bysameorbit projector, preserving dG. PrimitivePi andrhoLplusrho doNOTdependPhi inthisfrozenHref family: their potentialforce contribution0, nottheir dynamicalomission. Negativebandreference atTheta0 isdistinctsource; actualdifference retained. Exactsource forcecap usesNocc2V, unitarypaths, Cartopnorm<=Cartnorm, allactualscalarcoefficients; originalmeanchargedweights2^-8192 scaleSOURCEvsreferenceforcecap, so no sourcepressure silentlyabsorbedinto mean.',scope='ACTUALnative all45uniformnormal potential-force/sourcepressure diagnostic plusstate-independentchargedrowbound. NOTfullPhiGaussian/Noe compactlink/heavy force integral, kineticconnection force, variationalderivativeincludingchangedsourcefamily, new mean/minimum/quantumground/rootselection/pole/commoncone/Compton. Shapecontrols/native-to-macro notpromoted; fullPhi real-time oldcaps remainfrozen. Originalsourcefamily andmean/supplier quantumkernels untouched.',source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--join',type=Path,required=True)
    parser.add_argument('--samples',type=int,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run(args.join,args.samples,args.seed)
    if not all(result['source_unchanged_since_import'].values()):raise ArithmeticError('Source changed during actual calculation')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('input_sha256','actual_fullTEN_Gram_source_minus_reference_derivative','uniform_original_Lambda_scaled_SOURCE_vs_reference_charged_potential_force_cap','actual_literal120_onebody_direction_control_error','elapsed_seconds')},indent=2))
