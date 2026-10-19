#!/usr/bin/env python3
"""Simultaneous K/Pi/source/state/target dressing on one Gaussian reference.

This diagnoses the affine-reference conjugation convention. It does NOT
admit the microscopic supplier or retag the native shortest-tree family.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.linalg import eigh

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES_AT_IMPORT={name:(ROOT/name).read_bytes() for name in (
    'covariant_coupled_dressing_control.py','covariant_induced_photon.py',
    'covariant_native_haar_range_jets.py','covariant_source_band_halfstep_control.py',
    'covariant_photon_energy_control.py','sea_selection.py','rotation_band.py')}
from covariant_induced_photon import Parameters,phi_frame
from covariant_source_band_halfstep_control import exact_square_half


def local_half(gamma: np.ndarray,angle: float) -> tuple[np.ndarray,float]:
    B=np.eye(4,dtype=complex)/.18;F=np.zeros((4,4),complex)
    phase=np.exp(.25j*angle)
    for x,y,mu,orientation in ((0,1,1,1),(1,2,2,1),(2,3,1,-1),(3,0,2,-1)):
        B[x,y]-=.5*phase;B[y,x]-=.5*phase.conjugate()
        z=orientation*gamma[mu,0,1]/2
        F[x,y]+=z*phase;F[y,x]-=z*phase.conjugate()
    inverse=np.linalg.inv(B)
    T=np.block([[inverse,-inverse@F],[-F.conj().T@inverse,B+F.conj().T@inverse@F]])
    perm=[0,4,1,5,2,6,3,7];T=T[np.ix_(perm,perm)]
    values,U=eigh(T)
    if values[0]<=0:raise ArithmeticError('Positive local Wilson half required')
    return (U*np.sqrt(values))@U.conj().T,float(np.sqrt(np.linalg.det(B).real))


def block_diagonal(blocks: np.ndarray) -> np.ndarray:
    count,size=blocks.shape[:2];result=np.zeros((count*size,count*size),complex)
    for i,block in enumerate(blocks):result[i*size:(i+1)*size,i*size:(i+1)*size]=block
    return result


def square_root(positive: np.ndarray) -> np.ndarray:
    values,U=eigh(positive)
    if values[0]<=0:raise ArithmeticError('Actual whole reference transfer lost positivity; no clipping allowed')
    return (U*np.sqrt(values))@U.conj().T


def conjugate(phase: np.ndarray,matrix: np.ndarray) -> np.ndarray:
    return phase[:,None]*matrix*phase.conj()[None,:]


def relative(actual: np.ndarray,expected: np.ndarray) -> float:
    return float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))


def run() -> dict[str,Any]:
    gamma4,metric=phi_frame(Parameters(kappa_power=400))
    eig,U=eigh(1j*gamma4[0]@gamma4[1]@gamma4[2]);embedding=U[:,eig>0]
    _,frame=eigh(embedding.conj().T@gamma4[0]@embedding);embedding=embedding@frame[:,::-1]
    gamma=np.array([embedding.conj().T@g@embedding for g in gamma4[:3]])
    neutral=list(itertools.combinations(range(8),4));charged=list(itertools.combinations(range(8),5))
    charged_index:dict[tuple[int,...],int]={occupied:i for i,occupied in enumerate(charged)}
    g=.37;theta=np.linspace(-1.5,1.5,9);dx=float(theta[1]-theta[0]);count=len(theta)
    slope=np.array([0.,-.25,-.5,.25]);chi=g*theta[:,None]*slope[None,:]
    Dsite=np.repeat(np.exp(-1j*chi),2,axis=1)
    background=np.exp(1j*np.sum(chi,axis=1))
    def fock_D(states: list[tuple[int,...]]) -> tuple[np.ndarray,float]:
        exterior=np.array([np.prod(Dsite[:,occupied],axis=1) for occupied in states]).T
        literal=np.array([np.exp(-1j*np.sum(chi[:,np.array(occupied)//2],axis=1)+1j*np.sum(chi,axis=1)) for occupied in states]).T
        return literal,float(np.max(abs(literal-background[:,None]*exterior)))
    dn,background_neutral_defect=fock_D(neutral);dc,background_charged_defect=fock_D(charged)
    P=-1j*(np.diag(np.ones(count-1),1)-np.diag(np.ones(count-1),-1))/(2*dx)
    H0=(P@P+np.diag(theta*theta))/2
    ev,eu=eigh(H0);heat=(eu*np.exp(-ev))@eu.conj().T
    matrices={};matter_controls=[]
    for name,states,D in (('neutral',neutral,dn),('charged',charged,dc)):
        index=np.array(states);local=[];tree=[]
        for node,angle in enumerate(g*theta):
            half,pref=local_half(gamma,float(angle))
            # "Tree" here is ONLY the coordinate transform of full direct law.
            site=np.diag(Dsite[node]);transformed=site@half@site.conj().T
            local.append(pref*np.linalg.det(half[index[:,None,:,None],index[None,:,None,:]]))
            tree.append(pref*np.linalg.det(transformed[index[:,None,:,None],index[None,:,None,:]]))
        L=block_diagonal(np.array(local));T=block_diagonal(np.array(tree));d=D.reshape(-1)
        electric=np.kron(P,np.eye(len(states)))
        electric_tree=conjugate(d,electric)
        kernel_local=L@np.kron(heat,np.eye(len(states)))@L
        kernel_tree=T@conjugate(d,np.kron(heat,np.eye(len(states))))@T
        matter_controls.append(dict(sector=name,
            Gamma_Dsite_with_background_matches_Drho_defect=background_neutral_defect if name=='neutral' else background_charged_defect,
            actual_Fock_SFtree_D_SFlocal_Ddag_relative_defect=relative(T,conjugate(d,L)),
            whole_Ktree_D_Klocal_Ddag_relative_defect=relative(kernel_tree,conjugate(d,kernel_local))))
        matrices[name]=dict(local=kernel_local,tree=kernel_tree,Pi=electric,Pi_tree=electric_tree,D=d)
    mean,_=exact_square_half(gamma,0.,0.);_,mean_basis=eigh(mean@mean)
    def coefficients(packet: np.ndarray) -> np.ndarray:
        values,vectors=eigh(mean@mean);stable=vectors[:,values<1]
        raw=np.kron(packet[:,None],np.eye(2));psi=stable@stable.conj().T@raw
        return psi/np.linalg.norm(psi,'fro')@embedding.conj().T
    packet=np.array([.1,.8,.4,.1]);packet/=np.linalg.norm(packet)
    outpacket=np.array([.5,.4,.1,.6]);outpacket/=np.linalg.norm(outpacket)
    def source(weights: np.ndarray,power: int) -> np.ndarray:
        result=np.zeros((4*count*len(charged),count*len(neutral)),complex)
        for node in range(count):
            phase=np.exp(power*1j*chi[node])
            for column,occupied in enumerate(neutral):
                mask=sum(1<<mode for mode in occupied)
                for mode in range(8):
                    if mode in occupied:continue
                    row=charged_index[tuple(sorted((*occupied,mode)))];below=(mask&((1<<mode)-1)).bit_count()
                    for ref in range(4):
                        result[(ref*count+node)*len(charged)+row,node*len(neutral)+column]=(-1)**(below+ref)*weights[mode,ref]*phase[mode//2]
        return result
    source_matrices={power:source(coefficients(packet),power) for power in (0,1,2)}
    target_matrices={power:source(coefficients(outpacket),power) for power in (0,1,2)}
    n=matrices['neutral'];c=matrices['charged'];Dn=n['D'];Dc=np.tile(c['D'],4)
    values,U=eigh(n['tree']);lambdaN=float(values[-1]);Omega=U[:,-1]
    def execute(Kn: np.ndarray,Kc: np.ndarray,Pin: np.ndarray,Pout: np.ndarray,
                S: np.ndarray,Sout: np.ndarray,initial: np.ndarray) -> dict[str,Any]:
        rootn=square_root(Kn/lambdaN);rootc=square_root(Kc/lambdaN)
        photon=rootn@(Pin@initial);Zph=float(np.vdot(photon,photon).real)
        if Zph<=0:raise ArithmeticError('Actual reference incoming photon zero')
        photon/=np.sqrt(Zph)
        raw=(S@photon).reshape(4,Kc.shape[0]);Xi=(raw@rootc.T).reshape(-1)
        M0=float(np.vdot(Xi,Xi).real)
        if M0<=0:raise ArithmeticError('Actual reference source zero')
        Xi/=np.sqrt(M0)
        target=Sout@initial;target/=np.linalg.norm(target)
        received=np.vdot(target,((Xi.reshape(4,Kc.shape[0]))@Pout.T).reshape(-1))
        return dict(Zph=Zph,M0=M0,receiver_real=float(received.real),receiver_imag=float(received.imag),
            receiver_probability=float(abs(received)**2),Xi=Xi,target=target)
    literal=execute(n['tree'],c['tree'],n['Pi_tree'],c['Pi_tree'],source_matrices[1],target_matrices[1],Omega)
    orientations=[]
    for sign,label,power in ((-1,'Ddag: wholeK and Pi local; source/target DOUBLE Wilson',2),
                            (1,'D: literal source bare; wholeK and Pi DOUBLE dressed',0)):
        un=Dn**sign;uc=c['D']**sign;initial=un*Omega
        if sign==-1:
            Kn=n['local'];Kc=c['local'];Pin=n['Pi'];Pout=c['Pi']
        else:
            Kn=conjugate(Dn*Dn,n['local']);Kc=conjugate(c['D']*c['D'],c['local'])
            Pin=conjugate(Dn*Dn,n['Pi']);Pout=conjugate(c['D']*c['D'],c['Pi'])
        actual=execute(Kn,Kc,Pin,Pout,source_matrices[power],target_matrices[power],initial)
        transformed_source=(np.tile(uc,4)[:,None]*source_matrices[1])*un.conj()[None,:]
        orientations.append(dict(orientation=label,
            same_unitary_whole_K_neutral_relative_defect=relative(conjugate(un,n['tree']),Kn),
            same_unitary_whole_K_charged_relative_defect=relative(conjugate(uc,c['tree']),Kc),
            same_unitary_physical_Pi_neutral_relative_defect=relative(conjugate(un,n['Pi_tree']),Pin),
            same_unitary_source_expected_power=power,
            same_unitary_source_expected_operator_relative_defect=relative(transformed_source,source_matrices[power]),
            same_unitary_source_vs_BARE_operator_relative_difference=relative(transformed_source,source_matrices[0]),
            same_unitary_Xi_relative_defect=relative(actual['Xi'],np.tile(uc,4)*literal['Xi']),
            same_unitary_fixed_target_relative_defect=relative(actual['target'],np.tile(uc,4)*literal['target']),
            Zph=actual['Zph'],M0=actual['M0'],receiver_real=actual['receiver_real'],receiver_imag=actual['receiver_imag'],
            receiver_probability=actual['receiver_probability'],
            receiver_probability_invariant_absolute_defect=abs(actual['receiver_probability']-literal['receiver_probability'])))
    shortcut=execute(n['local'],c['local'],n['Pi'],c['Pi'],source_matrices[0],target_matrices[0],Dn.conj()*Omega)
    for row in orientations:
        if max(row['same_unitary_source_expected_operator_relative_defect'],row['same_unitary_whole_K_neutral_relative_defect'],row['same_unitary_fixed_target_relative_defect'])>1e-10:
            raise ArithmeticError('Coupled dressing identity failed')
    inputs=dict(reference_only=True,field_coupling=g,integer_body_source_charge=1,integer_Ref_hole_charge=-1,
        original_Ref=4,prepared_Ref=3,background_per_body_site=1,Ref_site=0,Ref_phase=0.,
        theta_nodes=theta.tolist(),coexact_source_path_slopes=slope.tolist(),active_kappa=.09,
        neutral_CAR_number=4,charged_CAR_number=5,field_reference_H='(Pi²+theta²)/2',transfer_duration=1.,
        source_packet=packet.tolist(),receiver_packet=outpacket.tolist(),
        tree_notation='Gauge-coordinate transform of full direct law, NOT native primitive matched-tree t0')
    return dict(input_descriptor=inputs,input_descriptor_sha256=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest(),
        source_at_import={name:hashlib.sha256(data).hexdigest() for name,data in SOURCE_BYTES_AT_IMPORT.items()},
        source_bytes_at_import={name:data.decode() for name,data in SOURCE_BYTES_AT_IMPORT.items()},physical_mean_descriptor=metric,
        actual_Fock_background_and_whole_K_controls=matter_controls,actual_simultaneous_orientation_consumer_rows=orientations,
        common_neutral_lambda_N=lambdaN,literal_consumer={key:value for key,value in literal.items() if key not in ('Xi','target')},
        rejected_all_boundaries_bare_shortcut={key:value for key,value in shortcut.items() if key not in ('Xi','target')},
        rejected_shortcut_consumer_probability_absolute_error=abs(shortcut['receiver_probability']-literal['receiver_probability']),
        conclusion='SAME D is Gamma(Dsite) times the density-background scalar, not its inverse. Ddag frees wholeK/Pi but literal source and fixed target retain DOUBLE Wilson phases. D frees literal source but wholeK/Pi retain DOUBLE dressing. Initial Xi is transformed in BOTH; only each consistent coupled consumer is invariant. NO all-boundaries-free claim.',
        microscopic_source_or_same_tree_family_admitted=False,
        scope='Executed finite Gaussian coexact reference sign/consumer discriminator with full CAR4/5 and coherent Ref4. No compact/harmonic omission is promoted to a microscopic theorem; −184 word estimate remains reference-only.')


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    start=time.monotonic();result=run();result['runtime_seconds']=time.monotonic()-start
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('source_at_import','input_descriptor_sha256',
        'actual_Fock_background_and_whole_K_controls','actual_simultaneous_orientation_consumer_rows',
        'literal_consumer','rejected_all_boundaries_bare_shortcut','rejected_shortcut_consumer_probability_absolute_error','runtime_seconds')}))


if __name__=='__main__':
    main()
