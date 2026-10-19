#!/usr/bin/env python3
"""Actual prepared charged/neutral source moments on the native full Gauss range.

This finite two-spin/cycle/one-normal-coordinate control does not replace the
macro full125 supplier or supply a full Cartesian/SU4 vacuum-phase proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.linalg import eigh

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES_AT_IMPORT={name:(ROOT/name).read_bytes() for name in (
    'covariant_native_source_gram_moments.py','covariant_native_haar_range_jets.py',
    'covariant_source_band_halfstep_control.py','covariant_photon_energy_control.py',
    'covariant_induced_photon.py','sea_selection.py','rotation_band.py')}
import covariant_native_haar_range_jets as native
from covariant_induced_photon import Parameters,phi_frame
from covariant_source_band_halfstep_control import exact_square_half


def matched_tree_half(gamma: np.ndarray,theta: float,coupling: float) -> tuple[np.ndarray,float]:
    """Same literal shortest source tree at0; original direct law at1."""
    B=np.eye(4,dtype=complex)/.18;F=np.zeros((4,4),complex)
    phases=(1.,1.,coupling+(1-coupling)*np.exp(-1j*theta),np.exp(1j*theta))
    for phase,(x,y,mu,orientation) in zip(phases,((0,1,1,1),(1,2,2,1),(2,3,1,-1),(3,0,2,-1))):
        B[x,y]-=.5*phase;B[y,x]-=.5*np.conjugate(phase)
        z=orientation*gamma[mu,0,1]/2
        F[x,y]+=z*phase;F[y,x]-=z*np.conjugate(phase)
    inverse=np.linalg.inv(B)
    T=np.block([[inverse,-inverse@F],[-F.conj().T@inverse,B+F.conj().T@inverse@F]])
    permutation=[0,4,1,5,2,6,3,7];T=T[np.ix_(permutation,permutation)]
    values,U=eigh(T)
    if values[0]<=0:raise ArithmeticError('Actual matched-tree onebody transfer is not positive')
    return (U*np.sqrt(values))@U.conj().T,float(np.sqrt(np.linalg.det(B).real))


def factor(gamma: np.ndarray,y: np.ndarray,normal_half: np.ndarray,
           physical: dict[str,Any],grid: int,homotopy_parameter: float,family: str) -> np.ndarray:
    middle=[physical['states'][i] for i in physical['middle']]
    light_columns=sorted(set(l for l,_ in middle));aux_columns=sorted(set(a for _,a in middle))
    li={l:i for i,l in enumerate(light_columns)};ai={a:i for i,a in enumerate(aux_columns)}
    columns_l=np.array([li[l] for l,_ in middle]);columns_a=np.array([ai[a] for _,a in middle])
    row_l=np.array(physical['light']);col_l=np.array(light_columns)
    row_a=np.array(physical['auxiliary']);col_a=np.array(aux_columns)
    row_state=np.array([i for i,_,_ in physical['physical']])
    evaluate_half={'long-tree-edge3':exact_square_half,'matched-source-tree-edge2':matched_tree_half}[family]
    row_flux=np.array([n%grid for _,n,_ in physical['physical']])
    blocks=[];tail=0.
    pure_tree=family=='matched-source-tree-edge2' and homotopy_parameter==0.
    physical['exact_single_electric_character']=pure_tree
    character=np.array([native.density(physical['states'][i][0],physical['states'][i][1],physical['charge'])[3]
        for i,_,_ in physical['physical']])
    actual_integer=np.array([n for _,n,_ in physical['physical']])
    for value in y:
        actual=gamma.copy();actual[1]*=1+float(value)
        if pure_tree:
            half,prefactor=matched_tree_half(actual,0.,0.)
            active=prefactor*np.linalg.det(half[row_l[:,None,:,None],col_l[None,:,None,:]])
            ah,ap=native.auxiliary_half(actual,.04,0.)
            aux=ap*np.linalg.det(ah[row_a[:,None,:,None],col_a[None,:,None,:]])
            full=(active[:,columns_l][:,None,:]*aux[None,:,columns_a]).reshape(len(physical['states']),len(middle))
            block=np.zeros((len(row_state),len(middle)),complex)
            selected=actual_integer==character
            block[selected]=full[row_state[selected]]
            blocks.append(block)
            continue
        samples=[]
        for theta in 2*math.pi*np.arange(grid)/grid:
            half,prefactor=evaluate_half(actual,float(theta),homotopy_parameter)
            active=prefactor*np.linalg.det(half[row_l[:,None,:,None],col_l[None,:,None,:]])
            ah,ap=native.auxiliary_half(actual,.04,float(theta))
            aux=ap*np.linalg.det(ah[row_a[:,None,:,None],col_a[None,:,None,:]])
            full=active[:,columns_l][:,None,:]*aux[None,:,columns_a]
            samples.append(full.reshape(len(physical['states']),len(middle)))
        coefficients=np.fft.fft(np.array(samples),axis=0)/grid
        blocks.append(coefficients[row_flux,row_state,:])
        higher=[j for j in range(grid) if min(j,grid-j)>2]
        tail=max(tail,float(np.linalg.norm(coefficients[higher])))
    rows=len(physical['physical']);columns=len(middle);nodes=len(y)
    F=np.zeros((nodes*rows,nodes*columns),complex)
    for out in range(nodes):
        for inp in range(nodes):
            F[out*rows:(out+1)*rows,inp*columns:(inp+1)*columns]=normal_half[out,inp]*blocks[inp]
    physical['actual_Fourier_norm_above_integer_degree2']=tail
    return F


def run(homotopy_parameter: float,family: str) -> dict[str,Any]:
    p=Parameters(kappa_power=400);gamma4,metric=phi_frame(p)
    values,U=eigh(1j*gamma4[0]@gamma4[1]@gamma4[2]);embedding=U[:,values>0]
    _,frame=eigh(embedding.conj().T@gamma4[0]@embedding);embedding=embedding@frame[:,::-1]
    gamma=np.array([embedding.conj().T@g@embedding for g in gamma4[:3]])
    y=np.array([-.09,0.,.09]);sigma=.03;radius=.12;dy=.09
    matched_identity=[]
    for normal in y:
        actual=gamma.copy();actual[1]*=1+float(normal)
        free,_=exact_square_half(actual,0.,0.)
        for theta in (-.4,-.1,.1,.4):
            G=np.diag(np.repeat(np.exp(1j*np.array([0.,0.,0.,theta])),2))
            tree,_=matched_tree_half(actual,theta,0.)
            direct,_=matched_tree_half(actual,theta,1.)
            original,_=exact_square_half(actual,theta,1.)
            tree_defect=float(np.linalg.norm(tree-G@free@G.conj().T,2))
            direct_defect=float(np.linalg.norm(direct-original,2))
            if max(tree_defect,direct_defect)>1e-12:raise ArithmeticError('Same-source-tree/free or direct endpoint identity failed')
            matched_identity.append(dict(normal_coordinate=float(normal),loop_angle=theta,
                same_literal_source_tree_positive_half_identity_defect=tree_defect,
                unchanged_original_direct_endpoint_defect=direct_defect))
    logk=-y*y/(4*sigma*sigma)-sigma*sigma*y*y/(radius*radius*(radius*radius-y*y))
    k=np.exp(logk);k/=np.linalg.norm(k)
    D=np.zeros((2,3))
    for j in range(2):D[j,j]=math.sqrt(k[j+1]/k[j])/dy;D[j,j+1]=-math.sqrt(k[j]/k[j+1])/dy
    Hphi=2*sigma*sigma*D.T@D;ev,evec=eigh(Hphi)
    normal_half=(evec*np.exp(-ev/2))@evec.T
    neutral=native.space(0,12);charged=native.space(1,12)
    Fn=factor(gamma,y,normal_half,neutral,32,homotopy_parameter,family)
    Fc=factor(gamma,y,normal_half,charged,32,homotopy_parameter,family)
    def positive_range(F: np.ndarray,physical: dict[str,Any]) -> tuple[np.ndarray,np.ndarray]:
        if physical['exact_single_electric_character']:
            support=np.tile(np.array([n==native.density(physical['states'][i][0],physical['states'][i][1],physical['charge'])[3]
                for i,n,_ in physical['physical']]),3)
            core,q,_=np.linalg.svd(F[support],full_matrices=False)
            U=np.zeros((F.shape[0],core.shape[1]),complex);U[support]=core
            return U,q
        U,q,_=np.linalg.svd(F,full_matrices=False)
        return U,q
    Un,qn=positive_range(Fn,neutral);Uc,qc=positive_range(Fc,charged)
    scale=float(qn[0]**2);qn/=math.sqrt(scale);qc/=math.sqrt(scale)
    if min(qn[-1],qc[-1])<=0:raise ArithmeticError('Actual positive native range is not invertible')
    En=-2*np.log(qn);Ec=-2*np.log(qc)
    moment_contract=dict(K_neutral='Fn Fn†',K_charged='Fc Fc† tensor identity_Ref4',
        common_neutral_lambda_N=scale,charged_lambda_N_is_same_neutral=True,
        normalized_transfer='T_C=K_C/lambda_N; T_N=K_N/lambda_N',
        physical_hbar=1.,primitive_transfer_duration=1.,spatial_spacing=1.,
        Hamiltonian='H_transfer=−log(K_C/lambda_N)/tau on its true positive range; tau=1 here',
        energy_units='inverse native transfer-time, hbar=1; NOT the independent continuum Dirac Hamiltonian',
        prepared_photon='Xi_ph=sqrt(T_N) B_N Omega/sqrt(Zph); Omega is actual neutral ground',
        prepared_source='Xi_ch=sqrt(T_C) S_literal Xi_ph',
        whole_charged_normalized_halfsteps_per_prepared_source_vector=1,
        M0='||Xi_ch||²; includes ONE WHOLE charged sqrt(K_C/lambda_N) on each norm side',
        energy_moment='E1=<Xi_ch,H_transfer Xi_ch>; E1/M0 is ENERGY, not transfer Rayleigh',
        transfer_moment='R1=<Xi_ch,(K_C/lambda_N) Xi_ch>; R1/M0 is TRANSFER Rayleigh',
        null_support='True nullspace of the DECLARED finite integer-electric/Phi Galerkin positive transfers; not an untruncated-lattice claim',
        neutral_state_kind='Actual native ground eigenvector; not a finite-Nt stationary density')
    vacuum=Un[:,0]
    packet=np.array([.1,.8,.4,.1]);packet/=np.linalg.norm(packet)
    outpacket=np.array([.5,.4,.1,.6]);outpacket/=np.linalg.norm(outpacket)
    S=native.source(neutral,charged,native.source_coefficients(gamma,packet),embedding,3)
    Sout=native.source(neutral,charged,native.source_coefficients(gamma,outpacket),embedding,3)
    def charged_prepare(state: np.ndarray) -> np.ndarray:
        raw=(S@state).reshape(4,Uc.shape[0])
        return np.array([qc*(Uc.conj().T@row) for row in raw])
    Bn=native.curl(neutral,3);Bc=native.curl(charged,3)
    inputs=dict(parameters={'kappa_power':400,'active_kappa':.09,'auxiliary_kappa':.04},
        law_ABI='native-Wilson-path-homotopy-2spin-aux6-integerGauss-Phi3-20261008',
        spatial_spacing=1.,primitive_transfer_duration=1.,
        spatial_graph_coordinates=[[0,0],[1,0],[1,1],[0,1]],
        charged_species_count=2,active_orbitals_per_site=2,
        auxiliary_orbitals=4,auxiliary_sites=[0,3],auxiliary_species_count=1,
        Ref_role='four static fermion modes with filled-charge background minus4; no Ref hopping/heat supplied',
        active_canonical_numbers=[4,5],auxiliary_canonical_number=2,all_auxiliary_occupations=6,
        original_Ref=4,prepared_Ref=3,integer_tagged_charge=1,integer_Ref_hole_charge=-1,
        normal_nodes=y.tolist(),normal_sigma=sigma,normal_radius=radius,normal_heat_eigenvalues=ev.tolist(),
        Haar_grid=32,electric_Galerkin_cutoff=12,homotopy_parameter=homotopy_parameter,source_packet=packet.tolist(),
        point_family=family,
        declared_link_phases={'long-tree-edge3':'edge3=1+t(exp(i theta)-1)',
            'matched-source-tree-edge2':'edge3=exp(i theta), edge2=t+(1-t)exp(-i theta)'}[family],
        source_paths=[[0,0,0,0],[-1,0,0,0],[-1,-1,0,0],[0,0,0,1]],
        charge_and_statistical_seam_scaled=False,
        finite_Galerkin_scope='Positive finite integer-electric Galerkin transfer. Direct t1 Fourier alias/outsidecut tail is NOT bounded by these source moments.',
        receiver_packet=outpacket.tolist(),incoming_photon_operator='coherent sin(integercycleangle), both orientations',
        neutral_normalization='actual largest neutral transfer eigenvalue; NO charged rest-energy subtraction',
        charged_band_thresholds=[.25,2.,4.],physical_times=[0.,.5,1.,2.])
    raw_photon=Bn@vacuum
    photon_amplitudes=qn*(Un.conj().T@raw_photon)
    Zph=float(np.vdot(photon_amplitudes,photon_amplitudes).real)
    photon_raw_Knull=raw_photon-Un@(Un.conj().T@raw_photon)
    photon_raw_Knull_norm_squared=float(np.vdot(photon_raw_Knull,photon_raw_Knull).real)
    if Zph==0:
        source_zero=charged_prepare(vacuum);Z0=float(np.vdot(source_zero,source_zero).real)
        compressed=Un.conj().T@(Bn@Un)
        if np.linalg.norm(compressed)!=0:raise ArithmeticError('Claimed exact tree photon obstruction has nonzero matrix')
        return dict(input_descriptor=inputs,input_descriptor_sha256=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest(),
            source_at_import={name:hashlib.sha256(data).hexdigest() for name,data in SOURCE_BYTES_AT_IMPORT.items()},
            source_bytes_at_import={name:data.decode() for name,data in SOURCE_BYTES_AT_IMPORT.items()},
            physical_mean_descriptor=metric,neutral_positive_dimension=Un.shape[1],charged_positive_dimension=Uc.shape[1],
            actual_same_literal_source_tree_endpoint_identity_rows=matched_identity,
            actual_joint_Phi_Gauss_auxiliary_Z0=Z0,actual_joint_Phi_Gauss_auxiliary_Zph=Zph,
            actual_joint_Phi_Gauss_auxiliary_M0=None,
            actual_energy_weighted_source_moment_Htransfer=None,
            actual_transfer_weighted_source_moment_Kcharged_over_lambda_neutral=None,
            actual_transfer_Rayleigh_source_mean=None,source_operator_contract=moment_contract,
            actual_common_neutral_lambda_N=scale,
            actual_raw_incoming_photon_Knull_norm_squared=photon_raw_Knull_norm_squared,
            actual_minimum_charged_transfer_energy=float(Ec.min()),actual_charged_band_rows=None,
            actual_coherent_source_receiver_consumer_rows=None,
            actual_full_positive_range_photon_compression_norm=float(np.linalg.norm(compressed)),
            normalizable_incoming_photon_exists=False,
            exact_physical_obstruction='Matched active tree and the single-edge auxiliary share ONE tree. Both halfsteps are total-density pure gauges. Every output matter state has the single integer character n=rho3. The coherent cycle curl shifts n by plus/minus1 at unchanged matter state, hence Ppositive B Ppositive=0 exactly even with all6aux occupations and Phi heat.',
            undefined_normalized_source_semantics='M0, energy moment, transfer moment and receiver are null because Zph is EXACTLY zero: no normalized photon/source state exists in this declared native family. No floating-point Fourier noise is normalized into a counterfeit photon.',
            microscopic_macro_admission_proved=False,
            scope='Exact native matched-tree/fullaux6/Gauss/Phi3 NO-photon obstruction. Full macro auxiliary spatial loops are NOT represented by this single-edge auxiliary, so this is not a macro impossibility theorem.')
    if Zph<=0:raise ArithmeticError('Actual neutral prepared incoming photon has zero norm')
    incoming=Un@(photon_amplitudes/math.sqrt(Zph))
    source_zero=charged_prepare(vacuum);source_photon=charged_prepare(incoming)
    M0=float(np.vdot(source_photon,source_photon).real)
    Z0=float(np.vdot(source_zero,source_zero).real)
    if min(M0,Z0)<=0:raise ArithmeticError('Actual charged prepared source norm is zero')
    moments=[float(np.sum(np.abs(source_photon)**2*Ec[None,:]**grade)) for grade in range(4)]
    transfer_moment=float(np.sum(np.abs(source_photon)**2*qc[None,:]**2))
    raw_source=(S@incoming).reshape(4,Uc.shape[0])
    source_Knull=np.array([row-Uc@(Uc.conj().T@row) for row in raw_source])
    prepared_source_physical=source_photon@Uc.T
    prepared_source_Knull=np.array([row-Uc@(Uc.conj().T@row) for row in prepared_source_physical])
    source_Knull_norm_squared=float(np.vdot(source_Knull,source_Knull).real)
    prepared_source_Knull_norm_squared=float(np.vdot(prepared_source_Knull,prepared_source_Knull).real)
    receiver_raw=(Sout@vacuum).reshape(4,Uc.shape[0])
    receiver=np.array([Uc.conj().T@(Bc@row) for row in receiver_raw])
    photon_low=np.array([Un.conj().T@(Bn@Un[:,j]) for j in range(Un.shape[1])]).T
    charged_curl=np.array([Uc.conj().T@(Bc@Uc[:,j]) for j in range(Uc.shape[1])]).T
    def marginals(name: str,state: np.ndarray,physical: dict[str,Any]) -> dict[str,Any]:
        wave=state.reshape(-1,3,len(physical['physical']))
        norm=float(np.vdot(wave,wave).real)
        if norm<=0:raise ArithmeticError('Actual quantum marginal has zero state norm')
        probabilities=np.sum(np.abs(wave)**2,axis=0)/norm
        node_probability=np.sum(probabilities,axis=1);row_probability=np.sum(probabilities,axis=0)
        mean_y=float(node_probability@y);variance_y=float(node_probability@(y*y)-mean_y**2)
        phi_energy=float(np.einsum('cnr,nm,cmr->',wave.conj(),Hphi,wave).real/norm)
        electric_squared=np.array([sum(e*e for e in electric) for _,_,electric in physical['physical']])
        aux_probability=np.zeros(6)
        aux_index={occupied:i for i,occupied in enumerate(physical['auxiliary'])}
        for probability,(index,_,_) in zip(row_probability,physical['physical']):
            aux_probability[aux_index[physical['states'][index][1]]]+=probability
        mean_frame=gamma4.copy();mean_frame[1]*=1+mean_y
        defect=max(float(np.linalg.norm(a@b+b@a-2*np.eye(4)*(i==j),2))
            for i,a in enumerate(mean_frame) for j,b in enumerate(mean_frame))
        return dict(state=name,actual_norm_squared=norm,normal_node_probabilities=node_probability.tolist(),
            actual_normal_coordinate_mean=mean_y,actual_normal_coordinate_variance=variance_y,
            actual_primitive_normal_heat_energy=phi_energy,
            actual_integer_electric_squared_sum_mean=float(row_probability@electric_squared),
            all_six_auxiliary_canonical_occupation_probabilities=aux_probability.tolist(),
            all_six_auxiliary_canonical_occupations=[list(a) for a in physical['auxiliary']],
            actual_mean_frame_Clifford_defect_relative_to_frozen_mean=defect,
            status='Actual native quantum expectation; NOT a new full-Cartesian mean, stationary-phase admission, or common-cone repair.')
    quantum_marginals=[
        marginals('actual_neutral_Gauss_Phi_auxiliary_ground',vacuum,neutral),
        marginals('actual_prepared_neutral_incoming_photon',incoming,neutral),
        marginals('actual_prepared_charged_source_without_photon',source_zero@Uc.T,charged),
        marginals('actual_prepared_charged_source_with_photon',source_photon@Uc.T,charged)]
    bands=[]
    for threshold in (.25,2.,4.):
        mask=Ec<=threshold
        selected=source_photon[:,mask]
        bands.append(dict(absolute_charged_energy_threshold=threshold,actual_band_rank=int(np.sum(mask)),
            actual_prepared_source_band_norm_squared=float(np.vdot(selected,selected).real),
            actual_prepared_source_band_fraction=float(np.vdot(selected,selected).real/M0)))
    consumers=[]
    for elapsed in (0.,.5,1.,2.):
        evolved=source_photon*np.exp(-1j*elapsed*Ec)[None,:]/math.sqrt(M0)
        received=np.vdot(receiver,evolved)
        consumers.append(dict(physical_time=elapsed,coherent_Ref4_receiver_real=float(received.real),
            coherent_Ref4_receiver_imag=float(received.imag),coherent_Ref4_receiver_probability=float(abs(received)**2),
            actual_evolved_source_norm_squared=float(np.vdot(evolved,evolved).real)))
    return dict(input_descriptor=inputs,input_descriptor_sha256=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest(),
        source_at_import={name:hashlib.sha256(data).hexdigest() for name,data in SOURCE_BYTES_AT_IMPORT.items()},
        source_bytes_at_import={name:data.decode() for name,data in SOURCE_BYTES_AT_IMPORT.items()},physical_mean_descriptor=metric,
        neutral_positive_dimension=Un.shape[1],charged_positive_dimension=Uc.shape[1],
        actual_same_literal_source_tree_endpoint_identity_rows=matched_identity,
        charged_Ref_hole_channels=4,actual_joint_Phi_Gauss_auxiliary_Z0=Z0,
        actual_joint_Phi_Gauss_auxiliary_Zph=Zph,actual_joint_Phi_Gauss_auxiliary_M0=M0,
        actual_energy_weighted_source_moment_Htransfer=moments[1],
        actual_transfer_weighted_source_moment_Kcharged_over_lambda_neutral=transfer_moment,
        actual_transfer_Rayleigh_source_mean=transfer_moment/M0,
        actual_raw_energy_moments=moments,source_operator_contract=moment_contract,
        actual_common_neutral_lambda_N=scale,
        actual_raw_incoming_photon_Knull_norm_squared=photon_raw_Knull_norm_squared,
        actual_raw_source_Knull_norm_squared=source_Knull_norm_squared,
        actual_prepared_source_Knull_norm_squared=prepared_source_Knull_norm_squared,
        actual_raw_source_positive_projection_norm_squared=float(np.vdot(raw_source,raw_source).real)-source_Knull_norm_squared,
        actual_neutral_Knull_dimension=Un.shape[0]-Un.shape[1],
        actual_charged_Knull_dimension_including_Ref4=4*(Uc.shape[0]-Uc.shape[1]),
        actual_prepared_neutral_incoming_photon_energy_mean=float(np.sum(np.abs(photon_amplitudes)**2*En)/Zph),
        actual_normalized_source_energy_mean=moments[1]/M0,
        actual_normalized_source_energy_variance=moments[2]/M0-(moments[1]/M0)**2,
        actual_minimum_charged_transfer_energy=float(Ec.min()),actual_charged_band_rows=bands,
        actual_coherent_source_receiver_consumer_rows=consumers,
        actual_joint_source_ground_quantum_marginals=quantum_marginals,
        actual_neutral_coherent_photon_operator_norm=float(np.linalg.norm(photon_low,2)),
        actual_charged_coherent_photon_operator_norm=float(np.linalg.norm(charged_curl,2)),
        actual_neutral_Fourier_norm_above_integer_degree2=neutral['actual_Fourier_norm_above_integer_degree2'],
        actual_charged_Fourier_norm_above_integer_degree2=charged['actual_Fourier_norm_above_integer_degree2'],
        equations=['Tn=Fn Fndag/lambda_neutral, Tc=Fc Fcdag/lambda_neutral are actual positive Gauss/Phi/fullaux transfers.',
            'Omega is the ACTUAL largest neutral eigenvector. Xi_ph=sqrtTn Bn Omega/sqrtZph, not a static auxiliary ground trial.',
            'Xi_ch=sqrtTc S Xi_ph includes ONE WHOLE charged normalized halfstep. Energy E1=<Xi_ch,H_transfer Xi_ch>; transfer R1=<Xi_ch,Tc Xi_ch>. E1/M0 must NEVER substitute for R1/M0.',
            'The consumer is <Sout Omega, Bc exp(-it(-logTc)) Xi_ch/sqrtM0>; both curl orientations and all4 Ref channels interfere coherently.',
            'No charged rest normalization is invented to populate the quarter band. Empty actual quarter band is a discriminator, not macro impossibility.'],
        microscopic_macro_admission_proved=False,
        scope='Actual full canonical auxiliary/integer U1 Gauss/three-normal-level native source moments and time consumer. NOT full Cartesian Phi/SU4 Gauss, macro full125, quantum heavy-vacuum phase, or an admitted quartercut source.')


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--homotopy-parameter',type=float,required=True)
    parser.add_argument('--family',choices=('long-tree-edge3','matched-source-tree-edge2'),required=True);args=parser.parse_args()
    started=time.monotonic();result=run(args.homotopy_parameter,args.family);result['runtime_seconds']=time.monotonic()-started
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('source_at_import','input_descriptor_sha256',
        'actual_joint_Phi_Gauss_auxiliary_Z0','actual_joint_Phi_Gauss_auxiliary_Zph',
        'actual_joint_Phi_Gauss_auxiliary_M0','actual_energy_weighted_source_moment_Htransfer',
        'actual_transfer_weighted_source_moment_Kcharged_over_lambda_neutral',
        'actual_transfer_Rayleigh_source_mean','actual_common_neutral_lambda_N',
        'actual_minimum_charged_transfer_energy','actual_charged_band_rows',
        'actual_coherent_source_receiver_consumer_rows','runtime_seconds')}))


if __name__=='__main__':
    main()
