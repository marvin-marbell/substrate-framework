#!/usr/bin/env python3
"""New original-current CTC/full-CAR and whole-photon OUT scientific CLI.

Consumes original cached state without a SOURCE/packet/receiver reset. Native
finite matrices diagnose literal operator identities, not LEGAL probability.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from fractions import Fraction
from pathlib import Path
from typing import Any
import numpy as np
from scipy.linalg import expm,block_diag
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply
from prepared_neutral_shared_photon_state import load_shared_native_cache
from prepared_neutral_changing_paired_pulse import source_point
from functional_source_photon_bank import FunctionalSourcePhotonBank,Full108ReferenceMarks
from reference_photon_current_scattering import (
    directed_currents,WeylContour,uniform_photon_number_bound,physical_OUT_transfer,
    packet_outside_action,full_car_transition,ordinary_actual_graphs,
    initial_photon_cloud,reference_Gram_admission,joined_physical_Born,leading_reference_factorization,
    distinct_block_car_Gram,native_joint_receiver_Gram,
)
from prepared_neutral_mean_packet_intertwiner import MeanPacketIntertwiner
from driven_packet_interaction import driven_packet_bound
from reference_photon_current_car import Factor,car_word,native_closed_contour


def portable(value: Any) -> Any:
    if isinstance(value,np.ndarray):return portable(value.tolist())
    if isinstance(value,np.generic):return portable(value.item())
    if isinstance(value,complex):return dict(real=value.real,imag=value.imag)
    if isinstance(value,dict):return {str(k):portable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [portable(v) for v in value]
    return value


def fock_bilinear(matrix: np.ndarray):
    """Independent full 2^12 occupation basis, not projected one-body products."""
    n=len(matrix);rows=[];cols=[];values=[]
    for state in range(1<<n):
        for j in range(n):
            if not (state>>j)&1:continue
            removed=state^(1<<j);sj=(-1)**((state&((1<<j)-1)).bit_count())
            for i in range(n):
                if (removed>>i)&1 or matrix[i,j]==0:continue
                sign=sj*(-1)**((removed&((1<<i)-1)).bit_count())
                rows.append(removed|(1<<i));cols.append(state);values.append(sign*matrix[i,j])
    return coo_matrix((values,(rows,cols)),shape=(1<<n,1<<n)).tocsr()


def direct_CAR_proof(h: np.ndarray,current: np.ndarray,source: np.ndarray) -> dict[str,Any]:
    # Actual two neighbouring light sites plus ALL Ref4. Restriction is an
    # algebra diagnostic, never a preparation/evolution of the LEGAL state.
    n=12;occupied=np.r_[np.arange(8)<4,np.ones(4,dtype=bool)]
    L=np.vstack((source,np.zeros((4,4),complex)));R=np.vstack((np.zeros((8,4),complex),np.eye(4)))
    B=Factor(L,R);Bd=Factor(R,L);K=np.asarray(block_diag(h,np.zeros((4,4))));J=np.asarray(block_diag(current,np.zeros((4,4))))
    source_matrix=L@R.conj().T
    matrices=[fock_bilinear(a) for a in (source_matrix.conj().T,K,J,source_matrix)]
    state=sum(1<<i for i in np.flatnonzero(occupied));psi=np.zeros(1<<n,complex);psi[state]=1
    value=psi
    for matrix in reversed(matrices):value=matrix@value
    exact=complex(np.vdot(psi,value));wick=car_word(occupied,[Bd,K,J,B])
    defect=abs(exact-wick);scale=max(1.,abs(exact),abs(wick))
    if defect>1e-10*scale:raise ValueError(f'Full ordered CAR contraction failed: {defect}')
    dense_projection=complex(np.vdot(psi,matrices[0]@matrices[3]@psi))
    # Actual nonstationary neutral bra, not a neutral-return substitution.
    time_step=.4;orbitals=np.eye(n,dtype=complex)[:,occupied]
    left=expm(-1j*time_step*K)@orbitals
    fock_left=expm_multiply(-1j*time_step*matrices[1],psi)
    transition=complex(np.vdot(fock_left,matrices[2]@matrices[1]@psi))
    transitioned=full_car_transition(left,orbitals,J,K)
    transition_defect=abs(transition-transitioned)
    if transition_defect>1e-10*max(1.,abs(transition)):
        raise ValueError('Evolving full Slater/CAR transition failed')
    return_defect=float(np.linalg.norm(fock_left-psi*np.vdot(psi,fock_left)))
    return dict(full_Fock_dimension=1<<n,full_Ref_count=4,exact_fock_word=exact,full_Wick_word=wick,
        absolute_defect=defect,source_norm_squared=dense_projection,
        evolving_neutral_transition=transition,independent_full_fock_transition_defect=transition_defect,
        actual_nonreturning_neutral_norm=return_defect,
        actual_literal_two_site_restriction=True,not_a_LEGAL_state_preparation=True)


def photon_CTC_proof() -> dict[str,Any]:
    # Two actual arbitrary orthogonal vectors, independent full oscillator
    # multiplication. This finite-cutoff comparison checks the closed form;
    # the closed form itself retains the complete supplied mode space.
    cut=14;a=np.diag(np.sqrt(np.arange(1,cut)),1);ident=np.eye(cut)
    a0=np.kron(a,ident);a1=np.kron(ident,a);vac=np.zeros(cut*cut);vac[0]=1
    gin=np.array([1.,0.],complex);gout=np.array([0.,1.],complex)
    omega=np.array([.7,1.1]);left=[(np.array([.13,-.09]),.4)];right=[(np.array([-.07,.11]),1.2)]
    def operator(vertices):
        out=np.eye(cut*cut,dtype=complex)
        for phase,t in vertices:
            z=phase*np.exp(1j*omega*t)/np.sqrt(2)
            X=sum((z[j]*op.conj().T+z[j].conjugate()*op for j,op in enumerate((a0,a1))),np.zeros_like(out))
            out=out@expm(1j*X)
        return out
    direct=complex(np.vdot(vac,operator(left)@a1@operator(right)@a0.conj().T@vac))
    contour=WeylContour(omega,gin,gout);closed=complex(contour.split(left,right)['one_photon'])
    wrong=complex(contour.product(left+right)['one_photon'])
    if abs(direct-closed)>1e-12:raise ValueError('Closed-time contour photon insertion formula failed')
    return dict(exact_closed_form=closed,independent_oscillator_word=direct,absolute_defect=abs(direct-closed),
        incorrectly_moving_annihilator_across_left_Weyl_defect=abs(wrong-direct),
        neutral_backward_branch_retained=True,finite_cutoff_only_checks_formula_NOT_physical_Hilbert_space=True)


def main() -> None:
    p=argparse.ArgumentParser()
    names=('cache','manifest','descriptor','filter','mean-packet','paired-current',
           'energy','packet','core','heavy-even','packet-tail','energy-radius','source-screen')
    for name in names:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--quadrature',type=int,default=8);args=p.parse_args();start=time.perf_counter()
    paths={name:getattr(args,name.replace('-','_')) for name in names}
    before={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()}
    desc=json.loads(args.descriptor.read_text());weighted=json.loads(args.filter.read_text())
    mean_capsule=json.loads(args.mean_packet.read_text())
    energy=json.loads(args.energy.read_text());packet_capsule=json.loads(args.packet.read_text())
    core=json.loads(args.core.read_text());paired_execution=json.loads(args.paired_current.read_text())
    paired=json.loads(paired_execution['stdout'])
    if not paired_execution['exit']==0:raise ValueError('Mean current supplier did not naturally complete')
    source_bound=mean_capsule['science']['global_original_SOURCE_domain_and_bound']
    if (source_bound['original_beta_squared_upper']>2 or
        source_bound['LEGAL_SOURCE19_eta_squared_upper']>2 or
        source_bound['LEGAL_SOURCE19_descriptor_sha256']!=before['descriptor'] or
        not source_bound['global_coexact_number_commutation']):
        raise ValueError('Original global 16sqrtV SOURCE multiplier admission unavailable')
    heavy_even=json.loads(args.heavy_even.read_text())
    if not heavy_even['natural_exit']:raise ValueError('Actual even heavy/Root SOURCE control unavailable')
    model,data=load_shared_native_cache(args.cache,args.manifest);bank=FunctionalSourcePhotonBank(model,data)
    q,em,_=source_point(model,data);sample=model.basis_at(np.zeros_like(q),np.zeros_like(em));point=sample['point'];A=np.zeros((model.law.volume,3))
    original=model.original.basis_at(sample['q'],sample['em'],point)
    source=bank.fibre.source_vectors_at(point['phi'],point['links'],original['theta'],A,model.source_bank.profiles['source19'])
    ordinary=directed_currents(model.law,point['phi'],point['links'],A,bank.velocity)
    driven=directed_currents(model.law,point['phi'],point['links'],A,bank.velocity,Omega=.125)
    maxward=max(ordinary['continuity_errors']+driven['continuity_errors'])
    if maxward>1e-10:raise ValueError('Original positive-edge current continuity failed')
    mean_h=MeanPacketIntertwiner(model).light(A)
    mean_h_defect=float(np.linalg.norm(mean_h-ordinary['H']))
    if mean_h_defect>1e-9:raise ValueError('Mean packet/current API uses a different original light law')
    leg=(model.em['legs']@bank.packets[0]/np.sqrt(2)).reshape(model.law.volume,3)
    J=np.einsum('e,eij->ij',leg.ravel(),ordinary['J_original'])
    sites=np.r_[np.arange(4),4*int(model.law.target[0,0])+np.arange(4)]
    car=direct_CAR_proof(ordinary['H'][np.ix_(sites,sites)],J[np.ix_(sites,sites)],np.asarray(source['full108_plus_coefficients'])[sites])
    marks=Full108ReferenceMarks(np.asarray(source['full108_plus_coefficients']),np.asarray(source['full108_minus_coefficients']))
    packet=packet_outside_action(bank.fibre,marks,(J+J.conj().T)/2)
    base=packet['source']['source_matrix'];target=packet['relative_matrix']
    packet_controls=dict(packet['original_full_Fock_CAR_and_beforecompression_PH_controls'])
    packet_controls.pop('relative')
    packet_controls.update(full108_Ref_mark_action_norm=float(np.linalg.norm(target)),
        old_initial_two_beta_column_outside_norm=float(np.linalg.norm(target-base@np.linalg.lstsq(base,target,rcond=None)[0])),
        current_changed_packet_NOT_recaptured=True,neutral_RIGHT_not_erased=True)
    V=int(desc['N'])**3;family=desc['physical_family']
    UE=Fraction(family['UE']);c=Fraction(family['c'])
    selected=int(weighted['ledger']['selected_log2_lambda_lower'])
    fresh=ordinary_actual_graphs(weighted,V);cloud=initial_photon_cloud(weighted,V)
    # Literal original higher Touchard pullthrough bounds, not a vacuum reset.
    if max(cloud['neutral_A1_log2_upper'],cloud['neutral_A2_log2_upper'])>=-512:
        raise ValueError('Original initial factorial source bound not selected')
    initial=[Fraction(1),Fraction(1,2**512),Fraction(1,2**512)]
    moment=uniform_photon_number_bound(V,initial,UE,c,304)
    gram=reference_Gram_admission(V,UE,c,fresh['initial_CAR_comparator_norm_error_log2_upper'],selected)
    transfer=physical_OUT_transfer(fresh['NEW_ordinary_unweighted_reference_coefficient_log2_upper'],
        fresh['late_SOURCE_norm_coefficient_log2_upper'],selected,V,Fraction(1,128))
    factorization=leading_reference_factorization(desc,paired,json.loads(args.source_screen.read_text()))
    born=joined_physical_Born(desc,weighted,paired,fresh,gram,transfer,factorization)
    # Proper reference/coefficient boundary only: the evolved FULL interacting
    # D/N kernels below are never replaced by a Gaussian-after-U Wick rule.
    output_profile=model.source_bank.profiles['source19']*np.exp(
        1j*((model.law.sites-1)@(2*np.pi/int(desc['N'])*np.array([1.,-1.,0.]))))
    output_source=bank.fibre.source_vectors_at(point['phi'],point['links'],original['theta'],A,output_profile)
    reference_car=[]
    for charge in (1,-1):
        O=bank.fibre.orbitals if charge==1 else bank.fibre.orbitals.conj()
        occupied_light=np.arange(bank.fibre.n)<bank.fibre.sea_count if charge==1 else np.arange(bank.fibre.n)>=bank.fibre.sea_count+4
        hq=ordinary['H'] if charge==1 else -ordinary['H'].T
        evolving_frame=expm(-1j*6*hq)@O
        P_L=evolving_frame[:,occupied_light]@evolving_frame[:,occupied_light].conj().T
        S=np.asarray(output_source['full108_plus_coefficients']) if charge==1 else -np.asarray(output_source['full108_minus_coefficients']).T
        occupied_right=np.arange(bank.fibre.n)<bank.fibre.sea_count+4 if charge==1 else np.arange(bank.fibre.n)>=bank.fibre.sea_count
        P_R=O[:,occupied_right]@O[:,occupied_right].conj().T
        selector=np.eye(bank.fibre.n,dtype=complex)[:,4*bank.fibre.reference_site+np.arange(4)]
        S_full=S@selector.conj().T
        contraction=distinct_block_car_Gram([S_full,S_full],P_L,P_R)
        left=np.vstack((evolving_frame.conj().T@S,np.zeros((bank.fibre.n,4),complex)))
        right=np.vstack((np.zeros((bank.fibre.n,4),complex),O.conj().T@selector))
        wick=car_word(np.r_[occupied_light,occupied_right],[Factor(right,left),Factor(left,right)])
        defect=abs(wick-contraction['D'][0,0])
        wrong_plus=abs(wick-contraction['wrong_plus_adverse_Gram'][0,0])
        if defect>1e-10 or wrong_plus<=1e-8:
            raise ValueError('Distinct-L/R minus blocking contraction/adverse control failed')
        reference_car.append(dict(charge=charge,**contraction,independent_full216_ordered_CAR=wick,
            correct_minus_defect=float(defect),wrong_plus_adverse_defect=float(wrong_plus),
            complete_negative_sea_and_positive_R4_not_R_only=True,
            actual_evolving_non_ground_fixed_Slater_reference=True,
            LEFT_occupied_count=int(np.sum(occupied_light)),RIGHT_occupied_count=int(np.sum(occupied_right)),
            RIGHT_covariance='Psea+R4' if charge==1 else '(1-Psea)^T in the original hole branch',
            LEFT_covariance='Psea' if charge==1 else '(1-Psea-R4)^T in the original hole branch',
            original_physical_reference_site_selector_retained=True,
            RIGHT_density_and_multipair_not_replaced_by_onebody_Wick=True))
    joint_gram=native_joint_receiver_Gram(model,data,bank,legal_size=int(desc['N']))
    driven_bound=driven_packet_bound(energy['science']['bound'],weighted,core)
    if driven_bound['missing_prerequisites']:raise ValueError('Joined long-pulse packet graph is incomplete')
    if float(driven_bound['joined_outward_upper'])>=1e-4:
        raise ValueError('Same-H physical group comparison exceeds its prescribed target')
    if abs(float(driven_bound['joined_outward_upper'])-float(packet_capsule['joined_outward_upper']))>1e-18:
        raise ValueError('Corrected original eta18 packet source does not match bound API')
    contour=native_closed_contour(model,data,'source19',args.quadrature,int(desc['N']))
    results=dict(input_sha256=before,positive_full_edge_current=dict(maximum_continuity_defect=maxward,
        ordinary_Omega=0,driven_orbital_derivative_control_Omega=.125,range2_intermediate_edges_retained=True,
        Noether_EM_frame_terms_not_replaced_by_scalar_orbital=True),full_CAR=car,closed_time_photon=photon_CTC_proof(),
        separate_flatCAR_reference_CTC_coefficient=contour,full_packet_outside_action=packet_controls,
        mean_packet_current_native_API_defect=mean_h_defect,
        whole_photon_number_moment=moment,NEW_ordinary_actual_Dirichlet_graphs=fresh,
        original_initial_photon_cloud=cloud,historical_conditional_reference_Gram_epoch=gram,
        historical_conditional_OUT_creator_transfer=transfer,
        historical_conditional_Born_epoch=born,
        historical_conditional_long_driven_halfspin=driven_bound,
        actual_twoY_CCR_joint_Gram=joint_gram,
        actual_compatible_twoY_CTC_Born=joint_gram['compatible_CTC_Born'],
        reference_distinct_LR_CAR_minus_blocking=reference_car,
        retained_terms=dict(full_interacting_D_N='exact unexpanded full-state kernels',
            RIGHT_Coulomb_density='full RIGHT density and PH/multipair retained; not Wick-after-U',
            contact_HC_mixed='separate native CTC and all-mode remainder, not erased',
            harmonic='macro theta D/N phases retained, no narrow harmonic reset',
            Peierls='original first/second vertices, range2 both primitive displacements retained',
            source='SOURCE19 physical eta18/Mc; RAW41 distinct',
            Ref='complete negative sea and occupied positive R4, Q=1-P'),
        ordinary_T6_NOT_retimed_to_long_pulse=True,
        actual_original_physical_receiver_NOT_replaced=True,physical_Born_probability_certified=False,
        physical_halfspin_relative_comparison_certified=False,
        same_H_Born_Maxwell_physical_closure='OPEN; corrected native/reference Gram is advisory, not closure',
        same_original_H_CURRENT_SOURCE_family=True,
        original_eta18_NOT_replaced_by_unit_mean=True,
        no_Poynting_surface_or_new_source_filter_receiver_Z=True,
        elapsed_seconds=time.perf_counter()-start)
    for name,path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=before[name]:raise RuntimeError('Input changed during science')
    print(json.dumps(portable(results),indent=2,sort_keys=True))


if __name__=='__main__':main()
