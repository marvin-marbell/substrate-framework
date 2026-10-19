#!/usr/bin/env python3
"""Independent physical Etheta preparation on the literal matched-tree model.

Physical sin(theta) was already independent of the matter homotopy and had
zero positive-halfstep norm. This executes a genuinely different electric
operator, with no source/Pe cut change and no assumed photon-pole admission.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.linalg import eigh

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES_AT_IMPORT={name:(ROOT/name).read_bytes() for name in (
    'covariant_native_independent_electric_preparation.py','covariant_native_source_gram_moments.py',
    'covariant_native_haar_range_jets.py','covariant_source_band_halfstep_control.py',
    'covariant_induced_photon.py','covariant_photon_energy_control.py','sea_selection.py','rotation_band.py')}
import covariant_native_haar_range_jets as native
from covariant_native_source_gram_moments import factor
from covariant_induced_photon import Parameters,phi_frame

PARENT_PLAN_REVISION='8d37944a271479ff0ea3f97aa4ad022c3a0aaa4779384dfc95deb089890f6547'
PINNED_GENERAL='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'
PLAN_PATH=Path('/home/administrator/.omp/agent/memory/shared/plans/issue-222-induced-photon-delta13-continuation-20261008.md')
PLAN_BYTES_AT_IMPORT=PLAN_PATH.read_bytes()
PLAN_REVISION_AT_IMPORT=hashlib.sha256(PLAN_BYTES_AT_IMPORT).hexdigest()


def run() -> dict[str,Any]:
    gamma4,metric=phi_frame(Parameters(kappa_power=400))
    values,U=eigh(1j*gamma4[0]@gamma4[1]@gamma4[2]);embedding=U[:,values>0]
    _,frame=eigh(embedding.conj().T@gamma4[0]@embedding);embedding=embedding@frame[:,::-1]
    gamma=np.array([embedding.conj().T@g@embedding for g in gamma4[:3]])
    y=np.array([-.09,0.,.09]);sigma=.03;radius=.12;dy=.09
    logk=-y*y/(4*sigma*sigma)-sigma*sigma*y*y/(radius*radius*(radius*radius-y*y))
    k=np.exp(logk);k/=np.linalg.norm(k)
    D=np.zeros((2,3))
    for j in range(2):D[j,j]=math.sqrt(k[j+1]/k[j])/dy;D[j,j+1]=-math.sqrt(k[j]/k[j+1])/dy
    Hphi=2*sigma*sigma*D.T@D;phi_values,phi_basis=eigh(Hphi)
    normal_half=(phi_basis*np.exp(-phi_values/2))@phi_basis.T
    incidence=np.zeros((4,4),int)
    for edge in range(4):incidence[edge,edge]=1;incidence[(edge+1)%4,edge]=-1
    cycle=np.ones(4,int)
    gauge=np.array([.17,-.31,.23,.47]);gauge_link_change=incidence.T@gauge
    gauge_invariant_cycle_angle_defect=float(abs(cycle@gauge_link_change))
    if np.any(incidence@cycle):raise ArithmeticError('Declared theta is not a physical Gauss cycle')
    def space_and_range(charge: int) -> dict[str,Any]:
        physical=native.space(charge,12)
        F=factor(gamma,y,normal_half,physical,32,0.,'matched-source-tree-edge2')
        character=np.array([native.density(physical['states'][i][0],physical['states'][i][1],charge)[3]
            for i,_,_ in physical['physical']])
        support=np.tile(np.array([n==rho3 for (_,n,_),rho3 in zip(physical['physical'],character)]),3)
        core,singular,_=np.linalg.svd(F[support],full_matrices=False)
        V=np.zeros((F.shape[0],core.shape[1]),complex);V[support]=core
        physical['V']=V;physical['singular']=singular
        physical['Etheta']=np.tile(np.array([sum(electric)/4 for _,_,electric in physical['physical']]),3)
        physical['E3']=np.tile(np.array([electric[3] for _,_,electric in physical['physical']]),3)
        physical['rho3']=np.tile(character,3)
        scalar_connections=np.tile(np.array([np.array(native.density(physical['states'][i][0],physical['states'][i][1],charge))@np.array([0.,-.25,-.5,.25])
            for i,_,_ in physical['physical']]),3)
        physical['static_density_connection']=scalar_connections
        physical['physical_Etheta_vs_static_density_connection_on_positive_range_defect']=float(np.max(abs(physical['Etheta'][support]-scalar_connections[support])))
        return physical
    neutral=space_and_range(0);charged=space_and_range(1)
    Vn=neutral['V'];Vc=charged['V'];qn=neutral['singular'];qc=charged['singular']
    lambdaN=float(qn[0]**2);qn=qn/math.sqrt(lambdaN);qc=qc/math.sqrt(lambdaN)
    if min(qn[-1],qc[-1])<=0:raise ArithmeticError('Actual full positive range cannot be inverted')
    En=-2*np.log(qn);Ec=-2*np.log(qc);Omega=Vn[:,0]
    physical_electric=neutral['Etheta'];Eomega=physical_electric*Omega
    # Remove the measured vacuum expectation, not the physical density connection.
    electric_mean=float(np.vdot(Omega,Eomega).real)
    centered_Eomega=Eomega-electric_mean*Omega
    incoming_coefficients=qn*(Vn.conj().T@centered_Eomega)
    Zph=float(np.vdot(incoming_coefficients,incoming_coefficients).real)
    if Zph<=0:raise ArithmeticError('Independent physical electric prepared state has exactly zero norm')
    incoming_coefficients/=math.sqrt(Zph);incoming=Vn@incoming_coefficients
    cycle_curl=native.curl(neutral,3)
    magnetic_positive=Vn.conj().T@(cycle_curl@Vn)
    positive_electric=Vn.conj().T@(physical_electric[:,None]*Vn)
    density_electric=Vn.conj().T@(neutral['static_density_connection'][:,None]*Vn)
    positive_E3=Vn.conj().T@(neutral['E3'][:,None]*Vn)
    positive_rho3=Vn.conj().T@(neutral['rho3'][:,None]*Vn)
    packet=np.array([.1,.8,.4,.1]);packet/=np.linalg.norm(packet)
    outpacket=np.array([.5,.4,.1,.6]);outpacket/=np.linalg.norm(outpacket)
    S=native.source(neutral,charged,native.source_coefficients(gamma,packet),embedding,3)
    Sout=native.source(neutral,charged,native.source_coefficients(gamma,outpacket),embedding,3)
    raw_source=(S@incoming).reshape(4,Vc.shape[0])
    source=np.array([qc*(Vc.conj().T@row) for row in raw_source])
    M0=float(np.vdot(source,source).real)
    if M0<=0:raise ArithmeticError('Independent electric charged source has zero norm')
    energy_moment=float(np.sum(np.abs(source)**2*Ec[None,:]))
    transfer_moment=float(np.sum(np.abs(source)**2*qc[None,:]**2))
    outside=np.array([row-Vc@(Vc.conj().T@row) for row in raw_source])
    prepared_physical=source@Vc.T
    prepared_outside=np.array([row-Vc@(Vc.conj().T@row) for row in prepared_physical])
    # Physical Etheta outgoing observer, independent of hopping-loop interpolation.
    raw_target=(Sout@Omega).reshape(4,Vc.shape[0])
    receiver=np.array([Vc.conj().T@(charged['Etheta']*row) for row in raw_target])
    consumers=[]
    for elapsed in (0.,.5,1.,2.):
        evolved=source*np.exp(-1j*elapsed*Ec)[None,:]/math.sqrt(M0)
        amplitude=np.vdot(receiver,evolved)
        consumers.append(dict(physical_time=elapsed,coherent_Ref4_receiver_real=float(amplitude.real),
            coherent_Ref4_receiver_imag=float(amplitude.imag),coherent_Ref4_receiver_probability=float(abs(amplitude)**2),
            evolved_source_norm_squared=float(np.vdot(evolved,evolved).real)))
    bands=[]
    for threshold in (.25,2.,4.):
        selected=source[:,Ec<=threshold]
        weight=float(np.vdot(selected,selected).real)
        bands.append(dict(absolute_charged_energy_threshold=threshold,actual_band_rank=int(np.sum(Ec<=threshold)),
            source_band_norm_squared=weight,source_band_fraction=weight/M0))
    inputs=dict(parent_declared_plan_revision=PARENT_PLAN_REVISION,pinned_general_revision=PINNED_GENERAL,
        governing_plan_id='issue-222-induced-photon-delta13-continuation-20261008',
        current_plan_file_sha256_at_IMPORT=PLAN_REVISION_AT_IMPORT,
        predecessor_plan_id='issue-222-induced-photon-continuation-20261008',
        predecessor_plan_read_revision='0b10e9dd9d9b76c6870998f29019b1cdee80efc45ad3be6fe5a5a9470bf85b16',
        plan_binding='Current DELTA13 direct file read/hash at IMPORT. Predecessor read is NOT a current-cache claim. NO plan write/bind.',
        law_ABI='native-matched-source-tree0-SAME-law-physical-electric-NEW-preparation-20261008',
        homotopy_parameter=0.,active_kappa=.09,auxiliary_kappa=.04,auxiliary_sites=[0,3],all_auxiliary_canonical_occupations=6,
        active_canonical_numbers=[4,5],auxiliary_canonical_number=2,original_Ref=4,prepared_Ref=3,
        integer_tagged_charge=1,integer_Ref_hole_charge=-1,spatial_spacing=1.,transfer_duration=1.,
        normal_nodes=y.tolist(),normal_sigma=sigma,normal_radius=radius,Phi_heat_eigenvalues=phi_values.tolist(),
        Haar_grid=32,electric_Galerkin_cutoff=12,photon_preparation='sqrt(T_N) (physical Etheta minus its actual vacuum mean) Omega/sqrtZph',
        Etheta='sum_e integer E_e/4, conjugate to actual gauge-invariant theta=sum_e A_e',
        E3_vs_Etheta='E3=n conjugate in long-tree gauge; coexact Etheta=sum(E)/4 differs by the specified longitudinal/density connection, not a change of charge.',
        no_division_by_hopping_homotopy=True,no_source_or_Pe_cut_change=True,
        source_packet=packet.tolist(),receiver_packet=outpacket.tolist(),absolute_charged_energy_thresholds=[.25,2.,4.])
    return dict(input_descriptor=inputs,input_descriptor_sha256=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest(),
        source_at_import={name:hashlib.sha256(data).hexdigest() for name,data in SOURCE_BYTES_AT_IMPORT.items()},
        source_bytes_at_import={name:data.decode() for name,data in SOURCE_BYTES_AT_IMPORT.items()},physical_mean_descriptor=metric,
        actual_gauge_invariant_cycle_angle_defect=gauge_invariant_cycle_angle_defect,
        actual_cycle_shift_Gauss_defect=int(np.max(abs(incidence@cycle))),
        physical_Etheta_cycle_shift_increment=float(cycle.sum()/4),
        actual_old_physical_sin_theta_positive_range_norm=float(np.linalg.norm(magnetic_positive,2)),
        actual_physical_Etheta_positive_range_norm=float(np.linalg.norm(positive_electric,2)),
        physical_Etheta_equals_matter_density_connection_on_positive_range_defect=float(np.linalg.norm(positive_electric-density_electric,2)),
        physical_E3_equals_total_matter_rho3_on_same_positive_range_defect=float(np.linalg.norm(positive_E3-positive_rho3,2)),
        physical_coexact_Etheta_vs_rho3_on_positive_range_operator_difference=float(np.linalg.norm(positive_electric-positive_rho3,2)),
        positive_range_column_completeness=dict(
            neutral_bare_Gauss_Haar_columns_per_Phi_node=len(neutral['middle']),
            charged_bare_Gauss_Haar_columns_per_Phi_node=len(charged['middle']),
            neutral_full_active_CAR_states=len(neutral['light']),charged_full_active_CAR_states=len(charged['light']),
            all_auxiliary_CAR_states=len(neutral['auxiliary']),Phi_nodes=3,
            physical_electric_Galerkin_cutoff=12,neutral_physical_rows=Vn.shape[0],charged_physical_rows_per_Ref=Vc.shape[0],
            limitation='Complete declared finite canonical/Gauss/Phi3 columns only. NOT an untruncated EM/fullCartesianPhi/SU4/macro-source rank claim.'),
        actual_physical_Etheta_vacuum_mean=electric_mean,
        actual_neutral_independent_electric_Zph=Zph,actual_common_neutral_lambda_N=lambdaN,
        actual_source_M0_including_whole_charged_halfstep=M0,
        actual_energy_weighted_source_moment_Htransfer=energy_moment,
        actual_transfer_weighted_source_moment_Kcharged_over_lambda_neutral=transfer_moment,
        actual_transfer_Rayleigh_source_mean=transfer_moment/M0,
        actual_source_Htransfer_energy_mean=energy_moment/M0,
        actual_neutral_electric_preparation_Htransfer_energy_mean=float(np.sum(abs(incoming_coefficients)**2*En)),
        actual_raw_source_Knull_norm_squared=float(np.vdot(outside,outside).real),
        actual_prepared_source_Knull_norm_squared=float(np.vdot(prepared_outside,prepared_outside).real),
        actual_charged_absolute_energy_band_rows=bands,actual_independent_electric_source_consumer_rows=consumers,
        actual_positive_range_ranks={'neutral':Vn.shape[1],'charged_per_Ref':Vc.shape[1]},
        classification='NONZERO actual electric-field/density-dressing preparation. On this pure-tree positive range Etheta is EXACTLY the static matter-density connection, while independent sin(theta) compression stays ZERO. This does NOT supply an independent radiative photon or a massless photon pole.',
        original_direct_t1_consumer_changed=False,old_matchedtree_zero_sin_capsule_retagged=False,
        microscopic_macro_source_or_photon_pole_admitted=False,
        scope='Actual fixed-Gauss mode and materially distinct physical-electric preparation with ALL auxiliary6/Phi3/Ref4, source and consumer. Not an independent-photon, fullCartesian/SU4 or macro full125 theorem.')


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    start=time.monotonic();result=run();result['runtime_seconds']=time.monotonic()-start
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('source_at_import','input_descriptor_sha256',
        'actual_gauge_invariant_cycle_angle_defect','actual_old_physical_sin_theta_positive_range_norm',
        'actual_neutral_independent_electric_Zph','actual_source_M0_including_whole_charged_halfstep',
        'actual_energy_weighted_source_moment_Htransfer','actual_transfer_Rayleigh_source_mean',
        'actual_independent_electric_source_consumer_rows','classification','runtime_seconds')}))


if __name__=='__main__':
    main()
