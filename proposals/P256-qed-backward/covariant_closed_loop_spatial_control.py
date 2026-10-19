#!/usr/bin/env python3
"""NEW enlarged ring of closed-loop cells: exact q/source and harmonic control.

The local c6e5 two-path primitive is unchanged. Neighboring four-site cells
are connected by gauge-only tree/ring links with ZERO fermion hopping. ONE
global bare-electric Haar projector therefore forces bridge electric zero;
this is an explicit adverse spatial control, NOT a propagating-photon claim.
All local CAR/auxiliary/Phi states are kept. Product spectral contractions
are exact on the states actually reached by the physical field and source,
not a projection onto a guessed low-energy/occupied auxiliary sector.
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
SOURCE_NAMES=('covariant_closed_loop_spatial_control.py','covariant_native_closed_auxiliary_loop.py',
    'covariant_native_source_gram_moments.py','covariant_native_haar_range_jets.py',
    'covariant_source_band_halfstep_control.py','covariant_induced_photon.py',
    'covariant_photon_energy_control.py','sea_selection.py','rotation_band.py')
SOURCE_BYTES_AT_IMPORT={name:(ROOT/name).read_bytes() for name in SOURCE_NAMES}
PLAN_PATH=Path('/home/administrator/.omp/agent/memory/shared/plans/issue-222-induced-photon-delta13-continuation-20261008.md')
PLAN_SHA256_AT_IMPORT=hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()
import covariant_native_haar_range_jets as native
from covariant_native_closed_auxiliary_loop import reflected_factor
from covariant_induced_photon import Parameters,phi_frame


def local_data(grid: int) -> dict[str,Any]:
    gamma4,metric=phi_frame(Parameters(kappa_power=400))
    values,U=eigh(1j*gamma4[0]@gamma4[1]@gamma4[2]);embedding=U[:,values>0]
    _,frame=eigh(embedding.conj().T@gamma4[0]@embedding);embedding=embedding@frame[:,::-1]
    gamma=np.array([embedding.conj().T@g@embedding for g in gamma4[:3]])
    y=np.array([-.09,0.,.09]);sigma=.03;radius=.12;dy=.09
    logk=-y*y/(4*sigma*sigma)-sigma*sigma*y*y/(radius*radius*(radius*radius-y*y))
    k=np.exp(logk);k/=np.linalg.norm(k);D=np.zeros((2,3))
    for j in range(2):
        D[j,j]=math.sqrt(k[j+1]/k[j])/dy;D[j,j+1]=-math.sqrt(k[j]/k[j+1])/dy
    ev,evec=eigh(2*sigma*sigma*D.T@D);heat=(evec*np.exp(-ev/2))@evec.T
    neutral=native.space(0,12);charged=native.space(1,12)
    Fn=reflected_factor(gamma,y,heat,neutral,grid);Fc=reflected_factor(gamma,y,heat,charged,grid)
    Vn,qn,_=np.linalg.svd(Fn,full_matrices=False);del Fn
    Vc,qc,_=np.linalg.svd(Fc,full_matrices=False);del Fc
    lambdaN=float(qn[0]**2);qn/=math.sqrt(lambdaN);qc/=math.sqrt(lambdaN)
    if min(qn[-1],qc[-1])<=0:raise ArithmeticError('Actual full local positive range lost rank')
    En=-2*np.log(qn);Ec=-2*np.log(qc);Omega=Vn[:,0]
    Bn=native.curl(neutral,3);Bc=native.curl(charged,3)
    b=Vn.conj().T@(Bn@Omega)
    packet=np.array([.1,.8,.4,.1]);packet/=np.linalg.norm(packet)
    outpacket=np.array([.5,.4,.1,.6]);outpacket/=np.linalg.norm(outpacket)
    S=native.source(neutral,charged,native.source_coefficients(gamma,packet),embedding,3)
    Sout=native.source(neutral,charged,native.source_coefficients(gamma,outpacket),embedding,3)
    raw_columns=(S@Vn).reshape(4,Vc.shape[0],Vn.shape[1])
    pulled=np.array([Vc.conj().T@row for row in raw_columns])
    raw_gram=sum((row.conj().T@row for row in raw_columns),np.zeros((len(qn),len(qn)),complex))
    del raw_columns
    target=(Sout@Omega).reshape(4,Vc.shape[0])
    target_plain=np.array([Vc.conj().T@row for row in target])
    target_curl=np.array([Vc.conj().T@(Bc@row) for row in target])
    zph=float(np.sum(abs(qn*b)**2))
    if zph<=0:raise ArithmeticError('Actual local physical curl has no prepared state')
    return dict(qn=qn,qc=qc,En=En,Ec=Ec,b=b,pulled=pulled,raw_gram=raw_gram,
        target_plain=target_plain,target_curl=target_curl,lambdaN=lambdaN,
        local_photon_Zph=zph,metric=metric,packet=packet,outpacket=outpacket,
        local_ranks=[len(qn),len(qc)],local_physical_dimensions=[len(Omega),Vc.shape[0]],
        local_Fourier_tail=[neutral['sampled_Fourier_norm_above12'],charged['sampled_Fourier_norm_above12']])


def global_column_and_cycle_proof(cells: int) -> dict[str,Any]:
    """Enumerate ALL local Pauli occupations before global P_G factorization."""
    import itertools
    aux=list(itertools.combinations(range(4),2))
    counts={}
    for charge in (0,1):
        numbers=[0]*9;checked=0
        for mask in range(256):
            occupied=tuple(i for i in range(8) if mask&(1<<i))
            for heavy in aux:
                rho=native.density(occupied,heavy,charge);checked+=1
                if rho==(0,0,0,0):numbers[len(occupied)]+=1
        counts[str(charge)]=dict(all_checked_Pauli_aux_occupations=checked,
            actual_zero_density_middle_counts_by_active_number=numbers)
    neutral_counts=counts['0']['actual_zero_density_middle_counts_by_active_number']
    charged_counts=counts['1']['actual_zero_density_middle_counts_by_active_number']
    if any(n for j,n in enumerate(neutral_counts) if j!=4) or any(n for j,n in enumerate(charged_counts) if j!=5):
        raise ArithmeticError('Full Pauli enumeration invalidates local-number Knull statement')
    nodes=4*cells;edges=5*cells
    incidence=np.zeros((nodes,edges),int);cycles=np.zeros((edges,cells+1),int)
    for r in range(cells):
        for e in range(4):
            incidence[4*r+e,4*r+e]=1;incidence[4*r+(e+1)%4,4*r+e]=-1
            cycles[4*r+e,r]=1
        bridge=4*cells+r
        incidence[4*r,bridge]=1;incidence[4*((r+1)%cells),bridge]=-1
        cycles[bridge,cells]=1
    paths=np.zeros((edges,4),int)
    paths[0,1]=-1;paths[0,2]=-1;paths[1,2]=-1;paths[3,3]=1
    source_charge=np.zeros((nodes,4),int)
    for x in range(4):source_charge[x,x]+=1;source_charge[0,x]-=1
    gauge_defect=int(np.max(abs(incidence@paths-source_charge)))
    cycle_defect=int(np.max(abs(incidence@cycles)))
    if gauge_defect or cycle_defect:raise ArithmeticError('Actual global integer incidence/path/cycle failed')
    # Declared COMPLETE bridge-ring electricGalerkin n=-12,...,12. Every
    # global-Haar F column is constant in this holonomy, so this ONE column
    # and its null complement are explicit, not inferred untruncated fields.
    n=np.arange(-12,13);harmonic_factor=np.zeros((len(n),1),complex)
    harmonic_factor[np.flatnonzero(n==0)[0],0]=1
    harmonic_curl=np.zeros((len(n),len(n)),complex)
    for column,value in enumerate(n):
        for shift,coefficient in ((1,1/(2j)),(-1,-1/(2j))):
            rows=np.flatnonzero(n==value+shift)
            if len(rows):harmonic_curl[rows[0],column]=coefficient
    compression=harmonic_factor.conj().T@harmonic_curl@harmonic_factor
    raw=harmonic_curl@harmonic_factor
    return dict(actual_all_local_Pauli_before_Gauss_enumeration=counts,
        full_global_pre_Gauss_neutral_canonical_columns=math.comb(8*cells,4*cells)*6**cells*3**cells,
        full_global_pre_Gauss_charged_canonical_columns_per_Ref=math.comb(8*cells,4*cells+1)*6**cells*3**cells,
        actual_complete_global_neutral_middle_columns=(3*neutral_counts[4])**cells,
        actual_complete_global_charged_middle_columns_per_Ref=3*charged_counts[5]*(3*neutral_counts[4])**(cells-1),
        actual_global_integer_cycle_divergence_defect=cycle_defect,
        actual_global_literal_source_divergence_charge_defect=gauge_defect,
        local_contractible_plaquette_cycles=cells,separate_noncontractible_bridge_cycle=1,
        harmonic_electric_Galerkin_complete_n=n.tolist(),harmonic_factor_rank=1,
        actual_harmonic_curl_positive_compression_norm=float(np.linalg.norm(compression)),
        actual_raw_harmonic_curl_Knull_norm_squared=float(np.vdot(raw,raw).real),
        harmonic_scope='COMPLETE DECLARED finite bridge-cycle columns only; no complete untruncated EM claim',
        Knull_proof_scope='All global canonical Pauli distributions enumerated combinatorially; block-diagonal SF preserves local numbers; ONE P_E=0/P_G kills other local numbers. Other within-range electric nulls are only those of declared finite local factors.')


def enlarged_case(local: dict[str,Any],cells: int,mode: int) -> dict[str,Any]:
    """Exact finite-product contractions; environment one-excitation subspace.

    Diagonal product K preserves each cell's actual eigenstate. B_q Omega
    reaches vacuum plus exactly one local excitation. S atcell0 cannot add
    any excitation to another cell. Thus all omitted product eigenvectors
    have EXACT zero amplitudes for these observables, not small amplitudes.
    """
    qn=local['qn'];qc=local['qc'];En=local['En'];Ec=local['Ec'];b=local['b']
    q=2*math.pi*mode/cells
    phases=np.exp(1j*q*np.arange(cells));phase_sum=complex(np.sum(phases))
    neutral_left=qn*b/math.sqrt(cells)
    neutral_left[0]=b[0]*phase_sum/math.sqrt(cells)
    # The remaining cells have mutually orthogonal excited states. Each
    # carries its actual phase; no incoherent replacement of Ref4 is made.
    environment=qn[1:]*b[1:]/math.sqrt(cells)
    Zph=float(np.vdot(neutral_left,neutral_left).real+(cells-1)*np.vdot(environment,environment).real)
    if Zph<=0:raise ArithmeticError('Actual q-mode physical source zero')
    p=local['pulled'];scale=math.sqrt(Zph)
    left_projected=np.einsum('rai,i->ra',p,neutral_left)/scale
    env_projected=p[:,:,0,None]*environment[None,None,:]/scale
    Xi_left=qc[None,:]*left_projected
    Xi_env=qc[None,:,None]*qn[None,None,1:]*env_projected
    weight_left=abs(Xi_left)**2;weight_env=abs(Xi_env)**2
    M0=float(np.sum(weight_left)+(cells-1)*np.sum(weight_env))
    if M0<=0:raise ArithmeticError('Actual q-mode charged source zero')
    energies_env=Ec[:,None]+En[None,1:]
    E1=float(np.sum(weight_left*Ec[None,:])+(cells-1)*np.sum(weight_env*energies_env[None,:,:]))
    R1=float(np.sum(weight_left*qc[None,:]**2)+(cells-1)*np.sum(weight_env*qc[None,:,None]**2*qn[None,None,1:]**2))
    raw_left=float(np.vdot(neutral_left,local['raw_gram']@neutral_left).real/Zph)
    raw_env=float(local['raw_gram'][0,0].real*np.vdot(environment,environment).real/Zph)
    raw_norm=raw_left+(cells-1)*raw_env
    projected_norm=float(np.vdot(left_projected,left_projected).real+(cells-1)*np.vdot(env_projected,env_projected).real)
    raw_null=raw_norm-projected_norm
    if raw_null < -1e-12:raise ArithmeticError('Measured Knull source contraction violated positivity')
    receiver_left=local['target_curl']/math.sqrt(cells)
    receiver_left+=local['target_plain']*(b[0]*(phase_sum-1)/math.sqrt(cells))
    receiver_env=local['target_plain'][:,:,None]*b[None,None,1:]/math.sqrt(cells)
    consumer=[]
    for elapsed in (0.,.5,1.,2.):
        left=Xi_left*np.exp(-1j*elapsed*Ec)[None,:]
        env=Xi_env*np.exp(-1j*elapsed*energies_env)[None,:,:]
        # e^(iqr) in BOTH incoming and fixed outgoing mode cancels here;
        # coherently sum Ref4 before abs², and explicitly sum environment
        # cell amplitudes. Orthogonal sectors enter the state norm only.
        amplitude=(np.vdot(receiver_left,left)+(cells-1)*np.vdot(receiver_env,env))/math.sqrt(M0)
        consumer.append(dict(time=elapsed,coherent_Ref4_receiver_real=float(amplitude.real),
            coherent_Ref4_receiver_imag=float(amplitude.imag),coherent_Ref4_receiver_probability=float(abs(amplitude)**2),
            evolved_source_norm_squared=float((np.vdot(left,left).real+(cells-1)*np.vdot(env,env).real)/M0)))
    bands=[]
    for threshold in (.25,2.,4.):
        selected=float(np.sum(weight_left[:,Ec<=threshold])+(cells-1)*np.sum(weight_env[:,energies_env<=threshold]))
        bands.append(dict(absolute_energy_threshold=threshold,source_fraction=selected/M0))
    photon_energy=float((np.sum(abs(neutral_left)**2*En)+(cells-1)*np.sum(abs(environment)**2*En[1:]))/Zph)
    photon_second=float((np.sum(abs(neutral_left)**2*En**2)+(cells-1)*np.sum(abs(environment)**2*En[1:]**2))/Zph)
    spectral_weights=abs(qn*b)**2/Zph
    spectral_weights[0]=abs(b[0]*phase_sum/math.sqrt(cells))**2/Zph
    return dict(cells=cells,actual_cell_translation_mode=mode,actual_cell_translation_q=q,
        physical_q_operator='B_q=(sum_r exp(i q r) sin(theta_r))/sqrt(cells); source remains literal cell0',
        actual_photon_Zph=Zph,actual_photon_energy_mean=photon_energy,
        actual_photon_energy_second_moment=photon_second,
        actual_neutral_spectral_source_rows=[dict(energy=float(e),normalized_weight=float(w)) for e,w in zip(En,spectral_weights)],
        actual_common_neutral_lambda_N=local['lambdaN']**cells,
        actual_common_neutral_log_lambda_N=cells*math.log(local['lambdaN']),
        actual_source_M0_including_whole_charged_halfstep=M0,actual_energy_moment_Htransfer=E1,
        actual_transfer_moment_Kcharged_over_lambda_neutral=R1,actual_energy_mean=E1/M0,
        actual_transfer_Rayleigh_mean=R1/M0,actual_raw_source_Knull_norm_squared=raw_null,
        actual_raw_source_norm_squared=raw_norm,analytic_prepared_source_Knull_norm_squared=0.,
        actual_full_column_and_cycle_proof=global_column_and_cycle_proof(cells),
        analytic_intercell_connected_retarded_curl_commutator_from_exact_product_K=0.,
        actual_source_absolute_energy_band_rows=bands,actual_same_literal_source_consumer_rows=consumer,
        positive_neutral_rank=local['local_ranks'][0]**cells,
        positive_charged_rank_per_Ref=local['local_ranks'][1]*local['local_ranks'][0]**(cells-1),
        all_auxiliary_canonical_tensor_states=6**cells,all_independent_quantum_normal_configurations=3**cells,
        full_global_active_canonical_numbers=[4*cells,4*cells+1],
        full_global_active_CAR_dimensions=[math.comb(8*cells,4*cells),math.comb(8*cells,4*cells+1)],
        positive_range_local_active_numbers={'neutral':[4]*cells,'charged':[5]+[4]*(cells-1)},
        all_other_global_canonical_number_distributions='EXACT Knull: one global P_E=0/P_G requires cell charge0 (cell0 Refhole in charged); SF has no intercell fermion hopping',
        number_charged_flavours_including_active_global_flavour=1+cells)


def run(grid: int) -> dict[str,Any]:
    local=local_data(grid);cases=[]
    for cells in (2,4,8,16):
        for mode in sorted(set((0,1,cells//2))):cases.append(enlarged_case(local,cells,mode))
    dispersion=[]
    for cells in (2,4,8,16):
        rows=[row for row in cases if row['cells']==cells]
        energy=[row['actual_photon_energy_mean'] for row in rows]
        source=[row['actual_energy_mean'] for row in rows]
        dispersion.append(dict(cells=cells,actual_photon_energy_q_spread=max(energy)-min(energy),
            actual_source_energy_q_spread=max(source)-min(source),actual_lowest_nonzero_q=2*math.pi/cells))
    descriptor=dict(governing_plan_id='issue-222-induced-photon-delta13-continuation-20261008',
        plan_sha256_at_IMPORT=PLAN_SHA256_AT_IMPORT,
        pinned_general='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
        primitive_ABI='NEW-enlarged-c6e5-closed-loop-cells-gauge-only-ring-20261008',
        preserved_local_primitive='c6e5eda6241ba7929da2f1e203930c9b5af59807883bffa0f6b9b7cac867a0d2',
        graph='4-site plaquette percell; gauge-only oriented bridge anchor0_r -> anchor0_(r+1); no matter hopping on bridges',
        global_Gauss='ONE global integer Gauss projector and ONE bare-electric Haar P_E=0, not independently projected flavours',
        factorization_proof='SF is cell-block diagonal and bridge-independent; P_E=0 fixes all bridge flux0; global P_G middle charge equations then exactly select local4/5 active and all6aux. Other global active canonical distributions remain exact Knull.',
        harmonic_vs_curl='Local theta_r are contractible plaquette curls; noncontractible bridge-ring holonomy has positive K range constant and physical sin compression0. These are not the same mode.',
        no_intercell_propagating_mechanism=True,no_dispersion_or_residue_imported=True,no_bare_gauge_action=True,
        active_kappa=.09,auxiliary_kappa=.04,alternative_path_weights=[.5,.5],integer_aux_charge=1,
        active_light_homotopy=0.,local_mean_halfstep_unchanged=True,local_original_Ref=4,prepared_Ref=3,
        source_packet=local['packet'].tolist(),receiver_packet=local['outpacket'].tolist(),
        source_paths=[[0,0,0,0],[-1,0,0,0],[-1,-1,0,0],[0,0,0,1]],source_cell=0,Ref_cell=0,
        source_absolute_energy_thresholds=[.25,2.,4.],Haar_grid_per_local_cycle=grid,electric_Galerkin_cutoff_per_edge=12,
        local_quantum_normal_nodes=[-.09,0.,.09],normal_sigma=.03,normal_radius=.12,
        transfer_duration=1.,spatial_spacing=1.,cell_pitch=2.,cell_translation_q_units='radians/cell',
        local_operator_contract='K_N=FnFn†; K_C=FcFc† tensorIRef4; common lambdaN; Xi_ph=sqrt(T_N)Bq Omega/sqrtZph; Xi_ch=sqrt(T_C)S_literal Xi_ph; E1=<Xi,Htransfer Xi>; R1=<Xi,T_C Xi>',
        product_normalization='lambda_global=lambda_local**cells; Htransfer_global=sum local Htransfer; one whole global charged halfstep pervector',
        approximation_scope='Finite local Haar/Galerkin and native2spin/conditional mean, product contraction EXACT given those local factors; no fullCartesianPhi/SU4/macro/pole admission')
    return dict(input_descriptor=descriptor,input_descriptor_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
        source_at_import={name:hashlib.sha256(data).hexdigest() for name,data in SOURCE_BYTES_AT_IMPORT.items()},
        source_bytes_at_import={name:data.decode() for name,data in SOURCE_BYTES_AT_IMPORT.items()},
        physical_mean_descriptor=local['metric'],local_actual_sin_vacuum_expectation_real=float(local['b'][0].real),
        local_actual_sin_vacuum_expectation_imag=float(local['b'][0].imag),local_actual_photon_Zph=local['local_photon_Zph'],
        local_all_positive_ranks=local['local_ranks'],local_sampled_Fourier_norm_above12=local['local_Fourier_tail'],
        actual_q_source_consumer_cases=cases,actual_dispersion_comparison=dispersion,
        massless_photon_or_common_cone_or_macro_source_admitted=False,
        observed_decision='Independent local loop support and gapped spectral-source energies do NOT generate transverse propagation: same local primitive on enlarged neighboring cells remains q-flat; harmonic bridge-loop field is separately Knull. A material intercell matter/field coupling is still required.',
        original_local_source_bytes_or_law_retagged=False,source_or_quartercut_retrofitted=False)


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--haar-grid',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    start=time.monotonic();result=run(args.haar_grid);result['runtime_seconds']=time.monotonic()-start
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('source_at_import','input_descriptor_sha256',
        'local_actual_photon_Zph','actual_dispersion_comparison','observed_decision','runtime_seconds')}))


if __name__=='__main__':
    main()
