#!/usr/bin/env python3
"""Actual same-law cut3 ALL36 source/current/contact/time continuation.

No convergence bound from tiny projected Knull or fixed-seed Wilson image.
All neutral S2 x C and charged S2 directions, one global neutral normalizer.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any
import numpy as np
import scipy
import covariant_native_collective_primitive_electric as base
import covariant_collective_paired_car as car
import covariant_collective_paired_source_current as current
import covariant_collective_cut3_neutral_serialized as neutral_input
import covariant_collective_cut3_charged_loader as charged_input
import covariant_collective_contact_tail as response

ROOT = Path(__file__).resolve().parent
PLAN = Path('/home/administrator/.omp/agent/memory/shared/plans/issue-222-induced-photon-delta13-continuation-20261008.md')
PLAN_BYTES_AT_IMPORT = PLAN.read_bytes()
FROZEN_INPUTS = {
    'cut1': (ROOT/'covariant_native_collective_primitive_electric_cut1_UE1_capsule_20261008.json',
             '6000a6b4a6d9e6e91c9b1f0c607a3b4708c41c878a7aa24cf08ff17182412b05'),
    'cut2': (ROOT/'covariant_collective_cut2_source_consumer_serialized_capsule_20261009.json',
             'ccf99b7a57b593335b2714952cbcfdd9fd144f542fefe778d639a8d18bfa617f'),
    'cut1_contact': (ROOT/'covariant_collective_fixed_frame_contact_capsule_20261009.json',
                     '8dd4526ed0be2e615648cd9fa9a27a2af631a8ab0e86da2124f69b0459d83cd2'),
    'neutral_cut3_source': (ROOT/'covariant_collective_cut3_neutral_serialized.py',
                            '0732671e13d0688a93ddb63afea6533a8a06394fce860bb8c7ad30c067faf1bf'),
    'charged_cut3_loader_source': (ROOT/'covariant_collective_cut3_charged_loader.py',
                            'e4eabafc35377f3755a3f36d027c0a7dca021710344b6b39717ebe7c04b96dad'),
    'charged_cut3_source': (ROOT/'covariant_collective_cut3_charged.py',
                            '74230c89f9975d80199c1dfd54e6eef767bc1033e4170d270cd482054efda9d8')}
FROZEN_BYTES_AT_IMPORT = {name:path.read_bytes() for name,(path,_) in FROZEN_INPUTS.items()}
for name,(_,expected) in FROZEN_INPUTS.items():
    if hashlib.sha256(FROZEN_BYTES_AT_IMPORT[name]).hexdigest() != expected:
        raise RuntimeError(f'Frozen physical input changed: {name}')
ARGUMENT_BYTES_AT_IMPORT: dict[str,bytes] = {}
for i,argument in enumerate(sys.argv[:-1]):
    if argument in ('--neutral-input','--charged-input','--car-proof'):
        ARGUMENT_BYTES_AT_IMPORT[argument] = Path(sys.argv[i+1]).read_bytes()
SOURCE_BYTES_AT_IMPORT: dict[str,bytes] = {}
for module in tuple(sys.modules.values()):
    filename = getattr(module,'__file__',None)
    if isinstance(filename,str):
        path = Path(filename)
        if path.suffix == '.py' and path.resolve().parent == ROOT:
            SOURCE_BYTES_AT_IMPORT[path.name] = path.read_bytes()
SOURCE_BYTES_AT_IMPORT[Path(__file__).name] = Path(__file__).read_bytes()


def signed_comparison(reference: Any, actual: Any) -> dict[str,Any]:
    old,new = np.asarray(reference,dtype=float),np.asarray(actual,dtype=float)
    if old.shape != new.shape or np.any(old == 0):
        raise ArithmeticError('Comparison shape or nonzero reference contract changed')
    delta = new-old
    return dict(reference=old.tolist(),actual=new.tolist(),signed_difference=delta.tolist(),
                absolute_difference=abs(delta).tolist(),relative_difference=(delta/old).tolist())


def source_comparison(reference: dict[str,Any], actual: dict[str,Any]) -> dict[str,Any]:
    keys = ('actual_photon_Zph','actual_photon_energy_mean','actual_M0','actual_E1','actual_R1',
            'actual_energy_mean','actual_transfer_Rayleigh_mean')
    result: dict[str,Any] = {key:signed_comparison(reference[key],actual[key]) for key in keys}
    old,new = reference['actual_same_literal_source_consumer_rows'],actual['actual_same_literal_source_consumer_rows']
    if [r['time'] for r in old] != [r['time'] for r in new]:
        raise ArithmeticError('Original consumer time coordinates changed')
    result['same_literal_coherent_receiver_probabilities'] = [dict(time=a['time'],
        **signed_comparison(a['coherent_Ref4_receiver_probability'],b['coherent_Ref4_receiver_probability'])) for a,b in zip(old,new)]
    return result


def complete_source_images(gamma: np.ndarray, embedding: np.ndarray, neutral: dict[str,Any],
                           addition: dict[str,Any], removal: dict[str,Any], Omega: np.ndarray,
                           photon: np.ndarray, packet: np.ndarray, receiver: np.ndarray) -> dict[str,Any]:
    """Complete literal source image of SPECIFIED retained cut3 seeds in cut4."""
    neutral_large = base.space(0,4)
    nr,nlarge = len(neutral['physical']),len(neutral_large['physical'])
    kept_neutral = np.array([neutral_large['index'][(s,n)] for s,n,_ in neutral['physical']])
    for old,new in zip(neutral['physical'],kept_neutral):
        if neutral_large['physical'][new][2] != old[2]:
            raise ArithmeticError('Larger source-image embedding changed an electric edge')
    seeds = {}
    for name,source in (('fixed_actual_Omega_cut3',Omega),('fixed_actual_photon_cut3',photon)):
        expanded = np.zeros((3,nlarge),dtype=complex);expanded[:,kept_neutral] = source.reshape(3,nr)
        seeds[name] = expanded.reshape(-1)
    Ebar = car.antiparticle_embedding(embedding)
    rows = []
    for charge,small in ((1,addition),(-1,removal)):
        large = base.space(charge,4)
        nc,nclarge = len(small['physical']),len(large['physical'])
        kept = np.array([large['index'][(s,n)] for s,n,_ in small['physical']])
        outside = np.ones(nclarge,dtype=bool);outside[kept] = False
        for packet_name,p in (('ORIGINAL_source',packet),('ORIGINAL_receiver',receiver)):
            if charge == 1:
                coefficients = base.collective.native.source_coefficients(gamma,p)
                oldS = base.collective.source(neutral,small,coefficients,embedding)
                largeS = base.collective.source(neutral_large,large,coefficients,embedding)
            else:
                coefficients = car.negative_coefficients(gamma,p)
                oldS = car.annihilator(neutral,small,coefficients,Ebar)
                largeS = car.annihilator(neutral_large,large,coefficients,Ebar)
            for seed,large_seed in seeds.items():
                original_seed = Omega if seed == 'fixed_actual_Omega_cut3' else photon
                image = (largeS@large_seed).reshape(4,3,nclarge)
                old_image = (oldS@original_seed).reshape(4,3,nc)
                norm = float(np.vdot(image,image).real)
                if norm <= 0:
                    raise ArithmeticError('Complete retained-vector source image vanished')
                lost = image[:,:,outside]
                loss = float(np.vdot(lost,lost).real)
                rows.append(dict(active_particle_number=4+charge,packet=packet_name,seed=seed,
                    complete_cut4_image_norm_squared=norm,raw_source_boundary_loss_norm_squared=loss,
                    raw_source_boundary_loss_fraction=loss/norm,
                    retained_cut3_operator_image_absolute_defect=float(np.linalg.norm(image[:,:,kept]-old_image))))
    return dict(rows=rows,contract='SPECIFIED actual retained compact-electric-cut3 Omega/photon vectors ONLY. Mean psi+/psi− and E/Ebar field-independent; literal integer Wilson changes each edge<=1, so cut4 contains THIS complete source image before chargedhalfstep. NOT Hermite grades or Omega4/OmegaInfinity; nonpolynomial SF already inside retained seed. No bound on unprojected SF, subsequent SF/time, continuum source or full cutoff convergence.')


def run(neutral_path: Path, charged_path: Path, car_path: Path) -> dict[str,Any]:
    for flag,path in (('--neutral-input',neutral_path),('--charged-input',charged_path),('--car-proof',car_path)):
        if path.read_bytes() != ARGUMENT_BYTES_AT_IMPORT[flag]:
            raise RuntimeError(f'Quantum argument bytes changed after IMPORT: {flag}')
    if hashlib.sha256(ARGUMENT_BYTES_AT_IMPORT['--car-proof']).hexdigest() != '517a539a51759fe1cb5e45668b3c760bbcf1201abfc83ea2cbf4884748900acf':
        raise RuntimeError('Actual full20CAR proof input changed')
    n,ncontrols = neutral_input.load_spectrum(neutral_path)
    plus,ccontrols = charged_input.load_spectrum(charged_path)
    gamma,embedding,nodes,heat,Hphi,metric = base.left.frame()
    neutral,addition,removal = base.space(0,3),base.space(1,3),base.space(-1,3)
    if (len(n.q),len(plus.q)) != (3*len(neutral['physical']),3*len(addition['physical'])):
        raise ArithmeticError('Cut3 canonical direction omitted')
    Cbody = car.physical_conjugation(removal,addition,False)
    Cquartet = car.physical_conjugation(removal,addition,True)
    Cn = car.physical_conjugation(neutral,neutral,False)
    minus = base.Spectrum([dict(V=b['V'],q=b['q'],P=Cbody.T@b['P'],sign=b['sign']) for b in plus.blocks],3*len(removal['physical']))
    packet = np.array([.1,.8,.4,.1]);packet /= np.linalg.norm(packet)
    receiver = np.array([.5,.4,.1,.6]);receiver /= np.linalg.norm(receiver)
    Ebar = car.antiparticle_embedding(embedding)
    Sa = base.collective.source(neutral,addition,base.collective.native.source_coefficients(gamma,packet),embedding)
    Sa_out = base.collective.source(neutral,addition,base.collective.native.source_coefficients(gamma,receiver),embedding)
    Sh = car.annihilator(neutral,removal,car.negative_coefficients(gamma,packet),Ebar)
    Sh_out = car.annihilator(neutral,removal,car.negative_coefficients(gamma,receiver),Ebar)
    print(json.dumps(dict(stage='actual_cut3_ALL36_source_current_contact_time',ALL_ranks=[len(n.q),len(plus.q),len(minus.q)])),flush=True)
    particle = base.observe(n,plus,neutral,addition,Sa,Sa_out,nodes,Hphi)
    hole = base.observe(n,minus,neutral,removal,Sh,Sh_out,nodes,Hphi)
    Omega,photon,Z = current.photon_state(n,neutral)
    lambdaN = float(n.q[0]**2)
    jplus,pstate = current.current_case(plus,addition,Sa,Sa_out,Omega,photon,lambdaN)
    jminus,hstate = current.current_case(minus,removal,Sh,Sh_out,Omega,photon,lambdaN)
    plus_response = response.analyze(plus,addition,pstate,lambdaN,jplus,.17)
    minus_response = response.analyze(minus,removal,hstate,lambdaN,jminus,-.17,independent_frechet=False)
    eta = complex(np.vdot(Omega,Cn@Omega))
    mapped = (Cquartet@hstate['sourcewave'].reshape(-1)).reshape(4,plus.dimension)
    mapped_coefficients = np.array([plus.project(w) for w in mapped])
    source_C = float(np.linalg.norm(mapped_coefficients-eta*pstate['Xi'])/math.sqrt(particle['actual_M0']))
    current_C = []
    for p,h in zip(pstate['currents'],hstate['currents']):
        physical = np.array([minus.wave(w) for w in h])
        transformed = (Cquartet@physical.reshape(-1)).reshape(4,plus.dimension)
        transformed_coefficients = np.array([plus.project(w) for w in transformed])
        current_C.append(float(np.linalg.norm(transformed_coefficients+eta*p)/math.sqrt(particle['actual_M0'])))
    source_operator = Cquartet@Sh@Cn.T+Sa
    receiver_operator = Cquartet@Sh_out@Cn.T+Sa_out
    vacuum = np.zeros(len(n.q));vacuum[0] = 1.
    photon_coefficients = n.project(photon)
    symmetry = [dict(copy_S2_sign=b['sign'],charge_C_sign=b['charge_C_sign'],ALL_original_sigma=len(b['q']),
        actual_global_vacuum_weight=float(np.sum(abs(vacuum[b['global_indices']])**2)),
        actual_photon_weight=float(np.sum(abs(photon_coefficients[b['global_indices']])**2))) for b in n.blocks]
    images = complete_source_images(gamma,embedding,neutral,addition,removal,Omega,photon,packet,receiver)
    cut1 = json.loads(FROZEN_BYTES_AT_IMPORT['cut1'])
    cut2 = json.loads(FROZEN_BYTES_AT_IMPORT['cut2'])
    old_contact = json.loads(FROZEN_BYTES_AT_IMPORT['cut1_contact'])
    comparisons = dict(cut1_to_cut3_source_and_time=source_comparison(cut1['actual_primitive_electric_ALL36_transfer'],particle),
        cut2_to_cut3_source_and_time=source_comparison(cut2['actual_particle_photon_source_and_time_receiver'],particle),
        cut1_to_cut3_connected_response=signed_comparison(old_contact['exact_LOG_connected_second_derivative'],plus_response['LOG_connected_second_derivative']),
        cut2_to_cut3_connected_response=signed_comparison(cut2['actual_particle_connected_fixed_frame_source_response']['LOG_connected_second_derivative'],plus_response['LOG_connected_second_derivative']))
    embedding_descriptor = dict(actual_mean_gamma3=current.complex_array(gamma),actual_selected_cliff2_into_Ref4_embedding=current.complex_array(embedding),
        actual_antiparticle_embedding=current.complex_array(Ebar),actual_ORIGINAL_positive_source_coefficients=current.complex_array(base.collective.native.source_coefficients(gamma,packet)),
        actual_ORIGINAL_negative_source_coefficients=current.complex_array(car.negative_coefficients(gamma,packet)),
        actual_ORIGINAL_positive_receiver_coefficients=current.complex_array(base.collective.native.source_coefficients(gamma,receiver)),
        actual_ORIGINAL_negative_receiver_coefficients=current.complex_array(car.negative_coefficients(gamma,receiver)),
        rule='SAME frozen kappa400 conditional metric; select positive eigenspace of i gamma0 gamma1 gamma2, then diagonalize gamma0 in selectedtwo-dimensionalspace. All source coefficients and E/Ebar are field-independent MEAN values, not moving full4Dirac/RootW frames.')
    embedding_sha = hashlib.sha256(json.dumps(embedding_descriptor,sort_keys=True).encode()).hexdigest()
    descriptor = dict(plan_id=PLAN.stem,raw_plan_sha256_at_IMPORT=hashlib.sha256(PLAN_BYTES_AT_IMPORT).hexdigest(),
        general_plan_pin='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
        quantum_argument_input_sha256={k:hashlib.sha256(v).hexdigest() for k,v in ARGUMENT_BYTES_AT_IMPORT.items()},
        frozen_input_sha256={k:hashlib.sha256(v).hexdigest() for k,v in FROZEN_BYTES_AT_IMPORT.items()},
        actual_embedding_descriptor_sha256=embedding_sha,geometry=dict(vertices=[0,1,2,3],oriented_edges=[[0,1],[1,2],[2,3],[3,0]],cycles=1),
        spacing_contract='Native labelled unit graph and one declared transfer-step clock. No additional dimensionful a is a parameter of this SF/rotor law; NOT continuous-branch a=.015625 or WilsonConsumer a=.01.',
        regulator=dict(kind='compact electric/Fourier ALL4edge cutoff',cut1=1,cut2=2,NEW_actual_cut3=3,source_image_cutoff=4,angular_grid=32),
        active_modes=8,active_CAR_particle_numbers=[3,4,5],canonical_state_counts=[56,70,56],auxiliary_copies=2,ALL_auxiliary_states=36,
        dynamic_charged_flavours_including_taggedactive=3,ONE_global_Gauss_after_ALL36=True,
        neutral_symmetry='ALL four S2copy± x chargeC±, globally sortedALLsigma',charged_symmetry='BOTH S2±, NO invalid N5-only chargeC halving',
        primitive_UE=1.,no_bare_B_squared=True,F='PhiheatHalf SF_fullcanonical exp[-sumALL4edgesE²/2]',K='FFdag',
        kappa_active=.09,kappa_aux_each=.04,light_homotopy=0.,auxiliary_path_weights=[.5,.5],
        Phi_nodes=nodes.tolist(),Phi_sigma=.03,Phi_radius=.12,Phi_rule='ONEcoordinate threelevels; frozenoldconditionalkappa400 mean/metric, NOTselfconsistentnewlaw/fullCartesianPhi',
        actual_common_neutral_lambda=lambdaN,original_Ref4_to3_source_preserved=True,removal='SAME4RefCAR NEWRef0/background0→Ref1; rdag Winverse a_negative Ebar=E*sigmax',
        original_source_packet=packet.tolist(),original_receiver_packet=receiver.tolist(),Pe=[.25,2.,4.],times=[0.,.5,1.,2.],
        original_paths=[[0,0,0,0],[-1,0,0,0],[-1,-1,0,0],[0,0,0,1]],
        current='-i[H_transfer,Ephys_e], ALL4edges/longitudinalrho',contact='fixedphysicalframe unitaryEdrive; independently calculated current-only/contact-only FrechetExp and actualdisconnected subtraction; no RootW/BKS movingframe',
        connected_probe=dict(direction_ALL4edges=[.1,.2,.3,.4],positive_imaginarytime=.5,separate_postsource_pulse=.17,proper_removal_mirrored_pulse=-.17),
        tail_partition_role='Diagnostic source/current/contact norm and actualsignedresponse contributions only; no source gate, sigma pruning or inferred cutoff-convergence bound',
        scope='Distinct nativecompact/Phi3/selectedCliff2/ALL36. NOT originalrealEM/full4Dirac/SU4/selfconsistentnewmean, massless/commoncone/mass/dispersion/fullCARretarded/RootW/BKS/0226/450a/on-shellCompton admission.')
    return dict(input_descriptor=descriptor,input_descriptor_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
        source_at_IMPORT={k:hashlib.sha256(v).hexdigest() for k,v in SOURCE_BYTES_AT_IMPORT.items()},source_bytes_at_IMPORT={k:v.decode() for k,v in SOURCE_BYTES_AT_IMPORT.items()},
        plan_bytes_at_IMPORT=PLAN_BYTES_AT_IMPORT.decode(),physical_mean_descriptor=metric,actual_embedding_descriptor=embedding_descriptor,
        runtime_packages=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),
        actual_generated_neutral_controls=ncontrols,actual_generated_charged_controls=ccontrols,actual_complete_neutral_symmetry_loads=symmetry,
        actual_particle_photon_source_and_time_receiver=particle,actual_proper_Ref0_to1_photon_source_and_time_receiver=hole,
        actual_particle_full_field_current=jplus,actual_proper_Ref0_to1_full_field_current=jminus,
        actual_particle_source_current_contact_tail_response=plus_response,actual_proper_Ref0_to1_source_current_contact_tail_response=minus_response,
        actual_ALLneutral_columns_source_C_relative_defect=float(np.linalg.norm(source_operator.data)/np.linalg.norm(Sa.data)),
        actual_ALLneutral_columns_receiver_C_relative_defect=float(np.linalg.norm(receiver_operator.data)/np.linalg.norm(Sa_out.data)),
        actual_neutral_C_overlap=current.complex_array(np.array(eta)),actual_neutral_C_orthogonal_residual=float(np.linalg.norm(Cn@Omega-eta*Omega)),
        actual_C_even_source_relative_defect=source_C,actual_ALL4edge_C_odd_current_relative_defects=current_C,
        actual_C_odd_connected_first_derivative_absolute_defects=abs(np.array(plus_response['LOG_first_derivative'])+np.array(minus_response['LOG_first_derivative'])).tolist(),
        actual_C_even_connected_second_derivative_absolute_defects=abs(np.array(plus_response['LOG_connected_second_derivative'])-np.array(minus_response['LOG_connected_second_derivative'])).tolist(),
        actual_fixed_seed_complete_source_image=images,actual_cut1_cut2_cut3_comparisons=comparisons,originalrealEM_or_Compton_admitted=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--neutral-input',type=Path,required=True)
    parser.add_argument('--charged-input',type=Path,required=True)
    parser.add_argument('--car-proof',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    start = time.monotonic()
    result = run(args.neutral_input,args.charged_input,args.car_proof)
    result['runtime_seconds'] = time.monotonic()-start
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(input_descriptor_sha256=result['input_descriptor_sha256'],source_at_IMPORT=result['source_at_IMPORT'],runtime_seconds=result['runtime_seconds'],
        actual_particle={k:result['actual_particle_photon_source_and_time_receiver'][k] for k in ('actual_photon_energy_mean','actual_M0','actual_energy_mean')},
        actual_connected=result['actual_particle_source_current_contact_tail_response']['LOG_connected_second_derivative'],
        actual_C_even_source_relative_defect=result['actual_C_even_source_relative_defect'])),flush=True)


if __name__ == '__main__':
    main()
