#!/usr/bin/env python3
"""Actual larger-regulator ALL36 photon/source/removal/current/time consumer.

Neutral ALL S2 x charge-C sectors and charged ALL S2 sectors are generated
quantum inputs. No hand-selected support, imported massless pole, scalar sea
or reinterpretation of prior cut1 source-boundary loss as an Omega bound.
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
import covariant_collective_cut2_neutral_loader as neutral_input
import covariant_collective_cut2_charged as charged_input
import covariant_collective_fixed_frame_contact_typed as contact

ROOT = Path(__file__).resolve().parent
PLAN = Path('/home/administrator/.omp/agent/memory/shared/plans/issue-222-induced-photon-delta13-continuation-20261008.md')
PLAN_BYTES_AT_IMPORT = PLAN.read_bytes()
FROZEN_INPUTS = {
    'N1_cut2': (ROOT / 'covariant_native_electric_heat_cut2_UE1_capsule_20261008.json',
                '79c285fe8252dbb4187bab2fff5bbc3b872a9d7e57856df17f2cc79d8cf4bf4e'),
    'N2_cut1': (ROOT / 'covariant_native_collective_primitive_electric_cut1_UE1_capsule_20261008.json',
                '6000a6b4a6d9e6e91c9b1f0c607a3b4708c41c878a7aa24cf08ff17182412b05'),
    'N2_cut1_contact': (ROOT / 'covariant_collective_fixed_frame_contact_capsule_20261009.json',
                       '8dd4526ed0be2e615648cd9fa9a27a2af631a8ab0e86da2124f69b0459d83cd2'),
    'fixed_frame_contact_source': (ROOT / 'covariant_collective_fixed_frame_contact_typed.py',
                                  '615761b13f7c0faa5be9c5e4d86f49bb36b9bdf1dc94afbd0f066da0368adbe6'),
    'N2_cut1_paired': (ROOT / 'covariant_collective_paired_source_current_cut1_UE1_capsule_20261009.json',
                       'af2fb68e0b98152aa34140bb0849b51db53a34c612ae056e9eeac9b8fc37f1de')}
FROZEN_BYTES_AT_IMPORT = {name: path.read_bytes() for name,(path,_) in FROZEN_INPUTS.items()}
for name,(_,expected) in FROZEN_INPUTS.items():
    if hashlib.sha256(FROZEN_BYTES_AT_IMPORT[name]).hexdigest() != expected:
        raise RuntimeError(f'Frozen scientific input changed: {name}')
ARGUMENT_BYTES_AT_IMPORT: dict[str, bytes] = {}
for i, argument in enumerate(sys.argv[:-1]):
    if argument in ('--neutral-input', '--charged-input', '--car-proof'):
        ARGUMENT_BYTES_AT_IMPORT[argument] = Path(sys.argv[i + 1]).read_bytes()
SOURCE_BYTES_AT_IMPORT: dict[str, bytes] = {}
for module in tuple(sys.modules.values()):
    filename = getattr(module, '__file__', None)
    if isinstance(filename, str):
        path = Path(filename)
        if path.suffix == '.py' and path.resolve().parent == ROOT:
            SOURCE_BYTES_AT_IMPORT[path.name] = path.read_bytes()
SOURCE_BYTES_AT_IMPORT[Path(__file__).name] = Path(__file__).read_bytes()


def complete_source_images(gamma: np.ndarray, embedding: np.ndarray, neutral: dict[str, Any],
                           addition: dict[str, Any], removal: dict[str, Any], Omega: np.ndarray,
                           photon: np.ndarray, packet: np.ndarray, receiver: np.ndarray) -> dict[str, Any]:
    """Cut2 fixed actual states, COMPLETE cut3 literal-source image only."""
    neutral3 = base.space(0, 3)
    nr2, nr3 = len(neutral['physical']), len(neutral3['physical'])
    rows = np.array([neutral3['index'][(s,n)] for s,n,_ in neutral['physical']])
    for old,new in zip(neutral['physical'], rows):
        if neutral3['physical'][new][2] != old[2]:
            raise ArithmeticError('Neutral source-image embedding changed a physical electric edge')
    expanded = {}
    for name,wave in (('fixed_actual_Omega_cut2',Omega),('fixed_actual_photon_cut2',photon)):
        big = np.zeros((3,nr3),dtype=complex); big[:,rows] = wave.reshape(3,nr2)
        expanded[name] = big.reshape(-1)
    results = []
    Ebar = car.antiparticle_embedding(embedding)
    for charge,small in ((1,addition),(-1,removal)):
        large = base.space(charge,3)
        nc2,nc3 = len(small['physical']),len(large['physical'])
        kept = np.array([large['index'][(s,n)] for s,n,_ in small['physical']])
        outside = np.ones(nc3,dtype=bool); outside[kept] = False
        for packet_name,p in (('ORIGINAL_source',packet),('ORIGINAL_receiver',receiver)):
            if charge == 1:
                band = base.collective.native.source_coefficients(gamma,p)
                oldS = base.collective.source(neutral,small,band,embedding)
                bigS = base.collective.source(neutral3,large,band,embedding)
            else:
                band = car.negative_coefficients(gamma,p)
                oldS = car.annihilator(neutral,small,band,Ebar)
                bigS = car.annihilator(neutral3,large,band,Ebar)
            for seed,bigwave in expanded.items():
                wave = Omega if seed == 'fixed_actual_Omega_cut2' else photon
                old = (oldS@wave).reshape(4,3,nc2)
                image = (bigS@bigwave).reshape(4,3,nc3)
                total = float(np.vdot(image,image).real)
                if total <= 0:
                    raise ArithmeticError('Complete source image is zero')
                tail = image[:,:,outside]
                loss = float(np.vdot(tail,tail).real)
                results.append(dict(active_number=4+charge,packet=packet_name,seed=seed,
                    complete_cut3_image_norm_squared=total,raw_source_boundary_loss_norm_squared=loss,
                    raw_source_boundary_loss_fraction=loss/total,
                    retained_cut2_operator_image_absolute_defect=float(np.linalg.norm(image[:,:,kept]-old))))
    return dict(rows=results,contract='SPECIFIED fixed retained compact-electric-cut2 Omega/photon vectors, NOT cut3 quantum states or Hermite grades. Source band coefficients and E/Ebar are field-independent MEAN constants. Literal Wilson characters shift each integer edge by atmost1, so compact-electric-cut3 contains THIS exact retained-vector/source image. Nonpolynomial SF/normalheat already folded into the retained seed and charged halfstep is afterward. No bound on unprojected SF tails, subsequent SF/time evolution, continuumsource or Omega∞; raw image loss is before charged halfstep, not Knull or K/energy convergence.')


def source_connected_probe(c: base.Spectrum, physical: dict[str,Any], state: dict[str,Any],
                           lambdaN: float, post_source_pulse: float) -> dict[str,Any]:
    """ALL columns/Ref, same external fixed-frame probe as actual cut1 proof."""
    direction = np.array([.1,.2,.3,.4])
    duration, epsilon = .5, .005
    energy = -2*np.log(c.q/math.sqrt(lambdaN))
    field = np.tile(np.array([E for _,_,E in physical['physical']])@direction,3)
    M0 = float(np.vdot(state['Xi'],state['Xi']).real)
    if M0 <= 0:
        raise ArithmeticError('Actual source-connected preparation vanished')
    original = state['sourcewave'].T/math.sqrt(M0)
    pulsed = np.exp(-1j*post_source_pulse*field)[:,None]*original
    sources = np.concatenate((original,pulsed),axis=1)
    X = contact.project(c,sources)
    EX = contact.project(c,field[:,None]*sources)
    E2X = contact.project(c,field[:,None]**2*sources)
    thermal = np.exp(-duration*energy)[:,None]
    def group(values: np.ndarray) -> np.ndarray:
        return np.array([np.sum(values[:4]),np.sum(values[4:])])
    G0 = group(np.real(np.sum(X.conj()*thermal*X,axis=0)))
    G1 = group(np.real(1j*np.sum(EX.conj()*thermal*X-X.conj()*thermal*EX,axis=0)))
    G2 = group(np.real(np.sum(-E2X.conj()*thermal*X+2*EX.conj()*thermal*EX-X.conj()*thermal*E2X,axis=0)))
    first,disconnected = G1/G0,(G1/G0)**2
    connected = G2/G0-disconnected
    plus = group(contact.exact_pulse(c,energy,field,sources,duration,epsilon))
    minus = group(contact.exact_pulse(c,energy,field,sources,duration,-epsilon))
    nonlinear = (np.log(plus)-2*np.log(G0)+np.log(minus))/epsilon**2
    jointly_transformed = group(contact.exact_pulse(c,energy,field,sources,duration,.2,True))
    return dict(duration_positive_imaginary_time=duration,direction_ALL4edges=direction.tolist(),
        separate_post_source_physical_pulse=post_source_pulse,ALL_charged_columns=len(c.q),ALL_Ref_channels=4,
        source_survival=G0.tolist(),LOG_first_derivative=first.tolist(),
        LOG_connected_second_derivative=connected.tolist(),actual_disconnected_subtraction=disconnected.tolist(),
        exact_nonlinear_LOG_pulse_finite_difference=nonlinear.tolist(),finite_difference_step=epsilon,
        nonlinear_LOG_vs_analytic_absolute_defect=abs(nonlinear-connected).tolist(),
        actual_joint_source_and_H_basischange_invariance_absolute_defect=abs(jointly_transformed-G0).tolist(),
        contract='SAME frozen cut1 fixedphysicalEframe externalpulse law/current/contact, NEW actualcut2 quantumsource/kernel. Positiveimaginarytime source survival, notoldreal-time receiver or on-shellCompton. Exact analytic connected response and nonlinear pulse computed afresh on ALLcolumns. Independent Frechet current+contact decomposition physically verified atcut1 ONLY; no claim rerunhere. Original source untouched; separate postsourcepulse is declared externaldrive, minus charge uses opposite pulse for actualC control.')


def compare(reference: dict[str,Any], actual: dict[str,Any]) -> dict[str,Any]:
    quantities = ('actual_photon_Zph','actual_photon_energy_mean','actual_M0','actual_E1','actual_R1',
                  'actual_energy_mean','actual_transfer_Rayleigh_mean')
    result: dict[str,Any] = {key:dict(reference=reference[key],actual=actual[key],
            absolute_difference=abs(actual[key]-reference[key]),
            relative_difference=(actual[key]-reference[key])/reference[key]) for key in quantities}
    result['same_literal_coherent_receiver_probabilities'] = [dict(time=b['time'],
            reference=a['coherent_Ref4_receiver_probability'],actual=b['coherent_Ref4_receiver_probability'],
            absolute_difference=abs(a['coherent_Ref4_receiver_probability']-b['coherent_Ref4_receiver_probability']))
            for a,b in zip(reference['actual_same_literal_source_consumer_rows'],actual['actual_same_literal_source_consumer_rows'])]
    return result


def run(neutral_path: Path, charged_path: Path, car_path: Path) -> dict[str,Any]:
    for flag,path in (('--neutral-input',neutral_path),('--charged-input',charged_path),('--car-proof',car_path)):
        if path.read_bytes() != ARGUMENT_BYTES_AT_IMPORT[flag]:
            raise RuntimeError(f'Quantum input changed after IMPORT: {flag}')
    if hashlib.sha256(ARGUMENT_BYTES_AT_IMPORT['--car-proof']).hexdigest() != '517a539a51759fe1cb5e45668b3c760bbcf1201abfc83ea2cbf4884748900acf':
        raise RuntimeError('Actual frozen ALL20CAR source/conjugation proof input changed')
    n,ncontrols = neutral_input.load_spectrum(neutral_path)
    plus,ccontrols = charged_input.load_spectrum(charged_path)
    gamma,embedding,nodes,heat,Hphi,metric = base.left.frame()
    neutral,addition,removal = base.space(0,2),base.space(1,2),base.space(-1,2)
    if (len(n.q),len(plus.q)) != (3*len(neutral['physical']),3*len(addition['physical'])):
        raise ArithmeticError('Generated quantum carrier omitted a canonical direction')
    Cbody = car.physical_conjugation(removal,addition,False)
    Cquartet = car.physical_conjugation(removal,addition,True)
    Cn = car.physical_conjugation(neutral,neutral,False)
    minus = base.Spectrum([dict(V=b['V'],q=b['q'],P=Cbody.T@b['P'],sign=b['sign']) for b in plus.blocks],
                          3*len(removal['physical']))
    packet = np.array([.1,.8,.4,.1]); packet /= np.linalg.norm(packet)
    receiver = np.array([.5,.4,.1,.6]); receiver /= np.linalg.norm(receiver)
    Ebar = car.antiparticle_embedding(embedding)
    Sa = base.collective.source(neutral,addition,base.collective.native.source_coefficients(gamma,packet),embedding)
    Sa_out = base.collective.source(neutral,addition,base.collective.native.source_coefficients(gamma,receiver),embedding)
    Sh = car.annihilator(neutral,removal,car.negative_coefficients(gamma,packet),Ebar)
    Sh_out = car.annihilator(neutral,removal,car.negative_coefficients(gamma,receiver),Ebar)
    source_operator = Cquartet@Sh@Cn.T + Sa
    receiver_operator = Cquartet@Sh_out@Cn.T + Sa_out
    print(json.dumps(dict(stage='actual_cut2_ALL36_photon_source_removal_current_time',
                          complete_ranks=[len(n.q),len(plus.q),len(minus.q)])),flush=True)
    particle = base.observe(n,plus,neutral,addition,Sa,Sa_out,nodes,Hphi)
    hole = base.observe(n,minus,neutral,removal,Sh,Sh_out,nodes,Hphi)
    Omega,photon,Z = current.photon_state(n,neutral)
    lambdaN = float(n.q[0]**2)
    jplus,pstate = current.current_case(plus,addition,Sa,Sa_out,Omega,photon,lambdaN)
    jminus,hstate = current.current_case(minus,removal,Sh,Sh_out,Omega,photon,lambdaN)
    plus_contact = source_connected_probe(plus,addition,pstate,lambdaN,.17)
    minus_contact = source_connected_probe(minus,removal,hstate,lambdaN,-.17)
    old_contact = json.loads(FROZEN_BYTES_AT_IMPORT['N2_cut1_contact'])
    connected_comparison = dict(
        cut1_ORIGINAL_and_separate_pulse_LOG_connected_second_derivative=old_contact['exact_LOG_connected_second_derivative'],
        cut2_ORIGINAL_and_separate_pulse_LOG_connected_second_derivative=plus_contact['LOG_connected_second_derivative'],
        absolute_difference=abs(np.array(plus_contact['LOG_connected_second_derivative'])-np.array(old_contact['exact_LOG_connected_second_derivative'])).tolist(),
        proper_Ref0_to1_C_odd_LOG_first_derivative_absolute_defect=abs(np.array(plus_contact['LOG_first_derivative'])+np.array(minus_contact['LOG_first_derivative'])).tolist(),
        proper_Ref0_to1_C_even_LOG_second_derivative_absolute_defect=abs(np.array(plus_contact['LOG_connected_second_derivative'])-np.array(minus_contact['LOG_connected_second_derivative'])).tolist())
    eta = complex(np.vdot(Omega,Cn@Omega))
    mapped = (Cquartet@hstate['sourcewave'].reshape(-1)).reshape(4,plus.dimension)
    mapped_coeff = np.array([plus.project(w) for w in mapped])
    source_C = float(np.linalg.norm(mapped_coeff-eta*pstate['Xi'])/math.sqrt(particle['actual_M0']))
    current_C = []
    for p,h in zip(pstate['currents'],hstate['currents']):
        wave = np.array([minus.wave(w) for w in h])
        transformed = (Cquartet@wave.reshape(-1)).reshape(4,plus.dimension)
        amplitudes = np.array([plus.project(w) for w in transformed])
        current_C.append(float(np.linalg.norm(amplitudes+eta*p)/math.sqrt(particle['actual_M0'])))
    photon_coeff = n.project(photon)
    vacuum = np.zeros(len(n.q)); vacuum[0] = 1.
    symmetry = [dict(copy_S2_sign=b['sign'],neutral_charge_C_sign=b['charge_C_sign'],
                    ALL_original_sigma_in_block=len(b['q']),
                    actual_vacuum_weight=float(np.sum(abs(vacuum[b['global_indices']])**2)),
                    actual_photon_weight=float(np.sum(abs(photon_coeff[b['global_indices']])**2))) for b in n.blocks]
    images = complete_source_images(gamma,embedding,neutral,addition,removal,Omega,photon,packet,receiver)
    N1 = json.loads(FROZEN_BYTES_AT_IMPORT['N1_cut2'])['actual_primitive_electric_heat_transfer']
    N2old = json.loads(FROZEN_BYTES_AT_IMPORT['N2_cut1'])['actual_primitive_electric_ALL36_transfer']
    descriptor = dict(plan_id=PLAN.stem,raw_plan_sha256_at_IMPORT=hashlib.sha256(PLAN_BYTES_AT_IMPORT).hexdigest(),
        quantum_neutral_input_sha256=hashlib.sha256(ARGUMENT_BYTES_AT_IMPORT['--neutral-input']).hexdigest(),
        quantum_charged_input_sha256=hashlib.sha256(ARGUMENT_BYTES_AT_IMPORT['--charged-input']).hexdigest(),
        actual_CAR_proof_input_sha256=hashlib.sha256(ARGUMENT_BYTES_AT_IMPORT['--car-proof']).hexdigest(),
        frozen_comparison_input_sha256={name:hashlib.sha256(data).hexdigest() for name,data in FROZEN_BYTES_AT_IMPORT.items()},
        active_modes=8,active_particle_numbers=[3,4,5],active_CAR_states=[56,70,56],
        auxiliary_copies=2,auxiliary_modes_each=4,auxiliary_particle_number_each=2,ALL_auxiliary_states=36,
        dynamic_charged_flavours_including_tagged_active=3,ONE_global_Gauss_after_ALL36=True,
        exact_neutral_symmetry='BOTH S2copy± × BOTHchargeC±',exact_charged_symmetry='BOTH S2copy±, no invalid N5-only C reduction',
        primitive_electric_input=1.,NEW_ALL4edge_electric_cutoff=2,actual_field_grid=32,
        F='PhiheatHalf SF_fullcanonical exp[-sum_ALL4edges E²/2], K=FFdag',
        no_bare_B_squared=True,no_scalar_sea_substitute=True,no_original_sigma_clipping_or_pruning=True,
        kappa_active=.09,kappa_aux_each=.04,Phi3='threelevelsONEcoord; oldconditionalκ400 mean/metric heldfixed, notselfconsistentnewlaw',
        original_Ref4_source_preserved=True,removal_reference='SAME4RefCAR NEWRef0/background0→Ref1, r†W_inverse a_negative Ebar=E*σx',
        neutral_reference_background_cancellation=True,common_actual_neutral_lambdaN=True,
        source_packet=packet.tolist(),receiver_packet=receiver.tolist(),Pe=[.25,2.,4.],times=[0.,.5,1.,2.],
        original_paths=[[0,0,0,0],[-1,0,0,0],[-1,-1,0,0],[0,0,0,1]],
        current='-i[H_transfer,Ephys_e], actualALL4edges including longitudinal densityconnection',
        connected_source_probe='SAME actual cut1 fixedEframe Hpulse/current/contact law atNEWcut2, allcolumns/4Ref, d=[.1,.2,.3,.4], positiveimaginarytime.5; separatepostsourcepulse.17, oppositeforN3; NOTreal-time coherentreceiver/on-shell',
        complete_source_image_cutoff=3,source_image_role='SPECIFIEDretainedcompact-E-cut2seeds/constantmean-source/literalintegerWilsonimage ONLY, notHermitegrade, subsequentSF/time orcontinuumtail',
        source_coefficients_and_E_embedding_field_independent_MEAN=True,
        nonpolynomial_SF_role='Already foldedinto specifiedretainedseed, chargedwholehalfstepAFTERliteral source; unprojectedSFtail NOTcertified bysourceimage',
        scope='Native compactonecycle/all36/electricfinitecut/Phi3/selectedCliff2; NOT originalrealEM/fullCartesianPhi/SU4/largeN/massless/commoncone/mass/dispersion/CARanticommutator admission')
    return dict(input_descriptor=descriptor,input_descriptor_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
        source_at_import={name:hashlib.sha256(data).hexdigest() for name,data in SOURCE_BYTES_AT_IMPORT.items()},
        source_bytes_at_import={name:data.decode() for name,data in SOURCE_BYTES_AT_IMPORT.items()},
        plan_bytes_at_import=PLAN_BYTES_AT_IMPORT.decode(),physical_mean_descriptor=metric,
        runtime_packages=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),
        actual_generated_neutral_controls=ncontrols,actual_generated_charged_controls=ccontrols,
        actual_complete_neutral_quantum_symmetry_loads=symmetry,
        actual_particle_photon_source_and_time_receiver=particle,actual_Ref0_to1_removal_photon_source_and_time_receiver=hole,
        actual_particle_full_field_current=jplus,actual_removal_full_field_current=jminus,
        actual_particle_connected_fixed_frame_source_response=plus_contact,
        actual_proper_Ref0_to1_connected_fixed_frame_source_response=minus_contact,
        actual_N2_cut1_to_cut2_connected_source_response_comparison=connected_comparison,
        actual_ALLneutral_columns_source_operator_C_relative_defect=float(np.linalg.norm(source_operator.data)/np.linalg.norm(Sa.data)),
        actual_ALLneutral_columns_receiver_operator_C_relative_defect=float(np.linalg.norm(receiver_operator.data)/np.linalg.norm(Sa_out.data)),
        actual_neutral_C_overlap=current.complex_array(np.array(eta)),
        actual_neutral_C_orthogonal_residual=float(np.linalg.norm(Cn@Omega-eta*Omega)),
        actual_C_even_source_relative_defect=source_C,actual_ALL4edge_C_odd_current_relative_defects=current_C,
        actual_fixed_seed_complete_source_image=images,
        actual_N2_cut1_to_cut2_source_and_time_comparison=compare(N2old,particle),
        actual_same_cut2_N1_vs_N2_source_and_time_comparison=compare(N1,particle),
        actual_same_neutral_photon_Zph=Z,originalrealEM_massless_commoncone_macro_admitted=False)


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
    print(json.dumps(dict(source_at_import=result['source_at_import'],input_descriptor_sha256=result['input_descriptor_sha256'],
        runtime_seconds=result['runtime_seconds'],
        actual_particle={key:result['actual_particle_photon_source_and_time_receiver'][key] for key in ('actual_photon_energy_mean','actual_M0','actual_E1','actual_R1','actual_energy_mean')},
        actual_removal={key:result['actual_Ref0_to1_removal_photon_source_and_time_receiver'][key] for key in ('actual_M0','actual_E1','actual_R1','actual_energy_mean')},
        actual_C_even_source_relative_defect=result['actual_C_even_source_relative_defect'],
        actual_ALL4edge_C_odd_current_relative_defects=result['actual_ALL4edge_C_odd_current_relative_defects'])),flush=True)


if __name__ == '__main__':
    main()
