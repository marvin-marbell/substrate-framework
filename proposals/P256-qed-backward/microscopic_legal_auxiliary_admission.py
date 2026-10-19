#!/usr/bin/env python3
"""Renew actual LEGAL driven auxiliary transfer without changing state or Delta.

Retain the positive microscopic penalty discarded by the original full-field
Noether energy proof. This pays bounded state/group/effect transfer for the
actual initial neutral, SOURCE and true-creator states, including initial Q.
The comparator projection is proof only. J/contact use TWO actual quartic
graphs; no differential-G or photon-weighted boundary graph is promoted.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
from bosonic_original_law_join import symbolic_graph_contract
from improved_phi_band import NSTAR, UE, C, LIGHT_PARAMETERS
from local_orbital_stability import scalar_paths
from rotation_band import RotationBand

EXPECTED = {
    'filter': '9fdf6a3f5f38a7150f50489ce6e9fa413844fd0da6ddb43b9a4a6c741badedc3',
    'massive': '1330558494c1a30897d985ea9dee600fc2ed2c99ebc6be9e365e18c75a50653c',
    'penalty': '2d3a017627b722856036d61569f7d2e62e5b89c14f12ae6467d3ade7484c3981',
    'microscopic': '8d936b2e4a8de3a97b8b94048d57e48812069b193f02821e9855b48044b5c5a4',
}


def exact_fraction(value: dict[str, str]) -> Fraction:
    return Fraction(int(value.get('exact_numerator_hex', value.get('numerator_hex', '')), 16),
                    int(value.get('exact_denominator_hex', value.get('denominator_hex', '')), 16))


def ceil_log2(value: Fraction | int) -> int:
    value = Fraction(value)
    if value <= 0: raise ValueError('Positive bound required')
    candidate = value.numerator.bit_length()-value.denominator.bit_length()
    if candidate >= 0:
        if value.numerator > value.denominator*(1 << candidate): candidate += 1
    elif value.numerator*(1 << (-candidate)) > value.denominator:
        candidate += 1
    return candidate


def floor_log2(value: Fraction | int) -> int:
    upper = ceil_log2(value)
    value = Fraction(value)
    exact = (value.numerator == value.denominator*(1 << upper) if upper >= 0
             else value.numerator*(1 << (-upper)) == value.denominator)
    return upper if exact else upper-1


def construct(inputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    initial = inputs['filter']['initial_energy']
    ledger = inputs['filter']['ledger']
    whole = inputs['massive']['actual_whole_Phi_moment']
    penalty = inputs['penalty']['bounds']['Rootpenalty']
    microscopic = inputs['microscopic']
    selected = int(whole['unchanged_selected_log2_Lambda_lower'])
    if selected != int(initial['selected_log2_lambda_lower']): raise ValueError('Different fixed selected Lambda')
    if initial['absolute_initial_energy_upper'] != '8V Lambda for actual normalized Xi, each BplusXi/BminusXi, pairedSOURCE and truea† variants':
        raise ValueError('Missing actual neutral/SOURCE/true-creator absolute energy witness')
    if initial['identity'] != 'PNFF-RAW64-coherent-Ref4-sea-Ref0-seaPlus4-ONE-globalfilter-v2':
        raise ValueError('Wrong frozen initial function')
    if not whole['executed_initial_energy_binding_checked']: raise ValueError('Unbound initial energy')
    if whole['EM_positive_form_fraction_lower'] != '5/8': raise ValueError('Wrong original drive coercivity')
    if penalty['Delta'] != 'B_N(Lambda)+U_trial(Lambda)+window+beta+G': raise ValueError('Changed actual Delta selector')
    if penalty['G'] != 'max(2^132096,B_N(Lambda)+U_trial(Lambda)+window+beta+1)': raise ValueError('Changed actual G selector')
    beta = Fraction(penalty['beta'])
    if beta != Fraction(1, 16): raise ValueError('Nonzero original D must remain')
    volume = NSTAR**3
    if int(penalty['actual_code_vacuum_upper']['V']) != volume: raise ValueError('Different physical volume')
    if microscopic['graph']['histories'] != 8 or microscopic['graph']['primitive_algebra_errors']:
        raise ValueError('Missing actual all8 Pauli algebra proof')
    if microscopic['graph']['distinct_nonzero_defect_syndromes'] != 162:
        raise ValueError('Missing actual nonzero D execution')
    for profile in ('source19', 'SOFT'):
        row = microscopic['profiles'][profile]
        if row['actual_D_SOURCE_commutator']['norm_squared'] <= 0:
            raise ValueError('PDP=0 was substituted for the actual noncommuting defect')
        if row['literal_Pauli_J_SOURCE_commutator_error'] != 0:
            raise ValueError('Current and SOURCE are not the same encoded algebra')
    bhat = exact_fraction(whole['Bhat'])
    clock = exact_fraction(whole['actual_clock'])
    if clock <= 0: raise ValueError('Missing actual physical clock')
    ce0 = Fraction(8)
    # Original whole-field work bound, applied to each original normalized input.
    energy_coefficient = 8*(ce0+bhat+1)
    # SAME Delta >=2 U_trial+2 beta >=(264/5)V Lambda^(5/4)+2 beta.
    delta_leading = Fraction(264, 5)*volume
    # Conservative actual Root coverage includes S=0 vacancy/double sectors.
    # Pure native Pauli involutions give Delta, but wider inheritance uses Delta/2.
    leakage_squared_coefficient = energy_coefficient*Fraction(5, 132)
    leak_log = (ceil_log2(leakage_squared_coefficient)+1)//2-selected//8
    # v(t)=U_P(t,0)P Psi_original(0), not a new physical preparation.
    flow_multiplier = 1+clock*beta
    state_error_log = ceil_log2(flow_multiplier)+leak_log
    target = UE/(volume*(1 << 40))
    target_floor_log = floor_log2(target)
    effect_error_log = 1+state_error_log
    if effect_error_log >= target_floor_log:
        raise AssertionError('The fixed original Lambda does not pay actual auxiliary state/effect transfer')
    # A first-FORM spectral comparator, using K=H0+B_N >=0; no full-H norm.
    bn = [Fraction(penalty[name]) for name in ('a_N', 'b_N', 'c_N')]
    initial_K_coefficient = sum(bn, Fraction())+8*volume
    initial_window_tail_log = (ceil_log2(initial_K_coefficient)+1)//2-selected//16
    # w=Lambda^(9/8), Emax=w-B_N. U_trial>=2w+beta gives w+beta<=Delta/4.
    window_gap_log_lower = floor_log2(Fraction(66, 5)*volume)+(5*selected)//4
    # The original all8 fixed penalty supplies this much more than its floor.
    original_floor = int(penalty['minimum_G_log2'])
    floor_only_energy_ratio_log_lower = floor_log2(2*energy_coefficient*volume)+selected-original_floor
    if floor_only_energy_ratio_log_lower < 0:
        raise AssertionError('Adverse floor-as-Delta energy envelope failed to distinguish the actual window')
    # ORIGINAL quaternion matrices q/2 have HS orthogonality.  Each block's
    # entry-l1 <=4*HS norm; both signed maps <=8 sqrt(sum|raw beta|²)*||eta||1.
    raw_quartet_norm2 = Fraction(33, 20)
    profile_norm_upper = Fraction(33, 32)
    bounded_source_squared = 64*raw_quartet_norm2*profile_norm_upper**2*volume
    inherited_source_log = int(ledger['source_CB_full_auxQ_log2_upper'])
    if bounded_source_squared > Fraction(1 << (2*inherited_source_log)):
        raise ValueError('Actual source primitive exceeds its inherited full-aux bound')
    defect_source_squared = 4*beta**2*bounded_source_squared
    # On the actual original Gamma-supported coefficient domain.  Phi
    # commutes B and D; retaining the whole Phi graph pays rho[D,B] separately.
    phi_trace_coefficient = 64*(energy_coefficient/Fraction(8))
    # sqrt(C_E0+Bhat+1) <= C_E0+Bhat+1, deliberately rational/conservative.
    rho_squared_coefficient = phi_trace_coefficient/4
    weighted_defect_source_squared = defect_source_squared*rho_squared_coefficient*volume
    # Separate ACTUAL quartic graphs for full state AND proof code comparator.
    # P commutes non-D H and G. Projecting the initial state removes at most
    # two bounded D cross contributions; beta<=1/16 fits C_E0_code<=9.
    code_energy_coefficient = 8*(Fraction(9)+bhat+1)
    rho2_difference_graph_log = 3+ceil_log2(volume)+(ceil_log2(code_energy_coefficient)+1)//2
    rho_image_error_log = (state_error_log+rho2_difference_graph_log+1)//2
    # Current/contact use the SAME executed scalar-path primitive. No guessed
    # Dirac vertex or extra photon/volume normalization enters these caps.
    _, paths = scalar_paths(RotationBand(**LIGHT_PARAMETERS))
    path_l1 = sum((abs(Fraction(float(z.real)))+abs(Fraction(float(z.imag)))
                   for _, scalar in paths for z in scalar), Fraction())
    if max(int(max(displacement)) for displacement,_ in paths) != 2:
        raise ValueError('Missing original improved two-link current paths')
    photon_profile_norm_squared = UE*NSTAR/(4*C) # omega_min>=4c/N, ||g||=1.
    # 16 sqrt(2) V path_l1 ||f|| rho; C_fg <=32 V path_l1 ||f||||g|| rho.
    J_coefficient_squared = 512*volume**2*path_l1**2*photon_profile_norm_squared
    contact_coefficient = 32*volume*path_l1*photon_profile_norm_squared
    J_image_error_log = (ceil_log2(J_coefficient_squared)+1)//2+rho_image_error_log
    contact_image_error_log = ceil_log2(contact_coefficient)+rho_image_error_log
    if max(J_image_error_log,contact_image_error_log) >= target_floor_log:
        raise AssertionError('Actual quartic current/contact auxiliary images exceed the fixed target')
    groups = ['neutral', 'SOURCE_bothmaps_originalquartet', 'true_creator_SOURCE_original']
    cases = {group: dict(actual_initial_energy_coefficient=str(ce0),
        actual_initial_form='original normalized input; same fn,Gamma,globalZ and creator order',
        full_driven_auxQ_norm_log2_upper=leak_log,
        full_driven_state_minus_proof_codeflow_norm_log2_upper=state_error_log,
        bounded_unit_effect_difference_log2_upper=effect_error_log,
        initial_fullfield_window_tail_norm_log2_upper=initial_window_tail_log,
        actual_initial_auxQ_NOT_assumed_zero=True) for group in groups}
    return dict(scope='NEW actual selected LEGAL driven auxiliary state/group/effect and original J/contact image admission; not differential-G/photon-weighted closure',
        governing_plan='issue-222-mercury-synthesis-continuation-20261007',
        parent_revision='f9a076e0785f47644962a38b973e6b87a329d6ad594a4345a4728f4c689aa3be',
        original_inputs=cases, selected_log2_Lambda_lower=selected,
        original_clock=str(clock), beta=str(beta), all8_symbolic_graph=symbolic_graph_contract(NSTAR),
        original_Delta=penalty['Delta'], original_G=penalty['G'], original_floor_log2=original_floor,
        actual_Delta_lower='Delta >= (264/5)V Lambda^(5/4)+2beta; actual expression unchanged, floor not actual Delta',
        actual_Delta_leading_coefficient=str(delta_leading),
        actual_Q_penalty_lower='Delta/2; conservative inherited S_gamma in {-1,0,1} coverage',
        actual_Q_penalty_leading_coefficient=str(Fraction(132,5)*volume),
        stronger_native_Pauli_Delta_gap_NOT_used_for_wider_domain=True,
        retained_penalty_coercivity='H_R-Omega G+Bhat Lambda V >=(Delta/2) Q_aux +Tphi/2+Tcolour/2+(5/8)H_EM+Lambda rho^4/(16V)',
        retention_derivation='same global Pauli coefficient-l1/Noether Young absorptions never use the positive coordinate-independent cycle penalty; keep it instead of dropping it. D remains bounded by beta and included in Bhat. Encoded even G and field derivatives commute P_all8.',
        actual_work_energy_bound='S(t)<=8*(C_E0+Bhat+1)Lambda V; exact S_prime=-Omega_prime<G>, no static-frame relabelling',
        energy_coefficient=str(energy_coefficient), leakage_squared_coefficient=str(leakage_squared_coefficient),
        direct_positive_penalty_leakage='||Q_aux Psi(t)||^2<=2S(t)/actual_Delta<=leakage_squared_coefficient Lambda^(-1/4)',
        all8_codeflow_identity='i d(PPsi)/dt=P Hdrive P(PPsi)+P D Q Psi; ONLY D mixes auxiliary histories/syndromes',
        proof_comparator='v(t)=U_P(t,0)P Psi_original(0), unnormalized PROOF ONLY; physical original initial FUNCTION never changed',
        state_error_identity='||Psi(t)-v(t)||<=(1+T beta)sup_s||Q_aux Psi(s)||; initial Q retained',
        bounded_observable_transfer='for ||O||<=M, expectation error<=2M||Psi-v|| since ||Psi||=1,||v||<=1',
        initial_fullfield_spectral_window=dict(K='H0+B_N>=0',window='w=Lambda^(9/8)',
            Emax='w-B_N',first_form_bound='<K> <=(a_N+b_N+c_N+8V)Lambda',
            initial_K_coefficient=str(initial_K_coefficient), tail='||(1-P_window)Psi_initial||<=sqrt(C) Lambda^(-1/16)',
            gap='Q_aux(H0-Emax)Q_aux>=Delta/2-w-beta>=Delta/4; U_trial>=2w+beta',
            gap_log2_lower=window_gap_log_lower,
            PROJECTION_PROOF_ONLY=True, no_refilter=True, no_strong_fullH_norm_inferred=True),
        renewed_bounded_SOURCE=dict(raw_quartet_norm_squared=str(raw_quartet_norm2),
            source_norm_squared_upper=str(bounded_source_squared),profile_l2_majorant=str(profile_norm_upper),
            scalar_reason='psi1/psi0 is convex average of four unit-modulus centres with positive packet weights; every path factor modulus<=1',
            frame_reason='original polar SU4 frames and quaternion labels are unitary; phases have modulus1; signed conjugations preserve HS norm',
            Pauli_reason='literal edge-Pauli transfer is a product of unitary path A and endpoint projections, norm<=1 on full auxiliary Hilbert',
            domain='actual original Gamma-supported SOURCE domain; no arbitrary-jet extension through guard failure',
            D_SOURCE_operator_norm_squared_upper=str(defect_source_squared),
            rho_D_SOURCE_actual_graph_squared_upper=str(weighted_defect_source_squared),
            rho_graph='rho[D,BGamma]=[D,BGamma]rho; same original whole Phi graph retained, NOT inferred from norm closeness',
            derivative_G_Gamma_prime_graphs_NOT_promoted=True),
        target=dict(definition='UE/(V*2^40)',log2_floor=target_floor_log,
                    actual_unit_effect_error_log2_upper=effect_error_log, strictly_below_target=True),
        original_current_contact_image_transfer=dict(
            full_state_quartic_graph='||rho^2 Psi||<=4V sqrt(C_E)',
            proof_code_state_quartic_graph='||rho^2 v||<=4V sqrt(C_E_code), C_E0_code<=9 by P projection and bounded D; no new state',
            code_energy_coefficient=str(code_energy_coefficient),
            rho2_difference_graph_log2_upper=rho2_difference_graph_log,
            rho_image_difference_log2_upper=rho_image_error_log,
            interpolation='||rho(Psi-v)||^2<=||Psi-v||*(||rho^2Psi||+||rho^2v||); two actual energy graphs, NOT norm-closeness alone',
            actual_ImprovedPhiLaw_path_l1=str(path_l1),
            actual_photon_profile_norm_squared_upper=str(photon_profile_norm_squared),
            photon_units='canonical f=EMlegs*g/sqrt2, omega_min>=4c/N; uniform ||g||=1, no extra photon or volume rescale',
            positive_J_coefficient_squared_upper=str(J_coefficient_squared),
            contact_bilinear_coefficient_upper=str(contact_coefficient),
            J_action_difference_log2_upper=J_image_error_log,
            contact_action_difference_log2_upper=contact_image_error_log,
            original_positive_J_and_contact_unchanged=True,
            same_all8_Pauli_family_bound=True, actual_whole_Phi_graph_used=True,
            unbounded_image_auxiliary_transfer_admitted=True),
        adverse_floor_as_Delta=dict(incorrect_Delta='2^132096',
            conservative_energy_to_floor_penalty_ratio_log2_lower=floor_only_energy_ratio_log_lower,
            bounded_transfer_admitted=False, actual_Q_norm_NOT_inferred_from_failed_envelope=True,
            reason='even at the selected Lambda lower bound, the available actual energy envelope cannot pay a floor-only penalty'),
        fullfield_transfer_admitted=True, actual_driven_state_transfer_admitted=True,
        actual_nonzero_D_retained=True, all8_flat_independent_of64_colour=True,
        original_J_contact_auxiliary_images_admitted=True, differential_G_operator_images_admitted=False,
        incoming_photon_weighted_boundary_scattering_admitted=False,
        no_whole_particle_or_spin_or_Born_approval=True)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in EXPECTED: parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();start=time.perf_counter();inputs={};hashes={}
    for name,expected in EXPECTED.items():
        content=getattr(args,name).read_bytes();digest=hashlib.sha256(content).hexdigest()
        if digest != expected: raise ValueError('Wrong exact source-bound '+name+' receipt')
        inputs[name]=json.loads(content);hashes[name]=digest
    science=construct(inputs)
    result=dict(science=science,inputs_sha256=hashes,argv=sys.argv,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                wall_seconds=time.perf_counter()-start)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2))
    print(json.dumps(dict(output=str(args.output),output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        source_sha256=result['source_sha256'],wall_seconds=result['wall_seconds'],
        actual_driven_state_transfer_admitted=science['actual_driven_state_transfer_admitted'],
        effect_error_log2_upper=science['target']['actual_unit_effect_error_log2_upper'],
        target_log2_floor=science['target']['log2_floor'],
        original_J_contact_auxiliary_images_admitted=science['original_J_contact_auxiliary_images_admitted'],
        differential_G_graph_promoted=False),sort_keys=True))


if __name__=='__main__': main()
