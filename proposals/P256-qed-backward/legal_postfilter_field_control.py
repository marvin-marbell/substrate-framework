#!/usr/bin/env python3
"""Consume SAME-law proof inputs for ACTUAL postfilter field control.

No Gaussian is assigned to the filtered state.  No astronomical matrix or
native clock is integrated.  The new result is a bounded whole-space tube and
capped union-distance estimate, plus a local magnetic-IMS mixed graph.  The
current/CCR initial-photon extension uses the original a^dagger Gamma B fn tau,
not a re-filtered or re-normalized substitute.  All supplier science is frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

from prepared_minset_normal_proof import hex_fraction_log2_upper
from raw_source_relative_noether_graph import runtime_binding

PLAN = dict(id='issue-222-mercury-synthesis-continuation-20261007',
            revision='f9a076e0785f47644962a38b973e6b87a329d6ad594a4345a4728f4c689aa3be',
            template_revision='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
            work_item='soft-spatial-closed-rotation',campaign_open=True)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ceil_log2_integer(value: int) -> int:
    if value <= 0:
        raise ValueError('Positive exact integer required')
    return (value-1).bit_length()


def fraction(value: dict[str,str]) -> Fraction:
    return Fraction(int(value.get('exact_numerator_hex',value.get('numerator_hex','')),16),
                    int(value.get('exact_denominator_hex',value.get('denominator_hex','')),16))


def sum_log_upper(logs: list[int]) -> int:
    """If terms<=2^logs, their sum<=2^returned exact integer."""
    if not logs:
        raise ValueError('At least one actual bound required')
    return max(logs)+ceil_log2_integer(len(logs))


def negative_power_log(coefficient: int,power: Fraction,lambda_lower: int) -> int:
    if power > 0:
        raise ValueError('A lower Lambda bound cannot evaluate a positive-power upper bound')
    drop=(-power.numerator*lambda_lower)//power.denominator
    return coefficient-drop


def energy_envelope(rows: list[dict[str,Any]],lambda_lower: int) -> dict[str,Any]:
    exponent=Fraction(3,8)
    terms=[]
    for row in rows:
        power=fraction(row['power'])
        if power>exponent:
            raise ValueError('Actual initial form has a larger Lambda power; do not hide it')
        terms.append(negative_power_log(int(row['log']),power-exponent,lambda_lower))
    return dict(Lambda_power='3/8',coefficient_log2_upper=sum_log_upper(terms),
                original_positive_form_rows=rows,normalization_cost_ALREADY_in_supplier_rows=True)


def initial_photon_fast_graph(filter_ledger: dict[str,Any],graph: dict[str,Any],
                              driver: dict[str,Any],normal: dict[str,Any],time_value: int) -> dict[str,Any]:
    """NEW strong fast graph from actual filter CURRENT pullthrough and CCR.

    Let x=Gamma B fn tau (original order), a tau=0. Kfast acts on Phi/colour,
    EM-neutral heavy, Root and local charge-preserving colour fibre only; EM and
    GorbLight are outside Kfast, hence [Kfast,a]=0. This is stronger than, and
    is not inferred merely from, [Kfast,H_EM]=0.

    a fn tau=O(Cf CL d/L); a h fn tau=O(Cf_uF CL d). Positivity/commutation of
    Kfast and H_EM bounds Kfast a fn tau by (Kfast+H_EM)a fn tau. Using
    Kfast+H_EM=h-hLight-HC+Cdrive Lambda^1/2 gives O(Cbar^2 Lambda^1/8).
    Gamma/SOURCE magnetic first/second derivatives retain the original
    Lambda^1/4 and Lambda^1/2 cutoff costs; they do not enlarge that power.
    CCR then gives ||Kfast a^dagger x||²=||Kfast x||²+||Kfast a x||².
    No photon-weighted evolution or boundary norm is asserted.
    """
    window=filter_ledger['Fourier_window']
    source_jets=filter_ledger['full_source_jets_raw_log2_upper']
    exact_cl=fraction(filter_ledger['CL_upper'])
    cl=hex_fraction_log2_upper(dict(numerator_hex=hex(exact_cl.numerator),
                                  denominator_hex=hex(exact_cl.denominator)))
    chart=hex_fraction_log2_upper(normal['quotient_chart_jet_upper'])
    gamma_first=int(filter_ledger['global_union']['cutoff_gradient_coefficient_log2'])
    gamma_second=int(filter_ledger['global_union']['cutoff_Laplacian_coefficient_log2'])
    costs=[int(graph['second_graph_norm_coefficient_log2']),
           int(driver['contracted_fast_constant_log2']),int(driver['driven_common_window_coefficient_log2']),
           int(window['individual_log2_ceil']['1'])+cl,
           int(window['weighted_uF_log2_ceil']['1'])+cl,
           int(source_jets[0])+5,int(source_jets[1])+chart+16,
           int(source_jets[2])+2*chart+32,gamma_first,gamma_second,
           ceil_log2_integer(int(normal['physical_normal_dimension']))+16,5]
    # Cbar covers the displayed certified products, source/gamma jets and
    # original inverse32.  d<=1 and omega_g<=16 are conservative SAME-law
    # estimates; no UE, clock, field precision or state is changed.
    cbar=max(costs)+64
    initial=5+5*cbar   # 17 Cbar^5 Lambda^1/2; log2(17)<5.
    peak_log=max(0,4-ceil_log2_integer(time_value)+1)
    propagated=2*max((initial+1)//2,int(driver['contracted_fast_constant_log2'])+peak_log+8)+4
    return dict(Cbar_log2_upper=cbar,certified_product_log2_costs=costs,
                initial_strong_Kfast_norm_Lambda_power='1/2',initial_strong_Kfast_norm_coefficient_log2_upper=initial,
                propagated_strong_Kfast_norm_coefficient_log2_upper=propagated,
                annihilation_fast_graph_Lambda_power='1/8',
                photon_creator='ORIGINAL a_g^dagger Gamma B fn tau, allcoexact g with ||g||=1; same SOURCE Z and all harmonic factors',
                exact_initial_CCR_identity='||Kfast a_g^dagger x||²=||Kfast x||²+||Kfast a_g x||²',
                original_CREATOR_norm_lower='||a_g^dagger x||>=||x||; observable division is not a preparation re-normalization',
                original_SOURCE_commutes_creator=True,
                fast_creator_commutator_derivation='literal Hfast/Phi/colour/EM-neutral heavy/allRoot beta and local colour charge algebra contain no coexact A/E; light/current/GorbLight excluded and retained as slow terms',
                actual_current_force='F_g=[a_g,h]-omega_g a_g; ORIGINAL J/contact pullthrough, not a Gaussian photon replacement',
                inherited_Root_SOURCE_penalty_commutation='SAME global-filter Root_requirement, transport primitive binding remains T-owned; NOT a new microscopic validation',
                NO_photon_weighted_boundary_or_scattering_graph_claim=True)


def uniform_controls(graph_log: int,kappa_log: int,r0_log: int,lambda_lower: int,
                     time_log: int,boundary_log: int,dimension: int,cometric_log: int) -> dict[str,Any]:
    """Observable bounds, not a declaration of a postfilter Gaussian covariance."""
    d2=negative_power_log(graph_log-kappa_log,Fraction(-1,2),lambda_lower)
    tail_d=negative_power_log(graph_log-kappa_log-2*r0_log,Fraction(-1,8),lambda_lower)
    delta=negative_power_log(time_log+boundary_log,Fraction(-1,8),lambda_lower)
    # For F=min(d_union²,r_tube²), ||F^1/2(Psi-Psi_D)||<=r_tube delta.
    # For projection Pout, triangle then (a+b)^2<=2a^2+2b^2.
    whole_tail=sum_log_upper([tail_d+1,2*delta+1])
    capped_error=negative_power_log(2*r0_log+2*(time_log+boundary_log)+1,Fraction(-5,8),lambda_lower)
    whole_capped=sum_log_upper([d2+1,capped_error])
    # eta_j=q_j theta(|q|/r_tube), summed over ALL D normals. Magnetic IMS:
    # ||q theta chiPboost Psi_D||² <=8 chi r_tube sqrt(<d²>)||KfastPsi_D||
    #                                     +C chi²(D+1).
    mixed_first=3+r0_log+(3*graph_log-kappa_log+1)//2
    mixed_second=12+cometric_log+ceil_log2_integer(dimension+256)
    mixed_square=sum_log_upper([negative_power_log(mixed_first,Fraction(-7,16),lambda_lower),
                                negative_power_log(mixed_second,Fraction(-1),lambda_lower)])
    return dict(Dirichlet_actual_union_distance_second_moment_log2_upper=d2,
                Dirichlet_actual_tube_exit_probability_log2_upper=tail_d,
                actual_whole_minus_D_norm_log2_upper=delta,
                WHOLE_actual_tube_exit_probability_log2_upper=whole_tail,
                WHOLE_actual_capped_union_distance_second_moment_log2_upper=whole_capped,
                DIRICHLET_inside_tube_q_chiPboost_squared_log2_upper=mixed_square,
                mixed_graph_squared_Lambda_powers=['-7/16','-1'],
                bare_chiPboost_squared_coefficient_log2_upper=graph_log+2,
                bare_chiPboost_squared_Lambda_power='0; adverse first-moment bound does NOT give -1/8',
                full_normal_dimension=dimension,probability_upper_is_for_ALL_dimensions=True,
                global_TrSigma_about_one_branch_NOT_asserted=True,
                unbounded_whole_field_moments_NOT_transferred_from_norm_error=True)


def construct(normal_receipt: dict[str,Any],filter_receipt: dict[str,Any],
              massive: dict[str,Any],bridge: dict[str,Any]) -> dict[str,Any]:
    normal=normal_receipt['legal_N'];ledger=filter_receipt['ledger'];energy=filter_receipt['initial_energy']
    graph=massive['actual_regular_union_unweighted_graph_bridge']
    driver=massive['normal_branch_comparator']
    selected=int(normal['selected_log2_lambda_conservative_lower'])
    if selected!=int(ledger['selected_log2_lambda_lower']) or selected!=int(graph['unchanged_selected_log2_Lambda_lower']):
        raise ValueError('Different selected finite Lambda suppliers cannot be joined')
    if filter_receipt['exit_code']!=0 or filter_receipt['natural_exit'] is not True:
        raise ValueError('Original physical postfilter state supplier did not naturally succeed')
    if graph['second_graph_norm_Lambda_exponent']!='1/2':
        raise ValueError('Do not silently replace the actual driven common Lambda^1/2 window')
    time_value=int(massive['selected_clock']);time_log=ceil_log2_integer(time_value)
    kappa_log=min(hex_fraction_log2_upper(normal['uniform_alpha']),hex_fraction_log2_upper(normal['all_q_stiff']))-8
    radius_log=hex_fraction_log2_upper(normal['quotient_radius'])-1
    r0_log=radius_log-6
    cometric_log=hex_fraction_log2_upper(normal['quotient_cometric_upper'])
    dimension=int(normal['physical_normal_dimension'])
    base_graph=int(graph['second_graph_norm_coefficient_log2'])
    boundary_log=int(graph['boundary_coefficient_log2'])
    initial={}
    for name,key in (('neutral','neutral_relative_energy_rows'),('SOURCE_bothmaps_originalquartet','source_relative_energy_rows'),
                     ('true_creator_SOURCE_original','true_creator_relative_energy_rows')):
        envelope=energy_envelope(energy[key],selected)
        ce=envelope['coefficient_log2_upper']
        initial[name]=dict(envelope,
            actual_postfilter_union_distance_second_moment_log2_upper=negative_power_log(ce-kappa_log,Fraction(-5,8),selected),
            actual_postfilter_inside_analysis_tube_exit_log2_upper=negative_power_log(ce-kappa_log-2*r0_log,Fraction(-1,4),selected),
            actual_initial_chiPboost_squared_log2_upper=negative_power_log(ce+2,Fraction(-1,8),selected))
    ordinary=uniform_controls(base_graph,kappa_log,r0_log,selected,time_log,boundary_log,dimension,cometric_log)
    photon=initial_photon_fast_graph(ledger,graph,driver,normal,time_value)
    photon_graph=int(photon['propagated_strong_Kfast_norm_coefficient_log2_upper'])
    photon_shell=(photon_graph-kappa_log-2*radius_log+3)//2+4
    photon_boundary=(photon_graph+photon_shell+3)//2-2*radius_log+32
    photon['NEW_unweighted_shell_coefficient_log2_upper']=photon_shell
    photon['NEW_unweighted_boundary_coefficient_log2_upper']=photon_boundary
    photon_controls=uniform_controls(photon_graph,kappa_log,r0_log,selected,time_log,
                                    photon_boundary,dimension,cometric_log)
    # Any additional source/current Lipschitz costs must be the actual supplied
    # all-normal operator jets.  The original source first jet is source-bound,
    # not a field Hessian reinterpreted as a carrier mass.
    source_first=int(ledger['global_union']['source_first_log2'])
    source_inside_error=2*source_first+ordinary['WHOLE_actual_capped_union_distance_second_moment_log2_upper']
    source_inside_error_photon=2*source_first+photon_controls['WHOLE_actual_capped_union_distance_second_moment_log2_upper']
    original_source_norm_lower=fraction(ledger['source_relative_to_actual_Xi_norm_lower'])
    source_inverse_log=ceil_log2_integer(original_source_norm_lower.denominator)
    initial_source_mean_error=2*source_first+2*source_inverse_log+initial['neutral']['actual_postfilter_union_distance_second_moment_log2_upper']
    original_gamma_log=int(ledger['global_union']['r0_log2'])
    if original_gamma_log-r0_log>=selected//16:
        raise AssertionError('Original preparation Gamma is not inside the NEW analysis-only tube')
    target_probability_log=-40
    if max(ordinary['WHOLE_actual_tube_exit_probability_log2_upper'],photon_controls['WHOLE_actual_tube_exit_probability_log2_upper'])>=target_probability_log:
        raise AssertionError('Selected SAME finite Lambda does not pay the new actual-state tube estimate')
    # A genuine adverse branch: a merely norm-close raw Gaussian does NOT
    # transfer q² or qP.  Only the positive BOUNDED observable argument above
    # transfers; the mixed graph remains explicitly Dirichlet.
    return dict(scope='NEW SAME-LEGAL-law actual postfilter energy/source/current/CCR field-control construction; analytic finite bounds, NOT astronomical simulation',
        plan=PLAN,selected_log2_Lambda_lower=selected,clock=time_value,
        kappa_log2_lower=kappa_log,original_regular_radius_log2_lower=radius_log,
        analysis_ONLY_tube_radius='2^r0_log Lambda^(-3/16)',r0_log2=r0_log,
        physical_normal_dimension=dimension,original_centres=512,
        actual_original_state=dict(definition='Xi=Gamma fn(H_D-e_global) RAWtau / ORIGINALoneGLOBALnorm; SOURCE/truea† applied in ORIGINALorder',
            RAWtau=ledger['RAWtau'],fn_spectral_support=energy['form_proof'],original_e_global=ledger['actual_global_floor'],
            original_norm_bounds=energy['norm_inputs'],all64_alias_floors_and_harmonic55_kept=True,
            no_new_ground_no_recenter_no_sigma_retune_no_refilter=True),
        initial_mean_body_SOURCE_comparison=dict(
            original_Gamma_support='distance_union<=2^original_gamma_log Lambda^(-1/4); ORIGINAL physical preparation, not a new cutoff',
            original_Gamma_radius_coefficient_log2=original_gamma_log,
            initial_SOURCE_same_Z_error_squared_log2_upper=initial_source_mean_error,
            source_norm_lower_original=energy['norm_inputs']['each_SOURCE_relative'],
            proof='Gamma has support inside the ORIGINAL regular union, so ||(B(q)-B0)Xi||²<=M1²<d_union²>. Divide by SAME original SOURCE norm only; no globalnoncompactB bound needed and no B0 renormalization.',
            exact_unitarity_preserves_INITIAL_error=True,
            all4_bothmaps_ordinary_root_and_Ref_words_kept=True,
            source_mean_label='actual local mean/body word per ORIGINAL MinSet chart/history, no branch floor removal or imposed superselection',
            covariant_kinetic_intertwining_NOT_claimed=True,
            required_actual_mean_connection_check='actual mean A_N, g0_NG and covariant nabla B0 in the FULLFock parallel frame; transform original initial FUNCTION by Udagger exactly'),
        initial_actual_postfilter_controls=initial,
        full_pulse_neutral_and_SOURCE_actual_controls=ordinary,
        NEW_true_photon_initial_fast_graph=photon,
        full_pulse_true_photon_unweighted_actual_controls=photon_controls,
        source_field_join=dict(INSIDE_tube_SOURCE_deviation_squared_log2_upper=source_inside_error,
            INSIDE_tube_SOURCE_with_true_photon_squared_log2_upper=source_inside_error_photon,
            source_first_jet_log2_upper=source_first,
            OUTSIDE_source_weighted_tail_NOT_replaced_by_probability=True,
            operator_identity='Btilde_j=Wdagger B_j W=B_j(Q(t)), ALL4wordorder/bothmaps/sameinitialSOURCEZ; retain d_t Btilde and neutralRIGHT',
            physical_observable_transfer='Pout_tilde=Wdagger Pout W; Fcap_tilde=Wdagger min(d_union²,r_tube²)W; expectations EXACT on Wdagger actualXi',
            current_identity='Jtilde=Wdagger Joriginal W, contact likewise; W commutes actual coexact a_g† but not SOURCE B(q)',
            G_identity='Gtilde=Wdagger Gphysical W; legitimate RAWcolour-Haar halfdiv cancels as common SOURCE scalar, NOT Gauss residual or forcedzero',
            Href_relative_geometry='after an actually constructed fullFock parallel frame removes/retains the actual constant normal gauge connection, geometry DIFFERENCES use distance_union chiPboost; actual mean A_N/g0_NG/covariant nabla B0 remain separate obligations, NOT g0I/cross0 assumptions',
            bosonic_curvature_is_NOT_particle_mass=True,common_Maslov_is_NOT_physical_spinhalf=True,
            same_original_Q_light_Ref_commutes_field_reference=True,
            actual_Gauss_constraints_conjugated_NOT_assumed_commuting=True),
        derivation=dict(energy='kappa Lambda <d_union²> <=<Kfast> <=||KfastPsi_D|| <=2^G Lambda^(1/2)',
            tube='r_tube²=2^(2r0) Lambda^(-3/8); Markov gives Pout_D<=2^(G-kappa-2r0) Lambda^(-1/8)',
            WHOLE='||PoutPsi||<=sqrt(Pout_D)+delta; ||Fcap^(1/2)Psi||<=sqrt(<Fcap>_D)+r_tube delta. BOUNDED operators only.',
            mixed='magneticIMS eta=d_union theta(d_union/r_tube), a globally Lipschitz scalar on the ORIGINAL regular union; ||eta chiPboostPsi_D||²<=8chi r_tube sqrt(<d²>)||KfastPsi_D||+Cchi²(D+1), allD, no discontinuous nearest-centre vector',
            actual_initial='original positive postfilter energy rows <=Ce Lambda^(3/8), hence d²<=Ce/kappa Lambda^(-5/8) and chiP²<=4Ce Lambda^(-1/8) only atinitial',
            adverse='driven firstmoment alone gives chiPboost O(1), NOT Lambda^(-1/16); no ordinary Lambda^3/8 midclock inheritance',
            photon='current pullthrough gives a fn tau and a h fn tau; use Kfast+H_EM positivity then originalGamma/SOURCE commutator and exactCCR, not photon-weighted normtransfer'),
        admission=dict(normal_supplier_local_scope=normal['scope'],normal_supplier_global_field_coercivity_proved=normal['global_field_coercivity_proved'],
            used_actual_regular_union_Kfast_and_whole_D_BOUND_ONLY=True,
            quantum_law_and_selected_parameter_and_frozen_proof_inputs_CONDITIONAL=True,
            no_global_ground_Gausskernel_photonweighted_boundary_Born_or_full2pi_claim=True,
            no_new_microscopic_Root_source_penalty_validation=True),
        next_decisive_construction='Consume these ACTUALpostfilter tube/capped/mixed controls in fullSOURCE-relative H/G/J all4word/current orbit family; keep actual photon current/contact and ROOT/PH/outside-source graph. Remaining spatial2pi sign must come from actual Gphysical word orbit, NOT covariance/Maslov phase.',
        inherited_source_relative_bridge_scope=bridge['conditional_scope'])


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--normal',type=Path,required=True)
    parser.add_argument('--filter',type=Path,required=True)
    parser.add_argument('--massive',type=Path,required=True)
    parser.add_argument('--bridge',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runtime-archive',type=Path,required=True)
    args=parser.parse_args();started=time.perf_counter()
    expected=dict(normal='07180616d355175aee1a4ff23c9b9dc1a2dd55a072144ed6073f28fce8560d77',
                  filter='9fdf6a3f5f38a7150f50489ce6e9fa413844fd0da6ddb43b9a4a6c741badedc3',
                  massive='1330558494c1a30897d985ea9dee600fc2ed2c99ebc6be9e365e18c75a50653c',
                  bridge='008231eb67c9c6b1620fe3e110504234646b8a934bc6501f89872fedf34dfcd2')
    inputs={}
    for name,digest in expected.items():
        path=getattr(args,name)
        actual=sha(path)
        if actual!=digest:
            raise ValueError('Not the authoritative unchanged supplier: '+name)
        inputs[name]=json.loads(path.read_bytes())
    science=construct(inputs['normal'],inputs['filter'],inputs['massive'],inputs['bridge'])
    runtime=runtime_binding(args.runtime_archive)
    receipt=dict(science=science,source_sha256=sha(Path(__file__)),runtime_binding=runtime,
                 inputs_sha256={name:sha(getattr(args,name)) for name in expected},argv=sys.argv,
                 wall_seconds=time.perf_counter()-started,actual_python=sys.version.split()[0],
                 metadata_allowlist='explicit argv/source/proof inputs/Python version only, no environment or credentials')
    args.output.write_text(json.dumps(receipt,indent=2))
    print(json.dumps(dict(output=str(args.output),output_sha256=sha(args.output),source_sha256=receipt['source_sha256'],
        WHOLE_original_SOURCE_tube_exit_log2_upper=science['full_pulse_neutral_and_SOURCE_actual_controls']['WHOLE_actual_tube_exit_probability_log2_upper'],
        WHOLE_true_photon_tube_exit_log2_upper=science['full_pulse_true_photon_unweighted_actual_controls']['WHOLE_actual_tube_exit_probability_log2_upper'],
        actual_postfilter_Gaussian_assigned=False,full_physical_spatial2pi_claim=False,
        wall_seconds=receipt['wall_seconds']),indent=2))


if __name__=='__main__':
    main()
