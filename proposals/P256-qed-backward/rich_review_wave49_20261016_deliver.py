"""Package independent executed reviews with bounded claims and immutable attempts."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parent;P='rich_review_wave49_20261016_'


def load(name):return json.loads((ROOT/(P+name+'.json')).read_text())


def file_hash(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def canonical_control_labels(value):
    """Project historical raw keys into the corrected current verdict contract."""
    def label(text):
        return text.replace('whole_charge_motion_OFF','field_decoupled_hopping_disabled').replace('whole_OFF','field_decoupled')
    if isinstance(value,dict):return {label(k):canonical_control_labels(v) for k,v in value.items()}
    if isinstance(value,list):return [canonical_control_labels(v) for v in value]
    return value


def run():
    local=load('local_benchmark_result');primitives=load('local_primitives_result')
    spatial=load('spatial_benchmark_result');field=load('field_primitives_result')
    active=load('spatial_consumer_result');combined=load('integrated_consumer_result');language=load('language_final')
    debts=load('spatial_debts_result')
    paired=load('spatial_paired_result');integrated_paired=load('integrated_paired_result')
    concordance=load('spatial_concordance_result')
    projection=load('projection_result')
    cubature=load('cubature_result');quadrature=load('quadrature_api_result');gaunt=load('gaunt_result')
    qscale=load('spatial_qscale_result')
    pulse_audit=load('pulse_audit_result')
    exact_picture=load('exact_picture_result');exact_means=load('exact_picture_means_result')
    exits={p.name:json.loads(p.read_text()) for p in ROOT.glob(P+'*_exit.json')}
    verdict=dict(plan_id='issue222-rich-backward-candidates-20261014',work_item='independent-review',local=dict(verdict='BOUNDED_EXECUTED_LOCAL_NOETHER_AND_CHARGE_TRANSPORT',independent_benchmark=local,active_operator_audit=primitives,actual_same_H_consumer=combined),spatial=dict(verdict='BOUNDED_EXECUTED_FINITE_Q_QUANTUM_FIELD_SOURCE_RESPONSE',independent_canonical_benchmark=spatial,active_field_kernel_audit=field,actual_consumer=active),integrated=dict(verdict='OWN_FRESH_ACTUAL47_COMBINED_H_CONSUMER_NOT_A_PLOT_COLLAGE',result=combined),process_exits=exits,typing=dict(natural_exit=language['natural_exit'],summary=language['diagnostics'].get('summary')),not_claimed=['Continuum/bulk/on-shell photon','Finite cavity mode set globally SO3 closed','Lorentz/inertial mass from rest cost or finite gap','Electron localization/stability or unique species','CARG/topology/central sewing derived from classical Euler rather than explicit quantization','Finite chi/angular/radial/field convergence'])
    verdict['spatial']['own_state_canonical_debt']=debts
    verdict['spatial']['calibrated_field_feedback']=paired
    verdict['spatial']['current_apply_API_concordance']=concordance
    verdict['spatial']['fixed_basis_q1_q0_charged_neutral_response']=qscale
    verdict['integrated']['calibrated_field_to_local_carrier_feedback']=integrated_paired
    verdict['integrated']['construction_contraction_equivalence']=projection
    verdict['local']['physical_assertion_assessment']='PASSED_AT_REPORTED_DT_CHI_NOT_CONVERGENCE' if all(v for row in combined['physics_assertions'].values() for v in row.values()) else 'FAILED_OR_UNRESOLVED_PHYSICAL_ASSERTIONS'
    verdict['numerical_cubature_correction']=dict(independent_fixed_R2_comparison=cubature,shared_SO3_API=quadrature,source_relevant_Gaunt=gaunt,old_matrix_outputs=P+'old_material_cubature_outputs.zip')
    verdict['field_units']='Pi and length-gauge Pi+P are canonical kinetic field-momentum amplitudes, not SI electric-field units; uncut qdot divides by the declared cavity inertia. Historical benchmark physical_E keys use these momentum units.'
    verdict['counterfactual_control_semantics']=load('control_semantics')
    verdict['pulse_preparation_semantics']=load('pulse_semantics')
    verdict['spatial']['own_existing_ray_initial_preparation_vs_temporal_response']=pulse_audit
    verdict['exact_electric_picture']=dict(verdict='OWN_FULL_FREE_FIELD_EVOLUTION_AND_SOURCE_KERNEL_NOT_COUPLED_BODY_REVIEW',independent_Fourier_source_kernel=exact_picture,own_full_unbounded_free_state_integrals=exact_means)
    verdict['integrated']['cone_metric_semantics']='Declared r³ dr dg and C_g/(2Mr²), C_g=j(j+1), choose the cone over the SU2 metric dual to physical LEFT. This is not the standard flat quaternion/Frobenius cone, whose angular term has 4C_g; no flat-cone derivation or metric change is inferred.'
    verdict['integrated']['zero_source_API_failure_scope']=load('integrated_lazy1_assessment')
    verdict['integrated']['zero_source_repair_smoke']=load('integrated_repair_smoke_result')
    verdict['integrated']['native_zero_zipup_failure_scope']=load('integrated_repair_smoke2_assessment')
    verdict['integrated']['own_physical_stage_receipts']=[p.name for p in sorted(ROOT.glob(P+'zero_source_repair2_*_stage.json'))]
    (ROOT/(P+'verdict.json')).write_text(json.dumps(canonical_control_labels(verdict),indent=2)+'\n')
    lines=['INDEPENDENT REVIEW49 — bounded physical verdicts, not parent-goal closure.',
        'Inputs: shared issue222-rich-backward-candidates-20261014 / independent-review; exact source bytes per attempt capsule.',
        'No ancestor producer CLI, solved-state replay, public/plan/graph/world mutation, or nested worker.',
        'LOCAL LEFT / CHARGE BENCHMARK',
        'Own two-cell source-image Hamiltonian; all CAR8 bands/cell, not fullCAR32 acceptance.',
        f"LEFT source commutator {local['LEFT_source_commutator_error']:.3g}; central seam {local['central_seam_error']:.3g}.",
        f"Excess charge {local['initial']['charge_excess']} -> {local['final']['charge_excess']}.",
        f"Actual local Noether vector change {local['physical_Noether_vector_density_change']}.",
        f"Pinned-global vector-density wrong case error {local['wrong_global_vector_density_error']:.6g}; n/N not an excitation PVM.",
        f"Active47 full55g source sewing {primitives['local_central_sewing_error']:.3g}; exterior eigenvalue max {primitives['source_angular_exterior_eigenvalue_range'][1]:.6g}, not erased.",
        'The small R1 mechanical benchmark below is explicitly its recorded finite-cubature H, not corrected-model exterior/convergence evidence.',
        'FINITE-Q CANONICAL / FIELD BENCHMARK',
        f"Independent path/rotation/endpoint errors {spatial['kernel_errors']}.",
        f"Nonzero Wilson-loop norm {spatial['wrong_all_finite_q_is_pure_gauge_error']:.6g} rejects all-finite-q-pure-gauge case; q0 residual0.",
        f"Fixed-canonical ORIGINAL-Lambda current FD {spatial['fixed_canonical_current_fd_error']:.3g}; mixed contact FD {spatial['mixed_contact_fd_error']:.3g}.",
        f"Full-beforecut versus compressed-contact error {spatial['full_contact_exterior_debt']:.6g}; omitted connection feedback {spatial['source_feedback_state_difference']:.6g}.",
        f"Quantum eight-real-field kick work {spatial['kick_work']:.6g}; same-H energy drift {spatial['same_H_energy_drift']:.3g}.",
        f"Field boundary weight {spatial['field_boundary_shell_weight']:.6g}; outgoing field-H norm {spatial['field_exterior_H_norm']:.6g}; angular exterior unbounded.",
        f"Own Hermite quadrature field errors max {max(field['errors'].values()):.3g}; wrong cut exponential {field['wrong_exponential_after_projection_error']:.6g}, cut Pi² {field['wrong_kinetic_after_projection_error']:.6g}.",
        'ACTUAL48 FRESH CONSUMER: fullCAR8 N3/N4/N5; global g control, not local47.']
    lines.append(f"Current apply/source API concordance after computation-only changes: {concordance['records']}; physical evolution retains its exact executed source capsules.")
    for branch,r in active['records'].items():
        c=r['checks'];lines.extend([f"{branch}: dimension {r['dimension']}; Gauss {r['Gauss_error']}; all8 source strengths {r['source_strengths_all_bands']}.",f"  Physical current/contact/mixed FD relative {c['physical_current_fd_relative']:.3g}/{c['full_contact_fd_relative']:.3g}/{c['mixed_contact_fd_relative']:.3g}.",f"  HoppingOFF convection {c['hopping_OFF_convection_current_norm']:.6g}; field-decoupled/hopping-disabled owning EM source {c['whole_charge_motion_OFF_current_norm']}; wrong omitted R connection {c['wrong_omitted_rotational_connection_current_error']:.6g}.",f"  Fresh field-kick work {c['source_kick_work']:.6g}, same-H drift {c['same_H_energy_drift']:.3g}; two physical current time orders {c['physical_current_time_orders']}."])
        lines.append(f"  Finite-cut canonical debts (not exact Pi-dot=current): {debts['records'][branch]}.")
        for row in paired['records'][branch]:
            audit=next(a for a in pulse_audit['records'][branch] if a['incoming']==row['incoming'])
            lines.append(f"  Own paired {row['incoming']}->{row['outgoing']}: raw final field response {row['field_momentum_response']:.6g}, initial preparation shift {audit['initial_outgoing_preparation_shift']:.6g}; temporal full/decoupled {audit['full_temporal_response_difference']:.6g}/{audit['field_decoupled_temporal_response_difference']:.6g}, calibrated {row['calibrated_field_response']:.6g}; occupation-charge response {row['calibrated_charge_response']}.")
        for row in qscale['excess'][branch]:
            lines.append(f"  Own same8field q1-vs0 {row['incoming']}->{row['outgoing']} charged-minus-neutral/full-minusOFF: {row['charged_minus_neutral_full_minus_OFF']}; finite-q field difference {row['finite_q_minus_uniform_field']:.6g}, not an IR pole or gap.")
    lines.extend(['ACTUAL47 INTEGRATED CONSUMER: ONE H, own source states and same-H evolving neutral observer.',f"Chosen law parameters {combined['parameters']}."])
    lines.append(f"History: Gram regroup {projection['relative_error']:.3g}; two cancellations then SIGKILL-9 atfullMPO boundary,causeunconfirmed; lazy1 EXIT1 atempty-source builder after8h,no savedneutral/chargedassert. Nativezero zip_up also EXIT1; repaired builder/exactnaive0 action/existingSUM exp0/nonzero-path/own-stage smoke passed. API failures are not carrier failures.")
    lines.append(f"Independent fixed35R/9field cubature: old-vs24-grid norm {cubature['old_vs_high_norm']:.6g};24vs28 norm {cubature['high_vs_fine_norm']:.3g}; integration alias, NOT angular exterior.")
    lines.append(f"Shared integerSO3 Wigner/LEFT errors {quadrature['integer_R_F_value_error']:.3g}/{quadrature['physical_LEFT_vector_commutator_error']:.3g}; old uniform positive control {quadrature['old_uniform_q0_fullproduct_positive_control_error']:.3g}; SU2g4pi untouched.")
    lines.append(f"Own source-relevant Gaunt/direct errors {gaunt['errors']}; auxiliaryl4 expands retained bra/ket products, not extra/dropped evolved states.")
    lines.append(f"Frozen operator {combined['compiled_operator_input']['sha256']}; own directproduct charge+LEFT hop {combined['operator_primitive_action_probe']['sewn_charge_PLUS_LEFT_transfer_matrix_element']}, hoppingOFF0; actual scalar-phase control error {combined['operator_primitive_action_probe']['centering_phase_control']['analytic_error']:.3g}.")
    for branch,r in combined['records'].items():
        lines.extend([f"{branch}: all8 source strengths {r['source_strengths_all8']}.",f"  Local excess charge change {r['excess_charge_change']}; local LEFT-Casimir change {r['excess_LEFT_change']}.",f"  Actual Noether-vector density change {r['physical_Noether_vector_density_change']}.",f"  Field-decoupled/hopping-disabled owning EM source {r['whole_charge_motion_OFF_current']}; hoppingOFF convection {r['hopping_OFF_convection_current']:.6g}.",f"  Field kick work {r['kick_work']:.6g}; own-H energy drift {r['same_H_energy_drift']:.6g}; field-decoupled overlap {r['final_whole_OFF_overlap']:.6g}.",f"  Finite-chi evolution diagnostics {r['evolution']}.",f"  Both physical source time orders {r['physical_current_time_orders']}; raw response norms preserved."])
        maxima={key:float(max(abs(x) for x in np.asarray(value).reshape(-1))) for key,value in r['dt_halfdt']['normalized_observable_differences'].items()}
        temporal=integrated_paired['source_transport_dt_assessment'][branch]
        signal_steps=[temporal[k] for k in ('other_cell_occupation_charge_motion_norm','other_cell_occupation_charge_dt_difference_norm','other_cell_Noether_motion_norm','other_cell_Noether_dt_difference_norm')]
        weighted_legs=[v['source_weighted_twoleg'] for v in r['dt_halfdt']['source_weighted_twoleg']]
        lines.append(f"  dt/halfdt maxobservable differences {maxima}; othercell neutral-subtracted (charge signal,stepdiff,Lg signal,stepdiff) {signal_steps}; weighted complex twoleg {weighted_legs}; raw/retained source norm2 {r['source_strength']:.6g}/{r['source_quality']['retained_source_norm2']:.6g}; coherent compressed Himage² {r['coherent_compressed_H_image_norm2']:.6g}, sum-piece² {r['piece_squared_norm_sum_NOT_Hnorm']:.6g} is NOT H². Step differences are not error bounds.")
        for row in integrated_paired['records'][branch]:
            lines.append(f"  Own PaP-pulse {row['incoming']}->{row['outgoing']}: initial Pi(all8) before{row['initial_reference_all_mode_Pi']} after{row['initial_kicked_all_mode_Pi']}; temporal full/decoupled {row['temporal_full_response_difference']:.6g}/{row['temporal_field_decoupled_response_difference']:.6g}, calibrated {row['calibrated_field_response']:.6g}; neutral-subtracted charge {row['excess_carrier_charge_response']}; Noether {row['excess_carrier_Noether_response']}.")
    lines.extend(['LIMITS / EVIDENCE',
        'No identical-profile/localization gate: physical LEFT moments and background-subtracted charge, not n/N or a global spin label.',
        'Microsites/ions, explicit chosen cavity boundary, quantum g/r/R and all eight field coordinates remain in the chosen H.',
        'Integrated47 longitudinal Gauss is eliminated via its grounded-sphere Green operator with integer ions and compensating boundary charge, not a sampled independent E lattice.',
        'Quantum r_v is local matrix amplitude, not microsite radius. Declared r³dr and C_g/(2Mr²) choose a nonflat SU2 cone metric, not the standard flat quaternion cone with4C_g; no metric cutover is inferred.',
        'No exact global SO3 from the unclosed finite-q set; source/current canonical and projection debts are not photon/mass claims.',
        'Pi is canonical field momentum, not SI E. Frozen47 pulses exp(i eps PaP) are NOT mode-pure full Weyl kicks: initial crossmode Pi shifts can dominate raw final response, not free propagation alone.',
        'Separate exact-electric-picture review: own Fourier source matrices/derivatives <=2.11e-10 and full U0/W fresh8field state Pi/q/work/norm integrals <=8.9e-16 at9/45labels; wrong cutH0 Pi drift .02481/.01721. This is independent free/source-kernel evidence, not new coupled-body evolution or closure.',
        'Field-decoupled/hopping-disabled control retains R kinetic: charges can still orbit. Zero owning EM source is not lab-current/continuity zero; only bare-echo calibration is inferred.',
        'Uniform q0 control preserves eight field coordinates/profile Gram; it is not the literal isotropic three-mode44 finite spectrum.',
        'Sewing/CAR quantization is an explicit quantum/topological input, not a classical-Euler derivation.',
        f"Typing selected-file natural exit {language['natural_exit']}; {language['diagnostics'].get('summary')}; rooted LSP exit {language['rooted_LSP']['natural_exit']}.",
        'All natural exits, warnings/failures, source-change flags and immutable source capsules retained in verdict.json and *_exit/input/stdout/stderr files.',
        'Own states/raw arrays are separate *_raw.npz; no borrowed solved state. Delivery manifest pins every included byte.'])
    assert len(lines)<=80,len(lines)
    (ROOT/(P+'synthesis.txt')).write_text('\n'.join(lines)+'\n')
    omitted_archives={P+'handoff_capsule.zip'}
    external_operators={p.name:dict(sha256=file_hash(p),bytes=p.stat().st_size) for p in sorted(ROOT.glob(P+'zero_source_repair2_*_operator.npz'))}
    (ROOT/(P+'external_derivative_operator_references.json')).write_text(json.dumps(external_operators,indent=2)+'\n')
    omitted_archives.update(external_operators)
    files=sorted(p for p in ROOT.glob(P+'*') if p.is_file() and p.name not in {P+'manifest.json',P+'delivery_capsule.zip',P+'delivery_receipt.json',*omitted_archives})
    manifest={p.name:dict(sha256=file_hash(p),bytes=p.stat().st_size) for p in files}
    (ROOT/(P+'manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(P+'delivery_capsule.zip'),'x',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.name)
        z.write(ROOT/(P+'manifest.json'),P+'manifest.json')
    capsule=ROOT/(P+'delivery_capsule.zip')
    receipt=dict(capsule=capsule.name,sha256=file_hash(capsule),bytes=capsule.stat().st_size,synthesis_lines=len(lines),typing_exit=language['natural_exit'],preserved_external_history_archives=sorted(omitted_archives),external_derivative_operator_references=P+'external_derivative_operator_references.json',consumer_source_bindings='Exact per-attempt input+source capsules, own physical-stage receipts, frozen base operator SHA and derivative-only array references. No duplicated Model/operator cache; original historical archive is also preserved separately.')
    (ROOT/(P+'delivery_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)

if __name__=='__main__':run()
