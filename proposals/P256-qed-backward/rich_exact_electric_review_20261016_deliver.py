"""Compact physical synthesis/interface and exact coupled/image case capsules."""
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parent
P='rich_exact_electric_review_20261016_'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def main():
    result=json.loads((ROOT/(P+'result.json')).read_text());images=json.loads((ROOT/(P+'images_result.json')).read_text());language=json.loads((ROOT/(P+'language_final.json')).read_text());exits={name:json.loads((ROOT/(P+name+'_exit.json')).read_text()) for name in ('coupled1','images2')}
    neutral=result['cases']['neutral'];charged=result['cases']['addition'];matched=charged['matched_old_preparation'];ray=result['initial_cut_ray_comparison'];mechanical=images['mechanical_control_scope']
    interface=dict(kind='INDEPENDENT_GENUINE_EXACT_ELECTRIC_COUPLED_REVIEW',law='OLD48 CAR8/global g/R/r5 plus eight real nonparallel transverse noncompact fields, NOT Common CAR64/native/local47',scope=result['scope'],parameters=result['parameters'],T=result['T'],dt=result['dt'],author_states_used=False,own_fresh_states=['N4 analytic neutral','N5 own canonical CAR/Wilson addition component0'],source=result['source'],charged_excess_Pi06=result['charged_minus_neutral']['0']['Pi'][6],charged_excess_Pi24=result['charged_minus_neutral']['2']['Pi'][4],initial_ray_comparison=ray,matched_preparation=matched,selected_own_image_bounds={name:images[name] for name in ('field_shell','angular_shell')},commands={name:record['command'] for name,record in exits.items()},source_input_capsules=[P+name+'_source_capsule.zip' for name in exits],verification_scope='Independent coupled propagation/observers/source preparation and physical protocol overlaps; pure frozen operator/image helpers reused with hashes, no producer CLI or author evolved-state inputs.',limits='Finite coupled moving Galerkin state family only. No full-H closure/angular-field convergence/photon/Maxwell/ground/physicalmass/local47 claim; N3/removal and cut2 coupled trajectories not independently rerun.')
    interface['reported_producer_dt02_comparison']=dict(source='rich_exact_electric_picture_20261016_synthesis.txt lines37-38, REPORTED not own author-state rerun',reported_Pi06=7.63796747e-6,reported_Pi24=-2.40401171e-6,own_dt01_minus_reported_Pi06=result['charged_minus_neutral']['0']['Pi'][6]-7.63796747e-6,own_dt01_minus_reported_Pi24=result['charged_minus_neutral']['2']['Pi'][4]+2.40401171e-6,interpretation='About1.1% response change; common finite positive scale/sign, NOT completed time-step convergence or exact agreement.')
    (ROOT/(P+'interface.json')).write_text(json.dumps(interface,indent=2)+'\n')
    lines=[
      'EXACT-ELECTRIC INDEPENDENT SCIENTIFIC EXECUTION REVIEW: smaller genuine finite coupled response, not just free/source audit.',
      'Only rich_exact_electric_review_20261016_* written; frozen48/current47/Common/producer files/capsules unchanged; no nested agents/producer CLI replay.',
      'Distinct OLD48 CAR8 full N4/N5 sectors/global quantumg,R/positive-r5 law, NOT Common CAR64/native or local-frame47.',
      'Same microscopic H/parameters: rotating ions/microsites, netQ=ions-fermions, straight quantum Peierls/Wilson paths, eight noncompact field modes.',
      'Both nonparallel input/output q, both T channels and cos/sin partners remain active. No added Maxwell B²,omega²a²,Pauli term or field/body freeze.',
      f'OWN selected case: INITIALfieldcut1=9 labels, materialI={result["parameters"]["material_inertia"]}, fieldI={result["parameters"]["cavity_inertia"]}, g/R regulatorlevels={result["parameters"]["angular_level"]}/{result["parameters"]["material_level"]}, r5.',
      'Own fresh analytic staggered-ion/Haar g,R/radial-cone/vacuum neutral; own literal8-bit CAR signs plus full rectangular Wilson source for N5.',
      f'Own selected source component0 rawweight{result["source"]["raw_weight"]:.12g}; own source versus frozen pure source action error{result["source"]["own_literal_CAR_Wilson_vs_pure_source_error"]:.3g}; no author state loaded.',
      f'Own dimensions neutral{neutral["dimension"]}, addition{charged["dimension"]}; T={result["T"]}, dt={result["dt"]}. N3/removal and cut2 COUPLED trajectories not independently rerun.',
      'Own reorthogonalized Arnoldi12/exponential-midpoint integration; static old-H comparisons use own Arnoldi24. No imported producer propagator/evolved output.',
      'H0=sumPi²/(2If) propagates exactly on fullL2 BEFORE label projection: basis U0(t)W|initial label>, not an exponential of cut H0.',
      'Initial9 coefficient labels are NOT hard physical photon occupations at t. H_I is coefficient generator, NOT physical full-H energy.',
      'b=a+t(Pi+impulse)/If, full Weyl/BCH phases and all Lambda²/b_l b_m products precede projection. Physical H energy adds H0.',
      f'Independent raising/lowering H0 action errors neutral/charged={neutral["ladder_before_cut"]["independent_full_H0_action_error"]:.3g}/{charged["ladder_before_cut"]["independent_full_H0_action_error"]:.3g}; wrong cut-b product difference{charged["ladder_before_cut"]["full_product_minus_square_of_cut_b06"]:.12g}.',
      f'Actual initial SAME-H action errors neutral/charged={neutral["t0_same_microscopic_H_action_error"]:.3g}/{charged["t0_same_microscopic_H_action_error"]:.3g}. Microscopic H unchanged; time-dependent Galerkin state family changed.',
      'Each full W field-only pulse has strength.23, zero initial other-mode Pi change, unchanged matter marginal and correct physical free-electric work.',
      'Source/receivers use ABSOLUTE time. Physical cross-protocol overlaps insert full Wleftdag Wright; independently checked by Gauss-Hermite Weyl integral, not coefficient-vdot.',
    ]
    for name,case in (('neutral',neutral),('addition',charged)):
        for axis in ('0','2'):
            protocol=case['protocols'][axis];out=protocol['out']
            lines.append(f'{name} {axis}->{out}: OWN full-minus-decoupled temporal Pi{protocol["full_minus_decoupled"]["Pi"][out]:.12g}, OFFPi{protocol["decoupled_temporal"]["Pi"][out]:.3g}; kickworkerror{protocol["kick_work_error"]:.3g}, initialcrossPierror{protocol["initial_cross_mode_Pi_error"]:.3g}.')
    lines.extend([
      f'OWN charged-minus-neutral excessPi: 0->6={result["charged_minus_neutral"]["0"]["Pi"][6]:.12g}; 2->4={result["charged_minus_neutral"]["2"]["Pi"][4]:.12g}. These are measured outputs, not copied producer target constants.',
      'Temporal comparison=end(pulse)-end(base)-initial(pulse)+initial(base), then own field-decoupled subtraction, then charged-minus-neutral; one absolute-age convention.',
      f'Charged protocol0 actual physical cross-overlap{charged["protocols"]["0"]["physical_cross_protocol_overlap"]}; WRONG coefficient overlap{charged["protocols"]["0"]["WRONG_coefficient_overlap"]}; difference{charged["protocols"]["0"]["physical_vs_coefficient_overlap_difference"]:.12g}.',
      'Producer REPORTED dt.02 excessPi06=7.63796747e-6/Pi24=-2.40401171e-6; own dt.01 differs by about1.1%. Same finite scale/sign, NOT completed time-step convergence/exact agreement.',
      'FIELD-DECOUPLED/HOPPING-DISABLED control is actually evolved under its OWN H, retains exact H0 plus R/g/r motion; its coefficient equation is protocol-independent.',
      'OFF source derivative zero follows removed field-dependent coupling, not a declaration that rotating charges have no lab convection or all physical positions freeze.',
      f'Own mechanical probe at coupled charged endpoint: with hoppingOFF but Lambda active source norm{mechanical["hopping_disabled_Lambda_still_coupled_source_norm"]:.12g}; fully field-decoupled source norm{mechanical["fully_field_decoupled_source_norm"]:.3g}; retained freeR-kinetic action norm{mechanical["retained_free_R_kinetic_action_norm"]:.12g}.',
      'THREE PREPARATIONS/FAMILIES independently kept separate, using fresh own charged source:',
      f'1 OLD fixed projected-H on OLD finitePaP ray: calibrated temporalPi6={matched["old_fixedH_same_oldray_temporal_full_minus_OFF"]:.12g}; old bare/OFF echo{matched["old_fixedH_OFF_temporal_echo"]:.12g}.',
      f'2 EXACTunboundedU0 on THAT SAME old finitePaP ray: calibrated temporalPi6={matched["exactU0_same_oldray_temporal_full_minus_OFF"]:.12g}; exactOFF echo{matched["exactU0_same_oldray_OFF_temporal_echo"]:.3g}.',
      f'3 EXACTunboundedU0 on DIFFERENT fullW ray: calibrated temporalPi6={matched["exactU0_DIFFERENT_fullWray_temporal_full_minus_OFF"]:.12g}.',
      f'Initial crossPi6 finitePaP shift{matched["initial_Pi6_finite_PaP_shift"]:.12g} versus fullW{matched["initial_Pi6_fullW_shift"]:.3g}; physical oldray/fullW squaredoverlap{matched["initial_oldray_vs_fullW_physical_squared_overlap"]:.12g}.',
      'Wrong finitePaP-as-fullW construction reaches and FAILS the actual cross-mode preparation assertion. The old regulated experiment itself is not retroactively invalidated.',
      'Attribution: large response reduction is mainly changing the PHYSICAL PREPARATION, not free-solver improvement on the same ray. ExactH0 still changes the coupled trial manifold.',
      f'Own source-fieldcut1/2 initial charged squaredoverlap{ray["charged_squared_overlap"]:.12g}; selected sourceweights{ray["selected_source_weight_cut1"]:.12g}/{ray["selected_source_weight_cut2"]:.12g}; neutral ray is identical by vacuum embedding.',
      'Cut2 here is ONLY fresh source-ray projection, no cut2 coupled evolution. Changed charged rays plus field/angular exterior images preclude calling response differences convergence.',
      f'Own absoluteT source sewing error{result["absolute_source"]["correct_own_vs_pure_error"]:.3g}; wrong time0 source differs by{result["absolute_source"]["wrong_time0_source_action_error"]:.12g}.',
      f'AbsoluteT actual current/contact derivative action errors{result["absolute_current_contact"]["source_derivative_action_error"]:.3g}/{result["absolute_current_contact"]["contact_derivative_action_error"]:.3g}; wrong time0 current differs by{result["absolute_current_contact"]["wrong_time0_current_action_error"]:.12g}.',
      f'Physical full-H baseline energy drift neutral/charged={neutral["baseline"]["physical_H_energy"]-neutral["initial"]["physical_H_energy"]:.12g}/{charged["baseline"]["physical_H_energy"]-charged["initial"]["physical_H_energy"]:.12g}; HI energy alone is not conserved.',
      f'Own interacting canonical Pi-dot/source debt norm{result["interacting_canonical_debt"]["Pi_dot_minus_actual_source_norm"]:.12g} versus source norm{result["interacting_canonical_debt"]["actual_source_norm"]:.12g}. Exact free propagation is not interacting canonical closure.',
    ])
    for name in ('field_shell','angular_shell'):
        image=images[name]
        lines.append(f'OWN charged-endpoint {name}: HI outgoing lowerbound{image["HI_outgoing_lower_bound"]:.12g}, H0 BASIS-CONNECTION exterior{image["H0_basis_connection_outside_norm_NOT_error"]:.12g}, full-H exterior{image["full_H_outside_norm_NOT_error"]:.12g}; HI compressionerror{image["HI_compression_error"]:.3g}.')
    lines.extend([
      'Own selected rectangular images use pure frozen operator helper on OWN coupled state, not producer image/state outputs; field/angular target shells are separate lower bounds.',
      'H0 exterior is REQUIRED moving-basis derivative, not itself cut error. HI exterior measures missing coupled directions; do not confuse it with full-H variance or a decay rate.',
      'Tiny compression/time-step diagnostics do not establish angular/field convergence, full-H closure, stationary ground or photon pole. No full exterior equality asserted.',
      'Positive finite mechanism: full-W electric input changes a(t), acts through actual Peierls paths and mechanical Lambda coupling, and produces charge-dependent transverse field/current feedback beyond exact free OFF.',
      'Smaller genuine feedback is retained, not repaired by adding bare Maxwell B²,oscillator omega²a² or Pauli B. Individual hopping/mechanical contributions are not uniquely isolated by this combined OFF trajectory.',
      'Meaningful next rich construction: preserve the SAME physical fullW/source ray while enriching with HI/current/mechanical exterior images and consistent receivers; derive any induced stiffness from the common current/field law.',
      'Do not merely compare newly projected source rays at bigger field caps, append a restoring term, or impose a new ground/range permission gate on active construction.',
      'No Maxwell/photon/mass/g2/thermal/full-H convergence or local-frame47/Nature claim; this global-frame CAR8 law remains distinct from Common/native and other branches.'
    ])
    for name,record in exits.items():lines.append(f'Own {name}: child rc{record["returncode"]}, natural={record["natural_exit"]}, seconds{record["elapsed_seconds"]:.6f}, peakRSS_KiB{record["child_peak_RSS_KiB"]}, exact source/input epoch unchanged={record["source_input_unchanged"]}.')
    lines.extend([f'Final selected diagnostics exit{language["diagnostic_exit"]}: {language["diagnostics"]["summary"]}; rootedLSP exit{language["rooted_LSP"]["natural_exit"]}; no unobserved green claim.','Own raw coupled/oldray states, image tensors, interface and exact source/input/exit capsules are manifest-bound; parent reads this compact synthesis, reviewer owns raw audit.'])
    if len(lines)>80:raise ValueError(('Compact physical synthesis exceeded80lines',len(lines)))
    (ROOT/(P+'synthesis.txt')).write_text('\n'.join(lines)+'\n');excluded={P+'manifest.json',P+'delivery_capsule.zip',P+'delivery_receipt.json'};files=sorted(path for path in ROOT.glob(P+'*') if path.is_file() and path.name not in excluded)
    manifest=dict(kind='EXACT_ELECTRIC_INDEPENDENT_COUPLED_CASE_DELIVERY',plan='issue222-rich-backward-candidates-20261014',captured_revision='143d2f6af58dec9b80596fa3055175e5c5c131992d1865f8bef9201f555e8920',template='general@89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',synthesis_lines=len(lines),files={path.name:dict(bytes=path.stat().st_size,sha256=digest(path)) for path in files},science_exits={name:{key:record[key] for key in ('returncode','natural_exit','elapsed_seconds','child_peak_RSS_KiB','source_input_unchanged')} for name,record in exits.items()},scope=result['scope'])
    path=ROOT/(P+'manifest.json');path.write_text(json.dumps(manifest,indent=2)+'\n');capsule=ROOT/(P+'delivery_capsule.zip')
    with zipfile.ZipFile(capsule,'w',zipfile.ZIP_DEFLATED) as archive:
        for entry in files+[path]:archive.write(entry,entry.name)
    receipt=dict(capsule=capsule.name,sha256=digest(capsule),bytes=capsule.stat().st_size,synthesis=P+'synthesis.txt',synthesis_lines=len(lines),interface=P+'interface.json');(ROOT/(P+'delivery_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
