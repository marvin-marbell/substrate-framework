"""Deliver fresh physics, mechanism interpretation and precise missing vertex."""
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parent
P='rich_common_species_review_20261016_'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def main():
    result=json.loads((ROOT/(P+'result.json')).read_text());exit_record=json.loads((ROOT/(P+'actual1_exit.json')).read_text());language=json.loads((ROOT/(P+'language_final.json')).read_text());joint=result['joint'];full=joint['models']['full'];coord=joint['models']['coordinate_only'];off=joint['models']['ALLhopOFF'];audit=result['source_audit'];algebra=joint['field_CAR_algebra']
    lines=[
      'Independent COMMON SPECIES review: fresh N33/N31 sources/motion and one-Fock N36 two-way causal coupling, with actual mechanism discriminator.',
      'New rich_common_species_review_20261016_* only; all producer, old Review36 and native-body review epochs immutable. No producer CLI/bodyblock rebuild.',
      'Same primitive CAR64/Phi7/R10/PW19/parameters/occupiedOmega INPUTS; sector-specific native Galerkin spans remain different finite approximations.',
      'Command: /home/administrator/substrate-framework/.venv/bin/python rich_common_species_review_20261016_capture.py actual1',
      'Workdir /home/administrator/substrate-222/proposals/P256-qed-backward; exact imported sources/operators/Gram/canonical probes/input hashes archived.',
      'No stored author prepared/evolved/lowstate arrays supplied the proof; each original source is freshly normalized and propagated in the full native coordinates.',
      f'Independent originalCAR source strengths N33={audit["sources"][0]["actual_raw_strength"]:.12g}, N31={audit["sources"][1]["actual_raw_strength"]:.12g}; sum-minus-Omega-norm={audit["addition_plus_removal_minus_omega_norm"]:.3g}.',
      'N33 is cdag of DOWN-WHITE,physical spinUP,Q-1. N31 is ORIGINAL c of that same orbital,Q+1,M-.5, not an intersector current or assumed C partner.',
    ]
    for number in ('33','31'):
        row=result['carriers'][number];event=row['trajectories']['full'][-1]
        lines.extend([
          f'N{number}: own full dimensions{row["body_dimension"]}x7={row["full_Phi_dimension"]}; retained raw sourceG{row["source"]["retained_raw_strength"]:.12g}; t4 totalQsite{event["Qsite"]}.',
          f'N{number}: same-age finite white pulse/current/contact exercised; Wardmax{max(v for values in row["Ward"].values() for v in values):.3g}; actual ALLhopOFF count-TVmax{max(v["count_TV"] for v in row["trajectories"]["ALLhopOFF"]):.3g}.',
        ])
    lines.extend([
      f'Joint source Bdag0 Ldag2 Omega, L FIRST, canonical epsilon/sqrt6 and white1, UNMASKED rawG{audit["sources"][2]["actual_raw_strength"]:.12g}; native retainedG{joint["source"]["retained_raw_strength"]:.12g}.',
      f'Joint N36 has J1,M1 by the actual aligned-halfspin source algebra; full dimensions{joint["body_dimension"]}x7={joint["full_Phi_dimension"]}. No singlet gate imposed.',
      'Actual source Sz/S² canonical commutators were freshly evaluated; primitive spin/charge/SU3/3+1/action are inputs, not derived field-only spin/QCD.',
      f'Joint whole-sea charges quark/white/total={joint["whole_sea_charge_totals"]}: these are +5/-5, NOT tagged added +1/-1 charges.',
      'Physical whole-quark compact counts and white addition/hole-count Psite are not unique tagged particle positions, netQ or breakup thresholds.',
      'White-only edge2 pulse measures quark edge0; reverse quark-only edge0 measures white edge2. theta.3/duration.4 unchanged before outcomes.',
      'During0,.025,.05,.1,.2,.4; afterwards.25,.5,1,2,4. Every no-pulse comparator is independently evolved at the SAME ABSOLUTE AGE.',
      'Finite-theta J=-dH/dtheta and contact=d²H/dtheta² retain exp(iqtheta), BOTH scalar+vector terms and original fractional charges, not baryoncharge1.',
    ])
    for source in ('white','quark'):
        response=full['response'][source];action=joint['initial_cross_actions'][source];fd=joint['finite_theta_derivatives'][source]
        lines.extend([
          f'FULL {source} pulse: max other-species |deltaJ|{response["max_abs_receiver_current"]:.12g}, |deltaQ|{response["max_abs_receiver_charge"]:.12g}; first t.025 pinned deltaJ{response["rows"][1]["delta_pinned_receiver_current"]:.12g}.',
          f'{source}: t0 receiver deltaJ/Q={response["rows"][0]["delta_receiver_current_all_edges"]}/{response["rows"][0]["delta_receiver_charge_all_sites"]}; state/receiver continuous, not instantaneous transfer.',
          f'{source}: finite-theta derivative action errors J/contact{fd["current_action_difference"]:.3g}/{fd["contact_action_difference"]:.3g}; wrongly dropping phase changes J action by{fd["wrong_dropped_finite_phase_current_action_error"]:.12g}.',
          f'{source}: actual [DeltaHsource,Jreceiver]psi norm²{action["cross_commutator_action_norm2"]:.12g}; i<comm>={action["initial_delta_receiver_current_slope"]:.12g}, independent state-derivative slope{action["independent_state_derivative_slope"]:.12g}.',
          f'{source}: coherent components [VV,VS,SV,SS] action norms²{action["component_action_norms2"]}; ALL interference retained via raw4x4Gram, not sum-of-piece-norms.',
        ])
    lines.extend([
      'All products/actions use independent full count-block sqrtG coordinates: weak product Aweak Ginv Bweak, NOT naive weak-matrix multiplication.',
      'Units: DeltaH action norm² is energy², cross-commutator action norm² energy4; current/contact are owning H derivatives per dimensionless chordtheta, NOT probabilities or SI force.',
      'Fixed native metric is not the separate hot-source EOM K/Gdot problem; GinvH never Hermitized and no moving-source connection silently omitted.',
      f'Actual microscopic map audit: {algebra["actual_quark_even_bilinears"]} quark/{algebra["actual_white_even_bilinears"]} white even bilinears; crossCAR contractions{algebra["cross_CAR_contraction_count"]}.',
      'Disjoint quark/white member modes make every cross even-bilinear CAR commutator zero. Full Phi/R coordinate multiplication commutes.',
      'Chosen quark receiver is TREE edge0 and white edge2 carries no U. Hence source/receiver commute BEFORE field/body projection, for ANY state.',
      f'Actual full pointwise R product commutator norm{algebra["unprojected_pointwise_rotation_product_commutator"]:.3g}; finite R10 coordinate commutator maximum{algebra["maximum_projected_R_commutator"]:.12g}.',
      'Finite projected R can therefore already transmit coupling; further body/source compression gives [PAP,PBP]=P[A,B]P-P A Q B P+P B Q A P.',
      'Nonzero SS native components identify body/source compression (Phi multiplication itself is fully retained/commuting). VV can combine finiteR and body paths.',
      'ACTIVE-SOURCE control removes Hphi KINETIC, Hrot and SU3 electric kinetic, retains scalar potential, plaquette and ALL quark/white scalar+vector hops.',
      'Its projected geometry remains quantum/noncommuting; this is NOT a c-number field freeze, allcolourOFF, or zero source by construction.',
    ])
    for source in ('white','quark'):
        row=coord['response'][source]
        lines.append(f'COORDINATE-ONLY {source} pulse: actual sourceDeltaH action norm²{coord["active_source_action_norms2"][source]:.12g}; max cross |deltaJ|{row["max_abs_receiver_current"]:.12g}, |deltaQ|{row["max_abs_receiver_charge"]:.12g}.')
    lines.extend([
      f'Coordinate-only baseline Phi² conservation error{coord["Phi2_baseline_conservation_error"]:.3g}; full quantum coordinate/body distribution remains, not a replacement scalar level.',
      'Mechanism verdict: '+joint['kinetic_necessity_hypothesis_assertion'],
      'Thus a finite noncommuting shared-geometry/Galerkin coupling is a POSITIVE candidate link here, not automatically a defect or proof of untruncated substrate interaction.',
      'Field-kinetic backreaction can coexist/change higher-time response in FULL H; accompanying field changes do not uniquely isolate it.',
      f'FULL white/quark max shared-field contrasts{full["response"]["white"]["max_abs_shared_fields"]}/{full["response"]["quark"]["max_abs_shared_fields"]}; orderPhi,Phi²,Pphi,Hphi,Hrot,loopB,loopC,HE.',
      f'Actual ALL scalar+vector hops OFF count-TVmax{max(row["count_TV"] for row in off["baseline"]):.3g}; sourceDeltaH norms²{[off["pulses"][source]["actual_source_DeltaH_action_norm2"] for source in ("white","quark")]}.',
      f'ALLhopOFF max crossJ/Q{[(off["response"][source]["max_abs_receiver_current"],off["response"][source]["max_abs_receiver_charge"]) for source in ("white","quark")]}; field terms remain active.',
      'That null tests actual controlled hopping/observer consistency ONLY: source itself vanishes, so it is NOT a mediator-specific null.',
      f'Joint species Wardmax{max(v for values in joint["Ward"].values() for v in values):.3g}; count/metric/current closure is MATERIAL, not labrotational convection/Maxwell.',
      'This is temporal causal finite-H sector response, not relativistic lightcone, dynamical photon/Coulomb force, binding/proton/lifetime or positive mass.',
      'Important remaining vertex: N36 response includes the WHOLE +5/-5 sea. It does not isolate interaction of the ADDED canonical B and L.',
      'Need matched evolving-Omega N32/N35/N33/N36 source/current kernels with a common controlled operator lift/virtual space and raw Gram normalizations.',
      'Then form the connected canonical B,Bdag,L,Ldag/current cumulant, subtract every disconnected partition consistently; only afterwards compare/amputate.',
      'Those matched controls are not supplied. Old N35 and new source-specific N33/N36 spans differ; normalized R36-R35-R33+R32 is NOT justified.',
      'Do not rebuild huge sectors as an approval gate. Current finite coupling and charged motion remain useful while the source-connected added-pair vertex is open.',
      'Native source/neutralonehop/body/R/PW/anchor cuts severe; full independentPhi7 retained. Hot negative offsets/reference RMS are not masses or decay.',
      'Other47/48 laws remain distinct; no solved-state splice, whole issue closure, derived field-only spin or freeCOM claim.',
      f'Own scientific child rc{exit_record["returncode"]}, natural={exit_record["natural_exit"]}; capture seconds{exit_record["elapsed_seconds"]:.6f}, peakRSS_KiB{exit_record["child_peak_RSS_KiB"]}, unchanged sources/inputs={exit_record["source_input_unchanged"]}.',
      'Scientific stderr retains a SciPy FutureWarning about integer density diagonals cast to float64; no runtime warning suppression or all-clean claim.',
      f'Final selected diagnostics exit{language["diagnostic_exit"]}: {language["diagnostics"]["summary"]}; rooted LSP exit{language["rooted_LSP"]["natural_exit"]}. No typing-clean claim.',
      'Own states/cross actions/field matrices/contact/current arrays separate NPZ; exact source/native input/exit capsules and manifest included in delivery archive.'
    ])
    if len(lines)>80:raise ValueError(('Physical synthesis exceeds80lines',len(lines)))
    (ROOT/(P+'synthesis.txt')).write_text('\n'.join(lines)+'\n');excluded={P+'manifest.json',P+'delivery_capsule.zip',P+'delivery_receipt.json'};files=sorted(path for path in ROOT.glob(P+'*') if path.is_file() and path.name not in excluded)
    manifest=dict(kind='EXACT_FRESH_COMMON_SPECIES_RUNTIME_AND_MECHANISM_REVIEW',plan='issue222-rich-backward-candidates-20261014',captured_revision='8e2d5c7c966a528442ea07f5160679554a3e24626f21a7602f5489668d499795',template='general@89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',synthesis_lines=len(lines),files={path.name:dict(bytes=path.stat().st_size,sha256=digest(path)) for path in files},science_exit={key:exit_record[key] for key in ('returncode','natural_exit','elapsed_seconds','child_peak_RSS_KiB','source_input_unchanged')},limits=result['limits'])
    helper=ROOT/'rich_review_wave36_20261016_language.py';manifest['frozen_language_helper']=dict(name=helper.name,sha256=digest(helper));m=ROOT/(P+'manifest.json');m.write_text(json.dumps(manifest,indent=2)+'\n');capsule=ROOT/(P+'delivery_capsule.zip')
    with zipfile.ZipFile(capsule,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in files+[m]:archive.write(path,path.name)
        archive.write(helper,'external_language_helper/'+helper.name)
    receipt=dict(capsule=capsule.name,sha256=digest(capsule),bytes=capsule.stat().st_size,synthesis=P+'synthesis.txt',synthesis_lines=len(lines));(ROOT/(P+'delivery_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
