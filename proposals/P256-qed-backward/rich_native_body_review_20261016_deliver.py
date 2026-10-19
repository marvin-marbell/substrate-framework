"""Bounded plain-physics synthesis plus exact source/input/raw/exit delivery."""
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parent
P='rich_native_body_review_20261016_'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def main():
    result=json.loads((ROOT/(P+'result.json')).read_text());exit_record=json.loads((ROOT/(P+'actual1_exit.json')).read_text());language=json.loads((ROOT/(P+'language_final.json')).read_text())
    gauge=json.loads((ROOT/(P+'gauge_result.json')).read_text());gauge_exit=json.loads((ROOT/(P+'gauge2_exit.json')).read_text())
    half=result['sectors']['1'];three=result['sectors']['3'];f=half['trajectories']['full'];off=half['trajectories']['electric_OFF'];freeze=half['trajectories']['ALLhopOFF'];pulse=half['events']['EM_pulse'];age=half['events']['same_age_no_EM'];kick=half['events']['Phi_only_kick'];primitive=result['primitive_audit']
    lines=[
      'Independent native rich-body review: fresh charged source, spin comparison, propagation, current and quantum-field controls.',
      'Only rich_native_body_review_20261016_* created; old Review36/model/capsules immutable. No parked producer wake, catalogue assembly or producer CLI.',
      'Pinned model correlated_dressing3; shared OWN one-hop hot Omega/source coordinates are declared inputs, not an independently solved ground.',
      'Root /home/administrator/substrate-222/proposals/P256-qed-backward; Python /home/administrator/substrate-framework/.venv/bin/python.',
      'Command: baseline Python rich_native_body_review_20261016_capture.py actual1. Every imported source/native input is hashed and archived.',
      f'Used original ten weak operators/all actual G; fresh halfspace{half["body_dimension"]}x7={half["full_Phi_dimension"]}, threehalf{three["body_dimension"]}x7={three["full_Phi_dimension"]}.',
      'No stored lowstates, author prepared/evolved state or1328-block reconstruction supplied the answer. Canonical probe coordinates define source only.',
      'Independent count-block sqrtG coordinates preserve the full generalized metric, without Hermitizing GinvH or conflating fixed-state H with source-EOM K.',
      f'Half metric eigenrange{half["metric"]["minimum_eigenvalue"]:.12g}..{half["metric"]["maximum_eigenvalue"]:.12g}; inverse-root residual{half["metric"]["inverse_sqrt_metric_residual"]:.3g}.',
      f'Selected raw primitive/source/Gram audit max error{max(primitive["independent_errors"].values()):.3g}; actual unmasked bare strength{primitive["actual_source_raw_strength"]:.12g}.',
      'Audit contracts full local colour electric square, R kinetic, plaquette and one tree01 up-quark scalar/vector hop before projection, plus source covector.',
      f'Raw chosen J² checks half/threehalf residual²{primitive["angular"]["1"]["raw_residual_norm2"]:.3g}/{primitive["angular"]["3"]["raw_residual_norm2"]:.3g}, eigenvalues.75/3.75.',
      'Physical chosen J=L_left+S_CAR rotates the quantum body; primitive CAR spin, colour, fractional charges and geometry/action are inputs, not derived QCD/field-only spin.',
      f'Fresh lowest energies half{half["fresh_energies"][0]:.12g}, threehalf{three["fresh_energies"][0]:.12g}; difference{result["threehalf_minus_half"]:.12g}.',
      f'Full-space eigen residual maxima{max(half["fresh_eigen_residuals"]):.3g}/{max(three["fresh_eigen_residuals"]):.3g}; finite distinct variational J spaces, not global spin ordering.',
      f'Offsets to hot selected E0 are{half["raw_offset_to_hot_selected_reference"]:.12g}/{three["raw_offset_to_hot_selected_reference"]:.12g}: NEGATIVE raw offsets, not positive mass/gap or a new vacuum.',
      f'Original unmasked half source retained strength{half["source_raw_retained_strength"]:.12g}; original compact0 conditioning strength{half["compact0_conditioning_strength"]:.12g}.',
      'Prepare original bare site0 source, condition original Csite0 and normalize using actual G. Same receivers0..3; independentPhi7 released throughout.',
      f'Half initial compact-site weights{f[0]["compact_site"]}; at t4{f[-1]["compact_site"]}, total{f[-1]["compact"]:.12g}.',
      f'Half physical Qsite initial{f[0]["Qsite"]}; at t4{f[-1]["Qsite"]}, total{sum(f[-1]["Qsite"]):.12g}.',
      f'Threehalf t4 compact-site weights{three["trajectories"]["full"][-1]["compact_site"]}, total{three["trajectories"]["full"][-1]["compact"]:.12g}.',
      'Positive result: the charged spin-half three-quarks-together COUNT COMPONENT now reaches other body vertices while quantum sea/core/field profiles may change.',
      'Csite is whole-sea colourcount(6,6,6,9), not tagged-three-particle position/capture PVM; Qsite expectation is different and not a probability.',
      'Bsite=(Ncolour-6)/3 also differs from Qsite. Falling total compact weight is neither a breakup measurement nor a binding/lifetime result.',
      f'AllhopOFF max material-count PVM-TV{max(row["material_count_TV"] for row in freeze):.3g}; max full norm drift{max(abs(row["norm"]-1) for row in f):.3g}.',
      'Correct allhopOFF removes quark+white scalar AND vector hops; R/Phi/PW/electric/plaquette dynamics remain. Count freeze is NOT every lab coordinate frozen.',
      f'Plausible WRONG allhopOFF removing scalar hops only reaches physical assertion and FAILS: t.25 count-TV{half["wrong_scalar_only_ALLhopOFF"]["material_count_TV"]:.12g}.',
      f'ElectricOFF t4 compact{off[-1]["compact"]:.12g}, site1{off[-1]["compact_site"][1]:.12g}; full t4 site1{f[-1]["compact_site"][1]:.12g}.',
      f'ElectricOFF retains nonzero plaquette Frobenius{half["electric_OFF_retained_colour"]["plaquette_Frobenius"]:.12g} and quantum U/loop response; NOT allcolourOFF.',
      f'Native physical-charge Ward max{max(half["material_Q_Ward_Frobenius"]):.3g}; actual finite-state Qdot+divJ error{half["actual_Qdot_vs_minus_divJ_error"]:.3g}.',
      f'Fresh local node3 graph-gauge covariance fullPhi defect{gauge["fullPhi_native_H_covariance_error"]:.3g}; wrong chord-only node gauge fails assertion with defect{gauge["wrong_one_edge_gauge_error"]:.12g}.',
      f'Chord current/contact finite-derivative checks{half["finite_Peierls_derivative_checks"]}; both vector and Phi hopping contributions retained.',
      'Current is original fractional-charge material Peierls chord current, not labrotational convection/Maxwell. Lab A would additionally couple mechanical rotation.',
      'EM protocol theta=.3 on chord3 for.4, then off; event age=.4+t compared with independently evolved SAME-AGE noEM, not unshifted baseline.',
      f'At afterpulse t4, compact contrast{pulse[-1]["compact"]-age[-1]["compact"]:.12g}, Qsite contrast{[a-b for a,b in zip(pulse[-1]["Qsite"],age[-1]["Qsite"])]}.',
      f'EM also changes actual white-sea site means at age4.4 to{pulse[-1]["white_site"]}, versus same-age noEM{age[-1]["white_site"]}; not a rigid-core/field-return detector.',
      f'Actual Phi-only exp(-i*.3P) initial count-TV{half["field_only_initial_count_TV"]:.3g}, Phi²{f[0]["Phi2"]:.12g}->{kick[0]["Phi2"]:.12g}.',
      f'At t4 fieldkick/no-kick compact contrast{kick[-1]["compact"]-f[-1]["compact"]:.12g}, loopB contrast{kick[-1]["loopB"]-f[-1]["loopB"]:.12g}, loopC contrast{kick[-1]["loopC"]-f[-1]["loopC"]:.12g}.',
      f'At t4 no-kick Phi purity{f[-1]["Phi_purity"]:.12g}, white-site means{f[-1]["white_site"]}; independent fullPhi/body/sea wakes, not field return postselection.',
      'Author C/F contrast1.65217 vs electricOFF-.95843 means E_min(F)-E_min(C) in NONNESTED count subspaces, not population ratio or asymptotic breakup threshold; not rerun.',
      'Finite rich body/source/sea/R/PW rank remains severe despite full independentPhi7. No neutral ground theorem used as permission gate.',
      'No new full-primitive residual/lifetime measured; old W RMS5.8717 and failed~1e-12 W transfer are historical DIFFERENT states, not inherited or rerun.',
      'Still missing: physical breakup/binding threshold, stationary common vacuum/positive mass, fullprimitive lifetime, free lab COM/Lorentz/continuum and full lab EM coupling.',
      'No proton/Nature/QCD/SU4 admission, RMS subtraction, tuned particle definition, fixed quark core or Omega-return requirement.',
      f'Actual child exit{exit_record["returncode"]}, natural={exit_record["natural_exit"]}; capture seconds{exit_record["elapsed_seconds"]:.6f}, child peakRSS_KiB{exit_record["child_peak_RSS_KiB"]}, unchanged source/input={exit_record["source_input_unchanged"]}.',
      f'Gauge command: baseline Python rich_native_body_review_20261016_capture.py gauge2 gauge.py; child rc{gauge_exit["returncode"]}, natural={gauge_exit["natural_exit"]}, unchanged inputs={gauge_exit["source_input_unchanged"]}.',
      f'Final selected diagnostics exit{language["diagnostic_exit"]}, summary{language["diagnostics"]["summary"]}; rooted LSP exit{language["rooted_LSP"]["natural_exit"]}. No all-clean claim.',
      'Raw own states/eigenvectors/primitive arrays are separate NPZ; source/native inputs and exit are preserved in exact source_capsule.zip, manifest and delivery capsule.'
    ]
    if len(lines)>80:raise ValueError('Plain physics synthesis exceeds80lines')
    (ROOT/(P+'synthesis.txt')).write_text('\n'.join(lines)+'\n')
    excluded={P+'manifest.json',P+'delivery_capsule.zip',P+'delivery_receipt.json'};files=sorted(path for path in ROOT.glob(P+'*') if path.is_file() and path.name not in excluded)
    manifest=dict(kind='FRESH_NATIVE_RICH_BODY_PHYSICAL_REVIEW_CAPSULE',plan='issue222-rich-backward-candidates-20261014',captured_revision='fe8a72a60be4e42a08270f54b24dd8f5e17ef263e30e6abf121eff19c0e736aa',template='general@89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',source_and_input_capsule=P+'actual1_source_capsule.zip',files={path.name:dict(bytes=path.stat().st_size,sha256=digest(path)) for path in files},science_exit=dict(returncode=exit_record['returncode'],natural_exit=exit_record['natural_exit'],source_input_unchanged=exit_record['source_input_unchanged']),synthesis_lines=len(lines),limits=result['limits'])
    manifest['gauge_science_exit']=dict(returncode=gauge_exit['returncode'],natural_exit=gauge_exit['natural_exit'],source_input_unchanged=gauge_exit['source_input_unchanged'])
    helper=ROOT/'rich_review_wave36_20261016_language.py';manifest['frozen_language_helper']=dict(name=helper.name,sha256=digest(helper))
    m=ROOT/(P+'manifest.json');m.write_text(json.dumps(manifest,indent=2)+'\n');capsule=ROOT/(P+'delivery_capsule.zip')
    with zipfile.ZipFile(capsule,'w',zipfile.ZIP_DEFLATED) as z:
        for path in files+[m]:z.write(path,path.name)
        z.write(helper,'external_language_helper/'+helper.name)
    receipt=dict(capsule=capsule.name,sha256=digest(capsule),bytes=capsule.stat().st_size,synthesis=P+'synthesis.txt',synthesis_lines=len(lines))
    (ROOT/(P+'delivery_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
