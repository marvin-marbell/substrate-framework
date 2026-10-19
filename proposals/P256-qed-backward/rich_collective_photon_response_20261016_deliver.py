"""Produce <=80-line physical synthesis and revision-bound source/raw capsule."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent
P='rich_collective_photon_response_20261016_'


def fmt(stat):
    return f"{stat['mean']:.6g}+/-{stat['block_SEM']:.2g}"


def main():
    result=json.loads((ROOT/(P+'response_summary.json')).read_text());cases=result['cases']
    probe=json.loads((ROOT/(P+'probe_v2_result.json')).read_text())
    stress=json.loads((ROOT/(P+'stress_result.json')).read_text())
    lang=json.loads((ROOT/(P+'language_result.json')).read_text())
    stages=['probe','stress','hot','cold','coldrefine','analyze','language']
    exits={stage:json.loads((ROOT/(P+stage+'_exit.json')).read_text()) for stage in stages}
    lines=[
        'COLLECTIVE FIELD RESPONSE46: SAME FULL COMPACT ROTOR H, NEW V64 DATA',
        'Plan issue222-rich-backward-candidates-20261014 @673b0253; work item photon-candidate; parent alone owns joined/public outcome.',
        'Frozen Rotor41 input capsule SHA25636d97496914d742caf5f73631ff87edc31c0e3c0cb937bc4f9fb40dfd03e8106; Review43 rotor verdict read.',
        'L4/V64/192 links, t1/U4/I16, full E integer and bilateral U, ions1. No bare B/Maxwell or inserted opposite-charge species.',
        'Original TWO SAME-charge flavours: exact down PH yields conjugate up/hole determinants, centered Hubbard+1/site; compact holonomy projects full Gauss.',
        'Gauss-only original totalN64 spin mixture, NOT canonical32up/32down. Pure frozen helpers imported, no ancestor producer CLI or old-array physics replay.',
        'Exact finite-M change-of-variable: Z[A]=int K(Btheta)W(theta+A), Z_ab/Z=-(B^T C_E B)_ab.',
        'For n!=0, C_E(q,n)=-<W_ab/W>_n/[4sin^2(pi n/M)]; D_E=dt*C_E is an actual electric-source susceptibility.',
        'First+second Frechet interval exp(-dt h), h_fg contact and all determinant virtual/conditional-mean feedback are included. NOT continuum(K-JJ)/omega^2.',
        'The full source/current response is NOT an irreducible polarization. Contact-only and full interval-current arrays are separate.',
        'n0 electric response is independently measured from compact kinetic Kprime/Kdoubleprime, including holonomy/winding; never inferred from source Ward zero.',
        'All temporal n measured independently; complex tau retained. qminus is exact conjugate-source exchange(q,n)<->(-q,-n), not forced per-path inversion.',
        f"Own physical6-state transfer: exact-H trace{probe['exact_H_trace']:.10g}; finite-M Ward error<={max(r['exact_discrete_identity_error'] for r in probe['finite_transfer']):.3g}.",
        f"At M4 wrong point vertex error{probe['finite_transfer'][1]['wrong_point_vertex_error']:.6g}; missing contact error{probe['finite_transfer'][1]['wrong_missing_contact_error']:.6g}; independent electric n0{probe['finite_transfer'][1]['independent_zero_Matsubara']:.6g}.",
        f"Own compact holonomy trace{probe['holonomy']['projected_trace']:.10g} matches actual M1 physical trace to{probe['holonomy']['error']:.2g}; omitted projector gives{probe['holonomy']['wrong_omitted_projector']:.8g}.",
        f"Winding adverse control: omitted images fail{probe['winding']['wrong_one_image_error']:.4g}; wide-image exact integer-kernel error{probe['winding']['compact_wide_image_error']:.2g}. Five-image helper is inadequate in that light-rotor case, not used there as exact.",
        f"Analytic/stable small Hessian discrepancy{probe['stable_vs_analytic_Hessian_error']:.3g}; independent real-source finite difference discrepancy{probe['independent_real_source_derivative_error']:.3g}.",
        'Cold method is MATERIAL fork: pivoted QDT determinant + stabilized cyclic-HS Green, exact local Frechet jets, numerical centered scalar derivatives h=.008 with selected h/2 controls.',
        f"80-digit actual V8 cold transfer control: max stabilized logdet error{max(r['stable_logabsdet_error'] for r in stress['rows']):.3g}; Green{max(r['stabilized_Green_error'] for r in stress['rows']):.3g}; wrong direct logdet reaches{max(r['direct_logabsdet_error'] for r in stress['rows']):.5g}.",
        'ACTUAL NEW D_E(n1): entries [qpi/2 Ty,Tz; qpi Ty,Tz], +/- exploratory8-block SEM; D_E(n0) range shown independently.'
    ]
    for c in cases:
        modes=c['modes'];D0=[m['D_E_n0']['mean'] for m in modes]
        lines.append(f"{c['name']}: D1=["+'; '.join(fmt(m['D_E_n1']) for m in modes)+f"]; D0={min(D0):.6g}..{max(D0):.6g}; N={c['metadata']['samples']}.")
    lines.append('SAME-PATH FIELD/MATTER LEGS: qpi/2 Ty electric tauhalf (exact-source estimator) vs direct noisy kinetic estimator; sin-plaquette B retention and full interval-J retention at tauhalf.')
    for c in cases:
        m=c['modes'][0];b=m['B_tauhalf']['mean']/m['B_tau0']['mean'];j=m['interval_J_tauhalf']['mean']/m['interval_J_tau0']['mean']
        lines.append(f"{c['name']}: EEhalf={fmt(m['EE_tauhalf'])}; direct={fmt(m['direct_EE_tauhalf'])}; Bhalf/B0={b:.6g}; Jhalf/J0={j:.6g}; <cos plaquette>={fmt(c['plaquette'])}.")
    lines.extend([
        'Compact B=sin(F_xpol) is measured on these SAME new paths, no B^2 energy inserted. D_B imaginary-time/Matsubara arrays are delivered with D_E and exact interval current/contact.',
        'Restoring mechanism is matter-mediated electric current/contact feedback in this H; field response is now measured by the exact discrete transfer identity instead of noise-limited direct subtraction.',
        'Slow B-coordinate retention and fast material-current decay are distinct observables; heavy free rotors can also retain B, so retention alone does NOT prove collective photon restoring.',
        'Hot beta.5/1 first Matsubara frequencies are matter-UV. Actual stabilized beta4/8 states lower the first frequency to1.571/.7854; no oscillator/pole fit is substituted for these data.',
        'Actual dt=.25/.125 cases exist at beta1 and beta4; these are finite-Trotter comparisons, NOT a controlled extrapolation. beta8 has dt=.25 only.',
        'Actual independent seeds exist at beta.5/1/4/8. Fresh all-link bridge, collective temporal, HS, holonomy and all-static-link moves improve scope, not certify mixing.',
        'Raw burn/move acceptance/history,8-block SEM, first/second-half estimates, static-source Ward discrepancy, source means and seed differences remain visible.',
        'Static-source n0 should average0 but cannot supply electric zero mode. Finite-chain Ward defects and seed disagreement are adverse diagnostics, not erased or relabelled equilibrium.',
        f"Largest selected cold h/2 source-Hessian change{max(c['maximum_selected_step_halving_difference'] for c in cases):.3g}; cold numerical derivative error remains distinct from MC/thermal/Trotter error.",
        f"Maximum sampled transverse temporal winding{max(c['maximum_transverse_temporal_winding'] for c in cases):.3g}; omitted-image bounds per chain are reported; this is NOT exhaustive winding-sector proof.",
        'Raw stabilized condition-number0 is explicitly UNMEASURED, final summary null; no clean-conditioning claim from that sentinel.',
        'No one-q curvature=photon mass; no real-time/analytic-continuation, Maxwell speed, Lorentz, groundstate, equilibrium, Nature or joined-Band40 conclusion.',
        'Failures retained: initial wrong matter-only finite-difference control, unavailable zip binary, corrected edit syntax, early per-path inversion API, deliberate light-rotor image failure.',
        f"Selected Pyright exit{lang['natural_exit']}, errors{lang['diagnostics']['summary']['errorCount']}, files{len(lang['selected_files'])}; actual rooted LSP symbols/definition exit{lang['rooted_LSP']['natural_exit']}; positive AST documented.",
        'Natural scientific exits: '+', '.join(f"{s}={exits[s]['natural_exit']}" for s in stages)+'. Source/input/capsule hashes and streams are retained per stage.',
        'Exact law/derivation: *_derivation.txt; pure API/field meanings: *_interface.json; all new sources: this prefix only, frozen ancestors unchanged.',
        'Raw thermal states and response arrays: *_b*_raw.npz; same-path compact field/current arrays: *_b*_field_raw.npz; complete numbers: *_response_summary.json.',
        'Delivery capsule contains source snapshots, raw physical arrays, exits/streams, diagnostics, failures and manifest. Parent retains ownership of physical synthesis and any publication.'
    ])
    assert len(lines)<=80,len(lines)
    (ROOT/(P+'synthesis.txt')).write_text('\n'.join(lines)+'\n')
    excluded={P+'delivery_capsule.zip',P+'manifest.json',P+'delivery_receipt.json'}
    paths=sorted(p for p in ROOT.glob(P+'*') if p.is_file() and p.name not in excluded)
    manifest=dict(kind='FULL_NEW_PHYSICAL_RESPONSE_DELIVERY',synthesis_lines=len(lines),
                  files={p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in paths})
    (ROOT/(P+'manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    capsule=ROOT/(P+'delivery_capsule.zip')
    with zipfile.ZipFile(capsule,'w',zipfile.ZIP_DEFLATED) as z:
        for path in paths:z.write(path,path.name)
        z.write(ROOT/(P+'manifest.json'),P+'manifest.json')
    receipt=dict(capsule=str(capsule),sha256=hashlib.sha256(capsule.read_bytes()).hexdigest(),bytes=capsule.stat().st_size,
                 manifest_sha256=hashlib.sha256((ROOT/(P+'manifest.json')).read_bytes()).hexdigest(),synthesis_lines=len(lines),
                 cases=len(cases),scientific_exits={s:exits[s]['natural_exit'] for s in stages})
    (ROOT/(P+'delivery_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
