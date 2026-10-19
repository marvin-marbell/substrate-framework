"""Package the two independently executed physical reviews with bounded claims."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent;P='rich_review_wave45_20261016_'


def load(suffix):return json.loads((ROOT/(P+suffix)).read_text())


def main():
    coupled=load('coupled_result.json');source=load('coupled_source_result.json');cf=load('coupled_final_result.json')
    collective=load('collective_final_result.json');add=load('collective_addendum_result.json');pre=load('prelim_result.json');lang=load('language_final.json')
    verdict44=dict(kind='BOUNDED_CONSTRUCTIVE_PHYSICAL_VERDICT_44',decision='Working finite single-law charged-frame/uniform-field/current mechanism; not a bulk photon or localized spin/cloud result.',
        source_case=source,charged_branches=coupled['branches'],generator_and_impulse_revision=cf,physical_continuity_cut=load('coupled_continuity_result.json'),
        independently_observed=['Both charges freshly prepared from own propagated neutral input','Actual field-only impulses with work, current/electric feedback and hopping-off/HR-omitted Hamiltonians','Canonical source derivatives and full quadratic contact before projection','Frozen source37 mechanical/contact discriminator executed before new44 API'],
        unresolved=['Finite angular/field Galerkin with explicit full outgoing image; no convergence','Uniform noncompact cavity does not supply bulk dispersion/spatial propagation','Global total angular charge does not demonstrate localized spin/cloud transport','No Nature electron/proton/photon, mass from source energy, or g2 conclusion'],
        scientific_exits={k:load(k+'_exit.json') for k in ('prelim','coupled','coupled_source','coupled_final','coupled_continuity')},typing=dict(summary=lang['diagnostics']['summary'],selected_files=lang['selected_files'],LSP_exit=lang['rooted_LSP']['natural_exit']))
    verdict46=dict(kind='BOUNDED_CONSTRUCTIVE_PHYSICAL_VERDICT_46',decision='Exact finite-transfer/source identity independently supported on actual physical trace; fresh V64 response estimable, not equilibrated/continuum photon.',
        own_compact_cycle_trace=pre['transfer'],own_PH_Gauss_holonomy_trace=add['trace'],own_V64=collective,
        wrong_continuum_frequency=load('transfer_wrong_frequency.json'),free_bridge_control=load('collective_result.json')['free_control'],
        independently_observed=['Full Frechet interval vertices/contact beat actual point-current wrongcase','Electric zero Matsubara independent from source Hessian zero','Own V64 raw theta/holonomy/HS paths; no producer-output replay','Finalized all-frequency estimator rechecked only invalidated layers; original source captures retained'],
        unresolved=['Only 10 saved samples per independently generated chain, not mixing or small true MC errors','Material beta1 M4/M8 dependence; not Trotter convergence','Direct kinetic EE noisy and separate from exact source-linked ensemble estimator','Full current response/logdet feedback is not irreducible stiffness','No pole fit, ground-state or photon/mass/continuum conclusion'],
        scientific_exits={k:load(k+'_exit.json') for k in ('collective','collective_addendum','collective_final')},typing=dict(summary=lang['diagnostics']['summary'],selected_files=lang['selected_files'],LSP_exit=lang['rooted_LSP']['natural_exit']))
    for name,value in [('coupled_verdict.json',verdict44),('collective_verdict.json',verdict46)]:
        (ROOT/(P+name)).write_text(json.dumps(value,indent=2)+'\n')
    lines=[
        'TWO SEPARATE OWN EXECUTED PHYSICAL REVIEWS — NEW44 AND NEW46; NOT A PERMISSION GATE.',
        'Contract0199; shared issue222-rich-backward-candidates-20261014 / revision673b0253; no child plan/public/graph changes.',
        '44: NEW Hlength=H40+(Pi+D(R,n))²/(2I), three noncompact uniform lab-vector modes; no Maxwell/q² restorer.',
        'X=R(site-.5), D=sum(n-ions)X; r remains matrix norm. x/y transverse to chosen z, not bulk photons.',
        'Own frozen40/37 preliminary: Gauss0, mechanical source norm .055998, premature-contact error .021889.',
        'Own canonical source finite-A H differentiation holds canonical momenta fixed, not mechanical momenta.',
        f"44 derivative/contact errors at step.001: {source['records'][-1]['full_first_derivative_error']:.3g}/{source['records'][-1]['full_contact_derivative_error']:.3g}; deleting actual HR source misses {source['records'][-1]['actual_missing_mechanical_H_source_error']:.6g}.",
        'V=d_AHv=+i[D,H40]; physical J=-V. D² and source contact products made before angular projection.',
        'Own neutral evolved .21 under actual coupled H, then both actual addition/removal sources normalized separately.',
    ]
    for b in coupled['branches']:
        lines.append(f"44 N{b['number']} Q{b['net_charge']:+d}, dim{b['dimension']}, source strength{b['source_strength']:.6g} (not Born probability), Gauss{b['Gauss_error']:.1g}.")
        lines.append(f"44 N{b['number']} source mechanical norm{b['mechanical_source_vertex_norm']:.6g}; contact compression debt{b['contact_full_vs_compressed_error']:.6g}; angular exterior Hnorm{b['initial_full_angular_exterior_H_norm']:.6g}.")
        lines.append(f"44 N{b['number']} full physical source vs projected i[D,HP] complement{b['source_mechanical_hop_decomposition_error']:.6g}; this is NOT a mechanical-vertex failure.")
        for p in b['protocols']:
            lines.append(f"44 N{b['number']} pol{p['polarization']}: kickwork{p['kick_work']:.6g}, dJ={p['final_delta_current']}, dE={p['final_delta_electric']}.")
            lines.append(f"44 N{b['number']} pol{p['polarization']}: true HR-omitted/hopping-off law differences{p['actual_omitted_HR_law_difference']:.3g}/{p['actual_hopping_off_law_difference']:.3g}; retarded FD error{p['retarded_timeorder_error']:.3g}; field exterior{p['exterior']['field_exterior_norm']:.3g}.")
    lines.extend([
        '44 impulses are exact field-only exp(i eps PqP) in the retained regulator; full-unbounded impulse images/tails separately recorded.',
        'Initial density/current versus later feedback distinguished; full-CCR work comparison kept, never silently equated.',
        '44 initial field J used prematurely projected q/p products; source snapshots retained, final only invalidated J observable rechecked.',
        'Final J uses full-before-P products; global rotational closure does not prove localized spin/cloud transport.',
        '44 full-source J versus projected dotE complement is explicitly measured on own actual states; no false exact Maxwell identity at these cuts.',
        '44 corrected total-J commutator <9e-15; premature q/p projection gave .0036-.0142. Full impulse discarded weight ~6.94e-6, retained-state difference~6.21e-5.',
        '44 limits: noncompact uniform finite cavity, finite Galerkin, no spatial photon propagation or regulator convergence.',
        '46 independent own physical compact-cycle transfer: two sites/two links, PH/Gauss dimension40, beta1.4 M5 cap3.',
        f"46 exact discrete Ward max{pre['transfer']['exact_discrete_Ward_error']:.3g}; wrong point-vertex error{pre['transfer']['wrong_point_vertex_Ward_error']:.6g}.",
        f"46 independent electric n0={pre['transfer']['zero_mode_EE']:.6g}, source Hessian n0~{pre['transfer']['source_Hessian_zero_mode']:.1g}; never divide n0 by frequency.",
        f"46 own same-charge PH/Gauss/holonomy exact trace rel{add['trace']['relative_error']:.3g}; wrong nonconjugate/unprojected rel{add['trace']['wrong_nonconjugate_relative_error']:.3g}/{add['trace']['wrong_unprojected_relative_error']:.3g}.",
        '46 CEE_n=-sourceHessian_n/[4sin²(pi n/M)] for n!=0, not continuum (K-JJ)/omega².',
        '46 source Hessian retains full determinant feedback, interval Frechet vertices/contact and compact temporal holonomy/winding.',
        '46 final all n independently measured; complex tau and qminus swap retained; early copied-negative-n arrays superseded only.',
    ])
    for c in collective['cases']:
        vals=[round(z[1]*c['dt'],7) for z in c['EE_matsubara']['mean_real']]
        lines.append(f"46 own V64 beta{c['beta']:g} M{c['M']} seed{c['seed']}: 1200 moves/400 burn/10 saves; dt*CEE_n1(qpi/2 y,z; qpi y,z)={vals}.")
    lines.extend([
        '46 beta1 M4/M8 materially differ; colder beta2 is a new physical state, not convergence or vacuum proof.',
        '46 direct kinetic EE remains much noisier; free bridge N1000 has exact n0=64 and nonzero frequencies compatible with zero within noise.',
        '46 five-block errors are descriptive only: short chains, no mixing/equilibration or winding coverage claim.',
        '46 full-vs-logdet Hessian differs; reducible current response is not irreducible stiffness; no fitted oscillator/photon/pole.',
        f"Selected diagnostics: {lang['diagnostics']['summary']}; rooted LSP natural exit {lang['rooted_LSP']['natural_exit']}; positive AST recorded. Not type-clean unless errors=0.",
        'All science exit receipts, immutable first/superseded attempts, raw states/arrays, exact source snapshots/inputs are in two capsules.',
    ])
    if len(lines)>80:raise ValueError('physical synthesis exceeds requested80 lines')
    (ROOT/(P+'synthesis.txt')).write_text('\n'.join(lines)+'\n')
    shared=[ROOT/(P+s) for s in ('synthesis.txt','language_final.json','language_preflight.json','prelim.py','prelim_result.json','prelim_input.json','prelim_exit.json','prelim_stdout.txt','prelim_stderr.txt','execute.py','language.py','deliver.py')]
    receipts={}
    for case,patterns in [('coupled',['coupled*','prelim_source_raw.npz','prelim_failed_zero_source_capsule.zip']),('collective',['collective*','prelim_transfer_raw.npz','transfer_wrong_frequency.json'])]:
        files={p for pat in patterns for p in ROOT.glob(P+pat)}|set(shared)
        files={p for p in files if p.is_file() and not p.name.endswith('_delivery_capsule.zip')}
        manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
        mp=ROOT/(P+case+'_manifest.json');mp.write_text(json.dumps(manifest,indent=2)+'\n');files.add(mp)
        target=ROOT/(P+case+'_delivery_capsule.zip')
        with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(files):z.write(p,p.name)
        receipts[case]=dict(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),files=len(files))
    (ROOT/(P+'delivery_receipt.json')).write_text(json.dumps(dict(capsules=receipts,synthesis_lines=len(lines)),indent=2)+'\n')
    print(json.dumps(dict(capsules=receipts,synthesis_lines=len(lines)),indent=2),flush=True)


if __name__=='__main__':main()
