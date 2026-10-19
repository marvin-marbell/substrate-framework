"""Compact honest physical synthesis plus exact artifact manifest/capsule."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent
P='rich_dressed_local_carrier_20261015_'


def load(suffix):
    return json.loads((ROOT/(P+suffix+'.json')).read_text())


def dump(suffix,value):
    (ROOT/(P+suffix+'.json')).write_text(json.dumps(value,indent=2)+'\n')


def main():
    interface=load('interface')
    dynamics=load('dynamics_result')
    branches={}
    for branch,prepared in interface['branches'].items():
        evolved=dynamics['branches'][branch]
        branches[branch]=dict(physical_source_strength=prepared['physical_source_strength'],
            filtered_source_strength=prepared['filtered_source_strength'],
            bare_energy=prepared['bare_energy'],dressed_energy=prepared['selected_energy'],
            initial_excess=evolved['excess'][0],final_excess=evolved['excess'][-1],
            coherent_bare_source_span_weights=[r['coherent_source_span_weight'] for r in evolved['source_resolution']],
            orthogonal_weights=[r['orthogonal_weight'] for r in evolved['source_resolution']],
            final_shell_population=evolved['absolute'][-1]['any_Ecap_shell'],
            initial_outer_radial=prepared['selected_initial']['outer_radial_population'],
            final_outer_radial=evolved['absolute'][-1]['outer_radial_population'],
            accuracy=evolved['accuracy'],filter_accuracy=prepared['cooling']['normalized_state_comparison'],
            source_partner_residual=prepared['bare_partner_lowering_residual'],
            Jhalf_residual=prepared['total_Jhalf_residual'],H_CSR_SHA256=evolved['H_CSR_SHA256'])
    synthesis=dict(kind='ACTUAL_FULL_H_QUANTUM_DRESSED_CHARGED_PACKET_WITH_BACKGROUND_AND_TAILS',
        decision='Unchanged finite physical law, genuinely new lower-energy correlated neutral plus charged source preparation. Dynamics and source-coherent/orthogonal channel weights below are measurements, not rigid all-field identity or measured electron.',
        state_file=interface['state_file'],state_SHA256=interface['state_SHA256'],
        reference_seed=interface['neutral']['seed'],new_reference=interface['neutral']['selected'],
        neutral_cooling=interface['neutral']['cooling'],neutral_J_residual=interface['neutral']['total_J_generator_residual'],
        fixed_law='rich_local_frame_carrier_20261015_model and literal fullCAR12 N5/6/7, product local frames, positive radial domain, material rotations; no constants changed',
        preparation='Omega=normalized exp(-3H6) seed. chi=exp[-.35(H_N-E_Omega)] B_v,alpha Omega, normalized only as conditional state. Common shift and raw norms preserved.',
        background_filter_caveat='The source energy filter also cools the nonstationary neutral background, not exclusively a local carrier cloud. Original beta3 baseline retained here; independent Review32 separately computes matched beta3+.35 neutral and ray-distance bounds.',
        preserved_correction='Attempt1 exact source/state/exit capsule retains wrong relative doublet phase. Final source divides total-J lowering by measured c=-1 and residuals are recorded; no evolved output injected.',
        times=dynamics['times'],branches=branches,
        field_only_intervention=dict(operator=dynamics['field_intervention']['operator'],
            work=dynamics['field_intervention']['work'],
            initial_response=dynamics['field_intervention']['same_state_full_response'][0],
            final_response=dynamics['field_intervention']['same_state_full_response'][-1],
            accuracy=dynamics['field_intervention']['accuracy']),
        decompositions='Raw coherent-source span / orthogonal / interference for H,current,r2 retained. Complement contains dressing, incoherent and possible radiative/ambient excitations; no false radiated-only projector on this finite triangle.',
        raw='preparation_result.json, dynamics_result.json, states.npz and dynamics_states.npz under owner prefix retain alternatives, counts and joint field/spin/count statistics; no small conditioned component normalized into whole particle.',
        controls='Cooling changes state, NOT same-state causal control. exp(i .31 E0) same-state field-only intervention has initial count/spin/r2 unchanged and actual later current response. Independent reviewer owns different-spin/time full-H and vector-off consumer.',
        diagnostics=load('language_final')['summary'],
        actual_process_exits=dict(prepare=load('prepare_exit'),dynamics=load('dynamics_exit')),
        limitations=['Prepared neutral remains conditional, not global interacting vacuum or stationary theorem; propagate it explicitly.',
            'Finite positive three-node radial grid with large outer-node occupation, electric cap2 with shell population, material l0+l1; no regulator convergence.',
            'CAR/statistics/physical frame bundle and ordered matrix orbit are constitutive assumptions; no measured electron, mass, positron, continuous stability, free COM, Lorentz or Maxwell claim.',
            'Full-H scalar energy filtering can spread initial charge. Source-coherent weights are one declared resolution, not unique particle probability; matrix norm may remain mostly ambient.'])
    dump('synthesis',synthesis)
    exclusions={P+'manifest.json',P+'delivery_receipt.json',P+'delivery_capsule.zip'}
    selected=sorted(p for p in ROOT.glob(P+'*') if p.is_file() and p.name not in exclusions and not p.name.endswith('_source_capsule.zip'))
    # Include exact inherited primitive source files, never their evolved results.
    selected += [ROOT/n for n in json.loads((ROOT/(P+'input.json')).read_text())['source_files'] if not n.startswith(P)]
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in selected}
    dump('manifest',dict(kind='EXACT_DRESSED_CARRIER_DELIVERY_MANIFEST',files=hashes))
    with zipfile.ZipFile(ROOT/(P+'delivery_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as capsule:
        for path in selected:
            capsule.write(path,path.name)
        capsule.write(ROOT/(P+'manifest.json'),P+'manifest.json')
    receipt=dict(capsule=P+'delivery_capsule.zip',SHA256=hashlib.sha256((ROOT/(P+'delivery_capsule.zip')).read_bytes()).hexdigest(),
        files=len(selected)+1,source_input_state_hashes_in_manifest=True,actual_prepare_returncode=load('prepare_exit')['returncode'],
        actual_dynamics_returncode=load('dynamics_exit')['returncode'])
    dump('delivery_receipt',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
