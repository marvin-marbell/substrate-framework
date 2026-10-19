"""Own raw-review extraction, once-final selected diagnostics and <=80-line synthesis."""
from __future__ import annotations
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
import zipfile
import numpy as np
import scipy
from rich_review_wave12_20261015_diagnose import language_server

ROOT = Path(__file__).resolve().parent
P = 'rich_review_wave34_20261016_'
PY = '/home/administrator/substrate-framework/.venv/bin/python'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array(value: Any) -> Any:
    return np.asarray(value['real'])+1j*np.asarray(value['imag'])


def packed(value: Any) -> dict:
    return dict(real=np.asarray(value).real.tolist(), imag=np.asarray(value).imag.tolist())


def dump(suffix: str, value: Any) -> None:
    (ROOT/(P+suffix+'.json')).write_text(json.dumps(value, indent=2)+'\n')


def main() -> None:
    result = json.loads((ROOT/(P+'result.json')).read_text())
    exit_receipt = json.loads((ROOT/(P+'exit.json')).read_text())
    if exit_receipt['returncode'] != 0 or not exit_receipt['sources_and_inputs_unchanged']:
        raise ValueError('Actual successful unchanged-source natural exit is required')
    selected = [P+name+'.py' for name in ['consume', 'capture', 'deliver']]
    command = ['/home/administrator/.local/bin/pyright', '--pythonpath', PY, '--outputjson', *selected]
    done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    checker = json.loads(done.stdout)
    diagnostics = dict(command=command, natural_exit=True, returncode=done.returncode, checker=checker, stderr=done.stderr,
        selected=selected, source_SHA256={name: sha(ROOT/name) for name in selected},
        versions=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
            pyright=subprocess.run(['/home/administrator/.local/bin/pyright', '--version'], capture_output=True, text=True, check=True).stdout.strip()),
        LSP_helper_SHA256=sha(ROOT/'rich_review_wave12_20261015_diagnose.py'),
        errors_by_file=dict(Counter(Path(item['file']).name for item in checker.get('generalDiagnostics', []) if item['severity']=='error')),
        rooted_LSP=language_server([P+'consume.py', P+'deliver.py']),
        positive_AST={name: [dict(name=node.name, line=node.lineno) for node in ast.walk(ast.parse((ROOT/name).read_text())) if isinstance(node, (ast.FunctionDef, ast.ClassDef))] for name in selected})
    dump('language_final', diagnostics)
    reviewed = {}
    partition_arrays = {}
    zero_partitions = ['123|4', '12|34', '12|3|4', '13|24', '13|2|4', '1|234', '1|23|4', '1|24|3', '1|2|34', '1|2|3|4']
    for branch, baselines in result['branches'].items():
        reviewed[branch] = {}
        for baseline, data in baselines.items():
            schedules = data['schedules']
            entry = dict(raw_source_weights=data['source_filtered_raw_weights'], bare_source_weights=data['source_bare_weights'],
                         source_spin=data['actual_source_spin'], own_H_cycle_probe=data['own_H_cycle_probe'], orders={})
            for order in ['J_E', 'E_J']:
                first, second = order.split('_')
                c0 = array(schedules['0.0']['orders'][order]['connected'])
                cf = array(schedules['0.42']['orders'][order]['connected'])
                entry['orders'][order] = dict(equal_time_connected_Frobenius=float(np.linalg.norm(c0)), finite_time_connected_Frobenius=float(np.linalg.norm(cf)),
                    changed_from_initial_cloud_Frobenius=float(np.linalg.norm(cf-c0)), finite_time_connected_matrix=packed(cf),
                    finite_time_raw_Frobenius=schedules['0.42']['orders'][order]['Frobenius_raw'],
                    spin_offdiagonal_max=schedules['0.42']['orders'][order]['spin_offdiagonal_max'],
                    spin_diagonal_difference_max=schedules['0.42']['orders'][order]['spin_diagonal_difference_max'])
                for total, schedule in schedules.items():
                    response = schedule['orders'][order]
                    pair = array(schedule['source_pair'])
                    m1, m2 = [array(v).item() for v in response['neutral_one_point']]
                    field_pair = array(response['neutral_two_point']).item()
                    products = {'1234': array(response['raw_four']), '14|23': -pair*field_pair,
                        '134|2': -array(schedule['three_point']['early_'+first])*m2,
                        '124|3': -array(schedule['three_point']['late_'+second])*m1,
                        '14|2|3': 2*pair*m1*m2}
                    products.update({name: np.zeros((4,4), complex) for name in zero_partitions})
                    residual = float(np.linalg.norm(sum(products.values())-array(response['connected'])))
                    key = ':'.join([branch, baseline, total, order])
                    partition_arrays[key] = dict(signed_cumulant_partition_products={name: packed(v) for name,v in products.items()},
                        reconstruction_residual=residual, odd_charge_blocks='Exact ZERO by orthogonality of fixed-N summands, not an approximation to their mean. Sources carry +/-1; no charge-changing field leg. All even blocks are actual new full-H measured contractions.')
            controls = data['controls']
            entry['control_delta_n'] = {name: value['delta_n'] for name, value in controls.items()}
            entry['field_pulse_minus_full_receiver_n_at_t0'] = controls['cycle_field_pulse']['observations'][0]['n'][1]-controls['full']['observations'][0]['n'][1]
            entry['field_pulse_minus_full_receiver_n_at_final'] = controls['cycle_field_pulse']['observations'][-1]['n'][1]-controls['full']['observations'][-1]['n'][1]
            entry['field_pulse_work'] = controls['cycle_field_pulse']['observations'][0]['energy']['energy']-controls['full']['observations'][0]['energy']['energy']
            entry['field_pulse_initial_current_change'] = (np.array(controls['cycle_field_pulse']['observations'][0]['currents'])-controls['full']['observations'][0]['currents']).tolist()
            entry['continuity_max'] = max(max(v['continuity_operator_on_initial_residual']) for v in controls.values())
            entry['Gauss_count_transfer_max'] = max(max(abs(a) for a in v['exact_Gauss_count_transfer_residual']) for v in controls.values())
            entry['quadrature_residual_max'] = max(max(abs(a) for a in v['count_minus_integrated_current']) for v in controls.values())
            entry['initial_total_n'] = controls['full']['observations'][0]['n']
            entry['contact_connected_Frobenius'] = float(np.linalg.norm(array(schedules['0.42']['contact_connected'])))
            reviewed[branch][baseline] = entry
    dump('graded_partitions', partition_arrays)
    propagation_errors = []
    filter_errors = []
    for report in result['numerics']:
        if 'same_state_accuracy_comparison' in report:
            propagation_errors.extend(report['same_state_accuracy_comparison']['state_norm_differences'])
        if 'normalized_state_comparison' in report:
            filter_errors.extend(report['normalized_state_comparison'])
    numerical = dict(propagation_ray_18vs24_or20vs28_max=max(propagation_errors), filter_ray_30vs40_max=max(filter_errors),
                     number_of_full_space_chains=len(result['numerics']), diagnostics_exit=done.returncode, diagnostics_summary=checker.get('summary'))
    findings = dict(kind='ACTUAL_NEW_DRESSED_NONFORWARD_TWO_FIELD_CURRENT_RUNTIME_REVIEW',
        physical_model_sha256=result['input']['expected_input_SHA256']['rich_local_frame_carrier_20261015_model.py'],
        declared_neutral_state_sha256=result['input']['expected_input_SHA256']['rich_dressed_local_carrier_20261015_states.npz'],
        provenance=result['input']['provenance'], original_neutral=result['original_neutral'], matched_neutral=result['matched_neutral'],
        matched_ray_distance=result['matched_ray_distance'], branches=reviewed, numerical=numerical,
        grading='All15 partitions explicitly retained, odd-charge block means exactly zero by direct-sum superselection. All nonzero 1/2/3/4-point blocks newly contracted with actual absolute-time evolving neutral; raw means and products in result/graded_partitions.',
        spin='Full2x2 physical LEFT spin and2x2 root orbital matrix; scalar cycle legs must not manufacture spinflip. Jminus source phase is measured -1; total J includes material angular.',
        interpretation='Finite time response differs from equal-time connected cloud; difference is NOT an asymptotic scattering amplitude. Both temporal placements E_late J_early and J_late E_early are explicit. Positive/negative charge branches are addition/hole, not Nature species.',
        controls='Same full-H filtered prep for full/pathcut/vectoroff/fieldpulse. Bare prep compared to own t0. Complete U-dressed paths cut, never barefermionhop without U. Origin freezing is meaningful transfer defeat; residual receiver redistribution can survive initial spread. Receiver n1 is actual total count, not exclusive particle probability; field pulse injects current/work. A deleted-current zero alone is not evidence.',
        limits='One triangle cycle, not two polarizations. Ecap2, radial3, material l0+l1; no regulator convergence, stationary vacuum, massshell crosssection, photon, Nature electron/positron or stability theorem.',
        raw_result=P+'result.json', raw_partitions=P+'graded_partitions.json', runtime_exit=exit_receipt)
    dump('physical_findings', findings)
    compact = {
        'kind': findings['kind'],
        'model': findings['physical_model_sha256'],
        'state': findings['declared_neutral_state_sha256'],
        'executed': 'NEW fullH N5/6/7 source0 -> receiver1, both root orbitals/spins, E/J both orders, times0/.12/.26/.42, contact and own-H cycle derivative probes.',
        'source': 'F_N B at source, receiver Bdag F_N on arbitrary U6(T)Omega; literal Gauss tree/boundary; raw filter weights are not Born probabilities. No intermediate source projection.',
        'neutral': dict(original=result['original_neutral'], matched=result['matched_neutral'], matched_ray_distance=result['matched_ray_distance']),
        'graded': findings['grading'],
        'spin': findings['spin'],
        'connected_norm_legend': 'Each JE/EJ triple = [equal-time cloud norm, finite-time norm, norm of difference]; Frobenius over full spin/root4x4.',
        'interpretation': findings['interpretation'],
        'controls': findings['controls'],
        'numerics': numerical,
        'limits': findings['limits'],
        'runtime': dict(natural_exit=exit_receipt['natural_exit'], returncode=exit_receipt['returncode'], unchanged=exit_receipt['sources_and_inputs_unchanged'], elapsed=exit_receipt['elapsed_seconds']),
        'raw': [P+name for name in ['physical_findings.json', 'result.json', 'graded_partitions.json', 'input.json', 'equations.json', 'exit.json', 'language_final.json']]
    }
    for branch, baselines in reviewed.items():
        for baseline, entry in baselines.items():
            key = branch+'_'+baseline
            compact[key+'_raw_filter_weights'] = entry['raw_source_weights']
            compact[key+'_connected_norms'] = {order: [measurement[name] for name in ['equal_time_connected_Frobenius', 'finite_time_connected_Frobenius', 'changed_from_initial_cloud_Frobenius']] for order,measurement in entry['orders'].items()}
            compact[key+'_spin_contact'] = dict(spinflip_max=max(v['spin_offdiagonal_max'] for v in entry['orders'].values()), spin_diagonal_difference=max(v['spin_diagonal_difference_max'] for v in entry['orders'].values()), contact_connected_norm=entry['contact_connected_Frobenius'])
            compact[key+'_transfer'] = dict(initial_n=entry['initial_total_n'], delta_n_full=entry['control_delta_n']['full'], delta_n_cut=entry['control_delta_n']['path_cut'], delta_n_vectoroff=entry['control_delta_n']['vector_off'])
            compact[key+'_field_feedback'] = dict(receiver_n_pulse_minus_full=entry['field_pulse_minus_full_receiver_n_at_final'], pulse_work=entry['field_pulse_work'], continuity_max=entry['continuity_max'], Gauss_transfer_max=entry['Gauss_count_transfer_max'])
    # One compact key per line keeps the synthesis below80 lines without hiding raw arrays in it.
    text = '{\n'+',\n'.join('  '+json.dumps(key)+': '+json.dumps(value) for key,value in compact.items())+'\n}\n'
    (ROOT/(P+'synthesis.json')).write_text(text)
    names = sorted(path.name for path in ROOT.glob(P+'*') if path.is_file() and not path.name.endswith(('delivery_capsule.zip', 'delivery_receipt.json', 'manifest.json')))
    manifest = dict(files={name: sha(ROOT/name) for name in names}, synthesis_lines=len(text.splitlines()))
    dump('manifest', manifest)
    capsule = ROOT/(P+'delivery_capsule.zip')
    with zipfile.ZipFile(capsule, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names+[P+'manifest.json']:
            archive.write(ROOT/name, name)
    dump('delivery_receipt', dict(capsule=capsule.name, sha256=sha(capsule), source_capsule=exit_receipt['source_capsule'],
        synthesis=P+'synthesis.json', synthesis_lines=len(text.splitlines()), runtime_returncode=exit_receipt['returncode'], diagnostics_returncode=done.returncode))
    print(json.dumps(dict(complete=True, synthesis_lines=len(text.splitlines()), diagnostics_exit=done.returncode, numerical=numerical, capsule_sha256=sha(capsule))), flush=True)


if __name__ == '__main__':
    main()
