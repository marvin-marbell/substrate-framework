"""Compact outcome synthesis plus exact manifest and sealed output capsule."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parent
P = 'rich_gauss_transverse_20261015_'


def read(name: str) -> Any:
    return json.loads((ROOT/(P+name+'.json')).read_text())


def write(name: str, value: Any) -> None:
    (ROOT/(P+name+'.json')).write_text(json.dumps(value,indent=2)+'\n')


def decode(value: Any) -> Any:
    return np.asarray(value['real'])+1j*np.asarray(value['imag'])


def z(value: Any) -> list[float]:
    return [float(np.real(value)),float(np.imag(value))]


def main() -> None:
    result,settings,exit_record,language = read('result'),read('input'),read('exit'),read('language_final')
    if exit_record['returncode'] != 0 or not exit_record['sources_and_inputs_unchanged']:
        raise ValueError('Cannot deliver successful scientific execution without exact exit/input receipt')
    full,cut = result['full_mixed_hodge'],result['cut_mixed_hodge']
    connected,raw = decode(full['connected']),decode(full['complete'])
    cutc = decode(cut['connected'])
    channel_names = ['TT','TL','LT','LL']
    components = {name:dict(connected=z(connected[i,j]),complete=z(raw[i,j]),current='T' if i==1 else 'L',field='T' if j==1 else 'L')
                  for name,(i,j) in zip(channel_names,[(1,1),(1,2),(2,1),(2,2)])}
    collective = result['collective']
    fc,cc = collective['full'],collective['cut']
    fcvalue = complex(decode(fc['connected'])[0,0])
    error = fc['numerical_resolution']['connected_max_abs_difference']
    synthesis = dict(kind='ACTUAL_EXACT_FULL_BOUNDARY_HODGE_AND_NEW_COLLECTIVE_TWO_FIELD_CONNECTED_PROCESS',
        governor={key:settings[key] for key in ('plan_id','captured_revision','template','work_item','parent_sole_plan_writer')},
        law=settings['law'],decision='Retain scoped finite microscopic independent-cycle field/charged-source connected dynamics; do not rename mixed edge vertex or cycle channels photon scattering.',
        input=dict(model=settings['model'],model_SHA256=settings['model_SHA256'],reference=settings['reference'],reference_SHA256=settings['reference_SHA256'],
                   scope='Declared shared stationary Omega input; no independent global ground solve, no borrowed evolved arrays; old I16 cube, not new triangle frame law.'),
        exact_Hodge=dict(incidence_rank=8,edge_count=13,cycle_rank=5,exact_rational_projector_identities=True,
            equations=P+'equations.json',meaning='All physical edges including unchanged outward boundary. EL is fixed by density; ET consists of independent graph-cycle coordinates, not statistically independent matter/field states. Hcut changes current-active edges, never graph/projectors.'),
        first_channel=dict(original_edge_complete=z(raw[0,0]),original_edge_connected=z(connected[0,0]),
            axis_order='Rows current [edge,T,L], columns F [edge,T,L]. TT/TL/LT/LL are additive COMPLEX amplitudes, not disjoint probabilities.',
            components=components,independent_field_T_with_full_current=z(connected[0,1]),Gauss_density_field_L_with_full_current=z(connected[0,2]),
            full_connected_reconstruction_error=full['connected_reconstruction_error'],full_complete_reconstruction_error=full['complete_reconstruction_error'],
            raw_observable_retained='Both F and jQ decomposed; no component discarded and no source-range constraint imposed. Review28 original complex connected edge vertex reconstructed.',
            numerical_resolution=full['numerical_resolution'],first_finding=P+'first_channel.json'),
        first_symmetry_null_retained=settings['null_retained'],
        cut_semantics=dict(full_edge_connected=z(cutc[0,0]),decomposed_four_connected={name:z(cutc[i,j]) for name,(i,j) in zip(channel_names,[(1,1),(1,2),(2,1),(2,2)])},
            reconstruction_error=cut['connected_reconstruction_error'],numerical_resolution=cut['numerical_resolution'],
            meaning='Actual active bridge current is zero, but nonlocal T/L pieces need not vanish and can cancel. Both neutral bra and disconnected terms evolved with Hcut; never full-H stationary phase shortcut.'),
        actual_NEW_collective_process=dict(operator=collective['protocol'],profile=collective['profile'],profile_divergence_max=collective['divergence_max'],
            complete=fc['complete'],connected=fc['connected'],subtraction=fc['subtraction'],
            magnitude=abs(fcvalue),connected_method_difference=error,resolved_nonzero=bool(abs(fcvalue)>100*max(error,1e-18)),
            raw_B0_strength=fc['raw_source_strength'],raw_B4_strength=fc['raw_sink_strength'],neutral_physical_F_strength=collective['neutral_F_raw_strength'],
            charged_physical_F_on_raw_source_strength=collective['charged_F_on_raw_source_strength'],
            numerical_resolution=fc['numerical_resolution'],equal_time_complete=fc['equal_time_all_zero']['complete'],equal_time_connected=fc['equal_time_all_zero']['connected'],
            distinct_minus_equal_connected=fc['distinct_minus_equal_connected'],
            field_H_commutator_max=collective['H_F_commutator_max'],source_field_commutator_max=collective['source_commutator_max'],sink_field_commutator_max=collective['sink_commutator_max'],
            dressing_interpretation='Equal-time cumulant retained separately; changing unequal-time amplitude and nonzero [H,F] demonstrate actual H dynamics, not a proof of asymptotic scattering. Source/field commutators are explicitly measured; equal-time reference correlations need not vanish even when they commute.',
            same_input_cut_complete=cc['complete'],same_input_cut_connected=cc['connected'],cut_numerical_resolution=cc['numerical_resolution'],
            cut_interpretation='A two-field correlation need not vanish under cut and is not used as a false transport-defeat test.',
            mode_scope=collective['mode_relation'],microscopic=collective['microscopic']),
        causal_evidence_scope=settings['prior_operational'],numerical_method=result['numerical_method'],
        execution=dict(returncode=exit_record['returncode'],natural_exit=exit_record['natural_exit'],elapsed_seconds=exit_record['elapsed_seconds'],sources_and_inputs_unchanged=exit_record['sources_and_inputs_unchanged']),
        language=dict(selected_files=language['selected_files'],returncode=language['returncode'],summary=language['summary'],rooted_LSP_exit=language['rooted_LSP']['server_exit'],scope=language['scope']),
        artifacts=dict(result=P+'result.json',generated_full_states=P+'states.npz',source_input_capsule=P+'source_capsule.zip',binding=P+'start.json',exit=P+'exit.json',interface=P+'interface.json'),
        limits=result['scope']['limits'])
    write('synthesis',synthesis)
    suffixes = ['input.json','consume.py','capture.py','deliver.py','interface.json','equations.json','first_channel.json','result.json','states.npz','start.json','exit.json','stdout.txt','stderr.txt','source_capsule.zip','language_pre.json','language_final.json','synthesis.json']
    names = [P+s for s in suffixes]
    manifest = {name:dict(SHA256=hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),bytes=(ROOT/name).stat().st_size) for name in names}
    write('manifest',dict(kind='EXACT_OWNED_DELIVERY_BYTES',files=manifest,source_input_manifest=read('start')['input_SHA256']))
    names.append(P+'manifest.json')
    capsule = ROOT/(P+'delivery_capsule.zip')
    with zipfile.ZipFile(capsule,'w',zipfile.ZIP_DEFLATED) as cap:
        for name in names:
            cap.write(ROOT/name,name)
    write('delivery_receipt',dict(capsule=capsule.name,SHA256=hashlib.sha256(capsule.read_bytes()).hexdigest(),bytes=capsule.stat().st_size,
        manifest=P+'manifest.json',synthesis=P+'synthesis.json',publication='Local ownprefix only; no graph/public/git/plan/settings/services action'))
    print(json.dumps(dict(capsule=capsule.name,SHA256=hashlib.sha256(capsule.read_bytes()).hexdigest(),collective_connected=fc['connected'],resolved_nonzero=synthesis['actual_NEW_collective_process']['resolved_nonzero'])),flush=True)


if __name__ == '__main__':
    main()
