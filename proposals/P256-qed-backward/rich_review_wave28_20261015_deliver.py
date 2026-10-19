"""Project this own executed review into compact synthesis and sealed capsule."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parent
P = 'rich_review_wave28_20261015_'


def load(name: str) -> Any:
    return json.loads((ROOT/(P+name+'.json')).read_text())


def save(name: str, value: Any) -> None:
    (ROOT/(P+name+'.json')).write_text(json.dumps(value,indent=2)+'\n')


def main() -> None:
    result,receipt,lang = load('result'),load('exit'),load('language_final')
    selected,selected_receipt = load('selected_result'),load('selected_exit')
    op: dict[str, Any] = {}
    for name in ('full','cut'):
        first = result['operational'][name][0]['observables']
        last = result['operational'][name][-1]['observables']
        changes = {key:(np.asarray(last[key])-first[key]).tolist() for key in ('rho_right','physical_Q_right','EL','ER','H','own_HR')}
        excess_change = {key:float(np.diff(np.asarray(value))[0]) for key,value in changes.items()}
        op[name] = dict(initial_matter_density_difference_max=float(np.max(abs(np.diff(np.asarray(first['rho']),axis=0)))),
                        initial_physical_Q_right=first['physical_Q_right'], initial_EL=first['EL'],
                        endpoint_physical_Q_right=last['physical_Q_right'],own_t0_changes=changes,
                        pulse_minus_unpulsed_change=excess_change,
                        endpoint_flux_current=last['flux_current'], endpoint_physical_charge_current=last['physical_charge_current'],
                        endpoint_right_electric_power=last['right_electric_power'],
                        preparation_work=first['H'][1]-first['H'][0])
    numerical = {name:{key:value for key,value in result['numerical_resolution'][name].items() if key not in ('alternate_endpoint','alternate_correlation')} for name in ('full','cut')}
    numerical['method'] = result['numerical_resolution']['method']
    synthesis = dict(kind='INDEPENDENT_SAME_I16_CHARGED_FIELD_OPERATION_AND_COMPLETE_CONNECTED_CORRELATION',
                     governor=result['settings']['governor'], law_id=result['law_id'],
                     input_binding=dict(model=result['settings']['model'],model_SHA256=result['settings']['model_SHA256'],
                                        reference=result['settings']['reference'],reference_SHA256=result['settings']['reference_SHA256'],
                                        basis_hash=result['basis_hash'],dimensions=result['dimensions'],reference_scope=result['reference']),
                     verdict='Retain an observed scoped finite same-H charged/field interaction: disjoint pure-field intervention changes own-t0 charge motion and remote field energy, while actual path cut defeats charge transport. Complete and connected correlators remain unamputated amplitudes, never event probabilities or on-shell scattering. Exact cut regional conservation does not imply all correlations vanish.',
                     preparation='Fresh normalized B_source†Ω, primitive up addition at anchor0; pure-field exp[-i0.8 ImW_right] on opposite right yz face. Identical charged input in full/cut arms. No evolved ancestor arrays read.',
                     raw_source_weight=result['source_weight'], operational=op,
                     connected={name:selected[name] for name in ('full','cut')},correlation_times=result['settings']['correlation_times'],
                     selected_correlation_scope='One selected ordered complete B_4(t3) [-J_xbridge(t2)] [E_right_edge(t1)/16] B_0†(0) and full charge-allowed cumulant. Positive-frequency addition; no -i or timeordering. Actual two-sided neutral/charged cut evolution.',
                     first_circulation_null=dict(full=result['correlation']['full'],cut=result['correlation']['cut'],selection=selected['selection']),
                     selected_numerical_method=selected['numerical_method'],
                     signs_and_units=result['signs'], exact_boundary_identities=result['identities'],numerical_resolution=numerical,
                     independent_raw_review=dict(artifact=P+'raw_review.json',
                         conclusion='Frozen source/current/Gauss meanings support scoped mechanism, not physical spin. Authoritative neutral effective controls are control_correction_result.json. Carrier archived control endpoint currents still need FULL-H-diagnostic qualification; not used as own-control motion here.'),
                     execution=dict(operational=receipt,selected_correlation=selected_receipt),
                     diagnostics=dict(returncode=lang['returncode'],summary=lang['summary'],errors_by_file=lang['errors_by_file'],
                                      rooted_LSP_exit=lang['LSP']['server_exit'],scope=lang['scope']),
                     limits=result['limitations'],
                     artifacts=dict(input=P+'input.json',result=P+'result.json',first_result=P+'first_result.json',
                                    source=P+'consume.py',states=P+'states.npz',exit=P+'exit.json',
                                    selected_source=P+'selected.py',selected_result=P+'selected_result.json',selected_exit=P+'selected_exit.json',
                                    raw_review=P+'raw_review.json',language=P+'language_final.json',
                                    manifest=P+'manifest.json',capsule=P+'delivery_capsule.zip',receipt=P+'delivery_receipt.json'))
    save('synthesis',synthesis)
    members = sorted(path for path in ROOT.glob(P+'*') if path.is_file() and path.name not in (P+'manifest.json',P+'delivery_capsule.zip',P+'delivery_receipt.json'))
    hashes = {path.name:dict(SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size) for path in members}
    save('manifest',dict(kind='OWNED_WAVE28_ARTIFACT_MANIFEST',members=hashes,ancestor_inputs=receipt['input_SHA256']))
    with zipfile.ZipFile(ROOT/(P+'delivery_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as capsule:
        for path in members:
            capsule.write(path,path.name)
        capsule.write(ROOT/(P+'manifest.json'),P+'manifest.json')
    cap = ROOT/(P+'delivery_capsule.zip')
    save('delivery_receipt',dict(capsule=cap.name,SHA256=hashlib.sha256(cap.read_bytes()).hexdigest(),
                                bytes=cap.stat().st_size,consumer_exit=receipt['returncode'],natural_exit=receipt['natural_exit'],
                                selected_exit=selected_receipt['returncode'],selected_natural_exit=selected_receipt['natural_exit'],
                                sources_and_inputs_unchanged=receipt['sources_and_inputs_unchanged'] and selected_receipt['sources_and_inputs_unchanged']))
    print(json.dumps(dict(synthesis=P+'synthesis.json',operational=op,connected={name:selected[name] for name in ('full','cut')},receipt=load('delivery_receipt'))),flush=True)


if __name__ == '__main__':
    main()
