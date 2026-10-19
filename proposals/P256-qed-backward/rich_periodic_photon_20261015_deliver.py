"""Summarize exact owned finite experiment and package revision-bound evidence."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_periodic_photon_20261015_'


def read(suffix):
    return json.loads((ROOT/(PREFIX+suffix+'.json')).read_text())


def q_summary(response):
    gram = np.asarray(response['source_gram'])
    groups = []
    for group in response['groups']:
        matrix = np.asarray(group['source_gram'])
        transverse = matrix[:2,:2]
        eigenvalues = np.linalg.eigvalsh(transverse)
        weight = float(np.trace(transverse))
        if weight < 1e-7*float(np.trace(gram[:2,:2])):
            continue
        groups.append(dict(omega=group['omega'], multiplicity=group['multiplicity'],
                           transverse_source_rank=int(np.sum(eigenvalues > 1e-7*float(np.trace(gram[:2,:2])))),
                           transverse_gram=transverse.tolist(),transverse_weight=weight,
                           transverse_weight_fraction=weight/float(np.trace(gram[:2,:2])),
                           longitudinal_weight=float(matrix[2,2]),ritz_residual=group['residual_max']))
    dominant = sorted(groups,key=lambda g:g['transverse_weight'],reverse=True)[:8]
    low = sorted(groups,key=lambda g:g['omega'])[:8]
    return dict(q=response['q'],polarizations=response['polarizations'],
                transverse_source_gram=gram[:2,:2].tolist(),
                transverse_source_eigenvalues=np.linalg.eigvalsh(gram[:2,:2]).tolist(),
                actual_transverse_source_rank=int(np.linalg.matrix_rank(gram[:2,:2],tol=1e-12)),
                longitudinal_field_strength=float(gram[2,2]),
                dominant_transverse_groups=dominant,lowest_supported_transverse_groups=low,
                density_longitudinal_identity=response['longitudinal_density_identity_max'],
                explicit_primitive_current_identity=response['explicit_primitive_charge_current_identity_max'],
                joint_source_eigenvalues=response['joint_source_gram_eigenvalues'],
                signed_contact=response['twist_contact'],hopping_off_proper_baseline_response=response['hopping_off_proper_baseline_retarded_max'],
                block_dimension=response['dimension'],block_depth=response['depth'],orthogonality_error=response['orthogonality_error'],
                spectral_arrays=response['joint_spectral_arrays'])


def main():
    variance = read('full_H_variance')
    states = {}
    for kind in ('hop1','loop_dressed','multi_pair4'):
        result = read(kind+'_result')
        selected = {key:result[key] for key in ('dimension','nonzeros','energy','projected_reference_residual',
                    'gauss_max','hermiticity_max','doublon_number','holon_number','field_energy','matter_field_renyi2','matter_field_purity','omitted_hop_probability_sum')}
        selected.update(variance[kind])
        selected['q'] = {q:q_summary(response) for q,response in result['responses'].items()}
        selected['maximum_simultaneous_doublons'] = result.get('maximum_simultaneous_doublons',1)
        selected['doublon_number_distribution'] = result.get('doublon_number_distribution')
        states[kind] = selected
    decision = dict(
        first_state='One-hop correlated state has only one high-frequency rank3 source group, independent of these q. Two named transverse probes alone would have been misleading.',
        material_loop_state='Adding closed wrapping/plaquette flux correlations creates a strong low-energy transverse rank2 doublet at q100, but only rank1 low support at q110. This changes the physics decision: retain a finite cycle-correlated response candidate, reject the assertion of two generic propagating photon branches.',
        multi_pair_reason='Exact full-H variance exposed severe one-pair reference error. A materially multi-pair correlated construction (four-hop closure union all loop-dressed configurations) was therefore actually executed, rather than polishing bounds or fitting a speed.',
        multi_pair_finding='477302-dimensional multi-pair state has an actual q100 transverse rank2 group at2.6693166 (selected-space Ritz residual0.01246). At q110, the two independent transverse directions have distinct dominant groups2.6636685 and4.2192366 (residuals0.01316/0.02445). Density/longitudinal strength remains a separate checked Gauss channel. These are finite variational collective bands, not established full-H propagating photons.',
        changed_decision='Admitting simultaneous0..4 holon/doublon pairs lowers the energy from-6.85628 to-11.65032 and moves the q100 low group from0.92139 to2.66932. Reject the one-pair low gap as robust physical evidence; retain the richer source-resolved finite-band candidate only with exact full-H variance and truncation uncertainties.',
        final_scope='Actual finite-3D correlated collective-state/response candidate plus source-resolved finite-q finding and stationarity defeat. Finite block poles are quadrature of PHP, not full-H Lehmann poles; large full-H external residual defeats their promotion to exact photon branches.',
        exclusions=['No low-q/thermodynamic/massless/Coulomb/Nature proof','No electron/proton/physical-Noether-spin claim','No bare B term or hidden Maxwell/Dirac law','No old/evolved result replay; only new own-H state operators','No independent photon-branch proof from hopping-OFF control'])
    input_data = read('input')
    synthesis = dict(kind='EXECUTED_FINITE_PERIODIC_CORRELATED_FIELD_CURRENT_RESPONSE_AND_SOURCE_RANK_DISCRIMINATOR',
                     governor=input_data['governor'],law=input_data['law'],ansatz=input_data['ansatz'],
                     representation='Gauge-exact selected-configuration correlated state (contract-permitted comparable-state route), NOT MPS/MPO and NOT complete field enumeration',
                     states=states,decision=decision,small_physical=read('small_physical'),
                     correction=read('correction'),material_construction_decision=read('multi_pair_decision'),
                     final_candidate_exit=read('exit'),language=read('language_final')['diagnostics'].get('summary'),
                     semantics=PREFIX+'equations.json',source_capsule=PREFIX+'source_capsule.zip')
    (ROOT/(PREFIX+'synthesis.json')).write_text(json.dumps(synthesis,indent=2)+'\n')
    selected = [file for file in ROOT.glob(PREFIX+'*') if file.is_file() and file.suffix in ('.py','.json','.npz','.txt','.zip')
                and file.name not in (PREFIX+'manifest.json',PREFIX+'delivery_capsule.zip',PREFIX+'delivery_receipt.json')]
    manifest = dict(kind='OWNED_ARTIFACT_SHA256_MANIFEST',files={file.name:dict(bytes=file.stat().st_size,SHA256=hashlib.sha256(file.read_bytes()).hexdigest()) for file in sorted(selected)},
                    law=input_data['law'],producer_exit=read('exit'),limits='Only owned files; failed attempt snapshots retained, not accepted physics.')
    (ROOT/(PREFIX+'manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(PREFIX+'delivery_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
        for file in selected+[ROOT/(PREFIX+'manifest.json')]:
            archive.write(file,file.name)
    capsule = ROOT/(PREFIX+'delivery_capsule.zip')
    receipt = dict(capsule=capsule.name,bytes=capsule.stat().st_size,SHA256=hashlib.sha256(capsule.read_bytes()).hexdigest(),
                   synthesis=PREFIX+'synthesis.json',manifest=PREFIX+'manifest.json')
    (ROOT/(PREFIX+'delivery_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(receipt=receipt,state_summary={k:{name:v[name] for name in ('dimension','energy','full_H_external_residual')} for k,v in states.items()})),flush=True)


if __name__ == '__main__':
    main()
