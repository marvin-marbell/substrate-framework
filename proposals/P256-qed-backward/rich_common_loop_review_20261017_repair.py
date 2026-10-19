"""Recheck the corrected CAR fixture; retain completed OWN quantum-word layers."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
from rich_common_loop_review_20261017_run import literal_car_fixture

ROOT=Path(__file__).resolve().parent;P='rich_common_loop_review_20261017_'


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    config=json.loads(Path(sys.argv[1]).read_text());stem=Path(sys.argv[1]).name.removesuffix('_input.json');prior=config['reuse_OWN_loop_epoch'];started=time.time()
    if not prior.startswith(P):raise ValueError('only this review own-epoch data may be resumed')
    for suffix,expected in config['reuse_OWN_files'].items():
        if digest(ROOT/(prior+suffix))!=expected:raise ValueError(('own prior payload changed',suffix))
    old=json.loads((ROOT/(prior+'_input.json')).read_text())
    for key in ('native_capsule_sha256','native_manifest_sha256','frozen_code_sha256','seed','field_coordinates','Schmidt_weights','relative_phase','operator_inputs'):
        if old[key]!=config[key]:raise ValueError(('unchanged quantum-layer input mismatch',key))
    completed=json.loads((ROOT/(prior+'_partial_result.json')).read_text());circuits=completed['completed']
    if set(circuits)!=set(('interior','entangled_boundary')) or any(len(rows)!=4 for rows in circuits.values()):raise ValueError('own quantum circuit layers incomplete')
    sys.path.insert(0,config['frozen_code'])
    import rich_common_quantum_em_20261016_operator as frozen
    import rich_composite_geometry_20261015_model as primitive
    model=frozen.Model(config['matter_prefix']);car=literal_car_fixture(primitive)
    gauss=float(np.max(abs(model.longitudinal@model.d.T-model.integer_q)));assert gauss<1e-11
    output=dict(kind='OWN_DISTINCT_LOOP_RICH_LITERAL_QUANTUM_CIRCUIT_REVIEW',config=config,model=model.receipt,Gauss_integer_basis_residual=gauss,packet_stats=completed['packet_stats'],circuits=circuits,independent_CAR64_exchange_fixtures=car,own_quantum_word_epoch=prior,own_quantum_word_exit_receipt=prior+'_exit.json',own_quantum_word_raw_file=prior+'_raw.npz',own_quantum_word_raw_sha256=config['reuse_OWN_files']['_raw.npz'],repaired_stage='CAR fixture only; eight completed own quantum-word layers retained unchanged from literal2. That full attempt exited1 LATER at a mistaken spin-flip exchange-sign expectation, not at loop/field assertions. This repair does not relabel its exit or rerun author/own unaffected loop layers.',meaning='Fresh own native interior and constructed body-field-entangled boundary packets. Raw word norms energy^6/energy^8, not probabilities/induced energies. Literal intermediate field P differs from net-only shortcut on boundary square; projected selected-body law remains fundamental/cut. Separate CAR exchange chronology01,23,12,30 returns SAME determinant and fixes spectator-dependent prior spin-flip expectation.',limits=['No author solved vector, stationary search or spectral approval replay','No first-C1 or47/native old state/law splice','Uncut-flux reference retains all fundamental body/Phi/R/colour cuts','Constructed input entanglement is not dynamical or ground-state evidence','No photon poles from loop strength, Gram, clamped curvature or broad Ritz nodes','Connected256CAR/localfield construction proceeds independently'],elapsed_seconds=time.time()-started)
    (ROOT/(stem+'_result.json')).write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(dict(kind=output['kind'],reused_own_quantum_word_epoch=prior,Gauss_residual=gauss,CAR_amplitudes=[v['exact_amplitude'] for v in car],elapsed_seconds=output['elapsed_seconds'])),flush=True)

if __name__=='__main__':main()
