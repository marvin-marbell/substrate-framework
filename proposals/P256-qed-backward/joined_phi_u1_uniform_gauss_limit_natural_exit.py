"""Archive the exact uniform-family source/inputs, run its CLI, then selected Pyright."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = 'joined_phi_u1_uniform_gauss_limit'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'
PRODUCER = PREFIX+'_consumer.py'
INPUTS = [
    'joined_phi_u1_stationary_flat_holonomy_typed_certified_20261010_receipt.json',
    'joined_phi_u1_global_ground_geometry_bound.py',
    'joined_phi_u1_global_ground_frobenius_bound.py',
    'joined_phi_u1_matrix_jordan_supplier.py',
    'induced_phi_uniform_relative_colour_quantum.py',
    'induced_phi_nonuniform_field_car.py',
    'bosonic_original_law_join.py',
    'joined_phi_u1_jordan_sector_gaussian_parent_manifest_20261010.json',
]
CONTRACT_ROOT = Path('/home/administrator/.omp/agent/sessions/-/2026-10-09T16-26-59-293Z_01a1217d-415d-7376-a17d-16be721cbe28/local')
EXTERNAL_INPUTS = [
    CONTRACT_ROOT/'issue222-ir-uniform-charged-critical-contract.json',
    CONTRACT_ROOT/'issue222-spectral-direction-review-20261010.json',
    CONTRACT_ROOT/'issue222-uniform-volume-electric-starting-control-20261010.json',
]


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def save(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2)+'\n')


def execute(command: list[str], environment: dict[str, str]) -> dict[str, Any]:
    start = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
    return {'command': command, 'exit_code': result.returncode, 'natural_exit': True,
            'wall_seconds': time.monotonic()-start, 'stdout': result.stdout, 'stderr': result.stderr}


def run(epoch: str) -> int:
    sys.set_int_max_str_digits(0)
    source = Path(__file__).resolve()
    names = [PRODUCER, source.name, *INPUTS, PREFIX+'_preedit_language_20261010.json']
    snapshots = {name: (ROOT/name).read_bytes() for name in names}
    snapshots.update({'contract/'+p.name: p.read_bytes() for p in EXTERNAL_INPUTS})
    prearchive = ROOT/(PREFIX+'_'+epoch+'_preexecution.zip')
    receipt_path = ROOT/(PREFIX+'_'+epoch+'_actual.json')
    capsule_path = ROOT/(PREFIX+'_'+epoch+'_natural_exit.json')
    checker_path = ROOT/(PREFIX+'_'+epoch+'_checker.json')
    with zipfile.ZipFile(prearchive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, raw in sorted(snapshots.items()):
            z.writestr(name, raw)
    environment = dict(os.environ)
    environment.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    scientific = execute([PYTHON, str(ROOT/PRODUCER), '--output', str(receipt_path)], environment)
    capsule: dict[str, Any] = {'kind': 'ACTUAL_CLI_NATURAL_CAPSULE_NOT_PHOTON', 'epoch': epoch,
                'executable': PYTHON, 'preexecution_archive': {'path': prearchive.name, 'sha256': digest(prearchive.read_bytes())},
                'preexecution_sha256': {name: digest(raw) for name, raw in snapshots.items()},
                'scientific': scientific}
    save(capsule_path, capsule)
    if scientific['exit_code'] != 0:
        save(checker_path, {'accepted': False, 'failed_stage': 'scientific', 'natural_capsule': capsule_path.name})
        print(json.dumps({'accepted': False, 'capsule': capsule_path.name, 'stderr': scientific['stderr']}))
        return 1
    receipt = json.loads(receipt_path.read_bytes())
    targets = [ROOT/PRODUCER, source]
    with tempfile.TemporaryDirectory(prefix=PREFIX+'_selected_') as directory:
        config_path = Path(directory)/'pyrightconfig.json'
        config = {'include': [os.path.relpath(p, directory) for p in targets], 'extraPaths': [str(ROOT)],
                  'typeCheckingMode': 'basic', 'pythonVersion': '3.12'}
        save(config_path, config)
        checking = execute(['/home/administrator/.local/bin/pyright', '--project', str(config_path),
                            '--pythonpath', PYTHON, '--outputjson', *map(str, targets)], environment)
        checking.update(external_config=config, config_sha256=digest(config_path.read_bytes()),
                        exact_target_files=list(map(str, targets)), source_sha256={p.name: digest(p.read_bytes()) for p in targets})
    save(ROOT/(PREFIX+'_'+epoch+'_selected_pyright.json'), checking)
    capsule['selected_pyright'] = checking
    capsule['receipt'] = {'path': receipt_path.name, 'sha256': digest(receipt_path.read_bytes())}
    save(capsule_path, capsule)
    checker: dict[str, Any] = {'accepted': False, 'natural_exit': scientific['natural_exit'],
                              'scientific_exit': scientific['exit_code'], 'selected_pyright_exit': checking['exit_code']}
    try:
        assert scientific['stderr'] == ''
        assert checking['exit_code'] == 0 and checking['stderr'] == ''
        diagnostics = json.loads(checking['stdout'])
        assert diagnostics['summary']['filesAnalyzed'] == 2
        assert diagnostics['summary']['errorCount'] == 0 and diagnostics['summary']['warningCount'] == 0
        assert diagnostics['generalDiagnostics'] == []
        assert receipt['runtime']['executable'] == PYTHON
        assert [r['N'] for r in receipt['actual_N_controls']] == [4, 6, 8, 12, 16]
        assert receipt['source_inputs_unchanged']
        for name, sha in receipt['source_input_sha256'].items():
            assert digest(snapshots[name]) == sha
        executed = ROOT/receipt['executed_archive']['path']
        assert digest(executed.read_bytes()) == receipt['executed_archive']['sha256']
        with zipfile.ZipFile(executed) as z:
            for name, sha in receipt['source_input_sha256'].items():
                assert digest(z.read(name)) == sha
        for name in names:
            assert (ROOT/name).read_bytes() == snapshots[name]
        assert len(receipt['wrong_cases_assertions_reached']) == 11
        assert all(row['physical_plaquette_spectral_controls']['local_first_energy_universal_bound'] == 256 for row in receipt['actual_N_controls'])
        assert all(row['physical_plaquette_spectral_controls']['Fourier_first_energy_pair_universal_bound'] == 1024 for row in receipt['actual_N_controls'])
        assert 'joint_limit_construction' in receipt['proof']
        checker.update(accepted=True, exact_actual_source_scope=True, executed_archive_matches_preexecution=True,
                       scientific_assertions_reached=receipt['wrong_cases_assertions_reached'],
                       limiting_state_scope='static weak state + positive canonical joint plaquette momentum-energy and local-current temporal measures; no closed local dynamics/IR pole')
    except (AssertionError, KeyError, ValueError) as error:
        checker['failure'] = repr(error)
    checker.update(receipt=capsule['receipt'], capsule={'path': capsule_path.name, 'sha256': digest(capsule_path.read_bytes())},
                   preexecution_archive=capsule['preexecution_archive'], executed_archive=receipt['executed_archive'])
    save(checker_path, checker)
    print(json.dumps({'accepted': checker['accepted'], 'receipt': receipt_path.name, 'capsule': capsule_path.name,
                      'checker': checker_path.name, 'Pyright_exit': checking['exit_code']}, indent=2))
    return 0 if checker['accepted'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.epoch))
