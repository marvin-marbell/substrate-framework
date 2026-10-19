"""Immutable BEFORE-execution archive for NEW native quantum consumers only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'
CONTRACT = Path('/home/administrator/.omp/agent/sessions/-/2026-10-09T16-26-59-293Z_01a1217d-415d-7376-a17d-16be721cbe28/local/issue222-all-auxiliary-ground-sector-fullquantum-loop-support-contract-20261012.json')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def create_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', required=True)
    parser.add_argument('--successor', action='store_true')
    args = parser.parse_args()
    assert args.epoch.startswith('ground_loop_support_')
    prefix = args.epoch
    contract_raw = CONTRACT.read_bytes()
    assert digest(contract_raw) == 'e37aaf344e17cb2bf2bb07623fdf6fd5571b0e840df74a3dcb3f51b751c38254'
    contract = json.loads(contract_raw)
    selected = ['ground_loop_support_consumer_20261012.py', 'ground_loop_support_execute_20261012.py',
                'ground_loop_support_method_20261012.json', 'ground_loop_support_language_20261012.json',
                'native_loop_dark_proof_20261012.json', 'native_loop_dark_auxiliary_proof_20261012.json',
                'native_field_price_review_proof_20261012.json', 'auxiliary_adjoint_review_proof_20261012.json',
                'common_singlet_energy_interface_20261012.json', 'full_loop_graph_proof_20261012.json',
                'loop_fractional_graph_proof_20261012.json', 'carrier_support_review_proof_20261012.json',
                'canonical_ground_parent_moving_dark_bundle_method_20261012.json',
                'joined_phi_u1_uniform_gauss_limit_epoch4_20261010_actual_executed_inputs.zip']
    if args.successor:
        selected[0] = 'ground_loop_support_successor_consumer_20261012.py'
        selected[1:1] = ['ground_loop_support_actual1_20261012_preexecution.zip',
                         'ground_loop_support_actual1_20261012_receipt.json',
                         'ground_loop_support_actual1_20261012_capsule.json']
        selected[1:1] = ['ground_loop_support_actual2_20261012_preexecution.zip',
                         'ground_loop_support_actual2_20261012_capsule.json',
                         'ground_loop_support_epoch2_correction_20261012.json']
    snapshots = {name: (ROOT/name).read_bytes() for name in selected}
    for name, expected in contract['inputs'].items():
        if name in snapshots:
            assert digest(snapshots[name]) == expected, name
    governor = {key: contract[key] for key in ('plan_id', 'captured_revision', 'accepted_scoped_science',
                'template_id', 'template_revision', 'work_item_id', 'requirement_ids', 'parent_only_plan_writer')}
    pre = ROOT/(prefix+'_preexecution.zip')
    request = {'epoch': prefix, 'governor': governor, 'contract_sha256': digest(contract_raw),
               'consumer': selected[0], 'input_sha256': {name: digest(raw) for name, raw in snapshots.items()},
               'preexecution_archive': str(pre),
               'profile_archive': selected[-1], 'profile_member': contract['profile_member'],
               'profile_member_sha256': contract['profile_member_sha256']}
    if args.successor:
        request['first_preexecution_archive'] = 'ground_loop_support_actual1_20261012_preexecution.zip'
    request_path = ROOT/(prefix+'_request.json')
    create_json(request_path, request)
    with zipfile.ZipFile(pre, 'x', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('request.json', request_path.read_bytes())
        archive.writestr('contract.json', contract_raw)
        for name, raw in snapshots.items():
            archive.writestr(name, raw)
    pre_sha = digest(pre.read_bytes())
    output = ROOT/(prefix+'_receipt.json')
    command = [PYTHON, '-B', '-c',
               'import sys,zipfile;z=zipfile.ZipFile(sys.argv[1]);raw=z.read(sys.argv[2]);source=sys.argv[3];sys.argv=[source,"--request",sys.argv[4],"--output",sys.argv[5]];exec(compile(raw,source,"exec"),{"__name__":"__main__","__file__":source})',
               str(pre), selected[0], str(ROOT/selected[0]), str(request_path), str(output)]
    started = time.time(); clock = time.monotonic()
    child = subprocess.run(command, cwd=ROOT, capture_output=True)
    stdout = ROOT/(prefix+'_stdout.txt'); stderr = ROOT/(prefix+'_stderr.txt')
    with stdout.open('xb') as stream:
        stream.write(child.stdout)
    with stderr.open('xb') as stream:
        stream.write(child.stderr)
    exit_path = ROOT/(prefix+'_exit.json')
    create_json(exit_path, {'exit_code': child.returncode, 'natural_exit': True,
                           'started_unix': started, 'elapsed_seconds': time.monotonic()-clock,
                           'stdout_sha256': digest(child.stdout), 'stderr_sha256': digest(child.stderr)})
    assert digest(pre.read_bytes()) == pre_sha
    unchanged = all((ROOT/name).read_bytes() == raw for name, raw in snapshots.items())
    executed = ROOT/(prefix+'_executed.zip')
    with zipfile.ZipFile(executed, 'x', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('request.json', request_path.read_bytes())
        archive.writestr('contract.json', contract_raw)
        for name, raw in snapshots.items():
            archive.writestr(name, raw)
        for path in (stdout, stderr, exit_path, output):
            if path.exists():
                archive.writestr('outputs/'+path.name, path.read_bytes())
    capsule = {'kind': 'NEW_NATIVE_QUANTUM_RESPONSE_TWO_ATTEMPT_NATURAL_EXIT', 'governor': governor,
               'argv': command, 'preexecution': {'path': pre.name, 'sha256': pre_sha, 'created_before_child': True},
               'executed': {'path': executed.name, 'sha256': digest(executed.read_bytes())},
               'request': {'path': request_path.name, 'sha256': digest(request_path.read_bytes())},
               'receipt': {'path': output.name, 'sha256': digest(output.read_bytes())} if output.exists() else None,
               'input_sha256': request['input_sha256'], 'natural_exit': True, 'exit_code': child.returncode,
               'inputs_unchanged': unchanged, 'source_selection': 'compile NEW consumer bytes from BEFORE-execution ZIP',
               'historical_main_solver_checker_eigensolver_executed': False,
               'standalone_build_test_lint_format_pyright_executed': False,
               'actual_ground_backend': False, 'scientific_admission': False, 'campaign_complete': False}
    create_json(ROOT/(prefix+'_capsule.json'), capsule)
    print(json.dumps({'capsule': prefix+'_capsule.json', 'natural_exit': True,
                      'exit_code': child.returncode, 'elapsed_seconds': time.monotonic()-clock}))
    sys.exit(child.returncode if unchanged else 2)


if __name__ == '__main__':
    main()
