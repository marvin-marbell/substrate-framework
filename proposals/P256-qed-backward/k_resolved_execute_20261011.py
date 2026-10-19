"""Archive and execute ONLY the new joint-energy implementation, once per epoch."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = 'k_resolved_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'
PINS = {
    'positive_residual_actual1_20261011_receipt.json': '4a5b0e7ca9f6aafe9a07588d02550cf35d45e7d59f8532b0fa1330c1c78cdef3',
    'positive_residual_global_proof_20261011.json': '206df7fe7a4bf76d55db7ae5cc0937e0fecf261b7d404fb9aef3d51ea0610ec7',
    'positive_residual_actual1_20261011_preexecution.zip': '60c6f8fb13b88ecc45a502737dcf73ae9f788fe2cba1171c78a88fd3a9ce7de7',
    'positive_residual_actual1_20261011_executed.zip': '084375cd0324b7d93abd8958d4c895a95cc0f2ec86c7bf6e624e6a96094dbd47',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def closure(names):
    queue = [n for n in names if n.endswith('.py')]
    seen = set(queue)
    while queue:
        name = queue.pop()
        tree = ast.parse((ROOT/name).read_text(), filename=name)
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
        for module in modules:
            candidate = module.split('.')[0]+'.py'
            if (ROOT/candidate).is_file() and candidate not in seen:
                seen.add(candidate); queue.append(candidate)
    return seen


def run(epoch, stage):
    suffixes = ('request.json', 'preexecution.zip', 'executed.zip', 'exit.json', 'stdout.txt', 'stderr.txt', 'capsule.json', 'receipt.json')
    if any((ROOT/f'{PREFIX}{epoch}_{s}').exists() for s in suffixes):
        raise FileExistsError('Immutable epoch exists; failed epochs cannot be replaced')
    historical = json.loads((ROOT/'positive_residual_actual1_20261011_capsule.json').read_bytes())['input_sha256']
    stage_prefix = '' if stage == 'primary' else stage+'_'
    science = f'{PREFIX}{stage_prefix}scientist_20261011.py'
    proof_name = f'{PREFIX}{stage_prefix}proof_20261011.json'
    local = {Path(__file__).name, science, proof_name, f'{PREFIX}language_20261011.py', f'{PREFIX}language_20261011.json'}
    runtime_names = local | closure({science}) | {'positive_residual_actual1_20261011_receipt.json', 'positive_residual_actual1_20261011_capsule.json', 'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json', 'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json'}
    if stage in ('centered', 'refined'):
        runtime_names |= {f'{PREFIX}actual1_20261011_{s}' for s in ('receipt.json', 'capsule.json', 'exit.json', 'preexecution.zip', 'executed.zip')}
    if stage == 'refined':
        runtime_names |= {f'{PREFIX}centered_actual1_20261011_{s}' for s in ('receipt.json', 'capsule.json', 'exit.json', 'preexecution.zip', 'executed.zip')}
    names = set(historical) | set(PINS) | runtime_names
    blobs = {}
    with zipfile.ZipFile(ROOT/'positive_residual_actual1_20261011_executed.zip') as oldarchive:
        for name in names:
            blobs[name] = oldarchive.read('inputs/'+name) if name in historical and name not in runtime_names else (ROOT/name).read_bytes()
    current = {name: hashlib.sha256(blob).hexdigest() for name, blob in sorted(blobs.items())}
    for name, pin in (historical | PINS).items():
        if current[name] != pin:
            raise ArithmeticError(('Frozen pin mismatch', name))
    runtime = {name: sha(ROOT/name) for name in sorted(runtime_names)}
    proof = json.loads((ROOT/proof_name).read_bytes())
    request = dict(kind='NEW_JOINT_K_ENERGY_EXECUTION_REQUEST', epoch=epoch, stage=stage, science_source=science, governing_plan=proof['governing_plan'], input_sha256=current, runtime_sha256=runtime, historical_auxiliary_bytes_from_immutable_archive=True, python=PYTHON, physical_hopping_exact='1', no_historical_entrypoint_eigensolver_root_checker=True, parent_diagnostics_only=proof['parent_diagnostics_only'])
    request_path = ROOT/f'{PREFIX}{epoch}_request.json'
    dump(request_path, request)
    pre = ROOT/f'{PREFIX}{epoch}_preexecution.zip'
    with zipfile.ZipFile(pre, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, blob in blobs.items():
            archive.writestr('inputs/'+name, blob)
        archive.write(request_path, 'request.json')
    began = time.monotonic()
    completed = subprocess.run([PYTHON, str(ROOT/science), '--epoch', epoch], cwd=ROOT, capture_output=True, check=False)
    stdout = ROOT/f'{PREFIX}{epoch}_stdout.txt'; stderr = ROOT/f'{PREFIX}{epoch}_stderr.txt'
    stdout.write_bytes(completed.stdout); stderr.write_bytes(completed.stderr)
    unchanged = all(sha(ROOT/name) == pin for name, pin in runtime.items())
    exit_path = ROOT/f'{PREFIX}{epoch}_exit.json'
    dump(exit_path, dict(kind='NEW_JOINT_K_ENERGY_NATURAL_EXIT', returncode=completed.returncode, natural_exit=True, elapsed_seconds=time.monotonic()-began, sources_inputs_unchanged=unchanged, stdout_sha256=sha(stdout), stderr_sha256=sha(stderr), stderr_empty=not completed.stderr))
    executed = ROOT/f'{PREFIX}{epoch}_executed.zip'
    receipt = ROOT/f'{PREFIX}{epoch}_receipt.json'
    with zipfile.ZipFile(executed, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, blob in blobs.items():
            archive.writestr('inputs/'+name, blob)
        for path in (request_path, exit_path, stdout, stderr):
            archive.write(path, path.name)
        if receipt.exists():
            archive.write(receipt, receipt.name)
    capsule = dict(kind='IMMUTABLE_NEW_JOINT_K_ENERGY_EXECUTED_CAPSULE', epoch=epoch, request_sha256=sha(request_path), preexecution_archive=pre.name, preexecution_sha256=sha(pre), executed_archive=executed.name, executed_sha256=sha(executed), receipt=receipt.name if receipt.exists() else None, receipt_sha256=sha(receipt) if receipt.exists() else None, exit=exit_path.name, exit_sha256=sha(exit_path), returncode=completed.returncode, natural_exit=True, sources_inputs_unchanged=unchanged, stderr_empty=not completed.stderr, input_sha256=current, no_historical_entrypoint_eigensolver_root_checker=True, diagnostics_for_parent_only=True)
    dump(ROOT/f'{PREFIX}{epoch}_capsule.json', capsule)
    print(json.dumps(dict(epoch=epoch, returncode=completed.returncode, natural_exit=True, receipt=capsule['receipt'], receipt_sha256=capsule['receipt_sha256'], preexecution_sha256=capsule['preexecution_sha256'], executed_sha256=capsule['executed_sha256'], sources_inputs_unchanged=unchanged, stdout=completed.stdout.decode(), stderr=completed.stderr.decode())), flush=True)
    return completed.returncode if unchanged else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', required=True)
    parser.add_argument('--stage', choices=('primary', 'centered', 'refined'), default='primary')
    args = parser.parse_args()
    sys.exit(run(args.epoch, args.stage))
