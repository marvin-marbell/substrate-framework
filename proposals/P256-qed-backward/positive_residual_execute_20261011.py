"""Run only the new scientific calculation, preserving each immutable epoch."""
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
PREFIX = 'positive_residual_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'
PINS = {
    'vacuum_connected_native_scientist_20261011.py': 'a676d89e6e87e31f92068bb136d4b5561d3c4299d3ba45b28c55a00130236b1d',
    'vacuum_connected_native_parent_budget3_20261011_receipt.json': 'c257989407a173b2e25ff560c55dda8bff33d8f98c78411db9c6f2d707f999b6',
    'vacuum_connected_native_proof_20261011.json': 'fb15b68e34da95e21a1cc80b00b87bfdfde58d952729c7dedd0bceddb1ba0e57',
    'vacuum_operator_review_verdict_20261011.json': '702132df2aae1efcbc19ba303ceba944dfe68b26d66a3ef864993f9c2ba696ad',
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


def run(epoch):
    suffixes = ('request.json', 'preexecution.zip', 'executed.zip', 'exit.json', 'stdout.txt', 'stderr.txt', 'capsule.json', 'receipt.json')
    if any((ROOT/f'{PREFIX}{epoch}_{s}').exists() for s in suffixes):
        raise FileExistsError('Immutable epoch exists, including failed epochs')
    base = json.loads((ROOT/'vacuum_connected_native_parent_budget3_20261011_receipt.json').read_bytes())
    historical = base['input_sha256']
    names = set(historical) | set(PINS) | {Path(__file__).name, f'{PREFIX}scientist_20261011.py', f'{PREFIX}proof_20261011.json', f'{PREFIX}language_20261011.py', f'{PREFIX}language_20261011.json'}
    names |= closure(names)
    current = {name: sha(ROOT/name) for name in sorted(names)}
    for name, pin in (historical | PINS).items():
        if current[name] != pin:
            raise ArithmeticError(('Frozen pin mismatch', name))
    proof = json.loads((ROOT/f'{PREFIX}proof_20261011.json').read_bytes())
    request = dict(kind='NEW_NATIVE_POSITIVE_RESIDUAL_REQUEST_NOT_HISTORICAL_REEXECUTION', epoch=epoch, input_sha256=current, governing_plan=proof['governing_plan'], python=PYTHON, no_old_main_solver_root_checker_rerun=True)
    request_path = ROOT/f'{PREFIX}{epoch}_request.json'
    dump(request_path, request)
    pre = ROOT/f'{PREFIX}{epoch}_preexecution.zip'
    with zipfile.ZipFile(pre, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in current:
            archive.write(ROOT/name, 'inputs/'+name)
        archive.write(request_path, 'request.json')
    began = time.monotonic()
    completed = subprocess.run([PYTHON, str(ROOT/f'{PREFIX}scientist_20261011.py'), '--epoch', epoch], cwd=ROOT, capture_output=True, check=False)
    stdout = ROOT/f'{PREFIX}{epoch}_stdout.txt'; stderr = ROOT/f'{PREFIX}{epoch}_stderr.txt'
    stdout.write_bytes(completed.stdout); stderr.write_bytes(completed.stderr)
    unchanged = all(sha(ROOT/name) == pin for name, pin in current.items())
    exit_path = ROOT/f'{PREFIX}{epoch}_exit.json'
    dump(exit_path, dict(kind='NEW_NATIVE_NATURAL_EXIT', returncode=completed.returncode, natural_exit=True, elapsed_seconds=time.monotonic()-began, sources_inputs_unchanged=unchanged, stdout_sha256=sha(stdout), stderr_sha256=sha(stderr), stderr_empty=not completed.stderr, no_old_main_solver_root_checker_rerun=True))
    executed = ROOT/f'{PREFIX}{epoch}_executed.zip'
    receipt = ROOT/f'{PREFIX}{epoch}_receipt.json'
    with zipfile.ZipFile(executed, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in current:
            archive.write(ROOT/name, 'inputs/'+name)
        for path in (request_path, exit_path, stdout, stderr):
            archive.write(path, path.name)
        if receipt.exists():
            archive.write(receipt, receipt.name)
    capsule = dict(kind='IMMUTABLE_NEW_WHOLE_OPERATOR_POSITIVE_RESIDUAL_CAPSULE', epoch=epoch, request_sha256=sha(request_path), preexecution_archive=pre.name, preexecution_sha256=sha(pre), executed_archive=executed.name, executed_sha256=sha(executed), receipt=receipt.name if receipt.exists() else None, receipt_sha256=sha(receipt) if receipt.exists() else None, exit=exit_path.name, exit_sha256=sha(exit_path), returncode=completed.returncode, natural_exit=True, sources_inputs_unchanged=unchanged, stderr_empty=not completed.stderr, input_sha256=current, no_old_main_eigensolver_root_checker_rerun=True, diagnostics_for_parent_only=True)
    dump(ROOT/f'{PREFIX}{epoch}_capsule.json', capsule)
    print(json.dumps(dict(epoch=epoch, returncode=completed.returncode, natural_exit=True, receipt=capsule['receipt'], receipt_sha256=capsule['receipt_sha256'], preexecution_sha256=capsule['preexecution_sha256'], executed_sha256=capsule['executed_sha256'], sources_inputs_unchanged=unchanged, stdout=completed.stdout.decode(), stderr=completed.stderr.decode())), flush=True)
    return completed.returncode if unchanged else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', required=True)
    sys.exit(run(parser.parse_args().epoch))
