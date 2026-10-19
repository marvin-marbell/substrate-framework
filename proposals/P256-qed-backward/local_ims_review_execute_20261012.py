"""Freeze the complete new independent consumer BEFORE launching; preserve natural exits."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PYTHON = Path('/home/administrator/substrate-framework/.venv/bin/python')
PREFIX = 'local_ims_review_actual1_20261012_'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main():
    request_path = ROOT / 'local_ims_review_request_20261012.json'
    request = json.loads(request_path.read_bytes())
    frozen = {}
    for name in ('local_ims_review_source_20261012.py', 'local_ims_review_proof_20261012.json',
                 'local_ims_review_language_20261012.json', request_path.name, Path(__file__).name):
        frozen[name] = (ROOT / name).read_bytes()
    for name, pin in request['pins'].items():
        data = Path(pin['path']).read_bytes()
        if sha(data) != pin['sha256']:
            raise ValueError(f'Pin mismatch before execution: {name}')
        frozen['inputs/' + name] = data
    pre = ROOT / (PREFIX + 'preexecution.zip')
    if pre.exists():
        raise FileExistsError('Never replace an existing immutable execution epoch')
    manifest = {name: sha(data) for name, data in frozen.items()}
    manifest_bytes = (json.dumps({'kind': 'TRUE_PREEXECUTION_BEFORE_CONSUMER_LAUNCH',
                                 'members': manifest, 'launch_has_occurred': False}, indent=2) + '\n').encode()
    with zipfile.ZipFile(pre, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, data in frozen.items():
            archive.writestr(name, data)
        archive.writestr('local_ims_review_preexecution_manifest.json', manifest_bytes)
    pre.chmod(0o444)
    pre_sha = sha(pre.read_bytes())
    receipt_path = ROOT / (PREFIX + 'receipt.json')
    command = [str(PYTHON), str(ROOT / 'local_ims_review_source_20261012.py'),
               '--request', str(request_path), '--output', str(receipt_path)]
    start = time.time()
    process = subprocess.run(command, cwd=ROOT, capture_output=True)
    elapsed = time.time() - start
    stdout = ROOT / (PREFIX + 'stdout.txt'); stderr = ROOT / (PREFIX + 'stderr.txt')
    stdout.write_bytes(process.stdout); stderr.write_bytes(process.stderr)
    same = all(sha(Path(pin['path']).read_bytes()) == pin['sha256'] for pin in request['pins'].values())
    exit_record = {'kind': 'NEW_INDEPENDENT_CONSUMER_OBSERVED_NATURAL_EXIT',
                   'command': command, 'returncode': process.returncode, 'natural_exit': True,
                   'elapsed_seconds': elapsed, 'preexecution_sha256': pre_sha,
                   'source_inputs_identical_after_execution': same,
                   'stdout_sha256': sha(process.stdout), 'stderr_sha256': sha(process.stderr),
                   'receipt_sha256': sha(receipt_path.read_bytes()) if receipt_path.exists() else None,
                   'failure_preserved': process.returncode != 0,
                   'original_producer_root_optimizer_eigensolver_checker_invoked': False}
    exit_path = ROOT / (PREFIX + 'natural_exit.json')
    exit_path.write_text(json.dumps(exit_record, indent=2) + '\n')
    executed = ROOT / (PREFIX + 'executed.zip')
    with zipfile.ZipFile(executed, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, data in frozen.items():
            archive.writestr(name, data)
        archive.writestr(pre.name, pre.read_bytes())
        archive.writestr('local_ims_review_preexecution_manifest.json', manifest_bytes)
        for path in (stdout, stderr, exit_path, receipt_path):
            if path.exists():
                archive.writestr(path.name, path.read_bytes())
    executed.chmod(0o444)
    print(json.dumps({'natural_exit': exit_record, 'executed': str(executed),
                      'executed_sha256': sha(executed.read_bytes())}, indent=2))
    sys.exit(process.returncode)


if __name__ == '__main__':
    main()
