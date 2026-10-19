"""Capture immutable review inputs and natural scientific-process exit once."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = 'contiguous_carrier_review_actual1_20261012'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def put_json(path: Path, value: object) -> None:
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')


def run() -> None:
    manifest_path = ROOT/'contiguous_carrier_review_input_manifest_20261012.json'
    manifest = json.loads(manifest_path.read_text())
    inputs = {name:(ROOT/name).read_bytes() for name in manifest['members']}
    assert {name:sha(data) for name, data in inputs.items()} == manifest['members']
    inputs[manifest_path.name] = manifest_path.read_bytes()
    input_archive = ROOT/'contiguous_carrier_review_inputs_20261012.zip'
    with zipfile.ZipFile(input_archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in inputs.items(): z.writestr(name, data)
    consumer = ROOT/'contiguous_carrier_review_consumer_20261012.py'
    result = ROOT/f'{PREFIX}_result.json'
    command = [PYTHON, str(consumer), '--input-archive', str(input_archive), '--output', str(result)]
    request = {'schema':1, 'command':command, 'input_archive_sha256':sha(input_archive.read_bytes()),
               'input_members':manifest['members'], 'source_sha256':sha(consumer.read_bytes()),
               'physical_hopping':1, 'old_entrypoints_executed':False, 'scientific_admission':False}
    request_path = ROOT/f'{PREFIX}_request.json'
    put_json(request_path, request)
    captured = dict(inputs)
    for name in ['contiguous_carrier_review_consumer_20261012.py', 'contiguous_carrier_review_execute_20261012.py',
                 'contiguous_carrier_review_proof_20261012.json', 'contiguous_carrier_review_language_20261012.json',
                 request_path.name]:
        captured[name] = (ROOT/name).read_bytes()
    pre = ROOT/f'{PREFIX}_preexecution.zip'
    with zipfile.ZipFile(pre, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in captured.items(): z.writestr(name, data)
    start = datetime.now(timezone.utc).isoformat(); clock = time.monotonic()
    process = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
    elapsed = time.monotonic()-clock
    end = datetime.now(timezone.utc).isoformat()
    stdout = ROOT/f'{PREFIX}_stdout.txt'; stderr = ROOT/f'{PREFIX}_stderr.txt'
    with stdout.open('xb') as f: f.write(process.stdout)
    with stderr.open('xb') as f: f.write(process.stderr)
    exit_record = {'command':command, 'returncode':process.returncode,
                   'natural_exit':process.returncode >= 0, 'signal':-process.returncode if process.returncode < 0 else None,
                   'started_at':start, 'completed_at':end, 'elapsed_seconds':elapsed,
                   'timeout':None, 'old_entrypoints_executed':False}
    exit_path = ROOT/f'{PREFIX}_exit.json'; put_json(exit_path, exit_record)
    post_hashes = {name:sha((ROOT/name).read_bytes()) for name in captured}
    pre_hashes = {name:sha(data) for name, data in captured.items()}
    assert post_hashes == pre_hashes
    executed = ROOT/f'{PREFIX}_executed.zip'
    execution_outputs = [stdout, stderr, exit_path]+([result] if result.exists() else [])
    with zipfile.ZipFile(executed, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in captured.items(): z.writestr(name, data)
        for path in execution_outputs: z.writestr(path.name, path.read_bytes())
    with zipfile.ZipFile(pre) as a, zipfile.ZipFile(executed) as b:
        identical = all(a.read(name) == b.read(name) for name in a.namelist())
    assert identical
    artifacts = [input_archive, pre, executed, *execution_outputs]
    capsule = {'schema':1, 'kind':'IMMUTABLE_INDEPENDENT_SCIENTIFIC_REVIEW_CAPSULE_NOT_ADMISSION',
               'epoch':PREFIX, 'execution':exit_record, 'preexecution_hashes':pre_hashes,
               'postexecution_hashes':post_hashes, 'inputs_unchanged':True,
               'pre_to_executed_members_identical':identical,
               'artifacts':{path.name:sha(path.read_bytes()) for path in artifacts},
               'scientific_admission':False, 'campaign_complete':False}
    put_json(ROOT/f'{PREFIX}_capsule.json', capsule)
    print(json.dumps({'returncode':process.returncode, 'natural_exit':exit_record['natural_exit'],
                      'elapsed_seconds':elapsed, 'inputs_unchanged':True, 'stderr':process.stderr.decode()}))
    if process.returncode: raise SystemExit(process.returncode)


if __name__ == '__main__':
    run()
