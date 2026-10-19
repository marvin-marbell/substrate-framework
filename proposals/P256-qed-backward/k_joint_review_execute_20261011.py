"""Capture only NEW independent review controls, retaining every attempt."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def run(epoch):
    prefix = 'k_joint_review_'+epoch
    request_path = ROOT/(prefix+'_request.json')
    request = json.loads(request_path.read_bytes())
    source = ROOT/'k_joint_review_control_20261011.py'
    sources = {source.name: sha(source), Path(__file__).name: sha(Path(__file__))}
    assert request['source_sha256'] == sources
    assert all(sha(ROOT/n) == pin for n, pin in request['input_sha256'].items())
    pre = ROOT/(prefix+'_preexecution.zip')
    receipt = ROOT/(prefix+'_receipt.json')
    exit_path = ROOT/(prefix+'_exit.json')
    stdout = ROOT/(prefix+'_stdout.txt')
    stderr = ROOT/(prefix+'_stderr.txt')
    executed = ROOT/(prefix+'_executed.zip')
    capsule = ROOT/(prefix+'_capsule.json')
    assert not any(p.exists() for p in (pre, receipt, exit_path, stdout, stderr, executed, capsule))
    archived_inputs = [n for n in request['input_sha256'] if not n.endswith('.zip')]
    with zipfile.ZipFile(pre, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.write(request_path, request_path.name)
        for n in sources:
            z.write(ROOT/n, 'source/'+n)
        for n in archived_inputs:
            z.write(ROOT/n, 'inputs/'+n)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    began = time.monotonic()
    with stdout.open('wb') as out, stderr.open('wb') as err:
        result = subprocess.run([PYTHON, str(source), request_path.name, receipt.name], cwd=ROOT, env=env, stdout=out, stderr=err, check=False)
    unchanged = (all(sha(ROOT/n) == pin for n, pin in request['input_sha256'].items())
                 and all(sha(ROOT/n) == pin for n, pin in sources.items()))
    dump(exit_path, dict(kind='NEW_INDEPENDENT_CONTROL_NATURAL_EXIT', returncode=result.returncode,
        natural_exit=True, elapsed_seconds=time.monotonic()-began,
        sources_inputs_unchanged=unchanged, stdout_sha256=sha(stdout), stderr_sha256=sha(stderr),
        stderr_empty=stderr.stat().st_size == 0))
    with zipfile.ZipFile(executed, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in (request_path, exit_path, stdout, stderr):
            z.write(p, p.name)
        if receipt.exists():
            z.write(receipt, receipt.name)
        for n in sources:
            z.write(ROOT/n, 'source/'+n)
        for n in archived_inputs:
            z.write(ROOT/n, 'inputs/'+n)
    payload = dict(kind='IMMUTABLE_INDEPENDENT_K_JOINT_REVIEW_CONTROL_CAPSULE', epoch=epoch,
        request=request_path.name, request_sha256=sha(request_path), source_sha256=sources,
        input_sha256=request['input_sha256'], upstream_archives_referenced_by_hash_not_duplicated=True,
        preexecution_archive=pre.name, preexecution_sha256=sha(pre),
        executed_archive=executed.name, executed_sha256=sha(executed),
        exit=exit_path.name, exit_sha256=sha(exit_path), stdout=stdout.name, stdout_sha256=sha(stdout),
        stderr=stderr.name, stderr_sha256=sha(stderr), returncode=result.returncode,
        natural_exit=True, sources_inputs_unchanged=unchanged, stderr_empty=stderr.stat().st_size == 0)
    if receipt.exists():
        payload.update(receipt=receipt.name, receipt_sha256=sha(receipt))
    dump(capsule, payload)
    print(json.dumps(dict(capsule=capsule.name, returncode=result.returncode, natural_exit=True,
                         sources_inputs_unchanged=unchanged)), flush=True)
    if stdout.stat().st_size:
        print(stdout.read_text(), end='')
    if stderr.stat().st_size:
        print(stderr.read_text(), end='', file=sys.stderr)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(run(sys.argv[1]))
