from __future__ import annotations

import hashlib
import json
import subprocess
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREFIX = 'matching_defect_actual1_20261011'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def exclusive_json(path: Path, data: object) -> None:
    with path.open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')


def main() -> None:
    request_path = ROOT/'matching_defect_request_20261011.json'
    request = json.loads(request_path.read_text())
    names = list(request['input_sha256']) + [request_path.name, Path(__file__).name,
                                           'matching_defect_scientist_20261011.py']
    pins = {name: digest(ROOT/name) for name in names}
    assert all(pins[name] == value for name, value in request['input_sha256'].items())
    manifest_path = ROOT/(PREFIX+'_manifest.json')
    command = [PYTHON, '-B', 'matching_defect_scientist_20261011.py']
    exclusive_json(manifest_path, {'command': command, 'cwd': str(ROOT), 'input_source_sha256': pins,
                                  'law': request['operator_contract']['law'],
                                  'historical_CLI_solver_checker_reruns': 0})
    pre = ROOT/(PREFIX+'_preexecution.zip')
    with zipfile.ZipFile(pre, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in names+[manifest_path.name]:
            archive.write(ROOT/name, name)
    pre_hash = digest(pre)
    start = time.monotonic()
    with (ROOT/(PREFIX+'_stdout.txt')).open('xb') as stdout, (ROOT/(PREFIX+'_stderr.txt')).open('xb') as stderr:
        process = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
    elapsed = time.monotonic()-start
    stable = all(digest(ROOT/name) == expected for name, expected in pins.items())
    exit_path = ROOT/(PREFIX+'_exit.json')
    interface_path = ROOT/'matching_defect_interface_20261011.json'
    exit_core = {'command':command,'returncode':process.returncode,'natural_exit':True,
                 'elapsed_seconds':elapsed,'sources_inputs_unchanged':stable,
                 'preexecution_zip':pre.name,'preexecution_sha256':pre_hash,
                 'input_source_sha256':pins,
                 'interface_sha256':digest(interface_path) if interface_path.exists() else None}
    exclusive_json(exit_path,exit_core)
    executed = ROOT/(PREFIX+'_executed.zip')
    with zipfile.ZipFile(executed,'x',zipfile.ZIP_DEFLATED) as archive:
        for name in names+[manifest_path.name,exit_path.name,PREFIX+'_stdout.txt',PREFIX+'_stderr.txt']:
            archive.write(ROOT/name,name)
        if interface_path.exists():
            archive.write(interface_path,interface_path.name)
    capsule = dict(exit_core,executed_zip=executed.name,executed_sha256=digest(executed),
                   exit_sha256=digest(exit_path),stdout_sha256=digest(ROOT/(PREFIX+'_stdout.txt')),
                   stderr_sha256=digest(ROOT/(PREFIX+'_stderr.txt')),
                   archive_convention='Immutable exit core in executed ZIP; sidecar adds archive hash to avoid circular self-hashing.')
    exclusive_json(ROOT/'matching_defect_capsule_20261011.json',capsule)
    print(json.dumps({'capsule':'matching_defect_capsule_20261011.json',
                      'returncode':process.returncode,'natural_exit':True,'sources_inputs_unchanged':stable,
                      'elapsed_seconds':elapsed}),flush=True)
    if process.returncode:
        raise SystemExit(process.returncode)
    assert stable


if __name__ == '__main__':
    main()
