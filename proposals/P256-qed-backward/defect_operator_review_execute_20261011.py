from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
PREFIX = 'defect_operator_review_actual1_20261011_'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name: str, payload: dict) -> None:
    with (ROOT/name).open('x') as stream:
        json.dump(payload, stream, indent=2)
        stream.write('\n')


def main() -> None:
    request_name = 'defect_operator_review_request_20261011.json'
    request = json.loads((ROOT/request_name).read_bytes())
    names = list(request['input_sha256']) + [request_name, 'defect_operator_review_control_20261011.py', Path(__file__).name]
    before = {name: digest(ROOT/name) for name in names}
    for name, expected in request['input_sha256'].items():
        assert before[name] == expected, name
    pre_name = PREFIX+'preexecution.zip'
    with ZipFile(ROOT/pre_name, 'x', compression=ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(ROOT/name, name)
    command = ['/home/administrator/substrate-framework/.venv/bin/python', '-B', 'defect_operator_review_control_20261011.py']
    started = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
    elapsed = time.monotonic()-started
    stdout_name, stderr_name = PREFIX+'stdout.txt', PREFIX+'stderr.txt'
    with (ROOT/stdout_name).open('xb') as stream:
        stream.write(result.stdout)
    with (ROOT/stderr_name).open('xb') as stream:
        stream.write(result.stderr)
    after = {name: digest(ROOT/name) for name in names}
    exit_name = PREFIX+'exit.json'
    core = {'command': command, 'returncode': result.returncode, 'natural_exit': True, 'elapsed_seconds': elapsed, 'source_input_sha256': before, 'sources_inputs_unchanged': before == after, 'preexecution_zip': pre_name, 'preexecution_sha256': digest(ROOT/pre_name), 'stdout_sha256': digest(ROOT/stdout_name), 'stderr_sha256': digest(ROOT/stderr_name), 'target_reruns': 0}
    receipt_name = 'defect_operator_review_receipt_20261011.json'
    if result.returncode == 0:
        core['receipt_sha256'] = digest(ROOT/receipt_name)
    save(exit_name, core)
    executed_name = PREFIX+'executed.zip'
    with ZipFile(ROOT/executed_name, 'x', compression=ZIP_DEFLATED) as archive:
        for name in names+[stdout_name, stderr_name, exit_name]+([receipt_name] if result.returncode == 0 else []):
            archive.write(ROOT/name, name)
    save('defect_operator_review_capsule_20261011.json', {**core, 'executed_zip': executed_name, 'executed_sha256': digest(ROOT/executed_name), 'exit_sha256': digest(ROOT/exit_name), 'archive_convention': 'Immutable exit core in ZIP; sidecar adds archive hash without circular hashing.'})
    print(json.dumps({'returncode': result.returncode, 'elapsed_seconds': elapsed, 'sources_inputs_unchanged': before == after, 'capsule': 'defect_operator_review_capsule_20261011.json'}), flush=True)
    if result.returncode != 0:
        print(result.stderr.decode(), flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
