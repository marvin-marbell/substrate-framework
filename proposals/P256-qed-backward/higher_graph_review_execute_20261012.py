"""Capture immutable selected bytes before and after the new scientific CLI."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = 'higher_graph_review_actual3_20261012'
PYTHON = Path('/home/administrator/substrate-framework/.venv/bin/python')


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def save(path: Path, obj: dict) -> None:
    with path.open('x') as stream:
        stream.write(json.dumps(obj, indent=2)+'\n')


def main() -> None:
    request_path = ROOT/(PREFIX+'_request.json')
    request = json.loads(request_path.read_bytes())
    selected = {name:(ROOT/name).read_bytes() for name in request['input_pins']}
    assert all(digest(raw) == request['input_pins'][name] for name,raw in selected.items())
    selected[request_path.name] = request_path.read_bytes()
    executable = PYTHON.read_bytes()
    pre_path = ROOT/(PREFIX+'_preexecution.zip')
    with zipfile.ZipFile(pre_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name,raw in selected.items():
            archive.writestr(name,raw)
    result_path = ROOT/(PREFIX+'_result.json')
    command = [str(PYTHON),str(ROOT/request['own_consumer']),'--request',str(request_path),'--output',str(result_path)]
    start = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
    elapsed = time.monotonic()-start
    stdout_path, stderr_path = ROOT/(PREFIX+'_stdout.txt'), ROOT/(PREFIX+'_stderr.txt')
    with stdout_path.open('xb') as out:
        out.write(completed.stdout)
    with stderr_path.open('xb') as out:
        out.write(completed.stderr)
    unchanged = all((ROOT/name).read_bytes() == raw for name,raw in selected.items())
    exit_record = {'command':command,'returncode':completed.returncode,'natural_exit':completed.returncode>=0,
                   'elapsed_seconds':elapsed,'selected_bytes_unchanged':unchanged,
                   'python_executable_unchanged':PYTHON.read_bytes()==executable,
                   'python_executable_sha256':digest(executable),
                   'preexecution_sha256':digest(pre_path.read_bytes()),'result_exists':result_path.exists()}
    if result_path.exists():
        exit_record['result_sha256'] = digest(result_path.read_bytes())
    exit_path = ROOT/(PREFIX+'_exit.json')
    save(exit_path,exit_record)
    executed_path = ROOT/(PREFIX+'_executed.zip')
    with zipfile.ZipFile(executed_path,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,raw in selected.items():
            archive.writestr(name,raw)
        for path in (stdout_path,stderr_path,exit_path,result_path):
            if path.exists():
                archive.writestr(path.name,path.read_bytes())
    capsule = dict(exit_record)
    capsule.update({'preexecution':str(pre_path),'executed':str(executed_path),
                    'executed_sha256':digest(executed_path.read_bytes()),'exit':str(exit_path),
                    'result':str(result_path),'selected_hashes':{name:digest(raw) for name,raw in selected.items()}})
    save(ROOT/(PREFIX+'_capsule.json'),capsule)
    print(json.dumps(capsule))
    if completed.returncode != 0 or not unchanged:
        raise SystemExit(completed.returncode or 1)


if __name__ == '__main__':
    main()
