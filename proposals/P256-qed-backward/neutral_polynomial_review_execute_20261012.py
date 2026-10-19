"""One independent mathematical run with honest before/after byte capture."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, data: dict) -> None:
    with path.open('x') as handle:
        json.dump(data,handle,indent=2)
        handle.write('\n')


def freeze(path: Path, members: list[Path]) -> None:
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for member in members:
            archive.write(member,member.name)
    path.chmod(0o444)


def main() -> None:
    root = Path(__file__).resolve().parent
    prefix = root/'neutral_polynomial_review_actual1_20261012'
    def artifact(suffix: str) -> Path:
        return prefix.with_name(prefix.name+suffix)
    request = artifact('_request.json')
    req = json.loads(request.read_text())
    source = root/'neutral_polynomial_review_consumer_20261012.py'
    inputs = [Path(item['path']) for item in req['inputs']]
    for item in req['inputs']:
        assert digest(Path(item['path'])) == item['sha256']
    runner = Path(__file__).resolve()
    before = artifact('_preexecution.zip')
    captured = list(dict.fromkeys([request,source,runner]+inputs))
    freeze(before,captured)
    receipt = artifact('_receipt.json')
    stdout,stderr = artifact('_stdout.txt'),artifact('_stderr.txt')
    command = [req['python'],str(source),str(request),str(receipt)]
    start = time.time_ns()
    # communicate waits for natural child exit; no timeout, signal, old consumer or mock.
    completed = subprocess.run(command,cwd=root,capture_output=True,check=False)
    finish = time.time_ns()
    with stdout.open('xb') as handle:
        handle.write(completed.stdout)
    with stderr.open('xb') as handle:
        handle.write(completed.stderr)
    exit_path = artifact('_exit.json')
    dump(exit_path,{'argv':command,'cwd':str(root),'start_unix_ns':start,
         'finish_unix_ns':finish,'returncode':completed.returncode,
         'natural_exit':True,'signal_sent':False,'preexecution_sha256':digest(before),
         'request_sha256':digest(request),'source_sha256':digest(source)})
    outputs = [stdout,stderr,exit_path]+([receipt] if receipt.exists() else [])
    executed = artifact('_executed.zip')
    freeze(executed,captured+outputs)
    capsule = artifact('_capsule.json')
    hashes = {p.name:digest(p) for p in captured+outputs+[before,executed]}
    dump(capsule,{'kind':'INDEPENDENT_MATHEMATICAL_CONTROL_IMMUTABLE_CAPSULE',
         'status':'NATURAL_EXIT_SUCCESS' if completed.returncode == 0 else 'NATURAL_EXIT_FAILURE',
         'governor':req['governor'],'natural_exit':True,'returncode':completed.returncode,
         'preexecution_archive_created_before_subprocess':True,
         'full_sha256':hashes,'receipt':receipt.name if receipt.exists() else None,
         'no_original_producer_rerun':True,'scientific_admission':False,
         'campaign_complete':False})
    for path in outputs+[capsule]:
        path.chmod(0o444)
    print(json.dumps({'returncode':completed.returncode,'capsule':capsule.name,
                      'capsule_sha256':digest(capsule)}))
    sys.exit(completed.returncode)


if __name__ == '__main__':
    main()
