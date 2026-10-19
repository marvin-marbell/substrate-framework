"""Own immutable one-shot capsule; executes only the independent method consumer."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--epoch', required=True)
    args = parser.parse_args()
    request_path = args.request.resolve()
    root = request_path.parent
    request = json.loads(request_path.read_text())
    epoch = args.epoch
    assert epoch.startswith('curvature_memory_review_') and '/' not in epoch
    selected = dict(request['input_sha256'])
    selected[request_path.name] = sha(request_path)
    selected[Path(__file__).name] = sha(Path(__file__))
    before = {name: sha(root/name) for name in selected}
    assert selected == before
    pre = root/f'{epoch}_preexecution.zip'
    executed = root/f'{epoch}_executed.zip'
    output = root/f'{epoch}_result.json'
    stdout = root/f'{epoch}_stdout.txt'
    stderr = root/f'{epoch}_stderr.txt'
    exit_path = root/f'{epoch}_exit.json'
    capsule = root/f'{epoch}_capsule.json'
    for path in (pre, executed, output, stdout, stderr, exit_path, capsule):
        assert not path.exists(), f'immutable epoch exists: {path}'
    with zipfile.ZipFile(pre, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in selected:
            archive.write(root/name, name)
    command = [sys.executable, str(root/request['consumer']), '--request', str(request_path), '--output', str(output)]
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    completed = subprocess.run(command, cwd=root, capture_output=True, check=False)
    elapsed = time.monotonic()-start
    stdout.write_bytes(completed.stdout)
    stderr.write_bytes(completed.stderr)
    natural = completed.returncode >= 0
    exit_data = {'command': command, 'started_at': started,
                 'completed_at': datetime.now(timezone.utc).isoformat(),
                 'elapsed_seconds': elapsed, 'returncode': completed.returncode,
                 'natural_exit': natural, 'signal': -completed.returncode if not natural else None,
                 'timeout': None, 'old_entrypoint_executed': False}
    save(exit_path, exit_data)
    after = {name: sha(root/name) for name in selected}
    with zipfile.ZipFile(executed, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in selected:
            archive.write(root/name, name)
        for path in (output, stdout, stderr, exit_path):
            if path.exists():
                archive.write(path, path.name)
    with zipfile.ZipFile(pre) as a, zipfile.ZipFile(executed) as b:
        identical = all(a.read(name) == b.read(name) for name in selected)
    artifacts = {p.name: sha(p) for p in (pre, executed, stdout, stderr, exit_path)}
    if output.exists():
        artifacts[output.name] = sha(output)
    save(capsule, {'schema': 1, 'kind': 'IMMUTABLE_INDEPENDENT_NONNATIVE_METHOD_REVIEW_CAPSULE_NOT_ADMISSION',
                   'epoch': epoch, 'request_sha256': sha(request_path),
                   'preexecution_input_hashes': before, 'postexecution_input_hashes': after,
                   'inputs_unchanged': before == after, 'pre_to_executed_members_identical': identical,
                   'execution': exit_data, 'artifacts': artifacts,
                   'scientific_admission': False, 'campaign_complete': False})
    print(json.dumps({'capsule': capsule.name, 'sha256': sha(capsule),
                      'returncode': completed.returncode, 'natural_exit': natural,
                      'elapsed_seconds': elapsed, 'unchanged': before == after, 'identical_members': identical}))
    assert before == after and identical
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
