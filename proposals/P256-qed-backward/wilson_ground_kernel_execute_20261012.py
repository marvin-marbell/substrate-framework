"""Own one-shot natural-exit scientific capsule; never execute frozen inputs."""
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
    parser.add_argument('--request', required=True)
    parser.add_argument('--epoch', required=True)
    args = parser.parse_args()
    request_path = Path(args.request).resolve()
    root = request_path.parent
    request = json.loads(request_path.read_text())
    epoch = args.epoch
    assert epoch.startswith('wilson_ground_kernel_') and '/' not in epoch
    selected = dict(request['inputs'])
    selected[request_path.name] = sha(request_path)
    selected[Path(__file__).name] = sha(Path(__file__))
    selected_paths = {name: root/name for name in selected}
    contract_member = 'wilson_ground_kernel_governing_contract_20261012.json'
    selected[contract_member] = request['contract_sha256']
    selected_paths[contract_member] = Path(request['contract_path'])
    before = {name: sha(path) for name, path in selected_paths.items()}
    assert before == selected
    pre = root/f'{epoch}_preexecution.zip'
    executed = root/f'{epoch}_executed.zip'
    output = root/f'{epoch}_result.json'
    stdout = root/f'{epoch}_stdout.txt'
    stderr = root/f'{epoch}_stderr.txt'
    exit_path = root/f'{epoch}_exit.json'
    capsule = root/f'{epoch}_capsule.json'
    for path in (pre, executed, output, stdout, stderr, exit_path, capsule):
        assert not path.exists(), f'immutable epoch already exists: {path}'
    with zipfile.ZipFile(pre, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in selected:
            archive.write(selected_paths[name], name)
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
    after = {name: sha(path) for name, path in selected_paths.items()}
    unchanged = before == after
    with zipfile.ZipFile(executed, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in selected:
            archive.write(selected_paths[name], name)
        for path in (output, stdout, stderr, exit_path):
            if path.exists(): archive.write(path, path.name)
    with zipfile.ZipFile(pre) as a, zipfile.ZipFile(executed) as b:
        identical = all(a.read(name) == b.read(name) for name in selected)
    artifacts = {p.name: sha(p) for p in (pre, executed, stdout, stderr, exit_path)}
    if output.exists(): artifacts[output.name] = sha(output)
    save(capsule, {'schema': 1, 'kind': 'IMMUTABLE_OWN_NATIVE_SCIENTIFIC_CONSUMER_CAPSULE_NOT_ADMISSION',
                   'epoch': epoch, 'request_sha256': sha(request_path),
                   'preexecution_input_hashes': before, 'postexecution_input_hashes': after,
                   'inputs_unchanged': unchanged, 'pre_to_executed_members_identical': identical,
                   'execution': exit_data, 'artifacts': artifacts,
                   'scientific_admission': False, 'campaign_complete': False})
    print(json.dumps({'capsule': capsule.name, 'sha256': sha(capsule),
                      'returncode': completed.returncode, 'natural_exit': natural,
                      'elapsed_seconds': elapsed, 'unchanged': unchanged, 'identical_members': identical}))
    assert unchanged and identical
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
