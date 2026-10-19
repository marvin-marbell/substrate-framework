"""Run only the new review consumer from an immutable captured source epoch."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(path: Path, members: list[Path]) -> list[dict[str, object]]:
    manifest = []
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for file in members:
            content = file.read_bytes()
            z.writestr(file.name, content)
            manifest.append({'member': file.name, 'path': str(file), 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)})
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--epoch', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    root = args.request.resolve().parent
    consumer = root / request['consumer']
    members = [args.request.resolve(), Path(__file__).resolve(), consumer]
    for pin in request['pins'].values():
        path = Path(pin['path'])
        assert path.name == pin['member'] and sha(path) == pin['sha256']
        members.append(path)
    members = list(dict.fromkeys(members))
    assert len({file.name for file in members}) == len(members)
    prefix = args.epoch.resolve()
    def product(suffix: str) -> Path:
        return prefix.with_name(prefix.name + suffix)
    pre = product('_preexecution.zip')
    before = archive(pre, members)
    receipt = product('_receipt.json')
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONHASHSEED='0')
    with tempfile.TemporaryDirectory(prefix='fast_physical_review_') as temporary:
        snapshot = Path(temporary)
        with zipfile.ZipFile(pre) as z:
            z.extractall(snapshot)
        assert all(sha(snapshot / str(item['member'])) == item['sha256'] for item in before)
        command = [request['python'], str(snapshot / consumer.name), '--request', str(snapshot / args.request.name), '--output', str(receipt)]
        start = time.monotonic()
        completed = subprocess.run(command, cwd=snapshot, env=environment, capture_output=True, check=False)
        elapsed = time.monotonic() - start
        captured_unchanged = all(sha(snapshot / str(item['member'])) == item['sha256'] for item in before)
    stdout, stderr = product('_stdout.txt'), product('_stderr.txt')
    stdout.write_bytes(completed.stdout)
    stderr.write_bytes(completed.stderr)
    original_unchanged = all(sha(Path(str(item['path']))) == item['sha256'] for item in before)
    exit_file = product('_exit.json')
    exit_data = {'kind': 'NATURAL_EXIT_INDEPENDENT_IMMUTABLE_REVIEW_CONSUMER', 'argv': command,
                 'snapshot_cwd': str(snapshot), 'snapshot_directory_removed_after_exit': True,
                 'python': request['python'], 'executor_python_version': sys.version,
                 'returncode': completed.returncode, 'natural_exit': completed.returncode >= 0,
                 'seconds': elapsed, 'captured_members_unchanged': captured_unchanged,
                 'original_members_unchanged': original_unchanged,
                 'stdout_sha256': sha(stdout), 'stderr_sha256': sha(stderr),
                 'preexecution_sha256': sha(pre), 'members': before}
    exit_file.write_text(json.dumps(exit_data, indent=2) + '\n')
    products = [stdout, stderr, exit_file] + ([receipt] if receipt.exists() else [])
    executed = product('_executed.zip')
    after = archive(executed, members + products)
    capsule = {'kind': 'NEW_REVISION_BOUND_INDEPENDENT_FAST_PHYSICAL_REVIEW_CAPSULE',
               'governor': request['governor'], 'returncode': completed.returncode,
               'natural_exit': completed.returncode >= 0, 'seconds': elapsed,
               'captured_members_unchanged': captured_unchanged, 'original_members_unchanged': original_unchanged,
               'preexecution': {'path': str(pre), 'sha256': sha(pre), 'members': before},
               'executed': {'path': str(executed), 'sha256': sha(executed), 'members': after},
               'exit': {'path': str(exit_file), 'sha256': sha(exit_file)},
               'receipt': {'path': str(receipt), 'sha256': sha(receipt)} if receipt.exists() else None,
               'scientific_admission': False, 'uniform_quantum_fast_gap': False, 'campaign_complete': False}
    capsule_file = product('_capsule.json')
    capsule_file.write_text(json.dumps(capsule, indent=2) + '\n')
    print(json.dumps({'capsule': str(capsule_file), 'returncode': completed.returncode,
                      'natural_exit': completed.returncode >= 0, 'seconds': elapsed}), flush=True)
    if not captured_unchanged or not original_unchanged:
        raise RuntimeError('captured or original epoch changed; natural result retained')
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
