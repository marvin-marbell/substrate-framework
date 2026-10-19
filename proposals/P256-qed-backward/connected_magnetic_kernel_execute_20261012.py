"""Immutable own-CLI epochs; preserve failures and never execute old entrypoints."""
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

ROOT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put_json(path: Path, value: object) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--epoch', required=True)
    args = parser.parse_args()
    assert args.epoch.startswith('connected_magnetic_kernel_')
    assert '/' not in args.epoch and '\\' not in args.epoch
    request_path = args.request.resolve()
    request = json.loads(request_path.read_text())
    manifest: dict[str, str] = {}
    selected: list[tuple[str, Path]] = []
    for name, expected in request['source_inputs'].items():
        path = Path(name) if name.startswith('/') else ROOT/name
        actual = sha(path)
        assert actual == expected, f'Preexecution input changed: {name}'
        member = 'bound_contract/'+path.name if name.startswith('/') else 'source_input/'+name
        assert member not in manifest
        manifest[member] = actual
        selected.append((member, path))
    prefix = ROOT/args.epoch
    pre = Path(str(prefix)+'_preexecution.zip')
    executed = Path(str(prefix)+'_executed.zip')
    receipt = Path(str(prefix)+'_receipt.json')
    stdout = Path(str(prefix)+'_stdout.txt')
    stderr = Path(str(prefix)+'_stderr.txt')
    exit_path = Path(str(prefix)+'_exit.json')
    capsule_path = Path(str(prefix)+'_capsule.json')
    for path in (pre, executed, receipt, stdout, stderr, exit_path, capsule_path):
        assert not path.exists(), f'Epoch is immutable: {path}'
    with zipfile.ZipFile(pre, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for member, path in selected:
            archive.write(path, member)
        archive.write(request_path, 'request/'+request_path.name)
    command = [sys.executable, str(ROOT/request['consumer']),
               '--request', str(request_path), '--output', str(receipt)]
    started = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    with stdout.open('xb') as out, stderr.open('xb') as err:
        child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err)
        pid = child.pid
        code = child.wait()  # Natural exit; no killed/timed-out success substitute.
    elapsed = time.monotonic()-clock
    exit_info = {'command': command, 'pid': pid, 'returncode': code,
                 'natural_exit': True, 'started_UTC': started,
                 'finished_UTC': datetime.now(timezone.utc).isoformat(),
                 'elapsed_seconds': elapsed, 'old_entrypoints_executed': False}
    put_json(exit_path, exit_info)
    after_manifest: dict[str, str] = {}
    for member, path in selected:
        after_manifest[member] = sha(path)
    unchanged = after_manifest == manifest
    with zipfile.ZipFile(executed, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        # Exact PREEXECUTION source bytes, not a later retagging of source.
        with zipfile.ZipFile(pre) as original:
            for member in original.namelist():
                archive.writestr(member, original.read(member))
        for path in (stdout, stderr, exit_path, receipt):
            if path.exists():
                archive.write(path, 'execution/'+path.name)
    capsule = {'kind': 'OWN_NATURAL_EXIT_IMMUTABLE_CONNECTED_MECHANISM_CAPSULE',
               'epoch': args.epoch, 'governor': request['governor'],
               'request': {'path': request_path.name, 'sha256': sha(request_path)},
               'source_input_SHA256': request['source_inputs'],
               'selected_member_SHA256': manifest,
               'source_inputs_unchanged_after_child': unchanged,
               'preexecution': {'path': pre.name, 'sha256': sha(pre)},
               'executed': {'path': executed.name, 'sha256': sha(executed)},
               'exit': {'path': exit_path.name, 'sha256': sha(exit_path), **exit_info},
               'stdout': {'path': stdout.name, 'sha256': sha(stdout)},
               'stderr': {'path': stderr.name, 'sha256': sha(stderr)},
               'receipt': {'path': receipt.name, 'sha256': sha(receipt)} if receipt.exists() else None,
               'failed_epoch_preserved': code != 0,
               'native_actual_ground_numeric_backend': False,
               'parent_review_required': True}
    put_json(capsule_path, capsule)
    print(json.dumps({'epoch': args.epoch, 'returncode': code,
                      'natural_exit': True, 'inputs_unchanged': unchanged,
                      'capsule': capsule_path.name}))
    if not unchanged:
        raise RuntimeError('Bound inputs changed during scientific execution')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
