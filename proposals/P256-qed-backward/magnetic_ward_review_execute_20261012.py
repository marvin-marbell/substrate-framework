"""Freeze own sources before executing only the independent review consumer."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def put(path: Path, value: Any) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--epoch', required=True)
    args = parser.parse_args()
    if not args.epoch.startswith('magnetic_ward_review_'):
        raise ValueError('Own epoch prefix required')
    request_path = args.request.resolve()
    request = json.loads(request_path.read_text())
    if request['consumer'] != 'magnetic_ward_review_consumer_20261012.py':
        raise ValueError('Only own independent consumer may run')
    paths = {name: Path(name) if Path(name).is_absolute() else ROOT/name for name in request['pins']}
    before = {name: sha(path) for name, path in paths.items()}
    if before != request['pins']:
        raise ValueError('Bound source changed before execution')
    epoch = ROOT/args.epoch
    pre = epoch.with_name(epoch.name+'_preexecution.zip')
    executed = epoch.with_name(epoch.name+'_executed.zip')
    receipt = epoch.with_name(epoch.name+'_receipt.json')
    exit_path = epoch.with_name(epoch.name+'_exit.json')
    stdout = epoch.with_name(epoch.name+'_stdout.txt')
    stderr = epoch.with_name(epoch.name+'_stderr.txt')
    capsule = epoch.with_name(epoch.name+'_capsule.json')
    for path in (pre, executed, receipt, exit_path, stdout, stderr, capsule):
        if path.exists():
            raise FileExistsError(path)
    members = {}
    with zipfile.ZipFile(pre, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in paths.items():
            member = 'source_input/'+name if not Path(name).is_absolute() else 'bound_contract/'+path.name
            archive.write(path, member)
            members[member] = before[name]
        archive.write(request_path, 'request/'+request_path.name)
        members['request/'+request_path.name] = sha(request_path)
    command = [sys.executable, str(ROOT/request['consumer']), '--request', str(request_path), '--output', str(receipt)]
    started = datetime.now(timezone.utc).isoformat()
    tick = time.monotonic()
    with stdout.open('xb') as out, stderr.open('xb') as err:
        child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err)
        code = child.wait()
    ended = datetime.now(timezone.utc).isoformat()
    after = {name: sha(path) for name, path in paths.items()}
    exit_record = {'command': command, 'pid': child.pid, 'returncode': code, 'natural_exit': True,
                   'started_UTC': started, 'finished_UTC': ended, 'elapsed_seconds': time.monotonic()-tick,
                   'source_inputs_unchanged': before == after, 'old_entrypoints_executed': False}
    put(exit_path, exit_record)
    with zipfile.ZipFile(executed, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        with zipfile.ZipFile(pre) as frozen:
            for info in frozen.infolist():
                archive.writestr(info, frozen.read(info.filename))
        for path in (receipt, exit_path, stdout, stderr):
            if path.exists():
                archive.write(path, 'execution/'+path.name)
                members['execution/'+path.name] = sha(path)
    def record(path: Path) -> dict[str, str]:
        return {'path': path.name, 'sha256': sha(path)}
    payload = {'kind': 'INDEPENDENT_REVIEW_OWN_NATURAL_EXIT_IMMUTABLE_CAPSULE', 'epoch': args.epoch,
               'preexecution': record(pre), 'executed': record(executed), 'exit': record(exit_path),
               'stdout': record(stdout), 'stderr': record(stderr), 'request': record(request_path),
               'source_input_SHA256': before, 'source_inputs_unchanged_after_child': before == after,
               'selected_member_SHA256': members, 'natural_exit': True, 'returncode': code,
               'scientific_admission': False, 'native_ground_computed': False}
    if receipt.exists():
        payload['receipt'] = record(receipt)
    put(capsule, payload)
    print(json.dumps({'capsule': capsule.name, 'sha256': sha(capsule), 'returncode': code, 'natural_exit': True}))
    if code or before != after:
        raise SystemExit(code or 1)


if __name__ == '__main__':
    main()
