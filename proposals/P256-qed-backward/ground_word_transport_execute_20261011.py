"""Freeze NEW word consumer inputs, execute synchronously, capture natural exit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def freeze(path: Path, members: dict[str, bytes]) -> str:
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(members.items()):
            archive.writestr(name, data)
    os.chmod(path, 0o444)
    return sha(path.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--stem', required=True)
    args = parser.parse_args()
    request_raw = args.request.read_bytes()
    request = json.loads(request_raw)
    raw = {name: (ROOT/name).read_bytes() for name in request['pins']}
    assert {name: sha(data) for name, data in raw.items()} == request['pins']
    members = {('external/'+Path(name).name if Path(name).is_absolute() else name): data for name, data in raw.items()}
    members[args.request.name] = request_raw
    pre = ROOT/(args.stem+'_preexecution.zip')
    prehash = freeze(pre, members)
    receipt = ROOT/(args.stem+'_receipt.json')
    command = [PYTHON, str(ROOT/request['consumer']), '--request', str(args.request), '--output', str(receipt)]
    started = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT, capture_output=True)
    elapsed = time.monotonic()-started
    stdout, stderr = ROOT/(args.stem+'_stdout.txt'), ROOT/(args.stem+'_stderr.txt')
    stdout.write_bytes(completed.stdout)
    stderr.write_bytes(completed.stderr)
    after = {name: sha((ROOT/name).read_bytes()) for name in request['pins']}
    stable = after == request['pins'] and args.request.read_bytes() == request_raw
    executed = dict(members)
    executed[stdout.name] = completed.stdout
    executed[stderr.name] = completed.stderr
    if receipt.exists():
        executed[receipt.name] = receipt.read_bytes()
    exe = ROOT/(args.stem+'_executed.zip')
    exehash = freeze(exe, executed)
    capsule = {'schema': 1, 'command': command, 'returncode': completed.returncode,
        'natural_completed_exit': completed.returncode == 0,
        'timeout_or_kill_used': False, 'elapsed_seconds': elapsed,
        'source_input_unchanged': stable, 'source_input_sha256_before': request['pins'],
        'source_input_sha256_after': after, 'request_sha256': sha(request_raw),
        'preexecution': {'path': str(pre), 'sha256': prehash},
        'executed': {'path': str(exe), 'sha256': exehash},
        'receipt': {'path': str(receipt), 'sha256': sha(receipt.read_bytes()) if receipt.exists() else None},
        'stdout': {'path': str(stdout), 'sha256': sha(completed.stdout)},
        'stderr': {'path': str(stderr), 'sha256': sha(completed.stderr)}}
    out = ROOT/(args.stem+'_capsule.json')
    out.write_text(json.dumps(capsule, indent=2)+'\n')
    print(json.dumps({'capsule': str(out), 'returncode': completed.returncode,
        'elapsed_seconds': elapsed, 'source_input_unchanged': stable}))
    if completed.returncode != 0 or not stable:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
