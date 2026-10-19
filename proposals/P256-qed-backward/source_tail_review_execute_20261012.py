"""Capture this independent CLI's exact inputs and natural subprocess exit."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PYTHON = Path('/home/administrator/substrate-framework/.venv/bin/python')


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def archive(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, raw in sorted(members.items()):
            bundle.writestr(name, raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    request: dict[str, Any] = json.loads(args.request.read_text())
    pins: dict[str, str] = request['pins']
    selected = {name: (ROOT/name).read_bytes() for name in pins}
    assert all(digest(raw) == pins[name] for name, raw in selected.items())
    selected[args.request.name] = args.request.read_bytes()
    pre = ROOT/(args.prefix+'_preexecution.zip')
    archive(pre, selected)
    python_sha = digest(PYTHON.read_bytes())
    output = ROOT/(args.prefix+'_result.json')
    command = [str(PYTHON), str(ROOT/request['consumer']), '--request', str(args.request), '--output', str(output)]
    start = time.monotonic()
    process = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
    elapsed = time.monotonic()-start
    unchanged = all((ROOT/name).read_bytes() == raw for name, raw in selected.items())
    stdout = ROOT/(args.prefix+'_stdout.txt')
    stderr = ROOT/(args.prefix+'_stderr.txt')
    stdout.write_bytes(process.stdout)
    stderr.write_bytes(process.stderr)
    exit_record = {'command': command, 'returncode': process.returncode,
                   'natural_exit': process.returncode >= 0, 'elapsed_seconds': elapsed,
                   'selected_bytes_unchanged': unchanged,
                   'python_executable_sha256': python_sha,
                   'python_executable_unchanged': digest(PYTHON.read_bytes()) == python_sha}
    exit_path = ROOT/(args.prefix+'_exit.json')
    exit_path.write_text(json.dumps(exit_record, indent=2)+'\n')
    members = dict(selected)
    for path in (stdout, stderr, exit_path, output):
        if path.exists():
            members[path.name] = path.read_bytes()
    executed = ROOT/(args.prefix+'_executed.zip')
    archive(executed, members)
    capsule = dict(exit_record, preexecution=str(pre), preexecution_sha256=digest(pre.read_bytes()),
                   executed=str(executed), executed_sha256=digest(executed.read_bytes()),
                   selected_hashes={name: digest(raw) for name, raw in selected.items()},
                   result=str(output), result_exists=output.exists(),
                   result_sha256=digest(output.read_bytes()) if output.exists() else None)
    capsule_path = ROOT/(args.prefix+'_capsule.json')
    capsule_path.write_text(json.dumps(capsule, indent=2)+'\n')
    print(json.dumps({'capsule': str(capsule_path), 'returncode': process.returncode,
                      'natural_exit': process.returncode >= 0, 'elapsed_seconds': elapsed,
                      'unchanged': unchanged}))
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
