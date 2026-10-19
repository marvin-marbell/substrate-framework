"""Execute only the independent review consumer with immutable before/after archives."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'
PREFIX = 'common_filter_form_review_actual1_20261012'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    snapshots = {name: (ROOT / name).read_bytes() for name in request['pins']}
    assert all(digest(raw) == request['pins'][name] for name, raw in snapshots.items())
    snapshots[args.request.name] = args.request.read_bytes()
    outputs = {suffix: ROOT / (PREFIX + suffix) for suffix in ('_preexecution.zip', '_executed.zip', '_receipt.json', '_stdout.txt', '_stderr.txt', '_exit.json', '_capsule.json')}
    assert not any(path.exists() for path in outputs.values()), 'Refuse to overwrite any epoch evidence'
    with zipfile.ZipFile(outputs['_preexecution.zip'], 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(snapshots.items()):
            archive.writestr(name, raw)
    outputs['_preexecution.zip'].chmod(0o444)
    command = [PYTHON, str(ROOT / 'common_filter_form_review_consumer_20261012.py'), '--request', str(args.request), '--output', str(outputs['_receipt.json'])]
    started = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
    elapsed = time.monotonic() - clock
    runtime = {'command': command, 'started_at': started, 'completed_at': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': elapsed, 'returncode': completed.returncode, 'natural_exit': True}
    outputs['_stdout.txt'].write_bytes(completed.stdout)
    outputs['_stderr.txt'].write_bytes(completed.stderr)
    save_json(outputs['_exit.json'], runtime)
    after = {name: digest((ROOT / name).read_bytes()) for name in request['pins']}
    assert after == request['pins'], 'Pinned review inputs changed during execution'
    assert args.request.read_bytes() == snapshots[args.request.name]
    with zipfile.ZipFile(outputs['_executed.zip'], 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(snapshots.items()):
            archive.writestr(name, raw)
        for suffix in ('_stdout.txt', '_stderr.txt', '_exit.json', '_receipt.json'):
            if outputs[suffix].exists():
                archive.writestr(outputs[suffix].name, outputs[suffix].read_bytes())
    capsule = {'kind': 'IMMUTABLE_INDEPENDENT_NATIVE_FILTER_FORM_REVIEW_CAPSULE', 'pins_before': request['pins'], 'pins_after': after, 'inputs_unchanged': True,
               'preexecution': {'path': outputs['_preexecution.zip'].name, 'sha256': digest(outputs['_preexecution.zip'].read_bytes())},
               'executed': {'path': outputs['_executed.zip'].name, 'sha256': digest(outputs['_executed.zip'].read_bytes())},
               'outputs': {path.name: digest(path.read_bytes()) for suffix, path in outputs.items() if suffix != '_capsule.json' and path.exists()},
               'runtime': runtime, 'originals_executed_or_modified': False, 'native_ground_or_phase_claimed': False}
    save_json(outputs['_capsule.json'], capsule)
    for path in outputs.values():
        if path.exists():
            path.chmod(0o444)
    print(json.dumps({'capsule': outputs['_capsule.json'].name, 'sha256': digest(outputs['_capsule.json'].read_bytes()), 'runtime': runtime}))
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
