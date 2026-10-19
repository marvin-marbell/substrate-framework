"""Create a new immutable before-execution archive and natural-exit capsule."""
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, payload):
    with path.open('x') as stream:
        json.dump(payload, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    root = args.request.resolve().parent
    request = json.loads(args.request.read_bytes())
    prefix = request['epoch']
    paths = [root / name for name in request['archive_inputs']]
    paths.append(args.request.resolve())
    before = {path.name: digest(path) for path in paths}
    for name, expected in request['expected_pins'].items():
        assert before[name] == expected, (name, before[name], expected)
    archive = root / (prefix + '_preexecution.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            z.write(path, path.name)
    archive_sha = digest(archive)
    receipt = root / (prefix + '_receipt.json')
    command = [sys.executable, str(root / request['consumer']), '--input', str(root / request['input']),
               '--request', str(args.request.resolve()), '--output', str(receipt)]
    started = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    completed = subprocess.run(command, cwd=root, capture_output=True, check=False)
    elapsed = time.monotonic() - clock
    ended = datetime.now(timezone.utc).isoformat()
    stdout = root / (prefix + '_stdout.txt')
    stderr = root / (prefix + '_stderr.txt')
    with stdout.open('xb') as stream:
        stream.write(completed.stdout)
    with stderr.open('xb') as stream:
        stream.write(completed.stderr)
    after = {path.name: digest(path) for path in paths}
    exit_path = root / (prefix + '_exit.json')
    save(exit_path, {'returncode': completed.returncode, 'natural_exit': completed.returncode >= 0,
                     'started_utc': started, 'ended_utc': ended, 'elapsed_seconds': elapsed,
                     'command': command, 'source_input_pins_unchanged': before == after})
    executed = root / (prefix + '_executed.zip')
    outputs = [stdout, stderr, exit_path]
    if receipt.exists():
        outputs.append(receipt)
    with zipfile.ZipFile(executed, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        z.write(archive, archive.name)
        for path in paths + outputs:
            z.write(path, path.name)
    capsule = {'kind': 'NEW_INDEPENDENT_TRIAL_FORM_NATURAL_EXIT_CAPSULE',
               'governor': request['governor'], 'before_execution_archive': archive.name,
               'before_execution_archive_sha256': archive_sha,
               'executed_archive': executed.name, 'executed_archive_sha256': digest(executed),
               'source_input_pins_before': before, 'source_input_pins_after': after,
               'source_input_pins_unchanged': before == after,
               'returncode': completed.returncode, 'natural_exit': completed.returncode >= 0,
               'started_utc': started, 'ended_utc': ended, 'elapsed_seconds': elapsed,
               'command': command, 'output_pins': {p.name: digest(p) for p in outputs},
               'original_snapshot_is_postexecution': True,
               'old_supplier_or_main_executed': False, 'ground_solver_executed': False}
    capsule_path = root / (prefix + '_capsule.json')
    save(capsule_path, capsule)
    print(json.dumps({'capsule': capsule_path.name, 'returncode': completed.returncode,
                      'elapsed_seconds': elapsed, 'natural_exit': completed.returncode >= 0,
                      'pins_unchanged': before == after}), flush=True)
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
