"""Capture immutable source/input bytes before executing only the new form controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    if not args.prefix.startswith('ground_weighted_form_review_'):
        raise ValueError('Only owned-prefix outputs are allowed')
    request_raw = args.input.read_bytes()
    request = json.loads(request_raw)
    names = list(request['pins']) + [args.input.name]
    snapshot = {name: (ROOT/name).read_bytes() for name in names}
    before = {name: digest(raw) for name, raw in snapshot.items()}
    if {name: before[name] for name in request['pins']} != request['pins']:
        raise ValueError('Preexecution pinned source/input mismatch')
    preexecution = ROOT/(args.prefix+'_preexecution.zip')
    executed = ROOT/(args.prefix+'_executed.zip')
    receipt = ROOT/(args.prefix+'_receipt.json')
    stdout = ROOT/(args.prefix+'_stdout.txt')
    stderr = ROOT/(args.prefix+'_stderr.txt')
    capsule = ROOT/(args.prefix+'_capsule.json')
    outputs = (preexecution, executed, receipt, stdout, stderr, capsule)
    if any(path.exists() for path in outputs):
        raise FileExistsError('Immutable execution outputs already exist')
    with zipfile.ZipFile(preexecution, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, raw in snapshot.items():
            archive.writestr(Path(name).name, raw)
    argv = [sys.executable, str(ROOT/request['consumer']), '--input', str(args.input), '--output', str(receipt)]
    started = time.monotonic()
    process = subprocess.run(argv, cwd=ROOT, capture_output=True, check=False)
    elapsed = time.monotonic()-started
    stdout.write_bytes(process.stdout); stderr.write_bytes(process.stderr)
    after = {name: digest((ROOT/name).read_bytes()) for name in names}
    with zipfile.ZipFile(executed, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, raw in snapshot.items():
            archive.writestr(Path(name).name, raw)
        for path in (receipt, stdout, stderr):
            if path.exists(): archive.writestr(path.name, path.read_bytes())
        archive.writestr(args.prefix+'_exit.json', json.dumps({'argv': argv,
            'natural_exit': True, 'returncode': process.returncode, 'elapsed_seconds': elapsed,
            'source_input_sha256_before': before, 'source_input_sha256_after': after}, indent=2)+'\n')
    result = {'kind': 'ACTUAL_INDEPENDENT_WEIGHTED_FORM_CONTROL_EXECUTION_CAPSULE',
        'argv': argv, 'natural_exit': True, 'returncode': process.returncode,
        'elapsed_seconds': elapsed, 'source_input_sha256_before': before,
        'source_input_sha256_after': after, 'sources_inputs_unchanged': before == after,
        'preexecution': {'path': preexecution.name, 'sha256': digest(preexecution.read_bytes())},
        'executed': {'path': executed.name, 'sha256': digest(executed.read_bytes())},
        'stdout': {'path': stdout.name, 'sha256': digest(process.stdout)},
        'stderr': {'path': stderr.name, 'sha256': digest(process.stderr)},
        'receipt': {'path': receipt.name, 'sha256': digest(receipt.read_bytes())} if receipt.exists() else None,
        'scientific_admission': False, 'old_producer_imported_or_executed': False}
    capsule.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'capsule': capsule.name, 'natural_exit': True,
        'returncode': process.returncode, 'sources_inputs_unchanged': before == after}))
    if before != after: raise RuntimeError('Source/input bytes changed during execution')
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
