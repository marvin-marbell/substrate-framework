"""Exclusive pre/execution capture for the new independent exact review CLI."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def save(path: Path, value: bytes) -> None:
    with path.open('xb') as stream:
        stream.write(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--epoch', required=True)
    args = parser.parse_args()
    assert args.epoch.startswith('electric_band_review_')
    request_raw = args.request.read_bytes()
    request = json.loads(request_raw)
    snapshots = {name: (ROOT/name).read_bytes() for name in request['pins']}
    before = {name: sha(raw) for name, raw in snapshots.items()}
    assert before == request['pins']
    snapshots[args.request.name] = request_raw
    mapping = {name: Path(name).name for name in snapshots}
    assert len(set(mapping.values())) == len(mapping)
    paths = {suffix: ROOT/(args.epoch+'_'+suffix) for suffix in [
        'preexecution.zip', 'executed.zip', 'receipt.json', 'stdout.txt', 'stderr.txt', 'exit.json', 'capsule.json']}
    assert not any(path.exists() for path in paths.values())
    captured = dt.datetime.now(dt.timezone.utc).isoformat()
    with zipfile.ZipFile(paths['preexecution.zip'], 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in snapshots.items():
            archive.writestr(mapping[name], raw)
    pre_completed = dt.datetime.now(dt.timezone.utc).isoformat()
    argv = [sys.executable, str(ROOT/request['consumer']), '--request', str(args.request),
            '--output', str(paths['receipt.json'])]
    started = time.monotonic()
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, check=False)
    runtime = {'returncode': result.returncode, 'natural_exit': True,
               'elapsed_seconds': time.monotonic()-started,
               'completed_at': dt.datetime.now(dt.timezone.utc).isoformat()}
    save(paths['stdout.txt'], result.stdout)
    save(paths['stderr.txt'], result.stderr)
    save(paths['exit.json'], (json.dumps(runtime, indent=2)+'\n').encode())
    after = {name: sha((ROOT/name).read_bytes()) for name in request['pins']}
    outputs = [paths[suffix] for suffix in ['stdout.txt', 'stderr.txt', 'exit.json']]
    if paths['receipt.json'].exists():
        outputs.append(paths['receipt.json'])
    with zipfile.ZipFile(paths['executed.zip'], 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in snapshots.items():
            archive.writestr(mapping[name], raw)
        for path in outputs:
            archive.writestr(path.name, path.read_bytes())
    capsule = {'kind': 'IMMUTABLE_INDEPENDENT_EXACT_ELECTRIC_BAND_REVIEW_EXECUTION',
               'epoch': args.epoch, 'argv': argv, 'cwd': str(ROOT),
               'python_executable': sys.executable, 'python_version': sys.version,
               'captured_before_execution': captured, 'preexecution_completed_at': pre_completed,
               'source_input_sha256_before': before, 'source_input_sha256_after': after,
               'request_sha256_before': sha(request_raw),
               'request_sha256_after': sha(args.request.read_bytes()),
               'archive_member_map': mapping,
               'preexecution': {'path': paths['preexecution.zip'].name,
                                'sha256': sha(paths['preexecution.zip'].read_bytes())},
               'executed': {'path': paths['executed.zip'].name,
                            'sha256': sha(paths['executed.zip'].read_bytes())},
               'outputs': {path.name: sha(path.read_bytes()) for path in outputs},
               'runtime': runtime,
               'sources_inputs_unchanged': before == after and request_raw == args.request.read_bytes(),
               'old_producer_root_solver_checker_rerun': False,
               'native_fullH_or_ground_backend': False, 'scientific_admission': False}
    save(paths['capsule.json'], (json.dumps(capsule, indent=2)+'\n').encode())
    print(json.dumps({'capsule': paths['capsule.json'].name, 'runtime': runtime,
                      'sources_inputs_unchanged': capsule['sources_inputs_unchanged']}))
    if result.returncode:
        sys.exit(result.returncode)


if __name__ == '__main__':
    main()
