"""Freeze inputs, execute only the new reviewer in isolation, preserve natural exit."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def save(path: Path, raw: bytes) -> None:
    with path.open('xb') as stream:
        stream.write(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--epoch', required=True)
    args = parser.parse_args()
    assert args.epoch.startswith('magnetic_band_review_')
    request_raw = args.request.read_bytes()
    request = json.loads(request_raw)
    assert request['consumer'] == 'magnetic_band_review_consumer_corrected_20261012.py'
    source_paths = {name: Path(request.get('source_paths', {}).get(name, str(ROOT/name))) for name in request['pins']}
    snapshots = {name: path.read_bytes() for name, path in source_paths.items()}
    before = {name: sha(raw) for name, raw in snapshots.items()}
    assert before == request['pins']
    assert all(Path(name).name == name for name in snapshots)
    snapshots[args.request.name] = request_raw
    paths = {kind: ROOT/(args.epoch+'_'+kind+suffix) for kind, suffix in
             [('preexecution', '.zip'), ('executed', '.zip'), ('receipt', '.json'),
              ('stdout', '.txt'), ('stderr', '.txt'), ('exit', '.json'), ('capsule', '.json')]}
    assert not any(path.exists() for path in paths.values())
    captured = dt.datetime.now(dt.timezone.utc).isoformat()
    with zipfile.ZipFile(paths['preexecution'], 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in snapshots.items():
            archive.writestr(name, raw)
    with tempfile.TemporaryDirectory(prefix=args.epoch+'_isolated_', dir=ROOT) as directory:
        sandbox = Path(directory)
        for name, raw in snapshots.items():
            save(sandbox/name, raw)
        argv = [sys.executable, str(sandbox/request['consumer']), '--request', str(sandbox/args.request.name),
                '--output', str(sandbox/paths['receipt'].name)]
        started = time.monotonic()
        child = subprocess.run(argv, cwd=sandbox, capture_output=True, check=False)
        elapsed = time.monotonic()-started
        isolated_after = {name: sha((sandbox/name).read_bytes()) for name in request['pins']}
        isolated_request_after = sha((sandbox/args.request.name).read_bytes())
        if (sandbox/paths['receipt'].name).exists():
            save(paths['receipt'], (sandbox/paths['receipt'].name).read_bytes())
    save(paths['stdout'], child.stdout)
    save(paths['stderr'], child.stderr)
    exit_data = {'returncode': child.returncode, 'natural_exit': child.returncode >= 0,
                 'elapsed_seconds': elapsed, 'completed_at': dt.datetime.now(dt.timezone.utc).isoformat(),
                 'isolated_process_completed': True, 'killed_or_timeout': child.returncode < 0}
    save(paths['exit'], (json.dumps(exit_data, indent=2)+'\n').encode())
    after = {name: sha(path.read_bytes()) for name, path in source_paths.items()}
    request_after = sha(args.request.read_bytes())
    outputs = [paths['stdout'], paths['stderr'], paths['exit']]
    if paths['receipt'].exists():
        outputs.append(paths['receipt'])
    with zipfile.ZipFile(paths['executed'], 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in snapshots.items():
            archive.writestr(name, raw)
        for path in outputs:
            archive.writestr(path.name, path.read_bytes())
    stable = before == after == isolated_after and sha(request_raw) == request_after == isolated_request_after
    capsule = {'schema': 1, 'kind': 'INDEPENDENT_ISOLATED_MAGNETIC_REVIEW_IMMUTABLE_NATURAL_EXIT_EPOCH',
               'epoch': args.epoch, 'argv': argv, 'captured_before_execution': captured,
               'source_paths': {name: str(path) for name, path in source_paths.items()},
               'source_input_sha256_before': before, 'source_input_sha256_after': after,
               'isolated_source_input_sha256_after': isolated_after,
               'request_sha256_before': sha(request_raw), 'request_sha256_after': request_after,
               'isolated_request_sha256_after': isolated_request_after,
               'preexecution': {'path': paths['preexecution'].name, 'sha256': sha(paths['preexecution'].read_bytes())},
               'executed': {'path': paths['executed'].name, 'sha256': sha(paths['executed'].read_bytes())},
               'outputs': {path.name: sha(path.read_bytes()) for path in outputs}, 'runtime': exit_data,
               'source_inputs_and_originals_unchanged': stable, 'temporary_execution_directory_removed': True,
               'original_producer_solver_root_checker_executed': False, 'native_numerical_ground_fullH_backend': False,
               'scientific_admission': False, 'campaign_complete': False}
    save(paths['capsule'], (json.dumps(capsule, indent=2)+'\n').encode())
    print(json.dumps({'capsule': paths['capsule'].name, 'runtime': exit_data, 'source_inputs_stable': stable}))
    if child.returncode:
        sys.exit(child.returncode)
    if not stable:
        sys.exit(2)


if __name__ == '__main__':
    main()
