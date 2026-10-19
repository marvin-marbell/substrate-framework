"""Immutable NEW principal-history consumer capsules; never rerun old producers."""
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


def save(path: Path, raw: bytes) -> None:
    with path.open('xb') as stream:
        stream.write(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--epoch', required=True)
    args = parser.parse_args()
    assert args.epoch.startswith('principal_history_')
    request_raw = args.request.read_bytes(); request = json.loads(request_raw)
    inputs = {name: (ROOT/name).read_bytes() for name in request['pins']}
    before = {name: sha(raw) for name, raw in inputs.items()}
    assert before == request['pins']
    snapshots = dict(inputs); snapshots[args.request.name] = request_raw
    member_map = {name: Path(name).name for name in snapshots}
    assert len(set(member_map.values())) == len(member_map)
    paths = {key: ROOT/(args.epoch+'_'+suffix) for key, suffix in
             [('pre', 'preexecution.zip'), ('executed', 'executed.zip'), ('receipt', 'receipt.json'),
              ('stdout', 'stdout.txt'), ('stderr', 'stderr.txt'), ('exit', 'exit.json'), ('capsule', 'capsule.json')]}
    assert not any(path.exists() for path in paths.values())
    captured = dt.datetime.now(dt.timezone.utc).isoformat()
    with zipfile.ZipFile(paths['pre'], 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in snapshots.items():
            archive.writestr(member_map[name], raw)
    argv = [sys.executable, str(ROOT/request['consumer']), '--request', str(args.request), '--output', str(paths['receipt'])]
    started = time.monotonic()
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, check=False)
    elapsed = time.monotonic()-started
    save(paths['stdout'], result.stdout); save(paths['stderr'], result.stderr)
    after = {name: sha((ROOT/name).read_bytes()) for name in request['pins']}
    exit_record = {'returncode': result.returncode, 'natural_exit': result.returncode >= 0,
                   'elapsed_seconds': elapsed, 'completed_at': dt.datetime.now(dt.timezone.utc).isoformat()}
    save(paths['exit'], (json.dumps(exit_record, indent=2)+'\n').encode())
    generated = [paths['stdout'], paths['stderr'], paths['exit']]
    if paths['receipt'].exists():
        generated.append(paths['receipt'])
    with zipfile.ZipFile(paths['executed'], 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in snapshots.items():
            archive.writestr(member_map[name], raw)
        for path in generated:
            archive.writestr(path.name, path.read_bytes())
    capsule = {'schema': 1, 'kind': 'IMMUTABLE_NEW_PRINCIPAL_HISTORY_NATURAL_EXIT_CAPSULE',
               'captured_before_execution_at': captured, 'argv': argv, 'cwd': str(ROOT),
               'source_input_sha256_before': before, 'source_input_sha256_after': after,
               'request_sha256_before': sha(request_raw), 'request_sha256_after': sha(args.request.read_bytes()),
               'archive_member_map': member_map,
               'preexecution': {'path': paths['pre'].name, 'sha256': sha(paths['pre'].read_bytes())},
               'executed': {'path': paths['executed'].name, 'sha256': sha(paths['executed'].read_bytes())},
               'outputs': {path.name: sha(path.read_bytes()) for path in generated}, 'runtime': exit_record,
               'sources_inputs_unchanged': before == after and request_raw == args.request.read_bytes(),
               'old_producer_root_solver_checker_run': False,
               'native_ground_or_fullH_numerics': False}
    save(paths['capsule'], (json.dumps(capsule, indent=2)+'\n').encode())
    print(json.dumps({'capsule': paths['capsule'].name, 'runtime': exit_record,
                      'unchanged': capsule['sources_inputs_unchanged']}))
    if result.returncode:
        sys.exit(result.returncode)


if __name__ == '__main__':
    main()
