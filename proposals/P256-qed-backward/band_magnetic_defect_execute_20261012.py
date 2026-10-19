"""Execute only the new band-defect consumer with immutable pre/post bytes."""
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
    assert args.epoch.startswith('band_magnetic_defect_')
    request_raw = args.request.read_bytes(); request = json.loads(request_raw)
    snapshots = {name: (ROOT/name).read_bytes() for name in request['pins']}
    before = {name: sha(raw) for name, raw in snapshots.items()}
    assert before == request['pins']
    snapshots[args.request.name] = request_raw
    members = {name: Path(name).name for name in snapshots}
    assert len(set(members.values())) == len(members)
    paths = {kind: ROOT/(args.epoch+'_'+kind+suffix) for kind, suffix in
             [('preexecution', '.zip'), ('executed', '.zip'), ('receipt', '.json'),
              ('stdout', '.txt'), ('stderr', '.txt'), ('exit', '.json'), ('capsule', '.json')]}
    assert not any(path.exists() for path in paths.values())
    with zipfile.ZipFile(paths['preexecution'], 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in snapshots.items():
            archive.writestr(members[name], raw)
    captured = dt.datetime.now(dt.timezone.utc).isoformat()
    argv = [sys.executable, str(ROOT/request['consumer']), '--request', str(args.request),
            '--output', str(paths['receipt'])]
    started = time.monotonic()
    run = subprocess.run(argv, cwd=ROOT, capture_output=True, check=False)
    elapsed = time.monotonic()-started
    save(paths['stdout'], run.stdout); save(paths['stderr'], run.stderr)
    after = {name: sha((ROOT/name).read_bytes()) for name in request['pins']}
    exit_data = {'returncode': run.returncode, 'natural_exit': True, 'elapsed_seconds': elapsed,
                 'completed_at': dt.datetime.now(dt.timezone.utc).isoformat()}
    save(paths['exit'], (json.dumps(exit_data, indent=2)+'\n').encode())
    generated = [paths['stdout'], paths['stderr'], paths['exit']]
    if paths['receipt'].exists(): generated.append(paths['receipt'])
    with zipfile.ZipFile(paths['executed'], 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in snapshots.items(): archive.writestr(members[name], raw)
        for path in generated: archive.writestr(path.name, path.read_bytes())
    data = {'schema': 1, 'kind': 'IMMUTABLE_BAND_DEFECT_NATURAL_EXIT_EPOCH',
            'epoch': args.epoch, 'argv': argv, 'captured_before_execution': captured,
            'source_input_sha256_before': before, 'source_input_sha256_after': after,
            'request_sha256_before': sha(request_raw), 'request_sha256_after': sha(args.request.read_bytes()),
            'archive_member_map': members, 'runtime': exit_data,
            'preexecution': {'path': paths['preexecution'].name, 'sha256': sha(paths['preexecution'].read_bytes())},
            'executed': {'path': paths['executed'].name, 'sha256': sha(paths['executed'].read_bytes())},
            'outputs': {path.name: sha(path.read_bytes()) for path in generated},
            'sources_inputs_unchanged': before == after and request_raw == args.request.read_bytes(),
            'native_ground_or_fullH_backend': False, 'old_producer_root_solver_checker_run': False}
    save(paths['capsule'], (json.dumps(data, indent=2)+'\n').encode())
    print(json.dumps({'capsule': paths['capsule'].name, 'returncode': run.returncode,
                      'natural_exit': True, 'elapsed_seconds': elapsed}))
    if run.returncode: sys.exit(run.returncode)


if __name__ == '__main__':
    main()
