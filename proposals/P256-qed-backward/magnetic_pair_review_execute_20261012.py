"""Freeze and run only this independent consumer in an isolated working directory."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def save(name: str, raw: bytes) -> None:
    with (ROOT / name).open('xb') as stream:
        stream.write(raw)


def json_save(name: str, value: object) -> None:
    save(name, (json.dumps(value, indent=2, sort_keys=True) + '\n').encode())


def archive(name: str, members: dict[str, bytes]) -> dict[str, str]:
    with zipfile.ZipFile(ROOT / name, 'x', compression=zipfile.ZIP_DEFLATED) as output:
        for member, raw in members.items(): output.writestr(member, raw)
    return {'path': name, 'sha256': digest((ROOT / name).read_bytes())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    request_raw = args.request.read_bytes()
    request = json.loads(request_raw)
    epoch = request['epoch']
    assert epoch.startswith('magnetic_pair_review_actual')
    snapshots = {name: (ROOT / name).read_bytes() for name in request['pins']}
    before = {name: digest(raw) for name, raw in snapshots.items()}
    assert before == request['pins'] and request['consumer'] in snapshots
    snapshots[args.request.name] = request_raw
    pre = archive(epoch + '_preexecution.zip', snapshots)
    captured_at = dt.datetime.now(dt.timezone.utc).isoformat()
    result_name = epoch + '_result.json'
    with tempfile.TemporaryDirectory(prefix='magnetic-pair-independent-') as directory:
        isolated = Path(directory)
        for name in (request['consumer'], request['input_archive']):
            with (isolated / name).open('xb') as stream: stream.write(snapshots[name])
        argv = [PYTHON, str(isolated / request['consumer']), '--input-archive',
                str(isolated / request['input_archive']), '--output', str(isolated / result_name)]
        started = time.monotonic()
        child = subprocess.run(argv, cwd=isolated, capture_output=True, check=False)
        elapsed = time.monotonic() - started
        result_raw = (isolated / result_name).read_bytes() if (isolated / result_name).exists() else None
        isolated_unchanged = all((isolated / name).read_bytes() == snapshots[name]
                                 for name in (request['consumer'], request['input_archive']))
    outputs = {epoch + '_stdout.txt': child.stdout, epoch + '_stderr.txt': child.stderr}
    runtime = {'returncode': child.returncode, 'natural_exit': True,
               'elapsed_seconds': elapsed, 'completed_at': dt.datetime.now(dt.timezone.utc).isoformat(),
               'isolated_child_completed': True, 'killed_or_timeout': False}
    outputs[epoch + '_exit.json'] = (json.dumps(runtime, indent=2) + '\n').encode()
    if result_raw is not None: outputs[result_name] = result_raw
    for name, raw in outputs.items(): save(name, raw)
    after = {name: digest((ROOT / name).read_bytes()) for name in request['pins']}
    original_after = {name: digest((ROOT / name).read_bytes()) for name in request['original_live_pins']}
    stable = before == after and args.request.read_bytes() == request_raw and isolated_unchanged
    original_stable = original_after == request['original_live_pins']
    executed = archive(epoch + '_executed.zip', {**snapshots, **outputs})
    capsule = {'kind': 'INDEPENDENT_IMMUTABLE_MAGNETIC_PAIR_REVIEW_SOURCE_SELECTED_NATURAL_EXIT',
               'argv': argv, 'isolated_cwd': directory, 'captured_before_execution': captured_at,
               'request_sha256_before': digest(request_raw), 'request_sha256_after': digest(args.request.read_bytes()),
               'source_input_sha256_before': before, 'source_input_sha256_after': after,
               'sources_inputs_unchanged': stable, 'original_target_live_pins_after': original_after,
               'original_targets_unchanged': original_stable, 'preexecution': pre, 'executed': executed,
               'output_sha256': {name: digest(raw) for name, raw in outputs.items()}, 'runtime': runtime,
               'old_producer_root_solver_checker_run': False, 'native_ground_fullH_backend': False,
               'scientific_admission': False, 'campaign_complete': False}
    json_save(epoch + '_capsule.json', capsule)
    if child.returncode: raise SystemExit(child.returncode)
    assert stable and original_stable and result_raw is not None
    print(json.dumps({'capsule': epoch + '_capsule.json', 'runtime': runtime,
                      'sources_inputs_unchanged': stable, 'original_targets_unchanged': original_stable}))


if __name__ == '__main__':
    main()
