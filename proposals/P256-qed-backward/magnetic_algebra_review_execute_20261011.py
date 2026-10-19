"""Preserve independent magnetic-review CLI epochs and immutable input bytes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'
SOURCE = 'magnetic_algebra_review_control_20261011.py'
REQUEST = 'magnetic_algebra_review_request_20261011.json'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(epoch: str) -> None:
    prefix = 'magnetic_algebra_review_' + epoch + '_20261011'
    request = json.loads((ROOT / REQUEST).read_text())
    members = [SOURCE, REQUEST, Path(__file__).name, *request['inputs']]
    outputs = [prefix + suffix for suffix in ('_preexecution.zip', '_executed.zip', '_receipt.json', '_stdout.txt', '_stderr.txt', '_exit.json', '_capsule.json')]
    if any((ROOT / name).exists() for name in outputs):
        raise FileExistsError('Epoch exists; preserve it and use a new epoch')
    before = {name: digest(ROOT / name) for name in members}
    with zipfile.ZipFile(ROOT / outputs[0], 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in members:
            archive.write(ROOT / name, arcname=name)
    command = [PYTHON, str(ROOT / SOURCE), '--request', str(ROOT / REQUEST), '--output', str(ROOT / outputs[2])]
    start = time.monotonic()
    with (ROOT / outputs[3]).open('xb') as stdout, (ROOT / outputs[4]).open('xb') as stderr:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        code = process.wait()
    elapsed = time.monotonic() - start
    (ROOT / outputs[5]).write_text(json.dumps(dict(command=command, exit_code=code, natural_exit=True, seconds=elapsed, signal=(-code if code < 0 else None)), indent=2) + '\n')
    after = {name: digest(ROOT / name) for name in members}
    with zipfile.ZipFile(ROOT / outputs[1], 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in members:
            archive.write(ROOT / name, arcname=name)
        for name in outputs[2:6]:
            if (ROOT / name).exists():
                archive.write(ROOT / name, arcname=name)
    capsule = dict(epoch=epoch, source_sha256=before[SOURCE], preexecution_input_sha256=before,
                   execution_input_sha256=after, all_input_bytes_unchanged=before == after,
                   exit_code=code, natural_exit=True, seconds=elapsed,
                   artifacts_sha256={name: digest(ROOT / name) for name in outputs[:-1] if (ROOT / name).exists()},
                   target_main_or_solver_run=False, full_ground_or_phase_or_pole_admitted=False)
    (ROOT / outputs[6]).write_text(json.dumps(capsule, indent=2) + '\n')
    print(json.dumps(capsule))
    if code != 0:
        raise SystemExit(code)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', required=True)
    run(parser.parse_args().epoch)
