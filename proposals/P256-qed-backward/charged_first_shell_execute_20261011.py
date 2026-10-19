"""Immutable launch/natural-exit evidence for the NEW charged first-shell CLI."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
SCRIPT = 'charged_first_shell_consumer_20261011.py'
NAMES = [SCRIPT, Path(__file__).name, 'charged_first_shell_proof_20261011.json',
         'local_so5_resum_actual04_20261011_state.npz',
         'local_so5_resum_actual04_20261011_receipt.json',
         'local_so5_resum_source_20261011.py',
         'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json',
         'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json',
         'local_so5_positive_output_row_inertia_certificate_20261011_receipt.json',
         'local_so5_ground_electric_memory_consumer_20261011_receipt.json']


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def capture(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, 'x', zipfile.ZIP_DEFLATED) as z:
        for name, raw in sorted(members.items()):
            z.writestr(name, raw)
    path.chmod(0o444)


def run(epoch: str) -> None:
    assert epoch.isalnum()
    base = 'charged_first_shell_'+epoch+'_20261011'
    source = {name: (ROOT/name).read_bytes() for name in NAMES}
    output = ROOT/(base+'_receipt.json')
    capture(ROOT/(base+'_launch.zip'), source)
    command = [sys.executable, SCRIPT, '--output', str(output)]
    began = time.monotonic()
    env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True)
    additions = {base+'_stdout.txt': result.stdout, base+'_stderr.txt': result.stderr}
    unchanged = {name: (ROOT/name).read_bytes() == raw for name, raw in source.items()}
    record = dict(command=command, returncode=result.returncode, natural_exit=result.returncode >= 0,
                  source_and_input_sha256={name: sha(raw) for name, raw in source.items()},
                  unchanged=unchanged, stdout_sha256=sha(result.stdout), stderr_sha256=sha(result.stderr),
                  runtime_executable=sys.executable, elapsed_seconds=time.monotonic()-began,
                  maxrss_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                  scope='NEW charged first-shell only; no old producer/Root/solver/checker execution')
    additions[base+'_exit.json'] = (json.dumps(record, indent=2)+'\n').encode()
    for name, raw in additions.items():
        path = ROOT/name
        with path.open('xb') as f:
            f.write(raw)
        path.chmod(0o444)
    if output.exists():
        additions[output.name] = output.read_bytes()
    capture(ROOT/(base+'_natural_execution.zip'), {**source, **additions})
    print(json.dumps(record, indent=2))
    print(result.stdout.decode(), end='')
    print(result.stderr.decode(), end='', file=sys.stderr)
    assert all(unchanged.values())
    sys.exit(result.returncode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', required=True)
    run(parser.parse_args().epoch)
