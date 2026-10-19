"""Capture natural subprocess exit and immutable nonlocal certificate bytes."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'joined_phi_u1_nonlocal_signal_interval_certificate.py'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--capsule', type=Path, required=True)
    args = parser.parse_args()
    before = SOURCE.read_bytes()
    own = Path(__file__).read_bytes()
    command = [sys.executable, str(SOURCE), '--output', str(args.output)]
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
    elapsed = time.monotonic()-started
    unchanged = before == SOURCE.read_bytes() and own == Path(__file__).read_bytes()
    capture = args.capsule.with_name(args.capsule.stem+'_runtime_bytes.zip')
    receipt_bytes = args.output.read_bytes() if args.output.exists() else b''
    with zipfile.ZipFile(capture,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(SOURCE.name,before)
        z.writestr(Path(__file__).name,own)
        z.writestr('stdout.txt',result.stdout)
        z.writestr('stderr.txt',result.stderr)
        z.writestr(args.output.name,receipt_bytes)
    capsule = dict(command=command,cwd=str(Path.cwd()),natural_exit_code=result.returncode,
        natural_exit_observed=True,timed_out=False,source_unchanged=unchanged,
        source_sha256=hashlib.sha256(before).hexdigest(),driver_sha256=hashlib.sha256(own).hexdigest(),
        receipt_path=str(args.output),receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        stdout=result.stdout.decode(),stderr=result.stderr.decode(),elapsed_seconds=elapsed,
        runtime_archive=dict(path=capture.name,sha256=hashlib.sha256(capture.read_bytes()).hexdigest()),
        runtime=dict(executable=sys.executable,python=sys.version))
    args.capsule.write_text(json.dumps(capsule,indent=2)+'\n')
    print(json.dumps(capsule),flush=True)
    if not unchanged:
        raise ArithmeticError('Source mutated during execution')
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
