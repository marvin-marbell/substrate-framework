"""Capture the actual NEW-Jordan all-flat CLI's natural exit and immutable bytes."""
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


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--starts', type=int, default=2)
    args = parser.parse_args()
    target = ROOT/'joined_phi_u1_matrix_jordan_flat_bounds.py'
    sources = {target.name: target.read_bytes(), Path(__file__).name: Path(__file__).read_bytes()}
    command = [sys.executable, str(target), '--output', str(args.output), '--iterations', str(args.iterations), '--starts', str(args.starts)]
    started = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
    stdout = args.output.with_name(args.output.stem+'_stdout.txt')
    stderr = args.output.with_name(args.output.stem+'_stderr.txt')
    stdout.write_bytes(completed.stdout)
    stderr.write_bytes(completed.stderr)
    execution_archive = args.output.with_name(args.output.stem+'_execution_bytes.zip')
    with zipfile.ZipFile(execution_archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, raw in sources.items():
            z.writestr(name, raw)
        z.writestr(stdout.name, completed.stdout)
        z.writestr(stderr.name, completed.stderr)
    stable = {name: (ROOT/name).read_bytes() == raw for name, raw in sources.items()}
    receipt = json.loads(args.output.read_text()) if completed.returncode == 0 else None
    capsule = dict(schema=1, command=command, cwd=str(ROOT), elapsed_seconds=time.monotonic()-started, natural_exit_code=completed.returncode, source_sha256_before={name: sha(raw) for name, raw in sources.items()}, source_unchanged=stable, stdout={'path': str(stdout), 'sha256': sha(completed.stdout)}, stderr={'path': str(stderr), 'sha256': sha(completed.stderr)}, execution_archive={'path': str(execution_archive), 'sha256': sha(execution_archive.read_bytes())}, receipt={'path': str(args.output), 'sha256': sha(args.output.read_bytes())} if receipt is not None else None, immutable_input_archive={'path': receipt['immutable_input_archive'], 'sha256': sha(Path(receipt['immutable_input_archive']).read_bytes())} if receipt is not None else None, verdict=receipt['verdict'] if receipt is not None else None)
    capsule_path = args.output.with_name(args.output.stem+'_natural_exit_capsule.json')
    capsule_path.write_text(json.dumps(capsule, indent=2)+'\n')
    print(json.dumps(capsule, indent=2))
    if not all(stable.values()):
        raise RuntimeError('Source epoch changed during calculation')
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
