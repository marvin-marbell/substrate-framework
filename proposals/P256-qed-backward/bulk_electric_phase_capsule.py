"""Execute only the owned new scientific program and retain every failed epoch."""
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
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', required=True)
    args = parser.parse_args()
    if sys.executable != PYTHON:
        raise RuntimeError('Actual project Python is required, not a fallback interpreter')
    if not args.epoch.startswith('bulk_electric_phase_') or '/' in args.epoch:
        raise ValueError('Only owned bulk_electric_phase_ epochs are permitted')
    stem = ROOT/args.epoch
    existing = list(ROOT.glob(args.epoch+'*'))
    if existing:
        raise FileExistsError(f'Preserve the old epoch: {existing}')
    output = stem.with_suffix('.json')
    argv = [PYTHON, '-B', str(ROOT/'bulk_electric_phase_consumer.py'), '--output', str(output)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    started = time.time()
    process = subprocess.run(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    paths = {'stdout': stem.with_name(stem.name+'_stdout.txt'),
             'stderr': stem.with_name(stem.name+'_stderr.txt')}
    paths['stdout'].write_bytes(process.stdout)
    paths['stderr'].write_bytes(process.stderr)
    artifacts = {}
    for path in sorted(ROOT.glob(args.epoch+'*')):
        if path.is_file():
            artifacts[path.name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
    receipt = dict(kind='OWNED_NEW_SCIENTIFIC_PROGRAM_NATURAL_PROCESS_EXIT', argv=argv,
                   project_python=sys.executable, epoch=args.epoch, returncode=process.returncode,
                   natural_exit=True, timeout_or_kill=False, started_unix=started,
                   elapsed_seconds=time.time()-started, artifacts=artifacts,
                   runner_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   no_old_producer_or_checker_invoked=True,
                   failed_epoch_preserved=process.returncode != 0)
    receipt_path = stem.with_name(stem.name+'_exit.json')
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    execution = stem.with_name(stem.name+'_natural_execution.zip')
    with zipfile.ZipFile(execution, 'x', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('exit.json', receipt_path.read_bytes())
        archive.writestr('runner.py', Path(__file__).read_bytes())
        for path in paths.values():
            archive.writestr(path.name, path.read_bytes())
        for suffix in ('_preexecution.zip', '_executed.zip'):
            path = stem.with_name(stem.name+suffix)
            if path.is_file():
                archive.write(path, path.name)
        if output.is_file():
            archive.write(output, output.name)
    execution.chmod(0o444)
    for path in (receipt_path, *paths.values(), output):
        if path.exists():
            path.chmod(0o444)
    print(json.dumps(receipt, indent=2))
    if process.stdout:
        print(process.stdout.decode(), end='')
    if process.stderr:
        print(process.stderr.decode(), file=sys.stderr, end='')
    return process.returncode


if __name__ == '__main__':
    raise SystemExit(main())
