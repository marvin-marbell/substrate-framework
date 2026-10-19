"""Exact owned inputs/source capsule and natural process exit, not just a stage marker."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_frame_gauss_join_20261015_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def digest(path: Path) -> dict:
    return dict(file=path.name, bytes=path.stat().st_size, SHA256=hashlib.sha256(path.read_bytes()).hexdigest())


def main() -> None:
    run = sys.argv[1]
    target = ROOT/(PREFIX+run)
    selected = [ROOT/(PREFIX+suffix) for suffix in ['model.py', 'consume.py', 'capture.py', 'input.json', 'equations.json', 'first_physical_synthesis.json']]
    before = {path.name:digest(path) for path in selected}
    source_capsule = Path(str(target)+'_source_capsule.zip')
    with zipfile.ZipFile(source_capsule, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in selected:
            archive.write(path, path.name)
    command = [PYTHON, PREFIX+'consume.py', PREFIX+'input.json', target.name]
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    receipt = dict(command=command, cwd=str(ROOT), environment_subset={k:environment[k] for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']},
                   input_source=before, source_capsule=digest(source_capsule), start_unix=time.time())
    Path(str(target)+'_start.json').write_text(json.dumps(receipt, indent=2)+'\n')
    start = time.monotonic()
    with Path(str(target)+'_stdout.txt').open('w') as stdout, Path(str(target)+'_stderr.txt').open('w') as stderr:
        completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
    receipt.update(natural_exit=True, returncode=completed.returncode, elapsed_seconds=time.monotonic()-start,
                   sources_and_inputs_unchanged=all(digest(path)==before[path.name] for path in selected))
    outputs = sorted(ROOT.glob(target.name+'_*'))
    receipt['outputs'] = [digest(path) for path in outputs if path.is_file() and path.suffix != '.zip']
    Path(str(target)+'_exit.json').write_text(json.dumps(receipt, indent=2)+'\n')
    with zipfile.ZipFile(str(target)+'_capsule.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in selected+outputs+[Path(str(target)+'_exit.json')]:
            if path.is_file() and path.suffix != '.zip':
                archive.write(path, path.name)
    print(json.dumps(dict(returncode=completed.returncode, natural_exit=True, elapsed_seconds=receipt['elapsed_seconds'], sources_and_inputs_unchanged=receipt['sources_and_inputs_unchanged'], exit=target.name+'_exit.json')), flush=True)
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
