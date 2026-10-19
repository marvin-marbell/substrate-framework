"""Unique-epoch source/input/stdout/stderr and natural-exit capture."""
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
PREFIX = 'rich_composite_local_flux_binding_20261017_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    epoch = sys.argv[1]
    stem = ROOT/(PREFIX+epoch+'_')
    if Path(str(stem)+'exit.json').exists() or Path(str(stem)+'source.zip').exists():
        raise FileExistsError('Never overwrite a captured physical epoch')
    input_path = ROOT/(PREFIX+epoch+'_input.json')
    command = [PYTHON, str(ROOT/(PREFIX+'execute.py')), str(input_path)]
    sources = sorted(ROOT.glob(PREFIX+'*.py'))+[ROOT/(PREFIX+'equations.json'), input_path]
    with zipfile.ZipFile(str(stem)+'source.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, path.name)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    start = time.time()
    with open(str(stem)+'stdout.txt', 'w') as stdout, open(str(stem)+'stderr.txt', 'w') as stderr:
        process = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
    receipt = dict(command=command, returncode=process.returncode, natural_exit=process.returncode >= 0,
        termination_signal=-process.returncode if process.returncode < 0 else None, elapsed_seconds=time.time()-start,
        input_sha256=sha(input_path), source_sha256={path.name:sha(path) for path in sources}, source_zip_sha256=sha(Path(str(stem)+'source.zip')))
    Path(str(stem)+'exit.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)
    sys.exit(process.returncode if process.returncode >= 0 else 128-process.returncode)


if __name__ == '__main__':
    main()
