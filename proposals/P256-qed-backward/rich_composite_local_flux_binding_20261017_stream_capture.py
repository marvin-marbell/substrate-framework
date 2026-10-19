"""Natural-exit capture for exact streamed representation and final diagnostics."""
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
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    stamp = sys.argv[1]; mode = sys.argv[2] if len(sys.argv) > 2 else 'streamed'
    stem = ROOT/(PREFIX+stamp+'_')
    if Path(str(stem)+'source.zip').exists() or Path(str(stem)+'exit.json').exists():
        raise FileExistsError('Captured epoch is immutable')
    input_path = ROOT/(PREFIX+stamp+'_input.json')
    inp = json.loads(input_path.read_text())
    if mode == 'streamed':
        parent = json.loads((ROOT/(PREFIX+inp['same_physics_as']+'_input.json')).read_text())
        for key in ['parameters', 'word_depth', 'wilson_loop_seed', 'metric_relative_tolerance', 'times']:
            if parent[key] != inp[key]:
                raise ValueError('Representation-only epoch changed declared physics/source/relative content')
        command = [PYTHON, str(ROOT/(PREFIX+'streamed.py')), str(input_path)]
    elif mode == 'language':
        command = [PYTHON, str(ROOT/(PREFIX+'language.py'))]
    else:
        raise ValueError(mode)
    sources = sorted(ROOT.glob(PREFIX+'*.py'))+[ROOT/(PREFIX+'equations.json'), input_path]
    with zipfile.ZipFile(str(stem)+'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, path.name)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    start = time.time()
    with open(str(stem)+'stdout.txt', 'w') as stdout, open(str(stem)+'stderr.txt', 'w') as stderr:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
    receipt = dict(command=command, returncode=result.returncode, natural_exit=result.returncode >= 0,
        termination_signal=-result.returncode if result.returncode < 0 else None, elapsed_seconds=time.time()-start,
        input_sha256=sha(input_path), source_sha256={path.name:sha(path) for path in sources}, source_zip_sha256=sha(Path(str(stem)+'source.zip')))
    Path(str(stem)+'exit.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)
    sys.exit(result.returncode if result.returncode >= 0 else 128-result.returncode)


if __name__ == '__main__':
    main()
