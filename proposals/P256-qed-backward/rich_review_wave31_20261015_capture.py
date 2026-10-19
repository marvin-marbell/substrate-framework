"""Own input/source binding and natural-exit capture for the Wave31 experiment."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_review_wave31_20261015_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def digest(path: Path) -> dict:
    return dict(file=path.name, bytes=path.stat().st_size,
                SHA256=hashlib.sha256(path.read_bytes()).hexdigest())


def main() -> None:
    config = json.loads((ROOT/(PREFIX+'input.json')).read_text())
    selected = [ROOT/name for name in config['bound_inputs']]
    selected += [ROOT/(PREFIX+suffix) for suffix in ['consume.py', 'capture.py', 'krylov.py', 'observe.py', 'input.json', 'equations.json']]
    selected = list(dict.fromkeys(selected))
    before = {path.name:digest(path) for path in selected}
    capsule = ROOT/(PREFIX+'source_capsule.zip')
    with zipfile.ZipFile(capsule, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in selected:
            archive.write(path, path.name)
    command = [PYTHON, PREFIX+'consume.py']
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    receipt = dict(command=command, cwd=str(ROOT), input_source=before,
                   source_capsule=digest(capsule), start_unix=time.time(),
                   environment_subset={key:environment[key] for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']})
    (ROOT/(PREFIX+'start.json')).write_text(json.dumps(receipt, indent=2)+'\n')
    start = time.monotonic()
    with (ROOT/(PREFIX+'stdout.txt')).open('w') as stdout, (ROOT/(PREFIX+'stderr.txt')).open('w') as stderr:
        completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
    receipt.update(natural_exit=True, returncode=completed.returncode,
                   elapsed_seconds=time.monotonic()-start,
                   sources_and_inputs_unchanged=all(digest(path)==before[path.name] for path in selected))
    outputs = [ROOT/(PREFIX+suffix) for suffix in ['result.json','states.npz','stdout.txt','stderr.txt']]
    receipt['outputs'] = [digest(path) for path in outputs if path.is_file()]
    (ROOT/(PREFIX+'exit.json')).write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(returncode=completed.returncode, natural_exit=True,
                         elapsed_seconds=receipt['elapsed_seconds'],
                         sources_and_inputs_unchanged=receipt['sources_and_inputs_unchanged'])), flush=True)
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
