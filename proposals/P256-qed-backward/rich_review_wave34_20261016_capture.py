"""Capture own finite process with input/source binding and natural subprocess exit."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent
P = 'rich_review_wave34_20261016_'
PY = '/home/administrator/substrate-framework/.venv/bin/python'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config = json.loads((ROOT/(P+'input.json')).read_text())
    names = list(config['expected_input_SHA256'])+[P+'input.json', P+'equations.json', P+'consume.py', P+'capture.py']
    before = {name: sha(ROOT/name) for name in names}
    capsule = ROOT/(P+'source_capsule.zip')
    with zipfile.ZipFile(capsule, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(ROOT/name, name)
    command = [PY, P+'consume.py']
    environment = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    start = time.time()
    receipt = dict(command=command, cwd=str(ROOT), source_SHA256=before, source_capsule=dict(file=capsule.name, sha256=sha(capsule)),
                   start_unix=start, environment={k: environment[k] for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']})
    (ROOT/(P+'start.json')).write_text(json.dumps(receipt, indent=2)+'\n')
    with (ROOT/(P+'stdout.txt')).open('w') as stdout, (ROOT/(P+'stderr.txt')).open('w') as stderr:
        done = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
    receipt.update(natural_exit=True, returncode=done.returncode, elapsed_seconds=time.time()-start,
                   sources_and_inputs_unchanged=all(sha(ROOT/name)==value for name,value in before.items()))
    receipt['outputs'] = {name: sha(ROOT/name) for name in [P+'result.json', P+'partial_result.json', P+'stdout.txt', P+'stderr.txt'] if (ROOT/name).exists()}
    (ROOT/(P+'exit.json')).write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(natural_exit=True, returncode=done.returncode, elapsed_seconds=receipt['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
