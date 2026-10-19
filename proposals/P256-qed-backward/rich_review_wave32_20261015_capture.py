"""Bind every selected source/input before new runtime; retain natural child exit."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent
P = 'rich_review_wave32_20261015_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def digest(path: Path) -> dict:
    return dict(file=path.name,bytes=path.stat().st_size,SHA256=hashlib.sha256(path.read_bytes()).hexdigest())


def main() -> None:
    config = json.loads((ROOT/(P+'input.json')).read_text())
    selected = [ROOT/name for name in config['bound_inputs']]
    selected += [ROOT/(P+suffix) for suffix in ['consume.py','observe.py','capture.py','input.json','equations.json','raw_review.json']]
    selected = list(dict.fromkeys(selected))
    before = {path.name:digest(path) for path in selected}
    capsule = ROOT/(P+'source_capsule.zip')
    with zipfile.ZipFile(capsule,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in selected:
            archive.write(path,path.name)
    command = [PYTHON,P+'consume.py']
    environment = dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    receipt = dict(command=command,cwd=str(ROOT),input_source=before,source_capsule=digest(capsule),
        start_unix=time.time(),environment_subset={key:environment[key] for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']})
    (ROOT/(P+'start.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    start = time.monotonic()
    with (ROOT/(P+'stdout.txt')).open('w') as stdout,(ROOT/(P+'stderr.txt')).open('w') as stderr:
        completed = subprocess.run(command,cwd=ROOT,env=environment,stdout=stdout,stderr=stderr)
    receipt.update(natural_exit=True,returncode=completed.returncode,elapsed_seconds=time.monotonic()-start,
        sources_and_inputs_unchanged=all(digest(path)==before[path.name] for path in selected))
    receipt['outputs'] = [digest(ROOT/(P+suffix)) for suffix in ['result.json','states.npz','partial_result.json','stdout.txt','stderr.txt'] if (ROOT/(P+suffix)).is_file()]
    (ROOT/(P+'exit.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(returncode=completed.returncode,natural_exit=True,elapsed_seconds=receipt['elapsed_seconds'],sources_and_inputs_unchanged=receipt['sources_and_inputs_unchanged'])),flush=True)
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
