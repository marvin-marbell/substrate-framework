"""Exclusive frozen review-control execution, never target producer execution."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = 'magnetic_history_review_actual1_20261011_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, data: object) -> None:
    with path.open('x') as stream:
        json.dump(data,stream,indent=2)
        stream.write('\n')


def freeze(path: Path, files: list[Path]) -> None:
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for source in files:
            archive.write(source,source.name)
    os.chmod(path,0o444)


def main() -> None:
    source = ROOT/'magnetic_history_review_control_20261011.py'
    request = ROOT/'magnetic_history_review_input_20261011.json'
    data = json.loads(request.read_text())
    files = [source,request,Path(__file__).resolve(),ROOT/'magnetic_history_review_language_20261011.json',Path(data['contract_path'])]
    files += [ROOT/name for name in data['pins']]
    files.append(ROOT/data['snapshot']['path'])
    before = {p.name:sha(p) for p in files}
    pre = ROOT/(PREFIX+'preexecution.zip')
    freeze(pre,files)
    receipt = ROOT/(PREFIX+'receipt.json')
    stdout = ROOT/(PREFIX+'stdout.txt')
    stderr = ROOT/(PREFIX+'stderr.txt')
    command = [PYTHON,str(source),'--input',str(request),'--output',str(receipt)]
    start = time.monotonic()
    with stdout.open('xb') as out, stderr.open('xb') as err:
        child = subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err)
        code = child.wait()
    elapsed = time.monotonic()-start
    after = {p.name:sha(p) for p in files}
    result = {'command':command,'cwd':str(ROOT),'pid':child.pid,'returncode':code,
              'natural_exit':code>=0,'elapsed_seconds':elapsed,'before_sha256':before,
              'after_sha256':after,'sources_inputs_unchanged':before==after,
              'stdout_sha256':sha(stdout),'stderr_sha256':sha(stderr),
              'receipt_sha256':sha(receipt) if receipt.exists() else None}
    exitfile = ROOT/(PREFIX+'exit.json')
    save(exitfile,result)
    executed = ROOT/(PREFIX+'executed.zip')
    freeze(executed,files+[stdout,stderr,exitfile]+([receipt] if receipt.exists() else []))
    capsule = {'kind':'IMMUTABLE_INDEPENDENT_HISTORY_CONTROL_CAPSULE_NOT_SCIENTIFIC_ADMISSION',
               'preexecution':{'path':pre.name,'sha256':sha(pre)},
               'executed':{'path':executed.name,'sha256':sha(executed)},
               'exit':{'path':exitfile.name,'sha256':sha(exitfile)},'result':result}
    save(ROOT/(PREFIX+'capsule.json'),capsule)
    print(json.dumps({'returncode':code,'natural_exit':code>=0,'unchanged':before==after,'elapsed_seconds':elapsed}))
    if code != 0 or before != after:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
