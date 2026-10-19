"""Exact immutable review capture, no producer CLI replay."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from typing import Any

ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'
PREFIX='rich_review_wave27_20261015_'


def pin(path: Path) -> dict[str, Any]:
    return dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)


def main() -> None:
    input_path=Path(sys.argv[1]);request=json.loads(input_path.read_text())
    run=ROOT/request.get('run_prefix',PREFIX+'run')
    result_path=ROOT/request.get('result_file',PREFIX+'result.json')
    for record in request['frozen_files']:
        if pin(ROOT/record['file'])['sha256']!=record['sha256']:
            raise ValueError('Frozen bytes changed '+record['file'])
    pending=[ROOT/(PREFIX+'experiment.py'),Path(__file__)]
    sources: dict[str, Path]={}
    while pending:
        path=pending.pop()
        if path.name in sources:
            continue
        sources[path.name]=path
        for node in ast.walk(ast.parse(path.read_text())):
            module=node.module if isinstance(node,ast.ImportFrom) else None
            modules=[module] if module else [x.name for x in node.names] if isinstance(node,ast.Import) else []
            for name in modules:
                child=ROOT/(name.split('.')[0]+'.py')
                if child.is_file():
                    pending.append(child)
    manifest=json.loads((ROOT/request['numeric_cache']['manifest']['file']).read_text())
    data=[ROOT/manifest['snapshots'][label]['file'] for label in request['consumed_snapshot_labels']]
    data += [ROOT/request['numeric_cache'][label]['file'] for label in ('manifest','states','loader')]
    data += [ROOT/request['physical_input'],ROOT/request['model_capsule']['file'],ROOT/request['off_factorization']['file']]
    parameters=json.loads((ROOT/request['physical_input']).read_text())
    data += [ROOT/parameters['native_body']]
    data += [ROOT/record['file'] for record in request['frozen_files']]
    files={p.name:p for p in [*sources.values(),*data,input_path]}
    bindings=[pin(p) for p in files.values()]
    capsule=run.with_name(run.name+'_inputs.zip')
    with zipfile.ZipFile(capsule,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,path in files.items():
            archive.write(path,name)
    command: list[str]=[PYTHON,PREFIX+'experiment.py',input_path.name,result_path.name]
    start=dict(kind='EXACT_OWN_WAVE27_CAPTURE',command=command,
        input=pin(input_path),capsule=pin(capsule),bindings=bindings,wall_start=time.time(),deadline_seconds=request['deadline_seconds'])
    run.with_name(run.name+'_start.json').write_text(json.dumps(start,indent=2)+'\n')
    before=time.monotonic()
    with run.with_name(run.name+'_stdout.txt').open('xb') as out,run.with_name(run.name+'_stderr.txt').open('xb') as err:
        process=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err)
        natural=True
        try:
            code=process.wait(timeout=request['deadline_seconds'])
        except subprocess.TimeoutExpired:
            natural=False
            process.terminate()
            try:
                code=process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill();code=process.wait()
    exit_record=dict(kind='OBSERVED_EXACT_OWN_RUN_EXIT',natural_exit=natural,returncode=code,seconds=time.monotonic()-before,
        exit_meaning='natural process return' if natural else 'deadline interrupted; not full completion',
        source_unchanged=all(pin(p)['sha256']==record['sha256'] for p,record in zip(files.values(),bindings)),
        input_capsule=pin(capsule),start=pin(run.with_name(run.name+'_start.json')))
    run.with_name(run.name+'_exit.json').write_text(json.dumps(exit_record,indent=2)+'\n')
    outputs=[result_path,run.with_name(run.name+'_stdout.txt'),run.with_name(run.name+'_stderr.txt'),run.with_name(run.name+'_exit.json')]
    with zipfile.ZipFile(run.with_name(run.name+'_outputs.zip'),'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in outputs:
            if path.exists():
                archive.write(path,path.name)
    print(json.dumps(exit_record),flush=True)


if __name__=='__main__':
    main()
