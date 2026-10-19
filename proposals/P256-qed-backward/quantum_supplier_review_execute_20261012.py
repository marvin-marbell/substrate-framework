"""One immutable NEW consumer run, with natural exit and unchanged inputs."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

ROOT=Path(__file__).resolve().parent
PREFIX=ROOT/'quantum_supplier_review_actual2_20261012'
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    request=PREFIX.with_name(PREFIX.name+'_request.json')
    data=json.loads(request.read_text())
    inputs=[Path(pin['path']) for pin in data['pins'].values()]+[request,Path(__file__).resolve()]
    inputs=list(dict.fromkeys(inputs))
    pre=PREFIX.with_name(PREFIX.name+'_preexecution.zip')
    before={str(path):digest(path) for path in inputs}
    with zipfile.ZipFile(pre,'x',zipfile.ZIP_DEFLATED) as z:
        for path in inputs:
            z.write(path,path.name)
    output=PREFIX.with_name(PREFIX.name+'_receipt.json')
    command=[PYTHON,str(ROOT/'quantum_supplier_review_consumer_20261012.py'),'--request',str(request),'--output',str(output)]
    start=dt.datetime.now(dt.timezone.utc).isoformat();clock=time.monotonic()
    result=subprocess.run(command,cwd=ROOT,capture_output=True,check=False)
    end=dt.datetime.now(dt.timezone.utc).isoformat();elapsed=time.monotonic()-clock
    stdout=PREFIX.with_name(PREFIX.name+'_stdout.txt');stderr=PREFIX.with_name(PREFIX.name+'_stderr.txt')
    stdout.write_bytes(result.stdout);stderr.write_bytes(result.stderr)
    unchanged=all(digest(Path(path))==sha for path,sha in before.items())
    exitpath=PREFIX.with_name(PREFIX.name+'_natural_exit.json')
    record={'command':command,'cwd':str(ROOT),'start_utc':start,'end_utc':end,'elapsed_seconds':elapsed,'returncode':result.returncode,'natural_exit':True,'timeout':False,'inputs_unchanged':unchanged,'source_input_sha256':before,'preexecution_sha256':digest(pre),'stdout_sha256':digest(stdout),'stderr_sha256':digest(stderr)}
    exitpath.write_text(json.dumps(record,indent=2)+'\n')
    executed=PREFIX.with_name(PREFIX.name+'_executed.zip')
    outputs=[exitpath,stdout,stderr]+([output] if output.exists() else [])
    with zipfile.ZipFile(executed,'x',zipfile.ZIP_DEFLATED) as z:
        for path in inputs+outputs:
            z.write(path,path.name)
        z.write(pre,pre.name)
    delivery={'epoch':PREFIX.name,'preexecution':{'path':str(pre),'sha256':digest(pre)},'executed':{'path':str(executed),'sha256':digest(executed)},'runtime':record,'receipt':{'path':str(output),'sha256':digest(output)} if output.exists() else None}
    PREFIX.with_name(PREFIX.name+'_delivery.json').write_text(json.dumps(delivery,indent=2)+'\n')
    print(json.dumps({'returncode':result.returncode,'natural_exit':True,'inputs_unchanged':unchanged,'elapsed_seconds':elapsed}),flush=True)

if __name__=='__main__':
    main()
