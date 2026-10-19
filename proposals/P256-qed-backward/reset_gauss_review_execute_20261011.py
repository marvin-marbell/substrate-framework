"""Create immutable review capsules around a new native consumer natural exit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'

def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def put(path: Path, value) -> None:
    with path.open('x') as f:
        json.dump(value,f,indent=2); f.write('\n')

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',type=Path,required=True)
    p.add_argument('--prefix',required=True)
    a=p.parse_args()
    if not a.prefix.startswith('reset_gauss_review_'): raise ValueError('owned prefix required')
    request_raw=a.request.read_bytes(); request=json.loads(request_raw)
    source={name:(ROOT/name).read_bytes() for name in request['pins']}
    before={n:sha(raw) for n,raw in source.items()}
    if before!=request['pins']: raise ValueError('preexecution source pins changed')
    prefix=ROOT/a.prefix
    out=prefix.with_name(prefix.name+'_capsule.json')
    stdout=prefix.with_name(prefix.name+'_stdout.txt')
    stderr=prefix.with_name(prefix.name+'_stderr.txt')
    exitpath=prefix.with_name(prefix.name+'_exit.json')
    receipt=prefix.with_name(prefix.name+'_receipt.json')
    pre=prefix.with_name(prefix.name+'_preexecution.zip')
    executed=prefix.with_name(prefix.name+'_executed.zip')
    command=[PYTHON,str(ROOT/request['consumer']),'--request',str(a.request),'--output',str(out)]
    with zipfile.ZipFile(pre,'x',zipfile.ZIP_DEFLATED) as z:
        for n,raw in sorted(source.items()): z.writestr(n,raw)
        z.writestr(a.request.name,request_raw)
    started=time.monotonic()
    with stdout.open('xb') as sout,stderr.open('xb') as serr:
        child=subprocess.Popen(command,cwd=ROOT,stdout=sout,stderr=serr)
        code=child.wait()
    after={n:sha((ROOT/n).read_bytes()) for n in source}
    exitinfo=dict(command=command,returncode=code,natural_exit=True,timeout_used=False,
                  terminated=False,elapsed_seconds=time.monotonic()-started,
                  source_input_sha256_before=before,source_input_sha256_after=after,
                  source_inputs_unchanged=before==after,request_unchanged=a.request.read_bytes()==request_raw)
    put(exitpath,exitinfo)
    receiptinfo=dict(kind='INDEPENDENT_NEW_NATIVE_REVIEW_NATURAL_EXIT_RECEIPT',
                     request=dict(path=a.request.name,sha256=sha(request_raw)),
                     preexecution=dict(path=pre.name,sha256=sha(pre.read_bytes())),
                     exit=dict(path=exitpath.name,sha256=sha(exitpath.read_bytes())),
                     stdout=dict(path=stdout.name,sha256=sha(stdout.read_bytes())),
                     stderr=dict(path=stderr.name,sha256=sha(stderr.read_bytes())),
                     capsule=dict(path=out.name,sha256=sha(out.read_bytes())) if out.exists() else None,
                     frozen_target_pins=request['frozen_target_pins'],
                     source_input_sha256_before=before,source_input_sha256_after=after,
                     returncode=code,natural_exit=True,source_inputs_unchanged=before==after)
    put(receipt,receiptinfo)
    with zipfile.ZipFile(executed,'x',zipfile.ZIP_DEFLATED) as z:
        for n,raw in sorted(source.items()): z.writestr(n,raw)
        z.writestr(a.request.name,request_raw)
        for path in [stdout,stderr,exitpath,receipt,*([out] if out.exists() else [])]:
            z.writestr(path.name,path.read_bytes())
    print(json.dumps(dict(returncode=code,natural_exit=True,unchanged=before==after,
                          preexecution_sha256=sha(pre.read_bytes()),executed_sha256=sha(executed.read_bytes()),
                          receipt_sha256=sha(receipt.read_bytes()))))
    if code!=0: raise SystemExit(code)
    if before!=after: raise RuntimeError('source mutation')

if __name__=='__main__':
    main()
