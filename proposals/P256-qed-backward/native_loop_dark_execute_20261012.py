"""Execute only own pinned native consumer; preserve every natural-exit epoch."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2);stream.write('\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args();request_raw=args.request.read_bytes();request=json.loads(request_raw)
    prefix=request['epoch'];captured={name:(ROOT/name).read_bytes() for name in request['pins']}
    assert all(sha(raw)==request['pins'][name] for name,raw in captured.items())
    mapping={name:'inputs/'+Path(name).name for name in captured}
    assert len(set(mapping.values()))==len(mapping)
    pre=ROOT/(prefix+'_preexecution.zip');before=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with zipfile.ZipFile(pre,'x',zipfile.ZIP_DEFLATED) as archive:
        for name,raw in sorted(captured.items()):archive.writestr(mapping[name],raw)
        archive.writestr('request/'+args.request.name,request_raw)
    receipt=ROOT/(prefix+'_receipt.json')
    code=('import sys,zipfile; z=zipfile.ZipFile(sys.argv[1]); raw=z.read(sys.argv[2]); source=sys.argv[3]; '
          'sys.argv=[source,"--request",sys.argv[4],"--output",sys.argv[5]]; '
          'exec(compile(raw,source,"exec"),{"__name__":"__main__","__file__":source})')
    argv=[request['python'],'-B','-c',code,str(pre),mapping[request['consumer']],str(ROOT/request['consumer']),str(args.request.resolve()),str(receipt)]
    started=time.monotonic();child=subprocess.run(argv,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
    elapsed=time.monotonic()-started;after=datetime.datetime.now(datetime.timezone.utc).isoformat()
    for suffix,raw in (('stdout.txt',child.stdout),('stderr.txt',child.stderr)):
        with (ROOT/(prefix+'_'+suffix)).open('xb') as stream:stream.write(raw)
    stable=all((ROOT/name).read_bytes()==raw for name,raw in captured.items())
    exit_path=ROOT/(prefix+'_exit.json')
    save(exit_path,{'exit_code':child.returncode,'natural_exit':True,'elapsed_seconds':elapsed,'started_utc':before,'finished_utc':after,'inputs_unchanged':stable})
    executed=ROOT/(prefix+'_executed.zip')
    outputs=[exit_path,ROOT/(prefix+'_stdout.txt'),ROOT/(prefix+'_stderr.txt')]+([receipt] if receipt.exists() else [])
    with zipfile.ZipFile(executed,'x',zipfile.ZIP_DEFLATED) as archive:
        with zipfile.ZipFile(pre) as source:
            for name in source.namelist():archive.writestr(name,source.read(name))
        for path in outputs:archive.writestr('outputs/'+path.name,path.read_bytes())
    capsule={'kind':'OWN_NATIVE_FULL64_CONSUMER_NATURAL_EXIT_NOT_GROUND','preexecution':{'path':pre.name,'sha256':sha(pre.read_bytes()),'created_before_child':True},
             'executed':{'path':executed.name,'sha256':sha(executed.read_bytes())},'source_selection':'compile own consumer bytes from preexecution ZIP',
             'request':{'path':args.request.name,'sha256':sha(request_raw)},'pins':request['pins'],'argv':argv,
             'exit_code':child.returncode,'natural_exit':True,'elapsed_seconds':elapsed,'inputs_unchanged':stable,'governor':request['governor']}
    if receipt.exists():capsule['receipt']={'path':receipt.name,'sha256':sha(receipt.read_bytes())}
    save(ROOT/(prefix+'_capsule.json'),capsule)
    print(json.dumps({'capsule':prefix+'_capsule.json','exit_code':child.returncode,'elapsed_seconds':elapsed,'inputs_unchanged':stable}))
    if child.returncode or not stable:raise SystemExit(child.returncode or 1)


if __name__=='__main__':main()
