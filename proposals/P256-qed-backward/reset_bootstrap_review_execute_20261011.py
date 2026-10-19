"""Create one immutable independent-review epoch with real child natural exit."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import time
import zipfile

ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'
def digest(raw):return hashlib.sha256(raw).hexdigest()
def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def bundle(path,names):
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in names:z.writestr(name,(ROOT/name).read_bytes())
def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--epoch',required=True);a=p.parse_args()
    req=json.loads((ROOT/a.request).read_text());prefix=a.epoch
    before={n:digest((ROOT/n).read_bytes()) for n in req['pins']}
    if before!=req['pins']:raise RuntimeError('preexecution pin mismatch')
    pre=ROOT/(prefix+'_preexecution.zip');bundle(pre,list(before)+[a.request])
    output=prefix+'_capsule.json';cmd=[PYTHON,str(ROOT/req['consumer']),'--request',str(ROOT/a.request),'--output',str(ROOT/output)]
    started=time.monotonic();proc=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    out,err=proc.communicate();elapsed=time.monotonic()-started
    for suffix,raw in [('stdout.txt',out),('stderr.txt',err)]:
        with (ROOT/(prefix+'_'+suffix)).open('xb') as f:f.write(raw)
    exit_name=prefix+'_exit.json';save(ROOT/exit_name,{'argv':cmd,'returncode':proc.returncode,'natural_exit':True,'signal':None if proc.returncode>=0 else -proc.returncode,'elapsed_seconds':elapsed})
    after={n:digest((ROOT/n).read_bytes()) for n in before}
    receipt_name=prefix+'_receipt.json'
    save(ROOT/receipt_name,{'before':before,'after':after,'unchanged':before==after,'request_sha256':digest((ROOT/a.request).read_bytes()),'preexecution_zip_sha256':digest(pre.read_bytes()),'returncode':proc.returncode,'natural_exit':True,'capsule_present':(ROOT/output).exists()})
    members=list(before)+[a.request,exit_name,receipt_name,prefix+'_stdout.txt',prefix+'_stderr.txt']
    if (ROOT/output).exists():members.append(output)
    post=ROOT/(prefix+'_executed.zip');bundle(post,members)
    print(json.dumps({'returncode':proc.returncode,'natural_exit':True,'unchanged':before==after,'preexecution_sha256':digest(pre.read_bytes()),'executed_sha256':digest(post.read_bytes())}))
    if proc.returncode:raise SystemExit(proc.returncode)
    if before!=after:raise RuntimeError('frozen source mutation')
if __name__=='__main__':main()
