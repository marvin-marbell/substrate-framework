"""Launch only the NEW independent CLI and retain immutable natural-exit epochs."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'

def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def exclusive(name: str, data: bytes) -> None:
    with (ROOT/name).open('xb') as f:f.write(data)

def main() -> None:
    request_name=sys.argv[1]
    request=json.loads((ROOT/request_name).read_text())
    epoch=request['epoch']
    paths={name:ROOT/name for name in request['pins']}
    paths[request_name]=ROOT/request_name
    paths[Path(__file__).name]=Path(__file__)
    paths['weighted_unitary_review_contract_20261011.json']=Path(request['contract'])
    originals={name:p.read_bytes() for name,p in paths.items()}
    before={name:digest(raw) for name,raw in originals.items()}
    assert all(before[name]==sha for name,sha in request['pins'].items())
    assert before['weighted_unitary_review_contract_20261011.json']==request['contract_sha256']
    def archive(name: str, members: dict) -> dict:
        with zipfile.ZipFile(ROOT/name,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for member,raw in members.items():z.writestr(member,raw)
        return {'path':name,'sha256':digest((ROOT/name).read_bytes()),'members':{k:digest(v) for k,v in members.items()}}
    pre=archive(epoch+'_preexecution.zip',originals)
    cmd=[PYTHON,'-B',str(ROOT/request['consumer']),'--request',request_name,'--output',str(ROOT/(epoch+'_receipt.json'))]
    started=time.monotonic()
    run=subprocess.run(cmd,cwd=ROOT,capture_output=True,check=False)
    elapsed=time.monotonic()-started
    exclusive(epoch+'_stdout.txt',run.stdout)
    exclusive(epoch+'_stderr.txt',run.stderr)
    after={name:digest(p.read_bytes()) for name,p in paths.items()}
    exit_info={'command':cmd,'returncode':run.returncode,'natural_exit':True,'elapsed_seconds':elapsed,
               'source_input_sha256_before':before,'source_input_sha256_after':after,'source_input_unchanged':before==after}
    exclusive(epoch+'_exit.json',(json.dumps(exit_info,indent=2)+'\n').encode())
    executed=dict(originals)
    for suffix in ('_stdout.txt','_stderr.txt','_exit.json','_receipt.json'):
        p=ROOT/(epoch+suffix)
        if p.exists():executed[p.name]=p.read_bytes()
    post=archive(epoch+'_executed.zip',executed)
    capsule={'preexecution':pre,'executed':post,'exit':exit_info,'scope':'NEW independent native coefficient controls plus synthetic limit mechanism; not native fullH run','old_producers_run':False}
    exclusive(epoch+'_capsule.json',(json.dumps(capsule,indent=2)+'\n').encode())
    print(json.dumps({'capsule':epoch+'_capsule.json','returncode':run.returncode,'natural_exit':True,'unchanged':before==after,'elapsed_seconds':elapsed}))
    if run.returncode!=0:sys.exit(run.returncode)
    if before!=after:sys.exit(2)

if __name__=='__main__':
    main()
