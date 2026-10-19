"""Natural subprocess execution with genuine preexecution and executed byte archives."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args()
    request_path=args.request.resolve()
    request=json.loads(request_path.read_bytes())
    stem=request['output_stem']
    folder=request_path.parent
    names={key:folder/(stem+'_'+key+suffix) for key,suffix in
           [('preexecution','.zip'),('executed','.zip'),('receipt','.json'),('stdout','.txt'),('stderr','.txt'),('capsule','.json')]}
    assert all(not p.exists() for p in names.values())
    before={path:sha(Path(path).read_bytes()) for path in request['pins']}
    assert before==request['pins']
    inputs={Path(path).name:Path(path).read_bytes() for path in request['pins']}
    assert len(inputs)==len(request['pins'])
    inputs[request_path.name]=request_path.read_bytes()
    def archive(path:Path,members:dict[str,bytes]):
        with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as z:
            for name,raw in sorted(members.items()):z.writestr(name,raw)
        path.chmod(0o444)
    archive(names['preexecution'],inputs)
    command=[request['python'],request['consumer'],'--request',str(request_path),'--output',str(names['receipt'])]
    start=time.monotonic()
    completed=subprocess.run(command,cwd=folder,capture_output=True)
    elapsed=time.monotonic()-start
    names['stdout'].write_bytes(completed.stdout);names['stderr'].write_bytes(completed.stderr)
    after={path:sha(Path(path).read_bytes()) for path in request['pins']}
    products={p.name:p.read_bytes() for key,p in names.items() if key in ('receipt','stdout','stderr') and p.exists()}
    archive(names['executed'],inputs|products)
    capsule={'schema':1,'command':command,'returncode':completed.returncode,
             'natural_completed_exit':True,'timeout_or_kill_used':False,'elapsed_seconds':elapsed,
             'source_input_sha256_before':before,'source_input_sha256_after':after,
             'source_input_unchanged':before==after,'request_sha256':sha(request_path.read_bytes()),
             'artifacts':{key:{'path':str(p),'sha256':sha(p.read_bytes())} for key,p in names.items() if key!='capsule' and p.exists()}}
    names['capsule'].write_text(json.dumps(capsule,indent=2)+'\n')
    for key,p in names.items():
        if p.exists():p.chmod(0o444)
    print(json.dumps({'returncode':completed.returncode,'natural_completed_exit':True,
                      'capsule':str(names['capsule']),'elapsed_seconds':elapsed}))
    if completed.returncode!=0 or before!=after:
        raise SystemExit(completed.returncode or 1)


if __name__=='__main__':
    main()
