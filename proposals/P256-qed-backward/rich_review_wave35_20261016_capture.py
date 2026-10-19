"""Preserve exact source/input and natural science process exits, including failures."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parent
P='rich_review_wave35_20261016_'
PY='/home/administrator/substrate-framework/.venv/bin/python'


def main():
    label=sys.argv[1];script=ROOT/sys.argv[2]
    base=ROOT/(P+label)
    paths=[script,Path(__file__).resolve()]+[ROOT/p for p in sys.argv[3:]]
    sha={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    command=[PY,str(script)]
    environment=dict(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    record=dict(command=command,cwd=str(ROOT),source_input_sha256=sha,environment=dict(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),started=time.time())
    Path(str(base)+'_input.json').write_text(json.dumps(record,indent=2)+'\n')
    with zipfile.ZipFile(str(base)+'_source_capsule.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,p.name)
        z.write(Path(str(base)+'_input.json'),Path(str(base)+'_input.json').name)
    env=os.environ.copy();env.update(environment)
    with open(str(base)+'_stdout.txt','wb') as out,open(str(base)+'_stderr.txt','wb') as err:
        child=subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=err)
    record.update(natural_exit=child.returncode,finished=time.time(),input_sources_unchanged=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in sha.items()))
    Path(str(base)+'_exit.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    return child.returncode

if __name__=='__main__':raise SystemExit(main())
