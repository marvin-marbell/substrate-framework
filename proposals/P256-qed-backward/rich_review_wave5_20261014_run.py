"""Exact revision capture for NEW reviewer supplemental consumers only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent
PREFIX='rich_review_wave5_20261014'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    label=sys.argv[1]
    script=PREFIX+'_'+label+'.py'
    request=json.loads((ROOT/(PREFIX+'_'+label+'_manifest_input.json')).read_text())
    names=[script,PREFIX+'_run.py',PREFIX+'_'+label+'_manifest_input.json']+request['inputs']
    before={name:(ROOT/name).read_bytes() for name in names}
    command=[sys.executable,str(ROOT/script)]
    start=time.time()
    run=subprocess.run(command,cwd=ROOT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),capture_output=True)
    prefix=PREFIX+'_'+label
    (ROOT/(prefix+'_stdout.txt')).write_bytes(run.stdout)
    (ROOT/(prefix+'_stderr.txt')).write_bytes(run.stderr)
    after={name:digest((ROOT/name).read_bytes()) for name in names}
    outputs=request['outputs']+[prefix+'_stdout.txt',prefix+'_stderr.txt']
    receipt={'command':command,'python':sys.version,'cwd':str(ROOT),'natural_exit':run.returncode>=0,'returncode':run.returncode,'elapsed_seconds':time.time()-start,'sources_inputs_before':{name:digest(data) for name,data in before.items()},'sources_inputs_after':after,'all_inputs_unchanged':all(digest(before[name])==after[name] for name in names),'outputs':{name:digest((ROOT/name).read_bytes()) for name in outputs if (ROOT/name).exists()}}
    (ROOT/(prefix+'_exit.json')).write_text(json.dumps(receipt,indent=2))
    with zipfile.ZipFile(ROOT/(prefix+'_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
        for name,data in before.items():
            archive.writestr('inputs_before/'+name,data)
        for name in outputs+[prefix+'_exit.json']:
            if (ROOT/name).exists():
                archive.write(ROOT/name,name)
    print(json.dumps(receipt,indent=2))
    raise SystemExit(run.returncode)


if __name__=='__main__':
    main()
