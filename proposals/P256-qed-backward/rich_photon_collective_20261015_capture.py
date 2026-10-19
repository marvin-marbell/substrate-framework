"""Natural finite scientific capture with immutable source/input hash receipts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent
PREFIX='rich_photon_collective_20261015_'
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return dict(sha256=h.hexdigest(),bytes=path.stat().st_size)

def run():
    mode=sys.argv[1];config=ROOT/(PREFIX+mode+'_input.json');request=json.loads(config.read_text())
    script=ROOT/request['script'];sources=[ROOT/name for name in request['sources']];inputs=[config]+[ROOT/name for name in request.get('inputs',[])]
    before={p.name:digest(p) for p in [*sources,*inputs]};start=time.time()
    command=[PYTHON,str(script),*[str(x) for x in request.get('arguments',[])]]
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (ROOT/(PREFIX+mode+'_stdout.txt')).open('w') as stdout,(ROOT/(PREFIX+mode+'_stderr.txt')).open('w') as stderr:
        child=subprocess.run(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
    after={p.name:digest(p) for p in [*sources,*inputs]}
    receipt=dict(command=command,returncode=child.returncode,process_return_observed=True,natural_exit=child.returncode>=0,scientific_completed=child.returncode==0,interrupted=child.returncode<0,wall_seconds=time.time()-start,wall_semantics='start through child return and after-input hashes; excludes subsequent ZIP packaging/manifest publication',source_input_before=before,source_input_after=after,changed=[name for name in before if before[name]!=after[name]],resource_semantics='scientific child reports its own ru_maxrss; interrupted/failed child resource report may be UNKNOWN,never inferred from archive size')
    exitpath=ROOT/(PREFIX+mode+'_exit.json');exitpath.write_text(json.dumps(receipt,indent=2)+'\n')
    outputs=[p for p in ROOT.glob(PREFIX+mode+'_*') if p.suffix not in ('.zip',) and p.name not in (PREFIX+mode+'_manifest.json',)]
    if mode=='probe':outputs.append(ROOT/(PREFIX+'probe_data.json'))
    selected=sorted(set([*sources,*inputs,*outputs,exitpath]))
    capsule=ROOT/(PREFIX+mode+'_capsule.zip')
    with zipfile.ZipFile(capsule,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        for p in selected:archive.write(p,p.name)
    manifest=dict(files={p.name:digest(p) for p in selected},capsule={capsule.name:digest(capsule)},receipt=receipt)
    (ROOT/(PREFIX+mode+'_manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(mode=mode,returncode=child.returncode,wall_seconds=receipt['wall_seconds'],changed=receipt['changed'],capsule=manifest['capsule'])),flush=True)
    sys.exit(child.returncode)

if __name__=='__main__':run()
