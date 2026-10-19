"""Capture exact source bytes, natural child exits, stdout/stderr and inputs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parent
P='rich_gauss_projected_photon_20261016_'
PY='/home/administrator/substrate-framework/.venv/bin/python'

def main():
    label=sys.argv[1] if len(sys.argv)>1 else 'attempt1'
    target=sys.argv[2] if len(sys.argv)>2 else 'run'
    sources=[ROOT/(P+x) for x in ('state.py','run.py','exact.py','capture.py','interface.json')]
    sources.append(ROOT/'rich_periodic_photon_20261015_model.py')
    target_path=ROOT/(P+target+'.py')
    if target_path not in sources:sources.append(target_path)
    if target=='observe':
        selected=ROOT/(P+'selected_state.json')
        sources.extend([selected,ROOT/json.loads(selected.read_text())['table_file']])
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    command=[PY,str(ROOT/(P+target+'.py'))]
    started=time.time()
    start=dict(command=command,cwd=str(ROOT),sources=manifest,python=PY,
               parameters='Exact constants and fresh candidate parameters in run.py main; no external evolved state inputs.',
               governor=dict(plan='issue222-rich-backward-candidates-20261014',revision='58589f21e86de29c925ce31215357cb25d8cde838ffebb9d53d8a38bc3f478d9',work_item='photon-candidate'),
               start_unix=started)
    (ROOT/(P+label+'_input.json')).write_text(json.dumps(start,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(P+label+'_source_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.write(p,p.name)
        z.write(ROOT/(P+label+'_input.json'),P+label+'_input.json')
    environment=os.environ.copy();environment.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (ROOT/(P+label+'_stdout.txt')).open('w') as out,(ROOT/(P+label+'_stderr.txt')).open('w') as err:
        child=subprocess.run(command,cwd=ROOT,env=environment,stdout=out,stderr=err)
    result=dict(science_child_natural_exit=child.returncode,elapsed_seconds=time.time()-started,source_hashes=manifest,
                sources_unchanged=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in manifest.items()))
    (ROOT/(P+label+'_exit.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return child.returncode

if __name__=='__main__':raise SystemExit(main())
