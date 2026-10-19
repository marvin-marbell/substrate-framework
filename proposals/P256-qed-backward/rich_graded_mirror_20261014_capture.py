"""Source/input/output-bound natural exit capture of this NEW consumer."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parent
prefix='rich_graded_mirror_20261014';run=sys.argv[1];stem=f'{prefix}_{run}'
input_name=sys.argv[2] if len(sys.argv)>2 else f'{prefix}_input.json'
files=[f'{prefix}_model.py',f'{prefix}_calculate.py',f'{prefix}_capture.py',input_name,
       'rich_parity_sea_field_20261014_model.py','rich_spatial_holonomy_20261014_model.py','rich_frame_holonomy_20261014_model.py','rich_frame_gauss_20261014_model.py']
source_bytes={name:(ROOT/name).read_bytes() for name in files}
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
command=[sys.executable,f'{prefix}_calculate.py',input_name,stem]
started=time.time();child=subprocess.run(command,cwd=ROOT,env=env,capture_output=True)
(ROOT/f'{stem}_stdout.txt').write_bytes(child.stdout);(ROOT/f'{stem}_stderr.txt').write_bytes(child.stderr)
outputs={}
for suffix in ('stdout.txt','stderr.txt','result.json','states.npz'):
    path=ROOT/f'{stem}_{suffix}'
    if path.exists():outputs[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
record=dict(command=command,cwd=str(ROOT),returncode=child.returncode,natural_exit=True,wall_seconds=time.time()-started,python=sys.version,
            sources_sha256={name:hashlib.sha256(data).hexdigest() for name,data in source_bytes.items()},outputs_sha256=outputs,
            sources_unchanged_during_run={name:(ROOT/name).read_bytes()==data for name,data in source_bytes.items()})
(ROOT/f'{stem}_exit.json').write_text(json.dumps(record,indent=2)+'\n')
with zipfile.ZipFile(ROOT/f'{stem}_capsule.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
    for name,data in source_bytes.items():archive.writestr(name,data)
    for suffix in ('stdout.txt','stderr.txt','exit.json','result.json','states.npz'):
        path=ROOT/f'{stem}_{suffix}'
        if path.exists():archive.write(path,path.name)
print(json.dumps(record));sys.exit(child.returncode)
