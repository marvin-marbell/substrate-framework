"""Capture exact own whole-H physical extension inputs and natural exit."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parent;prefix='rich_parity_sea_field_20261014'
consumer,input_name,run=sys.argv[1:4];stem=f'{prefix}_{run}'
config=json.loads((ROOT/input_name).read_text())
files=[f'{prefix}_{consumer}.py',f'{prefix}_extension_capture.py',input_name,config['law_input'],config['source_states'],f'{prefix}_model.py',f'{prefix}_calculate.py',
       'rich_spatial_holonomy_20261014_model.py','rich_frame_holonomy_20261014_model.py','rich_frame_gauss_20261014_model.py']
if 'source_result' in config:files.append(config['source_result'])
source_bytes={name:(ROOT/name).read_bytes() for name in files}
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
command=[sys.executable,f'{prefix}_{consumer}.py',input_name,stem];started=time.time()
child=subprocess.run(command,cwd=ROOT,env=env,capture_output=True)
(ROOT/f'{stem}_stdout.txt').write_bytes(child.stdout);(ROOT/f'{stem}_stderr.txt').write_bytes(child.stderr)
record=dict(command=command,cwd=str(ROOT),returncode=child.returncode,natural_exit=True,wall_seconds=time.time()-started,python=sys.version,
            sources_sha256={name:hashlib.sha256(data).hexdigest() for name,data in source_bytes.items()},sources_unchanged_during_run={name:(ROOT/name).read_bytes()==data for name,data in source_bytes.items()})
(ROOT/f'{stem}_exit.json').write_text(json.dumps(record,indent=2)+'\n')
with zipfile.ZipFile(ROOT/f'{stem}_capsule.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
    for name,data in source_bytes.items():archive.writestr(name,data)
    for suffix in ('stdout.txt','stderr.txt','exit.json','result.json','states.npz'):
        path=ROOT/f'{stem}_{suffix}'
        if path.exists():archive.write(path,path.name)
print(json.dumps(record));sys.exit(child.returncode)
