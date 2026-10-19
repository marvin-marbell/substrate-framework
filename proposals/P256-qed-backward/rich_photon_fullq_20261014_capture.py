#!/usr/bin/env python3
"""Natural scientific exit capsule for full-quantum connected Gauss candidate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--geometry',default='double_face');parser.add_argument('--cutoff',type=int,default=3)
parser.add_argument('--epoch',default='final');parser.add_argument('--background',default='integer_ions')
args=parser.parse_args();name=f'rich_photon_fullq_20261014_{args.geometry}_L{args.cutoff}_{args.background}_{args.epoch}'
source=root/'rich_photon_fullq_20261014_model.py';helper=root/'rich_photon_20261014_loop.py'
inputs={'geometry':args.geometry,'cutoff':args.cutoff,'hopping':1.,'repulsion':4.,'electric_inertia':1.,'reference_particles':2,'anchor':0,
        'background':args.background,
        'neutral_spin_numbers':[1,1],'addition_spin_numbers':[2,1],'removal_spin_numbers':[0,1],
        'q':[.7,0,0],'omega':[0.,.3,.7,1.2,2.],'eta':.05,'time_interval':[0.,8.],'time_samples':33,
        'charged_G_omega':[-2.,-.7,0.,.7,2.],'source_site_double_face':4,'alternate_Wilson_string':'tree_path minus xz face',
        'response':'Full-H sparse resolvents; low eigenstates18 neutral,12 charged/removal are partial spectra'}
source_bytes=source.read_bytes();helper_bytes=helper.read_bytes();input_bytes=json.dumps(inputs,sort_keys=True).encode()
command=[sys.executable,str(source),'--geometry',args.geometry,'--cutoff',str(args.cutoff),'--background',args.background,'--name',name]
start=time.time();result=subprocess.run(command,cwd=root,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},capture_output=True)
outputs={name+'_stdout.txt':result.stdout,name+'_stderr.txt':result.stderr}
for filename,data in outputs.items():(root/filename).write_bytes(data)
archive=root/(name+'_execution.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    z.writestr(source.name,source_bytes);z.writestr(helper.name,helper_bytes);z.writestr('inputs.json',input_bytes)
    for filename,data in outputs.items():z.writestr(filename,data)
    for suffix in ('_result.json','_states.npz'):
        path=root/(name+suffix)
        if path.exists():z.write(path,path.name)
receipt={'command':command,'cwd':str(root),'natural_returncode':result.returncode,'source_sha256':hashlib.sha256(source_bytes).hexdigest(),
         'helper_sha256':hashlib.sha256(helper_bytes).hexdigest(),'input_sha256':hashlib.sha256(input_bytes).hexdigest(),'inputs':inputs,
         'wall_seconds':time.time()-start,'stdout_sha256':hashlib.sha256(result.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(result.stderr).hexdigest(),
         'outputs':{suffix:hashlib.sha256((root/(name+suffix)).read_bytes()).hexdigest() for suffix in ('_result.json','_states.npz') if (root/(name+suffix)).exists()},
         'execution_zip_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
(root/(name+'_natural_exit.json')).write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2));print(result.stdout.decode());print(result.stderr.decode(),file=sys.stderr)
raise SystemExit(result.returncode)
