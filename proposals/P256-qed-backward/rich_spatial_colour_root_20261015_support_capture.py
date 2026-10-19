"""Capture exact raw-support consumer and its already-snapshotted physical state."""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def limit():
    maximum=3800*1024**2;resource.setrlimit(resource.RLIMIT_AS,(maximum,maximum))

def main():
    prefix=sys.argv[1];inp=ROOT/(prefix+'_input.json');request=json.loads(inp.read_text());producer=request['producer_prefix'];out=ROOT/(prefix+'_result.json')
    sources=[ROOT/('rich_spatial_colour_root_20261015_'+x+'.py') for x in ['support','support_capture','model']]
    sources+=[ROOT/x for x in ['rich_common_colour_lepton_20261015_model.py','rich_colour_vector_loop_20261015_model.py','rich_colour_composite_20261014_model.py','rich_quantum_common_field_20261015_sources.py','rich_common_vector_20261015_model.py']]
    inputs=[inp]+[ROOT/(producer+suffix) for suffix in ['_input.json','_operators.npz','_states.npz','_preflight.json']]
    before={p.name:digest(p) for p in sources+inputs};stdout=ROOT/(prefix+'_stdout.txt');stderr=ROOT/(prefix+'_stderr.txt')
    initial=ROOT/(prefix+'_before.json');initial.write_text(json.dumps(dict(kind='EXACT_SUPPORT_SOURCE_AND_STATE_BEFORE_DIAGNOSTIC',sha256=before),indent=2)+'\n')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0')
    command=[PYTHON,str(sources[0]),str(inp),str(out)];started=time.time()
    with stdout.open('wb') as fo,stderr.open('wb') as fe:
        process=subprocess.run(command,cwd=ROOT,env=env,stdout=fo,stderr=fe,check=False,preexec_fn=limit)
    files=sources+inputs+[initial,stdout,stderr]
    if out.exists():files.append(out)
    exit_path=ROOT/(prefix+'_exit.json')
    metadata=dict(kind='RAW_PHYSICAL_SUPPORT_NATURAL_CONSUMER_NOT_BUILD_OR_REVIEW',command=command,returncode=process.returncode,
        natural_exit=process.returncode>=0,signal=(-process.returncode if process.returncode<0 else None),started_unix=started,finished_unix=time.time(),
        source_state_input_before=before,source_state_input_after={p.name:digest(p) for p in sources+inputs},
        files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in files},address_space_limit_MiB=3800,
        scope='Actual new spatial full-CAR current/density/hopping images and finite electric bound on EXACT captured states. No producer replay, state refit, global monkeypatch, tests/build/lint/formatters or admission.')
    exit_path.write_text(json.dumps(metadata,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(prefix+'_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
        for path in files+[exit_path]:archive.write(path,path.name)
    print(json.dumps(dict(prefix=prefix,returncode=process.returncode,elapsed=time.time()-started,result=str(out),exit=str(exit_path))))
    sys.exit(process.returncode)

if __name__=='__main__':main()
