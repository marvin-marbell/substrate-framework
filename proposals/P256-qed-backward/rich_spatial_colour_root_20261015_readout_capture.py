"""Exact capture for bounded spectral/spatial readout of frozen graph snapshots."""
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
    cap=3800*1024**2;resource.setrlimit(resource.RLIMIT_AS,(cap,cap))

def main():
    prefix=sys.argv[1];inp=ROOT/(prefix+'_input.json');req=json.loads(inp.read_text());producer=req['producer_prefix'];out=ROOT/(prefix+'_result.json')
    sources=[ROOT/('rich_spatial_colour_root_20261015_'+x+'.py') for x in ['readout','readout_capture','calculate','model']]
    sources+=[ROOT/x for x in ['rich_common_colour_lepton_20261015_model.py','rich_colour_vector_loop_20261015_model.py','rich_colour_composite_20261014_model.py','rich_quantum_common_field_20261015_sources.py','rich_common_vector_20261015_model.py']]
    inputs=[inp]+[ROOT/(producer+suffix) for suffix in ['_input.json','_operators.npz','_states.npz','_result.npz','_result.json']]
    before={p.name:digest(p) for p in sources+inputs};initial=ROOT/(prefix+'_before.json')
    initial.write_text(json.dumps(dict(kind='EXACT_SOURCE_STATE_AND_READOUT_INPUT_BEFORE_EXECUTION',sha256=before),indent=2)+'\n')
    stdout=ROOT/(prefix+'_stdout.txt');stderr=ROOT/(prefix+'_stderr.txt');command=[PYTHON,str(sources[0]),str(inp),str(out)]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0');started=time.time()
    with stdout.open('wb') as fo,stderr.open('wb') as fe:
        process=subprocess.run(command,cwd=ROOT,env=env,stdout=fo,stderr=fe,check=False,preexec_fn=limit)
    files=sources+inputs+[initial,stdout,stderr]
    if out.exists():files.append(out)
    exit_path=ROOT/(prefix+'_exit.json')
    metadata=dict(kind='BOUNDED_RAW_SOURCE_AND_SPATIAL_READOUT_NATURAL_EXIT',command=command,returncode=process.returncode,natural_exit=process.returncode>=0,
        started_unix=started,finished_unix=time.time(),source_state_input_before=before,source_state_input_after={p.name:digest(p) for p in sources+inputs},
        files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in files},address_space_limit_MiB=3800,
        scope='Frozen exact new graph H/state readout, not a new fitted law or old producer replay. Full raw source resolvents and actual time amplitudes; no test/build/lint/formatter or transport/science admission.')
    exit_path.write_text(json.dumps(metadata,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(prefix+'_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
        for path in files+[exit_path]:archive.write(path,path.name)
    print(json.dumps(dict(prefix=prefix,returncode=process.returncode,elapsed=time.time()-started,result=str(out),exit=str(exit_path))))
    sys.exit(process.returncode)

if __name__=='__main__':main()
