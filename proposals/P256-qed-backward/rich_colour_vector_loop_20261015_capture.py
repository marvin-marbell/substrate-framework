"""Freeze NEW vector-law sources/inputs and capture real natural process exits."""
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

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def memory_limit():
    maximum=3800*1024**2
    resource.setrlimit(resource.RLIMIT_AS,(maximum,maximum))

def main():
    prefix=sys.argv[1];inp=ROOT/(prefix+'_input.json');result=ROOT/(prefix+'_result.json')
    sources=[ROOT/('rich_colour_vector_loop_20261015_'+x+'.py') for x in ['model','calculate','capture']]
    sources += [ROOT/x for x in ['rich_common_colour_lepton_20261015_model.py','rich_common_vector_20261015_model.py','rich_quantum_common_field_20261015_sources.py','rich_colour_composite_20261014_model.py']]
    before={p.name:digest(p) for p in sources+[inp]};stdout=ROOT/(prefix+'_stdout.txt');stderr=ROOT/(prefix+'_stderr.txt')
    command=[PYTHON,str(sources[1]),str(inp),str(result)]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0')
    started=time.time()
    with stdout.open('wb') as out,stderr.open('wb') as err:
        process=subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=err,check=False,preexec_fn=memory_limit)
    files=sources+[inp,stdout,stderr]
    for path in [result,result.with_suffix('.npz')]:
        if path.exists():files.append(path)
    exit_path=ROOT/(prefix+'_exit.json')
    metadata=dict(kind='NEW_COLOUR_VECTOR_H0_PHYSICAL_CONSUMER_NATURAL_EXIT_NOT_REVIEW',command=command,cwd=str(ROOT),returncode=process.returncode,
        natural_exit=True,started_unix=started,finished_unix=time.time(),source_input_before=before,source_input_after={p.name:digest(p) for p in sources+[inp]},
        files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in files},
        environment={key:env[key] for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONHASHSEED']},
        address_space_limit_MiB=3800,limits='No build/lint/tests/formatters, old producer replay, global monkeypatch or inherited source strengths. Pure frozen CAR/Gauss/PW19/full-mixed-source helpers captured transitively. Physical consumer exit evidence, not independent review/admission.')
    exit_path.write_text(json.dumps(metadata,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(prefix+'_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
        for path in files+[exit_path]:archive.write(path,path.name)
    print(json.dumps(dict(prefix=prefix,returncode=process.returncode,elapsed=time.time()-started,result=str(result),exit=str(exit_path))))
    sys.exit(process.returncode)

if __name__=='__main__':main()
