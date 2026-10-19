"""Capture NEW process natural exit and exact source/input/output bytes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prefix=sys.argv[1];inp=ROOT/(prefix+'_input.json');result=ROOT/(prefix+'_result.json')
    sources=[ROOT/('rich_spatial_quantum_sea_20261015_'+suffix+'.py') for suffix in ['model','calculate','capture']]
    sources += [ROOT/'rich_common_colour_lepton_20261015_model.py',ROOT/'rich_colour_composite_20261014_model.py']
    hashes={p.name:digest(p) for p in sources+[inp]}
    command=[PYTHON,str(sources[1]),str(inp),str(result)]
    environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0')
    stdout=ROOT/(prefix+'_stdout.txt');stderr=ROOT/(prefix+'_stderr.txt');start=time.time()
    with stdout.open('wb') as so,stderr.open('wb') as se:
        process=subprocess.run(command,cwd=ROOT,env=environment,stdout=so,stderr=se,check=False)
    files=sources+[inp,stdout,stderr]
    files += [p for p in [result,result.with_suffix('.npz')] if p.exists()]
    exitpath=ROOT/(prefix+'_exit.json')
    metadata=dict(kind='NEW_SPATIAL_SEA_PHYSICAL_CONSUMER_NATURAL_EXIT_NOT_REVIEW',command=command,cwd=str(ROOT),
        returncode=process.returncode,natural_exit=True,started_unix=start,finished_unix=time.time(),
        source_and_input_before=hashes,source_and_input_after={p.name:digest(p) for p in sources+[inp]},
        captured={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in files},
        environment={k:environment[k] for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONHASHSEED']},
        scope='Conditional actual3D filled-sea quantum variational computation; future Wave15 independent audit, not Wave14 admission.')
    exitpath.write_text(json.dumps(metadata,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(prefix+'_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as capsule:
        for p in files+[exitpath]:capsule.write(p,p.name)
    print(json.dumps(dict(prefix=prefix,returncode=process.returncode,elapsed=time.time()-start)))
    sys.exit(process.returncode)


if __name__=='__main__':main()
