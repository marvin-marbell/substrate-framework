"""Exact-byte capture of NEW collective physical computation natural exit."""
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


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    prefix=sys.argv[1];inp=ROOT/(prefix+'_input.json');result=ROOT/(prefix+'_result.json')
    sources=[ROOT/('rich_spatial_collective_20261015_'+s+'.py') for s in ['model','calculate','capture']]
    sources += [ROOT/n for n in ['rich_spatial_quantum_sea_20261015_model.py','rich_spatial_quantum_sea_20261015_calculate.py','rich_common_colour_lepton_20261015_model.py','rich_colour_composite_20261014_model.py']]
    inputs=[inp,ROOT/'rich_spatial_quantum_sea_20261015_actual2_input.json']
    before={p.name:digest(p) for p in sources+inputs}
    command=[PYTHON,str(sources[1]),str(inp),str(result)];env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0')
    stdout=ROOT/(prefix+'_stdout.txt');stderr=ROOT/(prefix+'_stderr.txt');start=time.time()
    with stdout.open('wb') as so,stderr.open('wb') as se:process=subprocess.run(command,cwd=ROOT,env=env,stdout=so,stderr=se,check=False)
    files=sources+inputs+[stdout,stderr]+[p for p in [result,result.with_suffix('.npz')] if p.exists()]
    exitpath=ROOT/(prefix+'_exit.json')
    exitpath.write_text(json.dumps(dict(kind='NEW_COLLECTIVE_GAUSSIAN_SAME_LAW_NATURAL_EXIT_NOT_REVIEW',returncode=process.returncode,natural_exit=True,command=command,cwd=str(ROOT),started_unix=start,finished_unix=time.time(),source_and_input_before=before,source_and_input_after={p.name:digest(p) for p in sources+inputs},captured={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in files},review='Future Wave15; original prefix/capsules immutable'),indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(prefix+'_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[exitpath]:z.write(p,p.name)
    print(json.dumps(dict(prefix=prefix,returncode=process.returncode,elapsed=time.time()-start)))
    sys.exit(process.returncode)


if __name__=='__main__':main()
