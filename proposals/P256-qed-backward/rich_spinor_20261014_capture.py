"""Bind NEW actual calculation to exact bytes and a natural process exit."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent
MODE=sys.argv[1] if len(sys.argv)>1 else 'actual1'
if MODE not in ('actual1','actual2'): raise ValueError('Expected actual1 or actual2')
PREFIX='rich_spinor_20261014_'+MODE
ENTRY='rich_spinor_20261014_response.py' if MODE=='actual2' else 'rich_spinor_20261014_calculate.py'
SOURCES=['rich_spinor_20261014_model.py',ENTRY,'rich_spinor_20261014_capture.py']
if MODE=='actual2': SOURCES.append('rich_spinor_20261014_calculate.py')

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    before={name:digest(ROOT/name) for name in SOURCES}
    command=[sys.executable,str(ROOT/SOURCES[1])]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start=time.time()
    process=subprocess.run(command,cwd=ROOT,env=env,capture_output=True)
    elapsed=time.time()-start
    (ROOT/(PREFIX+'_stdout.txt')).write_bytes(process.stdout)
    (ROOT/(PREFIX+'_stderr.txt')).write_bytes(process.stderr)
    output_names=[PREFIX+'_stdout.txt',PREFIX+'_stderr.txt']
    if process.returncode==0: output_names += [PREFIX+'_result.json',PREFIX+'_states.npz']
    receipt={'kind':'NEW_ACTUAL_SCIENTIFIC_CALCULATION_NATURAL_EXIT','command':command,'cwd':str(ROOT),
             'returncode':process.returncode,'natural_exit':True,'wall_seconds':elapsed,
             'python':sys.version,'platform':platform.platform(),'thread_environment':{a:env[a] for a in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
             'source_before':before,'source_after':{name:digest(ROOT/name) for name in SOURCES},
             'output_hashes':{name:digest(ROOT/name) for name in output_names}}
    exitname=PREFIX+'_exit.json'; (ROOT/exitname).write_text(json.dumps(receipt,indent=2))
    with zipfile.ZipFile(ROOT/(PREFIX+'_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for name in SOURCES+output_names+[exitname]: z.write(ROOT/name,arcname=name)
    print(json.dumps(receipt,indent=2));print(process.stdout.decode());print(process.stderr.decode())
    raise SystemExit(process.returncode)

if __name__=='__main__': main()
