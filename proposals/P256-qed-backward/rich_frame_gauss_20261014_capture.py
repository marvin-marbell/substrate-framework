"""Own-prefix natural-exit capture; never imports an old scientific producer."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_frame_gauss_20261014_'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    label = sys.argv[1] if len(sys.argv)>1 else 'actual1'
    source = [ROOT/(PREFIX+tail) for tail in ('model.py','calculate.py','capture.py','input.json')]
    before = {p.name:digest(p) for p in source}
    result = ROOT/(PREFIX+label+'_result.json')
    command = [sys.executable,str(ROOT/(PREFIX+'calculate.py')),str(ROOT/(PREFIX+'input.json')),str(result)]
    environment = dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start = time.time()
    completed = subprocess.run(command,cwd=ROOT,env=environment,capture_output=True)
    elapsed = time.time()-start
    stdout = ROOT/(PREFIX+label+'_stdout.txt'); stdout.write_bytes(completed.stdout)
    stderr = ROOT/(PREFIX+label+'_stderr.txt'); stderr.write_bytes(completed.stderr)
    outputs = [stdout,stderr]
    if result.exists(): outputs.append(result)
    if result.with_suffix('.npz').exists(): outputs.append(result.with_suffix('.npz'))
    receipt = ROOT/(PREFIX+label+'_exit.json')
    receipt.write_text(json.dumps({'kind':'NEW_NATURAL_EXIT_EXACT_SOURCE_INPUT_OUTPUT_BINDING','command':command,'cwd':str(ROOT),'python':sys.executable,'python_version':platform.python_version(),'start_unix':start,'elapsed_seconds':elapsed,'natural_exit':True,'returncode':completed.returncode,'sources_before':before,'sources_after':{p.name:digest(p) for p in source},'outputs':{p.name:digest(p) for p in outputs},'threads':{k:environment[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}},indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(PREFIX+label+'_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in source+outputs+[receipt]: z.write(p,p.name)
    print(receipt.read_text())
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
