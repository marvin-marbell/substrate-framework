"""Run only the NEW boundary_bulk CLI and preserve every epoch naturally."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile
ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'
def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
def archive(path: Path, files: list[Path]) -> None:
    if path.exists(): raise FileExistsError(path)
    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as z:
        for f in files: z.write(f,f.name)
    path.chmod(0o444)
def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);parser.add_argument('--epoch',required=True)
    args=parser.parse_args()
    if not args.epoch.startswith('boundary_bulk_'): raise ValueError('Owned epoch prefix required')
    request=json.loads(args.request.read_text()); files=[ROOT/name for name in request['input_sha256']]
    files.extend([args.request.resolve(),Path(__file__).resolve()])
    files=list(dict.fromkeys(files)); pins={str(f):digest(f) for f in files}
    base=ROOT/args.epoch; pre=Path(str(base)+'_preexecution.zip'); archive(pre,files)
    receipt=Path(str(base)+'_receipt.json'); stdout=Path(str(base)+'_stdout.txt'); stderr=Path(str(base)+'_stderr.txt')
    argv=[PYTHON,'-B',str(ROOT/'boundary_bulk_scientist_20261011.py'),'--request',str(args.request.resolve()),'--output',str(receipt)]
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    start=time.monotonic()
    process=subprocess.run(argv,cwd=ROOT,env=env,capture_output=True,check=False)
    elapsed=time.monotonic()-start; stdout.write_bytes(process.stdout);stderr.write_bytes(process.stderr)
    unchanged=all(digest(f)==pins[str(f)] for f in files)
    exitpath=Path(str(base)+'_exit.json')
    exitpath.write_text(json.dumps(dict(argv=argv,returncode=process.returncode,natural_exit=True,
        elapsed_seconds=elapsed,source_inputs_unchanged=unchanged,source_input_sha256=pins,
        preexecution_zip=pre.name,preexecution_sha256=digest(pre),thread_caps=1),indent=2)+'\n')
    outputs=[stdout,stderr,exitpath]+([receipt] if receipt.exists() else [])
    executed=Path(str(base)+'_executed.zip');archive(executed,files+outputs)
    capsule=dict(kind='IMMUTABLE_NEW_NATIVE_BOUNDARY_BULK_EXECUTION_CAPSULE',
        natural_exit=True,returncode=process.returncode,elapsed_seconds=elapsed,source_inputs_unchanged=unchanged,
        preexecution_zip=pre.name,preexecution_sha256=digest(pre),executed_zip=executed.name,
        executed_sha256=digest(executed),outputs_sha256={p.name:digest(p) for p in outputs},
        native_actual_Gibbs_pressure=False,native_state_spectral_identity=False)
    Path(str(base)+'_capsule.json').write_text(json.dumps(capsule,indent=2)+'\n')
    print(json.dumps(capsule));raise SystemExit(process.returncode if unchanged else 3)
if __name__=='__main__':main()
