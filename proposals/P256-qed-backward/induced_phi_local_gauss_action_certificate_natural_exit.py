"""Run the selected new kinetic producer/checker and bind their natural exits."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'
PRODUCER = ROOT/'induced_phi_local_gauss_action_certificate.py'
RECEIPT = ROOT/'induced_phi_local_gauss_action_certificate_receipt.json'
CAPSULE = ROOT/'induced_phi_local_gauss_action_certificate_natural_exit_capsule.json'

def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def execute(command: list[str]) -> dict[str, Any]:
    start = time.monotonic()
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    return dict(command=command,cwd=str(ROOT),exit_code=completed.returncode,
                stdout=completed.stdout,stderr=completed.stderr,
                wall_seconds=time.monotonic()-start,natural_exit=True)

def main() -> None:
    selected = [PRODUCER,Path(__file__).resolve()]
    original = {path.name:path.read_bytes() for path in selected}
    producer = execute([PYTHON,str(PRODUCER),'--output',str(RECEIPT)])
    # External selected config, paths relative to actual config directory. It is
    # removed automatically; retained evidence names the actual checked sources.
    with tempfile.TemporaryDirectory(prefix='induced_phi_local_gauss_action_certificate_') as temporary:
        cfg = Path(temporary)/'pyrightconfig.json'
        configuration = dict(include=[os.path.relpath(path,cfg.parent) for path in selected],
                             extraPaths=[str(ROOT)],typeCheckingMode='basic',
                             pythonVersion='3.12',venvPath='/home/administrator/substrate-framework',venv='.venv')
        cfg.write_text(json.dumps(configuration))
        checker = execute(['/home/administrator/.local/bin/pyright','--project',str(cfg),'--stats'])
    unchanged = {name:(ROOT/name).read_bytes()==raw for name,raw in original.items()}
    payload: dict[str, Any] = dict(kind='actual_new_primitive_kinetic_natural_exit',
        runtime_python=sys.executable,producer=producer,selected_checker=checker,
        selected_checker_configuration=configuration,selected_files=[str(p) for p in selected],
        source_at_execution={name:dict(sha256=digest(raw),bytes_utf8=raw.decode()) for name,raw in original.items()},
        source_unchanged=unchanged,ownhash=digest(original[Path(__file__).name]),
        success=producer['exit_code']==0 and checker['exit_code']==0 and all(unchanged.values()))
    if producer['exit_code']==0:
        receipt_bytes=RECEIPT.read_bytes()
        receipt=json.loads(receipt_bytes)
        payload['receipt_sha256']=digest(receipt_bytes)
        payload['input_source_bundle']=receipt['immutable_input_source_bundle']
        payload['input_sha256']=receipt['input_sha256']
    CAPSULE.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({k:v for k,v in payload.items() if k not in ('source_at_execution','selected_checker_configuration')},indent=2))
    if not payload['success']:
        raise SystemExit(1)

if __name__=='__main__':
    main()
