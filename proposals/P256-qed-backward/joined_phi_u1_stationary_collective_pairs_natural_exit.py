"""Run the actual scientific main and bind natural exit plus immutable native bytes."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import joined_phi_u1_stationary_collective_pairs as producer

ROOT = Path(__file__).resolve().parent
PREFIX = 'joined_phi_u1_stationary_collective_pairs_actual_20261010'


def digest(path: Path) -> dict:
    raw=path.read_bytes()
    return dict(path=path.name,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def main() -> None:
    source={**producer.CAPTURED,Path(__file__).name:Path(__file__).read_bytes()}
    names=['induced_phi_uniform_selfconsistent_full_normal_quantum_final_root_actual_receipt.json',
           'induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json']
    inputs={name:(ROOT/name).read_bytes() for name in names}
    runtime=ROOT/(PREFIX+'_runtime_bytes.zip')
    with zipfile.ZipFile(runtime,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,raw in {**source,**inputs}.items():archive.writestr(name,raw)
    receipt=ROOT/(PREFIX+'_receipt.json')
    stdout=ROOT/(PREFIX+'_stdout.txt');stderr=ROOT/(PREFIX+'_stderr.txt')
    command=[sys.executable,str(ROOT/'joined_phi_u1_stationary_collective_pairs.py'),'--output',str(receipt)]
    start=time.monotonic()
    with stdout.open('wb') as out,stderr.open('wb') as err:
        completed=subprocess.run(command,cwd=ROOT,stdout=out,stderr=err,check=False)
    result=dict(command=command,executable=sys.executable,natural_returncode=completed.returncode,
                elapsed_seconds=time.monotonic()-start,stdout=digest(stdout),stderr=digest(stderr),
                immutable_native_source_input_archive=digest(runtime),
                sources={name:dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),
                             unchanged=(ROOT/name).read_bytes()==raw) for name,raw in source.items()},
                inputs={name:dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),
                             unchanged=(ROOT/name).read_bytes()==raw) for name,raw in inputs.items()})
    if completed.returncode==0:
        result.update(receipt=digest(receipt),science_archive=digest(receipt.with_suffix('.zip')),
                      parent_arrays=digest(receipt.with_suffix('.npz')))
    capsule=ROOT/(PREFIX+'_natural_exit_capsule.json')
    capsule.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if completed.returncode!=0:sys.exit(completed.returncode)


if __name__=='__main__':main()
