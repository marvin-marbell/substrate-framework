"""Archive executed scientific source/inputs BEFORE running and preserve every epoch."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from typing import Any
import joined_phi_u1_collective_charged_ground_consumer as consumer

ROOT=Path(__file__).resolve().parent


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def run(stem: str) -> dict[str, Any]:
    source=Path(consumer.__file__).resolve();output=ROOT/(stem+'.json')
    capsule=ROOT/(stem+'_natural_exit.json');pre=ROOT/(stem+'_preexecution.zip')
    assert not capsule.exists() and not pre.exists(), 'Never overwrite a scientific epoch'
    raw=dict(consumer.field.CAPTURED)
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            path=Path(filename).resolve()
            if path.suffix=='.py' and path.parent==ROOT:raw[path.name]=path.read_bytes()
    raw[Path(__file__).name]=Path(__file__).read_bytes()
    for name in ['joined_phi_u1_neutral_ground_graph_delivery_receipt_20261010.json','joined_phi_u1_neutral_ground_graph_final_20261010.json','joined_phi_u1_neutral_ground_graph_final_20261010.npz','induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json','induced_phi_uniform_selfconsistent_full_normal_quantum_final_root_actual_receipt.json','induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json']:
        raw[name]=(ROOT/name).read_bytes()
    with zipfile.ZipFile(pre,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,contents in raw.items():archive.writestr(name,contents)
    started=time.monotonic()
    command=[sys.executable,str(source),'--output',str(output)]
    process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    unchanged=all((ROOT/name).read_bytes()==contents for name,contents in raw.items())
    record=dict(kind='ACTUAL_ALLNATIVE_SYMMETRIC63_FULL_H_NATURAL_EXIT_IMMUTABLE_EPOCH',command=command,returncode=process.returncode,natural_exit=True,wall_seconds=time.monotonic()-started,stdout=process.stdout,stderr=process.stderr,all_sources_inputs_unchanged=unchanged,preexecution=dict(path=pre.name,sha256=sha(pre.read_bytes())),sha256={name:sha(contents) for name,contents in raw.items()},outputs={path.name:sha(path.read_bytes()) for path in (output,output.with_suffix('.zip')) if path.exists()})
    capsule.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(returncode=process.returncode,wall_seconds=record['wall_seconds'],capsule=capsule.name,stderr=process.stderr,stdout=process.stdout),indent=2))
    if process.returncode!=0 or not unchanged:raise RuntimeError('Scientific failure or unstable epoch; capsule preserved')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stem',required=True);args=parser.parse_args();run(args.stem)
