"""Immutable preexecution/source/input capsules for character resummation epochs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile
import joined_phi_u1_charged_exterior_resummation_consumer as consumer

ROOT = Path(__file__).resolve().parent

def run(stem: str) -> None:
    output = ROOT/(stem+'.json');capsule = ROOT/(stem+'_natural_exit.json');pre = ROOT/(stem+'_preexecution.zip')
    assert not output.exists() and not capsule.exists() and not pre.exists(), 'Never overwrite an executed epoch'
    captured = dict(consumer.old.field.CAPTURED)
    for module in tuple(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name:
            path = Path(name).resolve()
            if path.suffix=='.py' and path.parent==ROOT:captured[path.name] = path.read_bytes()
    for name in consumer.INPUTS:captured[name] = (ROOT/name).read_bytes()
    with zipfile.ZipFile(pre,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,raw in captured.items():archive.writestr(name,raw)
    started = time.monotonic();command = [sys.executable,str(Path(consumer.__file__).resolve()),'--output',str(output)]
    process = subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    unchanged = all((ROOT/name).read_bytes()==raw for name,raw in captured.items())
    record = dict(command=command,returncode=process.returncode,natural_exit=True,wall_seconds=time.monotonic()-started,stdout=process.stdout,stderr=process.stderr,all_sources_inputs_unchanged=unchanged,preexecution=dict(path=pre.name,sha256=consumer.sha(pre.read_bytes())),sha256={k:consumer.sha(v) for k,v in captured.items()},outputs={path.name:consumer.sha(path.read_bytes()) for path in (output,output.with_suffix('.zip')) if path.exists()})
    capsule.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(returncode=process.returncode,capsule=capsule.name,wall_seconds=record['wall_seconds'],stdout=process.stdout,stderr=process.stderr),indent=2))
    if process.returncode!=0 or not unchanged:raise RuntimeError('Failed epoch retained in immutable capsule')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--stem',required=True);run(parser.parse_args().stem)
