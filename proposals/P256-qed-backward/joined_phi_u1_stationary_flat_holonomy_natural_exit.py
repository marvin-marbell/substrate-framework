"""Archive exact source/input bytes and naturally completed scientific CLI."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import zipfile

ROOT=Path(__file__).resolve().parent
PREFIX='joined_phi_u1_stationary_flat_holonomy'
LAUNCHER='''import importlib.abc, importlib.machinery, pathlib, runpy, sys
root=pathlib.Path(sys.argv[1]).parent
class Loader(importlib.machinery.SourceFileLoader):
 def get_code(self,fullname):
  raw=pathlib.Path(self.path).read_bytes()
  return self.source_to_code(raw,self.path)
class Finder(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
  if spec and spec.origin and pathlib.Path(spec.origin).parent==root and spec.origin.endswith('.py'):
   spec.loader=Loader(fullname,spec.origin)
  return spec
sys.meta_path.insert(0,Finder())
source=sys.argv[1];sys.argv=sys.argv[1:]
runpy.run_path(source,run_name='__main__')
'''


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    source=ROOT/(PREFIX+'.py');runner=Path(__file__).resolve();source_before=source.read_bytes();runner_before=runner.read_bytes()
    snapshots={source.name:source_before,runner.name:runner_before};pending=[source.name]
    while pending:
        name=pending.pop()
        for node in ast.walk(ast.parse(snapshots[name])):
            modules=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
            for module in modules:
                path=ROOT/(module.split('.')[0]+'.py')
                if path.is_file() and path.name not in snapshots:
                    snapshots[path.name]=path.read_bytes();pending.append(path.name)
    for name in ('induced_phi_uniform_selfconsistent_full_normal_quantum_final_root_actual_receipt.json','induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json'):
        snapshots[name]=(ROOT/name).read_bytes()
    command=[sys.executable,'-B','-c',LAUNCHER,str(source),'--output',str(args.output)]
    execution_archive=args.output.with_name(args.output.stem+'_execution_bytes.zip')
    with zipfile.ZipFile(execution_archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,raw in sorted(snapshots.items()):z.writestr(name,raw)
        z.writestr('launcher.py',LAUNCHER);z.writestr('command.json',json.dumps(command))
    started=time.monotonic();proc=subprocess.run(command,cwd=ROOT,capture_output=True)
    stdout=args.output.with_name(args.output.stem+'_stdout.txt');stderr=args.output.with_name(args.output.stem+'_stderr.txt');stdout.write_bytes(proc.stdout);stderr.write_bytes(proc.stderr)
    result=json.loads(args.output.read_bytes()) if proc.returncode==0 else None
    capsule: dict[str,Any]=dict(command=command,natural_exit=True,exit_code=proc.returncode,elapsed_seconds=time.monotonic()-started,stdout=str(stdout),stderr=str(stderr),stdout_sha256=digest(proc.stdout),stderr_sha256=digest(proc.stderr),execution_archive=str(execution_archive),execution_archive_sha256=digest(execution_archive.read_bytes()),source_sha256_before=digest(source_before),source_sha256_after=digest(source.read_bytes()),runner_sha256_before=digest(runner_before),runner_sha256_after=digest(runner.read_bytes()),receipt_sha256=digest(args.output.read_bytes()) if result is not None else None,input_archive_sha256=digest(Path(result['immutable_input_archive']).read_bytes()) if result is not None else None)
    capsule['source_input_sha256_before']={name:digest(raw) for name,raw in snapshots.items()}
    capsule['source_input_sha256_after']={name:digest((ROOT/name).read_bytes()) for name in snapshots}
    if capsule['source_input_sha256_before']!=capsule['source_input_sha256_after']:raise ArithmeticError('Archived source/input epoch changed')
    if result is not None:
        for name,sha in result['source_input_sha256_before'].items():
            if capsule['source_input_sha256_before'].get(name)!=sha:raise ArithmeticError('Executed source not bound before CLI: '+name)
    capsule_path=args.output.with_name(args.output.stem+'_natural_exit_capsule.json');capsule_path.write_text(json.dumps(capsule,indent=2)+'\n')
    if capsule['source_sha256_before']!=capsule['source_sha256_after'] or capsule['runner_sha256_before']!=capsule['runner_sha256_after']:raise ArithmeticError('Execution source changed')
    print(json.dumps(capsule,indent=2));sys.stdout.flush()
    if proc.returncode:sys.stderr.write(proc.stderr.decode());raise SystemExit(proc.returncode)


if __name__=='__main__':main()
