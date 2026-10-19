"""Capture own numerical consumer, exact inputs, dependency bytes and real exit."""
import ast
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parent
P='rich_review_wave36_20261016_'
O='rich_composite_geometry_20261015_'
PY='/home/administrator/substrate-framework/.venv/bin/python'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def dependencies(path, found):
    if path in found:return
    found.add(path)
    for node in ast.walk(ast.parse(path.read_text())):
        modules=[]
        if isinstance(node,ast.Import):modules=[x.name for x in node.names]
        elif isinstance(node,ast.ImportFrom) and node.module:modules=[node.module]
        for module in modules:
            local=ROOT/(module.split('.')[0]+'.py')
            if local.exists():dependencies(local,found)


def main():
    label=sys.argv[1] if len(sys.argv)>1 else 'actual1';base=ROOT/(P+label)
    script=ROOT/(P+(sys.argv[2] if len(sys.argv)>2 else 'consume.py'));paths={Path(__file__).resolve()};dependencies(script,paths)
    inputs=[O+'correlated_source_eom1_input.json',O+'correlated_cloud_transport1_transport_CP.pkl',O+'correlated_source_eom1_B_data.npz',O+'correlated_source_eom1_B_result.json']
    for state in ('charged','neutral_reference'):
        inputs.extend([O+'correlated_cloud_residual1_'+state+'_residual_data.npz',O+'correlated_cloud_residual1_'+state+'_result.json'])
    for name in ('tetra_poles1','correlated_onehop_poles1'):
        inputs.extend([O+name+'_exit.json',O+name+'_termination_decision.json'])
    config=json.loads((ROOT/inputs[0]).read_text())
    inputs.extend(config[key] for key in ('neutral_basis_snapshot','neutral_coupling_snapshot','neutral_states_snapshot'))
    paths.update(ROOT/name for name in inputs)
    hashes={path.name:digest(path) for path in sorted(paths)}
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    record=dict(command=[PY,str(script)],cwd=str(ROOT),source_input_sha256=hashes,environment={key:env[key] for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')},started=time.time())
    Path(str(base)+'_input.json').write_text(json.dumps(record,indent=2)+'\n')
    with zipfile.ZipFile(str(base)+'_source_capsule.zip','w',zipfile.ZIP_DEFLATED) as capsule:
        for path in sorted(paths):capsule.write(path,path.name)
        capsule.write(Path(str(base)+'_input.json'),base.name+'_input.json')
    with open(str(base)+'_stdout.txt','wb') as out,open(str(base)+'_stderr.txt','wb') as err:
        child=subprocess.run(record['command'],cwd=ROOT,env=env,stdout=out,stderr=err)
    record.update(returncode=child.returncode,natural_exit=child.returncode>=0,elapsed_seconds=time.time()-record['started'],child_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,source_input_unchanged=all(digest(ROOT/name)==sha for name,sha in hashes.items()))
    Path(str(base)+'_exit.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
    return child.returncode


if __name__=='__main__':raise SystemExit(main())
