"""Exact source/native input capsule and actual scientific process exit."""
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
from rich_composite_geometry_20261015_science import load
ROOT=Path(__file__).resolve().parent
P='rich_native_body_review_20261016_'
RUN='rich_composite_geometry_20261015_correlated_dressing3'
PY='/home/administrator/substrate-framework/.venv/bin/python'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def dependencies(path,seen):
    if path in seen:return
    seen.add(path)
    for node in ast.walk(ast.parse(path.read_text())):
        modules=[name.name for name in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
        for module in modules:
            candidate=ROOT/(module.split('.')[0]+'.py')
            if candidate.exists():dependencies(candidate,seen)


def main():
    label=sys.argv[1] if len(sys.argv)>1 else 'actual1';base=ROOT/(P+label);script=ROOT/(P+(sys.argv[2] if len(sys.argv)>2 else 'consume.py'));paths=set();dependencies(script,paths);dependencies(Path(__file__).resolve(),paths)
    meta=load(ROOT/(RUN+'_body_basis.pkl'));storage=meta['storage'];first=next(iter(storage.meta));j,count=first;changed=list(count);changed[0]-=1;changed[4]+=1
    selected=[first,(j,tuple(changed)),next(label for label in storage.meta if label[0]==3)]
    inputs=[RUN+'_body_basis.pkl',RUN+'_input.json','rich_composite_geometry_20261015_native_review_interface.json',RUN+'_assembly_receipt.json']+[storage.paths[label] for label in selected]
    config=json.loads((ROOT/(RUN+'_input.json')).read_text());inputs.extend(config[key] for key in ('neutral_basis_snapshot','neutral_coupling_snapshot','neutral_states_snapshot'))
    for name in ('Gram','rotation','electric','plaquette','quark_vector','white_vector','quark_scalar','white_scalar','loopB','loopC'):
        inputs.extend([RUN+'_op_'+name+'.npz',RUN+'_raw_'+name+'.npz'])
    paths.update(ROOT/name for name in inputs);hashes={path.name:digest(path) for path in sorted(paths)}
    interface=json.loads((ROOT/'rich_composite_geometry_20261015_native_review_interface.json').read_text());bindings={name:hashes[name]==entry['sha256'] for name,entry in interface['artifacts'].items() if name in hashes}
    if not all(bindings.values()):raise AssertionError(('Frozen native input hash mismatch',bindings))
    command=[PY,str(script)];started=time.time();record=dict(command=command,cwd=str(ROOT),source_input_sha256=hashes,interface_hash_bindings=bindings,governing_plan='issue222-rich-backward-candidates-20261014',captured_revision='fe8a72a60be4e42a08270f54b24dd8f5e17ef263e30e6abf121eff19c0e736aa',environment=dict(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),started=started,forbidden_state_inputs='No lowstates/prepared/evolved arrays loaded. Body probe coordinates are declared canonical source inputs; fresh G preparation/eigenstates/evolution executed.')
    Path(str(base)+'_input.json').write_text(json.dumps(record,indent=2)+'\n')
    with zipfile.ZipFile(str(base)+'_source_capsule.zip','w',zipfile.ZIP_DEFLATED) as capsule:
        for path in sorted(paths):capsule.write(path,path.name)
        capsule.write(Path(str(base)+'_input.json'),base.name+'_input.json')
    environment=os.environ.copy();environment.update(record['environment'])
    with open(str(base)+'_stdout.txt','wb') as out,open(str(base)+'_stderr.txt','wb') as err:child=subprocess.run(command,cwd=ROOT,env=environment,stdout=out,stderr=err)
    record.update(returncode=child.returncode,natural_exit=child.returncode>=0,elapsed_seconds=time.time()-started,child_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,source_input_unchanged=all(digest(ROOT/name)==sha for name,sha in hashes.items()))
    Path(str(base)+'_exit.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
    return child.returncode


if __name__=='__main__':raise SystemExit(main())
