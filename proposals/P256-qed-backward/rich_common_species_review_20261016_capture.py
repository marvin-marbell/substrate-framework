"""Pin exact native/source inputs; capture independent real process exit."""
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
P='rich_common_species_review_20261016_'
O='rich_common_native_species_20261016_'
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
    label=sys.argv[1] if len(sys.argv)>1 else 'actual1';base=ROOT/(P+label);script=ROOT/(P+'consume.py');paths=set();dependencies(script,paths);dependencies(Path(__file__).resolve(),paths)
    inputs=[O+'interface.json',O+'delivery1_manifest.json']
    for run in ('carrier1','joint1'):
        inputs.extend([O+run+'_body_basis.pkl',O+run+'_input.json'])
        for name in ('Gram','rotation','electric','plaquette','quark_vector','white_vector','quark_scalar','white_scalar','loopB','loopC'):inputs.append(O+run+'_op_'+name+'.npz')
    config=json.loads((ROOT/(O+'joint1_input.json')).read_text());inputs.extend(config[key] for key in ('neutral_basis_snapshot','neutral_coupling_snapshot','neutral_states_snapshot'))
    paths.update(ROOT/name for name in inputs);hashes={path.name:digest(path) for path in sorted(paths)}
    manifest=json.loads((ROOT/(O+'delivery1_manifest.json')).read_text());bindings={}
    for key,entry in manifest['native_entries'].items():
        name=key.split('/')[-1]
        if name in inputs:bindings[key]=hashes[name]==entry['SHA256']
    if not all(bindings.values()):raise AssertionError(('Frozen input hash mismatch',bindings))
    native=ROOT/(O+'delivery1_native.zip');native_hash=digest(native)
    if native_hash!='da12a723a26c7f6f100f3aa7bc6dbe4a5977991b7549186fb680b6a899464581':raise AssertionError('Native package epoch mismatch')
    command=[PY,str(script)];started=time.time();record=dict(command=command,cwd=str(ROOT),source_input_sha256=hashes,native_artifact_bindings=bindings,producer_native_zip_sha256=native_hash,plan='issue222-rich-backward-candidates-20261014',captured_revision='8e2d5c7c966a528442ea07f5160679554a3e24626f21a7602f5489668d499795',environment=dict(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),started=started,scope='Fresh source and state evolution from frozen native weak operators/Gram/canonical source coordinates; no producer CLI, bodyblock rebuild or author solved/prepared/evolved inputs.')
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
