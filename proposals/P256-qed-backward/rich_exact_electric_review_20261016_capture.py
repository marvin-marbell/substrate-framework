"""Real coupled science exits, exact operator/input epoch and own state pipeline."""
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
P='rich_exact_electric_review_20261016_'
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
    label=sys.argv[1] if len(sys.argv)>1 else 'coupled1';mode=sys.argv[2] if len(sys.argv)>2 else 'consume.py';base=ROOT/(P+label);script=ROOT/(P+mode);paths=set();dependencies(script,paths);dependencies(Path(__file__).resolve(),paths)
    interface=ROOT/'rich_exact_electric_picture_20261016_interface.json';paths.add(interface);own_inputs=[]
    if mode=='images.py':
        own_inputs=[ROOT/(P+'addition_case_raw.npz'),ROOT/(P+'result.json')];paths.update(own_inputs)
    hashes={path.name:digest(path) for path in sorted(paths)};contract=json.loads(interface.read_text());bindings={name:hashes[contract[name]['file']]==contract[name]['sha256'] for name in ('fields','model','frozen_H')}
    if not all(bindings.values()):raise AssertionError(('Frozen exact-electric operator interface mismatch',bindings))
    started=time.time();command=[PY,str(script)];record=dict(command=command,cwd=str(ROOT),source_input_sha256=hashes,operator_interface_bindings=bindings,own_prior_science_inputs=[path.name for path in own_inputs],author_state_inputs=[],plan='issue222-rich-backward-candidates-20261014',captured_revision='143d2f6af58dec9b80596fa3055175e5c5c131992d1865f8bef9201f555e8920',environment=dict(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),started=started,law='Distinct OLD48 CAR8/globalg/R8mode electric law. No Common CAR64/native47-state splice; original pure operator constructors only.')
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
