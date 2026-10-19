"""Capture own reconstructed words, raw input bindings and scientific exit."""
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
P='rich_primitive_vertex_review_20261016_'
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
    primary=['rich_composite_geometry_20261015_correlated_onehop_neutral1_neutral.pkl','rich_composite_geometry_20261015_correlated_onehop_reference1_J0_coupling.pkl','rich_composite_geometry_20261015_correlated_onehop_reference1_J0_states.npz']
    secondary=['rich_common_source_vertex_20261016_actual1_'+direction+'_raw_tensor.npz' for direction in ('white','quark')]
    records=['rich_common_source_vertex_20261016_actual1_exit.json','rich_common_source_vertex_20261016_interface.json']
    paths.update(ROOT/name for name in primary+secondary+records);hashes={path.name:digest(path) for path in sorted(paths)};producer=json.loads((ROOT/records[0]).read_text())
    bindings={name:hashes[name]==producer['source_input_before'][name] for name in primary}
    bindings.update({name:hashes[name]==producer['files'][name]['SHA256'] for name in secondary})
    if not all(bindings.values()):raise AssertionError(('Physical input/secondary output epoch mismatch',bindings))
    started=time.time();command=[PY,str(script)];record=dict(command=command,cwd=str(ROOT),source_input_sha256=hashes,primary_physical_inputs=primary,secondary_comparison_ONLY=secondary,epoch_bindings=bindings,plan='issue222-rich-backward-candidates-20261014',captured_revision='ba0876c0c0b85ffcded63bc7c1ec4caae9bf2eb48a20671379025f54ecd48761',environment=dict(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),started=started,independence='Own pivotedQR exact degree3 field basis, finitePhi7 derivatives, rawsourceCAR, independent32bit Fock checks and full-pair KdagK words. Producer tensors read only AFTER all own primitive moments/connected normalization are reconstructed; no Vpsi/27x27Gram input or producer CLI.')
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
