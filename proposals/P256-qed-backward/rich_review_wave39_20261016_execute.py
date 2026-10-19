"""Capture exact declared sources and natural scientific subprocess exits."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parent
P='rich_review_wave39_20261016_'

def main():
    target=ROOT/sys.argv[1]
    label=target.stem.removeprefix(P)
    files={target,Path(__file__).resolve()}
    pending=list(files)
    while pending:
        current=pending.pop()
        for node in ast.walk(ast.parse(current.read_text())):
            modules=([node.module] if isinstance(node,ast.ImportFrom) else [alias.name for alias in node.names] if isinstance(node,ast.Import) else [])
            for module in modules:
                if module and module.startswith('rich_'):
                    source=ROOT/(module+'.py')
                    if source.exists() and source not in files:files.add(source);pending.append(source)
    if label in ('spatial','london'):
        files.update(ROOT/name for name in ('rich_dressed_local_carrier_20261015_interface.json','rich_dressed_local_carrier_20261015_states.npz'))
    if label == 'observer':
        files.add(ROOT/(P+'dynamics_raw.npz'))
    if label == 'enriched_observer':
        files.add(ROOT/(P+'enriched_raw.npz'))
    files=sorted(files)
    archive=ROOT/(P+label+'_sources.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in files:z.write(path,path.name)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    command=[sys.executable,str(target)]
    child=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    (ROOT/(P+label+'_stdout.txt')).write_text(child.stdout)
    (ROOT/(P+label+'_stderr.txt')).write_text(child.stderr)
    receipt=dict(command=command,natural_exit=child.returncode,source_hashes=hashes,source_capsule_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),sources_unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[p.name] for p in files))
    (ROOT/(P+label+'_exit.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    if child.returncode:print(child.stderr,flush=True)
    sys.exit(child.returncode)
if __name__=='__main__':main()
