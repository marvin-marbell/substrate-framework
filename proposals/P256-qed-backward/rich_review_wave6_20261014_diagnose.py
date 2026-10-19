"""Final stable selected-file pass only; caller supplies every explicit target."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent

def main():
    selected=sys.argv[1:]
    if not selected: raise ValueError('Explicit selected files required')
    if any(not (ROOT/name).is_file() for name in selected): raise FileNotFoundError(selected)
    digest=lambda name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    before={name:digest(name) for name in selected}
    version=subprocess.run(['/home/administrator/.local/bin/pyright','--version'],capture_output=True,text=True,check=True).stdout.strip()
    command=['/home/administrator/.local/bin/pyright','--pythonpath','/home/administrator/substrate-framework/.venv/bin/python','--outputjson',*selected]
    start=time.monotonic();done=subprocess.run(command,cwd=ROOT,capture_output=True,text=True);elapsed=time.monotonic()-start
    parsed=json.loads(done.stdout)
    byfile=Counter(Path(d['file']).name for d in parsed.get('generalDiagnostics',[]) if d['severity']=='error')
    artifact={'kind':'ONCE_FINAL_EXPLICIT_SELECTED_FILES_DIAGNOSTIC_NOT_ALL_CLEAN','command':command,'root':str(ROOT),'version':version,
              'natural_exit':True,'returncode':done.returncode,'elapsed_seconds':elapsed,'selected_files':selected,
              'source_SHA256_before':before,'sources_unchanged':all(digest(name)==value for name,value in before.items()),
              'selected_file_count':len(selected),'reported_summary':parsed.get('summary'),'error_count_by_file':dict(byfile),
              'checker':parsed,'stderr':done.stderr}
    (ROOT/'rich_review_wave6_20261014_diagnostics.json').write_text(json.dumps(artifact,indent=2)+'\n')
    print(json.dumps({k:artifact[k] for k in ['version','returncode','elapsed_seconds','selected_file_count','reported_summary','error_count_by_file','sources_unchanged']},indent=2))

if __name__=='__main__':main()
