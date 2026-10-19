"""Once-stable new owned Python LSP/AST/Pyright evidence, not project-wide cleanup."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time
from rich_review_wave12_20261015_diagnose import language_server

ROOT=Path(__file__).resolve().parent
PREFIX='rich_photon_collective_20261015_'
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'

def run():
    selected=[PREFIX+name+'.py' for name in ('probe','capture','model','compute','coordinate','coordinate_energy','control','repair','complete_groups','recover_groups','energy_account','language')]
    before={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in selected};started=time.monotonic();version=subprocess.run(['/home/administrator/.local/bin/pyright','--version'],capture_output=True,text=True,check=True).stdout.strip();command=['/home/administrator/.local/bin/pyright','--pythonpath',PYTHON,'--outputjson',*selected];done=subprocess.run(command,cwd=ROOT,capture_output=True,text=True);checker=json.loads(done.stdout);lsp=language_server(selected);positive={name:[dict(name=node.name,line=node.lineno) for node in ast.walk(ast.parse((ROOT/name).read_text())) if isinstance(node,(ast.ClassDef,ast.FunctionDef))] for name in selected}
    artifact=dict(kind='FINAL_NEW_COLLECTIVE_OWNED_SCOPE_NOT_PROJECT_CLEAN',selected_files=selected,version=version,command=command,returncode=done.returncode,natural_exit=True,wall_seconds=time.monotonic()-started,summary=checker.get('summary'),errors_by_file=dict(Counter(Path(d['file']).name for d in checker.get('generalDiagnostics',[]) if d['severity']=='error')),checker=checker,stderr=done.stderr,LSP=lsp,positive_AST=positive,source_SHA256=before,sources_unchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in before.items()),scope='ONLY twelve new owned Python sources. Real installed Pyright LSP symbols/explicit diagnostics plus Python AST. Frozen helper debt not repaired or relabelled clean; no build/lint/tests/formatters.')
    (ROOT/(PREFIX+'language_final.json')).write_text(json.dumps(artifact,indent=2)+'\n');print(json.dumps(dict(returncode=done.returncode,summary=artifact['summary'],errors_by_file=artifact['errors_by_file'],server_exit=lsp['server_exit'],sources_unchanged=artifact['sources_unchanged'])),flush=True)

if __name__=='__main__':run()
