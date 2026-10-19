"""Once-final selected owned sources: rooted LSP, positive AST and Pyright."""
from __future__ import annotations
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time
from rich_review_wave12_20261015_diagnose import language_server

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_gauss_photon_phase_20261015_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def main() -> None:
    selected = [PREFIX+name+'.py' for name in ('spectral','consume','capture','language')]
    before = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in selected}
    version = subprocess.run(['/home/administrator/.local/bin/pyright','--version'],capture_output=True,text=True,check=True).stdout.strip()
    command = ['/home/administrator/.local/bin/pyright','--pythonpath',PYTHON,'--outputjson',*selected]
    start = time.monotonic()
    done = subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    checker = json.loads(done.stdout)
    lsp = language_server(selected)
    structural = {name:[dict(name=n.name,line=n.lineno) for n in ast.walk(ast.parse((ROOT/name).read_text())) if isinstance(n,(ast.ClassDef,ast.FunctionDef))] for name in selected}
    output = dict(kind='ONCE_FINAL_OWNED_CONSUMER_SCOPE_NOT_PROJECT_CLEAN',root=str(ROOT),version=version,
                  scientific_interpreter=PYTHON,command=command,returncode=done.returncode,natural_exit=True,
                  selected_files=selected,source_SHA256=before,
                  sources_unchanged=all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==p for n,p in before.items()),
                  elapsed_seconds=time.monotonic()-start,summary=checker.get('summary'),
                  errors_by_file=dict(Counter(Path(d['file']).name for d in checker.get('generalDiagnostics',[]) if d['severity']=='error')),
                  checker=checker,stderr=done.stderr,LSP=lsp,positive_AST=structural,
                  scope='Four new owned sources only. Frozen owning model and pure geometry/CAR imports not relabelled type-clean. Native LSP device unmounted; actual installed direct Pyright JSON-RPC rooted in project.')
    (ROOT/(PREFIX+'language_final.json')).write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(returncode=done.returncode,summary=output['summary'],errors_by_file=output['errors_by_file'],LSP_exit=lsp['server_exit'],sources_unchanged=output['sources_unchanged'])),flush=True)


if __name__ == '__main__':
    main()
