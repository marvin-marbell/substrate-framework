"""One final explicit-owned-source pass; no ancestor clean claim or repeat setup."""
from __future__ import annotations
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from rich_review_wave12_20261015_diagnose import language_server

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_review_wave31_20261015_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def main() -> None:
    selected = [PREFIX+name+'.py' for name in ['consume','krylov','observe','capture','deliver','diagnose']]
    before = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in selected}
    command = ['/home/administrator/.local/bin/pyright','--pythonpath',PYTHON,'--outputjson',*selected]
    completed = subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    checked = json.loads(completed.stdout)
    lsp = language_server(selected)
    structural = {name:[dict(name=node.name,line=node.lineno) for node in ast.walk(ast.parse((ROOT/name).read_text()))
                        if isinstance(node,(ast.FunctionDef,ast.ClassDef))] for name in selected}
    result = dict(kind='ONCE_FINAL_SIX_OWNED_SOURCES_NOT_PROJECT_CLEAN',root=str(ROOT),scientific_interpreter=PYTHON,
        pyright_version=subprocess.run(['/home/administrator/.local/bin/pyright','--version'],capture_output=True,text=True,check=True).stdout.strip(),
        command=command,selected_files=selected,source_SHA256=before,natural_exit=True,returncode=completed.returncode,
        checker=checked,stderr=completed.stderr,summary=checked.get('summary'),
        error_counts_by_file=dict(Counter(Path(d['file']).name for d in checked.get('generalDiagnostics',[]) if d['severity']=='error')),
        rooted_LSP=lsp,positive_AST=structural,
        sources_unchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value for name,value in before.items()),
        scope='Explicit real six owned Python source paths. Existing installed direct rooted Pyright JSON-RPC because xd LSP device absent. No native-editor claim. No build/test/formatter or ancestor producer/checker replay.')
    (ROOT/(PREFIX+'language_final.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(returncode=completed.returncode,summary=result['summary'],errors=result['error_counts_by_file'],LSP_exit=lsp['server_exit'])),flush=True)


if __name__ == '__main__':
    main()
