"""Stable final selected Wave28 diagnostics, rooted protocol and positive AST."""
from __future__ import annotations
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any
from rich_review_wave12_20261015_diagnose import language_server

ROOT = Path(__file__).resolve().parent
P = 'rich_review_wave28_20261015_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def main() -> None:
    selected = [P+name+'.py' for name in ('consume','capture','selected','diagnose','deliver')]
    before = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in selected}
    version = subprocess.run(['/home/administrator/.local/bin/pyright','--version'],capture_output=True,text=True,check=True).stdout.strip()
    command = ['/home/administrator/.local/bin/pyright','--pythonpath',PYTHON,'--outputjson',*selected]
    done = subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    checker = json.loads(done.stdout)
    lsp: Any = language_server(selected)
    structural = {name:[dict(name=node.name,line=node.lineno) for node in ast.walk(ast.parse((ROOT/name).read_text())) if isinstance(node,(ast.FunctionDef,ast.ClassDef))] for name in selected}
    result = dict(kind='STABLE_FINAL_FIVE_SELECTED_FILES_NOT_WHOLE_PROJECT',root=str(ROOT),version=version,
                  interpreter=PYTHON,command=command,natural_exit=done.returncode>=0,returncode=done.returncode,
                  summary=checker.get('summary'),selected_files=selected,checker=checker,stderr=done.stderr,
                  errors_by_file=dict(Counter(Path(d['file']).name for d in checker.get('generalDiagnostics',[]) if d['severity']=='error')),
                  source_SHA256=before,sources_unchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h for name,h in before.items()),
                  LSP=lsp,positive_AST=structural,scope='Five stable new sources only. Rooted installed actual Pyright JSON-RPC; native LSP device unavailable. Frozen helpers not relabelled clean.')
    (ROOT/(P+'language_final.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('returncode','summary','errors_by_file','sources_unchanged')}),flush=True)


if __name__ == '__main__':
    main()
