"""Once-final selected owned Python diagnostics, actual rooted LSP and positive AST."""
from __future__ import annotations
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from rich_review_wave12_20261015_diagnose import language_server

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_frame_gauss_join_20261015_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def main() -> None:
    selected = [PREFIX+suffix+'.py' for suffix in ['model', 'consume', 'capture', 'language']]
    before = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in selected}
    command = ['/home/administrator/.local/bin/pyright', '--pythonpath', PYTHON, '--outputjson', *selected]
    checked = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    checker = json.loads(checked.stdout)
    lsp = language_server(selected)
    structural = {name:[dict(name=n.name, line=n.lineno) for n in ast.walk(ast.parse((ROOT/name).read_text())) if isinstance(n, (ast.FunctionDef, ast.ClassDef))] for name in selected}
    result = dict(kind='ONCE_FINAL_OWNED_SCOPE_NOT_PROJECT_CLEAN', root=str(ROOT), scientific_interpreter=PYTHON,
                  version=subprocess.run(['/home/administrator/.local/bin/pyright', '--version'], capture_output=True, text=True, check=True).stdout.strip(),
                  command=command, natural_exit=True, returncode=checked.returncode, checker=checker, stderr=checked.stderr,
                  summary=checker.get('summary'), errors_by_file=dict(Counter(Path(d['file']).name for d in checker.get('generalDiagnostics', []) if d['severity']=='error')),
                  selected_files=selected, source_SHA256=before, sources_unchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value for name, value in before.items()),
                  rooted_LSP=lsp, positive_AST=structural,
                  scope='Four owned sources only; no old source type-clean claim. LSP device unavailable, actual installed direct Pyright JSON-RPC with project root. No builds, tests, formatters, or ancestor producer/checker replay.')
    (ROOT/(PREFIX+'language_final.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(returncode=checked.returncode, summary=result['summary'], errors_by_file=result['errors_by_file'], LSP_exit=lsp['server_exit'], sources_unchanged=result['sources_unchanged'])), flush=True)


if __name__ == '__main__':
    main()
