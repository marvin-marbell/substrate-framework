"""One final selected-source diagnostics pass, separate from the scientific runtime."""
from __future__ import annotations
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from rich_review_wave12_20261015_diagnose import language_server

ROOT = Path(__file__).resolve().parent
P = 'rich_review_wave32_20261015_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def main() -> None:
    selected = [P+name+'.py' for name in ['observe','consume','capture','diagnose','deliver']]
    hashes = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in selected}
    command = ['/home/administrator/.local/bin/pyright','--pythonpath',PYTHON,'--outputjson',*selected]
    completed = subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    checker = json.loads(completed.stdout)
    rooted = language_server(selected)
    structural = {name:[dict(name=node.name,line=node.lineno) for node in ast.walk(ast.parse((ROOT/name).read_text()))
                        if isinstance(node,(ast.FunctionDef,ast.ClassDef))] for name in selected}
    result = dict(kind='ONCE_FINAL_FIVE_OWNED_SOURCE_DIAGNOSTICS',root=str(ROOT),command=command,
        version=subprocess.run(['/home/administrator/.local/bin/pyright','--version'],capture_output=True,text=True,check=True).stdout.strip(),
        source_SHA256=hashes,natural_exit=True,returncode=completed.returncode,checker=checker,stderr=completed.stderr,
        summary=checker.get('summary'),error_counts_by_file=dict(Counter(Path(item['file']).name for item in checker.get('generalDiagnostics',[]) if item['severity']=='error')),
        rooted_LSP=rooted,positive_AST=structural,
        sources_unchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value for name,value in hashes.items()),
        scope='Explicit real owned sources; no ancestor checker/producer CLI replay, no native-device claim. xd LSP unmounted; installed rooted direct Pyright JSON-RPC exercised.')
    (ROOT/(P+'language_final.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(returncode=completed.returncode,summary=result['summary'],errors=result['error_counts_by_file'],LSP_exit=rooted['server_exit'])),flush=True)


if __name__ == '__main__':
    main()
