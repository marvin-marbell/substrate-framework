"""Once-final selected diagnostics and actual rooted LSP, never ancestor replay."""
from __future__ import annotations
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from rich_review_wave12_20261015_diagnose import language_server

ROOT=Path(__file__).resolve().parent
P='rich_dressed_local_carrier_20261015_'


def main():
    selected=[P+s+'.py' for s in ['source','prepare','capture','dynamics','language','deliver']]
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in selected}
    command=['/home/administrator/.local/bin/pyright','--pythonpath','/home/administrator/substrate-framework/.venv/bin/python','--outputjson',*selected]
    done=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    checker=json.loads(done.stdout)
    lsp=language_server([P+'source.py',P+'dynamics.py'])
    structural={n:[dict(name=a.name,line=a.lineno) for a in ast.walk(ast.parse((ROOT/n).read_text())) if isinstance(a,(ast.FunctionDef,ast.ClassDef))] for n in selected}
    result=dict(kind='ONCE_FINAL_OWNED_SCOPE_ONLY',selected=selected,command=command,
        natural_exit=True,returncode=done.returncode,checker=checker,stderr=done.stderr,
        summary=checker.get('summary'),errors_by_file=dict(Counter(Path(d['file']).name for d in checker.get('generalDiagnostics',[]) if d['severity']=='error')),
        rooted_LSP=lsp,positive_AST=structural,source_SHA256=hashes,
        sources_unchanged=all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v for n,v in hashes.items()))
    (ROOT/(P+'language_final.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(returncode=done.returncode,summary=result['summary'],errors_by_file=result['errors_by_file'],LSP_exit=lsp['server_exit'])),flush=True)


if __name__=='__main__':
    main()
