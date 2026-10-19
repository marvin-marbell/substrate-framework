"""Final selected diagnostics for new owning sources, never project-clean claim."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from rich_review_wave12_20261015_diagnose import language_server
ROOT=Path(__file__).resolve().parent
P='rich_open_gauss_carrier_20261015_'
def main():
    names=[P+n+'.py' for n in ('model','reference','charged','capture','language','deliver')]
    done=subprocess.run(['/home/administrator/.local/bin/pyright','--pythonpath','/home/administrator/substrate-framework/.venv/bin/python','--outputjson',*names],cwd=ROOT,capture_output=True,text=True)
    report=dict(scope='Only six new owning sources; frozen helper bytes preserved even if type diagnostics remain',returncode=done.returncode,checker=json.loads(done.stdout),stderr=done.stderr,LSP=language_server([P+'model.py',P+'charged.py']),positive_AST={n:[node.name for node in ast.walk(ast.parse((ROOT/n).read_text())) if isinstance(node,(ast.FunctionDef,ast.ClassDef))] for n in names},source_SHA256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names})
    (ROOT/(P+'language_final.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(returncode=done.returncode,summary=report['checker'].get('summary'),LSP_exit=report['LSP']['server_exit'])),flush=True)
if __name__=='__main__':
    main()
