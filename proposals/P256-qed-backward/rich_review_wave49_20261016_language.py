"""Final selected-file diagnostics and rooted LSP; report failures without claiming green."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from rich_review_wave45_20261016_language import rooted_lsp

ROOT=Path(__file__).resolve().parent;P='rich_review_wave49_20261016_'
PY='/home/administrator/substrate-framework/.venv/bin/python'


def run():
    paths=sorted(ROOT.glob(P+'*.py'))
    command=['/home/administrator/.local/bin/pyright','--pythonpath',PY,'--outputjson',*map(str,paths)]
    child=subprocess.run(command,cwd=ROOT.parents[1],capture_output=True,text=True)
    result=dict(command=command,selected_real_files=list(map(str,paths)),natural_exit=child.returncode,diagnostics=json.loads(child.stdout),stderr=child.stderr,rooted_LSP=rooted_lsp(ROOT/(P+'spatial_consumer.py')),positive_AST={p.name:[dict(name=n.name,line=n.lineno) for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,(ast.FunctionDef,ast.ClassDef))] for p in paths},source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (ROOT/(P+'language_final.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(pyright_exit=child.returncode,summary=result['diagnostics'].get('summary'),LSP_exit=result['rooted_LSP']['natural_exit'])),flush=True)

if __name__=='__main__':run()
