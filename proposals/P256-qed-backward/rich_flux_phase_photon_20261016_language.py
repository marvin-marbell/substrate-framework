"""Final selected Pyright, rooted existing LSP transport and positive Python AST."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from rich_bulk_photon_20261016_language import rooted_lsp
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
P='rich_flux_phase_photon_20261016_'
PY='/home/administrator/substrate-framework/.venv/bin/python'


def main():
    paths=sorted(ROOT.glob(P+'*.py'))
    command=['/home/administrator/.local/bin/pyright','--pythonpath',PY,'--outputjson',*map(str,paths)]
    child=subprocess.run(command,cwd=PROJECT,capture_output=True,text=True)
    lsp=rooted_lsp(ROOT/(P+'run.py'))
    result=dict(command=command,natural_exit=child.returncode,diagnostics=json.loads(child.stdout),stderr=child.stderr,rooted_LSP=lsp,positive_AST={path.name:[dict(name=node.name,line=node.lineno) for node in ast.walk(ast.parse(path.read_text())) if isinstance(node,(ast.FunctionDef,ast.ClassDef))] for path in paths},source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},pre_edit_LSP='artifact://1594; root /home/administrator/substrate-222; existing bulk run symbols and s.kernel definition',pre_edit_AST='xd://ast_grep positive evaluate function on pure imported bulk state')
    (ROOT/(P+'language_final.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(natural_exit=child.returncode,summary=result['diagnostics']['summary'],LSP_exit=lsp['natural_exit'])),flush=True)


if __name__=='__main__':main()
