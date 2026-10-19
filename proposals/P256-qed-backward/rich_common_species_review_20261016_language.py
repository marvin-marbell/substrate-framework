"""Final selected-file diagnostics and actual rooted source-symbol LSP."""
import ast
import json
from pathlib import Path
import subprocess
from rich_review_wave36_20261016_language import lsp
ROOT=Path(__file__).resolve().parent
P='rich_common_species_review_20261016_'
PY='/home/administrator/substrate-framework/.venv/bin/python'


def main():
    paths=sorted(ROOT.glob(P+'*.py'));command=['/home/administrator/.local/bin/pyright','--pythonpath',PY,'--outputjson',*map(str,paths)]
    child=subprocess.run(command,cwd=ROOT.parents[1],capture_output=True,text=True);diagnostics=json.loads(child.stdout);editor=lsp(ROOT/(P+'mechanism.py'))
    result=dict(command=command,diagnostic_exit=child.returncode,diagnostics=diagnostics,stderr=child.stderr,rooted_LSP=editor,positive_AST={path.name:[dict(name=n.name,line=n.lineno) for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,(ast.FunctionDef,ast.ClassDef))] for path in paths})
    (ROOT/(P+'language_final.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(diagnostic_exit=child.returncode,summary=diagnostics.get('summary'),LSP_exit=editor['natural_exit'])),flush=True)


if __name__=='__main__':main()
