"""Rooted real-file language-server diagnostics and selected actual-file Pyright."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
import rich_coupled_frame_photon_20261016_language as ancestor
ROOT=Path(__file__).resolve().parent
P='rich_spatial_frame_photon_20261016_'
PY='/home/administrator/substrate-framework/.venv/bin/python'


def main():
    paths=sorted(ROOT.glob(P+'*.py'))
    command=['/home/administrator/.local/bin/pyright','--pythonpath',PY,'--outputjson',*map(str,paths)]
    child=subprocess.run(command,cwd=ROOT.parents[1],capture_output=True,text=True)
    result: dict[str,Any]=dict(command=command,selected_files=list(map(str,paths)),natural_exit=child.returncode,diagnostics=json.loads(child.stdout),stderr=child.stderr,rooted_LSP=ancestor.rooted_lsp(ROOT/(P+'model.py')),positive_AST={f.name:[dict(name=n.name,line=n.lineno) for n in ast.walk(ast.parse(f.read_text())) if isinstance(n,(ast.FunctionDef,ast.ClassDef))] for f in paths},source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in paths})
    (ROOT/(P+'language_final.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(exit=child.returncode,summary=result['diagnostics']['summary'],LSP_exit=result['rooted_LSP']['natural_exit'])),flush=True)
    sys.exit(child.returncode)


if __name__=='__main__':main()
