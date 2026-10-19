"""Final actual-file diagnostics, reusing the installed rooted LSP client read-only."""
from pathlib import Path
from typing import Any
import ast,hashlib,inspect,json,subprocess
import rich_common_quantum_em_review_20261016_language as existing

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1];P='rich_full108_recoil_review_20261017_';PY='/home/administrator/substrate-framework/.venv/bin/python'


def main():
    paths=sorted(ROOT.glob(P+'*.py'));command=['/home/administrator/.local/bin/pyright','--pythonpath',PY,'--outputjson',*map(str,paths)];child=subprocess.run(command,cwd=PROJECT,capture_output=True,text=True)
    source=inspect.getsource(existing.rooted_lsp).replace("'neutral=frozen.Model('","'one=native.single_particle('").replace(".index('Model')",".index('single_particle')");namespace=dict(existing.__dict__);exec(source,namespace)
    result:dict[str,Any]=dict(command=command,selected_actual_files=[str(path) for path in paths],natural_exit=child.returncode,diagnostics=json.loads(child.stdout),stderr=child.stderr,rooted_LSP=namespace['rooted_lsp'](ROOT/(P+'run.py')),positive_AST={path.name:[dict(name=node.name,line=node.lineno) for node in ast.walk(ast.parse(path.read_text())) if isinstance(node,(ast.FunctionDef,ast.ClassDef))] for path in paths},source_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},scope='Selected new independent low-order consumer only; no reapproval or rerun of old/author dynamics. Existing client adapted in memory, old sources untouched.')
    (ROOT/(P+'language_final.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(pyright_natural_exit=child.returncode,summary=result['diagnostics'].get('summary'),rooted_LSP_natural_exit=result['rooted_LSP']['natural_exit'],LSP_server_diagnostic_events=len(result['rooted_LSP']['publish_diagnostics']))),flush=True)

if __name__=='__main__':main()
