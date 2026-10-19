"""Final scoped diagnostics only, after the already-running physical controls end."""
from pathlib import Path
from typing import Any
import ast,hashlib,inspect,json,subprocess
import rich_common_quantum_em_review_20261016_language as existing

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1];P='rich_review_wave49_20261016_';PY='/home/administrator/substrate-framework/.venv/bin/python'


def main():
    suffixes=['execute','integrated_consumer','integrated_storage','integrated_paired','priority_capture','priority_consumer','priority_storage','priority_observers','priority_raw_audit','priority_transport_raw_audit','priority_stream_current','final_language','final_deliver'];paths=[ROOT/(P+name+'.py') for name in suffixes]
    command=['/home/administrator/.local/bin/pyright','--pythonpath',PY,'--outputjson',*map(str,paths)];child=subprocess.run(command,cwd=PROJECT,capture_output=True,text=True)
    source=inspect.getsource(existing.rooted_lsp).replace("'neutral=frozen.Model('","'source_hash=hashlib.sha256(inspect.getsource(frozen.Model.lab_source)'").replace(".index('Model')",".index('lab_source')");namespace=dict(existing.__dict__);exec(source,namespace)
    result:dict[str,Any]=dict(command=command,selected_actual_files=[str(path) for path in paths],natural_exit=child.returncode,diagnostics=json.loads(child.stdout),stderr=child.stderr,rooted_LSP=namespace['rooted_lsp'](ROOT/(P+'priority_stream_current.py')),positive_AST={path.name:[dict(name=node.name,line=node.lineno) for node in ast.walk(ast.parse(path.read_text())) if isinstance(node,(ast.FunctionDef,ast.ClassDef))] for path in paths},source_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},scope='Final affected own helper/caller/measurement sources. Includes historical unexercised branches for diagnostics, NOT physical approval. No K, convergence, new dynamics or source/field campaign.')
    (ROOT/(P+'final_language_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(natural_exit=child.returncode,summary=result['diagnostics'].get('summary'),rooted_LSP_exit=result['rooted_LSP']['natural_exit'],server_diagnostics_events=len(result['rooted_LSP']['publish_diagnostics']))),flush=True)

if __name__=='__main__':main()
