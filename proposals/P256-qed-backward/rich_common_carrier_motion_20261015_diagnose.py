"""Once-stable selected owned Python LSP/AST diagnostics, not a build/test run."""
import ast,hashlib,json,subprocess,time
from pathlib import Path
from rich_review_wave12_20261015_diagnose import language_server
ROOT=Path(__file__).resolve().parent


def main():
    selected=sorted(p.name for p in ROOT.glob('rich_common_carrier_motion_20261015_*.py'));before={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in selected};start=time.monotonic()
    version=subprocess.run(['/home/administrator/.local/bin/pyright','--version'],capture_output=True,text=True,check=True).stdout.strip();record=language_server(selected)
    positive={n:[{'name':node.name,'line':node.lineno} for node in ast.walk(ast.parse((ROOT/n).read_text())) if isinstance(node,(ast.FunctionDef,ast.ClassDef))] for n in selected};errors={}
    for name,response in record['diagnostics'].items():
        items=response.get('result',{}).get('items',[]);errors[name]=sum(item.get('severity')==1 for item in items)
    result={'kind':'ONCE_FINAL_OWNED_SCOPED_LSP_AST_NOT_ASSUMED_CLEAN','root':str(ROOT),'ancestor_project_marker':str(ROOT.parents[1]/'pyproject.toml'),'python':'/home/administrator/substrate-framework/.venv/bin/python','Pyright':version,'native_LSP_device':'unmounted; actual direct protocol initialize/documentSymbol/textDocument-diagnostic/shutdown','source_SHA256_before':before,'source_bytes_unchanged':all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==d for n,d in before.items()),'positive_AST':positive,'LSP':record,'LSP_errors_by_file':errors,'total_LSP_errors':sum(errors.values()),'elapsed_seconds':time.monotonic()-start,'scope':'Only owned new carrier Python files; imported old helpers not re-certified; no builds/tests/linters/formatters or old checker CLI replay.'}
    (ROOT/'rich_common_carrier_motion_20261015_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'version':version,'files':len(selected),'total_LSP_errors':sum(errors.values()),'server_exit':record['server_exit'],'bytes_unchanged':result['source_bytes_unchanged']}),flush=True)
if __name__=='__main__':main()
