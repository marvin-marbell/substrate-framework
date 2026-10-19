"""Once-stable explicitly scoped final Pyright/LSP/AST receipt, no suppression."""
import ast,hashlib,json,subprocess
from pathlib import Path
from collections import Counter
from rich_review_wave12_20261015_diagnose import language_server
ROOT=Path(__file__).resolve().parent
FILES=['rich_frame_field_join_20261015_model.py','rich_frame_field_join_20261015_calculate.py','rich_frame_field_join_20261015_response.py','rich_frame_field_join_20261015_capture.py','rich_frame_field_join_20261015_language.py','rich_frame_field_join_20261015_summarize.py','rich_frame_field_join_20261015_diagnose.py']
def digest(name):return hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
def main():
    before={n:digest(n) for n in FILES};version=subprocess.run(['/home/administrator/.local/bin/pyright','--version'],capture_output=True,text=True,check=True).stdout.strip();command=['/home/administrator/.local/bin/pyright','--pythonpath','/home/administrator/substrate-framework/.venv/bin/python','--outputjson',*FILES];process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True);raw=json.loads(process.stdout);errors=Counter(x['file'] for x in raw.get('generalDiagnostics',[]) if x['severity']=='error');lsp=language_server(FILES)
    record={'version':version,'root':str(ROOT),'command':command,'returncode':process.returncode,'reported_summary':raw.get('summary'),'errors_by_file':dict(errors),'full_checker_output':raw,'stderr':process.stderr,'LSP':lsp,'positive_AST':{n:[{'kind':type(x).__name__,'name':x.name,'line':x.lineno} for x in ast.walk(ast.parse((ROOT/n).read_text())) if isinstance(x,(ast.FunctionDef,ast.ClassDef))] for n in FILES},'before':before,'after':{n:digest(n) for n in FILES},'scope':'Exactly seven new stable active Python files, no frozen supplier recheck, no formatter/build/test, no suppression.'};record['sources_unchanged']=record['before']==record['after'];(ROOT/'rich_frame_field_join_20261015_diagnostics.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:record[k] for k in ['version','returncode','reported_summary','errors_by_file','sources_unchanged']}))
if __name__=='__main__':main()
