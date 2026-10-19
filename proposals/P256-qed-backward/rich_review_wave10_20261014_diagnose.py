"""ONCE new stable Wave10 source scope; no earlier-file checker ritual."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import threading
import time

ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'
PREFIX='rich_review_wave10_20261014_diagnostics'


def language_server(selected):
    process=subprocess.Popen(['/home/administrator/.local/bin/pyright-langserver','--stdio'],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    if process.stdin is None or process.stdout is None:
        raise RuntimeError('Missing LSP streams')
    incoming=queue.Queue();stdin=process.stdin;stdout=process.stdout
    def reader():
        while True:
            line=stdout.readline()
            if not line:
                return
            if line.lower().startswith(b'content-length:'):
                size=int(line.split(b':')[1]);stdout.readline();incoming.put(json.loads(stdout.read(size)))
    threading.Thread(target=reader,daemon=True).start()
    def send(value):
        body=json.dumps(value).encode();stdin.write(f'Content-Length: {len(body)}\r\n\r\n'.encode()+body);stdin.flush()
    def response(identifier):
        while True:
            value=incoming.get(timeout=120)
            if value.get('method')=='workspace/configuration':
                send({'jsonrpc':'2.0','id':value['id'],'result':[{'pythonPath':PYTHON} for _ in value['params']['items']]})
            if value.get('id')==identifier:
                return value
    send({'jsonrpc':'2.0','id':1,'method':'initialize','params':{'processId':None,'rootUri':ROOT.as_uri(),'capabilities':{'workspace':{'configuration':True}}}})
    initialization=response(1);send({'jsonrpc':'2.0','method':'initialized','params':{}})
    identifier=2;symbols={};diagnostics={}
    for name in selected:
        path=ROOT/name
        send({'jsonrpc':'2.0','method':'textDocument/didOpen','params':{'textDocument':{'uri':path.as_uri(),'languageId':'python','version':1,'text':path.read_text()}}})
        send({'jsonrpc':'2.0','id':identifier,'method':'textDocument/documentSymbol','params':{'textDocument':{'uri':path.as_uri()}}});symbols[name]=response(identifier);identifier+=1
        send({'jsonrpc':'2.0','id':identifier,'method':'textDocument/diagnostic','params':{'textDocument':{'uri':path.as_uri()}}});diagnostics[name]=response(identifier);identifier+=1
    send({'jsonrpc':'2.0','id':identifier,'method':'shutdown','params':None});response(identifier)
    send({'jsonrpc':'2.0','method':'exit','params':None});process.wait(timeout=20)
    return dict(initialization=initialization,selected_files=selected,symbols=symbols,diagnostics=diagnostics,server_exit=process.returncode,native_device='unmounted; actual direct protocol')


def main():
    data=json.loads((ROOT/(PREFIX+'_input.json')).read_text());selected=data['selected_files']
    if not selected or any(not (ROOT/name).is_file() for name in selected):
        raise ValueError('Explicit existing selected files required')
    before={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in selected}
    version=subprocess.run(['/home/administrator/.local/bin/pyright','--version'],capture_output=True,text=True,check=True).stdout.strip()
    command=['/home/administrator/.local/bin/pyright','--pythonpath',PYTHON,'--outputjson',*selected]
    start=time.monotonic();done=subprocess.run(command,cwd=ROOT,capture_output=True,text=True);checker=json.loads(done.stdout)
    lsp=language_server(data['lsp_files'])
    structural={name:[dict(name=node.name,line=node.lineno) for node in ast.walk(ast.parse((ROOT/name).read_text())) if isinstance(node,(ast.ClassDef,ast.FunctionDef))] for name in data['lsp_files']}
    artifact=dict(kind='ONCE_FINAL_NEW_WAVE10_SCOPED_DIAGNOSTICS_NOT_ALL_CLEAN',root=str(ROOT),version=version,command=command,
                  selected_files=selected,source_SHA256_before=before,sources_unchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value for name,value in before.items()),
                  natural_exit=True,returncode=done.returncode,elapsed_seconds=time.monotonic()-start,reported_summary=checker.get('summary'),
                  errors_by_file=dict(Counter(Path(item['file']).name for item in checker.get('generalDiagnostics',[]) if item['severity']=='error')),
                  checker=checker,stderr=done.stderr,LSP=lsp,positive_AST=structural,
                  scope='New GradedMirrorVector/ColourSeaExcitation/owned Wave10 Python files ONLY. Older unchanged frame/colour/sea helpers not explicitly rechecked or inherited as clean.')
    (ROOT/(PREFIX+'.json')).write_text(json.dumps(artifact,indent=2)+'\n')
    print(json.dumps({key:artifact[key] for key in ('version','returncode','reported_summary','errors_by_file','sources_unchanged')},indent=2))


if __name__=='__main__':
    main()
