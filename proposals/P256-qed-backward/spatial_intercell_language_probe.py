"""Project-root Pyright JSON-RPC symbol/reference probe; no diagnostics run."""
import json
import pathlib
import subprocess
import threading
import queue
import time
ROOT=pathlib.Path(__file__).resolve().parent
p=subprocess.Popen(['/home/administrator/.local/bin/pyright-langserver','--stdio'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=ROOT)
q=queue.Queue()
def receive():
    while True:
        headers={}
        while True:
            line=p.stdout.readline()
            if not line:return
            if line==b'\r\n':break
            key,value=line.decode().split(':',1);headers[key.lower()]=value.strip()
        q.put(json.loads(p.stdout.read(int(headers['content-length']))))
threading.Thread(target=receive,daemon=True).start()
def send(obj):
    b=json.dumps(obj).encode();p.stdin.write(f'Content-Length: {len(b)}\r\n\r\n'.encode()+b);p.stdin.flush()
def request(i,method,params):
    send(dict(jsonrpc='2.0',id=i,method=method,params=params))
    deadline=time.monotonic()+60
    while time.monotonic()<deadline:
        obj=q.get(timeout=60)
        if 'method' in obj and 'id' in obj:
            send(dict(jsonrpc='2.0',id=obj['id'],result=[]))
        if obj.get('id')==i:return obj
    raise TimeoutError(method)
source=ROOT/'joined_phi_u1_spatial_linked_source.py'
init=request(1,'initialize',{'processId':None,'rootUri':ROOT.as_uri(),'capabilities':{'textDocument':{'documentSymbol':{'hierarchicalDocumentSymbolSupport':True}}}})
send(dict(jsonrpc='2.0',method='initialized',params={}))
send(dict(jsonrpc='2.0',method='textDocument/didOpen',params={'textDocument':{'uri':source.as_uri(),'languageId':'python','version':1,'text':source.read_text()}}))
symbols=request(2,'textDocument/documentSymbol',{'textDocument':{'uri':source.as_uri()}})
references=request(3,'textDocument/references',{'textDocument':{'uri':source.as_uri()},'position':{'line':69,'character':5},'context':{'includeDeclaration':True}})
result={'root':str(ROOT),'binary':'/home/administrator/.local/bin/pyright-langserver','initialize':init,'symbols':symbols,'references_cell':references,'diagnostics':'Forbidden by scoped assignment; not requested.'}
(ROOT/'spatial_intercell_language_actual_20261010.json').write_text(json.dumps(result,indent=2)+'\n')
assert symbols.get('result') and references.get('result')
request(4,'shutdown',None);send(dict(jsonrpc='2.0',method='exit',params=None));p.wait(timeout=10)
print(json.dumps({'symbols':len(symbols['result']),'cell_references':len(references['result']),'exit':p.returncode}))
