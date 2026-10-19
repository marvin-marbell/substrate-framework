"""Direct selected-root language navigation; no producer loads."""
import ast
import json
from pathlib import Path
import queue
import subprocess
import threading
ROOT = Path(__file__).resolve().parent
source = ROOT / 'rich_frame_holonomy_20261014_model.py'
p = subprocess.Popen(['/home/administrator/.local/bin/pyright-langserver', '--stdio'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=ROOT)
messages = queue.Queue()
def reader():
    while True:
        line = p.stdout.readline()
        if not line:
            return
        if line.lower().startswith(b'content-length:'):
            size = int(line.split(b':')[1]); p.stdout.readline()
            messages.put(json.loads(p.stdout.read(size)))
threading.Thread(target=reader, daemon=True).start()
def send(obj):
    data = json.dumps(obj).encode()
    p.stdin.write(f'Content-Length: {len(data)}\r\n\r\n'.encode() + data); p.stdin.flush()
def response(identifier):
    while True:
        obj = messages.get(timeout=40)
        if obj.get('method') == 'workspace/configuration':
            send({'jsonrpc':'2.0','id':obj['id'],'result':[{'pythonPath':'/home/administrator/substrate-framework/.venv/bin/python'} for _ in obj['params']['items']]})
        if obj.get('id') == identifier and ('result' in obj or 'error' in obj):
            return obj
send({'jsonrpc':'2.0','id':1,'method':'initialize','params':{'processId':None,'rootUri':ROOT.as_uri(),'capabilities':{'workspace':{'configuration':True}}}})
initial = response(1)
send({'jsonrpc':'2.0','method':'initialized','params':{}})
send({'jsonrpc':'2.0','method':'textDocument/didOpen','params':{'textDocument':{'uri':source.as_uri(),'languageId':'python','version':1,'text':source.read_text()}}})
answers = {}
for identifier, method, extra in [(2,'textDocument/documentSymbol',{}),(3,'textDocument/definition',{'position':{'line':33,'character':26}})]:
    send({'jsonrpc':'2.0','id':identifier,'method':method,'params':{'textDocument':{'uri':source.as_uri()},**extra}})
    answers[method] = response(identifier)
send({'jsonrpc':'2.0','id':4,'method':'shutdown','params':None}); response(4)
send({'jsonrpc':'2.0','method':'exit','params':None}); p.wait(timeout=10)
positive = [{'name':n.name,'line':n.lineno} for n in ast.walk(ast.parse(source.read_text())) if isinstance(n,ast.FunctionDef) and n.name == 'raw_frame']
result = {'root':str(ROOT),'native_lsp_device_unavailable':True,'server':'pyright-langserver','initialize':initial,'navigation':answers,'positive_AST':positive,'diagnostics':'Final selected diagnostics owned by next-wave reviewer; no midflight checks.'}
(ROOT/'rich_graded_mirror_20261014_language.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'positive_AST':positive,'navigation_success':{k:bool(v.get('result')) for k,v in answers.items()}}))
