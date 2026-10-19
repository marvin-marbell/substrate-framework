"""One final selected-file diagnostics pass; rooted actual JSON-RPC Pyright."""
import ast
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import threading
import time
ROOT=Path(__file__).resolve().parent
P='rich_periodic_tensor_20261015_'
PROJECT=ROOT.parents[1]
PYTHON=str(ROOT/(P+'env/bin/python'))


def lsp(path):
    proc=subprocess.Popen(['/home/administrator/.local/bin/pyright-langserver','--stdio'],cwd=PROJECT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    messages=queue.Queue()
    def receive():
        while True:
            headers={}
            while True:
                line=proc.stdout.readline()
                if not line:return
                if line==b'\r\n':break
                k,v=line.decode().split(':',1);headers[k.lower()]=v.strip()
            messages.put(json.loads(proc.stdout.read(int(headers['content-length']))))
    threading.Thread(target=receive,daemon=True).start()
    events=[]
    def send(method,params,identity=None):
        value=dict(jsonrpc='2.0',method=method,params=params)
        if identity is not None:value['id']=identity
        raw=json.dumps(value).encode();proc.stdin.write(f'Content-Length: {len(raw)}\r\n\r\n'.encode()+raw);proc.stdin.flush()
    def response(identity):
        deadline=time.time()+90
        while time.time()<deadline:
            value=messages.get(timeout=max(.1,deadline-time.time()));events.append(value)
            if value.get('id')==identity:return value
        raise TimeoutError(identity)
    send('initialize',dict(processId=None,rootUri=PROJECT.as_uri(),capabilities={},initializationOptions=dict(pythonPath=PYTHON)),1)
    initialized=response(1);send('initialized',{})
    send('textDocument/didOpen',dict(textDocument=dict(uri=path.as_uri(),languageId='python',version=1,text=path.read_text())))
    send('textDocument/documentSymbol',dict(textDocument=dict(uri=path.as_uri())),2);symbols=response(2)
    lines=path.read_text().splitlines();line=next(i for i,s in enumerate(lines) if 'model=GaugeTensor()' in s)
    send('textDocument/definition',dict(textDocument=dict(uri=path.as_uri()),position=dict(line=line,character=lines[line].index('GaugeTensor')+1)),3);definition=response(3)
    time.sleep(2)
    while not messages.empty():events.append(messages.get())
    send('shutdown',None,4);response(4);send('exit',None);proc.wait(timeout=20)
    return {'root':str(PROJECT),'initialize':initialized,'symbols':symbols,'definition':definition,'events':events,'natural_exit':proc.returncode}


if __name__=='__main__':
    selected=[str(p) for p in ROOT.glob(P+'*.py')]
    hashes={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in selected}
    r=subprocess.run(['/home/administrator/.local/bin/pyright','--pythonpath',PYTHON,'--outputjson',*selected],cwd=PROJECT,capture_output=True,text=True)
    result={'selected_source_SHA256':hashes,'diagnostic_command_exit':r.returncode,'diagnostics':json.loads(r.stdout),'stderr':r.stderr,'rooted_LSP':lsp(ROOT/(P+'run.py')),'positive_AST':{s:[{'name':n.name,'line':n.lineno} for n in ast.walk(ast.parse(Path(s).read_text())) if isinstance(n,(ast.FunctionDef,ast.ClassDef))] for s in selected},'sources_unchanged':all(hashlib.sha256(Path(s).read_bytes()).hexdigest()==h for s,h in hashes.items())}
    (ROOT/(P+'language_final.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'diagnostic_exit':r.returncode,'summary':result['diagnostics'].get('summary'),'LSP_exit':result['rooted_LSP']['natural_exit'],'definition':result['rooted_LSP']['definition']}))
