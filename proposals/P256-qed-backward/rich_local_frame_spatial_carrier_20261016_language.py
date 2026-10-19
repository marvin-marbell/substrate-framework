"""Final selected real-file Pyright, rooted LSP navigation and positive AST."""
import ast
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import threading
import time
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
P='rich_local_frame_spatial_carrier_20261016_'
PY='/home/administrator/substrate-framework/.venv/bin/python'


def lsp():
    child=subprocess.Popen(['/home/administrator/.local/bin/pyright-langserver','--stdio'],cwd=PROJECT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    sin=child.stdin;sout=child.stdout
    assert sin is not None and sout is not None
    messages=queue.Queue();events=[]
    def receive():
        while True:
            headers={}
            while True:
                line=sout.readline()
                if not line:return
                if line==b'\r\n':break
                key,value=line.decode().split(':',1);headers[key.lower()]=value.strip()
            messages.put(json.loads(sout.read(int(headers['content-length']))))
    threading.Thread(target=receive,daemon=True).start()
    def send(method,params,identity=None):
        value=dict(jsonrpc='2.0',method=method,params=params)
        if identity is not None:value['id']=identity
        data=json.dumps(value).encode();sin.write(f'Content-Length: {len(data)}\r\n\r\n'.encode()+data);sin.flush()
    def response(identity):
        deadline=time.time()+90
        while time.time()<deadline:
            value=messages.get(timeout=max(.1,deadline-time.time()));events.append(value)
            if value.get('id')==identity:return value
        raise TimeoutError(identity)
    send('initialize',dict(processId=None,rootUri=PROJECT.as_uri(),capabilities={},initializationOptions=dict(pythonPath=PY)),1);init=response(1);send('initialized',{})
    model=ROOT/(P+'model.py');run=ROOT/(P+'run.py')
    for f in (model,run):send('textDocument/didOpen',dict(textDocument=dict(uri=f.as_uri(),languageId='python',version=1,text=f.read_text())))
    send('textDocument/documentSymbol',dict(textDocument=dict(uri=model.as_uri())),2);symbols=response(2)
    lines=run.read_text().splitlines();i=next(i for i,line in enumerate(lines) if 'm=Model()' in line)
    send('textDocument/definition',dict(textDocument=dict(uri=run.as_uri()),position=dict(line=i,character=lines[i].index('Model')+1)),3);definition=response(3)
    send('textDocument/references',dict(textDocument=dict(uri=run.as_uri()),position=dict(line=i,character=lines[i].index('Model')+1),context=dict(includeDeclaration=True)),4);references=response(4)
    time.sleep(2)
    while not messages.empty():events.append(messages.get())
    send('shutdown',None,5);response(5);send('exit',None);child.wait(timeout=20)
    return dict(root=str(PROJECT),initialize=init,symbols=symbols,definition=definition,references=references,events=events,natural_exit=child.returncode)


def main():
    paths=sorted(ROOT.glob(P+'*.py'))
    command=['/home/administrator/.local/bin/pyright','--pythonpath',PY,'--outputjson',*map(str,paths)]
    child=subprocess.run(command,cwd=PROJECT,capture_output=True,text=True)
    result=dict(command=command,selected_files=list(map(str,paths)),natural_exit=child.returncode,diagnostics=json.loads(child.stdout),stderr=child.stderr,rooted_LSP=lsp(),positive_AST={f.name:[dict(name=n.name,line=n.lineno) for n in ast.walk(ast.parse(f.read_text())) if isinstance(n,(ast.FunctionDef,ast.ClassDef))] for f in paths},source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in paths})
    (ROOT/(P+'language_final.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(exit=child.returncode,summary=result['diagnostics']['summary'],LSP_exit=result['rooted_LSP']['natural_exit'])),flush=True)


if __name__=='__main__':main()
