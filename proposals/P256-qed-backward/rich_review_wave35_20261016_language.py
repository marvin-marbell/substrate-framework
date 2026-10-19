"""Selected interpreter-matched diagnostics, rooted LSP and positive Python AST."""
import ast
import json
from pathlib import Path
import queue
import subprocess
import threading
import time
from typing import Any
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
P='rich_review_wave35_20261016_'
PY='/home/administrator/substrate-framework/.venv/bin/python'


def lsp(path):
    proc=subprocess.Popen(['/home/administrator/.local/bin/pyright-langserver','--stdio'],cwd=PROJECT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stream=proc.stdout;sink=proc.stdin;assert stream is not None and sink is not None
    messages=queue.Queue();events=[]
    def receive():
        while True:
            headers={}
            while True:
                line=stream.readline()
                if not line:return
                if line==b'\r\n':break
                k,v=line.decode().split(':',1);headers[k.lower()]=v.strip()
            messages.put(json.loads(stream.read(int(headers['content-length']))))
    threading.Thread(target=receive,daemon=True).start()
    def send(method,params,identity=None):
        value=dict(jsonrpc='2.0',method=method,params=params)
        if identity is not None:value['id']=identity
        raw=json.dumps(value).encode();sink.write(f'Content-Length: {len(raw)}\r\n\r\n'.encode()+raw);sink.flush()
    def response(identity):
        deadline=time.time()+90
        while time.time()<deadline:
            value=messages.get(timeout=max(.1,deadline-time.time()));events.append(value)
            if value.get('id')==identity:return value
        raise TimeoutError(identity)
    send('initialize',dict(processId=None,rootUri=PROJECT.as_uri(),capabilities={},initializationOptions=dict(pythonPath=PY)),1)
    initialized=response(1);send('initialized',{})
    send('workspace/didChangeConfiguration',dict(settings=dict(python=dict(pythonPath=PY,analysis=dict(extraPaths=[str(ROOT)])))))
    send('textDocument/didOpen',dict(textDocument=dict(uri=path.as_uri(),languageId='python',version=1,text=path.read_text())))
    send('textDocument/documentSymbol',dict(textDocument=dict(uri=path.as_uri())),2);symbols=response(2)
    lines=path.read_text().splitlines();line=next(i for i,s in enumerate(lines) if 'out=contract(' in s)
    send('textDocument/definition',dict(textDocument=dict(uri=path.as_uri()),position=dict(line=line,character=lines[line].index('contract')+1)),3);definition=response(3)
    time.sleep(2)
    while not messages.empty():events.append(messages.get())
    send('shutdown',None,4);response(4);send('exit',None);proc.wait(timeout=20)
    return dict(root=str(PROJECT),initialize=initialized,symbols=symbols,definition=definition,events=events,natural_exit=proc.returncode)


def main():
    paths=sorted(ROOT.glob(P+'*.py'))
    command=['/home/administrator/.local/bin/pyright','--pythonpath',PY,'--outputjson',*map(str,paths)]
    child=subprocess.run(command,cwd=PROJECT,capture_output=True,text=True)
    value: dict[str,Any]
    value=dict(command=command,natural_exit=child.returncode,diagnostics=json.loads(child.stdout),stderr=child.stderr,rooted_LSP=lsp(ROOT/(P+'consume.py')),positive_AST={p.name:[dict(name=n.name,line=n.lineno) for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,(ast.FunctionDef,ast.ClassDef))] for p in paths})
    (ROOT/(P+'language_final.json')).write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(dict(diagnostic_exit=child.returncode,summary=value['diagnostics'].get('summary'),LSP_exit=value['rooted_LSP']['natural_exit'])),flush=True)

if __name__=='__main__':main()
