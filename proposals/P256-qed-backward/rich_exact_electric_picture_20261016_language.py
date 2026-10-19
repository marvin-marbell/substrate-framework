"""Real selected source diagnostics, rooted LSP navigation and positive AST."""
import ast
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
P='rich_exact_electric_picture_20261016_'
PY='/home/administrator/substrate-framework/.venv/bin/python'


def rooted_lsp(path):
    child=subprocess.Popen(['/home/administrator/.local/bin/pyright-langserver','--stdio'],cwd=PROJECT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdin=child.stdin;stdout=child.stdout
    assert stdin is not None and stdout is not None
    messages=queue.Queue();events=[]
    def receive():
        while True:
            headers={}
            while True:
                line=stdout.readline()
                if not line:return
                if line==b'\r\n':break
                key,value=line.decode().split(':',1);headers[key.lower()]=value.strip()
            messages.put(json.loads(stdout.read(int(headers['content-length']))))
    threading.Thread(target=receive,daemon=True).start()
    def send(method,params,identity=None):
        value=dict(jsonrpc='2.0',method=method,params=params)
        if identity is not None:value['id']=identity
        data=json.dumps(value).encode();stdin.write(f'Content-Length: {len(data)}\r\n\r\n'.encode()+data);stdin.flush()
    def response(identity):
        deadline=time.time()+90
        while time.time()<deadline:
            value=messages.get(timeout=max(.1,deadline-time.time()));events.append(value)
            if value.get('id')==identity:return value
        raise TimeoutError(identity)
    send('initialize',dict(processId=None,rootUri=PROJECT.as_uri(),capabilities={},initializationOptions=dict(pythonPath=PY)),1)
    initialize=response(1);send('initialized',{})
    send('textDocument/didOpen',dict(textDocument=dict(uri=path.as_uri(),languageId='python',version=1,text=path.read_text())))
    send('textDocument/documentSymbol',dict(textDocument=dict(uri=path.as_uri())),2);symbols=response(2)
    lines=path.read_text().splitlines();line=next(i for i,text in enumerate(lines) if 'self.fixed=f=frozen.Model' in text)
    send('textDocument/definition',dict(textDocument=dict(uri=path.as_uri()),position=dict(line=line,character=lines[line].index('Model')+1)),3);definition=response(3)
    time.sleep(2)
    while not messages.empty():events.append(messages.get())
    send('shutdown',None,4);response(4);send('exit',None);child.wait(timeout=20)
    return dict(root=str(PROJECT),initialize=initialize,symbols=symbols,definition=definition,events=events,natural_exit=child.returncode)


def main():
    files=sorted(ROOT.glob(P+'*.py'));command=['/home/administrator/.local/bin/pyright','--pythonpath',PY,'--outputjson',*map(str,files)]
    child=subprocess.run(command,cwd=PROJECT,capture_output=True,text=True)
    result: dict[str,Any]=dict(command=command,selected_files=list(map(str,files)),natural_exit=child.returncode,diagnostics=json.loads(child.stdout),stderr=child.stderr,rooted_LSP=rooted_lsp(ROOT/(P+'model.py')),positive_AST={f.name:[dict(name=node.name,line=node.lineno) for node in ast.walk(ast.parse(f.read_text())) if isinstance(node,(ast.ClassDef,ast.FunctionDef))] for f in files},source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (ROOT/(P+'language_final.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(exit=child.returncode,summary=result['diagnostics']['summary'],LSP_exit=result['rooted_LSP']['natural_exit'])),flush=True)
    sys.exit(child.returncode)


if __name__=='__main__':main()
