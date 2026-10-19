"""Final bounded diagnostics of actual prefix-owned files and rooted JSON-RPC LSP."""
import ast
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import threading
import time
from typing import Any
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
P='rich_gauss_projected_photon_20261016_'
PY='/home/administrator/substrate-framework/.venv/bin/python'


def lsp(path) -> dict[str,Any]:
    proc=subprocess.Popen(['/home/administrator/.local/bin/pyright-langserver','--stdio'],cwd=PROJECT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout=proc.stdout;stdin=proc.stdin
    assert stdout is not None and stdin is not None
    messages=queue.Queue()
    def receive():
        while True:
            headers={}
            while True:
                line=stdout.readline()
                if not line:return
                if line==b'\r\n':break
                k,v=line.decode().split(':',1);headers[k.lower()]=v.strip()
            messages.put(json.loads(stdout.read(int(headers['content-length']))))
    threading.Thread(target=receive,daemon=True).start();events=[]
    def send(method,params,identity=None):
        value=dict(jsonrpc='2.0',method=method,params=params)
        if identity is not None:value['id']=identity
        raw=json.dumps(value).encode();stdin.write(f'Content-Length: {len(raw)}\r\n\r\n'.encode()+raw);stdin.flush()
    def response(identity):
        deadline=time.time()+90
        while time.time()<deadline:
            value=messages.get(timeout=max(.1,deadline-time.time()));events.append(value)
            if value.get('id')==identity:return value
        raise TimeoutError(identity)
    send('initialize',dict(processId=None,rootUri=PROJECT.as_uri(),capabilities={},initializationOptions=dict(pythonPath=PY)),1)
    initialized=response(1);send('initialized',{})
    send('textDocument/didOpen',dict(textDocument=dict(uri=path.as_uri(),languageId='python',version=1,text=path.read_text())))
    send('textDocument/documentSymbol',dict(textDocument=dict(uri=path.as_uri())),2);symbols=response(2)
    lines=path.read_text().splitlines();line=next(i for i,text in enumerate(lines) if 'table=s.pair_table' in text)
    send('textDocument/definition',dict(textDocument=dict(uri=path.as_uri()),position=dict(line=line,character=lines[line].index('pair_table')+1)),3)
    definition=response(3);time.sleep(2)
    while not messages.empty():events.append(messages.get())
    send('shutdown',None,4);response(4);send('exit',None);proc.wait(timeout=20)
    return dict(root=str(PROJECT),initialize=initialized,symbols=symbols,definition=definition,events=events,natural_exit=proc.returncode)


def main():
    paths=sorted(ROOT.glob(P+'*.py'));hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    command=['/home/administrator/.local/bin/pyright','--pythonpath',PY,'--outputjson',*map(str,paths)]
    child=subprocess.run(command,cwd=PROJECT,capture_output=True,text=True)
    value:dict[str,Any]=dict(command=command,selected_files=[str(p) for p in paths],natural_exit=child.returncode,
               diagnostics=json.loads(child.stdout),stderr=child.stderr,
               rooted_LSP=lsp(ROOT/(P+'run.py')),
               positive_AST={p.name:[dict(name=node.name,line=node.lineno) for node in ast.walk(ast.parse(p.read_text())) if isinstance(node,(ast.ClassDef,ast.FunctionDef))] for p in paths},
               source_hashes=hashes,sources_unchanged=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items()))
    (ROOT/(P+'language_final.json')).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(diagnostic_exit=child.returncode,summary=value['diagnostics'].get('summary'),LSP_exit=value['rooted_LSP']['natural_exit'],definition=value['rooted_LSP']['definition'])),flush=True)


if __name__=='__main__':main()
