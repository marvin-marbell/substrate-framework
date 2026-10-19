"""Final selected diagnostics and rooted symbols/definition for own CAR consumer."""
import ast
import json
from pathlib import Path
import queue
import subprocess
import threading
import time
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
P='rich_primitive_vertex_review_20261016_'
PY='/home/administrator/substrate-framework/.venv/bin/python'


def editor(path):
    child=subprocess.Popen(['/home/administrator/.local/bin/pyright-langserver','--stdio'],cwd=PROJECT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    sink=child.stdin;stream=child.stdout;assert sink is not None and stream is not None
    messages=queue.Queue();events=[]
    def receive():
        while True:
            headers={}
            while True:
                line=stream.readline()
                if not line:return
                if line==b'\r\n':break
                key,value=line.decode().split(':',1);headers[key.lower()]=value.strip()
            messages.put(json.loads(stream.read(int(headers['content-length']))))
    threading.Thread(target=receive,daemon=True).start()
    def send(method,params,identity=None):
        message=dict(jsonrpc='2.0',method=method,params=params)
        if identity is not None:message['id']=identity
        raw=json.dumps(message).encode();sink.write(f'Content-Length: {len(raw)}\r\n\r\n'.encode()+raw);sink.flush()
    def response(identity):
        deadline=time.time()+90
        while time.time()<deadline:
            message=messages.get(timeout=max(.1,deadline-time.time()));events.append(message)
            if message.get('id')==identity:return message
        raise TimeoutError(identity)
    send('initialize',dict(processId=None,rootUri=PROJECT.as_uri(),capabilities={},initializationOptions=dict(pythonPath=PY)),1);initialized=response(1);send('initialized',{})
    send('workspace/didChangeConfiguration',dict(settings=dict(python=dict(pythonPath=PY,analysis=dict(extraPaths=[str(ROOT)])))))
    send('textDocument/didOpen',dict(textDocument=dict(uri=path.as_uri(),languageId='python',version=1,text=path.read_text())))
    send('textDocument/documentSymbol',dict(textDocument=dict(uri=path.as_uri())),2);symbols=response(2)
    lines=path.read_text().splitlines();line=next(i for i,text in enumerate(lines) if 'q=PairAlgebra(' in text)
    send('textDocument/definition',dict(textDocument=dict(uri=path.as_uri()),position=dict(line=line,character=lines[line].index('PairAlgebra')+1)),3);definition=response(3)
    time.sleep(2)
    while not messages.empty():events.append(messages.get())
    send('shutdown',None,4);response(4);send('exit',None);child.wait(timeout=20)
    return dict(root=str(PROJECT),python=PY,initialize=initialized,symbols=symbols,definition=definition,events=events,natural_exit=child.returncode)


def main():
    paths=sorted(ROOT.glob(P+'*.py'));command=['/home/administrator/.local/bin/pyright','--pythonpath',PY,'--outputjson',*map(str,paths)];child=subprocess.run(command,cwd=PROJECT,capture_output=True,text=True);diagnostics=json.loads(child.stdout);lsp=editor(ROOT/(P+'consume.py'))
    result=dict(command=command,diagnostic_exit=child.returncode,diagnostics=diagnostics,stderr=child.stderr,rooted_LSP=lsp,positive_AST={path.name:[dict(name=node.name,line=node.lineno) for node in ast.walk(ast.parse(path.read_text())) if isinstance(node,(ast.FunctionDef,ast.ClassDef))] for path in paths})
    (ROOT/(P+'language_final.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(diagnostic_exit=child.returncode,summary=diagnostics.get('summary'),LSP_exit=lsp['natural_exit'])),flush=True)


if __name__=='__main__':main()
