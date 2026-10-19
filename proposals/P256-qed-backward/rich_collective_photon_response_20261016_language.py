"""Rooted actual LSP symbols/definition and selected-file Pyright/AST evidence."""
import ast
import json
from pathlib import Path
import queue
import subprocess
import threading
import time

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
P='rich_collective_photon_response_20261016_'
PY='/home/administrator/substrate-framework/.venv/bin/python'


def rooted_lsp(path):
    child=subprocess.Popen(['/home/administrator/.local/bin/pyright-langserver','--stdio'],cwd=PROJECT,
                           stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    inp=child.stdin;out=child.stdout
    assert inp is not None and out is not None
    messages=queue.Queue();events=[]
    def receive():
        while True:
            headers={}
            while True:
                line=out.readline()
                if not line:return
                if line==b'\r\n':break
                k,v=line.decode().split(':',1);headers[k.lower()]=v.strip()
            messages.put(json.loads(out.read(int(headers['content-length']))))
    threading.Thread(target=receive,daemon=True).start()
    def send(method,params,identity=None):
        value=dict(jsonrpc='2.0',method=method,params=params)
        if identity is not None:value['id']=identity
        data=json.dumps(value).encode();inp.write(f'Content-Length: {len(data)}\r\n\r\n'.encode()+data);inp.flush()
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
    lines=path.read_text().splitlines();line=next(i for i,t in enumerate(lines) if 'model.interval_response(' in t)
    send('textDocument/definition',dict(textDocument=dict(uri=path.as_uri()),position=dict(line=line,character=lines[line].index('interval_response')+1)),3)
    definition=response(3);time.sleep(2)
    while not messages.empty():events.append(messages.get())
    send('shutdown',None,4);response(4);send('exit',None);child.wait(timeout=20)
    return dict(root=str(PROJECT),initialize=initialize,symbols=symbols,definition=definition,events=events,natural_exit=child.returncode)


def main():
    paths=sorted(ROOT.glob(P+'*.py'));command=['/home/administrator/.local/bin/pyright','--pythonpath',PY,'--outputjson',*map(str,paths)]
    process=subprocess.run(command,cwd=PROJECT,capture_output=True,text=True)
    result=dict(command=command,natural_exit=process.returncode,diagnostics=json.loads(process.stdout),stderr=process.stderr,
                selected_files=[str(p) for p in paths],rooted_LSP=rooted_lsp(ROOT/(P+'run.py')),
                positive_AST={p.name:[dict(name=n.name,line=n.lineno) for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,(ast.FunctionDef,ast.ClassDef))] for p in paths},
                pre_edit_LSP='artifact://1767 rooted symbols/definition from frozen bridge helper natural0',
                pre_edit_AST='xd://ast_grep path_data positive match frozen thermal.py')
    (ROOT/(P+'language_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(selected=len(paths),pyright_exit=process.returncode,summary=result['diagnostics']['summary'],LSP_exit=result['rooted_LSP']['natural_exit'])),flush=True)


if __name__=='__main__':main()
