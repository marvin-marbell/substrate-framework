"""Once-final owned Python diagnostics and actual rooted Pyright LSP/AST."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
PREFIX = 'rich_periodic_photon_20261015_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def rooted_lsp(path):
    server = subprocess.Popen(['/home/administrator/.local/bin/pyright-langserver','--stdio'],cwd=PROJECT,
                              stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    messages = queue.Queue()

    def receive():
        while True:
            headers = {}
            while True:
                line = server.stdout.readline()
                if not line:
                    return
                if line == b'\r\n':
                    break
                key, value = line.decode().split(':',1)
                headers[key.lower()] = value.strip()
            messages.put(json.loads(server.stdout.read(int(headers['content-length']))))

    threading.Thread(target=receive,daemon=True).start()

    def send(method, params, identity=None):
        message = dict(jsonrpc='2.0',method=method,params=params)
        if identity is not None:
            message['id'] = identity
        raw = json.dumps(message).encode()
        server.stdin.write(f'Content-Length: {len(raw)}\r\n\r\n'.encode()+raw)
        server.stdin.flush()

    events = []

    def response(identity):
        deadline = time.time()+90
        while time.time() < deadline:
            message = messages.get(timeout=max(.1,deadline-time.time()))
            events.append(message)
            if message.get('id') == identity:
                return message
        raise TimeoutError(identity)

    send('initialize',dict(processId=None,rootUri=PROJECT.as_uri(),capabilities={},
                           initializationOptions=dict(pythonPath=PYTHON)),1)
    initialized = response(1)
    send('initialized',{})
    send('textDocument/didOpen',dict(textDocument=dict(uri=path.as_uri(),languageId='python',version=1,text=path.read_text())))
    send('textDocument/documentSymbol',dict(textDocument=dict(uri=path.as_uri())),2)
    symbols = response(2)
    lines = path.read_text().splitlines()
    call_line = next(i for i,line in enumerate(lines) if 'basis(kind)' in line)
    send('textDocument/definition',dict(textDocument=dict(uri=path.as_uri()),position=dict(line=call_line,character=lines[call_line].index('basis')+1)),3)
    definition = response(3)
    time.sleep(3)
    while not messages.empty():
        events.append(messages.get())
    send('shutdown',None,4)
    response(4)
    send('exit',None)
    server.wait(timeout=20)
    return dict(root=str(PROJECT),initialize=initialized,symbols=symbols,definition=definition,events=events,server_exit=server.returncode)


def main():
    selected = [PREFIX+s+'.py' for s in ('model','experiment','capture','variance','deliver','language')]
    hashes = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in selected}
    command = ['/home/administrator/.local/bin/pyright','--pythonpath',PYTHON,'--outputjson',*selected]
    checked = subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    diagnostics = json.loads(checked.stdout)
    lsp = rooted_lsp(ROOT/(PREFIX+'experiment.py'))
    structural = {name:[dict(name=node.name,line=node.lineno) for node in ast.walk(ast.parse((ROOT/name).read_text())) if isinstance(node,(ast.FunctionDef,ast.ClassDef))] for name in selected}
    result = dict(kind='ONCE_FINAL_SELECTED_OWNED_FILES_NOT_REPOSITORY_CLEAN',command=command,
                  selected_files=selected,scientific_interpreter=PYTHON,source_SHA256=hashes,
                  natural_exit=True,returncode=checked.returncode,diagnostics=diagnostics,stderr=checked.stderr,
                  rooted_LSP=lsp,positive_AST=structural,
                  pyright_version=subprocess.run(['/home/administrator/.local/bin/pyright','--version'],capture_output=True,text=True,check=True).stdout.strip(),
                  sources_unchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value for name,value in hashes.items()))
    (ROOT/(PREFIX+'language_final.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(returncode=checked.returncode,summary=diagnostics.get('summary'),LSP_exit=lsp['server_exit'],sources_unchanged=result['sources_unchanged'])),flush=True)


if __name__ == '__main__':
    main()
