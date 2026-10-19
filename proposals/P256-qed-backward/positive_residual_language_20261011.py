"""Exercise actual root-scoped Pyright protocol before new scientific edits."""
import json
from pathlib import Path
import queue
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
SRC = ROOT / 'vacuum_connected_native_scientist_20261011.py'
p = subprocess.Popen(['/home/administrator/.local/bin/pyright-langserver', '--stdio'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=PROJECT)
q = queue.Queue()
notifications = []

def receive():
    assert p.stdout is not None
    while True:
        headers = {}
        while True:
            line = p.stdout.readline()
            if not line:
                return
            if line == b'\r\n':
                break
            k, v = line.decode().split(':', 1)
            headers[k.lower()] = v.strip()
        q.put(json.loads(p.stdout.read(int(headers['content-length']))))

threading.Thread(target=receive, daemon=True).start()

def send(obj):
    assert p.stdin is not None
    raw = json.dumps(obj).encode()
    p.stdin.write(f'Content-Length: {len(raw)}\r\n\r\n'.encode() + raw)
    p.stdin.flush()

def req(i, method, params):
    send(dict(jsonrpc='2.0', id=i, method=method, params=params))
    while True:
        obj = q.get(timeout=60)
        if 'method' in obj and 'id' in obj:
            send(dict(jsonrpc='2.0', id=obj['id'], result=None))
        elif 'method' in obj:
            notifications.append(obj)
        if obj.get('id') == i:
            return obj

init = req(1, 'initialize', dict(processId=None, rootUri=PROJECT.as_uri(), capabilities={}))
send(dict(jsonrpc='2.0', method='initialized', params={}))
send(dict(jsonrpc='2.0', method='textDocument/didOpen', params=dict(textDocument=dict(uri=SRC.as_uri(), languageId='python', version=1, text=SRC.read_text()))))
symbols = req(2, 'textDocument/documentSymbol', dict(textDocument=dict(uri=SRC.as_uri())))
definition = req(3, 'textDocument/definition', dict(textDocument=dict(uri=SRC.as_uri()), position=dict(line=286, character=45)))
assert symbols.get('result') and definition.get('result')
end = time.monotonic() + 3
while time.monotonic() < end:
    try:
        obj = q.get(timeout=end-time.monotonic())
        if 'method' in obj:
            notifications.append(obj)
    except queue.Empty:
        break
req(4, 'shutdown', None)
send(dict(jsonrpc='2.0', method='exit', params=None))
p.wait(timeout=10)
result = dict(project_root=str(PROJECT), source=str(SRC), binary='/home/administrator/.local/bin/pyright-langserver', initialize=init, symbols=symbols, definition=definition, diagnostic_notifications=[n for n in notifications if n.get('method') == 'textDocument/publishDiagnostics'], diagnostics_not_claimed_green=True, exit=p.returncode, AST='xd://ast_grep positive projected_product function match', mounted_lsp_device_unavailable=True)
(ROOT/'positive_residual_language_20261011.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(dict(symbols=len(symbols.get('result') or []), definition=True, diagnostics=len([n for n in notifications if n.get('method') == 'textDocument/publishDiagnostics']), exit=p.returncode)))
