"""Direct Pyright and positive Python AST preflight for the selected root."""
import ast
import json
from pathlib import Path
import queue
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parent
source = ROOT / 'rich_spinor_gauss_20261014_model.py'
p = subprocess.Popen(['/home/administrator/.local/bin/pyright-langserver', '--stdio'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
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
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        obj = messages.get(timeout=40)
        if obj.get('method') == 'workspace/configuration':
            send({'jsonrpc': '2.0', 'id': obj['id'], 'result': [{'pythonPath': '/home/administrator/substrate-framework/.venv/bin/python'} for _ in obj['params']['items']]})
        if obj.get('id') == identifier and ('result' in obj or 'error' in obj):
            return obj
    raise TimeoutError(identifier)
send({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'processId': None, 'rootUri': ROOT.as_uri(), 'capabilities': {'workspace': {'configuration': True}}}})
initial = response(1)
send({'jsonrpc': '2.0', 'method': 'initialized', 'params': {}})
send({'jsonrpc': '2.0', 'method': 'textDocument/didOpen', 'params': {'textDocument': {'uri': source.as_uri(), 'languageId': 'python', 'version': 1, 'text': source.read_text()}}})
send({'jsonrpc': '2.0', 'id': 2, 'method': 'textDocument/documentSymbol', 'params': {'textDocument': {'uri': source.as_uri()}}})
symbols = response(2)
send({'jsonrpc': '2.0', 'id': 3, 'method': 'textDocument/definition', 'params': {'textDocument': {'uri': source.as_uri()}, 'position': {'line': 37, 'character': 8}}})
definition = response(3)
send({'jsonrpc': '2.0', 'id': 4, 'method': 'textDocument/diagnostic', 'params': {'textDocument': {'uri': source.as_uri()}}})
diagnostics = response(4)
send({'jsonrpc': '2.0', 'id': 5, 'method': 'shutdown', 'params': None}); response(5)
send({'jsonrpc': '2.0', 'method': 'exit', 'params': None}); p.wait(timeout=10)
tree = ast.parse(source.read_text())
positive = [{'name': n.name, 'line': n.lineno} for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'Sector']
assert positive
result = {'root': str(ROOT), 'source': str(source), 'server': 'installed pyright-langserver', 'initialized': initial, 'symbols': symbols, 'definition': definition, 'diagnostic_request': diagnostics, 'positive_python_ast': positive, 'native_device_missing': True}
(ROOT / 'rich_induced_spin_20261014_language.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
