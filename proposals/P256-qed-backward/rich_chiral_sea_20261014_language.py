"""Direct selected-root Pyright navigation and positive structural preflight."""
import ast
import json
from pathlib import Path
import queue
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parent
source = ROOT / 'rich_filled_symmetry_20261014_model.py'
p = subprocess.Popen(['/home/administrator/.local/bin/pyright-langserver', '--stdio'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
messages = queue.Queue()
def reader():
    while True:
        line = p.stdout.readline()
        if not line:
            return
        if line.lower().startswith(b'content-length:'):
            size = int(line.split(b':')[1])
            p.stdout.readline()
            messages.put(json.loads(p.stdout.read(size)))
threading.Thread(target=reader, daemon=True).start()
def send(obj):
    data = json.dumps(obj).encode()
    p.stdin.write(f'Content-Length: {len(data)}\r\n\r\n'.encode() + data)
    p.stdin.flush()
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
answers = {}
for identifier, method, extra in [(2, 'textDocument/documentSymbol', {}), (3, 'textDocument/definition', {'position': {'line': 38, 'character': 10}}), (4, 'textDocument/diagnostic', {})]:
    send({'jsonrpc': '2.0', 'id': identifier, 'method': method, 'params': {'textDocument': {'uri': source.as_uri()}, **extra}})
    answers[method] = response(identifier)
send({'jsonrpc': '2.0', 'id': 5, 'method': 'shutdown', 'params': None})
response(5)
send({'jsonrpc': '2.0', 'method': 'exit', 'params': None})
p.wait(timeout=10)
positive = [{'name': n.name, 'line': n.lineno} for n in ast.walk(ast.parse(source.read_text())) if isinstance(n, ast.FunctionDef) and n.name == 'build']
assert positive
result = {'root': str(ROOT), 'source': str(source), 'server': 'pyright-langserver', 'native_device_unavailable': True, 'initialize': initial, 'probes': answers, 'positive_AST': positive, 'final_diagnostics': 'Reviewer-owned; no author build/lint/test/formatter.'}
(ROOT / 'rich_chiral_sea_20261014_language.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'positive_AST': positive, 'probe_results': {key: 'result' in value for key, value in answers.items()}}))
