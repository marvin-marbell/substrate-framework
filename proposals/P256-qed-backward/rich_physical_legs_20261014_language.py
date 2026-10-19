"""Positive pre-edit direct Pyright and Python AST navigation."""
import ast
import json
from pathlib import Path
import queue
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parent
source = ROOT / 'rich_field_packet_20261014_connected.py'
process = subprocess.Popen(['/home/administrator/.local/bin/pyright-langserver', '--stdio'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
messages = queue.Queue()
def reader():
    while True:
        line = process.stdout.readline()
        if not line:
            return
        if line.lower().startswith(b'content-length:'):
            length = int(line.split(b':')[1])
            process.stdout.readline()
            messages.put(json.loads(process.stdout.read(length)))
threading.Thread(target=reader, daemon=True).start()
def send(obj):
    data = json.dumps(obj).encode()
    process.stdin.write(f'Content-Length: {len(data)}\r\n\r\n'.encode() + data)
    process.stdin.flush()
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
for identifier, method, extra in [(2, 'textDocument/documentSymbol', {}), (3, 'textDocument/definition', {'position': {'line': 37, 'character': 24}}), (4, 'textDocument/diagnostic', {})]:
    send({'jsonrpc': '2.0', 'id': identifier, 'method': method, 'params': {'textDocument': {'uri': source.as_uri()}, **extra}})
    answers[method] = response(identifier)
send({'jsonrpc': '2.0', 'id': 5, 'method': 'shutdown', 'params': None})
response(5)
send({'jsonrpc': '2.0', 'method': 'exit', 'params': None})
process.wait(timeout=10)
positive = [{'name': n.name, 'line': n.lineno} for n in ast.walk(ast.parse(source.read_text())) if isinstance(n, ast.ClassDef) and n.name == 'ConnectedPacket']
assert positive and 'result' in answers['textDocument/documentSymbol'] and answers['textDocument/definition'].get('result')
(ROOT / 'rich_physical_legs_20261014_language.json').write_text(json.dumps({'root': str(ROOT), 'initialize': initial, 'probes': answers, 'positive_AST': positive, 'tooling': 'Direct installed pyright-langserver; built-in device unavailable. Standard Python AST.', 'final_diagnostics': 'Review8-owned; author actual scientific consumers are not build/lint/test runs.'}, indent=2) + '\n')
print(json.dumps({'positive_AST': positive, 'definition': answers['textDocument/definition']}))
