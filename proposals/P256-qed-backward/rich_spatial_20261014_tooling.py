"""Direct Pyright LSP pre-edit navigation receipt; no project checks."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'rich_photon_fullq_20261014_model.py'
server = subprocess.Popen(['/home/administrator/.local/bin/pyright-langserver', '--stdio'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

def send(payload):
    data = json.dumps(payload).encode()
    server.stdin.write(f'Content-Length: {len(data)}\r\n\r\n'.encode() + data)
    server.stdin.flush()

def receive():
    length = 0
    while True:
        line = server.stdout.readline()
        if not line:
            raise RuntimeError('LSP exited')
        if line == b'\r\n':
            break
        if line.lower().startswith(b'content-length:'):
            length = int(line.split(b':')[1])
    return json.loads(server.stdout.read(length))

def request(index, method, params):
    send({'jsonrpc': '2.0', 'id': index, 'method': method, 'params': params})
    while True:
        response = receive()
        if response.get('method') and 'id' in response:
            send({'jsonrpc': '2.0', 'id': response['id'], 'result': []})
        if response.get('id') == index:
            return response

init = request(1, 'initialize', {'processId': None, 'rootUri': ROOT.as_uri(), 'capabilities': {'textDocument': {'publishDiagnostics': {} }}, 'initializationOptions': {}})
send({'jsonrpc': '2.0', 'method': 'initialized', 'params': {}})
send({'jsonrpc': '2.0', 'method': 'textDocument/didOpen', 'params': {'textDocument': {'uri': SOURCE.as_uri(), 'languageId': 'python', 'version': 1, 'text': SOURCE.read_text()}}})
symbols = request(2, 'textDocument/documentSymbol', {'textDocument': {'uri': SOURCE.as_uri()}})
definition = request(3, 'textDocument/definition', {'textDocument': {'uri': SOURCE.as_uri()}, 'position': {'line': 82, 'character': 53}})
diag = request(4, 'textDocument/diagnostic', {'textDocument': {'uri': SOURCE.as_uri()}})
request(5, 'shutdown', {})
send({'jsonrpc': '2.0', 'method': 'exit', 'params': {}})
server.wait(timeout=10)
receipt = {'root': str(ROOT), 'server': '/home/administrator/.local/bin/pyright-langserver', 'source': str(SOURCE), 'initialize': init, 'symbols': symbols, 'definition': definition, 'diagnostics_request': diag, 'note': 'No mounted xd LSP device; direct protocol. Unsupported pull diagnostics is recorded, not a clean diagnostic assertion.'}
(ROOT / 'rich_spatial_20261014_tooling.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps({'symbol_count': len(symbols.get('result', [])), 'definition': definition, 'diagnostics': diag}))
