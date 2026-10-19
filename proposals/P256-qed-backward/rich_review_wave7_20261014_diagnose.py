"""ONCE stable explicit-file Pyright and representative direct LSP diagnostics."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def lsp_diagnostics(selected):
    process = subprocess.Popen(['/home/administrator/.local/bin/pyright-langserver', '--stdio'], cwd=ROOT,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if process.stdin is None or process.stdout is None:
        raise RuntimeError('Missing LSP streams')
    incoming = queue.Queue()
    stdout = process.stdout
    stdin = process.stdin

    def reader():
        while True:
            line = stdout.readline()
            if not line:
                return
            if line.lower().startswith(b'content-length:'):
                size = int(line.split(b':')[1])
                stdout.readline()
                incoming.put(json.loads(stdout.read(size)))

    threading.Thread(target=reader, daemon=True).start()

    def send(message):
        data = json.dumps(message).encode()
        stdin.write(f'Content-Length: {len(data)}\r\n\r\n'.encode() + data)
        stdin.flush()

    def response(identifier):
        while True:
            message = incoming.get(timeout=120)
            if message.get('method') == 'workspace/configuration':
                send({'jsonrpc': '2.0', 'id': message['id'], 'result': [{'pythonPath': PYTHON} for _ in message['params']['items']]})
            if message.get('id') == identifier:
                return message

    send({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'processId': None, 'rootUri': ROOT.as_uri(), 'capabilities': {'workspace': {'configuration': True}}}})
    initial = response(1)
    send({'jsonrpc': '2.0', 'method': 'initialized', 'params': {}})
    results = {}
    for identifier, name in enumerate(selected, 2):
        path = ROOT / name
        send({'jsonrpc': '2.0', 'method': 'textDocument/didOpen', 'params': {'textDocument': {'uri': path.as_uri(), 'languageId': 'python', 'version': 1, 'text': path.read_text()}}})
        send({'jsonrpc': '2.0', 'id': identifier, 'method': 'textDocument/diagnostic', 'params': {'textDocument': {'uri': path.as_uri()}}})
        results[name] = response(identifier)
    shutdown = len(selected) + 2
    send({'jsonrpc': '2.0', 'id': shutdown, 'method': 'shutdown', 'params': None})
    response(shutdown)
    send({'jsonrpc': '2.0', 'method': 'exit', 'params': None})
    process.wait(timeout=20)
    return {'initialize': initial, 'selected_files': selected, 'results': results, 'server_exit': process.returncode, 'native_device': 'unmounted; direct protocol used'}


def main():
    parameters = json.loads((ROOT / sys.argv[1]).read_text())
    selected = parameters['selected_files']
    if not selected or any(not (ROOT / name).is_file() for name in selected):
        raise ValueError('Explicit existing source files required')
    before = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in selected}
    version = subprocess.run(['/home/administrator/.local/bin/pyright', '--version'], capture_output=True, text=True, check=True).stdout.strip()
    command = ['/home/administrator/.local/bin/pyright', '--pythonpath', PYTHON, '--outputjson', *selected]
    start = time.monotonic()
    done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    checker = json.loads(done.stdout)
    lsp = lsp_diagnostics(parameters['lsp_files'])
    artifact = {'kind': 'ONCE_FINAL_STABLE_EXPLICIT_SCOPE_NOT_ALL_CLEAN', 'root': str(ROOT), 'version': version, 'command': command,
                'natural_exit': True, 'returncode': done.returncode, 'elapsed_seconds': time.monotonic() - start,
                'selected_files': selected, 'selected_file_count': len(selected), 'source_SHA256_before': before,
                'sources_unchanged': all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value for name, value in before.items()),
                'reported_summary': checker.get('summary'), 'error_count_by_file': dict(Counter(Path(item['file']).name for item in checker.get('generalDiagnostics', []) if item['severity'] == 'error')),
                'checker': checker, 'stderr': done.stderr, 'LSP': lsp}
    (ROOT / 'rich_review_wave7_20261014_diagnostics.json').write_text(json.dumps(artifact, indent=2) + '\n')
    print(json.dumps({key: artifact[key] for key in ('version', 'returncode', 'selected_file_count', 'reported_summary', 'error_count_by_file', 'sources_unchanged')}, indent=2))


if __name__ == '__main__':
    main()
