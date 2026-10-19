"""Explicit selected-file Pyright and real rooted protocol navigation."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
PREFIX = 'rich_recoil_current_wave_20261016_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def navigate(path: Path) -> dict[str, Any]:
    process = subprocess.Popen(['/home/administrator/.local/bin/pyright-langserver', '--stdio'], cwd=PROJECT,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdin, stdout = process.stdin, process.stdout
    assert stdin is not None and stdout is not None
    messages: queue.Queue[Any] = queue.Queue()
    events = []
    def receive() -> None:
        while True:
            headers = {}
            while True:
                line = stdout.readline()
                if not line:
                    return
                if line == b'\r\n':
                    break
                key, value = line.decode().split(':', 1)
                headers[key.lower()] = value.strip()
            messages.put(json.loads(stdout.read(int(headers['content-length']))))
    threading.Thread(target=receive, daemon=True).start()
    def send(method: str, parameters: Any, identity: int | None = None) -> None:
        item: dict[str, Any] = dict(jsonrpc='2.0', method=method, params=parameters)
        if identity is not None:
            item['id'] = identity
        raw = json.dumps(item).encode()
        stdin.write(f'Content-Length: {len(raw)}\r\n\r\n'.encode()+raw)
        stdin.flush()
    def response(identity: int) -> Any:
        deadline = time.time()+90
        while time.time() < deadline:
            value = messages.get(timeout=max(.1, deadline-time.time()))
            events.append(value)
            if value.get('id') == identity:
                return value
        raise TimeoutError(identity)
    send('initialize', dict(processId=None, rootUri=PROJECT.as_uri(), capabilities={},
        initializationOptions=dict(pythonPath=PYTHON)), 1)
    initialize = response(1)
    send('initialized', {})
    send('textDocument/didOpen', dict(textDocument=dict(uri=path.as_uri(), languageId='python', version=1, text=path.read_text())))
    send('textDocument/documentSymbol', dict(textDocument=dict(uri=path.as_uri())), 2)
    symbols = response(2)
    lines = path.read_text().splitlines()
    line = next(i for i, text in enumerate(lines) if 'geometry(law)' in text)
    send('textDocument/definition', dict(textDocument=dict(uri=path.as_uri()), position=dict(line=line, character=lines[line].index('geometry')+2)), 3)
    definition = response(3)
    send('shutdown', None, 4)
    response(4)
    send('exit', None)
    process.wait(timeout=20)
    return dict(root=str(PROJECT), marker=str(PROJECT/'pyproject.toml'), initialize=initialize,
                symbols=symbols, definition=definition, protocol_events=events, natural_exit=process.returncode)


def main() -> None:
    files = sorted(ROOT.glob(PREFIX+'*.py'))
    command = ['/home/administrator/.local/bin/pyright', '--pythonpath', PYTHON, '--outputjson', *map(str, files)]
    completed = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True)
    result: dict[str, Any] = dict(command=command, selected_files=list(map(str, files)), natural_exit=completed.returncode,
        diagnostics=json.loads(completed.stdout), stderr=completed.stderr, rooted_LSP=navigate(ROOT/(PREFIX+'model.py')),
        positive_AST={path.name: [dict(name=node.name, line=node.lineno) for node in ast.walk(ast.parse(path.read_text())) if isinstance(node, (ast.ClassDef, ast.FunctionDef))] for path in files},
        source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
    (ROOT/(PREFIX+'language_final.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(exit=completed.returncode, summary=result['diagnostics']['summary'], LSP_exit=result['rooted_LSP']['natural_exit'])), flush=True)
    sys.exit(completed.returncode)


if __name__ == '__main__':
    main()
