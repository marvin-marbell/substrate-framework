"""Exact local-import/source/input receipt for this independent consumer only."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_review_wave24_20261015_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def imports(path, found):
    if path in found:
        return
    found.add(path)
    for node in ast.walk(ast.parse(path.read_text())):
        names = ([node.module] if isinstance(node, ast.ImportFrom) else
                 [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
        for name in names:
            if name is not None:
                local = ROOT / (name.split('.')[0] + '.py')
                if local.is_file():
                    imports(local, found)


def run():
    consumer = ROOT / (PREFIX+'consumer.py')
    source_paths = set()
    imports(consumer, source_paths)
    source_paths.add(Path(__file__).resolve())
    request_path = ROOT / (PREFIX+'input.json')
    request = json.loads(request_path.read_text())
    selected = source_paths | {request_path, ROOT/request['state_json'], ROOT/request['state_npz']}
    before = {path.name: digest(path) for path in sorted(selected)}
    command = [PYTHON, str(consumer), str(request_path)]
    started = time.time()
    (ROOT/(PREFIX+'start.json')).write_text(json.dumps(dict(command=command, start_unix=started, source_input_sha256_before=before), indent=2)+'\n')
    with (ROOT/(PREFIX+'stdout.txt')).open('w') as stdout, (ROOT/(PREFIX+'stderr.txt')).open('w') as stderr:
        done = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)
    after = {path.name: digest(path) for path in sorted(selected)}
    output = {path.name: digest(path) for suffix in ['provisional.json','states.npz','stdout.txt','stderr.txt']
              if (path := ROOT/(PREFIX+suffix)).exists()}
    receipt = dict(command=command, natural_process_exit=True, returncode=done.returncode,
                   elapsed_seconds=time.time()-started, source_input_sha256_before=before,
                   source_input_sha256_after=after, immutable_inputs=before==after,
                   output_sha256=output, interruption=False)
    (ROOT/(PREFIX+'exit.json')).write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(returncode=done.returncode, immutable_inputs=before==after,
                          elapsed_seconds=receipt['elapsed_seconds'], output_sha256=output)), flush=True)
    raise SystemExit(done.returncode)


if __name__ == '__main__':
    run()
