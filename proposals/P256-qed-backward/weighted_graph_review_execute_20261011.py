"""Archive immutable review inputs, run ONLY new review CLI, retain natural exit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exclusive_json(path: Path, obj: dict) -> None:
    with path.open('x') as stream:
        json.dump(obj, stream, indent=2)
        stream.write('\n')


def archive(path: Path, names: list[str]) -> dict:
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in names:
            bundle.write(ROOT/name, name)
    return {'path': path.name, 'sha256': digest(path),
            'members': {name: digest(ROOT/name) for name in names}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    prefix = args.prefix
    if not prefix.startswith('weighted_graph_review_'):
        raise ValueError('review ownership prefix required')
    names = list(request['pins']) + [args.request.name]
    before = {name: digest(ROOT/name) for name in names}
    if {name: before[name] for name in request['pins']} != request['pins']:
        raise ValueError('review input pin mismatch before launch')
    pre = archive(ROOT/(prefix+'_preexecution.zip'), names)
    receipt = prefix+'_receipt.json'
    command = [PYTHON, '-B', str(ROOT/request['consumer']), '--request',
               str(args.request.resolve()), '--output', str(ROOT/receipt)]
    start = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
    elapsed = time.monotonic()-start
    for suffix, raw in (('_stdout.txt', completed.stdout), ('_stderr.txt', completed.stderr)):
        with (ROOT/(prefix+suffix)).open('xb') as stream:
            stream.write(raw)
    after = {name: digest(ROOT/name) for name in names}
    exit_record = {'command': command, 'returncode': completed.returncode,
                   'natural_exit': completed.returncode >= 0, 'elapsed_seconds': elapsed,
                   'source_input_sha256_before': before, 'source_input_sha256_after': after,
                   'source_input_unchanged': before == after,
                   'old_producers_imported_or_run': False}
    exclusive_json(ROOT/(prefix+'_exit.json'), exit_record)
    outputs = [prefix+'_stdout.txt', prefix+'_stderr.txt', prefix+'_exit.json']
    if (ROOT/receipt).exists():
        outputs.append(receipt)
    executed = archive(ROOT/(prefix+'_executed.zip'), names+outputs)
    exclusive_json(ROOT/(prefix+'_capsule.json'), {'schema': 1,
                   'kind': 'INDEPENDENT_GRAPH_REVIEW_NATURAL_EXIT_CAPSULE',
                   'preexecution': pre, 'executed': executed, 'exit': exit_record,
                   'scope': 'Native coefficient and separated analytical-method controls only'})
    print(json.dumps({'prefix': prefix, 'returncode': completed.returncode,
                      'natural_exit': completed.returncode >= 0, 'stable': before == after}))
    if before != after:
        raise RuntimeError('input instability retained in capsule')
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
