"""Pin inputs BEFORE a new owned consumer, retain every natural failure/exit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile
from typing import Any

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: dict[str, Any]) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def archive(path: Path, names: list[str]) -> dict[str, Any]:
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
    if not prefix.startswith('native_static_exterior_'):
        raise ValueError('owned prefix required')
    names = list(request['pins'])+[args.request.name]
    before = {name: digest(ROOT/name) for name in names}
    if {name: before[name] for name in request['pins']} != request['pins']:
        raise ValueError('pin mismatch before launch')
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
                   'natural_exit': completed.returncode >= 0,
                   'elapsed_seconds': elapsed, 'source_input_sha256_before': before,
                   'source_input_sha256_after': after, 'source_input_unchanged': before == after,
                   'old_producers_or_solvers_run': False}
    save(ROOT/(prefix+'_exit.json'), exit_record)
    outputs = [prefix+'_stdout.txt', prefix+'_stderr.txt', prefix+'_exit.json']
    if (ROOT/receipt).exists():
        outputs.append(receipt)
    executed = archive(ROOT/(prefix+'_executed.zip'), names+outputs)
    save(ROOT/(prefix+'_capsule.json'), {'schema': 1,
         'kind': 'NATIVE_STATIC_EXTERIOR_NATURAL_EXIT_CAPSULE_NOT_NATIVE_U',
         'preexecution': pre, 'executed': executed, 'exit': exit_record,
         'failed_branches_retained': True,
         'scope': 'native scaling/action/conditional projection controls; actual-ground exterior remains open'})
    print(json.dumps({'prefix': prefix, 'returncode': completed.returncode,
                      'natural_exit': completed.returncode >= 0,
                      'source_input_unchanged': before == after}))
    if before != after:
        raise RuntimeError('input instability retained')
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
