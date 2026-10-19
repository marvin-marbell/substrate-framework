"""Capture typed successor bytes; scientific request pins remain frozen separately."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def save(path: Path, value: dict) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    request_raw = args.request.read_bytes()
    request = json.loads(request_raw)
    prefix = request['typed_epoch']
    assert '_parenttyped_' in request['consumer']
    captured = {name: (ROOT / name).read_bytes() for name in request['capture_pins']}
    assert {name: sha(raw) for name, raw in captured.items()} == request['capture_pins']
    mapping = {name: Path(name).name for name in captured}
    assert len(set(mapping.values())) == len(mapping)
    assert args.request.name not in mapping.values()
    pre = ROOT / (prefix + '_preexecution.zip')
    executed = ROOT / (prefix + '_executed.zip')
    receipt = ROOT / (prefix + '_receipt.json')
    with zipfile.ZipFile(pre, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, raw in captured.items():
            archive.writestr(mapping[name], raw)
        archive.writestr(args.request.name, request_raw)
    source = str(ROOT / request['consumer'])
    cli = ['--request', str(args.request.resolve()), '--output', str(receipt), *request['extra_args']]
    code = ('import sys,zipfile; raw=zipfile.ZipFile(sys.argv[1]).read(sys.argv[2]); '
            'source=sys.argv[3]; sys.argv=[source]+sys.argv[4:]; '
            'exec(compile(raw,source,"exec"),{"__name__":"__main__","__file__":source})')
    argv = [sys.executable, '-B', '-c', code, str(pre), mapping[request['consumer']], source, *cli]
    started = time.monotonic()
    child = subprocess.run(argv, cwd=ROOT, capture_output=True, check=False)
    runtime = {'returncode': child.returncode, 'natural_exit': child.returncode >= 0,
               'elapsed_seconds': time.monotonic() - started}
    outputs = []
    for suffix, raw in (('stdout.txt', child.stdout), ('stderr.txt', child.stderr)):
        path = ROOT / (prefix + '_' + suffix)
        with path.open('xb') as stream:
            stream.write(raw)
        outputs.append(path)
    exit_path = ROOT / (prefix + '_exit.json')
    save(exit_path, runtime)
    outputs.append(exit_path)
    if receipt.exists():
        outputs.append(receipt)
    after = {name: sha((ROOT / name).read_bytes()) for name in captured}
    assert args.request.read_bytes() == request_raw
    with zipfile.ZipFile(executed, 'x', zipfile.ZIP_DEFLATED) as archive:
        with zipfile.ZipFile(pre) as original:
            for name in original.namelist():
                archive.writestr(name, original.read(name))
        for path in outputs:
            archive.writestr(path.name, path.read_bytes())
    capsule = {'kind': 'NEW_TYPED_SCIENTIFIC_CAPTURE_WITH_FROZEN_REQUEST_NOT_ORIGINAL_REPLAY',
               'pins_before': request['capture_pins'], 'pins_after': after,
               'inputs_unchanged': request['capture_pins'] == after, 'runtime': runtime,
               'preexecution': {'path': pre.name, 'sha256': sha(pre.read_bytes())},
               'executed': {'path': executed.name, 'sha256': sha(executed.read_bytes())},
               'outputs': {path.name: sha(path.read_bytes()) for path in outputs},
               'request': {'path': args.request.name, 'sha256': sha(request_raw)},
               'archive_member_map': mapping, 'originals_modified_or_rebound': False}
    save(ROOT / (prefix + '_capsule.json'), capsule)
    print(json.dumps({'epoch': prefix, 'runtime': runtime, 'inputs_unchanged': capsule['inputs_unchanged']}))
    if child.returncode or not capsule['inputs_unchanged']:
        raise SystemExit(child.returncode or 1)


if __name__ == '__main__':
    main()
