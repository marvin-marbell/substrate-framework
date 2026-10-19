"""Run only the new support review consumer, selected from its preexecution ZIP."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    request_raw = args.request.read_bytes()
    request = json.loads(request_raw)
    prefix = request['epoch']
    captured = {name: (ROOT / name).read_bytes() for name in request['pins']}
    assert all(sha(raw) == request['pins'][name] for name, raw in captured.items())
    assert sha(captured[request['consumer']]) == request['consumer_sha256']
    pre = ROOT / (prefix + '_preexecution.zip')
    mapping = {name: 'inputs/' + Path(name).name for name in captured}
    assert len(set(mapping.values())) == len(mapping)
    before = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with zipfile.ZipFile(pre, 'x', zipfile.ZIP_DEFLATED) as z:
        for name, raw in sorted(captured.items()):
            z.writestr(mapping[name], raw)
        z.writestr('request/' + args.request.name, request_raw)
    pre_sha = sha(pre.read_bytes())
    receipt = ROOT / (prefix + '_receipt.json')
    code = ('import pathlib,runpy,sys,zipfile; '
            'z=zipfile.ZipFile(sys.argv[1]); raw=z.read(sys.argv[2]); '
            'source=sys.argv[3]; sys.argv=[source,"--request",sys.argv[4],"--output",sys.argv[5]]; '
            'exec(compile(raw,source,"exec"),{"__name__":"__main__","__file__":source})')
    argv = [request['python'], '-B', '-c', code, str(pre), mapping[request['consumer']],
            str(ROOT / request['consumer']), str(args.request.resolve()), str(receipt)]
    started = time.monotonic()
    child = subprocess.run(argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    elapsed = time.monotonic() - started
    after = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for suffix, raw in (('stdout.txt', child.stdout), ('stderr.txt', child.stderr)):
        with (ROOT / (prefix + '_' + suffix)).open('xb') as stream:
            stream.write(raw)
    stable = all((ROOT / name).read_bytes() == raw for name, raw in captured.items())
    exit_path = ROOT / (prefix + '_exit.json')
    save(exit_path, {'exit_code': child.returncode, 'natural_exit': True, 'elapsed_seconds': elapsed,
                     'started_utc': before, 'finished_utc': after, 'inputs_unchanged': stable,
                     'stdout_sha256': sha(child.stdout), 'stderr_sha256': sha(child.stderr)})
    executed = ROOT / (prefix + '_executed.zip')
    artifacts = [exit_path, ROOT / (prefix + '_stdout.txt'), ROOT / (prefix + '_stderr.txt')]
    if receipt.exists():
        artifacts.append(receipt)
    with zipfile.ZipFile(executed, 'x', zipfile.ZIP_DEFLATED) as z:
        with zipfile.ZipFile(pre) as source:
            for name in source.namelist():
                z.writestr(name, source.read(name))
        for path in artifacts:
            z.writestr('outputs/' + path.name, path.read_bytes())
    capsule = {'kind': 'NEW_INDEPENDENT_SUPPORT_REVIEW_SOURCE_SELECTED_NATURAL_EXIT',
               'request': {'path': args.request.name, 'sha256': sha(request_raw)},
               'argv': argv, 'source_selection': 'compile bytes from immutable preexecution ZIP; no original producer import',
               'preexecution': {'path': pre.name, 'sha256': pre_sha, 'created_before_child': True},
               'executed': {'path': executed.name, 'sha256': sha(executed.read_bytes())},
               'consumer_sha256': request['consumer_sha256'], 'input_pins': request['pins'],
               'exit': {'path': exit_path.name, 'sha256': sha(exit_path.read_bytes())},
               'exit_code': child.returncode, 'natural_exit': True, 'elapsed_seconds': elapsed,
               'inputs_unchanged': stable, 'python': request['python'], 'launcher_runtime': sys.version,
               'governor': request['governor'], 'ground_numeric_or_phase_inference': False}
    if receipt.exists():
        capsule['receipt'] = {'path': receipt.name, 'sha256': sha(receipt.read_bytes())}
    save(ROOT / (prefix + '_capsule.json'), capsule)
    print(json.dumps({'exit_code': child.returncode, 'elapsed_seconds': elapsed,
                      'inputs_unchanged': stable, 'capsule': prefix + '_capsule.json'}))
    if child.returncode or not stable:
        raise SystemExit(child.returncode or 1)


if __name__ == '__main__':
    main()
