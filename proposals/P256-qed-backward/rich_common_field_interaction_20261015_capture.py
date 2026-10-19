"""Exact input/source/factor capsule and natural/interrupted subprocess receipt."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    request_path, prefix = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    request = json.loads(request_path.read_text())
    files = {request_path, Path(__file__).resolve(),
             ROOT / 'rich_common_field_interaction_20261015_consume.py'}
    if request.get('direct_action', False):
        consumer = request.get('direct_consumer', 'rich_common_field_interaction_20261015_direct.py')
        if consumer not in ('rich_common_field_interaction_20261015_direct.py', 'rich_common_field_interaction_20261015_transition.py'):
            raise ValueError('Unknown owned physical consumer')
        files.add(ROOT / consumer)
        files.add(ROOT / 'rich_common_field_interaction_20261015_direct.py')
        pins = [request['model_capsule'], request['off_factorization'], *request['source_pins']]
        base = ROOT
        if request.get('numeric_cache'):
            cache = request['numeric_cache']
            pins += [cache['manifest'], cache['states'], cache['loader']]
            cache_path = ROOT / cache['manifest']['file']
            if sha256(cache_path) != cache['manifest']['sha256']:
                raise ValueError('Changed shared numeric manifest')
            snapshots = json.loads(cache_path.read_text())['snapshots']
            labels = [name+suffix for name in ('neutral', 'lepton', 'antilepton')
                      for suffix in ('_H_zero', '_H_full_transverse_off', '_Q2', '_Q5')]
            labels += [name+'_source_site'+str(site) for name in ('lepton', 'antilepton')
                       for site in (request['source_site'], request['receiver_site'])]
            pins += [snapshots[label] for label in labels]
    else:
        consumer = 'rich_common_field_interaction_20261015_consume.py'
        manifest_path = ROOT / request['manifest']
        if sha256(manifest_path) != request['manifest_sha256']:
            raise ValueError('Changed manifest before capture')
        manifest = json.loads(manifest_path.read_text())
        files.add(manifest_path)
        pins = [manifest['reference'], manifest['factor_loader'], *manifest['provenance_pins'],
                *manifest['operators'].values()]
        base = manifest_path.parent
    for item in pins:
        path = (base / item['file']).resolve()
        if sha256(path) != item['sha256']:
            raise ValueError('Changed pinned member: ' + str(path))
        files.add(path)
    capsule = prefix.with_name(prefix.name + '_inputs.zip')
    members = []
    with zipfile.ZipFile(capsule, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in sorted(files):
            name = str(path.relative_to(ROOT))
            archive.write(path, name)
            members.append(dict(file=name, sha256=sha256(path), bytes=path.stat().st_size))
    start = time.monotonic()
    command = [PYTHON, str(ROOT / consumer),
               str(request_path), str(prefix.with_name(prefix.name + '_result'))]
    start_record = dict(command=command, capsule=capsule.name, capsule_sha256=sha256(capsule),
                        members=members, deadline_seconds=request['deadline_seconds'])
    prefix.with_name(prefix.name + '_start.json').write_text(json.dumps(start_record, indent=2) + '\n')
    interrupted = False
    with prefix.with_name(prefix.name + '_stdout.txt').open('wb') as stdout, prefix.with_name(prefix.name + '_stderr.txt').open('wb') as stderr:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            code = process.wait(timeout=request['deadline_seconds'])
        except subprocess.TimeoutExpired:
            interrupted = True
            process.terminate()
            try:
                code = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait(timeout=15)
        except KeyboardInterrupt:
            process.terminate()
            code = process.wait(timeout=15)
            interrupted = True
    outputs = []
    for path in sorted(prefix.parent.glob(prefix.name + '*')):
        if path.is_file() and path != capsule:
            outputs.append(dict(file=path.name, sha256=sha256(path), bytes=path.stat().st_size))
    receipt = dict(natural_exit=not interrupted, interrupted=interrupted, returncode=code,
                   elapsed_seconds=time.monotonic() - start, outputs=outputs,
                   input_capsule=capsule.name, input_capsule_sha256=sha256(capsule))
    exit_path = prefix.with_name(prefix.name + '_exit.json')
    exit_path.write_text(json.dumps(receipt, indent=2) + '\n')
    with zipfile.ZipFile(prefix.with_name(prefix.name + '_outputs.zip'), 'x', compression=zipfile.ZIP_STORED) as archive:
        for item in outputs:
            archive.write(prefix.parent / item['file'], item['file'])
        archive.write(exit_path, exit_path.name)
    print(json.dumps(receipt), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
