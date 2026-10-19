"""Capture independent reviewer inputs before execution and natural outputs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, obj):
    with path.open('x') as stream:
        stream.write(json.dumps(obj, indent=2) + '\n')


def archive(path, members):
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)
    path.chmod(0o444)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    root = args.request.parent
    request_bytes = args.request.read_bytes()
    request = json.loads(request_bytes)
    prefix = request['output_prefix']
    assert prefix.startswith('charged_schur_review_')
    before_members = {args.request.name: request_bytes}
    source_hashes = {}
    for name, sha in request['pins'].items():
        data = (root / name).read_bytes()
        assert digest(data) == sha, name
        before_members[name] = data
        source_hashes[name] = sha
    for path, sha in request['external_pins'].items():
        data = Path(path).read_bytes()
        assert digest(data) == sha, path
        before_members['external/' + Path(path).name] = data
        source_hashes[path] = sha
    before = root / (prefix + '_preexecution.zip')
    archive(before, before_members)
    receipt = root / (prefix + '_receipt.json')
    assert not receipt.exists()
    command = [request['python'], str(root / request['source']), '--request', str(args.request), '--output', str(receipt)]
    started = time.time()
    completed = subprocess.run(command, cwd=root, capture_output=True, check=False)
    elapsed = time.time() - started
    outputs = {}
    for suffix, data in [('_stdout.txt', completed.stdout), ('_stderr.txt', completed.stderr)]:
        path = root / (prefix + suffix)
        with path.open('xb') as stream:
            stream.write(data)
        outputs[path.name] = data
    exit_path = root / (prefix + '_exit.json')
    save_json(exit_path, {'natural_exit': True, 'returncode': completed.returncode,
                          'elapsed_seconds': elapsed, 'command': command,
                          'preexecution_archive_sha256': digest(before.read_bytes())})
    outputs[exit_path.name] = exit_path.read_bytes()
    if receipt.exists():
        outputs[receipt.name] = receipt.read_bytes()
    after = root / (prefix + '_executed.zip')
    archive(after, {**before_members, **outputs})
    unchanged = all(digest((root / name).read_bytes()) == sha for name, sha in request['pins'].items())
    unchanged = unchanged and all(digest(Path(path).read_bytes()) == sha for path, sha in request['external_pins'].items())
    capsule = root / (prefix + '_capsule.json')
    save_json(capsule, {'kind': 'INDEPENDENT_REVIEWER_NATURAL_EXIT_IMMUTABLE_CAPSULE',
                        'governor': request['governor'], 'command': command,
                        'natural_exit': True, 'returncode': completed.returncode,
                        'elapsed_seconds': elapsed, 'request_sha256': digest(request_bytes),
                        'source_and_target_pins': source_hashes,
                        'pins_unchanged': unchanged,
                        'preexecution_archive': {'path': before.name, 'sha256': digest(before.read_bytes())},
                        'executed_archive': {'path': after.name, 'sha256': digest(after.read_bytes())},
                        'outputs': {name: digest(data) for name, data in outputs.items()},
                        'scientific_admission': False, 'campaign_complete': False})
    for name in outputs:
        (root / name).chmod(0o444)
    capsule.chmod(0o444)
    args.request.chmod(0o444)
    print(json.dumps({'returncode': completed.returncode, 'natural_exit': True,
                      'pins_unchanged': unchanged, 'capsule': capsule.name,
                      'capsule_sha256': digest(capsule.read_bytes())}))
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    assert unchanged


if __name__ == '__main__':
    main()
