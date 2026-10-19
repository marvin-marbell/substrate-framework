"""Run only the new comparison CLI; immutable pre/executed/source/input/natural-exit capsule."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--epoch', required=True)
    args = parser.parse_args()
    assert args.epoch.startswith('ground_carrier_comparison_')
    request = json.loads(args.request.read_text())
    files = {name: (ROOT / name).read_bytes() for name in request['pins']}
    assert all(sha(raw) == request['pins'][name] for name, raw in files.items())
    files[args.request.name] = args.request.read_bytes()
    stem = ROOT / args.epoch
    pre = Path(str(stem) + '_preexecution.zip')
    executed = Path(str(stem) + '_executed.zip')
    receipt = Path(str(stem) + '_receipt.json')
    stdout = Path(str(stem) + '_stdout.txt')
    stderr = Path(str(stem) + '_stderr.txt')
    exit_path = Path(str(stem) + '_exit.json')
    capsule = Path(str(stem) + '_capsule.json')
    assert not any(p.exists() for p in (pre, executed, receipt, stdout, stderr, exit_path, capsule))
    archive_names = {}
    def snapshot(destination, members):
        with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, raw in members.items():
                member = name if not Path(name).is_absolute() else 'external/' + Path(name).name
                assert member not in archive_names or archive_names[member] == name
                archive_names[member] = name
                archive.writestr(member, raw)
        destination.chmod(0o444)
    snapshot(pre, files)
    command = [request['python'], str(ROOT / request['consumer']), '--request', str(args.request), '--output', str(receipt)]
    start = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT, capture_output=True)
    elapsed = time.monotonic() - start
    with stdout.open('xb') as stream:
        stream.write(completed.stdout)
    with stderr.open('xb') as stream:
        stream.write(completed.stderr)
    unchanged = all((ROOT / name).read_bytes() == raw for name, raw in files.items())
    exit_data = {'command': command, 'returncode': completed.returncode, 'natural_exit': True,
                 'elapsed_seconds': elapsed, 'inputs_unchanged': unchanged}
    with exit_path.open('x') as stream:
        stream.write(json.dumps(exit_data, indent=2) + '\n')
    outputs = {p.name: p.read_bytes() for p in (stdout, stderr, exit_path)}
    if receipt.exists():
        outputs[receipt.name] = receipt.read_bytes()
    snapshot(executed, {**files, **outputs})
    result = {'epoch': args.epoch, 'preexecution': {'path': pre.name, 'sha256': sha(pre.read_bytes())},
              'executed': {'path': executed.name, 'sha256': sha(executed.read_bytes())},
              'request': {'path': args.request.name, 'sha256': sha(args.request.read_bytes())},
              'source_input_pins': request['pins'], 'archive_member_sources': archive_names,
              'outputs': {name: sha(raw) for name, raw in outputs.items()}, 'exit': exit_data,
              'scientific_scope': 'New own exact complete-form/canonical/response controls; not numeric native quantum ground.'}
    with capsule.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    for path in (stdout, stderr, exit_path, capsule, receipt):
        if path.exists():
            path.chmod(0o444)
    print(json.dumps({'epoch': args.epoch, 'returncode': completed.returncode, 'natural_exit': True,
                      'elapsed_seconds': elapsed, 'capsule': capsule.name, 'inputs_unchanged': unchanged}))
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
