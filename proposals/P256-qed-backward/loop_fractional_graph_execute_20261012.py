"""Capture selected new consumer bytes before execution and natural exit afterward."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PYTHON = Path('/home/administrator/substrate-framework/.venv/bin/python')


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def save_json(path: Path, value: dict) -> None:
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2)+'\n')


def snapshot(path: Path, captured: dict[str, bytes], manifest: dict) -> None:
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in captured.items():
            # Absolute contract paths are preserved in manifest and made safe in ZIP.
            archive.writestr('selected/'+name.lstrip('/'), raw)
        archive.writestr('manifest.json', json.dumps(manifest, indent=2)+'\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--epoch', required=True)
    args = parser.parse_args()
    request_raw = args.request.read_bytes()
    request = json.loads(request_raw)
    captured = {name: (ROOT/name).read_bytes() for name in request['input_pins']}
    assert all(sha(raw) == request['input_pins'][name] for name, raw in captured.items())
    captured[str(args.request.resolve())] = request_raw
    python_resolved = PYTHON.resolve()
    python_hash = sha(python_resolved.read_bytes())
    prefix = ROOT/('loop_fractional_graph_'+args.epoch+'_20261012')
    result = Path(str(prefix)+'_result.json')
    command = [str(PYTHON), str(ROOT/request['consumer']), '--request', str(args.request.resolve()),
               '--output', str(result)]
    manifest = {'command': command, 'cwd': str(ROOT), 'selected_hashes': {n: sha(b) for n, b in captured.items()},
                'python_requested': str(PYTHON), 'python_resolved': str(python_resolved),
                'python_executable_sha256': python_hash, 'preexecution_wall_ns': time.time_ns()}
    pre = Path(str(prefix)+'_preexecution.zip')
    snapshot(pre, captured, manifest)
    started = time.monotonic()
    # No timeout, injected signal, harness stop, old entrypoint, or fabricated exit.
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
    elapsed = time.monotonic()-started
    for suffix, raw in [('stdout.txt', completed.stdout), ('stderr.txt', completed.stderr)]:
        with Path(str(prefix)+'_'+suffix).open('xb') as stream:
            stream.write(raw)
    after = {name: (ROOT/name).read_bytes() for name in captured}
    unchanged = after == captured
    exit_record = {'command': command, 'returncode': completed.returncode, 'natural_exit': completed.returncode >= 0,
                   'elapsed_seconds': elapsed, 'selected_bytes_unchanged': unchanged,
                   'python_executable_unchanged': sha(python_resolved.read_bytes()) == python_hash,
                   'preexecution_sha256': sha(pre.read_bytes()),
                   'result_exists': result.exists(), 'result_sha256': sha(result.read_bytes()) if result.exists() else None}
    exit_path = Path(str(prefix)+'_exit.json')
    save_json(exit_path, exit_record)
    executed = dict(captured)
    for path in [exit_path, Path(str(prefix)+'_stdout.txt'), Path(str(prefix)+'_stderr.txt')]:
        executed[path.name] = path.read_bytes()
    if result.exists():
        executed[result.name] = result.read_bytes()
    post = Path(str(prefix)+'_executed.zip')
    snapshot(post, executed, exit_record)
    capsule = dict(exit_record)
    capsule.update({'preexecution': str(pre), 'executed': str(post), 'executed_sha256': sha(post.read_bytes()),
                    'selected_hashes': manifest['selected_hashes'], 'exit': str(exit_path), 'result': str(result)})
    save_json(Path(str(prefix)+'_capsule.json'), capsule)
    print(json.dumps(capsule, indent=2))
    if completed.returncode != 0 or not unchanged or not capsule['python_executable_unchanged']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
