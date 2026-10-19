"""Freeze own scientific controls, execute once to natural exit, preserve the epoch."""
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


def save(path: Path, raw: bytes) -> None:
    with path.open('xb') as stream:
        stream.write(raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--python', required=True)
    args = parser.parse_args()
    request_raw = args.request.read_bytes()
    request = json.loads(request_raw)
    epoch = request['epoch']
    captured = {name: (ROOT/name).read_bytes() for name in request['source_input_sha256']}
    for name, raw in captured.items():
        if sha(raw) != request['source_input_sha256'][name]:
            raise ValueError('Input hash mismatch: '+name)
    captured[args.request.name] = request_raw
    pre = ROOT/(epoch+'_preexecution.zip')
    with zipfile.ZipFile(pre, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, raw in captured.items():
            archive.writestr(name, raw)
    receipt = ROOT/(epoch+'_receipt.json')
    command = [args.python, str(ROOT/request['consumer']), '--request', str(args.request.resolve()),
               '--output', str(receipt)]
    start = time.monotonic()
    process = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
    unchanged = all((ROOT/name).read_bytes() == raw for name, raw in captured.items())
    exit_record = {'command': command, 'returncode': process.returncode,
                   'natural_exit': process.returncode >= 0, 'elapsed_seconds': time.monotonic()-start,
                   'inputs_unchanged': unchanged, 'request_sha256': sha(request_raw)}
    output = {epoch+'_stdout.txt': process.stdout, epoch+'_stderr.txt': process.stderr,
              epoch+'_exit.json': (json.dumps(exit_record, indent=2)+'\n').encode()}
    if receipt.exists():
        output[receipt.name] = receipt.read_bytes()
    for name, raw in output.items():
        if name != receipt.name:
            save(ROOT/name, raw)
    executed = ROOT/(epoch+'_executed.zip')
    with zipfile.ZipFile(executed, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, raw in {**captured, **output}.items():
            archive.writestr(name, raw)
    capsule = {'kind': 'INDEPENDENT_FRACTIONAL_VIRIAL_REVIEW_IMMUTABLE_EPOCH', 'epoch': epoch,
               'exit': exit_record, 'source_input_sha256': {name: sha(raw) for name, raw in captured.items()},
               'output_sha256': {name: sha(raw) for name, raw in output.items()},
               'preexecution': {'path': pre.name, 'sha256': sha(pre.read_bytes())},
               'executed': {'path': executed.name, 'sha256': sha(executed.read_bytes())},
               'receipt_produced': receipt.exists(), 'failed_epoch_retained': process.returncode != 0,
               'native_ground_computed': False, 'native_evolution_computed': False,
               'scientific_admission': False, 'campaign_complete': False}
    save(ROOT/(epoch+'_capsule.json'), (json.dumps(capsule, indent=2)+'\n').encode())
    print(json.dumps({'epoch': epoch, 'exit': exit_record, 'capsule': epoch+'_capsule.json'}))
    if process.returncode != 0 or not unchanged or not receipt.exists():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
