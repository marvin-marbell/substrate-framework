"""Capture request/input bytes before the independent mathematical CLI executes."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2)+'\n')


def archive(path, root, members):
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as output:
        for member in members:
            output.write(root/member, arcname=member)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_bytes())
    root = args.request.parent
    epoch = request['epoch']
    members = list(request['input_pins'])+[args.request.name]
    before_pins = {name: sha(root/name) for name in members}
    assert all(before_pins[name] == value for name, value in request['input_pins'].items())
    before = root/(epoch+'_preexecution.zip')
    archive(before, root, members)
    receipt = epoch+'_receipt.json'
    command = [request['python'], str(root/request['consumer']), '--request', str(args.request), '--output', str(root/receipt)]
    started = datetime.now(timezone.utc).isoformat()
    tick = time.monotonic()
    process = subprocess.run(command, cwd=root, capture_output=True, check=False)
    elapsed = time.monotonic()-tick
    ended = datetime.now(timezone.utc).isoformat()
    (root/(epoch+'_stdout.txt')).write_bytes(process.stdout)
    (root/(epoch+'_stderr.txt')).write_bytes(process.stderr)
    exit_name = epoch+'_exit.json'
    save(root/exit_name, {'returncode': process.returncode, 'natural_exit': process.returncode >= 0,
                         'started_utc': started, 'ended_utc': ended, 'elapsed_seconds': elapsed,
                         'command': command, 'forced_termination': False})
    after_pins = {name: sha(root/name) for name in members}
    outputs = [epoch+'_stdout.txt', epoch+'_stderr.txt', exit_name]
    if (root/receipt).exists():
        outputs.append(receipt)
    executed = root/(epoch+'_executed.zip')
    archive(executed, root, members+outputs)
    capsule = {'kind': 'INDEPENDENT_CHARGED_HISTOGRAM_NATURAL_EXIT_CAPSULE',
               'governor': request['governor'], 'source_input_pins_before': before_pins,
               'source_input_pins_after': after_pins, 'pins_unchanged': before_pins == after_pins,
               'before_execution_archive': before.name, 'before_execution_sha256': sha(before),
               'executed_archive': executed.name, 'executed_sha256': sha(executed),
               'command': command, 'returncode': process.returncode, 'natural_exit': process.returncode >= 0,
               'started_utc': started, 'ended_utc': ended, 'elapsed_seconds': elapsed,
               'output_pins': {name: sha(root/name) for name in outputs},
               'original_producer_diagonalizer_rerun': False, 'scientific_admission': False}
    save(root/(epoch+'_capsule.json'), capsule)
    print(json.dumps({'returncode': process.returncode, 'natural_exit': process.returncode >= 0,
                      'pins_unchanged': capsule['pins_unchanged'], 'capsule': epoch+'_capsule.json'}))
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
