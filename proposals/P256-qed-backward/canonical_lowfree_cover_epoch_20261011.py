from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

from canonical_lowfree_cover_control_20261011 import PINS

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'
SOURCE = ROOT / 'canonical_lowfree_cover_control_20261011.py'
LANGUAGE = ROOT / 'canonical_lowfree_cover_language_20261011.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    assert args.tag.startswith('canonical_lowfree_cover_')
    command = [PYTHON, str(SOURCE), '--tag', args.tag]
    sources = [SOURCE, Path(__file__), LANGUAGE]
    source_pins = {path.name: digest(path) for path in sources}
    input_pins = {name: digest(ROOT / name) for name in PINS}
    assert input_pins == PINS
    pre = ROOT / (args.tag + '_preexecution.zip')
    with zipfile.ZipFile(pre, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, path.name)
        for name in PINS:
            archive.write(ROOT / name, name)
        archive.writestr('manifest.json', json.dumps({'command': command, 'source_sha256': source_pins,
                                                      'input_sha256': input_pins,
                                                      'old_solver_or_checker_execution': False}, indent=2))
    pre.chmod(0o444)
    started = time.time()
    stdout = ROOT / (args.tag + '_stdout.log')
    stderr = ROOT / (args.tag + '_stderr.log')
    with stdout.open('xb') as out, stderr.open('xb') as err:
        process = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err,
                                 env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'), check=False)
    ended = time.time()
    receipt = ROOT / (args.tag + '_receipt.json')
    outputs = [stdout, stderr] + ([receipt] if receipt.exists() else [])
    natural = {'command': command, 'started_unix': started, 'ended_unix': ended,
               'natural_exit_code': process.returncode, 'signal_exit': process.returncode < 0,
               'source_sha256': source_pins, 'input_sha256': input_pins,
               'sources_unchanged': all(digest(path) == source_pins[path.name] for path in sources),
               'inputs_unchanged': all(digest(ROOT / name) == pin for name, pin in input_pins.items()),
               'preexecution_zip_sha256': digest(pre),
               'output_sha256': {path.name: digest(path) for path in outputs}}
    exit_path = ROOT / (args.tag + '_natural_exit.json')
    with exit_path.open('x') as file:
        json.dump(natural, file, indent=2)
        file.write('\n')
    outputs.append(exit_path)
    for path in outputs:
        path.chmod(0o444)
    executed = ROOT / (args.tag + '_executed.zip')
    with zipfile.ZipFile(executed, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(pre, pre.name)
        for path in outputs:
            archive.write(path, path.name)
    executed.chmod(0o444)
    print(json.dumps({'tag': args.tag, 'natural_exit_code': process.returncode,
                      'elapsed_seconds': ended - started, 'executed_zip_sha256': digest(executed),
                      'receipt_present': receipt.exists()}), flush=True)
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
