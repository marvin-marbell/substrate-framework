"""Capture before/executed bytes and natural exits for the new charged consumer."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True, type=Path)
    args = parser.parse_args()
    request_path = args.request.resolve()
    root = request_path.parent
    request = json.loads(request_path.read_bytes())
    epoch = request['epoch']
    paths = [root/name for name in request['archive_inputs']] + [request_path]
    before = {path.name:digest(path) for path in paths}
    for name, expected in request['pins'].items():
        assert before[name] == expected
    archive = root/(epoch+'_preexecution.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in paths:
            zipped.write(path,path.name)
    receipt = root/(epoch+'_receipt.json')
    command = [sys.executable,str(root/request['consumer']),'--request',str(request_path),'--output',str(receipt)]
    started = datetime.now(timezone.utc).isoformat(); clock = time.monotonic()
    run = subprocess.run(command,cwd=root,capture_output=True,check=False)
    elapsed = time.monotonic()-clock; ended = datetime.now(timezone.utc).isoformat()
    stdout = root/(epoch+'_stdout.txt'); stderr = root/(epoch+'_stderr.txt')
    with stdout.open('xb') as stream:
        stream.write(run.stdout)
    with stderr.open('xb') as stream:
        stream.write(run.stderr)
    after = {path.name:digest(path) for path in paths}
    exit_path = root/(epoch+'_exit.json')
    save(exit_path,{'returncode':run.returncode,'natural_exit':run.returncode>=0,
                    'started_utc':started,'ended_utc':ended,'elapsed_seconds':elapsed,
                    'command':command,'pins_unchanged':before==after})
    outputs = [stdout,stderr,exit_path] + ([receipt] if receipt.exists() else [])
    executed = root/(epoch+'_executed.zip')
    with zipfile.ZipFile(executed,'x',compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.write(archive,archive.name)
        for path in paths+outputs:
            zipped.write(path,path.name)
    capsule = root/(epoch+'_capsule.json')
    save(capsule,{'kind':'NEW_NATIVE_CHARGED_BODY_IMMUTABLE_NATURAL_EXIT_CAPSULE',
                   'governor':request['governor'],'source_input_pins_before':before,
                   'source_input_pins_after':after,'pins_unchanged':before==after,
                   'before_execution_archive':archive.name,'before_execution_sha256':digest(archive),
                   'executed_archive':executed.name,'executed_sha256':digest(executed),
                   'returncode':run.returncode,'natural_exit':run.returncode>=0,
                   'command':command,'started_utc':started,'ended_utc':ended,
                   'elapsed_seconds':elapsed,'output_pins':{path.name:digest(path) for path in outputs},
                   'old_producer_optimizer_main_root_eigensolver_checker_rerun':False})
    print(json.dumps({'capsule':capsule.name,'returncode':run.returncode,'natural_exit':run.returncode>=0,'pins_unchanged':before==after}))
    raise SystemExit(run.returncode)


if __name__ == '__main__':
    main()
