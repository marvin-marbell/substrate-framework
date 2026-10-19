"""Immutable capture and natural-exit execution for the new scientific consumer."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--epoch',required=True)
    args=parser.parse_args()
    assert args.epoch.startswith('magnetic_connected_limit_')
    request=json.loads(args.request.read_text())
    pins=request['pins']
    for name,pin in pins.items():
        assert digest(ROOT/name) == pin,name
    members=list(pins)+[args.request.name]
    pre=ROOT/(args.epoch+'_preexecution.zip')
    with zipfile.ZipFile(pre,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in members:
            archive.write(ROOT/name,name)
    output=ROOT/(args.epoch+'_receipt.json')
    command=[request['python'],str(ROOT/request['consumer']),'--request',str(args.request),'--output',str(output)]
    start=time.monotonic()
    completed=subprocess.run(command,cwd=ROOT,capture_output=True,check=False)
    elapsed=time.monotonic()-start
    stdout=ROOT/(args.epoch+'_stdout.txt')
    stderr=ROOT/(args.epoch+'_stderr.txt')
    stdout.write_bytes(completed.stdout)
    stderr.write_bytes(completed.stderr)
    unchanged=all(digest(ROOT/name)==pin for name,pin in pins.items())
    exit_path=ROOT/(args.epoch+'_exit.json')
    exit_data={'command':command,'returncode':completed.returncode,'natural_exit':completed.returncode>=0,
               'elapsed_seconds':elapsed,'source_inputs_unchanged':unchanged,
               'request_sha256':digest(args.request),'source_input_sha256':pins,
               'preexecution_sha256':digest(pre),'receipt_sha256':digest(output) if output.exists() else None}
    with exit_path.open('x') as stream:
        json.dump(exit_data,stream,indent=2);stream.write('\n')
    executed=ROOT/(args.epoch+'_executed.zip')
    with zipfile.ZipFile(executed,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in members+[stdout.name,stderr.name,exit_path.name]+([output.name] if output.exists() else []):
            archive.write(ROOT/name,name)
    capsule={'kind':'IMMUTABLE_NEW_SCIENTIFIC_CONSUMER_EXECUTION_CAPSULE','epoch':args.epoch,
             'source_input_sha256':pins,'request_sha256':digest(args.request),
             'preexecution':pre.name,'preexecution_sha256':digest(pre),
             'executed':executed.name,'executed_sha256':digest(executed),
             'exit':exit_path.name,'exit_sha256':digest(exit_path),
             'receipt':output.name if output.exists() else None,
             'receipt_sha256':digest(output) if output.exists() else None,
             'returncode':completed.returncode,'natural_exit':completed.returncode>=0,
             'source_inputs_unchanged':unchanged,'old_entrypoints_rerun':False}
    with (ROOT/(args.epoch+'_capsule.json')).open('x') as stream:
        json.dump(capsule,stream,indent=2);stream.write('\n')
    print(json.dumps(capsule))
    if completed.returncode:
        print(completed.stderr.decode(errors='replace'))
        raise SystemExit(completed.returncode)
    if not unchanged:
        raise RuntimeError('Pinned input changed during new consumer execution')


if __name__ == '__main__':
    main()
