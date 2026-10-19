"""Freeze independent review inputs before its own CLI and preserve natural-exit evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def save_json(path: Path, value: dict) -> None:
    with path.open('x') as stream:
        stream.write(json.dumps(value,indent=2)+'\n')


def archive(path: Path, members: dict[str, bytes]) -> dict:
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as output:
        for name,raw in sorted(members.items()):
            output.writestr(name,raw)
    return {'path':path.name,'sha256':sha(path.read_bytes()),
            'members':{name:sha(raw) for name,raw in sorted(members.items())}}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--epoch',required=True)
    args=parser.parse_args()
    if not args.epoch.startswith('full_primitive_graph_review_') or '/' in args.epoch:
        raise ValueError('Only own-prefix simple epoch names are permitted')
    request_raw=args.request.read_bytes()
    request=json.loads(request_raw)
    sources={name:(ROOT/name).read_bytes() for name in request['pins']}
    assert all(sha(raw)==request['pins'][name] for name,raw in sources.items())
    pins_digest=sha(json.dumps(request['pins'],sort_keys=True,separators=(',',':')).encode())
    members={('absolute/'+name.lstrip('/') if name.startswith('/') else name):raw
             for name,raw in sources.items()}
    members[args.request.name]=request_raw
    pre=archive(ROOT/(args.epoch+'_preexecution.zip'),members)
    receipt=ROOT/(args.epoch+'_receipt.json')
    command=[PYTHON,str(ROOT/request['consumer']),'--request',str(args.request.resolve()),'--output',str(receipt)]
    start=time.monotonic()
    execution=subprocess.run(command,cwd=ROOT,capture_output=True,check=False)
    elapsed=time.monotonic()-start
    stdout=ROOT/(args.epoch+'_stdout.txt'); stderr=ROOT/(args.epoch+'_stderr.txt')
    with stdout.open('xb') as stream:stream.write(execution.stdout)
    with stderr.open('xb') as stream:stream.write(execution.stderr)
    unchanged=all((ROOT/name).read_bytes()==raw for name,raw in sources.items())
    exit_path=ROOT/(args.epoch+'_exit.json')
    exit_data={'command':command,'returncode':execution.returncode,'natural_exit':True,
               'elapsed_seconds':elapsed,'inputs_unchanged':unchanged,'source_input_pins_sha256':pins_digest,
               'request_sha256':sha(request_raw),'ground_computed':False,'original_producer_executed':False}
    save_json(exit_path,exit_data)
    for path in (stdout,stderr,exit_path):members[path.name]=path.read_bytes()
    if receipt.exists():members[receipt.name]=receipt.read_bytes()
    executed=archive(ROOT/(args.epoch+'_executed.zip'),members)
    capsule={'kind':'OWN_INDEPENDENT_FULL_PRIMITIVE_GRAPH_REVIEW_EXECUTION_CAPSULE',
             'epoch':args.epoch,'preexecution':pre,'executed':executed,'exit':exit_data,
             'source_input_pins_sha256':pins_digest,'request':args.request.name,
             'receipt':receipt.name if receipt.exists() else None,
             'successful':execution.returncode==0 and unchanged,
             'scientific_admission':False,'campaign_complete':False}
    capsule_path=ROOT/(args.epoch+'_capsule.json')
    save_json(capsule_path,capsule)
    print(json.dumps({'capsule':capsule_path.name,'returncode':execution.returncode,
                      'natural_exit':True,'inputs_unchanged':unchanged,'elapsed_seconds':elapsed}))
    if execution.returncode or not unchanged:
        raise SystemExit(execution.returncode or 1)


if __name__=='__main__':
    main()
