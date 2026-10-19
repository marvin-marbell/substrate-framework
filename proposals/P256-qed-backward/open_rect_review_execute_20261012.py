"""Freeze independent reviewer inputs before the new scientific consumer."""
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


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    args.request = args.request.resolve()
    request = json.loads(args.request.read_text())
    root = args.request.parent
    prefix = root/request['epoch']
    source = root/request['source']
    paths = [Path(p) for p in request['pins']]+[args.request]
    names = [p.name for p in paths]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate archive member')
    for p in paths[:-1]:
        if sha(p) != request['pins'][str(p)]:
            raise ValueError(f'Input changed before execution: {p}')
    before = {str(p):sha(p) for p in paths}
    pre = Path(str(prefix)+'_preexecution.zip')
    with zipfile.ZipFile(pre,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p,p.name)
    pre_hash = sha(pre)
    command = [sys.executable,str(source),'--request',str(args.request),'--output',str(prefix)+'_receipt.json']
    start = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    process = subprocess.run(command,cwd=root,capture_output=True,text=True)
    end = datetime.now(timezone.utc).isoformat()
    elapsed = time.monotonic()-clock
    stdout = Path(str(prefix)+'_stdout.txt')
    stderr = Path(str(prefix)+'_stderr.txt')
    stdout.write_text(process.stdout)
    stderr.write_text(process.stderr)
    after = {str(p):sha(p) for p in paths}
    unchanged = before == after and sha(pre) == pre_hash
    runtime = {'command':command,'cwd':str(root),'start_utc':start,'end_utc':end,'elapsed_seconds':elapsed,
               'returncode':process.returncode,'natural_exit':True,'timeout':False,'source_sha256':before[str(source)],
               'request_sha256':before[str(args.request)],'input_pins':request['pins'],'inputs_unchanged':unchanged,
               'preexecution_sha256':pre_hash,'stdout_sha256':sha(stdout),'stderr_sha256':sha(stderr)}
    exit_path = Path(str(prefix)+'_exit.json')
    exit_path.write_text(json.dumps(runtime,indent=2)+'\n')
    receipt = Path(str(prefix)+'_receipt.json')
    outputs = [stdout,stderr,exit_path]+([receipt] if receipt.exists() else [])
    executed = Path(str(prefix)+'_executed.zip')
    with zipfile.ZipFile(executed,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in paths+outputs:
            z.write(p,p.name)
    delivery = {'kind':'INDEPENDENT_OPEN_RECT_REVIEW_NEW_CONSUMER_IMMUTABLE_EPOCH',
                'epoch':request['epoch'],'runtime':runtime,'preexecution':{'path':str(pre),'sha256':pre_hash,'members':names},
                'executed':{'path':str(executed),'sha256':sha(executed),'members':[p.name for p in paths+outputs]},
                'outputs':{str(p):sha(p) for p in outputs},'proof':str(root/request['proof']),
                'scientific_admission':False,'campaign_complete':False}
    Path(str(prefix)+'_delivery.json').write_text(json.dumps(delivery,indent=2)+'\n')
    print(json.dumps({'returncode':process.returncode,'natural_exit':True,'inputs_unchanged':unchanged,'preexecution_sha256':pre_hash,'executed_sha256':sha(executed),'elapsed_seconds':elapsed}))
    if process.returncode:
        print(process.stderr)
    raise SystemExit(process.returncode if unchanged else 99)


if __name__ == '__main__':
    main()
