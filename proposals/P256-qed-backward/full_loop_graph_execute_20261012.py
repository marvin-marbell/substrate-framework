"""Capture immutable own source/input epochs and natural scientific CLI exit."""
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


def save(path: Path, raw: bytes) -> None:
    with path.open('xb') as stream:
        stream.write(raw)


def archive(path: Path, captured: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in captured.items():
            z.writestr(name,raw)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--epoch',required=True)
    args=parser.parse_args()
    assert args.epoch.startswith('full_loop_graph_') and '/' not in args.epoch
    request_raw=args.request.read_bytes()
    request=json.loads(request_raw)
    captured={name:(ROOT/name).read_bytes() for name in request['pins']}
    assert all(sha(raw)==request['pins'][name] for name,raw in captured.items())
    captured[args.request.name]=request_raw
    prefix=ROOT/args.epoch
    output=Path(str(prefix)+'_receipt.json')
    pre=Path(str(prefix)+'_preexecution.zip')
    executed=Path(str(prefix)+'_executed.zip')
    capsule=Path(str(prefix)+'_capsule.json')
    exit_path=Path(str(prefix)+'_exit.json')
    stdout_path=Path(str(prefix)+'_stdout.txt')
    stderr_path=Path(str(prefix)+'_stderr.txt')
    assert not any(path.exists() for path in (output,pre,executed,capsule,exit_path,stdout_path,stderr_path))
    archive(pre,captured)
    command=[PYTHON,str(ROOT/request['consumer']),'--request',str(args.request),'--output',str(output)]
    started=time.monotonic()
    process=subprocess.run(command,cwd=ROOT,capture_output=True,check=False)
    elapsed=time.monotonic()-started
    save(stdout_path,process.stdout);save(stderr_path,process.stderr)
    unchanged=all((ROOT/name).read_bytes()==raw for name,raw in captured.items())
    exit_record={'command':command,'returncode':process.returncode,
                 'natural_exit':process.returncode>=0,'elapsed_seconds':elapsed,
                 'inputs_unchanged':unchanged,'request_sha256':sha(request_raw)}
    save(exit_path,(json.dumps(exit_record,indent=2)+'\n').encode())
    outputs={path.name:path.read_bytes() for path in (stdout_path,stderr_path,exit_path)}
    if output.exists():
        outputs[output.name]=output.read_bytes()
    archive(executed,{**captured,**outputs})
    record={'kind':'OWN_SCIENTIFIC_IMMUTABLE_EPOCH_CAPSULE','epoch':args.epoch,
            'exit':exit_record,'source_input_sha256':{name:sha(raw) for name,raw in captured.items()},
            'output_sha256':{name:sha(raw) for name,raw in outputs.items()},
            'preexecution':{'path':pre.name,'sha256':sha(pre.read_bytes())},
            'executed':{'path':executed.name,'sha256':sha(executed.read_bytes())},
            'receipt_produced':output.exists(),'failed_epoch_retained':process.returncode!=0,
            'numerical_native_ground':False,'numerical_fullH_evolution':False,
            'scientific_admission':False,'campaign_complete':False}
    save(capsule,(json.dumps(record,indent=2)+'\n').encode())
    print(json.dumps({'capsule':capsule.name,'returncode':process.returncode,
                      'natural_exit':process.returncode>=0,'inputs_unchanged':unchanged}))
    if process.returncode !=0:
        raise SystemExit(process.returncode)
    assert unchanged and output.exists()


if __name__=='__main__':
    main()
