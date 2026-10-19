"""Run only the new Jordan/CAR consumer with immutable pre/post evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    with path.open('x') as handle:
        handle.write(json.dumps(value,indent=2,sort_keys=True)+'\n')


def archive(path, entries):
    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as handle:
        for name, raw in sorted(entries.items()):
            info = zipfile.ZipInfo(name, (2026,10,12,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            handle.writestr(info,raw)


def execute(request_path, attempt):
    request_raw = request_path.read_bytes()
    request = json.loads(request_raw)
    prefix = 'jordan_loop_form_'+attempt+'_20261012_'
    outputs = {key:ROOT/(prefix+suffix) for key,suffix in {
        'pre':'preexecution.zip','executed':'executed.zip','result':'result.json',
        'stdout':'stdout.txt','stderr':'stderr.txt','exit':'exit.json',
        'receipt':'receipt.json','capsule':'capsule.json'}.items()}
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError('Immutable epoch already exists')
    originals = {name:(ROOT/name).read_bytes() for name in request['input_pins']}
    for name, raw in originals.items():
        assert sha(raw) == request['input_pins'][name], name
    entries = {'inputs/'+Path(name).name:raw for name,raw in originals.items()}
    entries['request/'+request_path.name] = request_raw
    entries['runner/'+Path(__file__).name] = Path(__file__).read_bytes()
    archive(outputs['pre'],entries)
    command = [PYTHON,str(ROOT/request['consumer']),'--request',str(request_path),'--output',str(outputs['result'])]
    start = time.monotonic()
    completed = subprocess.run(command,cwd=ROOT,capture_output=True,check=False)
    elapsed = time.monotonic()-start
    for key, raw in [('stdout',completed.stdout),('stderr',completed.stderr)]:
        with outputs[key].open('xb') as handle:
            handle.write(raw)
    exit_record = {'command':command,'returncode':completed.returncode,
                   'natural_exit':completed.returncode>=0,'signal':-completed.returncode if completed.returncode<0 else None,
                   'elapsed_seconds':elapsed,'timeout':None,'old_producer_executed':False}
    save(outputs['exit'],exit_record)
    unchanged = {name:(ROOT/name).read_bytes()==raw for name,raw in originals.items()}
    unchanged[request_path.name] = request_path.read_bytes()==request_raw
    unchanged[Path(__file__).name] = Path(__file__).read_bytes()==entries['runner/'+Path(__file__).name]
    receipt = {'kind':'SOURCE_SELECTED_ACTUAL_NATURAL_EXIT_JORDAN_CAR_CAPSULE',
               'exit':exit_record,'input_pins':request['input_pins'],'unchanged_after':unchanged,
               'preexecution_zip':{'path':outputs['pre'].name,'sha256':sha(outputs['pre'].read_bytes())},
               'output_pins':{outputs[key].name:sha(outputs[key].read_bytes()) for key in ['stdout','stderr','exit','result'] if outputs[key].exists()}}
    save(outputs['receipt'],receipt)
    for key in ['stdout','stderr','exit','result','receipt']:
        if outputs[key].exists():
            entries['observed/'+outputs[key].name] = outputs[key].read_bytes()
    archive(outputs['executed'],entries)
    capsule = {'kind':'IMMUTABLE_NEW_SCIENCE_EXECUTED_CAPSULE_NOT_PHYSICS_ADMISSION',
               'natural_exit':exit_record,'input_pins':request['input_pins'],
               'artifacts':{outputs[key].name:sha(outputs[key].read_bytes()) for key in outputs if key!='capsule' and outputs[key].exists()},
               'all_pins_unchanged':all(unchanged.values()),'campaign_complete':False}
    save(outputs['capsule'],capsule)
    print(json.dumps({'capsule':outputs['capsule'].name,'returncode':completed.returncode,'elapsed_seconds':elapsed,'all_pins_unchanged':all(unchanged.values())}))
    return completed.returncode if completed.returncode else (0 if all(unchanged.values()) else 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--attempt',required=True)
    args = parser.parse_args()
    raise SystemExit(execute(args.request,args.attempt))
