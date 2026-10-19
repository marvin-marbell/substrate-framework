"""Archive a genuinely new review epoch before executing only its own consumer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = sys.argv[1]
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    request_path = ROOT/(PREFIX+'_request.json')
    request = json.loads(request_path.read_text())
    consumer = ROOT/'body_probability_review_consumer_20261012.py'
    members = {name:Path(pin['path']) for name,pin in request['pins'].items()}
    members.update({request_path.name:request_path, consumer.name:consumer,
                    Path(__file__).name:Path(__file__)})
    if len(members) != len(request['pins'])+3:
        raise ValueError('Duplicate epoch members')
    for name, pin in request['pins'].items():
        if digest(Path(pin['path'])) != pin['sha256']:
            raise ValueError(f'Input pin changed before preexecution: {name}')
    before = {name:digest(path) for name,path in members.items()}
    pre = ROOT/(PREFIX+'_preexecution.zip')
    executed = ROOT/(PREFIX+'_executed.zip')
    if pre.exists() or executed.exists():
        raise ValueError('Immutable epoch already exists; choose a successor rather than replay')
    with zipfile.ZipFile(pre,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,path in members.items(): z.write(path,name)
    output = ROOT/(PREFIX+'_receipt.json')
    command = [PYTHON,str(consumer),'--request',str(request_path),'--output',str(output)]
    started = time.time_ns()
    process = subprocess.run(command,cwd=ROOT,capture_output=True,check=False)
    ended = time.time_ns()
    stdout = ROOT/(PREFIX+'_stdout.txt'); stderr = ROOT/(PREFIX+'_stderr.txt')
    stdout.write_bytes(process.stdout); stderr.write_bytes(process.stderr)
    unchanged = {name:digest(path)==before[name] for name,path in members.items()}
    natural_path = ROOT/(PREFIX+'_natural_exit.json')
    natural = {'kind':'NEW_EXACT_REVIEW_CONSUMER_NATURAL_EXIT','command':command,'cwd':str(ROOT),
               'started_unix_ns':started,'ended_unix_ns':ended,'returncode':process.returncode,
               'natural_exit':process.returncode>=0,'killed':False,'timeout_used':False,
               'stdout_sha256':digest(stdout),'stderr_sha256':digest(stderr),'stderr_empty':not process.stderr,
               'preexecution_path':str(pre),'preexecution_sha256':digest(pre),
               'preexecution_is_preexecution':True,'member_pins_before':before,'member_pins_unchanged':unchanged,
               'source_sha256':digest(consumer),'request_sha256':digest(request_path)}
    natural_path.write_text(json.dumps(natural,indent=2)+'\n')
    executed_members = dict(members)
    executed_members.update({pre.name:pre,stdout.name:stdout,stderr.name:stderr,natural_path.name:natural_path})
    if output.exists():executed_members[output.name]=output
    with zipfile.ZipFile(executed,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,path in executed_members.items():z.write(path,name)
    delivery = {'kind':'IMMUTABLE_INDEPENDENT_BODY_PROBABILITY_REVIEW_EPOCH_DELIVERY',
                'returncode':process.returncode,'natural_exit':natural['natural_exit'],'stderr_empty':not process.stderr,
                'input_source_pins_unchanged':all(unchanged.values()),'preexecution':{'path':str(pre),'sha256':digest(pre),'is_preexecution':True,'unique_members':len(members)},
                'executed':{'path':str(executed),'sha256':digest(executed),'unique_members':len(executed_members)},
                'runtime':{'path':str(natural_path),'sha256':digest(natural_path)},
                'source':{'path':str(consumer),'sha256':digest(consumer)},
                'request':{'path':str(request_path),'sha256':digest(request_path)},
                'proof':{'path':str(ROOT/'body_probability_review_proof_20261012.json'),'sha256':digest(ROOT/'body_probability_review_proof_20261012.json')},
                'receipt':{'path':str(output),'sha256':digest(output)} if output.exists() else None,
                'scientific_admission':False,'originals_rerun':False}
    (ROOT/(PREFIX+'_delivery.json')).write_text(json.dumps(delivery,indent=2)+'\n')
    print(json.dumps(delivery,indent=2),flush=True)
    if process.returncode or not all(unchanged.values()):sys.exit(process.returncode or 1)


if __name__ == '__main__':
    main()
