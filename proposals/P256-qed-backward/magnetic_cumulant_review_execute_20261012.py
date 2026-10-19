"""Freeze and execute only the independent review scientific control."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, data: dict):
    with path.open('x') as stream:
        json.dump(data,stream,indent=2,sort_keys=True)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--epoch', required=True)
    args = parser.parse_args()
    assert args.epoch.startswith('magnetic_cumulant_review_') and '/' not in args.epoch
    request = json.loads(args.request.read_text())
    sources = {Path(name):expected for name,expected in request['source_input_SHA256'].items()}
    for path, expected in sources.items():
        assert digest(path) == expected, f'Frozen input mismatch: {path}'
    for path in (args.request,Path(__file__)):
        sources[path.resolve()] = digest(path)
    selected = {}
    pre = ROOT/(args.epoch+'_preexecution.zip')
    with zipfile.ZipFile(pre,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path, expected in sources.items():
            member = 'source_input/'+path.name
            assert member not in selected
            archive.writestr(member,path.read_bytes())
            selected[member] = expected
    output = ROOT/(args.epoch+'_receipt.json')
    command = [PYTHON,str(ROOT/'magnetic_cumulant_review_control_20261012.py'),'--request',str(args.request.resolve()),'--output',str(output)]
    start = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    child = subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout, stderr = child.communicate()
    elapsed = time.monotonic()-clock
    finished = datetime.now(timezone.utc).isoformat()
    logs = []
    for label, data in (('stdout',stdout),('stderr',stderr)):
        path = ROOT/(args.epoch+'_'+label+'.txt')
        with path.open('xb') as stream:
            stream.write(data)
        logs.append(path)
    unchanged = all(digest(path) == expected for path,expected in sources.items())
    exit_path = ROOT/(args.epoch+'_exit.json')
    save(exit_path,{'command':command,'pid':child.pid,'started_UTC':start,'finished_UTC':finished,
                    'elapsed_seconds':elapsed,'natural_exit':True,'returncode':child.returncode,
                    'source_inputs_unchanged_after_child':unchanged,'original_producer_executed':False})
    executed = ROOT/(args.epoch+'_executed.zip')
    outputs = logs+[exit_path]+([output] if output.exists() else [])
    with zipfile.ZipFile(executed,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        with zipfile.ZipFile(pre) as before:
            for name in before.namelist():
                archive.writestr(name,before.read(name))
        for path in outputs:
            archive.writestr('execution/'+path.name,path.read_bytes())
    def record(path):
        return {'path':path.name,'sha256':digest(path)}
    capsule = {'kind':'INDEPENDENT_EXACT_CUMULANT_REVIEW_NATURAL_EXIT_CAPSULE','epoch':args.epoch,
               'preexecution':record(pre),'executed':record(executed),'request':record(args.request),
               'exit':record(exit_path),'stdout':record(logs[0]),'stderr':record(logs[1]),
               'receipt':record(output) if output.exists() else None,
               'selected_member_SHA256':selected,'source_input_SHA256':request['source_input_SHA256'],
               'natural_exit':True,'returncode':child.returncode,'source_inputs_unchanged_after_child':unchanged,
               'original_targets_edited_or_executed':False,'actual_ground_numeric_backend':False,
               'scientific_admission':False,'photon_or_phase_or_adoption':False}
    capsule_path = ROOT/(args.epoch+'_capsule.json')
    save(capsule_path,capsule)
    print(json.dumps({'capsule':record(capsule_path),'returncode':child.returncode,'source_unchanged':unchanged}))
    if child.returncode != 0 or not unchanged:
        raise SystemExit(child.returncode or 1)


if __name__ == '__main__':
    main()
