"""Execute only the NEW independent review; retain every failed epoch."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'
CONTROL='vacuum_operator_review_control_20261011.py'
SNAPSHOT='vacuum_weyl_parent_review_original_targets_20261011.zip'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def encode(value):
    return (json.dumps(value,sort_keys=True,indent=2)+'\n').encode()

def deterministic_zip(path,members):
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(members.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,10,11,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,data)

def run(epoch):
    prefix=f'vacuum_operator_review_{epoch}_'
    suffixes=('request.json','preexecution.zip','result.json','stdout.txt','stderr.txt','exit.json','executed.zip','capsule.json')
    paths={suffix:ROOT/(prefix+suffix) for suffix in suffixes}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('Immutable review epoch exists')
    inputs={name:(ROOT/name).read_bytes() for name in (CONTROL,Path(__file__).name,SNAPSHOT)}
    original_capsule=json.loads(zipfile.ZipFile(ROOT/SNAPSHOT).read('vacuum_connected_native_actual2_20261011_capsule.json'))
    original_executed=original_capsule['executed']['path']
    pins={name:sha(data) for name,data in inputs.items()}
    pins[original_executed]=sha((ROOT/original_executed).read_bytes())
    request={'kind':'NEW_INDEPENDENT_REVIEW_NOT_TARGET_RERUN','epoch':epoch,'python':PYTHON,'source_input_sha256':pins,'plan_id':'issue-222-induced-photon-delta13-continuation-20261008','captured_revision':'74c9314ae6e7985f85615b94c5b190c8b8b356be00b78722a1ebd7444ef47190','template_revision':'89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe','workitem':'induced-photon'}
    paths['request.json'].write_bytes(encode(request))
    deterministic_zip(paths['preexecution.zip'],{**inputs,'request.json':paths['request.json'].read_bytes()})
    completed=subprocess.run([PYTHON,str(ROOT/CONTROL),'--output',str(paths['result.json'])],cwd=ROOT,capture_output=True,check=False)
    paths['stdout.txt'].write_bytes(completed.stdout);paths['stderr.txt'].write_bytes(completed.stderr)
    unchanged=all(sha((ROOT/name).read_bytes())==pin for name,pin in pins.items())
    exit_record={'kind':'NEW_REVIEW_NATURAL_EXIT','returncode':completed.returncode,'natural_exit':completed.returncode>=0,'sources_inputs_unchanged':unchanged,'source_input_sha256':pins,'stdout_sha256':sha(completed.stdout),'stderr_sha256':sha(completed.stderr),'old_entrypoint_checker_rerun':False}
    paths['exit.json'].write_bytes(encode(exit_record))
    members={**inputs,**{suffix:path.read_bytes() for suffix,path in paths.items() if suffix in ('request.json','result.json','stdout.txt','stderr.txt','exit.json') and path.exists()}}
    deterministic_zip(paths['executed.zip'],members)
    capsule={**exit_record,'preexecution':{'path':paths['preexecution.zip'].name,'sha256':sha(paths['preexecution.zip'].read_bytes())},'executed':{'path':paths['executed.zip'].name,'sha256':sha(paths['executed.zip'].read_bytes())},'result':{'path':paths['result.json'].name,'sha256':sha(paths['result.json'].read_bytes())} if paths['result.json'].exists() else None,'exit':{'path':paths['exit.json'].name,'sha256':sha(paths['exit.json'].read_bytes())},'conditional_machine_scope':'Exact new symbolic/rational controls. Historical target binary64/formation conditions are not retroactively attested.'}
    paths['capsule.json'].write_bytes(encode(capsule))
    print(json.dumps(capsule,sort_keys=True),flush=True)
    return completed.returncode if unchanged else 2

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--epoch',required=True)
    sys.exit(run(p.parse_args().epoch))
