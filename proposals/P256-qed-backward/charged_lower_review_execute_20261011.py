"""Exclusive raw-byte execution capsules for the independent review scientist."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'
CONTROL='charged_lower_review_control_20261011.py'
PINS={
 'charged_output_lower_certificate_20261011.py':'0ffdd14b7f968acb0170d9cb221723a558b1d9924a3de0ee57725eeb45e4255b',
 'charged_output_lower_attempt3_20261011_receipt.json':'427f5dbcff0fbfbb126c13b7d6c157a0ac6997248af53ce76f00ddc08384f7c2',
 'charged_image_proof_control_20261011.py':'de8173ae6d2a59eb7bab6f7457b04a98c6735941beafacc2e4ef50792430c500',
 'charged_image_proof_control_20261011_receipt.json':'c45fd28aee5c1cd50c2e7350fc3ba5a276029fd118073da80655d88007975590',
 'charged_rooted_source_inverse_guard_20261011_receipt.json':'4d215ba13e1b5e1057569f01b5646d0a7d7efc64f764ba34711454227f8316e4'}
EXTRA=[CONTROL,Path(__file__).name,
 'charged_rooted_source_inverse_guard_20261011.py',
 'charged_first_shell_consumer_20261011.py','charged_first_shell_actual04_20261011_receipt.json',
 'charged_output_lower_attempt3_20261011_natural_exit.json',
 'charged_image_proof_attempt2_20261011_preexecution.json',
 'charged_image_proof_attempt2_20261011_executed.json',
 'charged_image_proof_attempt2_20261011_exit.json',
 'charged_image_proof_attempt2_20261011_recovered_after_execution.zip']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def immutable(path,data):
    with path.open('xb') as out:out.write(data)
    os.chmod(path,0o444)


def encoded(value):
    return (json.dumps(value,indent=2)+'\n').encode()


def capsule(path,members):
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in members.items():archive.writestr(name,data)
    os.chmod(path,0o444)
    return digest(path.read_bytes())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--epoch',required=True);args=parser.parse_args()
    epoch=args.epoch;assert epoch.startswith('charged_lower_review_') and '/' not in epoch
    names=set(PINS)|set(EXTRA)
    lower=json.loads((ROOT/'charged_output_lower_attempt3_20261011_receipt.json').read_bytes())
    image=json.loads((ROOT/'charged_image_proof_control_20261011_receipt.json').read_bytes())
    names.update(lower['input_sha256']);names.update(image['input_hashes'])
    for b in lower['branches']:names.update([b['witness_path'],b['exact_matrix_path']])
    captured={name:(ROOT/name).read_bytes() for name in sorted(names)}
    pins={name:digest(data) for name,data in captured.items()}
    for name,pin in PINS.items():assert pins[name]==pin,(name,pins[name])
    for name,pin in lower['input_sha256'].items():assert pins[name]==pin
    for name,pin in image['input_hashes'].items():assert pins[name]==pin
    manifest_name=f'{epoch}_preexecution.json'
    command=[PYTHON,str(ROOT/CONTROL),'--manifest',manifest_name,'--epoch',epoch]
    manifest={'kind':'EXCLUSIVE_PREEXECUTION_BYTES','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'plan_id':'issue-222-induced-photon-delta13-continuation-20261008',
              'governing_revision':'9a5e0b2cbe54cf4e5fa035b7fec3e31ff0f59bb7ecb7bf97fcea4844f31de2ba',
              'template_id':'general','template_revision':'89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
              'work_items':['induced-photon:charged-lower-independent-review'],
              'command':command,'input_sha256':pins}
    immutable(ROOT/manifest_name,encoded(manifest))
    captured[manifest_name]=(ROOT/manifest_name).read_bytes()
    prearchive=ROOT/f'{epoch}_preexecution.zip';pre_sha=capsule(prearchive,captured)
    start=time.monotonic()
    completed=subprocess.run(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
    elapsed=time.monotonic()-start
    stdout_name=f'{epoch}_stdout.bin';stderr_name=f'{epoch}_stderr.bin'
    immutable(ROOT/stdout_name,completed.stdout);immutable(ROOT/stderr_name,completed.stderr)
    output_names=[p.name for p in ROOT.glob(f'{epoch}_*_proof.json')]
    receipt=ROOT/f'{epoch}_receipt.json'
    if receipt.exists():output_names.append(receipt.name)
    output_bytes={name:(ROOT/name).read_bytes() for name in output_names}
    for name in output_names:os.chmod(ROOT/name,0o444)
    unchanged=all(digest((ROOT/name).read_bytes())==pin for name,pin in pins.items())
    exit_name=f'{epoch}_exit.json'
    exit_record={'kind':'ACTUAL_NATURAL_CHILD_EXIT','returncode':completed.returncode,
                 'natural_exit':completed.returncode>=0,'elapsed_seconds':elapsed,
                 'source_inputs_unchanged':unchanged,'input_sha256':pins,
                 'preexecution_path':prearchive.name,'preexecution_sha256':pre_sha,
                 'stdout_path':stdout_name,'stdout_sha256':digest(completed.stdout),
                 'stderr_path':stderr_name,'stderr_sha256':digest(completed.stderr),
                 'outputs_sha256':{name:digest(data) for name,data in output_bytes.items()}}
    immutable(ROOT/exit_name,encoded(exit_record))
    executed={**captured,**output_bytes,stdout_name:completed.stdout,stderr_name:completed.stderr,
              exit_name:(ROOT/exit_name).read_bytes()}
    exarchive=ROOT/f'{epoch}_executed.zip';ex_sha=capsule(exarchive,executed)
    delivery={'kind':'REVISION_BOUND_REVIEW_EPOCH','returncode':completed.returncode,
              'natural_exit':completed.returncode>=0,'source_inputs_unchanged':unchanged,
              'preexecution_archive':prearchive.name,'preexecution_sha256':pre_sha,
              'executed_archive':exarchive.name,'executed_sha256':ex_sha,
              'exit_path':exit_name,'exit_sha256':digest((ROOT/exit_name).read_bytes()),
              'outputs_sha256':exit_record['outputs_sha256']}
    immutable(ROOT/f'{epoch}_delivery.json',encoded(delivery))
    print(json.dumps(delivery,indent=2),flush=True)
    if completed.returncode!=0:print(completed.stderr.decode(errors='replace'),flush=True)
    raise SystemExit(completed.returncode if completed.returncode>=0 else 1)


if __name__=='__main__':main()
