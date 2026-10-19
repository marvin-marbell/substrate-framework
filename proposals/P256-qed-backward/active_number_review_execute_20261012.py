"""Archive exact new consumer/input bytes before natural scientific execution."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    prefix = f'active_number_review_{args.epoch}_20261012'
    request_path = root/f'{prefix}_request.json'
    receipt = root/f'{prefix}_receipt.json'
    pre = root/f'{prefix}_preexecution.zip'
    executed = root/f'{prefix}_executed.zip'
    request = {
        'snapshot': 'auxiliary_sector_energy_parent_review_postexecution_snapshot_20261012.zip',
        'snapshot_sha256': '62b2887907844c87e5e1f7859c1f332708e62856484ad8c7fa9d1b64d4c17763',
        'frozen_helper_sha256': '63e92ef53f5f4906e0b1c253513a5187ba27e9c05cb942791329f3d6dcd22d23',
        'profile_member': 'joined_phi_u1_stationary_flat_holonomy_typed_certified_20261010_receipt.json',
        'profile_sha256': '1c79b1a74c4236079645ebf9765a48c1c2263ec7b7209e9f103704c98752da94',
        'contract_sha256': '91986c1abb6655873316a644b83b714ddc556286ffc0358d9946f8beb5279105',
        'governor': {'plan_id': 'issue-222-induced-photon-delta13-continuation-20261008', 'revision': 'c65d409ab76891a52043391a5637e80041f453aa60d4a0634ffe87f265d716d6', 'template_id': 'general', 'template_revision': '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe', 'work_item_id': 'induced-photon', 'requirements': ['outcome', 'verification'], 'parent_only_plan_writer': True},
        'epoch': args.epoch, 'consumer': 'active_number_review_consumer_20261012.py'
    }
    with request_path.open('x') as stream:
        json.dump(request, stream, indent=2); stream.write('\n')
    sources = [root/request['consumer'], Path(__file__).resolve(), request_path, root/request['snapshot'], root/'active_number_review_language_20261012.json']
    contract = Path('/home/administrator/.omp/agent/sessions/-/2026-10-09T16-26-59-293Z_01a1217d-415d-7376-a17d-16be721cbe28/local/issue222-original-auxiliary-trial-number-response-proof-review-contract-20261012.json')
    assert sha(contract) == request['contract_sha256']
    sources.append(contract)
    for previous in sorted(root.glob('active_number_review_actual*_20261012_capsule.json')):
        sources.append(previous)
    pins_before = {str(p): sha(p) for p in sources}
    with zipfile.ZipFile(pre, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, path.name)
    pre_sha = sha(pre)
    command = ['/home/administrator/substrate-framework/.venv/bin/python', str(root/request['consumer']), '--request', str(request_path), '--output', str(receipt)]
    started = time.time()
    proc = subprocess.run(command, cwd=root, capture_output=True, check=False)
    elapsed = time.time()-started
    stdout = root/f'{prefix}_stdout.txt'; stderr = root/f'{prefix}_stderr.txt'; exit_path = root/f'{prefix}_exit.json'
    with stdout.open('xb') as stream: stream.write(proc.stdout)
    with stderr.open('xb') as stream: stream.write(proc.stderr)
    exit_data = {'returncode': proc.returncode, 'natural_exit': True, 'elapsed_seconds': elapsed, 'command': command, 'started_unix': started, 'ended_unix': time.time()}
    with exit_path.open('x') as stream: json.dump(exit_data, stream, indent=2); stream.write('\n')
    assert pins_before == {str(p): sha(p) for p in sources}
    outputs = [stdout, stderr, exit_path]+([receipt] if receipt.exists() else [])
    with zipfile.ZipFile(executed, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(pre, pre.name)
        for path in sources+outputs: archive.write(path, path.name)
    capsule = {'kind': 'EXACT_NEW_SCIENTIFIC_EXECUTION_CAPSULE', 'preexecution': {'path': pre.name, 'sha256': pre_sha, 'created_before_execution': True}, 'executed': {'path': executed.name, 'sha256': sha(executed)}, 'input_pins_before_and_after_equal': True, 'inputs': pins_before, 'outputs': {p.name: sha(p) for p in outputs}, 'exit': exit_data, 'frozen_main_or_solver_run': False, 'scientific_admission': False}
    with (root/f'{prefix}_capsule.json').open('x') as stream: json.dump(capsule, stream, indent=2); stream.write('\n')
    print(json.dumps({'capsule': f'{prefix}_capsule.json', 'returncode': proc.returncode, 'elapsed_seconds': elapsed}))
    sys.exit(proc.returncode)


if __name__ == '__main__':
    main()
