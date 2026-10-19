"""Capture pre/executed bytes around only the NEW assigned independent CLI."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = 'quantum_boundary_sign_review_'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    request_path = ROOT/args.request
    req = json.loads(request_path.read_text())
    assert request_path.name.startswith(PREFIX)
    for name, expected in req['pins'].items():
        assert sha((ROOT/name).read_bytes()) == expected, name
    assert req['consumer'].startswith(PREFIX)
    assert req['python'] == '/home/administrator/substrate-framework/.venv/bin/python'
    selected = dict(req['pins'])
    selected[request_path.name] = sha(request_path.read_bytes())
    for name in req['supplemental']:
        selected[name] = sha((ROOT/name).read_bytes())
    started = now()
    pre = ROOT/req['preexecution']
    assert not pre.exists(), 'Do not overwrite an existing execution epoch'
    with zipfile.ZipFile(pre, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, expected in selected.items():
            data = (ROOT/name).read_bytes()
            assert sha(data) == expected
            archive.writestr(name, data)
    command = [req['python'], str(ROOT/req['consumer']), '--request', request_path.name]
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=600)
    (ROOT/req['stdout']).write_bytes(proc.stdout)
    (ROOT/req['stderr']).write_bytes(proc.stderr)
    exit_data = {'command': command, 'started': started, 'finished': now(),
                 'returncode': proc.returncode, 'natural_exit': True,
                 'preexecution_sha256': sha(pre.read_bytes()),
                 'source_and_input_hashes': selected}
    (ROOT/req['exit']).write_text(json.dumps(exit_data, indent=2)+'\n')
    with zipfile.ZipFile(ROOT/req['executed'], 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, expected in selected.items():
            data = (ROOT/name).read_bytes()
            assert sha(data) == expected, 'Input changed during independent execution: '+name
            archive.writestr(name, data)
        for name in (req['stdout'], req['stderr'], req['exit'], req['receipt']):
            if (ROOT/name).exists():
                archive.write(ROOT/name, name)
    artifacts = [req['preexecution'], req['executed'], req['exit'], req['stdout'], req['stderr']]
    if (ROOT/req['receipt']).exists():
        artifacts.append(req['receipt'])
    capsule = {'kind': 'NEW_INDEPENDENT_SOURCE_SELECTED_NATURAL_EXIT_CAPSULE',
               'natural_exit': True, 'returncode': proc.returncode, 'command': command,
               'captured_input_hashes': selected,
               'artifacts': {name: sha((ROOT/name).read_bytes()) for name in artifacts},
               'preexecution_backfill': False, 'original_producers_rerun': False,
               'scientific_admission': False, 'campaign_complete': False}
    (ROOT/req['capsule']).write_text(json.dumps(capsule, indent=2)+'\n')
    print(json.dumps({'returncode': proc.returncode, 'capsule': req['capsule']}))
    sys.exit(proc.returncode)


if __name__ == '__main__':
    main()
