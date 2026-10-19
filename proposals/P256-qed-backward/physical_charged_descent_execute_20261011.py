"""Execute one exclusive new physical_charged_descent scientist epoch.

Never reruns an old ground/root/checker. A failure is archived with its genuine
natural returncode; later epochs cannot replace its files or byte archives.
"""
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


def exclusive_json(path: Path, value: object) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    path.chmod(0o444)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', required=True)
    parser.add_argument('--eta-exponent', type=int, default=4000)
    parser.add_argument('--neutral-epsilon-exponent', type=int, default=6212)
    parser.add_argument('--tau', default='-1/128')
    args = parser.parse_args()
    assert args.epoch.replace('_','').isalnum()
    prefix = 'physical_charged_descent_'+args.epoch+'_20261011_'
    request = {'prefix':prefix,'eta_exponent':args.eta_exponent,'neutral_epsilon_exponent':args.neutral_epsilon_exponent,'tau':args.tau,'plan_id':'issue-222-induced-photon-delta13-continuation-20261008','task_governing_revision':'9a5e0b2cbe54cf4e5fa035b7fec3e31ff0f59bb7ecb7bf97fcea4844f31de2ba','template':'general','template_revision':'89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe','scope':'independent fullPhi charged-only sameH increment against identical neutral-circuited baseline'}
    request_path = ROOT/(prefix+'request.json')
    exclusive_json(request_path, request)
    command = [PYTHON,'physical_charged_descent_20261011.py','--request',request_path.name]
    started = time.monotonic()
    completed = subprocess.run(command,cwd=ROOT,capture_output=True)
    for suffix,raw in [('stdout.txt',completed.stdout),('stderr.txt',completed.stderr)]:
        path = ROOT/(prefix+suffix)
        with path.open('xb') as stream:
            stream.write(raw)
        path.chmod(0o444)
    exit_path=ROOT/(prefix+'natural_exit.json')
    exclusive_json(exit_path,{'command':command,'cwd':str(ROOT),'natural_returncode':completed.returncode,'runtime_seconds':time.monotonic()-started,'termination':'subprocess.run returned naturally; no timeout, signal, inferred success or polling','stdout_sha256':hashlib.sha256(completed.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(completed.stderr).hexdigest()})
    pre = ROOT/(prefix+'preexecution.zip')
    if pre.exists():
        with zipfile.ZipFile(pre) as archive:
            files={name:archive.read(name) for name in archive.namelist()}
        for path in ROOT.glob(prefix+'*'):
            if path.suffix != '.zip' and path.is_file():
                files[path.name]=path.read_bytes()
        with zipfile.ZipFile(ROOT/(prefix+'executed.zip'),'x',zipfile.ZIP_DEFLATED) as archive:
            for name,raw in files.items():
                archive.writestr(name,raw)
        (ROOT/(prefix+'executed.zip')).chmod(0o444)
    print(json.dumps({'prefix':prefix,'natural_returncode':completed.returncode,'stdout':completed.stdout.decode(errors='replace'),'stderr':completed.stderr.decode(errors='replace')}))
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
