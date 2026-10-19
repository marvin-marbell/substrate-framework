"""Capture exact NEW source/input bytes before a fresh physical calculation."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_composite_free_rot_20261014_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'
SOURCES = [PREFIX+'model.py', PREFIX+'current.py', PREFIX+'calculate.py', PREFIX+'capture.py', PREFIX+'request.json', 'rich_composite_rot_20261014_consumer.py']

def sha(data):
    return hashlib.sha256(data).hexdigest()

snapshot = {name: (ROOT/name).read_bytes() for name in SOURCES}
command = [PYTHON, str(ROOT/(PREFIX+'calculate.py')), '--request', str(ROOT/(PREFIX+'request.json')), '--output', str(ROOT/(PREFIX+'actual1_result'))]
start = time.monotonic()
with (ROOT/(PREFIX+'actual1_stdout.txt')).open('wb') as stdout, (ROOT/(PREFIX+'actual1_stderr.txt')).open('wb') as stderr:
    completed = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
elapsed = time.monotonic()-start
outputs = [PREFIX+'actual1_stdout.txt', PREFIX+'actual1_stderr.txt']
outputs.extend(name for name in [PREFIX+'actual1_result.json', PREFIX+'actual1_result.npz'] if (ROOT/name).exists())
receipt = dict(kind='NEW_PHYSICAL_CONSTRUCTION_NATURAL_EXIT_NOT_TEST_REPLAY', command=command, cwd=str(ROOT), python=PYTHON, returncode=completed.returncode, natural_exit=True, elapsed_seconds=elapsed, source_input_sha256={name: sha(data) for name, data in snapshot.items()}, source_stable_after_exit={name: (ROOT/name).read_bytes() == data for name, data in snapshot.items()}, output_sha256={name: sha((ROOT/name).read_bytes()) for name in outputs}, governing_plan='issue222-rich-backward-candidates-20261014', captured_plan_revision='87bedd8343641b2540426b62d725c656fa893e956a9ec2b56ba08082c518ab3e', template='general@89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe', binding='The precursor pure build/current/spectrum helpers are captured; no precursor CLI or state is replayed. New joint body/orientation state is calculated.')
receipt_path = ROOT/(PREFIX+'actual1_exit.json')
receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
with zipfile.ZipFile(ROOT/(PREFIX+'actual1_capsule.zip'), 'w', zipfile.ZIP_DEFLATED) as capsule:
    for name, data in snapshot.items():
        capsule.writestr('source/'+name, data)
    for name in outputs:
        capsule.write(ROOT/name, 'output/'+name)
    capsule.write(receipt_path, receipt_path.name)
print(json.dumps(receipt), flush=True)
raise SystemExit(completed.returncode)
