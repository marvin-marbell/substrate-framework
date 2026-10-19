from __future__ import annotations
import datetime
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'charged_image_proof_control_20261011.py'
PYTHON = Path('/home/administrator/substrate-framework/.venv/bin/python')
INPUTS = [ROOT/name for name in ['charged_spinor_joint_free_phase_control_20261011.py', 'charged_spinor_joint_free_phase_control_20261011_receipt.json', 'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json', 'local_so5_positive_output_row_inertia_certificate_20261011.py']]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def immutable(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2)+'\n')

attempt = sys.argv[1]
assert attempt.isdecimal()
prefix = ROOT/f'charged_image_proof_attempt{attempt}_20261011'
hashes = {str(path): digest(path) for path in [SOURCE, Path(__file__), *INPUTS]}
command = [str(PYTHON), str(SOURCE)]
pre = {'kind': 'IMMUTABLE_PREEXECUTION', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'command': command, 'cwd': str(ROOT), 'source_input_hashes': hashes, 'python_executable_sha256': digest(PYTHON.resolve()), 'receipt_expected': str(ROOT/'charged_image_proof_control_20261011_receipt.json')}
immutable(Path(str(prefix)+'_preexecution.json'), pre)
start = time.monotonic()
process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
unchanged = all(digest(Path(path)) == value for path, value in hashes.items())
executed = {'kind': 'IMMUTABLE_EXECUTED', 'preexecution_sha256': digest(Path(str(prefix)+'_preexecution.json')), 'stdout': process.stdout, 'stderr': process.stderr, 'returncode': process.returncode, 'elapsed_seconds': time.monotonic()-start, 'source_inputs_unchanged': unchanged}
immutable(Path(str(prefix)+'_executed.json'), executed)
receipt = ROOT/'charged_image_proof_control_20261011_receipt.json'
exit_record = {'kind': 'IMMUTABLE_NATURAL_EXIT', 'returncode': process.returncode, 'natural_exit': process.returncode >= 0, 'source_inputs_unchanged': unchanged, 'source_input_hashes': hashes, 'executed_sha256': digest(Path(str(prefix)+'_executed.json')), 'receipt_sha256': digest(receipt) if process.returncode == 0 else None}
immutable(Path(str(prefix)+'_exit.json'), exit_record)
print(json.dumps(executed, indent=2))
print(json.dumps(exit_record, indent=2))
sys.exit(process.returncode if process.returncode >= 0 else 1)
