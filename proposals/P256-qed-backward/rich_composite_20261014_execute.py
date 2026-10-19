"""Record one new scientific construction process with exact bytes and exit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tag', default='actual1')
parser.add_argument('--request', default='rich_composite_20261014_request.json')
parser.add_argument('--source', default='rich_composite_20261014_consumer.py')
args = parser.parse_args()
root = Path(__file__).resolve().parent
prefix = 'rich_composite_20261014_'+args.tag
source = root/args.source
request = root/args.request
out = root/(prefix+'_receipt.json')
files = list(dict.fromkeys([source, root/'rich_composite_20261014_consumer.py', request, Path(__file__).resolve()]))
pins = {p.name: digest(p) for p in files}
command = [sys.executable, str(source), '--request', str(request), '--output', str(out)]
started = time.time()
env = dict(os.environ)
env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
process = subprocess.run(command, cwd=root, env=env, capture_output=True)
finished = time.time()
stdout = root/(prefix+'_stdout.txt'); stderr = root/(prefix+'_stderr.txt')
stdout.write_bytes(process.stdout); stderr.write_bytes(process.stderr)
outputs = [stdout, stderr]
if out.exists():
    outputs += [out, out.with_suffix('.npz')]
exit_record = {'command': command, 'cwd': str(root), 'python': sys.executable,
               'python_version': sys.version, 'platform': platform.platform(),
               'start_unix': started, 'finish_unix': finished, 'elapsed_seconds': finished-started,
               'natural_returncode': process.returncode, 'input_source_sha256': pins,
               'inputs_unchanged': all(digest(root/name)==value for name, value in pins.items()),
               'output_sha256': {p.name: digest(p) for p in outputs},
               'thread_environment': {key: env[key] for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}}
exit_path = root/(prefix+'_exit.json'); exit_path.write_text(json.dumps(exit_record, indent=2)+'\n')
with zipfile.ZipFile(root/(prefix+'_capsule.zip'), 'w', zipfile.ZIP_DEFLATED) as archive:
    for p in files+outputs+[exit_path]:
        archive.write(p, p.name)
print(json.dumps(exit_record))
sys.exit(process.returncode)
