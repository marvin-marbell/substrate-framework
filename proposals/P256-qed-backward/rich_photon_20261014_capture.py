#!/usr/bin/env python3
"""Capture this new scientific execution, source/input identity and natural exit."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

root = Path(__file__).resolve().parent
source = root/'rich_photon_20261014_loop.py'
command = [sys.executable, str(source)]
inputs = {'law': 'rich-photon-hubbard-plaquette-links-v1-20261014',
          'hopping': 1.0, 'repulsion': 4.0, 'electric_inertia': 1.0,
          'particles_per_loop': 2, 'q_scales': [.08, .2, .5, 1.0],
          'directions': [[0, 0, 1], [1, 1, 1], [1, 2, 3]],
          'link_charge': 1.0, 'decoupled_control_link_charge': 0.0,
          'response_q': [.12, .17, .23], 'response_eta': .03,
          'response_omega': [0., .1, .5, 1., 2., 3., 4., 6., 10.],
          'feedback_q': [0., 0., .5], 'feedback_times': {'start': 0., 'stop': 20., 'count': 81},
          'feedback_initial_amplitude': .02}
input_bytes = json.dumps(inputs, sort_keys=True).encode()
source_bytes = source.read_bytes()
started = time.time()
result = subprocess.run(command, cwd=root, env={**os.environ, 'OPENBLAS_NUM_THREADS': '1',
                        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}, capture_output=True)
outputs = {'rich_photon_20261014_final_stdout.txt': result.stdout,
           'rich_photon_20261014_final_stderr.txt': result.stderr}
for name, data in outputs.items():
    (root/name).write_bytes(data)
output = root/'rich_photon_20261014_result.json'
archive = root/'rich_photon_20261014_final_execution.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    z.writestr(source.name, source_bytes)
    z.writestr('inputs.json', input_bytes)
    for name, data in outputs.items():
        z.writestr(name, data)
    if output.exists():
        z.writestr(output.name, output.read_bytes())
receipt = {'command': command, 'cwd': str(root), 'natural_returncode': result.returncode,
           'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
           'inputs_sha256': hashlib.sha256(input_bytes).hexdigest(),
           'inputs': inputs, 'wall_seconds': time.time()-started,
           'stdout_sha256': hashlib.sha256(result.stdout).hexdigest(),
           'stderr_sha256': hashlib.sha256(result.stderr).hexdigest(),
           'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest() if output.exists() else None,
           'execution_zip_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
(root/'rich_photon_20261014_final_natural_exit.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt, indent=2))
print(result.stdout.decode())
print(result.stderr.decode(), file=sys.stderr)
raise SystemExit(result.returncode)
