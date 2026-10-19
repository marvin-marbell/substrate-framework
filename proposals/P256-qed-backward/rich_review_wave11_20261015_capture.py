"""Natural exits and exact bounded source/data capsules for independent review."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    prefix = 'rich_review_wave11_20261015_' + sys.argv[1]
    inp = root / (prefix + '_input.json'); out = root / (prefix + '_result.json')
    request = json.loads(inp.read_text())
    own = ['rich_review_wave11_20261015_consumer.py', 'rich_review_wave11_20261015_capture.py', inp.name]
    paths = [root / name for name in dict.fromkeys(own + list(request['bindings']))]
    before = {path.name: sha(path) for path in paths}
    environment = dict(os.environ)
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        environment[key] = '1'
    command = [sys.executable, str(root / own[0]), str(inp), str(out)]
    started = time.time()
    process = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True)
    stdout = root / (prefix + '_stdout.txt'); stderr = root / (prefix + '_stderr.txt')
    stdout.write_text(process.stdout); stderr.write_text(process.stderr)
    after = {path.name: sha(path) for path in paths}
    outputs = [stdout, stderr] + [path for path in [out, out.with_suffix('.npz')] if path.exists()]
    receipt = dict(command=command, cwd=str(root), natural_exit=True, returncode=process.returncode, elapsed_seconds=time.time() - started, source_before=before, source_after=after, source_unchanged=before == after, outputs={path.name: sha(path) for path in outputs}, python=sys.version, executable=sys.executable, platform=platform.platform(), packages={name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'sympy']}, capsule_semantics='All consumed frozen sources/inputs/states/receipts and exact referenced producer capsules included; no old producer replay.')
    exitpath = root / (prefix + '_exit.json'); exitpath.write_text(json.dumps(receipt, indent=2) + '\n')
    capsule = root / (prefix + '_capsule.zip')
    with zipfile.ZipFile(capsule, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in paths + outputs + [exitpath]:
            archive.write(path, path.name)
    print(process.stdout, end='')
    if process.stderr:
        print(process.stderr, end='', file=sys.stderr)
    print(json.dumps(dict(returncode=process.returncode, natural_exit=True, source_unchanged=before == after, capsule=capsule.name, capsule_sha256=sha(capsule), seconds=receipt['elapsed_seconds'])))
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
