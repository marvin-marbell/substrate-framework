#!/usr/bin/env python3
"""Run one reviewer physical CLI; bind exact bytes, environment and natural exit."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_review_exec_20261014_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capsule_audit():
    specifications = [
        ('rich_photon_20261014_final_execution.zip', 'rich_photon_20261014_loop.py',
         'a6674ba9e4eab0cd66bef1bd0143d4c6d36a823297fe7bab3f960c60df0b0948'),
        ('rich_composite_20261014_motion2_capsule.zip', 'rich_composite_20261014_consumer.py',
         '9d13eb870860501f4c7e9dd2e7bba7c2ce002b900a7d3057f1591f9db80b6608'),
        ('rich_composite_20261014_motion2_capsule.zip', 'rich_composite_20261014_motion.py',
         '3aa6e3d3bae4da1a73559a8fff27d327e70e097f419a0397e94a30006f0d0379'),
        ('rich_interaction_20261014_execution.zip', 'rich_interaction_20261014_coupled.py',
         '44821a9c8bb5e5c77400af1af6f3f839691eac15247239b818a1c02098d1ef25')]
    rows = []
    for capsule, source, expected in specifications:
        with zipfile.ZipFile(ROOT/capsule) as z:
            members = [name for name in z.namelist() if Path(name).name == source]
            captured = {name: hashlib.sha256(z.read(name)).hexdigest() for name in members}
        rows.append({'capsule': capsule, 'capsule_sha256': digest(ROOT/capsule),
                     'source': source, 'current_sha256': digest(ROOT/source),
                     'declared_executed_source_sha256': expected,
                     'captured_members': captured,
                     'current_matches_declared': digest(ROOT/source) == expected,
                     'capsule_contains_declared_source': expected in captured.values()})
    charged = json.loads((ROOT/'rich_electron_20261014_hubbard_charged_receipt.json').read_text())
    a = charged['artifacts']
    rows.append({'author_receipt': 'rich_electron_20261014_hubbard_charged_receipt.json',
        'no_selected_charged_execution_zip_declared': True,
        'source_matches': digest(ROOT/a['source']) == a['source_sha256'],
        'input_matches': digest(ROOT/a['input']) == a['input_sha256'],
        'output_matches': digest(ROOT/a['output']) == a['output_sha256'],
        'natural_exit': charged['natural_exit_observed'],
        'evidence_scope': 'Receipt and current bytes inspected; selected charged run had no source ZIP. Independent reviewer invocation below captures its imported source.'})
    return rows


def main():
    sources = [PREFIX+'consumer.py', PREFIX+'execute.py', PREFIX+'input.json',
               'rich_photon_20261014_loop.py', 'rich_electron_20261014_hubbard_charged.py',
               'rich_interaction_20261014_coupled.py', 'rich_composite_20261014_consumer.py',
               'rich_composite_20261014_motion.py', 'rich_composite_20261014_request.json']
    before = {name: digest(ROOT/name) for name in sources}
    audit = capsule_audit()
    command = [PYTHON, str(ROOT/(PREFIX+'consumer.py')), '--input', PREFIX+'input.json',
               '--output', PREFIX+'result.json']
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONHASHSEED='0')
    start = time.time()
    process = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True)
    (ROOT/(PREFIX+'stdout.txt')).write_bytes(process.stdout)
    (ROOT/(PREFIX+'stderr.txt')).write_bytes(process.stderr)
    after = {name: digest(ROOT/name) for name in sources}
    outputs = [PREFIX+'stdout.txt', PREFIX+'stderr.txt']
    if (ROOT/(PREFIX+'result.json')).exists():
        outputs.append(PREFIX+'result.json')
    receipt = {'command': command, 'cwd': str(ROOT), 'natural_returncode': process.returncode,
        'natural_exit_observed': True, 'started_unix': start, 'elapsed_seconds': time.time()-start,
        'python': sys.executable, 'python_version': sys.version, 'platform': platform.platform(),
        'thread_environment': {k: environment[k] for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTHONHASHSEED')},
        'source_input_sha256': before, 'sources_inputs_unchanged': before == after,
        'output_sha256': {name: digest(ROOT/name) for name in outputs},
        'author_source_capsule_audit': audit,
        'scope': 'One reviewer CLI; independent physical consumer invokes only pure APIs, no author run()/main()/CLI, no cached physical result.'}
    receipt_path = ROOT/(PREFIX+'natural_exit.json')
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(PREFIX+'execution.zip'), 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name in sources+outputs+[receipt_path.name]:
            z.write(ROOT/name, arcname=name)
    print(json.dumps({'receipt': str(receipt_path), 'natural_returncode': process.returncode,
                      'sources_inputs_unchanged': before == after, 'elapsed_seconds': receipt['elapsed_seconds'],
                      'capsule_sha256': digest(ROOT/(PREFIX+'execution.zip'))}))
    if process.returncode:
        print(process.stderr.decode(), file=sys.stderr)
    return process.returncode


if __name__ == '__main__':
    raise SystemExit(main())
