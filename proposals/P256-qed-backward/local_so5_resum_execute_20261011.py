"""Capture actual scientific child natural exit, preserving failed epochs."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', required=True)
    parser.add_argument('--iterations', type=int, default=250)
    parser.add_argument('--response-iterations', type=int, default=100)
    parser.add_argument('--tolerance', type=float, default=2e-5)
    parser.add_argument('--delta', type=float, default=.1)
    parser.add_argument('--state-input')
    parser.add_argument('--state-sha256')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    base = 'local_so5_resum_'+args.epoch+'_20261011'
    own = Path(__file__)
    source = root/'local_so5_resum_source_20261011.py'
    raws = {p.name: p.read_bytes() for p in [own, source]}
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    target = root/(base+'_exit.json')
    if target.exists():
        raise RuntimeError('immutable epoch exists; choose a new epoch')
    with zipfile.ZipFile(root/(base+'_launch.zip'), 'w', zipfile.ZIP_DEFLATED) as capsule:
        for name, raw in raws.items():
            capsule.writestr(name, raw)
        prior_proof = root/'local_so5_resum_proof_20261011.json'
        if prior_proof.exists():
            capsule.writestr('prior_epoch/'+prior_proof.name, prior_proof.read_bytes())
    command = [sys.executable, str(source), '--epoch', args.epoch, '--iterations', str(args.iterations), '--response-iterations', str(args.response_iterations)]
    command += ['--tolerance', str(args.tolerance), '--delta', str(args.delta)]
    if args.state_input:
        command += ['--state-input', args.state_input, '--state-sha256', args.state_sha256]
    start = time.monotonic()
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    with (root/(base+'_stdout.txt')).open('wb') as out, (root/(base+'_stderr.txt')).open('wb') as err:
        result = subprocess.run(command, cwd=root, stdout=out, stderr=err, check=False, env=environment)
    unchanged = {name: (root/name).read_bytes() == raw for name, raw in raws.items()}
    receipt = {'command': command, 'returncode': result.returncode, 'natural_exit': True, 'runtime_seconds': time.monotonic()-start, 'source_sha256': {name: sha(raw) for name, raw in raws.items()}, 'source_unchanged_at_exit': unchanged, 'child_maxrss_kib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, 'stdout': base+'_stdout.txt', 'stderr': base+'_stderr.txt', 'launch_capsule': base+'_launch.zip'}
    target.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
