#!/usr/bin/env python3
"""Run real research entrypoints and preserve revision-bound numerical evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import scipy
import sympy


HERE = Path(__file__).resolve().parent
BASELINE = '14e0b08f6099698b6d117840e65f86cbe70d4efb'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Save the same JSON printed to stdout')
    args = parser.parse_args()
    results = {}
    for name in ('compton', 'constraint', 'quantum_supplier', 'lattice_compton', 'spin_transfer'):
        completed = subprocess.run([sys.executable, str(HERE / f'{name}.py')],
                                   cwd=HERE, check=True, text=True, capture_output=True)
        results[name] = json.loads(completed.stdout)
    artifacts = sorted(HERE.glob('*.py')) + sorted(HERE.glob('*.md')) + [HERE / 'proposal.yaml']
    evidence = dict(
        status='first_deliverable_conditional_not_full_issue_fulfillment',
        source_baseline=BASELINE,
        runtime=dict(python=platform.python_version(), executable=sys.executable,
                     numpy=np.__version__, scipy=scipy.__version__, sympy=sympy.__version__),
        artifact_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts},
        calculations=results)
    payload = json.dumps(evidence, indent=2, sort_keys=True) + '\n'
    if args.output is not None:
        args.output.write_text(payload)
    print(payload, end='')


if __name__ == '__main__':
    main()
