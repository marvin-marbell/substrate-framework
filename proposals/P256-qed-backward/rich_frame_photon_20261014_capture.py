"""Capture exact new producer bytes, environment, output and natural process exit."""
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


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parent
    label = sys.argv[1]
    prefix = 'rich_frame_photon_20261014_' + label
    input_path = root / (prefix + '_input.json')
    output_path = root / (prefix + '_result.json')
    names = ['rich_frame_photon_20261014_model.py', 'rich_frame_photon_20261014_calculate.py',
             'rich_frame_photon_20261014_capture.py', 'rich_frame_photon_20261014_equations.json',
             'rich_frame_gauss_20261014_model.py', input_path.name]
    paths = [root / name for name in names]
    before = {path.name: digest(path) for path in paths}
    captured_bytes = {path.name: path.read_bytes() for path in paths}
    command = [sys.executable, str(root / 'rich_frame_photon_20261014_calculate.py'), str(input_path), str(output_path)]
    environment = dict(os.environ)
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        environment[key] = '1'
    start = time.time()
    process = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True)
    elapsed = time.time() - start
    stdout = root / (prefix + '_stdout.txt')
    stderr = root / (prefix + '_stderr.txt')
    stdout.write_text(process.stdout)
    stderr.write_text(process.stderr)
    after = {path.name: digest(path) for path in paths}
    artifacts = [stdout, stderr] + [p for p in (output_path, output_path.with_suffix('.npz')) if p.exists()]
    receipt = dict(command=command, cwd=str(root), python=sys.version, executable=sys.executable,
                   platform=platform.platform(), started_unix=start, elapsed_seconds=elapsed,
                   packages={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'sympy')},
                   returncode=process.returncode, natural_exit=True, source_before=before, source_after=after,
                   source_unchanged=before == after,
                   environment={key: environment[key] for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
                   outputs={path.name: digest(path) for path in artifacts})
    exit_path = root / (prefix + '_exit.json')
    exit_path.write_text(json.dumps(receipt, indent=2) + '\n')
    capsule = root / (prefix + '_capsule.zip')
    with zipfile.ZipFile(capsule, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, content in captured_bytes.items():
            archive.writestr(name, content)
        for path in artifacts + [exit_path]:
            archive.write(path, path.name)
    print(process.stdout)
    if process.stderr:
        print(process.stderr, file=sys.stderr)
    print(json.dumps(dict(returncode=process.returncode, seconds=elapsed, capsule=str(capsule),
                          capsule_sha256=digest(capsule), sources_unchanged=before == after)))
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
