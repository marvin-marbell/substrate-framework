"""Capture one NEW cube consumer's actual natural/failure exit and exact bytes."""
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
PREFIX = 'rich_photon_gauss_transport_20261015_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    selected = [PREFIX + name for name in ('model.py', 'consume.py', 'capture.py', 'actual_input.json')]
    selected += ['rich_photon_fullq_20261014_model.py', 'rich_photon_20261014_loop.py']
    sources = {name: dict(sha256=digest(ROOT / name), bytes=(ROOT / name).stat().st_size) for name in selected}
    command = [PYTHON, PREFIX + 'consume.py', PREFIX + 'actual_input.json']
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    started = time.time()
    (ROOT / (PREFIX + 'actual_start.json')).write_text(json.dumps(dict(command=command, cwd=str(ROOT),
        started_unix=started, python=PYTHON, platform=platform.platform(), source_pins=sources,
        actual_launch_revision='b8b88c61c08e2622b9a2b7f5355f5259c1e37ca98c344bccf6263896549353d4',
        threads={key:environment[key] for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}), indent=2) + '\n')
    with zipfile.ZipFile(ROOT / (PREFIX + 'actual_source_capsule.zip'), 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in selected:
            archive.write(ROOT / name, name)
    interruption = None
    with (ROOT / (PREFIX + 'actual_stdout.txt')).open('w') as stdout, (ROOT / (PREFIX + 'actual_stderr.txt')).open('w') as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
        try:
            code = process.wait(timeout=600)
            natural = True
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            interruption = type(error).__name__
            process.terminate()
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
            natural = False
    result = dict(command=command, observed_returncode=code, natural_process_exit=natural,
                  interruption=interruption, elapsed_seconds=time.time() - started,
                  sources_unchanged=all(digest(ROOT / name) == item['sha256'] for name, item in sources.items()),
                  source_pins=sources, success=(code == 0 and natural))
    (ROOT / (PREFIX + 'actual_exit.json')).write_text(json.dumps(result, indent=2) + '\n')
    outputs = selected + [path.name for path in ROOT.glob(PREFIX + '*') if path.is_file()
                         and path.suffix not in ('.zip', '.py') and path.name not in selected]
    with zipfile.ZipFile(ROOT / (PREFIX + 'actual_capsule.zip'), 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(outputs)):
            archive.write(ROOT / name, name)
    result['capsule_sha256'] = digest(ROOT / (PREFIX + 'actual_capsule.zip'))
    print(json.dumps(result), flush=True)
    return 0 if result['success'] else 1


if __name__ == '__main__':
    sys.exit(main())
