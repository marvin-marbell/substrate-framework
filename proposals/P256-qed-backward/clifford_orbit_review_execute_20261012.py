"""Capture and execute only the independent orbit review at a new immutable epoch."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    base = Path(__file__).resolve().parent
    prefix = 'clifford_orbit_review_actual2_20261012'
    request_name = prefix + '_request.json'
    request = json.loads((base / request_name).read_text())
    names = [request_name, request['entrypoint'], request['proof'],
             'clifford_orbit_review_language_20261012.json', Path(__file__).name]
    names.extend(item['name'] for item in request['frozen_inputs'])
    captured = {name: (base / name).read_bytes() for name in names}
    for item in request['frozen_inputs']:
        assert digest(captured[item['name']]) == item['sha256']
    manifest = {name: {'sha256': digest(data), 'bytes': len(data)} for name, data in captured.items()}
    pre_path = base / (prefix + '_preexecution.zip')
    with zipfile.ZipFile(pre_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in captured.items():
            archive.writestr(name, data)
        archive.writestr('capture_manifest.json', json.dumps(manifest, indent=2) + '\n')
    with tempfile.TemporaryDirectory(prefix=prefix + '_') as temporary:
        run_root = Path(temporary)
        with zipfile.ZipFile(pre_path) as archive:
            archive.extractall(run_root)
        argv = [request['python'], str(run_root / request['entrypoint']),
                '--request', str(run_root / request_name), '--output', str(run_root / 'receipt.json')]
        started = time.time()
        clock = time.monotonic()
        result = subprocess.run(argv, cwd=run_root, capture_output=True, check=False)
        elapsed = time.monotonic() - clock
        unchanged = {name: digest((base / name).read_bytes()) == digest(data)
                     for name, data in captured.items()}
        exit_record = {'kind': 'NATURAL_EXIT_INDEPENDENT_CAPTURED_ORBIT_REVIEW',
                       'epoch': prefix, 'argv': argv, 'returncode': result.returncode,
                       'natural_exit': result.returncode >= 0, 'wall_seconds': elapsed,
                       'started_unix': started, 'preexecution_sha256': digest(pre_path.read_bytes()),
                       'manifest': manifest, 'inputs_unchanged': unchanged,
                       'stdout_sha256': digest(result.stdout), 'stderr_sha256': digest(result.stderr)}
        products = {'stdout.txt': result.stdout, 'stderr.txt': result.stderr,
                    'exit.json': (json.dumps(exit_record, indent=2) + '\n').encode()}
        receipt = run_root / 'receipt.json'
        if receipt.exists():
            products['receipt.json'] = receipt.read_bytes()
        for name, data in products.items():
            path = base / (prefix + '_' + name)
            with path.open('xb') as stream:
                stream.write(data)
        executed_path = base / (prefix + '_executed.zip')
        with zipfile.ZipFile(executed_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in captured.items():
                archive.writestr(name, data)
            archive.writestr('capture_manifest.json', json.dumps(manifest, indent=2) + '\n')
            for name, data in products.items():
                archive.writestr('execution/' + name, data)
        capsule = {'kind': 'IMMUTABLE_NEW_EXACT_ORBIT_REVIEW_CAPSULE_NOT_SOURCE_PROMOTION',
                   'epoch': prefix, 'preexecution': {'path': str(pre_path), 'sha256': digest(pre_path.read_bytes())},
                   'executed': {'path': str(executed_path), 'sha256': digest(executed_path.read_bytes())},
                   'captured_manifest': manifest, 'natural_exit': result.returncode >= 0,
                   'returncode': result.returncode, 'wall_seconds': elapsed,
                   'inputs_unchanged': all(unchanged.values()),
                   'products': {name: {'path': prefix + '_' + name, 'sha256': digest(data)}
                                for name, data in products.items()},
                   'scientific_admission': False, 'campaign_complete': False}
        with (base / (prefix + '_capsule.json')).open('x') as stream:
            stream.write(json.dumps(capsule, indent=2) + '\n')
        print(json.dumps({'epoch': prefix, 'natural_returncode': result.returncode,
                          'wall_seconds': elapsed, 'inputs_unchanged': all(unchanged.values())}), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode if result.returncode > 0 else 128 - result.returncode)
        assert all(unchanged.values())


if __name__ == '__main__':
    main()
