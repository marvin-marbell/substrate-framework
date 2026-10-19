"""Immutable capture and natural-exit execution of the NEW charged-frame consumer."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import zipfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.request.read_text())
    root = args.request.parent
    epoch = config['epoch']
    target = lambda suffix: root / f'{epoch}_{suffix}'
    request = Path(config['consumer_request'])
    spec = json.loads(request.read_text())
    inputs = list(dict.fromkeys(Path(p).resolve() for p in list(spec['pins']) + [str(request), str(args.request), __file__]))
    for name, expected in spec['pins'].items():
        assert digest(Path(name)) == expected, name
    hashes = {str(path): digest(path) for path in inputs}
    pre = target('preexecution.zip')
    with zipfile.ZipFile(pre, 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in inputs:
            archive.write(path, arcname='inputs/' + path.name)
    command = [config['python'], config['consumer'], '--request', str(request), '--output', str(target('receipt.json'))]
    start = time.time_ns()
    process = subprocess.run(command, cwd=root, capture_output=True, check=False)
    finish = time.time_ns()
    target('stdout.txt').write_bytes(process.stdout)
    target('stderr.txt').write_bytes(process.stderr)
    target('exit.json').write_text(json.dumps({'command': command, 'start_ns': start, 'end_ns': finish,
                                             'elapsed_seconds': (finish - start) / 1e9,
                                             'natural_returncode': process.returncode,
                                             'timeout_or_signal_intervention': False}, indent=2) + '\n')
    unchanged = {str(path): digest(path) == hashes[str(path)] for path in inputs}
    outputs = [target('stdout.txt'), target('stderr.txt'), target('exit.json')]
    if target('receipt.json').exists():
        outputs.append(target('receipt.json'))
    executed = target('executed.zip')
    with zipfile.ZipFile(executed, 'x', zipfile.ZIP_DEFLATED) as archive:
        with zipfile.ZipFile(pre) as source:
            for member in source.namelist():
                archive.writestr(member, source.read(member))
        for path in outputs:
            archive.write(path, arcname='outputs/' + path.name)
    capsule = {'kind': 'NEW_CHARGED_FRAME_SOURCE_IMMUTABLE_EXECUTED_CAPSULE', 'epoch': epoch,
               'command': command, 'natural_returncode': process.returncode,
               'elapsed_seconds': (finish - start) / 1e9, 'input_hashes': hashes,
               'post_input_unchanged': unchanged,
               'preexecution': {'path': str(pre), 'sha256': digest(pre)},
               'executed': {'path': str(executed), 'sha256': digest(executed)},
               'outputs': {str(path): digest(path) for path in outputs},
               'historical_entrypoint_import_or_replay': False,
               'diagnostics_build_lint_tests_formatters_run': False}
    target('capsule.json').write_text(json.dumps(capsule, indent=2) + '\n')
    print(json.dumps({'epoch': epoch, 'returncode': process.returncode,
                      'elapsed_seconds': (finish - start) / 1e9, 'capsule': str(target('capsule.json'))}))
    assert all(unchanged.values()), 'captured input changed during execution'
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
