"""Execute new type-only CLI epochs; never overwrite original reviews."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    epoch = request['epoch']
    pre = ROOT / (epoch + '_preexecution.zip')
    executed = ROOT / (epoch + '_executed.zip')
    capsule = ROOT / (epoch + '_capsule.json')
    assert not pre.exists() and not executed.exists() and not capsule.exists()
    members: dict[str, Path] = {name: Path(path) for name, path in request['members'].items()}
    members[args.request.name] = args.request.resolve()
    pins = {name: digest(path) for name, path in members.items()}
    for name, expected in request['pins'].items():
        assert pins[name] == expected, name
    with ZipFile(pre, 'x', compression=ZIP_DEFLATED) as archive:
        for name, path in members.items():
            archive.write(path, name)
    start = time.monotonic()
    process = subprocess.run(request['argv'], cwd=ROOT, capture_output=True, check=False)
    elapsed = time.monotonic() - start
    stdout, stderr, exit_path = (ROOT / (epoch + suffix) for suffix in ('_stdout.txt', '_stderr.txt', '_exit.json'))
    stdout.write_bytes(process.stdout)
    stderr.write_bytes(process.stderr)
    runtime = {'argv': request['argv'], 'cwd': str(ROOT), 'natural_exit': process.returncode >= 0,
               'returncode': process.returncode, 'elapsed_seconds': elapsed}
    exit_path.write_text(json.dumps(runtime, indent=2) + '\n')
    output = ROOT / request['output']
    outputs = [stdout, stderr, exit_path]
    if output.exists():
        outputs.append(output)
    with ZipFile(executed, 'x', compression=ZIP_DEFLATED) as archive:
        with ZipFile(pre) as before:
            for name in before.namelist():
                archive.writestr(name, before.read(name))
        archive.write(pre, pre.name)
        for path in outputs:
            archive.write(path, path.name)
    payload = {'kind': 'NEW_TYPE_ONLY_CLI_IMMUTABLE_EPOCH_NOT_HISTORICAL_REBINDING',
               'governor': request['governor'], 'runtime': runtime,
               'source_inputs_unchanged': pins == {name: digest(path) for name, path in members.items()},
               'preexecution': {'path': pre.name, 'sha256': digest(pre)},
               'executed': {'path': executed.name, 'sha256': digest(executed)},
               'input_member_pins': pins,
               'outputs': {path.name: digest(path) for path in outputs}}
    capsule.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'capsule': capsule.name, **runtime}))
    sys.exit(process.returncode)


if __name__ == '__main__':
    main()
