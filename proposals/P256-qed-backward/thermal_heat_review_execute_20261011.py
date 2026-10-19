"""Capture one new independent control process without rerunning old targets."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = 'thermal_heat_review_actual1_20261011'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def new_json(path: Path, data: dict) -> None:
    with path.open('x') as stream:
        json.dump(data, stream, indent=2); stream.write('\n')


def main() -> None:
    input_path = ROOT/'thermal_heat_review_input_20261011.json'
    source_path = ROOT/'thermal_heat_review_control_20261011.py'
    request = json.loads(input_path.read_text())
    frozen = {name: (ROOT/name).read_bytes() for name in request['pins']}
    assert all(sha(frozen[name]) == pin for name, pin in request['pins'].items())
    contract = Path(request['contract_path']).read_bytes()
    assert sha(contract) == request['contract_sha256']
    selected = {path.name: path.read_bytes() for path in (input_path, source_path, Path(__file__),
                                                        ROOT/'thermal_heat_review_language_20261011.json')}
    pre = ROOT/(PREFIX+'_preexecution.zip')
    with zipfile.ZipFile(pre, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, raw in selected.items():
            archive.writestr(name, raw)
        for name, raw in frozen.items():
            archive.writestr('frozen/'+name, raw)
        archive.writestr('thermal_heat_review_frozen_contract.json', contract)
    output = ROOT/(PREFIX+'_receipt.json')
    argv = [PYTHON, '-B', str(source_path), '--input', str(input_path), '--output', str(output)]
    started = time.monotonic()
    child = subprocess.run(argv, cwd=ROOT, capture_output=True, check=False)
    elapsed = time.monotonic()-started
    for suffix, raw in (('_stdout.txt', child.stdout), ('_stderr.txt', child.stderr)):
        with (ROOT/(PREFIX+suffix)).open('xb') as stream:
            stream.write(raw)
    unchanged = all((ROOT/name).read_bytes() == raw for name, raw in selected.items())
    unchanged = unchanged and all((ROOT/name).read_bytes() == raw for name, raw in frozen.items())
    exit_path = ROOT/(PREFIX+'_exit.json')
    exit_record = {'kind': 'INDEPENDENT_CONTROL_NATURAL_PROCESS_EXIT', 'argv': argv,
                   'returncode': child.returncode, 'natural_exit': child.returncode >= 0,
                   'elapsed_seconds': elapsed, 'sources_inputs_unchanged': unchanged,
                   'preexecution_sha256': sha(pre.read_bytes()),
                   'stdout_sha256': sha(child.stdout), 'stderr_sha256': sha(child.stderr)}
    new_json(exit_path, exit_record)
    executed = ROOT/(PREFIX+'_executed.zip')
    with zipfile.ZipFile(executed, 'x', zipfile.ZIP_DEFLATED) as archive:
        with zipfile.ZipFile(pre) as prior:
            for name in prior.namelist():
                archive.writestr(name, prior.read(name))
        for path in (output, exit_path, ROOT/(PREFIX+'_stdout.txt'), ROOT/(PREFIX+'_stderr.txt')):
            if path.exists():
                archive.writestr(path.name, path.read_bytes())
    capsule = dict(exit_record, epoch=PREFIX, source_sha256=sha(selected[source_path.name]),
                   input_sha256=sha(selected[input_path.name]),
                   executed_sha256=sha(executed.read_bytes()),
                   receipt_sha256=sha(output.read_bytes()) if output.exists() else None,
                   contract_sha256=request['contract_sha256'],
                   files={'source': source_path.name, 'input': input_path.name,
                          'preexecution': pre.name, 'executed': executed.name, 'receipt': output.name,
                          'exit': exit_path.name, 'stdout': PREFIX+'_stdout.txt', 'stderr': PREFIX+'_stderr.txt'},
                   no_old_entrypoint=True, no_native_numeric_partition=True)
    new_json(ROOT/(PREFIX+'_capsule.json'), capsule)
    print(json.dumps(capsule, indent=2))
    if child.returncode != 0 or not unchanged:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
