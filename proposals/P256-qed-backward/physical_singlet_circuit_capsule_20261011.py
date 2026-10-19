"""One literal scientific execution followed by one final selected Pyright."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = 'physical_singlet_circuit_epoch4_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'

def main() -> None:
    owned = [ROOT/('physical_singlet_circuit_'+n+'_20261011.py') for n in ('source', 'capsule', 'language')]
    frozen = {p.name: p.read_bytes() for p in owned}
    with zipfile.ZipFile(ROOT/(PREFIX+'launcher_preexecution_20261011.zip'), 'x', zipfile.ZIP_DEFLATED) as z:
        for n, raw in frozen.items(): z.writestr(n, raw)
    command = [PYTHON, owned[0].name]
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    start = time.monotonic()
    process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    (ROOT/(PREFIX+'stdout_20261011.txt')).write_text(process.stdout)
    (ROOT/(PREFIX+'stderr_20261011.txt')).write_text(process.stderr)
    exit_record = dict(command=command, cwd=str(ROOT), returncode=process.returncode, natural_exit=True, elapsed_seconds=time.monotonic()-start, owned_source_sha256={n: hashlib.sha256(raw).hexdigest() for n, raw in frozen.items()}, owned_sources_unchanged=all((ROOT/n).read_bytes() == raw for n, raw in frozen.items()))
    (ROOT/(PREFIX+'exit_20261011.json')).write_text(json.dumps(exit_record, indent=2)+'\n')
    assert exit_record['owned_sources_unchanged']
    if process.returncode != 0:
        print(json.dumps(exit_record))
        raise SystemExit(process.returncode)
    with tempfile.TemporaryDirectory(prefix=PREFIX+'pyright_') as tmp:
        directory = Path(tmp)
        config = dict(include=[os.path.relpath(p, directory) for p in owned], extraPaths=[str(ROOT)], typeCheckingMode='basic', pythonVersion='3.12')
        config_path = directory/'pyrightconfig.json'
        config_path.write_text(json.dumps(config))
        checker_command = ['/home/administrator/.local/bin/pyright', '--project', str(config_path), '--pythonpath', PYTHON, '--outputjson', *[str(p) for p in owned]]
        started = time.monotonic()
        check = subprocess.run(checker_command, cwd=ROOT, env=env, capture_output=True, text=True, check=False)
        record = dict(command=checker_command, config=config, selected_source_sha256=exit_record['owned_source_sha256'], returncode=check.returncode, natural_exit=True, elapsed_seconds=time.monotonic()-started, stdout=check.stdout, stderr=check.stderr, source_unchanged=all((ROOT/n).read_bytes() == raw for n, raw in frozen.items()))
        (ROOT/(PREFIX+'selected_pyright_20261011.json')).write_text(json.dumps(record, indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(PREFIX+'natural_execution_20261011.zip'), 'x', zipfile.ZIP_DEFLATED) as z:
        for n, raw in frozen.items(): z.writestr(n, raw)
        for suffix in ('exit_20261011.json', 'stdout_20261011.txt', 'stderr_20261011.txt', 'selected_pyright_20261011.json', 'preexecution_20261011.zip', 'executed_20261011.zip'):
            path = ROOT/(PREFIX+suffix)
            z.writestr(path.name, path.read_bytes())
    print(json.dumps(dict(scientific_returncode=process.returncode, checker_returncode=check.returncode, source_unchanged=record['source_unchanged'])))

if __name__ == '__main__':
    main()
