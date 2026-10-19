"""Natural finite child capture and exact source/input/output capsule, review-owned."""
from __future__ import annotations
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_review_wave22_20261015_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict:
    return {'sha256': sha(path), 'bytes': path.stat().st_size}


def dependencies(path: Path, seen: set[Path]) -> None:
    if path in seen:
        return
    seen.add(path)
    for node in ast.walk(ast.parse(path.read_text())):
        modules = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules = [node.module]
        for module in modules:
            child = ROOT/(module+'.py')
            if child.is_file():
                dependencies(child, seen)


sources: set[Path] = set()
dependencies(ROOT/(PREFIX+'probe.py'), sources)
sources.add(Path(__file__).resolve())
# Snapshot all potential frozen source-state bindings before launch; only actual
# consumed inputs are packed below. No old CLI executes.
inputs = set(ROOT.glob('rich_mobile_packet_20261015_actual1_*.npz')) | set(ROOT.glob('rich_mobile_packet_20261015_actual1_*.json')) | set(ROOT.glob('rich_mobile_packet_20261015_actual2_*state*.npz'))
inputs |= {ROOT/'rich_mobile_packet_20261015_actual2_result.json', ROOT/'rich_mobile_packet_20261015_actual2_exit.json', ROOT/'rich_mobile_packet_20261015_analysis_exit.json'}
before = {p.name: record(p) for p in sources | inputs}
start = time.time()
environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
with (ROOT/(PREFIX+'stdout.txt')).open('wb') as out, (ROOT/(PREFIX+'stderr.txt')).open('wb') as err:
    child = subprocess.Popen([PYTHON, PREFIX+'probe.py'], cwd=ROOT, env=environment, stdout=out, stderr=err)
    returncode = child.wait()
receipt = {'kind': 'NATURAL_INDEPENDENT_PHYSICAL_CHILD_EXIT', 'returncode': returncode, 'wall_seconds': time.time()-start, 'pid': child.pid, 'command': [PYTHON, PREFIX+'probe.py'], 'environment': {k: environment[k] for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']}, 'source_input_changed': [p.name for p in sources | inputs if record(p) != before[p.name]]}
result_path = ROOT/(PREFIX+'result.json')
checkpoint_path = ROOT/(PREFIX+'checkpoint.json')
actual = json.loads((result_path if returncode == 0 else checkpoint_path).read_text()) if (result_path.exists() or checkpoint_path.exists()) else {}
receipt['scientific_elapsed_seconds'] = actual.get('elapsed_seconds')
receipt['scientific_observed_peak_resident_KiB'] = actual.get('peak_resident_KiB')
(ROOT/(PREFIX+'exit.json')).write_text(json.dumps(receipt, indent=2)+'\n')
consumed = actual.get('consumed_sha256', {})
manifest = {'kind': 'EXACT_INDEPENDENT_SOURCE_INPUT_STATE_OUTPUT_CAPSULE', 'receipt': receipt, 'source_before': {p.name: before[p.name] for p in sources}, 'consumed_inputs': {name: before[name] for name in consumed}, 'consumer_first_read_vs_prelaunch_mismatches': [name for name, digest in consumed.items() if before[name]['sha256'] != digest], 'outputs': {p.name: record(p) for p in ROOT.glob(PREFIX+'*') if p.is_file() and p.suffix not in ['.zip'] and p.name not in {PREFIX+'manifest.json'}}}
(ROOT/(PREFIX+'runtime_manifest.json')).write_text(json.dumps(manifest, indent=2)+'\n')
with zipfile.ZipFile(ROOT/(PREFIX+'capsule.zip'), 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as capsule:
    members = sources | {ROOT/name for name in consumed} | {p for p in ROOT.glob(PREFIX+'*') if p.is_file() and p.suffix != '.zip'}
    for path in sorted(members):
        capsule.write(path, path.name)
print(json.dumps({'receipt': receipt, 'capsule': record(ROOT/(PREFIX+'capsule.zip'))}), flush=True)
raise SystemExit(returncode)
