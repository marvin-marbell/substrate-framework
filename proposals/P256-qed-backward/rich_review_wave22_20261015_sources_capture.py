"""Exact natural source-moment child capture, separate from completed probe."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile
ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_review_wave22_20261015_'


def record(path: Path) -> dict:
    return {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}


inputs = set(ROOT.glob('rich_mobile_packet_20261015_actual1_*.npz')) | set(ROOT.glob('rich_mobile_packet_20261015_actual1_*masks.json')) | {ROOT/'rich_mobile_packet_20261015_actual1_input.json', ROOT/'rich_mobile_packet_20261015_actual2_charged_incoming_state16.npz', ROOT/'rich_mobile_packet_20261015_actual2_neutral_incoming_state16.npz', ROOT/(PREFIX+'sources.py'), Path(__file__).resolve()}
before = {p.name: record(p) for p in inputs}
start = time.time()
with (ROOT/(PREFIX+'sources_stdout.txt')).open('wb') as out, (ROOT/(PREFIX+'sources_stderr.txt')).open('wb') as err:
    child = subprocess.Popen(['/home/administrator/substrate-framework/.venv/bin/python', PREFIX+'sources.py'], cwd=ROOT, env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'), stdout=out, stderr=err)
    code = child.wait()
result = json.loads((ROOT/(PREFIX+'sources_result.json')).read_text()) if code == 0 else {}
receipt = {'kind': 'NATURAL_INDEPENDENT_FIELD_SOURCE_CHILD_EXIT', 'returncode': code, 'wall_seconds': time.time()-start, 'scientific_elapsed_seconds': result.get('elapsed_seconds'), 'observed_scientific_peak_resident_KiB': result.get('peak_resident_KiB'), 'changed_source_inputs': [p.name for p in inputs if record(p) != before[p.name]], 'first_read_vs_before_mismatches': [name for name, digest in result.get('consumed_sha256', {}).items() if before[name]['sha256'] != digest]}
(ROOT/(PREFIX+'sources_exit.json')).write_text(json.dumps(receipt, indent=2)+'\n')
members = {ROOT/name for name in result.get('consumed_sha256', {})} | {ROOT/(PREFIX+'sources.py'), Path(__file__).resolve()} | set(ROOT.glob(PREFIX+'sources_*.json')) | set(ROOT.glob(PREFIX+'sources_*.txt')) | {ROOT/(PREFIX+'field_sources.npz')}
manifest = {'before': {p.name: before[p.name] for p in members if p.name in before}, 'outputs': {p.name: record(p) for p in members if p.name not in before}, 'receipt': receipt}
(ROOT/(PREFIX+'sources_manifest.json')).write_text(json.dumps(manifest, indent=2)+'\n')
members.add(ROOT/(PREFIX+'sources_manifest.json'))
with zipfile.ZipFile(ROOT/(PREFIX+'sources_capsule.zip'), 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as capsule:
    for path in sorted(members):
        capsule.write(path, path.name)
print(json.dumps({'receipt': receipt, 'capsule': record(ROOT/(PREFIX+'sources_capsule.zip'))}), flush=True)
raise SystemExit(code)
