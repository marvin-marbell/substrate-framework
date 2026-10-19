"""Immutable scientific input/source capture and honest natural child exits."""
from __future__ import annotations
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_recoil_current_wave_20261016_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def main() -> None:
    stamp = sys.argv[1]
    stage = sys.argv[2] if len(sys.argv) > 2 else 'run'
    if stage not in ['run', 'emission', 'seed_details', 'language']:
        raise ValueError('Only the physical producer, its current/emission consumers and final selected diagnostics are captured')
    stem = PREFIX+stamp+'_'
    input_path = ROOT/(stem+'input.json')
    if input_path.exists():
        raise FileExistsError('Scientific attempts are immutable; use a NEW stamp')
    law_path = ROOT/'rich_spatial_translation_20261015_actual3_input.json'
    law = json.loads(law_path.read_text())
    inputs = dict(stamp=stamp, stage=stage, analysis_of=sys.argv[3] if stage in ['emission', 'seed_details'] else None,
        plan_id='issue222-rich-backward-candidates-20261014',
        captured_plan_revision='813490dc61fdae6c7b9b049ec7f59e2f9fc63415db5461f0f80e39ffecd139e4',
        template='general@89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
        law=law, law_input_sha256=hashlib.sha256(law_path.read_bytes()).hexdigest(),
        short_start=.01, dt=.095, duration=.2, minimum_bond=64, numbers=[54, 55, 53], protocols=['baseline', 'xcos', 'ycos'],
        versions={package: importlib.metadata.version(package) for package in ['numpy', 'scipy', 'physics-tenpy']},
        physical_reference='Fresh explicitly prepared reference Slater, not any old solved neutral state; new matched neutral evolved at the same clock.',
        field_protocol='Finite-shell projected-Q-only unitary with strength0.8; no full-W/pure-photon claim.',
        algorithm='109-site full108-CAR+75-state scalar/EM MPS, exact physical C3 and N; first actual same-H short Galerkin evolution in psi,Hpsi,four before-cut QJpsi, then occupied-Schmidt two-site TDVP; no zero padding or hard particle-hole-rank cutoff.')
    input_path.write_text(json.dumps(inputs, indent=2)+'\n')
    sources = set(ROOT.glob(PREFIX+'*.py'))
    queue = list(sources)
    while queue:
        source = queue.pop()
        for node in ast.walk(ast.parse(source.read_text())):
            names = [node.module] if isinstance(node, ast.ImportFrom) else [x.name for x in node.names] if isinstance(node, ast.Import) else []
            for name in names:
                if name is None:
                    continue
                candidate = ROOT/(name+'.py')
                if candidate.exists() and candidate not in sources:
                    sources.add(candidate)
                    queue.append(candidate)
    hashes = {source.name: hashlib.sha256(source.read_bytes()).hexdigest() for source in sorted(sources)}
    with zipfile.ZipFile(ROOT/(stem+'source_capsule.zip'), 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(sources):
            archive.write(source, source.name)
        archive.write(law_path, law_path.name)
        archive.write(input_path, input_path.name)
        archive.writestr('source_sha256.json', json.dumps(hashes, indent=2)+'\n')
    started = time.time()
    arguments = [str(input_path)] if stage == 'run' else [sys.argv[3], stamp] if stage == 'emission' else [sys.argv[3]] if stage == 'seed_details' else []
    command = [PYTHON, str(ROOT/(PREFIX+stage+'.py')), *arguments]
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    with (ROOT/(stem+'stdout.txt')).open('w') as stdout, (ROOT/(stem+'stderr.txt')).open('w') as stderr:
        child = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
    exit_data = dict(returncode=child.returncode, natural_exit=child.returncode >= 0,
        termination_signal=signal.Signals(-child.returncode).name if child.returncode < 0 else None,
        seconds=time.time()-started, input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        source_sha256=hashes, command=command)
    (ROOT/(stem+'exit.json')).write_text(json.dumps(exit_data, indent=2)+'\n')
    print(json.dumps({k: v for k, v in exit_data.items() if k != 'source_sha256'}), flush=True)
    sys.exit(child.returncode if child.returncode >= 0 else 128-child.returncode)


if __name__ == '__main__':
    main()
