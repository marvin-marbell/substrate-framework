"""Freeze independent sources/inputs, execute only NEW exact review control."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = 'bulk_thermal_review_actual2_20261011'


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def archive(path: Path, contents: dict[str, bytes]) -> str:
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, raw in sorted(contents.items()):
            bundle.writestr(name, raw)
    return sha(path.read_bytes())


def main() -> None:
    request_name = PREFIX + '_request.json'
    request = json.loads((ROOT / request_name).read_bytes())
    names = [request_name, 'bulk_thermal_review_control_20261011.py', Path(__file__).name,
             'bulk_thermal_review_language_20261011.json', *request['input_sha256']]
    frozen = {name: (ROOT / name).read_bytes() for name in names}
    for name, expected in request['input_sha256'].items():
        assert sha(frozen[name]) == expected
    contract_path = Path('/home/administrator/.omp/agent/sessions/-/2026-10-09T16-26-59-293Z_01a1217d-415d-7376-a17d-16be721cbe28/local/issue222-thermal-bulk-independent-review-contract-20261011.json')
    contract_raw = contract_path.read_bytes()
    assert sha(contract_raw) == request['frozen_contract_sha256']
    frozen['bulk_thermal_review_frozen_contract_20261011.json'] = contract_raw
    primitive_archive = ROOT / 'joined_phi_u1_stationary_flat_holonomy_typed_certified_20261010_receipt_input_bytes.zip'
    primitive_pins = json.loads(frozen[request['profile']])['source_input_sha256_before']
    primitive_sources = ['induced_phi_uniform_relative_colour_quantum.py', 'bosonic_original_law_join.py', 'induced_phi_nonuniform_field_car.py']
    with zipfile.ZipFile(primitive_archive) as bundle:
        for name in primitive_sources:
            raw = bundle.read(name)
            assert sha(raw) == primitive_pins[name]
            frozen['primitive_sources/' + name] = raw
    with zipfile.ZipFile(ROOT / 'boundary_bulk_actual2_20261011_executed.zip') as bundle:
        for name in request['input_sha256']:
            if name in bundle.namelist():
                assert bundle.read(name) == frozen[name]
    pre_name = PREFIX + '_preexecution.zip'
    pre_hash = archive(ROOT / pre_name, frozen)
    output_name = PREFIX + '_receipt.json'
    argv = [sys.executable, '-B', str(ROOT / 'bulk_thermal_review_control_20261011.py'),
            '--request', str(ROOT / request_name), '--output', str(ROOT / output_name)]
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    started = time.monotonic()
    process = subprocess.run(argv, cwd=ROOT, env=environment, capture_output=True, check=False)
    elapsed = time.monotonic() - started
    stable = all((ROOT / name).read_bytes() == frozen[name] for name in names)
    exit_data = dict(argv=argv, returncode=process.returncode, natural_exit=True, elapsed_seconds=elapsed,
                     source_inputs_unchanged=stable, input_sha256={n: sha(raw) for n, raw in frozen.items()},
                     runtime=dict(executable=sys.executable, python=sys.version))
    outputs = {PREFIX + '_stdout.txt': process.stdout, PREFIX + '_stderr.txt': process.stderr,
               PREFIX + '_exit.json': (json.dumps(exit_data, indent=2) + '\n').encode()}
    if (ROOT / output_name).exists():
        outputs[output_name] = (ROOT / output_name).read_bytes()
    for name, raw in outputs.items():
        (ROOT / name).write_bytes(raw)
    exe_name = PREFIX + '_executed.zip'
    exe_hash = archive(ROOT / exe_name, {**frozen, **outputs})
    capsule = dict(kind='IMMUTABLE_INDEPENDENT_BULK_THERMAL_REVIEW_CONTROL_CAPSULE',
                   natural_exit=True, returncode=process.returncode, elapsed_seconds=elapsed,
                   source_inputs_unchanged=stable, preexecution_zip=pre_name, preexecution_sha256=pre_hash,
                   executed_zip=exe_name, executed_sha256=exe_hash,
                   outputs_sha256={n: sha(raw) for n, raw in outputs.items()},
                   native_pressure_computed=False, old_producer_entrypoint_executed=False)
    (ROOT / (PREFIX + '_capsule.json')).write_text(json.dumps(capsule, indent=2) + '\n')
    print(json.dumps(capsule, indent=2))
    if process.returncode != 0 or not stable:
        raise SystemExit(process.returncode or 1)


if __name__ == '__main__':
    main()
