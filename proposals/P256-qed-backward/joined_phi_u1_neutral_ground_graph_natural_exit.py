"""Archive exact inputs before executing the actual neutral graph consumer."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from typing import Any
import joined_phi_u1_neutral_ground_graph_consumer as consumer

ROOT = Path(__file__).resolve().parent


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def run(stem: str) -> dict[str, Any]:
    source = ROOT/'joined_phi_u1_neutral_ground_graph_consumer.py'
    output = ROOT/(stem+'.json')
    capsule = ROOT/(stem+'_natural_exit.json')
    pre = ROOT/(stem+'_preexecution.zip')
    raw: dict[str, bytes] = dict(consumer.field.CAPTURED)
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == '.py' and path.parent == ROOT:
                raw[path.name] = path.read_bytes()
    raw[Path(__file__).name] = Path(__file__).read_bytes()
    inputs = ['induced_phi_uniform_selfconsistent_full_normal_quantum_final_root_actual_receipt.json','induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json','induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json','joined_phi_u1_ground_photon_collective_delivery_receipt_20261010.json','joined_phi_u1_ground_photon_collective_delivered_actual_20261010.json','joined_phi_u1_stationary_jordan_parent_packet_20261010.json','joined_phi_u1_matrix_jordan_supplier_receipt_20261010.json','joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz']
    for name in inputs:
        raw[name] = (ROOT/name).read_bytes()
    with zipfile.ZipFile(pre, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, contents in raw.items():
            archive.writestr(name, contents)
    started = time.monotonic()
    process = subprocess.run([sys.executable, str(source), '--output', str(output)], cwd=ROOT, capture_output=True, text=True)
    elapsed = time.monotonic()-started
    unchanged = all((ROOT/name).read_bytes() == contents for name, contents in raw.items())
    record = dict(kind='ACTUAL_FILE_NATURAL_EXIT_IMMUTABLE_EPOCH',command=[sys.executable,str(source),'--output',str(output)],returncode=process.returncode,natural_exit=True,wall_seconds=elapsed,stdout=process.stdout,stderr=process.stderr,all_sources_inputs_unchanged=unchanged,preexecution=dict(path=pre.name,sha256=sha(pre.read_bytes())),sha256={name:sha(contents) for name,contents in raw.items()},outputs={path.name:sha(path.read_bytes()) for path in (output, output.with_suffix('.zip'), output.with_suffix('.npz')) if path.exists()})
    capsule.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(returncode=process.returncode,wall_seconds=elapsed,capsule=capsule.name,stderr=process.stderr,stdout=process.stdout), indent=2))
    if process.returncode != 0 or not unchanged:
        raise RuntimeError('Consumer failed or archived epoch changed; capsule preserved')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stem', required=True)
    args = parser.parse_args()
    run(args.stem)
