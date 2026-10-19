"""Archive exact supplier epoch, execute its real consumer, retain natural caps."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile
import joined_phi_u1_matrix_jordan_supplier as upstream

ROOT=Path(__file__).resolve().parent
SOURCE='joined_phi_u1_jordan_full_gaussian_supplier.py'
INPUTS=(
    'joined_phi_u1_matrix_jordan_supplier.py',
    'joined_phi_u1_matrix_jordan_supplier_receipt_20261010.npz',
    'joined_phi_u1_matrix_jordan_supplier_receipt_20261010.json',
    'induced_phi_uniform_selfconsistent_full_normal_quantum_final_root_actual_receipt.json',
    'induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json',
    'induced_phi_global_normal_gauss_chart_certificate_receipt.json',
    'induced_phi_global_normal_h_jet_control_receipt.json',
    'induced_phi_global_occupied_kato_derivative_control_receipt.json',
    'induced_phi_compact_source_global_domain_control_corrected_receipt_20261010.json',
    'joined_phi_u1_neutral_ground_graph_epoch2_20261010.json',
    'joined_phi_u1_neutral_ground_graph_epoch2_20261010.npz',
)


def execute(output: Path) -> dict[str, object]:
    captured={**upstream.old.CAPTURED,SOURCE:(ROOT/SOURCE).read_bytes(),Path(__file__).name:Path(__file__).read_bytes()}
    captured.update({name:(ROOT/name).read_bytes() for name in INPUTS})
    digest=lambda raw:hashlib.sha256(raw).hexdigest()
    archive=output.with_name(output.stem+'_preexecution_epoch.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in captured.items():z.writestr(name,raw)
    command=[sys.executable,str(ROOT/SOURCE),'--output',str(output)]
    started=time.monotonic()
    child=subprocess.run(command,cwd=ROOT,capture_output=True)
    elapsed=time.monotonic()-started
    stdout=output.with_name(output.stem+'_stdout.txt');stderr=output.with_name(output.stem+'_stderr.txt')
    stdout.write_bytes(child.stdout);stderr.write_bytes(child.stderr)
    unchanged={name:(ROOT/name).read_bytes()==raw for name,raw in captured.items()}
    result={'command':command,'natural_exit_code':child.returncode,'wall_seconds':elapsed,'source_at_execution':{name:digest(raw) for name,raw in captured.items()},'source_input_unchanged':unchanged,'preexecution_archive':{'path':archive.name,'sha256':digest(archive.read_bytes())},'stdout':{'path':stdout.name,'sha256':digest(child.stdout)},'stderr':{'path':stderr.name,'sha256':digest(child.stderr)},'output':{'path':output.name,'sha256':digest(output.read_bytes()) if output.exists() else None},'runtime':{'python':sys.executable,'version':sys.version}}
    capsule=output.with_name(output.stem+'_natural_exit.json');capsule.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'natural_exit_code':child.returncode,'wall_seconds':elapsed,'all_unchanged':all(unchanged.values()),'capsule':capsule.name},indent=2))
    if child.returncode:sys.stderr.buffer.write(child.stderr)
    if not all(unchanged.values()):raise RuntimeError('Source/input mutation during real scientific path')
    if child.returncode:raise SystemExit(child.returncode)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    execute(args.output)
