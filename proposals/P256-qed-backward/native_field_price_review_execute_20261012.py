"""Pin NEW review inputs before launching only the NEW scientific consumer."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = 'native_field_price_review_actual1_20261012'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'
CONTRACT = Path('/home/administrator/.omp/agent/sessions/-/2026-10-09T16-26-59-293Z_01a1217d-415d-7376-a17d-16be721cbe28/local/issue222-native-loop-field-price-auxiliary-adjoint-proof-review-contract-20261012.json')


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def create_json(path: Path, value: dict) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')


def main() -> None:
    contract_raw = CONTRACT.read_bytes()
    assert digest(contract_raw) == '8a191a68d6092a8cf89b05b4270683be298e2b94632ac4b5598c371db2035741'
    contract = json.loads(contract_raw)
    selected = [
        'native_field_price_review_consumer_20261012.py',
        'native_field_price_review_execute_20261012.py',
        'native_field_price_review_proof_20261012.json',
        'native_field_price_review_language_20261012.json',
        'native_loop_dark_delivery_20261012.json',
        'native_loop_dark_proof_20261012.json',
        'native_loop_dark_auxiliary_proof_20261012.json',
        'native_loop_dark_scientific_scope_corrections_20261012.json',
        'native_loop_dark_supported_epoch1_correction_20261012.json',
        'native_loop_dark_supported_actual1_20261012_capsule.json',
        'native_loop_dark_supported_actual2_20261012_capsule.json',
        'native_loop_dark_supported_actual2_20261012_receipt.json',
        'native_loop_dark_supported_actual3_20261012_capsule.json',
        'native_loop_dark_supported_actual3_20261012_receipt.json',
        'native_loop_dark_supported_consumer_20261012.py',
        'full_loop_graph_proof_20261012.json',
        'full_primitive_graph_review_proof_20261012.json',
        'loop_fractional_graph_proof_20261012.json',
        'higher_graph_review_proof_20261012.json',
        'source_tail_review_proof_20261012.json',
        'carrier_support_review_proof_20261012.json',
        'joined_phi_u1_uniform_gauss_limit_epoch4_20261010_actual_executed_inputs.zip',
        'native_loop_dark_parent_review_postexecution_snapshot_20261012.zip'
    ]
    snapshots = {name: (ROOT / name).read_bytes() for name in selected}
    for name, pin in contract['snapshot']['pins'].items():
        if name in snapshots:
            assert digest(snapshots[name]) == pin
    assert digest(snapshots[contract['snapshot']['path']]) == contract['snapshot']['sha256']
    governor = {key: contract[key] for key in ('plan_id', 'captured_revision', 'accepted_scoped_science', 'template_id', 'template_revision', 'work_item_id', 'requirement_ids', 'parent_only_plan_writer')}
    request = {'epoch': PREFIX, 'python': PYTHON, 'consumer': selected[0],
               'pins': {name: digest(raw) for name, raw in snapshots.items()},
               'contract_sha256': digest(contract_raw), 'governor': governor,
               'profile_archive': 'joined_phi_u1_uniform_gauss_limit_epoch4_20261010_actual_executed_inputs.zip',
               'profile_member': 'joined_phi_u1_stationary_flat_holonomy_typed_certified_20261010_receipt.json',
               'profile_member_sha256': '1c79b1a74c4236079645ebf9765a48c1c2263ec7b7209e9f103704c98752da94'}
    request_path = ROOT / (PREFIX + '_request.json')
    create_json(request_path, request)
    pre = ROOT / (PREFIX + '_preexecution.zip')
    with zipfile.ZipFile(pre, 'x', zipfile.ZIP_DEFLATED) as z:
        z.writestr('request.json', request_path.read_bytes())
        z.writestr('contract.json', contract_raw)
        for name, raw in snapshots.items():
            z.writestr('inputs/' + name, raw)
    pre_pin = digest(pre.read_bytes())
    receipt = ROOT / (PREFIX + '_receipt.json')
    command = [PYTHON, '-B', '-c',
               'import sys,zipfile;z=zipfile.ZipFile(sys.argv[1]);raw=z.read(sys.argv[2]);source=sys.argv[3];sys.argv=[source,"--snapshot",sys.argv[1],"--output",sys.argv[4]];exec(compile(raw,source,"exec"),{"__name__":"__main__","__file__":source})',
               str(pre), 'inputs/' + selected[0], str(ROOT / selected[0]), str(receipt)]
    started = time.time(); clock = time.monotonic()
    # No timeout/kill/success fallback: exact natural return including failures.
    child = subprocess.run(command, cwd=ROOT, capture_output=True)
    elapsed = time.monotonic() - clock
    stdout = ROOT / (PREFIX + '_stdout.txt'); stderr = ROOT / (PREFIX + '_stderr.txt')
    with stdout.open('xb') as stream: stream.write(child.stdout)
    with stderr.open('xb') as stream: stream.write(child.stderr)
    exit_path = ROOT / (PREFIX + '_exit.json')
    create_json(exit_path, {'exit_code': child.returncode, 'natural_exit': True, 'started_unix': started,
                           'elapsed_seconds': elapsed, 'stdout_sha256': digest(child.stdout),
                           'stderr_sha256': digest(child.stderr)})
    unchanged = all((ROOT / name).read_bytes() == raw for name, raw in snapshots.items())
    assert digest(pre.read_bytes()) == pre_pin
    executed = ROOT / (PREFIX + '_executed.zip')
    with zipfile.ZipFile(executed, 'x', zipfile.ZIP_DEFLATED) as z:
        z.writestr('request.json', request_path.read_bytes()); z.writestr('contract.json', contract_raw)
        for name, raw in snapshots.items(): z.writestr('inputs/' + name, raw)
        for path in (stdout, stderr, exit_path, receipt):
            if path.exists(): z.writestr('outputs/' + path.name, path.read_bytes())
    capsule = {'kind': 'NEW_INDEPENDENT_NATIVE_FIELD_PRICE_REVIEW_NATURAL_EXIT',
               'governor': governor, 'pins': request['pins'], 'argv': command,
               'source_selection': 'compile NEW review source bytes from BEFORE-execution ZIP; historical sources never executed',
               'preexecution': {'path': pre.name, 'sha256': pre_pin, 'created_before_child': True},
               'executed': {'path': executed.name, 'sha256': digest(executed.read_bytes())},
               'request': {'path': request_path.name, 'sha256': digest(request_path.read_bytes())},
               'receipt': {'path': receipt.name, 'sha256': digest(receipt.read_bytes())} if receipt.exists() else None,
               'exit_code': child.returncode, 'natural_exit': True, 'inputs_unchanged': unchanged,
               'elapsed_seconds': elapsed, 'failed_attempts': [] if child.returncode == 0 else [PREFIX],
               'actual_ground_or_backend': False, 'scientific_admission': False, 'campaign_complete': False}
    create_json(ROOT / (PREFIX + '_capsule.json'), capsule)
    print(json.dumps({'capsule': PREFIX + '_capsule.json', 'exit_code': child.returncode,
                      'natural_exit': True, 'inputs_unchanged': unchanged, 'elapsed_seconds': elapsed}))
    if child.returncode != 0: sys.exit(child.returncode)
    if not unchanged: sys.exit(2)


if __name__ == '__main__':
    main()
