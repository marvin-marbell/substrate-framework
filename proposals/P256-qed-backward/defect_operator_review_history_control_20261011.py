from __future__ import annotations

import hashlib
import itertools
import json
import math
import subprocess
import sys
import time
from fractions import Fraction as F
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
PREFIX = 'defect_operator_review_history_actual1_20261011_'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name: str, obj: dict) -> None:
    with (ROOT/name).open('x') as stream:
        json.dump(obj, stream, indent=2)
        stream.write('\n')


def worker() -> None:
    # Two canonical CAR masks 0011/1100, with a rational exact ground
    # direction. This is a small operator control, not actual native g.
    cells = 8
    g = {3: F(3, 5), 12: F(4, 5)}
    omega = {(0, masks): math.prod((g[m] for m in masks), start=F(1))
             for masks in itertools.product(g, repeat=cells)}
    def shift(state, sign):
        return {(flow+sign, masks): value for (flow, masks), value in state.items()}
    def ground_at_cell(state, cell):
        result = {}
        for (flow, masks), amplitude in state.items():
            if flow != 0:
                continue
            for mask, value in g.items():
                target = masks[:cell]+(mask,)+masks[cell+1:]
                key = (flow, target)
                result[key] = result.get(key, F(0))+amplitude*g[masks[cell]]*value
        return {key: value for key, value in result.items() if value}
    def count(state):
        result = {key: cells*value for key, value in state.items()}
        for cell in range(cells):
            for key, value in ground_at_cell(state, cell).items():
                result[key] = result.get(key, F(0))-value
        return {key: value for key, value in result.items() if value}
    assert sum(value*value for value in omega.values()) == 1
    first = shift(omega, 1)
    returned = shift(first, -1)
    assert returned == omega and count(returned) == {}
    assert count(first) == {key: cells*value for key, value in first.items()}
    # The false assertion is tested on the actual state count, not just w.
    try:
        assert count(returned) == {key: 2*value for key, value in returned.items()}
    except AssertionError:
        history_count_rejected = True
    else:
        raise RuntimeError('history-as-defect assertion unexpectedly passed')
    # The exact ground projection product is distinct from pure flow0.
    masks = (3,)*cells
    neutral_basis = {(0, masks): F(1)}
    assert count(neutral_basis) != {}
    receipt = {'scope': 'EXACT_SMALL_CAR_RATIONAL_GROUND_SHARED_ROTOR_CONTROL_NOT_NATIVE_G_CONSTRUCTION', 'cycle_cells': cells, 'history': ['W_cycle', 'W_cycle_adjoint'], 'rotor': 'integer Z with signed shift; no cutoff or cyclic alias', 'first_state_defects': cells, 'returned_state_defects': 0, 'historical_insertions': 2, 'history_count_rejected_by_actual_Delta_operator': history_count_rejected, 'zero_flow_neutral_memory_nonzero': True, 'source_sha256': sha(Path(__file__)), 'physical_law_claim': False}
    save('defect_operator_review_history_receipt_20261011.json', receipt)
    print(json.dumps(receipt), flush=True)


def main() -> None:
    names = [Path(__file__).name, 'defect_operator_review_capsule_20261011.json', 'matching_defect_interface_20261011.json', 'matching_defect_projector_notation_addendum_20261011.json']
    pins = {name: sha(ROOT/name) for name in names}
    assert pins['matching_defect_interface_20261011.json'] == 'e3adabe763cfc0c32716780950f2be6f61027cd926db9fd0039db2a924a306e4'
    pre = PREFIX+'preexecution.zip'
    with ZipFile(ROOT/pre, 'x', compression=ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(ROOT/name, name)
    command = ['/home/administrator/substrate-framework/.venv/bin/python', '-B', Path(__file__).name, '--worker']
    started = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
    stdout, stderr = PREFIX+'stdout.txt', PREFIX+'stderr.txt'
    for name, data in ((stdout, result.stdout), (stderr, result.stderr)):
        with (ROOT/name).open('xb') as stream:
            stream.write(data)
    core = {'command': command, 'returncode': result.returncode, 'natural_exit': True, 'elapsed_seconds': time.monotonic()-started, 'input_source_sha256': pins, 'unchanged': all(sha(ROOT/name) == pin for name, pin in pins.items()), 'preexecution_sha256': sha(ROOT/pre), 'stdout_sha256': sha(ROOT/stdout), 'stderr_sha256': sha(ROOT/stderr), 'target_reruns': 0}
    exit_name = PREFIX+'exit.json'
    save(exit_name, core)
    receipt = 'defect_operator_review_history_receipt_20261011.json'
    executed = PREFIX+'executed.zip'
    with ZipFile(ROOT/executed, 'x', compression=ZIP_DEFLATED) as archive:
        for name in names+[stdout, stderr, exit_name]+([receipt] if result.returncode == 0 else []):
            archive.write(ROOT/name, name)
    save('defect_operator_review_history_capsule_20261011.json', {**core, 'preexecution_zip': pre, 'executed_zip': executed, 'executed_sha256': sha(ROOT/executed), 'exit_sha256': sha(ROOT/exit_name), 'receipt_sha256': sha(ROOT/receipt) if result.returncode == 0 else None})
    print(json.dumps(core), flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        worker()
    else:
        main()
