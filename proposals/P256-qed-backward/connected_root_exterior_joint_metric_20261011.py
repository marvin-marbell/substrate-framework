"""Captured actual-source joint metric: physical shared normalization, not leg Z^2."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any
import zipfile

import sympy as symbolic

ROOT = Path(__file__).resolve().parent
PINS = {
    'connected_root_exterior_actual2_corrected_20261011_receipt.json': 'f11d0d91b937ed6ee1d7db3fb72ea6cb799b3a89cc797fa3ffcf3a9797cd9656',
    'connected_root_exterior_mixed_actual2_corrected_20261011_receipt.json': 'e86164b6e788f587febc8a5872bc77e4cb56cf9ace54f973efd11d5c3872edc3',
    'normalized_root_graph_actual3_corrected_20261011_receipt.json': '1a83209037cefa55140f7ff45f4efa95460f4d7523e4d12b0b8a530fdd82ac60',
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def joint(wa: float, wb: float, overlap: float, pair: float) -> dict[str, float]:
    Z = 1+wa+wb+2*overlap+pair
    first_a = 2*(wa+overlap+pair)
    first_b = 2*(wb+overlap+pair)
    second = 2*overlap+4*pair
    response = (Z*second-first_a*first_b)/(Z*Z)
    a, b, A, B, D, G = symbolic.symbols('a b A B D G', real=True)
    F = 1+A*a*a+B*b*b+2*D*a*b+G*a*a*b*b
    derivative = symbolic.diff(symbolic.log(F), a, b)
    exact = ((2*D+4*G)*(1+A+B+2*D+G)-4*(A+D+G)*(B+D+G))/(1+A+B+2*D+G)**2
    assert symbolic.simplify(derivative.subs({a: 1, b: 1})-exact) == 0
    wrong_Z = (1+wa)*(1+wb)
    wrong_response = (wrong_Z*second-first_a*first_b)/(wrong_Z*wrong_Z)
    return {'joint_physical_normalizer': Z, 'disconnected_leg_product_wrong_normalizer': wrong_Z,
            'physical_connected_log_tag_response': response,
            'wrong_independent_leg_response': wrong_response,
            'wrong_shared_normalizer_difference': Z-wrong_Z,
            'primitive_degree22_log_coefficient': pair-wa*wb-overlap*overlap,
            'G1_a': wa, 'G1_b': wb, 'actual_coherent_overlap': overlap, 'G21': pair}


def worker(epoch: str) -> None:
    captured = {name: (ROOT/name).read_bytes() for name in PINS}
    assert all(sha(captured[n]) == pin for n, pin in PINS.items())
    same = json.loads(captured['connected_root_exterior_actual2_corrected_20261011_receipt.json'])
    mixed = json.loads(captured['connected_root_exterior_mixed_actual2_corrected_20261011_receipt.json'])
    fitted = json.loads(captured['normalized_root_graph_actual3_corrected_20261011_receipt.json'])
    assert same['integer_tree_matching_offsets_by_orientation'] == {'+': [1, 1], '-': [-1, -1]}
    N0 = same['source_norm']
    first = same['paths'][0]['material_second_shell_graph_test']
    t = first['t']
    raw1 = first['raw_first_oriented_G']['+']/N0**2
    raw2 = first['raw_second_oriented_G_T_U']['+']['G']/N0**2
    repeated = joint(t*t*raw1, t*t*raw1, t*t*raw1, t**4*raw2)
    repeated['e_degree27_response_envelope_load'] = math.e*27*abs(repeated['physical_connected_log_tag_response'])
    repeated['constant_response_envelope_less_than_one'] = repeated['e_degree27_response_envelope_load'] < 1
    unequal = mixed['results'][0]
    t2 = same['paths'][1]['material_second_shell_graph_test']['t']
    Wa, Wb = unequal['physical_first_source_weights']
    D = unequal['actual_cross_path_Gram_D'][0]
    Gab = unequal['complete_all64_relative_G_T_U_and_selected_V']['G']
    shared = joint(t*t*Wa, t2*t2*Wb, t*t2*D, t*t*t2*t2*Gab)
    shared['e_degree39_response_envelope_load'] = math.e*39*abs(shared['physical_connected_log_tag_response'])
    shared['constant_response_envelope_less_than_one'] = shared['e_degree39_response_envelope_load'] < 1
    leg = fitted['graph_interface']
    fitted_component = leg['firstshell_t']**4*(raw2-2*raw1*raw1)/leg['normalizer_Z']**2
    assert abs(repeated['wrong_shared_normalizer_difference']) > .01
    assert abs(shared['wrong_shared_normalizer_difference']) > 1e-4
    outcome: dict[str, Any] = {
        'kind': 'ACTUAL_PHYSICAL1_SHARED_JOINT_GRAPH_NORMALIZATION_AND_CONNECTED_RESPONSE',
        'input_sha256': PINS, 'own_source_sha256': sha(Path(__file__).read_bytes()),
        'exact_graph_source': '(I+b P_b)(I+a P_a)Omega; P_a=-t_a T_a, actual signed all64 primitive CAR and shared integer flows. This is a normalized trial graph, not two independent graph isometries or a ground.',
        'exact_metric': 'F(a,b)=1+Wa a^2+Wb b^2+2Re(D)ab+Gab a^2b^2; Zjoint=F(1,1).',
        'exact_physical_response': '(a d_a)(b d_b)logF at a=b=1=[Z(2ReD+4Gab)-4(Wa+ReD+Gab)(Wb+ReD+Gab)]/Z^2.',
        'symbolic_identity': True, 'same_edge_physical1': repeated, 'unequal_shared_paths_physical1': shared,
        'materially_different_fitted_degree2_graph': {
            'receipt': 'normalized_root_graph_actual3_corrected_20261011_receipt.json',
            'source': 'peer NEW actual3 corrected complete-action fitted graph; not this consumer execution',
            't': leg['firstshell_t'], 'u': leg['secondshell_u'], 'normalizer_Z': leg['normalizer_Z'],
            'four_firsthop_insertion_component_with_fitted_leg_metric': fitted_component,
            'component_is_not_full_fitted_graph_cumulant': True,
        },
        'allorder_boundary': 'For actual physical-response cumulants, require literal all-n connected tree domination and paid whole-H graph residual. If uniform adjacent-response envelope c exists, e*D*c<1 gives rooted summability. The measured repeated-edge physical response forces c at least its magnitude, so failed e*27*c excludes THIS constant-response-envelope criterion at physical1. It does not contradict the small degree22 insertion coefficient or refute other allorder representations.',
        'full_exterior_binding': 'Use corrected actual2 complete H0 moments and the separately paid full physical64 V exterior, never PHP squared. Passing/failing this scalar metric does not discharge Vcharge3/neutralQ or the graph residual.',
        'whole_physical_graph_normalization_not_independent_legs': True,
        'source_and_inputs_unchanged': all((ROOT/n).read_bytes() == raw for n, raw in captured.items()),
        'stage': 'exact metric calculus + actual binary64 source contractions; literal conditional allorder boundary',
        'fullPhi_ground': False, 'phase': False, 'campaign_complete': False,
    }
    assert outcome['source_and_inputs_unchanged']
    (ROOT/f'connected_root_exterior_{epoch}_receipt.json').write_text(json.dumps(outcome, indent=2)+'\n')
    print(json.dumps({'same_edge_response': repeated['physical_connected_log_tag_response'],
                      'same_edge_load': repeated['e_degree27_response_envelope_load'],
                      'unequal_response': shared['physical_connected_log_tag_response'],
                      'unequal_load': shared['e_degree39_response_envelope_load']}), flush=True)


def outer(epoch: str) -> None:
    captured = {name: (ROOT/name).read_bytes() for name in PINS}
    assert all(sha(captured[n]) == pin for n, pin in PINS.items())
    captured[Path(__file__).name] = Path(__file__).read_bytes()
    stem = f'connected_root_exterior_{epoch}'
    def archive(path: Path, members: dict[str, bytes]) -> None:
        with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as z:
            for name, raw in members.items():
                z.writestr(name, raw)
        path.chmod(0o444)
    pre = ROOT/f'{stem}_preexecution.zip'
    archive(pre, captured)
    work = Path(tempfile.mkdtemp(prefix=stem+'_frozen_'))
    start = time.monotonic()
    try:
        with zipfile.ZipFile(pre) as z:
            z.extractall(work)
        command = ['/home/administrator/substrate-framework/.venv/bin/python', '-B', str(work/Path(__file__).name), '--epoch', epoch, '--worker']
        process = subprocess.run(command, cwd=work, env=os.environ | {'PYTHONDONTWRITEBYTECODE': '1'}, capture_output=True)
        additions = {p.name: p.read_bytes() for p in work.iterdir() if p.is_file() and p.name not in captured}
        for name, raw in additions.items():
            with (ROOT/name).open('xb') as out:
                out.write(raw)
        for suffix, raw in [('stdout.txt', process.stdout), ('stderr.txt', process.stderr)]:
            with (ROOT/f'{stem}_{suffix}').open('xb') as out:
                out.write(raw)
        result = {'natural_returncode': process.returncode, 'natural_exit': True, 'elapsed_seconds': time.monotonic()-start,
                  'command': command, 'captured_sha256': {n: sha(raw) for n, raw in captured.items()},
                  'output_sha256': {n: sha(raw) for n, raw in additions.items()},
                  'preexecution_archive_sha256': sha(pre.read_bytes()),
                  'frozen_sources_inputs_unchanged': all((work/n).read_bytes() == raw for n, raw in captured.items())}
        raw_result = (json.dumps(result, indent=2)+'\n').encode()
        with (ROOT/f'{stem}_exit.json').open('xb') as out:
            out.write(raw_result)
        archive(ROOT/f'{stem}_executed.zip', captured | additions | {f'{stem}_exit.json': raw_result,
            f'{stem}_stdout.txt': process.stdout, f'{stem}_stderr.txt': process.stderr})
        print(json.dumps(result, indent=2), flush=True)
        if process.returncode:
            raise SystemExit(process.returncode)
    finally:
        shutil.rmtree(work)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    worker(args.epoch) if args.worker else outer(args.epoch)
