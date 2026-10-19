"""Entire central-domain K+gamma energy form, consuming new native joint operators.
The positive weight is explicit: a pure K form bound cannot include Pg memory.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile
import sympy as sp

ROOT = Path(__file__).resolve().parent
PREFIX = 'positive_residual_global_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'
GAMMA = F(46955312929, 34359738368)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def calculate(epoch):
    request = json.loads((ROOT/f'{PREFIX}{epoch}_request.json').read_bytes())
    for name, pin in request['input_sha256'].items():
        assert sha(ROOT/name) == pin
    native = json.loads((ROOT/'positive_residual_actual1_20261011_receipt.json').read_bytes())
    axes = []
    for axis in native['axes']:
        gl, gu = (F(x) for x in axis['G_spectrum']['exact'])
        g0l, g0u = (F(x) for x in axis['G_ground_expectation']['exact'])
        l, u = gl-g0u, gu-g0l
        lq, uq = min(l, F(0)), max(u, F(0))
        sweeps = []
        for sweep in axis['scalar_anchor_sweep']:
            a = F(sweep['anchor_exact'])
            bU = F(sweep['signed_B_spectrum']['exact'][1])
            rows = []
            for row in sweep['energies']:
                E = F(row['E_exact']); A = a-E
                wL, wU = (F(x) for x in row['vacuum_w0']['exact'])
                dU = F(row['joint_minus_B_plus_Tgram_over_floor']['exact'][1])
                mL, mU = g0l/A-wU, g0u/A-wL
                # S=K+gamma I>0 on ALL neutral vectors, including g.
                symUpper = (uq+(uq-lq)/8)/(GAMMA*A)
                symAbs = max(abs(l), abs(u))/(GAMMA*A)
                jointUpper = max(dU, F(0))/(GAMMA*A*A)
                negativeB = max(bU, F(0))/(GAMMA*A*A)
                # k/[(k+A)(k+gamma)]<=1/(A+gamma) for all k>=0.
                scalarLower = g0u/(A*(A+GAMMA))
                upper = symUpper+max(mU, F(0))/GAMMA+jointUpper
                lower = min(wU/GAMMA, symAbs+scalarLower+max(-mL, F(0))/GAMMA+negativeB)
                rows.append(dict(E_exact=str(E), all_central_vectors_weight='K+gamma I (including Pg and arbitrary Q combinations)', upper_relative_exact=str(upper), lower_relative_exact=str(lower), absolute_relative_exact=str(max(upper, lower)), upper_display=float(upper), lower_display=float(lower), absolute_display=float(max(upper, lower)), exact_components=dict(noncommuting_centered_G_upper=str(symUpper), signed_joint_upper=str(jointUpper), scalar_lower=str(scalarLower), signed_B_lower=str(negativeB)), original_Q_upper=row['every_Q_one_sided_upper_display'], original_Q_absolute=row['every_Q_absolute_display']))
            sweeps.append(dict(anchor_exact=str(a), energies=rows))
        best = min(sweeps, key=lambda r: F(r['energies'][1]['absolute_relative_exact']))
        axes.append(dict(axis=axis['axis'], all_domain_scalar_anchor_sweep=sweeps, best_E0_absolute_anchor=best['anchor_exact'], best_E0_absolute_display=best['energies'][1]['absolute_display']))
    # Negative centered Gram does NOT imply a negative anticommutator.
    U = sp.diag(1, 0); V = sp.Matrix([[1, 1], [1, 1]])/2
    sym = (U*V+V*U)/2
    assert sym.det() < 0
    assert (sym+sp.eye(2)/8).is_positive_definite
    assert sp.expand((U+V-sp.eye(2)/2)**2 + U-U**2+V-V**2) == U*V+V*U+sp.eye(2)/4
    # Nonzero Q Wg prevents ANY absolute K-only bound on the FULL domain.
    K = sp.diag(0, 2)
    C = sp.Matrix([[0, 1], [1, 0]])
    assert (3*K-C).det() < 0 and (3*K+C).det() < 0
    result = dict(kind='NEW_ALL_CENTRAL_VECTOR_POSITIVE_WEIGHT_ENERGY_FORM', epoch=epoch, governing_plan=request['governing_plan'], input_sha256=request['input_sha256'], source_sha256=sha(Path(__file__)), native_receipt_sha256=sha(ROOT/'positive_residual_actual1_20261011_receipt.json'), axes=axes, noncommuting_anticommutator_wrongcontrol=True, pure_K_global_bound_with_nonzero_QWPg_rejected=True, definition_of_all_vector_scope='Native main Qg bound pays every excited canonical vector; this supplement pays EVERY central vector including Pg in explicit positive K+gamma weight. Neither is a ground/source-column replacement.', physical1_verdict='FAILED_SUFFICIENT_GUARD_NOT_NO_GO', no_original_producer_main_eigensolver_root_checker_rerun=True, historical_formation_and_ground_premises_conditional=True, interacting_multicharge_phase_physicalPhi_ground_paid=False, campaign_complete=False)
    result['physical1_verdict'] = 'PASS_SCOPED_POSITIVE_WEIGHT_ONLY' if all(F(min(a['all_domain_scalar_anchor_sweep'], key=lambda s: F(s['energies'][1]['absolute_relative_exact']))['energies'][1]['absolute_relative_exact']) < 1 for a in axes) else 'FAILED_SUFFICIENT_GUARD_NOT_NO_GO'
    result['sources_inputs_unchanged'] = all(sha(ROOT/n) == p for n, p in request['input_sha256'].items())
    assert result['sources_inputs_unchanged']
    dump(ROOT/f'{PREFIX}{epoch}_receipt.json', result)
    print(json.dumps(dict(axes=[dict(axis=a['axis'], global_absolute=a['best_E0_absolute_display']) for a in axes], native_Q_caps=[dict(axis=a['axis'], upper=a['best_E0_one_sided_upper_display'], absolute=a['best_E0_absolute_display']) for a in native['axes']])), flush=True)


def execute(epoch):
    suffixes = ('request.json', 'preexecution.zip', 'executed.zip', 'exit.json', 'stdout.txt', 'stderr.txt', 'capsule.json', 'receipt.json')
    assert not any((ROOT/f'{PREFIX}{epoch}_{s}').exists() for s in suffixes), 'Immutable epoch exists'
    native = json.loads((ROOT/'positive_residual_actual1_20261011_receipt.json').read_bytes())
    native_capsule = json.loads((ROOT/'positive_residual_actual1_20261011_capsule.json').read_bytes())
    assert native_capsule['returncode'] == 0 and native_capsule['natural_exit'] and native_capsule['sources_inputs_unchanged']
    assert sha(ROOT/native_capsule['receipt']) == native_capsule['receipt_sha256']
    names = (Path(__file__).name, 'positive_residual_global_proof_20261011.json', 'positive_residual_actual1_20261011_receipt.json', 'positive_residual_actual1_20261011_capsule.json', 'positive_residual_actual1_20261011_exit.json', 'positive_residual_scientist_20261011.py', 'positive_residual_proof_20261011.json')
    pins = {name: sha(ROOT/name) for name in names}
    assert pins['positive_residual_scientist_20261011.py'] == native['source_sha256']
    request_path = ROOT/f'{PREFIX}{epoch}_request.json'
    dump(request_path, dict(epoch=epoch, input_sha256=pins, governing_plan=native['governing_plan']))
    pre = ROOT/f'{PREFIX}{epoch}_preexecution.zip'
    with zipfile.ZipFile(pre, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in pins:
            archive.write(ROOT/name, 'inputs/'+name)
        archive.write(request_path, 'request.json')
    start = time.monotonic()
    completed = subprocess.run([PYTHON, str(Path(__file__)), '--epoch', epoch, '--calculate'], cwd=ROOT, capture_output=True, check=False)
    stdout, stderr = ROOT/f'{PREFIX}{epoch}_stdout.txt', ROOT/f'{PREFIX}{epoch}_stderr.txt'
    stdout.write_bytes(completed.stdout); stderr.write_bytes(completed.stderr)
    unchanged = all(sha(ROOT/n) == pin for n, pin in pins.items())
    exit_path = ROOT/f'{PREFIX}{epoch}_exit.json'
    dump(exit_path, dict(returncode=completed.returncode, natural_exit=True, elapsed_seconds=time.monotonic()-start, sources_inputs_unchanged=unchanged, stderr_empty=not completed.stderr))
    receipt = ROOT/f'{PREFIX}{epoch}_receipt.json'; executed = ROOT/f'{PREFIX}{epoch}_executed.zip'
    with zipfile.ZipFile(executed, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in pins:
            archive.write(ROOT/name, 'inputs/'+name)
        for path in (request_path, stdout, stderr, exit_path):
            archive.write(path, path.name)
        if receipt.exists():
            archive.write(receipt, receipt.name)
    capsule = dict(epoch=epoch, preexecution_sha256=sha(pre), executed_sha256=sha(executed), receipt_sha256=sha(receipt) if receipt.exists() else None, exit_sha256=sha(exit_path), returncode=completed.returncode, natural_exit=True, sources_inputs_unchanged=unchanged, stderr_empty=not completed.stderr, input_sha256=pins)
    dump(ROOT/f'{PREFIX}{epoch}_capsule.json', capsule)
    print(json.dumps(dict(capsule=capsule, stdout=completed.stdout.decode(), stderr=completed.stderr.decode())), flush=True)
    return completed.returncode if unchanged else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', required=True)
    parser.add_argument('--calculate', action='store_true')
    args = parser.parse_args()
    if args.calculate:
        calculate(args.epoch)
    else:
        sys.exit(execute(args.epoch))
