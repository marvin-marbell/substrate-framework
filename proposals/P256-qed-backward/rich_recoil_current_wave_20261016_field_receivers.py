"""Saved-state field receiver readout; no new preparation or coupled evolution."""
from __future__ import annotations
from itertools import product
import json
from pathlib import Path
import pickle
import sys
from typing import Any
import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_recoil_current_wave_20261016_'


def primitive_receivers(law: Any) -> dict[str, Any]:
    labels = law.field['raw']['shell_states']
    cutoff = law.parameters['shell_cutoff']
    target = sorted([s for s in product(range(cutoff+2), repeat=4) if sum(s) <= cutoff+1], key=lambda s: (sum(s), s))
    index = {s: i for i, s in enumerate(target)}
    width = law.parameters['shell_width']
    qmaps, pmaps = [], []
    for axis in range(4):
        q = np.zeros((len(target), len(labels)), complex)
        p = np.zeros_like(q)
        for j, state in enumerate(labels):
            if state[axis]:
                moved = list(state); moved[axis] -= 1
                q[index[tuple(moved)], j] += width*np.sqrt(state[axis]/2)
                p[index[tuple(moved)], j] -= 1j*np.sqrt(state[axis]/2)/width
            moved = list(state); moved[axis] += 1
            q[index[tuple(moved)], j] += width*np.sqrt((state[axis]+1)/2)
            p[index[tuple(moved)], j] += 1j*np.sqrt((state[axis]+1)/2)/width
        qmaps.append(q); pmaps.append(p)
    rows = [index[s] for s in labels]
    transform = law.field['transform']
    lift = lambda matrix: transform.conj().T @ np.kron(np.eye(law.field['raw']['nphi']), matrix) @ transform
    return dict(Q=[lift(q[rows]) for q in qmaps], Pi=[lift(p[rows]) for p in pmaps],
        QQ=[[lift(a.conj().T@b) for b in qmaps] for a in qmaps],
        PiPi=[[lift(a.conj().T@b) for b in pmaps] for a in pmaps],
        QPi=[[lift(a.conj().T@b) for b in pmaps] for a in qmaps])


def receive(rho: np.ndarray, operators: dict[str, Any]) -> dict[str, Any]:
    q = np.array([np.trace(rho@a).real for a in operators['Q']])
    p = np.array([np.trace(rho@a).real for a in operators['Pi']])
    qq = np.array([[np.trace(rho@a).real for a in row] for row in operators['QQ']])
    pp = np.array([[np.trace(rho@a).real for a in row] for row in operators['PiPi']])
    qp = np.array([[np.trace(rho@a).real for a in row] for row in operators['QPi']])
    return dict(norm=float(np.trace(rho).real), Q=q.tolist(), Pi=p.tolist(), QQ=qq.tolist(), PiPi=pp.tolist(),
        covariance_QQ=(qq-np.outer(q, q)).tolist(), covariance_PiPi=(pp-np.outer(p, p)).tolist(),
        symmetrized_covariance_QPi=(qp-np.outer(q, p)).tolist())


def differences(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {key: (np.array(a[key])-b[key]).tolist() for key in ['Q', 'Pi', 'QQ', 'PiPi', 'covariance_QQ', 'covariance_PiPi', 'symmetrized_covariance_QPi']}


def main() -> None:
    stamp = sys.argv[1]
    stem = ROOT/(PREFIX+stamp+'_')
    inputs = json.loads(Path(str(stem)+'input.json').read_text())
    completed = json.loads(Path(str(stem)+'result.json').read_text())
    with Path(str(stem)+'constructed_model.pkl').open('rb') as stream:
        law = pickle.load(stream)
    operators = primitive_receivers(law)
    snapshots = [('initial', 0.), ('short_start', inputs['short_start'])]
    steps = int(round((inputs['duration']-inputs['short_start'])/inputs['dt']))
    snapshots += [(f't{i}', inputs['short_start']+i*inputs['dt']) for i in range(1, steps+1)]
    cases = {}
    for case in completed['cases']:
        with np.load(Path(str(stem)+case+'_initial_arrays.npz')) as source:
            initial = source['field_density']
        rows = []
        for endpoint, absolute_time in snapshots:
            with np.load(Path(str(stem)+case+'_'+endpoint+'_arrays.npz')) as source:
                coupled = receive(source['field_density'], operators)
            # Exact analytic receiver of the ALREADY specified independent control
            # generator on its OWN saved clock0 density, not a coupled-state copy.
            u = expm(-1j*absolute_time*law.field['boson'])
            frozen = receive(u@initial@u.conj().T, operators)
            rows.append(dict(absolute_time=absolute_time, coupled=coupled, frozen=frozen,
                coupled_minus_frozen=differences(coupled, frozen)))
        cases[case] = rows
    contrasts = {}
    for number in [54, 55, 53]:
        for protocol in ['xcos', 'ycos']:
            name = f'N{number}_{protocol}'
            contrasts[name] = []
            for i, row in enumerate(cases[name]):
                base = cases[f'N{number}_baseline'][i]
                contrasts[name].append(dict(absolute_time=row['absolute_time'],
                    pulse_minus_baseline=differences(row['coupled'], base['coupled']),
                    frozen_pulse_minus_baseline=differences(row['frozen'], base['frozen']),
                    interaction_increment=differences(row['coupled_minus_frozen'], base['coupled_minus_frozen'])))
    excess = {}
    for number in [55, 53]:
        for protocol in ['baseline', 'xcos', 'ycos']:
            name = f'N{number}_{protocol}'
            excess[name] = [dict(absolute_time=a['absolute_time'], coupled_charge_minus_matched_neutral=differences(a['coupled'], b['coupled']))
                for a, b in zip(cases[name], cases[f'N54_{protocol}'])]
    result = dict(kind='EXISTING_SAVED_STATE_FIELD_RECEIVERS_NOT_NEW_EVOLUTION', receiver_order=['xcos','xsin','ycos','ysin'],
        model_units='hbar=1, Hermite width1; Q/Pi have reciprocal canonical field units, before-cut quadratic receivers are NOT projected operator squares.',
        cases=cases, pulse_contrasts=contrasts, matched_neutral_excess=excess,
        caveat='Actual finite projected-Q pulse is retained, NOT called a mode-pure unbounded Weyl source. Body invariance alone is not field-receiver invariance. Initial populations/means/covariances and frozen finite-regulator evolution are shown separately before assigning later changes to coupling/emission. No improved pulse, new TDVP, on-shell photon or emission-probability claim.')
    Path(str(stem)+'field_receivers_result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(cases=len(cases), initial_receiver_changes={key: values[0]['pulse_minus_baseline'] for key, values in contrasts.items()})), flush=True)


if __name__ == '__main__':
    main()
