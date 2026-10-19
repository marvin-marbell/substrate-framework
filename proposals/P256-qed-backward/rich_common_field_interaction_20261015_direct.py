"""Actual same-H direct-action first spatial, two-profile connected transition.

The two field legs share a declared middle time. BOTH ordered products are
retained because projected coordinates need not commute. This is an overlapping
finite-pulse transition diagnostic, not separated asymptotic photon scattering.
No factor compilation, cached evolved state, stationarity or external-A contact.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any
import zipfile

import numpy as np
from scipy.sparse.linalg import LinearOperator

from rich_common_field_interaction_20261015_consume import lanczos, complex_pair

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinned_sources(request: dict[str, Any]) -> None:
    for item in request['source_pins']:
        path = ROOT / item['file']
        if digest(path) != item['sha256']:
            raise ValueError('Changed direct source pin: ' + str(path))
    capsule = ROOT / request['model_capsule']['file']
    if digest(capsule) != request['model_capsule']['sha256']:
        raise ValueError('Changed corrected model capsule')
    with zipfile.ZipFile(capsule) as archive:
        for name in archive.namelist():
            # Owner's capture CLI is never imported/replayed by either backend.
            # Its later edits do not change the pinned scientific dependencies.
            if name == 'rich_common_carrier_motion_20261015_capture.py':
                continue
            if name.endswith('.py'):
                path = ROOT / name
                if not path.is_file() or path.read_bytes() != archive.read(name):
                    raise ValueError('Captured Python source changed: ' + name)
    # The direct H imports occur ONLY AFTER exact corrected dependency checks.


class Projection(LinearOperator):
    def __init__(self, operator: Any, target: Any, source: Any | None = None):
        self.operator, self.target = operator, target
        self.source = target if source is None else source
        super().__init__(dtype=np.dtype(complex), shape=(target.shape[1], self.source.shape[1]))

    def _matvec(self, x: Any) -> Any:
        return self.target.conj().T @ (self.operator @ (self.source @ x))

    def _rmatvec(self, x: Any) -> Any:
        return self.source.conj().T @ (self.operator.H @ (self.target @ x))


class NumericCache:
    """Exact owner DATA reuse, never owner-evolved source states."""
    def __init__(self, request: dict[str, Any]):
        from rich_common_carrier_motion_20261015_factors import load
        self.load = load
        pin = request['numeric_cache']
        for record in (pin['manifest'], pin['states'], pin['loader']):
            if digest(ROOT / record['file']) != record['sha256']:
                raise ValueError('Changed published numeric input')
        self.data = json.loads((ROOT / pin['manifest']['file']).read_text())
        if self.data['stage'] != 'NATURAL_COMPLETION':
            raise ValueError('Incomplete owner numeric export')
        if self.data['model_sha256'] != '9b8912187ff32a90dd70fb03343a99a70d07153e6ee1bb9eeaefab414e16e7cc':
            raise ValueError('Different owner H')
        parameters = json.loads((ROOT / request['physical_input']).read_text())
        protocol_keys = {'consumer', 'time', 'krylov_steps', 'fine_steps', 'deadline_seconds', 'single_decision_control', 'scope'}
        if any(self.data['parameters'].get(key) != value for key, value in parameters.items() if key not in protocol_keys):
            raise ValueError('Different owning law parameters')
        with np.load(ROOT / pin['states']['file']) as states:
            self.reference = np.array(states['neutral_prepared_t0'], dtype=complex)
        self.operators: dict[str, Any] = {}

    def operator(self, label: str) -> Any:
        if label not in self.operators:
            record = self.data['snapshots'][label]
            self.operators[label] = self.load(ROOT / record['file'], record['sha256'])
        return self.operators[label]


def main() -> None:
    path, out = Path(sys.argv[1]), Path(sys.argv[2])
    request: dict[str, Any] = json.loads(path.read_text())
    pinned_sources(request)
    from rich_common_carrier_motion_20261015_model import build, law, source, prepared_reference
    from rich_common_carrier_motion_20261015_operators import field_observable
    from rich_frame_field_join_20261015_calculate import m_isometry
    started = time.monotonic()
    cache = NumericCache(request) if request.get('numeric_cache') else None
    model = None
    v0: Any = None
    if cache is not None:
        if request['spins'] != [.5]:
            raise ValueError('Published cache is M+.5 only; never fake a down-spin evolution')
        z = cache.reference
        h0 = cache.operator('neutral_H_zero')
        q0 = [cache.operator('neutral_Q'+str(axis)) for axis in (2, 5)]
        embedding_loss = cache.data['original_preparation']['M0_embedding_loss']
    else:
        parameters = json.loads((ROOT / request['physical_input']).read_text())
        model = build(parameters)
        reference = prepared_reference(model)
        v0 = m_isometry(model, 'neutral', 0.)
        z = v0.conj().T @ reference
        h0 = Projection(law(model, 'neutral'), v0)
        q0 = [Projection(field_observable(model, 'neutral', 'Q', axis), v0) for axis in (2, 5)]
        embedding_loss = float(np.linalg.norm(reference - v0 @ z))
    proof_path = ROOT / request['off_factorization']['file']
    if digest(proof_path) != request['off_factorization']['sha256']:
        raise ValueError('Changed exact physical-OFF factorization witness')
    result: dict[str, Any] = dict(kind='ACTUAL_CAR36_OVERLAPPING_TE_TM_LEG_SPATIAL_TRANSITION',
        action_backend='EXACT_PUBLISHED_NUMERIC_FACTORS' if cache is not None else 'DIRECT_FULL_FORM_OPERATOR',
        numeric_cache=request.get('numeric_cache'),
        request_sha256=digest(path), model_capsule=request['model_capsule'],
        preparation=dict(norm_squared=float(np.linalg.norm(z)**2),
                         M0_embedding_loss=embedding_loss,
                         semantics='Original actual-H prepared t0, explicitly NONSTATIONARY'),
        field_times=[request['half_time'], request['half_time']],
        source_time=0., receiver_time=2*request['half_time'],
        source_site=request['source_site'], receiver_site=request['receiver_site'],
        control=dict(kind='EXACT_SAME_STATE_COMPLETE_TRANSVERSE_OFF_FACTORIZATION',
                     proof=request['off_factorization'], connected=[0., 0.],
                     meaning='Analytic tensor factorization, NOT a fabricated propagation result'),
        rows=[], decision='NOT_YET_OBSERVED')
    def save(stage: str) -> None:
        result['stage'] = stage
        result['elapsed_seconds'] = time.monotonic() - started
        out.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
        print(stage, flush=True)
    save('CORRECTED_DIRECT_H_AND_ORIGINAL_PREPARATION_AVAILABLE')
    before = time.monotonic()
    hz = h0 @ z
    result['actual_H_matvec_seconds'] = time.monotonic() - before
    result['prepared_nonstationarity_residual'] = float(np.linalg.norm(hz - np.vdot(z, hz)*z))
    save('ACTUAL_FULL_H_ACTION_TIMED')
    half = request['half_time']
    schedules = [(request['coarse_steps'], 1), (request['fine_steps'], 1), (request['fine_steps'], 2)]
    neutral: list[dict[str, Any]] = []
    for steps, split in schedules:
        receipts: list[dict[str, Any]] = []
        def evolve0(state: Any, duration: float) -> Any:
            for _ in range(split):
                state, receipt = lanczos(h0, state, duration/split, steps)
                receipts.append(dict(duration=duration/split, **receipt))
            return state
        middle = evolve0(z, half)
        endpoint = evolve0(z, 2*half)
        means = [complex(np.vdot(middle, q @ middle)) for q in q0]
        field_pairs = [complex(np.vdot(middle, q0[a] @ (q0[b] @ middle))) for a, b in ((0, 1), (1, 0))]
        neutral.append(dict(endpoint=endpoint, means=means, field_pairs=field_pairs, evolution=receipts))
    result['neutral_resolution'] = dict(
        endpoint_coarse_split_L2=float(np.linalg.norm(neutral[0]['endpoint']-neutral[2]['endpoint'])),
        endpoint_fine_split_L2=float(np.linalg.norm(neutral[1]['endpoint']-neutral[2]['endpoint'])))
    save('ACTUAL_TWO_SIDED_NEUTRAL_EVOLUTION')
    for branch in ('lepton', 'antilepton'):
        full_h = law(model, branch) if cache is None else None
        for spin in request['spins']:
            if cache is not None:
                h = cache.operator(branch+'_H_zero')
                sin = cache.operator(branch+'_source_site'+str(request['source_site']))
                sout = cache.operator(branch+'_source_site'+str(request['receiver_site']))
                qs = [cache.operator(branch+'_Q'+str(axis)) for axis in (2, 5)]
            else:
                v = m_isometry(model, branch, float(spin))
                h = Projection(full_h, v)
                sin = Projection(source(model, branch, request['source_site'], spin), v, v0)
                sout = Projection(source(model, branch, request['receiver_site'], spin), v, v0)
                qs = [Projection(field_observable(model, branch, 'Q', axis), v) for axis in (2, 5)]
            incoming: Any = sin @ z
            receiver0: Any = sout @ z
            outputs: list[list[dict[str, complex]]] = []
            state_errors: list[dict[str, Any]] = []
            for index, (steps, split) in enumerate(schedules):
                receipts = []
                def evolve(state: Any, duration: float) -> Any:
                    for _ in range(split):
                        state, receipt = lanczos(h, state, duration/split, steps)
                        receipts.append(dict(duration=duration/split, **receipt))
                    return state
                ket = evolve(incoming, half)
                bra = evolve(sout @ neutral[index]['endpoint'], -half)
                g = complex(np.vdot(bra, ket))
                triples = [complex(np.vdot(bra, q @ ket)) for q in qs]
                moments = []
                for ordering, (a, b) in enumerate(((0, 1), (1, 0))):
                    full = complex(np.vdot(bra, qs[a] @ (qs[b] @ ket)))
                    means = neutral[index]['means']
                    disconnected = g*neutral[index]['field_pairs'][ordering] + triples[a]*means[b] + triples[b]*means[a] - 2*g*means[a]*means[b]
                    moments.append(dict(full=full, disconnected=disconnected,
                                        connected=full-disconnected, pair=g,
                                        three_TE=triples[0], three_TM=triples[1],
                                        mean_TE=means[0], mean_TM=means[1],
                                        field_pair=neutral[index]['field_pairs'][ordering]))
                outputs.append(moments)
                state_errors.append(dict(ket=ket, bra=bra, evolution=receipts))
            for ordering, label in enumerate(('Q_TEz_Q_TMz', 'Q_TMz_Q_TEz')):
                moments = outputs[2][ordering]
                coarse = max(abs(outputs[0][ordering][key]-value) for key, value in moments.items())
                fine = max(abs(outputs[1][ordering][key]-value) for key, value in moments.items())
                connected_error = max(abs(outputs[index][ordering]['connected']-moments['connected']) for index in (0, 1))
                magnitude = abs(moments['connected'])
                result['rows'].append(dict(branch=branch, physical_frame_spin_M=spin,
                    field_operator_order=label, moments={key:complex_pair(value) for key,value in moments.items()},
                    source_retained_raw_norm_squared=float(np.linalg.norm(incoming)**2),
                    receiver_retained_raw_norm_squared=float(np.linalg.norm(receiver0)**2),
                    resolution=dict(all_moments_coarse_split_max=coarse,
                                    all_moments_fine_split_max=fine,
                                    connected_empirical_error=connected_error,
                                    ket_coarse_split_L2=float(np.linalg.norm(state_errors[0]['ket']-state_errors[2]['ket'])),
                                    ket_fine_split_L2=float(np.linalg.norm(state_errors[1]['ket']-state_errors[2]['ket'])),
                                    bra_coarse_split_L2=float(np.linalg.norm(state_errors[0]['bra']-state_errors[2]['bra'])),
                                    bra_fine_split_L2=float(np.linalg.norm(state_errors[1]['bra']-state_errors[2]['bra']))),
                    magnitude_against_exact_physical_OFF=magnitude,
                    resolved=magnitude>request['resolution_margin']*connected_error and magnitude>request['absolute_floor'],
                    direct_independent_spin_evolution=True,
                    evolution=state_errors[2]['evolution']))
            result['decision'] = 'POSITIVE_FINITE_SAME_H_TWO_LEG_MECHANISM' if any(row['resolved'] for row in result['rows']) else 'NO_RESOLVED_TWO_LEG_MECHANISM_IN_COMPLETED_CHANNELS'
            save('FIRST_ACTUAL_PHYSICAL_DISCRIMINATOR' if len(result['rows']) == 2 else 'ACTUAL_SIGNED_SPIN_CHANNEL_COMPLETE')
            synthesis = dict(kind='ACTUAL_DIRECT_SPATIAL_TWO_FIELD_LEG_PROVISIONAL',
                decision=result['decision'], completed_rows=len(result['rows']),
                observed_branches=sorted({row['branch'] for row in result['rows']}),
                observed_independently_evolved_frame_spins=sorted({row['physical_frame_spin_M'] for row in result['rows']}),
                maximum_connected=max(row['magnitude_against_exact_physical_OFF'] for row in result['rows']),
                maximum_connected_empirical_error=max(row['resolution']['connected_empirical_error'] for row in result['rows']),
                control=result['control'], source_site=request['source_site'], receiver_site=request['receiver_site'],
                field_times=result['field_times'], source_time=0., receiver_time=2*half,
                limits=['Two actual distinct TEz/TMz profile legs with declared coincident middle time and BOTH ordered products',
                        'Finite nonstationary unamputated transition, not asymptotic scattering or event probability',
                        'All compatible J retained, physical frame spins directly evolved, no cache compression',
                        'Source overlap and signed density are not particle/event probabilities'])
            out.with_name(out.name+'_provisional_synthesis.json').write_text(json.dumps(synthesis, indent=2)+'\n')
    result['peak_RSS_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    save('NATURAL_COMPLETION')


if __name__ == '__main__':
    main()
