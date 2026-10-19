"""Pinned same-CAR36 nonstationary, graded connected two-field-leg consumer.

Only explicit trusted-local numerical factor snapshots and the declared t0
reference are inputs. No owner model imports, evolved states, or mutable H.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from itertools import permutations, product
from pathlib import Path
import resource
import sys
import time
from typing import Any

import numpy as np
from scipy import linalg

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def complex_pair(z: complex) -> list[float]:
    return [float(np.real(z)), float(np.imag(z))]


def lanczos(h: Any, state: Any, duration: float, steps: int) -> tuple[Any, dict[str, Any]]:
    """Fully reorthogonalized Hermitian action; zero sources stay exactly zero."""
    length = float(np.linalg.norm(state))
    if duration == 0 or length == 0:
        return state.copy(), {'dimension': 0, 'endpoint_defect': 0.0}
    q = state / length
    basis: list[Any] = []
    diagonal: list[float] = []
    offdiagonal: list[float] = []
    previous = np.zeros_like(q)
    residual = 0.0
    for iteration in range(steps):
        basis.append(q)
        w = h @ q - (offdiagonal[-1] * previous if offdiagonal else 0)
        alpha = float(np.vdot(q, w).real)
        diagonal.append(alpha)
        w -= alpha * q
        for _ in range(2):
            for b in basis:
                w -= np.vdot(b, w) * b
        residual = float(np.linalg.norm(w))
        if residual < 1e-13:
            break
        previous, q = q, w / residual
        if iteration + 1 < steps:
            offdiagonal.append(residual)
    t = np.diag(diagonal) + np.diag(offdiagonal, 1) + np.diag(offdiagonal, -1)
    energies, vectors = linalg.eigh(t)
    coefficients = vectors @ (np.exp(-1j * duration * energies) * vectors[0].conj())
    result = sum((c * b for c, b in zip(coefficients, basis)), np.zeros_like(state)) * length
    return result, {'dimension': len(basis), 'endpoint_defect': residual * abs(coefficients[-1]) * length}


class Snapshots:
    def __init__(self, manifest_path: Path):
        self.path = manifest_path
        self.data = json.loads(manifest_path.read_text())
        self.base = manifest_path.parent
        expected = dict(model_sha256='9b8912187ff32a90dd70fb03343a99a70d07153e6ee1bb9eeaefab414e16e7cc',
                        physical_input_sha256='c2b2bb944f96806b3d15b257cd50d0c7f31a65aa8c5bc3dd644a94aaaf697888')
        if self.data['law_pin'] != expected:
            raise ValueError('Only the explicitly corrected actual2 law/input is assigned')
        if self.data['reference_semantics'] != 'prepared_reference_t0_NONSTATIONARY':
            raise ValueError('Explicit original nonstationary t0 reference required')
        helper = self.checked(self.data['factor_loader'])
        spec = importlib.util.spec_from_file_location('interaction_pinned_factor_loader', helper)
        if spec is None or spec.loader is None:
            raise ValueError('Missing explicit factor-loader specification')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.loader = module.load
        self.cache: dict[str, Any] = {}
        ref = self.data['reference']
        reference_path = self.checked(ref)
        with np.load(reference_path) as arrays:
            self.reference = np.array(arrays[ref['key']], dtype=complex)
        self.reference_norm_squared = float(np.vdot(self.reference, self.reference).real)
        if abs(self.reference_norm_squared - 1) > 1e-9:
            raise ValueError('Unchanged normalized prepared reference required')
        for pin in self.data['provenance_pins']:
            self.checked(pin)

    def checked(self, item: dict[str, Any]) -> Path:
        path = self.base / item['file']
        if digest(path) != item['sha256']:
            raise ValueError('Changed pinned input: ' + str(path))
        return path

    def operator(self, name: str) -> Any:
        if name not in self.cache:
            item = self.data['operators'][name]
            self.cache[name] = self.loader(self.checked(item), item['sha256'])
        return self.cache[name]


class Moments:
    """Absolute-time Heisenberg moments; never use a stationary-vacuum phase."""
    def __init__(self, snapshots: Snapshots, arm: str, branch: str, spin: float,
                 source: int, receiver: int, times: list[float], steps: int, split: int = 1):
        self.snapshots, self.arm, self.branch = snapshots, arm, branch
        self.spin, self.source, self.receiver = spin, source, receiver
        self.times, self.steps, self.split = times, steps, split
        self.psi = snapshots.reference
        self.values: dict[tuple[int, ...], complex] = {}
        self.bras: dict[float, Any] = {}
        self.evolution: list[dict[str, Any]] = []

    def propagate(self, sector: str, state: Any, duration: float) -> Any:
        h = self.snapshots.operator(self.arm + '/' + sector + '/H')
        result = state
        for _ in range(self.split):
            result, receipt = lanczos(h, result, duration / self.split, self.steps)
            self.evolution.append(dict(sector=sector, duration=duration / self.split, **receipt))
        return result

    def source_operator(self, branch: str, spin: float, site: int) -> Any:
        sign = 'up' if spin > 0 else 'down'
        return self.snapshots.operator(f'source/{branch}/{sign}/site{site}')

    def odd(self, slot: int, sector: str, state: Any) -> tuple[str, Any]:
        # Slot0=S_out^dagger, slot3=S_in. S_+=c^dagger,
        # S_-(m)=epsilon_m c(-m), epsilon_up=+1, epsilon_down=-1.
        site = self.receiver if slot == 0 else self.source
        sign = 'up' if self.spin > 0 else 'down'
        charged = self.branch + '_' + sign
        if sector == 'neutral' and slot == 3:
            return charged, self.source_operator(self.branch, self.spin, site) @ state
        if sector == charged and slot == 0:
            return 'neutral', self.source_operator(self.branch, self.spin, site).H @ state
        reverse = 'antilepton' if self.branch == 'lepton' else 'lepton'
        reverse_spin = -self.spin
        reverse_sector = reverse + ('_up' if reverse_spin > 0 else '_down')
        coefficient = (-1 if reverse_spin < 0 else 1) if self.branch == 'lepton' else (-1 if self.spin < 0 else 1)
        if sector == 'neutral' and slot == 0:
            return reverse_sector, coefficient * (self.source_operator(reverse, reverse_spin, site) @ state)
        if sector == reverse_sector and slot == 3:
            return 'neutral', coefficient * (self.source_operator(reverse, reverse_spin, site).H @ state)
        raise ValueError('Unbalanced odd operator sequence')

    def moment(self, slots: tuple[int, ...]) -> complex:
        slots = tuple(sorted(slots))
        if slots in self.values:
            return self.values[slots]
        if sum(slot in (0, 3) for slot in slots) % 2:
            return 0j  # Exact total-charge superselection, not assumed field means.
        ordered = sorted(slots, key=lambda slot: self.times[slot])
        original_spin = self.spin
        odd_slots = [slot for slot in ordered if slot in (0, 3)]
        rotated = bool(odd_slots) and (self.spin if odd_slots[0] == 3 else -self.spin) < 0
        if rotated:
            if not self.snapshots.data.get('rotation_receipt'):
                raise ValueError('Negative-M reconstruction requires pinned covariance receipt')
            self.spin = -self.spin
        sector, state, previous = 'neutral', self.psi, 0.0
        for slot in ordered:
            state = self.propagate(sector, state, self.times[slot] - previous)
            if slot in (0, 3):
                sector, state = self.odd(slot, sector, state)
            else:
                field = 'Q2' if slot == 1 else 'Q5'
                state = self.snapshots.operator(f'field/{sector}/{field}') @ state
            previous = self.times[slot]
        if sector != 'neutral':
            raise ValueError('Unbalanced physical moment')
        if previous not in self.bras:
            self.bras[previous] = self.propagate('neutral', self.psi, previous)
        value = complex(np.vdot(self.bras[previous], state))
        if rotated:
            value *= (-1) ** sum(slot in (1, 2) for slot in slots)
        self.spin = original_spin
        self.values[slots] = value
        return value

    def connected(self) -> dict[str, Any]:
        full = self.moment((0, 1, 2, 3))
        g = self.moment((0, 3))
        qq = self.moment((1, 2))
        q1, q2 = self.moment((1,)), self.moment((2,))
        gq1, gq2 = self.moment((0, 1, 3)), self.moment((0, 2, 3))
        # Full graded partition cumulant. Odd singleton blocks vanish by charge.
        disconnected = g * qq + gq1 * q2 + gq2 * q1 - 2 * g * q1 * q2
        sign = 1 if self.times[0] > self.times[3] else -1
        return dict(full=sign * full, disconnected=sign * disconnected,
                    connected=sign * (full - disconnected), pair=sign * g,
                    field_pair=qq, field_mean_TE=q1, field_mean_TM=q2,
                    three_TE=sign * gq1, three_TM=sign * gq2)


def main() -> None:
    input_path, output = Path(sys.argv[1]), Path(sys.argv[2])
    config = json.loads(input_path.read_text())
    snapshots = Snapshots(ROOT / config['manifest'])
    if digest(snapshots.path) != config['manifest_sha256']:
        raise ValueError('Manifest changed')
    started = time.monotonic()
    result: dict[str, Any] = dict(kind='SAME_CAR36_NONSTATIONARY_SPATIAL_TWO_FIELD_LEG_CONNECTED',
        input_sha256=digest(input_path), manifest_sha256=digest(snapshots.path),
        original_reference_norm_squared=snapshots.reference_norm_squared,
        source_norms={}, rows=[], decision='NOT_YET_OBSERVED')
    def save(stage: str) -> None:
        result['stage'] = stage
        result['elapsed_seconds'] = time.monotonic() - started
        output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
        print(stage, flush=True)
    save('PINNED_ORIGINAL_PREPARED_INPUT_LOADED')
    for branch, site in product(('lepton', 'antilepton'), range(4)):
        source = snapshots.operator(f'source/{branch}/up/site{site}') @ snapshots.reference
        result['source_norms'][f'{branch}/up/site{site}'] = dict(
            value=float(np.vdot(source, source).real), meaning='direct raw source action')
    # Earliest discriminator is causal B-Q-Q-Bdag and interchanged TE/TM;
    # reverse fermion ordering and the remaining24 are then retained if requested.
    orders: list[tuple[int, ...]] = [tuple(int(slot) for slot in order) for order in config['orders']]
    spins: list[float] = [float(spin) for spin in config.get('spins', [.5])]
    spatial_pairs: list[tuple[int, int]] = [(int(pair[0]), int(pair[1])) for pair in config['spatial_pairs']]
    if config.get('complete_24_orders', False):
        orders += [order for order in permutations(range(4)) if order not in orders]
    for order, branch, spin, pair in product(orders, ('lepton', 'antilepton'), spins, spatial_pairs):
        times = [0.0] * 4
        for index, slot in enumerate(order):
            times[slot] = (index + config.get('time_origin_steps', 0)) * config['spacing']
        arms: dict[str, Any] = {}
        for arm in config['arms']:
            if arm == config['control'] and snapshots.data.get('factorized_control'):
                proof_record = snapshots.data['factorized_control']
                proof_path = snapshots.checked(proof_record)
                proof = json.loads(proof_path.read_text())
                if proof['source_capsule_sha256'] != '2df43f25e3b8bb06e1ac3001b6bf60e80bc5293bbaa6228bc612eae6ee80aa60':
                    raise ValueError('Factorization witness belongs to another H input')
                for sector in ('neutral', 'lepton_up', 'antilepton_up'):
                    snapshots.checked(snapshots.data['operators'][arm+'/'+sector+'/H'])
                arms[arm] = dict(
                    evaluation='EXACT_COMPLETE_TRANSVERSE_OFF_TENSOR_FACTORIZATION_NOT_NUMERICAL_PROPAGATION',
                    proof=proof_record,
                    moments={key: [0., 0.] for key in ('full', 'disconnected', 'connected', 'field_pair', 'field_mean_TE', 'field_mean_TM', 'three_TE', 'three_TM')},
                    fermion_pair='Not evaluated: multiplies the rigorously zero field cross-pair; never asserted zero',
                    resolution=dict(empirical_connected_uncertainty=0., exact_algebraic_zero=True))
                continue
            rows = []
            engines = []
            for steps, split in [(config['coarse_steps'], 1), (config['fine_steps'], 1), (config['fine_steps'], 2)]:
                engine = Moments(snapshots, arm, branch, spin, pair[0], pair[1], times, steps, split)
                rows.append(engine.connected())
                engines.append(engine)
            fine = rows[2]
            estimate = max(abs(rows[0]['connected'] - fine['connected']), abs(rows[1]['connected'] - fine['connected']))
            arms[arm] = dict(moments={key: complex_pair(value) for key, value in fine.items()},
                resolution=dict(connected_coarse_split_difference=abs(rows[0]['connected'] - fine['connected']),
                    connected_fine_split_difference=abs(rows[1]['connected'] - fine['connected']),
                    all_moments_coarse_split_max=max(abs(rows[0][key] - fine[key]) for key in fine),
                    all_moments_fine_split_max=max(abs(rows[1][key] - fine[key]) for key in fine),
                    empirical_connected_uncertainty=estimate),
                evolution=engines[2].evolution)
        full = complex(*arms['full']['moments']['connected'])
        control = complex(*arms[config['control']]['moments']['connected'])
        uncertainty = sum(arm['resolution']['empirical_connected_uncertainty'] for arm in arms.values())
        difference = abs(full - control)
        row = dict(order_earliest_to_latest=list(order), times=times, branch=branch, frame_spin_M=spin,
            source_site=pair[0], receiver_site=pair[1], arms=arms,
            spin_evidence='Ry(pi) covariance reconstruction where negative charged M is generated' if spin < 0 or times[0] < times[3] else 'direct positive-M snapshot evolution',
            control_difference=difference, empirical_uncertainty=uncertainty,
            resolved_difference=difference > config['resolution_margin'] * uncertainty and difference > config['absolute_floor'])
        result['rows'].append(row)
        result['decision'] = 'POSITIVE_FINITE_COUPLED_MECHANISM' if any(r['resolved_difference'] for r in result['rows']) else 'NO_RESOLVED_MECHANISM_ON_COMPLETED_CHANNELS'
        save('FIRST_PHYSICAL_DISCRIMINATOR' if len(result['rows']) == 1 else 'PHYSICAL_CHANNEL_COMPLETE')
        synthesis = dict(kind='FIRST_SAME_LAW_TWO_LEG_PROVISIONAL_NOT_ASYMPTOTIC_SCATTERING',
            decision=result['decision'], completed_channels=len(result['rows']),
            maximum_control_difference=max(r['control_difference'] for r in result['rows']),
            maximum_empirical_uncertainty=max(r['empirical_uncertainty'] for r in result['rows']),
            observed_branches=sorted({r['branch'] for r in result['rows']}),
            last_channel={key: row[key] for key in ('branch', 'frame_spin_M', 'source_site', 'receiver_site', 'control_difference', 'empirical_uncertainty', 'resolved_difference', 'spin_evidence')},
            exact_manifest_sha256=result['manifest_sha256'],
            remaining_unobserved_channels=len(orders) * 2 * len(config.get('spins', [.5])) * len(config['spatial_pairs']) - len(result['rows']),
            limits=['Prepared nonstationary finite CAR36 correlator, no Nature/on-shell/event-probability claim',
                    'Coarse/fine/two-substep actual complex moments are empirical resolution evidence',
                    'Negative-M use is explicitly covariance reconstruction, not separate propagation'])
        output.with_name(output.name + '_provisional_synthesis.json').write_text(json.dumps(synthesis, indent=2) + '\n')
    result['limits'] = ['Finite unamputated transition correlator, not asymptotic photons or an event probability',
        'Original nonstationary prepared state; both bra and ket propagate at absolute times',
        'Physical Q_TE,z and Q_TM,z finite Gaussian profile legs, not imported photon D',
        'All fixed-M angular intermediates retained; z legs make unequal input/output M exactly forbidden',
        'Empirical coarse/fine/split resolution estimate, not a rigorous interval enclosure',
        'Finite CAR/root-charge/radial/field/material cuts; no Nature/QED/on-shell/particle closure claim']
    result['peak_RSS_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    save('NATURAL_COMPLETION')


if __name__ == '__main__':
    main()
