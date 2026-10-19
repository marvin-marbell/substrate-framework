#!/usr/bin/env python3
"""Diagnose winding and repair hot-start Wilson sampling with fixed global moves.

The Hodge fields are frozen BEFORE production. Each additive translation and its
negative have equal state-independent proposal probabilities; ordinary Metropolis
therefore preserves the original compact Wilson measure. Recomputing a field from
the current configuration during production would NOT justify that acceptance.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from compact_phase import (CUBES, PLANES, CompactGauge, action, curl, difference,
                           forward, principal, sample)


def backward(field, axis):
    return field - np.roll(field, 1, axis=axis)


def cubes_and_current(links):
    wrapped = {plane: principal(curl(links, *plane)) for plane in PLANES}
    cubes = {}
    for mu, nu, rho in CUBES:
        cubes[mu, nu, rho] = np.rint((difference(wrapped[nu, rho], mu)
                                    - difference(wrapped[mu, rho], nu)
                                    + difference(wrapped[mu, nu], rho)) / (2 * np.pi)).astype(int)
    current = np.array([forward(((-1) ** axis) * cubes[tuple(mu for mu in range(4) if mu != axis)], axis)
                        for axis in range(4)])
    divergence = sum(backward(current[axis], axis) for axis in range(4))
    assert np.max(np.abs(divergence)) == 0
    return cubes, current


def topology(links):
    size = links.shape[1]
    _, current = cubes_and_current(links)
    adjacency = {}
    for axis in range(4):
        for coordinate_array in np.argwhere(current[axis] != 0):
            coordinate = tuple(int(x) for x in coordinate_array)
            neighbor = tuple((x + int(mu == axis)) % size for mu, x in enumerate(coordinate))
            value = int(current[(axis,) + coordinate])
            source, target = (coordinate, neighbor) if value > 0 else (neighbor, coordinate)
            step = tuple((1 if value > 0 else -1) * int(mu == axis) for mu in range(4))
            outgoing = adjacency.setdefault(source, deque())
            for _ in range(abs(value)):
                outgoing.append((target, step))
    cycles = []
    while any(adjacency.values()):
        start = next(node for node, edges in adjacency.items() if edges)
        vertex, seen, path = start, {start: 0}, []
        while True:
            assert adjacency[vertex], 'Closed current cannot end at an unmatched vertex'
            target, step = adjacency[vertex][0]
            path.append((vertex, target, step))
            if target in seen:
                cycle = path[seen[target]:]
                displacement = np.sum([arc[2] for arc in cycle], axis=0)
                assert np.all(displacement % size == 0)
                for source, destination, delta in cycle:
                    assert adjacency[source].popleft() == (destination, delta)
                cycles.append({'length': len(cycle),
                               'winding': (displacement // size).tolist(),
                               'vertices': [list(arc[0]) for arc in cycle]})
                break
            seen[target] = len(path)
            vertex = target
    return {'absolute_dual_link_count': int(np.abs(current).sum()),
            'density': float(np.abs(current).mean()),
            'nonzero_winding_cycles': sum(any(cycle['winding']) for cycle in cycles),
            'cycles': cycles,
            'scope': 'A directed simple-cycle decomposition; at intersections the decomposition need not be unique.'}


def hodge_template(links):
    """-2pi Laplacian^-1 d^dagger n from frozen integer Dirac plaquette sheets."""
    size = links.shape[1]
    source = np.zeros_like(links)
    for mu, nu in PLANES:
        raw = curl(links, mu, nu)
        integers = np.rint((raw - principal(raw)) / (2 * np.pi))
        source[mu] += backward(integers, nu)
        source[nu] -= backward(integers, mu)
    momentum = np.indices((size,) * 4)
    laplacian = (4 * np.sin(np.pi * momentum / size)**2).sum(axis=0)
    transformed = np.fft.fftn(source, axes=(1, 2, 3, 4))
    solution = np.zeros_like(transformed)
    np.divide(-2 * np.pi * transformed, laplacian, out=solution, where=laplacian != 0)
    template = np.fft.ifftn(solution, axes=(1, 2, 3, 4)).real
    template.setflags(write=False)
    return template


@dataclass(frozen=True)
class FrozenTranslations:
    template: np.ndarray

    def propose(self, model):
        template = self.template
        # Both fixed-template and uniformly translated channels are symmetric.
        if model.rng.random() < 0.5:
            shifts = tuple(int(x) for x in model.rng.integers(model.size, size=4))
            template = np.roll(template, shifts, axis=(1, 2, 3, 4))
        sign = 1 if model.rng.integers(2) else -1
        candidate = principal(model.links + sign * template)
        change = action(candidate, model.beta) - action(model.links, model.beta)
        accepted = np.log(model.rng.random()) < -change
        if accepted:
            model.links = candidate
        return accepted


class TopologyGauge(CompactGauge):
    def __post_init__(self):
        super().__post_init__()
        self._frozen = None
        self.global_attempts = self.global_accepts = 0

    def freeze_global_moves(self, snapshot):
        if self._frozen is not None:
            raise ValueError('Global template cannot be adapted after production begins')
        self._frozen = FrozenTranslations(hodge_template(snapshot))

    def sweep(self):
        super().sweep()
        if self._frozen is not None:
            self.global_attempts += 1
            self.global_accepts += int(self._frozen.propose(self))


def run(size, seed, burn, samples, thinning):
    beta = 1 / 0.3**2
    hot = TopologyGauge(size, beta, seed, 'random')
    for _ in range(burn):
        hot.sweep()
    before = topology(hot.links)
    snapshot = hot.links.copy()
    template = hodge_template(snapshot)
    candidate = principal(snapshot + template)
    candidate_topology = topology(candidate)
    proposal_action = action(candidate, beta) - action(snapshot, beta)
    # Positive control on precisely the same topology map, not a zero-instance guess.
    strong = CompactGauge(4, .35, 991, 'random')
    strong_topology = topology(strong.links)
    assert strong_topology['absolute_dual_link_count'] > 0
    hot.freeze_global_moves(snapshot)
    hot_report = sample(hot, 0, samples, thinning)
    cold = CompactGauge(size, beta, seed - 1234, 'ordered')
    cold_report = sample(cold, burn, samples, thinning)
    after = topology(hot.links)
    return {'mechanism': 'Fixed preproduction Hodge translations with symmetric +/- proposals; original Wilson action unchanged.',
            'receipt': {'python': platform.python_version(), 'numpy': np.__version__,
                        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            'inputs': {'size': size, 'beta': beta, 'hot_seed': seed, 'preparation_sweeps': burn,
                       'samples': samples, 'thinning': thinning},
            'before_global_kernel': before,
            'fixed_remove_sheet_candidate': {'action_change': proposal_action, 'topology': candidate_topology},
            'after_production': after,
            'global_attempts': hot.global_attempts, 'global_accepts': hot.global_accepts,
            'strong_topology_positive_control': {'density': strong_topology['density'],
                                                  'absolute_dual_link_count': strong_topology['absolute_dual_link_count']},
            'hot_with_frozen_moves': hot_report, 'ordered_control': cold_report,
            'limits': ['A frozen kernel can accelerate known traps, not prove equilibrium in every sector.',
                       'Blocked errors do not quantify undetected trapping; compare starts and actual winding.',
                       'Pure Euclidean gauge theory only; no thermodynamic or unquenched sector proof.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=6)
    parser.add_argument('--seed', type=int, default=11351)
    parser.add_argument('--burn', type=int, default=600)
    parser.add_argument('--samples', type=int, default=400)
    parser.add_argument('--thinning', type=int, default=2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(args.size, args.seed, args.burn, args.samples, args.thinning)
    payload = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.write_text(payload + '\n')
        print(json.dumps({'output': str(args.output), 'receipt': result['receipt'],
                          'winding_before': result['before_global_kernel']['nonzero_winding_cycles'],
                          'winding_after': result['after_production']['nonzero_winding_cycles'],
                          'global_accepts': result['global_accepts']}))
    else:
        print(payload)
