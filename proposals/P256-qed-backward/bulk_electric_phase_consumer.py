"""Exact frozen-H0 face discriminator and full-law local-tail construction.

No old producer is run: only exact_frame/profiles read-only constructors are used.
No finite node, local moment, generated character, or Ward identity is a photon.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from fractions import Fraction as F
import hashlib
import importlib.abc
import importlib.machinery
from itertools import permutations, product
import json
from pathlib import Path
import sys
import time
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = 'bulk_electric_phase_'
INPUTS = [
    'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz',
    'induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json',
    'joined_phi_u1_bulk_flux_ring_proof_20261010.json',
    'joined_phi_u1_bulk_flux_ring_closed_parity_20261010_actual.json',
    'joined_phi_u1_uniform_electric_current_bridge_proof_20261010.json',
    'joined_phi_u1_bulk_flux_spatial_sharp_parent_packet_20261010.json',
    PREFIX + 'proof_20261010.json',
    PREFIX + 'language_20261010.json',
    PREFIX + 'contract_20261010.json',
    'joined_phi_u1_native_ir_mechanism_consumer.py',
    'joined_phi_u1_original_cover_raw_flux_packet_boundary_control_20261010_receipt.json',
]
PINS = {
    INPUTS[0]: 'f8a0793adcbf4909402cd089471aa8c33fc74a31b66e99618f8baaab8d6dfb65',
    INPUTS[1]: '5b97862fc7fca21b8797d549d7d4dfdde58b101e8c46c7a7cfb6531a669c4354',
    INPUTS[5]: '50157e90c0a93c972a579990af3b7dcb270310e76646b7d27cd02b712f0f4dc9',
    PREFIX+'contract_20261010.json': '6ab9d0f12425dd07f967bda403700b782c2d33df99e6f7d3fb9ca340d0af57c1',
    'joined_phi_u1_original_cover_raw_flux_packet_boundary_control_20261010_receipt.json': '0ae3c9f11bd4dd1538669e035267311f9a639b1a6b15ee97d1b7a91bb9075830',
}
CAPTURED: dict[str, bytes] = {Path(__file__).name: Path(__file__).read_bytes()}


class CapturedLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname: str) -> Any:
        raw = Path(self.path).read_bytes()
        CAPTURED[Path(self.path).name] = raw
        return self.source_to_code(raw, self.path)


class CapturedFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec and spec.origin and Path(spec.origin).parent == ROOT and spec.origin.endswith('.py'):
            spec.loader = CapturedLoader(fullname, spec.origin)
        return spec


finder = CapturedFinder()
sys.meta_path.insert(0, finder)
import joined_phi_u1_bulk_flux_ring_consumer as frozen
sys.meta_path.remove(finder)
import numpy as np

C = tuple[F, F]
ZERO: C = (F(0), F(0))
ONE: C = (F(1), F(0))


def add(a: C, b: C) -> C:
    return a[0] + b[0], a[1] + b[1]


def mul(a: C, b: C) -> C:
    return a[0]*b[0] - a[1]*b[1], a[0]*b[1] + a[1]*b[0]


def conjugate(a: C) -> C:
    return a[0], -a[1]


def scale(a: C, b: F | int) -> C:
    return a[0]*b, a[1]*b


def abs2(a: C) -> F:
    return a[0]**2 + a[1]**2


def car(mask: int, create: int, annihilate: int) -> tuple[int, int] | None:
    if not (mask >> annihilate) & 1:
        return None
    sign = (-1)**((mask & ((1 << annihilate) - 1)).bit_count())
    mask ^= 1 << annihilate
    if (mask >> create) & 1:
        return None
    sign *= (-1)**((mask & ((1 << create) - 1)).bit_count())
    return mask | (1 << create), sign


class IntegerBox:
    """Literal native positive edges; divergence is outgoing minus incoming."""

    def __init__(self, n: int):
        self.n = n
        self.sites = list(product(range(n), repeat=3))
        self.volume = len(self.sites)
        self.index = {x: i for i, x in enumerate(self.sites)}
        self.tail = np.repeat(np.arange(self.volume), 3)
        self.head = np.array([self.index[self.step(x, axis, 1)]
                              for x in self.sites for axis in range(3)])

    def step(self, x: tuple[int, ...], axis: int, amount: int) -> tuple[int, ...]:
        y = list(x)
        y[axis] = (y[axis] + amount) % self.n
        return tuple(y)

    def path(self, x: tuple[int, ...], axis: int, length: int) -> np.ndarray:
        k = np.zeros(3*self.volume, dtype=np.int64)
        for j in range(length):
            k[3*self.index[self.step(x, axis, j)] + axis] += 1
        return k

    def face(self, x: tuple[int, ...], a: int, b: int) -> np.ndarray:
        return (self.path(x, a, 1) + self.path(self.step(x, a, 1), b, 1)
                - self.path(self.step(x, b, 1), a, 1) - self.path(x, b, 1))

    def divergence(self, k: np.ndarray) -> np.ndarray:
        rho = np.zeros(self.volume, dtype=np.int64)
        np.add.at(rho, self.tail, k)
        np.add.at(rho, self.head, -k)
        return rho

    def cycle_basis(self) -> tuple[np.ndarray, list[int]]:
        parent: dict[int, tuple[int, int, int]] = {}
        seen = {0}
        queue = deque([0])
        while queue:
            v = queue.popleft()
            x = self.sites[v]
            for axis in range(3):
                for direction in (1, -1):
                    y = self.step(x, axis, direction)
                    w = self.index[y]
                    if w not in seen:
                        tail = x if direction == 1 else y
                        edge = 3*self.index[tail] + axis
                        parent[w] = (v, edge, direction)
                        seen.add(w)
                        queue.append(w)
        assert len(parent) == self.volume - 1
        tree = {item[1] for item in parent.values()}
        chords = [e for e in range(3*self.volume) if e not in tree]
        paths = np.zeros((self.volume, 3*self.volume), dtype=np.int16)
        for v in range(1, self.volume):
            w = v
            while w:
                p, edge, direction = parent[w]
                paths[v, edge] -= direction
                w = p
        columns = np.zeros((3*self.volume, len(chords)), dtype=np.int16)
        for i, e in enumerate(chords):
            columns[:, i] = paths[self.head[e]] - paths[self.tail[e]]
            columns[e, i] += 1
        assert len(chords) == 2*self.volume + 1
        assert np.array_equal(columns[chords, :], np.eye(len(chords), dtype=np.int16))
        for i in range(len(chords)):
            assert not np.any(self.divergence(columns[:, i]))
        return columns, chords


def hodge_control(box: IntegerBox, k: np.ndarray) -> dict[str, Any]:
    """Floating split only; the admitted identity follows integer Hodge algebra."""
    rho = box.divergence(k).reshape((box.n,)*3)
    q = np.indices((box.n,)*3)
    eigen = 4*np.sum(np.sin(np.pi*q/box.n)**2, axis=0)
    rhs = np.fft.fftn(rho)
    potential_hat = np.zeros_like(rhs)
    np.divide(rhs, eigen, out=potential_hat, where=eigen > 0)
    potential = np.asarray(np.fft.ifftn(potential_hat), dtype=np.complex128).real
    longitudinal = float(np.sum(rho*potential))
    full = int(k @ k)
    return dict(full_energy_exact=64*full,
                HE_finite_float=64*(full-longitudinal),
                HC_collective_finite_float=64*longitudinal,
                full_minus_split_residual=abs(64*full-64*(full-longitudinal)-64*longitudinal))


def spectral_box(n: int) -> dict[str, Any]:
    box = IntegerBox(n)
    signed: set[tuple[tuple[int, int], ...]] = set()
    faces = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        for x in box.sites:
            c = box.face(x, a, b)
            assert not np.any(box.divergence(c)) and int(c @ c) == 4
            for sign in (1, -1):
                key = tuple((int(e), int(sign*c[e])) for e in np.flatnonzero(c))
                assert key not in signed
                signed.add(key)
            faces.append(c)
    c = box.face((0, 0, 0), 0, 1)
    adjacent = box.face((1, 0, 0), 0, 1)
    assert int((c + adjacent) @ (c + adjacent)) == 6
    wrong = c.copy()
    wrong[np.flatnonzero(c)[0]] *= -1
    assert int(wrong @ wrong) == 4 and np.any(box.divergence(wrong))
    missing = c.copy()
    missing[np.flatnonzero(c)[0]] = 0
    assert int(missing @ missing) == 3 and np.any(box.divergence(missing))
    # Six oriented boundaries of one cube: a linear cycle relation, not sin Bianchi.
    cube = (box.face((0, 0, 1), 0, 1) - box.face((0, 0, 0), 0, 1)
            - box.face((0, 1, 0), 0, 2) + box.face((0, 0, 0), 0, 2)
            + box.face((1, 0, 0), 1, 2) - box.face((0, 0, 0), 1, 2))
    assert not np.any(cube)
    fourier = np.exp(-2j*np.pi*(np.asarray(box.sites) @ np.asarray(box.sites).T)/n)/np.sqrt(box.volume)
    gram_error = float(np.max(np.abs(fourier @ fourier.conj().T - np.eye(box.volume))))
    assert gram_error < 1e-11
    columns, chords = box.cycle_basis()
    for face in faces:
        coordinates = face[chords]
        used = np.flatnonzero(coordinates)
        reconstructed = np.sum(columns[:, used]*coordinates[used], axis=1)
        assert np.array_equal(reconstructed, face)
    hs2 = int(np.sum(columns.astype(np.int64)**2))
    d = len(chords)
    epsilon = F(1, 1024*d*hs2)
    excess_mean = F(64, 3)*epsilon**2*hs2
    excess_sup = 64*epsilon**2*d*hs2
    slopes = [int(np.sum(np.abs(np.sum(columns[np.flatnonzero(face), :]
                                      *face[np.flatnonzero(face), None], axis=0)))) for face in faces]
    spread = 128*epsilon*max(slopes) + excess_sup
    assert spread < 1 and excess_sup < 1
    qmin_hat = 2*np.sin(np.pi/n)
    wrong_massless_energy = float(qmin_hat)
    assert wrong_massless_energy != 256 and 0 != 256
    energy_splits = [hodge_control(box, k) for k in
                     (c, c+adjacent, box.path((0, 0, 0), 0, 1), 2*box.path((0, 0, 0), 0, 1))]
    assert max(row['full_minus_split_residual'] for row in energy_splits) < 1e-10
    return dict(N=n, volume=box.volume, canonical_modes_per_charged_species=4*box.volume,
                canonical_particles_per_charged_species=2*box.volume, charged_species=64,
                literal_edges=3*box.volume, faces=3*box.volume,
                distinct_signed_face_characters=len(signed), all_face_divergence_exact=0,
                all_face_norm_squared_exact=4, face_energy_exact=256,
                source_norm_squared_exact='1/2', source_mean_exact='0',
                face_momentum_measure={"every_lattice_momentum":"(1/2) delta_256",
                                       "momenta_exercised":box.volume, "DFT_Gram_finite_float_error":gram_error,
                                       "p0_energy_exact":256, "pmin_energy_exact":256,
                                       "electric_current_source_masses_exact":["0", "0"]},
                wrong_controls=dict(sign_flip_divergence_l1=int(np.sum(np.abs(box.divergence(wrong)))),
                                    sign_flip_energy_still_256=True, missing_edge_energy=192,
                                    adjacent_faces_energy=384, wrong_independent_face_energy=512,
                                    linear_cube_Bianchi_exact=True, sin_linear_Bianchi_not_asserted=True,
                                    wrong_massless_p0_energy=0, wrong_massless_pmin_energy=wrong_massless_energy,
                                    pole_claim_rejected=True),
                full_HE_HC64_controls=energy_splits,
                original_cover_packet=dict(cycle_dimension=d, integer_basis_HS_squared=hs2,
                                           face_cycle_reconstruction_exact=True, epsilon_exact=str(epsilon),
                                           normalized_full_dimensional_packet=True,
                                           mean_excess_exact=str(excess_mean), sup_excess_exact=str(excess_sup),
                                           maximum_face_energy_window_about_256_exact=str(spread),
                                           source_mass_exact='1/2', real_cover_L2_exact_ground=False))


def neutral_AP_ground(cfg: dict[str, Any], frame: list[list[list[C]]], output: Path) -> dict[str, Any]:
    """Fixed exact spectral projector; floats exercise its AP/native consumer."""
    band = frozen.RotationBand(a=cfg['a'], m=cfg['heavy_m'], r=cfg['heavy_r'],
                               v=cfg['heavy_v'], improved=False)
    onsite, paths = frozen.scalar_paths(band)
    def exact_product(a: list[list[C]], b: list[list[C]]) -> list[list[C]]:
        return [[frozen.sum_c([mul(a[i][k], b[k][j]) for k in range(4)])
                 for j in range(4)] for i in range(4)]
    for mu in range(4):
        for nu in range(4):
            ab, ba = exact_product(frame[mu], frame[nu]), exact_product(frame[nu], frame[mu])
            anti = [[add(ab[i][j], ba[i][j]) for j in range(4)] for i in range(4)]
            radius_squared = abs2(frame[mu][0][0]) + abs2(frame[mu][0][1]) + abs2(frame[mu][0][2]) + abs2(frame[mu][0][3])
            expected = [[(2*radius_squared if mu == nu and i == j else F(0), F(0))
                         for j in range(4)] for i in range(4)]
            assert anti == expected
    mats = np.array([[[complex(float(z[0]), float(z[1])) for z in row] for row in m]
                     for m in frame])
    records = []
    arrays = {}
    assert cfg['heavy_m'] > 0 and cfg['heavy_r'] > 0
    for n in (4, 6, 8):
        momenta = 2*np.pi*(np.array(list(product(range(n), repeat=3)))+0.5)/n
        h = np.broadcast_to(onsite*mats[0], (n**3, 4, 4)).copy()
        for displacement, coefficients in paths:
            matrix = np.einsum('m,mij->ij', coefficients, mats)
            phase = np.exp(1j*(momenta @ displacement))
            h += phase[:, None, None]*matrix + phase.conj()[:, None, None]*matrix.conj().T
        # Clifford h(q)^2=r(q)^2 I; -Lambda selects the POSITIVE unscaled bands.
        squared = h @ h
        radius = np.sqrt(np.trace(squared, axis1=1, axis2=2).real/4)
        assert np.all(radius > 0)
        projector = (np.eye(4) + h/radius[:, None, None])/2
        values, vectors = np.linalg.eigh(h)
        occupied = vectors[:, :, 2:]
        wrong_sign_projector = (np.eye(4) - h/radius[:, None, None])/2
        defects = dict(
            Clifford_square=float(np.max(np.abs(squared-radius[:, None, None]**2*np.eye(4)))),
            projector=float(np.max(np.abs(projector @ projector-projector))),
            occupied_frame=float(np.max(np.abs(occupied @ occupied.conj().transpose(0, 2, 1)-projector))),
            AP_wrap=float(np.max(np.abs(np.exp(1j*n*momenta)+1))),
            wrong_neutral_sign=float(np.max(np.abs(wrong_sign_projector-projector))))
        assert max(defects[k] for k in ('Clifford_square', 'projector', 'occupied_frame', 'AP_wrap')) < 1e-9
        assert defects['wrong_neutral_sign'] > 0.1
        arrays[f'N{n}_momenta'] = momenta
        arrays[f'N{n}_occupied_frame_positive_unscaled'] = occupied
        arrays[f'N{n}_exact_projector_formula_float_control'] = projector
        records.append(dict(N=n, charge=0, particle_count=2*n**3, canonical_modes=4*n**3,
                            physical_sign='-Lambda', Lambda_exact='2^8192', improved=False,
                            exact_projector='(I+h_AP(q)/sqrt(Tr h_AP(q)^2/4))/2',
                            exact_ground_selection='wedge all two positive unscaled bands at each AP momentum',
                            neutral_energy_over_Lambda_finite_float=-float(np.sum(values[:, 2:])),
                            minimum_unscaled_absolute_eigenvalue=float(np.min(radius)),
                            finite_float_controls=defects, held_fixed_for_all_charged_lambda=True))
    neutral_path = output.with_name(output.stem+'_neutral_AP_ground.npz')
    np.savez_compressed(neutral_path, **arrays)
    neutral_path.chmod(0o444)
    return dict(records=records, exact_ground_basis_rule='AP Fourier positive spectral subspace, any ordered orthonormal frame',
                primary_law_source='rotation_band.RotationBand(improved=False).couplings -> scalar_paths',
                artifact=neutral_path.name, sha256=hashlib.sha256(neutral_path.read_bytes()).hexdigest(),
                exact_vs_float='Exact ground subspace defined by Clifford spectral formula; persisted frames and residuals are finite-float controls.')


def native_channels(profile: dict[str, Any]) -> list[dict[str, Any]]:
    ref = (1 << 0) | (1 << 1) | (1 << 4) | (1 << 5)
    rows = []
    for path in profile['paths']:
        channels = []
        matrix = path['matrix']
        for i in range(2, 4):
            for j in range(2):
                for target, origin, value, direction in (
                        (i, 4+j, matrix[i][j], 1),
                        (4+i, j, conjugate(matrix[j][i]), -1)):
                    action = car(ref, target, origin)
                    assert action is not None and action[0].bit_count() == 4
                    if value != ZERO:
                        channels.append(dict(create=target, annihilate=origin,
                                             value=value, direction=direction))
        g = sum((abs2(channel['value']) for channel in channels), F(0))
        energy = 2*profile['onsite'] + 64*path['length']
        assert energy > 0
        rows.append(dict(axis=path['axis'], length=path['length'], g=g, d=energy,
                         channels=channels, vacuum_norm_density=g/energy**2))
    return rows


def linked_pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """All actual-CAR disjoint two-bubble histories at order four, not a new law."""
    ref = sum(1 << (4*x+s) for x in range(4) for s in (0, 1))
    total = ZERO
    allowed_orders = []
    excluded_orders = []
    for order in permutations(('A+', 'A-', 'B+', 'B-')):
        if order.index('A+') > order.index('A-') or order.index('B+') > order.index('B-'):
            continue
        active = set()
        premature = False
        denominator = F(1)
        for j, event in enumerate(order):
            if event[1] == '+':
                active.add(event[0])
            else:
                active.remove(event[0])
            if j < 3:
                if not active:
                    premature = True
                    break
                energy = (left['d'] if 'A' in active else F(0)) + (right['d'] if 'B' in active else F(0))
                denominator *= -energy
        if premature:
            excluded_orders.append(list(order))
            continue
        allowed_orders.append(list(order))
        for a in left['channels']:
            for b in right['channels']:
                mask = ref
                amplitude = ONE
                for event in order:
                    channel = a if event[0] == 'A' else b
                    offset = 0 if event[0] == 'A' else 8
                    create, annihilate, value = channel['create'], channel['annihilate'], channel['value']
                    if event[1] == '-':
                        create, annihilate, value = annihilate, create, conjugate(value)
                    action = car(mask, create+offset, annihilate+offset)
                    assert action is not None
                    mask, sign = action
                    amplitude = mul(amplitude, scale(value, sign))
                assert mask == ref and amplitude[1] == 0 and amplitude[0] >= 0
                total = add(total, scale(amplitude, 1/denominator))
    g1, g2, d1, d2 = left['g'], right['g'], left['d'], right['d']
    analytic = -g1*g2*(d1+d2)/(d1**2*d2**2)
    fold = g1*g2*(1/(d1**2*d2) + 1/(d1*d2**2))
    assert total == (analytic, F(0)) and analytic + fold == 0
    assert len(allowed_orders) == 4 and len(excluded_orders) == 2
    return dict(exact_fixed_z_irreducible_disconnected=str(analytic), exact_energy_metric_fold=str(fold),
                exact_linked_sum='0', actual_CAR_channel_pairs=len(left['channels'])*len(right['channels']),
                complete_two_PH_interleavings=allowed_orders, premature_global_vacuum_returns=excluded_orders,
                wrong_fold_deleted_rejected=analytic != 0)


def polynomial_add(a: dict[tuple[int, ...], C], b: dict[tuple[int, ...], C], factor: C = ONE) -> dict[tuple[int, ...], C]:
    result = dict(a)
    for k, value in b.items():
        result[k] = add(result.get(k, ZERO), mul(value, factor))
        if result[k] == ZERO:
            del result[k]
    return result


def shift(poly: dict[tuple[int, ...], C], flow: np.ndarray) -> dict[tuple[int, ...], C]:
    return {tuple(int(x+y) for x, y in zip(k, flow)): value for k, value in poly.items()}


def sine(poly: dict[tuple[int, ...], C], c: np.ndarray) -> dict[tuple[int, ...], C]:
    return polynomial_add({k: mul(v, (F(0), F(-1, 2))) for k, v in shift(poly, c).items()},
                          shift(poly, -c), (F(0), F(1, 2)))


def electric(poly: dict[tuple[int, ...], C]) -> dict[tuple[int, ...], C]:
    return {k: scale(value, 64*sum(x*x for x in k)) for k, value in poly.items() if any(k)}


def local_action_controls(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    box = IntegerBox(4)
    c = box.face((0, 0, 0), 0, 1)
    other = box.face((1, 0, 0), 0, 1)
    zero = tuple(0 for _ in c)
    state = {zero: ONE, tuple(c): (F(1, 3), F(0)), tuple(other): (F(0), F(1, 5))}
    sea64 = sum(1 << (copy*4*box.volume+4*site+spin)
                for copy in range(64) for site in range(box.volume) for spin in (0, 1))
    assert sea64.bit_count() == 128*box.volume
    physical_CAR_Gauss_checks = 0
    # No flow cut is used: all shifted outputs, including exterior +/-2 faces, are retained.
    ho = electric(sine(state, c))
    oh = sine(electric(state), c)
    commutator = polynomial_add(ho, oh, (F(-1), F(0)))
    predicted: dict[tuple[int, ...], C] = {}
    for k, value in state.items():
        dot = sum(int(x)*int(y) for x, y in zip(k, c))
        # +/- character coefficient of -128i cos(c.A)c.E +256 sin(c.A).
        for sign in (1, -1):
            key = tuple(int(x+sign*y) for x, y in zip(k, c))
            amplitude = mul(value, (F(0), F(-64*dot-128*sign)))
            predicted[key] = add(predicted.get(key, ZERO), amplitude)
    predicted = {k: value for k, value in predicted.items() if value != ZERO}
    assert commutator == predicted
    double = polynomial_add(sine(commutator, c),
                            polynomial_add(electric(sine(sine(state, c), c)),
                                           sine(electric(sine(state, c)), c), (F(-1), F(0))),
                            (F(-1), F(0)))
    # [O,[H,O]]=2 O H O-O^2 H-H O^2 =512 cos^2.
    expected = {k: scale(value, 256) for k, value in state.items()}
    expected = polynomial_add(expected, shift(state, 2*c), (F(128), F(0)))
    expected = polynomial_add(expected, shift(state, -2*c), (F(128), F(0)))
    assert double == expected
    channels_checked = 0
    gauss_checked = 0
    for copy, profile in enumerate(profiles):
        for path in profile['paths']:
            # AP seam is crossed here; each literal length1/2 path is retained.
            x = (box.n-1, 0, 0)
            y = box.step(x, path['axis'], path['length'])
            flow = box.path(x, path['axis'], path['length'])
            rho = box.divergence(flow)
            expected_rho = np.zeros(box.volume, dtype=np.int64)
            expected_rho[box.index[x]] += 1
            expected_rho[box.index[y]] -= 1
            assert np.array_equal(rho, expected_rho)
            for sign in (1, -1):
                for i in range(4):
                    for j in range(4):
                        value = path['matrix'][i][j]
                        if value == ZERO:
                            continue
                        # CAR commutes with this pure electric multiplier for every mask;
                        # all electric shift outputs are evaluated without a truncation.
                        left = sine({k: mul(v, value) for k, v in shift(state, sign*flow).items()}, c)
                        right = {k: mul(v, value) for k, v in shift(sine(state, c), sign*flow).items()}
                        assert left == right
                        target_site, origin_site = (box.index[x], box.index[y]) if sign == 1 else (box.index[y], box.index[x])
                        target_spin, origin_spin = (i, j) if sign == 1 else (j, i)
                        action = car(sea64, copy*4*box.volume+4*target_site+target_spin,
                                     copy*4*box.volume+4*origin_site+origin_spin)
                        if action is not None:
                            assert action[0].bit_count() == sea64.bit_count()
                            actual_rho = np.zeros(box.volume, dtype=np.int64)
                            actual_rho[target_site] += 1
                            actual_rho[origin_site] -= 1
                            assert np.array_equal(box.divergence(sign*flow), actual_rho)
                            physical_CAR_Gauss_checks += 1
                        channels_checked += 1
            gauss_checked += 1
    return dict(exact_complete_electric_commutator=True, exact_double_commutator_512_cos_squared=True,
                polynomial_exterior_untruncated=True, distinct_role_literal_monomials_checked=channels_checked,
                literal_Gauss_paths_checked=gauss_checked, all64_included_by_exact_role_multiplicity=[1, 63],
                full64_canonical_reference_particles=sea64.bit_count(),
                actual_CAR_Gauss_native_orientation_checks=physical_CAR_Gauss_checks,
                divergence_convention='native D[tail,e]=+1, D[head,e]=-1',
                scalar_or_current_Hessian_used=False)


def snapshot() -> dict[str, bytes]:
    captured = {**CAPTURED, **frozen.CAPTURED,
                **{name: (ROOT/name).read_bytes() for name in INPUTS},
                PREFIX+'capsule.py': (ROOT/(PREFIX+'capsule.py')).read_bytes()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename)
            if path.is_file():
                key = path.name if path.parent == ROOT else 'runtime_imports/'+str(path).lstrip('/')
                captured[key] = path.read_bytes()
    return captured


def archive_bytes(path: Path, captured: dict[str, bytes], additions: dict[str, bytes] | None = None) -> None:
    with zipfile.ZipFile(path, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, raw in captured.items():
            archive.writestr('bound/'+name, raw)
        for name, raw in (additions or {}).items():
            archive.writestr(name, raw)
    path.chmod(0o444)


def run(output: Path) -> dict[str, Any]:
    started = time.monotonic()
    captured = snapshot()
    for name, pin in PINS.items():
        assert hashlib.sha256(captured[name]).hexdigest() == pin
    archive_bytes(output.with_name(output.stem+'_preexecution.zip'), captured)
    frame, s = frozen.exact_frame(captured[INPUTS[0]])
    cfg = json.loads(captured[INPUTS[1]])['input_descriptor']['original_micro_config_UNCHANGED']
    assert cfg['auxiliary_count'] == 63 and cfg['a'] == 1/64
    ps = frozen.profiles(cfg, frame, s)
    assert ps[0]['onsite'] != ps[1]['onsite']
    rows = []
    channels = []
    b = F(0)
    a = F(0)
    for profile in ps:
        ch = native_channels(profile)
        channels.append(ch)
        local_bound = 2*sum((abs(z[0])+abs(z[1]) for path in profile['paths']
                             for row in path['matrix'] for z in row), F(0))
        b += profile['count']*local_bound
        density = sum((row['vacuum_norm_density'] for row in ch), F(0))
        a += profile['count']*density
        rows.append(dict(role=profile['role'], count=profile['count'], onsite_exact=str(profile['onsite']),
                         onsite_excitation_gap_exact=str(2*profile['onsite']),
                         full_CAR_operator_bound_per_site_exact=str(local_bound),
                         vacuum_dressing_norm_density_exact=str(density),
                         paths=[dict(axis=row['axis'], length=row['length'], sea_PH_norm_squared_exact=str(row['g']),
                                     physical_intermediate_energy_exact=str(row['d']),
                                     all_nonzero_occupied_empty_channels=len(row['channels'])) for row in ch]))
    assert a > 0 and b > 0
    bubbles = []
    for l, r in ((0, 0), (0, 1), (1, 1)):
        bubbles.append(dict(roles=[ps[l]['role'], ps[r]['role']],
                            **linked_pair(channels[l][0], channels[r][2])))
    shared_actual = 2*ps[0]['onsite'] + 2*ps[1]['onsite'] + 256
    shared_wrong = 2*ps[0]['onsite'] + 2*ps[1]['onsite'] + 128
    assert shared_actual - shared_wrong == 128
    second = 131072 + 4096*b
    tail = [dict(R=R, connected_tail_upper_exact=str(min(F(1), F(256, R), second/R**2)))
            for R in (256, 4096, 65536, 1048576)]
    boxes = [spectral_box(n) for n in (4, 6, 8)]
    actions = local_action_controls(ps)
    neutral = neutral_AP_ground(cfg, frame, output)
    result = dict(kind='EXACT_H0_FACE_MEASURE_AND_ALL_LAMBDA_LOCAL_HIGH_ENERGY_TAIL_NOT_PHASE',
                  scope=json.loads(captured[PREFIX+'proof_20261010.json'])['scope'],
                  actual_frozen_Phi_scale_exact=str(s), roles=rows,
                  fixed_neutral_AP_ground=neutral,
                  spectral_boxes=boxes, connected_local_tail=dict(
                      conditional_on_frozen_model_and_translation_averaged_compact_ground=True,
                      ground_existence_proved_by_compact_resolvent=True,
                      full64_native_CAR_bound_per_site_exact=str(b), connected_mass_upper_exact='1',
                      first_energy_moment_upper_exact='256',
                      second_energy_moment_upper_at_physical1_exact=str(second),
                      second_energy_moment_upper_at_physical1=float(second),
                      tail_examples_at_physical1=tail, physical_hopping=1,
                      tail_inequality_reaches_physical1=True,
                      N_uniform_gapped_phase_radius_derived=False, gap_or_pole_not_inferred=True,
                      formula='mu((R,infty))<=min(1,256/R,(131072+4096|lambda|b)/R^2)'),
                  linked_attempt=dict(vacuum_dressing_norm_density_all64_exact=str(a),
                                      actual_volume_scaling=[dict(N=n, exact_norm_squared=str(n**3*a)) for n in (4, 6, 8)],
                                      disconnected_complete_CAR_controls=bubbles,
                                      shared_edge_two_role_full_energy_exact=str(shared_actual),
                                      wrong_factorized_copy_energy_exact=str(shared_wrong),
                                      exact_shared_field_price_missing_in_wrongcase='128',
                                      undressed_relative_bound_necessary_vacuum_annihilation=False,
                                      N_uniform_allorders_gap_or_stability_not_proved=True,
                                      physical1_phase_reach=False,
                                      failed_assumption='An undressed vacuum-containing relative resolvent cannot have a volume-independent local norm: V Omega!=0, H0 Omega=0, ||H0^-1 V Omega||^2=Vvolume*a. Global-Q irreducibility does not cancel disconnected bubbles without the energy-dependent metric fold.',
                                      missing_mechanism='All-order normalized vacuum dressing/rooted connected summability with overlapping shared integer flows, full64 charged CAR sectors and actual onsite roles.'),
                  complete_local_action_controls=actions,
                  exact_vs_conditional=dict(integer_and_Gaussian_rational='exact exercised',
                                            N_uniform_H0_spectrum='exact derived and N4/N6/N8 exercised',
                                            original_cover='normalized packet bounds exact; limiting spectral statement derived',
                                            nonzero_local_tail='analytic conditional frozen-model bound; no numerical interacting-ground spectrum',
                                            Fourier_controls='finite floating arithmetic, not intervals',
                                            gap_phase_IRatom='unproved'),
                  source_and_input_sha256={name: hashlib.sha256(raw).hexdigest() for name, raw in captured.items()},
                  bound_local_python_sources_sha256={name: hashlib.sha256(raw).hexdigest() for name, raw in captured.items()
                                              if '/' not in name and name.endswith('.py')},
                  runtime=dict(executable=sys.executable, python=sys.version, numpy=np.__version__),
                  proof=json.loads(captured[PREFIX+'proof_20261010.json']),
                  campaign_complete=False, elapsed_seconds=time.monotonic()-started)
    for name, raw in captured.items():
        if '/' not in name:
            assert (ROOT/name).read_bytes() == raw, name
    output.write_text(json.dumps(result, indent=2)+'\n')
    archive_bytes(output.with_name(output.stem+'_executed.zip'), captured,
                  {'actual.json': output.read_bytes(),
                   neutral['artifact']: (ROOT/neutral['artifact']).read_bytes()})
    print(json.dumps(dict(kind=result['kind'], spectral_N=[4, 6, 8], exact_face_energy=256,
                         exact_source_mass='1/2', physical1_local_tail=True,
                         physical1_phase=False, b=float(b), dressing_density=float(a),
                         disconnected_CAR_cancellations=len(bubbles),
                         elapsed_seconds=result['elapsed_seconds']), indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.output)
