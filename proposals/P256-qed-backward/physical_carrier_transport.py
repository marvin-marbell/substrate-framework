#!/usr/bin/env python3
"""Positive-clock physical out/back contour, primitive Pauli and whole CAR sectors."""
from __future__ import annotations
import argparse
import ast
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from fractions import Fraction as F
import time
from typing import Any, cast
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
from scipy.integrate import solve_ivp
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply, LinearOperator
from bosonic_statistics import primitive_pauli_words
from improved_phi_band import ImprovedPhiLaw, selected_band, C, UE, NSTAR, physical_h
from improved_physical_compton import decode, encode, ImprovedBloch
from sea_selection import matrix_unit_basis, SELECTED, reframe
from unrestricted_field_minimum import BARE, local_potential
from original_total_angular_transfer import primitive_flow, scalar_orbital
from rotation_band import RotationBand
from noncompact_screened_gauss_dressing import frame_chart, centre_scalar
import noncompact_legal_volume_bloch as legal
import noncompact_harmonic_scattering_fiber as fibre

ROOT = Path(__file__).resolve().parent
SITES = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
EDGE_PATH = ((0, 1, 0), (1, 2, 1))


def require(value: bool | np.bool_, label: str) -> None:
    if not value:
        raise AssertionError(label)


def check(value: float, label: str, bound: float = 2e-8) -> float:
    require(math.isfinite(value) and value <= bound, f'{label}: {value} > {bound}')
    return float(value)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def multiply(left: tuple[complex, int, int], right: tuple[complex, int, int]):
    p, x, z = left
    q, y, w = right
    return p*q*(-1)**((z & y).bit_count()), x ^ y, z ^ w


def monomial(word: tuple[str, ...], sign: complex = 1):
    x = z = 0
    for e, letter in enumerate(word):
        if letter in ('X', 'Y'):
            x |= 1 << e
        if letter in ('Z', 'Y'):
            z |= 1 << e
        if letter == 'Y':
            sign *= 1j
    return complex(sign), x, z


class PrimitiveSector:
    """Root edge Pauli words, restricted only by conserved TOTAL particle number.

    Computational edge states, not a copied occupation-sign Hamiltonian. All
    loop sectors and all reference occupations 0..4 survive with finite costs.
    """
    def __init__(self, number: int):
        self.edges = tuple(sorted((*((i, i+1) for i in range(15)), (0, 15))))
        incident = tuple(tuple(e for e, edge in enumerate(self.edges) if v in edge)
                         for v in range(16))
        bw, aw = primitive_pauli_words(self.edges, incident)
        self.bwords = [monomial(word, sign) for sign, word in bw]
        self.awords = {edge: monomial(word) for edge, word in aw.items()}
        allstates = np.arange(1 << len(self.edges), dtype=np.int64)
        allocc = np.array([[(1-int(p.real)*(-1)**((int(s)&z).bit_count()))//2
                            for p, _, z in self.bwords] for s in allstates], dtype=np.int8)
        self.states = allstates[allocc.sum(axis=1) == number]
        self.occupation = allocc[allocc.sum(axis=1) == number]
        self.lookup = {int(s): k for k, s in enumerate(self.states)}
        self.dimension = len(self.states)
        self.number = number
        self.masks = np.sum(self.occupation.astype(np.int64)*(1 << np.arange(16)), axis=1)
        self.cycle_word = (1+0j, 0, 0)
        for source, target in zip(range(16), (*range(1, 16), 0)):
            self.cycle_word = multiply(self.cycle_word, self.a(source, target))
        self.cycle_word = (self.cycle_word[0]*1j**16, *self.cycle_word[1:])
        self.cycle = self.operator(self.cycle_word)
        self.identity = sparse.eye(self.dimension, format='csr')
        self.defect = self.operator((1+0j, 0, 1 << self.edges.index((0, 15))))
        self.nref = self.occupation[:, 12:].sum(axis=1)
        self.density = self.occupation[:, :12].reshape(-1, 3, 4).sum(axis=2)-2
        self._bilinears: dict[tuple[int, int], sparse.csr_matrix] = {}

    def a(self, source: int, target: int):
        phase, x, z = self.awords[tuple(sorted((source, target)))]
        return phase*(1 if source < target else -1), x, z

    def operator(self, word):
        p, x, z = word
        rows = [self.lookup[int(s)^x] for s in self.states]
        values = [p*(-1)**((int(s)&z).bit_count()) for s in self.states]
        return sparse.csr_matrix((values, (rows, np.arange(self.dimension))),
                                 shape=(self.dimension, self.dimension))

    def bilinear(self, target: int, source: int):
        key = target, source
        if key in self._bilinears:
            return self._bilinears[key]
        if source == target:
            result = sparse.diags(self.occupation[:, source], format='csr', dtype=float)
        else:
            step = 1 if target > source else -1
            path = tuple(range(source, target+step, step))
            word = (1+0j, 0, 0)
            for a, b in zip(path, path[1:]):
                word = multiply(word, self.a(a, b))
            p, x, z = word
            p *= -1j*1j**(len(path)-2)
            columns = np.flatnonzero((self.occupation[:, source] == 1) &
                                     (self.occupation[:, target] == 0))
            rows = [self.lookup[int(self.states[k])^x] for k in columns]
            values = [p*(-1)**((int(self.states[k])&z).bit_count()) for k in columns]
            result = sparse.csr_matrix((values, (rows, columns)),
                                       shape=(self.dimension, self.dimension))
        self._bilinears[key] = cast(sparse.csr_matrix, result)
        return self._bilinears[key]

    def lift(self, matrix: np.ndarray):
        result = sparse.csr_matrix((self.dimension, self.dimension), dtype=complex)
        for target, source in zip(*np.nonzero(abs(matrix) > 1e-15)):
            result += matrix[target, source]*self.bilinear(int(target), int(source))
        return result

    def code_vector(self, mask: int):
        """Cycle+ eigenvector, with Root's tree comparator phase alignment."""
        keys = np.flatnonzero(self.masks == mask)
        require(len(keys) == 2, 'one auxiliary qubit in every allowed whole-CAR occupation')
        v = np.zeros(self.dimension, complex)
        v[keys[0]] = 1
        v = (v+self.cycle@v)/math.sqrt(2)
        return v

    def neutral(self, orbital: np.ndarray, branch: int):
        """Actual six-particle Slater coefficients and wedge4/wedge0 reference.

        Alignment uses the primitive transfer amplitudes along the Root tree;
        CAR determinants enter only as the independently observable comparator.
        """
        reference = sum(1 << i for i in range(12, 16)) if branch == 1 else 0
        masks = [sum(1 << i for i in row) | reference
                 for row in itertools.combinations(range(12), 6)]
        vectors = {mask: self.code_vector(mask) for mask in masks}
        phases: dict[int, complex] = {masks[0]: 1+0j}
        queue = [masks[0]]
        while queue:
            mask = queue.pop()
            for source in range(11):
                for s, t in ((source, source+1), (source+1, source)):
                    if not mask >> s & 1 or mask >> t & 1:
                        continue
                    target = mask ^ (1 << s) ^ (1 << t)
                    if target in phases:
                        continue
                    amplitude = np.vdot(vectors[target], self.bilinear(t, s)@vectors[mask])
                    rest = mask ^ (1 << s)
                    sign = (-1)**((mask & ((1 << s)-1)).bit_count()+
                                   (rest & ((1 << t)-1)).bit_count())
                    require(abs(abs(amplitude)-1) < 1e-10, 'primitive tree transfer alignment')
                    phases[target] = amplitude*phases[mask]/sign
                    queue.append(target)
        require(len(phases) == len(masks), 'whole six-of-twelve neutral comparator connected')
        result = np.zeros(self.dimension, complex)
        for mask in masks:
            occupied = [i for i in range(12) if mask >> i & 1]
            result += np.linalg.det(orbital[occupied])*phases[mask]*vectors[mask]
        check(abs(np.linalg.norm(result)-1), 'primitive neutral normalization')
        return result

    def source(self, coefficients: np.ndarray, branch: int):
        result = sparse.csr_matrix((self.dimension, self.dimension), dtype=complex)
        for target in range(12):
            for ref in range(4):
                coefficient = coefficients[target, ref] if branch == 1 else coefficients[target, ref].conjugate()
                result += coefficient*(self.bilinear(target, 12+ref) if branch == 1
                                       else self.bilinear(12+ref, target))
        neutral = 4 if branch == 1 else 0
        charged = 3 if branch == 1 else 1
        return sparse.diags((self.nref == charged).astype(float))@result@sparse.diags((self.nref == neutral).astype(float))


def native_onebody(law: ImprovedPhiLaw, fields: np.ndarray, links: np.ndarray,
                   potential: np.ndarray, direction: np.ndarray | None = None,
                   second: np.ndarray | None = None):
    chosen = [law.index[tuple(site)] for site in SITES]
    modes = np.concatenate([np.arange(4*x, 4*x+4) for x in chosen])
    full = law.onebody(fields, links, potential, direction, second)
    return full[np.ix_(modes, modes)]


def observations(state: np.ndarray, sector: PrimitiveSector, nodes: int):
    weights = np.sum(abs(state.reshape(nodes, sector.dimension))**2, axis=0)
    return weights@sector.density


def expectation(state: np.ndarray, operator):
    return float(np.vdot(state, operator@state).real)


def angular_chart(law, fields, links, potential, velocity, cutoff, poses):
    """Actual primitive fullPhi/SU4/sin(curl A) integral curve, not spin copies."""
    shapes = (fields.shape, links.shape, potential.shape)
    lengths = [int(np.prod(shape)) for shape in shapes]
    def unpack(values):
        chunks = np.split(values, np.cumsum(lengths)[:-1])
        return tuple(v.reshape(shape) if k < 2 else v.real.reshape(shape)
                     for k, (v, shape) in enumerate(zip(chunks, shapes)))
    def rhs(theta, values):
        df, du, da = unpack(values)
        return np.concatenate([v.ravel() for v in primitive_flow(
            law, df, du, da, velocity, cutoff)])
    initial = np.concatenate([v.ravel() for v in (fields, links, potential)])
    solution = solve_ivp(rhs, (0., max(poses)), initial, t_eval=sorted(poses),
                         rtol=2e-10, atol=2e-12, max_step=.004)
    require(solution.success, 'physical Phi/SU4/EM chart natural integration')
    ordered = {float(theta): unpack(row) for theta, row in zip(solution.t, solution.y.T)}
    return [ordered[float(theta)] for theta in poses]


def native_source_coefficient(law, fields, links, potential, profile, beta):
    """Same Phi source, with REAL local SU4/U1 transport to the static Ref."""
    origin, corner, reference = [law.index[tuple(site)] for site in SITES]
    f0 = frame_chart(fields[origin])[0]
    fr = frame_chart(fields[reference])[0]
    path = links[origin, 0]@links[corner, 1]
    scalar = centre_scalar(f0.conj().T@path@fr, 18., law.generators)[0]
    labels = np.asarray([np.eye(4), *(SELECTED[j]@SELECTED[k]
                         for j, k in ((2, 3), (3, 1), (1, 2)))])/2
    mark = scalar*np.einsum('a,aij->ij', beta, f0@labels@fr.conj().T)
    transport = (np.eye(4), links[origin, 0], path)
    phase = (0., potential[origin, 0], potential[origin, 0]+potential[corner, 1])
    return np.concatenate([profile[j]*np.exp(1j*(phase[-1]-phase[j]))*
                           u.conj().T@mark for j, u in enumerate(transport)])/math.sqrt(2)


def primitive_gauss_source_probe(law, fields, links, potential, sector, profile, beta, omega, branch):
    """Noncentral full15 SU4 AND nonzero U1 covariance on the actual source."""
    angles = np.array([.006*(x+1)*law.generators[(x+11)%15] for x in range(law.volume)])
    phase = .005*np.arange(law.volume)
    gauge = np.array([expm(1j*x) for x in angles])
    moved_phi = gauge[:, None]@fields@gauge.conj().transpose(0, 2, 1)[:, None]
    moved_links = np.array([[gauge[x]@links[x, j]@gauge[law.target[x, j]].conj().T
                             for j in range(3)] for x in range(law.volume)])
    moved_potential = potential+phase[:, None]-phase[law.target]
    coefficient = native_source_coefficient(law, fields, links, potential, profile, beta)
    moved = native_source_coefficient(law, moved_phi, moved_links, moved_potential, profile, beta)
    reference = law.index[tuple(SITES[2])]
    expected = np.concatenate([np.exp(1j*(phase[x]-phase[reference]))*gauge[x]@
                               coefficient[4*j:4*j+4]@gauge[reference].conj().T
                               for j, x in enumerate([law.index[tuple(site)] for site in SITES])])
    error = float(np.linalg.norm(moved-expected))
    check(error, 'actual Phi/Wilson source fullSU4/U1 Gauss covariance')
    onebody = np.zeros((16, 16), complex)
    for j, site in enumerate((*SITES, SITES[2])):
        x = law.index[tuple(site)]
        onebody[4*j:4*j+4, 4*j:4*j+4] = angles[x]+phase[x]*np.eye(4)
    primitive = sector.lift(onebody)
    B = sector.source(coefficient, branch)
    Bg = sector.source(moved, branch)
    left = Bg@expm_multiply(1j*primitive, omega)
    right = expm_multiply(1j*primitive, B@omega)
    primitive_error = float(np.linalg.norm(left-right))
    check(primitive_error, 'full-sector primitive Pauli source Gauss covariance')
    return dict(full15_noncentral_SU4=True, nonzero_U1=True, actual_coefficient_error=error,
        primitive_full_CAR_ref_aux_error=primitive_error,
        source_and_neutral_transformed_together=True, reference_static=True,
        local_source_transport_retained=True, actual_harmonic_and_coexact_Wilson_phase=True)


def closed_phi_loop(law: ImprovedPhiLaw, fields: np.ndarray, links: np.ndarray,
                    potential: np.ndarray, sector: PrimitiveSector):
    """Actual Cartesian field action plus Phi-derived primitive Gauss lift.

    This is the relative intrinsic loop: the reference and orbital frame are
    clamped. It is not a claim that an arbitrary packet's TOTAL J is a doublet.
    """
    from moving_physical_noether import physical_generators
    frames = np.array([frame_chart(field)[0] for field in fields])
    poles = np.array([frame@np.asarray(SELECTED)@frame.conj().T for frame in frames])
    physical_axes = np.array([physical_generators(cast(Any, law), pole) for pole in poles])
    primitive_axes, reference_axes = [], []
    for axis in range(3):
        mode_spin = np.zeros((12, 12), complex)
        for j, site in enumerate(SITES):
            mode_spin[4*j:4*j+4, 4*j:4*j+4] = physical_axes[law.index[tuple(site)], axis]
        primitive_axes.append(sector.lift(mode_spin))
        reference_spin = np.zeros((16, 16), complex)
        reference_spin[12:, 12:] = physical_axes[law.index[tuple(SITES[2])], axis]
        reference_axes.append(sector.lift(reference_spin))
    spin = physical_axes[:, 0]
    quarter_fields = []
    full_turn = None
    for theta in np.linspace(0., 2*math.pi, 65):
        rotation = np.array([[1., 0., 0.], [0., math.cos(theta), -math.sin(theta)],
                             [0., math.sin(theta), math.cos(theta)]])
        gauges = np.array([expm(1j*theta*s) for s in spin])
        physical = fields.copy()
        physical[:, 1:] = np.einsum('jk,xkli->xjli', rotation, fields[:, 1:])
        restored = gauges.conj().transpose(0, 2, 1)[:, None]@physical@gauges[:, None]
        colour = np.array([[gauges[x].conj().T@links[x, j]@gauges[law.target[x, j]]
                            for j in range(3)] for x in range(law.volume)])
        photon = potential
        if abs(theta/(math.pi/2)-round(theta/(math.pi/2))) < 1e-10:
            quarter_fields.append(dict(angle=float(theta),
                actual_full_Phi_distance=float(np.linalg.norm(restored-fields)),
                actual_SU4_distance=float(np.linalg.norm(colour-links)),
                actual_real_EM_distance=float(np.linalg.norm(photon-potential))))
        full_turn = restored, colour, photon, gauges
    assert full_turn is not None
    restored, colour, photon, gauges = full_turn
    check(float(np.linalg.norm(restored-fields)), 'actual Cartesian Phi full-turn closure')
    check(float(np.linalg.norm(colour-links)), 'actual fullSU4 full-turn closure')
    check(float(np.linalg.norm(photon-potential)), 'actual realEM full-turn closure')
    check(float(np.linalg.norm(gauges+np.eye(4))), 'Phi-generated Spin Gauss centre, not copied SU2')
    # A positive-control quantum wave on the FULL Cartesian Phi coordinates.
    # Orthogonal field pullback preserves its entire L2 norm/covariance. Unlike
    # a frozen classical mark, its intermediate quantum overlap is not one.
    probe_angle = math.pi/2
    step = 1e-5
    def transformed(theta):
        rotation = np.array([[1., 0., 0.], [0., math.cos(theta), -math.sin(theta)],
                             [0., math.sin(theta), math.cos(theta)]])
        gauge = np.array([expm(1j*theta*s) for s in spin])
        value = fields.copy()
        value[:, 1:] = np.einsum('jk,xkli->xjli', rotation, fields[:, 1:])
        return gauge.conj().transpose(0, 2, 1)[:, None]@value@gauge[:, None]
    derivative = (transformed(step)-transformed(-step))/(2*step)
    sigma = .003
    displacement = float(np.linalg.norm(transformed(probe_angle)-fields))
    gaussian_overlap = math.exp(-displacement**2/(4*sigma**2))
    generator_second = float(np.linalg.norm(derivative)**2/(2*sigma**2))
    require(generator_second > 1e-8 and gaussian_overlap < .999999,
            'actual quantum Cartesian Phi rotation is not a spectator')
    return primitive_axes, reference_axes, dict(angle=2*math.pi, field_quantum_action='unitary SO3 Cartesian pullback, with Phi-derived fullSU4 Gauss restoration',
        field_quantum_fullturn_identity=True, field_action_applies_to_any_quantum_state=True,
        quantum_Gaussian_positive_control_only=dict(full_Phi_sites=law.volume, sigma=sigma,
            intermediate_overlap=gaussian_overlap, physical_field_generator_second_moment=generator_second,
            not_a_replacement_for_the_native_coupled_source=True),
        actual_field_quarter_word=quarter_fields, primitive_Phi_spin_norm=float(sparse.linalg.norm(primitive_axes[0])),
        reference=dict(position='static', spin='explicit opposite Phi-derived Ref connection clamps physical spin; no Ref hopping',
                       ordinary_mass_offsets_retained=True, fixed_Born_receiver_unchanged=True),
        orbital='intrinsic Phi-vector-index loop is onsite: no rigid spatial rotation of the fixed torus or Ref; separately controlled scalar_orbital participates only in the out/back current history',
        EM='intrinsic Phi rotation leaves actual real coexact A intact; separate full-Noether bump retains quadratic Maxwell and literal sine flow',
        harmonic='actual paired non-small R3/Bloch harmonic boundary is unchanged by intrinsic rotation, no oscillator coercivity',
        original_source='prepared ONCE; nonlinear eta18 scalar and local-birth curvature retained without time-dependent recreation or normalization',
        source_scalar_curvature_retained=True, arbitrary_total_J_is_not_three_quarters=True)


def spin_rotation(spin: sparse.csr_matrix, theta: float, value: np.ndarray):
    """Exact finite-spectrum exponential of ACTUAL primitive Phi Gauss spin.

    In these full16-mode TOTAL6/10 sectors, m runs from-3 to3 in half steps.
    The thirteen-point Chebyshev polynomial is the exact spectral function,
    not a low-band projection or a copied four-by-four spin matrix.
    """
    spectrum = np.arange(-6, 7)/2
    coefficients = np.polynomial.chebyshev.chebfit(spectrum/3, np.exp(-1j*theta*spectrum), 12)
    older = np.zeros_like(value)
    previous = np.zeros_like(value)
    for coefficient in coefficients[:0:-1]:
        current = (2/3)*(spin@previous)-older+coefficient*value
        older, previous = previous, current
    return (spin@previous)/3-older+coefficients[0]*value


def quantum_group_source_binding(law: ImprovedPhiLaw, snapshots, coefficients,
                                 sector: PrimitiveSector, branch: int, normalizer: float,
                                 Bfull: sparse.csr_matrix, source: np.ndarray,
                                 omega: np.ndarray, lifts):
    """Join TRUE quantum Phi/Ref differential words to THIS primitive source.

    The SOURCE coefficient action extends by the whole CAR operator identity;
    it is not a reprepare, a quaternion projection or a total-state Casimir.
    Independent endpoint frames use their actual Phi-frame bridge. Noncentral
    Wilson factors stay INSIDE the full16 matrix seed, including every Ref label.
    """
    from cartesian_noether_group import (
        cartesian_source_group, integer_orientation, seed_coefficients,
        source_embedding, source_matrix, source_word)
    cache, records = {}, []
    count = len(snapshots)
    first_coefficients = [[] for _ in range(3)]
    identity = np.eye(3)
    for (fields, _, _), coefficient in zip(snapshots, coefficients):
        frames = np.array([frame_chart(field)[0] for field in fields])
        ref = frames[law.index[tuple(SITES[2])]]
        pole = ref@np.asarray(SELECTED)@ref.conj().T
        key = pole.tobytes()
        if key not in cache:
            cache[key] = cartesian_source_group(pole, branch=branch)
        group = cache[key]
        row_coefficients = [[] for _ in range(3)]
        for j, site in enumerate(SITES):
            bridge = ref@frames[law.index[tuple(site)]].conj().T
            matrix = coefficient[4*j:4*j+4]
            actual = matrix if branch == 1 else matrix.conj().T
            seed = bridge@actual if branch == 1 else actual@bridge.conj().T
            reconstructed = source_matrix(identity, pole, seed, bridge, branch)
            check(float(np.linalg.norm(reconstructed-actual)),
                  'same actual Phi/Wilson/rawquartet source identity, no reprepare')
            vector = seed_coefficients(group, seed)
            values = group['points']@vector
            second = sum(source_word(group, seed, (i, i)) for i in range(3))
            true_second_error = check(float(np.linalg.norm(second-.75*values)),
                                      'literal quantum Phi/ref SOURCE second word on actual full16 seed', 2e-8)
            wave = group['embedding']@vector
            rotated = vector.copy()
            for axis, angle in ((0, math.pi), (1, math.pi), (0, -math.pi), (1, -math.pi)):
                rotated = expm(-1j*angle*group['K'][axis])@rotated
            rotated_wave = group['embedding']@rotated
            quantum_word_error = check(float(np.linalg.norm(rotated_wave+wave)),
                                       'genuine noncommuting quantum Phi/Ref ordered word on actual source', 2e-8)
            wave_norm_squared = float(np.vdot(wave, wave).real)
            observed_holonomy = complex(np.vdot(wave, rotated_wave)/wave_norm_squared) if wave_norm_squared else None
            check(float(np.linalg.norm(source_embedding(group, identity)@vector-
                                       group['creator_vectors']@seed.reshape(16))),
                  'quantum source embedding identity and actual primitive source')
            for axis in range(3):
                acted = (group['K'][axis]@vector).reshape(4, 4)
                transformed = bridge.conj().T@acted if branch == 1 else acted@bridge
                row_coefficients[axis].append(transformed if branch == 1 else transformed.conj().T)
            records.append(dict(node=len(records)//3, source_site=j,
                full16_seed_norm=float(np.linalg.norm(seed)), true_second_SOURCE_error=true_second_error,
                observed_quantum_ordered_word_holonomy=observed_holonomy,
                quantum_ordered_word_error=quantum_word_error,
                actual_primitive_field_embedding_dimension=group['embedding'].shape[0]))
        for axis in range(3):
            first_coefficients[axis].append(np.concatenate(row_coefficients[axis]))
    primitive_axes = [sparse.block_diag([row[0][axis] for row in lifts], format='csr') for axis in range(3)]
    first_errors, neutral_noise = [], []
    second = np.zeros_like(source)
    for axis, generator in enumerate(primitive_axes):
        coefficient_word = sparse.block_diag([sector.source(matrix, branch) for matrix in first_coefficients[axis]],
                                             format='csr')/normalizer
        n = generator@omega
        first_errors.append(check(float(np.linalg.norm(generator@source-Bfull@n-coefficient_word@omega)),
                                  'quantum physical SOURCE action extends to actual full CAR/current state', 2e-8))
        neutral_noise.append(float(np.linalg.norm(n)))
        second += generator@(generator@source)-2*generator@(Bfull@n)+Bfull@(generator@n)
    primitive_second_error = check(float(np.linalg.norm(second-.75*source)),
                                   'SOURCE operator second word retains SAME neutral right action', 2e-8)
    _, _, bosonic = integer_orientation()
    bosonic_word = np.eye(10, dtype=complex)
    for axis, angle in ((0, math.pi), (1, math.pi), (0, -math.pi), (1, -math.pi)):
        bosonic_word = expm(-1j*angle*bosonic[axis])@bosonic_word
    bosonic_error = check(float(np.linalg.norm(bosonic_word-np.eye(10))),
                         'actual integer quantum Phi closed noncommuting word')
    return dict(actual_native_source_nodes=count, source_records=records,
        true_quantum_Phi_group_controls=[group['controls'] for group in cache.values()],
        actual_SOURCE_first_word_full_CAR_errors=first_errors,
        actual_SOURCE_second_word_full_CAR_error=primitive_second_error,
        matched_neutral_spin_image_noise=neutral_noise, neutral_right_action_retained=True,
        bosonic_quantum_ordered_word_error=bosonic_error, source_ordered_word_holonomy=-1,
        physical_frame_scope='intrinsic onsite Phi-vector index, independent actual endpoint frames; no spatial torus/Ref rotation',
        scalar_scope='original eta18 local-birth curvature retained separately, not replaced by this pureClifford coefficient-sector Casimir',
        normal_scope='unrestricted off-Clifford endpoint deviations use the joined paid normal/field transfer; no exact arbitrary-normal restorer asserted',
        source_prepared_once=True, full16_noncentral_seed_retained=True)


def native_source_chart(frame: np.ndarray, band: RotationBand, descriptor: dict[str, Any],
                        branch: int, leg: float = .2, pulse_time: float = .01) -> dict[str, Any]:
    """Prepare the shared actual Pauli/Phi/source/current carrier ONCE.

    The compact native electrical reference is a finite prototype; an independent
    initial reference-Phi copy is exported for the selected far-RIGHT field.
    Physical source identification must not move that anchor with bulk site2.
    """
    sector = PrimitiveSector(10 if branch == 1 else 6)
    law = ImprovedPhiLaw(3, band)
    heavy = ImprovedPhiLaw(3, RotationBand(**BARE))
    rng = np.random.default_rng(22210691)
    basis = matrix_unit_basis()
    normal = np.einsum('xma,aij->xmij', rng.normal(size=(law.volume, 4, 15)), basis)
    normal /= np.linalg.norm(normal)
    nodes, modes = np.linalg.eigh(np.array([[0., 1.], [1., 0.]])/math.sqrt(2))
    p2 = modes.T@np.diag([.5, 1.5])@modes
    scales = (.02, .04, .07)
    ue, chi = .0025, .0004
    K = float(C)**2/(2*ue)
    angles = np.array([.71, -.46, .32])  # Deliberately NOT a small harmonic coordinate.
    beta = np.array([1., .3j, -.2+.4j, .6])
    ds = np.asarray(descriptor['compact_displacement'])
    eta = np.asarray(descriptor['source_eta'])
    profile = np.array([eta[np.flatnonzero(np.all(ds == site, axis=1))[0]] for site in SITES])
    require(np.linalg.norm(profile) > 0, 'literal SC19 restriction supports L contour')
    green = np.array([[5., -1., -4.], [-1., 2., -1.], [-4., -1., 5.]])/9
    # This is the Moore-Penrose Green of the actual two-link contour.
    incidence = np.array([[1., 0.], [-1., 1.], [0., -1.]])
    check(np.linalg.norm(incidence@incidence.T@green-np.eye(3)+np.ones((3, 3))/3),
          'full scalar-boundary Green with zero mode retained')
    from moving_physical_noether import field_noether_galerkin
    velocity = .025*np.column_stack((-law.sites[:, 1]+1, law.sites[:, 0]-1, np.zeros(law.volume)))
    cutoff = np.ones(law.volume)
    reference_site = law.index[tuple(SITES[2])]
    cutoff[reference_site] = 0
    velocity[reference_site] = 0
    chosen = [law.index[tuple(site)] for site in SITES]
    chosen_modes = np.concatenate([np.arange(4*x, 4*x+4) for x in chosen])
    angular_speed = .01/(2*leg+2*pulse_time)
    orbitals_physical = []
    full_bonds = [(x, int(law.target[x, j])) for x in range(law.volume) for j in range(3)]
    colour_magnetic = .7
    field_chart_cache = {}
    currents = [[], []]
    contacts = [[], []]
    reference_scalar_boundaries = []
    blocks, sources, omegas = [], [], []
    sources_ops = []
    offsets = []
    field_snapshots = []
    coefficient_snapshots = []
    for kappa_sign in (-1, 1):
        for z0, z1, z2 in itertools.product(nodes, repeat=3):
            chart_key = (kappa_sign, float(z1), float(z2))
            if chart_key not in field_chart_cache:
                base_fields = np.tile(frame, (law.volume, 1, 1, 1))+.003*normal
                base_links = np.asarray([[expm(1j*(.03+scales[1]*z1)*(1+x+j)*law.generators[3])
                                          for j in range(3)] for x in range(law.volume)])
                base_potential = np.tile(kappa_sign*angles/3, (law.volume, 1))
                coexact = np.cos(2*math.pi*law.sites[:, 0]/3)
                base_potential[:, 2] += scales[2]*z2*coexact
                poses = [.03+scales[0]*float(z) for z in nodes]
                field_chart_cache[chart_key] = angular_chart(
                    law, base_fields, base_links, base_potential, velocity, cutoff, poses)
            fields, links, potential = field_chart_cache[chart_key][int(np.argmin(abs(nodes-z0)))]
            field_snapshots.append((fields, links, potential))
            h = native_onebody(law, fields, links, potential)
            orbital_full = scalar_orbital(cast(Any, law), links, potential, velocity)
            orbital = orbital_full[np.ix_(chosen_modes, chosen_modes)]
            orbitals_physical.append(sector.lift(orbital))
            hh = native_onebody(heavy, fields, links, np.zeros_like(potential))
            eig, orbitals = np.linalg.eigh(hh)
            omega = sector.neutral(orbitals[:, :6], branch)
            coefficient = native_source_coefficient(law, fields, links, potential, profile, beta)
            coefficient_snapshots.append(coefficient)
            B = sector.source(coefficient, branch)
            source = B@omega
            # Whole reference charge, including finite-energy unwanted occupations.
            refcharge = sector.nref-(4 if branch == 1 else 0)
            rho = sector.density.copy().astype(float)
            rho[:, 2] += refcharge
            check(float(np.max(abs(rho.sum(axis=1)))), 'full reference Gauss compensation')
            coulomb = ue*np.einsum('xi,ij,xj->x', rho, green, rho)
            missing_ref_coulomb = ue*np.einsum('xi,ij,xj->x', sector.density, green, sector.density)
            reference_scalar_boundaries.append(sparse.diags(coulomb-missing_ref_coulomb))
            # ORIGINAL scalar/colour and real-quadratic Maxwell boundaries,
            # not the source's two-link Wilson product or a 9*q_A^2 proxy.
            scalar = float(sum(local_potential(fields, full_bonds, links.reshape(-1, 4, 4))))
            heavy_full = heavy.onebody(fields, links, np.zeros_like(potential))
            scalar -= float(np.linalg.eigvalsh(heavy_full)[2*law.volume:].sum())
            curl = []
            colour_loop = 0.
            for j in range(3):
                for k in range(j+1, 3):
                    curl.append(potential[:, j]+potential[law.target[:, j], k]-
                                potential[law.target[:, k], j]-potential[:, k])
                    plaquette = (links[:, j]@links[law.target[:, j], k]@
                                 links[law.target[:, k], j].conj().transpose(0, 2, 1)@
                                 links[:, k].conj().transpose(0, 2, 1))
                    colour_loop += float(np.sum(4-np.trace(plaquette, axis1=1, axis2=2).real))
            scalar += K/2*float(np.sum(np.asarray(curl)**2))+colour_magnetic*colour_loop
            # Ordinary harmonic p^2 and its actual +/-kappa pairing are retained.
            harmonic = ue/27*(kappa_sign*.37)**2
            unwanted = (sector.nref != (4 if branch == 1 else 0)) & (sector.nref != (3 if branch == 1 else 1))
            diagonal = coulomb+scalar+harmonic+11*unwanted
            H = sector.lift(h)+sparse.diags(diagonal)+9*(sector.identity-sector.cycle)/2+.023*sector.defect
            blocks.append(H)
            sources.append(source); omegas.append(omega); sources_ops.append(B)
            offsets.append(scalar+harmonic)
            for edge, _ in enumerate(EDGE_PATH):
                # Cut flux, not a nearest-only spectator: every range2 boundary
                # insertion crossing the actual region participates.
                region = np.zeros(law.volume)
                region[chosen[:edge+1]] = 1
                direction = region[:, None]-region[law.target]
                currents[edge].append(sector.lift(native_onebody(law, fields, links, potential, direction)))
                contacts[edge].append(sector.lift(native_onebody(law, fields, links, potential, direction, direction)))
    count = len(blocks)
    H = sparse.block_diag(blocks, format='csr')
    kinetic = sparse.csr_matrix((count*sector.dimension, count*sector.dimension), dtype=complex)
    for axis, coefficient in enumerate((chi/(2*scales[0]**2), .3*chi/(2*scales[1]**2), ue/(2*scales[2]**2))):
        factors = [sparse.csr_matrix(p2) if k == axis else sparse.eye(2) for k in range(3)]
        k3 = sparse.kron(sparse.kron(factors[0], factors[1]), factors[2])
        kinetic += coefficient*sparse.kron(sparse.eye(2), sparse.kron(k3, sector.identity), format='csr')
    H += kinetic
    derivative = modes.T@np.array([[0., 1.], [-1., 0.]])@modes/(math.sqrt(2)*scales[0])
    d8 = np.kron(np.kron(derivative, np.eye(2)), np.eye(2))
    d16 = np.kron(np.eye(2), d8)
    field_generator = field_noether_galerkin(np.ones((count, 1)), d16[None])
    Gfield = sparse.kron(sparse.csr_matrix(field_generator), sector.identity, format='csr')
    Gmatter = sparse.block_diag(orbitals_physical, format='csr')
    G = Gfield+Gmatter
    source = np.concatenate(sources)/math.sqrt(count)
    normalizer = float(np.linalg.norm(source))
    source /= normalizer  # One GLOBAL norm, never per fibre/ref whitening.
    omega = np.concatenate(omegas)/math.sqrt(count)
    Bfull = sparse.block_diag(sources_ops, format='csr')/normalizer
    J = [sparse.block_diag(rows, format='csr') for rows in currents]
    Q = [sparse.block_diag(rows, format='csr') for rows in contacts]
    midnumber = sector.occupation[:, 4:8].sum(axis=1)
    scalar_pulse = sparse.diags(np.tile(math.pi/pulse_time*midnumber, count))
    Hp = H+scalar_pulse
    return dict(sector=sector, law=law, H=H, Hp=Hp, G=G, Gfield=Gfield, Gmatter=Gmatter,
        source=source, omega=omega, B=Bfull, current=J, contact=Q,
        source_global_norm=normalizer, node_count=count, field_snapshots=field_snapshots,
        coefficient_snapshots=coefficient_snapshots, node_neutrals=omegas,
        independent_reference_phi=[phi[reference_site].copy() for phi, _, _ in field_snapshots],
        prototype_reference_coordinate=tuple(SITES[2]), selected_reference_coordinate=(0, NSTAR//2, NSTAR//2),
        profile=profile, raw_quartet=beta, scalar_pulse=scalar_pulse, incidence=incidence,
        ordinary_offsets=offsets, harmonic_angles=angles, UE=ue, K=K, chi=chi,
        colour_magnetic=colour_magnetic, full_bonds=full_bonds,
        reference_scalar_boundaries=reference_scalar_boundaries, angular_speed=angular_speed)


def native_contour(frame: np.ndarray, band: RotationBand, descriptor: dict[str, Any],
                   branch: int, intrinsic_receipt: dict[str, Any],
                   leg: float = .2, pulse_time: float = .01):
    from moving_physical_noether import source19_boundary_word, source19_nested_casimir
    chart = native_source_chart(frame, band, descriptor, branch, leg, pulse_time)
    from spatial_noether_identification import native_spatial_identification
    spatial_identification = native_spatial_identification(chart, frame, band, branch, intrinsic_receipt)
    sector, law = chart['sector'], chart['law']
    H, Hp, G, Gfield, Gmatter = (chart[key] for key in ('H', 'Hp', 'G', 'Gfield', 'Gmatter'))
    source, omega, Bfull = (chart[key] for key in ('source', 'omega', 'B'))
    J, Q = chart['current'], chart['contact']
    normalizer, count = chart['source_global_norm'], chart['node_count']
    field_snapshots, coefficient_snapshots = chart['field_snapshots'], chart['coefficient_snapshots']
    omegas, profile, beta = chart['node_neutrals'], chart['profile'], chart['raw_quartet']
    scalar_pulse, incidence = chart['scalar_pulse'], chart['incidence']
    offsets, angles = chart['ordinary_offsets'], chart['harmonic_angles']
    ue, K, chi, colour_magnetic = (chart[key] for key in ('UE', 'K', 'chi', 'colour_magnetic'))
    full_bonds, reference_scalar_boundaries = chart['full_bonds'], chart['reference_scalar_boundaries']
    angular_speed = chart['angular_speed']
    fields, links, potential = field_snapshots[-1]
    lifts = [closed_phi_loop(law, phi, u, a, sector) for phi, u, a in field_snapshots]
    loop = dict(angle=2*math.pi, quantum_field_nodes=count,
        actual_Phi_conditional_lifts=[row[2] for row in lifts],
        not_a_static_calibration_or_copied_SU2=True)
    gauss_covariance = primitive_gauss_source_probe(law, fields, links, potential, sector, profile, beta, omegas[-1], branch)
    S = cast(sparse.csr_matrix, sparse.block_diag([row[0][0] for row in lifts], format='csr'))
    Sref = cast(sparse.csr_matrix, sparse.block_diag([row[1][0] for row in lifts], format='csr'))
    quantum_group = quantum_group_source_binding(law, field_snapshots, coefficient_snapshots,
        sector, branch, normalizer, Bfull, source, omega, lifts)
    loop['genuine_quantum_cartesian_group_SOURCE_binding'] = quantum_group
    ordinary_lower = float(np.min(H.diagonal().real-np.asarray(abs(H).sum(axis=1)).ravel()+abs(H.diagonal())))
    pulse_bound = float(np.max(abs(scalar_pulse.diagonal())))
    Qordinary = H-ordinary_lower*sparse.eye(H.shape[0])
    source_first_initial = float(np.linalg.norm(Qordinary@source))
    neutral_first_initial = float(np.linalg.norm(Qordinary@omega))
    # The same ordinary Hamiltonian/source is pulled back through the REAL
    # quantum-field moving frame. It is NOT exp(-i(H-omegaG)T) factored by fiat.
    total_time = 2*leg+2*pulse_time
    def pose(t):
        u = t/total_time
        theta = 2*math.pi*(10*u**3-15*u**4+6*u**5)
        speed = 2*math.pi/total_time*(30*u**2-60*u**3+30*u**4)
        angle = .01*64*u**3*(1-u)**3
        rate = .01*192/total_time*u**2*(1-u)**2*(1-2*u)
        return theta, speed, angle, rate
    def frame_action(t, value):
        theta, _, angle, _ = pose(t)
        pulled = value if abs(angle) < 1e-15 else expm_multiply(-1j*angle*G, value)
        return spin_rotation(S, theta, pulled)
    cuts = [np.tile(sector.occupation[:, :4*(edge+1)].sum(axis=1), count) for edge in range(2)]
    times, states, neutral_states = [0.], [source], [omega]
    density_rows = [observations(source, sector, count)-observations(omega, sector, count)]
    flux = [np.zeros(2)]
    source_graph_history, neutral_graph_history, driver_energy_history, driver_second_history = [], [], [], []
    current_history, norm_history = [], []
    current_action_history, contact_action_history = [], []
    elapsed = 0.
    steps = 24
    pulse_count = 0
    for duration, generator in ((leg, H), (pulse_time, Hp), (leg, H), (pulse_time, Hp)):
        if generator is Hp:
            pulse_count += 1
        segment = expm_multiply(-1j*generator, states[-1], start=0., stop=duration, num=steps+1, endpoint=True)
        neutral_segment = expm_multiply(-1j*generator, neutral_states[-1], start=0., stop=duration, num=steps+1, endpoint=True)
        js = []
        for k, (v, w) in enumerate(zip(segment, neutral_segment)):
            at = elapsed+duration*k/steps
            _, speed, _, rate = pose(at)
            mapped = frame_action(at, np.column_stack((v, w, generator@v, generator@w, G@v, G@w)))
            lab, neutral_lab = mapped[:, 0], mapped[:, 1]
            charged_action = mapped[:, 2]+rate*mapped[:, 4]+speed*(S@lab)
            neutral_action = mapped[:, 3]+rate*mapped[:, 5]+speed*(S@neutral_lab)
            js.append(np.array([-2*(np.vdot(cut*lab, charged_action).imag-
                                    np.vdot(cut*neutral_lab, neutral_action).imag) for cut in cuts]))
            qsource = float(np.linalg.norm(Qordinary@v))
            qneutral = float(np.linalg.norm(Qordinary@w))
            require(qsource <= source_first_initial+2*pulse_count*pulse_bound+1e-7,
                    'actual driven source first graph, not static Omega moments')
            require(qneutral <= neutral_first_initial+2*pulse_count*pulse_bound+1e-7,
                    'actual driven neutral first graph, not reset Hn')
            if k:
                times.append(at); states.append(v); neutral_states.append(w)
                density_rows.append(observations(lab, sector, count)-observations(neutral_lab, sector, count))
                flux.append(flux[-1]+duration/steps*(js[k-1]+js[k])/2)
                source_graph_history.append(qsource); neutral_graph_history.append(qneutral)
                driver_energy_history.append(float(np.vdot(lab, charged_action).real))
                driver_second_history.append(float(np.linalg.norm(charged_action)**2))
                current_action_history.append([float(np.linalg.norm(operator@v)) for operator in J])
                contact_action_history.append([float(np.linalg.norm(operator@v)) for operator in Q])
                current_history.append(js[-1]); norm_history.append(abs(np.linalg.norm(lab)-1))
        elapsed += duration
    density = np.array(density_rows)
    flux = np.array(flux)
    continuity = np.max(abs(density-density[0]-flux@incidence.T))
    # Sign convention: dH/dA has positive oriented density divergence.
    alternate = np.max(abs(density-density[0]+flux@incidence.T))
    if alternate < continuity:
        flux = -flux
        continuity = alternate
    check(float(continuity), 'measured microscopic contour continuity', 3e-6)
    norm_error = max(norm_history)
    check(float(norm_error), 'positive-clock primitive quantum normalization')
    movement = float(np.max(np.linalg.norm(density-density[0], axis=1)))
    require(movement > 1e-5, 'spatial current/density actually moves on noncollinear contour')
    recovery = float(np.linalg.norm(density[-1]-density[0]))
    require(recovery < movement, 'positive-clock scalar echo improves spatial recovery')
    loopQ = sparse.kron(sparse.eye(count), (sector.identity-sector.cycle)/2, format='csr')
    leakage = float(np.linalg.norm(loopQ@states[-1]))
    require(leakage > 1e-7, 'nonzero auxiliary defect dynamically populates complement')
    charged_ref = 3 if branch == 1 else 1
    refweights = np.sum(abs(source.reshape(count, sector.dimension))**2, axis=0)
    check(abs(float(refweights@(sector.nref == charged_ref))-1), 'actual bar4/4 source embedding')
    components = np.array([refweights@((sector.nref == charged_ref)&
                            (sector.occupation[:, 12+ref] == (0 if branch == 1 else 1))) for ref in range(4)])
    require(float(np.min(components)) > 1e-4, 'all four actual reference components populated')
    photon_current = [float(np.linalg.norm(q@states[len(states)//2])) for q in Q]
    require(min(photon_current) > 1e-6, 'same moving carrier has actual photon contacts')
    boundary_defect = float(np.linalg.norm(sparse.block_diag(reference_scalar_boundaries, format='csr')@source))
    require(boundary_defect > 1e-8, 'missing actual scalar/reference Coulomb boundary Hamiltonian rejects')
    no_ref_charge = float(abs(np.sum(observations(source, sector, count))))
    require(no_ref_charge > .99, 'missing reference violates actual Gauss charge')
    # Actual Heisenberg-advected source on SAME moving neutral vector: B_t=Uc B Un*.
    # Compute Hn action via forward propagation, not Jn Omega=0.
    Hn_omega = H@omega
    propagated_Hn = expm_multiply(-1j*H*leg, Hn_omega)
    Bh = Bfull@Hn_omega
    propagated_Bh = expm_multiply(-1j*H*leg, Bh)
    evolved = frame_action(leg, states[steps])
    # Full ordinary+inertial right action from the SAME neutral history.
    theta, speed, angle, rate = pose(leg)
    neutral_pulled = frame_action(leg, neutral_states[steps])
    ordinary_right = propagated_Bh
    # B_t Hn Omega_t = R Uc B0 Un* R* Hn Omega_t, never zero.
    def inverse_frame(value):
        return expm_multiply(1j*angle*G, spin_rotation(S, -theta, value))
    neutral_extra = rate*(G@neutral_states[steps])+speed*inverse_frame(S@neutral_pulled)
    neutral_initial_extra = expm_multiply(1j*H*leg, neutral_extra)
    extra_right = expm_multiply(-1j*H*leg, Bfull@neutral_initial_extra)
    propagated_Bh = frame_action(leg, ordinary_right+extra_right)
    def driver_action(value):
        pulled = inverse_frame(value)
        return frame_action(leg, H@pulled+rate*(G@pulled))+speed*(S@value)
    driver = cast(Any, LinearOperator)(H.shape, matvec=driver_action, matmat=driver_action, dtype=complex)
    dB = -1j*(driver@evolved-propagated_Bh)
    from moving_physical_noether import timed_source
    timed = timed_source(evolved[:, None], dB[:, None], driver, driver,
                         right_action=propagated_Bh[:, None])
    residual = np.asarray(timed['residual']).reshape(-1)
    no_bdot = float(np.linalg.norm(driver@evolved-propagated_Bh))
    check(float(np.linalg.norm(residual)), 'same source physical -iBdot equation')
    require(no_bdot > 1e-5, 'missing Bdot changes actual moving equation')
    energy = expectation(source, H)
    secondmoment = float(np.linalg.norm(H@source)**2)
    require(math.isfinite(energy) and math.isfinite(secondmoment), 'finite physical energy and second graph')
    neutral_noise = G@omega
    first_word = G@source-Bfull@neutral_noise
    second_word = G@(G@source)-2*G@(Bfull@neutral_noise)+Bfull@(G@neutral_noise)
    static_first = Gmatter@source-Bfull@(Gmatter@omega)
    cloud_defect = float(np.linalg.norm(first_word-static_first))
    require(cloud_defect > 1e-6, 'missing physical field/cloud Noether contribution rejects')
    require(float(np.linalg.norm(neutral_noise)) > 1e-6, 'actual neutral Noether noise is nonzero')
    angular_state = expm_multiply(-.01j*G, source)
    angular_phase = complex(np.vdot(source, angular_state))
    static_angular = expm_multiply(-.01j*Gmatter, source)
    static_rotation_defect = float(np.linalg.norm(angular_state-static_angular))
    require(static_rotation_defect > 1e-6, 'mere static internal/orbital rotation differs from actual fullfield pulse')
    noether = dict(physical_angular_velocity=angular_speed, angle=.01,
        field_generator_hermiticity=check(float(sparse.linalg.norm(Gfield-Gfield.conj().T)),
                                         'actual field Noether Hermiticity'),
        fullfield_integral_curve=True, literal_sin_curl=True, quantum_source_multiplication=True,
        first_source_word_norm=float(np.linalg.norm(first_word)),
        true_second_source_word_norm=float(np.linalg.norm(second_word)),
        same_neutral_Noether_action=float(np.linalg.norm(neutral_noise)),
        phase_on_same_source=angular_phase, phase_argument=float(np.angle(angular_phase)),
        angular_recovery_norm_error=check(abs(float(np.linalg.norm(angular_state))-1),
                                         'physical Noether pulse normalization'),
        missing_field_cloud_firstword_defect=cloud_defect,
        static_internal_only_rotation_defect=static_rotation_defect,
        source19_boundary_word=source19_boundary_word(),
        source19_true_nested_casimir=source19_nested_casimir(),
        native_G_scope='Hermitian Galerkin compression of actual fullPhi/SU4/EM Lie derivative plus scalar_orbital; orthogonal quantum modes are owner analytic dependencies')
    charged_end = frame_action(total_time, states[-1])
    neutral_end = frame_action(total_time, neutral_states[-1])
    missing_clamp = spin_rotation(S+Sref, 2*math.pi, states[-1])
    clamp_defect = float(np.linalg.norm(missing_clamp-frame_action(total_time, states[-1])))
    require(clamp_defect > 1.9, 'dropping actual Noether Ref clamp destroys relative halfspin holonomy')
    loop.update(actual_Noether_reference_source_action=float(np.linalg.norm(Sref@source)),
        actual_Noether_reference_neutral_action=float(np.linalg.norm(Sref@omega)),
        full_Noether_spin_minus_Ref_control_is_actual_relative_spin=True,
        missing_reference_clamp_phase_defect=clamp_defect,
        full_sector_Gauss_source_covariance=gauss_covariance,
        all_ref_occupations_preserved_at_finite_penalty=True)
    charged_holonomy = complex(np.vdot(states[-1], charged_end))
    neutral_holonomy = complex(np.vdot(neutral_states[-1], neutral_end))
    check(abs(charged_holonomy+1), 'actual prepared primitive charged closed relative2pi holonomy')
    check(abs(neutral_holonomy-1), 'matched ordinary neutral closed relative2pi holonomy')
    loop.update(charged_holonomy=charged_holonomy, neutral_holonomy=neutral_holonomy,
        relative_holonomy=charged_holonomy/neutral_holonomy,
        ordinary_history_phase_not_discarded=True, positive_clock=total_time)
    graph_caps = dict(ordinary_generator='Htheta=Rtheta H0 Rtheta*, not fixed H0-omegaG',
        driven_generator='Rtheta(H0+Vpi)Rtheta*+iRdot Rtheta*',
        source_prepared_once=True, source_first_initial=source_first_initial,
        source_first_observed_max=max(source_graph_history), neutral_first_initial=neutral_first_initial,
        neutral_first_observed_max=max(neutral_graph_history), pulse_operator_bound=pulse_bound,
        source_first_proved_upper=source_first_initial+4*pulse_bound,
        neutral_first_proved_upper=neutral_first_initial+4*pulse_bound,
        rho_charge_squared_operator_bound=float(np.max(np.sum(sector.density**2, axis=1))),
        quartic_Phi_moment='recomputed from transported ordinary Q cap; charge rho and Phi radial rho are distinct',
        physical_current_observed_max=np.max(np.abs(current_history), axis=0),
        native_driver_energy_min=min(driver_energy_history), native_driver_energy_max=max(driver_energy_history),
        native_driver_second_energy_graph_max=max(driver_second_history),
        actual_SOURCE_ordinary_current_action_norm_max=np.max(current_action_history, axis=0),
        actual_SOURCE_ordinary_contact_action_norm_max=np.max(contact_action_history, axis=0),
        current_graph_semantics='Jtheta=R Jordinary R*: these are actual transported-state operator norms, while physical cut flux above comes separately from FULL Hdrive including all controls',
        ordinary_lower_bound_for_graph_verification_only=ordinary_lower,
        ordinary_H_offsets_NOT_shifted_in_propagation=True,
        W_intertwining='local total charge commutes every local SU4 restorer, heavy normal vacuum and auxiliary cycle; Vpi W0=W0 Veff',
        Q_W_current_source_all_transported_by_same_frame=True)
    return dict(branch=branch, representation='bar4' if branch == 1 else '4',
        scope='three actual noncollinear sites, full12-mode+fullref4, primitive16edge fixed TOTAL sector with BOTH loop sectors; 8 quantum Phi/SU4/A nodes and paired non-small harmonic coordinates; finite Galerkin witness, not huge coupled ground',
        dimension=H.shape[0], whole_CAR_ref_dimension=sector.dimension//2,
        quantum_nodes=count, coordinates=SITES, global_source_norm=normalizer,
        times=times, charge_density=density, integrated_oriented_current=flux,
        density_movement=movement, density_recovery_error=recovery,
        continuity_quadrature_error=float(continuity), norm_error=float(norm_error),
        current_peak=np.max(np.abs(current_history), axis=0),
        photon_contact_state_norm=photon_current, finite_energy=energy, energy_second_moment=secondmoment,
        reference_component_weights=components, auxiliary_complement_norm=leakage,
        actual_neutral_H_action=float(np.linalg.norm(propagated_Hn)),
        moving_source_equation_error=float(np.linalg.norm(residual)), missing_Bdot_defect=no_bdot,
        physical_Noether=noether,
        spatial_physical_Noether_identification=spatial_identification,
        actual_closed_relative_rotation=loop, driven_state_graphs=graph_caps,
        harmonic_angles=angles, kappa_pair=[-.37, .37], ordinary_offsets_range=[min(offsets), max(offsets)],
        native_parameters=dict(UE=ue, K=K, chi=chi, K4=colour_magnetic, selected_UE_is_not_replaced=True),
        scalar_boundary_law=dict(full_Phi_sites=law.volume, alignment_bonds=len(full_bonds),
            actual_whole_heavy_occupied_modes=2*law.volume, heavy_carries_U1=False,
            electromagnetic='K/2 sum(real curl A)^2, not sin^2 or a coarse mode proxy',
            colour='K4 sum(4-ReTr actual plaquette)', reference='full compensated Coulomb Green'),
        scalar_pi_pulse_duration=pulse_time, scalar_pi_pulse_amplitude=math.pi/pulse_time,
        native_clock=elapsed, positive_clock=True, static_SC19_detector_unchanged=True,
        adverse=dict(static_internal_only=dict(rejected=True, density_error=movement),
            missing_reference=dict(rejected=True, uncompensated_charge=no_ref_charge),
            missing_scalar_boundary=dict(rejected=True, boundary_insertion_norm=boundary_defect),
            missing_cloud=dict(rejected=True, physical_firstword_defect=cloud_defect),
            missing_Bdot=dict(rejected=True, defect=no_bdot)))


def lowk_worldline(frame: np.ndarray, band: RotationBand):
    """Normalized paired-Bloch mobile packet, controlled physical T6 clock.

    125 EXACT legal lattice momenta. Positive-energy components are a state
    preparation, not a projected microscopic Hamiltonian. The entire 4x4 law
    propagates; the cloud/ref and high SOURCE tails are supplied separately.
    """
    momentum = np.array(list(itertools.product(range(-2, 3), repeat=3)), float)*.006
    weights = np.exp(-np.sum(momentum**2, axis=1)/(.012**2))
    weights /= weights.sum()
    # Six positive-clock legs; uniform electric impulse changes momentum only.
    centers = np.array([[.018, 0, 0], [0, .018, 0], [-.018, 0, 0],
                        [0, -.018, 0], [.012, .012, 0], [-.012, -.012, 0]])
    require(np.max(np.linalg.norm(momentum[:, None, :]+centers[None, :, :], axis=2)) < .05,
            'all populated momenta in explicit controlled .05 regime')
    state = np.empty((len(momentum), 4), complex)
    for n, p in enumerate(momentum+centers[0]):
        _, vectors = np.linalg.eigh(physical_h(p, frame, band))
        state[n] = math.sqrt(weights[n])*vectors[:, -1]
    paired = np.stack([state, state.conj()])/math.sqrt(2)
    original = paired.copy()
    worldline = [np.zeros(3)]
    flux = []
    local_errors = []
    phase_errors = []
    for center in centers:
        currents = []
        for n, p in enumerate(momentum+center):
            h = physical_h(p, frame, band)
            continuum = band.m*frame[0]+np.einsum('j,jik->ik', band.v*p, frame[1:])
            U = expm(-1j*h)
            paired[0, n] = U@paired[0, n]
            paired[1, n] = U.conj()@paired[1, n]
            vertices = np.array([reframe(row, frame) for row in band.current(p, p)])
            currents.append([2*np.vdot(paired[0, n], j@paired[0, n]).real for j in vertices])
            local_errors.append(float(np.linalg.norm(h-continuum, ord=2)))
            phase_errors.append(float(np.linalg.norm(U-expm(-1j*continuum), ord=2)))
        current = np.sum(currents, axis=0)
        flux.append(current)
        worldline.append(worldline[-1]+current)
    norm_error = abs(float(np.linalg.norm(paired))-1)
    check(norm_error, 'normalized paired Bloch whole-law history')
    cone = band.v*math.sqrt(float(np.trace(frame[1]@frame[1]).real/4))
    upper = cone*.05**3/6+math.sqrt(float(np.trace(frame[0]@frame[0]).real/4))*band.r*.05**4/4
    check(max(local_errors), 'controlled physical lattice H remainder', upper)
    check(max(phase_errors), 'controlled physical clock remainder per leg', upper)
    require(np.max(np.linalg.norm(worldline, axis=1)) > 1e-4, 'nontrivial physical current worldline')
    return dict(clock=6, cutoff=.05, maximum_populated_momentum=float(np.max(np.linalg.norm(momentum[:, None, :]+centers[None, :, :], axis=2))),
        normalized_paired_history=True, norm_error=norm_error, current_by_leg=flux,
        worldline=worldline, return_displacement_error=float(np.linalg.norm(worldline[-1])),
        transport_phase_error_upper=6*upper, measured_local_H_error=max(local_errors),
        measured_clock_unitary_error=max(phase_errors), physical_mass=band.m*math.sqrt(float(np.trace(frame[0]@frame[0]).real/4)),
        c_exact=str(C), rest_g=2., g_scope='quadratic lowk physical Pauli coefficient, not hard-q Lorentz claim',
        initial_and_final_fidelity=float(abs(np.vdot(original.ravel(), paired.ravel()))**2),
        hard_Compton_q=2*math.pi/5, hard_Compton_outside_controlled_regime=True,
        momentum_units='native spacing .006 is finite witness; exact legal momentum rounding bound 2pi/N per component',
        momentum_rounding_upper=2*math.pi/NSTAR)


def mobile_sc19_current_packet(frame, band, descriptor):
    """The complete unwhitened SC19 creation state, its T6 flight AND photons.

    This finite full-BZ free-comparator observation is not the fixed huge-volume
    detector probability. The latter is bound unchanged from its completed owner.
    Pauli Q comes from creator action on the filled sea, not a Hamiltonian cut.
    """
    q = 5
    points = 2*math.pi*np.array(list(itertools.product(range(q), repeat=3)))/q
    kernel = ImprovedBloch(points, frame, 6., band)
    info = kernel.info(np.zeros(3))
    profile = fibre.compact_fourier(points, np.asarray(descriptor['compact_displacement']),
                                   np.asarray(descriptor['source_eta']))
    mark = np.asarray(descriptor['source_normalized_creation_matrix'])
    initial = profile[:, None, None]*(info['Q']@mark)/math.sqrt(q**3)
    actual_norm = float(np.linalg.norm(initial))
    initial /= actual_norm  # actual single physical norm; no ref/column whitening
    densities = []
    current_history = []
    states = []
    for clock in np.linspace(0., 6., 13):
        phase = fibre.bounded_phase(-clock*info['E'])
        U = phase[:, None, None]*info['Q']+phase.conj()[:, None, None]*info['P']
        state = U@initial
        check(abs(float(np.linalg.norm(state))-1), 'same SC19/ref whole-H flight norm')
        position = np.fft.ifftn(state.reshape(q, q, q, 4, 4), axes=(0, 1, 2), norm='ortho')
        densities.append(np.sum(abs(position)**2, axis=(3, 4)))
        vertices = np.array([[reframe(row, frame) for row in band.current(p, p)]
                             for p in points])
        current_history.append([float(np.einsum('par,pab,pbr->', state.conj(),
                                                  vertices[:, j], state).real) for j in range(3)])
        states.append(state)
    incoming, outgoing = legal.represented_packets(float(C))
    response = np.zeros((q**3, 4, 4), complex)
    cloud = np.zeros_like(response)
    for first in incoming['modes']:
        for last in outgoing['modes']:
            response += kernel.connected_pair(first, last)
            cloud += legal.cloud_pair(kernel, first, last)
    complete = (response+cloud)@states[-1]
    missing_cloud = response@states[-1]
    cloud_defect = float(np.linalg.norm(complete-missing_cloud))
    require(cloud_defect > 1e-5, 'same moving SC19/current packet missing cloud rejects')
    contactless = ImprovedBloch(points, frame, 6., band, 'contact')
    wrong = sum(contactless.connected_pair(first, last)
                for first in incoming['modes'] for last in outgoing['modes'])+cloud
    contact_defect = float(np.linalg.norm(complete-wrong@states[-1]))
    require(contact_defect > 1e-5, 'same moving SC19/current packet missing contact rejects')
    no_ref = states[-1].copy()
    no_ref[:, :, 1:] = 0
    ref_defect = float(np.linalg.norm(complete-(response+cloud)@no_ref))
    require(ref_defect > 1e-5, 'same moving SC19/current packet missing full reference rejects')
    density_motion = float(np.linalg.norm(densities[-1]-densities[0]))
    require(density_motion > 1e-5, 'complete actual SC19 packet has physical spatial flight')
    return dict(scope='N5 full-BZ free comparison of the ENTIRE SC19 source and allRef4; no new physical detector probability or full coupled-ground simulation',
        source_sites=19, global_actual_source_norm=actual_norm, input_whitened=False,
        paired_history='PH-conjugate same normalized source; legal kappa/-kappa normalizable envelope remains source-owned',
        clock=6, density_before=densities[0], density_after=densities[-1],
        density_flight=density_motion, physical_current_history=current_history,
        moving_current_packet_response_norm=float(np.linalg.norm(complete)),
        actual_photon_modes='four distinct two-by-two supplied x/y hard mode terms, z polarization',
        photon_current_contact_law='literal range2 ImprovedBloch, all negative-final terms and crossed cloud retained',
        fixed_SC19_receiver_changed=False, transport_observation_separate=True,
        no_fibre_source_reset=True, missing_cloud_defect=cloud_defect,
        missing_contact_defect=contact_defect, missing_reference_defect=ref_defect)


def bind_packets(paths: dict[str, Path], descriptor: dict[str, Any]):
    packet = json.loads(paths['compton'].read_text())
    require(packet['inputs']['source_geometry_descriptor_sha256'] == sha(paths['descriptor']),
            'completed actual current/Compton and transport share literal SC19 descriptor')
    require(packet['unchanged']['source_sites'] == 19 and packet['unchanged']['allRef4'] and
            not packet['unchanged']['input_whitened'], 'fixed actual source/reference/input contract')
    for name, digest in packet['owned_source_pair'].items():
        require(sha(ROOT/name) == digest, 'completed Compton source bytes: '+name)
    for name, digest in packet['inputs']['primitive_published_source_pair'].items():
        require(sha(ROOT/name) == digest, 'completed primitive law source bytes: '+name)
    require(len(descriptor['source_eta']) == 19, 'SC19 and RAW41 not relabelled')
    transfer = packet['renewed_actual_transfer']
    return dict(input_sha256={name: sha(path) for name, path in paths.items()},
        completed_current_Compton_probability=packet['actual_physical_probability'],
        actual_source_graphs=transfer['source_graphs_log2']['scattering19'],
        high_SOURCE_cutoff_log2=transfer['source_cutoff_log2'],
        high_SOURCE_tail_rule='norm<=first/2^75430; half<=first/2^37715; no low-P spectral truncation of source',
        source_photon_debt_log2=transfer['source_photon_combined_absolute_debt_log2'],
        fixed_detector=dict(source='SC19',outputs=2,global_output_whitening=True,input_whitened=False,
                            reference='RIGHT fixed',moving_transport_is_a_separate_observation=True),
        original_bare_branch_retained=True, RAW41_is_distinct=True,
        ordinary_H=dict(E0_kappa='retained exactly in neutral and charged clocks; relative cancellation only',
                        all64='same paired history in each well, exact vacuum+normal offsets retained',
                        harmonic='UE/V p_h^2, source conjugation p_h->p_h+ell; no harmonic-coordinate smallness'))


def driven_source_contract(paths, compton, metadata, original_source) -> dict[str, Any]:
    """Re-derive actual driven SOURCE/Phi/current caps; never reuse Omega moments."""
    from noncompact_original_source_flight import sqrt_upper
    from noncompact_quantitative_normal_budget import ceil_log2
    from noncompact_scattering_flight import compact_descriptor
    # Exact Bernstein coefficient bounds for BOTH chosen clocks (native.42 and
    # legalT6). Derivatives through6 are bounded by128**order, not handwaved.
    duration_lower = F(21, 50)
    quintic = [F(x) for x in (0, 0, 0, 1, 1, 1)]
    bump = [F(0), F(0), F(0), F(4, 125), F(0), F(0), F(0)]
    clock_rows = []
    for order in range(1, 7):
        quintic = [(len(quintic)-1)*(b-a) for a, b in zip(quintic, quintic[1:])]
        bump = [(len(bump)-1)*(b-a) for a, b in zip(bump, bump[1:])]
        theta_cap = F(44, 7)*max(map(abs, quintic), default=F(0))/duration_lower**order
        bump_cap = max(map(abs, bump), default=F(0))/duration_lower**order
        require(max(theta_cap, bump_cap) <= F(128)**order, 'exact chosen-clock derivative bound')
        clock_rows.append(dict(order=order, theta_upper=str(theta_cap),
            outback_upper=str(bump_cap), dyadic_uniform_log2=7*order))
    # Faa-di-Bruno through6: Bell(6)=203<256. Field coefficients have frequency
    # at most2; four connection sums, two graph bits and four Young-sum bits.
    # 50+6+12+2+4=74 bits.96 pays these once, with22 explicit unused bits.
    reserve = 96
    actual_amplification_upper_log2 = 6*7+8+6+6*2+2+4
    require(actual_amplification_upper_log2 <= reserve, 'derived dynamic reserve fits selected96 bits')
    original = original_source['selected_source_flight']['sources']['scattering19']
    improved = compton['renewed_actual_transfer']['source_graphs_log2']['scattering19']
    primitive = compact_descriptor(decode(json.loads(paths['descriptor'].read_text())))
    field = primitive['field_sum']*sqrt_upper(F(1, NSTAR**3))
    changed_current = 16*NSTAR**3*9*field
    families = []
    for label, graphs, rho0, rhoquarter, jcoef, zcoef, cutoff, quartic in (
        ('original103', {name:{kind:int(value['log2_upper']) for kind, value in row.items()
                              if kind in ('norm', 'first', 'half')} for name, row in original['graphs'].items()},
         F(2)**int(original['rho_constant']['log2_upper']),
         F(2)**int(original['rho_quarter_constant']['log2_upper']),
         F(2)**int(original['current_coefficient']['log2_upper']),
         F(2)**int(original['contact_coefficient']['log2_upper']),
         int(original_source['selected_source_flight']['high_source_spectral_cutoff_log2']), None),
        ('improved108', improved, F(metadata['bounds']['renewed_quartic']['Crho']),
         F(metadata['bounds']['renewed_quartic']['CQ']), changed_current, changed_current*field,
         int(compton['renewed_actual_transfer']['source_cutoff_log2']),
         (F(metadata['bounds']['A4_unchanged']), F(metadata['bounds']['CD_changed'])))):
        rows = []
        for name, graph in graphs.items():
            norm = F(2)**int(graph['norm'])
            initial = F(2)**int(graph['first'])
            # M<=4*pi/.01<2048; UNNORMALIZED cloud norm multiplies the pulse.
            new_first = initial+F(2)**13*norm
            require(new_first < 2*initial, 'actual prepared birth/cloud grows by at mostONE graph bit')
            half = sqrt_upper(norm*new_first)
            quarter = sqrt_upper(norm*half)
            if quartic is None:
                radial_squared = sqrt_upper(rhoquarter**4*norm*new_first+rho0**4*norm**2)
            else:
                a4, cd = quartic
                radial_squared = sqrt_upper(2/a4*(norm*new_first+(cd+3)*norm**2))
            jcap = jcoef*(rho0*norm+rhoquarter*quarter)
            zcap = zcoef*(rho0*norm+rhoquarter*quarter)
            cut_control = F(1024*NSTAR**3)*norm
            mode_control = cut_control*field
            require(mode_control < jcap and mode_control*field < zcap,
                    'actual bounded orbital-control current/contact fit all-state source image caps')
            rows.append(dict(source=name, norm_log2_upper=int(graph['norm']),
                initial_first_log2_upper=int(graph['first']),
                driven_first_log2_upper=ceil_log2(new_first),
                driven_half_log2_upper=ceil_log2(half),
                rederived_Phi_radial_squared_norm_log2_upper=ceil_log2(radial_squared),
                arbitrary_state_ordinary_current_cap_log2_upper=ceil_log2(jcap),
                arbitrary_state_ordinary_contact_cap_log2_upper=ceil_log2(zcap),
                actual_orbital_control_cut_current_cap_log2_upper=ceil_log2(cut_control),
                high_actual_SOURCE_norm_log2_upper=ceil_log2(new_first)-cutoff,
                high_actual_SOURCE_half_log2_upper=ceil_log2(new_first)-cutoff//2))
        families.append(dict(law=label, actual_unchanged_source_cutoff_log2=cutoff, sources=rows,
            reprepare_at_driven_Omega=False, Phi_radial_rho_is_not_charge_density=True,
            normalized_and_unnormalized_graphs_not_confused=True))
    return dict(clock=clock_rows, finite_native_clock='21/50', microscopic_clock='6',
        field_normal_and_EM_generators='same true quantum field action; no copied SU2 spectator',
        actual_dynamic_amplification_log2_upper=actual_amplification_upper_log2,
        paid_dynamic_reserve_bits=reserve, unused_reserve_bits=reserve-actual_amplification_upper_log2,
        families=families,
        derivation='Each bounded V pulse: ||Qwout||<=||Qwin||+2M||w||; two add4M||w||<2^13||w||. Rtheta transports Qtheta,Wtheta,current,source and every ordinary E0kappa.',
        W_intertwining='Vpi is local total charge. It commutes site-local SU4 restoration, heavy normal-vacuum factor and the auxiliary cycle, so Vpi W0=W0 Veff. No normal-transfer commutator is hidden.',
        time_dependent_normal_map='Hdrive Wt-Wt hdrive-i Wtdot = R(H0 W0-W0 h0)r* +R(Vpi W0-W0 veff)r*. ALL inertial/field/control terms cancel by this exact moving-map identity, not by assuming [H0,G]=0.',
        all_state_quartic_form='||rhoPhi^2 psi||^2<=2/A4*(||psi|| ||Qpsi||+(CD+3)||psi||^2); original rho0/rhoquarter are the same all-domain form coefficients.',
        control_current='Charge cut current comes from ACTUAL Hdrive. Phi/EM/reference spin controls commute charge density; compensated intrinsic orbital term is zero. The genuine small full-Noether orbital control is bounded by1024 V ||SOURCE||.',
        photon_control='Pure photon Lie/control vertices intertwine the SHARED full coexact photon sector exactly; no old ordinary Born probability is claimed during controls. Matched Compton observation is a separate unchanged-law window after controls end.',
        source_phase_no_recreation=True, actual_high_SOURCE_uses_transported_Qtheta=True,
        normal_Wtheta_and_all64_ordinary_harmonic_boundaries_preserved=True)


def verify_cartesian_successor(paths: dict[str, Path], packet: dict[str, Any]) -> dict[str, Any]:
    """Recompute the narrowly authorized successor, preserving executed115 bytes."""
    correction_path = paths['cartesian_correction']
    witness = json.loads(correction_path.read_text())
    def local_artifact(entry):
        uri = entry['uri']
        require(uri.startswith('local://') and Path(uri[8:]).name == uri[8:],
                'Cartesian correction artifact is an explicit local basename')
        path = correction_path.parent/uri[8:]
        require(sha(path) == entry['sha256'], 'Cartesian correction artifact exact bytes')
        return path
    old_path = local_artifact(witness['original_executed_source'])
    require(sha(paths['cartesian']) == witness['original_packet']['sha256'] and
            sha(paths['cartesian_receipt']) == witness['original_native_receipt']['sha256'],
            'Cartesian executed packet/receipt remain immutable')
    require(sha(old_path) == packet['owned_source_pair_sha256']['cartesian_noether_group.py'] and
            sha(ROOT/'cartesian_noether_group.py') == witness['corrected_source_sha256'] and
            sha(ROOT/'cartesian_noether_group.md') == witness['unchanged_md_sha256'] ==
            packet['owned_source_pair_sha256']['cartesian_noether_group.md'],
            'exact original execution and authorized current successor source pairs')
    old_tree = ast.parse(old_path.read_text())
    current_tree = ast.parse((ROOT/'cartesian_noether_group.py').read_text())
    functions = {node.name: node for node in current_tree.body if isinstance(node, ast.FunctionDef)}
    native = functions['native_history']
    expected_init = ast.dump(ast.parse('physical_rotation = None').body[0])
    expected_guard = ast.dump(ast.parse(
        "assert physical_rotation is not None, 'The fixed nine-node physical clock is nonempty'").body[0])
    initializers = [i for i, node in enumerate(native.body) if ast.dump(node) == expected_init]
    guards = [i for i, node in enumerate(native.body) if ast.dump(node) == expected_guard]
    iterator = ast.dump(ast.parse('np.linspace(0., 6., 9)', mode='eval').body)
    loops = [(i, node) for i, node in enumerate(native.body)
             if isinstance(node, ast.For) and ast.dump(node.iter) == iterator]
    require(len(initializers) == len(guards) == len(loops) == 1 and
            initializers[0]+1 == loops[0][0] and guards[0] == loops[0][0]+1,
            'ONLY approved initialization/guard flank the fixed nine-node loop')
    loop = loops[0][1]
    binding = ast.dump(ast.parse(
        "current_word, field_word, physical_rotation = pulse['core'], pulse['field'], pulse['Phi_rotation']").body[0])
    require(any(ast.dump(node) == binding for node in loop.body) and
            not any(isinstance(node, (ast.Break, ast.Continue, ast.Return)) for node in ast.walk(loop)),
            'nonempty nine-node loop unconditionally binds the physical rotation')
    native.body = [node for i, node in enumerate(native.body) if i not in initializers+guards]
    annotations = [('cartesian_source_group', 'resolved_model', 'NativeOriginalModel'),
                   ('_reference_controls', 'generators', 'np.ndarray')]
    for function, target, annotation in annotations:
        body = functions[function].body
        matches = [i for i, node in enumerate(body) if isinstance(node, ast.AnnAssign) and
                   isinstance(node.target, ast.Name) and node.target.id == target and
                   ast.dump(node.annotation) == ast.dump(ast.parse(annotation, mode='eval').body)]
        require(len(matches) == 1, 'ONLY approved concrete local assignment annotation')
        node = cast(ast.AnnAssign, body[matches[0]])
        value = node.value
        assert value is not None, 'local type refinement retains its executable assignment'
        body[matches[0]] = ast.Assign(targets=[node.target], value=value)
    class ResolveModelName(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == 'resolved_model':
                node.id = 'model'
            return node
    ResolveModelName().visit(functions['cartesian_source_group'])
    old_encoding = ast.dump(old_tree, include_attributes=False)
    current_encoding = ast.dump(current_tree, include_attributes=False)
    require(current_encoding == old_encoding, 'restricted approved successor AST equivalence')
    digest = hashlib.sha256(old_encoding.encode()).hexdigest()
    require(digest == witness['normalization']['original_normalized_runtime_AST_sha256'] ==
            witness['normalization']['successor_after_approved_normalization_AST_sha256'],
            'independently recomputed restricted AST digest')
    lsp_path = local_artifact(witness['postfix_real_LSP'])
    lsp = json.loads(lsp_path.read_text())
    require(lsp['source_sha256'] == witness['corrected_source_sha256'] and
            lsp['observed_exit'] == 0 and not lsp['diagnostics']['result']['items'],
            'actual source-bound postfix real LSP zero diagnostics')
    return dict(correction_sha256=sha(correction_path), original_source_sha256=sha(old_path),
        current_source_sha256=witness['corrected_source_sha256'], restricted_AST_sha256=digest,
        original_scientific_packet_and_receipt_unchanged=True,
        approved_nonempty_nine_node_guard_proved=True, blanket_type_only_AST_identity=False,
        postfix_real_LSP_sha256=sha(lsp_path))


def joined_supplier_consumer(paths):
    """Final current source bytes + exact ledger; no old scientific leaf rerun."""
    from noncompact_scattering_flight import decimal_bound
    from bosonic_original_law_join import gauss_trial_vacuum_upper_contract
    from spatial_noether_identification import consumed_api_sha256
    packets = {key: json.loads(paths[key].read_text()) for key in (
        'root', 'physical', 'band_metadata', 'em_addendum', 'compton', 'original',
        'noether', 'cartesian', 'cartesian_receipt', 'spatial', 'spatial_receipt', 'prepared_source')}
    root, physical = packets['root'], packets['physical']
    metadata, addendum = packets['band_metadata'], packets['em_addendum']
    compton, original = packets['compton'], packets['original']
    noether = packets['noether']
    cartesian = packets['cartesian']
    cartesian_receipt = packets['cartesian_receipt']
    spatial_receipt = packets['spatial_receipt']
    spatial = packets['spatial']
    cartesian_successor = verify_cartesian_successor(paths, cartesian)
    for label, mapping in (
            ('Root', root['source_hashes']), ('physical', physical['source_sha256']),
            ('corrected_band', metadata['published_source_sha256']),
            ('Compton', compton['owned_source_pair']),
            ('Noether', noether['owned_source_pair_sha256']),
            ('Noether_dependencies', noether['observed_direct_dependency_sha256']),
            ('physical_spatial_Noether', spatial['owned_source_pair_sha256'])):
        for name, digest in mapping.items():
            require(sha(ROOT/name) == digest, f'final {label} current source bytes: {name}')
    require(sha(paths['root']) == addendum['Root_owner_topic_sha256'], 'EM addendum actual Root packet')
    require(sha(paths['band_metadata']) == addendum['supplier_metadata_receipt_sha256'],
            'EM addendum actual corrected metadata receipt')
    require(sha(paths['band']) == addendum['supplier_packet_sha256'], 'EM addendum actual band packet')
    require(sha(paths['em_addendum']) == compton['root_EM_qualification']['sha256'],
            'same completed Compton actual EM addendum')
    require(sha(paths['band_metadata']) == compton['inputs']['primitive_supplier_receipt_sha256'],
            'same completed Compton corrected heavy/physical metric')
    require(sha(paths['noether_receipt']) == noether['receipt']['sha256'] and
            noether['receipt']['natural_exit'] == 0, 'source-bound new Noether natural receipt')
    require(sha(paths['cartesian_receipt']) == cartesian['receipt']['sha256'] and
            cartesian['receipt']['natural_exit'] == 0, 'actual new Cartesian quantum group natural receipt')
    for name, digest in cartesian_receipt['input_sha256'].items():
        require(sha(Path(name)) == digest, 'Cartesian group SAME frozen source/Noether input')
    require(cartesian_receipt['source_sha256'] == cartesian['owned_source_pair_sha256'],
            'Cartesian quantum group exact ORIGINAL executed source pair, not rewritten history')
    require(sha(paths['spatial_receipt']) == spatial['receipt']['sha256'] and
            spatial['receipt']['natural_exit'] == 0,
            'new physical spatial/current identification natural source-bound receipt')
    require(spatial_receipt['source_pair_sha256'] == spatial['owned_source_pair_sha256'] and
            spatial_receipt['consumed_api_sha256'] == consumed_api_sha256(),
            'same current physical identification source pair and exact consumed shared APIs')
    for key in ('descriptor', 'cartesian', 'cartesian_receipt', 'cartesian_correction', 'band_metadata'):
        matches = [digest for name, digest in spatial_receipt['input_sha256'].items()
                   if Path(name).resolve() == paths[key].resolve()]
        require(matches == [sha(paths[key])], 'physical identification SAME source/field/current input: '+key)
    group_reserve = int(cartesian_receipt['transfer']['q6_four_segment_clock_reserve_bits'])
    require(group_reserve == 54, 'independent actual three-axis quantum word reserve')
    descriptor_digest = sha(paths['descriptor'])
    require(noether['selected_original']['actual_geometry_descriptor_sha256'] == descriptor_digest,
            'Noether original same literal nineteen compact sites')
    require(original['input_sha256']['issue222-legal-volume-far-RIGHT-global-Gram-b4096-descriptor.json'] == descriptor_digest,
            'original103 same fixed SC19 receiver source')
    require(compton['inputs']['source_geometry_descriptor_sha256'] == descriptor_digest,
            'improved108 same fixed SC19 receiver source')
    require(int(original['selected_family']['N']) == NSTAR and int(compton['unchanged']['N']) == NSTAR and
            metadata['N'] == NSTAR and noether['selected_original']['N'] == NSTAR, 'exact common legal N')
    require(F(metadata['UE']) == UE and F(compton['unchanged']['UE']) == UE,
            'exact common microscopic UE, not native witness UE')
    require(F(metadata['c']) == C and F(compton['unchanged']['c']) == C,
            'exact common microscopic c')
    require(metadata['bounds']['Rootpenalty']['actual_code_vacuum_upper']['U_trial'] ==
            gauss_trial_vacuum_upper_contract()['U_trial'],
            'same admitted full-Gauss Gaussian trial from final Root API, not a code-only vacuum')
    aggregate = F(2)**-33025+F(35, 128)*F(2)**-66048+F(2)**-46687
    require(aggregate < F(2)**-33024, 'actual Phi3/4 plus coexactEM1/2 aggregate counted once')
    target = UE/(NSTAR**3*2**40)
    original_angular = int(noether['selected_original']['moving_source_current_secondword_transfer_log2_upper'])
    improved_angular = int(noether['selected_improved_independent']['new_moving_source_current_word_transfer_log2_upper'])
    require(original_angular == -5120 and improved_angular == -5484,
            'independent same-law moving SOURCE error suppliers')
    driven = driven_source_contract(paths, compton, metadata, packets['prepared_source'])
    reserve = driven['paid_dynamic_reserve_bits']+group_reserve
    driven.update(quantum_group_clock_reserve_bits=group_reserve,
        total_joined_dynamic_reserve_bits=reserve,
        total_actual_amplification_log2_upper=driven['actual_dynamic_amplification_log2_upper']+group_reserve,
        unused_joined_reserve_bits=reserve-driven['actual_dynamic_amplification_log2_upper']-group_reserve)
    original_angular += reserve
    improved_angular += reserve
    # Moving-word amplification already contains the full SOURCE/current/photon
    # payer: do not add that same payer again. Fixed static Born ledgers stay put.
    original_error = F(2)**original_angular
    improved_error = (F(2)**improved_angular+F(2)**(-33024+reserve)+
                      UE/(NSTAR**3)*(F(2)**(-648+reserve)+F(2)**(-198381+reserve)))
    require(original_error < target and improved_error < target,
            'joined original and improved SOURCE/word/body/Root/phase errors below exact target')
    identification = spatial_receipt['geometric_identification']
    require(identification['geometric_normal_stress_reserve_bits'] == 1024 and
            identification['additional_Compton_source_word_reserve_bits'] == 0,
            'physical spatial identification pays only its separate normal/stress reserve')
    require(identification['original_and_improved_amplitude_words_unchanged'] ==
            [original_angular, improved_angular],
            'physical Noether identification does not silently amplify the SOURCE word')
    normal_log2 = -33024+reserve+1024
    em_log2 = -47349+reserve+1024
    require(identification['original_and_improved_normal_log2_upper'] == normal_log2 and
            identification['coexact_EM_image_log2_upper'] == em_log2,
            'same ordinary Root normal and coexact EM image under the geometric stress map')
    require(F(2)**normal_log2 < target and F(2)**em_log2 < target,
            'separately qualified physical normal/EM errors below exact common target')
    identification_composition = dict(
        normal_stress_reserve_bits=1024, additional_SOURCE_word_bits=0,
        normal_log2_upper=normal_log2, coexact_EM_image_log2_upper=em_log2,
        normal_decimal_upper=decimal_bound(F(2)**normal_log2),
        coexact_EM_decimal_upper=decimal_bound(F(2)**em_log2),
        comparisons_pass=True, Root_Phi_EM_aggregate_not_added_twice=True,
        errors_are_field_identification_not_fixed_detector_probability=True,
        lowk_operator_projector_phase_errors_not_scattering_amplitude=True,
        weak_spatial_clock_distinct_from_T6=True)
    return dict(source_and_input_bindings_checked=True,
        driven_source_graph_contract=driven,
        physical_spatial_identification_composition=identification_composition,
        cartesian_authorized_successor=cartesian_successor,
        final_frozen_source_pair={name: sha(ROOT/name) for name in (
            'bosonic_original_law_join.py', 'bosonic_original_law_join.md',
            'bosonic_physical_transfer.py', 'bosonic_physical_transfer.md',
            'improved_phi_band.py', 'improved_phi_band.md',
            'improved_physical_compton.py', 'improved_physical_compton.md',
            'moving_physical_noether.py', 'moving_physical_noether.md',
            'cartesian_noether_group.py', 'cartesian_noether_group.md',
            'spatial_noether_identification.py', 'spatial_noether_identification.md')},
        original103=dict(law='original bare nearest Wilson heavy/light branch retained',
            published_fixed_receiver_probability_lower=original['exact_composition']['probability_decimal_lower'],
            underlying_static_source_and_phase_payer_log2=-8370,
            moving_source_word_payer_log2=original_angular,
            SOURCE_payer_already_inside_word_counted_once=True,
            joined_source_word_error_decimal_upper=decimal_bound(original_error),
            rest_g2_claimed=False),
        improved108=dict(law='original heavy/scalar + actual range2 light Phi',
            published_fixed_receiver_probability_lower=compton['actual_physical_probability']['probability_lower'],
            underlying_static_source_photon_payer_log2=-8746,
            underlying_static_body_volume_rescaled_payer_log2=-648,
            driven_body_volume_rescaled_payer_log2=-648+reserve,
            underlying_static_root_Phi_EM_aggregate_payer_log2=-33024,
            driven_root_Phi_EM_aggregate_payer_log2=-33024+reserve,
            underlying_static_paired_phase_volume_rescaled_payer_log2=-198381,
            driven_paired_phase_volume_rescaled_payer_log2=-198381+reserve,
            SOURCE_payer_already_inside_word_counted_once=True,
            moving_source_word_payer_log2=improved_angular,
            joined_source_word_error_decimal_upper=decimal_bound(improved_error),
            rest_g=2, common_c=str(C), physical_mass_lower=metadata['calibration']['legal_mass_gap_lower'],
            no_original_probability_copied=True),
        exact_joint_target_decimal=decimal_bound(target, False), both_exact_joint_comparisons_pass=True,
        geometry=dict(SC19_count=19, actual_word='D+2', eta=18, RAW41_count_upper=68921,
            RAW_word='D+60', auxiliary_flat_structures=8, colour_wells=64, source_relabelled=False),
        coordinates=dict(Phi_weighted_power='3/4', coexact_EM_weighted_power='1/2',
            harmonic_zero_frequency='separate balanced Bloch bundle, no oscillator coercivity',
            kinetic_Gram_owner='unchanged original heavy/scalar', source_and_receiver_Grams_are_not_kinetic=True),
        phase_and_error_semantics=dict(intrinsic_scalar_Casimir_correction=noether['selected_original']['source_valued_intrinsic_Casimir'],
            dimensionless_scalar_curvature_is_not_added_to_scattering_amplitude=True,
            calibration_cone_debt=metadata['calibration']['continuum_cone_calibration_debt_upper_float'],
            fixed_scattering_probability_not_redefined_by_new_transport_control=True,
            scalar_and_Noether_control_errors_are_native_observation_errors_not_old_detector_errors=True,
            ordinary_E0_kappa_and_all64_retained=True),
        noether_actual_native=noether['new_native_observations'],
        quantum_cartesian_group_native=cartesian['new_native_observations'],
        physical_spatial_supplier=spatial_receipt,
        original_selected_word=noether['selected_original'],
        improved_selected_word=noether['selected_improved_independent'],
        scope='finite native new moving physical contour + source-bound conditional large-family word composition, not astronomical numerical ground simulation',
        remaining_gaps=['Uniform full-field phase/normal and quantum-law hypotheses remain explicit premises, not inferred from finite samples.',
            'Chosen microscopic coefficients/penalty hierarchy are not derived uniquely from an underlying medium.',
            'Hard Compton q=2pi/5 is outside .05 continuum control; no exact all-momentum Lorentz statement.',
            'Ordinary total angular momentum of an arbitrary packet includes orbital/ref/vacuum noise and is not forced to spin1/2.'],
        corrective_successor='Preserve the verified source19/current/ref/field-Noether worldline and derive microscopic coefficient/phase hypotheses on the underlying dynamical construction; do not reopen fixed scattering suppliers or erase boundaries.')


def terminal_radial_receipt_consumer(paths: dict[str, Path]) -> dict[str, Any]:
    """Join independent RAW science without re-preparing the completed SC19 carrier."""
    from radial_band_noether import consumed_api_sha256, radial_receipt_contract
    input_before = {key: sha(path) for key, path in paths.items()}
    sc19 = json.loads(paths['sc19_receipt'].read_text())
    wrapper = json.loads(paths['sc19_wrapper'].read_text())
    raw_packet = json.loads(paths['radial_packet'].read_text())
    raw_receipt = json.loads(paths['radial_receipt'].read_text())
    source_before = {name: sha(ROOT/name) for name in sc19['source_hashes']}
    source_before.update({name: sha(ROOT/name) for name in raw_packet['owned_source_pair_sha256']})
    require(wrapper['natural_exit'] == 0 and wrapper['completion_status'] == 'completed' and
            sha(paths['sc19_receipt']) == wrapper['receipt']['sha256'] and
            sc19['source_unchanged_during_run'] and sc19['all_checks_pass'],
            'actual naturally completed immutable SC19 scientific execution')
    def capsule(entry):
        uri = entry['uri']
        require(uri.startswith('local://') and Path(uri[8:]).name == uri[8:],
                'explicit executed SC19 local capsule')
        path = paths['sc19_wrapper'].parent/uri[8:]
        require(sha(path) == entry['sha256'], 'exact executed SC19 capsule bytes')
        paths['sc19_capsule_'+path.suffix.lstrip('.')] = path
        input_before['sc19_capsule_'+path.suffix.lstrip('.')] = entry['sha256']
        return path
    old_source = capsule(wrapper['executed_source_capsule'])
    old_document = capsule(wrapper['executed_document_capsule'])
    require(sha(old_source) == sc19['source_hashes']['physical_carrier_transport.py'] and
            sha(old_document) == sc19['source_hashes']['physical_carrier_transport.md'],
            'honest distinction between executed SC19 and current terminal consumer source')
    old_tree = ast.parse(old_source.read_text())
    current_tree = ast.parse((ROOT/'physical_carrier_transport.py').read_text())
    dispatcher = ast.parse(
        "if __name__ == '__main__':\n"
        "    import sys\n"
        "    if sys.argv[1:2] == ['join-radial']:\n"
        "        terminal_radial_main(sys.argv[2:])\n"
        "    else:\n"
        "        main()\n").body[0]
    require(ast.dump(current_tree.body[-1]) == ast.dump(dispatcher),
            'ONLY explicit terminal subcommand dispatch replaces the original footer')
    additions = {'terminal_radial_receipt_consumer', 'terminal_radial_main'}
    current_tree.body = [node for node in current_tree.body[:-1]
                         if not (isinstance(node, ast.FunctionDef) and node.name in additions)]
    old_tree.body = old_tree.body[:-1]
    old_ast = ast.dump(old_tree, include_attributes=False)
    require(ast.dump(current_tree, include_attributes=False) == old_ast,
            'entire executed SC19 physics/import/constants/main AST unchanged')
    require((ROOT/'physical_carrier_transport.md').read_text().startswith(old_document.read_text()),
            'executed scientific documentation preserved; RAW composition is explicitly additive')
    for name, digest in sc19['source_hashes'].items():
        if name not in ('physical_carrier_transport.py', 'physical_carrier_transport.md'):
            require(sha(ROOT/name) == digest, 'unchanged SC19 frozen supplier source: '+name)
    sc19_inputs: dict[str, Path] = {}
    for key, digest in sc19['bindings']['input_sha256'].items():
        # Keys are semantic input names, not paths; exact paths come from the
        # executed argv or the immutable linked supplier receipts below.
        require(wrapper['input_sha256'][key] == digest, 'executed SC19 wrapper input ledger')
    flags = {'--descriptor': 'descriptor', '--compton-packet': 'compton',
             '--band-packet': 'band', '--root-packet': 'root',
             '--physical-packet': 'physical', '--band-metadata': 'band_metadata',
             '--em-addendum': 'em_addendum', '--original-packet': 'original',
             '--noether-packet': 'noether', '--noether-receipt': 'noether_receipt',
             '--cartesian-packet': 'cartesian', '--cartesian-receipt': 'cartesian_receipt',
             '--cartesian-correction': 'cartesian_correction', '--spatial-packet': 'spatial',
             '--spatial-receipt': 'spatial_receipt'}
    argv = wrapper['argv']
    for flag, key in flags.items():
        index = argv.index(flag)
        sc19_inputs[key] = Path(argv[index+1])
        require(sha(sc19_inputs[key]) == sc19['bindings']['input_sha256'][key],
                'terminal consumer actual frozen SC19 input: '+key)
    sc19_inputs['prepared_source'] = Path(next(name for name in
        json.loads(sc19_inputs['noether'].read_text())['input_sha256']
        if Path(name).name == 'issue222-original-source-flight-receipt.json'))
    correction = json.loads(sc19_inputs['cartesian_correction'].read_text())
    for key in ('original_executed_source', 'postfix_real_LSP'):
        sc19_inputs['cartesian_'+key] = sc19_inputs['cartesian_correction'].parent/correction[key]['uri'][8:]
    for i, name in enumerate(json.loads(sc19_inputs['spatial_receipt'].read_text())['input_sha256']):
        path = Path(name)
        if path not in sc19_inputs.values():
            sc19_inputs[f'spatial_dependency_{i}'] = path
    require({key: sha(path) for key, path in sc19_inputs.items()} ==
            sc19['bindings']['input_sha256'], 'ALL nineteen executed SC19 input dependencies remain exact')
    for key, path in sc19_inputs.items():
        paths['sc19_input_'+key] = path
        input_before['sc19_input_'+key] = sc19['bindings']['input_sha256'][key]
    sc19_lsp = json.loads(paths['sc19_lsp'].read_text())
    require(sc19_lsp['stable_diagnostics_total'] == 0 and
            sc19_lsp['source_sha256']['physical_carrier_transport.py'] == sha(old_source) and
            all(not row['diagnostics']['result']['items'] and row['exit'] == 0
                for row in sc19_lsp['files']), 'executed SC19 source-bound actual real LSP evidence')
    renewed_join = joined_supplier_consumer(sc19_inputs)
    require(renewed_join == sc19['joined_consumer'], 'current exact SC19 analytic ledger unchanged')
    require(raw_packet['scientific_receipt']['natural_exit'] == 0 and
            sha(paths['radial_receipt']) == raw_packet['scientific_receipt']['sha256'],
            'independent actual RAW preparation naturally completed')
    require(raw_receipt['source_pair_sha256'] == raw_packet['executed_source_pair_sha256'] and
            raw_receipt['consumed_api_sha256'] == consumed_api_sha256(),
            'executed independent RAW law and unchanged current consumed physical APIs')
    for name, digest in raw_packet['owned_source_pair_sha256'].items():
        require(sha(ROOT/name) == digest, 'current RAW supplier source: '+name)
    def radial_dependency(key, entry):
        uri = entry['uri']
        require(uri.startswith('local://') and Path(uri[8:]).name == uri[8:],
                'explicit radial successor local dependency')
        path = paths['radial_packet'].parent/uri[8:]
        require(sha(path) == entry['sha256'], 'exact radial successor dependency bytes')
        paths[key] = path
        input_before[key] = entry['sha256']
        return path
    correction_path = radial_dependency('radial_typing_correction', raw_packet['typing_correction'])
    radial_correction = json.loads(correction_path.read_text())
    executed_radial = radial_dependency('radial_executed_source', radial_correction['executed_source'])
    current_pair = raw_packet['owned_source_pair_sha256']
    require(radial_correction['executed_source_pair_sha256'] == raw_receipt['source_pair_sha256'] and
            radial_correction['current_source_pair_sha256'] == current_pair and
            sha(executed_radial) == raw_receipt['source_pair_sha256']['radial_band_noether.py'] and
            current_pair['radial_band_noether.md'] == raw_receipt['source_pair_sha256']['radial_band_noether.md'],
            'honest executed/current radial pair distinction with unchanged scientific documentation')
    radial_old_tree = ast.parse(executed_radial.read_text())
    radial_current_tree = ast.parse((ROOT/'radial_band_noether.py').read_text())
    imports = [node for node in radial_current_tree.body
               if isinstance(node, ast.ImportFrom) and node.module == 'typing']
    require(len(imports) == 1 and [(alias.name, alias.asname) for alias in imports[0].names] ==
            [('Any', None), ('cast', None)], 'ONLY exact additional typing.cast import')
    imports[0].names = [imports[0].names[0]]
    cast_targets = {
        'native_radial_source_chart': ('tuple[int, int]', 'H.shape'),
        'raw_graph_certificate': ('F', 'sum(terms.values())')}
    for function, (type_expression, value_expression) in cast_targets.items():
        functions = [node for node in radial_current_tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == function]
        require(len(functions) == 1, 'unique scoped radial identity-cast function')
        expected_type = ast.dump(ast.parse(type_expression, mode='eval').body)
        expected_value = ast.dump(ast.parse(value_expression, mode='eval').body)
        class EraseExactIdentityCast(ast.NodeTransformer):
            count = 0
            def visit_Call(self, node):
                if (isinstance(node.func, ast.Name) and node.func.id == 'cast' and
                        len(node.args) == 2 and not node.keywords and
                        ast.dump(node.args[0]) == expected_type and ast.dump(node.args[1]) == expected_value):
                    self.count += 1
                    return node.args[1]
                return self.generic_visit(node)
        eraser = EraseExactIdentityCast()
        eraser.visit(functions[0])
        require(eraser.count == 1, 'ONLY one exact runtime-identity cast per authorized radial function')
    radial_old_ast = ast.dump(radial_old_tree, include_attributes=False)
    radial_ast_sha = hashlib.sha256(radial_old_ast.encode()).hexdigest()
    require(ast.dump(radial_current_tree, include_attributes=False) == radial_old_ast and
            radial_ast_sha == radial_correction['normalization']['executed_normalized_AST_sha256'] ==
            radial_correction['normalization']['current_normalized_AST_sha256'],
            'independently recomputed WHOLE radial module restricted AST equality')
    import inspect
    import typing
    cast_source = inspect.getsource(typing.cast)
    cast_function = ast.parse(cast_source).body[0]
    if not isinstance(cast_function, ast.FunctionDef):
        raise AssertionError('stdlib typing.cast source must be the actual function')
    cast_body = cast_function.body[1:]
    require(hashlib.sha256(cast_source.encode()).hexdigest() ==
            radial_correction['runtime_identity']['stdlib_cast_sha256'] and
            len(cast_body) == 1 and ast.dump(cast_body[0]) == "Return(value=Name(id='val', ctx=Load()))",
            'actual stdlib typing.cast has exactly unchanged-value runtime semantics')
    radial_lsp_path = radial_dependency('radial_current_LSP', radial_correction['final_real_LSP'])
    radial_lsp = json.loads(radial_lsp_path.read_text())
    require(radial_lsp['natural_exit'] == 0 and radial_lsp['source_pair_sha256'] == current_pair and
            not radial_lsp['diagnostics']['result']['items'],
            'actual current corrected radial source-bound real LSP zero diagnostics')
    radial_successor = dict(correction_sha256=sha(correction_path),
        executed_source_sha256=sha(executed_radial), current_source_pair_sha256=current_pair,
        independently_recomputed_whole_AST_sha256=radial_ast_sha,
        exact_two_cast_runtime_identity_proved=True,
        current_corrected_bytes_claimed_scientifically_executed=False,
        healthy_native_science_repeated=False)
    for i, (name, digest) in enumerate(raw_receipt['input_sha256'].items()):
        path = Path(name)
        require(sha(path) == digest, 'RAW supplier actual input bytes')
        if path not in paths.values():
            paths[f'raw_dependency_{i}'] = path
            input_before[f'raw_dependency_{i}'] = digest
    raw_contract = radial_receipt_contract(
        raw_receipt, decode(json.loads(sc19_inputs['descriptor'].read_text())),
        json.loads(sc19_inputs['band_metadata'].read_text()))
    soft_reserve = int(raw_contract['SOFT_normal_stress_added_bits'])
    require(raw_contract['joined_root_log2'] == -31850+soft_reserve and
            raw_contract['contained_EM_log2'] == -46175+soft_reserve and
            raw_contract['additional_Compton_word_bits'] == 0,
            'independent SOFT normal stress amplifies errors; it does not borrow SC19 word bits')
    target = UE/(NSTAR**3*2**40)
    root_error = F(2)**int(raw_contract['joined_root_log2'])
    em_error = F(2)**int(raw_contract['contained_EM_log2'])
    require(root_error < target and em_error < target,
            'separately qualified RAW/SOFT field identification errors below exact target')
    from noncompact_scattering_flight import decimal_bound
    radial_identification = dict(
        qualified_root_log2_upper=raw_contract['joined_root_log2'],
        contained_EM_log2_upper=raw_contract['contained_EM_log2'],
        qualified_root_decimal_upper=decimal_bound(root_error),
        contained_EM_decimal_upper=decimal_bound(em_error),
        common_exact_target_decimal=decimal_bound(target, False),
        exact_comparisons_pass=True, contained_EM_not_paid_twice=True,
        not_a_new_fixed_detector_probability=True,
        lowk_operator_projector_phase_errors_are_not_amplitude_debts=True)
    return dict(executed_SC19_receipt_sha256=sha(paths['sc19_receipt']),
        executed_SC19_source_sha256=sha(old_source),
        current_consumer_source_sha256=sha(ROOT/'physical_carrier_transport.py'),
        unchanged_SC19_physics_AST_sha256=hashlib.sha256(old_ast.encode()).hexdigest(),
        independent_RAW_receipt_sha256=sha(paths['radial_receipt']),
        independent_RAW_source_pair=raw_packet['owned_source_pair_sha256'],
        radial_authorized_identity_cast_successor=radial_successor,
        actual_RAW_contract=raw_contract, preserved_SC19_join=renewed_join,
        radial_field_identification=radial_identification,
        input_sha256_before=input_before, source_sha256_before=source_before,
        SC19_fixed_TWO_outputs_and_probability_unchanged=True,
        RAW41_not_relabelled_SC19_and_no_new_filter=True,
        completed_SC19_native_science_not_duplicated=True)


def terminal_radial_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description='Independent RAW + completed SC19 physical receipt consumer')
    for flag in ('sc19-receipt', 'sc19-wrapper', 'sc19-lsp', 'radial-packet', 'radial-receipt'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    paths = dict(sc19_receipt=args.sc19_receipt, sc19_wrapper=args.sc19_wrapper,
        sc19_lsp=args.sc19_lsp, radial_packet=args.radial_packet, radial_receipt=args.radial_receipt)
    started = time.monotonic()
    joined = terminal_radial_receipt_consumer(paths)
    before = joined.pop('input_sha256_before')
    sources = joined.pop('source_sha256_before')
    require(before == {key: sha(path) for key, path in paths.items()} and
            all(sha(ROOT/name) == digest for name, digest in sources.items()),
            'all terminal joined input and current source bytes remain unchanged')
    result = dict(joined_consumer=joined, source_hashes=sources, input_sha256=before,
        elapsed_seconds=time.monotonic()-started, all_checks_pass=True,
        scope='actual independent RAW science plus immutable naturally executed SC19 physical science; analytic dependencies stay conditional')
    args.output.write_text(json.dumps(result, default=encode, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), elapsed_seconds=result['elapsed_seconds'],
                         SC19_not_duplicated=True, RAW_independent=True, all_checks_pass=True), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--descriptor', type=Path, required=True)
    parser.add_argument('--compton-packet', type=Path, required=True)
    parser.add_argument('--band-packet', type=Path, required=True)
    parser.add_argument('--root-packet', type=Path, required=True)
    parser.add_argument('--physical-packet', type=Path, required=True)
    parser.add_argument('--band-metadata', type=Path, required=True)
    parser.add_argument('--em-addendum', type=Path, required=True)
    parser.add_argument('--original-packet', type=Path, required=True)
    parser.add_argument('--noether-packet', type=Path, required=True)
    parser.add_argument('--noether-receipt', type=Path, required=True)
    parser.add_argument('--cartesian-packet', type=Path, required=True)
    parser.add_argument('--cartesian-receipt', type=Path, required=True)
    parser.add_argument('--cartesian-correction', type=Path, required=True)
    parser.add_argument('--spatial-packet', type=Path, required=True)
    parser.add_argument('--spatial-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    paths = dict(descriptor=args.descriptor, compton=args.compton_packet,
        band=args.band_packet, root=args.root_packet, physical=args.physical_packet,
        band_metadata=args.band_metadata, em_addendum=args.em_addendum,
        original=args.original_packet, noether=args.noether_packet,
        noether_receipt=args.noether_receipt, cartesian=args.cartesian_packet,
        cartesian_receipt=args.cartesian_receipt, cartesian_correction=args.cartesian_correction,
        spatial=args.spatial_packet, spatial_receipt=args.spatial_receipt)
    noether_inputs = json.loads(paths['noether'].read_text())['input_sha256']
    prepared_name = next(name for name in noether_inputs if Path(name).name == 'issue222-original-source-flight-receipt.json')
    paths['prepared_source'] = Path(prepared_name)
    require(sha(paths['prepared_source']) == noether_inputs[prepared_name],
            'actual original prepared SOURCE receipt is byte-bound by frozen Noether supplier')
    correction = json.loads(paths['cartesian_correction'].read_text())
    for key in ('original_executed_source', 'postfix_real_LSP'):
        entry = correction[key]
        require(entry['uri'].startswith('local://') and
                Path(entry['uri'][8:]).name == entry['uri'][8:],
                'explicit successor dependency local basename')
        paths['cartesian_'+key] = paths['cartesian_correction'].parent/entry['uri'][8:]
        require(sha(paths['cartesian_'+key]) == entry['sha256'],
                'successor dependency bytes included in whole-run binding')
    spatial_inputs = json.loads(paths['spatial_receipt'].read_text())['input_sha256']
    for i, (name, digest) in enumerate(spatial_inputs.items()):
        path = Path(name)
        require(sha(path) == digest, 'physical spatial supplier input bytes')
        if path not in paths.values():
            paths[f'spatial_dependency_{i}'] = path
    descriptor = decode(json.loads(args.descriptor.read_text()))
    bindings = bind_packets(paths, descriptor)
    joined = joined_supplier_consumer(paths)
    source_names = ('physical_carrier_transport.py', 'physical_carrier_transport.md',
        'moving_physical_noether.py', 'moving_physical_noether.md', 'improved_phi_band.py',
        'improved_phi_band.md', 'improved_physical_compton.py', 'improved_physical_compton.md',
        'bosonic_original_law_join.py', 'bosonic_original_law_join.md',
        'bosonic_physical_transfer.py', 'bosonic_physical_transfer.md', 'bosonic_statistics.py',
        'cartesian_noether_group.py', 'cartesian_noether_group.md',
        'noncompact_original_source_flight.py', 'noncompact_quantitative_normal_budget.py',
        'noncompact_scattering_flight.py', 'spatial_noether_identification.py',
        'spatial_noether_identification.md')
    before = {name: sha(ROOT/name) for name in source_names}
    band, frame, calibration = selected_band()
    intrinsic_receipt = json.loads(paths['cartesian_receipt'].read_text())
    result: dict[str, Any] = dict(topic='same-law primitive finite-energy physical mobile charged/ref/cloud contour',
                  bindings=bindings, joined_consumer=joined, calibration=calibration,
                  charged=native_contour(frame, band, descriptor, 1, intrinsic_receipt),
                  anticharged=native_contour(frame, band, descriptor, -1, intrinsic_receipt),
                  controlled_spacetime=lowk_worldline(frame, band),
                  same_mobile_SC19_photon_current=mobile_sc19_current_packet(frame, band, descriptor),
                  exact_family=dict(N=NSTAR, UE=str(UE), c=str(C), K='c^2/(2 UE)',
                                    source19=True, reference_word='D+2', eta=18, alpha='2^371'),
                  elapsed_seconds=time.monotonic()-start,
                  source_hashes=before,
                  limits=['Native representable UE/chi slice is not the huge selected coupled neutral ground.',
                          'Analytic phase/normal/Root and microscopic statistics hypotheses remain owner-supplied.',
                          'Controlled .05 spacetime estimate never covers hard Compton q=2pi/5.'],
                  all_checks_pass=True)
    require(before == {name: sha(ROOT/name) for name in source_names},
            'entire final joined supplier/source set unchanged during native execution')
    require(bindings['input_sha256'] == {name: sha(path) for name, path in paths.items()},
            'all exact joined input artifacts unchanged during native execution')
    result['source_unchanged_during_run'] = True
    args.output.write_text(json.dumps(result, default=encode, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), elapsed_seconds=result['elapsed_seconds'],
                         charged_movement=result['charged']['density_movement'],
                         charged_recovery=result['charged']['density_recovery_error'],
                         anticharged_movement=result['anticharged']['density_movement'],
                         anticharged_recovery=result['anticharged']['density_recovery_error'],
                         all_checks_pass=True), indent=2))


if __name__ == '__main__':
    import sys
    if sys.argv[1:2] == ['join-radial']:
        terminal_radial_main(sys.argv[2:])
    else:
        main()
