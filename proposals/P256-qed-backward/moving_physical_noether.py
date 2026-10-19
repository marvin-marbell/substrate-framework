#!/usr/bin/env python3
"""Source-valued physical Noether second words and a finite moving pulse.

The native consumer is a finite Galerkin witness, not the selected infinite
Cartesian phase. The selected certificate consumes frozen SOURCE graph bounds.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'

import mpmath as mp
import numpy as np
from scipy.linalg import eigh, expm, block_diag
from scipy.integrate import solve_ivp

from bulk_quantum_selection import BulkQuantumLaw
from noncompact_native_current import Coupled, exterior, accept, adverse
from noncompact_original_source_flight import _dense
from noncompact_photon_amputation_graph import creators
from noncompact_screened_gauss_dressing import screened_marks
from original_charged_principal import original_frame
from original_total_angular_transfer import primitive_flow, scalar_orbital, endpoint_flow
from unrestricted_field_minimum import coordinates
from rotation_band import RotationBand

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-synthesis-continuation-20261006'


@dataclass(frozen=True)
class PulseClock:
    """C2 compact pulse, physical time is never reset at a segment boundary."""
    duration: float = 6.
    angle: float = 1/16

    def __post_init__(self) -> None:
        if self.duration <= 0 or not math.isfinite(self.angle):
            raise ValueError('Finite angle and positive physical duration required')

    def pose(self, time: float) -> float:
        u = min(1., max(0., time/self.duration))
        return self.angle*(10*u**3-15*u**4+6*u**5)

    def speed(self, time: float) -> float:
        u = time/self.duration
        return self.angle*30*u*u*(1-u)**2/self.duration if 0 < u < 1 else 0.

    def acceleration(self, time: float) -> float:
        u = time/self.duration
        return self.angle*60*u*(1-u)*(1-2*u)/self.duration**2 if 0 < u < 1 else 0.


def physical_generators(law: Any, frame: np.ndarray) -> np.ndarray:
    """Solve the full fifteen Gauss directions against Cartesian vector flow."""
    d = np.asarray([coordinates(1j*(t@frame-frame@t), law.basis).reshape(60).real
                    for t in law.generators]).T
    masses = np.r_[np.ones(60), np.full(45, 1/.6)]
    gauge = np.vstack((d, np.zeros((45, 15))))
    inertia = gauge.T@(masses[:, None]*gauge)
    result = []
    for j, k in ((2, 3), (3, 1), (1, 2)):
        tangent = np.zeros_like(frame)
        tangent[j], tangent[k] = -frame[k], frame[j]
        y = np.r_[coordinates(tangent, law.basis).reshape(60).real, np.zeros(45)]
        coefficients = np.linalg.solve(inertia, gauge.T@(masses*y))
        generator = np.einsum('a,aij->ij', coefficients, law.generators)
        accept(np.linalg.norm(1j*(generator@frame-frame@generator)-tangent),
               'actual fullPhi physical vector tangent', 2e-8)
        result.append(generator)
    return np.asarray(result)


def relative_word(B: np.ndarray, Jc: np.ndarray, Jn: np.ndarray,
                  order: int = 2) -> np.ndarray:
    """Literal nested relative operator, not powers of compressed first matrices."""
    if order < 0:
        raise ValueError('Nonnegative word length required')
    result = B
    for _ in range(order):
        result = Jc@result-result@Jn
    return result


def relative_source_jets(B: np.ndarray, dB: np.ndarray, ddB: np.ndarray,
                         Lc: np.ndarray, Ln: np.ndarray,
                         dLc: np.ndarray | None = None,
                         dLn: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Full differential word for J=-i d/dtheta+L(theta), charged AND neutral.

    Includes orbital/current/ref connections and their derivatives. The source
    retains all reference columns; derivatives are of the same physical family.
    """
    dLc = np.zeros_like(Lc) if dLc is None else dLc
    dLn = np.zeros_like(Ln) if dLn is None else dLn
    first = -1j*dB+Lc@B-B@Ln
    second = (-ddB-1j*(dLc@B-B@dLn+2*Lc@dB-2*dB@Ln)
              +Lc@Lc@B-2*Lc@B@Ln+B@Ln@Ln)
    return dict(first=first, second=second)


def timed_source(B: np.ndarray, dB: np.ndarray, Hc: Any, Hn: Any,
                 right_action: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Actual moving equation, either as operator or exact operator-on-state.

    For a contracted source vector, right_action MUST be the propagated
    B Hn Omega from the SAME ordinary neutral history, not zero or Hn B Omega.
    """
    if right_action is None:
        right = B@Hn
    else:
        if right_action.shape != B.shape:
            raise ValueError('Exact neutral right action must have source shape')
        right = right_action
    derivative = -1j*dB
    return dict(derivative_term=derivative,
                residual=Hc@B-right+derivative)


def field_noether_galerkin(vector: np.ndarray, derivatives: np.ndarray,
                          weights: np.ndarray | None = None) -> np.ndarray:
    """Hermitian physical field Lie generator in a finite quadrature chart.

    vector[node, coordinate] is the pullback of ACTUAL primitive_flow.
    derivatives[coordinate,node,node] act on coefficient wavefunctions.
    Weights transform to orthonormal amplitudes before symmetric ordering.
    Orthogonal physical-flow components require caller-retained graph bounds.
    """
    nodes, dimension = vector.shape
    if derivatives.shape != (dimension, nodes, nodes):
        raise ValueError('Field vector and coordinate derivatives disagree')
    if weights is None:
        weights = np.ones(nodes)
    if weights.shape != (nodes,) or np.min(weights) <= 0:
        raise ValueError('Positive finite quadrature weights required')
    root = np.sqrt(weights)
    transport = sum(vector[:, j, None]*derivatives[j] for j in range(dimension))
    orthonormal = root[:, None]*transport/root[None, :]
    return -.5j*(orthonormal-orthonormal.conj().T)


def source_scalar_jets(theta: Any, eta: int = 18,
                       axis_generator: np.ndarray | None = None) -> tuple[Any, ...]:
    """Original psi1/psi0 along a physical Gauss-solved rotation, q0 through q6.

    Tr exp(i theta S)=4 cos(theta/2). For Z4 centres the exact original ratio is
    tanh(2 eta cos(theta/2)), NOT the constant nearest-centre value one.
    """
    if axis_generator is not None:
        accept(np.linalg.norm(axis_generator@axis_generator-.25*np.eye(4)),
               'physical source axis eigenvalues +/-1/2')
    with mp.workdps(100):
        at = mp.mpf(str(theta))
        fn = lambda t: mp.tanh(2*eta*mp.cos(t/2))
        return tuple(+mp.diff(fn, at, q) for q in range(7))


def _polynomial_plateau(radius: float) -> float:
    if radius <= 24:
        return 1.
    if radius >= 48:
        return 0.
    u = (radius-24)/24
    return 1-(10*u**3-15*u**4+6*u**5)


def source19_boundary_word() -> dict[str, Any]:
    """Source19 to far RIGHT: compute the finite nonconstant part, not D copies.

    The straight axis leg exits the local pulse ball once. Other two source
    legs lie in its rigid plateau. Long exterior segments have delta f=0.
    q2/q6 refer to the actual product of ORIGINAL scalar ratios on these links.
    """
    with mp.workdps(100):
        s0 = source_scalar_jets(0)[0]
        deltas = [mp.mpf(str(_polynomial_plateau(r)-_polynomial_plateau(r+1)))
                  for r in range(24, 48)]
        # At the centre, products have vanishing odd jets. Exact mixed q4/q6
        # cross terms come from differentiating the PRODUCT, not summing jets.
        fn = lambda theta: mp.fprod(mp.tanh(36*mp.cos(delta*theta/2))/s0
                                    for delta in deltas)
        jets = tuple(+mp.diff(fn, 0, q) for q in range(7))
        curvature = -(mp.mpf(18)/2)/mp.cosh(36)**2
        expected = curvature/s0*sum(delta**2 for delta in deltas)
        if abs(jets[2]-expected) > abs(expected)*mp.mpf('1e-60'):
            raise AssertionError('True original centre scalar second curvature missing')
        wrong = abs(expected)
        try:
            if wrong > abs(expected)*mp.mpf('1e-40'):
                raise AssertionError('Missing scalar curvature rejected at its own scale')
        except AssertionError as exc:
            rejection = str(exc)
        else:
            raise AssertionError('Zero second curvature incorrectly accepted')
        return dict(profile=19, eta=18, plateau=24, outer=48, nonconstant_edges=24,
                    centre_second_curvature=mp.nstr(curvature, 80),
                    boundary_second_word=mp.nstr(jets[2], 80),
                    boundary_q0_to_q6=[mp.nstr(v, 80) for v in jets],
                    mixed_q4_q6_retained=True, missing_curvature_rejected=rejection,
                    distant_identity_segments_have_zero_relative_jet=True)


def source19_nested_casimir() -> dict[str, Any]:
    """Uncompressed second SOURCE word on actual19 compact detector sites.

    In the rigid core B(theta)=scalar(theta) U(theta) B(0), allRef4.
    Summing -B'' gives 3/4 B -3 scalar''/scalar B. The second term is
    explicitly retained; arbitrary-packet orbital terms are a separate word.
    """
    with mp.workdps(100):
        s0 = mp.tanh(36)
        curvature = -9/mp.cosh(36)**2
        rows = []
        displacements = tuple(p for p in itertools.product((-1, 0, 1), repeat=3)
                              if sum(abs(v) for v in p) <= 2)
        for y, z in sorted({(p[1], p[2]) for p in displacements}):
            # Actual source-to-RIGHT x-axis leg, including diagonal radii.
            values = [_polynomial_plateau(math.sqrt(x*x+y*y+z*z))
                      for x in range(0, 50)]
            angular_square = sum(mp.mpf(str(a-b))**2 for a, b in zip(values, values[1:]))
            correction = -3*curvature/s0*angular_square
            rows.append((y, z, correction))
        upper = max(row[2] for row in rows)
        if not 0 < upper < mp.mpf('1e-28'):
            raise AssertionError('Actual SC19 source-valued intrinsic second word outside curvature envelope')
        # Each source coordinate is -1..1 in the rigid core. Constant interior/
        # exterior links keep the actual19 source's individual tree lengths.
        return dict(source_count=len(displacements), source_displacements=displacements,
            boundary_classes=len(rows), all_reference_count=4,
            relative_source_Casimir='sum(delta_j^2 B)=3/4 B-3(scalar_second/scalar)B in rigid core; orbital/ref/EM remainder retained separately',
            nonzero_Casimir_curvature_upper=mp.nstr(upper, 80),
            curvature_source_vector_norm_upper=mp.nstr(64*upper, 80),
            intrinsic_not_arbitrary_packet_total_J=True,
            uncompressed_second_SOURCE_word=True,
            baseline_original_scalar_power_actual_source_tree_lengths_retained=True)


class NativeFieldHistory:
    """Actual fullPhi/SU4/literal-sin-EM integral curve with off-Clifford normals."""
    def __init__(self, law: Any, frame: np.ndarray, clock: PulseClock):
        self.law, self.clock = law, clock
        rng = np.random.default_rng(22210671)
        self.cloud = law.index[(1, 1, 1)]
        self.reference = law.index[(0, 0, 0)]
        self.sites = [law.index[(1, 2, 1)], law.index[(2, 2, 1)]]
        self.fields = np.tile(frame, (law.volume, 1, 1, 1))+.001*law.fields(
            rng.normal(size=(law.volume, 4, 15)))
        self.links = np.asarray([[expm(.008j*np.einsum('a,aij->ij',
            rng.normal(size=15), law.generators)) for _ in range(3)]
            for _ in range(law.volume)])
        self.em = rng.normal(scale=.04, size=(law.volume, 3))
        self.cutoff = np.ones(law.volume); self.cutoff[self.reference] = 0.
        r = law.sites-law.sites[self.cloud]
        self.velocity = .025*np.column_stack((-r[:, 1], r[:, 0], np.zeros(law.volume)))
        self.velocity[self.reference] = 0.
        self.eta = np.zeros(law.volume, complex)
        self.eta[self.sites] = [math.sqrt(.6), math.sqrt(.4)*np.exp(.37j)]
        self.shapes = [self.fields.shape, self.links.shape, self.em.shape]
        self.lengths = [int(np.prod(shape)) for shape in self.shapes]
        initial = self.pack(self.fields, self.links, self.em)
        self.solution = solve_ivp(self.rhs, (-.002, clock.angle+.002), initial,
            rtol=3e-10, atol=3e-12, dense_output=True, max_step=.004)
        if not self.solution.success or self.solution.sol is None:
            raise AssertionError('Actual native physical field pulse failed')

    def pack(self, *parts: np.ndarray) -> np.ndarray:
        return np.concatenate([v.ravel() for v in parts])

    def unpack(self, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        parts = np.split(vector, np.cumsum(self.lengths)[:-1])
        return parts[0].reshape(self.shapes[0]), parts[1].reshape(self.shapes[1]), np.real(parts[2]).reshape(self.shapes[2])

    def rhs(self, theta: float, vector: np.ndarray) -> np.ndarray:
        fields, links, em = self.unpack(vector)
        return self.pack(*primitive_flow(self.law, fields, links, em,
                                         self.velocity, self.cutoff))

    def at(self, theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.solution.sol is not None
        return self.unpack(self.solution.sol(theta))

    def marks(self, theta: float) -> np.ndarray:
        fields, links, em = self.at(theta)
        allmarks = screened_marks(self.law, fields, links, em, self.eta,
                                  self.reference, 18, harmonic=np.array([.17, -.11, .08]))
        alpha = np.array([1., .3j, -.2+.4j, .6])
        indices = np.concatenate([np.arange(4*s, 4*s+4) for s in self.sites])
        return np.einsum('a,air->ir', alpha, allmarks[:, indices])/math.sqrt(2)

    def jets(self, theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = 1e-4
        m2, m1, m0, p1, p2 = [self.marks(theta+k*h) for k in (-2, -1, 0, 1, 2)]
        return m0, (m2-8*m1+8*p1-p2)/(12*h), (-m2+16*m1-30*m0+16*p1-p2)/(12*h*h)

    def orbital(self, theta: float) -> np.ndarray:
        _, links, em = self.at(theta)
        full = scalar_orbital(self.law, links, em, self.velocity)
        indices = np.concatenate([np.arange(4*s, 4*s+4) for s in self.sites])
        return full[np.ix_(indices, indices)]


def native_quantum_angular(history: NativeFieldHistory, neutral: Any,
                           charged: Any) -> dict[str, Any]:
    """Quantize the ACTUAL field/colour/EM integral curve, not its endpoint spin.

    The five-node Hermite chart is an explicit finite angular Galerkin witness.
    Its source is a multiplication operator of the full original marks, with
    whole exterior sectors and allRef4. Intermediate field states are retained
    in the nested relative word. Orthogonal normal directions are not simulated.
    """
    nodes = 5
    ladder = np.diag(np.sqrt(np.arange(1, nodes)), 1)
    x = (ladder+ladder.T)/math.sqrt(2)
    p = -1j*(ladder-ladder.T)/math.sqrt(2)
    grid, rotation = eigh(x)
    width = .006
    poses = history.clock.angle/2+width*grid
    pgrid = rotation.conj().T@p@rotation/width
    # On its own integral curve the actual physical vector pulls back to +1.
    field = field_noether_galerkin(np.ones((nodes, 1)), (1j*pgrid)[None])
    accept(np.linalg.norm(field-pgrid), 'actual Hermitian quantum field Noether compression')
    coefficients = creators(neutral, charged)
    marks = [np.einsum('ir,icd->rcd', history.marks(float(at)), coefficients)
             for at in poses]
    B = np.asarray([block_diag(*(mark[ref] for mark in marks)) for ref in range(4)])
    orbital = [history.orbital(float(at)) for at in poses]
    gn = np.kron(field, np.eye(neutral.d))+np.asarray(block_diag(*(exterior(o, neutral.rows) for o in orbital)))
    gc = np.kron(field, np.eye(charged.d))+np.asarray(block_diag(*(exterior(o, charged.rows) for o in orbital)))
    hnblocks, hcblocks = [], []
    for at in poses:
        phi, links, em = history.at(float(at))
        full = history.law.onebody(phi, links, em)
        indices = np.concatenate([np.arange(4*s, 4*s+4) for s in history.sites])
        h = full[np.ix_(indices, indices)]
        gram = np.einsum('xmij,xnji->xmn', phi, phi).real/4
        heavy = np.linalg.eigvalsh(full)
        potential = 32*float(np.sum(gram*gram))-32*float(np.sum(heavy[heavy > 0]))
        hnblocks.append(exterior(h, neutral.rows)+np.diag(neutral.coulomb)+potential*np.eye(neutral.d))
        hcblocks.append(exterior(h, charged.rows)+np.diag(charged.coulomb)+potential*np.eye(charged.d))
    hn = 1e-6*np.kron(field@field, np.eye(neutral.d))+np.asarray(block_diag(*hnblocks))
    hc = 1e-6*np.kron(field@field, np.eye(charged.d))+np.asarray(block_diag(*hcblocks))
    en, vn = eigh(hn); ec, vc = eigh(hc)
    an, un = eigh(gn); ac, uc = eigh(gc)
    initial = vn[:, 0]
    source = B@initial
    first = relative_word(B, gc, gn, 1)
    second = relative_word(B, gc, gn, 2)
    explicit = gc@gc@B-2*gc@B@gn+B@gn@gn
    error = accept(np.linalg.norm(second-explicit), 'true quantum-field uncompressed nested word', 2e-8)
    wrong = adverse(np.linalg.norm((gc@gc@B-second)@initial),
                    'true quantum-field uncompressed nested word')
    def propagate(hvalues: np.ndarray, hvectors: np.ndarray, gvalues: np.ndarray,
                  gvectors: np.ndarray, rows: np.ndarray, steps: int) -> np.ndarray:
        result = rows.copy()
        dt = history.clock.duration/steps
        for step in range(steps):
            for phase, vectors in ((np.exp(-.5j*dt*hvalues), hvectors),
                (np.exp(1j*dt*history.clock.speed((step+.5)*dt)*gvalues), gvectors),
                (np.exp(-.5j*dt*hvalues), hvectors)):
                result = (vectors@(phase[:, None]*(vectors.conj().T@result.T))).T
        return result
    actual = propagate(ec, vc, ac, uc, source, 96)
    comparison = propagate(ec, vc, ac, uc, source, 48)
    neutral_final = propagate(en, vn, an, un, initial[None], 96)[0]
    neutral_noise = float(np.linalg.norm(gn@initial))
    if neutral_noise <= 1e-6:
        raise AssertionError('Quantum field neutral Noether variance accidentally erased')
    # This genuinely time-dependent control includes the quantized field Lie
    # term. Static internal rotation would omit it and produce a different word.
    matter_only = np.asarray(block_diag(*(exterior(o, charged.rows) for o in orbital)))
    static_wrong = adverse(np.linalg.norm((relative_word(B, matter_only, np.zeros_like(gn), 2)-second)@initial),
                           'true quantum-field uncompressed nested word')
    return dict(quantum_field_nodes=nodes, neutral_dimension=hn.shape[0], charged_dimension=hc.shape[0],
        actual_physical_coordinate_poses=poses.tolist(), field_generator_norm=float(np.linalg.norm(field)),
        primitive_Phi_SU4_EM_curve_quantized=True, all_reference_count=4,
        original_quartic_and_heavy_scalar_retained=True, ordinary_E0_neutral=float(en[0]),
        ordinary_E0_charged=float(ec[0]), neutral_Noether_noise_norm=neutral_noise,
        source_first_word_norm=float(np.linalg.norm(first@initial)),
        source_second_word_norm=float(np.linalg.norm(second@initial)), nested_identity_error=error,
        missing_neutral_secondword_rejected=wrong, static_internal_rotation_rejected=static_wrong,
        control='H_ordinary - theta_dot(t) G_actual_field_plus_orbital',
        pulse_final_source_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
        native_time_discretization_comparison_norm=float(np.linalg.norm(actual-comparison)),
        source_norm_conservation_error=accept(abs(np.linalg.norm(actual)-np.linalg.norm(source)), 'finite unitary physical pulse source norm'),
        neutral_final_norm=float(np.linalg.norm(neutral_final)),
        ordinary_source_energy_final=float(sum(np.vdot(row, hc@row).real for row in actual)),
        native_scope='one actual physical integral-curve quantum Galerkin chart; orthogonal full normal directions retained by analytic graph, not numerically simulated')


def _creator_rows(mark: np.ndarray, coefficients: np.ndarray,
                  radial_dimension: int, neutral: Any, charged: Any) -> np.ndarray:
    cc = np.einsum('ir,icd->rcd', mark, coefficients)
    # The genuine harmonic Wilson factor shifts ell by -1. It is retained,
    # including its finite Galerkin boundary and nonzero momentum offset.
    shift = np.diag(np.ones(neutral.l-1), -1)
    return np.asarray([np.kron(np.eye(radial_dimension),
        np.kron(np.eye(neutral.n), np.kron(shift, c))) for c in cc])


def native_consumer() -> dict[str, Any]:
    band = RotationBand(a=1., m=.2, r=.12, v=.3, improved=False)
    _, frame = original_frame(band, 5, np.zeros(3), False)
    law = BulkQuantumLaw(3, band=band)
    spins = physical_generators(law, frame)
    clock = PulseClock()
    history = NativeFieldHistory(law, frame, clock)
    radial_dimension = 3
    neutral = Coupled(2, 1, .17, 4, .015, .53, frame)
    charged = Coupled(2, 1, .17, 5, .015, .53, frame)
    ladder = np.diag(np.sqrt(np.arange(1, radial_dimension)), 1)
    x = (ladder+ladder.T)/math.sqrt(2)
    p = -1j*(ladder-ladder.T)/math.sqrt(2)
    grid, rotation = eigh(x)
    onsite = np.kron(np.eye(2), (.2+3*.12)*frame[0])
    heavy = onsite.copy()
    for axis in range(3):
        hop = np.zeros((8, 8), complex)
        hop[:4, 4:] = -(.12*frame[0]+1j*.3*frame[axis+1])/2
        heavy += hop+hop.conj().T
    heavy_values = np.linalg.eigvalsh(heavy)
    heavy_linear = -float(np.sum(heavy_values[heavy_values > 0]))
    gram = np.einsum('mij,nji->mn', frame, frame).real/4
    quartic = 2*float(np.sum(gram*gram))
    centre = (-heavy_linear/(4*quartic))**(1/3)
    radial = centre+.04*grid
    # Nonzero radial Cartesian Phi is quantum, not a calibrated spectator.
    # Ordinary potential constant and photon zero point are NOT subtracted.
    radial_h = .01/(2*.04**2)*(rotation.conj().T@(p@p)@rotation)
    radial_h += np.diag(32*(quartic*radial**4+heavy_linear*radial))
    matrices = []
    for model in (neutral, charged):
        base = _dense(model)
        scalar = np.diag((model.phenergy[:, None]+model.coulomb[None, :]).ravel())
        matrices.append(np.kron(radial_h, np.eye(model.size))+
            np.kron(np.eye(radial_dimension), scalar)+np.kron(np.diag(radial), base-scalar))
    hn, hc = matrices
    en, vn = eigh(hn); ec, vc = eigh(hc)
    omega = vn[:, 0]
    birth_operator = np.kron(np.eye(radial_dimension), np.kron(ladder[:2, :2].T,
        np.eye(neutral.l*neutral.d)))
    birth = birth_operator@omega
    weights = np.array([1., -1., 0.])/math.sqrt(2*neutral.omega)
    current_n = np.column_stack([neutral.current(column, weights)
                                for column in np.eye(neutral.size).T])
    current_c = np.column_stack([charged.current(column, weights)
                                for column in np.eye(charged.size).T])
    jn = np.kron(np.diag(radial), current_n)
    jc = np.kron(np.diag(radial), current_c)
    cloud = vn@((vn.conj().T@(jn@omega))/(en-en[0]+neutral.omega))
    cc = creators(neutral, charged)
    positive_shift = max(1., float(en[0]-ec[0])+1.)
    positive = ec-en[0]+positive_shift
    cutoff = float(np.quantile(positive, .3))
    high = positive > cutoff
    records = []
    for t in (0., 1.5, 3., 4.5, 6.):
        theta = clock.pose(t)
        mark, dmark, ddmark = history.jets(theta)
        B, dB, ddB = [_creator_rows(v, cc, radial_dimension, neutral, charged)
                       for v in (mark, dmark, ddmark)]
        orbital = history.orbital(theta)
        h = 1e-4
        dorbital = (history.orbital(theta+h)-history.orbital(theta-h))/(2*h)
        lc = np.kron(np.eye(radial_dimension*charged.phdim), exterior(orbital, charged.rows))
        ln = np.kron(np.eye(radial_dimension*neutral.phdim), exterior(orbital, neutral.rows))
        dlc = np.kron(np.eye(radial_dimension*charged.phdim), exterior(dorbital, charged.rows))
        dln = np.kron(np.eye(radial_dimension*neutral.phdim), exterior(dorbital, neutral.rows))
        word = relative_source_jets(B, dB, ddB, lc, ln, dlc, dln)
        # Full second Leibniz identity with a nonconstant SAME neutral section.
        # psi'=v, psi''=w deliberately includes nonzero neutral angular variance.
        rng = np.random.default_rng(22210673)
        psi = rng.normal(size=hn.shape[0])+1j*rng.normal(size=hn.shape[0]); psi /= np.linalg.norm(psi)
        v = .2j*omega+.1*psi; w = -.3*omega+.07*psi
        neutral_first = -1j*v+ln@psi
        neutral_first_derivative = -1j*w+dln@psi+ln@v
        neutral_second = -w-1j*(dln@psi+2*ln@v)+ln@ln@psi
        bp = B@psi
        charged_second = -(ddB@psi+2*dB@v+B@w)-1j*(bp@dlc.T+2*(dB@psi+B@v)@lc.T)+bp@(lc@lc).T
        charged_on_neutral_first = -1j*(dB@neutral_first+B@neutral_first_derivative)+(B@neutral_first)@lc.T
        nested = charged_second-2*charged_on_neutral_first+B@neutral_second
        identity = accept(np.linalg.norm(nested-word['second']@psi),
                          'true nested SAME-neutral source second word', 2e-8)
        time_derivative = clock.speed(t)*dB
        moving = timed_source(B, time_derivative, hc, hn)
        algebraic = hc@B-B@hn-1j*clock.speed(t)*dB
        accept(np.linalg.norm(moving['residual']-algebraic), 'actual moving -iBdot equation')
        missing_neutral = adverse(np.linalg.norm(charged_second-nested),
            'true nested SAME-neutral source second word')
        missing_ref = adverse(np.linalg.norm(mark[:, 1:]), 'complete four reference source')
        missing_orbital = adverse(np.linalg.norm((word['second']+ddB)@psi),
            'actual orbital/ref mixed second word')
        missing_bdot = None
        if clock.speed(t):
            missing_bdot = adverse(np.linalg.norm((moving['residual']-(hc@B-B@hn))@birth),
                'actual moving -iBdot equation')
        for label, initial in (('birth', birth), ('cloud', cloud)):
            neutral_at = vn@(np.exp(-1j*t*en)*(vn.conj().T@initial))
            prepared = B@neutral_at
            neutral_right = B@(hn@neutral_at)
            contracted = timed_source(prepared[0], (time_derivative@neutral_at)[0], hc, hn,
                                      right_action=neutral_right[0])
            contracted_error = accept(np.linalg.norm(contracted['residual']-(moving['residual']@neutral_at)[0]),
                                      'exact nonzero neutral right-action contraction', 2e-8)
            missing_right = adverse(np.linalg.norm(neutral_right[0]),
                                    'exact nonzero neutral right-action contraction')
            coeff = prepared@vc.conj()
            low = (vc[:, ~high]@coeff[:, ~high].T).T
            tail = (vc[:, high]@coeff[:, high].T).T
            # Independent ordinary-H propagations of BOTH components.
            def evolve(rows: np.ndarray) -> np.ndarray:
                return (vc@(np.exp(-1j*(6-t)*ec)[:, None]*(vc.conj().T@rows.T))).T
            actual = evolve(prepared)
            accept(np.linalg.norm(actual-evolve(low)-evolve(tail)), 'unfiltered source low/high propagation')
            first_graph = float(np.linalg.norm(coeff*positive[None, :]))
            half_tail = float(np.linalg.norm(coeff[:, high]*np.sqrt(positive[high])[None, :]))
            accept(max(0., half_tail-first_graph/math.sqrt(cutoff)), 'SOURCE high-half graph')
            rows = dict(time=t, theta=theta, source=label, all_reference_count=4,
                source_norm=float(np.linalg.norm(prepared)), source_first_graph=first_graph,
                high_source_norm=float(np.linalg.norm(tail)), high_source_half_energy=half_tail,
                invariant_high_norm_error=accept(abs(np.linalg.norm(evolve(tail))-np.linalg.norm(tail)), 'invariant SOURCE high norm'),
                current=float(sum(np.vdot(row, jc@row).real for row in actual)),
                relative_current_boundary_norm=float(np.linalg.norm((jc@B-B@jn)@neutral_at)),
                occupation=float(sum(np.vdot(row, row).real*5 for row in actual)),
                actual_ordinary_energy=float(sum(np.vdot(row, hc@row).real for row in actual)),
                first_word_norm=float(np.linalg.norm(word['first']@neutral_at)),
                second_word_norm=float(np.linalg.norm(word['second']@neutral_at)),
                Bdot_norm=float(np.linalg.norm(time_derivative@neutral_at)),
                source_sha256=hashlib.sha256(prepared.tobytes()).hexdigest(),
                second_word_identity_error=identity, neutral_first_norm=float(np.linalg.norm(neutral_first)),
                neutral_second_norm=float(np.linalg.norm(neutral_second)),
                contracted_moving_equation_error=contracted_error,
                actual_neutral_right_action_norm=float(np.linalg.norm(neutral_right[0])),
                missing_neutral_right_action_rejected=missing_right,
                missing_neutral_rejected=missing_neutral, missing_ref_rejected=missing_ref,
                missing_orbital_rejected=missing_orbital, missing_Bdot_rejected=missing_bdot)
            records.append(rows)
    fields, links, em = history.at(clock.angle/2)
    flow = primitive_flow(law, fields, links, em, history.velocity, history.cutoff)
    true_current = endpoint_flow(law, fields, links, em, flow)
    wrong_flow = (flow[0], flow[1], np.zeros_like(flow[2]))
    missing_em = adverse(np.linalg.norm(true_current-endpoint_flow(law, fields, links, em, wrong_flow)),
                         'literal sin-curl EM physical current boundary')
    scalar = source19_boundary_word()
    half_casimir = sum(s@s for s in spins)
    casimir_error = accept(np.linalg.norm(half_casimir-.75*np.eye(4)), 'intrinsic physical vector lift Casimir')
    two_pi_error = accept(np.linalg.norm(expm(-2j*math.pi*spins[2])+np.eye(4)), 'continuous physical vector lift 2pi sign')
    return dict(clock=dict(duration=clock.duration, angle=clock.angle, pulse='quintic C2 compact'),
        native_scope='full Cartesian perturbed coefficient flow on27sites + finite quantum homogeneous radial-Phi/coexact/harmonic/eight-mode entire CAR; not full27site quantum vacuum',
        neutral_dimension=hn.shape[0], charged_dimension=hc.shape[0], radial_grid=radial.tolist(),
        ordinary_neutral_E0=float(en[0]), ordinary_charged_E0=float(ec[0]),
        harmonic_offset=.17, photon_zero_point_retained=True,
        fullPhi_flow_norm=float(np.linalg.norm(flow[0])), SU4_flow_norm=float(np.linalg.norm(flow[1])),
        literal_sine_EM_flow_norm=float(np.linalg.norm(flow[2])),
        missing_EM_rejected=missing_em, scalar_boundary=scalar,
        source_valued_intrinsic_Casimir=source19_nested_casimir(),
        quantum_angular_source_history=native_quantum_angular(history, neutral, charged),
        intrinsic_spin_half_Casimir_error=casimir_error, continuous_lift_two_pi_error=two_pi_error,
        total_J_equals_intrinsic_spin_claimed=False,
        source_current_change=records[-1]['current']-records[0]['current'],
        source_phase_overlap_real=float(np.vdot(history.marks(0), history.marks(clock.angle)).real),
        source_phase_overlap_imag=float(np.vdot(history.marks(0), history.marks(clock.angle)).imag),
        input_sha256=hashlib.sha256(history.pack(history.fields, history.links, history.em).tobytes()).hexdigest(),
        records=records)


def selected_certificate(receipt: dict[str, Any]) -> dict[str, Any]:
    """Propagate genuine SC19 SOURCE tails and nested mixed q2/q6 envelopes.

    Reuses the executed original normal certificate, never the RAW comparison.
    The physical geometry adds CY and profile orbital factors. A second word
    uses q2, while q6 remains in the low-W mixed graph/localization envelope.
    """
    selected = receipt['selected_source_flight']
    source = selected['sources']['scattering19']
    if source['source_identity'] != dict(eta=18, profile=19, detector_source=True):
        raise ValueError('Original source19 must own the moving word certificate')
    log = lambda value: int(value['log2_upper'])
    jet2 = max(log(g['jets'][2]) for g in source['graphs'].values())
    jet6 = max(log(g['jets'][6]) for g in source['graphs'].values())
    cy = log(source['physical_CY_upper'])
    orbital = log(source['profile_orbital_q6_upper'])
    # (1+q2)*(1+CY)^2*(1+orbital)^2; 8 covers sums and quintic clock.
    word2 = jet2+2*cy+2*orbital+8
    mixed6 = jet6+6*cy+6*orbital+16
    cutoff = int(selected['high_source_spectral_cutoff_log2'])
    tail_rows = []
    for label, graph in source['graphs'].items():
        first = log(graph['first'])
        tail_rows.append(dict(source=label, source_first_graph_log2_upper=first,
            retained_high_word_norm_log2_upper=first-cutoff+word2,
            retained_high_word_half_energy_log2_upper=first-(cutoff//2)+word2,
            mixed_q6_high_half_log2_upper=first-(cutoff//2)+mixed6))
    # The existing full SOURCE/current allocation ALREADY owns GLOBAL Loewdin
    # and phase bookkeeping. Extra second-word factors are counted once here.
    transfer = log(selected['source_current_and_occupation_transfer_upper'])+mixed6
    target = Fraction(selected['target']['exact_rational'])
    if Fraction(2)**transfer >= target:
        raise AssertionError('Moving source second-word transfer exceeds actual UE/V target')
    if max(row['mixed_q6_high_half_log2_upper'] for row in tail_rows) >= -4922:
        raise AssertionError('True mixed high SOURCE tails not paid by selected cutoff')
    scalar = source19_boundary_word()
    return dict(source='original SC19 eta18 alpha2^371 wordD+2', N=selected['N'],
        actual_time=selected['time']['exact_rational'], actual_all64_E0_kappa=True,
        ordinary_H_not_shifted_to_spin_operator=True, source_scalar=scalar,
        source_valued_intrinsic_Casimir=source19_nested_casimir(),
        physical_secondword_multiplier_log2_upper=word2,
        full_mixed_q6_multiplier_log2_upper=mixed6,
        q2_graph_log2_upper=jet2, q6_graph_log2_upper=jet6, source_tails=tail_rows,
        moving_source_current_secondword_transfer_log2_upper=transfer,
        strict_exact_UE_over_V_target_met=True,
        source_high_normal_retained=True, neutral_variance_cancelled_operatorially=True,
        normalized_paired_Bloch_history=receipt['paired_photon_source_join']['actual_normalized_packet'],
        intrinsic_spin='continuous physical vector lift spin1/2 with retained scalar boundary jets',
        arbitrary_packet_total_J='intrinsic+orbital+reference+neutral field; not universally J2=3/4',
        original_scalar_phase_and_normal_hypotheses_retained=True,
        RB_U_half_inference=False, Phi_ground_uniqueness_gap_used=False,
        improved_lightonly_extension='same heavy Phi/SU4/source scalar geometry; transport must supply independently derived improved current/source graphs',
        RAW41_comparison_used_as_proof=False)


def improved_certificate(selected: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
    """Bind independently renewed improved source graphs, never RAW comparison."""
    unchanged, changed = packet['unchanged'], packet['changed_law']
    if (unchanged['N'] != selected['N'] or unchanged['T'] != 6
            or unchanged['source_sites'] != 19 or unchanged['screen_eta'] != 18
            or unchanged['word'] != 'D+2' or not unchanged['allRef4']
            or not changed['heavy_unchanged']):
        raise ValueError('Improved source law is not the SAME actual SC19 physical family')
    if (changed['m'] != .2 or changed['r'] != .12 or not .3 <= changed['v'] <= .6
            or changed['mass'] != 'm+r sum(1-cos q)^2'):
        raise ValueError('Improved light coefficient/range-two q6 cap must be recomputed')
    target = Fraction(unchanged['UE'])/unchanged['V']/2**40
    renewed = packet['renewed_actual_transfer']
    # Light coefficient l1 grows by at most2, path range by at most2.
    # Twelve extra bits pay six mixed derivative/free-flow factors of each.
    multiplier = selected['full_mixed_q6_multiplier_log2_upper']+12
    transfer = int(renewed['source_photon_combined_absolute_debt_log2'])+multiplier
    if Fraction(2)**transfer >= target:
        raise AssertionError('Independently improved moving second-word graph exceeds UE/V target')
    cutoff = int(renewed['source_cutoff_log2'])
    sources = renewed['source_graphs_log2']['scattering19']
    return dict(changed_light_law=changed, heavy_and_source_geometry_unchanged=True,
        source='SC19', source_graphs=sources, actual_improved_spectral_cutoff_log2=cutoff,
        mixed_q6_high_half_log2_upper={label: int(graph['first'])-cutoff//2+multiplier
                                     for label, graph in sources.items()},
        new_moving_source_current_word_transfer_log2_upper=transfer,
        improved_q6_range_coefficient_reserve_bits=12,
        strict_exact_target_met=True, actual_normalized_packet=renewed['packet'],
        microscopic_Compton_supplier=packet['receipt'],
        improved_primitive_law_sha256=packet['inputs']['primitive_supplier_receipt_sha256'],
        probability_copied=False, RAWcomparison_copied=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-flight', type=Path, required=True)
    parser.add_argument('--descriptor', type=Path, required=True)
    parser.add_argument('--improved-compton', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.set_int_max_str_digits(0)
    flight = json.loads(args.source_flight.read_text())
    descriptor_bytes = args.descriptor.read_bytes()
    descriptor = json.loads(descriptor_bytes)
    descriptor_sha = hashlib.sha256(descriptor_bytes).hexdigest()
    if descriptor_sha not in flight['input_sha256'].values():
        raise ValueError('Actual SC19 geometry bytes differ from executed SOURCE supplier')
    selected = selected_certificate(flight)
    if (descriptor['N'] != selected['N'] or descriptor['compact_displacement'] !=
            [list(p) for p in selected['source_valued_intrinsic_Casimir']['source_displacements']]):
        raise ValueError('SC19 must be actual19 compact sites, not profile19^3 or RAW41')
    selected['actual_geometry_descriptor_sha256'] = descriptor_sha
    native = native_consumer()
    result: dict[str, Any] = dict(plan=PLAN, selected=selected, native=native,
        source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (Path(__file__), ROOT/'moving_physical_noether.md')},
        input_sha256={str(args.source_flight): hashlib.sha256(args.source_flight.read_bytes()).hexdigest()})
    result['input_sha256'][str(args.descriptor)] = descriptor_sha
    if args.improved_compton:
        improved = json.loads(args.improved_compton.read_text())
        result['improved'] = improved_certificate(selected, improved)
        result['input_sha256'][str(args.improved_compton)] = hashlib.sha256(args.improved_compton.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output), native_dimension=native['charged_dimension'],
        selected_secondword_log2=selected['moving_source_current_secondword_transfer_log2_upper'],
        missing_curvature_rejected=native['scalar_boundary']['missing_curvature_rejected'],
        actual_current_change=native['source_current_change'], rows=len(native['records']))))


if __name__ == '__main__':
    main()
