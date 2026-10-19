"""Continuous-transfer primitive-link sea tensor and subthreshold photon modes."""
from __future__ import annotations

from collections import OrderedDict
from typing import TypedDict

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq

from rotation_band import RotationBand, HOPPING, MIXING


class InfraredResponse(TypedDict):
    electric: float
    magnetic: float
    electric_axes: tuple[float, ...]
    magnetic_axes: tuple[float, ...]
    electric_cubic_error: float
    magnetic_cubic_error: float
    domain: str


def brillouin_quadrature(order: int, rule: str = 'midpoint'):
    """Cubic normalized BZ measure; None means uniform midpoint weights."""
    if rule == 'midpoint':
        axis = -np.pi + (np.arange(order) + .5) * (2 * np.pi / order)
        weights = None
    elif rule == 'split_gauss':
        edges = np.array([-np.pi, -.6, -.12, 0., .12, .6, np.pi])
        nodes, legendre_weights = leggauss(order)
        widths = np.diff(edges)
        axis = (edges[:-1, None] + widths[:, None] * (nodes + 1) / 2).ravel()
        axis_weights = (widths[:, None] * legendre_weights / (4 * np.pi)).ravel()
        weights = np.einsum('i,j,k->ijk', axis_weights, axis_weights, axis_weights).ravel()
    else:
        raise ValueError('Unknown BZ quadrature rule.')
    points = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1).reshape(-1, 3)
    return points, weights


class ContinuousSea:
    """Genuine continuous-transfer BZ response using exact Clifford traces.

    Internal units have a=1. The selected RotationBand's matrix-unit law
    determines all four coefficient functions; no per-momentum diagonalization
    or off-grid torus shift supplies its spectral weights.
    """
    def __init__(self, order: int, ma: float = .4, *, band: RotationBand | None = None,
                 chi_e: float = 1 / .09, chi_b: float = 1 / .09,
                 quadrature: str = 'midpoint'):
        self.band = RotationBand(a=1., m=ma, e=1., improved=True) if band is None else band
        if order < 8 or self.band.m <= 0 or chi_e <= 0 or chi_b <= 0:
            raise ValueError('Require order >=8, positive mass and positive photon stiffnesses.')
        if self.band.a != 1. or self.band.e != 1.:
            raise ValueError('Spectral units require a=1 and microscopic integer source magnitude=1.')
        self.order, self.ma, self.quadrature = order, self.band.m, quadrature
        self.p, self.weights = brillouin_quadrature(order, quadrature)
        gamma = np.array([MIXING, *HOPPING])
        onsite, terms = self.band.couplings()
        self._onsite = np.einsum('mij,ji->m', gamma, onsite).real / 4
        self._terms = [(d, np.einsum('mij,ji->m', gamma, t) / 4) for d, t in terms]
        self._d, self._energy = self._components(self.p)
        self._direction = self._d / self._energy[:, None]
        self._contact_terms = self._periodic_contact_terms()
        self.chi_e, self.chi_b = chi_e, chi_b
        self._poles = OrderedDict()

    def _components(self, p):
        d = np.broadcast_to(self._onsite, (len(p), 4)).copy()
        for displacement, coefficient in self._terms:
            phase = np.exp(1j * (p @ displacement))
            d += 2 * (phase[:, None] * coefficient).real
        energy = np.linalg.norm(d, axis=1)
        if np.any(energy <= 0):
            raise ValueError('Selected occupied and empty bands must remain gapped.')
        return d, energy

    def _average(self, values):
        return float(values.mean() if self.weights is None else np.dot(self.weights, values))

    def _periodic_contact_terms(self):
        """Exact periodic-BZ integration by parts of the primitive contact.

        Tr(P_- h'') = -Tr(P_-' h'). This representation cancels the
        zero-transfer pair response pointwise, rather than amplifying a
        quadrature error by 1/|q|^2. It is not a finite-torus replacement.
        """
        derivative = np.zeros((len(self.p), 3, 4))
        pieces = []
        for displacement, coefficient in self._terms:
            axis = int(np.flatnonzero(displacement)[0])
            steps = int(displacement[axis])
            term = -2 * steps * (np.exp(1j * (self.p @ displacement))[:, None] * coefficient).imag
            derivative[:, axis] += term
            pieces.append((axis, steps, term))
        longitudinal = np.einsum('nm,nim->ni', self._direction, derivative)
        tangent = (derivative - self._direction[:, None, :] * longitudinal[:, :, None])
        tangent /= self._energy[:, None, None]
        self._infrared_response = self._infrared_from_derivatives(derivative, tangent)
        return [(axis, steps, 2 * self.band.charge**2 * self._average(
            np.sum(tangent[:, axis] * term, axis=1))) for axis, steps, term in pieces]

    def _infrared_from_derivatives(self, derivative, tangent) -> InfraredResponse:
        """Exact transfer-zero electric and q^2 magnetic coefficients."""
        energy, direction = self._energy, self._direction
        electric = []
        magnetic = []
        for axis in range(3):
            polarization = (axis + 1) % 3
            electric.append(self.band.charge**2 / 2 * self._average(
                np.sum(derivative[:, axis] * tangent[:, axis], axis=1) / energy**2))
            second = np.zeros_like(self._d)
            for displacement, coefficient in self._terms:
                if displacement[axis]:
                    second -= 2 * displacement[axis]**2 * (
                        np.exp(1j * (self.p @ displacement))[:, None] * coefficient).real
            first_radius = np.sum(direction * derivative[:, axis], axis=1)
            second_radius = (np.sum(direction * second, axis=1)
                             + np.sum(tangent[:, axis] * derivative[:, axis], axis=1))
            second_direction = (second - direction * second_radius[:, None]
                                - 2 * tangent[:, axis] * first_radius[:, None]) / energy[:, None]
            vertex = -self.band.charge / 2 * derivative[:, polarization]
            projected = np.sum(direction * vertex, axis=1)
            weight = 2 * (np.sum(vertex**2, axis=1) - projected**2)
            first_weight = -2 * np.sum(tangent[:, axis] * vertex, axis=1) * projected
            second_weight = (-np.sum(tangent[:, axis]**2, axis=1) * np.sum(vertex**2, axis=1)
                             - 2 * np.sum(second_direction * vertex, axis=1) * projected)
            curvature = (second_weight / (2 * energy)
                         - first_weight * first_radius / (2 * energy**2)
                         + weight * (first_radius**2 / (4 * energy**3)
                                     - second_radius / (4 * energy**2)))
            magnetic.append(-4 * self._average(curvature))
        return InfraredResponse(electric=float(np.mean(electric)), magnetic=float(np.mean(magnetic)),
                    electric_axes=tuple(electric), magnetic_axes=tuple(magnetic),
                    electric_cubic_error=float(np.ptp(electric)),
                    magnetic_cubic_error=float(np.ptp(magnetic)),
                    domain='Analytic derivatives of the same thermodynamic BZ integral at q=0.')

    def infrared(self) -> InfraredResponse:
        return self._infrared_response.copy()

    def _contact(self, q):
        contact = np.zeros((3, 3))
        for axis, steps, value in self._contact_terms:
            path = sum(np.cos(q[axis] * (r - (steps - 1) / 2)) for r in range(steps))
            contact[axis, axis] += value * (path / steps)**2
        return contact

    def spectrum(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (3,) or not 0 < np.linalg.norm(q) < .6:
            raise ValueError('Continuous low-momentum domain: 0<|Q|<.6.')
        contact = self._contact(q)
        channels = []
        for sign in (-1, 1):
            vertex = np.zeros((len(self.p), 3, 4))
            for displacement, coefficient in self._terms:
                axis = int(np.flatnonzero(displacement)[0])
                steps = int(displacement[axis])
                # Straight-path midpoint phases cancel exactly, leaving a real
                # Clifford coefficient, including every elementary long-path link.
                path = sum(np.cos(q[axis] * (r - (steps - 1) / 2)) for r in range(steps))
                phase = np.exp(1j * (self.p @ displacement))
                middle = phase * np.exp(.5j * sign * (q @ displacement))
                vertex[:, axis] += self.band.charge * path * (middle[:, None] * coefficient).imag
            shifted_d, shifted_energy = self._components(self.p + sign * q)
            shifted_direction = shifted_d / shifted_energy[:, None]
            gaps = shifted_energy + self._energy
            inner = np.sum(shifted_direction * self._direction, axis=1)
            left = np.einsum('nm,nim->ni', shifted_direction, vertex)
            right = np.einsum('nm,nim->ni', self._direction, vertex)
            # Tr(P+ J_i P- J_j) from the actual four-gamma trace:
            # (1+a.b) u_i.u_j -(a.u_i)(b.u_j)-(a.u_j)(b.u_i).
            weights = (1 + inner[:, None, None]) * np.einsum('nim,njm->nij', vertex, vertex)
            weights -= left[:, :, None] * right[:, None, :] + right[:, :, None] * left[:, None, :]
            if self.weights is not None:
                weights *= self.weights[:, None, None]
            channels.append((gaps, weights))
        normalization = 1 / len(self.p) if self.weights is None else 1.
        return SeaTensor(q, contact, channels, self.chi_e, self.chi_b, normalization)

    def modes(self, q):
        q = np.asarray(q, dtype=float)
        key = tuple(float(x) for x in q)
        if key not in self._poles:
            self._poles[key] = self.spectrum(q).modes()
            if len(self._poles) > 64:
                self._poles.popitem(last=False)
        return self._poles[key]

    def axis_energy(self, radius: float, axis: int = 0):
        if radius == 0:
            return 0.
        modes = self.modes(radius * np.eye(3)[axis])
        if modes['tensor_axis_defect'] > 1e-10:
            raise ValueError('Cubic-axis degeneracy failed; an unpolarized axis supplier is invalid.')
        return modes['branches'][0]['energy']

    def axis_velocity(self, radius: float, axis: int = 0, step: float = 2e-4):
        if radius <= step:
            raise ValueError('Velocity stencil must stay inside the positive radial domain.')
        return (self.axis_energy(radius + step, axis) - self.axis_energy(radius - step, axis)) / (2 * step)


class SeaTensor:
    def __init__(self, q, contact, channels, chi_e, chi_b, normalization):
        self.q, self.contact, self.channels = q, contact, channels
        self.chi_e, self.chi_b, self.normalization = chi_e, chi_b, normalization
        self.khat = 2 * np.sin(q / 2)
        n = self.khat / np.linalg.norm(self.khat)
        seed = np.eye(3)[np.argmin(abs(n))]
        first = np.cross(n, seed)
        first /= np.linalg.norm(first)
        self.basis = np.column_stack((first, np.cross(n, first)))
        self.gap = min(float(g.min()) for g, _ in channels)
        self.static = contact - self.spectral(0., 'static')

    def spectral(self, s, kind):
        result = np.zeros((3, 3))
        for gaps, weights in self.channels:
            if kind == 'static':
                factor = 1 / gaps
            elif kind == 'change':
                factor = s / (gaps * (gaps**2 - s))
            elif kind == 'derivative':
                factor = gaps / (gaps**2 - s)**2
            else:
                raise ValueError('Unknown spectral kernel.')
            # Real PSD projector trace weights already sum both band doublets.
            result += 4 * self.normalization * np.einsum('nij,n->ij', weights, factor)
        return (result + result.T) / 2

    def inverse(self, s):
        response = self.static - self.spectral(s, 'change')
        return self.basis.T @ response @ self.basis + (
            self.chi_b * np.dot(self.khat, self.khat) - self.chi_e * s) * np.eye(2)

    def modes(self):
        static = self.inverse(0.)
        eigen = np.linalg.eigvalsh(static)
        upper = float(eigen[-1] / self.chi_e)
        if eigen[0] <= 0 or upper >= self.gap**2:
            raise ValueError('Positive restoring tensor and sufficient subthreshold bound failed.')
        axis = np.count_nonzero(self.q) == 1
        defect = float(np.linalg.norm(static - np.trace(static) / 2 * np.eye(2))) if axis else None
        branches = []
        for index in range(2):
            solved = brentq(lambda s: np.linalg.eigvalsh(self.inverse(s))[index],
                            0., upper, xtol=1e-15, rtol=np.float64(1e-13), full_output=True)
            root = float(solved[0])
            _, vectors = np.linalg.eigh(self.inverse(root))
            # Cubic C4 symmetry enforces an exact scalar transverse block on an
            # axis. Choose the deterministic symmetry basis in this degenerate
            # space, rather than numerically fluctuating eigenvectors.
            vector = np.eye(2)[:, index] if axis else vectors[:, index]
            polarization = self.basis @ vector
            derivative = self.chi_e + float(polarization @ self.spectral(root, 'derivative') @ polarization)
            branches.append(dict(energy=float(np.sqrt(root)), residue=1 / derivative,
                                 source=1 / np.sqrt(derivative), polarization=polarization.tolist(),
                                 pole_residual=float(np.linalg.norm(self.inverse(root) @ vector)),
                                 inverse_residue=derivative))
        longitudinal = float(self.khat @ self.static @ self.khat)
        longitudinal_scale = max(abs(float(self.khat @ self.contact @ self.khat)), 1e-30)
        return dict(momentum=self.q.tolist(), lattice_momentum=self.khat.tolist(),
                    pair_threshold=self.gap, transverse_static_tensor=static.tolist(),
                    static_sea_tensor=self.static.tolist(), contact_tensor=self.contact.tolist(),
                    gradient_static_curvature=longitudinal,
                    gradient_relative_residual=abs(longitudinal) / longitudinal_scale,
                    tensor_axis_defect=defect, branches=branches,
                    branch_splitting=abs(branches[0]['energy'] - branches[1]['energy']),
                    sufficient_pole_squared_upper=upper)
