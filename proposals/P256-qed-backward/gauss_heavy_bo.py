#!/usr/bin/env python3
"""Physical heavy-only Haar BO/Schur bounds; no slow-spectrum or phase claim.

The CLI is an evidence consumer. Its finite probes check the primitive operator;
its continuum inequalities come from the accompanying derivation, not sampling.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy import sparse
from scipy.linalg import expm

from gauss_charge_intertwiner import boundary_word
from heavy_sea_centre_potential import HeavySeaCentrePotential, sqrt_upper
from quantum_su4_gauss import (HaarLink, PauliCode, PhysicalHamiltonian,
                               PrimitiveMatter, norm, require_small, sparse_sum)
from quantum_su4_photon import CompactGaussPhoton

HEAVY_SUPPLIER_SHA256 = 'de1e32fb43f9be142d227d9073b44dc02cc15bb20bff8f0606c58d699cbccf3a'
DIMENSION = 15


def upper(value: Fraction) -> float:
    x = float(value)
    return math.nextafter(x, math.inf) if Fraction(x) < value else x


def lower(value: Fraction) -> float:
    x = float(value)
    return math.nextafter(x, -math.inf) if Fraction(x) > value else x


def rational(value) -> Fraction:
    """Match the supplier's decimal bare-parameter convention, not binary drift."""
    return Fraction(str(value))


def sources(potential) -> dict[str, str]:
    root = Path(__file__).resolve().parent
    bound = potential.model
    from heavy_sea_centre_potential import source_hashes
    result = source_hashes()
    for name in ('gauss_heavy_bo.py', 'gauss_heavy_bo.md',
                 'quantum_su4_photon.py', 'quantum_su4_gauss.py'):
        result[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if result['heavy_sea_centre_potential.py'] != HEAVY_SUPPLIER_SHA256:
        raise ValueError('Whole-heavy proof supplier changed; review before using its certificate')
    if bound.scale != 1.:
        raise ValueError('Potential provider must expose the unscaled whole-heavy onebody')
    return result


def constants(potential, inertia, ue4, ue1, magnetic) -> dict[str, Any]:
    """All arithmetic of bound endpoints is rational until outward display."""
    certificate = potential.certificate
    beta, spatial = [tuple(Fraction(x) for x in certificate['amplitude_intervals'][key])
                     for key in ('beta', 'spatial')]
    m, r, v, j = map(rational, (potential.band.m, potential.band.r,
                                potential.band.v, potential.prefactor))
    I, E4, E1, K = map(rational, (inertia, ue4, ue1, magnetic))
    if min(I, E4, E1) <= 0 or K < 0:
        raise ValueError('Positive I, UE4, UE1 and nonnegative magnetic K required')
    a = (m + r) * beta[1]
    f = r * beta[1] / 2 + v * spatial[1] / 2
    g = 2 * ((m + r) * beta[0] - f)
    c = Fraction(certificate['exact_margin_rational'])
    if min(c, g) <= 0:
        raise ValueError('Supplier must certify positive actual potential coercivity and gap')
    # Rank-one NH=4 determinant projection on the actual 70-state heavy fibre.
    # Deliberately conservative invariant-direction bounds; ||Ta|| <= 1.
    h1 = h2 = 4 * f
    p1 = 2 * h1 / g
    p2 = 2 * h2 / g + 32 * h1 * h1 / (g * g)
    section1 = 4 * p1
    radius = min(Fraction(1, 2), 1 / (4 * p1))
    sinc = 1 - radius * radius / 24
    haar_lower, metric_upper = sinc ** 12, sinc ** -2
    # V'(I)=0. Heavy ground Hessian <= h2+2*h1^2/g.
    potential_quadratic = j * (beta[1] ** 2 + 3 * spatial[1] ** 2) / 4 + h2 / 2 + h1 * h1 / g
    kappa = E4 + 1 / I
    packet_gradient = Fraction(DIMENSION * (DIMENSION + 8), 3)
    A = 2 * kappa * metric_upper * packet_gradient / (haar_lower * radius * radius) + potential_quadratic * radius * radius
    # [D,P] bounds: endpoint heavy charge norm <=4; light and B commute with P.
    endpoint1 = p1 + 8
    endpoint2 = p2 + 16 * p1 + 64
    S1 = DIMENSION * (E4 * p1 * p1 + endpoint1 * endpoint1 / I)
    S0 = DIMENSION * (E4 * p2 + endpoint2 / I)
    c1, c0 = 8 * S1, 2 * S0 * S0
    raw = dict(a=a, f=f, g=g, c=c, p1=p1, p2=p2, section1=section1,
               radius=radius, haar_lower=haar_lower, metric_upper=metric_upper,
               potential_quadratic=potential_quadratic, kappa=kappa, A=A,
               c1=c1, c0=c0, light_norm=a + f)
    charges = {}
    for q in (0, -1, 1):
        nl, boundary_norm = 4 + q, int(q != 0)
        loss = nl * (a + f) + K
        angular = DIMENSION * (2 * E4 * section1 ** 2 +
                   ((section1 + 4 + nl) ** 2 +
                    (section1 + 4 + nl + boundary_norm) ** 2) / I)
        B = angular + loss + E1 * (4 + q * q) / 2
        # epsilon=1/2 scalar lower comparison; each endpoint ||Q|| <= NH+NL.
        Dmax = DIMENSION * ((4 + nl) ** 2 + (4 + nl + boundary_norm) ** 2)
        charges[str(q)] = dict(NL=nl, boundary={0: '1', -1: '4', 1: 'bar4'}[q],
            retained_light_dimension=math.comb(8, nl),
            retained_light_boundary_dimension=math.comb(8, nl) * (1 if q == 0 else 4),
            infinite_EM_loop_retained=True, Bq_upper=upper(loss), B_upper=upper(B),
            exact_Bq=str(loss), exact_B=str(B), Dmax_upper=upper(Fraction(Dmax)),
            scalar_lower_kappa=lower(E4 + 1 / (2 * I)),
            scalar_lower_loss_upper=upper(Fraction(Dmax) / (2 * I)))
    return dict(exact={key: str(x) for key, x in raw.items()},
                upper={key: upper(x) for key, x in raw.items() if key not in ('g', 'c', 'haar_lower')},
                lower={key: lower(raw[key]) for key in ('g', 'c', 'haar_lower')},
                charges=charges, EFT_inertia=inertia,
                inertia_interpretation='Declared isotropic EFT I; not induced Cartesian field inertia',
                projector_derivative_domain='All SU4 invariant left/right unit directions; P projects only NH=4 heavy determinant',
                trial_domain='Lambda>=1; radius=rho0*Lambda^(-1/4); four disjoint equivariantly related charts',
                packet='f_rho(x)=(1-|x|^2/rho^2)_+^2; unit light occupation, unit B vector, ell=0',
                exact_normalization='N_rho^2=4*c_H*integral_{|x|<rho} f_rho(x)^2*j(x) dx; psi=f_rho*omega/N_rho on each chart',
                normalization_lower='4*c_H*j_* * |S^14|*rho^15*Beta(15/2,5)/2',
                Schur_derivative_form='||QHP psi||^2 <= c1*Kcolour[psi]+c0*||psi||^2',
                proof='gauss_heavy_bo.md; rational/outward endpoints, not a sampled potential bound')


class HeavyPhysicalFibre:
    """Actual marker-word Pauli embedding of one heavy ray and ALL light states.

    No CAR matrix builds an operator. Every lifted matrix uses PrimitiveMatter.
    A small heavy block is a restriction of that actual matrix, not a new law.
    """

    def __init__(self, potential, q, link, inertia, ue4, ue1, magnetic):
        if q not in (0, -1, 1):
            raise ValueError('Only NL=4,3,5 sectors are supplied')
        self.potential, self.q = potential, q
        self.code = PauliCode()
        self.matter = PrimitiveMatter(self.code, 4 + q, potential.generators)
        self.link = link
        self.base = PhysicalHamiltonian(self.matter, link, potential.frame,
                    potential.band, 1., inertia, ue4, ue1)
        self.photon = CompactGaussPhoton(self.base, 1, magnetic)
        self.light_dimension = math.comb(8, 4 + q)
        self.bd = self.base.bd
        self.reference_indices = np.arange(70) * self.light_dimension
        heavy_modes = tuple(range(4)) + tuple(range(8, 12))
        light_modes = tuple(range(4, 8)) + tuple(range(12, 16))
        hs = list(itertools.combinations(heavy_modes, 4))
        ls = list(itertools.combinations(light_modes, 4 + q))
        self.units = {}
        for i, source in enumerate(heavy_modes):
            for j, target in enumerate(heavy_modes):
                op = self.matter.transfer(self.code, source, target)
                self.units[j, i] = op
        self.small_units = {ij: op[self.reference_indices][:, self.reference_indices].tocsr()
                            for ij, op in self.units.items()}
        words = [boundary_word(self.code, i, True) for i in range(16)]
        phases = np.zeros((70, self.light_dimension), complex)
        for hi, h in enumerate(hs):
            for li, l in enumerate(ls):
                mask, amplitude = 0, 1. + 0j
                for orbital in reversed(h + l):
                    new = mask | (1 << orbital)
                    amplitude *= self.code.element(mask, new, words[orbital], sign=1)
                    mask = new
                if self.matter.index[mask] != hi * self.light_dimension + li:
                    raise AssertionError('Provider fixed-number tensor ordering changed')
                phases[hi, li] = amplitude
        self.phase_ratios = phases / phases[:, :1]
        h = potential.value(np.eye(4))['h']
        H = -self.small_lift(h)
        values, vectors = np.linalg.eigh(H)
        self.seed = vectors[:, 0]
        require_small(abs(values[0] - potential.centre_energy), 'actual heavy centre energy', 2e-9)
        self.heavy_left = self.matter.heavy_left[self.reference_indices]
        self.inertia, self.ue4, self.ue1 = inertia, ue4, ue1
        self.light_units = {(i, j): self.matter.transfer(self.code, light_modes[j], light_modes[i])
                            for i in range(8) for j in range(8)}

    def small_lift(self, h):
        return sparse_sum(h[i, j] * self.small_units[i, j]
                          for i in range(8) for j in range(8)).toarray()

    def embed(self, omega) -> sparse.csr_matrix:
        rows = np.arange(self.matter.dimension)
        cols = np.tile(np.arange(self.light_dimension), 70)
        data = (omega[:, None] * self.phase_ratios).ravel()
        return sparse.csr_matrix((data, (rows, cols)),
              shape=(self.matter.dimension, self.light_dimension))

    def ray(self, U, centre=0) -> tuple[np.ndarray, sparse.csr_matrix, tuple, dict[str, Any]]:
        result = self.potential.value(U)
        H = -self.small_lift(result['h'])
        values, vectors = np.linalg.eigh(H)
        seed = (1j ** centre) ** self.heavy_left * self.seed
        overlap = np.vdot(seed, vectors[:, 0])
        if abs(overlap) < .5:
            raise ValueError('Outside the certified centre chart; no global chart fallback')
        omega = vectors[:, 0] * overlap.conjugate() / abs(overlap)
        require_small(abs(values[0] + result['sea_positive_sum']), 'actual whole-heavy Ky Fan energy', 3e-9)
        require_small(norm(H @ omega - values[0] * omega), 'actual primitive heavy eigenvector', 3e-9)
        J = self.embed(omega)
        if centre:
            # Light is NOT projected: each occupation column carries its own character.
            light_left = self.matter.light_left[:self.light_dimension]
            J = sparse.csr_matrix(J @ sparse.diags((1j ** centre) ** light_left, format='csr'))
        return omega, J, (values, vectors, seed), result

    def derivative(self, omega, spectral, dh):
        values, vectors, seed = spectral
        dH = -self.small_lift(dh)
        coeff = vectors[:, 1:].conj().T @ dH @ omega
        horizontal = vectors[:, 1:] @ (coeff / (values[0] - values[1:]))
        overlap = np.vdot(seed, omega).real
        phase = -np.vdot(seed, horizontal).imag / overlap
        return horizontal + 1j * phase * omega

    def centre_controls(self, U):
        omega, J, _, _ = self.ray(U)
        state = J[:, 0].toarray().ravel()
        residues = []
        for k in range(4):
            z = 1j ** k
            _, Jz, _, rz = self.ray(z * U, k)
            target = z ** self.matter.count_left * state
            residual = norm(Jz[:, 0].toarray().ravel() - target)
            require_small(residual, 'physical centre-equivariant heavy-only section', 3e-8)
            residues.append(residual)
            if k:
                D = np.diag([z] * 4 + [1.] * 4)
                require_small(norm(rz['projector'] - D @ self.potential.value(U)['projector'] @ D.conj().T),
                              'actual onebody heavy projector centre covariance', 3e-8)
        # An extra scalar character is rejected on an actual nonzero section.
        _, Ji, _, _ = self.ray(1j * U, 1)
        wrong = norm(1j * Ji[:, 0].toarray().ravel() - 1j ** self.matter.count_left * state)
        if wrong < 1.:
            raise AssertionError('Wrong centre character did not fail physical equivariance')
        nleft = self.matter.count_left
        nright = 8 + self.q - nleft
        right_ok = bool(np.all((nleft + nright + self.base.bc) % 4 == 0))
        omit_B_ok = bool(np.all((nleft + nright) % 4 == 0))
        if not right_ok or (self.q != 0 and omit_B_ok):
            raise AssertionError('Charged boundary four-ality control failed')
        return dict(equivariance_residuals=residues, wrong_character_residual=wrong,
                    wrong_character_rejected=True, right_endpoint_consistent=right_ok,
                    omitted_B_consistent=omit_B_ok, omitted_charged_B_rejected=self.q != 0,
                    actual_heavy_ray_norm=norm(omega), light_projector_used=False)

    def compressed(self, U, seed=222, include_matrices=False):
        omega, J, spectral, result = self.ray(U)
        Jd = J.toarray()
        identity_B = np.eye(self.bd)
        identity_LB = np.eye(self.light_dimension * self.bd)
        weights, actions = [], []
        labels = []
        zero_fields = np.zeros_like(self.potential.fields)
        for side in ('left', 'right'):
            for a, t in enumerate(self.potential.generators):
                dU = 1j * t @ U if side == 'left' else -1j * U @ t
                dh = self.potential.model.onebody_derivative(self.potential.fields, U,
                          (0., 0.), zero_fields, dU, np.zeros(2))[0]
                dw = self.derivative(omega, spectral, dh)
                dJ = self.embed(dw).toarray()
                if side == 'left':
                    weights.append(self.ue4)
                    actions.append(-1j * dJ)
                    labels.append(('EL', a))
                QJ = self.matter.Q[0 if side == 'left' else 1][a] @ Jd
                weights.append(1 / (2 * self.inertia))
                actions.append(-1j * dJ - QJ)
                labels.append(('DL' if side == 'left' else 'DR', a))
        A, born = [], []
        born_total = np.zeros_like(identity_LB, dtype=complex)
        primitive_sum = np.zeros_like(identity_LB, dtype=complex)
        for weight, action, (kind, a) in zip(weights, actions, labels):
            connection = Jd.conj().T @ action
            residual = action - Jd @ connection
            b = residual.conj().T @ residual
            connection_B = np.kron(connection, identity_B)
            if kind == 'DR':
                connection_B -= np.kron(np.eye(self.light_dimension), self.base.boundary[a])
            bB = np.kron(b, identity_B)
            # Tensor contractions retain B without a huge dense J tensor identity_B.
            gram = np.kron(action.conj().T @ action, identity_B)
            if kind == 'DR':
                qb = self.base.boundary[a]
                cross = action.conj().T @ Jd
                gram -= np.kron(cross, qb) + np.kron(cross.conj().T, qb)
                gram += np.kron(Jd.conj().T @ Jd, qb @ qb)
            primitive_sum += weight * gram
            born_total += weight * bB
            A.append(connection_B)
            born.append(bB)
            require_small(norm(connection_B - connection_B.conj().T), 'Hermitian covariant connection', 2e-8)
        reconstructed = sum(w * (a.conj().T @ a + b) for w, a, b in zip(weights, A, born))
        identity_residual = norm(primitive_sum - reconstructed)
        require_small(identity_residual, 'endpoint/B-inclusive compressed kinetic plus Born identity', 3e-8)
        born_min = float(np.linalg.eigvalsh(born_total).min())
        if born_min < -2e-9:
            raise AssertionError('Positive Born energy became negative')
        # Exercise the derivative part too, not only the constant-fibre connection.
        rng = np.random.default_rng(seed)
        f = rng.normal(size=identity_LB.shape[0]) + 1j * rng.normal(size=identity_LB.shape[0])
        f /= norm(f)
        d = rng.normal(size=(len(weights), len(f))) + 1j * rng.normal(size=(len(weights), len(f)))
        lhs = rhs = 0.
        for n, (w, action, (kind, a), connection, b) in enumerate(zip(weights, actions, labels, A, born)):
            fm = f.reshape(self.light_dimension, self.bd)
            dm = d[n].reshape(self.light_dimension, self.bd)
            primitive_f = action @ fm
            if kind == 'DR':
                primitive_f -= (Jd @ fm) @ self.base.boundary[a].T
            lhs += w * norm(-1j * Jd @ dm + primitive_f) ** 2
            rhs += w * (norm(-1j * d[n] + connection @ f) ** 2 + np.vdot(f, b @ f).real)
        require_small(abs(lhs - rhs), 'primitive differentiated kinetic identity', 3e-8)
        # Finite-difference the chart-fixed HEAVY ray; the light fibre is unchanged.
        t = self.potential.generators[0]
        step = 1e-5
        wp = self.ray(expm(1j * step * t) @ U)[0]
        wm = self.ray(expm(-1j * step * t) @ U)[0]
        dh = self.potential.model.onebody_derivative(self.potential.fields, U,
                   (0., 0.), zero_fields, 1j * t @ U, np.zeros(2))[0]
        analytic = self.derivative(omega, spectral, dh)
        derivative_error = norm((wp - wm) / (2 * step) - analytic)
        require_small(derivative_error, 'actual heavy chart derivative', 2e-7)
        report: dict[str, Any] = dict(point_V=result['V'], point_Q=result['Q'], primitive_kinetic_identity_residual=identity_residual,
            primitive_differentiated_identity_residual=abs(lhs - rhs), heavy_derivative_error=derivative_error,
            covariant_connection_count=len(A), retained_light_boundary_dimension=len(f),
            weighted_Born_eigenvalues=np.linalg.eigvalsh(born_total).tolist(),
            weighted_Born_trace=float(np.trace(born_total).real), Born_positive=True,
            connection_Frobenius_norms=[norm(a) for a in A],
            scope='Compression PH K PH, not the complete H; exact Schur subtraction is separate')
        if include_matrices:
            report['operators'] = dict(connections=A, Born_terms=born, weights=weights,
                                       weighted_Born=born_total, embedding=J)
        return report

    def retained_slow_multiplication(self, U, phi=.37, include_matrices=False):
        """Full light Pauli matrix, B and infinite rotor; no light-band projection.

        The infinite ell operator is the differential expression below, not a
        truncated matrix. Matrices expose its entire finite internal fibre.
        """
        _, J, _, _ = self.ray(U)
        Jd = J.toarray()
        hL = self.potential.model.onebody(self.potential.fields, U, (0., phi))[1]
        lifted = sparse_sum(hL[i, j] * self.light_units[i, j]
                            for i in range(8) for j in range(8)).tocsr()
        light = Jd.conj().T @ (lifted @ Jd)
        residual = norm(lifted @ Jd - Jd @ light)
        require_small(residual, 'entire light fibre retained under primitive light Hamiltonian', 3e-8)
        levels = np.linalg.eigvalsh(hL)
        many = np.linalg.eigvalsh(light)
        nl = 4 + self.q
        edge_residual = max(abs(many[0] - sum(levels[:nl])), abs(many[-1] - sum(levels[-nl:])))
        require_small(edge_residual, 'actual primitive light Ky Fan extrema', 3e-8)
        n = Jd.conj().T @ ((self.matter.light_left - 2)[:, None] * Jd)
        n2 = Jd.conj().T @ ((self.matter.light_left - 2)[:, None] ** 2 * Jd)
        require_small(norm(n2 - n @ n), 'light electric density preserves heavy compression', 3e-8)
        report: dict[str, Any] = dict(phi=phi, primitive_light_invariance_residual=residual,
            primitive_light_extrema_residual=edge_residual,
            actual_light_manybody_spectrum=many.tolist(), all_light_states_retained=True,
            infinite_loop_expression='hL(U,phi) + UE1/2*((-i*dphi+n)^2+(-i*dphi)^2+q^2) - K*cos(phi)',
            boundary='Tensor identity_B on multiplication; QB remains in DR, not absorbed into ell')
        if include_matrices:
            report['operators'] = dict(light=np.kron(light, np.eye(self.bd)),
                n=np.kron(n, np.eye(self.bd)), n2=np.kron(n2, np.eye(self.bd)))
        return report

    def photon_orientation_control(self, seed=223):
        name = next(n for n in self.photon.blocks if n != '1')
        block = self.photon.blocks[name]
        rng = np.random.default_rng(seed)
        x = rng.normal(size=block['shape']) + 1j * rng.normal(size=block['shape'])
        x /= norm(x)
        rep = self.link.reps[name]
        def DL(a, y):
            el = np.einsum('ca,mabzl->mcbzl', rep[a].T, y, optimize=True)
            return el - self.base.matter_action(block['Q'][0][a], y)
        def DR(a, y, boundary=True):
            er = -np.einsum('cb,mabzl->maczl', rep[a], y, optimize=True)
            z = er - self.base.matter_action(block['Q'][1][a], y)
            if boundary:
                z -= np.einsum('zw,mabwl->mabzl', self.base.boundary[a], y, optimize=True)
            return z
        explicit = sum(DL(a, DL(a, x)) + DR(a, DR(a, x)) for a in range(15)) / (2 * self.inertia)
        residual = norm(explicit - self.photon.orientation(name, x))
        require_small(residual, 'actual photon orientation including QR.QB cross term', 2e-8)
        without = sum(DL(a, DL(a, x)) + DR(a, DR(a, x, False), False) for a in range(15)) / (2 * self.inertia)
        omitted_difference = norm(explicit - without)
        if self.q != 0 and omitted_difference < .01:
            raise AssertionError('Omitted charged B failed to change actual orientation energy')
        return dict(exact_photon_orientation_residual=residual,
                    omitted_B_operator_difference=omitted_difference,
                    loop_cutoff_used_for_algebra_probe_only=1,
                    continuum_bounds_do_not_use_a_loop_cutoff=True)


def energy_bounds(potential, bounds, scale):
    """Outward bounds at requested scale, keeping V0 symbolic in all differences."""
    if not math.isfinite(scale) or scale < 1:
        raise ValueError('The explicit shrinking trial requires finite Lambda>=1')
    L = Fraction(scale)
    root_L = sqrt_upper(L)
    A = Fraction(bounds['exact']['A'])
    c, g = (Fraction(bounds['exact'][key]) for key in ('c', 'g'))
    c1, c0 = (Fraction(bounds['exact'][key]) for key in ('c1', 'c0'))
    rho = Fraction(bounds['exact']['radius'])
    answer = {}
    for key, charge in bounds['charges'].items():
        loss = Fraction(charge['exact_Bq'])
        B = Fraction(charge['exact_B'])
        excess = A * root_L + B
        budget = excess + loss
        delta = L * g - budget
        comparison = dict(delta_at_trial_energy_lower=lower(delta),
                          below_complementary_threshold=delta > 0)
        if delta > 0:
            comparison.update(Schur_retained_kinetic_relative_upper=upper(c1 / delta),
                              Schur_constant_upper=upper(c0 / delta),
                              positive_retained_kinetic_coefficient_certified=c1 < delta)
        answer[key] = dict(energy_upper_relative_to_Lambda_V0=upper(excess),
            energy_lower_relative_to_Lambda_V0=-upper(loss),
            expected_Q_upper=upper(budget / (L * c)),
            heavy_off_band_probability_upper=min(1., upper(budget / (L * g))),
            outside_Q_ge_tau_probability='min(1,(A*sqrt(Lambda)+B+Bq)/(Lambda*c*tau))',
            Schur=comparison)
    return dict(Lambda=scale, V0=potential.centre_energy,
                V0_exact_definition='-4*sqrt(((m+r)*beta)^2+(v*spatial/2)^2)',
                chart_radius_upper=upper(rho * sqrt_upper(sqrt_upper(1 / L))),
                charge_bounds=answer,
                interpretation='Ground energy/infimum trial and form bounds; not observed eigenvalues, multiplicity or phase')


def run(args):
    potential = HeavySeaCentrePotential()
    identities = sources(potential)
    bounds = constants(potential, args.inertia, args.ue4, args.ue1, args.magnetic)
    link = HaarLink('minimal')
    coordinates = np.arange(1., 16.)
    coordinates /= norm(coordinates)
    radius = float(Fraction(bounds['exact']['radius']))
    U = expm(1j * radius / 3 * np.einsum('a,aij->ij', coordinates, potential.generators))
    evidence = {}
    for q in (0, -1, 1):
        fibre = HeavyPhysicalFibre(potential, q, link, args.inertia, args.ue4, args.ue1, args.magnetic)
        evidence[str(q)] = dict(centre=fibre.centre_controls(U),
                    centre_compression=fibre.compressed(np.eye(4)),
                    deformed_compression=fibre.compressed(U),
                    retained_slow=fibre.retained_slow_multiplication(U),
                    photon_orientation=fibre.photon_orientation_control())
    # Actual photon onebody owner at phi=pi: the charged light cut is NOT separated.
    hL = potential.model.onebody(potential.fields, np.eye(4), (0., math.pi))[1]
    light = np.linalg.eigvalsh(hL)
    charged_gaps = {str(q): float(light[4 + q] - light[3 + q]) for q in (-1, 1)}
    require_small(max(charged_gaps.values()), 'charged light cut closes at photon pi', 3e-9)
    return dict(scope='Fixed selected-frame continuous-Haar heavy-only physical BO/Schur theorem',
        source_sha256=identities, coefficients=dict(m=.2, r=.12, v=.3, stiffness=2., Jpref=.2,
                       I=args.inertia, UE4=args.ue4, UE1=args.ue1, K=args.magnetic),
        constants=bounds, at_scale=energy_bounds(potential, bounds, args.scale), primitive_evidence=evidence,
        charged_light_cut_gaps_at_phi_pi=charged_gaps,
        heavy_supplier_global_certificate=potential.certificate,
        retained_operator='PH H PH with full light+B+ell plus exact energy-dependent heavy Schur subtraction',
        centre_interpretation='One residual gauge orbit, not four physical vacua or rank-four photon ladder',
        remaining=['Actual-V angular excitation/remainder isolation',
                   'Coupled entire-light/boundary/infinite-loop slow spectral comparison across q=0,+/-1',
                   'Endpoint-dressed current/carrier compression and joined particle interpretation',
                   'Full-field phase and microscopic quantum law/parameter closure'],
        science_executed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale', type=float, default=1e12)
    parser.add_argument('--inertia', type=float, default=2.)
    parser.add_argument('--ue4', type=float, default=.3)
    parser.add_argument('--ue1', type=float, default=.4)
    parser.add_argument('--magnetic', type=float, default=.7)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(args)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
