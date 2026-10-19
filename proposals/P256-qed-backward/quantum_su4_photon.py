#!/usr/bin/env python3
"""Same-H continuous SU4 Gauss plus one genuine compact U1 loop EFT."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg, eigsh

from exact_haar_alignment import ExactHaarAlignment

from quantum_su4_gauss import (CUTOFFS, HaarLink, PauliCode, PhysicalHamiltonian,
                               PrimitiveMatter, RotationBand, matrix_unit_basis,
                               norm, require_small, source_hashes, uniform_data)


def columns(action, x):
    return np.column_stack([action(x[:, j]) for j in range(x.shape[1])])


def invariant(x):
    return dict(norm=norm(x), trace=[float(np.trace(x).real), float(np.trace(x).imag)],
                singular_values=np.linalg.svd(x, compute_uv=False).tolist())


class CompactGaussPhoton:
    """Complete old physical domain tensor ell=-R..R; old blocks are immutable."""

    def __init__(self, base: Any, R, magnetic, alignment=None, alignment_coefficient=0.):
        if int(R) != R or R < 1 or magnetic < 0:
            raise ValueError('Require integer R>=1 and K>=0')
        self.base, self.R, self.magnetic = base, int(R), float(magnetic)
        if not np.isfinite(alignment_coefficient) or alignment_coefficient < 0:
            raise ValueError('Alignment coefficient must be finite and nonnegative')
        if alignment_coefficient and alignment is None:
            raise ValueError('Nonzero alignment requires its exact continuous Haar projection')
        if alignment is not None and (alignment.link is not base.link
                or not np.array_equal(alignment.frame, base.frame)):
            raise ValueError('Alignment must use the same Haar link and physical orbital frame')
        self.alignment_coefficient = float(alignment_coefficient)
        self.alignment_blocks = alignment.blocks if alignment_coefficient and alignment is not None else {}
        self.loop_values = np.arange(-self.R, self.R + 1)
        self.q, self.matter, self.link, self.bd = base.q, base.matter, base.link, base.bd
        self.blocks, self.offset, self.diagonal, self.preconditioner_diagonal = {}, {}, {}, {}
        offset = 0
        for name, old in base.blocks.items():
            block: dict[str, Any] = dict(old, shape=(*old['shape'], len(self.loop_values)))
            self.blocks[name], self.offset[name] = block, offset
            offset += int(np.prod(block['shape']))
            n = self.matter.light_left[block['keep']] - 2
            ell = self.loop_values[None, :]
            # The base already contains UE1/2*(n^2+q^2).
            self.diagonal[name] = base.diagonal[name][:, None] + base.ue1 * (ell**2 + n[:, None] * ell)
            if np.any(-(ell + n[:, None]) + ell + self.q != self.matter.light_number - n[:, None] - 4):
                raise AssertionError('Two-path U1 Gauss law')
            diag = np.broadcast_to(self.diagonal[name][:, None, None, None, :], block['shape']).copy()
            orient = np.full(old['shape'], 2 * base.link.casimir[name] + base.link.casimir[base.boundary_name])
            q2_diagonal = block['Q2'][0].diagonal().real + block['Q2'][1].diagonal().real
            orient += q2_diagonal[:, None, None, None]
            for a, t in enumerate(base.link.reps[name]):
                left = np.real(np.diag(t))[None, :, None, None]
                right = -np.real(np.diag(t))[None, None, :, None]
                qb = np.real(np.diag(base.boundary[a]))[None, None, None, :]
                ql = block['Q'][0][a].diagonal().real[:, None, None, None]
                qr = block['Q'][1][a].diagonal().real[:, None, None, None]
                orient += -2 * (ql * left + qr * right + qb * right) + 2 * qr * qb
            diag += (orient / (2 * base.inertia) + block['onsite'].diagonal().real[:, None, None, None])[..., None]
            self.preconditioner_diagonal[name] = diag
        self.dimension = offset
        # Actual primitive directed products, cached ONCE. B and ell are spectators
        # here, but never merged for boundary-charge orientation actions.
        directed = {}
        for r, s, flavour, mm, lm in base.edges:
            product = sparse.kron(mm, lm, format='csr')
            key = (r, s, flavour)
            directed[key] = directed[key] + product if key in directed else product
        self.directed = [(r, s, f, t, t.getH().tocsr()) for (r, s, f), t in directed.items()]
        for (target, source), potential in self.alignment_blocks.items():
            if not np.array_equal(self.blocks[target]['keep'], self.blocks[source]['keep']):
                raise AssertionError('Centre-neutral alignment changed the matter fibre')
            if target == source:
                size = self.link.reps[target].shape[1]
                self.preconditioner_diagonal[target] += self.alignment_coefficient * (
                    potential.diagonal().real.reshape(size, size)[None, :, :, None, None])
        self.diag_vector = self.pack(self.preconditioner_diagonal)

    def unpack(self, vector):
        return {name: np.asarray(vector[self.offset[name]:self.offset[name] + int(np.prod(b['shape']))]).reshape(b['shape'])
                for name, b in self.blocks.items()}

    def pack(self, blocks):
        return np.concatenate([blocks[n].ravel() for n in self.blocks])

    def orientation(self, name, x, wrong_sign=False):
        base, block = self.base, self.blocks[name]
        rep = base.link.reps[name]
        result = (2 * base.link.casimir[name] + base.link.casimir[base.boundary_name]) * x
        result += sum(base.matter_action(q2, x) for q2 in block['Q2'])
        sign = 1 if wrong_sign else -1
        for a, t in enumerate(rep):
            el = np.einsum('ca,mabzl->mcbzl', t.T, x, optimize=True)
            er = -np.einsum('cb,mabzl->maczl', t, x, optimize=True)
            qb = np.einsum('zw,mabwl->mabzl', base.boundary[a], x, optimize=True)
            result += 2 * sign * (base.matter_action(block['Q'][0][a], el)
                      + base.matter_action(block['Q'][1][a], er)
                      + np.einsum('zw,mabwl->mabzl', base.boundary[a], er, optimize=True))
            result += 2 * base.matter_action(block['Q'][1][a], qb)
        return result / (2 * base.inertia)

    @staticmethod
    def shift(x, direction, wrapping=False):
        if wrapping:
            return np.roll(x, direction, axis=-1)
        y = np.zeros_like(x)
        if direction == 1:
            y[..., 1:] = x[..., :-1]
        elif direction == -1:
            y[..., :-1] = x[..., 1:]
        else:
            raise ValueError('Shift must be +/-1')
        return y

    def loop_action(self, vector, observable='sin', background=0., wrapping=False):
        result = {}
        for name, x in self.unpack(vector).items():
            if observable == 'ell':
                y = x * self.loop_values
            elif observable in ('sin', 'cos'):
                plus = np.exp(1j * background) * self.shift(x, 1, wrapping)
                minus = np.exp(-1j * background) * self.shift(x, -1, wrapping)
                y = (plus - minus) / (2j) if observable == 'sin' else (plus + minus) / 2
            else:
                raise ValueError('Unknown loop observable')
            result[name] = y
        return self.pack(result)

    def density_action(self, vector):
        return self.pack({n: self.matter.light_left[self.blocks[n]['keep']][:, None, None, None, None] * v
                          for n, v in self.unpack(vector).items()})

    def hop_action(self, vector, phases=(0., 0.), left=None, right=None,
                   flavour=None, formal_identity=False, wrapping=False):
        """Hopping, or exact light-only J(left)/C(left,right) at real phases."""
        x = self.unpack(vector)
        result = {n: np.zeros_like(v, dtype=complex) for n, v in x.items()}
        a0, a1 = phases
        factors = [np.exp(1j * a0) / 2, np.exp(-1j * a1) / 2]
        if left is not None:
            factors = [factors[0] * 1j * left[0], factors[1] * -1j * left[1]]
            if right is not None:
                factors = [factors[0] * 1j * right[0], factors[1] * -1j * right[1]]
        for r, s, f, t, adj in self.directed:
            if flavour is not None and flavour != f or f == 0 and left is not None:
                continue
            ys = (t @ x[s].reshape(t.shape[1], -1)).reshape(self.blocks[r]['shape'])
            yr = (adj @ x[r].reshape(t.shape[0], -1)).reshape(self.blocks[s]['shape'])
            if f == 0:
                result[r] += ys
                result[s] += yr
            else:
                result[r] += factors[0] * ys + factors[1] * (ys if formal_identity else self.shift(ys, -1, wrapping))
                result[s] += np.conjugate(factors[0]) * yr + np.conjugate(factors[1]) * (yr if formal_identity else self.shift(yr, 1, wrapping))
        return self.pack(result)

    def _add_alignment(self, source_blocks, target_blocks):
        for (target, source), potential in self.alignment_blocks.items():
            x = source_blocks[source]
            matter, _, _, boundary, loop = x.shape
            size = self.link.reps[target].shape[1]
            columns = x.transpose(1, 2, 0, 3, 4).reshape(potential.shape[1], -1)
            result = (potential @ columns).reshape(size, size, matter, boundary, loop)
            target_blocks[target] += self.alignment_coefficient * result.transpose(2, 0, 1, 3, 4)

    def apply(self, vector, phases=(0., 0.), magnetic_background=None):
        x = self.unpack(vector)
        y = {n: self.orientation(n, v) + self.diagonal[n][:, None, None, None, :] * v
                + self.base.matter_action(self.blocks[n]['onsite'], v) for n, v in x.items()}
        if self.alignment_blocks:
            self._add_alignment(x, y)
        # None is the primary matter-only source: quantum magnetic potential fixed.
        B = 0. if magnetic_background is None else magnetic_background
        return self.pack(y) + self.hop_action(vector, phases) - self.magnetic * self.loop_action(vector, 'cos', B)

    def linear_operator(self):
        factory: Any = LinearOperator
        return factory((self.dimension, self.dimension), matvec=self.apply, rmatvec=self.apply, dtype=complex)

    def operator_evidence(self, seed):
        rng = np.random.default_rng(seed)
        x, y = [rng.normal(size=self.dimension) + 1j * rng.normal(size=self.dimension) for _ in range(2)]
        x /= norm(x); y /= norm(y)
        hx = self.apply(x)
        hermitian = float(abs(np.vdot(x, self.apply(y)) - np.vdot(hx, y)))
        theta, phase = .193, (.137, -.081)
        W = self.pack({n: np.broadcast_to(np.exp(1j * theta * self.matter.light_left[b['keep']])[:, None, None, None, None], b['shape'])
                       for n, b in self.blocks.items()})
        ward = norm(self.apply(x, (phase[0] + theta, phase[1] - theta)) - W * self.apply(W.conj() * x, phase))
        gradient = (1., -1.)
        j = self.hop_action(x, phase, gradient)
        c = self.hop_action(x, phase, gradient, gradient)
        step = 2e-4
        plus = self.apply(x, (phase[0] + step, phase[1] - step))
        minus = self.apply(x, (phase[0] - step, phase[1] + step))
        fdj = norm((plus - minus) / (2 * step) - j)
        fdc = norm((plus - 2 * self.apply(x, phase) + minus) / step**2 - c)
        dx = self.density_action(x)
        comm = self.density_action(hx) - self.apply(dx)
        gradient_j = norm(self.hop_action(x, left=gradient) - 1j * comm)
        double = self.density_action(comm) - (self.density_action(self.apply(dx)) - self.apply(self.density_action(dx)))
        gradient_c = norm(self.hop_action(x, left=gradient, right=gradient) + double)
        orientation_slots = norm(self.pack({n: self.orientation(n, v) - np.stack(
            [self.base.orientation(n, v[..., k]) for k in range(len(self.loop_values))], axis=-1)
            for n, v in self.unpack(x).items()}))
        tensors = self.unpack(x)
        gauss_square = 0.
        gauss_action = {}
        for name, v in tensors.items():
            block = self.blocks[name]
            square_action = np.zeros_like(v)
            for a, t in enumerate(self.link.reps[name]):
                def dl(z):
                    return self.base.matter_action(block['Q'][0][a], z) - np.einsum('ca,mabzl->mcbzl', t.T, z, optimize=True)
                def dr(z):
                    return (self.base.matter_action(block['Q'][1][a], z)
                        + np.einsum('zw,mabwl->mabzl', self.base.boundary[a], z, optimize=True)
                        + np.einsum('cb,mabzl->maczl', t, z, optimize=True))
                left_charge, right_charge = dl(v), dr(v)
                gauss_square += (norm(left_charge)**2 + norm(right_charge)**2) / (2 * self.base.inertia)
                square_action += dl(left_charge) + dr(right_charge)
            gauss_action[name] = square_action / (2 * self.base.inertia) - self.orientation(name, v)
        gauss_factorization = norm(self.pack(gauss_action))
        heavy_current = norm(self.hop_action(x, left=(.31, -.27), flavour=0))
        old_slots = [self.base.unpack(self.base.hop_action(self.base.pack(
            {m: v[..., k] for m, v in tensors.items()}))) for k in range(len(self.loop_values))]
        old = self.pack({n: np.stack([slot[n] for slot in old_slots], axis=-1) for n in self.blocks})
        formal = norm(old - self.hop_action(x, formal_identity=True))
        wrong = norm(self.pack({n: self.orientation(n, v, True) - self.orientation(n, v) for n, v in self.unpack(x).items()}))
        heavy_wrong = self.pack({n: (self.base.ue1 / 2 * ((self.loop_values[None, :] + self.matter.light_left[b['keep']][:, None]
            + self.matter.heavy_left[b['keep']][:, None] - 4)**2 - (self.loop_values[None, :] + self.matter.light_left[b['keep']][:, None] - 2)**2))[:, None, None, None, :] * self.unpack(x)[n] for n, b in self.blocks.items()})
        wrap = norm(self.hop_action(x, wrapping=True) - self.hop_action(x)) + norm(self.loop_action(x, wrapping=True) - self.loop_action(x))
        for value, label in ((hermitian, 'same-H Hermiticity'), (ward, 'gradient covariance'),
                             (gradient_j, 'density J Ward'), (gradient_c, 'density C Ward'),
                             (orientation_slots, 'boundary versus ell'), (formal, 'zero phase formal old hop'),
                             (gauss_factorization, 'full Lie Gauss square factorization'), (heavy_current, 'heavy EM neutrality')):
            require_small(value, label, 3e-8)
        require_small(fdj, 'current finite difference', 2e-7)
        require_small(fdc, 'contact finite difference', 2e-6)
        if min(wrong, norm(heavy_wrong), wrap) < 1e-6:
            raise AssertionError('Actual wrong Gauss/heavy EM/wrapping controls did not discriminate')
        return dict(hermiticity=hermitian, pure_gradient_covariance=ward, current_derivative=fdj,
                    contact_derivative=fdc, density_current_Ward=gradient_j, density_contact_Ward=gradient_c,
                    orientation_matches_individual_ell_slots=orientation_slots, formal_S_identity_old_hop=formal,
                    full_Lie_Gauss_square_residual=gauss_factorization,
                    angular_Gauss_positive_square_expectation=gauss_square,
                    heavy_source_current_norm=heavy_current,
                    wrong_Gauss_defect=wrong, wrong_heavy_EM_defect=norm(heavy_wrong), wrapping_defect=wrap,
                    no_loop_wrapping=True, heavy_EM_neutral=True, complete_old_domain_each_ell=True)


def ritz(model, vectors, tolerance):
    vectors, _ = np.linalg.qr(vectors, mode='reduced')
    actions = columns(model.apply, vectors)
    projected = vectors.conj().T @ actions
    values, rotation = np.linalg.eigh((projected + projected.conj().T) / 2)
    vectors, actions = vectors @ rotation, actions @ rotation
    residuals = np.linalg.norm(actions - vectors * values, axis=0)
    orth = norm(vectors.conj().T @ vectors - np.eye(len(values)))
    if not np.all(np.isfinite(residuals)) or np.max(residuals / np.maximum(1., abs(values))) > tolerance:
        raise RuntimeError('Actual same-H Ritz residual exceeds declared tolerance')
    if orth > 256 * np.finfo(float).eps * len(values):
        raise RuntimeError('Actual same-H Ritz frame is not orthonormal')
    return values, vectors, residuals, orth


def solve_ground(model, levels, tolerance, maxiter, seed, max_ground_rank=32):
    """Independent deflated solves complete the measured multiplet, not a T guess."""
    rng = np.random.default_rng(seed)
    k = min(max(4, levels), model.dimension - 2)
    initial = rng.normal(size=model.dimension) + 1j * rng.normal(size=model.dimension)
    _, vectors = eigsh(model.linear_operator(), k=k, which='SA', v0=initial,
                       tol=tolerance / 20, maxiter=maxiter, ncv=min(model.dimension, max(2*k+1, 24)))
    values, vectors, residuals, orth = ritz(model, vectors, tolerance)
    uncertainty = max(20 * float(residuals.max()), 20 * tolerance * max(1., abs(float(values[0]))))
    rank = int(np.count_nonzero(values - values[0] <= uncertainty))
    if rank == k:
        raise RuntimeError('All computed eigenvectors are ground-degenerate; increase --levels')
    completion = []
    while True:
        ground = vectors[:, :rank]
        shift = max(10., 2 * float(values[-1] - values[0]))
        factory: Any = LinearOperator
        op = factory((model.dimension, model.dimension), dtype=complex,
                     matvec=lambda x: model.apply(x) + shift * ground @ (ground.conj().T @ x))
        start = rng.normal(size=model.dimension) + 1j * rng.normal(size=model.dimension)
        _, candidate = eigsh(op, k=2, which='SA', v0=start, tol=tolerance / 20,
                             maxiter=maxiter, ncv=min(model.dimension, 24))
        projected = candidate - ground @ (ground.conj().T @ candidate)
        candidate_norms = np.linalg.norm(projected, axis=0)
        if np.max(candidate_norms) < .9:
            raise RuntimeError('Deflation shift insufficient to separate actual complement')
        # Select the lowest non-ground candidate, rather than arbitrary ordering.
        choices = []
        for j in range(2):
            if candidate_norms[j] > .9:
                v = projected[:, j] / candidate_norms[j]
                hv = model.apply(v)
                e = float(np.vdot(v, hv).real)
                choices.append((e, v, norm(hv - e * v)))
        ec, chosen, rc = min(choices, key=lambda row: row[0])
        if rc > tolerance * max(1., abs(ec)):
            raise RuntimeError('Measured complement residual insufficient')
        completion.append(dict(candidate_energy=ec, full_H_residual=rc, existing_rank=rank))
        if ec - values[0] > uncertainty:
            gap = min(ec, float(values[rank])) - float(values[0])
            if gap <= 10 * uncertainty:
                raise RuntimeError('Ground/complement separation unresolved at declared residual')
            break
        if rank >= max_ground_rank:
            raise RuntimeError('Ground rank unresolved; increase --max-ground-rank')
        values, vectors, residuals, orth = ritz(model, np.column_stack((vectors, chosen)), tolerance)
        rank = int(np.count_nonzero(values - values[0] <= uncertainty))
        if rank == len(values):
            raise RuntimeError('No computed separated eigenvalue after completion; increase --levels')
    return values, vectors, vectors[:, :rank], gap, dict(
        energies=values.tolist(), full_H_residuals=residuals.tolist(), residual_limit=tolerance,
        residual_scale='max(1,abs(E))', orthogonality=orth, ground_rank=rank,
        measured_complement_gap=gap, degeneracy_resolution=uncertainty,
        independent_deflated_ground_completion=completion,
        interpretation='finite same-H Ritz multiplet and measured complement, not infinite gap')


class FullResolvent:
    def __init__(self, model, energy, ground, gap, tolerance, maxiter):
        self.model, self.energy, self.ground, self.gap = model, energy, ground, gap
        self.tolerance, self.maxiter, self.receipts = tolerance, maxiter, []

    def project(self, x):
        return x - self.ground @ (self.ground.conj().T @ x)

    def solve(self, source, frequency):
        if abs(frequency) >= self.gap:
            raise ValueError('Frequency must be below measured complement gap')
        factory: Any = LinearOperator
        def action(x):
            q = self.project(x)
            return self.project(self.model.apply(q) - (self.energy + frequency) * q) + self.gap * (x - q)
        diagonal = np.maximum(abs(self.model.diag_vector - self.energy - frequency), self.gap - abs(frequency))
        def precondition(x):
            q = self.project(x)
            return self.project(q / diagonal) + (x - q) / self.gap
        A = factory((self.model.dimension, self.model.dimension), matvec=action, dtype=complex)
        M = factory(A.shape, matvec=precondition, dtype=complex)
        rhs = self.project(source)
        answers = []
        for j in range(rhs.shape[1]):
            calls = [0]
            def count(_):
                calls[0] += 1
            answer, status = cg(A, rhs[:, j], M=M, rtol=self.tolerance, atol=0., maxiter=self.maxiter, callback=count)
            residual = norm(action(answer) - rhs[:, j])
            relative = residual / max(norm(rhs[:, j]), np.finfo(float).tiny)
            if status != 0 or relative > 8 * self.tolerance:
                raise RuntimeError(f'Full Gauss+ell complement solve failed: {status}, {relative}')
            answers.append(self.project(answer))
            self.receipts.append(dict(frequency=float(frequency), column=j, iterations=calls[0],
                status=int(status), absolute_residual=residual, relative_residual=relative,
                ground_leakage=norm(self.ground.conj().T @ answer)))
        return np.column_stack(answers)

    def kernel(self, left, right, contact, frequency):
        g = self.ground
        jl, jr = columns(left, g), columns(right, g)
        first = g.conj().T @ columns(left, self.solve(jr, frequency))
        second = g.conj().T @ columns(right, self.solve(jl, -frequency))
        local = g.conj().T @ columns(contact, g)
        lg, rg = g.conj().T @ jl, g.conj().T @ jr
        if frequency == 0:
            if max(norm(lg), norm(rg)) > 1e-6:
                raise RuntimeError('Static ground poles cannot be silently discarded')
            born = np.zeros_like(local)
        else:
            born = (lg @ rg - rg @ lg) / frequency
        return local - first - second + born, local, first + second, born


def photon_krylov(model, seed, ground, energy, steps):
    """Source-active finite projection, with full-H residual of EVERY pole."""
    def project(x):
        return x - ground @ (ground.conj().T @ x)
    rank = ground.shape[1]
    receipts = []
    for column in range(rank):
        source = seed[:, column]
        weight = norm(source)**2 / rank
        if weight <= np.finfo(float).tiny:
            receipts.append(dict(column=column, weight=weight, poles=[]))
            continue
        basis = [source / norm(source)]
        actions = []
        full_actions = []
        for step in range(steps):
            full_action = model.apply(basis[-1]) - energy * basis[-1]
            full_actions.append(full_action)
            action = project(full_action)
            actions.append(action)
            if step == steps - 1:
                break
            q = np.column_stack(basis)
            remainder = action - q @ (q.conj().T @ action)
            remainder -= q @ (q.conj().T @ remainder)
            beta = norm(remainder)
            if beta <= 128 * np.finfo(float).eps * max(1., norm(action)):
                break
            basis.append(remainder / beta)
        q, hq = np.column_stack(basis), np.column_stack(actions)
        projected = q.conj().T @ hq
        values, rotation = np.linalg.eigh((projected + projected.conj().T) / 2)
        weights = weight * abs(rotation[0])**2
        residuals = np.linalg.norm(np.column_stack(full_actions) @ rotation - (q @ rotation) * values, axis=0)
        if np.min(values) <= 0:
            raise RuntimeError('Source-active photon Ritz projection has nonpositive poles')
        receipts.append(dict(column=column, weight=weight,
            projected_zeroth_moment=float(weights.sum()),
            projected_first_moment=float(weights @ values),
            poles=[dict(energy=float(e), weight=float(w), full_H_residual=float(r))
                   for e, w, r in zip(values, weights, residuals)]))
    return dict(finite_source_Krylov_projection=True, requested_steps=steps,
                columns=receipts,
                interpretation='Positive Ritz quadrature, not exact eigenstate poles. Full-action zeroth/first moments fix total source weight; finite quadrature does not resolve the unseen exact spectral measure.')


def photon_response(model, energies, vectors, ground, energy, observable, steps):
    action = lambda x: model.loop_action(x, observable)
    acted = columns(action, ground)
    elastic = ground.conj().T @ acted
    seed = acted - ground @ elastic
    rank = ground.shape[1]
    total, ground_weight = norm(acted)**2 / rank, norm(elastic)**2 / rank
    zeroth = norm(seed)**2 / rank
    hseed = columns(model.apply, seed) - energy * seed
    first = float(np.trace(seed.conj().T @ hseed).real / rank)
    double = float((np.trace(acted.conj().T @ columns(model.apply, acted)).real
                   - np.trace(columns(action, acted).conj().T @ columns(model.apply, ground)).real) / rank)
    poles = []
    visible = 0.
    for j in range(rank, len(energies)):
        weight = float(norm(vectors[:, j].conj() @ seed)**2 / rank)
        visible += weight
        if weight > 1e-14 * max(1., zeroth):
            pole = float(energies[j] - energy)
            if pole <= 0:
                raise AssertionError('Nonpositive same-H inelastic photon pole')
            poles.append(dict(energy=pole, weight=weight,
                              full_H_residual=norm(model.apply(vectors[:, j]) - energies[j] * vectors[:, j])))
    unseen = zeroth - visible
    require_small(abs(total - ground_weight - zeroth), 'photon equal-time closure', 1e-7)
    require_small(abs(first - double), 'photon full-action first sumrule', 2e-6)
    if unseen < -1e-8 or first < -1e-8:
        raise AssertionError('Photon spectral positivity/weight closure failed')
    return dict(observable=observable, finite_low_eigenstate_projection=True, active_poles=poles,
                full_action_zeroth_moment=zeroth, full_action_first_moment=first,
                half_double_commutator=double, energy_sumrule_residual=abs(first-double),
                total_equal_time_weight=total, elastic_weight=ground_weight,
                equal_time_closure_residual=abs(total-ground_weight-zeroth),
                seen_inelastic_weight=visible, unseen_inelastic_weight=float(max(0., unseen)),
                source_active_Krylov=photon_krylov(model, seed, ground, energy, steps),
                interpretation='Positive same-H low eigenstate Lehmann poles; unseen weight retained, no complete spectral claim')


def ground_observables(model, ground):
    rank = ground.shape[1]
    distribution = np.zeros(len(model.loop_values))
    joint = np.zeros((5, len(model.loop_values)))
    colour_joint = {n: np.zeros_like(joint) for n in model.blocks}
    rho = np.zeros((len(model.loop_values), len(model.loop_values)), complex)
    colour = lambda x: model.pack({n: model.link.casimir[n] * v for n, v in model.unpack(x).items()})
    for j in range(rank):
        for name, x in model.unpack(ground[:, j]).items():
            probabilities = np.sum(abs(x)**2, axis=(1, 2, 3)) / rank
            distribution += probabilities.sum(axis=0)
            for row, keep in enumerate(model.blocks[name]['keep']):
                joint[model.matter.light_left[keep]] += probabilities[row]
                colour_joint[name][model.matter.light_left[keep]] += probabilities[row]
            flat = x.reshape(-1, len(model.loop_values))
            rho += flat.T @ flat.conj() / rank
    actions = dict(ell=lambda x: model.loop_action(x, 'ell'), sinB=lambda x: model.loop_action(x, 'sin'),
                   nleft=model.density_action, colour_C2=colour)
    values = {name: columns(action, ground) for name, action in actions.items()}
    means = {name: float(np.trace(ground.conj().T @ v).real / rank) for name, v in values.items()}
    correlations = {}
    for a, b in (('ell', 'nleft'), ('ell', 'colour_C2'), ('sinB', 'nleft'), ('sinB', 'colour_C2')):
        correlations[a + '_' + b] = float(np.trace(values[a].conj().T @ values[b]).real / rank - means[a] * means[b])
    joint_defect = norm(joint - joint.sum(axis=1)[:, None] * distribution[None, :])
    combined_joint = np.stack(list(colour_joint.values()))
    combined_defect = norm(combined_joint - combined_joint.sum(axis=2)[..., None] * distribution[None, None, :])
    sin_operator = np.zeros((len(model.loop_values), len(model.loop_values)), complex)
    for j in range(len(model.loop_values)-1):
        sin_operator[j+1, j], sin_operator[j, j+1] = 1/(2j), -1/(2j)
    sine_values, sine_frame = np.linalg.eigh(sin_operator)
    sine_probabilities = np.real(np.diag(sine_frame.conj().T @ rho @ sine_frame))
    return dict(ensemble='normalized projector onto measured full ground rank',
        ell_values=model.loop_values.tolist(), ell_probabilities=distribution.tolist(),
        sinB_eigenvalues=sine_values.tolist(), sinB_probabilities=sine_probabilities.tolist(),
        ell_nleft_joint_probabilities=joint.tolist(), photon_reduced_purity=float(np.trace(rho @ rho).real),
        ell_nleft_colour_joint_probabilities={n:v.tolist() for n,v in colour_joint.items()},
        ell_joint_matter_colour_independence_defect=combined_defect,
        measured_photon_matter_colour_correlated=bool(combined_defect > 1e-8),
        means=means, connected_correlations=correlations,
        ell_matter_joint_independence_defect=joint_defect,
        measured_ell_matter_correlated=bool(joint_defect > 1e-8),
        ell_cutoff_shell_probability=float(distribution[0]+distribution[-1]))


def matter_response(model, ground, energy, gap, args):
    gradient, curl = np.array((1., -1.)), np.array((1., 1.))
    J = lambda source: lambda x: model.hop_action(x, left=source)
    C = lambda left, right: lambda x: model.hop_action(x, left=left, right=right)
    solver = FullResolvent(model, energy, ground, gap, args.resolvent_tol, args.maxiter)
    ward = solver.kernel(J(gradient), J(gradient), C(gradient, gradient), 0.)
    error, contact, pair = norm(ward[0]), norm(ward[1]), norm(ward[2])
    numerical = max(args.identity_tol, 100 * error)
    if error > args.identity_tol or min(contact, pair) <= numerical:
        raise AssertionError('Complete same-H Ward/contact-pair cancellation failed or nondiscriminating')
    density = columns(model.density_action, ground)
    exact = -1j * solver.project(density)
    actual = solver.solve(columns(J(gradient), ground), 0.)
    identity = norm(columns(model.apply, exact) - energy * exact - columns(J(gradient), ground))
    agreement = norm(exact - actual)
    require_small(identity, 'pure-gradient exact density solution', args.identity_tol)
    require_small(agreement, 'full solve versus exact density solution', args.identity_tol)
    frequency = args.gap_fraction * gap
    sources = [gradient, curl]
    sumrules = []
    for source in sources:
        action = J(source)
        acted = columns(action, ground)
        elastic = ground.conj().T @ acted
        inelastic = solver.project(acted)
        total = acted.conj().T @ acted
        ground_weight = elastic.conj().T @ elastic
        complement_weight = inelastic.conj().T @ inelastic
        moment = inelastic.conj().T @ (columns(model.apply, inelastic) - energy * inelastic)
        hground = columns(model.apply, ground)
        second_action = columns(action, acted)
        double = acted.conj().T @ columns(model.apply, acted) - (
            second_action.conj().T @ hground + hground.conj().T @ second_action) / 2
        closure = norm(total - ground_weight - complement_weight)
        first_error = norm(moment - double)
        require_small(closure, 'matter equal-time sumrule', args.identity_tol)
        require_small(first_error, 'matter first-moment sumrule', args.identity_tol)
        sumrules.append(dict(source=source.tolist(), full_action_total=invariant(total),
            elastic=invariant(ground_weight), complete_complement_zeroth=invariant(complement_weight),
            complete_complement_first=invariant(moment), half_double_commutator=invariant(double),
            closure_residual=closure, first_sumrule_residual=first_error))
    kernel = np.zeros((2, 2, ground.shape[1], ground.shape[1]), complex)
    entries = []
    for a, left in enumerate(sources):
        for b, right in enumerate(sources):
            result = solver.kernel(J(left), J(right), C(left, right), frequency)
            kernel[a, b] = result[0]
            entries.append(dict(left=a, right=b, kernel=invariant(result[0]),
                                contact=invariant(result[1]), pair=invariant(result[2]), born=invariant(result[3])))
    if norm(kernel) <= args.identity_tol:
        raise AssertionError('Chosen independent two-source kernel is numerically trivial')
    return dict(source_owner='Light matter Peierls derivatives only; dynamical -K cosB unchanged',
                source_vectors=[s.tolist() for s in sources], source_meaning=['gradient', 'curl'],
                finite_external_source=True, on_shell_photon_S_matrix=False,
                frequency=frequency, measured_gap=gap, entries=entries, whole_kernel_norm=norm(kernel),
                pure_gradient=dict(kernel_norm=error, contact_norm=contact, pair_norm=pair,
                    exact_density_solution_residual=identity, solve_agreement=agreement,
                    omitted_contact_wrong_norm=pair, frozen_ground_intermediates_wrong_norm=contact,
                    omitted_contact_breaks_Ward=bool(pair > numerical), frozen_ground_breaks_Ward=bool(contact > numerical)),
                preconditioner='full diagonal H, projected on both sides; positive absolute shifted diagonal',
                full_action_source_sumrules=sumrules,
                complete_domain_dimension=model.dimension, resolvent_receipts=solver.receipts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoffs', nargs='+', choices=tuple(CUTOFFS), default=['minimal'])
    parser.add_argument('--R', nargs='+', type=int, default=[1])
    parser.add_argument('--numbers', nargs='+', type=int, choices=(3,4,5), default=[3,4,5])
    parser.add_argument('--lambdas', nargs='+', type=float, default=[1.])
    parser.add_argument('--inertias', nargs='+', type=float, default=[2.])
    parser.add_argument('--heavy-sign', type=int, choices=(-1,1), default=-1)
    parser.add_argument('--Jpref', dest='alignment', type=float, default=0.)
    for name, default in (('ue4', .3), ('ue1', .4), ('magnetic', .7), ('stiffness', 2.),
                          ('m', .2), ('r', .12), ('v', .3)):
        parser.add_argument('--'+name, type=float, default=default)
    for name in ('m', 'r', 'v'):
        parser.add_argument('--heavy-'+name, type=float)
    parser.add_argument('--levels', type=int, default=8)
    parser.add_argument('--max-ground-rank', type=int, default=32)
    parser.add_argument('--tol', type=float, default=1e-10)
    parser.add_argument('--photon-steps', type=int, default=12)
    parser.add_argument('--resolvent-tol', type=float, default=1e-9)
    parser.add_argument('--identity-tol', type=float, default=2e-6)
    parser.add_argument('--gap-fraction', type=float, default=.2)
    parser.add_argument('--maxiter', type=int, default=6000)
    parser.add_argument('--seed', type=int, default=222)
    parser.add_argument('--operators-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.stiffness, args.tol, args.resolvent_tol, args.identity_tol) <= 0 or not 0 < args.gap_fraction < 1:
        parser.error('Positive tolerances/stiffness and 0<gap-fraction<1 required')
    if args.levels < 4 or args.maxiter < 1 or args.max_ground_rank < 1 or args.photon_steps < 1:
        parser.error('levels>=4, maxiter>=1, max-ground-rank>=1 required')
    if not np.isfinite(args.alignment) or args.alignment < 0:
        parser.error('Jpref must be finite and nonnegative')
    band = RotationBand(a=1., m=args.m, r=args.r, v=args.v, improved=False)
    heavy_band = RotationBand(a=1., m=args.m if args.heavy_m is None else args.heavy_m,
        r=args.r if args.heavy_r is None else args.heavy_r, v=args.v if args.heavy_v is None else args.heavy_v, improved=False)
    data = uniform_data(band, 3, args.stiffness, 0., matrix_unit_basis())
    result: dict[str, Any] = dict(model='Two-cell actual continuous SU4 Gauss + compact U1 two-path EFT',
        parameters={k:v for k,v in vars(args).items() if k!='output'},
        hypotheses=['Fundamental simultaneous adjoint orbit declared EFT, not full60 field quantum phase',
            'Two parallel EM paths; colour U and U dagger tied by explicit flat-colour-loop EFT constraint',
            'One colour canonical momentum; all native physical states retained for every independent ell',
            'Primitive Pauli fullCAR/code and neutral boundary marker unchanged',
            'One colour bond has exact continuous Haar alignment Lambda*Jpref*W/8; Jpref=0 omits it'],
        source_interpretation='PRIMARY independent finite external light-matter sources; quantum magnetic potential stays dynamical and source-independent. Optional apply magnetic_background explicitly chooses additional external curl.',
        excluded_claims=['3D photon', 'Lorentz invariance', 'on-shell S matrix', 'recoil scattering', 'full phase', 'canonical QED'],
        selected_frame_real=data['frame'].real.tolist(), selected_frame_imaginary=data['frame'].imag.tolist(),
        selected_gram=np.diag(data['selected']['G']).tolist(), links=[], alignments=[], runs=[], charged_differences=[])
    code = PauliCode()
    for cutoff in args.cutoffs:
        link = HaarLink(cutoff)
        link_evidence = link.evidence()
        alignment = ExactHaarAlignment(link, data['frame']) if args.alignment else None
        link_evidence.update(alignment_J=args.alignment,
            alignment_scope='exact projected continuous W; Lambda*Jpref/8 per run' if alignment is not None else 'omitted, no compressed PUP product')
        result['links'].append(link_evidence)
        if alignment is not None:
            result['alignments'].append(alignment.evidence())
        matters = {number:PrimitiveMatter(code, number, link.T) for number in args.numbers}
        for inertia in args.inertias:
            for heavy in args.lambdas:
                for R in args.R:
                    energies_by_number = {}
                    for number, matter in matters.items():
                        base = PhysicalHamiltonian(matter, link, data['frame'], band, heavy, inertia,
                            args.ue4, args.ue1, heavy_sign=args.heavy_sign, heavy_band=heavy_band)
                        model = CompactGaussPhoton(base, R, args.magnetic, alignment, heavy * args.alignment / 8)
                        run = dict(cutoff=cutoff, R=R, Lambda=heavy, inertia=inertia, NH=4, NL=number,
                            charge=model.q, dimension=model.dimension, base_dimension=base.dimension,
                            boundary_rep=base.boundary_name, boundary_dimension=base.bd,
                            Jpref=args.alignment, alignment_coefficient=model.alignment_coefficient,
                            tensor_shapes={n:list(b['shape']) for n,b in model.blocks.items()},
                            operators=model.operator_evidence(args.seed))
                        if not args.operators_only:
                            values, vectors, ground, gap, receipt = solve_ground(model, args.levels, args.tol,
                                args.maxiter, args.seed, args.max_ground_rank)
                            energy = float(values[0])
                            energies_by_number[number] = energy
                            run.update(spectrum=receipt, ground_photon=ground_observables(model, ground),
                                photon_response=[photon_response(model, values, vectors, ground, energy, source, args.photon_steps)
                                                 for source in ('ell', 'sin')],
                                matter_current_response=matter_response(model, ground, energy, gap, args))
                        result['runs'].append(run)
                    if 4 in energies_by_number:
                        for number in (3,5):
                            if number in energies_by_number:
                                result['charged_differences'].append(dict(cutoff=cutoff,R=R,Lambda=heavy,inertia=inertia,Jpref=args.alignment,
                                    NL=number,charge=number-4,neutral_energy=energies_by_number[4],
                                    charged_energy=energies_by_number[number],
                                    finite_energy_difference=energies_by_number[number]-energies_by_number[4],
                                    interpretation='Actual same-parameter finite coupled-H ground Ritz energy difference, not positive infinite-gap theorem'))
    result['source_sha256'] = source_hashes()
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.write_text(payload+'\n')
    print(payload)


if __name__ == '__main__':
    main()
