#!/usr/bin/env python3
"""Two-cell selected-orbit SU(4) Gauss EFT, with actual Pauli matter.

The finite matrices are exact Haar Galerkin matrices, not group quadrature.
Higgs alignment is explicitly absent (J=0); see quantum_su4_gauss.md.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator, eigsh

from local_orbital_stability import scalar_paths, uniform_data
from quantum_spinor_plaquette import PauliCode
from rotation_band import RotationBand
from sea_selection import matrix_unit_basis


CUTOFFS = {'minimal': ('1', '4', 'bar4'),
           'medium': ('1', '4', 'bar4', '6', '15'),
           'full': ('1', '4', 'bar4', '6', '10', 'bar10', '15')}


def norm(x):
    if sparse.issparse(x):
        return float(np.linalg.norm(x.data))
    return float(np.linalg.norm(x))


def sparse_sum(terms) -> Any:
    """Sum a nonempty operator family without the integer-zero type ambiguity."""
    iterator = iter(terms)
    result = next(iterator)
    for term in iterator:
        result = result + term
    return result


def require_small(value, name, tolerance=2e-10):
    if value > tolerance:
        raise AssertionError(f'{name}: residual {value} exceeds {tolerance}')


def tensor_basis(symmetric):
    pairs = list(itertools.combinations_with_replacement(range(4), 2) if symmetric
                 else itertools.combinations(range(4), 2))
    result = np.zeros((4, 4, len(pairs)), complex)
    for k, (i, j) in enumerate(pairs):
        result[i, j, k] = 1 if i == j else 1 / np.sqrt(2)
        if i != j:
            result[j, i, k] = (1 if symmetric else -1) / np.sqrt(2)
    return result, pairs


def epsilon(indices):
    if len(set(indices)) != 4:
        return 0
    return (-1) ** sum(indices[i] > indices[j] for i in range(4) for j in range(i + 1, 4))


class HaarLink:
    """Peter-Weyl basis sqrt(d) D_ab; generators act on FUNCTION coefficients."""

    def __init__(self, cutoff):
        self.names = CUTOFFS[cutoff]
        self.T = matrix_unit_basis() / np.sqrt(8)
        self.f = np.real(np.array([[-2j * np.einsum('cij,ji->c', self.T,
                             a @ b - b @ a) for b in self.T] for a in self.T]))
        self.wedge, self.pairs = tensor_basis(False)
        self.sym, _ = tensor_basis(True)
        def two_rep(basis):
            flat = basis.reshape(16, -1)
            return np.array([flat.conj().T @ (np.kron(t, np.eye(4)) +
                             np.kron(np.eye(4), t)) @ flat for t in self.T])
        adjoint = np.array([2 * np.einsum('bij,ji->b', self.T,
                                        t @ c - c @ t) for t in self.T for c in self.T])
        adjoint = adjoint.reshape(15, 15, 15).transpose(0, 2, 1)
        symrep = two_rep(self.sym)
        self.reps = {'1': np.zeros((15, 1, 1), complex), '4': self.T,
                     'bar4': -self.T.conj(), '6': two_rep(self.wedge),
                     '10': symrep, 'bar10': -symrep.conj(), '15': adjoint}
        self.centre = {'1': 0, '4': 1, 'bar4': 3, '6': 2,
                       '10': 2, 'bar10': 2, '15': 0}
        self.casimir = {name: float(np.trace(sum(t @ t for t in rep)).real / rep.shape[1])
                        for name, rep in self.reps.items()}
        self.cg = {}
        self.cg['4', '1'] = np.eye(4).reshape(4, 1, 4)
        self.cg['6', '4'] = self.wedge
        self.cg['10', '4'] = self.sym
        self.cg['1', 'bar4'] = np.eye(4).reshape(4, 4, 1) / 2
        self.cg['15', 'bar4'] = np.sqrt(2) * self.T.transpose(1, 2, 0)
        self.cg['bar4', '6'] = np.array([[[epsilon((d, i, a, b)) / np.sqrt(3)
                                           for d in range(4)] for a, b in self.pairs]
                                         for i in range(4)], complex)
        self.cg['bar4', 'bar10'] = np.sqrt(2 / 5) * self.sym.transpose(0, 2, 1)
        self.cg['4', '15'] = np.sqrt(8 / 15) * self.T.transpose(1, 0, 2)
        self.offset = {}
        offset = 0
        for name in self.names:
            self.offset[name] = offset
            offset += self.reps[name].shape[1] ** 2
        self.dimension = offset
        self.U: list[list[Any]] = [[sparse.csr_matrix((offset, offset), dtype=complex)
                                   for _ in range(4)] for _ in range(4)]
        self.blocks = {}
        for (target, source), cg in self.cg.items():
            if target not in self.names or source not in self.names:
                continue
            ds, dr = self.reps[source].shape[1], self.reps[target].shape[1]
            blocks = [[sparse.csr_matrix(np.sqrt(ds / dr) *
                       np.einsum('ca,db->abcd', cg[i], cg[j].conj()).reshape(dr**2, ds**2))
                       for j in range(4)] for i in range(4)]
            self.blocks[target, source] = blocks
            for i in range(4):
                for j in range(4):
                    coo = blocks[i][j].tocoo()
                    self.U[i][j] += sparse.coo_matrix((coo.data,
                        (coo.row + self.offset[target], coo.col + self.offset[source])),
                        shape=(offset, offset)).tocsr()
        self.left: list[Any] = [sparse.block_diag([
            sparse.kron(sparse.csr_matrix(self.reps[n][a].T),
                        sparse.eye(self.reps[n].shape[1], format='csr'), format='csr')
            for n in self.names], format='csr') for a in range(15)]
        self.right: list[Any] = [sparse.block_diag([
            sparse.kron(sparse.eye(self.reps[n].shape[1], format='csr'),
                        sparse.csr_matrix(-self.reps[n][a]), format='csr')
            for n in self.names], format='csr') for a in range(15)]

    def evidence(self):
        lie = hermitian = casimir = intertwiner = isometry = centre = 0.
        centre_log = np.pi / 2 * np.diag([1., 1., 1., -3.])
        centre_coordinates = 2 * np.einsum('aij,ji->a', self.T, centre_log)
        for name, rep in self.reps.items():
            identity = np.eye(rep.shape[1])
            casimir = max(casimir, norm(sum(t @ t for t in rep) - self.casimir[name] * identity))
            centre = max(centre, norm(expm(1j * np.einsum('a,aij->ij', centre_coordinates, rep)) -
                                      1j**self.centre[name] * identity))
            for a, ta in enumerate(rep):
                hermitian = max(hermitian, norm(ta - ta.conj().T))
                for b, tb in enumerate(rep):
                    lie = max(lie, norm(ta @ tb - tb @ ta -
                                       1j * np.einsum('c,cij->ij', self.f[a, b], rep)))
        for (r, s), c in self.cg.items():
            cflat = c.reshape(4 * self.reps[s].shape[1], self.reps[r].shape[1])
            isometry = max(isometry, norm(cflat.conj().T @ cflat - np.eye(cflat.shape[1])))
            centre = max(centre, abs(1j**(1 + self.centre[s]) - 1j**self.centre[r]))
            for a in range(15):
                product = np.kron(self.T[a], np.eye(self.reps[s].shape[1])) + np.kron(np.eye(4), self.reps[s][a])
                intertwiner = max(intertwiner, norm(product @ cflat - cflat @ self.reps[r][a]))
        multiplication = left_right = adjoints = electric_lie = 0.
        for a in range(15):
            for i in range(4):
                for j in range(4):
                    u = self.U[i][j]
                    multiplication = max(multiplication,
                        norm(self.left[a] @ u - u @ self.left[a] -
                             sum(self.T[a, i, k] * self.U[k][j] for k in range(4))),
                        norm(self.right[a] @ u - u @ self.right[a] +
                             sum(self.U[i][k] * self.T[a, k, j] for k in range(4))))
                    # Hermitian E implies [E,U^dagger]=-[E,U]^dagger.
                    adjoints = max(adjoints, norm(self.left[a] @ u.getH() - u.getH() @ self.left[a] +
                                                 (self.left[a] @ u - u @ self.left[a]).getH()))
            for b in range(15):
                left_right = max(left_right, norm(self.left[a] @ self.right[b] - self.right[b] @ self.left[a]))
                for operators in (self.left, self.right):
                    electric_lie = max(electric_lie, norm(operators[a] @ operators[b] -
                        operators[b] @ operators[a] +
                        1j * sparse_sum(self.f[a, b, c] * operators[c] for c in range(15))))
        for value, label in ((lie, 'Lie representations'), (hermitian, 'Hermitian representations'),
                             (casimir, 'Casimirs'), (intertwiner, 'CG intertwining'),
                             (isometry, 'CG isometry'), (centre, 'CG centres'),
                             (multiplication, 'electric multiplication commutators'),
                             (adjoints, 'adjoint commutators'), (left_right, 'left/right commute'),
                             (electric_lie, 'electric negative Lie algebra')):
            require_small(value, label)
        # This is deliberately NONZERO: PUP is not unitary at a representation boundary.
        compression_defect = max(norm(sum(self.U[k][i].getH() @ self.U[k][j] for k in range(4)) -
                                      (sparse.eye(self.dimension) if i == j else
                                       sparse.csr_matrix((self.dimension, self.dimension))))
                                 for i in range(4) for j in range(4))
        if compression_defect < 1e-5:
            raise AssertionError('Adverse compressed-unitarity control unexpectedly vanished')
        return dict(cutoff=list(self.names), dimension=self.dimension,
                    dimensions={n: self.reps[n].shape[1] for n in self.names},
                    casimirs=self.casimir, lie_residual=lie, hermiticity_residual=hermitian,
                    electric_lie_residual=electric_lie,
                    casimir_residual=casimir, cg_intertwiner_residual=intertwiner,
                    cg_isometry_residual=isometry, centre_residual=centre,
                    multiplication_commutator_residual=multiplication,
                    adjoint_commutator_residual=adjoints, left_right_residual=left_right,
                    compressed_unitarity_defect=compression_defect,
                    haar_inner_product='exact Schur delta_rr delta_ac delta_bd; no quadrature',
                    alignment_J=0., alignment_scope='omitted, not factorized compressed U')


class PrimitiveMatter:
    """Modes 0:4 LH, 4:8 LL, 8:12 RH, 12:16 RL; unchanged code S=+."""

    def __init__(self, code, light_number, generators):
        heavy_modes = tuple(range(4)) + tuple(range(8, 12))
        light_modes = tuple(range(4, 8)) + tuple(range(12, 16))
        self.masks = [sum(1 << i for i in h + l)
                      for h in itertools.combinations(heavy_modes, 4)
                      for l in itertools.combinations(light_modes, light_number)]
        self.index = {mask: i for i, mask in enumerate(self.masks)}
        self.dimension = len(self.masks)
        self.light_number = light_number
        self.count_left = np.array([(mask & 255).bit_count() for mask in self.masks])
        self.light_left = np.array([((mask >> 4) & 15).bit_count() for mask in self.masks])
        self.heavy_left = np.array([(mask & 15).bit_count() for mask in self.masks])
        self.units = {}
        for cell in range(2):
            for flavour in range(2):
                for i in range(4):
                    for j in range(4):
                        self.units[cell, flavour, i, j] = self.transfer(code, 8 * cell + 4 * flavour + j,
                                                                                  8 * cell + 4 * flavour + i)
        self.forward = {(f, i, j): self.transfer(code, 8 + 4 * f + j, 4 * f + i)
                        for f in range(2) for i in range(4) for j in range(4)}
        self.Q = [[sparse_sum(t[i, j] * self.units[cell, f, i, j]
                       for f in range(2) for i in range(4) for j in range(4))
                   for t in generators] for cell in range(2)]
        self.Q2 = [sparse_sum(q @ q for q in qs).tocsr() for qs in self.Q]

    def transfer(self, code, source, target):
        rows, cols, values = [], [], []
        for col, mask in enumerate(self.masks):
            if not mask >> source & 1:
                continue
            if source == target:
                row, value = col, 1.
            elif mask >> target & 1:
                continue
            else:
                new, value = code.transfer(mask, source, target, sign=1)
                row = self.index[new]
            rows.append(row); cols.append(col); values.append(value)
        return sparse.coo_matrix((values, (rows, cols)), shape=(self.dimension, self.dimension), dtype=complex).tocsr()

    def algebra_evidence(self, f):
        residual = 0.
        for qs in self.Q:
            for a in range(15):
                for b in range(15):
                    residual = max(residual, norm(qs[a] @ qs[b] - qs[b] @ qs[a] -
                                                  1j * sum(f[a, b, c] * qs[c] for c in range(15))))
        require_small(residual, 'actual Pauli local Lie algebra', 2e-8)
        return residual


class PhysicalHamiltonian:
    """Complete fixed-number/centre fibre; matrix-free in sparse tensor blocks."""

    def __init__(self, matter, link, frame, band, heavy_scale, inertia, ue4, ue1,
                 heavy_sign=-1, heavy_band=None):
        if min(heavy_scale, inertia, ue4, ue1) <= 0:
            raise ValueError('Lambda, I, UE4 and UE_U1 must be positive')
        if heavy_sign not in (-1, 1):
            raise ValueError('Heavy whole-H polarity must be -1 or +1')
        self.matter, self.link = matter, link
        self.frame = frame
        self.inertia, self.ue4, self.ue1 = inertia, ue4, ue1
        self.heavy_scale = heavy_scale
        self.heavy_sign = heavy_sign
        self.heavy_band = band if heavy_band is None else heavy_band
        self.heavy_multiplier = heavy_sign * heavy_scale
        self.q = matter.light_number - 4
        self.boundary_name = {0: '1', -1: '4', 1: 'bar4'}[self.q]
        self.boundary = link.reps[self.boundary_name]
        self.bd = self.boundary.shape[1]
        self.bc = link.centre[self.boundary_name]
        self.blocks, self.offset = {}, {}
        offset = 0
        for name in link.names:
            n = link.centre[name]
            keep = np.flatnonzero(matter.count_left % 4 == n)
            right = 4 + matter.light_number - matter.count_left[keep]
            if np.any((right + self.bc + n) % 4):
                raise AssertionError('Right centre restriction does not match charged boundary')
            if not len(keep):
                continue
            d = link.reps[name].shape[1]
            shape = (len(keep), d, d, self.bd)
            self.offset[name] = offset
            self.blocks[name] = dict(keep=keep, shape=shape,
                Q=[[q[keep][:, keep].tocsr() for q in qs] for qs in matter.Q],
                Q2=[q[keep][:, keep].tocsr() for q in matter.Q2])
            offset += int(np.prod(shape))
        self.dimension = offset
        # One active x-axis, not either supplier's three-axis Wilson count.
        onsites, endpoints = [], []
        for flavour_band in (self.heavy_band, band):
            onsite, paths = scalar_paths(flavour_band)
            onsite -= 2 * flavour_band.r / flavour_band.a
            xpaths = [s for displacement, s in paths if tuple(displacement) == (1, 0, 0)]
            if len(xpaths) != 1 or flavour_band.improved:
                raise ValueError('This consumer uses unimproved one-bond nearest x paths')
            onsites.append(onsite)
            endpoints.append(np.einsum('m,mij->ij', xpaths[0], frame))
        self.onsite_scalar = onsites[1]
        self.heavy_onsite_scalar = onsites[0]
        multipliers = (self.heavy_multiplier, 1.)
        self.onsite = sparse_sum(multipliers[f] * onsites[f] * frame[0, i, j] *
                          matter.units[c, f, i, j] for c in range(2) for f in range(2)
                          for i in range(4) for j in range(4)).tocsr()
        self.edges = []
        for (r, s), u in link.blocks.items():
            if r not in self.blocks or s not in self.blocks:
                continue
            left, right = self.blocks[r]['keep'], self.blocks[s]['keep']
            for flavour in range(2):
                endpoint = endpoints[flavour]
                for i in range(4):
                    for j in range(4):
                        mm = matter.forward[flavour, i, j][left][:, right].tocsr()
                        lm = sparse_sum((endpoint[i, k] * u[k][j] + u[i][k] * endpoint[k, j]) / 2
                                        for k in range(4)).tocsr()
                        if mm.nnz and lm.nnz:
                            self.edges.append((r, s, flavour, mm, lm * multipliers[flavour]))
        self.diagonal = {}
        for name, block in self.blocks.items():
            keep = block['keep']
            block['onsite'] = self.onsite[keep][:, keep].tocsr()
            e = matter.light_left[keep] - 2
            # Oriented L->R tree field E=e; -E+q=n_L(R)-2 at R.
            if np.any(-e + self.q != matter.light_number - matter.light_left[keep] - 2):
                raise AssertionError('U1 tree Gauss law')
            self.diagonal[name] = ue4 * link.casimir[name] + ue1 * (e**2 + self.q**2) / 2

    def unpack(self, vector):
        return {name: np.asarray(vector[self.offset[name]:self.offset[name] + int(np.prod(block['shape']))]).reshape(block['shape'])
                for name, block in self.blocks.items()}

    def pack(self, blocks):
        return np.concatenate([blocks[name].ravel() for name in self.blocks])

    @staticmethod
    def matter_action(matrix, x):
        return (matrix @ x.reshape(x.shape[0], -1)).reshape((matrix.shape[0], *x.shape[1:]))

    def orientation(self, name, x, wrong_sign=False):
        block = self.blocks[name]
        rep = self.link.reps[name]
        c = self.link.casimir[name]
        cb = self.link.casimir[self.boundary_name]
        result = (2 * c + cb) * x + sum(self.matter_action(q2, x) for q2 in block['Q2'])
        sign = 1 if wrong_sign else -1
        for a, t in enumerate(rep):
            el = np.einsum('ca,mabz->mcbz', t.T, x, optimize=True)
            er = -np.einsum('cb,mabz->macz', t, x, optimize=True)
            qb = np.einsum('zw,mabw->mabz', self.boundary[a], x, optimize=True)
            result += 2 * sign * (self.matter_action(block['Q'][0][a], el) +
                        self.matter_action(block['Q'][1][a], er) +
                        np.einsum('zw,mabw->mabz', self.boundary[a], er, optimize=True))
            result += 2 * self.matter_action(block['Q'][1][a], qb)
        return result / (2 * self.inertia)

    def hop_action(self, vector, phase=0., derivative=0):
        x = self.unpack(vector)
        result = {n: np.zeros_like(v) for n, v in x.items()}
        for r, s, flavour, mm, lm in self.edges:
            factor = (np.exp(1j * phase) * (1j ** derivative)) if flavour == 1 else (1 if derivative == 0 else 0)
            if not factor:
                continue
            # Tensor factors commute. No full physical sparse matrix is assembled.
            y = self.matter_action(mm, x[s])
            m, _, _, b = y.shape
            dr = self.link.reps[r].shape[1]
            result[r] += factor * (lm @ y.reshape(m, -1, b).transpose(1, 0, 2).reshape(lm.shape[1], -1)).reshape(dr, dr, m, b).transpose(2, 0, 1, 3)
            y = self.matter_action(mm.getH(), x[r])
            m, _, _, b = y.shape
            ds = self.link.reps[s].shape[1]
            result[s] += np.conjugate(factor) * (lm.getH() @ y.reshape(m, -1, b).transpose(1, 0, 2).reshape(lm.shape[0], -1)).reshape(ds, ds, m, b).transpose(2, 0, 1, 3)
        return self.pack(result)

    def apply(self, vector, phase=0.):
        x = self.unpack(vector)
        result = {}
        for name, v in x.items():
            block = self.blocks[name]
            result[name] = self.orientation(name, v) + self.diagonal[name][:, None, None, None] * v + self.matter_action(block['onsite'], v)
        return self.pack(result) + self.hop_action(vector, phase)

    def linear_operator(self):
        # SciPy's generic base-class stub omits the documented factory kwargs.
        operator_factory: Any = LinearOperator
        return operator_factory((self.dimension, self.dimension), matvec=self.apply,
                                rmatvec=self.apply, dtype=np.complex128)

    def operator_evidence(self, seed=222):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=self.dimension) + 1j * rng.normal(size=self.dimension)
        y = rng.normal(size=self.dimension) + 1j * rng.normal(size=self.dimension)
        x /= norm(x); y /= norm(y)
        hx, hy = self.apply(x), self.apply(y)
        hermitian = abs(np.vdot(x, hy) - np.vdot(hx, y))
        theta, phase = .193, .137
        w = self.pack({n: np.exp(1j * theta * self.matter.light_left[b['keep']])[:, None, None, None] * np.ones(b['shape'])
                       for n, b in self.blocks.items()})
        ward = norm(self.apply(x, phase + theta) - w * self.apply(np.conjugate(w) * x, phase))
        step = 1e-4
        current = self.hop_action(x, phase, 1)
        contact = self.hop_action(x, phase, 2)
        fdcurrent = norm((self.hop_action(x, phase + step) - self.hop_action(x, phase - step)) / (2 * step) - current)
        fdcontact = norm((self.hop_action(x, phase + step) - 2 * self.hop_action(x, phase) + self.hop_action(x, phase - step)) / step**2 - contact)
        wrong = self.pack({n: self.orientation(n, v, True) - self.orientation(n, v) for n, v in self.unpack(x).items()})
        wrong_gauss = norm(wrong)
        heavy_wrong = self.pack({n: (self.ue1 / 2 * ((self.matter.light_left[b['keep']] + self.matter.heavy_left[b['keep']] - 4)**2 -
                                                    (self.matter.light_left[b['keep']] - 2)**2))[:, None, None, None] * self.unpack(x)[n]
                                for n, b in self.blocks.items()})
        heavy_defect = norm(heavy_wrong)
        missing = sum(len(b['keep']) for n, b in self.blocks.items()
                      if np.any((4 + self.matter.light_number - self.matter.count_left[b['keep']] + self.link.centre[n]) % 4))
        require_small(hermitian, 'physical Hamiltonian Hermiticity', 2e-8)
        require_small(ward, 'actual pure-gradient operator covariance', 2e-8)
        if wrong_gauss < 1e-5 or heavy_defect < 1e-5 or (self.q and not missing):
            raise AssertionError('An actual adverse Gauss/EM/boundary control vanished')
        return dict(hermiticity_residual=float(hermitian), pure_gradient_covariance_residual=ward,
                    same_H_current_finite_difference_residual=fdcurrent,
                    same_H_contact_finite_difference_residual=fdcontact,
                    wrong_orientation_gauss_sign_defect=wrong_gauss,
                    wrong_heavy_EM_count_defect=heavy_defect,
                    missing_boundary_rejected_matter_fibres=missing,
                    current_contact_owner='analytic derivatives of this exact light Peierls hopping',
                    full_resolvent_ward='not computed; exact operator covariance instead')

    def spectrum(self, levels, tolerance, maxiter, seed):
        k = min(levels, self.dimension - 2)
        if k < 2:
            raise ValueError('Request at least two eigenlevels')
        rng = np.random.default_rng(seed)
        initial = rng.normal(size=self.dimension) + 1j * rng.normal(size=self.dimension)
        operator = self.linear_operator()
        values, vectors = eigsh(operator, k=k, which='SA', tol=tolerance,
                                maxiter=maxiter, v0=initial, ncv=min(self.dimension, max(2 * k + 1, 20)))
        # Reorthogonalize the returned subspace before its Hermitian Ritz solve.
        vectors, _ = np.linalg.qr(vectors, mode='reduced')
        actions = np.column_stack([self.apply(vectors[:, i]) for i in range(k)])
        projected = vectors.conj().T @ actions
        values, rotation = np.linalg.eigh((projected + projected.conj().T) / 2)
        vectors, actions = vectors @ rotation, actions @ rotation
        residuals = np.linalg.norm(actions - vectors * values, axis=0)
        relative_residuals = residuals / np.maximum(1., np.abs(values))
        orthogonality = norm(vectors.conj().T @ vectors - np.eye(k))
        orthogonality_limit = 128 * np.finfo(float).eps * k
        if (not np.all(np.isfinite(relative_residuals))
                or float(relative_residuals.max()) > tolerance
                or orthogonality > orthogonality_limit):
            raise RuntimeError('Physical eigenpair residual or orthogonality exceeds its declared limit')
        state = self.unpack(vectors[:, 0])
        gauss_kinetic = sum(float(np.vdot(x, self.orientation(name, x)).real)
                           for name, x in state.items())
        rest_polarization = {}
        for flavour, label in enumerate(('heavy', 'light')):
            rest_polarization[label] = []
            for cell in range(2):
                observable = sparse_sum(self.frame[0, i, j] * self.matter.units[cell, flavour, i, j]
                                        for i in range(4) for j in range(4)).tocsr()
                value = sum(float(np.vdot(x, self.matter_action(
                    observable[self.blocks[name]['keep']][:, self.blocks[name]['keep']], x)).real)
                            for name, x in state.items())
                rest_polarization[label].append(value)
        return dict(energies=values.tolist(), eigen_residuals=residuals.tolist(),
                    scaled_eigen_residuals=relative_residuals.tolist(),
                    eigen_residual_scale='max(1, abs(E)); energy in declared lattice units',
                    eigen_residual_limit=tolerance,
                    eigenvector_orthogonality_residual=orthogonality,
                    eigenvector_orthogonality_limit=orthogonality_limit,
                    lowest_Ritz_vector_observables=dict(
                        orientation_gauss_kinetic_expectation=gauss_kinetic,
                        rest_field_polarization=rest_polarization,
                        meaning='Gauge-invariant field-relative c†Phi0 c at each cell; one finite lowest Ritz vector, not an infinite ground certificate'),
                    ritz_first_spacing=float(values[1] - values[0]),
                    spacing_interpretation='finite Galerkin Ritz spacing, not infinite gap or degeneracy certificate')


def source_hashes():
    here = Path(__file__).resolve().parent
    sources = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.parent == here and path.suffix == '.py':
                sources.add(path)
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(sources)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoffs', nargs='+', choices=tuple(CUTOFFS), default=['minimal'])
    parser.add_argument('--numbers', nargs='+', type=int, choices=(3, 4, 5), default=[3, 4, 5])
    parser.add_argument('--lambdas', nargs='+', type=float, default=[1., 8., 64.])
    parser.add_argument('--heavy-sign', type=int, choices=(-1, 1), default=-1,
                        help='Whole-H polarity: -1 balanced-complement fork; +1 original same-sign probe')
    parser.add_argument('--heavy-m', type=float, help='Heavy bare m; default exactly the light m')
    parser.add_argument('--heavy-r', type=float, help='Heavy bare r; default exactly the light r')
    parser.add_argument('--heavy-v', type=float, help='Heavy bare v; default exactly the light v')
    parser.add_argument('--inertias', nargs='+', type=float, default=[2.])
    parser.add_argument('--ue4', type=float, default=.3)
    parser.add_argument('--ue1', type=float, default=.4)
    parser.add_argument('--stiffness', type=float, default=2.)
    parser.add_argument('--m', type=float, default=.2)
    parser.add_argument('--r', type=float, default=.12)
    parser.add_argument('--v', type=float, default=.3)
    parser.add_argument('--levels', type=int, default=3)
    parser.add_argument('--tol', type=float, default=1e-8)
    parser.add_argument('--maxiter', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=222)
    parser.add_argument('--operators-only', action='store_true', help='No spectrum: still builds and applies the real complete Hamiltonians')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.stiffness <= 0 or args.tol <= 0 or args.maxiter < 1 or args.levels < 2:
        parser.error('Require positive stiffness/tolerance/maxiter and levels>=2')
    band = RotationBand(a=1., m=args.m, r=args.r, v=args.v, improved=False)
    heavy_band = RotationBand(a=1.,
        m=args.m if args.heavy_m is None else args.heavy_m,
        r=args.r if args.heavy_r is None else args.heavy_r,
        v=args.v if args.heavy_v is None else args.heavy_v, improved=False)
    equal_shapes = (heavy_band.m, heavy_band.r, heavy_band.v) == (band.m, band.r, band.v)
    basis = matrix_unit_basis()
    data = uniform_data(band, 3, args.stiffness, 0., basis)
    frame = data['frame']
    if data['orbit'].shape[1] != 15:
        raise AssertionError('Selected four-field frame does not have SU4/Z4 Lie orbit')
    result: dict[str, Any] = dict(model='two-cell two-flavour selected-orbit quantum SU4 Gauss EFT',
                  parameters={key: value for key, value in vars(args).items() if key != 'output'},
                  selected_source='local_orbital_stability.uniform_data(band,size=3,preferred=0)',
                  selected_frame_real=frame.real.tolist(), selected_frame_imaginary=frame.imag.tolist(),
                  selected_gram_diagonal=np.diag(data['selected']['G']).tolist(), orbit_dimension=15,
                  selected_gram_stationarity_residual=data['selected']['gradient_residual'],
                  heavy_base_coefficients=dict(a=heavy_band.a, m=heavy_band.m, r=heavy_band.r, v=heavy_band.v),
                  light_base_coefficients=dict(a=band.a, m=band.m, r=band.r, v=band.v),
                  heavy_coefficient_relation=dict(whole_H_sign=args.heavy_sign, equal_bare_shapes=equal_shapes,
                      balanced_complement_fork=args.heavy_sign == -1 and equal_shapes,
                      selection_status='explicit coefficient hypothesis, not dynamically proved'),
                  hypotheses=['fixed simultaneous adjoint orbit SU4/Z4 is an EFT restriction, not quantization of all 60 field coordinates',
                              'one genuine SU4 link; exact compact Haar measure and Peter-Weyl projection',
                              'two physical cells, two primitive flavours, four orbitals per flavour; raw Pauli code S=+',
                              'heavy NH=4 EM-neutral; light NL=3/4/5 charge q=NL-4',
                              'positive finite inertia and electric coefficients; Higgs alignment J=0'],
                  excluded_claims=['full 60-coordinate quantum dynamics', 'harmonic shapes', 'full three-dimensional photon',
                                   'infinite-volume or infinite-cutoff spectral gap', 'full phase or continuum S-matrix', 'full QED'],
                  geometry=dict(active_bonds=1, axis='x', onsite_Wilson_axis_count=1,
                                onsite_scalar=args.m + args.r, alignment_J=0.), links=[], runs=[])
    code = PauliCode()
    matter_cache = {}
    for cutoff in args.cutoffs:
        link = HaarLink(cutoff)
        result['links'].append(link.evidence())
        for number in args.numbers:
            if number not in matter_cache:
                matter = PrimitiveMatter(code, number, link.T)
                matter_cache[number] = matter, matter.algebra_evidence(link.f)
            matter, lie_residual = matter_cache[number]
            for inertia in args.inertias:
                for heavy in args.lambdas:
                    h = PhysicalHamiltonian(matter, link, frame, band, heavy, inertia, args.ue4, args.ue1,
                                            heavy_sign=args.heavy_sign, heavy_band=heavy_band)
                    run = dict(cutoff=cutoff, NH=4, NL=number, charge=h.q, Lambda=heavy, inertia=inertia,
                               heavy_sign=args.heavy_sign, heavy_whole_H_multiplier=h.heavy_multiplier,
                               matter_dimension=matter.dimension, physical_dimension=h.dimension,
                               boundary_rep=h.boundary_name, boundary_dimension=h.bd,
                               blocks={n: dict(matter_fibres=len(b['keep']), shape=list(b['shape'])) for n, b in h.blocks.items()},
                               raw_marker_policy='unchanged parity marker at mode16; marker has no SU4 or EM charge',
                               pauli_lie_residual=lie_residual, operators=h.operator_evidence(args.seed))
                    if not args.operators_only:
                        run['spectrum'] = h.spectrum(args.levels, args.tol, args.maxiter, args.seed)
                    result['runs'].append(run)
    result['source_sha256'] = source_hashes()
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.write_text(payload + '\n')
    print(payload)


if __name__ == '__main__':
    main()
