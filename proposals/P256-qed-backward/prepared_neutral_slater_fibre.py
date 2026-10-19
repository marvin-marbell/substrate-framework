#!/usr/bin/env python3
"""Fixed full108 Slater/reference weak fibre for the SAME ambient law.

SOURCE is evaluated as a coordinate-dependent physical function, not projected
onto a finite field-polynomial basis. Weak bilinear products use Wick BEFORE
projection. This is a native variational fibre, not full Fock dynamics.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'

import numpy as np
from scipy import sparse
from scipy.linalg import expm, null_space

from prepared_neutral_coherent_bank import (IDENTITY, canonical_bank_coordinate_marks,
                                           positive_spectator_modes)
from prepared_neutral_full_field import BETA
from prepared_neutral_full_onebody import full_coordinate_light_operators
from improved_phi_band import ImprovedPhiLaw, selected_band
from original_charged_principal import portable


def slater_expectation(projector: np.ndarray, a: np.ndarray,
                       b: np.ndarray | None = None) -> complex:
    """Exact full Slater expectation; suitable also for an occupied heavy sea.

    <dGamma(A)dGamma(B)> = Tr(PA)Tr(PB)+Tr(PA(I-P)B).
    No normal-order subtraction and no actual-ground assertion.
    """
    p = np.asarray(projector, complex)
    a = np.asarray(a, complex)
    ta = np.trace(p @ a)
    if b is None:
        return complex(ta)
    b = np.asarray(b, complex)
    return complex(ta * np.trace(p @ b) + np.trace(p @ a @ (np.eye(len(p)) - p) @ b))


def _light_weak(a: np.ndarray, occupied: np.ndarray, active: np.ndarray,
                particle: bool) -> np.ndarray:
    trace = np.trace(a[np.ix_(occupied, occupied)])
    block = a[np.ix_(active, active)]
    return trace * np.eye(len(active)) + (block if particle else -block.T)


def _light_product(a: np.ndarray, b: np.ndarray, occupied: np.ndarray,
                   active: np.ndarray, particle: bool) -> np.ndarray:
    """Exact single particle/hole Slater weak product, including outside pairs."""
    empty = np.setdiff1d(np.arange(len(a)), occupied, assume_unique=True)
    ta = np.trace(a[np.ix_(occupied, occupied)])
    tb = np.trace(b[np.ix_(occupied, occupied)])
    cross = np.einsum('ij,ji->', a[np.ix_(occupied, empty)], b[np.ix_(empty, occupied)])
    aa = a[np.ix_(active, active)]
    bb = b[np.ix_(active, active)]
    if particle:
        outside = b[np.ix_(active, occupied)] @ a[np.ix_(occupied, active)]
        return (ta * tb + cross) * np.eye(len(active)) + ta * bb + tb * aa + aa @ bb - outside
    outside = a[np.ix_(active, empty)] @ b[np.ix_(empty, active)]
    return ((ta * tb + cross) * np.eye(len(active)) - ta * bb.T - tb * aa.T
            + (bb @ aa - outside).T)


def _car_transfer(mask: int, annihilated: int, created: int) -> tuple[int, int]:
    """Actual even CAR product c_created^dagger c_annihilated, right first."""
    if not mask & (1 << annihilated) or mask & (1 << created):
        raise ValueError('Pauli-blocked basis transfer')
    phase = (-1) ** ((mask & ((1 << annihilated) - 1)).bit_count())
    mask ^= 1 << annihilated
    phase *= (-1) ** ((mask & ((1 << created) - 1)).bit_count())
    return mask | (1 << created), phase


class FullSlaterFibre:
    """One once-prepared sea54/Ref4 + sea58/Ref0 fibre; no field recapture.

    Coordinates, Hamiltonians, SOURCE and current may change. The occupied
    orbitals, basis CAR products and coherent raw coefficients never change.
    Charged ordering is reference index outer, light empty/hole index inner.
    """
    def __init__(self, law: Any, mean_fields: np.ndarray, mean_links: np.ndarray,
                 reference_site: int, profile: np.ndarray | None = None,
                 quartet: np.ndarray | None = None,
                 spectator_momenta: tuple | None = None):
        self.law = law
        self.reference_site = int(reference_site)
        self.mean_fields = np.array(mean_fields, complex, copy=True)
        self.mean_links = np.array(mean_links, complex, copy=True)
        self.profile = np.array(np.ones(law.volume) if profile is None else profile, complex, copy=True)
        self.quartet = np.array(BETA if quartet is None else quartet, complex, copy=True)
        if self.profile.shape != (law.volume,) or self.quartet.shape != (4,):
            raise ValueError('Original profile/quartet shapes required')
        if not 0 <= self.reference_site < law.volume:
            raise ValueError('Reference site is outside the SAME lattice')
        self.h0 = law.onebody(self.mean_fields, self.mean_links, np.zeros((law.volume, 3)))
        values, modes = np.linalg.eigh(self.h0)
        negative = modes[:, values < 0]
        if negative.shape[1] != 2 * law.volume or np.min(abs(values)) < 1e-10:
            raise ValueError('Actual h0 must have the entire gapped negative2V cluster')
        k = max(1, law.size // 4)
        momenta = ((k, 0, 0), (-k, 0, 0)) if spectator_momenta is None else spectator_momenta
        self.spectators = positive_spectator_modes(self.h0, law.sites, law.size, momenta)
        extra = self.spectators['modes']
        empty = null_space(np.column_stack((negative, extra)).conj().T)
        self.orbitals = np.column_stack((negative, extra, empty))
        self.n = len(self.h0)
        self.sea_count = negative.shape[1]
        self.occupied_indices = {1: np.arange(self.sea_count), -1: np.arange(self.sea_count + 4)}
        self.active_indices = {1: np.arange(self.sea_count, self.n), -1: np.arange(self.sea_count + 4)}
        self.occupied = {q: self.orbitals[:, ids] for q, ids in self.occupied_indices.items()}
        self.projectors = {q: u @ u.conj().T for q, u in self.occupied.items()}
        self.empty = self.orbitals[:, self.active_indices[1]]
        self.dimensions = {1: 4 * len(self.active_indices[1]), -1: 4 * len(self.active_indices[-1])}
        self.charged_dimension = sum(self.dimensions.values())
        self.charged_slices = {1: slice(0, self.dimensions[1]),
                               -1: slice(self.dimensions[1], self.charged_dimension)}
        self.raw_neutral = np.ones(2, complex) / math.sqrt(2)
        ref4 = 15 << self.n
        sea = (1 << self.sea_count) - 1
        sea_extra = (1 << (self.sea_count + 4)) - 1
        self.neutral_masks = (sea | ref4, sea_extra)
        self.charged_masks = []
        self.charged_car_phases = []
        self.charged_labels = []
        for q in (1, -1):
            neutral_mask = self.neutral_masks[0 if q == 1 else 1]
            for r, i in itertools.product(range(4), self.active_indices[q]):
                first, second = (self.n + r, int(i)) if q == 1 else (int(i), self.n + r)
                mask, phase = _car_transfer(neutral_mask, first, second)
                self.charged_masks.append(mask)
                self.charged_car_phases.append(phase)
                self.charged_labels.append((q, r, int(i)))
        gram_error = np.linalg.norm(self.orbitals.conj().T @ self.orbitals - np.eye(self.n))
        if gram_error > 1e-10 or len(set(self.charged_masks)) != self.charged_dimension:
            raise ValueError('Full Slater orbital/basis independence failed')
        self.orbital_gram_error = float(gram_error)
        # Prevent accidental preparation mutation; coordinate methods allocate only outputs.
        for array in (self.mean_fields, self.mean_links, self.profile, self.quartet,
                      self.h0, self.orbitals, self.raw_neutral, *self.occupied.values(),
                      *self.projectors.values(), self.empty):
            array.setflags(write=False)

    def overlap(self, sector: str = 'charged') -> np.ndarray:
        return np.eye(self._dimension(sector), dtype=complex)

    def _dimension(self, sector: str) -> int:
        if sector == 'neutral':
            return 2
        if sector == 'charged':
            return self.charged_dimension
        raise ValueError('Sector must be neutral or charged')

    def _rotate(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, complex)
        if a.shape != (self.n, self.n):
            raise ValueError('Actual FULL4V light operator required; no active cropping')
        return self.orbitals.conj().T @ a @ self.orbitals

    def _light_blocks(self, a: np.ndarray) -> dict[int, np.ndarray]:
        return {q: _light_weak(a, self.occupied_indices[q], self.active_indices[q], q == 1)
                for q in (1, -1)}

    @staticmethod
    def _reference_blocks(r: np.ndarray) -> dict[int, np.ndarray]:
        r = np.asarray(r, complex)
        if r.shape != (4, 4):
            raise ValueError('All FOUR physical reference modes required')
        return {1: np.trace(r) * np.eye(4) - r.T, -1: r}

    def onebody(self, a: np.ndarray, sector: str = 'charged') -> np.ndarray:
        """P dGamma(A) P, including the occupied-sea trace, arbitrary complex A."""
        rotated = self._rotate(a)
        if sector == 'neutral':
            return np.diag([np.trace(rotated[np.ix_(self.occupied_indices[q], self.occupied_indices[q])])
                            for q in (1, -1)])
        self._dimension(sector)
        blocks = self._light_blocks(rotated)
        out = np.zeros((self.charged_dimension, self.charged_dimension), complex)
        for q in (1, -1):
            out[self.charged_slices[q], self.charged_slices[q]] = np.kron(np.eye(4), blocks[q])
        return out

    def reference(self, r: np.ndarray, sector: str = 'charged') -> np.ndarray:
        """Derived full Ref dGamma(R): Ref3 holes use Tr(R)I-R^T, Ref1 uses R."""
        blocks = self._reference_blocks(r)
        if sector == 'neutral':
            return np.diag([np.trace(r), 0]).astype(complex)
        self._dimension(sector)
        out = np.zeros((self.charged_dimension, self.charged_dimension), complex)
        for q in (1, -1):
            out[self.charged_slices[q], self.charged_slices[q]] = np.kron(blocks[q], np.eye(len(self.active_indices[q])))
        return out

    def bilinear_product(self, a: np.ndarray, b: np.ndarray, sector: str = 'charged',
                         reference_A: np.ndarray | None = None,
                         reference_B: np.ndarray | None = None) -> np.ndarray:
        """P(dGamma(A)+dGamma(RA))(dGamma(B)+dGamma(RB))P BEFORE compression."""
        aa, bb = self._rotate(a), self._rotate(b)
        ra = np.zeros((4, 4), complex) if reference_A is None else np.asarray(reference_A, complex)
        rb = np.zeros((4, 4), complex) if reference_B is None else np.asarray(reference_B, complex)
        ref_a, ref_b = self._reference_blocks(ra), self._reference_blocks(rb)
        if sector == 'neutral':
            entries = []
            for q in (1, -1):
                ids = self.occupied_indices[q]
                empty = np.setdiff1d(np.arange(self.n), ids, assume_unique=True)
                ta, tb = np.trace(aa[np.ix_(ids, ids)]), np.trace(bb[np.ix_(ids, ids)])
                cross = np.einsum('ij,ji->', aa[np.ix_(ids, empty)], bb[np.ix_(empty, ids)])
                rta, rtb = (np.trace(ra), np.trace(rb)) if q == 1 else (0, 0)
                entries.append((ta + rta) * (tb + rtb) + cross)
            return np.diag(entries)
        self._dimension(sector)
        light_a, light_b = self._light_blocks(aa), self._light_blocks(bb)
        out = np.zeros((self.charged_dimension, self.charged_dimension), complex)
        for q in (1, -1):
            ids, active = self.occupied_indices[q], self.active_indices[q]
            light_ab = _light_product(aa, bb, ids, active, q == 1)
            # Reference holes/particles exhaust their fixed-number Ref sectors.
            ref_ab = ref_a[q] @ ref_b[q]
            block = (np.kron(np.eye(4), light_ab) + np.kron(ref_ab, np.eye(len(active)))
                     + np.kron(ref_b[q], light_a[q]) + np.kron(ref_a[q], light_b[q]))
            out[self.charged_slices[q], self.charged_slices[q]] = block
        return out

    def gauss(self, sector: str = 'charged') -> np.ndarray:
        """Absolute totalGauss = Nlight-2V+Nref-4, BOTH channels."""
        return (self.onebody(np.eye(self.n), sector) + self.reference(np.eye(4), sector)
                - (2 * self.law.volume + 4) * self.overlap(sector))

    def coulomb(self, green: np.ndarray, ue: float, sector: str = 'charged') -> np.ndarray:
        """Original HC=UE rho^T Green rho; exact Wick BEFORE weak projection.

        rho_x=n_x-2+(Nref-4)delta_xR. Green is the original lattice L^+.
        """
        green = np.asarray(green, float)
        if green.shape != (self.law.volume, self.law.volume) or ue < 0:
            raise ValueError('SAME lattice Green and nonnegative UE required')
        if np.linalg.norm(green - green.T) > 1e-10:
            raise ValueError('Coulomb Green must be symmetric')
        values, vectors = np.linalg.eigh(green)
        if values.min() < -1e-10:
            raise ValueError('Coulomb Green must be positive semidefinite')
        out = np.zeros_like(self.overlap(sector))
        for value, vector in zip(values, vectors.T):
            if value <= 0:
                continue
            a = np.diag(np.repeat(vector, 4))
            r = vector[self.reference_site] * np.eye(4)
            offset = -2 * np.sum(vector) - 4 * vector[self.reference_site]
            first = self.onebody(a, sector) + self.reference(r, sector)
            square = self.bilinear_product(a, a, sector, r, r)
            out += ue * value * (square + 2 * offset * first + offset ** 2 * self.overlap(sector))
        return out

    def source_vectors_from_marks(self, marks: dict[int, np.ndarray]) -> dict[str, Any]:
        """Exact all108 SOURCE rows in the actual evenCAR-labelled charged basis."""
        plus = np.asarray(marks[1], complex).reshape(self.n, 4)
        minus = np.asarray(marks[-1], complex).transpose(1, 0, 2).reshape(4, self.n)
        rows = {1: (self.empty.conj().T @ plus).T.reshape(-1),
                -1: (minus @ self.occupied[-1]).reshape(-1)}
        matrix = np.zeros((self.charged_dimension, 2), complex)
        matrix[self.charged_slices[1], 0] = rows[1]
        matrix[self.charged_slices[-1], 1] = rows[-1]
        vectors = {}
        for q in (1, -1):
            vectors[q] = np.zeros(self.charged_dimension, complex)
            vectors[q][self.charged_slices[q]] = rows[q] / math.sqrt(2)
        exact_norms = {1: float(np.trace(plus.conj().T @ (np.eye(self.n) - self.projectors[1]) @ plus).real / 2),
                       -1: float(np.trace(minus @ self.projectors[-1] @ minus.conj().T).real / 2)}
        total = vectors[1] + vectors[-1]
        gram = matrix.conj().T @ matrix
        z2 = float(np.vdot(total, total).real)
        if max(abs(np.vdot(vectors[q], vectors[q]).real - exact_norms[q]) for q in (1, -1)) > 1e-10:
            raise AssertionError('All108 SOURCE coefficients disagree with FULL Slater norm')
        return dict(full_SOURCE_marks=marks,full108_plus_coefficients=plus,full108_minus_coefficients=minus,
                    charged_source_rows=rows,source_matrix=matrix,source_vectors=vectors,total_source=total,
                    bank_source_Gram=gram,raw_SOURCE_norm_squared=exact_norms,total_SOURCE_norm_squared=z2,
                    SOURCE_normalizer=math.sqrt(z2),raw_bank_coefficients=self.raw_neutral.copy(),
                    projection='full SOURCE fibre exactly represented; field FUNCTION is NOT polynomial-projected')

    def source_vectors_at(self, fields: np.ndarray, links: np.ndarray, theta: np.ndarray,
                          a: np.ndarray, profile: np.ndarray | None = None,
                          quartet: np.ndarray | None = None) -> dict[str, Any]:
        """B(q) on FIXED prepared banks. NO h(q) diagonalization or recapture."""
        theta, a = np.asarray(theta, float), np.asarray(a, float)
        if theta.shape != (3,) or a.shape != (self.law.volume, 3):
            raise ValueError('Physical theta/A shapes required')
        if np.linalg.norm(np.mean(a, axis=0) * self.law.size - theta) > 1e-11:
            raise ValueError('theta must equal N mean(A), not a second harmonic input')
        marks = canonical_bank_coordinate_marks(self.law, fields, links, theta, self.reference_site,
                                               self.profile if profile is None else profile,
                                               self.quartet if quartet is None else quartet, a)
        return self.source_vectors_from_marks(marks)

    def coordinate(self, fields: np.ndarray, links: np.ndarray, theta: np.ndarray,
                   profile: np.ndarray, actual_a: np.ndarray,
                   quartet: np.ndarray | None = None) -> dict[str, Any]:
        source = self.source_vectors_at(fields, links, theta, actual_a, profile, quartet)
        h = self.law.onebody(fields, links, actual_a)
        return dict(source=source,full108_H=h,neutral_H=self.onebody(h, 'neutral'),
                    charged_H=self.onebody(h),unrepresented_onebody=self.onebody_graph_diagnostic(h),
                    fixed_basis=True,state_recaptured=False)

    def current_rows(self, a: np.ndarray, source: dict[str, Any]) -> dict[str, Any]:
        """Actual full108 current primitive and its exact weak/source response rows."""
        projected = self.onebody(a)
        second = self.bilinear_product(np.asarray(a).conj().T, a)
        v = source['total_source']
        return dict(full108_operator=np.asarray(a),charged_weak=projected,
                    charged_SOURCE_response=projected @ v,
                    full_SOURCE_response_norm_squared=float(np.vdot(v, second @ v).real),
                    unrepresented_SOURCE_response_norm_squared=float(np.vdot(v, (second - projected.conj().T @ projected) @ v).real))

    def onebody_graph_diagnostic(self, a: np.ndarray) -> dict[str, Any]:
        """Unrepresented dynamic particle-hole/multipair complement, not erased."""
        rotated = self._rotate(a)
        neutral_variance = {}
        for q in (1, -1):
            ids = self.occupied_indices[q]
            outside = np.setdiff1d(np.arange(self.n), ids, assume_unique=True)
            neutral_variance[q] = float(np.linalg.norm(rotated[np.ix_(outside, ids)]) ** 2)
        weak = self.onebody(a)
        product = self.bilinear_product(np.asarray(a).conj().T, a)
        remainder = product - weak.conj().T @ weak
        return dict(neutral_particle_hole_norm_squared=neutral_variance,
                    charged_unrepresented_graph_trace=float(np.trace(remainder).real),
                    charged_unrepresented_graph_frobenius=float(np.linalg.norm(remainder)),
                    scope='static mean Slater basis; omitted particle-hole/multipair dynamics remain outside weak fibre')

    def descriptor(self) -> dict[str, Any]:
        return dict(identity=IDENTITY,oneparticle_dimension=self.n,neutral_dimension=2,
                    charged_dimension=self.charged_dimension,charged_dimensions=self.dimensions,
                    neutral_light_counts=[self.sea_count,self.sea_count+4],neutral_reference_counts=[4,0],
                    charged_light_counts=[self.sea_count+1,self.sea_count+3],charged_reference_counts=[3,1],
                    fixed_total_number=self.sea_count+4,reference_charge='Nref-4 BOTH physical channels',
                    raw_neutral_coefficients=self.raw_neutral,orbital_Gram_error=self.orbital_gram_error,
                    neutral_masks_hex=[hex(m) for m in self.neutral_masks],
                    charged_masks_hex=[hex(m) for m in self.charged_masks],
                    charged_evenCAR_phases=self.charged_car_phases,charged_labels=self.charged_labels,
                    charged_plus_basis='a_empty^dagger r_r |negative2V Ref4>',
                    charged_minus_basis='r_r^dagger a_occupied |negative2V+R4 Ref0>',
                    spectator_momenta=self.spectators['momentum_blocks'],
                    scope='native SAME-law variational weak H; NOT full Fock dynamics/Gauss-kernel evolution/physical closed-loop proof')


def lattice_green(law: Any) -> np.ndarray:
    lap = np.zeros((law.volume, law.volume))
    for x in range(law.volume):
        for axis in range(3):
            y = int(law.target[x, axis])
            lap[x, x] += 1; lap[y, y] += 1; lap[x, y] -= 1; lap[y, x] -= 1
    values, modes = np.linalg.eigh(lap)
    positive = values > 1e-10
    return (modes[:, positive] / values[positive]) @ modes[:, positive].T


def _small_car_control() -> dict[str, Any]:
    """Independent exact10-mode CAR control of uncertain Wick/Ref/source signs."""
    n, sea = 6, 2
    dimension = 1 << (n + 4)
    annihilators = []
    for j in range(n + 4):
        columns = np.array([m for m in range(dimension) if m & (1 << j)])
        rows = columns ^ (1 << j)
        phases = [(-1) ** ((int(m) & ((1 << j) - 1)).bit_count()) for m in columns]
        annihilators.append(sparse.csr_matrix((phases, (rows, columns)), shape=(dimension, dimension)))
    rng = np.random.default_rng(22210858)
    a, b = [rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n)) for _ in range(2)]
    ra, rb = [rng.normal(size=(4,4)) + 1j*rng.normal(size=(4,4)) for _ in range(2)]
    def dgamma(light, ref):
        coefficients = np.zeros((n+4,n+4), complex)
        coefficients[:n,:n] = light; coefficients[n:,n:] = ref
        return sum((coefficients[i,j] * (annihilators[i].conj().T @ annihilators[j])
                    for i,j in itertools.product(range(n+4), repeat=2)),
                   sparse.csr_matrix((dimension,dimension),dtype=complex))
    da, db = dgamma(a,ra), dgamma(b,rb)
    masks = ((1 << sea)-1 | (15 << n), (1 << (sea+4))-1)
    errors = {}; wrong_transpose = []; wrong_normalorder = []
    source_coefficients = {1:rng.normal(size=(n,4))+1j*rng.normal(size=(n,4)),
                           -1:rng.normal(size=(4,n))+1j*rng.normal(size=(4,n))}
    seed = np.zeros(dimension, complex); seed[list(masks)] = 1/math.sqrt(2)
    source_errors = {}
    for q in (1,-1):
        occupied = np.arange(sea if q==1 else sea+4)
        active = np.arange(sea,n) if q==1 else occupied
        rows=[]; signs=[]
        for r,i in itertools.product(range(4),active):
            first,second=(n+r,int(i)) if q==1 else (int(i),n+r)
            mask,phase=_car_transfer(masks[0 if q==1 else 1],first,second)
            rows.append(mask);signs.append(phase)
        basis=sparse.csr_matrix((signs,(rows,np.arange(len(rows)))),shape=(dimension,len(rows)))
        la,lb=_light_weak(a,occupied,active,q==1),_light_weak(b,occupied,active,q==1)
        raa=FullSlaterFibre._reference_blocks(ra)[q];rbb=FullSlaterFibre._reference_blocks(rb)[q]
        weak=np.kron(np.eye(4),la)+np.kron(raa,np.eye(len(active)))
        product=(np.kron(np.eye(4),_light_product(a,b,occupied,active,q==1))
                 +np.kron(raa@rbb,np.eye(len(active)))+np.kron(rbb,la)+np.kron(raa,lb))
        errors[q]=[float(np.linalg.norm((basis.conj().T@da@basis).toarray()-weak)),
                   float(np.linalg.norm((basis.conj().T@da@db@basis).toarray()-product))]
        light_exact=(basis.conj().T@dgamma(a,np.zeros((4,4)))@basis).toarray()
        wrong=np.kron(np.eye(4),np.trace(a[np.ix_(occupied,occupied)])*np.eye(len(active))
                      + (a[np.ix_(active,active)] if q==1 else -a[np.ix_(active,active)]))
        wrong_transpose.append(float(np.linalg.norm(light_exact-wrong)))
        wrong_normalorder.append(float(np.linalg.norm(light_exact-np.kron(np.eye(4),la-np.trace(a[np.ix_(occupied,occupied)])*np.eye(len(active))))))
        coeff=source_coefficients[q]
        operator=sum((coeff[i,r]*(annihilators[i].conj().T@annihilators[n+r]) if q==1
                      else coeff[r,i]*(annihilators[n+r].conj().T@annihilators[i])
                      for i,r in itertools.product(range(n),range(4))),sparse.csr_matrix((dimension,dimension),dtype=complex))
        predicted=(coeff[active,:].T.reshape(-1) if q==1 else coeff[:,active].reshape(-1))/math.sqrt(2)
        source_errors[q]=float(np.linalg.norm(basis.conj().T@(operator@seed)-predicted))
    expectation_errors=[]
    light_a=dgamma(a,np.zeros((4,4)));light_b=dgamma(b,np.zeros((4,4)))
    for q in (1,-1):
        occupied=np.arange(sea if q==1 else sea+4)
        projector=np.diag(np.isin(np.arange(n),occupied).astype(float))
        mask=masks[0 if q==1 else 1]
        exact_first=light_a[mask,mask]
        exact_second=(light_a@light_b)[mask,mask]
        expectation_errors.extend([float(abs(exact_first-slater_expectation(projector,a))),
                                   float(abs(exact_second-slater_expectation(projector,a,b)))])
    if max(expectation_errors)>1e-10:
        raise AssertionError('Full Slater expectation helper disagrees with exact CAR')
    if max(v for row in errors.values() for v in row)>1e-10 or max(source_errors.values())>1e-12:
        raise AssertionError('Independent exact CAR contradicts Wick/hole/Ref/source signs')
    if wrong_transpose[-1]<1 or min(wrong_normalorder)<1:
        raise AssertionError('Adverse hole transpose/normal-order controls are not discriminating')
    return dict(actual_modes=n+4,full_Fock_dimension=dimension,Wick_light_Ref_product_errors=errors,
                exact_evenCAR_SOURCE_errors=source_errors,wrong_hole_transpose_defect=wrong_transpose[-1],
                full_Slater_expectation_errors=expectation_errors,
                wrong_normalorder_trace_removal_defects=wrong_normalorder)


def native_science() -> dict[str, Any]:
    started = time.perf_counter()
    band,frame,_ = selected_band()
    law=ImprovedPhiLaw(3,band);v=law.volume
    fields=np.tile(frame,(v,1,1,1));links=np.tile(np.eye(4,dtype=complex),(v,3,1,1))
    xyz=law.sites.astype(float)-1
    eta=np.exp(-np.sum(xyz*xyz,axis=1)/(2*.65**2))
    profiles=np.stack([xyz[:,0]*eta,xyz[:,1]*eta],axis=1)
    profiles/=np.linalg.norm(profiles,axis=0)
    ref=law.index[(0,1,1)]
    fibre=FullSlaterFibre(law,fields,links,ref,profiles[:,0])
    theta=np.array([.21,-.13,.09]);a=np.tile(theta/law.size,(v,1))
    # A real longitudinal/coexact perturbation retains the actual harmonic mean.
    a[:,2]+=.07*np.sin(2*np.pi*law.sites[:,0]/law.size)
    source_rows=[]
    for profile in (profiles[:,0],profiles[:,1],(profiles[:,0]+1j*profiles[:,1])/math.sqrt(2)):
        source=fibre.source_vectors_at(fields,links,theta,a,profile)
        source_rows.append(dict(profile_sha256=hashlib.sha256(profile.tobytes()).hexdigest(),
                                coefficient_norm_squared={q:float(np.linalg.norm(source['source_vectors'][q])**2) for q in (1,-1)},
                                full_Slater_norm_squared=source['raw_SOURCE_norm_squared'],
                                total_SOURCE_norm_squared=source['total_SOURCE_norm_squared'],
                                SOURCE_normalizer=source['SOURCE_normalizer'],bank_Gram=source['bank_source_Gram']))
    source=fibre.source_vectors_at(fields,links,theta,a)
    basis_hash=hashlib.sha256(fibre.orbitals.tobytes()).hexdigest()
    varied_fields=fields.copy()
    varied_fields[:,0]+=.017*np.sin(2*np.pi*law.sites[:,0]/law.size)[:,None,None]*frame[1]
    varied_links=links.copy()
    varied_links[:,2]=expm(.05j*frame[2])
    varied_source=fibre.source_vectors_at(varied_fields,varied_links,theta,a)
    field_function_change=float(np.linalg.norm(varied_source['total_source']-source['total_source']))
    fixed_basis_unchanged=basis_hash==hashlib.sha256(fibre.orbitals.tobytes()).hexdigest()
    if field_function_change<1e-5 or not fixed_basis_unchanged:
        raise AssertionError('Actual SOURCE field function was frozen or mean fibre recaptured')
    h=law.onebody(fields,links,a)
    h0diagnostic=fibre.onebody_graph_diagnostic(fibre.h0)
    actualdiagnostic=fibre.onebody_graph_diagnostic(h)
    if max(h0diagnostic['neutral_particle_hole_norm_squared'].values())>1e-20:
        raise AssertionError('h0 sea control is not stationary')
    if min(actualdiagnostic['neutral_particle_hole_norm_squared'].values())<1e-6:
        raise AssertionError('Actual h(A theta) wrongly substituted by stationary h0')
    first=np.zeros((v,3));first[:,1]=np.cos(2*np.pi*law.sites[:,0]/law.size)
    current=full_coordinate_light_operators(law,fields,links,a,first)['J']
    current_rows=fibre.current_rows(current,source)
    green=lattice_green(law);ue=.0025
    coulomb={sector:fibre.coulomb(green,ue,sector) for sector in ('neutral','charged')}
    positivity={sector:float(np.linalg.eigvalsh((matrix+matrix.conj().T)/2).min()) for sector,matrix in coulomb.items()}
    wrong=np.zeros_like(coulomb['charged'])
    values,modes=np.linalg.eigh(green)
    for value,mode in zip(values,modes.T):
        if value<=0:continue
        primitive=np.diag(np.repeat(mode,4));reference=mode[ref]*np.eye(4)
        offset=-2*sum(mode)-4*mode[ref]
        projected=fibre.onebody(primitive)+fibre.reference(reference)+offset*fibre.overlap()
        wrong+=ue*value*(projected@projected)
    correction=coulomb['charged']-wrong
    before_square_defect=float(np.linalg.norm(correction))
    neutral_coulomb=fibre.raw_neutral.conj()@coulomb['neutral']@fibre.raw_neutral
    source_coulomb=np.vdot(source['total_source'],coulomb['charged']@source['total_source'])/source['total_SOURCE_norm_squared']
    gauss_errors={s:float(np.linalg.norm(fibre.gauss(s))) for s in ('neutral','charged')}
    old_bare_ref0_gauss=2*v-2*v-4
    if min(positivity.values())<-1e-10 or before_square_defect<1e-4 or max(gauss_errors.values())>1e-10 or old_bare_ref0_gauss!=-4:
        raise AssertionError('HC positivity/preprojection/full reference absolute Gauss control failed')
    # Original plus map is quartet-linear; original minus map conjugates it.
    # Keep BOTH maps, never impose an invented shared complex-linear SOURCE.
    basis_source={q:[] for q in (1,-1)}
    for j in range(4):
        quartet=np.eye(4,dtype=complex)[j]
        column=fibre.source_vectors_at(fields,links,theta,a,quartet=quartet)
        for q in (1,-1):
            basis_source[q].append(column['source_vectors'][q])
    columns={q:np.column_stack(basis_source[q]) for q in (1,-1)}
    grams={q:columns[q].conj().T@columns[q] for q in (1,-1)}
    source_gram=grams[1]+grams[-1].T
    predicted=columns[1]@BETA+columns[-1]@BETA.conj()
    raw_identity_error=float(np.linalg.norm(predicted-source['total_source']))
    wrong_shared_linear_error=float(np.linalg.norm(columns[1]@BETA+columns[-1]@BETA-source['total_source']))
    if raw_identity_error>1e-11 or wrong_shared_linear_error<1e-3:
        raise AssertionError('Original linear/conjugate-linear raw SOURCE maps were changed')
    return dict(descriptor=fibre.descriptor(),SOURCE_profile_controls=source_rows,
                full_quartet_SOURCE_Gram=source_gram,channel_quartet_Grams=grams,
                raw_quartet_linear_conjugate_identity_error=raw_identity_error,
                wrong_shared_complex_linear_SOURCE_defect=wrong_shared_linear_error,
                actual_SOURCE_field_function_change=field_function_change,
                SOURCE_field_function_fixed_basis_unchanged=fixed_basis_unchanged,
                varied_SOURCE_full_Slater_norm_squared=varied_source['raw_SOURCE_norm_squared'],
                absolute_total_Gauss_errors=gauss_errors,old_bare_sea54_Ref0_Gauss=old_bare_ref0_gauss,
                original_Coulomb_min_eigenvalues=positivity,HC_project_before_square_defect=before_square_defect,
                HC_outside_pair_correction_min_eigenvalue=float(np.linalg.eigvalsh((correction+correction.conj().T)/2).min()),
                raw_neutral_Coulomb_expectation=complex(neutral_coulomb),normalized_SOURCE_Coulomb_expectation=complex(source_coulomb),
                full108_current_norm=float(np.linalg.norm(current)),
                full_SOURCE_current_response_norm_squared=current_rows['full_SOURCE_response_norm_squared'],
                unrepresented_SOURCE_current_response_norm_squared=current_rows['unrepresented_SOURCE_response_norm_squared'],
                h0_stationary_control=h0diagnostic,actual_h_A_theta_nonstationary=actualdiagnostic,
                four_positive_spectator_energy=fibre.spectators['extra_energy'],
                spectator_eigen_residuals=fibre.spectators['eigen_residuals'],
                independent_small_CAR=_small_car_control(),science_walltime_seconds=time.perf_counter()-started,
                not_claimed=['full Fock dynamics','full Gauss-kernel dynamics','full512 realization','actual ground','physical closed loop'])


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    root=Path(__file__).resolve().parent
    loaded={}
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if not isinstance(filename,str):
            continue
        path=Path(filename)
        if path.parent==root and path.suffix=='.py':
            loaded[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    started=time.perf_counter()
    result=native_science()
    result['loaded_scientific_source_sha256']=loaded
    result['loaded_sources_unchanged']=all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in loaded.items())
    result['runtime']=dict(python=sys.executable,version=sys.version,argv=sys.argv,cwd=os.getcwd(),
                           walltime_seconds=time.perf_counter()-started)
    result['native_inputs']=dict(N=3,UE=.0025,theta=[.21,-.13,.09],reference_site=[0,1,1],
                                 profile='original .65 Gaussian x/y spatial orbitals and their coherent complex combination',
                                 raw_quartet=BETA,sea_preparation='original h0 at A=0; entire54 negative modes',
                                 spectators='actual two positive spin pairs native +/-qx')
    print(json.dumps(portable(result),indent=2,sort_keys=True))


if __name__ == '__main__':
    main()
