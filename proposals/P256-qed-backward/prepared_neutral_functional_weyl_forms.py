"""Exact all52 weak forms for the original functional SOURCE/creator ABI.

The elementary base polynomials are constants. Incoming columns are their
actual canonical creators, not multiplication-only photons. A Fourier
operator shifts a phase without recreating the incoming photon polynomial.
"""
from __future__ import annotations
from dataclasses import replace
from typing import Any
import math
import numpy as np
from functional_source_photon_bank import CoexactOperatorPack, coefficient_block


def kernels(left: CoexactOperatorPack, right: CoexactOperatorPack,
            shift: np.ndarray | None = None) -> tuple[np.ndarray, ...]:
    """Four exact Gaussian blocks for a right Fourier multiplication operator."""
    l = np.asarray([term.function.phase for term in left.terms[:len(left.terms)//2]])
    r = np.asarray([term.function.phase for term in right.terms[:len(right.terms)//2]])
    k = np.zeros(52) if shift is None else np.asarray(shift)
    shifted = r+k
    distance = np.sum(l*l, axis=1)[:, None]+np.sum(shifted*shifted, axis=1)[None, :]-2*l@shifted.T
    base = np.exp(-np.maximum(distance, 0)/4)
    right_mean = 1j*((k-l)@right.packet)[:, None]/math.sqrt(2)
    left_mean = 1j*((r+k)@left.packet.conj())[None, :]/math.sqrt(2)
    photon = np.vdot(left.packet, right.packet)+left_mean*right_mean
    return base, right_mean*base, left_mean*base, photon*base


def oscillator_kernels(left: CoexactOperatorPack, right: CoexactOperatorPack,
                       omega: np.ndarray) -> tuple[np.ndarray, ...]:
    """Literal oscillator sum omega/2*(−d²+y²), zero point ONCE."""
    l = np.asarray([term.function.phase for term in left.terms[:len(left.terms)//2]])
    r = np.asarray([term.function.phase for term in right.terms[:len(right.terms)//2]])
    metric = kernels(left, right)
    energy = float(np.sum(omega))/2+(l*omega)@r.T/2
    a = -1j*(l@right.packet)[:, None]/math.sqrt(2)
    b = 1j*(r@left.packet.conj())[None, :]/math.sqrt(2)
    aw = -1j*((l*omega)@right.packet)[:, None]/math.sqrt(2)
    bw = 1j*((r*omega)@left.packet.conj())[None, :]/math.sqrt(2)
    return (energy*metric[0], energy*metric[1]+aw*metric[0],
            energy*metric[2]+bw*metric[0],
            energy*metric[3]+(np.vdot(left.packet, omega*right.packet)+a*bw+aw*b)*metric[0])


def directional_kernels(left: CoexactOperatorPack, right: CoexactOperatorPack,
                        direction: np.ndarray, shift: np.ndarray) -> tuple[np.ndarray, ...]:
    """Exact <L, exp(i shift.y) D_direction(physical right Gaussian R)>.

    The derivative is BEFORE weak Hermitian symmetrization. Its half-divergence
    is therefore not inserted a second time. Harmonic derivatives are separate.
    """
    l = np.asarray([term.function.phase for term in left.terms[:len(left.terms)//2]])
    r = np.asarray([term.function.phase for term in right.terms[:len(right.terms)//2]])
    shifted = r+shift
    distance = np.sum(l*l, axis=1)[:, None]+np.sum(shifted*shifted, axis=1)[None, :]-2*l@shifted.T
    base = np.exp(-np.maximum(distance, 0)/4)
    fmean = .5j*((r@direction)[None, :]+shift@direction-(l@direction)[:, None])
    lmean = 1j*((r+shift)@left.packet.conj())[None, :]/math.sqrt(2)
    rmean = 1j*((shift-l)@right.packet)[:, None]/math.sqrt(2)
    lflow = math.sqrt(2)*(left.packet.conj()@direction)
    rflow = math.sqrt(2)*(right.packet@direction)
    lr = np.vdot(left.packet, right.packet)+lmean*rmean
    plain = 1j*(r@direction)[None, :]-fmean
    return (plain*base, (rflow+plain*rmean-.5*rflow)*base,
            (plain*lmean-.5*lflow)*base,
            (rflow*lmean+plain*lr-.5*lflow*rmean-.5*rflow*lmean)*base)


def bilinear(left: CoexactOperatorPack, right: CoexactOperatorPack,
             operator: np.ndarray | None = None, shift: np.ndarray | None = None,
             scalar_kernels: tuple[np.ndarray, ...] | None = None) -> np.ndarray:
    """Exact coexact moment; no field-node Gaussian or quadrature is used."""
    if left.fibre_dimension != right.fibre_dimension:
        raise ValueError('Physical fibre dimensions must agree')
    for pack in (left, right):
        for term in pack.terms[:len(pack.terms)//2]:
            if term.function.polynomial.terms != {(): 1}:
                raise ValueError('This compiler requires the actual constant-base/full-creator ABI')
    rhs = right
    if operator is not None:
        terms = tuple(replace(term, fibre_coefficients=operator@term.fibre_coefficients)
                      for term in right.terms[:len(right.terms)//2])
        # coefficient_block consumes base half only. The incoming coefficients
        # are identical, their FULL polynomials remain in the scalar kernels.
        incoming = tuple(replace(term, fibre_coefficients=base.fibre_coefficients)
                         for term, base in zip(right.terms[len(terms):], terms))
        rhs = replace(right, terms=terms+incoming)
    blocks = kernels(left, right, shift) if scalar_kernels is None else scalar_kernels
    nl, nr = left.base_channel_count, right.base_channel_count
    result = np.zeros((2*nl, 2*nr), complex)
    for start in range(0, left.fibre_dimension, 24):
        rows = slice(start, min(start+24, left.fibre_dimension))
        a = coefficient_block(left, rows)
        b = coefficient_block(rhs, rows)
        for kernel, i, j in ((blocks[0], 0, 0), (blocks[1], 0, 1),
                             (blocks[2], 1, 0), (blocks[3], 1, 1)):
            applied = (kernel@b.reshape(len(b), -1)).reshape(len(a), len(range(left.fibre_dimension)[rows]), nr)
            result[i*nl:(i+1)*nl, j*nr:(j+1)*nr] += np.einsum('tfk,tfl->kl', a.conj(), applied, optimize=True)
    return result


def linear_pack(packs: tuple[CoexactOperatorPack, ...], weights: tuple[complex, ...],
                operator: np.ndarray | None = None) -> CoexactOperatorPack:
    """Linear combination at fixed literal Fourier/channel structure.

    q/theta dependence belongs to the coefficient fibres; altered frequencies
    or channel maps are rejected rather than silently paired by list position.
    """
    reference = packs[0]
    count = len(reference.terms)//2
    result = []
    for index, term in enumerate(reference.terms[:count]):
        coefficient = np.zeros_like(term.fibre_coefficients, dtype=complex)
        for pack, weight in zip(packs, weights):
            if len(pack.terms)//2 != count or not np.array_equal(pack.packet, reference.packet):
                raise ValueError('Changed functional Fourier/creator structure')
            other = pack.terms[index]
            if not (np.array_equal(other.function.phase, term.function.phase)
                    and np.array_equal(other.channel_coefficients, term.channel_coefficients)
                    and np.array_equal(other.channels, term.channels)):
                raise ValueError('Changed Fourier/channel map needs a general polynomial compiler')
            coefficient += weight*other.fibre_coefficients
        if operator is not None:
            coefficient += operator@term.fibre_coefficients
        result.append(replace(term, fibre_coefficients=coefficient))
    incoming = tuple(replace(term, fibre_coefficients=base.fibre_coefficients)
                     for term, base in zip(reference.terms[count:], result))
    return replace(reference, terms=tuple(result)+incoming)
